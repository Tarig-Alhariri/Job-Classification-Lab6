"""
Model Training and Hyperparameter Tuning Module
Модуль обучения моделей и настройки гиперпараметров
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import csv
from datetime import datetime
from text_processor import TextProcessor, split_dataset, create_job_categories_from_skills
from ml_model import MLModel, ModelComparator


class ModelTrainer:
    """
    Model training and hyperparameter tuning class
    Класс для обучения моделей и настройки гиперпараметров
    """
    
    def __init__(self, results_dir: str = "training_results"):
        """
        Initialize model trainer
        Инициализация тренера моделей
        """
        self.results_dir = results_dir
        self.text_processor = TextProcessor()
        self.training_history = []
        os.makedirs(results_dir, exist_ok=True)
    
    def balance_categories(self, df: pd.DataFrame, target_column: str, min_samples: int = 2) -> pd.DataFrame:
        """
        Balance categories by removing classes with too few samples
        """
        print("Балансировка категорий...")
        
        category_counts = df[target_column].value_counts()
        valid_categories = category_counts[category_counts >= min_samples].index
        balanced_df = df[df[target_column].isin(valid_categories)]
        
        return balanced_df
    
    def load_and_prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data and prepare for training
        Загрузка данных и подготовка к обучению
        تحميل البيانات والتحضير للتدريب
        """
        print("Загрузка данных из CSV файла...")
        
        # Use the same method as data_analysis.py
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', sep=';' ,quoting=csv.QUOTE_NONE)
                print("Файл загружен с кодировкой: utf-8, разделитель: ;")
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    df = pd.read_csv(csv_path, encoding='cp1251', sep=';',quoting=csv.QUOTE_NONE)
                    print("Файл загружен с кодировкой: cp1251, разделитель: ;")
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        df = pd.read_csv(csv_path, encoding='latin-1', sep=';',quoting=csv.QUOTE_NONE)
                        print("Файл загружен с кодировкой: latin-1, разделитель: ;")
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        # If semicolon fails, try with comma
                        # Если точка с запятой не работает, попробовать запятую
                        # إذا فشلت الفاصلة المنقوطة، جرب الفاصلة
                        try:
                            df = pd.read_csv(csv_path, encoding='utf-8',quoting=csv.QUOTE_NONE)
                            print("Файл загружен с кодировкой: utf-8, разделитель: ,")
                        except UnicodeDecodeError:
                            try:
                                df = pd.read_csv(csv_path, encoding='cp1251')
                                print("Файл загружен с кодировкой: cp1251, разделитель: ,")
                            except UnicodeDecodeError:
                                df = pd.read_csv(csv_path, encoding='latin-1')
                                print("Файл загружен с кодировкой: latin-1, разделитель: ,")
        
        except Exception as e:
            raise ValueError(f"Не удалось загрузить файл: {str(e)}")
        
        print(f"Доступные столбцы: {list(df.columns)}")
        print(f"Размер данных: {df.shape}")
        
        # Find skills column automatically
        skills_column = None
        for col in df.columns:
            if 'skill' in col.lower() or 'навык' in col.lower():
                skills_column = col
                break
        
        if skills_column is None:
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                skills_column = text_columns[0]
                print(f"Столбец навыков не найден, используется: {skills_column}")
            else:
                raise ValueError("Не найден подходящий столбец для анализа")
        else:
            print(f"Найден столбец навыков: {skills_column}")
        
        print("Создание категорий вакансий на основе навыков...")
        # df['category'] = create_job_categories_from_skills(df[skills_column])
        df['category'] = create_job_categories_from_skills(df[skills_column].fillna(''))
        
        # Balance categories
        df = self.balance_categories(df, 'category', min_samples=2)
        
        print("Разделение данных на тренировочные, валидационные и демонстрационные...")
        df = self.balance_categories(df, 'category', min_samples=2)
        train_df, val_df, demo_df = split_dataset(df, 'category')
        
        return train_df, val_df, demo_df
    
    def preprocess_text_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           demo_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess text data and convert to TF-IDF
        Предобработка текстовых данных и преобразование в TF-IDF
        """
        print("Предобработка текстовых данных...")
        
        all_skills = pd.concat([train_df['skills'], val_df['skills'], demo_df['skills']])
        processed_skills = all_skills.apply(
            lambda x: self.text_processor.preprocess_text(str(x))
        )
        
        print("Обучение TF-IDF векторизатора...")
        self.text_processor.fit_tfidf(processed_skills.tolist())
        
        print("Преобразование данных в TF-IDF векторы...")
        X_train = self.text_processor.transform_tfidf(
            train_df['skills'].apply(lambda x: self.text_processor.preprocess_text(str(x))).tolist()
        )
        X_val = self.text_processor.transform_tfidf(
            val_df['skills'].apply(lambda x: self.text_processor.preprocess_text(str(x))).tolist()
        )
        X_demo = self.text_processor.transform_tfidf(
            demo_df['skills'].apply(lambda x: self.text_processor.preprocess_text(str(x))).tolist()
        )
        
        y_train = train_df['category'].values
        y_val = val_df['category'].values
        y_demo = demo_df['category'].values
        
        return X_train, X_val, X_demo, y_train, y_val, y_demo
    
    def create_initial_models(self) -> ModelComparator:
        """
        Create initial models for comparison
        Создание начальных моделей для сравнения
        """
        print("Создание моделей для сравнения...")
        comparator = ModelComparator()
        
        models_config = [
            ('logistic_regression_1', 'logistic_regression', {'C': 1.0, 'max_iter': 1000}),
            ('random_forest_1', 'random_forest', {'n_estimators': 100, 'max_depth': 10}),
            ('svm_1', 'svm', {'C': 1.0, 'kernel': 'linear'}),
            ('naive_bayes_1', 'naive_bayes', {'alpha': 1.0})
        ]
        
        for name, model_type, params in models_config:
            model = MLModel(model_type, **params)
            comparator.add_model(name, model)
        
        return comparator
    
    def train_initial_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Train initial models and compare results
        Обучение начальных моделей и сравнение результатов
        """
        print("Начало обучения начальных моделей...")
        comparator = self.create_initial_models()
        results_df = comparator.train_and_compare(X_train, y_train, X_val, y_val)
        
        best_model, best_name = comparator.get_best_model()
        if best_model:
            print(f"Лучшая модель: {best_name}")
            self.best_model = best_model
            self.best_model_name = best_name
        
        self.save_training_results(comparison_results=results_df, stage="initial")
        return results_df
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Perform hyperparameter tuning
        Выполнение настройки гиперпараметров
        """
        print("Начало настройки гиперпараметров...")
        
        hyperparameter_configs = [
            # Logistic Regression configurations
            ('logistic_regression_C_0.1', 'logistic_regression', {'C': 0.1, 'max_iter': 1000}),
            ('logistic_regression_C_10', 'logistic_regression', {'C': 10.0, 'max_iter': 1000}),
            
            # Random Forest configurations
            ('random_forest_50_trees', 'random_forest', {'n_estimators': 50, 'max_depth': 5}),
            ('random_forest_200_trees', 'random_forest', {'n_estimators': 200, 'max_depth': 15}),
            
            # SVM configurations
            ('svm_C_0.1', 'svm', {'C': 0.1, 'kernel': 'linear'}),
            ('svm_rbf', 'svm', {'C': 1.0, 'kernel': 'rbf'}),
            
            # Naive Bayes configurations
            ('naive_bayes_alpha_0.5', 'naive_bayes', {'alpha': 0.5}),
            ('naive_bayes_alpha_2.0', 'naive_bayes', {'alpha': 2.0})
        ]
        
        comparator = ModelComparator()
        
        for name, model_type, params in hyperparameter_configs:
            print(f"Обучение модели с гиперпараметрами: {name}")
            model = MLModel(model_type, **params)
            comparator.add_model(name, model)
        
        results_df = comparator.train_and_compare(X_train, y_train, X_val, y_val)
        
        best_model, best_name = comparator.get_best_model()
        if best_model:
            print(f"Лучшая модель после настройки: {best_name}")
            self.best_model = best_model
            self.best_model_name = best_name
        
        self.save_training_results(comparison_results=results_df, stage="hyperparameter_tuning")
        return results_df
    
    def test_on_demo_data(self, X_demo: np.ndarray, y_demo: np.ndarray) -> Dict[str, Any]:
        """
        Test best model on demonstration data
        Тестирование лучшей модели на демонстрационных данных
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Лучшая модель не определена. Сначала выполните обучение.")
        
        print("Тестирование лучшей модели на демонстрационных данных...")
        results = self.best_model.evaluate(X_demo, y_demo)
        
        demo_results = {
            'model_name': self.best_model_name,
            'demo_accuracy': results['accuracy'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_demo_results(demo_results)
        return demo_results
    
    def save_training_results(self, comparison_results: pd.DataFrame, stage: str) -> None:
        """
        Save training results to file
        Сохранение результатов обучения в файл
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage}_results_{timestamp}.txt"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Результаты {stage}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(comparison_results.to_string())
            f.write("\n\nЛучшая модель:\n")
            
            if hasattr(self, 'best_model_name'):
                f.write(f"Имя: {self.best_model_name}\n")
                best_result = comparison_results[comparison_results['model_name'] == self.best_model_name]
                if not best_result.empty:
                    f.write(f"Точность на валидации: {best_result['val_accuracy'].iloc[0]:.4f}\n")
        
        print(f"Результаты сохранены в: {filepath}")
        
        record = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'results_file': filename,
            'best_model': getattr(self, 'best_model_name', None),
            'best_accuracy': float(comparison_results['val_accuracy'].max()) if not comparison_results.empty else 0.0
        }
        self.training_history.append(record)
    
    def save_demo_results(self, demo_results: Dict[str, Any]) -> None:
        """
        Save demonstration results to file
        Сохранение демонстрационных результатов в файл
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_results_{timestamp}.txt"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Результаты тестирования на демонстрационных данных\n")
            f.write("=" * 60 + "\n")
            f.write(f"Время: {demo_results['timestamp']}\n")
            f.write(f"Модель: {demo_results['model_name']}\n")
            f.write(f"Точность: {demo_results['demo_accuracy']:.4f}\n\n")
            
            f.write("Отчет по классификации:\n")
            for class_name, metrics in demo_results['classification_report'].items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    if class_name == 'accuracy':
                        f.write(f"Точность: {metrics:.4f}\n")
                    else:
                        f.write(f"{class_name}: precision={metrics['precision']:.4f}, "
                               f"recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}\n")
                elif isinstance(metrics, dict):
                    f.write(f"Класс {class_name}: precision={metrics['precision']:.4f}, "
                           f"recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}\n")
            
            f.write(f"\nМатрица ошибок:\n")
            for row in demo_results['confusion_matrix']:
                f.write(f"{row}\n")
        
        print(f"Демонстрационные результаты сохранены в: {filepath}")
    
    def save_best_model(self, model_path: str = "models/best_model.joblib") -> bool:
        """
        Save the best trained model
        Сохранение лучшей обученной модели
        """
        if not hasattr(self, 'best_model'):
            print("Лучшая модель не определена")
            return False
        
        success = self.best_model.save_model(model_path)
        if success:
            print(f"Лучшая модель сохранена: {model_path}")
        return success
    
    def generate_hyperparameter_analysis(self) -> str:
        """
        Generate hyperparameter influence analysis
        Генерация анализа влияния гиперпараметров
        """
        print("Генерация анализа влияния гиперпараметров...")
        
        analysis = "Анализ влияния гиперпараметров на результаты\n"
        analysis += "=" * 50 + "\n\n"
        
        if len(self.training_history) < 2:
            analysis += "Недостаточно данных для анализа гиперпараметров\n"
            return analysis
        
        initial_results = [r for r in self.training_history if r['stage'] == 'initial']
        tuning_results = [r for r in self.training_history if r['stage'] == 'hyperparameter_tuning']
        
        if initial_results and tuning_results:
            initial_acc = initial_results[0]['best_accuracy']
            tuning_acc = tuning_results[0]['best_accuracy']
            improvement = tuning_acc - initial_acc
            
            analysis += f"Сравнение до и после настройки гиперпараметров:\n"
            analysis += f"Лучшая точность до настройки: {initial_acc:.4f}\n"
            analysis += f"Лучшая точность после настройки: {tuning_acc:.4f}\n"
            analysis += f"Улучшение: {improvement:.4f} ({improvement*100:.2f}%)\n\n"
            
            analysis += "Выводы:\n"
            if improvement > 0:
                analysis += "• Настройка гиперпараметров улучшила производительность модели\n"
                analysis += "• Разные комбинации параметров значительно влияют на точность\n"
                analysis += "• Важно экспериментировать с различными значениями параметров\n"
            else:
                analysis += "• Настройка гиперпараметров не дала значительного улучшения\n"
                analysis += "• Начальные параметры были уже близки к оптимальным\n"
        
        # analysis += "\nРекомендации по гиперпараметрам:\n"
        # analysis += "• Logistic Regression: Поэкспериментируйте с параметром C (0.1-10)\n"
        # analysis += "• Random Forest: Увеличивайте n_estimators для лучшей производительности\n"
        # analysis += "• SVM: Попробуйте разные ядра (linear, rbf) и значения C\n"
        # analysis += "• Naive Bayes: Параметр alpha влияет на сглаживание\n"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparameter_analysis_{timestamp}.txt"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(analysis)
        
        print(f"Анализ гиперпараметров сохранен в: {filepath}")
        return analysis


def main():
    """
    Main function for testing
    Основная функция для тестирования
    """
    print("Тестирование модуля обучения моделей...")
    
    trainer = ModelTrainer()
    
    try:
        train_df, val_df, demo_df = trainer.load_and_prepare_data("jobs.csv")
        X_train, X_val, X_demo, y_train, y_val, y_demo = trainer.preprocess_text_data(
            train_df, val_df, demo_df
        )
        
        initial_results = trainer.train_initial_models(X_train, y_train, X_val, y_val)
        tuning_results = trainer.hyperparameter_tuning(X_train, y_train, X_val, y_val)
        demo_results = trainer.test_on_demo_data(X_demo, y_demo)
        
        trainer.save_best_model()
        analysis = trainer.generate_hyperparameter_analysis()
        
        print("Обучение и тестирование завершены успешно!")
        
    except Exception as e:
        print(f"Ошибка при выполнении: {str(e)}")


if __name__ == "__main__":
    main()