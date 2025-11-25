"""
Machine Learning Models Module for Laboratory Work 6
Модуль машинного обучения для лабораторной работы 6
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import os


class MLModel:
    """
    Machine Learning Model Class
    Класс модели машинного обучения
    """
    
    def __init__(self, model_type: str = 'logistic_regression', **model_params):
        """
        Initialize ML model
        Инициализация модели машинного обучения
        
        Args:
            model_type (str): Type of model to use
            model_type (str): Тип используемой модели
            **model_params: Additional model parameters
            **model_params: Дополнительные параметры модели
        """
        self.model_type = model_type
        self.model = None
        self.model_params = model_params
        self.is_trained = False
        self.initialize_model()
    
    def initialize_model(self) -> None:
        """
        Initialize the selected model
        Инициализация выбранной модели
        """
        model_map = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'naive_bayes': MultinomialNB
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
        
        model_class = model_map[self.model_type]
        
        # Set default parameters for each model type
        # Установка параметров по умолчанию для каждого типа модели
        default_params = self._get_default_params()
        default_params.update(self.model_params)
        
        self.model = model_class(**default_params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for each model type
        Получение параметров по умолчанию для каждого типа модели
        
        Returns:
            Dict[str, Any]: Default parameters dictionary
            Dict[str, Any]: Словарь параметров по умолчанию
        """
        default_params = {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'linear',
                'random_state': 42
            },
            'naive_bayes': {
                'alpha': 1.0
            }
        }
        
        return default_params.get(self.model_type, {})
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the model
        Обучение модели
        
        Args:
            X_train (np.ndarray): Training features
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Training labels
            y_train (np.ndarray): Обучающие метки
            
        Returns:
            Dict[str, Any]: Training results
            Dict[str, Any]: Результаты обучения
        """
        try:
            print(f"Начало обучения модели: {self.model_type}")
            print(f"Размер обучающих данных: {X_train.shape}")
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate training accuracy
            # Расчет точности на обучающих данных
            train_predictions = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            
            # Cross-validation
            # Кросс-валидация
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            results = {
                'train_accuracy': train_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model_type': self.model_type,
                'is_trained': True
            }
            
            print(f"Обучение завершено. Точность на обучающих данных: {train_accuracy:.4f}")
            print(f"Кросс-валидация (среднее): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return results
            
        except Exception as e:
            print(f"Ошибка при обучении модели: {str(e)}")
            return {
                'train_accuracy': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'model_type': self.model_type,
                'is_trained': False,
                'error': str(e)
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        Выполнение прогнозов
        
        Args:
            X (np.ndarray): Input features
            X (np.ndarray): Входные признаки
            
        Returns:
            np.ndarray: Predictions
            np.ndarray: Прогнозы
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала выполните обучение.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        Получение вероятностей прогнозов
        
        Args:
            X (np.ndarray): Input features
            X (np.ndarray): Входные признаки
            
        Returns:
            np.ndarray: Prediction probabilities
            np.ndarray): Вероятности прогнозов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала выполните обучение.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            print("Предупреждение: Модель не поддерживает вероятности прогнозов")
            return np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        Оценка производительности модели
        
        Args:
            X_test (np.ndarray): Test features
            X_test (np.ndarray): Тестовые признаки
            y_test (np.ndarray): Test labels
            y_test (np.ndarray): Тестовые метки
            
        Returns:
            Dict[str, Any]: Evaluation results
            Dict[str, Any]: Результаты оценки
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала выполните обучение.")
        
        try:
            predictions = self.predict(X_test)
            probabilities = self.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            class_report = classification_report(y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predictions)
            
            results = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': predictions.tolist(),
                'has_probabilities': len(probabilities) > 0
            }
            
            if len(probabilities) > 0:
                results['probabilities'] = probabilities.tolist()
            
            print(f"Точность на тестовых данных: {accuracy:.4f}")
            print("Отчет по классификации:")
            print(classification_report(y_test, predictions))
            
            return results
            
        except Exception as e:
            print(f"Ошибка при оценке модели: {str(e)}")
            return {
                'accuracy': 0.0,
                'classification_report': {},
                'confusion_matrix': [],
                'error': str(e)
            }
    
    def save_model(self, file_path: str) -> bool:
        """
        Save trained model to file
        Сохранение обученной модели в файл
        
        Args:
            file_path (str): Path to save the model
            file_path (str): Путь для сохранения модели
            
        Returns:
            bool: Success status
            bool: Статус успеха
        """
        if not self.is_trained:
            print("Предупреждение: Модель не обучена. Нечего сохранять.")
            return False
        
        try:
            # Create directory if it doesn't exist
            # Создание директории, если она не существует
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            joblib.dump(self.model, file_path)
            print(f"Модель сохранена: {file_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении модели: {str(e)}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """
        Load trained model from file
        Загрузка обученной модели из файла
        
        Args:
            file_path (str): Path to the model file
            file_path (str): Путь к файлу модели
            
        Returns:
            bool: Success status
            bool: Статус успеха
        """
        try:
            if not os.path.exists(file_path):
                print(f"Файл модели не найден: {file_path}")
                return False
            
            self.model = joblib.load(file_path)
            self.is_trained = True
            print(f"Модель загружена: {file_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {str(e)}")
            return False


class ModelComparator:
    """
    Compare multiple machine learning models
    Сравнение нескольких моделей машинного обучения
    """
    
    def __init__(self):
        """
        Initialize model comparator
        Инициализация компаратора моделей
        """
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model: MLModel) -> None:
        """
        Add model to comparator
        Добавление модели в компаратор
        
        Args:
            name (str): Model name
            name (str): Имя модели
            model (MLModel): ML model instance
            model (MLModel): Экземпляр модели машинного обучения
        """
        self.models[name] = model
    
    def train_and_compare(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Train and compare all models
        Обучение и сравнение всех моделей
        
        Args:
            X_train (np.ndarray): Training features
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Training labels
            y_train (np.ndarray): Обучающие метки
            X_val (np.ndarray): Validation features
            X_val (np.ndarray): Валидационные признаки
            y_val (np.ndarray): Validation labels
            y_val (np.ndarray): Валидационные метки
            
        Returns:
            pd.DataFrame: Comparison results
            pd.DataFrame): Результаты сравнения
        """
        comparison_results = []
        
        print("Начало сравнения моделей...")
        print(f"Количество моделей для сравнения: {len(self.models)}")
        
        for name, model in self.models.items():
            print(f"\n--- Обучение модели: {name} ---")
            
            # Train model
            # Обучение модели
            train_results = model.train(X_train, y_train)
            
            # Evaluate on validation set
            # Оценка на валидационном наборе
            if model.is_trained:
                val_results = model.evaluate(X_val, y_val)
                
                result = {
                    'model_name': name,
                    'model_type': model.model_type,
                    'train_accuracy': train_results.get('train_accuracy', 0),
                    'val_accuracy': val_results.get('accuracy', 0),
                    'cv_mean': train_results.get('cv_mean', 0),
                    'cv_std': train_results.get('cv_std', 0),
                    'is_trained': True
                }
                
                self.results[name] = {
                    'train_results': train_results,
                    'val_results': val_results,
                    'model': model
                }
                
            else:
                result = {
                    'model_name': name,
                    'model_type': model.model_type,
                    'train_accuracy': 0,
                    'val_accuracy': 0,
                    'cv_mean': 0,
                    'cv_std': 0,
                    'is_trained': False,
                    'error': train_results.get('error', 'Unknown error')
                }
            
            comparison_results.append(result)
        
        # Create comparison dataframe
        # Создание dataframe для сравнения
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('val_accuracy', ascending=False)
        
        print("\n--- Результаты сравнения моделей ---")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[Optional[MLModel], Optional[str]]:
        """
        Get the best performing model
        Получение лучшей модели по производительности
        
        Returns:
            Tuple[Optional[MLModel], Optional[str]]: Best model and its name
            Tuple[Optional[MLModel], Optional[str]]: Лучшая модель и ее имя
        """
        if not self.results:
            return None, None
        
        best_name = None
        best_accuracy = -1
        
        for name, result in self.results.items():
            accuracy = result['val_results'].get('accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_name = name
        
        if best_name:
            return self.results[best_name]['model'], best_name
        else:
            return None, None


# Example usage and testing
# Пример использования и тестирования
if __name__ == "__main__":
    # Test the ML model class
    # Тестирование класса модели машинного обучения
    print("Тестирование модуля машинного обучения...")
    
    # Create sample data
    # Создание примерных данных
    X_sample = np.random.rand(100, 10)
    y_sample = np.random.randint(0, 3, 100)
    
    # Test individual model
    # Тестирование отдельной модели
    model = MLModel('logistic_regression', C=0.1)
    results = model.train(X_sample, y_sample)
    
    print("Тестирование завершено успешно!")