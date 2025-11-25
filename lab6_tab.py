"""
Lab 6 Tab Module - Machine Learning Interface
Модуль вкладки для лабораторной работы 6 - Интерфейс машинного обучения
وحدة تبويب المختبر 6 - واجهة التعلم الآلي
"""

import sys
import os
import pandas as pd
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
                               QComboBox, QLineEdit, QProgressBar, QFileDialog, 
                               QMessageBox, QGroupBox, QSplitter)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont

from model_trainer import ModelTrainer


class TrainingThread(QThread):
    """
    Training thread to prevent GUI freezing
    Поток обучения для предотвращения зависания GUI
    ثريد التدريب لمنع تجمد الواجهة
    """
    
    finished = Signal(bool)
    progress = Signal(str)
    results_ready = Signal(dict)
    
    def __init__(self, csv_path: str):
        """
        Initialize training thread
        Инициализация потока обучения
        تهيئة ثريد التدريب
        """
        super().__init__()
        self.csv_path = csv_path
        self.trainer = None
        self.results = {}
    
    def run(self):
        """
        Main thread execution
        Основное выполнение потока
        التنفيذ الرئيسي للثريد
        """
        try:
            self.progress.emit("Инициализация тренера моделей...")
            self.trainer = ModelTrainer()
            
            self.progress.emit("Загрузка и подготовка данных...")
            train_df, val_df, demo_df = self.trainer.load_and_prepare_data(self.csv_path)
            
            self.progress.emit("Предобработка текстовых данных...")
            X_train, X_val, X_demo, y_train, y_val, y_demo = self.trainer.preprocess_text_data(
                train_df, val_df, demo_df
            )
            
            self.progress.emit("Обучение начальных моделей...")
            initial_results = self.trainer.train_initial_models(X_train, y_train, X_val, y_val)
            
            self.progress.emit("Настройка гиперпараметров...")
            tuning_results = self.trainer.hyperparameter_tuning(X_train, y_train, X_val, y_val)
            
            self.progress.emit("Тестирование на демонстрационных данных...")
            demo_results = self.trainer.test_on_demo_data(X_demo, y_demo)
            
            self.progress.emit("Сохранение лучшей модели...")
            self.trainer.save_best_model()
            
            self.progress.emit("Генерация анализа...")
            analysis = self.trainer.generate_hyperparameter_analysis()
            
            self.results = {
                'success': True,
                'initial_results': initial_results,
                'tuning_results': tuning_results,
                'demo_results': demo_results,
                'analysis': analysis,
                'trainer': self.trainer
            }
            
            self.finished.emit(True)
            
        except Exception as e:
            self.progress.emit(f"Ошибка: {str(e)}")
            self.results = {'success': False, 'error': str(e)}
            self.finished.emit(False)


class Lab6Tab(QWidget):
    """
    Lab 6 tab for machine learning operations
    Вкладка лабораторной работы 6 для операций машинного обучения
    تبويب المختبر 6 لعمليات التعلم الآلي
    """
    
    def __init__(self):
        """
        Initialize lab 6 tab
        Инициализация вкладки лабораторной работы 6
        تهيئة تبويب المختبر 6
        """
        super().__init__()
        self.trainer = None
        self.current_model = None
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup user interface
        Настройка пользовательского интерфейса
        إعداد واجهة المستخدم
        """
        layout = QVBoxLayout(self)
        
        
        # Splitter для разделения интерфейса
        # مقسم لتقسيم الواجهة
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - Управление обучением
        # اللوحة اليسرى - إدارة التدريب
        left_panel = self.create_training_panel()
        
        # Правая панель - Результаты и тестирование
        # اللوحة اليمنى - النتائج والاختبار
        right_panel = self.create_results_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
    
    def create_training_panel(self) -> QWidget:
        """
        Create training management panel
        Создание панели управления обучением
        إنشاء لوحة إدارة التدريب
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа выбора данных
        # مجموعة اختيار البيانات
        data_group = QGroupBox("Выбор данных")
        data_layout = QVBoxLayout(data_group)
        
        file_layout = QHBoxLayout()
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("Выберите CSV файл с данными вакансий...")
        browse_btn = QPushButton("Обзор")
        browse_btn.clicked.connect(self.select_data_file)
        
        file_layout.addWidget(self.data_path)
        file_layout.addWidget(browse_btn)
        data_layout.addLayout(file_layout)
        
        # Группа управления обучением
        # مجموعة إدارة التدريب
        training_group = QGroupBox("Управление обучением")
        training_layout = QVBoxLayout(training_group)
        
        self.train_btn = QPushButton("Запустить полное обучение")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 12px; }")
        
        self.stop_btn = QPushButton("Остановить обучение")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; padding: 12px; }")
        
        # Прогресс-бар
        # شريط التقدم
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        training_layout.addWidget(self.train_btn)
        training_layout.addWidget(self.stop_btn)
        training_layout.addWidget(self.progress_bar)
        
        # Группа загрузки модели
        # مجموعة تحميل النموذج
        model_group = QGroupBox("Управление моделью")
        model_layout = QVBoxLayout(model_group)
        
        load_model_btn = QPushButton("Загрузить сохраненную модель")
        load_model_btn.clicked.connect(self.load_saved_model)
        load_model_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px; }")
        
        model_layout.addWidget(load_model_btn)
        
        # Журнал выполнения
        # سجل التنفيذ
        log_group = QGroupBox("Журнал выполнения")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(400)
        self.log_text.setPlaceholderText("Здесь будет отображаться журнал выполнения...")
        
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(data_group)
        layout.addWidget(training_group)
        layout.addWidget(model_group)
        layout.addWidget(log_group)
        
        return panel
    
    def create_results_panel(self) -> QWidget:
        """
        Create results and testing panel
        Создание панели результатов и тестирования
        إنشاء لوحة النتائج والاختبار
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Группа результатов
        # مجموعة النتائج
        results_group = QGroupBox("Результаты обучения")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Результаты обучения будут отображены здесь...")
        
        results_layout.addWidget(self.results_text)
        
        # Группа тестирования
        # مجموعة الاختبار
        test_group = QGroupBox("Тестирование модели")
        test_layout = QVBoxLayout(test_group)
        
        # Поля ввода для тестирования
        # حقول الإدخال للاختبار
        input_layout = QVBoxLayout()
        
        skills_layout = QHBoxLayout()
        skills_layout.addWidget(QLabel("Навыки:"))
        self.skills_input = QLineEdit()
        self.skills_input.setPlaceholderText("Введите навыки для классификации...")
        skills_layout.addWidget(self.skills_input)
        
        input_layout.addLayout(skills_layout)
        
        # Кнопка тестирования
        # زر الاختبار
        test_btn_layout = QHBoxLayout()
        self.test_btn = QPushButton("Классифицировать вакансию")
        self.test_btn.clicked.connect(self.test_model)
        self.test_btn.setEnabled(False)
        self.test_btn.setStyleSheet("QPushButton { background-color: #9b59b6; color: white; font-weight: bold; padding: 10px; }")
        
        test_btn_layout.addWidget(self.test_btn)
        
        # Результаты тестирования
        # نتائج الاختبار
        self.test_result = QTextEdit()
        self.test_result.setMaximumHeight(150)
        self.test_result.setPlaceholderText("Результаты классификации будут отображены здесь...")
        
        test_layout.addLayout(input_layout)
        test_layout.addLayout(test_btn_layout)
        test_layout.addWidget(self.test_result)
        
        layout.addWidget(results_group, 3)
        layout.addWidget(test_group, 3)
        # layout.addWidget(table_group)
        
        return panel
    
    def select_data_file(self):
        """
        Select data file for training
        Выбор файла данных для обучения
        اختيار ملف البيانات للتدريب
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите CSV файл с данными вакансий", 
            "", 
            "CSV Files (*.csv)"
        )
        
        if file_path:
            self.data_path.setText(file_path)
            self.log_text.append(f"Выбран файл данных: {file_path}")
    
    def start_training(self):
        """
        Start model training process
        Запуск процесса обучения модели
        بدء عملية تدريب النموذج
        """
        data_path = self.data_path.text()
        
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите правильный файл данных")
            return
        
        self.log_text.clear()
        self.results_text.clear()
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.training_thread = TrainingThread(data_path)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()
        
        self.log_text.append("Начало процесса обучения...")
    
    def stop_training(self):
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            # احذف هذا السطر:
            # self.training_thread.wait()
            
            self.log_text.append("Обучение остановлено пользователем")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
    
    def update_progress(self, message: str):
        """
        Update progress log
        Обновление журнала прогресса
        تحديث سجل التقدم
        """
        self.log_text.append(message)
    
  
    def on_training_finished(self, success: bool):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log_text.append("Обучение завершено успешно!")
            self.test_btn.setEnabled(True)
        
            if hasattr(self, 'training_thread') and hasattr(self.training_thread, 'trainer'):
                self.trainer = self.training_thread.trainer
            # أضف هذين السطرين:
            if hasattr(self, 'training_thread') and hasattr(self.training_thread, 'results'):
                self.display_training_results(self.training_thread.results)
    
    def on_results_ready(self, results: dict):
        if results.get('success'):
            self.trainer = results.get('trainer')
            self.display_training_results(results)
            self.test_btn.setEnabled(True)
            # أضف هذا السطر:
            QMessageBox.information(self, "Успех", "Обучение завершено успешно!")
            
    def display_training_results(self, results: dict):
        """
        Display training results
        Отображение результатов обучения
        عرض نتائج التدريب
        """
        report = "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ\n"
        report += "=" * 50 + "\n\n"
        
        # Initial results
        # Начальные результаты
        # النتائج الأولية
        initial_results = results.get('initial_results')
        if initial_results is not None:
            report += "НАЧАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n"
            report += "-" * 30 + "\n"
            report += initial_results.to_string() + "\n\n"
        
        # Tuning results
        # Результаты настройки
        # نتائج الضبط
        tuning_results = results.get('tuning_results')
        if tuning_results is not None:
            report += "РЕЗУЛЬТАТЫ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ:\n"
            report += "-" * 45 + "\n"
            report += tuning_results.to_string() + "\n\n"
        
        # Demo results
        # Демонстрационные результаты
        # نتائج العرض التوضيحي
        demo_results = results.get('demo_results')
        if demo_results is not None:
            report += "РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:\n"
            report += "-" * 25 + "\n"
            report += f"Модель: {demo_results.get('model_name', 'N/A')}\n"
            report += f"Точность: {demo_results.get('demo_accuracy', 0):.4f}\n\n"
        
        # Analysis
        # Анализ
        # التحليل
        analysis = results.get('analysis')
        if analysis:
            report += "АНАЛИЗ ВЛИЯНИЯ ГИПЕРПАРАМЕТРОВ:\n"
            report += "-" * 40 + "\n"
            report += analysis + "\n"
        
        self.results_text.setText(report)
        
        # Update results table if demo data available
        # Обновление таблицы результатов, если доступны демонстрационные данные
        # تحديث جدول النتائج إذا كانت بيانات العرض التوضيحي متاحة
        self.update_demo_results_table(results)
    
    def update_demo_results_table(self, results: dict):
        """
        Update demonstration results table
        Обновление таблицы демонстрационных результатов
        تحديث جدول نتائج العرض التوضيحي
        """
        # This would be implemented to show actual demo predictions
        # Это будет реализовано для показа фактических демонстрационных прогнозов
        # سيتم تنفيذ هذا لعرض التوقعات التوضيحية الفعلية
        pass
    
    def load_saved_model(self):
        """
        Load saved model
        Загрузка сохраненной модели
        تحميل النموذج المحفوظ
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            "",
            "Model Files (*.joblib)"
        )
        
        if file_path:
            try:
                # Here you would implement model loading logic
                # Здесь вы реализуете логику загрузки модели
                # هنا ستقوم بتنفيذ منطق تحميل النموذج
                self.log_text.append(f"Модель загружена: {file_path}")
                self.test_btn.setEnabled(True)
                QMessageBox.information(self, "Успех", "Модель успешно загружена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки модели: {str(e)}")
    
    def test_model(self):
        """
        Test model with user input
        Тестирование модели с пользовательским вводом
        اختبار النموذج بإدخال المستخدم
        """
        skills = self.skills_input.text().strip()
        
        if not skills:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите навыки для классификации")
            return
        
        if self.trainer is None or not hasattr(self.trainer, 'text_processor'):
            QMessageBox.warning(self, "Ошибка", "Сначала выполните обучение модели")
            return
        # if self.trainer is None:
        #     QMessageBox.warning(self, "Ошибка", "Модель не обучена или не загружена")
        #     return
        
        try:
            # Preprocess input
            # Предобработка ввода
            # معالجة الإدخال
            processed_skills = self.trainer.text_processor.preprocess_text(skills)
            
            # Transform to TF-IDF
            # Преобразование в TF-IDF
            # التحويل إلى TF-IDF
            features = self.trainer.text_processor.transform_tfidf([processed_skills])
            
            # Make prediction
            # Выполнение прогноза
            # إجراء التوقع
            prediction = self.trainer.best_model.predict(features)[0]
            
            # Get probabilities if available
            # Получение вероятностей, если доступно
            # الحصول على الاحتمالات إذا كانت متاحة
            probabilities = self.trainer.best_model.predict_proba(features)
            
            # Display results
            # Отображение результатов
            # عرض النتائج
            result_text = f"Результаты классификации:\n"
            result_text += f"Входные навыки: {skills}\n"
            result_text += f"Предсказанная категория: {prediction}\n"
            
            if len(probabilities) > 0:
                result_text += f"\nВероятности:\n"
                classes = self.trainer.best_model.model.classes_
                for i, prob in enumerate(probabilities[0]):
                    result_text += f"  {classes[i]}: {prob:.4f}\n"
            
            self.test_result.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка классификации: {str(e)}")


# Example usage
# Пример использования
# مثال للاستخدام
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = Lab6Tab()
    window.show()
    sys.exit(app.exec())