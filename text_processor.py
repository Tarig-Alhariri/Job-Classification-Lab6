"""
Модуль обработки текста для лабораторной работы 6
وحدة معالجة النصوص للعمل المختبري 6
Этот модуль обрабатывает предобработку текста, преобразование TF-IDF и разделение данных
هذه الوحدة تعالج المعالجة المسبقة للنصوص، تحويل TF-IDF، وتقسيم البيانات
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Автоматическая загрузка данных NLTK
try:
    nltk.data.find('corpora/stopwords')
    print("stopwords")
except LookupError:
    print("stopwords..")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
    print("punkt ")
except LookupError:
    print("punkt...")
    nltk.download('punkt')


class TextProcessor:
    """
    Text preprocessing and transformation class
    Класс для предобработки и преобразования текста
    فئة لمعالجة وتحويل النصوص
    
    Handles text cleaning, stopword removal, stemming, and TF-IDF transformation
    Обрабатывает очистку текста, удаление стоп-слов, стемминг и преобразование TF-IDF
    تعالج تنظيف النص، إزالة الكلمات غير الضرورية، التصريف، وتحويل TF-IDF
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize text processor
        Инициализация процессора текста
        تهيئة معالج النصوص
        
        Args:
            language (str): Language for stopwords (english/russian/arabic)
            language (str): Язык для стоп-слов (english/russian/arabic)
            language (str): اللغة للكلمات غير الضرورية (english/russian/arabic)
        """
        self.language = language
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = None
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        Очистка и предобработка текста
        تنظيف ومعالجة مسبقة للنص
        
        Args:
            text (str): Input text to clean
            text (str): Входной текст для очистки
            text (str): النص المدخل لتنظيفه
            
        Returns:
            str: Cleaned text
            str: Очищенный текст
            str: النص المنظف
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase - Приведение к нижнему регистру - التحويل إلى أحرف صغيرة
        text = text.lower()
        
        # Remove punctuation and special characters - Удаление пунктуации и специальных символов - إزالة علامات الترقيم والرموز الخاصة
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers - Удаление чисел - إزالة الأرقام
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace - Удаление лишних пробелов - إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        Удаление стоп-слов из текста
        إزالة الكلمات غير الضرورية من النص
        
        Args:
            text (str): Input text
            text (str): Входной текст
            text (str): النص المدخل
            
        Returns:
            str: Text without stopwords
            str: Текст без стоп-слов
            str: النص بدون كلمات غير ضرورية
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def apply_stemming(self, text: str) -> str:
        """
        Apply stemming to text
        Применение стемминга к тексту
        تطبيق التصريف على النص
        
        Args:
            text (str): Input text
            text (str): Входной текст
            text (str): النص المدخل
            
        Returns:
            str: Stemmed text
            str: Текст после стемминга
            str: النص بعد التصريف
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def preprocess_text(self, text: str, use_stemming: bool = True) -> str:
        """
        Полный конвейер предобработки текста
        خطوة المعالجة الكاملة للنص
        
        Args:
            text (str): Входной текст
            text (str): النص المدخل
            use_stemming (bool): Применять ли стемминг
            use_stemming (bool): هل يتم تطبيق التصريف
            
        Returns:
            str: Полностью обработанный текст
            str: النص المعالج بالكامل
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        
        if use_stemming:
            text = self.apply_stemming(text)
            
        return text
    
    def fit_tfidf(self, texts: List[str], **tfidf_params) -> None:
        """
        Fit TF-IDF vectorizer on texts
        Обучение векторизатора TF-IDF на текстах
        تدريب محول TF-IDF على النصوص
        
        Args:
            texts (List[str]): Список текстовых документов
            texts (List[str]): قائمة المستندات النصية
            **tfidf_params: Параметры для TfidfVectorizer
            **tfidf_params: معاملات لـ TfidfVectorizer
        """
        default_params = {
            'max_features': 1000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': (1, 2)
        }
        default_params.update(tfidf_params)
        
        self.vectorizer = TfidfVectorizer(**default_params)
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors
        Преобразование текстов в векторы TF-IDF
        تحويل النصوص إلى متجهات TF-IDF
        
        Args:
            texts (List[str]): List of text documents
            texts (List[str]): Список текстовых документов
            texts (List[str]): قائمة المستندات النصية
            
        Returns:
            np.ndarray: TF-IDF feature matrix
            np.ndarray: Матрица признаков TF-IDF
            np.ndarray: مصفوفة خصائص TF-IDF
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet. Call fit_tfidf first.")
        
        return self.vectorizer.transform(texts)

def split_dataset(df: pd.DataFrame, target_column: str, 
                  test_size: float = 0.2, val_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training, validation, and demonstration sets
    """
    # تقسيم بسيط بدون stratified
    train_df = df.sample(frac=0.6, random_state=random_state)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=random_state)
    demo_df = temp_df.drop(val_df.index)
    
    return train_df, val_df, demo_df


def create_job_categories_from_skills(skills_series: pd.Series) -> pd.Series:
    """
    Create job categories based on skills (automatic categorization)
    Создание категорий вакансий на основе навыков (автоматическая категоризация)
    إنشاء فئات الوظائف بناءً على المهارات (تصنيف تلقائي)
    
    Args:
        skills_series (pd.Series): Series containing job skills
        skills_series (pd.Series): Series с навыками работы
        skills_series (pd.Series): سلسلة تحتوي على مهارات الوظيفة
        
    Returns:
        pd.Series: Series with job categories
        pd.Series): Series с категориями вакансий
        pd.Series): سلسلة بفئات الوظائف
    """
    
    # Define skill patterns for different job categories
    # Определение шаблонов навыков для разных категорий вакансий
    # تعريف أنماط المهارات لفئات الوظائف المختلفة
    category_patterns = {
        'web_development': [
            'javascript', 'html', 'css', 'react', 'angular', 'vue', 'node', 
            'php', 'laravel', 'django', 'flask', 'frontend', 'backend'
        ],
        'data_science': [
            'python', 'r', 'sql', 'pandas', 'numpy', 'machine learning', 
            'data analysis', 'statistics', 'tableau', 'powerbi'
        ],
        'devops': [
            'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'ci/cd',
            'linux', 'bash', 'terraform', 'ansible'
        ],
        'mobile_development': [
            'android', 'ios', 'swift', 'kotlin', 'react native', 'flutter',
            'mobile development'
        ],
        'database': [
            'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'redis'
        ]
    }
    # return category_patterns
    
    def categorize_skills(skills_text: str) -> str:
        """Categorize job based on skills text"""
        if not isinstance(skills_text, str):
            return 'other'
        
        skills_lower = skills_text.lower()
        category_scores = {}
        
        for category, patterns in category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in skills_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'other'
    
    return skills_series.apply(categorize_skills)


# Example usage and testing
# Пример использования и тестирования
# مثال للاستخدام والاختبار
if __name__ == "__main__":
    # Test the text processor
    # Тестирование процессора текста
    # اختبار معالج النصوص
    processor = TextProcessor()
    
    sample_text = "Python developer with experience in Django and Flask frameworks. Knowledge of REST APIs."
    cleaned_text = processor.preprocess_text(sample_text)
    
    print("Оригинал:", sample_text)
    # print("النص الأصلي:", sample_text)
    print("Обработанный:", cleaned_text)
    # print("النص المعالج:", cleaned_text)