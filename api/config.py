"""
Configuration settings for FinTech Risk Analyzer Flask API
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///risk_analyzer.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ML Model settings
    MODEL_PATH = Path('models/risk_analyzer.pkl')
    SCALER_PATH = Path('models/scaler.pkl')
    
    # API settings
    API_TITLE = 'FinTech Risk Analyzer API'
    API_VERSION = '1.0.0'
    API_DESCRIPTION = 'API for analyzing financial transaction risk using ML and rule-based approaches'
    
    # CORS settings
    CORS_ORIGINS = [
        'http://localhost:3000',  # React dev server
        'http://localhost:4200',  # Angular dev server
        'http://localhost:8080',  # Vue dev server
    ]
    
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URL = "memory://"
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/api.log'
    
    # Security
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Feature flags
    ENABLE_ML_PREDICTIONS = True
    ENABLE_RULE_BASED_ANALYSIS = True
    ENABLE_ANOMALY_DETECTION = True
    ENABLE_REAL_TIME_ALERTS = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Development database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///risk_analyzer_dev.db'
    
    # Development logging
    LOG_LEVEL = 'DEBUG'
    
    # Development CORS
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:4200', 'http://localhost:8080']

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    
    # Test database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///risk_analyzer_test.db'
    
    # Test logging
    LOG_LEVEL = 'DEBUG'
    
    # Disable ML for testing
    ENABLE_ML_PREDICTIONS = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production database (use environment variable)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Production CORS (restrict to actual domains)
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])
