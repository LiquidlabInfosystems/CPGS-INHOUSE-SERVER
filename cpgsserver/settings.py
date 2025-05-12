# Developed By Tecktrio At Liquidlab Infosystems
# Project: Settings
# Version: 1.0
# Date: 2025-03-08
# Description: Setting of the project


# IMPORTING NECCESSARY MODULES
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure--zu(ol$h_h30$!vvoa)-f))s-_7rm@x(+%1$v*#4yq5(5)2m1l'
DEBUG = True
ALLOWED_HOSTS = ["*"]


# INSTALLED APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    # 'channels',
    'corsheaders',
    'cpgsapp',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework'
]


# MIDDLEWARES USED
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# ROOT URL
ROOT_URLCONF = 'cpgsserver.urls'


# FRONT END FILES AND CONFIGURATION
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [ 'build'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]



# SERVER MODE
WSGI_APPLICATION = 'cpgsserver.wsgi.application'



# DATABASE CONNECTION 
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# VALIDATION VALIDATORS
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# INTERNATIONALISATION
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# STATIC CONFIGURATIONS
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    'build/static'
]
STATIC_ROOT = 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS SETTINGS
CORS_ALLOW_ALL_ORIGINS = True
USER_VALIDATE_TOKEN = 'tokenoftrio'
IS_PI_CAMERA_SOURCE = True
CONSISTENCY_LEVEL = 5
CONFIDENCE_LEVEL = 80

# END OF PROGRAM