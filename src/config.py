# src/config.py
"""
Configuración global del proyecto PickleLab
"""
import os
from pathlib import Path

# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Subdirectorios
RAW_VIDEOS_DIR = DATA_DIR / "raw"
PROCESSED_VIDEOS_DIR = DATA_DIR / "processed"
CALIBRATION_DIR = DATA_DIR / "calibration"

# Archivos importantes
HOMOGRAPHY_FILE = CALIBRATION_DIR / "homography_matrix.npy"
DEFAULT_MODEL = MODELS_DIR / "yolov8n.pt"
CUSTOM_MODEL = MODELS_DIR / "pickleball_ball_trained.pt"

# Parámetros de video
VIDEO_FPS = 30

# Umbrales de detección
BALL_CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.35

# Dimensiones oficiales de la cancha (metros)
COURT_WIDTH_METERS = 6.10
COURT_LENGTH_METERS = 13.41
KITCHEN_LENGTH_METERS = 2.13

# Colores para visualización (BGR)
COLOR_BALL = (0, 0, 255)          # Rojo
COLOR_PERSON = (0, 255, 0)        # Verde
COLOR_TRAJECTORY = (255, 200, 0)  # Cyan
COLOR_SPEED = (0, 255, 255)       # Amarillo

# Rango HSV para pelota amarilla (Ajustado para tu video corto)
HSV_YELLOW_MIN = [20, 140, 160]
HSV_YELLOW_MAX = [36, 255, 255]

# Crear directorios si no existen
for directory in [RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, 
                  CALIBRATION_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("✅ Configuración PickleLab cargada")
print(f"📁 Proyecto en: {PROJECT_ROOT}")