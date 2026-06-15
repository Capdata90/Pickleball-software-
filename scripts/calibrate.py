# scripts/01_calibrate.py
import cv2
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para poder importar src
sys.path.append(str(Path(__file__).parent.parent))

from src.calibration import CourtCalibrator
from src import config

def main():
    print("PICKLELAB - Calibración de Cámara")
    print("=" * 50)
    
    # Buscar video de calibración
    video_path = config.RAW_VIDEOS_DIR / "calibracion.mp4"
    
    if not video_path.exists():
        print(f"No existe: {video_path}")
        print("Coloca tu video de calibración en: data/raw/calibracion.mp4")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Leer primer frame
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el video")
        cap.release()
        return
    
    cap.release()
    
    print("Video cargado correctamente")
    print(f"Resolución: {frame.shape[1]}x{frame.shape[0]}")
    
    # Crear calibrador
    calibrator = CourtCalibrator()
    
    # Calibración interactiva
    print("\nSigue las instrucciones en la ventana de OpenCV.")
    success = calibrator.interactive_calibration(frame.copy())
    
    if success:
        # Guardar calibración
        calibrator.save_calibration()
        print("\nCalibración completada y guardada")
        print(f"Archivo: {config.HOMOGRAPHY_FILE}")
    else:
        print("\nCalibración cancelada")

if __name__ == "__main__":
    main()