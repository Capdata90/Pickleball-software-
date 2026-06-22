# scripts/01_calibrate.py
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.calibration import CourtCalibrator
from src import config

def main():
    print("PICKLELAB - Calibración de Cámara")
    print("=" * 50)
    
    # Video de calibración
    video_path = config.RAW_VIDEOS_DIR / "partido1.mp4"
    
    if not video_path.exists():
        print(f"❌ No existe: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f" Error: No se pudo abrir el video")
        return

    # Saltar al segundo 5 para evitar el frame negro
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)

    ret, frame = cap.read()
    
    # ️ ¡AQUÍ HABÍA UN 'return' QUE MATABA LA FUNCIÓN! Ya no está.

    if not ret:
        print("❌ Error: No se pudo leer el frame")
        cap.release()
        return

    cap.release()

    print("✅ Video cargado correctamente")
    print(f"Resolución: {frame.shape[1]}x{frame.shape[0]}")

    # Crear calibrador
    calibrator = CourtCalibrator()

    print("\n Sigue las instrucciones en la ventana de OpenCV.")
    print("   Haz click en las 4 esquinas de la cancha.")
    print("   Presiona 's' para guardar, 'q' para salir")
    
    success = calibrator.interactive_calibration(frame.copy())

    if success:
        calibrator.save_calibration()
        print("\n✅ Calibración completada y guardada")
        print(f"Archivo: {config.HOMOGRAPHY_FILE}")
    else:
        print("\n❌ Calibración cancelada")

# ⚠️ ¡AQUÍ HABÍA UN ERROR DE SINTAXIS! (faltaban los guiones bajos)
if __name__ == "__main__":
    main()