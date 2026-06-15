# scripts/02_analyze_video.py
import cv2
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.calibration import CourtCalibrator
from src.detection import BallAndPlayerDetector
from src.tracking import KalmanBallTracker
from src.analysis import SpeedAnalyzer
from src.visualization import VideoAnnotator

def main():
    print("PICKLELAB - Análisis de Partido")
    print("=" * 50)
    
    # 1. Cargar calibración
    calibrator = CourtCalibrator()
    try:
        calibrator.load_calibration()
        print("Calibración cargada correctamente.")
    except FileNotFoundError:
        print("Error: No se encontró la calibración. Ejecuta primero 01_calibrate.py")
        return
    
    # 2. Inicializar módulos
    detector = BallAndPlayerDetector()
    tracker = KalmanBallTracker()
    annotator = VideoAnnotator()
    speed_analyzer = SpeedAnalyzer(calibrator.homography_matrix, fps=30)
    
    # 3. Abrir video de entrada
    video_path = RAW_VIDEOS_DIR / "partido.mp4"
    if not video_path.exists():
        print(f"Error: No existe el video en {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps | Frames: {total_frames}")
    
    # 4. Configurar video de salida
    output_path = PROCESSED_VIDEOS_DIR / "partido_anotado.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("\nProcesando video...")
    frame_num = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps
        
        # 5. Pipeline de procesamiento
        # Detección
        ball_center, detections, method = detector.detect_fused(frame)
        
        # Tracking
        if ball_center:
            ball_smooth = tracker.update(ball_center)
        else:
            ball_smooth = tracker.update(None)
        
        # Análisis de velocidad
        speed = 0.0
        if ball_smooth:
            speed_analyzer.add_position(ball_smooth[0], ball_smooth[1], timestamp)
            speed = speed_analyzer.calculate_speed()
        
        # 6. Visualización
        frame = annotator.draw_ball(frame, ball_smooth, speed)
        frame = annotator.draw_trajectory(frame, tracker.history)
        
        for i, person in enumerate(detections["persons"]):
            frame = annotator.draw_player(frame, person["bbox"], 
                                         player_id=i, 
                                         confidence=person["confidence"])
        
        frame = annotator.draw_timestamp(frame, timestamp, fps)
        
        # Guardar frame
        out.write(frame)
        
        # Mostrar progreso
        frame_num += 1
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            fps_real = 100 / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_num}/{total_frames} ({fps_real:.1f} fps reales)")
    
    # 7. Limpieza
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nAnálisis completado.")
    print(f"Video guardado en: {output_path}")
    print(f"Tiempo total: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()