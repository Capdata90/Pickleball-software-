# scripts/analyze_video.py
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
        print("✅ Calibración cargada correctamente.")
    except FileNotFoundError:
        print("❌ Error: No se encontró la calibración. Ejecuta primero 01_calibrate.py")
        return
    
    # 2. Inicializar módulos
    detector = BallAndPlayerDetector()
    tracker = KalmanBallTracker()
    annotator = VideoAnnotator()
    speed_analyzer = SpeedAnalyzer(calibrator.homography_matrix, fps=30)
    
    # 3. Abrir video de entrada
    video_path = RAW_VIDEOS_DIR / "partido1.mp4"
    if not video_path.exists():
        print(f"❌ Error: No existe el video en {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps | Frames: {total_frames}")
    
    # 4. Configurar video de salida
    output_path = PROCESSED_VIDEOS_DIR / "partido_20s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("\n⏳ Procesando video (primeros 20 segundos)...")
    frame_num = 0
    start_time = time.time()
    frames_without_detection = 0
    
    # Calcular cuántos frames son 20 segundos
    max_frames = int(fps * 20)
    
    # OPTIMIZACIÓN: Procesar solo 1 de cada 3 frames para acelerar
    skip_frames = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detener el análisis si pasamos los 20 segundos
        if frame_num >= max_frames:
            print(f"⏹️ Límite de 20 segundos alcanzado ({max_frames} frames).")
            break
        
        # Saltar frames para acelerar (procesar solo 1 de cada 3)
        if frame_num % skip_frames != 0:
            # Escribir el frame sin procesar para mantener el video completo
            out.write(frame)
            frame_num += 1
            continue
            
        timestamp = frame_num / fps
        
        # 5. Pipeline de procesamiento
        # Detección
        ball_center, detections, method = detector.detect_fused(frame)
        
        # Tracking
        if ball_center:
            ball_smooth = tracker.update(ball_center)
            frames_without_detection = 0
        else:
            ball_smooth = tracker.update(None)
            frames_without_detection += 1
            
            # Si pasan 15 frames sin detectar, reiniciar el tracker
            if frames_without_detection > 15:
                tracker.reset()
                ball_smooth = None
        
        # Análisis de velocidad (se calcula pero NO se dibuja en el video)
        speed = 0.0
        if ball_smooth:
            speed_analyzer.add_position(ball_smooth[0], ball_smooth[1], timestamp)
            speed = speed_analyzer.calculate_speed()
        
        # 6. Visualización
        # Pasamos 0.0 para no dibujar la velocidad en el video (irá en el PDF después)
        frame = annotator.draw_ball(frame, ball_smooth, 0.0)
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
        if frame_num % 30 == 0:
            elapsed = time.time() - start_time
            fps_real = 30 / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_num}/{max_frames} ({fps_real:.1f} fps reales)")
    
    # 7. Limpieza
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Análisis completado.")
    print(f" Video guardado en: {output_path}")
    print(f"⏱️ Tiempo total: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()