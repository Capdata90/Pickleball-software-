# src/detection.py
import cv2
import numpy as np
from ultralytics import YOLO
from src import config

class BallAndPlayerDetector:
    def __init__(self):
        # Modelo para detectar jugadores (personas) - clase 0 = person
        self.model_persons = YOLO("yolo11n.pt")
        
        # Modelo local entrenado para pelota de pickleball
        self.model_ball = YOLO("models/pickleball_ball_trained.pt")
        
        # Configuración más agresiva para detectar la pelota
        self.ball_confidence = 0.15
        self.min_ball_size = 6
        self.max_ball_size = 60
        self.min_aspect_ratio = 0.6
        self.max_aspect_ratio = 1.4
        
        # Filtro de consistencia temporal
        self.ball_detection_history = []
        self.min_consecutive_detections = 2
        
        print("✅ Detector inicializado:")
        print("   - YOLO11n (personas)")
        print("   - Modelo local (pelota) con imgsz=320")

    def _is_consistent_detection(self, cx, cy):
        """
        Verifica si la detección es consistente con las últimas detecciones.
        """
        self.ball_detection_history.append((cx, cy))
        
        # Mantener solo las últimas 5 detecciones
        if len(self.ball_detection_history) > 5:
            self.ball_detection_history.pop(0)
        
        # Si no hay suficientes detecciones previas, aceptar
        if len(self.ball_detection_history) < self.min_consecutive_detections:
            return True
        
        # Calcular distancia promedio con las detecciones anteriores
        distances = []
        for prev_cx, prev_cy in self.ball_detection_history[:-1]:
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Si la distancia promedio es menor a 300 píxeles, es consistente
        return avg_distance < 300

    def detect_fused(self, frame):
        """
        Detecta jugadores con YOLO y pelota con modelo local entrenado.
        Retorna: ball_center, detections_dict, method_used
        """
        detections = {"persons": [], "ball": None}
        ball_center = None
        method = "fused"
        
        frame_height, frame_width = frame.shape[:2]
        
        # ==========================================
        # 1. DETECTAR JUGADORES CON YOLO
        # ==========================================
        results_persons = self.model_persons(frame, classes=[0], verbose=False)
        
        for result in results_persons:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Calcular centro del jugador
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # VALIDACIÓN: Filtrar espectadores y árbitro
                # - cy < 15% superior: gradas
                # - cx < 8% o cx > 92%: laterales extremos (árbitro/banderines)
                if cy < frame_height * 0.15:
                    continue
                if cx < frame_width * 0.08 or cx > frame_width * 0.92:
                    continue
                
                detections["persons"].append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })
        
        # ==========================================
        # 2. DETECTAR PELOTA CON MODELO LOCAL (imgsz=320 para velocidad)
        # ==========================================
        results_ball = self.model_ball(
            frame, 
            verbose=False, 
            conf=self.ball_confidence,
            imgsz=320
        )
        
        best_ball = None
        best_confidence = 0
        
        for result in results_ball:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Calcular centro y tamaño
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                width = x2 - x1
                height = y2 - y1
                
                # VALIDACIÓN 1: Filtrar logo de la esquina inferior derecha
                if cx > frame_width * 0.75 and cy > frame_height * 0.75:
                    continue
                
                # VALIDACIÓN 2: Filtrar zona de espectadores (15% superior)
                if cy < frame_height * 0.15:
                    continue
                
                # VALIDACIÓN 3: Tamaño razonable de pelota
                if width < self.min_ball_size or width > self.max_ball_size:
                    continue
                if height < self.min_ball_size or height > self.max_ball_size:
                    continue
                
                # VALIDACIÓN 4: Relación de aspecto (aproximadamente cuadrada)
                aspect_ratio = width / max(height, 1)
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # VALIDACIÓN 5: Consistencia temporal
                if not self._is_consistent_detection(cx, cy):
                    continue
                
                # Si pasa todas las validaciones y tiene mejor confianza, guardar
                if conf > best_confidence:
                    best_confidence = conf
                    best_ball = {
                        "center": (cx, cy),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf
                    }
        
        # Usar la mejor detección válida
        if best_ball:
            ball_center = best_ball["center"]
            detections["ball"] = best_ball
            method = "local_model"
        
        return ball_center, detections, method