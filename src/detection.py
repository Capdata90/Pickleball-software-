# src/detection.py
import cv2
import numpy as np
from ultralytics import YOLO
from src import config

class BallAndPlayerDetector:
    """
    Detecta pelotas de pickleball y jugadores.
    Usa YOLO + detección por color HSV.
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.DEFAULT_MODEL
        
        self.model = YOLO(model_path)
        self.hsv_min = np.array(config.HSV_YELLOW_MIN)
        self.hsv_max = np.array(config.HSV_YELLOW_MAX)
    
    def detect_color_ball(self, frame):
        """
        Detecta pelota amarilla usando color HSV.
        Retorna: (cx, cy) o None, y el área.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        
        # Limpieza morfológica
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Buscar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por tamaño (ajustar según resolución)
            if 15 < area < 400:
                # Verificar circularidad
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.55:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    return (cx, cy), area
        
        return None, 0
    
    def detect_with_yolo(self, frame):
        """
        Detecta con YOLO (personas y pelotas).
        Retorna: dict con boxes, clases y confianzas.
        """
        results = self.model(frame, verbose=False)
        
        detections = {
            "balls": [],
            "persons": []
        }
        
        if results[0].boxes is None:
            return detections
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            class_name = self.model.names[cls]
            
            if class_name == "sports ball" and conf > config.BALL_CONFIDENCE_THRESHOLD:
                detections["balls"].append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "confidence": conf
                })
            
            elif class_name == "person" and conf > config.PERSON_CONFIDENCE_THRESHOLD:
                # Verificar proporción (persona es más alta que ancha)
                height = y2 - y1
                width = x2 - x1
                aspect_ratio = height / width if width > 0 else 0
                
                if aspect_ratio > 1.2:  # Persona vertical
                    detections["persons"].append({
                        "bbox": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "confidence": conf
                    })
        
        return detections
    
    def detect_fused(self, frame):
        """
        Fusión de YOLO + Color.
        Prioriza YOLO, usa color como respaldo.
        """
        # Detección YOLO
        yolo_dets = self.detect_with_yolo(frame)
        
        # Detección por color
        color_ball, area = self.detect_color_ball(frame)
        
        # Fusión
        if yolo_dets["balls"]:
            # Usar detección de YOLO si existe
            best_ball = max(yolo_dets["balls"], key=lambda x: x["confidence"])
            return best_ball["center"], yolo_dets, "yolo"
        elif color_ball:
            # Usar detección por color
            return color_ball, yolo_dets, "color"
        else:
            return None, yolo_dets, None