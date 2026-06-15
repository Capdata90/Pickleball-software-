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
            
            # Filtrar por tamaño (pelota de pickleball en video 1080p)
            if 8 < area < 150:
                # Verificar circularidad
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.55:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    
                    # Filtrar por