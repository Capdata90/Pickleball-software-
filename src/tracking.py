# src/tracking.py
import numpy as np
import cv2
from collections import deque

class KalmanBallTracker:
    def __init__(self):
        self.kalman = None
        self.predicted = None
        self.history = deque(maxlen=30)  # Últimos 30 frames
        self.positions = []
        self.frame_count = 0
        self.consecutive_misses = 0
        
        # Parámetros de validación
        self.max_distance = 300  # Distancia máxima desde la predicción (píxeles)
        self.max_misses = 20 
        self.reset()

    def reset(self):
        """Reiniciar el tracker"""
        self.kalman = None
        self.history.clear()
        self.positions = []
        self.frame_count = 0
        self.consecutive_misses = 0

    def _init_kalman(self, measurement):
        """Inicializar el filtro de Kalman"""
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Matriz de transición de estado [x, y, vx, vy]
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Matriz de medición
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Ruido del proceso
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Ruido de medición
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Estado inicial (CORREGIDO para evitar el error de NumPy)
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePre[0] = measurement[0]
        self.kalman.statePre[1] = measurement[1]

    def update(self, ball_center):
        """
        Actualizar el tracker con la nueva detección.
        Retorna: posición suavizada o None
        """
        self.frame_count += 1
        
        if ball_center is None:
            self.consecutive_misses += 1
            
            # Si no hay detección pero tenemos Kalman, usar predicción
            if self.kalman is not None and self.consecutive_misses < 10:
                self.predicted = self.kalman.predict()
                return (int(self.predicted[0]), int(self.predicted[1]))
            else:
                return None
        
        x, y = ball_center
        measurement = np.array([[x], [y]], np.float32)
        
        # Si es la primera detección, inicializar Kalman
        if self.kalman is None:
            self._init_kalman(measurement)
            self.history.append((x, y))
            self.positions.append((x, y, self.frame_count))
            self.consecutive_misses = 0
            return (x, y)
        
        # Predecir posición
        self.predicted = self.kalman.predict()
        pred_x, pred_y = int(self.predicted[0]), int(self.predicted[1])
        
        # Calcular distancia desde la predicción
        distance = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
        
        # VALIDACIÓN: Si la detección está muy lejos de la predicción, rechazarla
        if distance > self.max_distance:
            # print(f"️ Detección rechazada (distancia: {distance:.1f} > {self.max_distance})")
            self.consecutive_misses += 1
            return (pred_x, pred_y)
        
        # Actualizar Kalman con la medición
        self.kalman.correct(measurement)
        
        # Obtener posición corregida
        corrected = self.kalman.statePost
        corr_x, corr_y = int(corrected[0]), int(corrected[1])
        
        # Guardar en historial
        self.history.append((corr_x, corr_y))
        self.positions.append((corr_x, corr_y, self.frame_count))
        
        # Limitar el tamaño de positions
        if len(self.positions) > 60:
            self.positions.pop(0)
        
        self.consecutive_misses = 0
        
        return (corr_x, corr_y)