# src/tracking.py
import cv2
import numpy as np
from collections import deque

class KalmanBallTracker:
    """
    Filtro de Kalman para suavizar la trayectoria de la pelota
    y predecir su posición cuando se pierde la detección.
    Estado: [x, y, vx, vy] (posición y velocidad en píxeles)
    """
    
    def __init__(self, process_noise=1e-2, measurement_noise=5.0):
        # 4 variables de estado (x, y, vx, vy), 2 variables medidas (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Matriz de transición (modelo de movimiento constante)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Matriz de medición (solo medimos x, y)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Ruido del proceso (qué tanto confiamos en nuestro modelo de movimiento)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Ruido de la medición (qué tanto confiamos en la detección de la cámara)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Covarianza del error inicial
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
        self.history = deque(maxlen=30)  # Guardar los últimos 30 puntos
    
    def update(self, measurement=None):
        """
        Actualiza el filtro de Kalman.
        measurement: tupla (x, y) en píxeles, o None si no hay detección.
        Retorna: (x, y) suavizado o predicho.
        """
        if measurement is not None:
            meas = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
            
            if not self.initialized:
                # Inicializar el estado con la primera medición
                self.kalman.statePost[:2] = meas
                self.initialized = True
            else:
                # Corregir la predicción con la nueva medición
                self.kalman.correct(meas)
        
        # Predecir la siguiente posición
        prediction = self.kalman.predict()
        
        x, y = int(prediction[0, 0]), int(prediction[1, 0])
        
        # Guardar en el historial
        self.history.append((x, y))
        
        return (x, y)
    
    def get_velocity(self, fps=30):
        """
        Calcula la velocidad actual en píxeles/segundo usando el historial.
        """
        if len(self.history) < 5:
            return (0.0, 0.0)
        
        # Usar los últimos 5 frames para calcular la velocidad
        recent = list(self.history)[-5:]
        x0, y0 = recent[0]
        x1, y1 = recent[-1]
        
        dt = len(recent) / fps
        if dt == 0:
            return (0.0, 0.0)
            
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        
        return (vx, vy)
    
    def reset(self):
        """Reinicia el tracker para una nueva pelota o secuencia."""
        self.initialized = False
        self.history.clear()
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)