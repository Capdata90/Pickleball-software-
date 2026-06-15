# src/analysis.py
import cv2
import numpy as np
from collections import deque

class SpeedAnalyzer:
    """
    Calcula velocidades reales (km/h) y estadísticas de trayectoria
    utilizando la matriz de homografía para convertir píxeles a metros.
    """
    
    def __init__(self, homography_matrix, fps=30, window_size=10):
        self.H = homography_matrix
        self.fps = fps
        self.window_size = window_size
        # Historial de posiciones en metros y tiempo
        self.position_history = deque(maxlen=window_size)
    
    def add_position(self, pixel_x, pixel_y, timestamp):
        """
        Agrega una posición en píxeles y su tiempo,
        convirtiéndola automáticamente a metros.
        """
        if self.H is None:
            raise ValueError("La matriz de homografía no está cargada")
        
        # Convertir píxeles a metros
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        meters = cv2.perspectiveTransform(point, self.H)[0][0]
        
        self.position_history.append({
            "time": timestamp,
            "meters": meters,
            "pixels": (pixel_x, pixel_y)
        })
    
    def calculate_speed(self):
        """
        Calcula la velocidad actual usando una ventana deslizante.
        Retorna: velocidad en km/h (float).
        """
        if len(self.position_history) < 4:
            return 0.0
        
        # Usar el primer y último punto de la ventana actual
        first = self.position_history[0]
        last = self.position_history[-1]
        
        dt = last["time"] - first["time"]
        if dt <= 0:
            return 0.0
        
        # Distancia euclidiana en metros
        dist = np.linalg.norm(last["meters"] - first["meters"])
        
        # Velocidad en m/s y luego en km/h
        speed_m_s = dist / dt
        speed_km_h = speed_m_s * 3.6
        
        return speed_km_h
    
    def get_trajectory_stats(self):
        """
        Calcula estadísticas generales de la trayectoria actual en el buffer.
        """
        if len(self.position_history) < 2:
            return None
        
        positions = np.array([p["meters"] for p in self.position_history])
        
        # Distancia total recorrida en la ventana
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        # Desplazamiento máximo desde el primer punto
        max_displacement = np.max(np.linalg.norm(positions - positions[0], axis=1))
        
        return {
            "total_distance_m": total_distance,
            "max_displacement_m": max_displacement,
            "avg_position_m": np.mean(positions, axis=0)
        }
    
    def clear_history(self):
        """Limpia el historial, útil cuando la pelota sale de la cancha."""
        self.position_history.clear()