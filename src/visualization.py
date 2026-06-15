# src/visualization.py
import cv2
from src import config

class VideoAnnotator:
    """
    Dibuja anotaciones en el video (pelota, jugadores, trayectoria, tiempo).
    """
    
    @staticmethod
    def draw_ball(frame, center, speed=None, color=(0, 0, 255)):
        """Dibuja la pelota y su velocidad."""
        if center is None:
            return frame
        
        x, y = int(center[0]), int(center[1])
        
        cv2.circle(frame, (x, y), 10, color, 2)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        
        if speed is not None:
            label = f"{speed:.1f} km/h"
            cv2.putText(frame, label, (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       config.COLOR_SPEED, 2)
        
        return frame
    
    @staticmethod
    def draw_player(frame, bbox, player_id=None, confidence=None):
        """Dibuja el bounding box del jugador."""
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_PERSON, 2)
        
        label = "Jugador"
        if player_id is not None:
            label += f" #{player_id}"
        if confidence is not None:
            label += f" {confidence:.0%}"
        
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_PERSON, 2)
        
        return frame
    
    @staticmethod
    def draw_trajectory(frame, history, color=(255, 200, 0)):
        """Dibuja la línea de trayectoria de la pelota."""
        if len(history) < 2:
            return frame
        
        for i in range(1, len(history)):
            pt1 = (int(history[i-1][0]), int(history[i-1][1]))
            pt2 = (int(history[i][0]), int(history[i][1]))
            cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
    
    @staticmethod
    def draw_timestamp(frame, timestamp, fps=30):
        """Dibuja el tiempo transcurrido en la esquina superior derecha."""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"Tiempo: {minutes:02d}:{seconds:02d}"
        
        cv2.putText(frame, time_str, (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame