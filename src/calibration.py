# src/calibration.py
import cv2
import numpy as np
import json
from src import config

class CourtCalibrator:
    """
    Calcula la matriz de homografía para convertir
    píxeles a metros reales de la cancha.
    """
    
    def __init__(self):
        self.homography_matrix = None
        self.court_dimensions = {
            "width": config.COURT_WIDTH_METERS,
            "length": config.COURT_LENGTH_METERS,
            "kitchen": config.KITCHEN_LENGTH_METERS
        }
    
    def set_pixel_points(self, points):
        """
        points: lista de 4 tuplas (x, y) en píxeles.
        Orden: [esquina_inf_izq, esquina_inf_der, 
                esquina_sup_der, esquina_sup_izq]
        """
        if len(points) != 4:
            raise ValueError("Se necesitan exactamente 4 puntos")
        
        self.pixel_points = np.array(points, dtype=np.float32)
        
        self.real_points = np.array([
            [0, 0],
            [self.court_dimensions["width"], 0],
            [self.court_dimensions["width"], self.court_dimensions["length"]],
            [0, self.court_dimensions["length"]]
        ], dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(
            self.pixel_points, self.real_points
        )
        
        print("✅ Matriz de homografía calculada.")
        return self.homography_matrix
    
    def save_calibration(self, filepath=None):
        if filepath is None:
            filepath = config.HOMOGRAPHY_FILE
        
        if self.homography_matrix is None:
            raise ValueError("Primero debes calcular la homografía")
        
        np.save(filepath, self.homography_matrix)
        
        json_file = filepath.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(self.court_dimensions, f, indent=2)
        
        print(f"✅ Calibración guardada en: {filepath}")
    
    def load_calibration(self, filepath=None):
        if filepath is None:
            filepath = config.HOMOGRAPHY_FILE
        
        if not filepath.exists():
            raise FileNotFoundError(f"No existe: {filepath}")
        
        self.homography_matrix = np.load(filepath)
        
        json_file = filepath.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.court_dimensions = json.load(f)
        
        print(f"✅ Calibración cargada desde: {filepath}")
        return self.homography_matrix
    
    def pixels_to_meters(self, pixel_x, pixel_y):
        if self.homography_matrix is None:
            raise ValueError("Primero carga o calcula la homografía")
        
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        meters = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return meters[0][0]
    
    def interactive_calibration(self, image):
        print("📍 Haz click en las 4 esquinas de la cancha:")
        print("   1. Esquina inferior izquierda")
        print("   2. Esquina inferior derecha")
        print("   3. Esquina superior derecha")
        print("   4. Esquina superior izquierda")
        print("   Presiona 's' para guardar, 'q' para salir")
        
        points = []
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, str(len(points)), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Calibración", image)
                print(f"Punto {len(points)}: ({x}, {y})")
        
        cv2.imshow("Calibración", image)
        cv2.setMouseCallback("Calibración", click_event)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(points) == 4:
                cv2.destroyAllWindows()
                self.set_pixel_points(points)
                return True
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        return False