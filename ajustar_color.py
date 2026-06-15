# ajustar_color.py
import cv2
import numpy as np
from tkinter import Tk, filedialog

def encontrar_color_pelota():
    print("=== AJUSTAR COLOR DE PELOTA ===")
    
    # Cargar video corto directamente
    video_path = "data/raw/partido_corto.mp4"
    print(f"Video: {video_path}")
    
    # Cargar frame
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_path}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # Frame 200 (ajusta si necesitas)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("No se pudo leer el frame")
        return
    
    print(f"Frame cargado: {frame.shape}")
    
    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def nothing(x):
        pass
    
    cv2.namedWindow("Ajustar HSV")
    cv2.createTrackbar("H min", "Ajustar HSV", 20, 180, nothing)
    cv2.createTrackbar("S min", "Ajustar HSV", 100, 255, nothing)
    cv2.createTrackbar("V min", "Ajustar HSV", 150, 255, nothing)
    cv2.createTrackbar("H max", "Ajustar HSV", 35, 180, nothing)
    cv2.createTrackbar("S max", "Ajustar HSV", 255, 255, nothing)
    cv2.createTrackbar("V max", "Ajustar HSV", 255, 255, nothing)
    
    while True:
        h_min = cv2.getTrackbarPos("H min", "Ajustar HSV")
        s_min = cv2.getTrackbarPos("S min", "Ajustar HSV")
        v_min = cv2.getTrackbarPos("V min", "Ajustar HSV")
        h_max = cv2.getTrackbarPos("H max", "Ajustar HSV")
        s_max = cv2.getTrackbarPos("S max", "Ajustar HSV")
        v_max = cv2.getTrackbarPos("V max", "Ajustar HSV")
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Mascara (blanco = detectado)", mask)
        cv2.imshow("Resultado", result)
        
        print(f"\rValores: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]", end="")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\n\nValores finales para config.py:")
    print(f"HSV_YELLOW_MIN = [{h_min}, {s_min}, {v_min}]")
    print(f"HSV_YELLOW_MAX = [{h_max}, {s_max}, {v_max}]")

if __name__ == "__main__":
    encontrar_color_pelota()
