import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# --- BLOQUE 1: INICIALIZACIÓN ---
ruta_video = "C:\\picklelab_app\\partido.mp4"
# ⚠️ RECOMENDACIÓN: entrena tu propio modelo con Roboflow (ver abajo)
modelo = YOLO("yolov8n.pt")
video = cv2.VideoCapture(ruta_video)

if not video.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
print(f"Video cargado: {ancho}x{alto} a {fps} fps")

# --- FILTRO DE KALMAN para la pelota ---
# Estado: [x, y, vx, vy]  (posición y velocidad en píxeles/frame)
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman_kalman_inicializado = False

# Buffer para calcular velocidad con ventana deslizante (menos ruido)
HISTORIAL_PELOTA = deque(maxlen=10)  # últimos 10 frames

# --- Rango HSV para pelota de pickleball amarilla ---
# Ajusta estos valores según tu video (iluminación del club)
AMARILLO_MIN = np.array([18, 80, 150])
AMARILLO_MAX = np.array([40, 255, 255])

# --- BLOQUE 2: LOOP PRINCIPAL ---
posiciones_pelota = []
posiciones_jugadores = []
velocidad_salida = []
pelota_anterior = None
numero_frame = 0
frame_relativo = 0  # frames dentro del segmento

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_salida = cv2.VideoWriter(
    'C:\\picklelab_app\\partido_anotado.mp4',
    fourcc, fps, (ancho, alto)
)

# Factor de conversión (m/px). Idealmente calibrar con homografía.
M_POR_PIXEL = 0.02

while True:
    ret, frame = video.read()
    if not ret:
        print("Video terminado")
        break

    tiempo_actual = numero_frame / fps

    if tiempo_actual < 720:
        numero_frame += 1
        continue
    if tiempo_actual > 735:
        print("Segmento analizado")
        break

    frame_relativo += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- 1) DETECCIÓN POR COLOR (rápida, buena para pelota amarilla) ---
    mascara_amarillo = cv2.inRange(hsv, AMARILLO_MIN, AMARILLO_MAX)
    # Limpieza morfológica
    mascara_amarillo = cv2.GaussianBlur(mascara_amarillo, (5, 5), 0)
    mascara_amarillo = cv2.morphologyEx(mascara_amarillo, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
    contornos_color, _ = cv2.findContours(
        mascara_amarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    deteccion_color = None
    for c in contornos_color:
        area = cv2.contourArea(c)
        if 15 < area < 400:  # pelota de pickleball en píxeles
            perimetro = cv2.arcLength(c, True)
            if perimetro == 0:
                continue
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            if circularidad > 0.55:
                x, y, w, h = cv2.boundingRect(c)
                deteccion_color = (x + w // 2, y + h // 2)
                break  # nos quedamos con la primera válida

    # --- 2) YOLO + TRACKER (ByteTrack integrado) ---
    # El tracker mantiene IDs consistentes entre frames
    resultados = modelo.track(
        frame, verbose=False, persist=True,
        tracker="bytetrack.yaml",
        classes=[0, 32]  # 0=person, 32=sports ball
    )

    deteccion_yolo_pelota = None
    confianza_pelota = 0.0

    for resultado in resultados:
        if resultado.boxes is None or resultado.boxes.id is None:
            continue
        ids = resultado.boxes.id.cpu().numpy().astype(int)
        boxes = resultado.boxes.xyxy.cpu().numpy()
        confs = resultado.boxes.conf.cpu().numpy()
        clases = resultado.boxes.cls.cpu().numpy().astype(int)

        for box, conf, track_id, cls in zip(boxes, confs, ids, clases):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            nombre = modelo.names[cls]

            if nombre == "person" and conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"P{track_id} {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                posiciones_jugadores.append((tiempo_actual, cx, cy))

            if nombre == "sports ball" and conf > 0.15:
                deteccion_yolo_pelota = (cx, cy)
                confianza_pelota = conf

    # --- 3) FUSIÓN: YOLO tiene prioridad, color como respaldo ---
    medicion = deteccion_yolo_pelota if deteccion_yolo_pelota else deteccion_color

    # --- 4) FILTRO DE KALMAN: suaviza y predice si se pierde ---
    if medicion is not None:
        medida = np.array([[np.float32(medicion[0])],
                           [np.float32(medicion[1])]])
        if not kalman_kalman_inicializado:
            kalman.statePost[:2] = medida
            kalman_kalman_inicializado = True
        else:
            kalman.correct(medida)
        prediccion = kalman.predict()
        px, py = int(prediccion[0,0]), int(prediccion[1,0])
    else:
        # Si no hay detección, usamos la predicción del Kalman
        prediccion = kalman.predict()
        px, py = int(prediccion[0,0]), int(prediccion[1,0])
        medicion = (px, py)

    # Guardamos en el historial para calcular velocidad
    HISTORIAL_PELOTA.append((tiempo_actual, px, py))

    # --- 5) DIBUJO de la pelota (círculos rojo/amarillo) ---
    cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)        # rojo
    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)      # amarillo centro
    cv2.putText(frame, f"Pelota ({px},{py})", (px + 15, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    posiciones_pelota.append((tiempo_actual, px, py))

    # --- 6) CÁLCULO DE VELOCIDAD con ventana deslizante ---
    # En vez de 2 frames, usamos los últimos N → mucho más estable
    if len(HISTORIAL_PELOTA) >= 4:
        t0, x0, y0 = HISTORIAL_PELOTA[0]
        t1, x1, y1 = HISTORIAL_PELOTA[-1]
        dt = t1 - t0
        if dt > 0:
            dx = x1 - x0
            dy = y1 - y0
            dist_px = np.sqrt(dx**2 + dy**2)
            vel_px_s = dist_px / dt
            vel_m_s = vel_px_s * M_POR_PIXEL
            vel_km_h = vel_m_s * 3.6

            # Filtramos valores imposibles (una pelota de pickleball rara vez > 160 km/h)
            if 5 < vel_km_h < 180:
                velocidad_salida.append((tiempo_actual, vel_km_h))
                color_texto = (0, 255, 255) if vel_km_h > 80 else (255, 255, 255)
                cv2.putText(frame, f"{vel_km_h:.1f} km/h",
                            (px, py - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

    # --- 7) TRAYECTORIA (últimos 30 puntos) ---
    for i in range(1, len(posiciones_pelota)):
        if i < 30:
            pt1 = (posiciones_pelota[-i-1][1], posiciones_pelota[-i-1][2])
            pt2 = (posiciones_pelota[-i][1], posiciones_pelota[-i][2])
            cv2.line(frame, pt1, pt2, (255, 200, 0), 2)

    # Timestamp
    tiempo_texto = f"Tiempo: {int(tiempo_actual//60):02d}:{int(tiempo_actual%60):02d}"
    cv2.putText(frame, tiempo_texto, (ancho - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("PickleLab Analysis", frame)
    video_salida.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    numero_frame += 1

video_salida.release()
video.release()
cv2.destroyAllWindows()

print(f"\n=== RESUMEN ===")
print(f"Frames procesados: {numero_frame}")
print(f"Detecciones de jugadores: {len(posiciones_jugadores)}")
print(f"Detecciones de pelota: {len(posiciones_pelota)}")

if velocidad_salida:
    velocidades = [v[1] for v in velocidad_salida]
    print(f"Velocidad máxima: {max(velocidades):.1f} km/h")
    print(f"Velocidad promedio: {sum(velocidades)/len(velocidades):.1f} km/h")
    print(f"Golpes medidos: {len(velocidades)}")