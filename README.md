# PickleLab - Sports Analytics

Sistema de tracking y análisis de datos deportivos para pickleball usando visión por computadora e inteligencia artificial.

---

## Descripción

PickleLab es una herramienta desarrollada para clubes de pickleball que permite:
- Detectar y trackear la pelota en tiempo real
- Detectar jugadores y sus posiciones
- Calcular velocidades de saque y golpes
- Analizar trayectorias y estadísticas del juego
- Generar videos anotados automáticamente

---

## Características

- Detección dual: YOLOv8 + detección por color HSV para máxima precisión
- Tracking avanzado: Filtro de Kalman + ByteTrack
- Calibración precisa: Homografía para conversión píxeles a metros reales
- Fácil de usar: Solo necesitas un smartphone y una cancha de pickleball

---

## Estructura del Proyecto

picklelab/
├── src/ # Código fuente
│ ├── init.py
│ ├── config.py # Configuración global
│ ├── calibration.py # Homografía y calibración
│ ├── detection.py # Detección de pelota y jugadores
│ ├── tracking.py # Filtro de Kalman
│ ├── analysis.py # Cálculo de velocidades
│ └── visualization.py # Visualización de resultados
│
├── scripts/ # Scripts ejecutables
│ ├── 01_calibrate.py # Calibrar cámara
│ ├── 02_analyze_video.py # Analizar partido
│ └── 03_train_model.py # Entrenar modelo YOLO
│
├── data/ # Datos
│ ├── raw/ # Videos originales
│ ├── processed/ # Videos anotados
│ └── calibration/ # Matrices de calibración
│
├── models/ # Modelos de YOLO
├── notebooks/ # Experimentos y pruebas
└── docs/ # Documentación adicional

## Instalación

### Requisitos
- Python 3.8 o superior
- Anaconda (recomendado)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/Capdata90/Pickleball-software-.git
cd Pickleball-software- 

## CREAR ENTORNO CONDA

conda create -n picklelab python=3.11
conda activate picklelab

## INSTALAR DEPENDENCIAS

pip install opencv-python
pip install ultralytics
pip install numpy


```markdown
# PickleLab - Sports Analytics

Sistema de tracking y análisis de datos deportivos para pickleball usando visión por computadora e inteligencia artificial.

---

## Descripción

PickleLab es una herramienta desarrollada para clubes de pickleball que permite:
- Detectar y trackear la pelota en tiempo real
- Detectar jugadores y sus posiciones
- Calcular velocidades de saque y golpes
- Analizar trayectorias y estadísticas del juego
- Generar videos anotados automáticamente

---

## Características

- Detección dual: YOLOv8 + detección por color HSV para máxima precisión
- Tracking avanzado: Filtro de Kalman + ByteTrack
- Calibración precisa: Homografía para conversión píxeles a metros reales
- Fácil de usar: Solo necesitas un smartphone y una cancha de pickleball

---

## Estructura del Proyecto

```
picklelab/
├── src/                        # Código fuente
│   ├── __init__.py
│   ├── config.py              # Configuración global
│   ├── calibration.py         # Homografía y calibración
│   ├── detection.py           # Detección de pelota y jugadores
│   ├── tracking.py            # Filtro de Kalman
│   ├── analysis.py            # Cálculo de velocidades
│   └── visualization.py       # Visualización de resultados
│
├── scripts/                    # Scripts ejecutables
│   ├── 01_calibrate.py        # Calibrar cámara
│   ├── 02_analyze_video.py    # Analizar partido
│   └── 03_train_model.py      # Entrenar modelo YOLO
│
├── data/                       # Datos
│   ├── raw/                   # Videos originales
│   ├── processed/             # Videos anotados
│   └── calibration/           # Matrices de calibración
│
├── models/                     # Modelos de YOLO
├── notebooks/                  # Experimentos y pruebas
└── docs/                       # Documentación adicional
```

---

## Instalación

### Requisitos
- Python 3.8 o superior
- Anaconda (recomendado)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/Capdata90/Pickleball-software-.git
cd Pickleball-software-
```

2. Crear entorno conda:
```bash
conda create -n picklelab python=3.11
conda activate picklelab
```

3. Instalar dependencias:
```bash
pip install opencv-python
pip install ultralytics
pip install numpy
```

---

## Uso Básico (EN TERMINAL)

### 1. Calibrar la cámara



```bash
python scripts/01_calibrate.py
```

- Graba un video de la cancha vacía
- Ejecuta el script
- Haz click en las 4 esquinas de la cancha
- Presiona 's' para guardar

### 2. Analizar un partido

```bash
python scripts/02_analyze_video.py
```

- Coloca tu video en `data/raw/partido.mp4`
- Ejecuta el script
- El video anotado se guardará en `data/processed/`

---

## Protocolo de Grabación

### Posición de la cámara:
- Altura: 2.5 - 3.5 metros del suelo
- Ángulo: 30-45° hacia abajo
- Posición: Detrás de la línea de fondo, centrada
- Resolución: 1080p mínimo (4K preferible)
- FPS: 60 fps (30 fps mínimo)

### Requisitos técnicos:
- Trípode obligatorio (sin movimiento)
- Iluminación uniforme
- Toda la cancha visible
- Enfoque manual bloqueado

---

## Configuración

Edita `src/config.py` para personalizar:
- Dimensiones de la cancha
- Umbrales de detección
- Colores de visualización
- Rango HSV para la pelota amarilla

---

## Métricas Generadas

- Velocidad de saque (km/h)
- Velocidad promedio de golpes
- Trayectoria de la pelota
- Posición de jugadores
- Zonas de impacto

---

## Contribuciones

Este proyecto está en desarrollo activo. Si encuentras bugs o tienes sugerencias:
1. Abre un issue
2. Envía un pull request

---

## Licencia

MIT License - ver archivo LICENSE

---

## Desarrollador

Alonso de Leo  
Proyecto desarrollado para análisis deportivo de pickleball

---

## Soporte

Para dudas o soporte técnico, abre un issue en GitHub o contacta directamente.

---

Versión: 1.0  
Última actualización: Junio 2026
```