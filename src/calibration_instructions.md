# 📋 INSTRUCCIONES DE CALIBRACIÓN - PickleLab

## 🎯 Objetivo
Marcar las 4 esquinas de la cancha en el frame de calibración para calcular la homografía (conversión píxeles → metros).

---

## 📍 ORDEN DE LOS CLICKS (MUY IMPORTANTE)

Debes hacer click en las esquinas **EN ESTE ORDEN EXACTO**:

    CLICK 4 ←───────→ CLICK 3
    (sup-izq)         (sup-der)
        │               │
        │               │
        │               │
    CLICK 1 ←───────→ CLICK 2
    (inf-izq)         (inf-der)


    
### **Paso a paso:**

1. **CLICK 1** → Esquina **inferior izquierda** (la más cercana a la cámara, lado izquierdo)
2. **CLICK 2** → Esquina **inferior derecha** (la más cercana a la cámara, lado derecho)
3. **CLICK 3** → Esquina **superior derecha** (la más lejana, lado derecho)
4. **CLICK 4** → Esquina **superior izquierda** (la más lejana, lado izquierdo)

---

## ⌨️ TECLAS

- **`s`** → Guardar calibración (solo funciona si marcaste 4 puntos)
- **`q`** → Salir sin guardar

---

## ✅ CHECKLIST ANTES DE EMPEZAR

- [ ] El video muestra TODA la cancha (las 4 esquinas visibles)
- [ ] La iluminación es buena (sin sombras fuertes en las esquinas)
- [ ] La cámara está fija (sin movimiento)
- [ ] No hay jugadores tapando las esquinas

---

## 🎥 DÓNDE HACER CLICK EXACTAMENTE

Haz click en la **intersección exacta** de las líneas:

- **Esquina inferior:** Donde se juntan la línea de fondo y la línea lateral
- **Esquina superior:** Donde se juntan la línea del fondo (lado red) y la línea lateral

**Consejo:** Si la cancha tiene líneas blancas gruesas, haz click en el **borde interior** de la línea (el lado de la cancha, no el de afuera).

---

## 🔧 CÓMO USAR

```python
from src.calibration import CourtCalibrator
import cv2

# 1. Cargar frame de calibración
cap = cv2.VideoCapture("data/raw/calibracion.mp4")
ret, frame = cap.read()
cap.release()

# 2. Crear calibrador
calibrador = CourtCalibrator()

# 3. Calibración interactiva
exito = calibrador.interactive_calibration(frame)

# 4. Guardar
if exito:
    calibrador.save_calibration()
    print("✅ Calibración guardada en data/calibration/")