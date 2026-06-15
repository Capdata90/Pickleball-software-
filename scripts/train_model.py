# scripts/03_train_model.py
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))

from src import config

def main():
    print("PICKLELAB - Entrenamiento de Modelo YOLO")
    print("=" * 50)

    # Ruta al archivo de configuración del dataset (YAML)
    # Este archivo se genera al exportar el dataset desde Roboflow
    dataset_yaml = Path(__file__).parent.parent / "data" / "dataset" / "data.yaml"

    if not dataset_yaml.exists():
        print(f"Error: No se encontró el archivo del dataset en {dataset_yaml}")
        print("Asegurate de exportar tu dataset desde Roboflow en formato YOLOv8")
        print("y colocarlo en data/dataset/data.yaml")
        return

    print("Iniciando entrenamiento...")
    print(f"Dataset: {dataset_yaml}")

    # Cargar el modelo base (pre-entrenado)
    model = YOLO(config.DEFAULT_MODEL)

    # Entrenar el modelo
    results = model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        name="pickleball_train",
        project=str(Path(__file__).parent.parent / "models" / "runs")
    )

    # Guardar el mejor modelo en la carpeta models/
    best_model_path = Path(__file__).parent.parent / "models" / "runs" / "pickleball_train" / "weights" / "best.pt"
    custom_model_path = config.CUSTOM_MODEL

    if best_model_path.exists():
        shutil.copy(best_model_path, custom_model_path)
        print(f"\nEntrenamiento completado.")
        print(f"Mejor modelo guardado en: {custom_model_path}")
    else:
        print("\nEl entrenamiento terminó, pero no se encontró el archivo 'best.pt'.")

if __name__ == "__main__":
    main()