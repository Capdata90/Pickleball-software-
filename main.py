# main.py
"""
PICKLELAB - Sistema de Análisis de Pickleball
Punto de entrada principal con menú interactivo.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def show_menu():
    print("\n" + "=" * 50)
    print("  PICKLELAB - Sports Analytics")
    print("=" * 50)
    print("\nSelecciona una opción:")
    print("  1. Calibrar cámara")
    print("  2. Analizar partido")
    print("  3. Entrenar modelo YOLO")
    print("  0. Salir")
    print()

def main():
    while True:
        show_menu()
        choice = input("Opción: ").strip()
        
        if choice == "1":
            from scripts import calibrate
            calibrate.main()
        
        elif choice == "2":
            from scripts import analyze_video
            analyze_video.main()
        
        elif choice == "3":
            from scripts import train_model
            train_model.main()
        
        elif choice == "0":
            print("Hasta luego!")
            break
        
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()