# cortar_video.py
import subprocess
from pathlib import Path
import imageio_ffmpeg

def cortar_video():
    print("Cortando video...")
    
    # Ruta del video original (ajusta si está en otra ubicación)
    video_original = Path("partido.mp4")
    
    if not video_original.exists():
        print(f"Error: No encontré {video_original}")
        print("Asegúrate de que el video esté en C:\\picklelab_app")
        return
    
    # Crear carpeta de salida si no existe
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ruta de salida
    video_corto = output_dir / "partido.mp4"
    
    # Comando de ffmpeg
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        '-i', str(video_original),
        '-ss', '00:12:00',
        '-to', '00:12:25',
        '-c', 'copy',
        '-y',  # Sobrescribir sin preguntar
        str(video_corto)
    ]
    
    # Ejecutar
    subprocess.run(cmd, check=True)
    
    print(f"✅ Video cortado guardado en: {video_corto}")
    print("Duración: 25 segundos (minuto 12:00 - 12:25)")

if __name__ == "__main__":
    cortar_video()