# cam.py
import cv2
import time
from dotenv import load_dotenv
import os

load_dotenv()  # Cargar variables de entorno desde el archivo .env

URL = os.getenv("URL")

def get_frames(url=URL, fps=20):
    """
    Generador que devuelve frames de la cámara IP a una frecuencia aproximada de `fps`.
    """
    video = cv2.VideoCapture(url)
    if not video.isOpened():
        raise Exception("No se pudo conectar a la cámara.")

    delay = 1.0 / fps  # tiempo entre frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame
        time.sleep(delay)

    video.release()

if __name__ == "__main__":
    for frame in get_frames(fps=20):
        cv2.imshow("Frame de la cámara", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()