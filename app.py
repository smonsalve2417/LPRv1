import cv2 
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import time
from cam import get_frames

model = YOLO(r'C:\Users\alvar\Documents\LPR\YoloV8\best.pt')

def get_frame_source():
    mode = int(input("Selecciona el modo de entrada (1 para cámara IP, 2 para video local): "))
    if mode == 1:
        return get_frames(fps=4)
    elif mode == 2:
        video_path = r'C:\Users\alvar\Videos\bambu.mp4' #input("Introduce la ruta del video local: ")
        cap = cv2.VideoCapture(video_path)
        def frame_gen():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
        return frame_gen()
    else:
        raise ValueError("Modo no válido")
