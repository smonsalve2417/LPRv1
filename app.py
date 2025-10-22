from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import time
import numpy as np
from threading import Thread, Lock, Event
from queue import Queue, Empty
from collections import deque
from ultralytics import YOLO
import requests
from cam import get_frames
from dotenv import load_dotenv
import os

load_dotenv()

API = os.getenv("API_KEY")
Skey = os.getenv("SECRET_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = Skey 
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- CONFIGURACIÓN ---
GEMINI_API_KEY = API 
#Define el URL al que llama la API, aquí también defines el modelo a usar, en mi caso 2.5
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

YOLO_FRAME_INTERVAL = 8
MIN_YOLO_DETECTIONS = 6
OCR_SEND_INTERVAL = 3
MAX_QUEUE_SIZE = 10
NUM_YOLO_WORKERS = 2
NUM_OCR_WORKERS = 1

model = YOLO(r'C:\Users\alvar\Documents\LPR\YoloV8\best.pt')

# Variables globales
detection_system = None
video_source = None
processing_thread = None
system_lock = Lock()


class LPRDetectionSystem:
    """Sistema de detección de placas con Flask/SocketIO."""
    
    def __init__(self):
        self.frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.yolo_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.ocr_queue = Queue(maxsize=5)
        
        self.detection_state = {
            'best_crop': None,
            'max_area': 0,
            'detection_counter': 0,
            'ocr_send_counter': 0,
            'ocr_active': False,
            'yolo_frame_counter': 0
        }
        self.state_lock = Lock()
        
        # Event para detener workers limpiamente
        self.stop_event = Event()
        self.threads = []
        
        self.placas_detectadas = []
        self.placas_lock = Lock()
        
        self.current_frame = None
        self.frame_lock = Lock()
        
        self.metrics = {
            'yolo_times': deque(maxlen=30),
            'ocr_times': deque(maxlen=10),
            'frames_processed': 0,
            'frames_dropped': 0,
            'fps': 0
        }
        self.metrics_lock = Lock()
        
        # Para streaming de video
        self.latest_annotated_frame = None
        self.annotated_frame_lock = Lock()
    
    def start(self):
        """Inicia todos los workers."""
        self.stop_event.clear()
        socketio.emit('system_log', {'message': '[SISTEMA] Iniciando workers...', 'type': 'info'})
        
        for i in range(NUM_YOLO_WORKERS):
            t = Thread(target=self.yolo_worker, args=(i,), daemon=True, name=f"YOLO-{i}")
            t.start()
            self.threads.append(t)
        
        t = Thread(target=self.ocr_worker, daemon=True, name="OCR-Worker")
        t.start()
        self.threads.append(t)
        
        t = Thread(target=self.decision_worker, daemon=True, name="Decision-Worker")
        t.start()
        self.threads.append(t)
        
        socketio.emit('system_log', {'message': f'[SISTEMA] {len(self.threads)} workers activos', 'type': 'success'})
    
    def stop(self):
        """Detiene todos los workers de forma limpia."""
        print("[SISTEMA] Enviando señal de stop a workers...")
        self.stop_event.set()
        
        # Limpiar colas para desbloquear workers
        self._clear_queues()
        
        # Esperar a que terminen los threads
        for t in self.threads:
            t.join(timeout=3.0)
            if t.is_alive():
                print(f"[WARNING] Thread {t.name} no terminó a tiempo")
        
        self.threads.clear()
        print("[SISTEMA] Workers detenidos correctamente")
        socketio.emit('system_log', {'message': '[SISTEMA] Workers detenidos', 'type': 'warning'})
    
    def _clear_queues(self):
        """Limpia todas las colas."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        while not self.yolo_queue.empty():
            try:
                self.yolo_queue.get_nowait()
            except Empty:
                break
        
        while not self.ocr_queue.empty():
            try:
                self.ocr_queue.get_nowait()
            except Empty:
                break
    
    def yolo_worker(self, worker_id):
        """Worker YOLO."""
        print(f"[YOLO-{worker_id}] Iniciado")
        
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
                
                # Verificar si se debe detener
                if self.stop_event.is_set():
                    break
                
                frame, frame_number = frame_data
                
                start_time = time.time()
                results = model(frame, verbose=False)
                elapsed = time.time() - start_time
                
                with self.metrics_lock:
                    self.metrics['yolo_times'].append(elapsed)
                
                best_detection = None
                max_area = 0
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > max_area:
                            max_area = area
                            placa_crop = frame[y1:y2, x1:x2]
                            best_detection = {
                                'crop': placa_crop,
                                'area': area,
                                'bbox': (x1, y1, x2, y2),
                                'frame_number': frame_number
                            }
                
                if not self.stop_event.is_set():
                    self.yolo_queue.put({
                        'detection': best_detection,
                        'frame_number': frame_number,
                        'worker_id': worker_id
                    })
                
                self.frame_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[YOLO-{worker_id}] Error: {e}")
        
        print(f"[YOLO-{worker_id}] Detenido")
    
    def decision_worker(self):
        """Worker de decisión."""
        print("[DECISION] Iniciado")
        
        while not self.stop_event.is_set():
            try:
                yolo_result = self.yolo_queue.get(timeout=0.5)
                
                if self.stop_event.is_set():
                    break
                
                with self.state_lock:
                    detection = yolo_result['detection']
                    
                    if detection:
                        self.detection_state['detection_counter'] += 1
                        
                        if detection['area'] > self.detection_state['max_area']:
                            self.detection_state['max_area'] = detection['area']
                            self.detection_state['best_crop'] = detection['crop']
                        
                        if (not self.detection_state['ocr_active'] and 
                            self.detection_state['detection_counter'] >= MIN_YOLO_DETECTIONS):
                            
                            self.detection_state['ocr_active'] = True
                            self.detection_state['ocr_send_counter'] = 0
                            
                            socketio.emit('system_log', {'message': '[DECISIÓN] OCR ACTIVADO - Placa estable', 'type': 'info'})
                            
                            if self.detection_state['best_crop'] is not None and not self.stop_event.is_set():
                                self.ocr_queue.put({
                                    'crop': self.detection_state['best_crop'].copy(),
                                    'area': self.detection_state['max_area'],
                                    'type': 'INICIAL'
                                })
                                self.detection_state['best_crop'] = None
                                self.detection_state['max_area'] = 0
                        
                        elif self.detection_state['ocr_active']:
                            self.detection_state['ocr_send_counter'] += 1
                            
                            if (self.detection_state['ocr_send_counter'] >= OCR_SEND_INTERVAL and
                                self.detection_state['best_crop'] is not None and
                                not self.stop_event.is_set()):
                                
                                self.ocr_queue.put({
                                    'crop': self.detection_state['best_crop'].copy(),
                                    'area': self.detection_state['max_area'],
                                    'type': 'PERIÓDICO'
                                })
                                self.detection_state['ocr_send_counter'] = 0
                                self.detection_state['best_crop'] = None
                                self.detection_state['max_area'] = 0
                    
                    else:
                        self.detection_state['detection_counter'] = 0
                        if self.detection_state['ocr_active']:
                            socketio.emit('system_log', {'message': '[DECISIÓN] OCR DESACTIVADO', 'type': 'warning'})
                        self.detection_state['ocr_active'] = False
                        self.detection_state['best_crop'] = None
                        self.detection_state['max_area'] = 0
                
                self.yolo_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[DECISION] Error: {e}")
        
        print("[DECISION] Detenido")
    
    def ocr_worker(self):
        """Worker OCR."""
        print("[OCR] Iniciado")
        
        while not self.stop_event.is_set():
            try:
                ocr_task = self.ocr_queue.get(timeout=0.5)
                
                if self.stop_event.is_set():
                    break
                
                crop = ocr_task['crop']
                area = ocr_task['area']
                task_type = ocr_task['type']
                
                start_time = time.time()
                texto_placa = self.ocr_with_gemini(crop)
                elapsed = time.time() - start_time
                
                with self.metrics_lock:
                    self.metrics['ocr_times'].append(elapsed)
                
                if texto_placa and len(texto_placa) >= 4:
                    with self.placas_lock:
                        if texto_placa not in self.placas_detectadas:
                            self.placas_detectadas.append(texto_placa)
                            
                            socketio.emit('new_plate', {
                                'plate': texto_placa,
                                'total': len(self.placas_detectadas),
                                'timestamp': time.strftime('%H:%M:%S')
                            })
                            
                            socketio.emit('system_log', {
                                'message': f'[OCR] *** PLACA REGISTRADA: {texto_placa} ***',
                                'type': 'success'
                            })
                
                self.ocr_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[OCR] Error: {e}")
        
        print("[OCR] Detenido")
    
    def ocr_with_gemini(self, cropped_image):
        """OCR con Gemini."""
        if GEMINI_API_KEY == "" or self.stop_event.is_set():
            return None
        
        base64_image = self.image_to_base64(cropped_image)
        if not base64_image:
            return None
        
        headers = {'Content-Type': 'application/json'}
        system_instruction = "You are an expert vehicle license plate reader. Analyze the provided image of a vehicle plate. Extract ONLY the alphanumeric text from the plate. Do not include any punctuation, spaces, or extra text. The output must be only the detected license plate number."
        user_query = "Extract the license plate number from this vehicle. If it cannot be read, respond only 'NO_READ'."
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": user_query},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}
                ]
            }],
            "systemInstruction": {"parts": [{"text": system_instruction}]}
        }

        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                headers=headers, 
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                return None

            result = response.json()
            detected_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if detected_text and detected_text.upper() != "NO_READ":
                return ''.join(filter(str.isalnum, detected_text)).upper()
            
            return None
                
        except Exception as e:
            return None
    
    @staticmethod
    def image_to_base64(image_np):
        """Convierte imagen a base64."""
        if image_np is None or not image_np.size > 0:
            return None
        
        is_success, buffer = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not is_success:
            return None
        return base64.b64encode(buffer).decode('utf-8')
    
    def add_frame(self, frame, frame_number):
        """Añade frame a procesamiento."""
        if self.stop_event.is_set():
            return
        
        try:
            self.frame_queue.put((frame, frame_number), timeout=0.01)
            with self.metrics_lock:
                self.metrics['frames_processed'] += 1
        except:
            with self.metrics_lock:
                self.metrics['frames_dropped'] += 1
    
    def update_annotated_frame(self, frame):
        """Actualiza frame anotado para streaming."""
        with self.annotated_frame_lock:
            self.latest_annotated_frame = frame.copy()
    
    def get_annotated_frame(self):
        """Obtiene último frame anotado."""
        with self.annotated_frame_lock:
            return self.latest_annotated_frame
    
    def get_metrics(self):
        """Obtiene métricas."""
        with self.metrics_lock:
            avg_yolo = sum(self.metrics['yolo_times']) / len(self.metrics['yolo_times']) if self.metrics['yolo_times'] else 0
            avg_ocr = sum(self.metrics['ocr_times']) / len(self.metrics['ocr_times']) if self.metrics['ocr_times'] else 0
            return {
                'avg_yolo_time': avg_yolo,
                'avg_ocr_time': avg_ocr,
                'frames_processed': self.metrics['frames_processed'],
                'frames_dropped': self.metrics['frames_dropped'],
                'fps': self.metrics['fps'],
                'queue_sizes': {
                    'frame': self.frame_queue.qsize(),
                    'yolo': self.yolo_queue.qsize(),
                    'ocr': self.ocr_queue.qsize()
                }
            }
    
    def get_plates(self):
        """Obtiene lista de placas."""
        with self.placas_lock:
            return self.placas_detectadas.copy()


class VideoProcessor:
    """Procesa video y genera frames para streaming."""
    
    def __init__(self, source_type, source_path=None):
        self.source_type = source_type
        self.source_path = source_path
        self.cap = None
        self.stop_event = Event()
        self.original_fps = 24
        
    def start(self):
        """Inicia captura de video."""
        if self.source_type == 'file':
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {self.source_path}")
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[VIDEO] FPS original del archivo: {self.original_fps}")
        elif self.source_type == 'camera':
            camera_index = int(self.source_path) if self.source_path else 0
            self.cap = cv2.VideoCapture(camera_index)
        elif self.source_type == 'ip':
            self.cap = cv2.VideoCapture(self.source_path)
        
        self.stop_event.clear()
        return True
    
    def stop(self):
        """Detiene captura."""
        print("[VIDEO] Deteniendo captura...")
        self.stop_event.set()
        time.sleep(0.2)  # Dar tiempo para que el último frame se procese
        if self.cap:
            self.cap.release()
        print("[VIDEO] Captura detenida")
    
    def get_frame(self):
        """Obtiene siguiente frame."""
        if self.stop_event.is_set() or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            # Reiniciar video si es archivo
            if self.source_type == 'file' and not self.stop_event.is_set():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        
        return frame if ret else None


def video_processing_thread():
    """Thread principal de procesamiento de video."""
    global detection_system, video_source
    
    print("[PROCESSING] Thread iniciado")
    frame_number = 0
    yolo_frame_counter = 0
    fps_start_time = time.time()
    fps_counter = 0
    
    # Calcular delay basado en FPS del video
    if video_source and video_source.original_fps > 0:
        target_delay = 1.0 / video_source.original_fps
    else:
        target_delay = 1.0 / 24.0
    
    last_frame_time = time.time()
    
    try:
        while not video_source.stop_event.is_set():
            current_time = time.time()
            
            # Control de FPS basado en el video original
            if current_time - last_frame_time < target_delay:
                time.sleep(0.001)
                continue
            
            frame = video_source.get_frame()
            if frame is None:
                if video_source.stop_event.is_set():
                    break
                continue
            
            last_frame_time = current_time
            frame_number += 1
            yolo_frame_counter += 1
            fps_counter += 1
            
            # Calcular FPS
            if time.time() - fps_start_time >= 1.0:
                with detection_system.metrics_lock:
                    detection_system.metrics['fps'] = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Enviar a YOLO cada N frames
            if yolo_frame_counter >= YOLO_FRAME_INTERVAL:
                detection_system.add_frame(frame.copy(), frame_number)
                yolo_frame_counter = 0
            
            # Anotar frame para display
            display_frame = frame.copy()
            
            plates = detection_system.get_plates()
            if plates:
                text = f"Ultima: {plates[-1]}"
                cv2.putText(display_frame, text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Total: {len(plates)}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Buscando placa...", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            metrics = detection_system.get_metrics()
            cv2.putText(display_frame, f"FPS: {metrics['fps']}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detection_system.update_annotated_frame(display_frame)
    
    except Exception as e:
        print(f"[PROCESSING] Error: {e}")
    finally:
        print("[PROCESSING] Thread detenido")


def generate_frames():
    """Generador de frames para streaming."""
    global detection_system, video_source
    
    while True:
        if detection_system is None or video_source is None:
            time.sleep(0.1)
            continue
        
        if video_source.stop_event.is_set():
            break
        
        frame = detection_system.get_annotated_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Codificar frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS para streaming web


# --- RUTAS FLASK ---

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_system():
    """Inicia el sistema."""
    global detection_system, video_source, processing_thread
    
    with system_lock:
        if detection_system is not None and not detection_system.stop_event.is_set():
            return jsonify({'error': 'Sistema ya está corriendo'}), 400
        
        data = request.json
        source_type = data.get('source_type', 'file')
        source_path = data.get('source_path', r'C:\Users\alvar\Documents\LPR\Demos\LWV344.mp4')
        
        try:
            print(f"[API] Iniciando sistema con fuente: {source_type}")
            
            # Inicializar sistema
            detection_system = LPRDetectionSystem()
            detection_system.start()
            
            # Inicializar video
            video_source = VideoProcessor(source_type, source_path)
            video_source.start()
            
            # Iniciar thread de procesamiento
            processing_thread = Thread(target=video_processing_thread, daemon=True, name="Processing-Thread")
            processing_thread.start()
            
            return jsonify({
                'status': 'success',
                'message': 'Sistema iniciado correctamente',
                'video_fps': video_source.original_fps
            })
        
        except Exception as e:
            print(f"[API] Error al iniciar: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Detiene el sistema."""
    global detection_system, video_source, processing_thread
    
    with system_lock:
        if detection_system is None:
            return jsonify({'error': 'Sistema no está corriendo'}), 400
        
        try:
            print("[API] Deteniendo sistema...")
            
            # 1. Detener video primero
            if video_source:
                video_source.stop()
            
            # 2. Esperar a que termine el thread de procesamiento
            if processing_thread and processing_thread.is_alive():
                processing_thread.join(timeout=2.0)
            
            # 3. Detener sistema de detección
            if detection_system:
                detection_system.stop()
            
            # 4. Limpiar referencias
            detection_system = None
            video_source = None
            processing_thread = None
            
            print("[API] Sistema detenido correctamente")
            
            return jsonify({
                'status': 'success',
                'message': 'Sistema detenido correctamente'
            })
        
        except Exception as e:
            print(f"[API] Error al detener: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def get_metrics():
    """Obtiene métricas del sistema."""
    global detection_system
    
    if detection_system is None:
        return jsonify({'error': 'Sistema no iniciado'}), 400
    
    metrics = detection_system.get_metrics()
    plates = detection_system.get_plates()
    
    return jsonify({
        'metrics': metrics,
        'plates': plates,
        'total_plates': len(plates)
    })

@app.route('/api/plates')
def get_plates():
    """Obtiene lista de placas detectadas."""
    global detection_system
    
    if detection_system is None:
        return jsonify([])
    
    plates = detection_system.get_plates()
    return jsonify(plates)

@app.route('/api/status')
def get_status():
    """Obtiene estado del sistema."""
    is_running = detection_system is not None and not detection_system.stop_event.is_set()
    
    return jsonify({
        'running': is_running,
        'has_detection_system': detection_system is not None,
        'has_video_source': video_source is not None
    })


if __name__ == '__main__':
    print("="*60)
    print("🚗 Sistema LPR - Detección de Placas Vehiculares")
    print("="*60)
    print("Servidor iniciando en http://localhost:5000")
    print("Presiona Ctrl+C para detener el servidor")
    print("="*60)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)