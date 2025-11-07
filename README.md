# 🚀 Proyecto Flask + YOLOv8 + SocketIO

Aplicación basada en **Flask** con soporte para **SocketIO**, visión por computadora usando **OpenCV** y modelos **YOLOv8 (Ultralytics)**.
Este README describe dos formas de instalación y ejecución:

1. Instalación **local** con Python.
2. Ejecución con **Docker Compose**.

---

## 📦 Requisitos del sistema

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.11+**
- **pip** (gestor de paquetes de Python)
- **Git** (opcional)
- **Docker** y **Docker Compose** (para la segunda opción)

---

## 🧩 Dependencias principales

### Python (requirements.txt)

```
# --- CORE ---
Flask==3.1.2
Flask-SocketIO==5.5.1
Werkzeug==3.1.3
itsdangerous==2.2.0
Jinja2==3.1.6
python-dotenv==1.1.1
requests==2.32.5

# --- COMPUTER VISION & ML ---
opencv-python-headless==4.12.0.88
numpy==2.2.6
ultralytics==8.3.203

# --- UTILIDADES ---
bidict==0.23.1
typing_extensions==4.15.0
tqdm==4.67.1
pillow==11.3.0
```

### Dependencias del sistema (para Docker o instalación local en Linux)

```
libgl1
libglib2.0-0
```

---

## 🛠️ 1. Instalación local (entorno Python)

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/MugnoA/LPRv1.git
cd LPRv1
```

### 2️⃣ Instalar dependencias del sistema (solo Linux)

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

### 3️ Instalar dependencias de Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Ejecutar la aplicación

```bash
python app.py
```

La aplicación se ejecutará por defecto en:
👉 [http://localhost:5000](http://localhost:5000)

---

## 🐳 2. Instalación con Docker Compose

Esta es la forma más rápida de levantar el proyecto sin instalar dependencias manualmente.

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/MugnoA/LPRv1.git
cd LPRv1
```

### 2️⃣ Construir la imagen

```bash
docker compose build
```

### 3️⃣ Levantar los contenedores

```bash
docker compose up
```

> Si deseas ejecutar en segundo plano:

```bash
docker compose up -d
```

La aplicación estará disponible en:
👉 [http://localhost:5000](http://localhost:5000) (según la configuración del `docker-compose.yml`)

### 4️⃣ Detener los contenedores

```bash
docker compose down
```

---

## 🧰 Estructura básica del proyecto

```
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

---

## 🧠 Notas

- Asegúrate de colocar tu modelo YOLOv8 (`best.pt`) en la ruta configurada dentro de `app.py`.
- Si cambias el puerto dentro del contenedor, actualiza el `docker-compose.yml` para reflejarlo.
- En sistemas Windows, Docker Desktop debe estar activo antes de ejecutar los comandos.

---

## 🪄 Comandos útiles

```bash
# Ver logs del contenedor
docker compose logs -f

# Reconstruir desde cero
docker compose build --no-cache

# Eliminar imágenes y contenedores antiguos
docker system prune -af
```
