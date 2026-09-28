# 🚀 ISL Two-Way Translator — Deployment Guide

This guide provides step-by-step instructions for deploying the **Indian Sign Language (ISL) ↔ Speech Two-Way Translator** across multiple environments: Local Machine, Docker Container, Cloud Platforms (Hugging Face Spaces, Render, Railway, AWS/GCP), and Android Mobile.

---

## 📋 Table of Contents
1. [System Architecture & Features](#system-architecture--features)
2. [Local Deployment (Windows / macOS / Linux)](#1-local-deployment)
3. [Docker Container Deployment](#2-docker-container-deployment)
4. [Cloud Hosting Deployments](#3-cloud-hosting-deployments)
   - [Hugging Face Spaces (Zero-Cost / GPU optional)](#hugging-face-spaces)
   - [Render / Railway](#render--railway)
   - [AWS EC2 / DigitalOcean / GCP](#aws-ec2--digitalocean--gcp)
5. [Android Mobile App Deployment](#4-android-mobile-app-deployment)
6. [API Reference & Health Endpoints](#5-api-reference--health-endpoints)

---

## 🧠 System Architecture & Features

- **Bidirectional Translation**:
  - **Deaf/Mute → Spoken English/Hindi**: Real-time camera capture → MediaPipe holistic landmark extraction → CNN-LSTM Sign Recognition & FER Facial Expression models → Transformer NLP sentence grammar correction → Speech synthesis.
  - **Hearing Person → Sign Avatar**: Speech recognition / text input → ISL Gloss converter & Fingerspelling synthesizer → Regional Dialect mapper (South / North / Maharashtra / Generic) → 2D/3D Avatar keyframe animation.
- **Microservices & UI**:
  - **FastAPI** backend with asynchronous REST & WebSocket inference.
  - **Modern Glassmorphic Web App** with offline neural model execution, live gesture visualizer, dictionary explorer, and interactive learning studio.

---

## 1. Local Deployment

### 🪟 Windows (1-Click Start)
Double-click `start_server.bat` or run:
```powershell
.\start_server.ps1
```
Or manually:
```bash
cd isl_two_way_translator
python run_app.py
```

### 🐧 Linux / macOS
Make the start script executable and run:
```bash
chmod +x start_server.sh
./start_server.sh
```

Once started, open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**  
👉 API Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 2. Docker Container Deployment

The project includes a production-ready `Dockerfile` and `docker-compose.yml`.

### Quick Start with Docker Compose:
```bash
docker-compose up --build -d
```

### Or using standard Docker CLI:
```bash
# 1. Build the Docker image
docker build -t isl-two-way-translator:latest .

# 2. Run the container
docker run -d \
  --name isl_translator \
  -p 8000:8000 \
  --restart unless-stopped \
  isl-two-way-translator:latest
```

Check health status:
```bash
curl http://localhost:8000/health
```

---

## 3. Cloud Hosting Deployments

### 🤗 Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Docker** as the SDK.
3. Push this repository to your Space repository.
4. Hugging Face will automatically build the `Dockerfile` and expose port `7860` or `8000`. (The app dynamically binds to `$PORT`).

### 🌐 Render / Railway
1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Create a new **Web Service**.
3. Set environment variables:
   - `PORT=8000`
   - `HOST=0.0.0.0`
4. Build command: `pip install -r requirements.txt` (or select Docker deployment).
5. Start command: `python src/backend/app.py`.

### ☁️ AWS EC2 / GCP Compute Engine / DigitalOcean
1. Launch an Ubuntu 22.04 LTS instance (2+ vCPUs, 4GB+ RAM recommended).
2. Clone repository & install Docker:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose
   git clone <your-repo-url>
   cd isl_two_way_translator
   sudo docker-compose up -d --build
   ```
3. Configure Nginx reverse proxy with SSL (Let's Encrypt) to proxy requests to `http://127.0.0.1:8000`.

---

## 4. Android Mobile App Deployment

For offline edge deployment on Android phones:

1. Open the `android_app/` folder in **Android Studio** (Electric Eel or newer).
2. The project comes pre-configured with CameraX, MediaPipe Tasks Vision, and TensorFlow Lite Android support.
3. Ensure the `.tflite` models and label mapping are present in `android_app/app/src/main/assets/`:
   - `sign_model.tflite`
   - `face_model.tflite`
   - `sign_label_map.json`
4. Build the APK:
   - In Android Studio: **Build > Build Bundle(s) / APK(s) > Build APK(s)**
   - Or via CLI:
     ```bash
     cd android_app
     ./gradlew assembleRelease
     ```
5. Install on any Android device:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

---

## 5. API Reference & Health Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main Interactive Web Application |
| `/health` | `GET` | Health check probe for Docker / Load Balancers |
| `/api/status` | `GET` | Model load status, active vocab, and regional dialect |
| `/api/predict_sign` | `POST` | Inference on landmark sequence frames |
| `/api/predict_face` | `POST` | Facial expression / non-manual grammar inference |
| `/api/text_to_gloss` | `POST` | Translates English/Hindi text → ISL Gloss + Avatar Keyframes |
| `/api/gloss_to_text` | `POST` | Formulates raw ISL gloss tokens into natural sentences |
| `/api/dictionary` | `GET` | Fetches full dictionary categorized with keyframe metadata |
| `/api/sign_keyframes/{word}` | `GET` | Keyframe animation coordinates for a specific sign |
