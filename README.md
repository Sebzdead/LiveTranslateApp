# LiveTranslate

LiveTranslate is a local, real-time speech transcription and parallel translation application designed to run seamlessly on MacOS with Metal Performance Shaders (MPS) acceleration. 

It captures live audio from your microphone, transcribes it instantly, and translates it simultaneously into multiple target languages using state-of-the-art on-device machine learning models.

## Architecture

The project is split into two main components:

1. **Backend (Python + FastAPI)**
   - **Dynamic Audio Chunking**: Audio streams are processed using a state-machine driven by **Silero VAD (v5)**. The algorithm evaluates 32ms micro-frames to detect natural speech pauses, seamlessly buffering audio between 1.5 and 4.0 seconds. 
   - **ASR (Speech-to-Text)**: Powered by the [Moonshine Medium Streaming](https://github.com/moonshine-ai/moonshine) model. If a speaker talks continuously for 4.0 seconds, the backend employs a "Guillotine" recovery method that evaluates Moonshine's word-level timestamps to safely drop truncated words and prepend them to the next audio buffer, completely eliminating cut-off words.
   - **Translation**: Powered by the base FP16 model of [AngelSlim's HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B). It runs natively on your Mac's GPU (MPS) via Hugging Face `transformers` to deliver fast and accurate translations.
   - **Communication**: A FastAPI WebSocket server bridges the backend ML pipelines with the frontend UI.

2. **Frontend (React + Vite)**
   - A modern, responsive React application built with Vite.
   - Features a premium glassmorphism aesthetic with smooth micro-animations.
   - Implements an auto-scrolling, typewriter-style chat interface with a 1,000-chunk history and dynamically resizing flex channels.
   - Allows users to select the speaker's language and seamlessly toggle between target languages without overloading local GPU memory.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for running the frontend)
- **MacOS** (M-series Apple Silicon is highly recommended for MPS acceleration)

## Installation & Setup

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/sebastian/LiveTranslate.git
cd LiveTranslate/LiveTranslateApp
```

### 1. Backend Setup

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn websockets moonshine-voice transformers torch torchaudio accelerate huggingface_hub sounddevice numpy
```

### 2. Frontend Setup

Navigate to the frontend directory and install the Node modules:

```bash
cd frontend
npm install
```

## Running the Application

You will need two terminal windows to run both the backend and frontend simultaneously. 

*Note: The very first time you start the backend, it will automatically download the necessary AI models from Hugging Face (~3.6GB for the translation model and ~50MB for the transcription models).*

**Terminal 1: Start the Backend**
```bash
cd LiveTranslateApp
source venv/bin/activate
python backend/server.py
```

**Terminal 2: Start the Frontend**
```bash
cd LiveTranslateApp/frontend
npm run dev
```

## Usage

1. Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).
2. Ensure your Mac's default microphone is enabled and working.
3. Select the **Speaker Language** from the dropdown menu (Defaults to English).
4. Select the **Target Language** you want to translate into (Defaults to Spanish). To prevent GPU inference bottlenecking, only one target language is translated at a time.
5. Click the **Start** button. The button will pulse to indicate it is listening.
6. Speak into your microphone. Your transcription and its translation will appear on the screen in real-time, accumulating naturally as you converse.

## Acknowledgements

- [Moonshine Voice](https://github.com/moonshine-ai/moonshine) for their incredible real-time STT engine.
- [Tencent Hunyuan / AngelSlim](https://huggingface.co/tencent/HY-MT1.5-1.8B) for the powerful and compact translation foundation model.
