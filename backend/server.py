import asyncio
import json
import os
import glob
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from contextlib import asynccontextmanager
import sounddevice as sd
import numpy as np
import queue
import threading
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.moonshine_api import ModelArch

# Global state
main_loop = None
translation_lock = None
latest_sentence_id = 0
active_connections = []
llm = None
tokenizer = None
vad_model = None
asr = None
target_langs = ["Spanish", "French", "German"]
speaker_lang = "en"
is_recording = False
audio_queue = queue.Queue()
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop, translation_lock, llm, tokenizer, vad_model, asr, device
    main_loop = asyncio.get_running_loop()
    translation_lock = asyncio.Lock()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Loading Silero VAD...")
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    
    print("Loading Moonshine Medium Streaming ASR Model...")
    model_path = "/Users/sebastian/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
    asr = Transcriber(model_path, ModelArch.MEDIUM_STREAMING)
    
    print("Loading AngelSlim Base FP16 Model via Transformers...")
    tokenizer = AutoTokenizer.from_pretrained("tencent/HY-MT1.5-1.8B", trust_remote_code=True)
    # Ensure padding token exists and side is left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    llm = AutoModelForCausalLM.from_pretrained("tencent/HY-MT1.5-1.8B", dtype=torch.float16, trust_remote_code=True).to(device)
    print(f"Models successfully loaded on {device.upper()}")
    
    yield
    
    global is_recording
    is_recording = False

app = FastAPI(lifespan=lifespan)

async def process_translation(text, current_speaker_lang, current_target_langs, sentence_id):
    msg = {
        "type": "transcription_start",
        "source_text": text,
        "speaker_lang": current_speaker_lang,
        "target_langs": current_target_langs
    }
    for conn in active_connections:
        try:
            await conn.send_json(msg)
        except:
            pass
            
    if not translation_lock:
        return
        
    async with translation_lock:
        loop = asyncio.get_running_loop()
        
        async def translate_lang(lang):
            prompt = f"Translate the following segment into {lang}, without additional explanation. {text}"
            
            def do_generate():
                messages = [{"role": "user", "content": prompt}]
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_input, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = llm.generate(**inputs, max_new_tokens=256, temperature=0.1)
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            result = await loop.run_in_executor(None, do_generate)
            
            print(f"  -> [{lang}] {result}")
            update_msg = {
                "type": "translation_update",
                "lang": lang,
                "text": result,
                "source_text": text
            }
            for conn in active_connections:
                try:
                    await conn.send_json(update_msg)
                except:
                    pass

        tasks = [translate_lang(lang) for lang in current_target_langs]
        await asyncio.gather(*tasks)

def process_chunk_loop():
    global latest_sentence_id, is_recording
    
    frame_samples = 512 
    raw_buffer = np.zeros((0,), dtype=np.float32)
    chunk_buffer = []
    
    min_threshold_samples = int(16000 * 1.5)
    max_threshold_samples = int(16000 * 4.0)
    
    while is_recording:
        try:
            data = audio_queue.get(timeout=0.1)
            raw_buffer = np.concatenate((raw_buffer, data.flatten()))
            
            while len(raw_buffer) >= frame_samples:
                frame = raw_buffer[:frame_samples]
                raw_buffer = raw_buffer[frame_samples:]
                
                chunk_buffer.append(frame)
                current_length = len(chunk_buffer) * frame_samples
                
                if current_length < min_threshold_samples:
                    continue
                    
                tensor = torch.from_numpy(frame)
                prob = vad_model(tensor, 16000).item()
                is_pause = prob < 0.2
                
                force_cut = current_length >= max_threshold_samples
                
                if is_pause or force_cut:
                    audio_data = np.concatenate(chunk_buffer)
                    
                    try:
                        res = asr.transcribe_without_streaming(audio_data.tolist(), 16000)
                        
                        if res and res.lines:
                            words_list = []
                            for line in res.lines:
                                if line.words:
                                    words_list.extend(line.words)
                            
                            if len(words_list) > 0:
                                if force_cut:
                                    last_word = words_list[-1]
                                    if last_word.end >= 3.7:
                                        words_list.pop()
                                        leftover = audio_data[-int(16000 * 0.5):]
                                        chunk_buffer = [leftover]
                                    else:
                                        chunk_buffer = []
                                else:
                                    chunk_buffer = []
                                
                                text = " ".join([w.word.strip() for w in words_list]).strip()
                            else:
                                # Fallback: No word timestamps available in this Moonshine build
                                text = " ".join([line.text for line in res.lines]).strip()
                                if force_cut and text:
                                    text_parts = text.split()
                                    if len(text_parts) > 1:
                                        text_parts.pop()
                                        text = " ".join(text_parts)
                                    leftover = audio_data[-int(16000 * 0.5):]
                                    chunk_buffer = [leftover]
                                else:
                                    chunk_buffer = []
                            
                            if text and len(text) > 2:
                                latest_sentence_id += 1
                                current_id = latest_sentence_id
                                print(f"[{speaker_lang}] {text}")
                                
                                if main_loop:
                                    asyncio.run_coroutine_threadsafe(
                                        process_translation(text, speaker_lang, target_langs.copy(), current_id), 
                                        main_loop
                                    )
                        else:
                            chunk_buffer = []
                            
                    except Exception as e:
                        print("Moonshine error:", e)
                        chunk_buffer = []
        except queue.Empty:
            continue

def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_queue.put(indata.copy())

current_stream = None
processor_thread = None

def start_recording(lang):
    global is_recording, current_stream, processor_thread, speaker_lang, audio_queue
    if is_recording:
        stop_recording()
        
    speaker_lang = lang
    is_recording = True
    
    while not audio_queue.empty():
        audio_queue.get_nowait()
        
    print(f"Starting Audio Stream for {lang}...")
    current_stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, dtype=np.float32)
    current_stream.start()
    
    processor_thread = threading.Thread(target=process_chunk_loop, daemon=True)
    processor_thread.start()

def stop_recording():
    global is_recording, current_stream
    is_recording = False
    if current_stream:
        current_stream.stop()
        current_stream.close()
        current_stream = None
        print("Microphone stream stopped.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("action") == "start":
                global target_langs
                target_langs = msg.get("target_langs", target_langs)
                new_lang = msg.get("speaker_lang", "en")
                
                start_recording(new_lang)
                
            elif msg.get("action") == "stop":
                stop_recording()
                    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        if len(active_connections) == 0:
            stop_recording()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
