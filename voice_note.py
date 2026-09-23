"""
================================================================================
VOICE-TO-MARKDOWN AUTOMATION SCRIPT (PRO VERSION - STABLE)
================================================================================

EDITOR SUGGESTION: 
Use 'vim' to edit this script for a fast, terminal-based workflow.
Example: vim voice_note.py

REQUIREMENTS
- Homebrew (https://brew.sh/)
- Python

SETUP (Recommended Workflow):
Copy and run these commands in your terminal for the first-time setup:

brew install ffmpeg portaudio && \
python3 -m venv .venv && \
source .venv/bin/activate && \
python -m pip install --upgrade pip && \
python -m pip install -r requirements.txt

USAGE:
1. Run the script: python3 voice_note.py
2. When recording starts, press [ENTER] to stop recording early.
3. If you do nothing, it will automatically stop after 10 minutes.

TROUBLESHOOTING:
- ERROR: 'module whisper has no attribute load_model'
  FIX: You installed the wrong package. Run:
       python -m pip uninstall whisper
       python -m pip install openai-whisper

- ERROR: 'PortAudio not found'
  FIX: Run: brew install portaudio

- ERROR: 'Permission Denied' (Microphone)
  FIX: Go to System Settings > Privacy & Security > Microphone and ensure 
       your Terminal/IDE is allowed to access it.
================================================================================
"""

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import datetime
import os
import numpy as np
import threading
import time
import torch  # Required for Mac GPU (MPS) acceleration
import re     # Required for cleaning filename slugs

# --- CONFIGURATION ---

# 1. Model Selection
# 'tiny', 'base', 'small', 'medium', 'large-v3'
WHISPER_MODEL_TYPE = "large-v3"

# 2. Hardware Acceleration
# Set to False to skip the MPS check and go straight to CPU (saves time on Mac)
# Set to True to attempt to use the Mac GPU
USE_MPS = False

# 3. Transcription Context (Improves spelling of specific terms)
INITIAL_PROMPT = "Python, Vim, Markdown, coding, and technical documentation."

# 4. Audio Settings
MAX_RECORDING_SECONDS = 600 
SAMPLE_RATE = 16000  # Optimized for Speech-to-Text AI (standard is 16kHz)

# Global flag to signal the recording loop to stop
stop_event = threading.Event()

def listen_for_stop():
    """Runs in a separate background thread to listen for the Enter key."""
    input("🎤 Recording... Press [ENTER] to stop recording early.\n")
    stop_event.set()

def record_audio(max_duration, filename):
    print(f"⏳ Max duration: {max_duration // 60} minutes.")
    
    listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
    listener_thread.start()

    chunks = []
    start_time = time.time()
    chunk_duration = 1.0 

    try:
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_duration:
                print("\n⏰ Time limit reached.")
                break
            
            chunk = sd.rec(int(chunk_duration * SAMPLE_RATE), 
                           samplerate=SAMPLE_RATE, 
                           channels=1)
            sd.wait() 
            chunks.append(chunk)
            
    except Exception as e:
        print(f"❌ Recording error: {e}")

    if stop_event.is_set():
        print("\n🛑 Stop signal received.")
    
    if not chunks:
        print("⚠️ No audio was captured.")
        return False

    full_recording = np.concatenate(chunks, axis=0)
    print("✅ Recording finished. Saving file...")
    write(filename, SAMPLE_RATE, full_recording)
    return True

def transcribe_and_save(audio_file, model_type):
    # Step 1: Determine device
    if USE_MPS and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Step 2: Precision Optimization
    # CPUs do not support FP16. Using FP16 on CPU causes a slow conversion delay.
    # We set fp16=False for CPU and fp16=True for MPS (GPU) to maximize speed.
    use_fp16 = True if device == "mps" else False
    
    try:
        print(f"🧠 Attempting transcription on {device} (FP16={use_fp16})...")
        model = whisper.load_model(model_type, device=device)
        result = model.transcribe(
            audio_file, 
            language="en", 
            initial_prompt=INITIAL_PROMPT,
            fp16=use_fp16
        )
    except Exception as e:
        # Step 3: FALLBACK to CPU if MPS fails
        if device == "mps":
            print(f"⚠️ MPS GPU error detected. Falling back to CPU for stability...")
            print(f"Details: {e}")
            device = "cpu"
            use_fp16 = False # Ensure FP16 is disabled on fallback
            model = whisper.load_model(model_type, device=device)
            result = model.transcribe(
                audio_file, 
                language="en", 
                initial_prompt=INITIAL_PROMPT,
                fp16=use_fp16
            )
        else:
            print(f"❌ Transcription failed: {e}")
            return

    text = result["text"].strip()
    
    if not text:
        print("⚠️ No speech detected.")
        return

    # --- FILENAME SLUG LOGIC ---
    words = text.split()[:5] # first 5 words of transcription
    slug = "-".join(words).lower() #kebab case
    slug = re.sub(r'[^a-z0-9-]', '', slug.replace(' ', '-')) # Remove non-word chars
    
    # Simple fallback if slug is empty
    if not slug:
        slug = "voice-note"

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_full = now.strftime("%Y-%m-%d %H:%M:%S")

    filename = f"{date_str}-{slug}.md"
    filepath = os.path.join(".", filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Transcript\n\n**Captured:** {timestamp_full}\n\n{text}")
        
        print(f"🚀 Success! Created: {filepath}")
        print(f"📝 Content: {text[:50]}...")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

def main():
    temp_audio = "temp_voice_note.wav"
    
    try:
        success = record_audio(MAX_RECORDING_SECONDS, temp_audio)
        if success:
            transcribe_and_save(temp_audio, WHISPER_MODEL_TYPE)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    main()

# ================================================================================
# REQUIREMENTS.TXT
# ================================================================================
'''
openai-whisper
sounddevice
scipy
numpy
torch
'''
