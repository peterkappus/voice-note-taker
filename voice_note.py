"""
================================================================================
VOICE-TO-MARKDOWN AUTOMATION SCRIPT
================================================================================

EDITOR SUGGESTION: 
Use 'vim' to edit this script for a fast, terminal-based workflow.
Example: vim voice_note.py

SETUP (Recommended Workflow):
1. Install System Dependencies (via Homebrew):
   brew install ffmpeg portaudio

2. Create a Virtual Environment:
   python3 -m venv venv

3. Activate the Environment:
   source venv/bin/activate

4. Install Python Dependencies:
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

- ERROR: 'No such file or directory'
  FIX: Ensure the 'MARKDOWN_FILE_PATH' or 'SINGLE_FILE_DIR' variables 
       below point to valid folders.
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

# --- CONFIGURATION ---

# 1. The Master Log: All notes are appended here chronologically
MARKDOWN_FILE_PATH = os.path.expanduser("~/Documents/VoiceNotes_Master_Log.md")

# 2. Individual Files: Each note gets its own unique file here
# (e.g., 2023-10-27-title-here-1698412345.md)
SINGLE_FILE_DIR = os.path.expanduser("~/Documents/VoiceNotes_Individual/")

# Model options: 'tiny', 'base', 'small', 'medium', 'large'
WHISPER_MODEL_TYPE = "base"

# Max duration (10 minutes)
MAX_RECORDING_SECONDS = 600 
SAMPLE_RATE = 44100 

# Global flag to stop recording
stop_event = threading.Event()

def listen_for_stop():
    """Function run in a separate thread to listen for the Enter key."""
    input("🎤 Recording... Press [ENTER] to stop recording early.\n")
    stop_event.set()

def record_audio(max_duration, filename):
    print(f"⏳ Max duration: {max_duration // 60} minutes.")
    
    # Start the listener thread
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

def transcribe_and_save(audio_file, master_md_path, single_dir, model_type):
    print(f"🧠 Transcribing using '{model_type}' model...")
    
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_file)
    text = result["text"].strip()
    
    if not text:
        print("⚠️ No speech detected.")
        return

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_filename_str = now.strftime("%Y-%m-%d")
    unix_seconds = int(time.time())

    # --- PART 1: APPEND TO MASTER LOG ---
    try:
        with open(master_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp_str}\n{text}\n")
        print(f"✅ Appended to Master Log: {master_md_path}")
    except Exception as e:
        print(f"❌ Failed to append to Master Log: {e}")

    # --- PART 2: SAVE INDIVIDUAL UNIQUE FILE ---
    try:
        # Ensure the directory exists
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
            print(f"📁 Created directory: {single_dir}")

        # Construct filename: yyyy-mm-dd-title-here-%s.md
        unique_filename = f"{date_filename_str}-title-here-{unix_seconds}.md"
        unique_filepath = os.path.join(single_dir, unique_filename)

        with open(unique_filepath, "w", encoding="utf-8") as f:
            f.write(f"# Transcript\n\n**Date:** {timestamp_str}\n\n{text}")
        
        print(f"🚀 Individual file created: {unique_filepath}")
        print(f"📝 Content preview: {text[:50]}...")
    except Exception as e:
        print(f"❌ Failed to save individual file: {e}")

def main():
    temp_audio = "temp_voice_note.wav"
    
    try:
        success = record_audio(MAX_RECORDING_SECONDS, temp_audio)
        
        if success:
            transcribe_and_save(
                temp_audio, 
                MARKDOWN_FILE_PATH, 
                SINGLE_FILE_DIR, 
                WHISPER_MODEL_TYPE
            )
        
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
'''
