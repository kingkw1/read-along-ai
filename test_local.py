import time
from local_inference import (
    local_transcribe_audio, 
    local_ask_minicpm_judge, 
    local_synthesize_speech
)

# Replace this with the path to one of your real audio files
TEST_AUDIO = "data/processed_audio/01_scientist.wav" 
TEST_TARGET = "scientist"

print("--- 1. Testing Faster-Whisper (ASR) ---")
start = time.time()
transcript = local_transcribe_audio(TEST_AUDIO)
print(f"Transcript: '{transcript}' (Took {time.time() - start:.2f}s)\n")

print("--- 2. Testing llama.cpp (MiniCPM Judge) ---")
# Let's test a deliberate failure scenario to see if the GGUF model reasons correctly
start = time.time()
verdict = local_ask_minicpm_judge(TEST_TARGET, "scientists") 
print(f"Verdict for '{TEST_TARGET}' vs 'scientists': {verdict} (Took {time.time() - start:.2f}s)\n")

print("--- 3. Testing VoxCPM (TTS) ---")
start = time.time()
audio_bytes = local_synthesize_speech("You did a great job!")
print(f"Generated {len(audio_bytes)} bytes of audio. (Took {time.time() - start:.2f}s)\n")

print("✅ ALL LOCAL MODELS LOADED AND EXECUTED SUCCESSFULLY!")