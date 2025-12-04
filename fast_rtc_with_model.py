"""
Minimal FastRTC + Gemini voice-to-text assistant
- Captures your audio
- Converts to text using Gemini STT
- Responds with text (no audio output)
"""

import numpy as np
from fastrtc import Stream, ReplyOnPause
import google.generativeai as genai
import wave
import io
import soundfile as sf
from general_utils import get_secret  # your existing secret loader

# ---------- Configure Gemini ----------
GEMINI_KEY_PATH = r"D:\Documents\Secrets\gemini_secret.txt"
GEMINI_API_KEY = get_secret(GEMINI_KEY_PATH)
if not GEMINI_API_KEY:
    raise SystemExit("Gemini API key missing. Put the key in the file and try again.")

genai.configure(api_key=GEMINI_API_KEY)

# Choose your Gemini models
STT_MODEL = "gemini-1.5"
LLM_MODEL = "gemini-2.0-pro"


# ---------- Helpers ----------
def pcm_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    """
    Convert a numpy audio array (float32 in [-1,1] or int16) into WAV bytes (PCM16).
    Uses the standard library wave module (no libsndfile), so it avoids BytesIO issues.
    Returns WAV file bytes.
    """
    # Debugging output (optional)
    # print("pcm dtype:", pcm.dtype, "shape:", pcm.shape, "sr:", sr)

    # Ensure array is numpy
    pcm = np.asarray(pcm)

    # If float, convert from [-1,1] to int16
    if pcm.dtype == np.float32 or pcm.dtype == np.float64:
        # Clip and convert
        clipped = np.clip(pcm, -1.0, 1.0)
        int_data = (clipped * 32767.0).round().astype(np.int16)
    elif pcm.dtype == np.int16:
        int_data = pcm
    else:
        # If some other dtype, try convert to float then to int16
        int_data = (np.clip(pcm.astype(np.float32), -1.0, 1.0) * 32767.0).round().astype(np.int16)

    # Ensure shape is (nsamples, nchannels)
    if int_data.ndim == 1:
        nchannels = 1
        frames = int_data.tobytes()
    elif int_data.ndim == 2:
        nchannels = int_data.shape[1]
        # wave expects interleaved frames; numpy is already interleaved in row-major
        frames = int_data.tobytes()
    else:
        raise ValueError("Unsupported audio shape: {}".format(int_data.shape))

    # Write WAV with wave module into BytesIO
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sr)
        wf.writeframes(frames)

    bio.seek(0)
    return bio.read()


# ---------- Handler ----------
def gemini_text_handler(audio: tuple[int, np.ndarray]):
    """
    FastRTC handler: receives (sr, audio array)
    Returns a text response from Gemini.
    """
    sr, pcm = audio
    wav_bytes = pcm_to_wav_bytes(pcm, sr)

    # 1) Speech -> text
    try:
        if hasattr(genai, "speech_to_text"):
            resp = genai.speech_to_text(model=STT_MODEL, audio=wav_bytes, sample_rate=sr)
            transcript = resp.get("text", "") or resp.get("transcript", "")
        else:
            # Generic fallback using generate_content (SDK-dependent)
            model = genai.GenerativeModel(STT_MODEL)
            transcript = model.generate_content([{"mime_type": "audio/wav", "data": wav_bytes}]).text
    except Exception as e:
        print("STT error:", e)
        transcript = ""

    if not transcript:
        reply_text = "Sorry, I didn't catch that."
    else:
        # 2) LLM response
        try:
            llm = genai.GenerativeModel(LLM_MODEL)
            response = llm.generate_content([{"role": "user", "content": transcript}])
            reply_text = getattr(response, "text", str(response))
        except Exception as e:
            print("LLM error:", e)
            reply_text = "Sorry, I couldn't generate a response."

    # Yield the response as text
    yield reply_text


# ---------- FastRTC Stream ----------
stream = Stream(
    handler=ReplyOnPause(gemini_text_handler),
    modality="audio",
    mode="send-receive",
)

if __name__ == "__main__":
    print("Launching FastRTC Gemini text assistant UI...")
    stream.ui.launch()
