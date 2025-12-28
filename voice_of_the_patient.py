# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup Audio recorder (ffmpeg & sounddevice)
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, duration=10, sample_rate=16000):
    """
    Record audio using sounddevice and save as WAV.
    Python 3.13 compatible (NO pydub, NO audioop).
    """
    try:
        logging.info("Recording started... Speak now")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()

        write(file_path, sample_rate, audio)
        logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"Recording error: {e}")

def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3"):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text



if __name__ == "__main__":
    audio_filepath = "patient_voice_test.wav"
    stt_model = "whisper-large-v3"
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    record_audio(audio_filepath, duration=10)

    text = transcribe_with_groq(
        stt_model=stt_model,
        audio_filepath=audio_filepath,
        api_key=GROQ_API_KEY
    )

    """ print("\nTRANSCRIPTION:")
    print(text)"""
