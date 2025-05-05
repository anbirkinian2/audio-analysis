import os
import numpy as np
import tensorflow as tf
import wave
import pydub
from scipy.signal import resample
import pandas as pd

model = tf.keras.models.load_model('./Saved_Models/CNN_Model.h5')
window_size = 8000
step_size = 4000
sr = 8000

def preprocess_audio(audio_path, sr=8000):
    if audio_path.lower().endswith('.wav'):
        return audio_path
    
    if audio_path is None:
        raise ValueError("audio_path cannot be None.")
    
    base_name = os.path.basename(audio_path)
    output_file_name = os.path.splitext(base_name)[0] + '.wav'
    output_file_path = "./testAudioConverted/" + output_file_name
    
    try:
        audio = pydub.AudioSegment.from_file(audio_path)
        audio.export(output_file_path, format="wav")
        print(f"Conversion successful! WAV file saved at: {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None
        

def process_wav_file_for_prediction(wav_path, window_size, step_size, target_sr=8000):
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        audio_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

        if sample_rate != target_sr:
            num_target_samples = int(len(audio_data) * target_sr / sample_rate)
            audio_data = resample(audio_data, num_target_samples)
            sample_rate = target_sr

    audio_segments = []
    num_windows = (len(audio_data) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = audio_data[start:end]

        if len(segment) == window_size:
            audio_segments.append(segment)

    audio_segments = np.array(audio_segments).reshape(-1, window_size, 1)
    return audio_segments

def extract_pitch_vector(audio_file):

    song_wav_path = audio_file

    song_audio_segments = process_wav_file_for_prediction(song_wav_path, window_size, step_size)

    predicted_song_pitches = model.predict(song_audio_segments)

    step_duration = step_size / sr
    timestamps = np.arange(0, len(predicted_song_pitches) * step_duration, step_duration)[:len(predicted_song_pitches)]
    pitch_time_dict = {float(timestamps[i]): float(predicted_song_pitches[i]) for i in range(len(timestamps))}

    pitch_data = pd.DataFrame(list(pitch_time_dict.items()), columns=["Timestamp (s)", "Pitch Value"])

    return pitch_data