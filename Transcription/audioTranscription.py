import pandas as pd
import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_path):
    
    result = model.transcribe(audio_path, word_timestamps=True)

    data = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            data.append({
                "Start Timestamp": word_info["start"],
                "End Timestamp": word_info["end"],
                "Word": word_info.get("word", "")
            })

    transcript_data = pd.DataFrame(data)
    return transcript_data
