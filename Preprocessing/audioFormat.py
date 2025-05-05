from pydub import AudioSegment
import os

def convert_to_wav(input_file, output_folder = "./draftAudios"):
    base_name = os.path.basename(input_file).rsplit('.', 1)[0]
    output_file = os.path.join(output_folder, f"{base_name}.wav")
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

    return output_file
