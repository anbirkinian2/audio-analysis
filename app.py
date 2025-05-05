from flask import Flask, request, render_template, jsonify
import os
import CNN_Usage.pv_process as pv_process
import Transcription.audioTranscription as at
import AdaBoost_Usage.classifier as abc
import Preprocessing.audioFormat as af
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Handle the transcription request.
    - Save uploaded audio file.
    - Transcribe the audio and return the transcript.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    transcript_df = at.transcribe_audio(file_path)
    print(transcript_df.columns)

    transcript = " ".join(transcript_df['Word'].tolist())

    return jsonify({"transcript": transcript})

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("Starting analyze endpoint")
    
    if 'audio' not in request.files:
        logger.error("No audio file in request")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.error("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, audio_file.filename)
    
    try:
        logger.info(f"Saving uploaded file to {temp_path}")
        audio_file.save(temp_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return jsonify({"error": f"Error saving uploaded file: {str(e)}"}), 500

    audio_file_wav = None
    try:
        # Convert to WAV
        logger.info("Converting to WAV")
        audio_file_wav = af.convert_to_wav(temp_path)
        logger.info(f"WAV file created at: {audio_file_wav}")

        # Extract pitch vector
        logger.info("Extracting pitch vector")
        pitch_df = pv_process.extract_pitch_vector(audio_file_wav)
        logger.info(f"Pitch vector extracted with shape: {pitch_df.shape}")

        # Transcribe audio
        logger.info("Transcribing audio")
        transcript_df = at.transcribe_audio(audio_file_wav)
        logger.info(f"Transcription completed with {len(transcript_df)} words")

        # Map pitch to words
        logger.info("Mapping pitch to words")
        word_pitch_df = abc.map_pitch_to_words(pitch_df, transcript_df)
        logger.info(f"Pitch mapping completed with {len(word_pitch_df)} words")

        # Use AdaBoost classifier
        logger.info("Using AdaBoost classifier")
        key_words, non_key_words = abc.use_adaboost(word_pitch_df)
        logger.info(f"Classification completed. Key words: {len(key_words)}, Non-key words: {len(non_key_words)}")

        result = {
            "key_words": key_words,
            "non_key_words": non_key_words
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            if audio_file_wav and os.path.exists(audio_file_wav):
                os.remove(audio_file_wav)
                logger.info(f"Removed WAV file: {audio_file_wav}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=3000, debug=True)
