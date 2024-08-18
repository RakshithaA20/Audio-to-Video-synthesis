import transformers
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import os
import shutil

# Function to convert audio to video
def convert_audio_to_video(input_audio_path, output_video_path):
    # Load the pre-trained models
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Ensure models are downloaded before proceeding
    try:
        video_pipeline = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
    except Exception as e:
        print(f"Error loading video synthesis model: {e}")
        return

    # Load and process the audio file
    audio, sampling_rate = librosa.load(input_audio_path, sr=16000)
    
    # Tokenize and predict
    input_values = processor(audio, return_tensors='pt', sampling_rate=sampling_rate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Generate video based on the transcription
    video_input = {'text': transcription}
    try:
        video_output = video_pipeline(video_input)
        generated_video_path = video_output[OutputKeys.OUTPUT_VIDEO]
        # Move the generated video to the desired output path
        shutil.move(generated_video_path, output_video_path)
        print(f"Video saved to {output_video_path}")
    except Exception as e:
        print(f"Error generating video: {e}")

# Example usage
if __name__ == "__main__":
    input_audio_path = "C:/Users/lenovo/OneDrive/Desktop/final project/Voice 001.m4a"
    output_video_path = "C:/Users/lenovo/OneDrive/Desktop/final project/Voice 001_output.mp4"
    
    convert_audio_to_video(input_audio_path, output_video_path)
