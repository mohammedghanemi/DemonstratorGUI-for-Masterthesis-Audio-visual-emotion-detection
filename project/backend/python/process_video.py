import sys
import cv2
import numpy as np
import tensorflow as tf
from decord import VideoReader
from moviepy.editor import AudioFileClip
from scipy.io import wavfile
from scipy.fftpack import dct
import json

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def calculate_mfcc(signal, sample_rate):
    NFFT = 512
    hop_size = 256
    window = np.hanning(NFFT)
    
    stft = np.abs(np.fft.rfft(signal, NFFT, axis=-1))
    mel_filterbank = np.linspace(0, sample_rate//2, 40)
    mfcc = dct(stft, type=2, axis=-1, norm='ortho')[:13]
    
    return mfcc

def format_frames(frames, output_size):
    frames = frames.astype(np.uint8)
    frames_resized = np.array([cv2.resize(frame, (output_size[0], output_size[1])) 
                              for frame in frames])
    frames_resized = frames_resized[:16]
    return frames_resized

def read_video(file_path, num_frames=16):
    vr = VideoReader(file_path)
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()
    return format_frames(frames, output_size=(224, 224))

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2]

    try:
        # Load the model
        model = tf.saved_model.load(model_path)

        # Process video
        video_frames = read_video(video_path)
        video_frames = np.expand_dims(video_frames, axis=0)
        video_frames = tf.convert_to_tensor(video_frames, dtype=tf.float32)

        # Get prediction
        predictions = model(video_frames)
        predicted_class = np.argmax(predictions, axis=-1)[0]

        # Map prediction to emotion
        emotions = ['anger', 'happiness', 'surprise', 'disgust', 'fear', 'sadness']
        emotion = emotions[predicted_class]

        # Return result
        result = {
            "emotion": emotion,
            "confidence": float(predictions[0][predicted_class]),
            "frameCount": video_frames.shape[1],
            "frameSize": [video_frames.shape[2], video_frames.shape[3]]
        }
        
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()