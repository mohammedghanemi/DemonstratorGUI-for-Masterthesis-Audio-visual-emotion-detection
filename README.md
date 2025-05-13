# Real-Time Emotion Detection

This project uses a combination of computer vision and audio analysis techniques to detect emotions in real-time from video files or live camera streams. The system extracts visual frames and audio features (MFCC) from video files, processes them through a pre-trained deep learning model, and outputs the predicted emotion.

## Features

- **Video Input**: Load a video from your local storage or use the webcam to detect emotions in real-time.
- **Emotion Detection**: Detects one of the following emotions: Anger, Happiness, Surprise, Disgust, Fear, and Sadness.
- **Real-Time Camera Detection**: Capture video from the camera, process it, and detect emotions in real-time.
- **Pre-trained Model**: Utilizes a deep learning model for emotion recognition that processes audio and visual features from videos.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyQt5
- OpenCV
- TensorFlow
- Keras
- MoviePy
- Decord
- SciPy
- Matplotlib
- PIL (Pillow)

You can install the required dependencies using `pip`:

```bash
pip install PyQt5 opencv-python tensorflow keras moviepy scipy matplotlib pillow decord
