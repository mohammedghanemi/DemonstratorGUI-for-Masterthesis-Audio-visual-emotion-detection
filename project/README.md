# Emotion Detection Application

This application provides real-time emotion detection using a pre-trained deep learning model. It consists of a React frontend and a Node.js/Python backend.

## Prerequisites

- Node.js v14.21.2
- Python 3.x with the following packages:
  - tensorflow
  - opencv-python
  - numpy
  - decord
  - moviepy
  - scipy

## Project Structure

```
.
├── backend/
│   ├── python/
│   │   └── process_video.py
│   ├── services/
│   │   └── emotionDetection.js
│   ├── server.js
│   └── package.json
├── src/
│   ├── services/
│   │   └── api.ts
│   ├── App.tsx
│   └── ...
└── package.json
```

## Setup Instructions

1. Install frontend dependencies:
   ```bash
   npm install
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   npm install
   ```

3. Install Python dependencies:
   ```bash
   pip install tensorflow opencv-python numpy decord moviepy scipy
   ```

4. Start the backend server:
   ```bash
   cd backend
   npm start
   ```

5. Start the frontend development server:
   ```bash
   npm run dev
   ```

## Features

- Real-time emotion detection using webcam
- Video file upload and analysis
- Support for multiple emotions: anger, happiness, surprise, disgust, fear, sadness
- Beautiful and responsive UI
- Server status monitoring
- Error handling and loading states

## Model Information

- Input Shape: [16, 224, 224, 3]
- Frame Rate: 6 fps
- Model Path: D:/saved_model

## Notes

- Ensure the Python model path (D:/saved_model) is correctly set up before running the application
- The backend server runs on port 5000 by default
- The frontend development server typically runs on port 3000 or 5173