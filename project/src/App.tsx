import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Upload, Play, Pause, RefreshCw, AlertCircle } from 'lucide-react';
import { detectEmotion, checkServerHealth } from './services/api';
import { EmotionDisplay } from './components/EmotionDisplay';

type Emotion = 'anger' | 'happiness' | 'surprise' | 'disgust' | 'fear' | 'sadness';

function App() {
  const [mode, setMode] = useState<'webcam' | 'upload'>('webcam');
  const [isRecording, setIsRecording] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentEmotion, setCurrentEmotion] = useState<Emotion | null>(null);
  const [isServerConnected, setIsServerConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const webcamRef = useRef<Webcam>(null);
  const recordingInterval = useRef<NodeJS.Timeout>();

  useEffect(() => {
    checkServerConnection();
  }, []);

  const checkServerConnection = async () => {
    const isConnected = await checkServerHealth();
    setIsServerConnected(isConnected);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setIsProcessing(true);
      setError(null);

      try {
        const result = await detectEmotion(file);
        setCurrentEmotion(result.emotion as Emotion);
      } catch (err) {
        setError('Failed to process video. Please try again.');
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const captureFrame = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], 'webcam-frame.jpg', { type: 'image/jpeg' });

        try {
          const result = await detectEmotion(file);
          setCurrentEmotion(result.emotion as Emotion);
        } catch (err) {
          console.error('Error processing frame:', err);
        }
      }
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      recordingInterval.current = setInterval(captureFrame, 2000);
    } else if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 p-6">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Camera className="w-8 h-8" />
            Real-time Emotion Detection
          </h1>
          <div className="mt-2 flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isServerConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm">Server Status: {isServerConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-6">
        {!isServerConnected && (
          <div className="mb-6 bg-red-900/50 p-4 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <p>Server is not connected. Please ensure the backend server is running.</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Side - Webcam */}
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Webcam Feed</h2>
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="mt-4 flex justify-center">
                <button
                  onClick={toggleRecording}
                  disabled={!isServerConnected}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg transition ${
                    isRecording ? 'bg-red-600' : 'bg-green-600'
                  } ${!isServerConnected && 'opacity-50 cursor-not-allowed'}`}
                >
                  {isRecording ? (
                    <>
                      <Pause className="w-5 h-5" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Start Recording
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right Side - Upload and Results */}
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Upload Video</h2>
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <label className="w-full h-full flex items-center justify-center cursor-pointer">
                  <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileUpload}
                    disabled={!isServerConnected || isProcessing}
                  />
                  <div className="flex flex-col items-center gap-4">
                    <Upload className="w-12 h-12 text-gray-400" />
                    <span className="text-gray-400">
                      {selectedFile ? selectedFile.name : 'Click to upload video'}
                    </span>
                  </div>
                </label>
              </div>
            </div>

            {/* Emotion Results */}
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Detected Emotion</h2>
                <button
                  onClick={() => setCurrentEmotion(null)}
                  className="p-2 hover:bg-gray-700 rounded-full transition"
                >
                  <RefreshCw className="w-5 h-5" />
                </button>
              </div>
              <EmotionDisplay
                emotion={currentEmotion}
                isProcessing={isProcessing}
                error={error}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;