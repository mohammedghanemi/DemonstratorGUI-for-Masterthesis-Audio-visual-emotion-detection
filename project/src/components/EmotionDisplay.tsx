import React from 'react';

type Emotion = 'anger' | 'happiness' | 'surprise' | 'disgust' | 'fear' | 'sadness';

interface EmotionDisplayProps {
  emotion: Emotion | null;
  isProcessing: boolean;
  error: string | null;
}

const emotionEmojis: Record<Emotion, string> = {
  anger: '😠',
  happiness: '😊',
  surprise: '😲',
  disgust: '🤢',
  fear: '😨',
  sadness: '😢',
};

export function EmotionDisplay({ emotion, isProcessing, error }: EmotionDisplayProps) {
  if (isProcessing) {
    return (
      <div className="flex items-center gap-3">
        <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent" />
        <span>Processing...</span>
      </div>
    );
  }

  if (error) {
    return <div className="text-red-400">{error}</div>;
  }

  if (!emotion) {
    return (
      <p className="text-gray-400">
        Start recording or upload a video to detect emotions
      </p>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <span className="text-4xl">{emotionEmojis[emotion]}</span>
      <span className="text-2xl font-bold capitalize">{emotion}</span>
    </div>
  );
}