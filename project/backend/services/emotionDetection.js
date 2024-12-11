import { spawn } from 'child_process';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_PATH = "D:/saved_model";
const INPUT_SIZE = 224;
const NUM_FRAMES = 16;
const SAMPLING_RATE = 6;

const UC_LABEL2ID = {
  'anger': 0,
  'happiness': 1,
  'surprise': 2,
  'disgust': 3,
  'fear': 4,
  'sadness': 5
};

const UC_ID2LABEL = Object.fromEntries(
  Object.entries(UC_LABEL2ID).map(([k, v]) => [v, k])
);

export async function processVideoAndAudio(videoPath) {
  return new Promise((resolve, reject) => {
    // Spawn Python process to handle the video processing
    const pythonProcess = spawn('python', [
      join(__dirname, '../python/process_video.py'),
      videoPath,
      MODEL_PATH
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${error}`));
        return;
      }

      try {
        const processedResult = JSON.parse(result);
        resolve(processedResult);
      } catch (err) {
        reject(new Error('Failed to parse Python output'));
      }
    });
  });
}