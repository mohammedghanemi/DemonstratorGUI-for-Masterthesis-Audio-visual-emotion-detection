const API_URL = 'http://localhost:5000/api';

export async function detectEmotion(videoFile: File) {
  const formData = new FormData();
  formData.append('video', videoFile);

  try {
    const response = await fetch(`${API_URL}/detect-emotion`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to process video');
    }

    return await response.json();
  } catch (error) {
    console.error('Error detecting emotion:', error);
    throw error;
  }
}

export async function checkServerHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
}