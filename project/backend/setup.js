import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// Create necessary directories
const dirs = ['uploads', 'logs'];
dirs.forEach(dir => {
  const dirPath = join(process.cwd(), dir);
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath);
    console.log(`Created directory: ${dir}`);
  }
});

// Install Python dependencies
try {
  console.log('Installing Python dependencies...');
  execSync('pip install tensorflow opencv-python numpy decord moviepy scipy', { stdio: 'inherit' });
  console.log('Python dependencies installed successfully');
} catch (error) {
  console.error('Failed to install Python dependencies:', error.message);
  process.exit(1);
}

console.log('\nBackend setup completed successfully!');