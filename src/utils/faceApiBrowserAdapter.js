// This file ensures face-api.js works correctly in browser environments
// by providing a mock for the 'fs' module and other Node.js specifics

// Import this file before using face-api.js
import * as faceapi from 'face-api.js';

// Apply browser-friendly environment settings
if (typeof window !== 'undefined') {
  // Ensure the env is properly configured for browsers
  // This helps prevent errors related to Node.js specific features
  faceapi.env.monkeyPatch({
    Canvas: HTMLCanvasElement,
    Image: HTMLImageElement,
    ImageData: window.ImageData,
    Video: HTMLVideoElement,
    createCanvasElement: () => document.createElement('canvas'),
    createImageElement: () => document.createElement('img')
  });

  console.log('Face API browser environment configured successfully');
}

export default faceapi;
