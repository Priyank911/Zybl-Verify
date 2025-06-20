# IMPORTANT: Bug Fixes

## `navigator.getUserMedia`

`navigator.getUserMedia` is now deprecated and is replaced by `navigator.mediaDevices.getUserMedia`. To fix this bug replace all versions of `navigator.getUserMedia` with `navigator.mediaDevices.getUserMedia`

## Low-end Devices Bug

The video eventListener for `play` fires up too early on low-end machines, before the video is fully loaded, which causes errors to pop up from the Face API and terminates the script (tested on Debian [Firefox] and Windows [Chrome, Firefox]). Replaced by `playing` event, which fires up when the media has enough data to start playing.

# Face Detection App

## Setup Instructions

### Firebase Configuration

1. Copy `src/config/firebase.config.template.js` to `src/config/firebase.config.js`
2. Update the configuration values in `firebase.config.js` with your Firebase project details:
   - Get these values from your Firebase Console -> Project Settings -> Web App
   - Generate a secure encryption key for face vector encryption

```javascript
// Example configuration
export const firebaseConfig = {
  projectId: "your-project-id",
  authDomain: "your-project-id.firebaseapp.com",
  storageBucket: "your-project-id.appspot.com",
  apiKey: "your-api-key",
  messagingSenderId: "your-messaging-sender-id",
  appId: "your-app-id"
};

export const ENCRYPTION_KEY = "your-encryption-key";
```

### Security Notes

- Never commit `firebase.config.js` to version control
- Keep your encryption key secure
- Use Firebase security rules to protect your data

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

### `npm test`

Launches the test runner in the interactive watch mode.

### `npm run build`

Builds the app for production to the `build` folder.

## Source Map Warnings

If you're seeing warnings like:
```
Failed to parse source map from '...\node_modules\face-api.js\src\...' file: Error: ENOENT: no such file or directory
```

These are harmless warnings related to missing source map files in the face-api.js library. They don't affect functionality but can clutter your console output.

### How to Fix the Warnings

We've added environment configuration files that disable source maps. This is the simplest solution:

- `.env.development` - Disables source maps during development
- `.env.production` - Disables source maps for production builds

After adding these files, restart your development server:

```bash
npm start
```

For advanced users who've ejected from Create React App and have direct access to webpack configuration, you can modify it to exclude the face-api.js library from source map processing:

```javascript
// In your webpack.config.js
module.exports = {
  // ... other webpack configuration
  module: {
    rules: [
      {
        test: /\.js$/,
        enforce: 'pre',
        use: ['source-map-loader'],
        exclude: /node_modules\/face-api\.js/
      }
    ]
  }
};
```

## Node.js 'fs' Module Error Fix

If you're seeing warnings like:
```
Module not found: Error: Can't resolve 'fs' in '...node_modules\face-api.js\build\es6\env'
```

This is because face-api.js tries to use Node.js's 'fs' module which is not available in browser environments. We've implemented these fixes:

1. Added webpack configuration to provide fallbacks for Node.js core modules
2. Created a browser adapter for face-api.js (`src/utils/faceApiBrowserAdapter.js`)
3. Updated React components to use the browser adapter

These changes are implemented through:
- `config-overrides.js` - Provides webpack configuration overrides
- Using `react-app-rewired` instead of `react-scripts` to apply the overrides

No action is needed as these fixes are already in place.
