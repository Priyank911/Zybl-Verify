// This file helps document how to suppress source map warnings
// when using face-api.js in your project

// Option 1: For Create React App projects
// Create .env.development and .env.production files with:
// GENERATE_SOURCEMAP=false

// Option 2: For webpack projects
// Add this to your webpack.config.js:
/*
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
*/

// Option 3: In your HTML file, when loading face-api.min.js directly
// Make sure to remove any source map references:
/*
<script src="face-api.min.js"></script>
*/
// Instead of:
/*
<script src="face-api.min.js" sourcemap="face-api.min.js.map"></script>
*/

console.log('Source map warning suppression guide loaded');
