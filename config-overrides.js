// config-overrides.js
const webpack = require('webpack');

module.exports = function override(config, env) {
  // Add fallbacks for node.js core modules
  config.resolve.fallback = {
    ...config.resolve.fallback,
    fs: false,  // Provides an empty module for 'fs'
    path: require.resolve('path-browserify'),
    stream: require.resolve('stream-browserify'),
    crypto: require.resolve('crypto-browserify'),
    buffer: require.resolve('buffer/'),
    process: require.resolve('process/browser'),
  };

  // Add webpack plugins to provide process and Buffer polyfills
  config.plugins = [
    ...config.plugins,
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer'],
    }),
  ];

  // Disable source map warnings
  if (config.module && config.module.rules) {
    config.module.rules = config.module.rules.map(rule => {
      if (rule.use && rule.use.some && rule.use.some(use => use.loader && use.loader.includes('source-map-loader'))) {
        rule.exclude = /node_modules\/face-api\.js/;
      }
      return rule;
    });
  }

  return config;
};
