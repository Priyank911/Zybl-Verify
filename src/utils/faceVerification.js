import faceapi from './faceApiBrowserAdapter';
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, getDocs, query, where } from 'firebase/firestore';
import { firebaseConfig, ENCRYPTION_KEY } from '../config/firebase.config';

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Function to generate face vector from detection data with quality metrics
export const generateFaceVector = (detection) => {
  if (!detection || !detection.descriptor) {
    throw new Error('Invalid face detection data');
  }
  
  // Check that descriptor is a valid Float32Array
  if (!(detection.descriptor instanceof Float32Array) || detection.descriptor.length === 0) {
    throw new Error('Invalid descriptor format');
  }
  
  // First convert to regular array for easier processing
  const vectorArray = Array.from(detection.descriptor);
  
  // Enhanced validation with strict quality metrics
  let sum = 0;
  let sumSquares = 0;
  let hasInvalidValues = false;
  let outOfRangeCount = 0;
  let zeroCount = 0;
  
  for (let i = 0; i < vectorArray.length; i++) {
    const value = vectorArray[i];
    
    // Check for invalid values
    if (isNaN(value) || !isFinite(value)) {
      hasInvalidValues = true;
      break;
    }
    
    // Check for extreme values (potential quality issues)
    if (Math.abs(value) > 0.5) {
      outOfRangeCount++;
    }
    
    // Check for too many zero values (could indicate poor quality)
    if (Math.abs(value) < 0.001) {
      zeroCount++;
    }
    
    sum += value;
    sumSquares += value * value;
  }
  
  const mean = sum / vectorArray.length;
  const variance = (sumSquares / vectorArray.length) - (mean * mean);
  const stdDev = Math.sqrt(variance);
  const norm = Math.sqrt(sumSquares);
  
  // Log vector quality metrics
  console.log('Face vector quality metrics:', {
    length: vectorArray.length,
    mean: mean.toFixed(4),
    variance: variance.toFixed(4),
    stdDev: stdDev.toFixed(4),
    norm: norm.toFixed(4),
    outOfRangeValues: outOfRangeCount,
    zeroValues: zeroCount,
    isNormalized: Math.abs(norm - 1.0) < 0.1
  });
  
  // Stricter quality checks
  if (hasInvalidValues) {
    throw new Error('Face vector contains invalid values');
  }
  
  // Reject vectors with poor distribution
  if (zeroCount > vectorArray.length * 0.2) { // More than 20% near-zero values
    throw new Error('Face vector contains too many near-zero values - poor quality detection');
  }
  
  // Reject vectors with extreme variance
  if (stdDev > 0.3 || stdDev < 0.05) {
    throw new Error('Face vector has abnormal standard deviation - likely a poor quality image');
  }
  
  // Strictly enforce normalization
  if (Math.abs(norm - 1.0) > 0.1) {
    console.warn('Face vector is not properly normalized, normalizing now');
    
    // Normalize the vector
    return vectorArray.map(x => x / norm);
  }
  
  return vectorArray;
};

// Function to encrypt face vector
export const encryptFaceVector = (vector) => {
  if (!vector || !Array.isArray(vector)) {
    throw new Error('Invalid face vector');
  }
  // For now, we'll just store the vector as is
  // In production, you should implement proper encryption
  return vector;
};

// Function to decrypt face vector
export const decryptFaceVector = (encryptedVector) => {
  if (!encryptedVector || !Array.isArray(encryptedVector)) {
    throw new Error('Invalid encrypted vector');
  }
  // For now, we'll just return the vector as is
  // In production, you should implement proper decryption
  return encryptedVector;
};

// Function to calculate similarity between two vectors
export const calculateSimilarity = (vector1, vector2) => {
  try {
    // Ensure vectors are arrays and have the same length
    if (!Array.isArray(vector1) || !Array.isArray(vector2)) {
      console.error('Invalid vector format:', { 
        vector1Type: typeof vector1, 
        vector2Type: typeof vector2 
      });
      throw new Error('Vectors must be arrays');
    }

    if (vector1.length !== vector2.length) {
      console.error('Vector length mismatch:', { 
        vector1Length: vector1.length, 
        vector2Length: vector2.length 
      });
      throw new Error('Vectors must be of same length');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    // Vector normalization and dot product calculation
    for (let i = 0; i < vector1.length; i++) {
      const v1 = Number(vector1[i]);
      const v2 = Number(vector2[i]);
      
      if (isNaN(v1) || isNaN(v2)) {
        console.error('Invalid vector values at index', i, { v1, v2 });
        throw new Error('Invalid vector values');
      }

      dotProduct += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) {
      console.warn('Zero norm detected:', { norm1, norm2 });
      return 0;
    }

    // Cosine similarity calculation
    const similarity = dotProduct / (norm1 * norm2);
    
    // Ensure the result is within valid range [-1, 1]
    const clampedSimilarity = Math.max(-1, Math.min(1, similarity));
    
    // Log similarity for debugging
    console.log(`Face similarity: ${clampedSimilarity.toFixed(4)}`);
    
    return clampedSimilarity;
  } catch (error) {
    console.error('Error in calculateSimilarity:', error);
    throw error;
  }
};

// Function to store face vector in Firebase with strict duplicate checking
export const storeFaceVector = async (faceVector) => {
  try {
    if (!Array.isArray(faceVector) || faceVector.length < 128) {
      throw new Error('Invalid face vector format');
    }

    // ULTRA-STRICT duplicate check before storing
    console.log('Running ultra-strict duplicate check before storing...');
    
    // Step 1: Use our regular check first
    const existingCheck = await checkExistingFaceVector(faceVector);
    
    if (existingCheck) {
      console.log('Face already exists in database with high confidence, skipping storage');
      return { success: true, status: 'existing' };
    }
    
    // Step 2: Additional direct vector comparisons
    // Get all face vectors again for direct comparison
    const faceVectorsRef = collection(db, 'faceVectors');
    const q = query(faceVectorsRef);
    const querySnapshot = await getDocs(q);
    
    // Enhanced duplicate detection - multiple metrics
    for (const doc of querySnapshot.docs) {
      const storedData = doc.data();
      const storedVector = storedData.vector;
      
      // Skip invalid vectors
      if (!Array.isArray(storedVector) || storedVector.length !== faceVector.length) {
        continue;
      }
      
      // Check multiple similarity metrics 
      const similarity = await calculateCosineSimilarity(faceVector, storedVector);
      const distance = calculateEuclideanDistance(faceVector, storedVector);
      const manhattanDistance = calculateManhattanDistance(faceVector, storedVector);
      const angleDegreeDiff = calculateVectorAngleDifference(faceVector, storedVector);
      
      // Compare source and stored vector statistics
      const sourceVectorStats = calculateVectorStats(faceVector);
      const storedVectorStats = calculateVectorStats(storedVector);
      const meanDiff = Math.abs(sourceVectorStats.mean - storedVectorStats.mean);
      const stdDevDiff = Math.abs(sourceVectorStats.stdDev - storedVectorStats.stdDev);
      
      console.log(`Direct vector comparison: similarity=${similarity.toFixed(4)}, distance=${distance.toFixed(4)}, manhattan=${manhattanDistance.toFixed(2)}, angle=${angleDegreeDiff.toFixed(2)}°, meanDiff=${meanDiff.toFixed(4)}, stdDevDiff=${stdDevDiff.toFixed(4)}`);
      
      // Any vector with high similarity could be a duplicate - be very strict
      if (similarity > 0.9) {
        console.log(`Potential duplicate detected with similarity ${similarity.toFixed(4)}`);
        
        // If we have extremely high similarity and other metrics confirm
        if (
          (similarity > 0.95 && distance < 0.5) || 
          (similarity > 0.92 && distance < 0.6 && manhattanDistance < 20 && angleDegreeDiff < 8 && meanDiff < 0.01 && stdDevDiff < 0.01)
        ) {
          console.log(`Direct vector comparison found existing match with similarity ${similarity.toFixed(4)}`);
          return { success: true, status: 'existing' };
        }
      }
    }
    
    // If we get here, this is confirmed to be a new face
    console.log('Confirmed new face. Storing in database...');
    
    // Generate unique IDs
    const timestamp = new Date().toISOString();
    const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const vectorId = `face_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Calculate vector quality metrics for storage
    const vectorStats = calculateVectorStats(faceVector);
    
    // Store the face vector with metadata
    const docRef = await addDoc(faceVectorsRef, {
      vector: faceVector,
      timestamp: timestamp,
      vectorId: vectorId,
      userId: userId,
      vectorStats: {
        mean: vectorStats.mean,
        stdDev: vectorStats.stdDev,
        min: vectorStats.min,
        max: vectorStats.max
      },
      checksum: calculateVectorChecksum(faceVector)
    });

    console.log(`New face vector successfully stored with ID: ${docRef.id}`);
    return { success: true, status: 'new', id: docRef.id };
  } catch (error) {
    console.error('Error storing face vector:', error);
    throw new Error(`Failed to store face vector: ${error.message}`);
  }
};

// Helper function to calculate a simple checksum for a vector
const calculateVectorChecksum = (vector) => {
  if (!Array.isArray(vector)) return '0';
  
  // Sum all values in the vector and create a hash
  const sum = vector.reduce((acc, val) => acc + Number(val), 0);
  return sum.toString(16) + '_' + vector.length;
}

// Function to check for existing face vectors with extreme strictness
export const checkExistingFaceVector = async (faceVector) => {
  try {
    if (!Array.isArray(faceVector) || faceVector.length < 128) {
      console.error('Invalid face vector format:', { type: typeof faceVector, length: faceVector?.length });
      throw new Error('Invalid face vector format');
    }

    console.log('Starting extremely strict face verification check...');
    
    // Log source vector statistics for debugging
    const sourceVectorStats = calculateVectorStats(faceVector);
    console.log('Source face vector stats:', sourceVectorStats);
    
    // Get all face vectors from database
    const faceVectorsRef = collection(db, 'faceVectors');
    const q = query(faceVectorsRef);
    const querySnapshot = await getDocs(q);
    
    const totalVectors = querySnapshot.docs.length;
    console.log(`Found ${totalVectors} existing vectors in database`);
    
    // If no vectors exist in database, return false
    if (totalVectors === 0) {
      console.log('No existing vectors found in database, this is definitely a new user');
      return false;
    }
    
    // Track all face comparisons for logging
    const allComparisons = [];
    
    // UPDATED THRESHOLDS - Making verification MUCH stricter
    const STRICTEST_MATCH_THRESHOLD = 0.97;   // Previously 0.95 - Now requires even higher similarity
    const STRICT_MATCH_THRESHOLD = 0.92;      // Previously 0.9 - Raising this too
    const MAX_DISTANCE_THRESHOLD = 0.4;       // Previously 0.5 - Requiring closer distance
    const MAX_SECONDARY_DISTANCE = 0.6;       // Previously 0.7 - Tightening secondary threshold
    
    // Process each vector
    for (const doc of querySnapshot.docs) {
      const storedData = doc.data();
      const storedVector = storedData.vector;
      const vectorId = storedData.vectorId || doc.id;
      
      // Validate stored vector
      if (!Array.isArray(storedVector) || storedVector.length !== faceVector.length) {
        console.warn(`Invalid stored vector format in document: ${doc.id}`);
        continue;
      }
      
      // Deep comparison of vectors using cosine similarity
      try {
        // Calculate cosine similarity
        const similarity = await calculateCosineSimilarity(faceVector, storedVector);
        
        // Calculate Euclidean distance as secondary check (smaller is more similar)
        const distance = calculateEuclideanDistance(faceVector, storedVector);
        
        // Get stats for the stored vector
        const storedVectorStats = calculateVectorStats(storedVector);
        
        // Calculate Manhattan distance as tertiary check
        const manhattanDistance = calculateManhattanDistance(faceVector, storedVector);
        
        // Calculate vector angle difference in degrees
        const angleDegreeDiff = calculateVectorAngleDifference(faceVector, storedVector);
        
        // Log the comparison with detailed metrics
        allComparisons.push({
          id: vectorId,
          similarity,
          distance,
          manhattanDistance,
          angleDegreeDiff,
          sourceStats: sourceVectorStats,
          storedStats: storedVectorStats
        });
        
        // IF we have an extremely high similarity (almost certainly the same person)
        if (similarity >= STRICTEST_MATCH_THRESHOLD) {
          console.log(`POTENTIAL MATCH found! Similarity: ${similarity.toFixed(4)}, Distance: ${distance.toFixed(4)}`);
          
          // Additional safeguards - Must pass ALL checks for a definite match
          if (distance < MAX_DISTANCE_THRESHOLD && 
              manhattanDistance < 15 && 
              angleDegreeDiff < 5) {
            
            // New: Also verify vector statistics are very close
            if (Math.abs(sourceVectorStats.mean - storedVectorStats.mean) < 0.008 &&
                Math.abs(sourceVectorStats.stdDev - storedVectorStats.stdDev) < 0.008) {
                
              console.log("Match confirmed by multiple metrics! This is definitely an existing user.");
              
              // Log all comparisons for debugging before returning
              logComparisonsSummary(allComparisons);
              return true;
            }
          } else {
            console.log(`High similarity but failed secondary distance checks. Treating as new user for safety.`);
          }
        }
        
        // Check for high confidence matches with second threshold (now with more checks)
        if (similarity >= STRICT_MATCH_THRESHOLD) {
          console.log(`Possible match (high confidence): ${similarity.toFixed(4)}`);
          
          // For high confidence, verify with additional metrics
          if (distance < MAX_SECONDARY_DISTANCE && 
              manhattanDistance < 20 &&
              angleDegreeDiff < 8 &&
              Math.abs(sourceVectorStats.mean - storedVectorStats.mean) < 0.009 &&
              Math.abs(sourceVectorStats.stdDev - storedVectorStats.stdDev) < 0.009) {
            
            console.log("Match confirmed by multiple metrics! This is very likely an existing user.");
            
            // Log all comparisons for debugging before returning
            logComparisonsSummary(allComparisons);
            return true;
          }
        }
      } catch (error) {
        console.warn(`Error comparing vector ${vectorId}:`, error);
      }
    }
    
    // If we get here, no definite matches were found
    logComparisonsSummary(allComparisons);
    console.log(`No matches found after checking ${totalVectors} vectors with strict criteria.`);
    return false;
  } catch (error) {
    console.error('Error in face vector verification:', error);
    throw new Error(`Face verification failed: ${error.message}`);
  }
};

// Helper function to log comparison summary
function logComparisonsSummary(comparisons) {
  if (comparisons.length === 0) {
    console.log('No comparisons to summarize');
    return;
  }
  
  // Sort by similarity (highest first)
  comparisons.sort((a, b) => b.similarity - a.similarity);
  
  console.log('===== FACE COMPARISON SUMMARY =====');
  console.log(`Total comparisons: ${comparisons.length}`);
  console.log('Top 3 closest matches:');
  
  for (let i = 0; i < Math.min(3, comparisons.length); i++) {
    const c = comparisons[i];
    console.log(`${i+1}. ID: ${c.id}, Similarity: ${c.similarity.toFixed(4)}, Euclidean: ${c.distance.toFixed(4)}, Manhattan: ${c.manhattanDistance?.toFixed(2) || 'N/A'}, AngleDiff: ${c.angleDegreeDiff?.toFixed(2) || 'N/A'}°`);
    
    // Add vector statistics comparison for top match
    if (i === 0) {
      const sourceStats = c.sourceStats;
      const storedStats = c.storedStats;
      if (sourceStats && storedStats) {
        console.log(`  Source Stats - Mean: ${sourceStats.mean.toFixed(4)}, StdDev: ${sourceStats.stdDev.toFixed(4)}`);
        console.log(`  Stored Stats - Mean: ${storedStats.mean.toFixed(4)}, StdDev: ${storedStats.stdDev.toFixed(4)}`);
        console.log(`  Diff - Mean: ${Math.abs(sourceStats.mean - storedStats.mean).toFixed(4)}, StdDev: ${Math.abs(sourceStats.stdDev - storedStats.stdDev).toFixed(4)}`);
      }
    }
  }
  
  console.log('=================================');
}

// Calculate vector statistics for additional verification
function calculateVectorStats(vector) {
  const sum = vector.reduce((acc, val) => acc + val, 0);
  const mean = sum / vector.length;
  
  const squaredDiffs = vector.map(val => Math.pow(val - mean, 2));
  const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / vector.length;
  const stdDev = Math.sqrt(variance);
  
  return {
    mean,
    stdDev,
    min: Math.min(...vector),
    max: Math.max(...vector),
    length: vector.length
  };
}

// Calculate Euclidean distance between two vectors (smaller = more similar)
function calculateEuclideanDistance(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let sumOfSquaredDifferences = 0;
  for (let i = 0; i < vector1.length; i++) {
    const diff = vector1[i] - vector2[i];
    sumOfSquaredDifferences += diff * diff;
  }
  
  return Math.sqrt(sumOfSquaredDifferences);
}

// Calculate cosine similarity using a more rigorous approach
async function calculateCosineSimilarity(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let dotProduct = 0;
  let mag1 = 0;
  let mag2 = 0;
  
  for (let i = 0; i < vector1.length; i++) {
    dotProduct += vector1[i] * vector2[i];
    mag1 += Math.pow(vector1[i], 2);
    mag2 += Math.pow(vector2[i], 2);
  }
  
  mag1 = Math.sqrt(mag1);
  mag2 = Math.sqrt(mag2);
  
  if (mag1 === 0 || mag2 === 0) {
    throw new Error('Zero magnitude vector encountered');
  }
  
  const similarity = dotProduct / (mag1 * mag2);
  
  // Cosine similarity should be between -1 and 1
  if (similarity < -1 || similarity > 1) {
    console.warn('Abnormal cosine similarity computed:', similarity);
    return Math.max(-1, Math.min(1, similarity)); // Clamp to valid range
  }
  
  return similarity;
}

// Calculate Manhattan distance between two vectors (sum of absolute differences)
function calculateManhattanDistance(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let sumOfAbsoluteDifferences = 0;
  for (let i = 0; i < vector1.length; i++) {
    sumOfAbsoluteDifferences += Math.abs(vector1[i] - vector2[i]);
  }
  
  return sumOfAbsoluteDifferences;
}

// Calculate the angle between two vectors in degrees
function calculateVectorAngleDifference(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let dotProduct = 0;
  let mag1 = 0;
  let mag2 = 0;
  
  for (let i = 0; i < vector1.length; i++) {
    dotProduct += vector1[i] * vector2[i];
    mag1 += Math.pow(vector1[i], 2);
    mag2 += Math.pow(vector2[i], 2);
  }
  
  mag1 = Math.sqrt(mag1);
  mag2 = Math.sqrt(mag2);
  
  if (mag1 === 0 || mag2 === 0) {
    throw new Error('Zero magnitude vector encountered');
  }
  
  const cosine = dotProduct / (mag1 * mag2);
  // Clamp to handle floating point errors
  const clampedCosine = Math.max(-1, Math.min(1, cosine));
  const angleRadians = Math.acos(clampedCosine);
  const angleDegrees = angleRadians * (180 / Math.PI);
  
  return angleDegrees;
}

// Function to check if database is accessible
export const checkDatabaseAccess = async () => {
  try {
    console.log('Checking database access...');
    const faceVectorsRef = collection(db, 'faceVectors');
    
    // Try to get just one document to verify connectivity
    const q = query(faceVectorsRef, where('test', '==', 'test'));
    const querySnapshot = await getDocs(q);
    
    console.log('Database access check succeeded');
    return true;
  } catch (error) {
    // If there's a permission error, the database is accessible but we don't have permission
    if (error.code && (error.code === 'permission-denied' || error.code === 'resource-exhausted')) {
      console.log('Database is accessible (permission error indicates DB exists)');
      return true;
    }
    
    console.error('Database access check failed:', error);
    return false;
  }
};

// Helper function for testing - clears verification state
export const resetVerificationState = async () => {
  try {
    // This function would typically be used in development only
    console.log('Verification state reset requested');
    
    // In a production environment, you might want to clear cookies/localStorage
    localStorage.removeItem('face_verification_state');
    
    return true;
  } catch (error) {
    console.error('Error resetting verification state:', error);
    return false;
  }
};