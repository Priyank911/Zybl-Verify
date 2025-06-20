import faceapi from './faceApiBrowserAdapter';
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, getDocs, query, where } from 'firebase/firestore';
import { firebaseConfig, ENCRYPTION_KEY } from '../config/firebase.config';

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Function to generate face vector from detection data
export const generateFaceVector = (detection) => {
  if (!detection || !detection.descriptor) {
    throw new Error('Invalid face detection data');
  }
  
  // Check that descriptor is a valid Float32Array
  if (!(detection.descriptor instanceof Float32Array) || detection.descriptor.length === 0) {
    throw new Error('Invalid descriptor format');
  }
  
  // Validate descriptor values (should be normalized)
  let sum = 0;
  let hasInvalidValues = false;
  
  for (let i = 0; i < detection.descriptor.length; i++) {
    const value = detection.descriptor[i];
    
    // Check for invalid values
    if (isNaN(value) || !isFinite(value)) {
      hasInvalidValues = true;
      break;
    }
    
    sum += value * value;
  }
  
  // Verify that the vector is normalized (sum of squares should be close to 1.0)
  if (hasInvalidValues || Math.abs(Math.sqrt(sum) - 1.0) > 0.1) {
    console.warn('Face descriptor is not properly normalized:', {
      sum: sum,
      length: detection.descriptor.length,
      norm: Math.sqrt(sum)
    });
    // Normalize the vector anyway
    const norm = Math.sqrt(sum);
    if (norm > 0) {
      return Array.from(detection.descriptor).map(x => x / norm);
    }
  }
  
  // Convert to regular array for firebase storage
  return Array.from(detection.descriptor);
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

// Function to store face vector in Firebase
export const storeFaceVector = async (faceVector) => {
  try {
    if (!Array.isArray(faceVector)) {
      throw new Error('Invalid face vector format');
    }

    // Double-check if this face is already in the database with strict criteria
    console.log('Final verification before storing new face...');
    const existingStrict = await checkExistingFaceVector(faceVector);
    if (existingStrict) {
      console.log('Face already exists in database with high confidence, skipping storage');
      return true;
    }

    const faceVectorsRef = collection(db, 'faceVectors');
    const timestamp = new Date().toISOString();
    const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const vectorId = `face_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    console.log(`Storing new face vector with ID: ${vectorId} for user: ${userId}`);
    
    // Store the primary face vector
    await addDoc(faceVectorsRef, {
      vector: faceVector,
      timestamp: timestamp,
      vectorId: vectorId,
      userId: userId,
      isPrimary: true,
      checksum: calculateVectorChecksum(faceVector)
    });

    console.log('Primary face vector successfully stored in database');
    return true;
  } catch (error) {
    console.error('Error storing face vector:', error);
    throw new Error('Failed to store face vector');
  }
};

// Helper function to calculate a simple checksum for a vector
const calculateVectorChecksum = (vector) => {
  if (!Array.isArray(vector)) return '0';
  
  // Sum all values in the vector and create a hash
  const sum = vector.reduce((acc, val) => acc + Number(val), 0);
  return sum.toString(16) + '_' + vector.length;
}

// Function to check for existing face vectors
export const checkExistingFaceVector = async (faceVector) => {
  try {
    if (!Array.isArray(faceVector)) {
      throw new Error('Invalid face vector format');
    }

    console.log('Checking for existing face vectors...');
    const faceVectorsRef = collection(db, 'faceVectors');
    const q = query(faceVectorsRef);
    const querySnapshot = await getDocs(q);
    
    console.log(`Found ${querySnapshot.docs.length} existing vectors in database`);
    
    // If no vectors exist in database, return false
    if (querySnapshot.docs.length === 0) {
      console.log('No existing vectors found in database');
      return false;
    }
    
    let highestSimilarity = 0;
    let bestMatchId = null;
    let matchCount = 0;
    
    // Critical: Use multiple thresholds for different confidence levels
    const DEFINITE_MATCH_THRESHOLD = 0.92;  // Extremely high confidence match
    const POSSIBLE_MATCH_THRESHOLD = 0.85;  // High confidence match, needs confirmation
    const MIN_VECTORS_FOR_CONFIRMATION = 1; // Require at least this many matches above threshold
    
    // Track all similarities for logging
    const allSimilarities = [];
    
    for (const doc of querySnapshot.docs) {
      const storedData = doc.data();
      const storedVector = storedData.vector;
      const vectorId = storedData.vectorId || doc.id;
      
      if (!Array.isArray(storedVector) || storedVector.length !== faceVector.length) {
        console.warn(`Invalid stored vector format in document: ${doc.id}. Expected array of length ${faceVector.length}`);
        continue;
      }

      try {
        const similarity = calculateSimilarity(faceVector, storedVector);
        allSimilarities.push({ id: vectorId, similarity });
        
        // Track highest similarity found for debugging
        if (similarity > highestSimilarity) {
          highestSimilarity = similarity;
          bestMatchId = vectorId;
        }
        
        // Instant match for extremely high confidence
        if (similarity >= DEFINITE_MATCH_THRESHOLD) {
          console.log(`DEFINITE MATCH FOUND! Similarity: ${similarity.toFixed(4)} (Threshold: ${DEFINITE_MATCH_THRESHOLD}) with vector ${vectorId}`);
          return true;
        }
        
        // Count possible matches for multiple-confirmation approach
        if (similarity >= POSSIBLE_MATCH_THRESHOLD) {
          matchCount++;
        }
      } catch (error) {
        console.warn(`Error comparing with vector ${vectorId}:`, error);
        continue;
      }
    }
    
    // Sort and log all similarities for debugging
    allSimilarities.sort((a, b) => b.similarity - a.similarity);
    console.log('All similarity scores:', 
      allSimilarities.map(s => `${s.id}: ${s.similarity.toFixed(4)}`).join(', ')
    );
    
    // Match if we have enough high-confidence matches
    if (matchCount >= MIN_VECTORS_FOR_CONFIRMATION) {
      console.log(`CONFIRMED MATCH! Found ${matchCount} vectors above threshold ${POSSIBLE_MATCH_THRESHOLD}`);
      return true;
    }
    
    console.log(`No definite match found. Highest similarity: ${highestSimilarity.toFixed(4)} with vector ${bestMatchId}`);
    return false;
  } catch (error) {
    console.error('Error checking existing face vector:', error);
    throw new Error(`Failed to check existing face vector: ${error.message}`);
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