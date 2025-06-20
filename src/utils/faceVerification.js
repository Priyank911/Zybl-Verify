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
      console.error('Invalid vector format:', { vector1, vector2 });
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

    const similarity = dotProduct / (norm1 * norm2);
    return similarity;
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

    const faceVectorsRef = collection(db, 'faceVectors');
    const timestamp = new Date().toISOString();
    
    await addDoc(faceVectorsRef, {
      vector: faceVector,
      timestamp: timestamp
    });

    return true;
  } catch (error) {
    console.error('Error storing face vector:', error);
    throw new Error('Failed to store face vector');
  }
};

// Function to check for existing face vectors
export const checkExistingFaceVector = async (faceVector) => {
  try {
    if (!Array.isArray(faceVector)) {
      throw new Error('Invalid face vector format');
    }

    console.log('Checking for existing face vector...');
    const faceVectorsRef = collection(db, 'faceVectors');
    const q = query(faceVectorsRef);
    const querySnapshot = await getDocs(q);
    
    console.log('Found', querySnapshot.docs.length, 'existing vectors');
    
    for (const doc of querySnapshot.docs) {
      const storedData = doc.data();
      const storedVector = storedData.vector;
      
      if (!Array.isArray(storedVector)) {
        console.warn('Invalid stored vector format in document:', doc.id);
        continue;
      }

      try {
        const similarity = calculateSimilarity(faceVector, storedVector);
        console.log('Vector similarity:', similarity);
        
        // If similarity is above threshold, consider it a match
        if (similarity > 0.6) {
          return true;
        }
      } catch (error) {
        console.warn('Error comparing vectors:', error);
        continue;
      }
    }
    
    return false;
  } catch (error) {
    console.error('Error checking existing face vector:', error);
    throw new Error('Failed to check existing face vector');
  }
}; 