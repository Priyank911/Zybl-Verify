import React, { useEffect, useRef, useState } from 'react';
import faceapi from '../utils/faceApiBrowserAdapter';
import { generateFaceVector, encryptFaceVector, storeFaceVector, checkExistingFaceVector, calculateSimilarity, checkDatabaseAccess } from '../utils/faceVerification';
import { getFirestore, collection, addDoc, getDocs, query, where } from 'firebase/firestore';
import { initializeApp } from 'firebase/app';
import { firebaseConfig } from '../config/firebase.config';

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

const FaceDetection = ({ walletAddress, onVerificationComplete }) => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [blinkCount, setBlinkCount] = useState(0);
  const [currentEAR, setCurrentEAR] = useState(0);
  const [headVerification, setHeadVerification] = useState({
    left: false,
    right: false
  });
  const headVerificationRef = useRef({
    left: false,
    right: false
  });
  const [currentHeadTilt, setCurrentHeadTilt] = useState(0);
  const [verificationComplete, setVerificationComplete] = useState(false);
  const [eyesClosed, setEyesClosed] = useState(false);
  const [message, setMessage] = useState('Please blink twice (0/2)');
  const [lastBlinkTime, setLastBlinkTime] = useState(0);
  const [blinkStatus, setBlinkStatus] = useState('EYES OPEN');
  const [debugInfo, setDebugInfo] = useState('');
  const blinkCooldown = 1000;
  const blinkDoneRef = useRef(false);
  const [isCheckingExisting, setIsCheckingExisting] = useState(false);
  const [verificationError, setVerificationError] = useState(null);
  const detectionIntervalRef = useRef(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 720, height: 560 });
  const [isAlreadyVerified, setIsAlreadyVerified] = useState(false);
  const [isReturningUser, setIsReturningUser] = useState(false);
  const faceVectorRef = useRef(null);
  const initialCheckDoneRef = useRef(false);

  // New states to track verification progress
  const [verificationStep, setVerificationStep] = useState('blink');
  const [stepsCompleted, setStepsCompleted] = useState({
    blink: false,
    turnLeft: false,
    turnRight: false,
    faceStored: false
  });
  const [verification, setVerification] = useState({
    blinks: 0,
    requiredBlinks: 2,
    leftTurn: false,
    rightTurn: false,
    verificationStrength: 0 // 0-100 percentage of confidence
  });
  const [verificationSuccess, setVerificationSuccess] = useState(false);
  // Add a state to track if initial database check is complete
  const [initialDbCheckComplete, setInitialDbCheckComplete] = useState(false);
  const [isLoadingDatabase, setIsLoadingDatabase] = useState(true);
  const [hasCheckedDatabase, setHasCheckedDatabase] = useState(false);
  // State to show temporary recognition feedback
  const [showRecognitionFeedback, setShowRecognitionFeedback] = useState(false);

  // Load models and start video
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models'
          )
        ]);
        startVideo();
      } catch (error) {
        console.error('Error loading models:', error);
      }
    };

    loadModels();

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, []);

  // Handle video stream
  const startVideo = () => {
    // Detect if on mobile (portrait)
    const isMobile = window.innerWidth <= 600;
    const mobileDimensions = { width: 360, height: 480 };
    const desktopDimensions = { width: 720, height: 560 };
    const chosenDimensions = isMobile ? mobileDimensions : desktopDimensions;
    setVideoDimensions(chosenDimensions);
    navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: chosenDimensions.width },
        height: { ideal: chosenDimensions.height },
        facingMode: 'user'
      } 
    })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Ensure video starts playing
          videoRef.current.play().catch(err => {
            console.error('Error playing video:', err);
          });
        }
      })
      .catch(err => {
        console.error('Error accessing webcam:', err);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setVerificationError('Camera access was denied. Please allow camera permissions in your browser settings and reload the page. If you are on mobile, close any overlays or bubbles from other apps and try again.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setVerificationError('No camera device found. Please connect a camera and try again.');
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          setVerificationError('Camera is already in use by another application. Please close other apps that might be using the camera and try again.');
        } else {
          setVerificationError('Error accessing webcam. Please ensure you have granted camera permissions and no other apps are using the camera.');
        }
      });
  };
  // Setup face detection after initial database check
  useEffect(() => {
    // Don't start main detection until initial database check is complete
    if (!initialDbCheckComplete || !videoRef.current || !canvasRef.current) return;
    
    // Skip face detection if already verified
    if (isAlreadyVerified || verificationSuccess) {
      console.log('User is already verified, skipping verification process');
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    const handleVideoPlay = () => {
      console.log('Video started playing');
      canvas.width = videoDimensions.width;
      canvas.height = videoDimensions.height;
    };

    video.addEventListener('play', handleVideoPlay);
    video.addEventListener('loadedmetadata', () => {
      console.log('Video metadata loaded');
      video.play().catch(err => {
        console.error('Error playing video after metadata loaded:', err);
      });
    });

    const detectFaces = async () => {
      if (!video || !canvas || video.readyState !== 4) {
        console.log('Video not ready:', video?.readyState);
        return;
      }
      
      try {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceExpressions()
          .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, {
          width: videoDimensions.width,
          height: videoDimensions.height
        });

        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
        
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

        if (detections && detections.length > 0) {
          // Always update the current face vector for future use
          if (detections[0].descriptor) {
            faceVectorRef.current = generateFaceVector(detections[0]);
          }          // Check if user is already verified (only if not already checked)
          if (!initialCheckDoneRef.current && !isAlreadyVerified) {
            const vectorQuality = checkFaceDetectionQuality(detections[0]);
            
            // Only check high-quality face vectors
            if (vectorQuality && vectorQuality.isGoodQuality) {
              await checkForExistingVerification(detections);
              
              // If user is verified, immediately show success UI
              if (isAlreadyVerified) {
                // Mark check as done
                initialCheckDoneRef.current = true;
                return; // Skip the rest of the verification process
              }
            }
          }

          const landmarks = detections[0].landmarks;
          const leftEye = landmarks.getLeftEye();
          const rightEye = landmarks.getRightEye();
          const nose = landmarks.getNose();
          const jawOutline = landmarks.getJawOutline();

          const leftEAR = calculateEAR(leftEye);
          const rightEAR = calculateEAR(rightEye);
          const averageEAR = (leftEAR + rightEAR) / 2.0;
          
          if (Math.abs(currentEAR - averageEAR) > 0.01) {
            setCurrentEAR(averageEAR);
          }

          const isBlinking = averageEAR < 0.280;
          const currentTime = Date.now();
          
          if (isBlinking !== (blinkStatus === 'BLINK DETECTED')) {
            setBlinkStatus(isBlinking ? 'BLINK DETECTED' : 'EYES OPEN');
          }

          // Debug information
          context.font = '16px Arial';
          context.fillStyle = '#000000';
          context.fillText(`Current EAR: ${averageEAR.toFixed(3)} (Threshold: 0.290)`, 10, canvas.height - 60);
          context.fillText(`Left Eye: ${leftEAR.toFixed(3)} | Right Eye: ${rightEAR.toFixed(3)}`, 10, canvas.height - 40);
          context.fillText(`Status: ${isBlinking ? 'BLINK DETECTED' : 'EYES OPEN'}`, 10, canvas.height - 20);          // Blink detection
          if (isBlinking && !eyesClosed && currentTime - lastBlinkTime > blinkCooldown && !stepsCompleted.blink) {
            setEyesClosed(true);
            setLastBlinkTime(currentTime);
            
            const newBlinkCount = Math.min(blinkCount + 1, 2);
            setBlinkCount(newBlinkCount);
            
            // Update verification state
            setVerification(prev => ({
              ...prev,
              blinks: newBlinkCount,
              verificationStrength: prev.verificationStrength + 10 // Increase confidence
            }));
            
            if (newBlinkCount === 2) {
              console.log('Blink verification complete!');
              setStepsCompleted(prev => ({ ...prev, blink: true }));
              setVerificationStep('turnLeft');
              setMessage('Blink verification complete! Now turn your head left');
              blinkDoneRef.current = true;
            } else {
              setMessage(`Please blink twice (${newBlinkCount}/2)`);
            }
          } else if (!isBlinking && eyesClosed) {
            setEyesClosed(false);
          }          // Head movement detection
          if ((stepsCompleted.blink || blinkDoneRef.current) && !verificationComplete) {
            const jawCenter = jawOutline[8];
            const noseTop = nose[0];

            if (!jawCenter || !noseTop || !leftEye || !rightEye) {
              return;
            }

            const leftEyeCenter = {
              x: leftEye.reduce((sum, point) => sum + point.x, 0) / leftEye.length,
              y: leftEye.reduce((sum, point) => sum + point.y, 0) / leftEye.length
            };
            const rightEyeCenter = {
              x: rightEye.reduce((sum, point) => sum + point.x, 0) / rightEye.length,
              y: rightEye.reduce((sum, point) => sum + point.y, 0) / rightEye.length
            };

            const eyeAngle = Math.atan2(rightEyeCenter.y - leftEyeCenter.y, rightEyeCenter.x - leftEyeCenter.x);
            const headTilt = Math.sin(eyeAngle);
            
            if (Math.abs(currentHeadTilt - headTilt) > 0.01) {
              setCurrentHeadTilt(headTilt);
            }

            const newDebugInfo = `Tilt: ${headTilt.toFixed(3)} | Left: ${headVerificationRef.current.left} | Right: ${headVerificationRef.current.right}`;
            if (newDebugInfo !== debugInfo) {
              setDebugInfo(newDebugInfo);
            }

            context.font = '16px Arial';
            context.fillStyle = '#000000';
            context.fillText(`Head Tilt: ${headTilt.toFixed(3)}`, 10, canvas.height - 80);
            context.fillText(`Debug: ${newDebugInfo}`, 10, canvas.height - 60);

            // LEFT HEAD MOVEMENT
            if (!headVerificationRef.current.left && headTilt < -0.10 && verificationStep === 'turnLeft') {
              headVerificationRef.current.left = true;
              setHeadVerification(prev => ({ ...prev, left: true }));
              
              // Update verification state
              setVerification(prev => ({
                ...prev,
                leftTurn: true,
                verificationStrength: prev.verificationStrength + 15 // Increase confidence
              }));
              
              setStepsCompleted(prev => ({ ...prev, turnLeft: true }));
              setVerificationStep('turnRight');
              setMessage('Great! Now turn your head right');
            }
            
            // RIGHT HEAD MOVEMENT
            if (headVerificationRef.current.left && !headVerificationRef.current.right && headTilt > 0.10 && verificationStep === 'turnRight') {
              // Mark right turn as completed
              headVerificationRef.current.right = true;
              setHeadVerification(prev => ({ ...prev, right: true }));
              
              // Update verification state
              setVerification(prev => ({
                ...prev,
                rightTurn: true,
                verificationStrength: 100 // Set directly to 100% since all steps are done
              }));
              
              setStepsCompleted(prev => ({ ...prev, turnRight: true }));
              
              // Skip the storeData step and immediately mark verification as complete
              setVerificationStep('complete');
              setVerificationComplete(true);
              setVerificationSuccess(true);
              
              // Show success message immediately
              setMessage('Verification successful! Storing face data...');
              
              // Directly call handleVerificationComplete with no delay
              handleVerificationComplete();
            } else if (verificationStep === 'turnLeft') {
              setMessage(`Turn your head LEFT (Current tilt: ${headTilt.toFixed(3)})`);
            } else if (verificationStep === 'turnRight') {
              setMessage(`Turn your head RIGHT (Current tilt: ${headTilt.toFixed(3)})`);
            }
          }

          // Draw status text
          context.font = '24px Arial';
          context.fillStyle = '#000000';
          context.fillText(message, 10, 30);
          context.font = '16px Arial';
          context.fillText(`Current Tilt: ${currentHeadTilt.toFixed(3)}`, 10, 60);
          context.fillText(`Left Verified: ${headVerificationRef.current.left}`, 10, 90);
          context.fillText(`Right Verified: ${headVerificationRef.current.right}`, 10, 120);
        }
      } catch (error) {
        console.error('Error in face detection:', error);
      }
    };

    detectionIntervalRef.current = setInterval(detectFaces, 100);    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
      video.removeEventListener('play', handleVideoPlay);
      video.removeEventListener('loadedmetadata', () => {});
    };
  }, [
    currentEAR, 
    blinkStatus, 
    eyesClosed, 
    lastBlinkTime, 
    blinkCount, 
    verificationComplete, 
    debugInfo, 
    message, 
    videoDimensions, 
    isAlreadyVerified, 
    verificationStep, 
    stepsCompleted,
    verificationSuccess
  ]);
  // Check if user is already verified when face is detected
  const checkForExistingVerification = async (detections) => {
    if (!detections || detections.length === 0) {
      console.log('No detections available for verification check');
      return;
    }

    // Only perform this check once per session
    if (initialCheckDoneRef.current) {
      return;
    }

    try {
      console.log('Starting multi-step verification check process...');
      // Set flag to prevent duplicate checks
      initialCheckDoneRef.current = true;
      
      // Step 1: Get initial face vector
      const faceVector = generateFaceVector(detections[0]);
      
      // Store for future use
      faceVectorRef.current = faceVector; 
      
      setIsCheckingExisting(true);
      setMessage('Checking if you are already verified...');
      
      // Step 2: Get multiple face detections to ensure stability
      const faceVectors = [];
      let stableDetectionCount = 0;
      
      // Collect multiple face detections to ensure stability
      for (let i = 0; i < 3; i++) {
        try {
          // Wait a moment to get different frames
          await new Promise(resolve => setTimeout(resolve, 300));
          
          const newDetections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptors();
            
          if (newDetections && newDetections.length > 0) {
            const newFaceVector = generateFaceVector(newDetections[0]);
            faceVectors.push(newFaceVector);
            
            // Check if this detection is consistent with our original
            const selfSimilarity = calculateSimilarity(faceVector, newFaceVector);
            console.log(`Face stability check ${i+1}: similarity = ${selfSimilarity.toFixed(4)}`);
            
            if (selfSimilarity > 0.92) {
              stableDetectionCount++;
            }
          }
        } catch (err) {
          console.warn(`Error during stability check ${i+1}:`, err);
        }
      }
        // Step 3: Verify stability before checking database
      if (stableDetectionCount < 3) {
        console.warn(`Face detection not stable enough (${stableDetectionCount}/3 stable detections)`);
        setMessage('Please keep your face centered and still');
        setIsCheckingExisting(false);
        return;
      }
        // Enhanced: Check for self-consistency between all vectors
      let consistentVectors = true;
      for (let i = 0; i < faceVectors.length - 1; i++) {
        for (let j = i + 1; j < faceVectors.length; j++) {
          const selfSimilarity = calculateSimilarity(faceVectors[i], faceVectors[j]);
          console.log(`Self-consistency check between vectors ${i} and ${j}: ${selfSimilarity.toFixed(4)}`);
          
          if (selfSimilarity < 0.97) { // Require extremely high self-consistency
            consistentVectors = false;
            console.warn(`Inconsistent face vectors detected: ${selfSimilarity.toFixed(4)}`);
          }
        }
      }
      
      if (!consistentVectors) {
        console.warn("Face vectors are not consistent enough between frames");
        setMessage('Please hold your face steady and look directly at the camera');
        setIsCheckingExisting(false);
        return;
      }
      
      console.log('Face detection is stable and consistent. Proceeding with verification check...');
      
      // Step 4: Check all collected vectors against the database with double check
      // We'll only count it as a match if multiple vectors match the SAME person
      let matchCount = 0;
      
      // First, try checking with the strictest setting
      for (const vector of faceVectors) {
        try {
          const exists = await checkExistingFaceVector(vector);
          if (exists) {
            matchCount++;
          }
        } catch (error) {
          console.warn('Error during database check:', error);
        }
      }
        // Step 5: Only verify if MOST checks succeed (heightened requirement)
      if (matchCount >= faceVectors.length - 1) {
        console.log(`User verified with high confidence (${matchCount}/${faceVectors.length} checks passed)`);
        setIsAlreadyVerified(true);
        setIsReturningUser(true);
        setVerificationSuccess(true);
        setMessage('Welcome back! Your face has been recognized.');
        setVerificationComplete(true);
        setStepsCompleted({
          blink: true,
          turnLeft: true,
          turnRight: true,
          faceStored: true
        });
        
        // Update verification state
        setVerification(prev => ({
          ...prev,
          blinks: 2,
          leftTurn: true,
          rightTurn: true,
          verificationStrength: 100
        }));
        
        // Show recognition feedback
        setShowRecognitionFeedback(true);
        
        // Stop the detection process
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
        
        // Call the completion callback
        if (onVerificationComplete) {
          onVerificationComplete();
        }
      } else {
        console.log(`User not verified (only ${matchCount}/${faceVectors.length} checks passed)`);
        setMessage('Please blink twice (0/2) to begin verification');
      }
    } catch (error) {
      console.error('Error during verification check:', error);
      setMessage('Error checking verification. Please try again.');
    } finally {
      setIsCheckingExisting(false);
    }
  };
  // Check for existing verification on first solid face detection
  useEffect(() => {
    let initialCheckTimer = null;
    let checkAttempts = 0;
    const MAX_CHECK_ATTEMPTS = 15; // Increased from 5 to allow more chances to recognize
    
    if (!initialCheckDoneRef.current) {
      initialCheckTimer = setInterval(async () => {
        // Give up after several attempts
        if (checkAttempts >= MAX_CHECK_ATTEMPTS || initialCheckDoneRef.current) {
          clearInterval(initialCheckTimer);
          return;
        }
        
        checkAttempts++;
        
        try {
          // Only attempt if we have a valid face vector
          if (faceVectorRef.current) {
            console.log(`Attempt ${checkAttempts}: Checking for existing verification`);
            
            // Check if this face is already verified
            const exists = await checkExistingFaceVector(faceVectorRef.current);
            
            if (exists) {
              console.log('User verified from initial check! Updating state...');
              setIsAlreadyVerified(true);
              setIsReturningUser(true);
              setVerificationSuccess(true);
              setVerificationComplete(true);
              
              setStepsCompleted({
                blink: true,
                turnLeft: true,
                turnRight: true,
                faceStored: true
              });
              
              setMessage('Welcome back! Your face has been recognized.');
              
              // Call completion callback
              if (onVerificationComplete) {
                onVerificationComplete();
              }
              
              // Mark check as done
              initialCheckDoneRef.current = true;
              clearInterval(initialCheckTimer);
            } else if (checkAttempts === MAX_CHECK_ATTEMPTS) {
              // If we've tried the maximum number of times and still no match,
              // let's proceed with verification
              console.log('No existing verification found after multiple attempts');
              initialCheckDoneRef.current = true;
              
              // Call completion callback
              if (onVerificationComplete) {
                onVerificationComplete();
              }
              
              clearInterval(initialCheckTimer);
            }
            
            // Mark as checked even if not verified
            initialCheckDoneRef.current = true;
            clearInterval(initialCheckTimer);
          }
        } catch (error) {
          console.error('Error in initial verification check:', error);
        }
      }, 2000); // Check every 2 seconds
    }
    
    return () => {
      if (initialCheckTimer) {
        clearInterval(initialCheckTimer);
      }
    };
  }, []);

  // Initial check on component mount - first check if database is available
  useEffect(() => {
    const checkDatabaseConnection = async () => {
      try {
        setIsLoadingDatabase(true);
        setMessage('Connecting to face database...');
        
        // Simple check to see if we can access Firestore
        const checkResult = await checkDatabaseAccess();
        
        if (checkResult) {
          console.log('Successfully connected to face database');
          setIsLoadingDatabase(false);
          setMessage('Please look at the camera');
        } else {
          console.error('Could not connect to face database');
          setIsLoadingDatabase(false);
          setVerificationError('Could not connect to face verification database');
        }
      } catch (error) {
        console.error('Database connection error:', error);
        setIsLoadingDatabase(false);
        setVerificationError('Error connecting to verification system');
      }
    };
    
    checkDatabaseConnection();
  }, []);

  // Two-phase face detection and recognition process with improved quality checks
  useEffect(() => {
    // Don't start face recognition until database check is complete
    if (isLoadingDatabase || hasCheckedDatabase || !videoRef.current || !canvasRef.current) {
      return;
    }
    
    let faceCheckCount = 0;
    const maxFaceChecks = 10; // More attempts to ensure quality
    const minRequiredDetections = 5; // More detections for better verification
    let successfulDetections = 0;
    let detectedFaceVectors = [];
    let recognitionTimer = null;
    let lowQualityDetections = 0;
    
    const performFaceRecognition = async () => {
      if (faceCheckCount >= maxFaceChecks || initialDbCheckComplete) {
        clearInterval(recognitionTimer);
        
        // If we didn't get enough good detections, just continue as a new user
        if (successfulDetections < minRequiredDetections && !initialDbCheckComplete) {
          console.log(`Insufficient quality face detections (${successfulDetections}/${minRequiredDetections}). Continuing as new user.`);
          setHasCheckedDatabase(true);
          setInitialDbCheckComplete(true);
          setMessage('Welcome! Please complete verification steps.');
        }
        
        return;
      }
      
      try {
        faceCheckCount++;
        console.log(`Face recognition attempt ${faceCheckCount}/${maxFaceChecks}...`);
        
        // Message to show process is working
        if (faceCheckCount === 1) {
          setMessage('Scanning your face...');
        } else if (faceCheckCount === 3) {
          setMessage('Running face recognition...');
        } else if (faceCheckCount === 5) {
          setMessage('Almost there...');
        }
        
        const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.7 }))
          .withFaceLandmarks()
          .withFaceExpressions()
          .withFaceDescriptors();
          
        if (detections && detections.length === 1) {
          // Quality checks for the face detection
          const detection = detections[0];
          const landmarks = detection.landmarks;
          const expressions = detection.expressions;
            // Check face detection quality factors
          const qualityCheck = checkFaceDetectionQuality(detection);
          
          if (qualityCheck && qualityCheck.isGoodQuality) {
            successfulDetections++;
            console.log(`High quality face detection: ${successfulDetections}/${minRequiredDetections}`);
            
            // IMPORTANT IMPROVEMENT: Try to recognize returning users during the process
            if (faceCheckCount % 2 === 0 && !isAlreadyVerified) {
              // Generate a face vector for checking
              const faceVector = generateFaceVector(detection);
              
              // Try to verify if this is a returning user
              try {
                const exists = await checkExistingFaceVector(faceVector);
                if (exists) {
                  console.log('RETURNING USER RECOGNIZED DURING VERIFICATION PROCESS!');
                  setIsAlreadyVerified(true);
                  setIsReturningUser(true);
                  setVerificationSuccess(true);
                  setVerificationComplete(true);
                  setShowRecognitionFeedback(true);
                  
                  setStepsCompleted({
                    blink: true,
                    turnLeft: true,
                    turnRight: true,
                    faceStored: true
                  });
                  
                  setMessage('Welcome back! Your face has been recognized.');
                  
                  // Call completion callback
                  if (onVerificationComplete) {
                    onVerificationComplete();
                  }
                  
                  // No need to continue checking
                  setInitialDbCheckComplete(true);
                  setHasCheckedDatabase(true);
                  return;
                }
              } catch (error) {
                console.error('Error checking for returning user during process:', error);
              }
            }
            
            try {
              const faceVector = generateFaceVector(detection);
              detectedFaceVectors.push(faceVector);
              
              // Store the latest face vector for future use
              faceVectorRef.current = faceVector;
              
              // When we have enough good detections, check if this face exists in database
              if (successfulDetections >= minRequiredDetections) {
                clearInterval(recognitionTimer);
                await checkFaceInDatabase(detectedFaceVectors);
              }
            } catch (err) {
              console.warn('Error generating face vector:', err);
              lowQualityDetections++;
            }
          } else {            console.log('Low quality face detection, skipping');
            lowQualityDetections++;
          }
        } else if (detections && detections.length > 1) {
          console.log('Multiple faces detected, skipping this frame');
        } else {
          console.log('No face detected in this frame');
        }
      } catch (error) {
        console.error('Error during face recognition:', error);
      }
    };
    
    // Start periodic face recognition checks
    setMessage('Looking for your face...');
    recognitionTimer = setInterval(performFaceRecognition, 500);
    
    return () => {
      if (recognitionTimer) {
        clearInterval(recognitionTimer);
      }
    };  }, [isLoadingDatabase, hasCheckedDatabase]);

  // Helper function to check if face detection is high quality enough for verification
  const checkFaceDetectionQuality = (detection) => {
    if (!detection || !detection.detection || !detection.landmarks || !detection.descriptor) {
      return { isGoodQuality: false, reason: 'Incomplete detection data' };
    }
    
    try {
      // Check confidence score
      const confidence = detection.detection.score;
      if (confidence < 0.85) {
        return { isGoodQuality: false, reason: `Low detection confidence: ${confidence.toFixed(2)}` };
      }
      
      // Check if descriptor has expected length
      if (!(detection.descriptor instanceof Float32Array) || detection.descriptor.length !== 128) {
        return { isGoodQuality: false, reason: 'Invalid descriptor format' };
      }
      
      // Check if face is too small (width < 100px is likely too far away)
      const box = detection.detection.box;
      if (box.width < 100 || box.height < 100) {
        return { isGoodQuality: false, reason: `Face too small: ${box.width}x${box.height}` };
      }
      
      // Check if face is too close to edge of frame
      if (box.x < 10 || box.y < 10) {
        return { isGoodQuality: false, reason: 'Face too close to edge of frame' };
      }
      
      // Check if landmarks are visible (minimum number of points)
      const landmarks = detection.landmarks;
      const jawPoints = landmarks.getJawOutline().length;
      const nosePoints = landmarks.getNose().length;
      const eyePoints = landmarks.getLeftEye().length + landmarks.getRightEye().length;
      
      if (jawPoints < 15 || nosePoints < 5 || eyePoints < 10) {
        return { isGoodQuality: false, reason: 'Incomplete facial landmarks' };
      }
      
      // Generate vector and check its statistics
      const vector = generateFaceVector(detection);
      const stats = calculateVectorStats(vector);
      
      // Check if the vector has good statistical properties (based on typical values)
      if (Math.abs(stats.mean) > 0.2 || stats.stdDev < 0.05) {
        return { isGoodQuality: false, reason: 'Unusual vector statistics' };
      }
      
      return { isGoodQuality: true, vector: vector };
    } catch (error) {
      console.error('Error checking face quality:', error);
      return { isGoodQuality: false, reason: `Error: ${error.message}` };
    }
  };
  
  // Helper function to calculate vector statistics
  const calculateVectorStats = (vector) => {
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
  };

  // Function to check face vectors against the database and update state
  const checkFaceInDatabase = async (faceVectors) => {
    try {
      setMessage('Checking if you are already verified...');
      setHasCheckedDatabase(true);
      
      console.log(`Checking ${faceVectors.length} high-quality face vectors against database`);
      
      // Only proceed with database check if we have at least 3 consistent face vectors
      if (faceVectors.length < 3) {
        console.warn('Not enough high-quality face vectors to perform reliable verification');
        setInitialDbCheckComplete(true);
        setMessage('Please blink twice (0/2) to begin verification');
        return;
      }
      
      // First check: ensure the face vectors are self-consistent
      let selfConsistencyPassed = true;
      for (let i = 0; i < faceVectors.length - 1; i++) {
        for (let j = i + 1; j < faceVectors.length; j++) {
          const selfSimilarity = calculateSimilarity(faceVectors[i], faceVectors[j]);
          console.log(`Self-consistency check between vectors ${i} and ${j}: ${selfSimilarity.toFixed(4)}`);
          
          if (selfSimilarity < 0.97) { // Require extremely high self-consistency
            selfConsistencyPassed = false;
            console.warn(`Inconsistent face vectors detected: ${selfSimilarity.toFixed(4)}`);
          }
        }
      }
      
      if (!selfConsistencyPassed) {
        console.warn("Face vectors are not consistent enough between frames");
        setInitialDbCheckComplete(true);
        setMessage('Please hold your face steady and look directly at the camera');
        return;
      }
      
      // Now check each vector against the database
      let matchCount = 0;
      let matchedVectors = [];
      
      for (const vector of faceVectors) {
        try {
          const exists = await checkExistingFaceVector(vector);
          if (exists) {
            matchCount++;
            matchedVectors.push(vector);
          }
        } catch (error) {
          console.warn('Error during database check:', error);
        }
      }
      
      // We need MOST vectors to match to consider this a verified user
      // This is stricter than before
      const requiredMatchThreshold = Math.ceil(faceVectors.length * 0.75); // At least 75% must match
      
      if (matchCount >= requiredMatchThreshold) {
        console.log(`User verified with high confidence (${matchCount}/${faceVectors.length} checks passed)`);
        
        // Use the best matching vector for future recognition
        if (matchedVectors.length > 0) {
          // Use the middle vector to avoid extreme samples
          const bestVectorIndex = Math.floor(matchedVectors.length / 2);
          faceVectorRef.current = matchedVectors[bestVectorIndex];
        }        // Mark as verified
        setIsAlreadyVerified(true);
        setIsReturningUser(true);
        setVerificationSuccess(true);
        setMessage('Welcome back! Your face has been recognized.');
        setVerificationComplete(true);
        setStepsCompleted({
          blink: true,
          turnLeft: true,
          turnRight: true,
          faceStored: true
        });
        
        // Update verification state
        setVerification(prev => ({
          ...prev,
          blinks: 2,
          leftTurn: true,
          rightTurn: true,
          verificationStrength: 100
        }));
        
        // Call the completion callback
        if (onVerificationComplete) {
          onVerificationComplete();
        }
      } else {
        console.log(`User not verified (only ${matchCount}/${faceVectors.length} checks passed)`);
        setInitialDbCheckComplete(true);
        setMessage('Please blink twice (0/2) to begin verification');
      }
    } catch (error) {
      console.error('Error during database check:', error);
      setInitialDbCheckComplete(true);
      setMessage('Please blink twice (0/2) to begin verification');
    }
  };

  // Calculate progress percentage for the progress bar
  const getProgress = () => {
    // Count completed steps
    const completedSteps = Object.values(stepsCompleted).filter(Boolean).length;
    const totalSteps = Object.keys(stepsCompleted).length;
    
    // Show 100% progress if already verified or verification is successful
    if (completedSteps === totalSteps || isAlreadyVerified || verificationSuccess) {
      return 1;
    }
    
    // Base progress is the number of completed steps divided by total steps
    const baseProgress = completedSteps / totalSteps;
    
    // Add partial progress for current step
    if (verificationStep === 'blink' && verification.blinks > 0) {
      return baseProgress + (verification.blinks / (verification.requiredBlinks * totalSteps));
    } else if (verificationStep === 'turnLeft' && stepsCompleted.blink) {
      return baseProgress + (verification.leftTurn ? 0.25 : 0) / totalSteps;
    } else if (verificationStep === 'turnRight' && stepsCompleted.turnLeft) {
      return baseProgress + (verification.rightTurn ? 0.25 : 0) / totalSteps;
    }
    
    return baseProgress;
  };

  const calculateEAR = (eye) => {
    try {
      const p2_p6 = euclideanDistance(eye[1], eye[5]);
      const p3_p5 = euclideanDistance(eye[2], eye[4]);
      const p1_p4 = euclideanDistance(eye[0], eye[3]);
      
      if (p1_p4 === 0) return 0.35;
      
      const ear = (p2_p6 + p3_p5) / (2.0 * p1_p4);
      return Math.min(Math.max(ear, 0.1), 0.45);
    } catch (error) {
      console.error('Error calculating EAR:', error);
      return 0.35;
    }
  };

  const euclideanDistance = (point1, point2) => {
    return Math.sqrt(
      Math.pow(point2.x - point1.x, 2) + 
      Math.pow(point2.y - point1.y, 2)
    );
  };  // Store multiple face vectors for the same person to improve recognition
  const storeMultipleFaceVectors = async (faceVectors) => {
    try {
      // Check if we have enough vectors
      if (!faceVectors || faceVectors.length < 3) {
        console.error('Not enough face vectors to store');
        return false;
      }
      
      console.log(`Storing ${faceVectors.length} face vectors for improved recognition`);
      
      // First, check if any of these vectors already exist in the database
      for (const vector of faceVectors) {
        const exists = await checkExistingFaceVector(vector);
        if (exists) {
          console.log('Face already exists in database. No need to store additional vectors.');
          return true;
        }
      }
      
      // Generate a unique user ID to link all vectors
      const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Store each vector with the same userId to link them
      let successCount = 0;
      
      for (let i = 0; i < faceVectors.length; i++) {
        try {
          const vector = faceVectors[i];
          
          // Generate unique vector ID
          const vectorId = `face_${Date.now() + i}_${Math.random().toString(36).substr(2, 9)}`;
            // Calculate vector quality metrics
          const vectorStats = calculateVectorStats(vector);
          
          // Store in Firestore
          const faceVectorsRef = collection(db, 'faceVectors');
          await addDoc(faceVectorsRef, {
            vector: vector,
            timestamp: new Date().toISOString(),
            vectorId: vectorId,
            userId: userId,
            sampleIndex: i, // To track which sample this is
            totalSamples: faceVectors.length,
            vectorStats: {
              mean: vectorStats.mean,
              stdDev: vectorStats.stdDev,
              min: vectorStats.min,
              max: vectorStats.max
            },
            checksum: calculateVectorChecksum(vector)
          });
          
          successCount++;
        } catch (error) {
          console.error(`Error storing face vector ${i}:`, error);
        }
      }
      
      console.log(`Successfully stored ${successCount}/${faceVectors.length} face vectors`);
      return successCount > 0;
    } catch (error) {
      console.error('Error in storeMultipleFaceVectors:', error);
      return false;
    }
  };
  
  // Helper function to calculate a simple checksum for a vector
  const calculateVectorChecksum = (vector) => {
    if (!Array.isArray(vector)) return '0';
    
    // Sum all values in the vector and create a hash
    const sum = vector.reduce((acc, val) => acc + Number(val), 0);
    return sum.toString(16) + '_' + vector.length;
  };

  // Extremely robust verification completion handler
  const handleVerificationComplete = async () => {
    try {
      // If we're already verified, skip the process
      if (isAlreadyVerified) {
        setMessage('You are already verified!');
        setVerificationSuccess(true);
        setVerificationComplete(true);
        setStepsCompleted({
          blink: true,
          turnLeft: true,
          turnRight: true,
          faceStored: true
        });
        if (onVerificationComplete) {
          onVerificationComplete();
        }
        return;
      }

      // Double check if all steps are completed
      const allStepsCompleted = headVerificationRef.current.left && 
                              headVerificationRef.current.right && 
                              blinkCount >= 2;
                              
      if (!allStepsCompleted) {
        console.log("Warning: Not all verification steps are complete");
        setMessage('Please complete all verification steps first');
        return;
      }

      // Immediately update the UI to show processing state
      setVerificationComplete(true);
      setVerificationStep('storeData');
      setMessage('Processing verification...');
      
      // STEP 1: Collect multiple high-quality face vectors
      const faceVectors = [];
      let captureAttempts = 0;
      const maxCaptureAttempts = 5;
      
      setMessage('Capturing face data...');
      
      while (faceVectors.length < 3 && captureAttempts < maxCaptureAttempts) {
        captureAttempts++;
        
        try {
          // Wait briefly between captures to get different frames
          if (captureAttempts > 1) {
            await new Promise(resolve => setTimeout(resolve, 300));
          }
          
          const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptors();

          if (detections && detections.length === 1) {            // Use our quality check function
            const qualityCheck = checkFaceDetectionQuality(detections[0]);
            
            if (qualityCheck && qualityCheck.isGoodQuality) {
              const vector = generateFaceVector(detections[0]);
              faceVectors.push(vector);
              console.log(`Captured high-quality face vector ${faceVectors.length}/3`);
            } else {
              console.log('Low quality detection, skipping');
            }
          }
        } catch (error) {
          console.warn(`Error capturing face vector (attempt ${captureAttempts}):`, error);
        }
      }
      
      // STEP 2: Ensure we have enough quality vectors
      if (faceVectors.length < 2) {
        throw new Error("Could not capture enough high-quality face images. Please try again in better lighting.");
      }
      
      // STEP 3: Check face consistency
      setMessage('Verifying face consistency...');
      let consistencyPassed = true;
      
      for (let i = 0; i < faceVectors.length - 1; i++) {
        for (let j = i + 1; j < faceVectors.length; j++) {
          const similarity = await calculateSimilarity(faceVectors[i], faceVectors[j]);
          if (similarity < 0.95) {
            consistencyPassed = false;
            break;
          }
        }
      }
      
      if (!consistencyPassed) {
        throw new Error("Face verification failed: inconsistent face detections. Please try again.");
      }
      
      // Use the highest quality vector (middle one to avoid extremes)
      const middleIndex = Math.floor(faceVectors.length / 2);
      const faceVector = faceVectors[middleIndex];
      faceVectorRef.current = faceVector; // Store for future use
        // STEP 4: Final check if this face is already in the database
      setMessage('Checking for existing registration...');
      
      // Instead of just storing one vector, store multiple for better recognition
      const storageSuccess = await storeMultipleFaceVectors(faceVectors);
      
      if (!storageSuccess) {
        // Fall back to single vector storage if multiple storage fails
        const singleVectorResult = await storeFaceVector(faceVector);
        if (singleVectorResult.status === 'existing') {
          console.log("Face already exists in database");
          setMessage('This face is already registered in our system!');
        } else {
          setMessage('Face data successfully stored!');
        }
      } else {
        setMessage('Face data successfully stored with multiple samples!');
      }
      
      // Final success state
      setVerificationSuccess(true);
      setStepsCompleted(prev => ({ ...prev, faceStored: true }));
      
      // Update verification strength to 100%
      setVerification(prev => ({
        ...prev,
        verificationStrength: 100
      }));
      
      // Notify parent component
      if (onVerificationComplete) {
        onVerificationComplete();
      }
    } catch (error) {
      console.error('Verification error:', error);
      setVerificationError(error.message || 'Error during verification. Please try again.');
      setVerificationSuccess(false);
      setVerificationComplete(false);
    }
  };

  // Auto-hide recognition feedback after delay
  useEffect(() => {
    let feedbackTimer;
    
    if (showRecognitionFeedback) {
      feedbackTimer = setTimeout(() => {
        setShowRecognitionFeedback(false);
      }, 3000); // Hide after 3 seconds
    }
    
    return () => {
      if (feedbackTimer) {
        clearTimeout(feedbackTimer);
      }
    };
  }, [showRecognitionFeedback]);

  return (
    <div className="face-detection-outer">
      {/* Development controls for testing - hidden in production */}
      {process.env.NODE_ENV === 'development' && (
        <div className="dev-controls">
          <button 
            onClick={() => {
              console.log('Resetting verification state...');
              initialCheckDoneRef.current = false;
              setBlinkCount(0);
              setHeadVerification({ left: false, right: false });
              headVerificationRef.current = { left: false, right: false };
              setVerificationComplete(false);
              setVerificationSuccess(false);
              setIsAlreadyVerified(false);
              setStepsCompleted({
                blink: false,
                turnLeft: false,
                turnRight: false,
                faceStored: false
              });
              setVerificationStep('blink');
              setMessage('Please blink twice (0/2) to begin verification');
            }}
            className="dev-reset-btn"
          >
            Reset Verification (Dev Only)
          </button>
        </div>
      )}

      {/* Status/Error/Progress above video */}
      <div className="face-info-top">
        {isLoadingDatabase && (
          <div className="status-overlay loading">
            <div className="loading-spinner"></div>
            Connecting to verification system...
          </div>
        )}
        
        {/* Only show status if there is no error */}
        {!isLoadingDatabase && !verificationError && message && (
          <div className={`status-overlay ${isAlreadyVerified || verificationSuccess ? 'verified' : ''}`}>
            {message}
          </div>
        )}
        
        {verificationError && (
          <div className="error-message">
            {verificationError}
            <button 
              onClick={() => {
                setVerificationError(null);
                window.location.reload();
              }}
              className="retry-button"
            >
              Retry
            </button>
          </div>
        )}
        
        {/* Progress bar with steps */}
        <div className="verification-progress">
          <div className="progress-bar">
            <div className="progress-bar-inner" style={{ width: `${getProgress() * 100}%` }} />
          </div>
          <div className="progress-steps">
            <div className={`progress-step ${stepsCompleted.blink ? 'completed' : ''} ${verificationStep === 'blink' ? 'active' : ''}`}>
              <div className="step-number">1</div>
              <div className="step-name">Blink</div>
            </div>
            <div className={`progress-step ${stepsCompleted.turnLeft ? 'completed' : ''} ${verificationStep === 'turnLeft' ? 'active' : ''}`}>
              <div className="step-number">2</div>
              <div className="step-name">Left</div>
            </div>
            <div className={`progress-step ${stepsCompleted.turnRight ? 'completed' : ''} ${verificationStep === 'turnRight' ? 'active' : ''}`}>
              <div className="step-number">3</div>
              <div className="step-name">Right</div>
            </div>
            <div className={`progress-step ${stepsCompleted.faceStored ? 'completed' : ''} ${verificationStep === 'storeData' ? 'active' : ''}`}>
              <div className="step-number">4</div>
              <div className="step-name">Store</div>
            </div>
          </div>
        </div>
      </div>

      {/* Video/Canvas area only */}
      <div className="face-canvas-area">
        <video
          ref={videoRef}
          width={videoDimensions.width}
          height={videoDimensions.height}
          autoPlay
          muted
          playsInline
          className="face-video"
        />
        <canvas
          ref={canvasRef}
          width={videoDimensions.width}
          height={videoDimensions.height}
          className="face-canvas"
        />        {/* Show verified overlay when already verified or verification successful */}
        {(isAlreadyVerified || verificationSuccess) && (
          <div className={`verified-overlay ${isReturningUser ? 'returning-user' : ''}`}>
            <div className="verified-icon">✓</div>
            <div className="verified-text">
              {isReturningUser ? 'Welcome Back!' : 'Verification Complete'}
            </div>
            <div className="verified-details">
              {isReturningUser ? (
                <>
                  You have been successfully recognized.<br/>
                  Thank you for returning to our system.
                </>
              ) : (
                <>
                  Your face data has been securely stored.<br/>
                  You will be automatically recognized on future visits.
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Show current verification step instructions */}
        {!isAlreadyVerified && !verificationSuccess && (
          <div className="step-instructions">
            {verificationStep === 'blink' && (
              <div className="instruction-item">
                <div className="instruction-icon">👁️</div>
                <div className="instruction-text">Blink twice {verification.blinks}/2</div>
              </div>
            )}
            {verificationStep === 'turnLeft' && (
              <div className="instruction-item">
                <div className="instruction-icon">👈</div>
                <div className="instruction-text">Turn your head LEFT</div>
              </div>
            )}
            {verificationStep === 'turnRight' && (
              <div className="instruction-item">
                <div className="instruction-icon">👉</div>
                <div className="instruction-text">Turn your head RIGHT</div>
              </div>
            )}
            {(verificationStep === 'storeData' || verificationStep === 'complete') && (
              <div className="instruction-item">
                <div className="instruction-icon">⏳</div>
                <div className="instruction-text">Processing verification...</div>
              </div>
            )}
          </div>
        )}
        
        {/* Show recognition feedback when detected */}
        {showRecognitionFeedback && !isAlreadyVerified && !verificationSuccess && (
          <div className="recognition-feedback">
            <div className="recognition-icon">👋</div>
            <div className="recognition-text">Welcome Back!</div>
            <div className="recognition-message">You've been recognized. Processing verification...</div>
          </div>
        )}
      </div>

      {/* Info/Controls below video */}
      <div className="face-info-bottom">
        {!isAlreadyVerified && !verificationSuccess && (
          <>
            <div className="verification-strength">
              <div className="strength-label">Verification Progress:</div>
              <div className="strength-bar">
                <div 
                  className="strength-bar-inner"
                  style={{ width: `${verification.verificationStrength}%` }}
                ></div>
              </div>
              <div className="strength-percentage">{verification.verificationStrength}%</div>
            </div>
            <div className="verification-steps">
              <div className={`verification-step ${verification.blinks >= 2 ? 'completed' : ''}`}>
                Blink Detection: {verification.blinks}/{verification.requiredBlinks}
              </div>
              <div className={`verification-step ${verification.leftTurn ? 'completed' : ''}`}>
                Left Turn: {verification.leftTurn ? 'Completed' : 'Pending'}
              </div>
              <div className={`verification-step ${verification.rightTurn ? 'completed' : ''}`}>
                Right Turn: {verification.rightTurn ? 'Completed' : 'Pending'}
              </div>
            </div>
            <div className="debug-info">
              {`Current Tilt: ${currentHeadTilt.toFixed(3)}\nLeft Verified: ${headVerification.left}\nRight Verified: ${headVerification.right}\nCurrent EAR: ${currentEAR.toFixed(3)} (Threshold: 0.290)\nStatus: ${blinkStatus}\n${debugInfo}`}
            </div>
          </>
        )}
        
        {(isAlreadyVerified || verificationSuccess) && (
          <div className="verification-complete-info">
            <div className="complete-message">Verification Successful!</div>
            <div className="complete-details">
              Your face data has been securely stored in the database.<br/>
              You will be automatically recognized on your next visit.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FaceDetection;
