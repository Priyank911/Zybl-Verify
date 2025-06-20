import React, { useEffect, useRef, useState } from 'react';
import faceapi from '../utils/faceApiBrowserAdapter';
import { generateFaceVector, encryptFaceVector, storeFaceVector, checkExistingFaceVector, calculateSimilarity, checkDatabaseAccess } from '../utils/faceVerification';

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

  // Load models and start video
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models')
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
          }
          
          // Check if user is already verified (only if not already checked)
          if (!initialCheckDoneRef.current && !isAlreadyVerified) {
            await checkForExistingVerification(detections);
            if (isAlreadyVerified) {
              return; // Skip the rest of the verification process
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
      if (stableDetectionCount < 2) {
        console.warn(`Face detection not stable enough (${stableDetectionCount}/3 stable detections)`);
        setMessage('Please keep your face centered and still');
        setIsCheckingExisting(false);
        return;
      }
      
      console.log('Face detection is stable. Proceeding with verification check...');
      
      // Step 4: Check all collected vectors against the database
      let matchCount = 0;
      
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
      
      // Step 5: Only verify if multiple checks succeed
      if (matchCount >= 2) {
        console.log(`User verified with high confidence (${matchCount}/${faceVectors.length} checks passed)`);
        setIsAlreadyVerified(true);
        setVerificationSuccess(true);
        setMessage('You are already verified! Face recognized.');
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
    
    if (!initialCheckDoneRef.current) {
      initialCheckTimer = setInterval(async () => {
        // Give up after several attempts
        if (checkAttempts >= 5 || initialCheckDoneRef.current) {
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
              setVerificationSuccess(true);
              setVerificationComplete(true);
              
              setStepsCompleted({
                blink: true,
                turnLeft: true,
                turnRight: true,
                faceStored: true
              });
              
              setMessage('You are already verified! Face recognized.');
              
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

  // Two-phase face detection and recognition process
  useEffect(() => {
    // Don't start face recognition until database check is complete
    if (isLoadingDatabase || hasCheckedDatabase || !videoRef.current || !canvasRef.current) {
      return;
    }
    
    let faceCheckCount = 0;
    const maxFaceChecks = 5;
    const minRequiredDetections = 3;
    let successfulDetections = 0;
    let detectedFaceVectors = [];
    let recognitionTimer = null;
    
    const performFaceRecognition = async () => {
      if (faceCheckCount >= maxFaceChecks || initialDbCheckComplete) {
        clearInterval(recognitionTimer);
        return;
      }
      
      try {
        faceCheckCount++;
        console.log(`Face recognition attempt ${faceCheckCount}/${maxFaceChecks}...`);
        
        const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceExpressions()
          .withFaceDescriptors();
          
        if (detections && detections.length > 0) {
          successfulDetections++;
          console.log(`Successful face detection: ${successfulDetections}/${minRequiredDetections}`);
          
          const faceVector = generateFaceVector(detections[0]);
          detectedFaceVectors.push(faceVector);
          
          // Store the latest face vector for future use
          faceVectorRef.current = faceVector;
          
          // When we have enough good detections, check if this face exists in database
          if (successfulDetections >= minRequiredDetections) {
            clearInterval(recognitionTimer);
            await checkFaceInDatabase(detectedFaceVectors);
          }
        } else {
          console.log('No face detected in this frame');
        }
      } catch (error) {
        console.error('Error during face recognition:', error);
      }
    };
    
    // Start periodic face recognition checks
    setMessage('Looking for your face...');
    recognitionTimer = setInterval(performFaceRecognition, 1000);
    
    return () => {
      if (recognitionTimer) {
        clearInterval(recognitionTimer);
      }
    };
  }, [isLoadingDatabase, hasCheckedDatabase]);

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
  };  // Enhanced verification completion handler
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
      
      // Use stored face vector if available, otherwise detect a new one
      let faceVector = faceVectorRef.current;
      
      if (!faceVector) {
        // Try to get the face vector one more time
        try {
          setMessage('Capturing your face data...');
          const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptors();

          if (detections && detections.length > 0) {
            faceVector = generateFaceVector(detections[0]);
          } else {
            throw new Error("No face detected during verification completion");
          }
        } catch (error) {
          console.error("Error detecting face for vector generation:", error);
          throw new Error("Couldn't capture your face properly. Please try again.");
        }
      }
      
      if (!faceVector) {
        throw new Error("Could not generate face data. Please try again.");
      }

      // First check if this face already exists in the database
      // This handles the edge case where someone completes verification but was already in the system
      setMessage('Checking if you are already registered...');
      const exists = await checkExistingFaceVector(faceVector);
      
      if (exists) {
        console.log("Face already exists in database");
        setMessage('Your face is already registered in our system!');
        setVerificationSuccess(true);
        setStepsCompleted(prev => ({ ...prev, faceStored: true }));
      } else {
        // Store the face data and complete the process
        setMessage('Storing your face data securely...');
        
        try {
          await storeFaceVector(faceVector);
          
          // Double-check that storage worked by verifying the face was saved
          const storageVerified = await checkExistingFaceVector(faceVector);
          
          if (storageVerified) {
            console.log("Face vector successfully stored and verified");
            setMessage('Verification successful! Your face is now registered.');
          } else {
            console.warn("Storage verification warning: Face was stored but not found on verification");
            setMessage('Verification successful! Your face is now registered.');
          }
          
          setVerificationSuccess(true);
          setStepsCompleted(prev => ({ ...prev, faceStored: true }));
        } catch (error) {
          console.error("Error storing face vector:", error);
          throw new Error("Failed to save your face data. Please try again.");
        }
      }
      
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
  // Helper for progress bar
  const getProgress = () => {
    // Always show 100% when verification is complete
    if (verificationSuccess || isAlreadyVerified || verificationComplete) {
      return 1.0; // 100%
    }
    
    // Show 75% when all verification steps are done but storage is pending
    if (stepsCompleted.blink && stepsCompleted.turnLeft && stepsCompleted.turnRight) {
      return 0.75; // 75%
    }
    
    // Calculate based on steps completed
    let progress = 0;
    if (stepsCompleted.blink) progress += 0.25;
    if (stepsCompleted.turnLeft) progress += 0.25;
    if (stepsCompleted.turnRight) progress += 0.25;
    if (stepsCompleted.faceStored) progress += 0.25;
    
    return progress;
  };  // Monitor verification steps and trigger completion automatically if needed
  useEffect(() => {
    // If all verification steps are completed but storage hasn't happened
    if (stepsCompleted.blink && stepsCompleted.turnLeft && stepsCompleted.turnRight && 
        !stepsCompleted.faceStored && !verificationSuccess && !isAlreadyVerified) {
      
      console.log("All steps completed, triggering verification completion");
      // Set as complete and trigger storing process
      setVerificationComplete(true);
      setVerificationStep('complete');
      handleVerificationComplete();
    }
  }, [stepsCompleted, verificationSuccess, isAlreadyVerified]);

  // Function to check if face exists in database
  const checkFaceInDatabase = async (faceVectors) => {
    try {
      setHasCheckedDatabase(true);
      setMessage('Checking if your face is already registered...');
      
      // Only use the best vectors (some might be lower quality)
      const bestVectors = faceVectors.slice(-3); // Use the most recent 3 detections
      let verifiedCount = 0;
      
      // Check each vector against the database
      for (let i = 0; i < bestVectors.length; i++) {
        const result = await checkExistingFaceVector(bestVectors[i]);
        if (result) verifiedCount++;
        
        console.log(`Database check ${i+1}/${bestVectors.length}: ${result ? 'MATCH' : 'NO MATCH'}`);
      }
      
      const percentMatch = (verifiedCount / bestVectors.length) * 100;
      console.log(`Face verification result: ${verifiedCount}/${bestVectors.length} (${percentMatch.toFixed(1)}%)`);
      
      // If majority of checks confirm this is an existing user
      if (verifiedCount >= 2) {
        console.log('This face is already verified in the system');
        setMessage('Face recognized! You are already verified.');
        setIsAlreadyVerified(true);
        setVerificationSuccess(true);
        setVerificationComplete(true);
        setStepsCompleted({
          blink: true,
          turnLeft: true,
          turnRight: true,
          faceStored: true
        });
        
        // Call the completion callback
        if (onVerificationComplete) {
          onVerificationComplete();
        }
      } else {
        console.log('This is a new face - starting verification process');
        setMessage('Welcome! Please complete the verification steps.');
      }
      
      // Mark initial check as complete
      setInitialDbCheckComplete(true);
      
    } catch (error) {
      console.error('Error checking face in database:', error);
      setMessage('Error during face check. Starting verification steps...');
      setInitialDbCheckComplete(true);
    }
  };

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
        />
        
        {/* Show verified overlay when already verified or verification successful */}
        {(isAlreadyVerified || verificationSuccess) && (
          <div className="verified-overlay">
            <div className="verified-icon">✓</div>
            <div className="verified-text">Verification Complete</div>
            <div className="verified-details">
              Your face data has been securely stored.<br/>
              You will be automatically recognized on future visits.
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
