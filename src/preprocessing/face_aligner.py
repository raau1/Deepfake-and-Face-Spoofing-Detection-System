"""
Face Aligner Module
Uses MediaPipe for facial landmark detection and alignment.
As specified in Section 1.2 and 4.2.3 of the Interim Report.

This module handles:
- Facial landmark detection using MediaPipe FaceMesh
- Face alignment based on eye positions
- Standardized face cropping for consistent model input
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path

# Handles different MediaPipe versions
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_LEGACY = False
except ImportError:
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_LEGACY = True
    except (ImportError, AttributeError):
        MEDIAPIPE_AVAILABLE = False
        print("WARNING: MediaPipe not available. Face alignment will use simple cropping.")


class FaceAligner:
    """
    Face aligner using MediaPipe FaceMesh for landmark-based alignment.
    
    Aligns faces based on eye positions to ensure consistent orientation
    for improved model performance.
    """
    
    # MediaPipe FaceMesh landmark indices for key points
    # Left eye in image coordinates (subject's right eye)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Right eye in image coordinates (subject's left eye)
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Nose tip
    NOSE_TIP_INDEX = 1
    # Mouth landmarks
    MOUTH_INDICES = [61, 291, 0, 17]
    
    def __init__(
        self,
        output_size: int = 299,
        margin: float = 0.3,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the face aligner.
        
        Args:
            output_size: Size of output aligned face image
            margin: Margin around face as fraction
            static_image_mode: Whether to treat each image independently
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.output_size = output_size
        self.margin = margin
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        
        if MEDIAPIPE_AVAILABLE:
            try:
                # Initialize MediaPipe FaceMesh
                if MEDIAPIPE_LEGACY:
                    self.face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=static_image_mode,
                        max_num_faces=max_num_faces,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        refine_landmarks=True
                    )
                else:
                    self.face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=static_image_mode,
                        max_num_faces=max_num_faces,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        refine_landmarks=True
                    )
                print(f"FaceAligner initialized with MediaPipe, output_size={output_size}")
            except Exception as e:
                print(f"WARNING: Could not initialize MediaPipe FaceMesh: {e}")
                print("Face alignment will use simple cropping.")
                self.mediapipe_available = False
                self.face_mesh = None
        else:
            self.face_mesh = None
            print(f"FaceAligner initialized without MediaPipe (simple cropping), output_size={output_size}")
    
    def get_landmarks(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in an image.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            landmarks: Array of landmarks [468, 3] or None if no face detected
        """
        if not self.mediapipe_available or self.face_mesh is None:
            return None
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Process image
        try:
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            h, w = image.shape[:2]
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z * w]
                for lm in face_landmarks.landmark
            ])
            
            return landmarks
        except Exception as e:
            return None
    
    def get_eye_centers(
        self,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate eye center positions from landmarks.
        
        Args:
            landmarks: Facial landmarks [468, 3]
            
        Returns:
            left_eye: Left eye center [2]
            right_eye: Right eye center [2]
        """
        left_eye = np.mean(landmarks[self.LEFT_EYE_INDICES, :2], axis=0)
        right_eye = np.mean(landmarks[self.RIGHT_EYE_INDICES, :2], axis=0)
        
        return left_eye, right_eye
    
    def calculate_alignment_matrix(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        desired_left_eye: Tuple[float, float] = (0.35, 0.35),
        output_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate the affine transformation matrix for face alignment.
        
        Args:
            left_eye: Left eye center position
            right_eye: Right eye center position
            desired_left_eye: Desired position of left eye in output (fractions)
            output_size: Size of output image
            
        Returns:
            M: Affine transformation matrix [2, 3]
        """
        if output_size is None:
            output_size = self.output_size
        
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate distance between eyes
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        
        # Calculate desired distance based on output size
        desired_right_eye_x = 1.0 - desired_left_eye[0]
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * output_size
        
        # Calculate scale
        scale = desired_dist / dist if dist > 0 else 1.0
        
        # Calculate center point between eyes
        eyes_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update translation component
        tX = output_size * 0.5
        tY = output_size * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        return M
    
    def align_face(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        output_size: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Align a face using landmark-based transformation.
        
        Args:
            image: Input image (BGR)
            landmarks: Pre-computed landmarks (optional)
            output_size: Size of output image
            
        Returns:
            aligned: Aligned face image or None if alignment fails
        """
        if output_size is None:
            output_size = self.output_size
        
        # Get landmarks if not provided
        if landmarks is None:
            landmarks = self.get_landmarks(image)
        
        if landmarks is None:
            return None
        
        # Get eye centers
        left_eye, right_eye = self.get_eye_centers(landmarks)
        
        # Calculate alignment matrix
        M = self.calculate_alignment_matrix(left_eye, right_eye, output_size=output_size)
        
        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            M,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return aligned
    
    def align_face_simple(
        self,
        image: np.ndarray,
        box: np.ndarray,
        output_size: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Simple face alignment using bounding box (fallback method).
        
        Args:
            image: Input image
            box: Face bounding box [x1, y1, x2, y2]
            output_size: Output size
            
        Returns:
            aligned: Cropped and resized face
        """
        if output_size is None:
            output_size = self.output_size
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        
        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_w = int(face_w * self.margin)
        margin_h = int(face_h * self.margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        # Crop and resize
        face = image[y1:y2, x1:x2]
        
        if face.size > 0:
            aligned = cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        else:
            aligned = None
        
        return aligned
    
    def process_image(
        self,
        image: np.ndarray,
        use_alignment: bool = True,
        fallback_box: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Process an image to extract and align face.
        
        Args:
            image: Input image (BGR)
            use_alignment: Whether to use landmark-based alignment
            fallback_box: Bounding box to use if alignment fails
            
        Returns:
            face: Aligned face image or None
        """
        if use_alignment and self.mediapipe_available:
            aligned = self.align_face(image)
            if aligned is not None:
                return aligned
        
        # Fallback to simple crop if alignment fails
        if fallback_box is not None:
            return self.align_face_simple(image, fallback_box)
        
        return None
    
    def close(self):
        """Release resources."""
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass
    
    def __del__(self):
        """Destructor to release resources."""
        try:
            self.close()
        except:
            pass


# Test function
def test_face_aligner():
    """Test the face aligner."""
    print("Testing FaceAligner...")
    
    aligner = FaceAligner(output_size=299)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test landmark detection
    landmarks = aligner.get_landmarks(dummy_image)
    
    if landmarks is None:
        print("No face detected (expected for random noise)")
    else:
        print(f"Detected {len(landmarks)} landmarks")
    
    aligner.close()
    print("FaceAligner test complete!")


if __name__ == '__main__':
    test_face_aligner()
