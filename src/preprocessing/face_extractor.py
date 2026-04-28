"""
Face Extractor Module
Uses MTCNN for face detection as specified in Section 4.2.3 of the Interim Report.

This module handles:
- Face detection in images and video frames
- Bounding box extraction with configurable margins
- Batch processing for efficiency
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image


class FaceExtractor:
    """
    Face extractor using MTCNN for robust face detection.
    
    Attributes:
        device: PyTorch device (cuda/cpu)
        mtcnn: MTCNN face detector
        min_face_size: Minimum face size to detect
        margin: Margin around detected face (fraction)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        min_face_size: int = 60,
        margin: float = 0.3,
        selection_method: str = 'largest',
        keep_all: bool = False,
        post_process: bool = False
    ):
        """
        Initialize the face extractor.
        
        Args:
            device: Device to run detection on ('cuda' or 'cpu')
            min_face_size: Minimum face size in pixels
            margin: Margin to add around face (as fraction of face size)
            selection_method: How to select face when multiple detected
                             ('largest', 'center', 'probability')
            keep_all: Whether to return all detected faces
            post_process: Whether to apply post-processing to faces
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.min_face_size = min_face_size
        self.margin = margin
        self.selection_method = selection_method
        self.keep_all = keep_all
        
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds for 3 stages
            factor=0.709,
            post_process=post_process,
            select_largest=True,
            keep_all=keep_all,
            device=self.device
        )
        
        print(f"FaceExtractor initialized on {self.device}")
    
    def detect_faces(
        self,
        image: Union[np.ndarray, Image.Image],
        return_landmarks: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR numpy array or PIL Image)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            boxes: Detected face bounding boxes [N, 4] (x1, y1, x2, y2)
            probs: Detection probabilities [N]
            landmarks: Facial landmarks [N, 5, 2] if requested, else None
        """
        # Convert BGR to RGB if numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Detect faces - handle different return formats
        if return_landmarks:
            result = self.mtcnn.detect(image, landmarks=True)
            if result is None:
                return None, None, None
            if len(result) == 3:
                boxes, probs, landmarks = result
            else:
                boxes, probs = result
                landmarks = None
        else:
            result = self.mtcnn.detect(image, landmarks=False)
            if result is None:
                return None, None, None
            if isinstance(result, tuple):
                if len(result) == 3:
                    boxes, probs, landmarks = result
                elif len(result) == 2:
                    boxes, probs = result
                    landmarks = None
                else:
                    return None, None, None
            else:
                return None, None, None
        
        return boxes, probs, landmarks
    
    def extract_face(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: int = 299,
        return_box: bool = False
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Extract a single face from an image.
        
        Args:
            image: Input image (BGR numpy array or PIL Image)
            output_size: Size of output face image
            return_box: Whether to return bounding box
            
        Returns:
            face: Extracted face image [output_size, output_size, 3] or None
            box: Bounding box [4] if return_box=True, else None
        """
        # Store original for cropping
        if isinstance(image, Image.Image):
            original_image = np.array(image)
            is_rgb = True
        else:
            original_image = image.copy()
            is_rgb = False
        
        # Detect faces
        boxes, probs, _ = self.detect_faces(image)
        
        if boxes is None or len(boxes) == 0:
            return (None, None) if return_box else None
        
        # Handle case where boxes might be None or empty
        try:
            if boxes is None:
                return (None, None) if return_box else None
            boxes = np.atleast_2d(boxes)
            if boxes.shape[0] == 0:
                return (None, None) if return_box else None
        except:
            return (None, None) if return_box else None
        
        # Select best face
        if probs is not None:
            probs = np.atleast_1d(probs)
        else:
            probs = np.ones(len(boxes))
            
        box = self._select_face(boxes, probs, original_image.shape)
        
        if box is None:
            return (None, None) if return_box else None
        
        # Extract face with margin
        face = self._crop_face(original_image, box, output_size, is_rgb)
        
        if return_box:
            return face, box
        return face
    
    def extract_faces_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        output_size: int = 299
    ) -> List[Optional[np.ndarray]]:
        """
        Extract faces from a batch of images.
        
        Args:
            images: List of input images
            output_size: Size of output face images
            
        Returns:
            faces: List of extracted face images (None for failed detections)
        """
        faces = []
        for image in images:
            face = self.extract_face(image, output_size)
            faces.append(face)
        return faces
    
    def _select_face(
        self,
        boxes: np.ndarray,
        probs: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Select the best face from multiple detections.
        
        Args:
            boxes: Detected bounding boxes [N, 4]
            probs: Detection probabilities [N]
            image_shape: Shape of input image (H, W, C)
            
        Returns:
            box: Selected bounding box [4]
        """
        if boxes is None or len(boxes) == 0:
            return None
            
        if self.selection_method == 'largest':
            # Select largest face by area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            
        elif self.selection_method == 'center':
            # Select face closest to image center
            h, w = image_shape[:2]
            center = np.array([w / 2, h / 2])
            face_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            distances = np.linalg.norm(face_centers - center, axis=1)
            idx = np.argmin(distances)
            
        elif self.selection_method == 'probability':
            # Select face with highest detection probability
            idx = np.argmax(probs)
            
        else:
            idx = 0
        
        return boxes[idx]
    
    def _crop_face(
        self,
        image: np.ndarray,
        box: np.ndarray,
        output_size: int,
        is_rgb: bool = False
    ) -> Optional[np.ndarray]:
        """
        Crop and resize face from image with margin.
        
        Args:
            image: Input image
            box: Bounding box [x1, y1, x2, y2]
            output_size: Output size
            is_rgb: Whether image is RGB (vs BGR)
            
        Returns:
            face: Cropped and resized face image (RGB)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        
        # Calculate face dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Add margin
        margin_w = face_w * self.margin
        margin_h = face_h * self.margin
        
        # New coordinates with margin
        x1 = max(0, int(x1 - margin_w))
        y1 = max(0, int(y1 - margin_h))
        x2 = min(w, int(x2 + margin_w))
        y2 = min(h, int(y2 + margin_h))
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        # Resize to output size
        if face.size > 0:
            face = cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB if input was BGR
            if not is_rgb:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            return face
        else:
            return None
    
    def process_video_frames(
        self,
        frames: List[np.ndarray],
        output_size: int = 299
    ) -> List[Tuple[Optional[np.ndarray], int]]:
        """
        Process multiple video frames and extract faces.
        
        Args:
            frames: List of video frames (BGR)
            output_size: Output face size
            
        Returns:
            results: List of (face, frame_index) tuples
        """
        results = []
        for idx, frame in enumerate(frames):
            face = self.extract_face(frame, output_size)
            if face is not None:
                results.append((face, idx))
        return results


# Test function
def test_face_extractor():
    """Test the face extractor with a sample image."""
    print("Testing FaceExtractor...")
    
    extractor = FaceExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection (won't find faces in random noise, but tests the pipeline)
    boxes, probs, _ = extractor.detect_faces(dummy_image)
    
    if boxes is None:
        print("No faces detected (expected for random noise)")
    else:
        print(f"Detected {len(boxes)} faces")
    
    print("FaceExtractor test complete!")


if __name__ == '__main__':
    test_face_extractor()
