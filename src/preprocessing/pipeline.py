"""
Preprocessing Pipeline
Combines video processing, face extraction, and alignment into a unified pipeline.
As specified in Section 1.2 and 4.2.3 of the Interim Report.

This is the main entry point for preprocessing videos into face datasets
suitable for training deepfake detection models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from tqdm import tqdm
import json
import os

from src.preprocessing.face_extractor import FaceExtractor
from src.preprocessing.face_aligner import FaceAligner
from src.preprocessing.video_processor import VideoProcessor


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for deepfake detection.
    
    Combines:
    - Video frame extraction (VideoProcessor)
    - Face detection (MTCNN via FaceExtractor)
    - Face alignment (MediaPipe via FaceAligner)
    """
    
    def __init__(
        self,
        output_size: int = 299,
        frames_per_video: int = 32,
        sampling_strategy: str = 'uniform',
        margin: float = 0.3,
        use_alignment: bool = True,
        device: str = 'cuda',
        min_face_size: int = 60
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            output_size: Size of output face images (299 for XceptionNet)
            frames_per_video: Number of frames to extract per video
            sampling_strategy: Frame sampling strategy ('uniform', 'random')
            margin: Margin around detected face
            use_alignment: Whether to use landmark-based alignment
            device: PyTorch device for face detection
            min_face_size: Minimum face size to detect
        """
        self.output_size = output_size
        self.frames_per_video = frames_per_video
        self.use_alignment = use_alignment
        
        # Initialize components
        self.video_processor = VideoProcessor(
            frames_per_video=frames_per_video,
            sampling_strategy=sampling_strategy
        )
        
        self.face_extractor = FaceExtractor(
            device=device,
            min_face_size=min_face_size,
            margin=margin
        )
        
        if use_alignment:
            self.face_aligner = FaceAligner(
                output_size=output_size,
                margin=margin
            )
        else:
            self.face_aligner = None
        
        print(f"PreprocessingPipeline initialized:")
        print(f"  Output size: {output_size}")
        print(f"  Frames per video: {frames_per_video}")
        print(f"  Alignment: {'enabled' if use_alignment else 'disabled'}")
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Process a single frame: detect and align face.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            face: Processed face image or None if no face detected
        """
        # Detect face and get bounding box
        face, box = self.face_extractor.extract_face(
            frame,
            output_size=self.output_size,
            return_box=True
        )
        
        if face is None:
            return None
        
        # Apply alignment if enabled
        if self.use_alignment and self.face_aligner is not None:
            aligned = self.face_aligner.align_face(frame, output_size=self.output_size)
            if aligned is not None:
                # Convert BGR to RGB
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                return aligned
        
        # Return detected face (already RGB from FaceExtractor)
        return face
    
    def process_video(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Process a single video: extract frames and faces.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (uses default if None)
            
        Returns:
            faces: List of extracted face images
            metadata: Processing metadata
        """
        video_path = Path(video_path)
        num_frames = num_frames or self.frames_per_video
        
        # Get video info
        try:
            video_info = self.video_processor.get_video_info(video_path)
        except Exception as e:
            return [], {'error': str(e), 'video_path': str(video_path)}
        
        # Extract frames
        try:
            frames = self.video_processor.extract_frames(video_path, num_frames)
        except Exception as e:
            return [], {'error': str(e), 'video_path': str(video_path)}
        
        if not frames:
            return [], {'error': 'No frames extracted', 'video_path': str(video_path)}
        
        # Process each frame
        faces = []
        face_indices = []
        
        for idx, frame in enumerate(frames):
            face = self.process_frame(frame)
            if face is not None:
                faces.append(face)
                face_indices.append(idx)
        
        metadata = {
            'video_path': str(video_path),
            'video_name': video_path.stem,
            'total_frames': video_info['frame_count'],
            'frames_extracted': len(frames),
            'faces_detected': len(faces),
            'face_indices': face_indices,
            'fps': video_info['fps'],
            'duration': video_info['duration']
        }
        
        return faces, metadata
    
    def process_dataset(
        self,
        input_dirs: Dict[str, Path],
        output_dir: Path,
        save_format: str = 'jpg',
        quality: int = 95
    ) -> Dict:
        """
        Process an entire dataset of videos.
        
        Args:
            input_dirs: Dictionary mapping labels to input directories
                       e.g., {'real': Path('...'), 'fake': Path('...')}
            output_dir: Root output directory
            save_format: Image format for saving ('jpg', 'png')
            quality: JPEG quality (1-100)
            
        Returns:
            stats: Processing statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_faces': 0,
            'per_class': {}
        }
        
        all_metadata = []
        
        for label, input_dir in input_dirs.items():
            input_dir = Path(input_dir)
            label_output_dir = output_dir / label
            label_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing {label} videos from: {input_dir}")
            
            # Find all video files
            video_files = []
            for ext in VideoProcessor.SUPPORTED_FORMATS:
                video_files.extend(input_dir.glob(f'*{ext}'))
                video_files.extend(input_dir.glob(f'*{ext.upper()}'))
            # De-duplicate: on case-insensitive filesystems (Windows, default
            # macOS) the lower- and upper-case globs match the same files.
            video_files = sorted({p.resolve() for p in video_files})
            
            stats['per_class'][label] = {
                'total': len(video_files),
                'processed': 0,
                'failed': 0,
                'faces': 0
            }
            stats['total_videos'] += len(video_files)
            
            for video_path in tqdm(video_files, desc=f"Processing {label}"):
                try:
                    faces, metadata = self.process_video(video_path)
                    
                    if faces:
                        # Create output directory for this video
                        video_output_dir = label_output_dir / video_path.stem
                        video_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save faces
                        for idx, face in enumerate(faces):
                            face_path = video_output_dir / f"face_{idx:04d}.{save_format}"
                            
                            # Convert RGB to BGR for OpenCV saving
                            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                            
                            if save_format == 'jpg':
                                cv2.imwrite(str(face_path), face_bgr, 
                                           [cv2.IMWRITE_JPEG_QUALITY, quality])
                            else:
                                cv2.imwrite(str(face_path), face_bgr)
                        
                        metadata['label'] = label
                        metadata['output_dir'] = str(video_output_dir)
                        all_metadata.append(metadata)
                        
                        stats['processed_videos'] += 1
                        stats['total_faces'] += len(faces)
                        stats['per_class'][label]['processed'] += 1
                        stats['per_class'][label]['faces'] += len(faces)
                    else:
                        stats['failed_videos'] += 1
                        stats['per_class'][label]['failed'] += 1
                        
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    stats['failed_videos'] += 1
                    stats['per_class'][label]['failed'] += 1
        
        # Save metadata
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'stats': stats,
                'videos': all_metadata
            }, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_path}")
        
        return stats
    
    def close(self):
        """Release resources."""
        if self.face_aligner is not None:
            self.face_aligner.close()
