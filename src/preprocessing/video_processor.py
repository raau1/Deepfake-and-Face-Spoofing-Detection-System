"""
Video Processor Module
Handles video frame extraction using OpenCV.
As specified in Section 3.1 and 4.2.3 of the Interim Report.

Supports:
- MP4, AVI, MOV video formats (Section 3.1)
- Uniform and random frame sampling
- Batch processing with progress tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Union
from tqdm import tqdm


class VideoProcessor:
    """
    Video processor for extracting frames from video files.
    
    Handles various video formats and provides flexible frame sampling
    strategies for deepfake detection preprocessing.
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    def __init__(
        self,
        frames_per_video: int = 32,
        sampling_strategy: str = 'uniform',
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            frames_per_video: Number of frames to extract per video
            sampling_strategy: How to sample frames ('uniform', 'random', 'all')
            max_frames: Maximum frames to extract (overrides frames_per_video if smaller)
            resize: Optional resize dimensions (width, height)
        """
        self.frames_per_video = frames_per_video
        self.sampling_strategy = sampling_strategy
        self.max_frames = max_frames
        self.resize = resize
        
        print(f"VideoProcessor initialized: {frames_per_video} frames, {sampling_strategy} sampling")
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            info: Dictionary with video properties
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': str(video_path),
            'filename': video_path.name,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': None,
            'codec': None
        }
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return info
    
    def _get_frame_indices(
        self,
        total_frames: int,
        num_frames: int,
        strategy: str
    ) -> List[int]:
        """
        Calculate which frame indices to extract.
        
        Args:
            total_frames: Total frames in video
            num_frames: Number of frames to extract
            strategy: Sampling strategy
            
        Returns:
            indices: List of frame indices to extract
        """
        if strategy == 'all':
            return list(range(total_frames))
        
        if num_frames >= total_frames:
            return list(range(total_frames))
        
        if strategy == 'uniform':
            # Uniformly spaced frames
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return indices.tolist()
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (uses default if None)
            strategy: Sampling strategy (uses default if None)
            
        Returns:
            frames: List of extracted frames (BGR format)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        num_frames = num_frames or self.frames_per_video
        strategy = strategy or self.sampling_strategy
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Get frame indices to extract
        if self.max_frames is not None:
            num_frames = min(num_frames, self.max_frames)
        
        indices = self._get_frame_indices(total_frames, num_frames, strategy)
        
        frames = []
        current_idx = 0
        
        for target_idx in indices:
            # Seek to target frame
            if target_idx != current_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Resize if specified
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            current_idx = target_idx + 1
        
        cap.release()
        
        return frames
    
    def extract_frames_generator(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Generator that yields frames from a video one at a time.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            strategy: Sampling strategy
            
        Yields:
            (frame, frame_index): Tuple of frame and its index
        """
        video_path = Path(video_path)
        num_frames = num_frames or self.frames_per_video
        strategy = strategy or self.sampling_strategy
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.max_frames is not None:
            num_frames = min(num_frames, self.max_frames)
        
        indices = self._get_frame_indices(total_frames, num_frames, strategy)
        
        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            
            if ret:
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
                yield frame, target_idx
        
        cap.release()
    
    def process_video_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_format: str = 'jpg',
        num_frames: Optional[int] = None
    ) -> dict:
        """
        Process all videos in a directory and save extracted frames.
        
        Args:
            input_dir: Directory containing videos
            output_dir: Directory to save extracted frames
            save_format: Image format for saving ('jpg', 'png')
            num_frames: Number of frames per video
            
        Returns:
            stats: Processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in self.SUPPORTED_FORMATS:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        stats = {
            'total_videos': len(video_files),
            'processed': 0,
            'failed': 0,
            'total_frames': 0
        }
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                frames = self.extract_frames(video_path, num_frames)
                
                # Create output subdirectory for this video
                video_name = video_path.stem
                video_output_dir = output_dir / video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save frames
                for idx, frame in enumerate(frames):
                    frame_path = video_output_dir / f"frame_{idx:04d}.{save_format}"
                    cv2.imwrite(str(frame_path), frame)
                
                stats['processed'] += 1
                stats['total_frames'] += len(frames)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                stats['failed'] += 1
        
        return stats
    
    @staticmethod
    def is_supported_format(file_path: Union[str, Path]) -> bool:
        """Check if a file has a supported video format."""
        return Path(file_path).suffix.lower() in VideoProcessor.SUPPORTED_FORMATS


# Test function
def test_video_processor():
    """Test the video processor."""
    print("Testing VideoProcessor...")
    
    processor = VideoProcessor(frames_per_video=16, sampling_strategy='uniform')
    
    # Test with a sample video path
    test_path = Path("C:/FINAL YEAR PROJECT/data/FaceForensics++/DeepFakeDetection")
    
    if test_path.exists():
        video_files = list(test_path.glob("*.mp4"))[:1]
        
        if video_files:
            video_path = video_files[0]
            print(f"Testing with: {video_path}")
            
            # Get video info
            info = processor.get_video_info(video_path)
            print(f"Video info: {info['frame_count']} frames, {info['fps']:.2f} FPS")
            
            # Extract frames
            frames = processor.extract_frames(video_path)
            print(f"Extracted {len(frames)} frames")
            
            if frames:
                print(f"Frame shape: {frames[0].shape}")
    else:
        print(f"Test path not found: {test_path}")
    
    print("VideoProcessor test complete!")


if __name__ == '__main__':
    test_video_processor()
