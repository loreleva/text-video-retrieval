"""Vision encoder."""

from pathlib import Path

from internvideo2_multi_modality.inference import load_vision_encoder
import numpy as np
import torch
import torch.nn.functional as F

from .utils import preprocess_videos


class VisionEncoder:
    """Encode videos as unit-normalized embeddings."""
    
    def __init__(
        self,
        config_path: str,
        *,
        dense_sampling: bool,
    ):
        """Load the vision encoder and configure video sampling.
        
        Args:
            config_path: Path to the encoder configuration file.
            dense_sampling: Sample one frame per second and
                group frames into clips if True;
                otherwise sample one clip uniformly per video.
        """
        self.vision_encoder = load_vision_encoder(config_path)
        self.target_num_frames = self.vision_encoder.num_frames
        self.target_size = (self.vision_encoder.img_size,) * 2
        self.dtype = self.vision_encoder.dtype
        self.dense_sampling = dense_sampling

    def encode_videos(
        self,
        videos_path: str | Path,
    ) -> tuple[list[Path], np.ndarray]:
        """Encode a video file or the videos in a directory.
        
        With dense sampling, sum the normalized clip embeddings
        and normalize the result to produce one embedding per video.
        
        Args:
            videos_path: Path to a video file or a directory
                containing videos.
        
        Returns:
            Video paths and corresponding unit-normalized embeddings
            with shape (N, D).
        """
        videos_path, videos_tensor = preprocess_videos(
            videos_path,
            target_num_frames=self.target_num_frames,
            target_size=self.target_size,
            dtype=self.dtype,
            dense_sampling=self.dense_sampling,
        )
        videos_feat = []
        for video_tensor in videos_tensor:
            video_feat = self.vision_encoder(video_tensor)
            video_feat = F.normalize(video_feat, dim=-1)
            if self.dense_sampling:
                video_feat = video_feat.sum(dim=0, keepdim=True)
                video_feat = F.normalize(video_feat, dim=-1)
            videos_feat.append(video_feat)
        videos_feat = torch.concat(videos_feat, dim=0)
        return videos_path, np.asarray(videos_feat.cpu())