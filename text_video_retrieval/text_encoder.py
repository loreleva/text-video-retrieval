"""Text encoder."""

from internvideo2_multi_modality.inference import load_text_encoder
import numpy as np
import torch.nn.functional as F


class TextEncoder:
    """Encode text as unit-normalized embeddings."""
    
    def __init__(self, config_path: str):
        """Load the text encoder.

        Args:
            config_path: Path to the encoder configuration file.
        """
        self.text_encoder = load_text_encoder(config_path)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text as a unit-normalized NumPy array.

        Args:
            text: Text query to encode.

        Returns:
            Unit-normalized text embedding array of shape (1, D).
        """
        text_feat = self.text_encoder(text)
        text_feat = F.normalize(text_feat, dim=-1)
        return np.asarray(text_feat.cpu())