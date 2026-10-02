# server/pipeline/vad_filter.py

import torch
import numpy as np
from silero_vad import load_silero_vad


class VADFilter:
    """
    Handles Voice Activity Detection (VAD) using Silero VAD.
    
    Processes arbitrary-length numpy audio arrays by chunking them into strict
    512-sample frames required by Silero VAD v5 at 16kHz.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the Silero VAD model and pipeline parameters.

        Args:
            threshold (float): Speech confidence boundary (0.0 to 1.0) above 
                               which an audio slice is classified as speech.
        """
        # Load the pre-trained Silero VAD PyTorch model
        self.model = load_silero_vad()
        
        # Set PyTorch model to evaluation mode (disables dropout, batch norm updates)
        self.model.eval()
        
        # Store the decision threshold for speech classification
        self.threshold = threshold
        
        # Silero VAD v5 strictly requires exactly 512 samples per frame at 16,000 Hz
        self.window_size = 512

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Evaluates an incoming audio array for speech presence.

        Divides large audio buffers into 512-sample windows, computes model predictions
        per frame, and determines overall speech presence based on the average probability.

        Args:
            audio_chunk (np.ndarray): 1D float32 array containing mono PCM audio samples.
            sample_rate (int): Sampling rate of the input audio in Hz (defaults to 16000).

        Returns:
            bool: True if average speech probability exceeds or equals threshold, False otherwise.
        """
        # If the input buffer contains fewer samples than required for one evaluation frame, ignore it
        if len(audio_chunk) < self.window_size:
            return False

        probabilities = []

        # Disable gradient calculations to reduce CPU/RAM usage during inference
        with torch.no_grad():
            # Step through the raw audio buffer in non-overlapping 512-sample strides
            for i in range(0, len(audio_chunk) - self.window_size + 1, self.window_size):
                # Extract a 512-sample slice
                chunk = audio_chunk[i : i + self.window_size]
                
                # Convert 1D numpy array into a 2D PyTorch FloatTensor: shape (1, 512)
                tensor_chunk = torch.from_numpy(chunk).float().unsqueeze(0)
                
                # Run inference on the single frame and extract the probability scalar
                speech_prob = self.model(tensor_chunk, sample_rate).item()
                probabilities.append(speech_prob)

        # Handle edge case where no full 512-sample windows were extracted
        if not probabilities:
            return False

        # Calculate average probability across all processed 512-sample windows
        avg_speech_prob = float(np.mean(probabilities))
        
        # Console diagnostic output for monitoring VAD performance per audio buffer
        print(f"[VAD DEBUG] Window Chunks: {len(probabilities)} | Avg Speech Prob: {avg_speech_prob:.4f}")

        # Return True if average probability meets or exceeds the configured confidence threshold
        return avg_speech_prob >= self.threshold