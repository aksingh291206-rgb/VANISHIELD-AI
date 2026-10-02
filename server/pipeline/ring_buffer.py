# server/pipeline/ring_buffer.py

# =========================================================================
# SYSTEM & NUMPY IMPORTS
# Importing NumPy for fast, contiguous memory array operations
# =========================================================================
import numpy as np


class RingBuffer:
    def __init__(self, capacity_samples: int = 48000, stride_samples: int = 3200):
        """
        capacity_samples: Total memory window size (48,000 samples = 3.0s window at 16kHz).
        stride_samples: Step interval for inference triggers (3,200 samples = 200ms step).
        """
        # Set max rolling buffer capacity (3.0 seconds at 16kHz)
        self.capacity = capacity_samples
        
        # Set stride triggering interval (200ms step size)
        self.stride = stride_samples
        
        # Pre-allocate fixed float32 memory array initialized to zeros
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        
        # Tracks how many samples have filled the buffer on initial startup
        self.filled_samples = 0
        
        # Tracks accumulated samples toward the next 200ms stride trigger
        self.accumulated_since_stride = 0

    def append(self, pcm_chunk: np.ndarray) -> bool:
        """
        Appends new float32 audio samples to the ring buffer.
        Returns True when >= 1.0s of audio is primed AND a 200ms stride interval occurs.
        """
        # Determine number of incoming samples in current chunk
        chunk_len = len(pcm_chunk)
        
        # If incoming array is empty, return False immediately
        if chunk_len == 0:
            return False

        # If incoming chunk exceeds capacity, keep only the newest samples
        if chunk_len >= self.capacity:
            self.buffer[:] = pcm_chunk[-self.capacity:]
            self.filled_samples = self.capacity
        else:
            # Shift existing buffer left by chunk_len to drop old samples
            self.buffer = np.roll(self.buffer, -chunk_len)
            
            # Append new chunk at the end of the buffer
            self.buffer[-chunk_len:] = pcm_chunk
            
            # Update total count of samples buffered
            self.filled_samples = min(self.capacity, self.filled_samples + chunk_len)

        # Update counter tracking progress toward the next 200ms stride
        self.accumulated_since_stride += chunk_len

        # Require at least 16,000 samples (1 second) of audio before firing first stride
        if self.filled_samples >= 16000 and self.accumulated_since_stride >= self.stride:
            # Keep leftover samples for exact stride tracking across frames
            self.accumulated_since_stride %= self.stride
            return True

        return False

    def get_window(self) -> np.ndarray:
        """Returns a snapshot copy of the current 3-second audio window array."""
        return self.buffer.copy()