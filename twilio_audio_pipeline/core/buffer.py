# core/buffer.py

import threading  # Threading lock to ensure thread safety across concurrent network/inference threads
import numpy as np  # NumPy for high-performance memory buffer management


class AudioRingBuffer:
  """A thread-safe circular ring buffer designed to store streaming audio

  and slice out fixed-size windows for model inference.
  """

  def __init__(self, capacity_samples: int):
    """Initializes the ring buffer with a fixed maximum capacity in samples.

    Args:
        capacity_samples (int): Maximum number of audio samples the buffer can
          hold (e.g., 5 seconds * 16000Hz = 80000 samples).
    """
    # Step 1: Pre-allocate a contiguous block of zeros in memory to prevent runtime memory allocation overhead
    self.buffer = np.zeros(capacity_samples, dtype=np.float32)

    # Step 2: Store the total maximum capacity attribute
    self.capacity = capacity_samples

    # Step 3: Initialize the head pointer (where reading/popping data starts)
    self.head = 0

    # Step 4: Initialize the tail pointer (where incoming data is written)
    self.tail = 0

    # Step 5: Track the current count of unread samples available in the buffer
    self.size = 0

    # Step 6: Create a threading Lock object to prevent race conditions between the WebSocket ingestion thread and inference thread
    self.lock = threading.Lock()

  def write(self, data: np.ndarray):
    """Writes incoming audio chunks into the ring buffer, wrapping around if full.

    Args:
        data (np.ndarray): 1D NumPy float32 array of audio samples to append.
    """
    # Step 7: Acquire the thread lock to ensure atomic write operations
    with self.lock:
      # Step 8: Get the length of the incoming data array
      n = len(data)

      # Step 9: Edge protection - if incoming data exceeds total buffer capacity, truncate to keep only the newest window
      if n > self.capacity:
        data = data[-self.capacity :]
        n = self.capacity

      # Step 10: Calculate remaining contiguous space available from the tail pointer to the end of the array
      end_space = self.capacity - self.tail

      # Step 11: Check if the incoming data fits entirely within the remaining tail end space
      if n <= end_space:
        # Copy data directly into the tail block
        self.buffer[self.tail : self.tail + n] = data
      else:
        # Split data: fill the remaining tail space first...
        self.buffer[self.tail :] = data[:end_space]
        # ...and wrap around the remaining bytes to the beginning of the buffer array
        self.buffer[: n - end_space] = data[end_space:]

      # Step 12: Update the tail pointer using modulo arithmetic to loop back to index 0 if it overflows
      self.tail = (self.tail + n) % self.capacity

      # Step 13: Increment current size count, capped strictly at maximum capacity
      self.size = min(self.capacity, self.size + n)

  def read(self, num_samples: int) -> np.ndarray:
    """Reads and removes an exact window size (W) of samples from the buffer.

    Args:
        num_samples (int): The exact number of samples needed (e.g., window size
          W = 1600).

    Returns:
        np.ndarray or None: A 1D float32 array of length num_samples, or None if
        not enough data is available yet.
    """
    # Step 14: Acquire the thread lock for atomic read operations
    with self.lock:
      # Step 15: Check if there are enough accumulated samples to fulfill the requested window size
      if self.size < num_samples:
        return None  # Return None if buffer underflows the requested chunk size

      # Step 16: Allocate a temporary output array for the exact window size
      out = np.zeros(num_samples, dtype=np.float32)

      # Step 17: Calculate contiguous space available from the head pointer to the end of the array
      end_space = self.capacity - self.head

      # Step 18: Check if the requested window can be read contiguously without wrapping around
      if num_samples <= end_space:
        out[:] = self.buffer[self.head : self.head + num_samples]
      else:
        # Handle wrap-around reading: read from head to end of buffer...
        out[:end_space] = self.buffer[self.head :]
        # ...and read the remainder from the beginning of the buffer array
        out[end_space:] = self.buffer[: num_samples - end_space]

      # Step 19: Advance the head pointer forward using modulo arithmetic
      self.head = (self.head + num_samples) % self.capacity

      # Step 20: Decrement the active tracked buffer size by the number of consumed samples
      self.size -= num_samples

      # Step 21: Return the extracted audio window slice
      return out