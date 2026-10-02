# core/resampler.py

import numpy as np  # NumPy for handling float32 audio arrays
import soxr  # High-quality 1D sample-rate conversion library (libsoxr bindings)


class StreamingResampler:
  """Handles real-time streaming audio sample rate conversion

  from telephony standard (8kHz) to deep learning model requirement (16kHz).
  """

  def __init__(
      self,
      input_sample_rate: int = 8000,
      output_sample_rate: int = 16000,
  ):
    """Initializes the streaming resampler instance with target rates.

    Args:
        input_sample_rate (int): Incoming audio sample rate from Twilio (default
          8000 Hz).
        output_sample_rate (int): Target model input sample rate (default 16000
          Hz).
    """
    # Step 1: Store the input sample rate attribute for processing checks
    self.in_sr = input_sample_rate

    # Step 2: Store the target output sample rate attribute
    self.out_sr = output_sample_rate

    # Step 3: Initialize a persistent streaming resampler state object using soxr
    # Using 'HQ' (High Quality) profile to maintain audio frequency fidelity without massive CPU latency
    self.resample_stream = soxr.ResampleStream(
        self.in_sr, self.out_sr, num_channels=1, quality="HQ"
    )

  def resample_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
    """Resamples a single live audio data chunk from input rate to output rate.

    Args:
        audio_chunk (np.ndarray): 1D NumPy array of float32 audio samples at
          8kHz.

    Returns:
        np.ndarray: 1D NumPy array of float32 audio samples up-sampled to 16kHz.
    """
    # Step 4: Check if the input chunk is empty; if so, return an empty array immediately to avoid errors
    if audio_chunk.size == 0:
      return np.array([], dtype=np.float32)

    # Step 5: Ensure the incoming numpy array is formatted strictly as float32
    audio_float32 = audio_chunk.astype(np.float32)

    # Step 6: Pass the chunk through the persistent soxr streaming resampler state
    resampled_output = self.resample_stream.resample_chunk(audio_float32)

    # Step 7: Ensure output is strictly typed as a float32 NumPy array before handing off to the buffer
    return resampled_output.astype(np.float32) 