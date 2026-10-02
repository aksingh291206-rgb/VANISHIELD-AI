# core/decoder.py

import base64  # Standard library for decoding base64 strings sent by Twilio
import numpy as np  # NumPy for efficient numerical array manipulation


class AudioDecoder:
  """A utility class to decode base64 encoded mu-law audio packets

  received from telephony CPaaS providers into normalized float32 arrays.
  """

  @staticmethod
  def decode_mulaw_packet(base64_payload: str) -> np.ndarray:
    """Decodes a single base64 string payload from a Twilio WebSocket message.

    Args:
        base64_payload (str): The raw base64 encoded audio string from Twilio JSON.

    Returns:
        np.ndarray: A 1D NumPy array of float32 values scaled between -1.0 and
        1.0.
    """
    # Step 1: Decode the base64 string back into raw binary bytes
    raw_bytes = base64.b64decode(base64_payload)

    # Step 2: Convert raw bytes into a NumPy array of unsigned 8-bit integers
    mulaw_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Step 3: Invert the bits of the mu-law byte (telephony standard format requirement)
    mulaw_bytes = ~mulaw_bytes

    # Step 4: Extract sign, exponent, and mantissa from the 8-bit mu-law value
    # Isolate the sign bit (the 8th bit)
    sign = np.bitwise_and(mulaw_bytes, 0x80)
    # Isolate the exponent bits (bits 5, 6, 7) and shift them down
    exponent = np.bitwise_and(np.right_shift(mulaw_bytes, 4), 0x07)
    # Isolate the mantissa bits (the lower 4 bits)
    mantissa = np.bitwise_and(mulaw_bytes, 0x0F)

    # Step 5: Reconstruct the original 16-bit linear PCM audio magnitude using the mu-law expansion formula
    # Formula: 33 * (2^exponent + mantissa * 2^(exponent - 4) - 1)
    # Using float operations to prevent integer overflow during bit shifts
    linear_pcm = (33.0 * (np.left_shift(1, exponent) + (mantissa * np.left_shift(1, np.maximum(exponent - 4, 0))) - 1.0))

    # Step 6: Apply the sign bit back to the reconstructed linear value (negative if sign bit is set)
    linear_pcm = np.where(sign != 0, -linear_pcm, linear_pcm)

    # Step 7: Normalize the 16-bit integer values (-32768 to 32767) into standard ML float range [-1.0, 1.0]
    normalized_audio = linear_pcm / 32768.0

    # Step 8: Ensure array is explicitly cast to float32 for PyTorch / ONNX compatibility
    return normalized_audio.astype(np.float32)