# core/session.py

import threading  # Threading lock to protect session dictionary modifications across concurrent threads
from core.buffer import AudioRingBuffer  # Import our thread-safe ring buffer class
from config import WINDOW_SIZE_SAMPLES  # Import window size constant (e.g., 1600 samples)


class CallSession:
  """Represents an individual active phone call session,

  encapsulating its own ring buffer and recurrent neural network hidden state.
  """

  def __init__(self, call_id: str, buffer_capacity: int = 80000):
    """Initializes a new call session container.

    Args:
        call_id (str): Unique identifier for the live call (e.g., Twilio
          StreamSid).
        buffer_capacity (int): Total sample capacity for this session's ring
          buffer (default 80000 = 5 seconds at 16kHz).
    """
    # Step 1: Store the unique string identifier for this specific phone call
    self.call_id = call_id

    # Step 2: Instantiate a dedicated AudioRingBuffer unique to this call session
    self.ring_buffer = AudioRingBuffer(capacity_samples=buffer_capacity)

    # Step 3: Initialize the recurrent network hidden state as None (will be initialized by the RNN on the first chunk)
    self.hidden_state = None

    # Step 4: Record a timestamp or track packet counters if session health monitoring is needed
    self.is_active = True


class SessionManager:
  """Manages multiple concurrent live call sessions safely,

  preventing data leakage between users and handling clean memory disposal on disconnect.
  """

  def __init__(self):
    """Initializes a thread-safe dictionary container to hold all active call sessions."""
    # Step 5: Dictionary mapping call_id strings to respective CallSession objects
    self.sessions = {}

    # Step 6: Thread lock to prevent race conditions when creating or deleting sessions concurrently
    self.lock = threading.Lock()

  def get_or_create_session(self, call_id: str) -> CallSession:
    """Retrieves an existing session or creates a new one if it doesn't exist yet.

    Args:
        call_id (str): Unique call identifier from the WebSocket stream.

    Returns:
        CallSession: The session object assigned to this call ID.
    """
    # Step 7: Acquire thread lock to ensure thread-safe dictionary access
    with self.lock:
      # Step 8: Check if the call_id is not yet tracked in our active sessions dictionary
      if call_id not in self.sessions:
        # Step 9: Instantiate a brand new CallSession and store it against the call_id
        self.sessions[call_id] = CallSession(call_id=call_id)

      # Step 10: Return the existing or newly created session object
      return self.sessions[call_id]

  def remove_session(self, call_id: str):
    """Safely deletes a call session and purges its buffers/hidden states from RAM.

    Args:
        call_id (str): Unique identifier of the disconnected call.
    """
    # Step 11: Acquire thread lock before modifying the session dictionary
    with self.lock:
      # Step 12: Check if the session exists in memory to avoid KeyError exceptions
      if call_id in self.sessions:
        # Step 13: Mark the session as inactive to halt any pending background processes
        self.sessions[call_id].is_active = False

        # Step 14: Delete the session object from the dictionary, triggering garbage collection of its ring buffer and hidden state tensors
        del self.sessions[call_id]