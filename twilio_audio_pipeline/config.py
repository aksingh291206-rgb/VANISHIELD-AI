# config.py

# Telephony input sample rate from Twilio Media Streams (standard landline/cellular audio)
TELEPHONY_SAMPLE_RATE = 8000

# Model target sample rate (deep learning audio frontends typically expect 16kHz)
MODEL_SAMPLE_RATE = 16000

# Window size (W) in samples: 100ms of audio at 16kHz = 1600 samples per chunk
WINDOW_SIZE_SAMPLES = 1600

# Hop size (H) in samples: 50% overlap (800 samples) or custom stride
HOP_SIZE_SAMPLES = 800

# Dimension of the feature embedding vector output by EmbNet
EMBEDDING_DIM = 128

# Hidden dimension size for the recurrent state layer (GRU / LSTM)
HIDDEN_DIM = 256