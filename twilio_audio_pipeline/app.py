# app.py

import base64  # Standard library for parsing base64 encoded strings
import json  # Library for parsing JSON frames sent over WebSockets by Twilio
import numpy as np  # NumPy for audio array manipulation
import torch  # PyTorch for tensor/sigmoid calculations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # FastAPI web framework for async WebSocket endpoints

# Import custom modules built in previous steps
from .config import WINDOW_SIZE_SAMPLES  # Import window size constant W (1600 samples)
from .core.buffer import AudioRingBuffer  # Thread-safe ring buffer class
from .core.decoder import AudioDecoder  # Telephony mu-law to PCM decoder utility
from .core.resampler import StreamingResampler  # Real-time streaming resampler (8kHz to 16kHz)
from .core.session import SessionManager  # Manager tracking concurrent call states and buffers
from .models.inference import QuantizedInferenceEngine  # Quantized ONNX inference execution wrapper

# Step 1: Initialize the FastAPI application instance
app = FastAPI()

# Step 2: Initialize global session manager to handle concurrent phone calls safely
session_manager = SessionManager()

# Step 3: Initialize the shared Quantized ONNX Inference Engine globally to load model weights into RAM once at startup
inference_engine = QuantizedInferenceEngine(model_path="models/streaming_model_int8.onnx")


@app.websocket("/ws/live-call/{call_id}")
async def live_call_websocket_endpoint(websocket: WebSocket, call_id: str):
    """Asynchronous WebSocket endpoint handling live bi-directional audio streams
    initiated by Twilio Media Streams when a user calls your phone number.
    """
    # Step 4: Accept the incoming WebSocket connection handshake from Twilio
    await websocket.accept()
    print(f"Live phone call connected | Stream Session ID: {call_id}")

    # Step 5: Retrieve or initialize a dedicated session for this specific call ID
    call_session = session_manager.get_or_create_session(call_id)

    # Step 6: Instantiate a dedicated real-time streaming resampler for this connection (8kHz input -> 16kHz model target)
    resampler = StreamingResampler(input_sample_rate=8000, output_sample_rate=16000)

    try:
        # Step 7: Enter an infinite asynchronous loop to process incoming stream packets frame-by-frame
        while True:
            # Step 8: Await incoming text data frames from the Twilio WebSocket connection
            message = await websocket.receive_text()

            # Step 9: Parse the incoming JSON text string into a Python dictionary
            data = json.loads(message)

            # Step 10: Check the event type of the incoming Twilio message frame
            event_type = data.get("event")

            # Step 11: Handle media audio payload frames sent during active conversation
            if event_type == "media":
                # Extract the nested media dictionary containing the base64 audio string
                media_payload = data.get("media", {})
                base64_audio_string = media_payload.get("payload")

                if base64_audio_string:
                    # Step 12: Decode telephony 8-bit mu-law base64 packet into a float32 array at 8kHz
                    decoded_chunk_8k = AudioDecoder.decode_mulaw_packet(base64_audio_string)

                    # Step 13: Up-sample the 8kHz audio chunk to 16kHz to match model expectations
                    resampled_chunk_16k = resampler.resample_chunk(decoded_chunk_8k)

                    # Step 14: Push the resampled chunk into this call session's thread-safe ring buffer
                    call_session.ring_buffer.write(resampled_chunk_16k)

                    # Step 15: Extract exact window sizes (W = 1600 samples) continuously as data accumulates
                    while call_session.ring_buffer.size >= WINDOW_SIZE_SAMPLES:
                        # Pop the exact window slice from the ring buffer
                        window_window_data = call_session.ring_buffer.read(WINDOW_SIZE_SAMPLES)

                        # Step 16: Reshape array into tensor dimensions expected by model: [Batch=1, Channels=1, W=1600]
                        tensor_input = np.expand_dims(np.expand_dims(window_window_data, axis=0), axis=0)

                        # Step 17: Run inference and propagate the recurrent hidden state forward
                        prediction_logits, call_session.hidden_state = inference_engine.run(
                            audio_chunk_tensor=tensor_input,
                            hidden_state=call_session.hidden_state
                        )

                        # Step 18: Analyze classification scores with the 0.7 threshold
                        raw_output = torch.tensor(prediction_logits)
                        probability = torch.sigmoid(raw_output).item()
                        
                        THRESHOLD = 0.7
                        is_spoof = probability >= THRESHOLD
                        
                        if is_spoof:
                            print(f"🚨 [CALL {call_id}] SPOOF / DEEPFAKE DETECTED! Score: {probability:.4f}")
                        else:
                            print(f"✅ [CALL {call_id}] Real Voice. Score: {probability:.4f}")

            # Step 19: Handle call termination events sent gracefully by Twilio
            elif event_type == "stop":
                print(f"Call stop event received from Twilio for session: {call_id}")
                break

    except WebSocketDisconnect:
        # Step 20: Handle unexpected client disconnection or network drops gracefully
        print(f"Live call disconnected abruptly | Session ID: {call_id}")

    finally:
        # Step 21: Clean up and purge session resources from RAM to prevent memory leaks on server
        session_manager.remove_session(call_id)
        print(f"Session resources cleaned up for call ID: {call_id}")