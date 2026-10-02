import os
import glob
import numpy as np
import torch
from twilio_audio_pipeline.config import WINDOW_SIZE_SAMPLES
from twilio_audio_pipeline.core.buffer import AudioRingBuffer
from twilio_audio_pipeline.core.decoder import AudioDecoder
from twilio_audio_pipeline.core.resampler import StreamingResampler
from twilio_audio_pipeline.models.inference import QuantizedInferenceEngine

def float_to_mulaw(pcm_data, mu=255):
    x = np.clip(pcm_data, -1.0, 1.0)
    encoded = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    quantized = ((encoded + 1) / 2 * mu).astype(np.uint8)
    return quantized.tobytes()

def run_batch_evaluation():
    print("🚀 Initializing Batch Evaluation Pipeline...")
    
    # 1. Initialize Inference Engine and Resampler
    inference_engine = QuantizedInferenceEngine(model_path="models/streaming_model_int8.onnx")
    THRESHOLD = 0.56
    
    G_DRIVE_DIR = r"G:\My Drive\Audio_Training_Data"
    real_files = glob.glob(os.path.join(G_DRIVE_DIR, "real", "*.npy"))
    spoof_files = glob.glob(os.path.join(G_DRIVE_DIR, "spoof", "*.npy"))
    
    # Select a balanced batch size (e.g., 5 real, 5 spoof)
    batch_real = real_files[:500]
    batch_spoof = spoof_files[:500]
    
    test_dataset = [(f, 0) for f in batch_real] + [(f, 1) for f in batch_spoof]
    
    if not test_dataset:
        print("⚠️ No dataset files found! Check your G Drive path.")
        return

    print(f"📊 Running evaluation across {len(test_dataset)} files (500 Real, 500 Spoof)...")
    print("-" * 60)
    print(f"{'File Type':<10} | {'File Name':<30} | {'Avg Score':<10} | {'Verdict'}")
    print("-" * 60)

    correct_predictions = 0
    total_files = len(test_dataset)

    for file_path, true_label in test_dataset:
        filename = os.path.basename(file_path)
        label_str = "SPOOF (1)" if true_label == 1 else "REAL (0)"
        
        try:
            audio_samples = np.load(file_path).astype(np.float32)
            max_val = np.max(np.abs(audio_samples))
            if max_val > 0:
                audio_samples /= max_val
        except Exception as e:
            print(f"Skipping {filename} due to load error: {e}")
            continue

        # Simulate streaming pipeline components
        ring_buffer = AudioRingBuffer(capacity_samples=WINDOW_SIZE_SAMPLES * 4)
        resampler = StreamingResampler(input_sample_rate=8000, output_sample_rate=16000)
        
        file_probabilities = []
        chunk_size = 160  # 20ms chunks at 8kHz
        
        for i in range(0, len(audio_samples) - chunk_size, chunk_size):
            chunk = audio_samples[i:i+chunk_size]
            
            # Simulate Telephony pipeline: PCM -> Mu-law bytes -> Decode -> Resample -> Ring Buffer
            mulaw_bytes = float_to_mulaw(chunk)
            decoded_8k = AudioDecoder.decode_mulaw_packet(base64_encode_sim(mulaw_bytes))
            resampled_16k = resampler.resample_chunk(decoded_8k)
            ring_buffer.write(resampled_16k)
            
            while ring_buffer.size >= WINDOW_SIZE_SAMPLES:
                window_data = ring_buffer.read(WINDOW_SIZE_SAMPLES)
                tensor_input = np.expand_dims(np.expand_dims(window_data, axis=0), axis=0)
                
                logits, _ = inference_engine.run(tensor_input)
                prob = torch.sigmoid(torch.tensor(logits)).item()
                file_probabilities.append(prob)

        if not file_probabilities:
            continue

        # Average score across all sliding windows for this file
        avg_score = np.percentile(file_probabilities,90)
        predicted_label = 1 if avg_score >= THRESHOLD else 0
        
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
            
        verdict_status = "✅ CORRECT" if is_correct else "❌ WRONG"
        pred_str = "SPOOF" if predicted_label == 1 else "REAL"
        
        print(f"{label_str:<10} | {filename[:28]:<30} | {avg_score:.4f}     | {pred_str} ({verdict_status})")

    accuracy = (correct_predictions / total_files) * 100
    print("-" * 60)
    print(f"🎯 Batch Evaluation Complete! Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_files} correct)")

def base64_encode_sim(mulaw_bytes):
    import base64
    return base64.b64encode(mulaw_bytes).decode('utf-8')

if __name__ == "__main__":
    run_batch_evaluation()