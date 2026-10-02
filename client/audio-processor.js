// client/audio-processor.js

/**
 * PCMProcessor extends AudioWorkletProcessor to run audio DSP(Digital Signal Processing) off the main thread.
 * It intercepts raw microphone audio, converts it to 16kHz mono, and packs it as 16-bit PCM (16-bit signed integers).
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    // Call the parent AudioWorkletProcessor constructor
    super();
    
    // Target sample rate expected by deepfake detection models (16kHz)
    this.targetSampleRate = 16000;
  }

  /**
   * process() is called automatically by the Web Audio API engine every ~128 frames.
   * @param {Array} inputs - Array of input channels containing Float32 raw audio.
   * @param {Array} outputs - Array of output channels (unused here as we don't play back audio).
   * @param {Object} parameters - Dynamic parameters passed to the worklet (unused).
   * @returns {boolean} Must return true to keep the worklet thread alive.
   */
  process(inputs, outputs, parameters) {
    // Grab the first input device (the user's active microphone)
    const input = inputs[0];

    // Debug check: If input is empty, uninitialized, or muted, return true to keep processor listening
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    // Reference the primary channel array to get frame length (usually 128 float samples)
    const inputChannel = input[0];
    
    // Get total input channels provided by hardware (1 for mono, 2 for stereo, etc.)
    const numChannels = input.length;
    
    // Allocate a temporary Float32 array to store downmixed mono samples
    const monoSamples = new Float32Array(inputChannel.length);

    // =========================================================================
    // STEP 1: DOWNMIX MULTI-CHANNEL STEREO TO SINGLE-CHANNEL MONO
    // Formula: Mono = (Channel_1 + Channel_2 + ... + Channel_N) / N
    // =========================================================================
    for (let i = 0; i < inputChannel.length; i++) {
      let channelSum = 0;
      
      // Sum the amplitudes across all physical hardware channels
      for (let ch = 0; ch < numChannels; ch++) {
        channelSum += input[ch][i];
      }
      
      // Compute the average value across channels to produce mono audio
      monoSamples[i] = channelSum / numChannels;
    }

    // =========================================================================
    // STEP 2: LINEAR DOWNSAMPLING TO 16,000 Hz
    // Native browser rates are usually 44,100Hz or 48,000Hz.
    // =========================================================================
    // Calculate downsampling ratio (e.g., 48000 / 16000 = 3.0)
    const resampleRatio = sampleRate / this.targetSampleRate;
    
    // Compute the resulting sample array length after downsampling
    const resampledLength = Math.floor(monoSamples.length / resampleRatio);
    
    // Allocate buffer for 16kHz resampled Float32 audio samples
    const resampledSamples = new Float32Array(resampledLength);

    // Step through the downsampled array and map source indices back to monoSamples
    for (let i = 0; i < resampledLength; i++) {
      // Find corresponding sample index in the original native-rate array
      const originIndex = Math.floor(i * resampleRatio);
      
      // Assign mapped sample value to downsampled array
      resampledSamples[i] = monoSamples[originIndex];
    }

    // =========================================================================
    // STEP 3: QUANTIZATION (Float32 [-1.0, 1.0] -> Signed Int16 [-32768, 32767])
    // Reduces binary size by 50% compared to transmitting raw 32-bit floats.
    // =========================================================================
    // Allocate 16-bit signed integer buffer matching resampled sample length
    const pcm16Buffer = new Int16Array(resampledSamples.length);

    for (let i = 0; i < resampledSamples.length; i++) {
      // Clamp float sample value to strict [-1.0, 1.0] range to prevent digital clipping overflow
      const clampedSample = Math.max(-1, Math.min(1, resampledSamples[i]));
      
      // Scale negative float to Int16 minimum (-32768) or positive float to Int16 maximum (+32767)
      pcm16Buffer[i] = clampedSample < 0 ? clampedSample * 0x8000 : clampedSample * 0x7fff;
    }

    // =========================================================================
    // STEP 4: EMIT RAW BINARY BUFFER TO MAIN CLIENT THREAD
    // Transfer the underlying ArrayBuffer ownership directly without copying memory.
    // =========================================================================
    this.port.postMessage(pcm16Buffer.buffer, [pcm16Buffer.buffer]);

    // Keep processor active for continuous streaming
    return true;
  }
}

// Register the processor name so main thread web page can load it via AudioWorkletNode
registerProcessor("pcm-processor", PCMProcessor);