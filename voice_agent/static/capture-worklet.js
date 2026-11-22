/**
 * AudioWorkletProcessor that captures browser microphone audio,
 * resamples from AudioContext sampleRate to 16 kHz mono, and emits
 * 512-sample (32 ms) little-endian Int16 PCM frames.
 */
class CaptureWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.frameSize = 512;
    this.ratio = sampleRate / this.targetRate;
    this.srcTail = new Float32Array(0);
    this.srcPos = 0.0;
    this.outBuf = new Float32Array(this.frameSize);
    this.outIdx = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) {
      return true;
    }

    const buf = new Float32Array(this.srcTail.length + input.length);
    buf.set(this.srcTail, 0);
    buf.set(input, this.srcTail.length);

    while (this.srcPos + 1 < buf.length) {
      const i0 = Math.floor(this.srcPos);
      const frac = this.srcPos - i0;
      const sample = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac;
      this.outBuf[this.outIdx++] = sample;
      this.srcPos += this.ratio;

      if (this.outIdx >= this.frameSize) {
        const pcm16 = new Int16Array(this.frameSize);
        let sumSq = 0.0;
        for (let i = 0; i < this.frameSize; i++) {
          const s = Math.max(-1.0, Math.min(1.0, this.outBuf[i]));
          sumSq += s * s;
          pcm16[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
        }
        const rms = Math.sqrt(sumSq / this.frameSize);
        this.port.postMessage(
          { type: "pcm16", buffer: pcm16.buffer, rms },
          [pcm16.buffer]
        );
        this.outIdx = 0;
      }
    }

    const consumed = Math.floor(this.srcPos);
    this.srcTail = buf.slice(consumed);
    this.srcPos -= consumed;
    return true;
  }
}

registerProcessor("capture-worklet", CaptureWorkletProcessor);
