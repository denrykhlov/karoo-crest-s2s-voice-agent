/**
 * AudioWorkletProcessor that buffers 24 kHz mono Int16 PCM audio from the server,
 * starts playback immediately after a 40 ms (960-sample < 1440-sample Frame-1) prebuffer,
 * emits `started` with turnId so perceived TTFA can be timed accurately, and supports
 * instant zero-tail clearing on user barge-in.
 */
class PlaybackWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceRate = 24000;
    this.step = this.sourceRate / sampleRate;
    this.capacity = this.sourceRate * 30; // 30s ring buffer
    this.ring = new Float32Array(this.capacity);
    this.writePos = 0;
    this.readPos = 0;
    this.available = 0;
    this.srcFrac = 0.0;

    // 40 ms prebuffer (960 samples < 1440-sample Frame 1 => immediate audible onset on Chunk 1)
    this.prebufferSamples = 960;
    this.started = false;
    this.endOfStream = false;
    this.played24k = 0;
    this.samplesSinceReport = 0;

    this.activeTurnId = null;
    this.reportedStartedForTurn = null;

    this.port.onmessage = (event) => {
      const msg = event.data;
      if (!msg) return;
      if (msg.type === "new_turn") {
        this.activeTurnId = msg.turnId || null;
        this.reportedStartedForTurn = null;
        this.endOfStream = false;
        this.played24k = 0;
        this.samplesSinceReport = 0;
        if (this.available === 0) {
          this.started = false;
        }
      } else if (msg.type === "audio" && msg.buffer) {
        const pcm16 = new Int16Array(msg.buffer);
        this._pushSamples(pcm16);
      } else if (msg.type === "clear") {
        this._clear();
      } else if (msg.type === "end_of_stream") {
        this.endOfStream = true;
      }
    };
  }

  _clear() {
    this.writePos = 0;
    this.readPos = 0;
    this.available = 0;
    this.srcFrac = 0.0;
    this.started = false;
    this.endOfStream = false;
    this.played24k = 0;
    this.samplesSinceReport = 0;
    this.reportedStartedForTurn = null;
  }

  _pushSamples(pcm16) {
    for (let i = 0; i < pcm16.length; i++) {
      if (this.available >= this.capacity) {
        this.readPos = (this.readPos + 1) % this.capacity;
        this.available--;
      }
      this.ring[this.writePos] = pcm16[i] / 32768.0;
      this.writePos = (this.writePos + 1) % this.capacity;
      this.available++;
    }
  }

  process(_inputs, outputs) {
    const outputChannels = outputs[0];
    if (!outputChannels || outputChannels.length === 0) {
      return true;
    }
    const out = outputChannels[0];

    if (!this.started) {
      if (
        this.available >= this.prebufferSamples ||
        (this.endOfStream && this.available > 0)
      ) {
        this.started = true;
        if (
          this.activeTurnId &&
          this.reportedStartedForTurn !== this.activeTurnId
        ) {
          this.reportedStartedForTurn = this.activeTurnId;
          this.port.postMessage({
            type: "started",
            turnId: this.activeTurnId,
            bufferedSamples: this.available,
          });
        }
      } else {
        out.fill(0);
        if (this.endOfStream && this.available === 0) {
          this.endOfStream = false;
          this.port.postMessage({
            type: "drained",
            playedSamples: this.played24k,
          });
        }
        return true;
      }
    }

    if (
      this.started &&
      this.activeTurnId &&
      this.reportedStartedForTurn !== this.activeTurnId &&
      this.available > 0
    ) {
      this.reportedStartedForTurn = this.activeTurnId;
      this.port.postMessage({
        type: "started",
        turnId: this.activeTurnId,
        bufferedSamples: this.available,
      });
    }

    let sumSq = 0.0;
    for (let i = 0; i < out.length; i++) {
      if (this.available === 0) {
        out[i] = 0.0;
        continue;
      }
      const s0 = this.ring[this.readPos];
      const nextIdx = (this.readPos + 1) % this.capacity;
      const s1 = this.available > 1 ? this.ring[nextIdx] : s0;
      const val = s0 + (s1 - s0) * this.srcFrac;
      out[i] = val;
      sumSq += val * val;

      this.srcFrac += this.step;
      while (this.srcFrac >= 1.0 && this.available > 0) {
        this.srcFrac -= 1.0;
        this.readPos = (this.readPos + 1) % this.capacity;
        this.available--;
        this.played24k++;
      }
    }

    for (let ch = 1; ch < outputChannels.length; ch++) {
      outputChannels[ch].set(out);
    }

    this.samplesSinceReport += out.length;
    if (this.samplesSinceReport >= 3840) {
      this.samplesSinceReport = 0;
      const rms = Math.sqrt(sumSq / out.length);
      this.port.postMessage({
        type: "progress",
        playedSamples: this.played24k,
        rms,
      });
    }

    if (this.started && this.endOfStream && this.available === 0) {
      this.started = false;
      this.endOfStream = false;
      this.port.postMessage({
        type: "drained",
        playedSamples: this.played24k,
      });
    }

    return true;
  }
}

registerProcessor("playback-worklet", PlaybackWorkletProcessor);
