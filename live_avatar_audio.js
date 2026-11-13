class AvatarMicrophone extends AudioWorkletProcessor {
  constructor() {
    super();
    this.packet = new Float32Array(640);
    this.offset = 0;
    this.outputPort = null;
    this.port.onmessage = event => { if (event.data.bridge) this.outputPort = event.data.bridge; };
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.packet[this.offset++] = channel[i];
      if (this.offset === this.packet.length) {
        if (this.outputPort) this.outputPort.postMessage({samples:this.packet.buffer,capturedAt:Date.now(),sampleRate},[this.packet.buffer]);
        else this.port.postMessage(this.packet.buffer, [this.packet.buffer]);
        this.packet = new Float32Array(640);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("avatar-microphone", AvatarMicrophone);
