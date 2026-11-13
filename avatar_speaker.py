"""Local voice embedding. No identity service or external audio transmission."""
from avatar_paths import MODELS
import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf
MODEL=MODELS/'speaker/campplus-zh-en-advanced.onnx'

class VoiceEmbedding:
    def __init__(self):
        config=ort.SessionOptions();config.intra_op_num_threads=1;config.inter_op_num_threads=1
        self.model=ort.InferenceSession(str(MODEL),config,providers=['CPUExecutionProvider'])
        metadata=self.model.get_modelmeta().custom_metadata_map
        assert metadata['feature_normalize_type']=='global-mean' and metadata['normalize_samples']=='1'
        self.options=knf.FbankOptions()
        self.options.frame_opts.dither=0.;self.options.frame_opts.snip_edges=False
        self.options.mel_opts.num_bins=80;self.options.mel_opts.high_freq=-400
    def compute(self,audio):
        fbank=knf.OnlineFbank(self.options)
        fbank.accept_waveform(16000,np.asarray(audio,np.float32));fbank.input_finished()
        features=np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features-=features.mean(axis=0,keepdims=True)
        vector=self.model.run(None,{'x':features[None]})[0].reshape(-1)
        return vector/(np.linalg.norm(vector)+1e-12)

class OwnerVoiceGate:
    """Speaker decisions on source time, with explicit bounded lookahead.

    Does not separate simultaneous voices. Unknown identity stays closed. A
    decision is never latched for an entire sentence or carried across sessions.
    """
    def __init__(self,model,reference,threshold=.30,window_ms=1120,lookahead_ms=480,period_ms=160):
        self.model=model;self.reference=reference;self.threshold=threshold
        self.window_samples=window_ms*16;self.lookahead_samples=lookahead_ms*16;self.period_samples=period_ms*16
        self.audio=np.zeros(0,np.float32);self.received=0;self.decisions=()
        self.score=None;self.compute_ms=0.
    def push(self,samples):
        import time
        self.received+=len(samples);self.audio=np.concatenate([self.audio,samples])[-self.window_samples:]
        if self.received%self.period_samples or len(self.audio)<6400:return
        started=time.perf_counter()
        self.score=float(self.model.compute(self.audio)@self.reference)
        # Only already received packets participate; timing is explicit and
        # matched to the configured video deadline in the caller.
        at=max(0,self.received-self.lookahead_samples)
        self.decisions=tuple((p,w) for p,w in self.decisions if p>at-16000*10)+((at,float(self.score>=self.threshold)),)
        self.compute_ms=(time.perf_counter()-started)*1000
    def weight(self,frame):
        if frame<0:return 0.
        sample=(frame+.5)*640
        choices=self.decisions
        # Short symmetric fade is within the fixed lookahead; no long identity
        # hangover that would let the next speaker animate the avatar.
        for i in range(len(choices)-1,-1,-1):
            at,w=choices[i]
            if sample>=at:
                previous=choices[i-1][1] if i else 0.
                return float(previous+(w-previous)*min(1.,max(0.,(sample-at)/640)))
        return 0.
