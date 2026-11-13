"""Silero speech activity, independent of articulation or volume-driven lips.

CPU only. Audio remains local. Source-time intervals include onset pre-roll and
short-pause hangover; no neutral switch on brief unvoiced Russian consonants.
"""
from collections import deque
from avatar_paths import MODELS
import numpy as np
import onnxruntime as ort
MODEL=MODELS/'asr/silero-vad-5.1.2.onnx'

def load_model():
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    return ort.InferenceSession(str(MODEL),options,providers=['CPUExecutionProvider'])

class SpeechActivity:
    def __init__(self,model):
        self.model=model;self.pending=np.zeros(0,np.float32);self.context=np.zeros(64,np.float32)
        self.state=np.zeros((2,1,128),np.float32);self.rate=np.array(16000,np.int64)
        self.samples=0;self.received=0;self.hits=0;self.probability=0.;self.rms=0.
        self.intervals=deque();self.triggered=False;self.last_voice=-100000
    def push(self,samples):
        assert samples.ndim==1 and len(samples)==640 and np.isfinite(samples).all()
        self.rms=float(np.sqrt(np.mean(samples*samples)));self.received+=len(samples)
        self.pending=np.concatenate([self.pending,samples])
        while len(self.pending)>=512:
            chunk=self.pending[:512];self.pending=self.pending[512:]
            window=np.concatenate([self.context,chunk])[None]
            prob,self.state=self.model.run(None,{'input':window,'state':self.state,'sr':self.rate})
            self.context=chunk[-64:];self.samples+=512;self.probability=float(prob.item())
            # Two confident windows reject isolated clicks. The known future
            # buffer lets us recover the beginning rather than clip consonants.
            self.hits=self.hits+1 if self.probability>=.55 else 0
            if not self.triggered and self.hits>=2:
                start=max(0,self.samples-2*512-1600)
                if self.intervals and start<=self.intervals[-1][1]:self.intervals[-1][1]=self.samples+4800
                else:self.intervals.append([start,self.samples+4800])
                self.triggered=True;self.last_voice=self.samples
            if self.triggered:
                if self.probability>=.35:
                    self.last_voice=self.samples;self.intervals[-1][1]=self.samples+4800
                if self.samples-self.last_voice>4800:self.triggered=False
            while self.intervals and self.intervals[0][1]<self.samples-16000*12:self.intervals.popleft()
    def weight(self,frame):
        if frame<0:return 0.
        sample=(frame+.5)*640;result=0.
        for start,end in self.intervals:
            if start<=sample<=end:
                attack=min(1.,max(0.,(sample-start)/640))
                release=min(1.,max(0.,(end-sample)/1600))
                result=max(result,min(attack,release))
        return float(result)
