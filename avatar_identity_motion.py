"""Small autonomous gestures separated by long quiet holds; no video loop."""
import numpy as np

class ProceduralMotion:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed)
        self.blink_rng=np.random.default_rng(seed^0x5929a37)
        self.blinks=[float(self.blink_rng.uniform(2.5,4.5))]
        self.gestures=[];self.target=np.zeros(7);self.next_start=float(self.rng.uniform(2.5,4.))
    def closure(self,t):
        while self.blinks[-1]<t+1:self.blinks.append(self.blinks[-1]+float(self.blink_rng.uniform(3.,6.)))
        for at in self.blinks[-3:]:
            dt=t-at
            if -.10<=dt<0:return float(.5+.5*np.cos(np.pi*dt/.10))
            if 0<=dt<=.18:return float(.5+.5*np.cos(np.pi*dt/.18))
        return 0.
    @staticmethod
    def ease(u):
        u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))
    def pose(self,t):
        while self.next_start<t+1:
            start=self.next_start
            duration=float(self.rng.uniform(2.1,3.1))
            kind=int(self.rng.integers(0,3));sign=float(self.rng.choice([-1,1]))
            if kind==0:peak=np.array([self.rng.uniform(-.5,.5),sign*self.rng.uniform(2.4,3.2),self.rng.uniform(-.4,.4)])
            elif kind==1:peak=np.array([self.rng.uniform(.9,1.7),sign*self.rng.uniform(.7,1.5),sign*.35])
            else:peak=np.array([self.rng.uniform(-.6,.6),sign*self.rng.uniform(1.2,2.2),sign*self.rng.uniform(.5,.8)])
            # A new relaxed posture is held, rather than returning to exactly
            # the same neutral photo after every movement. Shoulders follow.
            goal=np.r_[peak,.07*peak[1]+.18*peak[2],.5*peak[1],self.rng.uniform(-.6,.6),self.rng.uniform(-.002,.002)]
            self.gestures.append((start,duration,self.target.copy(),goal))
            self.target=goal;self.next_start=start+duration+float(self.rng.uniform(5.,9.))
        while len(self.gestures)>2 and self.gestures[1][0]<t:self.gestures.pop(0)
        posture=np.zeros(7)
        for start,duration,previous,goal in self.gestures:
            if t<start:break
            posture=previous+(goal-previous)*self.ease((t-start)/duration)
        angles=posture[:3]+np.array([.055*np.sin(.31*t),.065*np.sin(.23*t),.03*np.sin(.19*t)])
        breath=np.sin(1.3*t+.15*np.sin(.21*t))
        # Breathing remains subtle; no continuous large side-to-side sway.
        body=np.array([posture[3]+.035*np.sin(.13*t),posture[4]+.2*np.sin(.17*t),posture[5]-.75*breath,1.+posture[6]+.0007*np.sin(.21*t),breath],np.float32)
        translation=np.array([[.0007*np.sin(.23*t),.0005*breath,0]],np.float32)
        return angles,translation,1.+.0008*np.sin(.29*t),self.closure(t),body


class Motion(ProceduralMotion):
    """Recombine measured personal gestures; never replay source video pixels."""
    def __init__(self,seed,bank_path):
        super().__init__(seed)
        with np.load(bank_path,allow_pickle=False) as bank:
            self.clips=bank['clips'];self.clip_fps=float(bank['fps'])
        if len(self.clips)<2 or self.clips.ndim!=3 or self.clips.shape[-1]!=9:
            raise ValueError('Invalid gesture bank')
        self.events=[]
        self.current=np.zeros(9);self.next_start=float(self.rng.uniform(1.5,2.5));self.last_clip=-1
    def pose(self,t):
        while self.next_start<t+1:
            choices=[i for i in range(len(self.clips)) if i!=self.last_clip]
            index=int(self.rng.choice(choices));self.last_clip=index
            clip=self.clips[index].copy()
            # Mirroring a small turn avoids always favouring the recording's side.
            if self.rng.random()<.5:clip[:,[1,2,3,4,7]]*=-1
            duration=(len(clip)-1)/self.clip_fps/float(self.rng.uniform(.9,1.1))
            previous=self.current.copy();u=np.linspace(0,1,len(clip))
            clip+=previous[None]*(1-self.ease(u))[:,None]
            clip=np.clip(clip,[-4,-6,-2.2,-.8,-5,-3,-.008,-.01,-.01],[4,6,2.2,.8,5,3,.008,.01,.01])
            self.events.append((self.next_start,duration,previous,clip));self.current=clip[-1]
            self.next_start+=duration+float(self.rng.uniform(3.5,6.5))
        while len(self.events)>2 and self.events[1][0]<t:self.events.pop(0)
        posture=np.zeros(9)
        for start,duration,previous,clip in self.events:
            if t<start:break
            pos=np.clip((t-start)/duration,0,1)*(len(clip)-1);lo=int(pos);hi=min(lo+1,len(clip)-1)
            posture=clip[lo]+(clip[hi]-clip[lo])*(pos-lo)
        breath=np.sin(1.3*t+.15*np.sin(.21*t))
        angles=posture[:3]+[.055*np.sin(.31*t),.065*np.sin(.23*t),.03*np.sin(.19*t)]
        body=np.array([posture[3],posture[4],posture[5]-.75*breath,1+posture[6],breath],np.float32)
        translation=np.array([[posture[7],posture[8]+.0005*breath,0]],np.float32)
        return angles,translation,1.,self.closure(t),body
