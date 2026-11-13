"""Loopback-only live identity avatar for OBS; physical capture starts in the UI."""
import argparse
import asyncio
from collections import deque
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time

from avatar_paths import DH, ROOT
from aiohttp import web,WSMsgType
import cv2
import numpy as np
from avatar_speech_activity import SpeechActivity,load_model,MODEL as ACTIVITY_MODEL
from avatar_speaker import VoiceEmbedding,OwnerVoiceGate,MODEL as SPEAKER_MODEL

STATIC=DH/'web_demo/static'
FPS=25
RENDER_LEAD=.03


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Feed:
    def __init__(self,loop,seed,profile,video_delay_ms=360,cable_delay_ms=280,fixture_dir=None,direct_camera=False):
        self.loop=loop;self.seed=seed;self.stop=threading.Event();self.lock=threading.Lock()
        self.delay=video_delay_ms/1000;self.cable_delay_ms=cable_delay_ms
        self.fixture_dir=fixture_dir
        self.direct_camera=direct_camera
        self.profile=profile
        self.motion_path=profile / "gestures.npz"
        self.owner_path=profile / "owner.npy"
        self.owner_reference=None
        self.speaker_model=None
        self.owner_lookahead_ms=video_delay_ms-40;self.render_lead=RENDER_LEAD if video_delay_ms>=380 else .015
        self.updated=asyncio.Event();self.shutdown=asyncio.Event()
        self.heads={};self.coeffs={};self.latest=None;self.ready=False;self.error=None
        self.origin=time.monotonic()+30;self.audio_epoch=None;self.mic=None;self.stream_id=None
        self.frames=0;self.head_frames=0;self.head_late=0;self.render_late=0;self.coeff_late=0
        self.head_ms=deque(maxlen=400);self.render_ms=deque(maxlen=400);self.arrivals=deque(maxlen=400)
        self.audio_received_frames=0;self.last_coefficient_input=None
        self.activity=None;self.speech_weight=0.
        self.owner=None;self.owner_only=True
        self.slow_frames=deque(maxlen=64)
        self.capture_clock=None;self.stream_resets=0
        self.head_resyncs=0;self.last_head_slot=None
        self.version={p.name:digest(p) for p in [Path(__file__),ROOT/'avatar_identity_renderer.py',ROOT/'avatar_identity_motion.py',ROOT/'avatar_speaker.py',ROOT/'avatar_speech_activity.py',ROOT/'avatar_identity_live.js',ROOT/'avatar_identity_live.html',ROOT/'avatar_identity_audio.js',ROOT/'avatar_identity_worker.js',ROOT/'avatar_identity_feed.js',ROOT/'live_avatar_audio.js']}

    def health(self):
        now=time.monotonic()
        def stat(xs):return {'median':float(np.median(xs)),'p95':float(np.percentile(xs,95))} if xs else None
        with self.lock:
            return {'ready':self.ready,'error':self.error,'pid':os.getpid(),'version':'portrait-preview-1','code_sha256':self.version,
                'mode':'microphone + continuous procedural pose + LivePortrait + original detail + DH WASM coefficients',
                'replay_loop':False,'fps_target':25,'audio_video_buffer_ms':round(self.delay*1000),'cable_audio_buffer_ms':self.cable_delay_ms,'seed':self.seed,
                'camera_transport':'direct-obs-driver' if self.direct_camera else 'obs-browser-source',
                'output_width':720,'owner_available':self.owner_reference is not None,
                'fixture_mode':self.fixture_dir is not None,
                'render_lead_ms':round(self.render_lead*1000),
                'capture_clock':self.capture_clock,'stream_resets':self.stream_resets,
                'capture_frontier_lag_ms':max(0.,(now-self.audio_epoch-self.activity.received/16000)*1000) if self.audio_epoch is not None and self.activity else None,
                'coefficient_age_ms':(now-self.last_coefficient_input)*1000 if self.last_coefficient_input is not None else None,
                'frames':self.frames,'head_frames':self.head_frames,'head_late':self.head_late,'render_late':self.render_late,'coefficient_late':self.coeff_late,
                'head_ms':stat(self.head_ms),'frame_ms':stat(self.render_ms),'head_buffer_frames':len(self.heads),
                'head_resyncs':self.head_resyncs,'head_lead_ms':(self.origin+self.last_head_slot/FPS-now)*1000 if self.last_head_slot is not None else None,
                'slow_frames':list(self.slow_frames),
                'stream_active':self.mic is not None,'audio_received_frames':self.audio_received_frames,'stream_id':self.stream_id,
                'microphone_rms':self.activity.rms if self.activity else 0.,'speech_probability':self.activity.probability if self.activity else 0.,'speech_weight':self.speech_weight,
                'owner_filter':self.owner_only,'owner_score':self.owner.score if self.owner else None,'owner_compute_ms':self.owner.compute_ms if self.owner else None,
                'motion':'recorded gestures' if self.motion_path.is_file() else 'procedural gestures',
                'articulation_gain':1.30,'lip_support_radius':0,'upper_teeth_vertical_fit':.045,
                'owner_gate':{'window_ms':1120,'lookahead_ms':self.owner_lookahead_ms,'period_ms':80,'threshold':.30},
                'texture':'native source sampling; original skin at native mouth resolution',
                'model_hashes':getattr(self,'model_hashes',{}),'last_output_age_ms':(now-self.latest['time'])*1000 if self.latest else None,
                'acceptance':'PENDING: OBS/Zoom and perceptual review are separate from process health'}

    async def initialize(self):
        from avatar_identity_renderer import Head
        self.head=await asyncio.to_thread(Head,self.profile)
        self.activity_model=await asyncio.to_thread(load_model)
        self.model_hashes=dict(self.head.model.hashes)
        self.model_hashes['DHLiveMini.wasm']=digest(STATIC/'DHLiveMini.wasm')
        self.model_hashes['DINet_mini']=digest(DH/'checkpoint/DINet_mini/epoch_40_new.pth')
        self.model_hashes['silero_vad']=digest(ACTIVITY_MODEL)
        if self.owner_path.is_file():
            self.speaker_model=await asyncio.to_thread(VoiceEmbedding)
            self.owner_reference=np.load(self.owner_path,allow_pickle=False)
            if self.owner_reference.shape!=(192,) or not np.isfinite(self.owner_reference).all():
                raise ValueError('Invalid owner voice profile')
            self.model_hashes['speaker_embedding']=digest(SPEAKER_MODEL)
        self.head_ready=threading.Event();self.mouth_ready=threading.Event()
        self.head_thread=threading.Thread(target=self.head_worker,daemon=True,name='identity-head')
        self.mouth_thread=threading.Thread(target=self.mouth_worker,daemon=True,name='identity-mouth')
        self.origin=time.monotonic()+1.2
        self.head_thread.start();self.mouth_thread.start()
        await asyncio.to_thread(self.mouth_ready.wait,40)
        if self.error:raise RuntimeError(self.error)
        if not self.mouth_ready.is_set():raise RuntimeError('Mouth initialization timed out')
        self.ready=True

    def fail(self,error):
        logging.exception('Live renderer failure')
        self.error=str(error);self.ready=False;self.stop.set()
        if hasattr(self,'head_ready'):self.head_ready.set()
        if hasattr(self,'mouth_ready'):self.mouth_ready.set()
        self.loop.call_soon_threadsafe(self.shutdown.set)

    def head_worker(self):
        from avatar_identity_motion import Motion,ProceduralMotion
        from avatar_identity_renderer import torch
        try:
            motion=Motion(self.seed,self.motion_path) if self.motion_path.is_file() else ProceduralMotion(self.seed)
            slot=0;previous=None;previous_slot=None
            stream=torch.cuda.Stream(priority=0)
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                while not self.stop.is_set():
                    current=math.floor((time.monotonic()-self.origin)*FPS)
                    if slot<current:
                        # A GPU stall must not leave us computing a permanently
                        # obsolete animation queue. Rejoin the current pose clock.
                        slot=current+4;previous=None;previous_slot=None;self.head_resyncs+=1
                    if slot>current+30:
                        self.stop.wait(.01);continue
                    head=self.head.render(motion.pose(slot/FPS))
                    midpoint=self.head.midpoint(previous,head) if previous is not None and slot-previous_slot==2 else None
                    with self.lock:
                        self.heads[slot]=head
                        if midpoint is not None:self.heads[slot-1]=midpoint
                        self.head_frames+=1;self.head_ms.append(head['head_ms'])
                        self.last_head_slot=slot
                        for old in [k for k in self.heads if k<current-4]:del self.heads[old]
                    self.head_ready.set();previous=head;previous_slot=slot
                    # Compute every blink frame. Only slow head motion is interpolated.
                    near_blink=any(motion.closure((slot+j)/FPS)>0 for j in range(4))
                    slot+=1 if near_blink else 2
        except Exception as e:self.fail(e)

    def notify(self):
        self.updated.set();self.updated=asyncio.Event()

    def mouth_worker(self):
        from avatar_identity_renderer import Mouth,torch
        mouth=None;camera=None
        try:
            stream=torch.cuda.Stream(priority=-1);stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                mouth=Mouth(self.head.combined)
                self.head_ready.wait(30)
                # Prime DINet/CUDA before accepting the first word. Silence now
                # bypasses DINet; without this warm-up the first word stalls.
                with self.lock:warm_head=self.heads[min(self.heads)]
                for _ in range(3):mouth.render(warm_head,np.zeros(12,np.float32),1.)
                if self.direct_camera:
                    import pyvirtualcam
                    camera=pyvirtualcam.Camera(1280,720,FPS,fmt=pyvirtualcam.PixelFormat.BGR,backend='obs')
                    camera_frame=np.zeros((720,1280,3),np.uint8)
                self.mouth_ready.set()
                slot=0;last_head=None;previous_coeff=np.zeros(12,np.float32)
                while not self.stop.is_set():
                    target=self.origin+slot/FPS
                    # Coefficients arrive before the display deadline. Compute
                    # ahead of it, then publish at the deadline instead of only
                    # starting expensive rendering when the frame is due.
                    self.stop.wait(max(0,target-self.render_lead-time.monotonic()))
                    if self.stop.is_set():break
                    started=time.monotonic()
                    with self.lock:
                        head=self.heads.get(slot)
                        if head is None:
                            head=last_head or (self.heads.get(min(self.heads)) if self.heads else None)
                            if last_head is not None:self.head_late+=1
                        audio_slot=math.floor((target-self.audio_epoch-self.delay)*FPS+.00001) if self.audio_epoch is not None else -1
                        if audio_slot>=0:
                            coeff=self.coeffs.get(audio_slot)
                            if coeff is None:
                                coeff=previous_coeff
                                self.coeff_late+=1
                            else:previous_coeff=coeff
                        else:coeff=np.zeros(12,np.float32)
                        for old in [k for k in self.coeffs if k<audio_slot-10]:del self.coeffs[old]
                    if head is None:
                        self.stop.wait(.02);continue
                    last_head=head
                    # Prediction is independent of the gate; apply the freshest
                    # source-time voice decision after GPU work, near display.
                    articulated=mouth.render(head,coeff,1.) if audio_slot>=0 else head['frame']
                    with self.lock:
                        voice_weight=self.activity.weight(audio_slot) if self.activity else 0.
                        owner_weight=self.owner.weight(audio_slot) if self.owner_only and self.owner else (0. if self.owner_only else 1.)
                        weight=voice_weight*owner_weight;self.speech_weight=weight
                    # Keep the tiny mouth network warm on idle frames too. Its
                    # prediction is discarded in silence; VAD controls display.
                    frame=articulated if weight>=1 else cv2.addWeighted(articulated,weight,head['frame'],1-weight,0) if weight>0 else head['frame'].copy()
                    mouth_ended=time.monotonic()
                    frame=self.head.compose_body(frame,head['body'])
                    if self.fixture_dir and 0<=audio_slot<4095:
                        # Machine-readable frame ID on test frames only. Verify
                        # the camera output clock without a lip-sync classifier.
                        left=0
                        for k,bit in enumerate((1,0,1,0,*((audio_slot+1>>j)&1 for j in range(12)))):
                            frame[10:20,left+10+k*10:left+20+k*10]=245 if bit else 10
                    body_ended=time.monotonic()
                    ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,94]);assert ok
                    computed=time.monotonic()
                    self.stop.wait(max(0,target-computed))
                    if self.stop.is_set():break
                    if camera is not None:
                        camera_frame[:,280:1000]=frame
                        camera.send(camera_frame)
                    ended=time.monotonic()
                    with self.lock:
                        self.frames+=1;self.render_ms.append((computed-started)*1000)
                        if computed-started>.04:self.slow_frames.append({'slot':slot,'audio_frame':audio_slot,'mouth_ms':(mouth_ended-started)*1000,'body_ms':(body_ended-mouth_ended)*1000,'jpeg_ms':(computed-body_ended)*1000,'display_late_ms':max(0,(ended-target)*1000)})
                        self.latest={'jpeg':encoded.tobytes(),'sequence':slot,'time':ended,'published_ms':round(time.time()*1000),'target':target,'audio_frame':audio_slot,'stream_id':self.stream_id,'compute_ms':(computed-started)*1000,'speech_weight':weight,'owner_weight':owner_weight,'coeff':np.asarray(coeff).tolist()}
                    self.loop.call_soon_threadsafe(self.notify)
                    slot+=1
                    if ended>self.origin+slot/FPS+.04:
                        skip=math.floor((ended-self.origin)*FPS)-slot
                        if skip>0:self.render_late+=skip;slot+=skip
        except Exception as e:self.fail(e)
        finally:
            if camera is not None:camera.close()
            if mouth:mouth.close()

    def new_owner(self):
        if self.owner_reference is None:
            return None
        return OwnerVoiceGate(self.speaker_model,self.owner_reference,
                              lookahead_ms=self.owner_lookahead_ms,period_ms=80)

    async def microphone(self,request):
        if request.headers.get('Origin')!=f'http://{request.host}':raise web.HTTPForbidden()
        if self.fixture_dir and request.query.get('fixture')!='wav':raise web.HTTPForbidden(text='Fixture server accepts recorded tests only')
        if self.mic is not None:raise web.HTTPConflict(text='Микрофон уже подключён в другой вкладке.')
        ws=web.WebSocketResponse(heartbeat=15,max_msg_size=128*1024)
        await ws.prepare(request)
        self.mic=ws;self.stream_id=secrets.token_hex(6)
        self.audio_received_frames=0;self.audio_epoch=None;self.coeffs.clear();self.capture_clock=None;self.stream_resets=0
        self.activity=SpeechActivity(self.activity_model)
        trace=[];trace_starts=[]
        owner=self.new_owner();self.owner=owner;self.owner_only=owner is not None
        await ws.send_json({'type':'ready','stream_id':self.stream_id,'delay_ms':round(self.delay*1000),'server_monotonic':time.monotonic()})
        try:
            async for msg in ws:
                if msg.type==WSMsgType.BINARY:
                    samples=np.frombuffer(msg.data,dtype='<f4')
                    if self.fixture_dir and len(trace)<3000:trace.append(samples.copy())
                    with self.lock:self.activity.push(samples)
                    # Speaker inference must not hold the render/head lock or
                    # block aiohttp. This stream is processed in packet order.
                    if self.owner_only and owner is not None:await asyncio.to_thread(owner.push,samples)
                    continue
                if msg.type!=WSMsgType.TEXT:continue
                data=json.loads(msg.data)
                if data.get('type')=='configure':
                    if type(data.get('owner_only')) is not bool:raise ValueError('Invalid owner filter setting')
                    self.owner_only=data['owner_only'] and self.owner_reference is not None
                elif data.get('type')=='start':
                    captured=data.get('captured_at_ms')
                    if self.fixture_dir:trace_starts.append(captured)
                    age=0. if captured is None else (time.time()-float(captured)/1000)
                    if not math.isfinite(age) or abs(age)>2:raise ValueError('Invalid capture timestamp')
                    with self.lock:
                        if self.audio_epoch is not None:self.stream_resets+=1
                        self.audio_epoch=time.monotonic()-max(0.,age)-.04
                        self.coeffs.clear();self.audio_received_frames=0
                        self.activity=SpeechActivity(self.activity_model)
                        owner=self.new_owner();self.owner=owner
                    await ws.send_json({'type':'started','server_epoch':self.audio_epoch})
                elif data.get('type')=='capture_clock':
                    self.capture_clock={k:data.get(k) for k in ['frame','client_age_ms','client_dropped','client_resets','sample_rate','compute_ms']}
                    self.capture_clock['server_age_ms']=time.time()*1000-float(data['captured_at_ms'])
                elif data.get('type')=='coeff':
                    first=int(data['first']);rows=np.array(data['rows'],dtype=np.float32)
                    if rows.ndim!=2 or rows.shape[1]!=12 or len(rows)>12 or not np.isfinite(rows).all() or np.abs(rows).max()>1000:raise ValueError('Invalid articulation packet')
                    with self.lock:
                        for i,row in enumerate(rows):self.coeffs[first+i]=row
                        self.audio_received_frames=max(self.audio_received_frames,int(data['available']))
                        self.last_coefficient_input=time.monotonic()
                    await ws.send_json({'type':'ack','first':first,'server_received':time.monotonic()})
        except Exception as e:
            logging.exception('Microphone session failed');await ws.close(code=1011,message=str(e).encode()[:100])
        finally:
            if self.fixture_dir and trace:
                import wave
                self.fixture_dir.mkdir(parents=True,exist_ok=True)
                with wave.open(str(self.fixture_dir/'worklet-source.wav'),'wb') as wav:
                    wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes((np.clip(np.concatenate(trace),-1,.999969)*32768).astype('<i2').tobytes())
                (self.fixture_dir/'worklet-clock.json').write_text(json.dumps(trace_starts),encoding='utf8')
            if self.mic is ws:
                with self.lock:self.mic=None;self.audio_epoch=None;self.coeffs.clear();self.stream_id=None;self.activity=None;self.owner=None
        return ws


async def serve(args):
    loop=asyncio.get_running_loop();feed=Feed(loop,args.seed,args.profile,args.video_delay_ms,args.cable_delay_ms,args.fixture_dir,args.direct_camera)
    @web.middleware
    async def local(request,handler):
        if request.host!=f'127.0.0.1:{args.port}':raise web.HTTPForbidden()
        response=await handler(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response
    app=web.Application(middlewares=[local])
    async def page(request):return web.FileResponse(ROOT/'avatar_identity_live.html')
    async def asset(request):
        names={'avatar_identity_live.js':ROOT/'avatar_identity_live.js','worklet.js':ROOT/'live_avatar_audio.js','engine.js':ROOT/'avatar_identity_audio.js','speech-worker.js':ROOT/'avatar_identity_worker.js','feed.js':ROOT/'avatar_identity_feed.js'}
        name=request.match_info['name']
        if name not in names:raise web.HTTPNotFound()
        return web.FileResponse(names[name])
    async def static(request):
        name=request.match_info['name']
        allow={'js/DHLiveMini.js','js/MiniMateLoader.js','DHLiveMini.wasm'}
        if name not in allow:raise web.HTTPNotFound()
        return web.FileResponse(STATIC/name)
    async def engine(request):
        return web.Response(text='''<!doctype html><meta charset="utf-8"><div id="loadingSpinner"></div><div id="screen"></div><div id="screen2"></div><script src="/model/js/DHLiveMini.js"></script><script src="/model/js/MiniMateLoader.js"></script><script src="/static/engine.js"></script>''',content_type='text/html')
    async def wasm(request):return web.FileResponse(STATIC/'DHLiveMini.wasm')
    async def health(request):return web.json_response(feed.health())
    async def frame(request):
        if not feed.latest:raise web.HTTPServiceUnavailable()
        return web.Response(body=feed.latest['jpeg'],content_type='image/jpeg')
    async def video(request):
        response=web.StreamResponse(headers={'Content-Type':'multipart/x-mixed-replace; boundary=avatar-frame'})
        await response.prepare(request);last=-1
        try:
            while not feed.stop.is_set():
                event=feed.updated;packet=feed.latest
                if packet and packet['sequence']!=last:
                    last=packet['sequence'];jpg=packet['jpeg']
                    headers=f"--avatar-frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\nX-Avatar-Frame: {last}\r\nX-Avatar-Monotonic: {packet['time']:.6f}\r\nX-Avatar-Published-Ms: {packet['published_ms']}\r\nX-Audio-Frame: {packet['audio_frame']}\r\nX-Speech-Weight: {packet['speech_weight']:.4f}\r\nX-Owner-Weight: {packet['owner_weight']:.4f}\r\n\r\n"
                    await response.write(headers.encode()+jpg+b'\r\n')
                await asyncio.wait_for(event.wait(),10)
        except (ConnectionError,asyncio.TimeoutError,asyncio.CancelledError):pass
        return response
    async def obs(request):
        return web.Response(text='''<!doctype html><meta charset="utf-8"><style>html,body{margin:0;width:100%;height:100%;background:#000;overflow:hidden;display:grid;place-items:center}canvas{width:auto;height:auto;max-width:100vw;max-height:100vh;object-fit:contain}</style><canvas width="720" height="720"></canvas><script src="/static/feed.js?native-size=1"></script>''',content_type='text/html')
    async def shutdown(request):
        if request.headers.get('Origin')!=f'http://{request.host}':raise web.HTTPForbidden()
        feed.shutdown.set();return web.json_response({'stopping':True})
    app.add_routes([web.get('/',page),web.get('/static/{name}',asset),web.get('/model/{name:.*}',static),web.get('/engine',engine),web.get('/api/health',health),web.get('/frame',frame),web.get('/video',video),web.get('/feed',obs),web.get('/microphone',feed.microphone),web.post('/api/shutdown',shutdown)])
    app.router.add_get('/DHLiveMini.wasm',wasm)
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1);sock.bind(('127.0.0.1',args.port));sock.listen(128);sock.setblocking(False)
    runner=web.AppRunner(app);await runner.setup();site=web.SockSite(runner,sock);await site.start()
    try:
        await feed.initialize()
        logging.info('Identity live ready at http://127.0.0.1:%s PID=%s',args.port,os.getpid())
        await feed.shutdown.wait()
    finally:
        feed.stop.set()
        if feed.mic:await feed.mic.close()
        await runner.cleanup()
        for name in ['head_thread','mouth_thread']:
            if hasattr(feed,name):await asyncio.to_thread(getattr(feed,name).join,5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8776);parser.add_argument('--seed',type=int,default=secrets.randbits(32));parser.add_argument('--log',type=Path)
    parser.add_argument('--video-delay-ms',type=int,choices=(360,380,400),default=360)
    parser.add_argument('--cable-delay-ms',type=int,default=280)
    parser.add_argument('--fixture-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--direct-camera',action='store_true',help='Publish directly to the installed OBS virtual-camera driver')
    parser.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args()
    args.profile=args.profile.resolve()
    if not 250<=args.cable_delay_ms<=650:parser.error('Cable delay must be 250–650 ms')
    if args.fixture_dir is not None:args.fixture_dir=args.fixture_dir.resolve()
    logging.basicConfig(level=logging.INFO,filename=args.log,encoding='utf8' if args.log else None,format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(serve(args))
