// Isolated DH WASM inference: no Qt UI and no main-thread model computation.
importScripts('/model/js/DHLiveMini.js');
let engine,bsPtr,session=null;
function analyze(samples){
  const bytes=new Uint8Array(44+samples.length*2),v=new DataView(bytes.buffer);
  const tag=(s,k)=>[...s].forEach((c,j)=>bytes[k+j]=c.charCodeAt(0));
  tag('RIFF',0);v.setUint32(4,36+samples.length*2,true);tag('WAVEfmt ',8);v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);tag('data',36);v.setUint32(40,samples.length*2,true);
  for(let i=0;i<samples.length;i++)v.setInt16(44+i*2,Math.round(Math.max(-1,Math.min(1,samples[i]))*32767),true);
  const realNow=performance.now.bind(performance),dateNow=Date.now.bind(Date);let clock=0;
  const started=realNow();Object.defineProperty(performance,'now',{value:()=>clock,configurable:true});Date.now=()=>clock;
  try{
    engine._clearAudio();const ptr=engine._malloc(bytes.length);
    try{engine.HEAPU8.set(bytes,ptr);engine._setAudioBuffer(ptr,bytes.length);}finally{engine._free(ptr);}
    const rows=[];
    for(let i=0;i<203;i++){clock=i*40;engine._updateBlendShape(bsPtr,48);if(i>=200)rows.push(Array.from(new Float32Array(engine.HEAPU8.buffer,bsPtr,12)));}
    return {rows,computeMs:realNow()-started};
  }finally{Object.defineProperty(performance,'now',{value:realNow,configurable:true});Date.now=dateNow;}
}
function startClock(s,capturedAt){
  s.packets.clear();s.available=0;s.next=0;s.epoch=capturedAt-40;
  s.ws.send(JSON.stringify({type:'start',captured_at_ms:capturedAt}));
}
function packet(data){
  const s=session;if(!s||s.closed||s.ws.readyState!==WebSocket.OPEN)return;
  const samples=new Float32Array(data.samples),now=Date.now(),age=now-data.capturedAt;
  if(data.sampleRate!==16000||samples.length!==640)throw Error('Unexpected microphone sample rate');
  // Never spend time rendering a backlog of obsolete speech. A discontinuity
  // resets source-time state and is exposed in diagnostics, not counted as PASS.
  if(age>160){s.dropped++;s.needsReset=true;return;}
  const drift=s.epoch==null?0:data.capturedAt-(s.epoch+(s.available+1)*40);
  if(s.epoch==null||s.needsReset||Math.abs(drift)>120){
    if(s.epoch!=null)s.resets++;startClock(s,data.capturedAt);s.needsReset=false;
  }
  if(s.available%5===0)s.ws.send(JSON.stringify({type:'capture_clock',captured_at_ms:data.capturedAt,frame:s.available,client_age_ms:age,client_dropped:s.dropped,client_resets:s.resets,sample_rate:data.sampleRate,compute_ms:s.computeMs||0}));
  s.ws.send(samples.buffer);s.packets.set(s.available,samples);s.available++;
  while(s.available>=s.next+8){
    const pcm=new Float32Array(208*640);
    for(let j=s.next-200;j<s.next+8;j++){const p=s.packets.get(j);if(p)pcm.set(p,(j-s.next+200)*640);}
    const computed=analyze(pcm);s.computeMs=computed.computeMs;
    s.ws.send(JSON.stringify({type:'coeff',first:s.next,rows:computed.rows,available:s.available}));
    s.next+=3;
    for(const key of s.packets.keys())if(key<s.next-200)s.packets.delete(key);
  }
}
self.onmessage=e=>{
  const data=e.data;
  try{
    if(data.type==='connect'){
      const s={closed:false,packets:new Map(),available:0,next:0,epoch:null,dropped:0,resets:0};session=s;
      s.ws=new WebSocket(data.url);
      s.ws.onopen=()=>{s.ws.send(JSON.stringify({type:'configure',owner_only:data.ownerOnly}));self.postMessage({type:'connected'});};
      s.ws.onerror=()=>self.postMessage({error:'Не удалось подключить обработку речи'});
      s.ws.onclose=()=>{if(!s.closed)self.postMessage({error:'Соединение прервано. Включите микрофон снова.'});};
    }else if(data.type==='stream_port'){
      data.port.onmessage=event=>{try{packet(event.data);}catch(error){self.postMessage({error:String(error)});}};data.port.start();
    }else if(data.type==='stop'){
      if(session){session.closed=true;session.ws.close();session=null;}
    }else self.postMessage({id:data.id,...analyze(new Float32Array(data.samples))});
  }catch(error){self.postMessage({id:data.id,error:String(error)});}
};
createQtAppInstance({noInitialRun:true,locateFile:p=>'/DHLiveMini.wasm'}).then(m=>{
  engine=m;bsPtr=engine._malloc(48);self.postMessage({type:'ready'});
}).catch(error=>self.postMessage({error:String(error)}));
