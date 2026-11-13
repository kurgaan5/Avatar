// Dedicated same-origin iframe. Selected DH WASM, bounded audio context only.
window.newVideoTask=async()=>{window.bsPtr=Module._malloc(48);window.engineReady=true;};
window.analyzeAudio=(samples)=>{
  if(!window.engineReady)throw new Error('Модель речи ещё загружается');
  const bytes=new Uint8Array(44+samples.length*2),v=new DataView(bytes.buffer);
  const tag=(s,k)=>[...s].forEach((c,j)=>bytes[k+j]=c.charCodeAt(0));
  tag('RIFF',0);v.setUint32(4,36+samples.length*2,true);tag('WAVEfmt ',8);v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);tag('data',36);v.setUint32(40,samples.length*2,true);
  for(let i=0;i<samples.length;i++)v.setInt16(44+i*2,Math.round(Math.max(-1,Math.min(1,samples[i]))*32767),true);
  const realNow=performance.now.bind(performance),dateNow=Date.now.bind(Date);let clock=0;
  const start=realNow();Object.defineProperty(performance,'now',{value:()=>clock,configurable:true});Date.now=()=>clock;
  try{
    Module._clearAudio();const ptr=Module._malloc(bytes.length);
    try{Module.HEAPU8.set(bytes,ptr);Module._setAudioBuffer(ptr,bytes.length);}finally{Module._free(ptr);}
    const rows=[];
    // 8 s history + 120 ms emitted block + 200 ms explicitly received lookahead.
    for(let i=0;i<203;i++){clock=i*40;Module._updateBlendShape(bsPtr,48);if(i>=200)rows.push(Array.from(new Float32Array(Module.HEAPU8.buffer,bsPtr,12)));}
    return {rows,computeMs:realNow()-start};
  }finally{Object.defineProperty(performance,'now',{value:realNow,configurable:true});Date.now=dateNow;}
};
