// Paint complete JPEG packets, reconnect after EOF, and retain the last good frame.
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d',{alpha:false});
const wait=ms=>new Promise(r=>setTimeout(r,ms));
window.avatarFeed={frames:0,reconnects:0,lastSequence:null,lastDraw:0,stale:0,lastAgeMs:null};
let pending=null,decoding=false;
async function paint(){
  if(decoding)return;
  decoding=true;
  try{
    while(pending){
      const frame=pending;pending=null;
      if(frame.published&&Date.now()-frame.published>160){window.avatarFeed.stale++;continue;}
      const bitmap=await createImageBitmap(frame.blob);
      const age=frame.published?Date.now()-frame.published:null;
      if(age===null||age<=160){
        if(canvas.width!==bitmap.width||canvas.height!==bitmap.height){
          canvas.width=bitmap.width;canvas.height=bitmap.height;
        }
        ctx.drawImage(bitmap,0,0);window.avatarFeed.frames++;
        window.avatarFeed.lastSequence=frame.sequence;window.avatarFeed.lastDraw=performance.now();window.avatarFeed.lastAgeMs=age;
      }else window.avatarFeed.stale++;
      bitmap.close();
    }
  }catch(e){window.avatarFeed.lastError=String(e);}
  finally{decoding=false;if(pending)void paint();}
}
async function receive(){
  for(;;){
    const controller=new AbortController();let lastByte=performance.now();
    const watchdog=setInterval(()=>{if(performance.now()-lastByte>3500)controller.abort();},1000);
    try{
      const response=await fetch('/video',{signal:controller.signal,cache:'no-store'});
      if(!response.ok)throw Error('Stream unavailable');
      const reader=response.body.getReader();let buffer=new Uint8Array(0),length=null,sequence=null,published=null;
      for(;;){
        const packet=await reader.read();if(packet.done)break;lastByte=performance.now();
        const joined=new Uint8Array(buffer.length+packet.value.length);joined.set(buffer);joined.set(packet.value,buffer.length);buffer=joined;
        if(buffer.length>2000000)throw Error('Invalid frame size');
        for(;;){
          if(length===null){
            let end=-1;for(let i=0;i<buffer.length-3;i++)if(buffer[i]===13&&buffer[i+1]===10&&buffer[i+2]===13&&buffer[i+3]===10){end=i;break;}
            if(end<0)break;
            const header=new TextDecoder().decode(buffer.subarray(0,end));
            const found=header.match(/Content-Length: (\d+)/i);if(!found)throw Error('Missing length');
            length=Number(found[1]);sequence=Number(header.match(/X-Avatar-Frame: (\d+)/i)?.[1]);published=Number(header.match(/X-Avatar-Published-Ms: (\d+)/i)?.[1])||null;buffer=buffer.slice(end+4);
          }
          if(buffer.length<length)break;
          // Do not let JPEG decoding serialize the network reader. If several
          // frames arrived together, only the newest pending image is useful.
          pending={blob:new Blob([buffer.slice(0,length)],{type:'image/jpeg'}),sequence,published};
          buffer=buffer.slice(length);length=null;
        }
        void paint();
      }
    }catch(e){window.avatarFeed.lastError=String(e);}
    finally{clearInterval(watchdog);controller.abort();}
    window.avatarFeed.reconnects++;await wait(500);
  }
}
receive();
