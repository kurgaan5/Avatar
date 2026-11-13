const $=s=>document.querySelector(s);let active=null,health=null,lastError=null;
$('#url').textContent=location.origin+'/feed';
const state=s=>$('#state').textContent=s;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const engine=()=>$('#engine').contentWindow;
$('#cable').checked=localStorage.getItem('avatarCable')==='true';
$('#cable').onchange=()=>localStorage.setItem('avatarCable',String($('#cable').checked));
$('#owner').checked=localStorage.getItem('avatarOwnerOnly')!=='false';
$('#owner').onchange=()=>localStorage.setItem('avatarOwnerOnly',String($('#owner').checked));
async function stop(){
  if(active){active.closed=true;active.ws?.close();active.worker?.postMessage({type:'stop'});active.stream?.getTracks().forEach(t=>t.stop());await active.context?.close().catch(()=>{});active.worker?.terminate();active=null;}
  $('#stop').disabled=true;$('#input').disabled=false;$('#cable').disabled=false;$('#owner').disabled=false;$('#mic').textContent='Включить микрофон';$('#state').classList.remove('recording');state('Микрофон выключен. Аватар продолжает спокойно двигаться.');
}
async function connectSession(){
  if(active)await stop();
  const session={closed:false,packets:new Map(),available:0,next:0,feeds:[],acks:[]};active=session;
  session.ws=new WebSocket(location.origin.replace('http','ws')+'/microphone');
  const ready=new Promise((resolve,reject)=>{session.ws.onopen=resolve;session.ws.onerror=()=>reject(new Error('Не удалось подключиться к аватару'));});
  session.ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='ack')session.acks.push(msg);if(msg.type==='started')session.server_epoch=msg.server_epoch;};
  session.ws.onclose=()=>{if(!session.closed&&active===session){lastError='Соединение прервано. Включите микрофон снова.';void stop().then(()=>state(lastError));}};
  await ready;
  session.ws.send(JSON.stringify({type:'configure',owner_only:$('#owner').checked}));
  session.push=packet=>{
    if(session.closed)return;
    if(!session.available)session.ws.send(JSON.stringify({type:'start'}));
    session.ws.send(packet.buffer.slice(packet.byteOffset,packet.byteOffset+packet.byteLength));
    session.packets.set(session.available,packet);session.available++;
    while(session.available>=session.next+8){
      const pcm=new Float32Array(208*640);
      for(let j=session.next-200;j<session.next+8;j++){
        const p=session.packets.get(j);if(p)pcm.set(p,(j-session.next+200)*640);
      }
      const computed=engine().analyzeAudio(pcm);
      session.ws.send(JSON.stringify({type:'coeff',first:session.next,rows:computed.rows,available:session.available}));
      session.feeds.push({first:session.next,available:session.available,compute_ms:computed.computeMs,rows:computed.rows});
      session.next+=3;
      for(const key of session.packets.keys())if(key<session.next-200)session.packets.delete(key);
      if(session.feeds.length>1500)session.feeds.splice(0,500);
      if(session.acks.length>1500)session.acks.splice(0,500);
    }
  };
  return session;
}
async function connectWorkerSession(){
  if(active)await stop();
  const session={closed:false,worker:new Worker('/static/speech-worker.js')};active=session;
  await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(Error('Модель речи не запустилась')),20000);
    session.worker.onmessage=e=>{
      if(e.data.type==='ready')session.worker.postMessage({type:'connect',url:location.origin.replace('http','ws')+'/microphone',ownerOnly:$('#owner').checked});
      if(e.data.type==='connected'){clearTimeout(timeout);resolve();}
      if(e.data.error){clearTimeout(timeout);reject(Error(e.data.error));if(session.running&&!session.closed){lastError=e.data.error;void stop().then(()=>state(lastError));}}
    };
    session.worker.onerror=e=>{clearTimeout(timeout);reject(Error(e.message));};
  });
  return session;
}
$('#mic').onclick=async()=>{
  lastError=null;
  $('#mic').disabled=true;
  $('#mic').textContent='Подключаю микрофон…';state('Подключаю микрофон. Если браузер запросит доступ, разрешите его.');
  try{
    const session=await connectWorkerSession();
    let wanted=$('#input').value;
    const constraints=id=>({audio:{deviceId:id==='default'?undefined:{exact:id},echoCancellation:!$('#cable').checked,noiseSuppression:true,autoGainControl:false},video:false});
    session.stream=await navigator.mediaDevices.getUserMedia(constraints(wanted));
    if(/CABLE|Stereo Mix/i.test(session.stream.getAudioTracks()[0].label)){
      session.stream.getTracks().forEach(t=>t.stop());
      const available=await navigator.mediaDevices.enumerateDevices();
      const physical=available.find(d=>d.kind==='audioinput'&&d.deviceId!=='default'&&d.deviceId!=='communications'&&!/CABLE|Stereo Mix/i.test(d.label));
      if(!physical)throw new Error('Выберите физический микрофон, а не CABLE Output.');
      wanted=physical.deviceId;session.stream=await navigator.mediaDevices.getUserMedia(constraints(wanted));
    }
    session.context=new AudioContext({sampleRate:16000,latencyHint:'interactive'});
    if(session.context.sampleRate!==16000)throw Error('Браузер не поддерживает нужную частоту микрофона');
    await session.context.audioWorklet.addModule('/static/worklet.js');
    const devices=await navigator.mediaDevices.enumerateDevices();
    const selected=wanted;$('#input').replaceChildren(new Option('Системный микрофон','default'));
    devices.filter(d=>d.kind==='audioinput'&&!/CABLE|Stereo Mix/i.test(d.label)).forEach(d=>$('#input').add(new Option(d.label||'Микрофон',d.deviceId)));
    $('#input').value=selected;
    if($('#cable').checked){
      const output=devices.find(d=>d.kind==='audiooutput'&&/CABLE Input/i.test(d.label));
      if(!output||!session.context.setSinkId)throw new Error('CABLE Input пока недоступен. Установите VB-CABLE для синхронного голоса.');
      await session.context.setSinkId(output.deviceId);
    }
    const source=session.context.createMediaStreamSource(session.stream);
    const worklet=new AudioWorkletNode(session.context,'avatar-microphone');
    const channel=new MessageChannel();
    session.worker.postMessage({type:'stream_port',port:channel.port1},[channel.port1]);
    worklet.port.postMessage({bridge:channel.port2},[channel.port2]);source.connect(worklet);
    const silent=session.context.createGain();silent.gain.value=0;worklet.connect(silent).connect(session.context.destination);
    if($('#cable').checked){const delay=session.context.createDelay(2);delay.delayTime.value=(health.cable_audio_buffer_ms??health.audio_video_buffer_ms)/1000;source.connect(delay).connect(session.context.destination);}
    await session.context.resume();session.running=true;$('#stop').disabled=false;$('#input').disabled=true;$('#cable').disabled=true;$('#owner').disabled=true;$('#mic').textContent='Микрофон включён';$('#state').classList.add('recording');
    state($('#cable').checked?'Микрофон включён. Голос направлен в CABLE Input.':'Микрофон включён. Передача голоса в CABLE Input выключена.');
  }catch(e){await stop();lastError=e.message;state(lastError);}
};
$('#stop').onclick=stop;
window.addEventListener('pagehide',()=>{active?.stream?.getTracks().forEach(t=>t.stop());active?.ws?.close();active?.worker?.postMessage({type:'stop'});active?.worker?.terminate();});
setInterval(async()=>{try{
  health=await(await fetch('/api/health')).json();
  $('#ownerRow').hidden=!health.owner_available;
  if(!health.owner_available)$('#owner').checked=false;
  $('#obsStep').hidden=$('#obsSource').hidden=health.camera_transport==='direct-obs-driver';
  $('#stats').textContent=JSON.stringify({ready:health.ready,frames:health.frames,delay_ms:health.audio_video_buffer_ms,head_late:health.head_late,render_late:health.render_late,coefficient_late:health.coefficient_late,speech_probability:health.speech_probability,capture_clock:health.capture_clock,capture_frontier_lag_ms:health.capture_frontier_lag_ms},null,2);
  $('#mic').disabled=!!active||!health.ready||!engine().engineReady;
  $('#level').value=active?.running?Math.min(1,Math.sqrt(health.microphone_rms||0)*3):0;
  $('#activity').textContent=active?.running?(health.speech_probability<.35?'Тишина — рот отдыхает':health.owner_filter&&health.speech_weight<=0?'Голос не подтверждён — рот закрыт':'Слышу речь'):'Микрофон выключен';
  if(!active&&health.ready&&engine().engineReady)state(lastError||'Аватар готов. Микрофон выключен.');
  if(active?.running&&health.capture_frontier_lag_ms>250)state('Обработка речи задерживается. Синхронизация сейчас нарушена.');
  if(health.error)state(health.error);
}catch{if(!active)state('Ожидание локального сервера…');}},250);
// Deterministic recorded-input test API. It never opens a physical device.
window.avatarRecordedTest=async (samples,repeats=1)=>{
  const session=await connectSession();session.closed=false;
  const start=performance.now();
  const count=samples.length/640*repeats;
  for(let i=0;i<count+20;i++){
    await sleep(Math.max(0,start+(i+1)*40-performance.now()));
    const offset=(i%(samples.length/640))*640;
    session.push(i<count?new Float32Array(samples.slice(offset,offset+640)):new Float32Array(640));
  }
  const result={feeds:session.feeds,acks:session.acks,server_epoch:session.server_epoch,elapsed_ms:performance.now()-start};await stop();return result;
};
