const frameImg = document.getElementById('rtspFrame');
const overlay = document.getElementById('rtspOverlay');
const logEl = document.getElementById('rtspLog');
const ctx = overlay.getContext('2d');
let ws=null;

function draw(dets){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  dets.forEach(d=>{
    const [x1,y1,x2,y2]=d.bbox;
    ctx.strokeStyle='lime';ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle='lime';ctx.font='12px sans-serif';ctx.fillText(d.class,x1,y1-4);
  });
}

document.getElementById('connectBtn').addEventListener('click',()=>{
  const url = document.getElementById('rtspUrl').value.trim();
  if(!url){alert('Enter RTSP URL');return;}
  if(ws){ws.close();}
  const API_BASE = (window.API_BASE || 'http://localhost:8001');
  ws = new WebSocket(API_BASE.replace('http','ws') + '/ws/rtsp');
  ws.onopen=()=>{ws.send(url);};
  ws.onmessage=(e)=>{
    try{const data=JSON.parse(e.data);if(data.frame){frameImg.src='data:image/jpeg;base64,'+data.frame;draw(data.detections || []);logEl.textContent=e.data+'\n'+logEl.textContent;}}catch(err){/* ignore */}
  };
  ws.onclose=()=>{};
});
