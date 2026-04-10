const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const loader = document.getElementById('loader');
const summary = document.getElementById('summary');
const summaryText = document.getElementById('summaryText');
const downloadLink = document.getElementById('downloadLink');
let selectedFile = null;

function human(obj){return JSON.stringify(obj,null,2);} // quick pretty

dropzone.addEventListener('click',()=>fileInput.click());

dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('bg-teal-50');});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('bg-teal-50'));
dropzone.addEventListener('drop',e=>{
  e.preventDefault();
  dropzone.classList.remove('bg-teal-50');
  if(e.dataTransfer.files.length){
    selectedFile = e.dataTransfer.files[0];
    dropzone.textContent = selectedFile.name;
    processBtn.disabled = false;
  }
});
fileInput.addEventListener('change',e=>{
  if(fileInput.files.length){
    selectedFile = fileInput.files[0];
    dropzone.textContent = selectedFile.name;
    processBtn.disabled = false;
  }
});

processBtn.addEventListener('click', async ()=>{
  if(!selectedFile) return;
  loader.classList.remove('hidden');
  processBtn.disabled = true;
  const form = new FormData();
  form.append('file', selectedFile);
  try {
    const API_BASE = (window.API_BASE || 'http://localhost:8001');
    console.log('Uploading to:', `${API_BASE}/upload-video`);
    const res = await fetch(`${API_BASE}/upload-video`,{method:'POST', body: form});
    console.log('Response status:', res.status);
    if(!res.ok) {
      const errorText = await res.text();
      console.error('Error response:', errorText);
      throw new Error(`Upload failed: ${res.status} ${res.statusText} - ${errorText}`);
    }
    const json = await res.json();
    console.log('Success:', json);
    summary.classList.remove('hidden');
    summaryText.textContent = human(json.summary) + '\nDetections sample:\n' + human(json.detections.slice(0,10));
    downloadLink.href = API_BASE + json.processed_video;
    downloadLink.textContent = 'Download processed video';
  } catch(err){
    console.error('Upload error:', err);
    alert('Error: ' + err.message + '\n\nCheck browser console (F12) for details.');
  } finally {
    loader.classList.add('hidden');
    processBtn.disabled = false;
  }
});
