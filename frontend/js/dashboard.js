const totalsEl = document.getElementById('totals');
const alertsEl = document.getElementById('recentAlerts');
let chartInstance=null;

// Placeholder: In a full system this would fetch aggregated stats from backend.
// For now simulate by polling a health endpoint and building dummy data.
async function fetchData(){
  const API_BASE = (window.API_BASE || 'http://localhost:8010');
  try {await fetch(API_BASE + '/health');} catch { /* ignore */ }
  // Dummy data; integrate real endpoint later.
  const threatCounts = { 'Normal': 45, 'Violence': 8, 'Weaponized': 5 };
  const recent = [
    { t: Date.now()-40000, class: 'Violence', conf: 0.91 },
    { t: Date.now()-30000, class: 'Weaponized', conf: 0.87 },
    { t: Date.now()-20000, class: 'Normal', conf: 0.95 },
    { t: Date.now()-10000, class: 'Violence', conf: 0.89 }
  ];
  render(threatCounts, recent);
}

function render(counts, recent){
  totalsEl.innerHTML = Object.entries(counts).map(([k,v])=>`<li><span class='font-semibold'>${k}</span>: ${v}</li>`).join('');
  alertsEl.innerHTML = recent.map(r=>`<li>${new Date(r.t).toLocaleTimeString()} - ${r.class} (${(r.conf*100).toFixed(1)}%)</li>`).join('');
  const data = {labels:Object.keys(counts), datasets:[{label:'Threats', data:Object.values(counts), backgroundColor:['#0d9488','#0369a1','#dc2626','#9333ea']}]};
  if(chartInstance){chartInstance.destroy();}
  chartInstance = new Chart(document.getElementById('chart'), {type:'bar', data});
}

fetchData();
setInterval(fetchData, 15000);
