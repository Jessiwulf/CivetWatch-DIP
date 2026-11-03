"""CivetWatch app for zoo staff (runs in the browser, on this computer only).

Home: every observation day + "Process new footage" (pick folders or upload,
see the time estimate, watch the progress). For each day: Overview, Sheet,
Charts, Events & video (watch any event and correct it), Behavior check
(urine / feces / marking / grooming, with video), Spot check (accuracy),
Files. Answers are saved to runs/<day>/review.json and used when the report
is rebuilt (one button) and to train the behavior model (one button).
"""
import json
import os
import shutil
import threading
from datetime import datetime

import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from jinja2 import DictLoader

from . import ethogram as E
from . import jobs, spotcheck
from .config import PROJECT_ROOT, load_manifest, resolve
from .report import fmt_hm

# ------------------------------------------------------------------ templates ---
BASE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}CivetWatch{% endblock %}</title>
<style>
:root{--bg:#f5efdf;--panel:#fdf9ef;--soft:#e4e9d4;--ink:#1d2b22;--ink2:#52574c;--muted:#7a8a6a;--green:#183a2e;
--olive:#6e8452;--orange:#c8693a;--line:#c9d3ae;--warn-bg:#f7e3d4;--ok-bg:#e3ecd3}
@media (prefers-color-scheme:dark){:root{--bg:#111a15;--panel:#18241d;--soft:#1f2e25;--ink:#eef0e6;--ink2:#c3c8b8;
--muted:#93a383;--green:#0c1712;--line:#2f4236;--warn-bg:#3a2519;--ok-bg:#1f3322}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Segoe UI",sans-serif}
a{color:var(--orange)}
header.top{background:var(--green);color:#f5efdf;padding:10px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:20}
header.top .brand{font-weight:800;font-size:18px;color:#f5efdf;text-decoration:none;letter-spacing:.02em}
header.top .brand small{font-weight:400;color:#a7b87a;font-size:12px;margin-left:6px}
header.top select{background:#234a3b;color:#f5efdf;border:1px solid #3f6b56;border-radius:6px;padding:4px 8px;font:inherit}
header.top .sp{flex:1}
header.top a.btn{background:var(--orange);color:#fff;text-decoration:none;padding:6px 12px;border-radius:6px;font-weight:700;font-size:14px}
#jobpill{display:none;background:#234a3b;border:1px solid #3f6b56;border-radius:16px;padding:3px 12px;font-size:13px;color:#f5efdf;text-decoration:none}
#jobpill .bar{display:inline-block;width:70px;height:6px;background:#3f6b56;border-radius:3px;vertical-align:middle;margin-left:6px;overflow:hidden}
#jobpill .bar i{display:block;height:100%;background:#e5b04a}
nav.tabs{background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:2px;padding:0 16px;overflow-x:auto;position:sticky;top:52px;z-index:19}
nav.tabs a{padding:10px 14px;color:var(--ink2);text-decoration:none;font-weight:600;font-size:14px;white-space:nowrap;border-bottom:3px solid transparent}
nav.tabs a.on{color:var(--ink);border-bottom-color:var(--orange)}
nav.tabs a .n{background:var(--orange);color:#fff;border-radius:9px;padding:0 6px;font-size:11px;margin-left:4px}
main{max-width:1280px;margin:0 auto;padding:18px 16px 60px}
h1{font-size:24px;margin:4px 0 12px}h2{font-size:19px;margin:26px 0 10px;padding-bottom:5px;border-bottom:1px solid var(--line)}
.m{color:var(--ink2);font-size:13px}
.panel{background:var(--panel);border-radius:8px;padding:14px 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.tile{background:var(--panel);padding:14px 16px;border-radius:8px}
.tile .v{font-size:30px;font-weight:700;line-height:1.15}.tile .l{color:var(--ink2);font-size:13px}
.tile.o .v{color:var(--orange)}.tile.g .v{color:var(--olive)}
.alert{background:var(--warn-bg);border-left:4px solid var(--orange);padding:8px 12px;border-radius:4px;margin-top:6px}
.okmsg{background:var(--ok-bg);border-left:4px solid var(--olive);padding:8px 12px;border-radius:4px;margin-top:6px}
button,.button{font:inherit;font-size:14px;border-radius:6px;border:1px solid var(--line);background:var(--soft);color:var(--ink);padding:6px 12px;cursor:pointer;text-decoration:none;display:inline-block}
button.main,.button.main{background:var(--orange);border-color:var(--orange);color:#fff;font-weight:700}
button.ok{background:var(--olive);border-color:var(--olive);color:#fff}button.no{background:var(--orange);border-color:var(--orange);color:#fff}
button:disabled{opacity:.5;cursor:default}
input[type=text],input[type=date],select.f{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.steps{display:grid;gap:10px;counter-reset:s}.step{background:var(--panel);border-radius:8px;padding:12px 16px 12px 52px;position:relative}
.step:before{counter-increment:s;content:counter(s);position:absolute;left:14px;top:12px;width:26px;height:26px;border-radius:13px;background:var(--green);color:#f5efdf;text-align:center;font-weight:700;line-height:26px}
.step.done:before{content:"✓";background:var(--olive)}.step b{font-size:15px}.step .act{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.progress{height:14px;background:var(--soft);border-radius:7px;overflow:hidden}.progress i{display:block;height:100%;background:var(--orange);transition:width .6s}
.progress.ind i{width:30%!important;animation:ind 1.4s infinite ease-in-out}@keyframes ind{0%{margin-left:-30%}100%{margin-left:100%}}
.spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--orange);border-radius:50%;animation:sp 1s linear infinite;vertical-align:-2px}@keyframes sp{to{transform:rotate(360deg)}}
pre.log{background:#0e1511;color:#cfe0c4;padding:10px 12px;border-radius:6px;font-size:12px;overflow-x:auto;max-height:260px}
.tablewrap{overflow:auto;background:var(--panel);border-radius:8px;max-height:70vh}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--soft);z-index:1}td.n{text-align:right;font-variant-numeric:tabular-nums}
tr.sel td{background:var(--warn-bg)}tbody tr{cursor:pointer}tbody tr:hover td{background:var(--soft)}
.pill{display:inline-block;padding:0 8px;border-radius:10px;font-size:12px;background:var(--soft)}
.pill.check{background:var(--warn-bg);color:var(--orange);font-weight:700}.pill.done{background:var(--ok-bg);color:var(--olive);font-weight:700}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.chips button.on{background:var(--green);color:#f5efdf;border-color:var(--green)}
.gloss dt{font-weight:700;margin-top:8px}.gloss dd{margin:0;color:var(--ink2)}
.vbox{position:relative;overflow:hidden;background:#000;border-radius:8px;aspect-ratio:16/9}
.vbox video{width:100%;height:100%;display:block;object-fit:contain;transition:transform .3s}
.vbox .none{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:14px}
.vctl{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:6px;font-size:13px}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;overflow:auto;cursor:zoom-in}
#lb.open{display:block}#lb img{display:block;margin:auto;max-width:100vw;max-height:100vh;object-fit:contain}
#lb.big{cursor:zoom-out}#lb.big img{max-width:none;max-height:none;width:250vw}
#lb .x{position:fixed;top:10px;right:16px;color:#fff;background:rgba(0,0,0,.6);padding:6px 12px;border-radius:6px;cursor:pointer}
img.zoomable{cursor:zoom-in}
.tl{display:flex;align-items:center;gap:6px;margin-top:6px}.tl button{padding:4px 9px}#pp{min-width:40px}
.seekw{position:relative;flex:1;height:22px}.seekw .trk,.seekw .band{position:absolute;top:50%;height:8px;transform:translateY(-50%);border-radius:4px}
.seekw .trk{left:0;right:0;background:var(--soft);border:1px solid var(--line)}.seekw .band{background:var(--orange);opacity:.55}
.seekw input{position:absolute;inset:0;width:100%;margin:0;background:transparent;-webkit-appearance:none;appearance:none;cursor:pointer}
.seekw input::-webkit-slider-runnable-track{height:22px;background:transparent}
.seekw input::-webkit-slider-thumb{-webkit-appearance:none;width:6px;height:22px;border-radius:3px;background:var(--ink)}
.seekw input::-moz-range-track{background:transparent}.seekw input::-moz-range-thumb{width:6px;height:22px;border-radius:3px;background:var(--ink);border:0}
.clock{font:600 14px ui-monospace,Consolas,monospace;min-width:72px;text-align:right}
kbd{display:inline-block;min-width:20px;text-align:center;background:var(--soft);border:1px solid var(--line);border-radius:4px;padding:0 5px;font:600 12px system-ui}
#langbtn{background:#234a3b;color:#f5efdf;border:1px solid #3f6b56;border-radius:6px;padding:4px 10px;font-weight:700}
</style>{{ i18n_script|safe }}{% block head %}{% endblock %}</head><body>
<div id="lb" onclick="lbClick(event)"><span class="x" onclick="lbClose(event)">Esc · close ✕</span><img alt=""></div>
<header class="top"><a class="brand" href="{{ url_for('home') }}">CivetWatch<small>Chiang Mai Zoo</small></a>
{% if run %}<select onchange="location.href=this.value" title="Observation day">{% for r in runs %}<option value="{{ url_for(request.endpoint, run=r) }}" {{ 'selected' if r==run }}>{{ r }}</option>{% endfor %}</select>{% endif %}
<span class="sp"></span><button id="langbtn" type="button">ไทย</button><a id="jobpill" href="#"><span class="spin"></span> <span id="jobtxt"></span><span class="bar"><i id="jobbar"></i></span></a>
<a class="btn" href="{{ url_for('new_page') }}">+ Process new footage</a></header>
{% if run %}<nav class="tabs">
{% for ep, name in [('run_page','Overview'),('sheet_page','Observation sheet'),('charts_page','Charts'),('events_page','Events & video'),('check_page','Behavior check'),('spot_page','Spot check'),('files_page','Files')] %}
<a href="{{ url_for(ep, run=run) }}" class="{{ 'on' if request.endpoint==ep or (ep=='spot_page' and request.endpoint=='spot_results') }}">{{ name }}{% if ep=='check_page' and nav_left %}<span class="n">{{ nav_left }}</span>{% endif %}</a>{% endfor %}</nav>{% endif %}
<main>{% block main %}{% endblock %}</main>
<script>
function lbOpen(src){const lb=document.getElementById('lb');lb.querySelector('img').src=src;lb.className='open';}
function lbClose(e){if(e)e.stopPropagation();document.getElementById('lb').className='';}
function lbClick(e){const lb=document.getElementById('lb');if(e.target.tagName!=='IMG'){lbClose();return;}
  const r=e.target.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;lb.classList.toggle('big');
  if(lb.classList.contains('big')){const i=lb.querySelector('img');lb.scrollLeft=fx*i.width-innerWidth/2;lb.scrollTop=fy*i.height-innerHeight/2;}}
document.addEventListener('keydown',e=>{if(e.key==='Escape')lbClose();});
document.addEventListener('click',e=>{const t=e.target.closest('img.zoomable,a.zoomable');if(t){e.preventDefault();lbOpen(t.tagName==='A'?t.href:t.src);}});
function fmtS(s){if(s==null)return'?';s=Math.round(s);if(s<90)return s+' s';if(s<3600)return Math.round(s/60)+' min';
  return Math.floor(s/3600)+' h '+String(Math.round((s%3600)/60)).padStart(2,'0')+' min';}
async function jobPoll(){try{const r=await fetch("{{ url_for('api_job') }}");const j=await r.json();const p=document.getElementById('jobpill');
  if(j&&j.state==='running'){p.style.display='inline-block';p.href=j.url;
    document.getElementById('jobtxt').textContent=j.title+(j.progress!=null?' · '+Math.round(j.progress*100)+'%':'')+(j.eta!=null?' · ~'+fmtS(j.eta)+' left':'');
    document.getElementById('jobbar').style.width=((j.progress||0)*100)+'%';}else p.style.display='none';
  if(window.onJob)window.onJob(j);}catch(e){} setTimeout(jobPoll,3000);}
jobPoll();
async function startJob(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  const j=await r.json();if(!r.ok){alert(j.error||'Could not start');return null;}location.href=j.url;return j;}
</script>{% block script %}{% endblock %}</body></html>"""

HOME = r"""{% extends "base" %}{% block main %}
<h1>Observation days</h1>
{% if job %}<div class="okmsg"><span class="spin"></span> <b>{{ job.title }}</b> is running · <a href="{{ url_for('job_page', job_id=job.id) }}">see progress</a></div>{% endif %}
{% if not days %}<div class="panel">No days yet. Click <b>+ Process new footage</b> to add the first one.</div>{% endif %}
<div class="tiles" style="margin-top:12px">{% for d in days %}
<a class="tile" style="text-decoration:none;color:inherit;position:relative" href="{{ url_for('run_page', run=d.run) }}">
<button class="rm" title="Remove this day" style="position:absolute;top:8px;right:8px;font-size:12px;padding:2px 8px"
 onclick='event.preventDefault();event.stopPropagation();removeDay({{ d.run|tojson }},{{ d.size|tojson }},{{ d.video_size|tojson }})'>Remove</button>
<div class="l">{{ d.weekday }}</div><div class="v">{{ d.day }}</div>
<div class="l">{% if d.ok %}active {{ d.active }} · inactive {{ d.inactive }}<br>{{ d.feeding }} feeding · {{ d.drinking }} drinking visits{% else %}not finished: {{ d.state }}{% endif %}</div>
{% if d.left %}<div style="margin-top:6px"><span class="pill check">{{ d.left }} to check</span></div>{% elif d.ok %}<div style="margin-top:6px"><span class="pill done">all checked</span></div>{% endif %}
</a>{% endfor %}</div>
{% if history %}<h2>Recent jobs</h2><div class="tablewrap"><table><thead><tr><th>Started</th><th>Job</th><th>Result</th><th>Took</th></tr></thead><tbody>
{% for j in history %}<tr onclick="location.href='{{ url_for('job_page', job_id=j.id) }}'"><td>{{ j.started_txt }}</td><td>{{ j.title }}</td><td>{{ j.state }}</td><td>{{ j.took }}</td></tr>{% endfor %}</tbody></table></div>{% endif %}
{% endblock %}{% block script %}<script>
async function removeDay(run,size,vsize){
  let msg=L(`Remove ${run} from CivetWatch?\n\nIts results (${size}: sheet, charts, events, your answers) go to the Recycle Bin - you can restore them from there.`,
    `ลบ ${run} ออกจาก CivetWatch?\n\nผลลัพธ์ (${size}: ตาราง กราฟ เหตุการณ์ คำตอบของคุณ) จะถูกย้ายไปถังรีไซเคิล - กู้คืนได้จากที่นั่น`);
  if(!confirm(msg))return;
  let videos=false;
  if(vsize)videos=confirm(L(`Also remove its prepared videos (${vsize}, data/${run})?\n\nOK = remove them too (frees the space)\nCancel = keep them (needed to process this day again)`,
    `ลบวิดีโอที่เตรียมไว้ด้วยหรือไม่ (${vsize}, data/${run})?\n\nOK = ลบด้วย (ได้พื้นที่คืน)\nCancel = เก็บไว้ (ต้องใช้ถ้าจะประมวลผลวันนี้ใหม่)`));
  const r=await fetch(`{{ url_for('api_remove_day', run='RUN') }}`.replace('RUN',encodeURIComponent(run)),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({videos})});
  const j=await r.json();if(!r.ok){alert(j.error||'Could not remove');return;}location.reload();}
</script>{% endblock %}"""

NEW = r"""{% extends "base" %}{% block main %}
<h1>Process new footage</h1>
<p class="m">CivetWatch reads the 4 cameras' recordings of one day, finds the civet in every frame and fills the observation sheet.
The videos stay where they are; a small 2-frames-per-second copy is made in <code>data/&lt;date&gt;</code>. Zones and settings are copied from the last day (the cameras do not move).</p>
<div class="panel">
<p><b>1 · Day</b><br><input type="date" id="date" value="{{ today }}"> <span class="m">filled in from the file names when you pick a folder</span></p>
<p><b>2 · Where are the videos?</b></p>
<div class="chips" id="mode"><button class="on" data-m="folder">A folder on this computer (recommended)</button><button data-m="upload">Upload video files</button></div>
<div id="m-folder">
<p class="m">Paste one folder: CivetWatch recognises camera sub-folders, or splits a flat folder from file names such as <code>A01_20260405000000.mp4</code> through <code>A04_...</code>.</p>
<p><input type="text" id="source" style="width:min(640px,100%)" placeholder="e.g. D:\CMU_Work\Image processing\ชะมดแผงสันหางดำ" value="{{ last_source }}"></p>
<p class="m" id="autofill" aria-live="polite"></p>
<details><summary class="m">or one folder per camera</summary>{% for i in range(1,5) %}
<p>Camera {{ i }} <input type="text" id="cam{{ i }}" style="width:min(560px,90%)"></p>{% endfor %}</details>
</div>
<div id="m-upload" style="display:none">
<p class="m">The files are copied into <code>data/&lt;date&gt;/upload</code> (needs free space for the full recordings). File names must contain the date and time, like <code>A01_20260405000000.mp4</code>.</p>
<p><b>Select all videos</b><br><input type="file" id="upall" multiple accept="video/*"> <span class="m" id="upallnote"></span></p>
<p class="m">Files named <code>A01_...</code> through <code>A04_...</code> are placed into the matching camera automatically. You can still add files to a camera manually below.</p>
{% for i in range(1,5) %}<p>Camera {{ i }} <input type="file" id="up{{ i }}" multiple accept="video/*"> <span class="m" id="up{{ i }}note"></span></p>{% endfor %}
<div id="uplist" class="panel" style="display:none;margin:12px 0 0"><b>Files ready to upload</b><div id="uplistitems"></div></div>
<div id="upbox" style="display:none"><div class="progress"><i id="upbar" style="width:0"></i></div><p class="m" id="uptxt"></p></div>
</div>
<p><button id="checkbtn" onclick="check()">Check footage and estimate time</button></p>
<div id="est" style="display:none"></div>
</div>
{% endblock %}{% block script %}<script>
let mode='folder';
document.querySelectorAll('#mode button').forEach(b=>b.onclick=()=>{mode=b.dataset.m;
  document.querySelectorAll('#mode button').forEach(x=>x.classList.toggle('on',x===b));
  document.getElementById('m-folder').style.display=mode==='folder'?'':'none';
  document.getElementById('m-upload').style.display=mode==='upload'?'':'none';});
function body(){const b={date:document.getElementById('date').value,source:document.getElementById('source').value.trim(),cams:{}};
  for(let i=1;i<=4;i++){const v=document.getElementById('cam'+i).value.trim();if(v)b.cams['cam'+i]=v;}return b;}
async function scanSource(){const source=document.getElementById('source').value.trim(),note=document.getElementById('autofill');
  if(!source){note.textContent='';return;}
  note.textContent='Reading video file names ...';
  try{const r=await fetch("{{ url_for('api_footage_metadata') }}",{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source})}),m=await r.json();
    if(!r.ok){note.textContent=m.error||'Could not read this folder.';return;}
    if(m.date&&!document.getElementById('date').dataset.touched)document.getElementById('date').value=m.date;
    const cams=Object.entries(m.cameras).map(([camera,info])=>`${camera.replace('cam','Camera ')}: ${info.files} video${info.files===1?'':'s'}`).join(' · ');
    note.textContent=`Detected ${cams}${m.date?' · date '+m.date:''}.`;
  }catch(e){note.textContent='Could not read this folder.';}}
document.getElementById('source').addEventListener('change',scanSource);
document.getElementById('source').addEventListener('blur',scanSource);
let autoUploadFiles={};
function uploadMeta(name){const cam=name.match(/(?:^|[_\-\s])(?:cam(?:era)?|a)\s*0?([1-8])(?=$|[_\-\s.])/i),compact=name.match(/(20\d{2})(\d{2})(\d{2})/),split=name.match(/(20\d{2})[-_](\d{2})[-_](\d{2})/);
  const date=compact||split;return {cam:cam?'cam'+Number(cam[1]):'',date:date?`${date[1]}-${date[2]}-${date[3]}`:''};}
function assignUploads(files){autoUploadFiles={};let unknown=[],dates=[];
  for(const f of files){const meta=uploadMeta(f.name);if(!meta.cam){unknown.push(f.name);continue;}(autoUploadFiles[meta.cam]??=[]).push(f);if(meta.date)dates.push(meta.date);}
  for(let i=1;i<=4;i++){const n=(autoUploadFiles['cam'+i]||[]).length;document.getElementById('up'+i+'note').textContent=n?`${n} file${n===1?'':'s'} added automatically`:'';}
  const counts={};for(const date of dates)counts[date]=(counts[date]||0)+1;const date=Object.keys(counts).sort((a,b)=>counts[b]-counts[a])[0];
  if(date&&!document.getElementById('date').dataset.touched)document.getElementById('date').value=date;
  const added=Object.values(autoUploadFiles).reduce((n,files)=>n+files.length,0),note=document.getElementById('upallnote');
  note.textContent=`${added} file${added===1?'':'s'} assigned automatically${date?' · date '+date:''}${unknown.length?' · '+unknown.length+' file(s) need a camera name':''}.`;renderUploadList();}
document.getElementById('upall').addEventListener('change',e=>assignUploads(e.target.files));
function pendingUploads(){let files=[],seen=new Set();
  const add=(cam,f)=>{const key=`${f.name}:${f.size}:${f.lastModified}`;if(!seen.has(key)){seen.add(key);files.push([cam,f]);}};
  for(const [cam,items] of Object.entries(autoUploadFiles))for(const f of items)add(cam,f);
  for(let i=1;i<=4;i++)for(const f of document.getElementById('up'+i).files)add('cam'+i,f);
  return files;}
function renderUploadList(){const files=pendingUploads(),box=document.getElementById('uplist'),items=document.getElementById('uplistitems');items.replaceChildren();box.style.display=files.length?'':'none';
  const groups={};for(const [cam,file] of files)(groups[cam]??=[]).push(file.name);
  for(const cam of Object.keys(groups).sort()){const title=document.createElement('p'),label=document.createElement('b');title.style.margin='8px 0 2px';label.textContent=cam.replace('cam','Camera ');title.append(label);items.append(title);const list=document.createElement('ul');list.style.margin='0';for(const name of groups[cam].sort()){const item=document.createElement('li');item.textContent=name;list.append(item);}items.append(list);}}
for(let i=1;i<=4;i++)document.getElementById('up'+i).addEventListener('change',renderUploadList);
async function upload(){const date=document.getElementById('date').value,files=pendingUploads();
  if(!files.length){alert('Pick the video files first');return null;}
  const total=files.reduce((a,x)=>a+x[1].size,0);let sent=0;document.getElementById('upbox').style.display='';
  for(const [cam,f] of files){await new Promise((ok,bad)=>{const x=new XMLHttpRequest();
    x.open('PUT',`{{ url_for('api_upload', date='DATE', cam='CAM', name='NAME') }}`.replace('DATE',date).replace('CAM',cam).replace('NAME',encodeURIComponent(f.name)));
    x.upload.onprogress=e=>{document.getElementById('upbar').style.width=((sent+e.loaded)/total*100)+'%';
      document.getElementById('uptxt').textContent=`Uploading ${f.name} · ${Math.round((sent+e.loaded)/1e9*10)/10} of ${Math.round(total/1e9*10)/10} GB`;};
    x.onload=()=>x.status<300?ok():bad(x.responseText);x.onerror=bad;x.send(f);});sent+=f.size;}
  document.getElementById('uptxt').textContent='Upload finished';
  const r=await fetch(`{{ url_for('api_uploaded', date='DATE') }}`.replace('DATE',date));return await r.json();}
async function check(){const btn=document.getElementById('checkbtn');btn.disabled=true;btn.innerHTML='<span class="spin"></span> Checking the videos ...';
  let b=body();
  if(mode==='upload'){const up=await upload();if(!up){btn.disabled=false;btn.textContent='Check footage and estimate time';return;}b={date:b.date,cams:up.cams,source:''};}
  const r=await fetch("{{ url_for('api_estimate') }}",{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const e=await r.json();btn.disabled=false;btn.textContent='Check footage and estimate time';const box=document.getElementById('est');box.style.display='';
  if(!r.ok){box.innerHTML=`<div class="alert">${e.error}</div>`;return;}
  if(e.date&&!document.getElementById('date').dataset.touched)document.getElementById('date').value=e.date;
  let rows=Object.entries(e.cameras).map(([c,v])=>`<tr><td>${c}</td><td>${v.folder}</td><td class="n">${v.files}</td><td class="n">${v.hours}</td></tr>`).join('');
  box.innerHTML=`<h2>Found</h2><div class="tablewrap"><table><thead><tr><th>Camera</th><th>Folder</th><th>Videos</th><th>Hours</th></tr></thead><tbody>${rows}</tbody></table></div>
  <h2>Estimated time</h2><div class="tiles"><div class="tile o"><div class="v">~${fmtS(e.total_s)}</div><div class="l">in total, in the background</div></div>
  <div class="tile"><div class="v">${fmtS(e.prepare_s)}</div><div class="l">prepare videos</div></div><div class="tile"><div class="v">${fmtS(e.analyse_s)}</div><div class="l">find the civet + behaviors</div></div>
  <div class="tile"><div class="v">${e.disk_gb} GB</div><div class="l">disk needed · ${e.free_gb} GB free</div></div></div>
  ${e.warnings.map(w=>`<div class="alert">${w}</div>`).join('')}
  <p class="m">The laptop stays usable: the job runs at low priority and pauses when the GPU is hot. You can close this page; you can stop and later continue - finished hours are kept.</p>
  <p><button class="main" ${e.can_start?'':'disabled'} onclick='startJob("{{ url_for('api_start_process') }}",${JSON.stringify(b)}).then(()=>0)'>Start processing</button></p>`;}
document.getElementById('date').oninput=e=>e.target.dataset.touched=1;
</script>{% endblock %}"""

JOB = r"""{% extends "base" %}{% block main %}
<h1 id="title">{{ job.title }}</h1>
<div class="panel">
<div class="progress" id="pbox"><i id="pbar" style="width:{{ (job.progress or 0) * 100 }}%"></i></div>
<p style="margin:10px 0 0"><b id="stage">{{ job.status.stage or 'Starting ...' }}</b></p>
<div class="tiles" style="margin-top:10px"><div class="tile o"><div class="v" id="pct">-</div><div class="l">done</div></div>
<div class="tile"><div class="v" id="eta">-</div><div class="l">left (estimate)</div></div>
<div class="tile"><div class="v" id="el">-</div><div class="l">running for</div></div>
<div class="tile"><div class="v" id="st">{{ job.state }}</div><div class="l">state</div></div></div>
<p class="vctl"><button id="cancel" class="no" onclick="cancelJob()">Stop</button>
<span class="m">Stopping is safe: finished parts are kept and starting the same job again continues from there.</span></p>
<div id="next"></div>
<details open><summary class="m">Details (log)</summary><pre class="log" id="log"></pre></details></div>
{% endblock %}{% block script %}<script>
const JID="{{ job.id }}";
async function refresh(){const r=await fetch("{{ url_for('api_job_one', job_id=job.id) }}");const j=await r.json();
  document.getElementById('stage').textContent=j.stage||'Starting ...';document.getElementById('st').textContent=j.state_txt;
  const bar=document.getElementById('pbar'),box=document.getElementById('pbox');
  if(j.progress==null){box.classList.add('ind');}else{box.classList.remove('ind');bar.style.width=(j.progress*100)+'%';}
  document.getElementById('pct').textContent=j.progress==null?'…':Math.round(j.progress*100)+'%';
  document.getElementById('eta').textContent=j.state==='running'?(j.eta!=null?'~'+fmtS(j.eta):'…'):'-';
  document.getElementById('el').textContent=fmtS(j.elapsed);
  document.getElementById('log').textContent=j.log.join('\n');
  document.getElementById('cancel').style.display=j.state==='running'?'':'none';
  if(j.state!=='running'){document.getElementById('next').innerHTML=j.state==='done'?
     `<div class="okmsg">Finished. ${j.day?`<a href="${j.day_url}"><b>Open ${j.day}</b></a>`:''}</div>`:
     `<div class="alert">The job ${j.state_txt}. ${j.restart_url?`<button onclick='startJob("${j.restart_url}",{})'>Start again (continues where it stopped)</button>`:''} See the log below.</div>`;return;}
  setTimeout(refresh,2500);}
async function cancelJob(){if(!confirm('Stop this job? Finished parts are kept.'))return;await fetch("{{ url_for('api_cancel', job_id=job.id) }}",{method:'POST'});refresh();}
refresh();
</script>{% endblock %}"""

OVERVIEW = r"""{% extends "base" %}{% block main %}
<h1>{{ s.civet }} · {{ s.day }}</h1>
<div class="tiles">
 <div class="tile"><div class="v">{{ s.active_hm }}</div><div class="l">Active · {{ s.active_pct }} of the time it was seen</div></div>
 <div class="tile"><div class="v">{{ s.inactive_hm }}</div><div class="l">Inactive (resting)</div></div>
 <div class="tile"><div class="v">{{ s.not_seen_hm }}</div><div class="l">Not seen by any camera</div></div>
 <div class="tile o"><div class="v">{{ s.feeding }} / {{ s.drinking }}</div><div class="l">feeding / drinking visits</div></div>
 <div class="tile g"><div class="v">{{ s.rare }}</div><div class="l">territorial marking · grooming · urination · defecation (checked)</div></div>
</div>
{% for a in s.alerts %}<div class="alert">{{ a }}</div>{% endfor %}
{% if job %}<div class="okmsg"><span class="spin"></span> {{ job.title }} is running · <a href="{{ url_for('job_page', job_id=job.id) }}">progress</a></div>{% endif %}
{% if stale %}<div class="alert">You have answers that are not in the sheet yet. <button class="main" onclick='startJob("{{ url_for('api_start_report', run=run) }}")'>Rebuild report now</button></div>{% endif %}

<h2>What to do</h2>
<div class="steps">
 <div class="step {{ 'done' if not check_left }}"><b>Behavior check</b> - {{ check_left }} of {{ check_total }} left<br>
 <span class="m">Short videos of every visit to the sleeping box top, every moment the civet stood still, and every moment it moved
 its body while staying in place. You say if it was urination, defecation, territorial marking, grooming or nothing. Only what you confirm is counted.</span>
 <div class="act"><a class="button main" href="{{ url_for('check_page', run=run) }}">Open Behavior check</a></div></div>
 <div class="step {{ 'done' if not unsure_left }}"><b>Check unsure feeding / drinking</b> - {{ unsure_left }} left<br>
 <span class="m">Watch each one and say if it was right.</span>
 <div class="act"><a class="button" href="{{ url_for('events_page', run=run, f='check') }}">Open the list</a></div></div>
 <div class="step {{ 'done' if not stale and answered }}"><b>Rebuild the report</b> so your answers are in the sheet and charts (about {{ report_time }}).
 <div class="act"><button onclick='startJob("{{ url_for('api_start_report', run=run) }}")'>Rebuild report</button></div></div>
 <div class="step {{ 'done' if model }}"><b>Teach the app</b> (optional, after the Behavior check) - trains the behavior model on your answers (about 15 min).
 {% if model %}<br><span class="m">Last training: {{ model }}</span>{% endif %}
 <div class="act"><button onclick='startJob("{{ url_for('api_start_train', run=run) }}")'>Train behavior model</button></div></div>
 <div class="step {{ 'done' if spot_done }}"><b>Spot check</b> (optional) - random moments you answer without seeing the app's answer; gives the real accuracy. {{ spot or '' }}
 <div class="act"><a class="button" href="{{ url_for('spot_page', run=run) }}">Open Spot check</a></div></div>
</div>
<details class="panel gloss" style="margin-top:16px"><summary><b>What the words mean</b></summary>
<dl>{% for w, m in glossary %}<dt>{{ w }}</dt><dd>{{ m }}</dd>{% endfor %}</dl></details>
<p class="m" style="margin-top:20px">Models: {{ s.models }}</p>
{% endblock %}"""

SHEET = r"""{% extends "base" %}{% block main %}
<h1>Observation sheet</h1>
<p class="m">The same tables as <code>sheet.xlsx</code>. <a href="{{ url_for('run_file', run=run, path='sheet.xlsx') }}" download>Download sheet.xlsx</a>
{% if stale %} · <b style="color:var(--orange)">your newest answers are not in it yet</b> - <a href="#" onclick='startJob("{{ url_for('api_start_report', run=run) }}");return false'>rebuild report</a>{% endif %}</p>
<div class="chips" id="tabs">{% for name, _ in sheets %}<button class="{{ 'on' if loop.first }}" data-i="{{ loop.index0 }}">{{ name }}</button>{% endfor %}</div>
{% for name, html in sheets %}<div class="sheet panel" style="overflow:auto;max-height:75vh;{{ '' if loop.first else 'display:none' }}">{{ html|safe }}</div>{% endfor %}
{% endblock %}{% block head %}<style>
h3.xt{margin:0 0 10px;font-size:16px}
table.xl{border-collapse:collapse;width:auto;font-size:13px}
table.xl td{border:1px solid var(--line);padding:4px 10px;white-space:nowrap;text-align:left;min-width:44px}
table.xl.c td{text-align:center}
table.xl td.wrap{white-space:normal;max-width:720px}
table.xl td.h{background:var(--soft);font-weight:700}
table.xl td.n{font-variant-numeric:tabular-nums}
table.xl tr:hover td{background:inherit}
</style>{% endblock %}{% block script %}<script>
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('.sheet').forEach((s,i)=>s.style.display=i==b.dataset.i?'':'none');});
</script>{% endblock %}"""

CHARTS = r"""{% extends "base" %}{% block main %}
<h1>Charts</h1>
<p class="m">Click the picture to enlarge, click again to zoom in. Filled dots = counted, hollow dots = still to check.
<a href="{{ url_for('run_file', run=run, path='charts.png') }}" download>Download charts.png</a></p>
<img class="zoomable" style="width:100%;border-radius:8px;background:#fcfcfb" src="{{ url_for('run_file', run=run, path='charts.png') }}?v={{ v }}" alt="Day timeline, time per room, hourly counts and time budget">
{% endblock %}"""

EVENTS = r"""{% extends "base" %}{% block head %}<style>
.ev{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:900px){.ev{grid-template-columns:1fr}}
.side{position:sticky;top:110px}
</style>{% endblock %}{% block main %}
<h1>Events & video</h1>
<p class="m">Click any row to watch that moment from the camera (<kbd>↑</kbd> <kbd>↓</kbd> next event, <kbd>←</kbd> <kbd>→</kbd> 5 s back / forward, <kbd>Space</kbd> play / pause). If the app was wrong, correct it on the right; then rebuild the report.</p>
<div class="chips" id="filters">{% for k, n in filters %}<button data-f="{{ k }}" class="{{ 'on' if k==f }}">{{ n }}</button>{% endfor %}
<input type="text" id="q" placeholder="search (time, behavior, room)" style="margin-left:8px"></div>
<div class="ev"><div class="tablewrap"><table id="events"><thead><tr><th>Start</th><th>Stop</th><th>Length</th><th>Behavior</th><th>Room</th><th>Sure</th><th></th></tr></thead><tbody>
{% for e in events %}<tr data-i="{{ loop.index0 }}" data-cat="{{ e.category }}" data-check="{{ 1 if e.check else 0 }}" data-done="{{ 1 if e.decision else 0 }}">
<td>{{ e.time }}</td><td>{{ e.stop }}</td><td class="n">{{ e.dur }}</td><td>{{ e.name }}</td><td>{{ e.room }}</td><td class="n">{{ e.conf }}</td>
<td>{% if e.decision %}<span class="pill done">{{ e.decision }}</span>{% elif e.check %}<span class="pill check">please check</span>{% endif %}</td></tr>{% endfor %}
</tbody></table></div>
<div class="side panel" id="side"><p class="m">Pick an event on the left.</p></div></div>
{% endblock %}{% block script %}
<script>const EVENTS={{ events|tojson }};const LABELS={{ labels|tojson }};const RUN="{{ run }}";
const LOCATE="{{ url_for('api_locate', run=run) }}",SAVE="{{ url_for('save_review', run=run) }}";</script>
<script>{% include "player_js" %}</script>
<script>
let cur=-1;
function show(i){cur=i;const e=EVENTS[i];document.querySelectorAll('#events tbody tr').forEach(tr=>tr.classList.toggle('sel',+tr.dataset.i===i));
  const side=document.getElementById('side');
  side.innerHTML=`<div class="vbox" id="vb"><video id="vid" muted playsinline></video><div class="none" id="vnone" style="display:none"></div></div>
  ${P.bar()}<div class="vctl"><button id="zb" onclick="P.zoom()">Zoom on civet</button>
   <button onclick="P.speed()">Speed <span id="spd">1×</span></button>${e.clip?`<a href="${e.clip}" class="m">labelled clip</a>`:''}${e.grid?` <a class="m zoomable" href="${e.grid}">4 pictures</a>`:''}</div>
  <h3 style="margin:12px 0 2px">${e.name}</h3><div class="m">${e.day} ${e.time} - ${e.stop} · ${e.dur} · ${e.room} (${e.camera||'no camera'}) · sure ${e.conf} · ${e.cond}</div>
  <div class="m" id="evnote">${e.note||''}</div>
  ${e.editable?`<div class="vctl" style="margin-top:10px"><button class="ok" onclick="decide('confirmed')">✓ Correct</button><button class="no" onclick="decide('rejected')">✗ Not this behavior</button>
  <select class="f" onchange="decide('corrected',this.value)"><option value="">It was …</option>${LABELS.map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></div>
  <div class="m" id="saved">${e.decision?'Saved: '+e.decision:''}</div>`:`<p class="m">${e.category==='stop'?'Answer short stops in <a href="{{ url_for('check_page', run=run) }}">Behavior check</a>.':''}</p>`}`;
  P.load(e.camera,e.start_iso,e.stop_iso,e.box,2);}
async function decide(status,label){if(status==='corrected'&&!label)return;const e=EVENTS[cur];
  const r=await fetch(SAVE,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx:e.idx,status,label:label||null})});
  const txt=status==='corrected'?'it was '+LABELS.find(x=>x[0]===label)[1]:(status==='confirmed'?'correct':'not this behavior');
  document.getElementById('saved').textContent=r.ok?'Saved: '+txt+' (rebuild the report to update the sheet)':'Save failed';
  if(r.ok){e.decision=txt;e.note='checked by a person: '+txt+' - new answer, rebuild the report to update the sheet';
    document.getElementById('evnote').textContent=e.note;const tr=document.querySelector(`#events tr[data-i="${cur}"]`);tr.dataset.done=1;tr.lastElementChild.innerHTML=`<span class="pill done">${txt}</span>`;}}
document.querySelectorAll('#events tbody tr').forEach(tr=>tr.onclick=()=>show(+tr.dataset.i));
let F="{{ f }}";function filt(){const q=document.getElementById('q').value.toLowerCase();
  document.querySelectorAll('#events tbody tr').forEach(tr=>{const ok=(F==='all'||tr.dataset.cat===F||(F==='check'&&tr.dataset.check==='1'&&tr.dataset.done==='0')||(F==='done'&&tr.dataset.done==='1'));
   tr.style.display=ok&&(!q||tr.textContent.toLowerCase().includes(q))?'':'none';});}
document.querySelectorAll('#filters button').forEach(b=>b.onclick=()=>{F=b.dataset.f;document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('on',x===b));filt();});
document.getElementById('q').oninput=filt;filt();
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const vis=[...document.querySelectorAll('#events tbody tr')].filter(t=>t.style.display!=='none').map(t=>+t.dataset.i);const k=vis.indexOf(cur);
  if(e.key==='ArrowDown'){e.preventDefault();show(vis[Math.min(vis.length-1,k+1)]);}if(e.key==='ArrowUp'){e.preventDefault();show(vis[Math.max(0,k-1)]);}
  if(e.key==='ArrowLeft'){e.preventDefault();P.jump(-5);}if(e.key==='ArrowRight'){e.preventDefault();P.jump(5);}
  if(e.key===' '){e.preventDefault();P.toggle();}});
</script>{% endblock %}"""

# shared video player: plays the prepared camera video around an event, loops, zooms on the civet
PLAYER_JS = r"""
const P={v:null,a:0,b:0,ea:0,eb:0,w0:0,w1:1,t0:0,z:false,box:null,rate:1,loop:true,drag:false,
 bar(){return `<div class="tl"><button id="pp" onclick="P.toggle()" title="Play / pause (Space)">▶</button>
  <button onclick="P.jump(-5)" title="5 s back">−5 s</button>
  <div class="seekw"><div class="trk"></div><div class="band" id="band" title="the event"></div>
  <input type="range" id="seek" min="0" max="1" step="0.5" value="0" aria-label="Video time"></div>
  <button onclick="P.jump(5)" title="5 s forward">+5 s</button><span class="clock" id="clock">--:--:--</span></div>
  <div class="vctl"><button onclick="P.back()">↺ Back to the event</button>
  <label class="m"><input type="checkbox" id="loop" checked onchange="P.loop=this.checked"> loop the event</label>
  <span class="m">orange = the event · drag the bar to see before / after</span></div>`;},
 async load(cam,startIso,stopIso,box,pad){this.v=document.getElementById('vid');this.box=box;const none=document.getElementById('vnone');
  if(!cam){none.style.display='';none.textContent='No camera saw the civet here';return;}
  const r=await fetch(`${LOCATE}?cam=${cam}&t=${encodeURIComponent(startIso)}&t2=${encodeURIComponent(stopIso)}`);const j=await r.json();
  if(!r.ok||!j.url){none.style.display='';none.textContent=j.error||'Video not found';return;}
  none.style.display='none';this.ea=j.offset;this.eb=j.offset+j.length;this.a=Math.max(0,this.ea-pad);this.b=this.eb+pad;
  const ctx=Math.max(30,j.length);this.w0=Math.max(0,this.ea-ctx);this.w1=this.eb+ctx;
  this.t0=new Date(startIso.replace(' ','T')).getTime()-this.ea*1000;
  this.loop=true;const lp=document.getElementById('loop');if(lp)lp.checked=true;
  const v=this.v;if(!v.src.endsWith(j.url)){v.src=j.url;}v.playbackRate=this.rate;
  const go=()=>{if(v.duration)this.w1=Math.min(this.w1,v.duration);this.draw();v.currentTime=this.a;v.play().catch(()=>{});};
  if(v.readyState>=1)go();else v.onloadedmetadata=go;
  v.ontimeupdate=()=>{const c=v.currentTime;if(this.loop&&(c>this.b||c<this.a-1)){v.currentTime=this.a;return;}
    if(c>this.w1){this.w1=Math.min(v.duration||c+30,c+30);this.draw();}if(c<this.w0){this.w0=Math.max(0,c-30);this.draw();}this.tick();};
  v.onplay=v.onpause=()=>{const b=document.getElementById('pp');if(b)b.textContent=v.paused?'▶':'❚❚';};
  const sk=document.getElementById('seek');
  if(sk){sk.oninput=()=>{this.drag=true;const t=+sk.value;v.currentTime=t;if(t<this.a||t>this.b)this.setLoop(false);this.tick(t);};
    sk.onchange=()=>{this.drag=false;};}
  this.draw();this.apply();},
 draw(){const sk=document.getElementById('seek'),bd=document.getElementById('band');if(!sk)return;
  sk.min=this.w0;sk.max=this.w1;const w=Math.max(1e-6,this.w1-this.w0);
  if(bd){bd.style.left=((this.ea-this.w0)/w*100)+'%';bd.style.width=(Math.max(0.5,(this.eb-this.ea)/w*100))+'%';}this.tick();},
 tick(t){const v=this.v;if(!v)return;t=t??v.currentTime;const sk=document.getElementById('seek');if(sk&&!this.drag)sk.value=t;
  const c=document.getElementById('clock');if(c){const d=new Date(this.t0+t*1000);c.textContent=d.toTimeString().slice(0,8);}},
 setLoop(on){this.loop=on;const lp=document.getElementById('loop');if(lp)lp.checked=on;},
 jump(s){if(!this.v)return;const t=Math.max(0,this.v.currentTime+s);if(t<this.a||t>this.b)this.setLoop(false);this.v.currentTime=t;this.tick(t);},
 back(){if(!this.v)return;this.setLoop(true);this.v.currentTime=this.a;this.v.play().catch(()=>{});},
 toggle(){if(!this.v)return;this.v.paused?this.v.play():this.v.pause();},
 replay(){this.back();},
 speed(){this.rate=this.rate===1?2:(this.rate===2?0.5:1);if(this.v)this.v.playbackRate=this.rate;const s=document.getElementById('spd');if(s)s.textContent=this.rate+'×';},
 zoom(){this.z=!this.z;this.apply();},
 apply(){const v=this.v;if(!v)return;const zb=document.getElementById('zb');if(zb)zb.textContent=this.z?'Whole picture':'Zoom on civet';
  if(!this.z||!this.box){v.style.transform='';return;}const [x1,y1,x2,y2]=this.box,W=960,H=540;
  const s=Math.max(1,Math.min(4,0.6*H/Math.max(x2-x1,y2-y1,1)));v.style.transformOrigin=`${(x1+x2)/2/W*100}% ${(y1+y2)/2/H*100}%`;v.style.transform=`scale(${s})`;}};
"""

CHECK = r"""{% extends "base" %}{% block head %}<style>
.ck{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(260px,1fr);gap:16px;align-items:start}
@media (max-width:900px){.ck{grid-template-columns:1fr}}
.list{max-height:calc(100vh - 230px);overflow:auto;background:var(--panel);border-radius:8px}
.it{display:flex;gap:8px;padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer;align-items:center;font-size:13px}
.it img{width:76px;height:76px;object-fit:cover;border-radius:4px;background:#000;flex:none}
.it.cur{background:var(--warn-bg)}.it .a{font-weight:700;color:var(--olive)}.it .g{color:var(--orange)}
.keys{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:6px;margin-top:10px}
.keys button{text-align:left;padding:8px 10px}.keys button.picked{background:var(--olive);color:#fff;border-color:var(--olive)}
.keys button b{display:inline-block;min-width:22px}
</style>{% endblock %}{% block main %}
<h1>Behavior check <span class="m" style="font-size:15px;font-weight:400">· <span id="cnt">{{ done }}</span> of {{ items|length }} checked</span></h1>
<div class="chips" id="filters">
 <button data-f="todo" class="on">Not checked yet</button><button data-f="all">All</button><button data-f="box_top">Sleeping box top</button><button data-f="short_stop">Stood still</button>
 <button data-f="busy_still">Moving in place</button><button data-f="found">Found behaviors</button>
 <span class="m" style="align-self:center">Sorted: most likely behaviors first{{ ' (your trained model)' if trained else '' }}</span></div>
{% if not items %}<div class="panel">Nothing to check. Rebuild the report first.</div>{% else %}
<div class="ck"><div>
 <div class="vbox" id="vb"><video id="vid" muted playsinline></video><div class="none" id="vnone" style="display:none"></div></div>
 <div id="tlbox"></div>
 <div class="vctl"><button id="zb" onclick="P.zoom()"><kbd>Z</kbd> zoom</button><button onclick="P.speed()">speed <span id="spd">1×</span></button>
  <a class="m zoomable" id="gridlink" href="#">4 pictures (enlarge)</a></div>
 <div class="m" id="info" style="margin-top:6px"></div>
 <div class="keys">{% for k, name, what in choices %}<button data-k="{{ k }}" title="{{ what }}"><b>{{ k }}</b>{{ name }}</button>{% endfor %}</div>
 <p class="m"><kbd>←</kbd> <kbd>→</kbd> previous / next · <kbd>Shift</kbd>+<kbd>←</kbd> <kbd>→</kbd> 5 s back / forward · <kbd>Space</kbd> play / pause · <kbd>R</kbd> back to the event · the answer is saved at once and the next one opens.</p>
 <details class="panel" style="margin-top:8px"><summary><b>How to tell them apart</b></summary>
 <dl class="gloss">{% for k, name, what in choices %}<dt>{{ k }} · {{ name }}</dt><dd>{{ what }}</dd>{% endfor %}</dl></details>
</div>
<div class="list" id="list">{% for it in items %}
<div class="it" data-i="{{ loop.index0 }}"><img src="{{ url_for('run_file', run=run, path=it.grid) }}" loading="lazy" alt="">
<div>{{ it.time }} · {{ it.room }}<br><span class="m">{{ it.kind_txt }} {{ it.dur }} s</span>{% if it.guess %}<br><span class="g">app: {{ it.guess }}</span>{% endif %}
<br><span class="a">{{ it.answer }}</span></div></div>{% endfor %}</div></div>
{% endif %}{% endblock %}{% block script %}
<script>const ITEMS={{ items|tojson }};const CH={{ ch|tojson }};
const LOCATE="{{ url_for('api_locate', run=run) }}",SAVE="{{ url_for('save_review', run=run) }}",FILE="{{ url_for('run_file', run=run, path='') }}";</script>
<script>{% include "player_js" %}</script>
<script>
document.getElementById('tlbox').innerHTML=P.bar();
const els=[...document.querySelectorAll('.it')];let cur=-1,F='todo';
P.z=true;
function visible(){return els.filter(x=>x.style.display!=='none').map(x=>+x.dataset.i);}
function filt(){els.forEach(x=>{const it=ITEMS[+x.dataset.i];let ok=F==='all'||(F==='todo'&&!it.answer)||(F===it.kind)||(F==='found'&&it.found);x.style.display=ok?'':'none';});}
function show(i){if(i==null||i<0||!ITEMS[i])return;cur=i;const it=ITEMS[i];els.forEach(x=>x.classList.toggle('cur',+x.dataset.i===i));
  els[i].scrollIntoView({block:'nearest'});
  document.getElementById('info').innerHTML=`<b>${it.time}</b> · ${it.room} (${it.camera}) · ${it.kind_txt} ${it.dur} s${it.change?` · box top changed ${it.change} (red in the 'after' picture)`:''}${it.guess?` · <span style="color:var(--orange)">app's guess: ${it.guess}</span>`:''}${it.answer?` · your answer: <b>${it.answer}</b>`:''}`;
  document.getElementById('gridlink').href=FILE+it.grid;
  document.querySelectorAll('.keys button').forEach(b=>b.classList.toggle('picked',CH[b.dataset.k][0]===it.answer));
  P.load(it.camera,it.start_iso,it.stop_iso,it.box,2);}
function next(){const v=visible();const k=v.indexOf(cur);show(v[k+1]!==undefined?v[k+1]:v[k]);}
function prev(){const v=visible();const k=v.indexOf(cur);show(v[Math.max(0,k-1)]);}
async function answer(k){const it=ITEMS[cur];if(!it)return;const [name,status,label]=CH[k];const was=it.answer;
  const el=els[cur];const i=cur;next();
  const r=await fetch(SAVE,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx:it.idx,status,label:label||null})});
  if(!r.ok){el.querySelector('.a').textContent='save failed';return;}
  it.answer=name;it.found=['urination','defecation','territorial_marking','grooming','toileting'].includes(label);el.querySelector('.a').textContent=name;
  if(!was){const c=document.getElementById('cnt');c.textContent=+c.textContent+1;}
  if(F==='todo'&&i!==cur)el.style.display='none';}
els.forEach(x=>x.onclick=()=>show(+x.dataset.i));
document.querySelectorAll('.keys button').forEach(b=>b.onclick=()=>answer(b.dataset.k));
document.querySelectorAll('#filters button').forEach(b=>b.onclick=()=>{F=b.dataset.f;document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('on',x===b));filt();show(visible()[0]);});
document.addEventListener('keydown',e=>{if(document.getElementById('lb').className.includes('open')){return;}
  if(e.key===' '){e.preventDefault();P.toggle();return;}
  if(e.shiftKey&&e.key==='ArrowLeft'){e.preventDefault();P.jump(-5);return;}if(e.shiftKey&&e.key==='ArrowRight'){e.preventDefault();P.jump(5);return;}
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){e.preventDefault();next();return;}if(e.key==='ArrowLeft'||e.key==='ArrowUp'){e.preventDefault();prev();return;}
  if(e.key==='z'||e.key==='Z'){P.zoom();return;}if(e.key==='r'||e.key==='R'){P.replay();return;}
  if(CH[e.key])answer(e.key);});
filt();show(visible()[0]??0);
</script>{% endblock %}"""

SPOT = r"""{% extends "base" %}{% block main %}
<h1>Spot check <span class="m" style="font-size:15px;font-weight:400">· {{ done }} of {{ total }} checked · <a href="{{ url_for('spot_results', run=run) }}">results</a></span></h1>
{% if not ready %}
<div class="panel"><p>A spot check picks {{ n }} random moments of the day. You answer where the civet is and what it does - without seeing the app's answer -
and the app measures how often it agrees with you. That is its real accuracy.</p>
<p><button class="main" onclick='startJob("{{ url_for('api_start_spot', run=run) }}")'>Make the spot check (about 5 min)</button></p></div>
{% elif item %}
<p class="m">Moment {{ item.id + 1 }} · {{ item.t[11:19] }} · rows = cameras / rooms, columns = 1 s before, the moment (t), 1 s after. The app's answer stays hidden.</p>
<img class="zoomable" style="width:100%;border-radius:6px;background:#000" src="{{ url_for('run_file', run=run, path=item.image) }}" alt="All cameras around {{ item.t }}">
<form method="post" class="panel" style="margin-top:10px"><input type="hidden" name="id" value="{{ item.id }}">
<p><b>1 · Where is the civet at t?</b> <span class="m">Tick every room you see it in - near the R2 / R3 opening it can show in both cameras at once.</span><br>
{% set prev_rooms = (prev.room or '').split('+') if prev else [] %}
{% for r in rooms %}<label class="button"><input type="checkbox" name="room" value="{{ r }}" class="rm" {{ 'checked' if r in prev_rooms }}> {{ r }}</label> {% endfor %}
<label class="button"><input type="checkbox" name="room" value="none" class="rn" {{ 'checked' if 'none' in prev_rooms }}> Not visible in any camera</label></p>
<p><b>2 · What is it doing?</b> (skip if not visible)<br>{% for k, v in labels %}<label class="button" style="margin:2px"><input type="radio" name="label" value="{{ k }}" {{ 'checked' if prev and prev.label==k }}> {{ v }}</label> {% endfor %}</p>
<p><input type="text" name="checker" placeholder="your name" value="{{ checker }}"> <button class="main">Save and next</button></p></form>
<script>
const rm=[...document.querySelectorAll('input.rm')],rn=document.querySelector('input.rn');
rm.forEach(b=>b.onchange=()=>{if(b.checked)rn.checked=false;});rn.onchange=()=>{if(rn.checked)rm.forEach(b=>b.checked=false);};
document.querySelector('form').onsubmit=e=>{if(!rn.checked&&!rm.some(b=>b.checked)){e.preventDefault();alert('Tick where the civet is (or "Not visible").');}};
</script>
{% else %}<div class="okmsg">All moments are checked. <a href="{{ url_for('spot_results', run=run) }}">See the results.</a></div>{% endif %}
{% endblock %}"""

SPOT_RESULTS = r"""{% extends "base" %}{% block main %}
<h1>Spot check results</h1><p><a href="{{ url_for('spot_page', run=run) }}">continue checking</a></p>
<div class="tablewrap"><table>{% for k, v in summary.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table></div>
<h2>Behavior: checker vs CivetWatch</h2><div class="tablewrap">{{ confusion|safe }}</div>
<h2>Disagreements</h2><div class="tablewrap">{{ wrong|safe }}</div>
{% endblock %}"""

FILES = r"""{% extends "base" %}{% block main %}
<h1>Files</h1>
<p class="m">Everything is in <code>{{ folder }}</code>. <button onclick="fetch('{{ url_for('api_open_folder', run=run) }}',{method:'POST'})">Open the folder</button></p>
<div class="tablewrap"><table><thead><tr><th>File</th><th>What it is</th><th>Size</th><th></th></tr></thead><tbody>
{% for f, what, size in files %}<tr><td>{{ f }}</td><td>{{ what }}</td><td class="n">{{ size }}</td><td><a href="{{ url_for('run_file', run=run, path=f) }}" download>download</a></td></tr>{% endfor %}
</tbody></table></div>{% endblock %}"""

FILE_INFO = {"sheet.xlsx": "Observation sheet for Excel (zoo layout, Read me tab)",
             "charts.png": "Day timeline, time per room, hourly counts, time budget",
             "events.csv": "Every event with start, stop, room, camera",
             "frames.csv": "The app's answer for every half second (large)",
             "summary.json": "Day totals", "review.json": "Your answers (Behavior check, corrections)",
             "behavior_model.json": "Test result of the trained behavior model",
             "behavior_model_test.csv": "Behavior model answers on the test stops"}

QUICK_CHOICES = [
    ("1", "Urination", "Stands or half-squats with the hindquarters lowered and the tail lifted; usually a few seconds. "
                        "On the box top: look for a new wet (darker) patch in the 'after' picture."),
    ("2", "Defecation", "Deep squat, back hunched, tail raised, held still longer (often 5 s or more). "
                         "On the box top: look for new droppings (marked red) in the 'after' picture."),
    ("3", "Territorial marking", "Presses or rubs its rear end (scent gland under the tail) on the box, a log, wall, door frame or fence."),
    ("4", "Grooming", "Licks, nibbles or scratches its own fur; head turned back to its body or a hind leg scratching."),
    ("5", "Lying down", "Resting or napping on its belly or side, head down or on its paws (Inactive)."),
    ("6", "Stretched out", "Lying with the body and legs stretched long (Inactive)."),
    ("7", "Curled up / sleeping", "Curled into a ball, tail around the body, usually asleep (Inactive)."),
    ("8", "Sitting / standing", "Awake and staying in place: sitting, standing, looking around (Active)."),
    ("0", "Nothing special", "Walking, sniffing, turning or anything else - not one of the behaviors above."),
    ("9", "Can't see", "The civet is hidden or the picture is too dark: skipped, not used for training."),
]
QUICK_STATUS = {"1": ("corrected", "urination"), "2": ("corrected", "defecation"),
                "3": ("corrected", "territorial_marking"), "4": ("corrected", "grooming"), "0": ("rejected", ""),
                "5": ("corrected", "lying_down"), "6": ("corrected", "stretched_out"),
                "7": ("corrected", "curling_up"), "8": ("corrected", "active_other"),
                "9": ("corrected", "cant_see")}
RARE_LABELS = ("urination", "defecation", "territorial_marking", "grooming", "toileting")
GUESS_TXT = {"urination": "Urination", "defecation": "Defecation", "territorial_marking": "Territorial marking",
             "grooming": "Grooming", "nothing": ""}

TEMPLATES = {"base": BASE, "home": HOME, "new": NEW, "job": JOB, "overview": OVERVIEW, "sheet": SHEET, "charts": CHARTS,
             "events": EVENTS, "player_js": PLAYER_JS, "check": CHECK, "spot": SPOT, "spot_results": SPOT_RESULTS,
             "files": FILES}


# -------------------------------------------------------------------- helpers ---
def _runs(root):
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not name.startswith("_") and os.path.isdir(os.path.join(root, name)) and (
                os.path.exists(os.path.join(root, name, "summary.json")) or os.path.exists(os.path.join(root, name, "run.json"))):
            out.append(name)
    return sorted(out, reverse=True)


_review_lock = threading.Lock()


def review_key(start, camera, category):
    return f"{str(start)[:19]}|{camera}|{'stop' if category == 'stop' else 'event'}"


def _load_review(d):
    """review.json, keyed by time|camera|kind (older files keyed by row number are converted)."""
    p = os.path.join(d, "review.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    out = {}
    for k, v in r.items():
        if "|" not in k and v.get("start"):
            b = v.get("behavior", "")
            cat = "stop" if b.startswith("possible_") or b in ("short_stop", "doorway_pass") else "event"
            k = review_key(v["start"], v.get("camera", ""), cat)
        out[k] = v
    return out


def _folder_bytes(path):
    total = 0
    for base, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def _size_txt(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _cell_text(v):
    import datetime as _dt
    if v is None:
        return ""
    if isinstance(v, _dt.datetime):
        return f"{v:%H:%M}" if v.second == 0 else f"{v:%H:%M:%S}"
    if isinstance(v, _dt.time):
        return f"{v:%H:%M}" if v.second == 0 else f"{v:%H:%M:%S}"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.2f}"
    return str(v)


def _sheet_html(path):
    """sheet.xlsx as HTML tables that look like the Excel file: merged cells,
    borders, bold headers. A title row spanning the whole sheet becomes a heading."""
    from html import escape
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    out = []
    for ws in wb:
        span, hidden = {}, set()
        for m in ws.merged_cells.ranges:
            span[(m.min_row, m.min_col)] = (m.max_row - m.min_row + 1, m.max_col - m.min_col + 1)
            for r in range(m.min_row, m.max_row + 1):
                for c in range(m.min_col, m.max_col + 1):
                    if (r, c) != (m.min_row, m.min_col):
                        hidden.add((r, c))
        ncol = ws.max_column
        # last row that has any value
        last = max([c.row for row in ws.iter_rows() for c in row if c.value not in (None, "")] or [0])
        title, first = "", 1
        rs, cs = span.get((1, 1), (1, 1))
        if ncol > 2 and (1, 1) in span and cs >= ncol - 1 and ws.cell(1, 1).value:
            title, first = _cell_text(ws.cell(1, 1).value), 2
        rows, notes = [], []
        for r in range(first, last + 1):
            vals = [(c, ws.cell(r, c).value) for c in range(1, ncol + 1) if ws.cell(r, c).value not in (None, "")]
            if len(vals) == 1 and isinstance(vals[0][1], str) and len(vals[0][1]) > 40 and (r, vals[0][0]) not in span:
                notes.append(vals[0][1])          # a note that spills over empty cells in Excel: below the table here
                continue
            cells = []
            for c in range(1, ncol + 1):
                if (r, c) in hidden:
                    continue
                cell = ws.cell(r, c)
                rs, cs = span.get((r, c), (1, 1))
                attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                bold = bool(cell.font and cell.font.b)
                fill = cell.fill.fgColor.rgb if cell.fill and cell.fill.fill_type == "solid" else None
                cls = "h" if bold or (isinstance(fill, str) and fill not in ("00000000", "FFFFFFFF")) else ""
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    cls += " n"
                if isinstance(cell.value, str) and len(cell.value) > 80:
                    cls += " wrap"
                cells.append(f'<td class="{cls.strip()}"{attrs}>{escape(_cell_text(cell.value))}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        html = (f'<h3 class="xt">{escape(title)}</h3>' if title else "") + \
            f'<table class="xl{" c" if span else ""}">' + "".join(rows) + "</table>" + \
            "".join(f'<p class="m" style="max-width:900px">{escape(n)}</p>' for n in notes)
        out.append((ws.title, html))
    return out


def _fmt_dur(s):
    s = float(s)
    return f"{s:.0f} s" if s < 90 else (f"{s / 60:.1f} min" if s < 3600 else fmt_hm(s))


def create_app(root, cfg):
    app = Flask(__name__)
    app.jinja_loader = DictLoader(TEMPLATES)
    root = os.path.abspath(root)
    cache = {}

    def run_dir(run):
        if run not in _runs(root):
            abort(404)
        return os.path.join(root, run)

    def run_info(run):
        """manifest + settings of a day (run.json, or the data/<day>/ convention)."""
        d = run_dir(run)
        info = {}
        p = os.path.join(d, "run.json")
        if os.path.exists(p):
            info = json.load(open(p, encoding="utf-8"))
        man = info.get("manifest") or f"data/{run}/manifest.yaml"
        conf = info.get("config") or (f"data/{run}/config.yaml" if os.path.exists(resolve(f"data/{run}/config.yaml")) else None)
        return {"manifest": resolve(man), "config": resolve(conf) if conf else None,
                "manifest_rel": man, "config_rel": conf}

    def events(run):
        d = run_dir(run)
        p = os.path.join(d, "events.csv")
        key = ("ev", run, os.path.getmtime(p))
        if key not in cache:
            cache[key] = pd.read_csv(p, parse_dates=["start", "stop"], keep_default_na=False)
        return cache[key]

    def boxes(run):
        """Median civet box of every event (for 'zoom on civet'), from frames.csv, cached."""
        d = run_dir(run)
        p = os.path.join(d, "frames.csv")
        key = ("box", run, os.path.getmtime(p) if os.path.exists(p) else 0, os.path.getmtime(os.path.join(d, "events.csv")))
        if key not in cache:
            out = {}
            if os.path.exists(p):
                fr = pd.read_csv(p, usecols=["t", "camera", "x1", "y1", "x2", "y2"], parse_dates=["t"], low_memory=False)
                fr = fr[fr["camera"].fillna("") != ""].set_index("t").sort_index()
                for i, e in events(run).iterrows():
                    if not e["camera"]:
                        continue
                    seg = fr.loc[e["start"]:e["stop"]]
                    seg = seg[seg["camera"] == e["camera"]]
                    if len(seg):
                        out[i] = [round(float(seg[c].median()), 1) for c in ("x1", "y1", "x2", "y2")]
            cache[key] = out
        return cache[key]

    def readers(run):
        info = run_info(run)
        key = ("rd", run, os.path.getmtime(info["manifest"]) if os.path.exists(info["manifest"]) else 0)
        if key not in cache:
            from .video import CameraReader
            m = load_manifest(info["manifest"])
            cache[key] = {c["id"]: CameraReader(c) for c in m["cameras"]}
        return cache[key]

    def is_stale(d):
        r, s = os.path.join(d, "review.json"), os.path.join(d, "events.csv")
        return os.path.exists(r) and os.path.exists(s) and os.path.getmtime(r) > os.path.getmtime(s)

    def check_items(run, with_boxes=True):
        ev = events(run)
        review = _load_review(run_dir(run))
        names = {v: n for k, n, _ in QUICK_CHOICES for kk, v in [(k, QUICK_STATUS[k])]}
        bx = boxes(run) if with_boxes else {}
        items = []
        for idx, e in ev.iterrows():
            if e["category"] != "stop" or not e.get("grid"):
                continue
            dec = review.get(review_key(e["start"], e["camera"], "stop"))
            answer = ""
            if dec:
                answer = names.get((dec["status"], dec.get("label") or ""), (dec.get("label") or dec["status"]).replace("_", " "))
            guess = GUESS_TXT.get(str(e.get("guess", "")), "")
            gp = float(e.get("guess_prob") or 0)
            kind = e.get("kind") or "short_stop"
            change = float(e.get("box_change") or 0)
            score = (gp if guess else 0) + float(e.get("toilet_score") or 0) + (10 + 100 * change if kind == "box_top" else 0)
            items.append({"idx": int(idx), "grid": e["grid"], "time": f"{e['start']:%H:%M:%S}", "room": e["room"],
                          "camera": e["camera"], "dur": f"{float(e['duration_s']):.0f}", "answer": answer,
                          "found": bool(dec and dec.get("label") in RARE_LABELS),
                          "change": f"{change:.1%}" if kind == "box_top" else "",
                          "kind": kind, "kind_txt": "on the sleeping box top" if kind == "box_top" else
                          "moving in place" if kind == "busy_still" else (
                              "passing a door, still" if e["behavior"] == "doorway_pass" else "stood still"),
                          "guess": f"{guess} {gp:.0%}" if guess else "", "score": score,
                          "doorway": e["behavior"] == "doorway_pass",
                          "start_iso": f"{e['start']:%Y-%m-%d %H:%M:%S}", "stop_iso": f"{e['stop']:%Y-%m-%d %H:%M:%S}",
                          "box": bx.get(idx)})
        items.sort(key=lambda i: (i["doorway"], -i["score"]))      # doorway passes last
        return items

    @app.context_processor
    def i18n():
        from .i18n import script
        return {"i18n_script": script()}

    @app.context_processor
    def common():
        run = request.view_args.get("run") if request.view_args else None
        left = None
        if run and run in _runs(root) and os.path.exists(os.path.join(root, run, "events.csv")):
            try:
                left = sum(1 for i in check_items(run, False) if not i["answer"]) or None
            except Exception:
                left = None
        return {"run": run, "runs": _runs(root), "nav_left": left}

    # ------------------------------------------------------------- pages ---
    @app.route("/")
    def home():
        days = []
        for r in _runs(root):
            d = os.path.join(root, r)
            vd = os.path.join(PROJECT_ROOT, "data", r)
            item = {"run": r, "day": r, "weekday": "", "ok": os.path.exists(os.path.join(d, "summary.json")), "left": 0,
                    "size": _size_txt(_folder_bytes(d)),
                    "video_size": _size_txt(_folder_bytes(vd)) if os.path.isdir(vd) else None}
            try:
                t = datetime.strptime(r[:10], "%Y-%m-%d")
                item["day"], item["weekday"] = f"{t:%d %b %Y}", f"{t:%A}"
            except ValueError:
                pass
            if item["ok"]:
                s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
                item.update(active=fmt_hm(s["active_s"]), inactive=fmt_hm(s["inactive_s"]),
                            feeding=s["counts"].get(E.FEEDING, 0), drinking=s["counts"].get(E.DRINKING, 0))
                try:
                    item["left"] = sum(1 for i in check_items(r, False) if not i["answer"])
                except Exception:
                    pass
            else:
                parts = len([p for p in os.listdir(os.path.join(d, "parts")) if p.endswith(".pkl")]) \
                    if os.path.isdir(os.path.join(d, "parts")) else 0
                item["state"] = f"{parts} hour(s) analysed - start it again to continue"
            days.append(item)
        hist = []
        for j in jobs.list_jobs(8):
            j["started_txt"] = f"{datetime.fromtimestamp(j['started']):%d %b %H:%M}"
            j["took"] = jobs.fmt_duration(j["elapsed"])
            hist.append(j)
        return render_template("home", days=days, job=jobs.running(), history=hist)

    @app.route("/new")
    def new_page():
        last = ""
        for r in _runs(root):
            p = os.path.join(root, r, "run.json")
            if os.path.exists(p):
                src = json.load(open(p, encoding="utf-8")).get("sources", {})
                if src:
                    first = list(src.values())[0]
                    last = os.path.dirname(first[0]) if isinstance(first, list) and first else os.path.dirname(first)
                    break
        if not last:
            guess = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "ชะมดแผงสันหางดำ")
            last = guess if os.path.isdir(guess) else ""
        return render_template("new", today=f"{datetime.now():%Y-%m-%d}", last_source=last)

    @app.route("/job/<job_id>")
    def job_page(job_id):
        try:
            job = jobs.load(job_id)
        except OSError:
            abort(404)
        return render_template("job", job=job)

    @app.route("/run/<run>")
    def run_page(run):
        d = run_dir(run)
        if not os.path.exists(os.path.join(d, "summary.json")):
            return redirect(url_for("home"))
        s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
        ev = events(run)
        review = _load_review(d)
        start = pd.Timestamp(s["start"])
        nv = float(ev.loc[ev["category"] == "not_visible", "duration_s"].sum())
        rare = sum(s["counts"].get(k, 0) for k in (E.GROOMING, E.MARKING, E.URINATION, E.DEFECATION))
        view = {"civet": s.get("civet", "Civet"), "day": f"{start:%A %d %B %Y}",
                "active_hm": fmt_hm(s["active_s"]), "inactive_hm": fmt_hm(s["inactive_s"]), "not_seen_hm": fmt_hm(nv),
                "active_pct": f"{s['active_frac']:.0%}", "feeding": s["counts"].get(E.FEEDING, 0),
                "drinking": s["counts"].get(E.DRINKING, 0), "rare": rare,
                "alerts": [a for a in s.get("alerts", []) if "to check by a person" not in a],
                "models": ", ".join(f"{k}={v}" for k, v in s.get("models", {}).items() if k != "cameras")}
        items = check_items(run, False)
        unsure = [i for i, e in ev.iterrows() if e["category"] != "stop" and str(e["needs_review"]) == "True"
                  and review_key(e["start"], e["camera"], "event") not in review]
        spot, spot_done = None, False
        if os.path.exists(os.path.join(d, "spotcheck", "items.json")):
            it, answers = spotcheck.load(d)
            spot, spot_done = f"{len(answers)} of {len(it['items'])} moments checked.", len(answers) >= len(it["items"])
        model = None
        mp = os.path.join(d, "behavior_model.json")
        if os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            model = f"{datetime.fromtimestamp(os.path.getmtime(mp)):%d %b %H:%M}, " + (
                f"{m['accuracy']:.0%} right on {m['test_stops']} test stops" if "accuracy" in m else "trained on all answers")
        from .glossary import GLOSSARY
        return render_template("overview", s=view, check_left=sum(1 for i in items if not i["answer"]),
                               check_total=len(items), unsure_left=len(unsure), stale=is_stale(d), answered=bool(review),
                               report_time=jobs.fmt_duration(jobs.speeds()["report_s"]), model=model,
                               spot=spot, spot_done=spot_done, glossary=GLOSSARY, job=jobs.running())

    @app.route("/run/<run>/sheet")
    def sheet_page(run):
        d = run_dir(run)
        p = os.path.join(d, "sheet.xlsx")
        sheets = _sheet_html(p) if os.path.exists(p) else []
        return render_template("sheet", sheets=sheets, stale=is_stale(d))

    @app.route("/run/<run>/charts")
    def charts_page(run):
        p = os.path.join(run_dir(run), "charts.png")
        return render_template("charts", v=int(os.path.getmtime(p)) if os.path.exists(p) else 0)

    @app.route("/run/<run>/events")
    def events_page(run):
        d = run_dir(run)
        ev = events(run)
        review = _load_review(d)
        bx = boxes(run)
        built = f"{datetime.fromtimestamp(os.path.getmtime(os.path.join(d, 'events.csv'))):%Y-%m-%d %H:%M:%S}"
        check_names = {v: n for k, n, _ in QUICK_CHOICES for v in [QUICK_STATUS[k]]}
        rows = []
        for idx, e in ev.iterrows():
            cat = e["category"]
            dec = review.get(review_key(e["start"], e["camera"], cat))
            decision = ""
            if dec:
                decision = {"confirmed": "correct", "rejected": "not this behavior"}.get(
                    dec["status"], "it was " + E.DISPLAY.get(dec.get("label") or "", str(dec.get("label")).replace("_", " ")))
            name = E.DISPLAY.get(e["behavior"], e["behavior"].replace("_", " "))
            note = str(e.get("note", ""))
            # the note shows the latest answer at once (events.csv only changes when the report is rebuilt);
            # a counted grooming / marking / ... row made from a checked stop follows that stop's answer
            live = dec or (review.get(review_key(e["start"], e["camera"], "stop"))
                           if cat == "frequency" and note.startswith("checked by") else None)
            if live:
                who = live.get("checker") or "a person"
                ans = check_names.get((live["status"], live.get("label") or "")) if live.get("category") == "stop" \
                    else None
                ans = ans or {"confirmed": "correct", "rejected": "not this behavior"}.get(
                    live["status"], E.DISPLAY.get(live.get("label") or "", str(live.get("label")).replace("_", " ")))
                note = f"checked by {who}: {ans}"
                if live.get("saved", "") > built:
                    note += " - new answer, rebuild the report to update the sheet"
            rows.append({"idx": int(idx), "day": f"{e['start']:%d %b}", "time": f"{e['start']:%H:%M:%S}",
                         "stop": f"{e['stop']:%H:%M:%S}", "dur": _fmt_dur(e["duration_s"]), "name": name,
                         "room": e["room"], "camera": e["camera"], "conf": f"{float(e['confidence'] or 0):.2f}",
                         "cond": f"{str(e['light']).replace('_', ' ')}, {str(e['visibility']).replace('_', ' ')}",
                         "category": cat, "check": str(e["needs_review"]) == "True" and cat != "stop",
                         "note": note, "decision": decision,
                         "editable": cat in ("frequency", "duration", "removed") and bool(e["camera"]),
                         "clip": url_for("run_file", run=run, path=e["clip"]) if e.get("clip") else "",
                         "grid": url_for("run_file", run=run, path=e["grid"]) if e.get("grid") else "",
                         "start_iso": f"{e['start']:%Y-%m-%d %H:%M:%S}", "stop_iso": f"{e['stop']:%Y-%m-%d %H:%M:%S}",
                         "box": bx.get(idx)})
        labels = [(k, E.DISPLAY[k]) for k in E.FREQUENCY_LABELS + E.INACTIVE_LABELS + [E.WALKING, E.ACTIVE_OTHER]]
        filters = [("all", "All"), ("frequency", "Counted (feeding, drinking ...)"), ("duration", "Resting / walking"),
                   ("check", "Please check"), ("done", "Checked by you"), ("stop", "Short stops"),
                   ("not_visible", "Not visible"), ("removed", "Removed")]
        return render_template("events", events=rows, labels=labels, filters=filters, f=request.args.get("f", "all"))

    @app.route("/run/<run>/check")
    def check_page(run):
        items = check_items(run)
        ch = {k: [n, QUICK_STATUS[k][0], QUICK_STATUS[k][1]] for k, n, _ in QUICK_CHOICES}
        trained = "guess" in events(run) and events(run)["guess"].astype(str).str.len().gt(0).any()
        return render_template("check", items=items, choices=QUICK_CHOICES, ch=ch, trained=trained,
                               done=sum(1 for i in items if i["answer"]))

    @app.route("/run/<run>/quick")
    def quick_page(run):                       # old link
        return redirect(url_for("check_page", run=run))

    @app.route("/run/<run>/spotcheck", methods=["GET", "POST"])
    def spot_page(run):
        d = run_dir(run)
        if not os.path.exists(os.path.join(d, "spotcheck", "items.json")):
            return render_template("spot", ready=False, n=120, done=0, total=0)
        checker = request.args.get("checker", "")
        if request.method == "POST":
            picked = [r for r in request.form.getlist("room") if r != "none"]
            room = "+".join(picked) if picked else "none"
            label = request.form.get("label") or ("" if room == "none" else spotcheck.CANT_TELL)
            checker = request.form.get("checker", "")
            spotcheck.save_answer(d, int(request.form["id"]), room, label, checker)
            return redirect(url_for("spot_page", run=run, checker=checker))
        items, answers = spotcheck.load(d)
        want = request.args.get("id")
        todo = [it for it in items["items"] if (want is not None and it["id"] == int(want))
                or (want is None and str(it["id"]) not in answers)]
        item = todo[0] if todo else None
        labels = [(k, E.DISPLAY.get(k, "Can't tell")) for k in spotcheck.ANSWER_LABELS]
        return render_template("spot", ready=True, item=item, rooms=items["rooms"], labels=labels,
                               done=len(answers), total=len(items["items"]), checker=checker,
                               prev=answers.get(str(item["id"])) if item else None)

    @app.route("/run/<run>/spotcheck/results")
    def spot_results(run):
        summary, df, confusion = spotcheck.score(run_dir(run))
        fmt = {k: ((f"{v:.0%}" if v == v else "n/a") if isinstance(v, float) else v) for k, v in summary.items()}
        wrong = ""
        if len(df):
            bad = df[(~df["room_ok"]) | ((df["true_label"] != df["pred_label"])
                                                             & (df["true_label"] != spotcheck.CANT_TELL)
                                                             & (df["true_room"] != spotcheck.NO_CIVET))]
            bad = bad.assign(link=[f'<a href="{url_for("spot_page", run=run, id=i)}">look again</a>' for i in bad["id"]])
            wrong = bad.to_html(index=False, escape=False)
        return render_template("spot_results", summary=fmt, confusion=confusion.to_html() if len(confusion) else "",
                               wrong=wrong)

    @app.route("/run/<run>/files")
    def files_page(run):
        d = run_dir(run)
        files = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                n = os.path.getsize(p)
                files.append((f, FILE_INFO.get(f, ""), f"{n / 1e6:.1f} MB" if n > 1e5 else f"{n / 1e3:.0f} kB"))
        return render_template("files", files=files, folder=d)

    @app.route("/run/<run>/file/<path:path>")
    def run_file(run, path):
        return send_from_directory(run_dir(run), path)

    # --------------------------------------------------------------- video ---
    @app.route("/api/run/<run>/locate")
    def api_locate(run):
        """Which prepared video file shows camera `cam` at time t, and where in it."""
        try:
            rd = readers(run).get(request.args["cam"])
        except Exception as e:
            return jsonify(error=f"Videos not found ({e.__class__.__name__})"), 404
        if rd is None:
            return jsonify(error="Unknown camera"), 404
        t = pd.Timestamp(request.args["t"]).to_pydatetime()
        t2 = pd.Timestamp(request.args.get("t2") or request.args["t"]).to_pydatetime()
        f, idx = rd.locate(t)
        if f is None:
            return jsonify(error="No video at this time"), 404
        offset = idx / f["fps"]
        return jsonify(url=url_for("run_video", run=run, cam=request.args["cam"], name=os.path.basename(f["path"])),
                       offset=offset, length=max(0.5, (t2 - t).total_seconds()))

    @app.route("/run/<run>/video/<cam>/<name>")
    def run_video(run, cam, name):
        rd = readers(run).get(cam)
        if rd is None:
            abort(404)
        for f in rd.files:
            if os.path.basename(f["path"]) == name:
                return send_file(f["path"], mimetype="video/mp4", conditional=True)
        abort(404)

    # ------------------------------------------------------------- answers ---
    @app.route("/run/<run>/review", methods=["POST"])
    def save_review(run):
        d = run_dir(run)
        body = request.get_json(force=True)
        if body.get("status") not in ("confirmed", "rejected", "corrected"):
            abort(400)
        if body["status"] == "corrected" and body.get("label") not in E.FRAME_LABELS + ["toileting", "cant_see"]:
            abort(400)
        with _review_lock:                       # one answer at a time: fast clicking must not lose any
            review = _load_review(d)
            row = events(run).iloc[int(body["idx"])]
            review[review_key(row["start"], row["camera"], row["category"])] = {
                "status": body["status"], "label": body.get("label"), "start": str(row["start"]),
                "camera": str(row["camera"]), "behavior": str(row["behavior"]), "category": str(row["category"]),
                "checker": "a person",
                "saved": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
            tmp = os.path.join(d, "review.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(review, f, indent=1)
            os.replace(tmp, os.path.join(d, "review.json"))
        return jsonify(ok=True)

    # ---------------------------------------------------------------- jobs ---
    def _job_json(j):
        j = dict(j)
        j["url"] = url_for("job_page", job_id=j["id"])
        j["stage"] = j.get("status", {}).get("stage", "")
        j["state_txt"] = {"running": "running", "done": "finished", "failed": "stopped with an error",
                          "cancelled": "was stopped"}.get(j["state"], j["state"])
        j["log"] = jobs.log_tail(j, 14)
        j["day_url"] = url_for("run_page", run=j["day"]) if j.get("day") and j["day"] in _runs(root) else ""
        j["restart_url"] = url_for("api_restart", job_id=j["id"])
        return j

    @app.route("/api/job")
    def api_job():
        j = jobs.running()
        return jsonify(_job_json(j) if j else None)

    @app.route("/api/job/<job_id>")
    def api_job_one(job_id):
        return jsonify(_job_json(jobs.load(job_id)))

    @app.route("/api/job/<job_id>/cancel", methods=["POST"])
    def api_cancel(job_id):
        return jsonify(_job_json(jobs.cancel(job_id)))

    @app.route("/api/job/<job_id>/restart", methods=["POST"])
    def api_restart(job_id):
        j = jobs.load(job_id)
        return _start(j["kind"], j["title"], j["args"], j.get("day"), j.get("estimate_s"))

    def _start(kind, title, args, day, est=None):
        try:
            j = jobs.start(kind, title, args, day=day, estimate_s=est)
        except RuntimeError as e:
            return jsonify(error=str(e)), 409
        return jsonify(_job_json(jobs.load(j["id"])))

    def _cfg_args(run):
        info = run_info(run)
        return (["--config", info["config_rel"]] if info["config_rel"] else []), info["manifest_rel"]

    @app.route("/api/run/<run>/report", methods=["POST"])
    def api_start_report(run):
        c, man = _cfg_args(run)
        return _start("report", f"Rebuild report {run}",
                      [*c, "report", "--frames", f"runs/{run}/frames.csv", "--manifest", man, "--out", f"runs/{run}"],
                      run, jobs.speeds()["report_s"])

    @app.route("/api/run/<run>/train", methods=["POST"])
    def api_start_train(run):
        c, man = _cfg_args(run)
        return _start("train", f"Train behavior model ({run})",
                      [*c, "train-behavior", "--run", f"runs/{run}", "--manifest", man], run, 900)

    @app.route("/api/run/<run>/spotcheck", methods=["POST"])
    def api_start_spot(run):
        c, man = _cfg_args(run)
        return _start("spotcheck", f"Make spot check ({run})",
                      [*c, "spotcheck", "--run", f"runs/{run}", "--manifest", man], run, 300)

    def _folders(body):
        cams = {k: v for k, v in (body.get("cams") or {}).items() if v}
        src = (body.get("source") or "").strip().strip('"')
        if not cams and src:
            for name in sorted(os.listdir(src)) if os.path.isdir(src) else []:
                digits = "".join(ch for ch in name if ch.isdigit())
                if os.path.isdir(os.path.join(src, name)) and digits:
                    cams[f"cam{int(digits)}"] = os.path.join(src, name)
            if not cams:
                cams = jobs.group_videos_by_camera(src)
        return cams

    def _footage_metadata(cams):
        from .config import timestamp_from_filename
        starts = [timestamp_from_filename(f) for source in cams.values() for f in jobs.find_videos(source)]
        starts = [t for t in starts if t]
        date = f"{pd.Series(starts).dt.date.mode().iloc[0]:%Y-%m-%d}" if starts else ""
        return date, starts

    @app.route("/api/footage-metadata", methods=["POST"])
    def api_footage_metadata():
        cams = _folders(request.get_json(force=True))
        if not cams:
            return jsonify(error="No camera names found. Use camera sub-folders or file names like A01_20260405000000.mp4."), 400
        date, _ = _footage_metadata(cams)
        return jsonify(date=date, cameras={camera: {"files": len(jobs.find_videos(source))}
                                            for camera, source in cams.items()})

    @app.route("/api/estimate", methods=["POST"])
    def api_estimate():
        body = request.get_json(force=True)
        cams = _folders(body)
        if not cams:
            return jsonify(error="No camera folders found. Give a folder with 'camera 1' ... 'camera 4' inside, "
                                 "or each camera's folder."), 400
        est = jobs.estimate(cams)
        inferred_date, starts = _footage_metadata(cams)
        date = inferred_date or body.get("date") or ""
        free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        warn = []
        if est["camera_hours"] == 0:
            warn.append("No readable videos were found in these folders.")
        if not starts:
            warn.append("The file names have no date and time (like A01_20260405000000.mp4), so the app cannot tell when they were recorded.")
        if est["disk_gb"] > free - 5:
            warn.append(f"Not enough free disk space: about {est['disk_gb']} GB is needed, {free:.0f} GB is free.")
        if os.path.exists(os.path.join(root, date, "summary.json")):
            warn.append(f"{date} was already processed. Starting again keeps the finished hours and your answers.")
        from .process import latest_setup
        if not latest_setup(exclude=date) and not os.path.isdir(os.path.join(PROJECT_ROOT, "data", date, "zones")):
            warn.append("No zones from an earlier day: feeding / drinking need the zones (python -m civetwatch zones ...).")
        if jobs.running():
            warn.append("Another job is running: wait for it to finish first.")
        est.update(date=date, free_gb=round(free), warnings=warn,
                   can_start=bool(est["camera_hours"]) and bool(starts) and est["disk_gb"] <= free - 5 and not jobs.running())
        return jsonify(est)

    @app.route("/api/process", methods=["POST"])
    def api_start_process():
        body = request.get_json(force=True)
        cams = _folders(body)
        date = body.get("date")
        if not cams or not date:
            return jsonify(error="Folders and date are needed"), 400
        est = jobs.estimate(cams)
        args = ["process", "--date", date]
        source = (body.get("source") or "").strip().strip('"')
        if source and not (body.get("cams") or {}):
            args += ["--source", source]
        else:
            for k, v in cams.items():
                args += [f"--{k}", v]
        return _start("process", f"Process {date}", args, date, est["total_s"])

    @app.route("/api/upload/<date>/<cam>/<name>", methods=["PUT"])
    def api_upload(date, cam, name):
        """Streams the file straight to data/<date>/upload/<cam>/ (no temporary copy on drive C)."""
        if not cam.startswith("cam") or "/" in name or "\\" in name or ".." in name:
            abort(400)
        datetime.strptime(date, "%Y-%m-%d")
        dst = os.path.join(PROJECT_ROOT, "data", date, "upload", cam)
        os.makedirs(dst, exist_ok=True)
        part = os.path.join(dst, name + ".part")
        with open(part, "wb") as f:
            while True:
                chunk = request.stream.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(part, os.path.join(dst, name))
        return jsonify(ok=True)

    @app.route("/api/uploaded/<date>")
    def api_uploaded(date):
        base = os.path.join(PROJECT_ROOT, "data", date, "upload")
        cams = {c: os.path.join(base, c) for c in sorted(os.listdir(base))} if os.path.isdir(base) else {}
        return jsonify(cams=cams)

    @app.route("/api/run/<run>/remove", methods=["POST"])
    def api_remove_day(run):
        """Moves a day's results (and, if asked, its prepared videos) to the Recycle Bin."""
        d = run_dir(run)
        j = jobs.running()
        if j and j.get("day") == run:
            return jsonify(error="A job for this day is still running. Stop it first."), 409
        from send2trash import send2trash
        try:
            send2trash(d)
            videos = os.path.join(PROJECT_ROOT, "data", run)
            if (request.get_json(silent=True) or {}).get("videos") and os.path.isdir(videos):
                send2trash(videos)
        except OSError as e:
            return jsonify(error=f"Could not remove: {e}"), 500
        cache.clear()
        return jsonify(ok=True)

    @app.route("/api/run/<run>/open", methods=["POST"])
    def api_open_folder(run):
        d = run_dir(run)
        if hasattr(os, "startfile"):
            os.startfile(d)
        return jsonify(ok=True)

    return app


def serve(root, port, cfg):
    app = create_app(root, cfg)
    print(f"CivetWatch: open http://127.0.0.1:{port} in your browser  (days from {os.path.abspath(root)})")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
