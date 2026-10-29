"""Keeper dashboard: the day's numbers, charts, the observation sheet, and a
review list where keepers confirm or correct unsure events. Corrections are
saved to review.json and become training labels (`civetwatch export-labels`)."""
import json
import os
import threading

import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_from_directory, url_for

from . import ethogram as E
from . import spotcheck
from .report import fmt_hm

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CivetWatch · {{ run }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Source+Serif+4:wght@500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#f5efdf;--panel:#fdf9ef;--soft:#e4e9d4;--ink:#1d2b22;--ink2:#52574c;--muted:#7a8a6a;--green:#183a2e;
--olive:#6e8452;--orange:#c8693a;--line:#c9d3ae;--warn-bg:#f7e3d4}
@media (prefers-color-scheme:dark){:root{--bg:#111a15;--panel:#18241d;--soft:#1f2e25;--ink:#eef0e6;--ink2:#c3c8b8;
--muted:#93a383;--green:#0c1712;--line:#2f4236;--warn-bg:#3a2519}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Lato,system-ui,sans-serif}
header{background:var(--green);color:#f5efdf;padding:20px 24px}
header .kicker{letter-spacing:.14em;font-size:12px;color:#a7b87a;font-weight:700;text-transform:uppercase}
header h1{font:600 30px/1.2 "Source Serif 4",Georgia,serif;margin:4px 0 0}
header select{margin-top:10px;background:#234a3b;color:#f5efdf;border:1px solid #3f6b56;border-radius:6px;padding:4px 8px;font:inherit}
main{max-width:1200px;margin:0 auto;padding:20px 16px 60px}
h2{font:600 22px "Source Serif 4",Georgia,serif;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--panel);padding:16px 18px;border-radius:8px}
.tile .v{font:600 34px/1.1 "Source Serif 4",Georgia,serif}.tile .l{color:var(--ink2);font-size:13px}
.tile.o .v{color:var(--orange)}.tile.g .v{color:var(--olive)}
.alerts{margin-top:14px;display:grid;gap:6px}.alert{background:var(--warn-bg);border-left:4px solid var(--orange);padding:8px 12px;border-radius:4px}
.downloads a{display:inline-block;margin:0 8px 8px 0;padding:6px 12px;border-radius:6px;background:var(--green);color:#f5efdf;text-decoration:none;font-weight:700;font-size:13px}
img.chart{width:100%;border-radius:8px;background:#fcfcfb}
.review{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--panel);border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.card video,.card .novideo,.card img.grid{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:contain}
.card .novideo{display:flex;align-items:center;justify-content:center;color:#aaa;font-size:13px}
.card .b{padding:10px 12px;display:grid;gap:6px}.card .t{font-weight:700}.card .m{color:var(--ink2);font-size:13px}
.card .act{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button,select.lab{font:inherit;font-size:13px;border-radius:6px;border:1px solid var(--line);background:var(--soft);color:var(--ink);padding:4px 10px;cursor:pointer}
button.ok{background:var(--olive);color:#fff;border-color:var(--olive)}button.no{background:var(--orange);color:#fff;border-color:var(--orange)}
.status{font-size:12px;font-weight:700;color:var(--olive)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.filters button.on{background:var(--green);color:#f5efdf}
.tablewrap{overflow-x:auto;background:var(--panel);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--soft)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;background:var(--soft)}
.pill.check{background:var(--warn-bg);color:var(--orange);font-weight:700}
footer{color:var(--muted);font-size:12px;margin-top:40px}
.todo{background:var(--panel);border-radius:8px;padding:12px 16px 12px 36px;margin:0}.todo li{margin:6px 0}
.todo a{color:var(--orange)}code{font-size:12px;background:var(--soft);padding:1px 4px;border-radius:4px;word-break:break-all}
.gloss{background:var(--panel);border-radius:8px;padding:10px 16px;margin-top:12px}.gloss summary{cursor:pointer}
.gloss dt{font-weight:700;margin-top:8px}.gloss dd{margin:0;color:var(--ink2)}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;overflow:auto;cursor:zoom-in}
#lb.open{display:block}#lb img{display:block;margin:auto;max-width:100vw;max-height:100vh;object-fit:contain}
#lb.big{cursor:zoom-out}#lb.big img{max-width:none;max-height:none;width:250vw}
#lb .x{position:fixed;top:10px;right:16px;color:#fff;font:600 15px system-ui,sans-serif;background:rgba(0,0,0,.6);
padding:6px 12px;border-radius:6px;cursor:pointer}img.zoomable{cursor:zoom-in}
</style></head><body>
<div id="lb" onclick="lbClick(event)"><span class="x" onclick="lbClose(event)">Esc · close ✕</span><img alt=""></div>
<header><div class="kicker">Chiang Mai Zoo · Smart Zoo project</div>
<h1>CivetWatch · {{ s.civet }} · {{ s.day }}</h1>
<select onchange="location.href=this.value">{% for r in runs %}<option value="{{ url_for('run_page', run=r) }}" {{ 'selected' if r==run }}>{{ r }}</option>{% endfor %}</select>
</header>
<main>
<div class="tiles">
 <div class="tile"><div class="v">{{ s.active_hm }}</div><div class="l">active {{ s.active_pct }} ({{ s.inactive_hm }} inactive)</div></div>
 <div class="tile o"><div class="v">{{ s.feeding }}</div><div class="l">feeding visits</div></div>
 <div class="tile g"><div class="v">{{ s.to_review }}</div><div class="l">things to check by a person</div></div>
 <div class="tile"><div class="v">{{ s.observed_hm }}</div><div class="l">observed · {{ s.cameras }} cameras · {{ s.night_pct }} night IR</div></div>
</div>
{% if s.alerts %}<div class="alerts">{% for a in s.alerts %}<div class="alert">{{ a }}</div>{% endfor %}</div>{% endif %}

<h2>What to do</h2>
<ol class="todo">
 <li><a href="{{ url_for('quick_page', run=run) }}"><b>Quick check the short stops</b></a> ({{ stops_left }} of {{ stops_total }} left) -
  pictures of every moment the civet stood still between walking. You pick: urinating, defecating, scent marking,
  grooming or nothing. Only what you pick is counted as these behaviors.</li>
 <li><b>Check the feeding / drinking cards</b> below ("Please check") - the app was unsure about these.</li>
 <li><b>Rebuild the report</b> so your answers appear in the sheet and charts:<br>
  <code>python -m civetwatch --config data/{{ run }}/config.yaml report --frames runs/{{ run }}/frames.csv --manifest data/{{ run }}/manifest.yaml --out runs/{{ run }}</code></li>
 <li>{% if spot %}<a href="{{ url_for('spot_page', run=run) }}"><b>Spot check</b></a>: {{ spot }}{% else %}<b>Spot check</b> (optional):
  <code>python -m civetwatch spotcheck --run runs/{{ run }} --manifest data/{{ run }}/manifest.yaml</code>{% endif %}
  - random moments you answer without seeing the app's answer; gives the real accuracy.</li>
</ol>
<details class="gloss"><summary><b>What the words mean</b> (Not visible, visits, short stop, ...)</summary>
<dl>{% for w, m in glossary %}<dt>{{ w }}</dt><dd>{{ m }}</dd>{% endfor %}</dl></details>

<h2>Output files</h2>
<div class="downloads">{% for f in files %}<a href="{{ url_for('run_file', run=run, path=f) }}" download>{{ f }}</a>{% endfor %}</div>

<h2>Charts</h2>
<img class="chart" src="{{ url_for('run_file', run=run, path='charts.png') }}" alt="Timeline, time per room, hourly counts and time budget">

<h2>Please check ({{ review|length }})</h2>
<p class="m">Events the app was unsure about. Short stops are not listed here - use Quick check above.</p>
{% if not review %}<p class="m">Nothing to review.</p>{% endif %}
<div class="review">
{% for e in review %}<div class="card" id="ev{{ e.idx }}">
 {% if e.clip %}<video src="{{ url_for('run_file', run=run, path=e.clip) }}" controls preload="metadata" muted></video>
 {% elif e.grid %}<img class="grid zoomable" title="click to enlarge" src="{{ url_for('run_file', run=run, path=e.grid) }}" alt="4 frames of the stop" loading="lazy">
 {% else %}<div class="novideo">no clip</div>{% endif %}
 <div class="b"><div class="t">{{ e.name }} · {{ e.time }}</div>
 <div class="m">{{ e.room }} · {{ e.camera }} · {{ e.dur }} · confidence {{ e.conf }} · {{ e.cond }}</div>
 {% if e.note %}<div class="m">{{ e.note }}{% if e.vlm %}: “{{ e.vlm }}”{% endif %}</div>{% endif %}
 {% if e.grid %}<a class="m zoomable" href="{{ url_for('run_file', run=run, path=e.grid) }}">4 frames of the stop (enlarge)</a>{% endif %}
 <div class="act">
  <button class="ok" onclick="decide({{ e.idx }},'confirmed')">Correct</button>
  <button class="no" onclick="decide({{ e.idx }},'rejected')">Not this behavior</button>
  <select class="lab" onchange="decide({{ e.idx }},'corrected',this.value)"><option value="">It was …</option>
   {% for k,v in labels %}<option value="{{ k }}">{{ v }}</option>{% endfor %}</select>
  <span class="status" id="st{{ e.idx }}">{{ e.decision }}</span></div></div></div>{% endfor %}
</div>

<h2>All events</h2>
<div class="filters" id="filters">
 <button class="on" data-f="all">All</button><button data-f="frequency">Counts</button>
 <button data-f="duration">Durations</button><button data-f="stop">Short stops</button><button data-f="not_visible">Not visible</button><button data-f="check">Please check</button></div>
<div class="tablewrap"><table id="events"><thead><tr><th>Start</th><th>Stop</th><th>Minutes</th><th>Behavior</th><th>Room</th>
<th>Confidence</th><th>Condition</th><th></th></tr></thead><tbody>
{% for e in events %}<tr data-cat="{{ e.category }}" data-check="{{ 1 if e.check else 0 }}"><td>{{ e.start }}</td><td>{{ e.stop }}</td>
<td class="n">{{ e.minutes }}</td><td>{{ e.name }}</td><td>{{ e.room }}</td><td class="n">{{ e.conf }}</td><td>{{ e.cond }}</td>
<td>{% if e.check %}<span class="pill check">please check</span>{% endif %}{% if e.clip %} <a href="{{ url_for('run_file', run=run, path=e.clip) }}">clip</a>{% endif %}</td></tr>{% endfor %}
</tbody></table></div>
<footer>Models: {{ s.models }} · every number links back to the video moment it came from (frames.csv).</footer>
</main>
<script>
function lbOpen(src){const lb=document.getElementById('lb');lb.querySelector('img').src=src;lb.className='open';}
function lbClose(e){if(e)e.stopPropagation();document.getElementById('lb').className='';}
function lbClick(e){const lb=document.getElementById('lb');if(e.target.tagName!=='IMG'){lbClose();return;}
  const r=e.target.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;
  lb.classList.toggle('big');
  if(lb.classList.contains('big')){const i=lb.querySelector('img');lb.scrollLeft=fx*i.width-innerWidth/2;lb.scrollTop=fy*i.height-innerHeight/2;}}
document.addEventListener('keydown',e=>{if(e.key==='Escape')lbClose();});
document.addEventListener('click',e=>{const t=e.target.closest('img.zoomable,a.zoomable');
  if(t){e.preventDefault();lbOpen(t.tagName==='A'?t.href:t.src);}});
</script>
<script>
async function decide(idx,status,label){
  if(status==='corrected'&&!label)return;
  const r=await fetch("{{ url_for('save_review', run=run) }}",{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({idx,status,label:label||null})});
  document.getElementById('st'+idx).textContent=r.ok?(status==='corrected'?'saved: '+label.replaceAll('_',' '):'saved: '+status):'save failed';
}
document.querySelectorAll('#filters button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('on',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('#events tbody tr').forEach(tr=>{
    tr.style.display=(f==='all'||tr.dataset.cat===f||(f==='check'&&tr.dataset.check==='1'))?'':'none';});
});
</script></body></html>"""


QUICK_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CivetWatch quick check</title>
<style>
:root{--bg:#f5efdf;--panel:#fdf9ef;--soft:#e4e9d4;--ink:#1d2b22;--ink2:#52574c;--green:#183a2e;--olive:#6e8452;--orange:#c8693a;--line:#c9d3ae}
@media (prefers-color-scheme:dark){:root{--bg:#111a15;--panel:#18241d;--soft:#1f2e25;--ink:#eef0e6;--ink2:#c3c8b8;--line:#2f4236}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{background:var(--green);color:#f5efdf;padding:12px 20px;position:sticky;top:0;z-index:5}
header a{color:#a7b87a}header b{font-size:17px}
.keys{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:6px;font-size:13px}
.keys span b{display:inline-block;min-width:20px;text-align:center;background:#f5efdf;color:#183a2e;border-radius:4px;padding:0 5px;margin-right:4px}
main{max-width:1300px;margin:0 auto;padding:14px 16px 60px}
.help{background:var(--panel);border-radius:8px;padding:10px 14px;font-size:14px;color:var(--ink2)}
.help dt{font-weight:700;color:var(--ink);margin-top:4px}.help dd{margin:0 0 2px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin-top:14px}
.t{background:var(--panel);border-radius:8px;overflow:hidden;border:3px solid transparent}
.t.cur{border-color:var(--orange)}.t.done{opacity:.55}.t img{width:100%;display:block;cursor:zoom-in;background:#000}
.t .c{padding:6px 10px;font-size:13px}.t .a{font-weight:700;color:var(--olive)}
.t .b{display:flex;flex-wrap:wrap;gap:4px;padding:0 10px 10px}
.t .b button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--soft);color:var(--ink);border-radius:5px;padding:2px 7px;cursor:pointer}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;overflow:auto;cursor:zoom-in}
#lb.open{display:block}#lb img{display:block;margin:auto;max-width:100vw;max-height:100vh;object-fit:contain}
#lb.big{cursor:zoom-out}#lb.big img{max-width:none;max-height:none;width:250vw}
#lb .x{position:fixed;top:10px;right:16px;color:#fff;background:rgba(0,0,0,.6);padding:6px 12px;border-radius:6px;cursor:pointer}
</style></head><body>
<div id="lb" onclick="lbClick(event)"><span class="x" onclick="lbClose(event)">Esc · close ✕</span><img alt=""></div>
<header><b>Quick check · {{ run }}</b> · <span id="cnt">{{ done }}</span> / {{ items|length }} checked ·
<a href="{{ url_for('run_page', run=run) }}">back to the day</a> ·
<a href="#" onclick="toggleDone();return false" id="tg">hide checked</a>
<div class="keys">{% for k, name, _ in choices %}<span><b>{{ k }}</b>{{ name }}</span>{% endfor %}
<span><b>←→</b>move</span><span><b>Space</b>enlarge</span></div></header>
<main>
<div class="help"><b>How to check.</b> Each picture shows 4 moments of one short stop (left to right, top to bottom, a few
seconds apart) where the civet stood still between walking. They are sorted so the ones that look most like
toileting come first - only a hint from a weak model, so please go through all of them. Pick what it was doing; the app counts it in the
observation sheet when you rebuild the report. Click a picture (or press Space) to enlarge it, click again to zoom in.
<dl>{% for k, name, what in choices %}<dt>{{ k }} · {{ name }}</dt><dd>{{ what }}</dd>{% endfor %}</dl></div>
<div class="grid" id="g">
{% for it in items %}<div class="t {{ 'done' if it.answer }}" data-idx="{{ it.idx }}" tabindex="0">
<img src="{{ url_for('run_file', run=run, path=it.grid) }}" alt="4 frames of the stop at {{ it.time }}" loading="lazy">
<div class="c">{{ it.time }} · {{ it.room }} · stood still {{ it.dur }} s · <span class="a">{{ it.answer }}</span></div>
<div class="b">{% for k, name, _ in choices %}<button data-k="{{ k }}">{{ k }} {{ name }}</button>{% endfor %}</div></div>
{% endfor %}</div></main>
<script>
const CH={ {% for k, name, _ in choices %}"{{ k }}":["{{ name }}","{{ statuses[k][0] }}","{{ statuses[k][1] }}"],{% endfor %} };
const tiles=[...document.querySelectorAll('.t')];let cur=tiles.findIndex(t=>!t.classList.contains('done'));if(cur<0)cur=0;
function focusTile(i){if(!tiles.length)return;cur=Math.max(0,Math.min(tiles.length-1,i));tiles.forEach(t=>t.classList.remove('cur'));
  tiles[cur].classList.add('cur');tiles[cur].scrollIntoView({block:'nearest',behavior:'smooth'});}
async function answer(t,k){const [name,status,label]=CH[k];
  const r=await fetch("{{ url_for('save_review', run=run) }}",{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({idx:+t.dataset.idx,status,label:label||null})});
  if(!r.ok){t.querySelector('.a').textContent='save failed';return;}
  if(!t.classList.contains('done')){const c=document.getElementById('cnt');c.textContent=+c.textContent+1;}
  t.classList.add('done');t.querySelector('.a').textContent=name;}
function lbOpen(src){const lb=document.getElementById('lb');lb.querySelector('img').src=src;lb.className='open';}
function lbClose(e){if(e)e.stopPropagation();document.getElementById('lb').className='';}
function lbClick(e){const lb=document.getElementById('lb');if(e.target.tagName!=='IMG'){lbClose();return;}
  const r=e.target.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;lb.classList.toggle('big');
  if(lb.classList.contains('big')){const i=lb.querySelector('img');lb.scrollLeft=fx*i.width-innerWidth/2;lb.scrollTop=fy*i.height-innerHeight/2;}}
let hide=false;function toggleDone(){hide=!hide;document.getElementById('tg').textContent=hide?'show checked':'hide checked';
  tiles.forEach(t=>t.style.display=hide&&t.classList.contains('done')?'none':'');}
tiles.forEach((t,i)=>{t.querySelector('img').onclick=()=>{focusTile(i);lbOpen(t.querySelector('img').src);};
  t.querySelectorAll('button').forEach(b=>b.onclick=()=>{focusTile(i);answer(t,b.dataset.k);});});
document.addEventListener('keydown',async e=>{const lbOpenNow=document.getElementById('lb').className.includes('open');
  if(e.key==='Escape'){lbClose();return;}
  if(e.key===' '||e.code==='Space'){e.preventDefault();if(lbOpenNow)lbClose();else lbOpen(tiles[cur].querySelector('img').src);return;}
  if(e.key==='ArrowRight'){focusTile(cur+1);return;} if(e.key==='ArrowLeft'){focusTile(cur-1);return;}
  if(CH[e.key]){lbClose();const t=tiles[cur];let n=cur+1;           // move on at once, save in the background
    while(hide&&n<tiles.length&&tiles[n].classList.contains('done'))n++;focusTile(n);answer(t,e.key);}});
focusTile(cur);
</script></body></html>"""

QUICK_CHOICES = [
    ("1", "Urinating", "Stands or half-squats with the hindquarters lowered and the tail lifted; usually a few seconds."),
    ("2", "Defecating", "Deep squat, back hunched, tail raised, held still longer (often 5 s or more)."),
    ("3", "Toileting, not sure which", "Clearly urinating or defecating but you cannot tell which: the app decides by length (5 s or more = defecating)."),
    ("4", "Scent marking", "Presses or rubs its rear end (scent gland under the tail) on a log, wall, door frame or fence."),
    ("5", "Grooming", "Licks, nibbles or scratches its own fur; head turned back to its body or a hind leg scratching."),
    ("0", "Nothing special", "Standing, sniffing, looking around, sitting or turning - not one of the behaviors above."),
]
QUICK_STATUS = {"1": ("corrected", "urination"), "2": ("corrected", "defecation"), "3": ("corrected", "toileting"),
                "4": ("corrected", "territorial_marking"), "5": ("corrected", "grooming"), "0": ("rejected", "")}


SPOT_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CivetWatch spot check</title>
<style>
:root{--bg:#f5efdf;--panel:#fdf9ef;--soft:#e4e9d4;--ink:#1d2b22;--ink2:#52574c;--green:#183a2e;--olive:#6e8452;--orange:#c8693a;--line:#c9d3ae}
@media (prefers-color-scheme:dark){:root{--bg:#111a15;--panel:#18241d;--soft:#1f2e25;--ink:#eef0e6;--ink2:#c3c8b8;--line:#2f4236}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{background:var(--green);color:#f5efdf;padding:14px 20px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header a{color:#a7b87a}main{max-width:1100px;margin:0 auto;padding:16px}
img{width:100%;border-radius:6px;background:#000}
.q{background:var(--panel);border-radius:8px;padding:12px 14px;margin-top:12px}
.q h3{margin:0 0 8px;font-size:15px}.opts{display:flex;flex-wrap:wrap;gap:6px}
.opts label{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
.opts input{margin-right:5px}button{font:inherit;background:var(--orange);color:#fff;border:0;border-radius:6px;padding:8px 18px;margin-top:12px;cursor:pointer}
.hint{color:var(--ink2);font-size:13px}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;overflow:auto;cursor:zoom-in}
#lb.open{display:block}#lb img{display:block;margin:auto;max-width:100vw;max-height:100vh;object-fit:contain}
#lb.big{cursor:zoom-out}#lb.big img{max-width:none;max-height:none;width:250vw}
#lb .x{position:fixed;top:10px;right:16px;color:#fff;font:600 15px system-ui,sans-serif;background:rgba(0,0,0,.6);
padding:6px 12px;border-radius:6px;cursor:pointer}img.zoomable{cursor:zoom-in}
</style></head><body>
<div id="lb" onclick="lbClick(event)"><span class="x" onclick="lbClose(event)">Esc · close ✕</span><img alt=""></div>
<header><b>Spot check &middot; {{ run }}</b><span>{{ done }} / {{ total }} checked</span>
<a href="{{ url_for('spot_results', run=run) }}">results</a><a href="{{ url_for('run_page', run=run) }}">back to the day</a></header>
<main>
{% if item %}
<p class="hint">Moment {{ item.id + 1 }} &middot; {{ item.t[11:19] }} &middot; rows = cameras/rooms, columns = 1 s before, the moment (t), 1 s after.
CivetWatch's answer stays hidden so it cannot influence you.</p>
<img class="zoomable" title="click to enlarge" src="{{ url_for('run_file', run=run, path=item.image) }}" alt="All cameras around {{ item.t }}">
<form method="post">
<input type="hidden" name="id" value="{{ item.id }}">
<div class="q"><h3>1 &middot; Where is the civet at t?</h3><div class="opts">
{% for r in rooms %}<label><input type="radio" name="room" value="{{ r }}" required {{ 'checked' if prev and prev.room==r }}>{{ r }}</label>{% endfor %}
<label><input type="radio" name="room" value="none" {{ 'checked' if prev and prev.room=='none' }}>Not visible in any camera</label></div></div>
<div class="q"><h3>2 &middot; What is it doing? (skip if not visible)</h3><div class="opts">
{% for k, v in labels %}<label><input type="radio" name="label" value="{{ k }}" {{ 'checked' if prev and prev.label==k }}>{{ v }}</label>{% endfor %}</div></div>
<input type="text" name="checker" placeholder="your name" value="{{ checker }}" style="margin-top:12px;padding:7px;border-radius:6px;border:1px solid var(--line)">
<button>Save and next</button>
</form>
{% else %}<p>All moments are checked. <a href="{{ url_for('spot_results', run=run) }}">See the results.</a></p>{% endif %}
</main><script>
function lbOpen(src){const lb=document.getElementById('lb');lb.querySelector('img').src=src;lb.className='open';}
function lbClose(e){if(e)e.stopPropagation();document.getElementById('lb').className='';}
function lbClick(e){const lb=document.getElementById('lb');if(e.target.tagName!=='IMG'){lbClose();return;}
  const r=e.target.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;
  lb.classList.toggle('big');
  if(lb.classList.contains('big')){const i=lb.querySelector('img');lb.scrollLeft=fx*i.width-innerWidth/2;lb.scrollTop=fy*i.height-innerHeight/2;}}
document.addEventListener('keydown',e=>{if(e.key==='Escape')lbClose();});
document.addEventListener('click',e=>{const t=e.target.closest('img.zoomable,a.zoomable');
  if(t){e.preventDefault();lbOpen(t.tagName==='A'?t.href:t.src);}});
</script></body></html>"""

SPOT_RESULTS = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spot check results</title><style>body{font:15px/1.5 system-ui,sans-serif;background:#f5efdf;color:#1d2b22;margin:0}
@media (prefers-color-scheme:dark){body{background:#111a15;color:#eef0e6}}
main{max-width:1100px;margin:0 auto;padding:16px;overflow-x:auto}table{border-collapse:collapse;margin:8px 0 20px}
td,th{padding:5px 10px;border-bottom:1px solid #c9d3ae;text-align:left}a{color:#c8693a}</style></head><body><main>
<h2>Spot check &middot; {{ run }}</h2><p><a href="{{ url_for('spot_page', run=run) }}">continue checking</a> &middot; <a href="{{ url_for('run_page', run=run) }}">back to the day</a></p>
<table>{% for k, v in summary.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>
<h3>Behavior: checker vs CivetWatch</h3>{{ confusion|safe }}
<h3>Disagreements</h3>{{ wrong|safe }}
</main></body></html>"""


def _runs(root):
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if os.path.exists(os.path.join(root, name, "summary.json")):
            out.append(name)
    return sorted(out, key=lambda n: os.path.getmtime(os.path.join(root, n, "summary.json")), reverse=True)


def create_app(root, cfg):
    app = Flask(__name__)
    root = os.path.abspath(root)

    def run_dir(run):
        d = os.path.join(root, run)
        if run not in _runs(root):
            abort(404)
        return d

    @app.route("/")
    def index():
        runs = _runs(root)
        if not runs:
            return f"No runs in {root} yet. Run <code>python -m civetwatch demo</code> first."
        return redirect(url_for("run_page", run=runs[0]))

    @app.route("/run/<run>")
    def run_page(run):
        d = run_dir(run)
        s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
        ev = pd.read_csv(os.path.join(d, "events.csv"), parse_dates=["start", "stop"], keep_default_na=False)
        review = _load_review(d)
        by_event = {(v["start"][:19], v["camera"]): v for v in review.values() if v.get("start")}
        cond = lambda e: f"{e['light'].replace('_', ' ')}, {e['visibility'].replace('_', ' ')}"
        rows, cards = [], []
        for idx, e in ev.iterrows():
            check = str(e["needs_review"]) == "True"
            row = {"idx": idx, "start": f"{e['start']:%H:%M:%S}", "stop": f"{e['stop']:%H:%M:%S}",
                   "minutes": f"{float(e['duration_s']) / 60:.1f}", "name": E.DISPLAY.get(e["behavior"], e["behavior"])
                   + (f" ({e['note']})" if e.get("note") and e["category"] != "stop" else ""),
                   "note": e.get("note", "") if e["category"] == "stop" else "", "room": e["room"], "camera": e["camera"],
                   "conf": f"{float(e['confidence']):.2f}", "cond": cond(e), "category": e["category"],
                   "check": check, "clip": e.get("clip", ""), "time": f"{e['start']:%H:%M:%S}",
                   "dur": f"{float(e['duration_s']):.0f} s",
                   "grid": e.get("grid", ""), "vlm": str(e.get("vlm_answer", ""))[:70]}
            rows.append(row)
            if check and e["category"] != "stop":
                dec = by_event.get((f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"])) or review.get(str(idx))
                row["decision"] = "" if not dec else ("saved: " + (dec["label"] or dec["status"]).replace("_", " "))
                cards.append(row)
        files = [f for f in ("sheet.xlsx", "events.csv", "frames.csv", "charts.png", "summary.json")
                 if os.path.exists(os.path.join(d, f))]
        start = pd.Timestamp(s["start"])
        view = {
            "civet": s.get("civet", "Civet"), "day": f"{start:%d %b %Y}",
            "active_hm": fmt_hm(s["active_s"]), "inactive_hm": fmt_hm(s["inactive_s"]),
            "active_pct": f"{s['active_frac']:.0%}", "observed_hm": fmt_hm(s["observed_s"]),
            "feeding": s["counts"].get(E.FEEDING, 0), "to_review": s["to_review"],
            "cameras": len(s.get("models", {}).get("cameras", [])) or "?",
            "night_pct": f"{s['night_frac']:.0%}", "alerts": s.get("alerts", []),
            "models": ", ".join(f"{k}={v}" for k, v in s.get("models", {}).items() if k != "cameras"),
        }
        labels = [(k, E.DISPLAY[k]) for k in E.FREQUENCY_LABELS + E.INACTIVE_LABELS + [E.WALKING, E.ACTIVE_OTHER]]
        spot = None
        if os.path.exists(os.path.join(d, "spotcheck", "items.json")):
            items, answers = spotcheck.load(d)
            spot = f"{len(answers)} of {len(items['items'])} moments checked"
        stop_ev = ev[(ev["category"] == "stop") & (ev["behavior"] != "doorway_pass")]
        stops_done = sum(1 for _, e in stop_ev.iterrows()
                         if (f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"]) in by_event)
        from .glossary import GLOSSARY
        return render_template_string(PAGE, run=run, runs=_runs(root), s=view, files=files, review=cards,
                                      stops_total=len(stop_ev), stops_left=len(stop_ev) - stops_done, glossary=GLOSSARY,
                                      events=rows, labels=labels, spot=spot)

    @app.route("/run/<run>/file/<path:path>")
    def run_file(run, path):
        return send_from_directory(run_dir(run), path)

    @app.route("/run/<run>/review", methods=["POST"])
    def save_review(run):
        d = run_dir(run)
        body = request.get_json(force=True)
        if body.get("status") not in ("confirmed", "rejected", "corrected"):
            abort(400)
        if body["status"] == "corrected" and body.get("label") not in E.FRAME_LABELS + ["toileting"]:
            abort(400)
        with _review_lock:                       # one answer at a time: fast clicking must not lose any
            return _save_decision(d, body)

    def _save_decision(d, body):
        review = _load_review(d)
        ev = pd.read_csv(os.path.join(d, "events.csv"), keep_default_na=False)
        row = ev.iloc[int(body["idx"])]
        review[str(int(body["idx"]))] = {"status": body["status"], "label": body.get("label"),
                                         "start": str(row["start"]), "camera": str(row["camera"]),
                                         "behavior": str(row["behavior"])}
        with open(os.path.join(d, "review.json"), "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)
        return jsonify(ok=True)

    @app.route("/run/<run>/quick")
    def quick_page(run):
        d = run_dir(run)
        ev = pd.read_csv(os.path.join(d, "events.csv"), parse_dates=["start"], keep_default_na=False)
        review = _load_review(d)
        by_event = {(v["start"][:19], v["camera"]): v for v in review.values() if v.get("start")}
        name = {QUICK_STATUS[k]: n for k, n, _ in QUICK_CHOICES}
        items = []
        for idx, e in ev.iterrows():
            if e["category"] != "stop" or e["behavior"] == "doorway_pass" or not e.get("grid"):
                continue
            dec = by_event.get((f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"]))
            answer = ""
            if dec:
                answer = name.get((dec["status"], dec.get("label") or ""), dec.get("label") or dec["status"])
            items.append({"idx": idx, "grid": e["grid"], "time": f"{e['start']:%H:%M:%S}", "room": e["room"],
                          "dur": f"{float(e['duration_s']):.0f}", "answer": answer,
                          "score": float(e.get("toilet_score") or 0)})
        items.sort(key=lambda i: -i["score"])        # likely toileting first (a hint only - check them all)
        return render_template_string(QUICK_PAGE, run=run, items=items, choices=QUICK_CHOICES, statuses=QUICK_STATUS,
                                      done=sum(1 for i in items if i["answer"]))

    @app.route("/run/<run>/spotcheck", methods=["GET", "POST"])
    def spot_page(run):
        d = run_dir(run)
        if not os.path.exists(os.path.join(d, "spotcheck", "items.json")):
            return f"No spot check yet. Run: python -m civetwatch spotcheck --run runs/{run} --manifest ..."
        checker = request.args.get("checker", "")
        if request.method == "POST":
            room = request.form.get("room", "none")
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
        return render_template_string(SPOT_PAGE, run=run, item=item, rooms=items["rooms"], labels=labels,
                                      done=len(answers), total=len(items["items"]), checker=checker,
                                      prev=answers.get(str(item["id"])) if item else None)

    @app.route("/run/<run>/spotcheck/results")
    def spot_results(run):
        summary, df, confusion = spotcheck.score(run_dir(run))
        fmt = {k: ((f"{v:.0%}" if v == v else "n/a") if isinstance(v, float) else v) for k, v in summary.items()}
        wrong = ""
        if len(df):
            bad = df[(df["true_room"] != df["pred_room"]) | ((df["true_label"] != df["pred_label"])
                                                             & (df["true_label"] != spotcheck.CANT_TELL)
                                                             & (df["true_room"] != spotcheck.NO_CIVET))]
            bad = bad.assign(link=[f'<a href="{url_for("spot_page", run=run, id=i)}">look again</a>' for i in bad["id"]])
            wrong = bad.to_html(index=False, escape=False)
        return render_template_string(SPOT_RESULTS, run=run, summary=fmt,
                                      confusion=confusion.to_html() if len(confusion) else "", wrong=wrong)

    return app


_review_lock = threading.Lock()


def _load_review(d):
    p = os.path.join(d, "review.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def serve(root, port, cfg):
    app = create_app(root, cfg)
    print(f"CivetWatch dashboard: http://127.0.0.1:{port}  (runs from {os.path.abspath(root)})")
    app.run(host="127.0.0.1", port=port, debug=False)
