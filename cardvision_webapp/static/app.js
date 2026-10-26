const $ = id => document.getElementById(id);
const ranks = ['UNKNOWN', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
let state = {}, events = [], resultId = null, dirty = false;
let regions = [], previewImage = null, dragging = null;
const busyStates = ['STARTING', 'RECORDING', 'STOPPING', 'PROCESSING'];
function error(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function api(path, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch (_) { throw new Error(`Request failed (${response.status})`); }
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function changed() { dirty = true; $('confirmReview').checked = false; renderCount(); }
function renderCount() {
  const included = events.filter(e => !e.excluded);
  const unknown = included.filter(e => e.rank === 'UNKNOWN').length;
  const subtotal = included.reduce((total,e) => total + (['A','10','J','Q','K'].includes(e.rank) ? -1 : ['2','3','4','5','6'].includes(e.rank) ? 1 : 0),0);
  const reviewed = !dirty && state.count_verified;
  $('countChange').textContent = resultId ? (subtotal > 0 ? `+${subtotal}` : `${subtotal}`) : '—';
  $('countHeading').textContent = resultId && unknown ? 'Known-card subtotal' : 'Round count change';
  $('countBadge').textContent = !resultId ? (state.session_status === 'PROCESSING' ? 'Analyzing' : 'Waiting') : unknown ? 'Incomplete' : dirty ? 'Unsaved edits' : reviewed ? 'Reviewed by you' : 'Provisional';
  $('countBadge').dataset.status = reviewed ? 'reviewed' : resultId ? 'provisional' : 'waiting';
  $('countStatus').textContent = !resultId ? (state.session_status === 'PROCESSING' ? 'Calculating the count from your saved screenshots…' : 'The count is calculated after you stop recording and analyze the round.') : unknown ? `${unknown} unknown card${unknown === 1 ? '' : 's'} — the full round count is still unknown.${dirty ? ' Edits are not saved.' : ''}` : dirty ? 'Count reflects your edits. Save the review to keep them.' : reviewed ? `${included.length} included cards. You confirmed this card list.` : `${included.length} included cards. Check the card list before trusting this count.`;
}
function renderCards() {
  $('cards').replaceChildren();
  if (!events.length) { const p = document.createElement('p'); p.className = 'empty-state'; p.textContent = 'No cards found. Check the selected regions and the full round.'; $('cards').append(p); }
  events.forEach((event,index) => {
    const card = document.createElement('article'); card.className = 'card-row' + (event.excluded ? ' excluded' : '');
    const head = document.createElement('div'); head.className = 'card-heading';
    const title = document.createElement('span'); title.textContent = `${index+1}. ${event.region} · ${event.observations || 0} views`; head.append(title);
    const select = document.createElement('select'); select.setAttribute('aria-label',`Rank of card ${index+1}`);
    ranks.forEach(rank => { const option = document.createElement('option'); option.value = rank; option.textContent = rank === 'UNKNOWN' ? 'Unknown' : rank; select.append(option); });
    select.value = event.rank; select.onchange = () => { event.rank = select.value; changed(); }; head.append(select);
    const excludeLabel = document.createElement('label'); const check = document.createElement('input'); check.type = 'checkbox'; check.checked = !!event.excluded;
    check.onchange = () => { event.excluded = check.checked; changed(); renderCards(); }; excludeLabel.append(check, ' Exclude'); head.append(excludeLabel);
    ['↑','↓'].forEach((arrow,n) => { const button = document.createElement('button'); button.textContent = arrow; button.setAttribute('aria-label',`${n ? 'Move down' : 'Move up'} card ${index+1}`); const target = index + (n ? 1 : -1); button.disabled = target < 0 || target >= events.length; button.onclick = () => { [events[index],events[target]] = [events[target],events[index]]; changed(); renderCards(); }; head.append(button); });
    card.append(head);
    const strip = document.createElement('div'); strip.className = 'evidence';
    (event.evidence || []).forEach((item,i) => { const button = document.createElement('button'); button.className = 'crop'; button.title = `Open original screenshot ${item.source}`; const image = document.createElement('img'); image.src = `/api/evidence/${resultId}/${encodeURIComponent(item.image)}`; image.alt = ['First','Middle','Last'][i] + ' card crop'; button.append(image); button.onclick = () => { $('fullFrame').src = `/api/frame/${resultId}/${item.frame}`; $('frameDialog').showModal(); }; strip.append(button); });
    card.append(strip);
    const caption = document.createElement('p'); caption.className = 'muted'; caption.textContent = event.observations ? `${event.recognition_source === "ocr_consensus" ? "OCR consensus" : "Model suggestion"}: ${event.suggested_rank} (${Math.round(event.confidence*100)}%). Click a crop to inspect the screenshot.` : 'Added during review.'; card.append(caption);
    if (event.possible_duplicates?.length) { const warning = document.createElement('p'); warning.className = 'muted'; warning.textContent = 'Check possible duplicate card(s): ' + event.possible_duplicates.join(', ') + '. This is a suggestion, not an automatic exclusion.'; card.append(warning); }
    $('cards').append(card);
  });
  renderCount();
}
function render(next) {
  state = next;
  const busy = busyStates.includes(state.session_status);
  $('startButton').disabled = busy; $('stopButton').disabled = state.session_status !== 'RECORDING'; $('resetButton').disabled = busy;
  $('replayButton').disabled = busy || !$('sourceSelect').value;
  ['saveRegions','saveReview','addCard','confirmReview'].forEach(id => $(id).disabled = busy || (['saveReview','addCard','confirmReview'].includes(id) && !state.result_id));
  const names = {IDLE:'Ready', STARTING:`Recording starts in ${state.countdown ?? 3}…`, RECORDING:'Recording', STOPPING:'Saving recording…', PROCESSING:'Analyzing saved frames…', COMPLETE:'Analysis complete', ERROR:'Action needed'};
  $('status').textContent = names[state.session_status] || state.session_status;
  $('progress').textContent = state.session_status === 'RECORDING' ? `${state.frames_recorded} frames saved` : state.session_status === 'PROCESSING' ? `${state.processed} / ${state.total || '…'} frames processed` : '';
  if (state.error) error(state.error);
  if (!regions.length && state.regions) regions = structuredClone(state.regions);
  if (state.result_id !== resultId) {
    resultId = state.result_id; events = structuredClone(state.events || []); dirty = false; $('confirmReview').checked = !!state.review_confirmed;
    $('warnings').textContent = (state.warnings || []).join(' '); $('warnings').hidden = !state.warnings?.length; $('download').hidden = !resultId;
    renderCards();
  }
  renderCount();
}
async function refresh() { try { render(await api('/api/state')); } catch (e) { error(e.message); } }
async function action(path, body = {}) { try { error(null); await api(path,body); await refresh(); } catch(e) { error(e.message); } }
$('startButton').onclick = () => action('/api/start');
$('stopButton').onclick = () => action('/api/stop');
$('resetButton').onclick = () => action('/api/reset');
$('replayButton').onclick = () => action('/api/process',{source:$('sourceSelect').value});
$('sourceSelect').onchange = () => render(state);
$('addCard').onclick = () => { events.push({id:`new-${crypto.randomUUID()}`,rank:'UNKNOWN',region:'manual',observations:0,evidence:[],excluded:false}); changed(); renderCards(); };
$('saveReview').onclick = async () => {
  try { const data = await api('/api/review',{result_id:resultId,events:events.map(e => ({id:e.id,rank:e.rank,excluded:!!e.excluded})),confirmed:$('confirmReview').checked}); events = structuredClone(data.events); dirty = false; render(data); renderCards(); }
  catch(e) { error(e.message); }
};
$('closeFrame').onclick = () => $('frameDialog').close();
function paint() {
  if (!previewImage) return;
  const canvas = $('preview'), ctx = canvas.getContext('2d'); canvas.width = previewImage.naturalWidth; canvas.height = previewImage.naturalHeight;
  ctx.drawImage(previewImage,0,0);
  regions.forEach((r,i) => { const [x,y,x2,y2] = r.rect; ctx.strokeStyle = i ? '#ffb55d' : '#63dfc3'; ctx.lineWidth = Math.max(2,canvas.width/400); ctx.strokeRect(x*canvas.width,y*canvas.height,(x2-x)*canvas.width,(y2-y)*canvas.height); ctx.fillStyle = ctx.strokeStyle; ctx.font = `${Math.max(16,canvas.width/60)}px sans-serif`; ctx.fillText(r.name,x*canvas.width+4,y*canvas.height+25); });
}
$('previewButton').onclick = async () => {
  try { const source = $('sourceSelect').value || 'live'; const response = await fetch(`/api/preview?source=${encodeURIComponent(source)}`); if (!response.ok) throw new Error((await response.json()).error); const blob = await response.blob(); const url = URL.createObjectURL(blob); previewImage = new Image(); previewImage.onload = () => { URL.revokeObjectURL(url); $('preview').hidden = false; paint(); }; previewImage.src = url; } catch(e) { error(e.message); }
};
function point(event) { const r = $('preview').getBoundingClientRect(); return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))]; }
$('preview').onpointerdown = e => { dragging = point(e); $('preview').setPointerCapture(e.pointerId); };
$('preview').onpointerup = e => { if (!dragging) return; const end = point(e); const i = Number($('regionSelect').value); regions[i] = {name:i ? 'player':'dealer',rect:[Math.min(dragging[0],end[0]),Math.min(dragging[1],end[1]),Math.max(dragging[0],end[0]),Math.max(dragging[1],end[1])]}; dragging = null; paint(); };
$('saveRegions').onclick = () => action('/api/regions',{regions});
(async () => { try { const data = await api('/api/sources'); $('sourceSelect').replaceChildren(); data.replays.forEach(name => { const option = document.createElement('option'); option.value = name; option.textContent = name; $('sourceSelect').append(option); }); } catch(e) { error(e.message); } await refresh(); })();
setInterval(refresh,1000);
