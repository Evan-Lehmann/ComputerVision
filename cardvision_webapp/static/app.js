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
function changed() { dirty = true; $('confirmReview').checked = false; $('confirmRounds').checked = false; renderCount(); }
const signed = value => value > 0 ? `+${value}` : `${value}`;
function renderRounds() {
  $('roundPanel').hidden = !resultId;
  const numbers = [...new Set(events.map(e => e.round_number || 1))].sort((a,b) => a-b);
  $('roundConfirmation').hidden = numbers.length < 2 && !state.boundaries?.length && !state.uncertain_transitions?.length;
  $('roundRows').replaceChildren();
  let running = 0, runningUnknown = 0;
  numbers.forEach(number => {
    const cards = events.filter(e => !e.excluded && (e.round_number || 1) === number);
    const unknown = cards.filter(e => e.rank === 'UNKNOWN').length;
    const change = cards.reduce((sum,e) => sum + (['A','10','J','Q','K'].includes(e.rank) ? -1 : ['2','3','4','5','6'].includes(e.rank) ? 1 : 0),0);
    running += change; runningUnknown += unknown;
    const row = document.createElement('tr');
    [number, cards.map(e => e.rank === 'UNKNOWN' ? '?' : e.rank).join(', ') || 'None included',
      unknown ? `${signed(change)} subtotal (${unknown} unknown)` : signed(change),
      runningUnknown ? `${signed(running)} subtotal (${runningUnknown} unknown)` : signed(running)].forEach(value => {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      });
    $('roundRows').append(row);
  });
  $('roundTransitions').replaceChildren();
  function transition(text, frames) {
    const p = document.createElement('p'); p.textContent = text + ' ';
    frames.forEach(([label,index]) => {
      const button = document.createElement('button'); button.textContent = label;
      button.onclick = () => { $('fullFrame').src = `/api/frame/${resultId}/${index}`; $('frameDialog').showModal(); };
      p.append(button);
    });
    $('roundTransitions').append(p);
  }
  (state.boundaries || []).forEach(b => transition(`Suggested boundary before Round ${b.round_number}.`, [['Before',b.previous_last_frame],['Cleared table',b.clear_end],['After',b.first_frame]]));
  (state.uncertain_transitions || []).forEach(b => transition('Uncertain transition: no verified empty table. Check whether a new round began.', [['Gap',b.gap_end],['After',b.first_frame]]));
  return numbers.length;
}
function renderCount() {
  const roundCount = renderRounds();
  const included = events.filter(e => !e.excluded);
  const unknown = included.filter(e => e.rank === 'UNKNOWN').length;
  const subtotal = included.reduce((total,e) => total + (['A','10','J','Q','K'].includes(e.rank) ? -1 : ['2','3','4','5','6'].includes(e.rank) ? 1 : 0),0);
  const reviewed = !dirty && state.count_verified;
  $('countChange').textContent = resultId ? (subtotal > 0 ? `+${subtotal}` : `${subtotal}`) : '—';
  $('countHeading').textContent = resultId && unknown ? 'Known-card subtotal' : roundCount > 1 ? 'Running count change' : 'Captured-card count change';
  $('countBadge').textContent = !resultId ? (state.session_status === 'PROCESSING' ? 'Analyzing' : 'Waiting') : unknown ? 'Incomplete' : dirty ? 'Unsaved edits' : reviewed ? 'Reviewed by you' : state.review_confirmed && state.needs_round_review ? 'Check rounds' : 'Provisional';
  $('countValueNote').textContent = !resultId ? '' : unknown ? 'subtotal only' : 'captured cards';
  $('downloadCrops').hidden = dirty || !reviewed || !events.some(e => !e.excluded && e.evidence?.length);
  $('countBadge').dataset.status = reviewed ? 'reviewed' : resultId ? 'provisional' : 'waiting';
  $('countStatus').textContent = !resultId ? (state.session_status === 'PROCESSING' ? 'Calculating the count from your saved screenshots…' : 'The count is calculated after you stop recording and analyze.') : unknown ? `${unknown} unknown card${unknown === 1 ? '' : 's'} — possible change for these detections: ${signed(subtotal-unknown)} to ${signed(subtotal+unknown)}.${dirty ? ' Edits are not saved.' : ''}` : dirty ? 'Count reflects your edits. Save the review to keep them.' : reviewed ? `${included.length} included cards. You confirmed this card list and its round divisions.` : state.review_confirmed && state.needs_round_review ? 'Check and confirm the round divisions, then save the review again.' : `${included.length} included cards across ${roundCount} round${roundCount === 1 ? '' : 's'}. Check the card list and round divisions before trusting this count.`;
}
function renderCards() {
  $('cards').replaceChildren();
  if (!events.length) { const p = document.createElement('p'); p.className = 'empty-state'; p.textContent = 'No cards found. Check the selected regions and the full round.'; $('cards').append(p); }
  events.forEach((event,index) => {
    const card = document.createElement('article'); card.className = 'card-row' + (event.excluded ? ' excluded' : '');
    const head = document.createElement('div'); head.className = 'card-heading';
    const title = document.createElement('span'); title.textContent = `${index+1}. ${event.region} · ${event.observations || 0} observations`; head.append(title);
    const select = document.createElement('select'); select.setAttribute('aria-label',`Rank of card ${index+1}`);
    ranks.forEach(rank => { const option = document.createElement('option'); option.value = rank; option.textContent = rank === 'UNKNOWN' ? 'Unknown' : rank; select.append(option); });
    select.value = event.rank; select.onchange = () => { event.rank = select.value; changed(); }; head.append(select);
    const roundLabel = document.createElement('label'); roundLabel.textContent = 'Round ';
    const roundInput = document.createElement('input'); roundInput.type = 'number'; roundInput.min = 1; roundInput.max = 1000; roundInput.value = event.round_number || 1; roundInput.setAttribute('aria-label',`Round of card ${index+1}`);
    roundInput.onchange = () => { const number = Number(roundInput.value); if (!Number.isInteger(number) || number < 1 || number > 1000) { error('Choose a whole round number between 1 and 1000.'); roundInput.value = event.round_number || 1; return; } event.round_number = number; changed(); };
    roundLabel.append(roundInput); head.append(roundLabel);
    const excludeLabel = document.createElement('label'); const check = document.createElement('input'); check.type = 'checkbox'; check.checked = !!event.excluded;
    check.onchange = () => { event.excluded = check.checked; changed(); renderCards(); }; excludeLabel.append(check, ' Exclude'); head.append(excludeLabel);
    ['↑','↓'].forEach((arrow,n) => { const button = document.createElement('button'); button.textContent = arrow; button.setAttribute('aria-label',`${n ? 'Move down' : 'Move up'} card ${index+1}`); const target = index + (n ? 1 : -1); button.disabled = target < 0 || target >= events.length; button.onclick = () => { [events[index],events[target]] = [events[target],events[index]]; changed(); renderCards(); }; head.append(button); });
    card.append(head);
    const strip = document.createElement('div'); strip.className = 'evidence';
    (event.evidence || []).forEach((item,i) => { const button = document.createElement('button'); button.className = 'crop'; button.title = `Open original screenshot ${item.source}`; const image = document.createElement('img'); image.src = `/api/evidence/${resultId}/${encodeURIComponent(item.image)}`; image.alt = ['First','Middle','Last'][i] + ' card crop'; button.append(image); button.onclick = () => { $('fullFrame').src = `/api/frame/${resultId}/${item.frame}`; $('frameDialog').showModal(); }; strip.append(button); });
    card.append(strip);
    const caption = document.createElement('p'); caption.className = 'muted'; caption.textContent = event.observations ? event.recognition_source === 'reference_consensus' ? `Reference agreement: ${event.suggested_rank} (image match ${event.confidence.toFixed(2)}). Click a crop to inspect the screenshot.` : `${event.recognition_source === "ocr_consensus" ? "OCR consensus" : "Model suggestion"}: ${event.suggested_rank} (score ${event.confidence.toFixed(2)}). Click a crop to inspect the screenshot.` : 'Added during review.'; card.append(caption);
    const duplicateRows = (event.possible_duplicates || [])
      .map(id => events.findIndex(other => String(other.id) === String(id)))
      .filter(i => i >= 0 && !events[i].excluded && i !== index)
      .map(i => i + 1);
    if (!event.excluded && duplicateRows.length) { const warning = document.createElement('p'); warning.className = 'muted'; warning.textContent = 'Compare with candidate row(s): ' + duplicateRows.join(', ') + '. These may be the same physical card; both remain included until you exclude one.'; card.append(warning); }
    $('cards').append(card);
  });
  renderCount();
}
function render(next) {
  state = next;
  const busy = busyStates.includes(state.session_status);
  $('startButton').disabled = busy; $('stopButton').disabled = state.session_status !== 'RECORDING'; $('resetButton').disabled = busy;
  $('replayButton').disabled = busy || !$('sourceSelect').value;
  ['saveRegions','saveReview','addCard','confirmReview','confirmRounds'].forEach(id => $(id).disabled = busy || (['saveReview','addCard','confirmReview','confirmRounds'].includes(id) && !state.result_id));
  const names = {IDLE:'Ready', STARTING:`Recording starts in ${state.countdown ?? 3}…`, RECORDING:'Recording', STOPPING:'Saving recording…', PROCESSING:'Analyzing saved frames…', COMPLETE:'Analysis complete', ERROR:'Action needed'};
  $('status').textContent = names[state.session_status] || state.session_status;
  $('progress').textContent = state.session_status === 'RECORDING' ? `${state.frames_recorded} frames saved` : state.session_status === 'PROCESSING' ? `${state.processed} / ${state.total || '…'} frames processed` : '';
  if (state.error) error(state.error);
  if (!regions.length && state.regions) regions = structuredClone(state.regions);
  if (state.result_id !== resultId) {
    resultId = state.result_id; events = structuredClone(state.events || []); dirty = false; $('confirmReview').checked = !!state.review_confirmed;
    $('confirmRounds').checked = !!state.rounds_confirmed;
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
$('addCard').onclick = () => { events.push({id:`new-${crypto.randomUUID()}`,rank:'UNKNOWN',region:'manual',round_number:Math.max(1,...events.map(e => e.round_number || 1)),observations:0,evidence:[],excluded:false}); changed(); renderCards(); };
$('saveReview').onclick = async () => {
  try { const data = await api('/api/review',{result_id:resultId,events:events.map(e => ({id:e.id,rank:e.rank,round_number:e.round_number || 1,excluded:!!e.excluded})),confirmed:$('confirmReview').checked,rounds_confirmed:$('confirmRounds').checked}); events = structuredClone(data.events); dirty = false; render(data); renderCards(); }
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
