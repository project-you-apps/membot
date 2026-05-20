
const $=s=>document.querySelector(s);
let _mounted=null, _memories=0;
function applyReadOnly(ro){
  const ta=$('#storeContent'),btn=document.querySelector('.store-row button'),ti=$('#storeTags');
  if(ro){
    if(ta){ta.value='THIS SERVICE NOT YET AVAILABLE';ta.disabled=true;ta.style.opacity='0.5';}
    if(ti){ti.disabled=true;ti.style.opacity='0.5';}
    if(btn){btn.disabled=true;btn.style.opacity='0.5';btn.style.cursor='not-allowed';}
  }else{
    if(ta){ta.value='';ta.disabled=false;ta.style.opacity='1';}
    if(ti){ti.disabled=false;ti.style.opacity='1';}
    if(btn){btn.disabled=false;btn.style.opacity='1';btn.style.cursor='pointer';}
  }
}
const BASE=()=>{
  const loc=location.pathname.replace(/\/app\/?$/,'');
  return location.protocol==='file:'?'http://137.184.227.79:8000':location.origin+loc;
};
async function checkStatus(){
  const dot=$('#statusDot'),txt=$('#statusText');
  try{
    const r=await fetch(BASE()+'/api/status',{signal:AbortSignal.timeout(5000)});
    const d=await r.json();
    dot.className='status-dot connected';
    _mounted=d.cartridge; _memories=d.memories||0;
    txt.textContent=(_mounted||'No cart')+' ('+_memories.toLocaleString()+' memories)';
    applyReadOnly(d.read_only);
  }catch(e){ dot.className='status-dot error'; txt.textContent='Disconnected'; }
}
async function loadCartridges(){
  const bar=$('#cartBar');
  try{
    const [cr,sr]=await Promise.all([
      fetch(BASE()+'/api/cartridges',{signal:AbortSignal.timeout(5000)}).then(r=>r.json()),
      fetch(BASE()+'/api/status',{signal:AbortSignal.timeout(5000)}).then(r=>r.json())
    ]);
    _mounted=sr.cartridge; _memories=sr.memories||0;
    const dot=$('#statusDot'),txt=$('#statusText');
    dot.className='status-dot connected';
    txt.textContent=(_mounted||'No cart')+' ('+_memories.toLocaleString()+' memories)';
    applyReadOnly(sr.read_only);
    const carts=cr.cartridges||[];
    if(carts.length===0){bar.innerHTML='<div class="cart-chip"><div class="name" style="color:var(--text-dim)">No cartridges found</div></div>';return;}
    bar.innerHTML=carts.map(c=>{
      const active=_mounted&&c.name===_mounted;
      const cls='cart-chip'+(active?' active':'');
      const mem=active?' &middot; <span class="count">'+_memories.toLocaleString()+' memories</span>':'';
      return '<div class="'+cls+'" data-name="'+esc(c.name)+'"><div class="name">'+esc(c.name)+'</div><div class="meta">'+c.size_mb+' MB &middot; '+c.format.toUpperCase()+(c.has_brain?' &middot; GPU':'')+mem+'</div></div>';
    }).join('');
    bar.querySelectorAll('.cart-chip').forEach(el=>el.addEventListener('click',()=>mountCart(el.dataset.name)));
  }catch(e){bar.innerHTML='<div class="cart-chip"><div class="name" style="color:var(--red)">Connection failed</div></div>';}
}
async function mountCart(name){
  if(name===_mounted)return;
  $('#resultsEl').innerHTML='<div class="empty-state"><div class="icon">&#x1F9E0;</div><p>Search your brain cartridge</p><div class="hint" style="font-size:12px;margin-top:8px">Type a query to find memories by meaning</div></div>';
  $('#searchMeta').textContent=''; _lastResults=[]; _lastQuery='';
  const chips=document.querySelectorAll('.cart-chip');
  chips.forEach(c=>{if(c.dataset.name===name)c.classList.add('mounting');});
  toast('Mounting '+name+'...');
  try{
    const r=await fetch(BASE()+'/api/mount',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    const d=await r.json();
    if(d.error){toast(d.error,'error');}
    else{toast('Mounted '+name,'success');}
    await loadCartridges();
  }catch(e){toast('Mount failed: '+e.message,'error');chips.forEach(c=>c.classList.remove('mounting'));}
}
var _lastResults=[], _lastQuery='';
async function doSearch(){
  const query=$('#searchInput').value.trim();
  if(!query)return;
  _lastQuery=query;
  const el=$('#resultsEl'),loading=$('#loadingEl'),meta=$('#searchMeta');
  el.innerHTML=''; loading.className='loading active'; meta.textContent='';
  const t0=performance.now();
  try{
    const r=await fetch(BASE()+'/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:8})});
    const data=await r.json();
    const elapsed=Math.round(performance.now()-t0);
    loading.className='loading';
    var items=data.results;
    if(!items&&typeof data.result==='string'){
      items=[];
      var lines=data.result.split(/\n\n/);
      for(var li=0;li<lines.length;li++){
        var m=lines[li].match(/^#\d+\s+\[([\d.]+)\]\s+(.*)/s);
        if(m)items.push({score:parseFloat(m[1]),text:m[2],full_text:m[2],tags:''});
      }
    }
    _lastResults=items||[];
    console.log('[HIPPO] Search results:', JSON.stringify(items?.map(r=>({idx:r.index,prev:r.prev_idx,next:r.next_idx}))));
    const haystack=_memories?' of '+_memories.toLocaleString():'';
    if(!items||items.length===0){
      el.innerHTML='<div class="empty-state"><div class="icon">&#x1F914;</div><p>No results found</p></div>';
      meta.textContent='0 results'+haystack+' in '+elapsed+'ms'; return;
    }
    meta.textContent=items.length+haystack+' results in '+elapsed+'ms';
    el.innerHTML=items.map((item,i)=>{
      const text=esc(item.text||item.content||'');
      const score=item.score!=null?item.score.toFixed(4):'';
      const tags=item.tags?item.tags.split(',').map(t=>'<span class="tag">'+esc(t.trim())+'</span>').join(''):'';
      const provBits=[];
      if(item.source_db)provBits.push(esc(item.source_db));
      if(item.paper_id)provBits.push('id: '+esc(item.paper_id));
      const prov=provBits.length?'<span class="prov">'+provBits.join(' · ')+'</span>':'';
      return '<div class="result-card" data-idx="'+i+'"><div class="result-header"><span class="result-rank">#'+(i+1)+'</span>'+(score?'<span class="result-score">score: '+score+'</span>':'')+'</div><div class="result-text">'+highlight(linkify(text),query)+'</div>'+((tags||prov)?'<div class="result-footer">'+tags+prov+'</div>':'')+'</div>';
    }).join('');
    el.querySelectorAll('.result-card').forEach(c=>c.addEventListener('click',()=>openPassage(parseInt(c.dataset.idx))));
  }catch(e){ loading.className='loading'; el.innerHTML='<div class="empty-state"><div class="icon">&#x26A0;</div><p>Search failed</p><div style="font-size:12px;margin-top:8px">'+esc(e.message)+'</div></div>'; }
}
var _currentPassage=null;
function openPassage(resultIdx){
  if(resultIdx<0||resultIdx>=_lastResults.length)return;
  const item=_lastResults[resultIdx];
  console.log('[HIPPO] openPassage resultIdx='+resultIdx+' item.index='+item.index+' prev_idx='+item.prev_idx+' next_idx='+item.next_idx, item);
  _showPassage({index:item.index,full_text:item.full_text||item.text||'',prev_idx:item.prev_idx,next_idx:item.next_idx,score:item.score,rank:resultIdx+1,paper_id:item.paper_id,source_db:item.source_db});
}
function _showPassage(p){
  _currentPassage=p;
  $('#modalRank').textContent=p.rank?'#'+p.rank:'idx:'+p.index;
  $('#modalScore').textContent=p.score!=null?'score: '+p.score.toFixed(4):'';
  $('#modalText').innerHTML=highlight(linkify(esc(p.full_text)),_lastQuery);
  // Provenance state machine: split-cart preview (source_db, no paper_id) shows the
  // load CTA; loaded state (paper_id present) shows the source line; non-split shows
  // neither.
  const isSplit=!!p.source_db;
  const isLoaded=!!p.paper_id;
  const cta=$('#modalCta'),src=$('#modalSource');
  if(cta){
    if(isSplit&&!isLoaded){
      cta.innerHTML='<button class="modal-cta-btn" onclick="loadSource()">&#x1F4C2; Load full passage from '+esc(p.source_db)+' &rarr;</button>';
      cta.style.display='block';
    } else { cta.style.display='none'; cta.innerHTML=''; }
  }
  if(src){
    if(isLoaded){
      const bits=['source: '+p.source_db];
      if(p.paper_id)bits.push('id: '+p.paper_id);
      src.textContent=bits.join(' · ');
      src.style.display='block';
    } else { src.style.display='none'; }
  }
  const btnP=$('#btnPrev'),btnN=$('#btnNext');
  if(p.prev_idx!=null){btnP.className='passage-nav-btn enabled';btnP.onclick=()=>navigatePassage(p.prev_idx);}
  else{btnP.className='passage-nav-btn';btnP.onclick=null;}
  if(p.next_idx!=null){btnN.className='passage-nav-btn enabled';btnN.onclick=()=>navigatePassage(p.next_idx);}
  else{btnN.className='passage-nav-btn';btnN.onclick=null;}
  $('#passageOverlay').classList.add('open');
  document.addEventListener('keydown',_passageKeys);
}
async function loadSource(){
  if(!_currentPassage)return;
  const cta=$('#modalCta'),idx=_currentPassage.index;
  if(cta)cta.innerHTML='<span class="modal-cta-loading">Loading from source database&hellip;</span>';
  try{
    const r=await fetch(BASE()+'/api/passage?idx='+idx);
    const data=await r.json();
    if(data.error){toast(data.error,'error');return;}
    const p=data.passage;
    _showPassage({index:p.index,full_text:p.full_text,prev_idx:p.prev_idx,next_idx:p.next_idx,score:_currentPassage.score,rank:_currentPassage.rank,source_db:p.source_db,paper_id:p.paper_id});
  }catch(e){
    toast('Source load failed: '+e.message,'error');
    if(cta)cta.innerHTML='<button class="modal-cta-btn" onclick="loadSource()">Retry</button>';
  }
}
async function navigatePassage(idx){
  try{
    const r=await fetch(BASE()+'/api/passage?idx='+idx);
    const data=await r.json();
    if(data.error){toast(data.error,'error');return;}
    const p=data.passage;
    _showPassage({index:p.index,full_text:p.full_text,prev_idx:p.prev_idx,next_idx:p.next_idx,score:null,rank:null,source_db:p.source_db,paper_id:p.paper_id});
  }catch(e){toast('Navigation failed: '+e.message,'error');}
}
function closePassage(){
  $('#passageOverlay').classList.remove('open');
  _currentPassage=null;
  document.removeEventListener('keydown',_passageKeys);
}
function _passageKeys(e){
  if(e.key==='Escape')closePassage();
  if(e.key==='ArrowLeft'&&_currentPassage&&_currentPassage.prev_idx!=null)navigatePassage(_currentPassage.prev_idx);
  if(e.key==='ArrowRight'&&_currentPassage&&_currentPassage.next_idx!=null)navigatePassage(_currentPassage.next_idx);
}
async function doStore(){
  const content=$('#storeContent').value.trim();
  const tags=$('#storeTags').value.trim();
  if(!content){toast('Enter content to store','error');return;}
  try{
    const r=await fetch(BASE()+'/api/store',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content,tags})});
    const data=await r.json();
    if(data.error){toast(data.error,'error');}
    else{toast('Memory stored','success');$('#storeContent').value='';$('#storeTags').value='';checkStatus();}
  }catch(e){toast('Store failed: '+e.message,'error');}
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
const _STOP=new Set(['the','and','are','was','were','for','that','this','with','from','not','but','has','had','have','does','did','will','can','its','who','what','how','why','about','tell','you','your','some','than','them','then','they','been','more','also','into','would','could','should','just','like','very','much','many','only','other','over','such','after','before','between','through','where','when','which','while','each','there','their','these','those','being','because','during','both','same','own','most','well','way','all','out','one','two','may']);
function highlight(html,query){
  if(!query)return html;
  const words=query.split(/\s+/).filter(w=>w.length>2&&!_STOP.has(w.toLowerCase()));
  let r=html;
  // Negative lookahead (?![^<]*>) skips matches inside open tags / attribute values
  // (e.g. inside href="https://arxiv.org") so URL hrefs don't break.
  for(const w of words){const re=new RegExp('('+w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')(?![^<]*>)','gi');r=r.replace(re,'<mark>$1</mark>');}
  return r;
}
function linkify(escapedText){
  // Wrap http(s) URLs in clickable links. Input must already be HTML-escaped.
  return escapedText.replace(/(https?:\/\/[^\s<]+?)([.,;:!?)\]]?(?=\s|$|<))/g,'<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>$2');
}
function toast(msg,type='success'){
  const el=$('#toastEl');el.textContent=msg;el.className='toast '+type+' show';
  setTimeout(()=>el.className='toast',3000);
}
function toggleTheme(){
  const html=document.documentElement;
  const next=html.getAttribute('data-theme')==='light'?'dark':'light';
  html.setAttribute('data-theme',next);
  $('#themeBtn').innerHTML=next==='light'?'&#x1F319;':'&#x2600;';
  localStorage.setItem('membot-theme',next);
}
(function(){const s=localStorage.getItem('membot-theme');if(s==='light'){document.documentElement.setAttribute('data-theme','light');setTimeout(()=>$('#themeBtn').innerHTML='&#x1F319;',0);}})();
loadCartridges();

/* =========================================================
 * Mempack Dashboard
 *   - View toggle (Search vs Mempack)
 *   - Owner UUID persistence in localStorage
 *   - Mempack list + selection
 *   - Pattern I editor (POST /api/mempack/<id>/pattern-i)
 *   - Activity feed (GET /api/mempack/<id>/activity, polling)
 *   - Settings toggle (PATCH /api/mempack/<id>)
 * ========================================================= */
const _SEARCH_VIEW_IDS = ['searchIntro','cartBar','storeSection'];
const _SEARCH_VIEW_SELECTORS = ['.search-box','.search-meta','#loadingEl','#resultsEl'];
let _currentMempack = null;     // currently selected mempack row
let _activitySinceTs = null;    // pagination cursor for delta polling
let _activityPollHandle = null; // setInterval handle
const _ACTIVITY_POLL_MS = 5000;

async function setView(name){
  const tabs = document.querySelectorAll('.view-tab');
  tabs.forEach(t => t.classList.toggle('active', t.dataset.view === name));
  const showSearch = (name === 'search');
  const showMempack = (name === 'mempack');
  _SEARCH_VIEW_IDS.forEach(id => { const el = document.getElementById(id); if(el) el.style.display = showSearch ? '' : 'none'; });
  _SEARCH_VIEW_SELECTORS.forEach(sel => { const el = document.querySelector(sel); if(el) el.style.display = showSearch ? '' : 'none'; });
  const mv = document.getElementById('mempackView'); if(mv) mv.style.display = showMempack ? '' : 'none';
  if (showMempack) {
    // Server-side identity check — Supabase cookies are HttpOnly so JS can't
    // read them, but they're auto-sent with same-origin fetches. /api/whoami
    // reads them server-side and tells us who's signed in.
    const detected = await detectSupabaseUser();
    const override = localStorage.getItem('membot-owner-override');
    const uuid = override || detected;
    if (uuid && $('#ownerUuid').value !== uuid) {
      $('#ownerUuid').value = uuid;
      loadMempacks();
    } else if (!uuid) {
      $('#ownerUuid').placeholder = 'No Supabase session detected. Sign in at project-you.app or use Override.';
      $('#mempackList').innerHTML = '<div class="dash-empty"><div class="icon">&#x1F511;</div>No Supabase session in this browser.<br><small>Sign in at <a href="https://project-you.app/" style="color:var(--accent)">project-you.app</a> first, then return.</small></div>';
    }
    loadTemplates();  // populates the Pattern I template dropdown (cached)
    if (_currentMempack) startActivityPoll();
  } else {
    stopActivityPoll();
  }
  localStorage.setItem('membot-active-view', name);
}

/* Identify the signed-in user via the server, since Supabase cookies are
 * HttpOnly and not readable from document.cookie. The cookie IS sent with
 * every same-origin fetch automatically, so /api/whoami can read it
 * server-side and tell us the user_id + email + avatar + name.
 *
 * getWhoami() returns the full {signed_in, user_id, email, full_name,
 * avatar_url} object; detectSupabaseUser() is a back-compat wrapper that
 * returns just the user_id string (or null) for callers that haven't been
 * upgraded yet.
 */
let _whoamiCache = { ts: 0, whoami: null };
const _WHOAMI_CACHE_MS = 3000;
async function getWhoami(){
  const now = Date.now();
  if (now - _whoamiCache.ts < _WHOAMI_CACHE_MS) return _whoamiCache.whoami;
  try {
    const r = await fetch(BASE() + '/api/whoami', { credentials: 'same-origin' });
    const d = await r.json();
    _whoamiCache = { ts: now, whoami: d };
    if (!d || !d.user_id) console.warn('[membot] getWhoami: server reports no signed-in user (cookie missing or invalid JWT)');
    return d;
  } catch(e) {
    console.warn('[membot] getWhoami /api/whoami failed:', e);
    return null;
  }
}
async function detectSupabaseUser(){
  const w = await getWhoami();
  return w && w.user_id ? w.user_id : null;
}

/* =========================================================
 * Auth chip (Option D — sign-in via popup to /vps/app)
 *   - Signed out: "Sign in" button → opens a popup to project-you.app
 *   - Signed in:  avatar (or initial) → dropdown with email + Profile (soon) + Sign out
 *   - Popup-open mode polls /api/whoami fast (every 2s); reverts to slow poll on close
 * ========================================================= */
let _authMenuOpen = false;
let _signinPopupRef = null;
let _signinPopupPoll = null;

function renderOwnerEmailLabel(whoami){
  // Updates the "Signed in as <email>" label at the top of the Mempack tab.
  // Falls through to "Anonymous" italic when no session is detected.
  const el = document.getElementById('ownerEmailLabel');
  if (!el) return;
  if (whoami && whoami.signed_in) {
    el.textContent = whoami.email || whoami.full_name || (whoami.user_id ? whoami.user_id.slice(0, 8) : 'Signed in');
    el.classList.remove('anonymous');
  } else {
    el.textContent = 'Anonymous';
    el.classList.add('anonymous');
  }
}

function renderAuthChip(whoami){
  // Side-effect: also refresh the owner-email label since both surfaces are
  // informed by the same whoami payload.
  renderOwnerEmailLabel(whoami);
  const chip = document.getElementById('authChip');
  if (!chip) return;
  const signedIn = whoami && whoami.signed_in;
  if (!signedIn) {
    chip.innerHTML =
      '<button class="auth-signin-btn" onclick="doSignIn()" title="Sign in to access your Mempack">'
      + '<span>&#x1F511;</span> Sign in'
      + '</button>';
    _authMenuOpen = false;
    return;
  }
  const email = whoami.email || '(no email)';
  const name = whoami.full_name || email;
  const avatar = whoami.avatar_url;
  const initial = (whoami.email && whoami.email[0] ? whoami.email[0] : (whoami.user_id || '?')[0]).toUpperCase();
  const avatarHtml = avatar
    ? '<img src="' + esc(avatar) + '" alt="">'
    : '<span class="auth-initial">' + esc(initial) + '</span>';
  const menuHtml = _authMenuOpen
    ? ('<div class="auth-menu" id="authMenu">'
      +   '<div class="auth-menu-header">'
      +     '<div class="label">Signed in as</div>'
      +     '<div class="email" title="' + esc(name) + '">' + esc(email) + '</div>'
      +   '</div>'
      +   '<button class="auth-menu-item disabled" onclick="event.stopPropagation()">'
      +     '<span>&#x1F464;</span> Profile <span class="soon-badge">soon</span>'
      +   '</button>'
      +   '<button class="auth-menu-item danger" onclick="doSignOut()">'
      +     '<span>&#x21AA;</span> Sign out'
      +   '</button>'
      + '</div>')
    : '';
  chip.innerHTML =
    '<button class="auth-avatar-btn" onclick="toggleAuthMenu(event)" title="' + esc(name) + '">'
    + avatarHtml
    + '</button>'
    + menuHtml;
}

async function refreshAuthChip(){
  // Bypass cache so we see current state
  _whoamiCache = { ts: 0, whoami: null };
  const w = await getWhoami();
  renderAuthChip(w);
  return w;
}

function toggleAuthMenu(ev){
  if (ev) ev.stopPropagation();
  _authMenuOpen = !_authMenuOpen;
  // Re-render with cached state; no network round-trip
  renderAuthChip(_whoamiCache.whoami);
}

// Close menu on outside click
document.addEventListener('mousedown', (ev) => {
  if (!_authMenuOpen) return;
  const chip = document.getElementById('authChip');
  if (chip && !chip.contains(ev.target)) {
    _authMenuOpen = false;
    renderAuthChip(_whoamiCache.whoami);
  }
});

function doSignIn(){
  // Stamp APP_ID so the central auth router (project-you.app/) lands the user
  // back at /membot/app/ after sign-in completes inside VPS's modal.
  try { localStorage.setItem('auth_return_app', 'membot'); } catch(_) {}

  const url = 'https://project-you.app/vps/app/';
  const w = 480;
  const h = 720;
  const left = (screen.width  - w) / 2;
  const top  = (screen.height - h) / 2;
  const features = [
    'width=' + w, 'height=' + h,
    'left=' + left, 'top=' + top,
    'resizable=yes', 'scrollbars=yes', 'status=no', 'toolbar=no', 'menubar=no',
  ].join(',');
  _signinPopupRef = window.open(url, 'membot-signin', features);
  if (!_signinPopupRef) {
    toast('Popup blocked — allow popups for project-you.app, or sign in at project-you.app/vps/app and come back', 'error');
    return;
  }
  _signinPopupRef.focus();

  // While the popup is open, poll /api/whoami every 2s. When the cookie
  // shows signed-in, update the chip and stop. When the popup closes (with
  // or without success), do one more refresh and stop.
  if (_signinPopupPoll) clearInterval(_signinPopupPoll);
  let ticks = 0;
  _signinPopupPoll = setInterval(async () => {
    ticks += 1;
    const popupClosed = !_signinPopupRef || _signinPopupRef.closed;
    const w = await refreshAuthChip();
    const signedIn = w && w.signed_in;
    if (signedIn) {
      // Got auth — close popup if still open, stop polling, refresh dependent UI
      if (_signinPopupRef && !_signinPopupRef.closed) {
        try { _signinPopupRef.close(); } catch(_) {}
      }
      clearInterval(_signinPopupPoll);
      _signinPopupPoll = null;
      _signinPopupRef = null;
      toast('Signed in as ' + (w.email || 'user'), 'success');
      // Kick the cookie watcher so Mempack tab re-loads under the new user
      _userWatchTick();
    } else if (popupClosed) {
      clearInterval(_signinPopupPoll);
      _signinPopupPoll = null;
      _signinPopupRef = null;
    } else if (ticks > 300) {
      // 10-minute hard cap — give up polling even if popup still open
      clearInterval(_signinPopupPoll);
      _signinPopupPoll = null;
    }
  }, 2000);
}

async function doSignOut(){
  _authMenuOpen = false;
  try {
    const r = await fetch(BASE() + '/api/auth/signout', {
      method: 'POST', credentials: 'same-origin',
    });
    const d = await r.json();
    if (d.status !== 'ok') { toast('Sign out failed: ' + (d.error || 'unknown'), 'error'); return; }
    toast('Signed out (' + (d.count || 0) + ' cookies cleared)', 'success');
  } catch(e) {
    toast('Sign out failed: ' + e.message, 'error');
    return;
  }
  // Refresh chip + cookie watcher (which will clear the Mempack tab)
  await refreshAuthChip();
  _userWatchTick();
}

// Initial chip render at page load (before first _userWatchTick)
refreshAuthChip();

function overrideOwner(){
  const current = $('#ownerUuid').value;
  const next = prompt('Override owner UUID (admin/debug). Leave blank to clear:', current || '');
  if (next === null) return;  // cancel
  if (next.trim() === '') {
    localStorage.removeItem('membot-owner-override');
    toast('Override cleared — using detected session', 'success');
  } else {
    if (!/^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$/.test(next.trim())) {
      toast('Not a UUID', 'error'); return;
    }
    localStorage.setItem('membot-owner-override', next.trim());
    toast('Override set — reloading', 'success');
  }
  $('#ownerUuid').value = '';
  setView('mempack');  // re-detect with new override
}

function fmtBytes(n){
  if (n == null) return '?';
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/(1024*1024)).toFixed(2) + ' MB';
}
function fmtTs(iso){
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const pad = n => String(n).padStart(2,'0');
    return d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate())
         + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  } catch(e) { return iso; }
}

async function loadMempacks(){
  const uuid = ($('#ownerUuid').value || '').trim();
  const list = $('#mempackList');
  if (!uuid) { list.innerHTML = '<div class="dash-empty"><div class="icon">&#x1F511;</div>No Supabase session detected. Sign in at <a href="https://project-you.app/" style="color:var(--accent)">project-you.app</a>.</div>'; return; }
  if (!/^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$/.test(uuid)) {
    toast('Detected value is not a UUID', 'error'); return;
  }
  list.innerHTML = '<div class="dash-empty"><div class="spinner"></div> Loading Mempacks...</div>';
  try {
    const r = await fetch(BASE() + '/api/mempacks?owner_id=' + encodeURIComponent(uuid));
    const d = await r.json();
    if (d.error || d.status === 'error') { list.innerHTML = '<div class="dash-empty"><div class="icon">&#x26A0;</div>' + esc(d.error || 'Load failed') + '</div>'; return; }
    const items = d.mempacks || [];
    if (items.length === 0) {
      list.innerHTML = '<div class="dash-empty"><div class="icon">&#x1F5C3;</div>No Mempacks yet for this user.</div>';
      return;
    }
    list.innerHTML = items.map(mp => {
      const sizeMb = (mp.size_bytes || 0) / (1024*1024);
      const capMb = 10;  // free-tier cap from migration 002 trigger
      const pct = Math.min(100, (sizeMb / capMb) * 100);
      const fillCls = pct > 90 ? 'full' : (pct > 70 ? 'warn' : '');
      const statusCls = mp.storage_status === 'ready' ? 'ready' : 'pending';
      return '<div class="mempack-card" data-id="' + esc(mp.id) + '">'
        + '<div class="name">' + esc(mp.name) + '</div>'
        + '<div class="meta">'
        +   '<span class="pill ' + statusCls + '">' + esc(mp.storage_status || 'unknown') + '</span>'
        +   '<span class="pill">' + (mp.pattern_count || 0) + ' patterns</span>'
        +   '<span class="pill">' + fmtBytes(mp.size_bytes) + ' / ' + capMb + ' MB</span>'
        + '</div>'
        + '<div class="quota-bar"><div class="quota-fill ' + fillCls + '" style="width:' + pct.toFixed(1) + '%"></div></div>'
        + '<div class="quota-label">' + pct.toFixed(1) + '% of ' + capMb + ' MB used</div>'
        + '</div>';
    }).join('');
    list.querySelectorAll('.mempack-card').forEach(el => {
      el.addEventListener('click', () => selectMempack(el.dataset.id, items));
    });
    if (d.auto_provisioned) toast('Provisioned starter Mempack ("primary")', 'success');
    // Auto-select if exactly one (common case after lazy-provision)
    if (items.length === 1) selectMempack(items[0].id, items);
  } catch(e) {
    list.innerHTML = '<div class="dash-empty"><div class="icon">&#x26A0;</div>' + esc(e.message) + '</div>';
  }
}

function selectMempack(id, items){
  const mp = (items || []).find(m => m.id === id);
  if (!mp) { toast('Mempack ' + id + ' not in list', 'error'); return; }
  _currentMempack = mp;
  _activitySinceTs = null;
  document.querySelectorAll('.mempack-card').forEach(c => c.classList.toggle('active', c.dataset.id === id));
  $('#mempackDetail').style.display = '';
  // Connect-an-agent snippets (owner_id + name baked in)
  populateConnectPanel(mp);
  // Pattern I
  $('#patternIText').value = mp.pattern_i_text || '';
  $('#patternIMeta').textContent = mp.pattern_count + ' patterns total · idx=1 reserved';
  $('#patternIStatus').textContent = '';
  $('#patternIStatus').className = 'saved-msg';
  // Settings
  $('#loggingToggle').checked = mp.activity_logging_enabled !== false;
  // Dispatch textbox: pre-fill with the most recent DISPATCH-tagged pattern
  // so the user can see what's currently queued without digging into the
  // Patterns Browser. Falls back to whatever the user had typed previously
  // if the fetch fails or no DISPATCH exists.
  prefillDispatchFromMempack(mp.id);
  // Patterns browser (resets state — filters cleared, offset back to 0)
  _patternsCurrentTag = '';
  _patternsCurrentQ = '';
  if ($('#patternsTagFilter')) $('#patternsTagFilter').value = '';
  if ($('#patternsSearchInput')) $('#patternsSearchInput').value = '';
  loadPatterns(true);
  // Activity (initial fetch + start polling)
  $('#activityFeed').innerHTML = '<div class="dash-empty">Loading activity...</div>';
  loadActivity(true);
  startActivityPoll();
}

/* Connect-an-agent panel — bakes the selected Mempack's name + owner_id into
 * a ready-to-paste mcp.json snippet and a starter prompt. Owner_id is per-
 * tool-call (mount_cartridge(name, owner_id)), not per-connection, so the
 * snippet itself is just the SSE URL; the owner_id rides in the prompt.
 */
function populateConnectPanel(mp){
  const ownerId = mp.owner_id || ($('#ownerUuid').value || '').trim() || '<your-uuid>';
  const name = mp.name || 'primary';
  // Droplet membot runs `--transport http` (StreamableHTTP) — endpoint is /mcp,
  // NOT /sse. The 5/12 mcp.json.example is stale. Most modern MCP clients
  // (Claude Code, Cursor, Windsurf) accept either "streamableHttp" or "http"
  // as the type field; "sse" type pointing at /mcp will NOT connect.
  // (We build snippets via array-join to keep this Python-templated JS safe.)
  const mcpUrl = (location.protocol === 'https:' ? 'https:' : 'http:') + '//' + location.host + '/membot/mcp';
  const mcpSnippet = [
    '{',
    '  "mcpServers": {',
    '    "membot": {',
    '      "type": "streamableHttp",',
    '      "url": "' + mcpUrl + '"',
    '    }',
    '  }',
    '}'
  ].join(String.fromCharCode(10));
  // Pull a friendly user label off whoami if available — the prompt nudges
  // the agent to address the user by name/email rather than the UUID it
  // had to pass to mount_cartridge. UUID stays in the tool args because
  // the API requires it.
  const _wp = (_whoamiCache && _whoamiCache.whoami) || {};
  const userLabel = _wp.full_name || _wp.email || (ownerId && ownerId !== '<your-uuid>' ? ownerId.slice(0, 8) : 'me');
  const prompt = [
    'Mount the Mempack named "' + name + '" with owner_id ' + ownerId + '.',
    'Then read Pattern I (mempack_read_pattern_i) and search for tag DISPATCH',
    '(memory_search query="DISPATCH" top_k=10). Acknowledge any dispatches you',
    'find back to me with a one-liner before starting work.',
    '',
    'If there are no DISPATCH patterns, tell me in chat — do not invent work.',
    '',
    'When you complete the dispatch, store every finding via memory_store with',
    'an appropriate tag (FINDING / SUMMARY / METHOD / etc.). Sign each stored',
    'pattern by appending a signature line at the end of the content body:',
    '  [signed: <your-model-name>@<your-host>, <ISO-timestamp>]',
    'e.g. [signed: qwen3:8b@goose, 2026-05-19T18:45:00Z]',
    '',
    'Do NOT report completion until memory_store has actually been called and',
    'returned a "Stored as passage #N" confirmation. Describing findings in',
    'chat is not storing them.',
    '',
    'Note: the owner_id above is a database key. Address me as "' + userLabel + '"',
    'in conversation — Pattern I has the user-facing identifier.'
  ].join(String.fromCharCode(10));
  $('#connectMcpJson').textContent = mcpSnippet;
  $('#connectPrompt').textContent = prompt;
  // Reset copy status indicators
  $('#connectMcpJsonStatus').textContent = '';
  $('#connectPromptStatus').textContent = '';
}

async function copyConnectSnippet(which){
  const sourceId = which === 'mcpJson' ? 'connectMcpJson' : 'connectPrompt';
  const statusId = which === 'mcpJson' ? 'connectMcpJsonStatus' : 'connectPromptStatus';
  const text = $('#' + sourceId).textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    $('#' + statusId).textContent = 'Copied!';
    setTimeout(() => { $('#' + statusId).textContent = ''; }, 2000);
  } catch(e) {
    // Fallback for older browsers / restrictive contexts: select-and-execCommand
    const pre = $('#' + sourceId);
    const range = document.createRange();
    range.selectNode(pre);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    try {
      document.execCommand('copy');
      $('#' + statusId).textContent = 'Copied (fallback)';
      setTimeout(() => { $('#' + statusId).textContent = ''; }, 2000);
    } catch(e2) {
      $('#' + statusId).textContent = 'Copy failed — select + Ctrl+C';
    }
    window.getSelection().removeAllRanges();
  }
}

async function savePatternI(){
  if (!_currentMempack) return;
  const text = $('#patternIText').value;
  const status = $('#patternIStatus');
  status.textContent = 'Saving...';
  status.className = 'saved-msg';
  try {
    const r = await fetch(BASE() + '/api/mempack/' + _currentMempack.id + '/pattern-i', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    if (d.status !== 'ok') { status.textContent = 'Error: ' + (d.error || 'unknown'); return; }
    status.textContent = 'Saved (' + fmtBytes(d.blob_bytes) + ' on disk)';
    status.className = 'saved-msg success';
    _currentMempack.pattern_i_text = text;
  } catch(e) {
    status.textContent = 'Save failed: ' + e.message;
  }
}

/* Pattern I templates — loaded once per page lifetime, cached client-side.
 * The dropdown is populated from /api/mempack/templates; selecting an enabled
 * entry replaces the Pattern I textarea (with confirm if non-empty).
 */
let _templatesCache = null;  // [{id, label, description, body, disabled}, ...]
async function loadTemplates(){
  if (_templatesCache) return _templatesCache;
  try {
    const r = await fetch(BASE() + '/api/mempack/templates');
    const d = await r.json();
    if (d.status !== 'ok') return null;
    _templatesCache = d.templates || [];
    const sel = $('#templateSelect');
    if (sel) {
      // Wipe existing options except the placeholder first option
      while (sel.options.length > 1) sel.remove(1);
      _templatesCache.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.textContent = t.label + (t.disabled ? '' : '');
        if (t.disabled) opt.disabled = true;
        sel.appendChild(opt);
      });
    }
    return _templatesCache;
  } catch(e) {
    console.warn('[membot] loadTemplates failed:', e);
    return null;
  }
}

function _applyTemplateVars(body){
  // The server returns template placeholders intact ({owner_id}, etc.). Fill
  // them client-side from the selected Mempack + the signed-in user so the
  // agent reads a human-friendly greeting instead of "operating on behalf of
  // 3579e6ee". Falls back to the UUID prefix when no email/name available.
  //
  // {owner_id_short}: human-greeting form
  //     1st: full_name (from OAuth)        e.g. "Andy Grossberg"
  //     2nd: email local-part              e.g. "andy.grossberg"
  //     3rd: UUID prefix                   e.g. "3579e6ee"
  // {owner_id}: identifier line
  //     1st: email                         e.g. "andy.grossberg@gmail.com"
  //     2nd: full UUID                     e.g. "3579e6ee-6412-..."
  //     3rd: "unknown"
  //
  // Note: the UUID is kept as the authoritative owner_id in the Mempack
  // schema; we just stop SHOWING it in the agent-facing template body.
  if (!body || !_currentMempack) return body || '';
  const ownerId  = _currentMempack.owner_id || ($('#ownerUuid').value || '').trim();
  const whoami   = _whoamiCache.whoami;
  const fullName = whoami ? whoami.full_name : null;
  const email    = whoami ? whoami.email : null;
  const ownerShort = fullName
    ? fullName
    : (email ? email.split('@')[0] : (ownerId ? ownerId.slice(0, 8) : 'unknown'));
  const ownerFull  = email || ownerId || 'unknown';
  const createdAt = new Date().toISOString();
  return body
    .replace(/\{owner_id_short\}/g, ownerShort)
    .replace(/\{owner_id\}/g, ownerFull)
    .replace(/\{created_at\}/g, createdAt)
    .replace(/\{name\}/g, _currentMempack.name || 'primary');
}

function onTemplatePick(){
  const sel = $('#templateSelect');
  const tid = sel.value;
  if (!tid) return;
  const t = (_templatesCache || []).find(x => x.id === tid);
  if (!t || !t.body) { sel.value = ''; return; }
  const current = $('#patternIText').value;
  if (current && current.trim().length > 0) {
    if (!confirm('Replace current Pattern I with the "' + t.label + '" template? Your current text will be overwritten in the editor (not yet saved to Supabase until you click "Save Pattern I").')) {
      sel.value = '';
      return;
    }
  }
  $('#patternIText').value = _applyTemplateVars(t.body);
  $('#patternIStatus').textContent = 'Template loaded — review then click "Save Pattern I"';
  $('#patternIStatus').className = 'saved-msg';
  sel.value = '';  // reset so picking the same template again still fires
}

/* Pre-fill the Dispatch textbox with the most-recent DISPATCH-tagged
 * pattern in the selected Mempack. Surfaces "what's currently queued" at
 * the surface where the user types — no need to dig into the Patterns
 * Browser to see what's in flight. Strips the `[DISPATCH] ` prefix from
 * the body and tags the textbox with a small `most recent dispatch in
 * this Mempack` hint until the user edits.
 */
async function prefillDispatchFromMempack(mempackId){
  const ta = $('#dispatchText');
  const status = $('#dispatchStatus');
  if (!ta) return;
  try {
    const r = await fetch(BASE() + '/api/mempack/' + mempackId + '/patterns?tag=DISPATCH&limit=1', { credentials: 'same-origin' });
    const d = await r.json();
    if (d.status !== 'ok' || !d.patterns || d.patterns.length === 0) {
      // No DISPATCH in this Mempack — leave textbox in whatever state it had
      // and clear the status line.
      if (status && status.textContent.startsWith('most recent dispatch')) status.textContent = '';
      return;
    }
    const newest = d.patterns[0];
    let body = newest.text || '';
    if (body.startsWith('[') && body.indexOf(']') > 0) {
      body = body.slice(body.indexOf(']') + 1).trimStart();
    }
    ta.value = body;
    if (status) {
      status.className = 'saved-msg';
      status.textContent = 'most recent dispatch in this Mempack (idx:' + newest.idx + ')';
    }
  } catch(e) {
    // Silently fail — pre-fill is a convenience, not a load-bearing surface.
  }
}

async function doDispatch(){
  if (!_currentMempack) { toast('Select a Mempack first', 'error'); return; }
  const ta = $('#dispatchText');
  const status = $('#dispatchStatus');
  const text = (ta.value || '').trim();
  if (!text) { toast('Type a dispatch first', 'error'); ta.focus(); return; }
  status.textContent = 'Sending...';
  status.className = 'saved-msg';
  try {
    const r = await fetch(BASE() + '/api/mempack/' + _currentMempack.id + '/dispatch', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    if (d.status !== 'ok') {
      status.textContent = 'Error: ' + (d.error || 'unknown');
      status.className = 'saved-msg';
      return;
    }
    status.textContent = 'Dispatched (idx=' + d.new_idx + ', ' + fmtBytes(d.blob_bytes) + ') — text kept; edit or clear to send again';
    status.className = 'saved-msg success';
    // Text intentionally NOT cleared — so a repeat dispatch is one click, not
    // a re-type. User clears the textarea themselves when they're done.
    // Refresh quota meter on the selected card (size changed). Cheapest path
    // is a full mempack-list reload, but that re-renders cards and loses the
    // selection animation; instead, mutate in place + tick activity feed so
    // the user immediately sees the dispatch row appear.
    _currentMempack.pattern_count = d.pattern_count;
    _currentMempack.size_bytes    = d.blob_bytes;
    loadActivity(false);
  } catch(e) {
    status.textContent = 'Dispatch failed: ' + e.message;
    status.className = 'saved-msg';
  }
}

async function toggleLogging(enabled){
  if (!_currentMempack) return;
  try {
    const r = await fetch(BASE() + '/api/mempack/' + _currentMempack.id, {
      method: 'PATCH', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({activity_logging_enabled: !!enabled})
    });
    const d = await r.json();
    if (d.status !== 'ok') { toast('Toggle failed: ' + (d.error || 'unknown'), 'error'); return; }
    _currentMempack.activity_logging_enabled = !!enabled;
    toast('Activity logging ' + (enabled ? 'enabled' : 'disabled'), 'success');
  } catch(e) { toast('Toggle failed: ' + e.message, 'error'); }
}

async function loadActivity(reset){
  if (!_currentMempack) return;
  const feed = $('#activityFeed');
  let url = BASE() + '/api/mempack/' + _currentMempack.id + '/activity?limit=100';
  if (!reset && _activitySinceTs) url += '&since=' + encodeURIComponent(_activitySinceTs);
  try {
    const r = await fetch(url);
    const d = await r.json();
    if (d.status !== 'ok') { feed.innerHTML = '<div class="dash-empty">' + esc(d.error || 'Failed to load activity') + '</div>'; return; }
    _activitySinceTs = d.server_time;
    if (reset) {
      feed.innerHTML = (d.activity || []).length === 0
        ? '<div class="dash-empty"><div class="icon">&#x1F4DD;</div>No activity yet. Mount this Mempack from an agent to see events.</div>'
        : (d.activity || []).map(_renderActivityRow).join('');
    } else if (d.activity && d.activity.length > 0) {
      // Prepend newer events to the top of the feed
      feed.insertAdjacentHTML('afterbegin', d.activity.map(_renderActivityRow).join(''));
    }
  } catch(e) { /* polling — silent on error */ }
}

function _renderActivityRow(row){
  const meta = row.metadata || {};
  const preview = meta.preview || '';
  // Previous-version expander: present on pattern_i_update / pattern_update
  // events where we captured the prior text (Fix A — recoverability for
  // accidental or agent-driven overwrites).
  const prevText = meta.previous_text || '';
  const prevLen  = meta.previous_text_length || 0;
  const prevTrunc = !!meta.previous_text_truncated;
  let prevBlock = '';
  if (prevText && prevLen > 0) {
    const label = 'View previous version (' + prevLen + ' chars)';
    const truncNote = prevTrunc
      ? ('<span class="prev-version-truncated-note">'
        + 'NOTE: previous body truncated at 8 KB in this activity row '
        + '(full was ' + prevLen + ' chars). For the complete prior text, '
        + 'check earlier activity rows or restore from a manual backup.'
        + '</span>')
      : '';
    prevBlock =
      '<button type="button" class="prev-version-toggle" onclick="togglePrevVersion(this)">'
      + '<span class="caret">&#x25B8;</span> ' + esc(label)
      + '</button>'
      + '<div class="prev-version-block">'
      +   truncNote
      +   '<pre class="prev-version-text">' + esc(prevText) + '</pre>'
      +   '<div class="prev-version-actions">'
      +     '<button type="button" class="prev-version-copy" onclick="copyPrevVersion(this)">Copy previous text</button>'
      +   '</div>'
      + '</div>';
  }
  return '<div class="activity-row">'
    + '<div class="ts">' + esc(fmtTs(row.created_at)) + '</div>'
    + '<div class="body">'
    +   '<div class="summary">' + esc(row.summary || '') + '</div>'
    +   (preview ? '<div class="preview">' + esc(preview) + '</div>' : '')
    +   prevBlock
    + '</div>'
    + '<span class="type ' + esc(row.event_type) + '">' + esc(row.event_type) + '</span>'
    + '</div>';
}

function togglePrevVersion(btn){
  const block = btn.nextElementSibling;
  if (!block) return;
  const isOpen = block.classList.toggle('open');
  btn.classList.toggle('open', isOpen);
}

async function copyPrevVersion(btn){
  // Find the <pre> sibling that holds the previous text within this block.
  const block = btn.closest('.prev-version-block');
  if (!block) return;
  const pre = block.querySelector('.prev-version-text');
  if (!pre) return;
  const text = pre.textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    btn.classList.add('copied');
    const original = btn.textContent;
    btn.textContent = 'Copied to clipboard';
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.textContent = original;
    }, 2000);
  } catch(e) {
    // Fallback for restrictive contexts (file://, older browsers)
    const range = document.createRange();
    range.selectNode(pre);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    try {
      document.execCommand('copy');
      btn.classList.add('copied');
      btn.textContent = 'Copied (select-and-copy fallback)';
      setTimeout(() => { btn.classList.remove('copied'); btn.textContent = 'Copy previous text'; }, 2000);
    } catch(e2) {
      btn.textContent = 'Copy failed — text is selected, hit Ctrl+C';
    }
  }
}

function startActivityPoll(){
  stopActivityPoll();
  _activityPollHandle = setInterval(() => loadActivity(false), _ACTIVITY_POLL_MS);
}
function stopActivityPoll(){
  if (_activityPollHandle) { clearInterval(_activityPollHandle); _activityPollHandle = null; }
}
// Stop polling when tab loses visibility (saves bandwidth + battery)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopActivityPoll();
  else if (_currentMempack && document.querySelector('.view-tab.active')?.dataset.view === 'mempack') startActivityPoll();
});

/* =========================================================
 * Patterns Browser
 *   - Lists every stored pattern in the selected Mempack
 *   - Filter by tag (FINDING / SUMMARY / DISPATCH / METHOD / ...)
 *   - Substring text search (debounced)
 *   - Expand-to-read with Copy button
 *   - Paginated; "Load more" appends
 *   - Skips Pattern 0 (header marker) + Pattern I (handled by the editor above)
 * ========================================================= */
let _patternsOffset = 0;
let _patternsTotal = 0;
let _patternsCurrentTag = '';
let _patternsCurrentQ = '';
let _patternsSearchTimer = null;
const _PATTERNS_PAGE_SIZE = 50;

async function loadPatterns(reset){
  if (!_currentMempack) return;
  const list = $('#patternsList');
  const pag  = $('#patternsPagination');
  const cnt  = $('#patternsCount');
  if (reset) {
    _patternsOffset = 0;
    list.innerHTML = '<div class="dash-empty"><div class="spinner"></div> Loading patterns&hellip;</div>';
    if (cnt) cnt.textContent = '';
    if (pag) pag.style.display = 'none';
  }
  const params = new URLSearchParams();
  params.set('offset', String(_patternsOffset));
  params.set('limit',  String(_PATTERNS_PAGE_SIZE));
  if (_patternsCurrentTag) params.set('tag', _patternsCurrentTag);
  if (_patternsCurrentQ)   params.set('q',   _patternsCurrentQ);
  const url = BASE() + '/api/mempack/' + _currentMempack.id + '/patterns?' + params.toString();
  try {
    const r = await fetch(url);
    const d = await r.json();
    if (d.status !== 'ok') {
      list.innerHTML = '<div class="dash-empty"><div class="icon">&#x26A0;</div>' + esc(d.error || 'Load failed') + '</div>';
      return;
    }
    _patternsTotal = d.total || 0;
    const rows = (d.patterns || []).map(_renderPatternRow).join('');
    if (reset) {
      list.innerHTML = rows || '<div class="dash-empty"><div class="icon">&#x1F4DD;</div>No patterns match.</div>';
    } else {
      list.insertAdjacentHTML('beforeend', rows);
    }
    _patternsOffset = _patternsOffset + (d.patterns || []).length;
    if (cnt) {
      const filter_bits = [];
      if (_patternsCurrentTag) filter_bits.push(_patternsCurrentTag);
      if (_patternsCurrentQ)   filter_bits.push('"' + _patternsCurrentQ + '"');
      const filter_str = filter_bits.length ? ' (' + filter_bits.join(' + ') + ')' : '';
      cnt.textContent = 'Showing ' + _patternsOffset + ' of ' + _patternsTotal + filter_str;
    }
    if (pag) pag.style.display = (_patternsOffset < _patternsTotal) ? '' : 'none';
    // Show the download bar only when there's something downloadable.
    const dl = $('#patternsDownload');
    if (dl) dl.style.display = (_patternsTotal > 0) ? '' : 'none';
  } catch(e) {
    list.innerHTML = '<div class="dash-empty"><div class="icon">&#x26A0;</div>' + esc(e.message) + '</div>';
  }
}

function _renderPatternRow(p){
  const tagChips = (p.tags || []).map(t =>
    '<span class="pattern-tag-chip tag-' + esc(t) + '">' + esc(t) + '</span>'
  ).join('');
  const noTagFallback = (p.tags || []).length === 0
    ? '<span class="pattern-tag-chip">(no tags)</span>'
    : '';
  // Preview text: strip the leading "[TAG] " bracket so the inline preview
  // doesn't waste characters re-stating what the chips already show.
  let preview = p.preview || '';
  if (preview.startsWith('[') && preview.indexOf(']') > 0) {
    preview = preview.slice(preview.indexOf(']') + 1).trim();
  }
  return '<div class="pattern-row" data-idx="' + p.idx + '">'
    + '<div class="pattern-header" onclick="togglePatternBody(this)">'
    +   '<span class="pattern-idx">idx:' + p.idx + '</span>'
    +   '<span class="pattern-tags">' + tagChips + noTagFallback + '</span>'
    +   '<span class="pattern-length">' + p.length + ' chars</span>'
    +   '<span class="pattern-caret">&#x25B8;</span>'
    + '</div>'
    + '<div class="pattern-preview">' + esc(preview) + '</div>'
    + '<div class="pattern-body">'
    +   '<pre class="pattern-text">' + esc(p.text || '') + '</pre>'
    +   '<div class="pattern-actions">'
    +     '<button class="pattern-copy-btn" onclick="copyPatternBody(this)">Copy</button>'
    +     '<span class="pattern-actions-sep">&middot;</span>'
    +     '<span class="pattern-actions-label">Download:</span>'
    +     '<button class="pattern-download-btn pattern-row-dl-btn" data-fmt="md"   onclick="onPatternDl(this)">.md</button>'
    +     '<button class="pattern-download-btn pattern-row-dl-btn" data-fmt="txt"  onclick="onPatternDl(this)">.txt</button>'
    +     '<button class="pattern-download-btn pattern-row-dl-btn" data-fmt="docx" onclick="onPatternDl(this)">.docx</button>'
    +     '<button class="pattern-download-btn pattern-row-dl-btn" data-fmt="pdf"  onclick="onPatternDl(this)">.pdf</button>'
    +     '<span class="pattern-download-row-status"></span>'
    +   '</div>'
    + '</div>'
    + '</div>';
}

function onPatternDl(btn){
  // Tiny bridge: reads format + idx from data attrs to avoid the
  // single-quote string-collision that comes with inline arg literals.
  const fmt = btn.dataset.fmt;
  const row = btn.closest('.pattern-row');
  if (!row) return;
  const idx = parseInt(row.dataset.idx, 10);
  if (isNaN(idx)) return;
  downloadPatterns(fmt, idx);
}

function togglePatternBody(headerEl){
  const row = headerEl.closest('.pattern-row');
  if (!row) return;
  row.classList.toggle('open');
}

async function copyPatternBody(btn){
  const row = btn.closest('.pattern-row');
  if (!row) return;
  const pre = row.querySelector('.pattern-text');
  if (!pre) return;
  const text = pre.textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    btn.classList.add('copied');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.classList.remove('copied'); btn.textContent = 'Copy'; }, 2000);
  } catch(e) {
    // Fallback: select + execCommand
    const range = document.createRange();
    range.selectNode(pre);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    try {
      document.execCommand('copy');
      btn.classList.add('copied');
      btn.textContent = 'Copied (fallback)';
      setTimeout(() => { btn.classList.remove('copied'); btn.textContent = 'Copy'; }, 2000);
    } catch(e2) {
      btn.textContent = 'Select + Ctrl+C';
    }
    window.getSelection().removeAllRanges();
  }
}

function onPatternsTagChange(){
  _patternsCurrentTag = $('#patternsTagFilter').value;
  loadPatterns(true);
}

function onPatternsSearchInput(){
  clearTimeout(_patternsSearchTimer);
  _patternsSearchTimer = setTimeout(() => {
    _patternsCurrentQ = $('#patternsSearchInput').value.trim();
    loadPatterns(true);
  }, 300);
}

function loadMorePatterns(){
  loadPatterns(false);
}

/* Download Patterns Browser content as a document.
 *
 *   downloadPatterns('md')        — bulk: filtered set (current tag + q)
 *   downloadPatterns('md', 5)     — single pattern by idx
 *
 * Single-pattern mode wires status messages to the per-row status span
 * inside that row's .pattern-actions block; bulk mode uses the global
 * #patternsDownloadStatus span under the patterns list.
 */
async function downloadPatterns(format, idx){
  if (!_currentMempack) return;
  const isSingle = (idx !== undefined && idx !== null);

  // Locate the right status element + button group for this scope.
  let statusEl, btns;
  if (isSingle) {
    const row = document.querySelector('.pattern-row[data-idx="' + idx + '"]');
    if (!row) return;
    statusEl = row.querySelector('.pattern-download-row-status');
    btns = row.querySelectorAll('.pattern-row-dl-btn, .pattern-copy-btn');
  } else {
    statusEl = $('#patternsDownloadStatus');
    btns = document.querySelectorAll('.patterns-download-btn');
  }

  const params = new URLSearchParams();
  params.set('format', format);
  if (isSingle) {
    params.set('idx', String(idx));
  } else {
    if (_patternsCurrentTag) params.set('tag', _patternsCurrentTag);
    if (_patternsCurrentQ)   params.set('q',   _patternsCurrentQ);
  }
  const url = BASE() + '/api/mempack/' + _currentMempack.id + '/patterns/export?' + params.toString();

  // Disable buttons while in flight (PDF can take 2-5 seconds)
  btns.forEach(b => { b.disabled = true; });
  if (statusEl) {
    statusEl.className = isSingle ? 'pattern-download-row-status' : 'patterns-download-status';
    statusEl.textContent = format === 'pdf' ? 'Rendering PDF…' : 'Preparing…';
  }

  try {
    const r = await fetch(url, { credentials: 'same-origin' });
    if (!r.ok) {
      let detail = 'HTTP ' + r.status;
      try { const errd = await r.json(); if (errd && errd.error) detail = errd.error; } catch(_) {}
      if (statusEl) {
        statusEl.className = (isSingle ? 'pattern-download-row-status' : 'patterns-download-status') + ' error';
        statusEl.textContent = 'Download failed: ' + detail;
      }
      return;
    }
    const blob = await r.blob();
    const cd = r.headers.get('content-disposition') || '';
    const fnMatch = cd.match(/filename="([^"]+)"/);
    const filename = fnMatch ? fnMatch[1] : ('mempack.' + format);
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1500);
    if (statusEl) {
      statusEl.className = (isSingle ? 'pattern-download-row-status' : 'patterns-download-status') + ' success';
      statusEl.textContent = 'Saved as ' + filename;
      setTimeout(() => {
        if (statusEl.textContent.startsWith('Saved as ')) {
          statusEl.className = isSingle ? 'pattern-download-row-status' : 'patterns-download-status';
          statusEl.textContent = '';
        }
      }, 4000);
    }
  } catch(e) {
    if (statusEl) {
      statusEl.className = (isSingle ? 'pattern-download-row-status' : 'patterns-download-status') + ' error';
      statusEl.textContent = 'Download error: ' + e.message;
    }
  } finally {
    btns.forEach(b => { b.disabled = false; });
  }
}

// Restore last view on page load (UUID auto-detected from Supabase cookie inside setView)
(function(){
  const lastView = localStorage.getItem('membot-active-view');
  if (lastView === 'mempack') setView('mempack');
})();

/* Cookie-disappearance / user-change watcher.
 *
 * Polls /api/whoami every 10s. If the detected UUID changes between polls
 * (user signed out at project-you.app, or signed in as a different user),
 * clear local Mempack state and re-enter the view so the empty-state /
 * sign-in prompt renders for the new auth context.
 *
 * Critical for closing the "logged in then out, dashboard stale" leak from
 * the user-facing side (server-side leak is closed via per-user session_id).
 */
let _lastDetectedUser = null;
const _USER_POLL_MS = 10000;
async function _userWatchTick(){
  // Bypass the whoami cache so the poll always sees fresh server-side state
  _whoamiCache = { ts: 0, whoami: null };
  const whoami = await getWhoami();
  const now = whoami && whoami.user_id ? whoami.user_id : null;
  const override = localStorage.getItem('membot-owner-override');
  const effective = override || now;
  const lastEffective = override || _lastDetectedUser;
  // Always keep the auth chip in sync with current state.
  renderAuthChip(whoami);
  if (effective !== lastEffective) {
    console.log('[membot] auth state changed:', lastEffective ? lastEffective.slice(0,8) : 'anon',
                '->', effective ? effective.slice(0,8) : 'anon');
    _lastDetectedUser = now;
    _currentMempack = null;
    stopActivityPoll();
    $('#mempackList').innerHTML = '';
    $('#mempackDetail').style.display = 'none';
    $('#ownerUuid').value = '';
    const activeTab = document.querySelector('.view-tab.active')?.dataset.view;
    if (activeTab === 'mempack') setView('mempack');
    if (!effective && lastEffective) toast('Signed out — Mempack dashboard cleared', 'success');
  } else {
    _lastDetectedUser = now;
  }
}
// Prime the cache once at startup so the first poll has a baseline
detectSupabaseUser().then(uid => { _lastDetectedUser = uid; });
setInterval(_userWatchTick, _USER_POLL_MS);
