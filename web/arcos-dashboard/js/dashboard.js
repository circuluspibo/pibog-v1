/**
 * ARCos Integrated Intelligence Dashboard — dashboard.js v4.0
 * ─────────────────────────────────────────────────────────────────
 * Handles:
 *  • Heartbeat polling → state update
 *  • OD canvas rendering (bounding boxes)
 *  • LIDAR radar animation
 *  • Navigation map (embeds FleetMap from map.js if present)
 *  • Chat with /v2/img2chat (streaming SSE)
 *  • TTS via /v1/tts
 *  • Voice input via Web Speech API
 */

'use strict';

const q  = (sel, fn) => { const el = document.querySelector(sel); if (el && fn) fn(el); return el; };
const qs = (sel) => [...document.querySelectorAll(sel)];

// ── Config ─────────────────────────────────────────────────────────
const CFG = {
  BASE_URL:       '',
  AGENT_URL:      `http://${location.hostname}:59532`,
  GATEWAY_URL:    `http://192.168.12.19:3001`,   // MQTT Gateway Socket.IO
  HEARTBEAT_URL:  '/heartbeat',
  CHAT_URL:       '/v1/rag/txt2chat',
  TTS_URL:        '/v1/tts',
  HB_INTERVAL:    800,
  SLAM_CLI:       'slam_cli',
};

// ── State ──────────────────────────────────────────────────────────
let robotState = null;
let hbTimer    = null;
let mapInst    = null;         // FleetMap instance

// ── Clock ──────────────────────────────────────────────────────────
(function clock() {
  function tick() {
    const now  = new Date();
    const hh   = String(now.getHours()).padStart(2,'0');
    const mm   = String(now.getMinutes()).padStart(2,'0');
    const ss   = String(now.getSeconds()).padStart(2,'0');
    const days = ['SUN','MON','TUE','WED','THU','FRI','SAT'];
    q('#clock').textContent = `${hh}:${mm}:${ss}`;
    q('#clock-date').textContent = `${days[now.getDay()]} ${now.getFullYear()}.${String(now.getMonth()+1).padStart(2,'0')}.${String(now.getDate()).padStart(2,'0')}`;
  }
  tick();
  setInterval(tick, 1000);
})();

// ── Heartbeat ──────────────────────────────────────────────────────
async function pollHeartbeat() {
  try {
    const res  = await fetch(CFG.BASE_URL + CFG.HEARTBEAT_URL, { signal: AbortSignal.timeout(1500) });
    if (!res.ok) throw new Error('non-ok');
    const json = await res.json();
    if (json.result && json.data) {
      robotState = json.data;
      updateUI(robotState);
      q('#hb-dot').className = 'hst-dot alive';
      q('#hb-lbl').textContent = 'HEARTBEAT';
      q('#sys-badge').textContent = 'ONLINE';
    }
  } catch {
    q('#hb-dot').className = 'hst-dot dead';
    q('#hb-lbl').textContent = 'NO SIGNAL';
    q('#sys-badge').textContent = 'OFFLINE';
  }
  hbTimer = setTimeout(pollHeartbeat, CFG.HB_INTERVAL);
}

// ── OD Canvas ──────────────────────────────────────────────────────
// boxes format: [{label, conf, x1, y1, x2, y2}] OR [{label, conf, box:[x1,y1,x2,y2]}]
const OD_COLORS = ['#00ffe7','#ff6b00','#39ff14','#ffd600','#e040fb','#00b4ff'];

function renderODCanvas(boxes) {
  const canvas = q('#od-canvas');
  if (!canvas) return;
  const wrap   = canvas.parentElement;
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  const ctx = canvas.getContext('2d');
  const W   = canvas.width;
  const H   = canvas.height;

  // Dark background grid
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(0,255,231,0.06)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 20) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

  if (!boxes || boxes.length === 0) {
    ctx.fillStyle = 'rgba(0,255,231,0.15)';
    ctx.font = '9px "Share Tech Mono"';
    ctx.textAlign = 'center';
    ctx.fillText('NO DETECTIONS', W/2, H/2);
    return;
  }

  boxes.forEach((b, i) => {
    const color = OD_COLORS[i % OD_COLORS.length];
    // normalised coords expected 0..1; if raw pixel, pass img dims
    let x1, y1, x2, y2;
    if (b.box) { [x1, y1, x2, y2] = b.box; }
    else       { x1=b.x1; y1=b.y1; x2=b.x2; y2=b.y2; }

    // Assume 0..1 normalised
    const px1 = x1 * W, py1 = y1 * H;
    const pw  = (x2 - x1) * W, ph = (y2 - y1) * H;

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = color; ctx.shadowBlur = 6;
    ctx.strokeRect(px1, py1, pw, ph);
    ctx.shadowBlur = 0;

    // Corner marks
    const cs = 6;
    ctx.lineWidth = 2;
    [[px1,py1,1,1],[px1+pw,py1,-1,1],[px1,py1+ph,1,-1],[px1+pw,py1+ph,-1,-1]].forEach(([cx,cy,sx,sy]) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy + sy*cs); ctx.lineTo(cx, cy); ctx.lineTo(cx + sx*cs, cy);
      ctx.stroke();
    });

    // Label
    const label = `${b.label || '?'} ${b.conf != null ? (b.conf * 100).toFixed(0) + '%' : ''}`;
    ctx.font = '9px "Share Tech Mono"';
    ctx.fillStyle = color;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(px1, py1 - 14, tw + 6, 14);
    ctx.fillStyle = '#000';
    ctx.fillText(label, px1 + 3, py1 - 3);
  });
}

function renderODList(boxes) {
  const list = q('#od-list');
  if (!list) return;
  if (!boxes || boxes.length === 0) {
    list.innerHTML = '<div class="od-empty">NO OBJECTS DETECTED</div>';
    return;
  }
  list.innerHTML = boxes.map((b, i) => {
    const color = OD_COLORS[i % OD_COLORS.length];
    const conf  = b.conf != null ? (b.conf * 100).toFixed(0) + '%' : '—';
    return `<div class="od-item">
      <div class="od-item-dot" style="background:${color};box-shadow:0 0 5px ${color}"></div>
      <span class="od-item-label">${b.label || '?'}</span>
      <span class="od-item-conf">${conf}</span>
    </div>`;
  }).join('');
}

// ── LIDAR Radar ────────────────────────────────────────────────────
(function initLidar() {
  const canvas = q('#lidar-canvas');
  if (!canvas) return;
  const ctx    = canvas.getContext('2d');
  let angle    = 0;
  let points   = [];  // [{r, a}] — filled by data feed

  // Demo points — replace with actual slam_cli lidar feed
  function genDemoPoints() {
    const pts = [];
    for (let i = 0; i < 120; i++) {
      const a = (i / 120) * Math.PI * 2;
      const r = 0.2 + Math.random() * 0.6 + Math.sin(a * 3) * 0.15;
      pts.push({ a, r, age: 0 });
    }
    return pts;
  }
  points = genDemoPoints();

  function resize() {
    const wrap = canvas.parentElement;
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    const W = canvas.width, H = canvas.height;
    const cx = W/2, cy = H/2;
    const R  = Math.min(W, H) / 2 - 4;

    ctx.clearRect(0,0,W,H);

    // Background
    ctx.fillStyle = '#000';
    ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.fill();

    // Concentric rings
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath(); ctx.arc(cx,cy,R*i/4,0,Math.PI*2);
      ctx.strokeStyle = `rgba(0,255,231,${0.07 * i})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Cross hairs
    ctx.strokeStyle = 'rgba(0,255,231,0.06)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cx-R,cy); ctx.lineTo(cx+R,cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx,cy-R); ctx.lineTo(cx,cy+R); ctx.stroke();

    // Sweep gradient
    const sweep = ctx.createConicalGradient
      ? null  // non-standard, skip
      : null;

    // Draw sweep as a wedge
    const sweepLen = Math.PI / 6;
    const g = ctx.createRadialGradient(cx,cy,0,cx,cy,R);
    g.addColorStop(0, 'rgba(0,255,231,0.3)');
    g.addColorStop(1, 'rgba(0,255,231,0.02)');

    ctx.save();
    ctx.translate(cx,cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0,0);
    ctx.arc(0,0,R,-sweepLen,0);
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();

    // Sweep line
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(R,0);
    ctx.strokeStyle = 'rgba(0,255,231,0.9)';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#00ffe7'; ctx.shadowBlur = 4;
    ctx.stroke(); ctx.shadowBlur = 0;
    ctx.restore();

    // Points
    points.forEach(p => {
      const a   = p.a;
      const r   = p.r * R;
      const px  = cx + Math.cos(a) * r;
      const py  = cy + Math.sin(a) * r;

      // Age-based fade  (how far behind the sweep)
      let diff = ((a - angle) % (Math.PI*2) + Math.PI*2) % (Math.PI*2);
      const alpha = Math.max(0, 1 - diff / (Math.PI*2));

      ctx.beginPath();
      ctx.arc(px,py,2,0,Math.PI*2);
      ctx.fillStyle = `rgba(0,255,231,${alpha * 0.85})`;
      ctx.shadowColor = '#00ffe7'; ctx.shadowBlur = alpha * 5;
      ctx.fill(); ctx.shadowBlur = 0;
    });

    // Border circle
    ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
    ctx.strokeStyle = 'rgba(0,255,231,0.2)'; ctx.lineWidth = 1; ctx.stroke();

    angle += 0.025;
    if (angle > Math.PI * 2) angle -= Math.PI * 2;
    requestAnimationFrame(draw);
  }
  draw();

  // Expose for data injection
  window.lidarSetPoints = (pts) => { points = pts; };
})();

// ── Navigation Map ─────────────────────────────────────────────────
(function initNavMap() {
  if (typeof FleetMap === 'undefined') return;
  try {
    mapInst = new FleetMap('map-canvas');

    // Controls
    q('#nmc-grid',  el => el.addEventListener('click', () => {
      mapInst.showGrid = !mapInst.showGrid; el.classList.toggle('active');
    }));
    q('#nmc-trail', el => el.addEventListener('click', () => {
      mapInst.showTrails = !mapInst.showTrails; el.classList.toggle('active');
    }));
    q('#nmc-2d5',   el => el.addEventListener('click', () => {
      mapInst.tilt = mapInst.tilt > 0 ? 0 : 1; el.classList.toggle('active');
      const tc = q('#tilt-ctrl');
      if (tc) tc.style.display = mapInst.tilt > 0 ? 'flex' : 'none';
    }));
    q('#nmc-fit',   el => el.addEventListener('click', () => {
      if (mapInst.panX !== undefined) { mapInst.panX = 0; mapInst.panY = 0; mapInst.zoom = 50; }
    }));
    q('#nmc-r0',    el => el.addEventListener('click', () => { mapInst.rotation = 0;              }));
    q('#nmc-r90',   el => el.addEventListener('click', () => { mapInst.rotation = Math.PI/2;      }));
    q('#nmc-r180',  el => el.addEventListener('click', () => { mapInst.rotation = Math.PI;        }));
    q('#nmc-r270',  el => el.addEventListener('click', () => { mapInst.rotation = 3*Math.PI/2;    }));

    // Update position display
    setInterval(() => {
      const robots = Object.values(mapInst.robots || {});
      if (robots.length > 0) {
        const r = robots[0];
        const p = r.position;
        if (p) {
          q('#nm-coords').textContent = `${p.x.toFixed(2)} · ${p.y.toFixed(2)}`;
          q('#ns-x').textContent      = p.x.toFixed(3);
          q('#ns-y').textContent      = p.y.toFixed(3);
          if (p.heading != null)
            q('#ns-hdg').textContent  = (p.heading * 180 / Math.PI).toFixed(1) + '°';
        }
        q('#ns-mode').textContent = r.status ? r.status.toUpperCase() : 'IDLE';
      }
    }, 500);
  } catch(e) {
    console.warn('FleetMap init failed:', e);
  }
})();

// ── MQTT Gateway Socket.IO 연결 ─────────────────────────────────────────────────
// Gateway(:3001)에 연결해 실시간 텔레메트리 수신
// fleet server Socket.IO와 병렬 운영 — 둘 다 받아서 최신값 우선 적용
let gwSocket = null;
let gwOnline = false;

function connectGateway() {
  // socket.io 라이브러리는 vendor/socket.io.min.js 또는 cdn에서 로드
  if (typeof io === 'undefined') return;
  try {
    gwSocket = io(CFG.GATEWAY_URL, {
      transports: ['websocket', 'polling'],
      reconnectionDelay: 3000,
      timeout: 5000,
    });

    gwSocket.on('connect', () => {
      gwOnline = true;
      q('#gw-indicator')?.classList.add('online');
      console.log('[gateway] Connected to MQTT Gateway');
    });

    gwSocket.on('disconnect', () => {
      gwOnline = false;
      q('#gw-indicator')?.classList.remove('online');
      console.log('[gateway] Disconnected from MQTT Gateway');
    });

    // 로봇 목록 수신 (Gateway 등록 정보)
    gwSocket.on('gateway:robots', (robots) => {
      console.log(`[gateway] ${robots.length} robots registered`);
    });

    // 위치 — fleet server와 동일한 이벤트명, 동일하게 처리
    gwSocket.on('robot:position', ({ robotId, position }) => {
      if (!position) return;
      robotState = { ...robotState, position };
      // 맵 업데이트
      if (mapInst && position) {
        if (!mapInst.robots) mapInst.robots = {};
        mapInst.robots[robotId] = { ...(mapInst.robots[robotId]||{}), position };
        q('#nm-coords').textContent = `${position.x.toFixed(2)} · ${position.y.toFixed(2)}`;
        q('#ns-x').textContent      = position.x.toFixed(3);
        q('#ns-y').textContent      = position.y.toFixed(3);
      }
    });

    // ctrl_info — 네비게이션 상태 실시간 갱신
    gwSocket.on('robot:ctrl_info', ({ robotId, ctrlInfo }) => {
      if (!ctrlInfo) return;
      updateCtrlInfo(ctrlInfo);
    });

    // robot_data → sysinfo → updateRobotData
    gwSocket.on('robot:sysinfo', ({ robotId, sysinfo }) => {
      if (!sysinfo) return;
      robotState = { ...robotState, sysinfo };
      updateRobotData(sysinfo);
      // 기존 updateUI bars도 갱신
      if (sysinfo.battery != null || sysinfo.batteryPower != null) {
        const pct = sysinfo.batteryPower ?? sysinfo.battery ?? 0;
        const el = q('#bar-charge'); if (el) el.style.width = `${Math.min(100,pct)}%`;
        const vl = q('#val-charge'); if (vl) vl.textContent = pct.toFixed(0) + '%';
      }
    });

    // 장애물
    gwSocket.on('robot:obstacle', (data) => {
      const { robotId, type, obsAccumSec } = data;
      const indicator = q('#obs-indicator');
      if (type === 'detected' || type === 'ongoing') {
        indicator?.classList.add('active');
        robotState = { ...robotState, obstacle: { active: true, accTime: obsAccumSec || 0 } };
      } else if (type === 'cleared') {
        indicator?.classList.remove('active');
        robotState = { ...robotState, obstacle: { active: false } };
      }
    });

  } catch (e) {
    console.warn('[gateway] Connection failed:', e);
  }
}

// Gateway 연결 시도 (로드 후 2초 대기 — fleet server 연결 우선)
setTimeout(connectGateway, 2000);

// Gateway 상태 인디케이터 업데이트
setInterval(() => {
  const el = q('#gw-indicator');
  if (el) el.title = gwOnline ? 'MQTT Gateway: 연결됨' : 'MQTT Gateway: 오프라인';
}, 3000);

// ── Waypoints (from fleet server) ───────────────────────────────────
async function loadWaypoints() {
  try {
    const res = await fetch('http://192.168.12.19:3000/api/waypoints');
    if (!res.ok) return;
    const data = await res.json();
    renderWaypointList(data);
  } catch {}
}

function renderWaypointList(waypoints) {
  // nav-status-bar의 waypoint 목록 업데이트
  const bar = q('#ns-wp-list-bar');
  const entries = Object.entries(waypoints || {});
  if (bar) {
    bar.innerHTML = entries.slice(0, 6).map(([name, wp]) =>
      `<button class="nsb-wp-btn" onclick="gotoWP('${name}')" title="${name} (${wp.x?.toFixed(1)||'?'},${wp.y?.toFixed(1)||'?'})">${name}</button>`
    ).join('');
  }

  // 사이드 nav-strip 업데이트 (있으면)
  const list = q('#ns-wp-list');
  if (list) {
    list.innerHTML = entries.length === 0
      ? '<div class="ns-wp-empty">No waypoints</div>'
      : entries.map(([name, wp]) =>
          `<div class="ns-wp-item" data-wp="${name}">
            <span>${name}</span>
            <span style="font-size:8px;color:var(--tx2)">${(wp.x||0).toFixed(1)}, ${(wp.y||0).toFixed(1)}</span>
            <button class="ns-wp-btn" onclick="gotoWP('${name}')">→</button>
          </div>`
        ).join('');
  }
}

window.gotoWP = async (name) => {
  // fleet server에 등록된 첫 번째 로봇 ID 사용
  try {
    const fleet = await fetch('http://192.168.12.19:3000/api/fleet').then(r => r.json());
    if (!fleet.length) return;
    const robotId = fleet[0].id;
    // GET v2 API 사용
    await fetch(`http://192.168.12.19:3000/api/v2/robot/${robotId}/goto?poseName=${encodeURIComponent(name)}`);
  } catch (e) { console.warn('gotoWP failed:', e); }
};

loadWaypoints();
// 30초마다 웨이포인트 갱신
setInterval(loadWaypoints, 30000);

// ── Nav 빠른 명령 버튼 (fleet server API 연결) ──────────────────────
async function getFirstRobotId() {
  try {
    const fl = await fetch('http://192.168.12.19:3000/api/fleet').then(r => r.json());
    return fl.length ? fl[0].id : null;
  } catch { return null; }
}

q('#ncb-estop',  el => el.addEventListener('click', async () => {
  const id = await getFirstRobotId(); if (!id) return;
  fetch(`http://192.168.12.19:3000/api/v2/robot/${id}/stop`);
}));
q('#ncb-home',   el => el.addEventListener('click', async () => {
  // 등록된 첫 웨이포인트로 이동 (또는 별도 home 웨이포인트)
  const id = await getFirstRobotId(); if (!id) return;
  const wps = await fetch('http://192.168.12.19:3000/api/waypoints').then(r=>r.json()).catch(()=>({}));
  const homeName = wps['home'] ? 'home' : Object.keys(wps)[0];
  if (homeName) fetch(`http://192.168.12.19:3000/api/v2/robot/${id}/goto?poseName=${encodeURIComponent(homeName)}`);
}));
q('#ncb-pause',  el => el.addEventListener('click', async () => {
  const id = await getFirstRobotId(); if (!id) return;
  fetch(`http://192.168.12.19:3000/api/v2/robot/${id}/pause`);
}));
q('#ncb-resume', el => el.addEventListener('click', async () => {
  const id = await getFirstRobotId(); if (!id) return;
  fetch(`http://192.168.12.19:3000/api/v2/robot/${id}/resume`);
}));

// ── Chat ────────────────────────────────────────────────────────────
let isStreaming = false;

const chatEl   = q('#chat-messages');
const inputEl  = q('#chat-input');
const sendBtn  = q('#send-btn');

function addMessage(role, text, extra = {}) {
  const div    = document.createElement('div');
  div.className = `chat-msg ${role}`;

  const ts = new Date().toLocaleTimeString('ko-KR', {hour:'2-digit',minute:'2-digit'});
  div.innerHTML = `
    <div class="cm-avatar">${role === 'agent' ? '⬡' : '◈'}</div>
    <div class="cm-bubble">
      <div class="cm-text ${extra.streaming ? 'streaming' : ''}">${text}</div>
      <div class="cm-time">${ts}</div>
    </div>`;

  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div.querySelector('.cm-text');
}

async function sendMessage(text) {
  if (!text.trim() || isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;

  addMessage('user', text);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  const bubbleEl = addMessage('agent', '', { streaming: true });
  let fullText   = '';

  // 로봇 상태 컨텍스트 주입
  const stateCtx = robotState
    ? `[로봇상태: 배터리=${robotState.charge}%, 온도=${robotState.temp}°C, ` +
      `감지객체=${robotState.cnt_object}, 생물=${robotState.cnt_live}, ` +
      `인물=${JSON.stringify(robotState.human)}] `
    : '';

  const fullPrompt = stateCtx + text;
  const lang = q('#tts-lang')?.value || 'ko';
  const isPlay = q('#tts-toggle')?.classList.contains('on') ? 1 : 0;

  // /v1/rag/txt2chat GET 스트리밍 — generate() 패턴
  const url = `${CFG.AGENT_URL}${CFG.CHAT_URL}?prompt=${encodeURIComponent(fullPrompt)}&lang=${lang}&isPlay=${isPlay}`;

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let sentenceBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });

      // playNext — TTS 청크 단위 처리
      if (isPlay && chunk.trim()) playNext(chunk);

      // 텍스트 누적 및 UI 업데이트
      fullText += chunk;
      bubbleEl.textContent = fullText;
      chatEl.scrollTop = chatEl.scrollHeight;

      // 클라이언트 TTS (isPlay=0일 때만 sentence 단위로)
      if (!isPlay && q('#tts-toggle')?.classList.contains('on')) {
        sentenceBuffer += chunk;
        if (/[.!?。！？\n]/.test(sentenceBuffer)) {
          triggerTTS(sentenceBuffer.trim());
          sentenceBuffer = '';
        }
      }
    }

    // 잔여 TTS 버퍼 플러시
    if (!isPlay && sentenceBuffer.trim() && q('#tts-toggle')?.classList.contains('on')) {
      triggerTTS(sentenceBuffer.trim());
    }

  } catch (err) {
    bubbleEl.textContent = `[연결 오류: ${err.message}]`;
    console.error('[Chat] error:', err);
  }

  bubbleEl.classList.remove('streaming');
  isStreaming = false;
  sendBtn.disabled = false;
}

// playNext — 서버 isPlay=1 시 오디오 청크 재생 (base64 또는 URL)
const playQueue = [];
let playPlaying = false;

function playNext(chunk) {
  // chunk가 base64 오디오이면 재생, 아니면 TTS 큐에 넣기
  if (!chunk || !chunk.trim()) return;
  // chunk가 순수 텍스트인 경우 — triggerTTS로 처리
  if (chunk.length < 500 && !/^[A-Za-z0-9+/=]+$/.test(chunk.trim())) {
    // 텍스트 조각 — sentence 감지 후 TTS
    return;
  }
  // base64 오디오 데이터인 경우
  try {
    const binary = atob(chunk.trim());
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/wav' });
    const url  = URL.createObjectURL(blob);
    playQueue.push(url);
    if (!playPlaying) drainPlayQueue();
  } catch { /* 텍스트 청크 무시 */ }
}

function drainPlayQueue() {
  if (!playQueue.length) { playPlaying = false; return; }
  playPlaying = true;
  const url   = playQueue.shift();
  const audio = new Audio(url);
  audio.onended = () => { URL.revokeObjectURL(url); drainPlayQueue(); };
  audio.onerror = () => drainPlayQueue();
  audio.play().catch(() => drainPlayQueue());
}

// ── TTS ─────────────────────────────────────────────────────────────
const ttsQueue = [];
let ttsPlaying = false;

async function triggerTTS(text) {
  if (!text.trim()) return;
  ttsQueue.push(text);
  if (!ttsPlaying) playNextTTS();
}

async function playNextTTS() {
  if (ttsQueue.length === 0) { ttsPlaying = false; return; }
  ttsPlaying = true;
  const text  = ttsQueue.shift();
  const voice = q('#tts-voice').value || 31;
  const lang  = q('#tts-lang').value  || 'ko';
  try {
    const url  = `${CFG.BASE_URL}${CFG.TTS_URL}?text=${encodeURIComponent(text)}&voice=${voice}&lang=${lang}&isPlay=0`;
    const res  = await fetch(url);
    if (res.ok) {
      const blob = await res.blob();
      const audio = new Audio(URL.createObjectURL(blob));
      audio.onended = playNextTTS;
      audio.play();
    } else { playNextTTS(); }
  } catch { playNextTTS(); }
}

// TTS toggle
q('#tts-toggle', el => {
  el.addEventListener('click', () => {
    const on = el.classList.toggle('on');
    el.dataset.on = on;
  });
});

// Send button
sendBtn.addEventListener('click', () => sendMessage(inputEl.value));
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(inputEl.value); }
});

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 80) + 'px';
});

// ── Voice Input ──────────────────────────────────────────────────────
(function initVoice() {
  const micBtn  = q('#mic-btn');
  const micIcon = q('#mic-icon');
  if (!micBtn) return;

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { micBtn.title = '이 브라우저는 음성 인식을 지원하지 않습니다.'; micBtn.style.opacity='0.4'; return; }

  const rec  = new SR();
  rec.lang   = 'ko-KR';
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  let listening = false;

  micBtn.addEventListener('click', () => {
    if (listening) { rec.stop(); }
    else           { rec.start(); }
  });

  rec.onstart = () => {
    listening = true;
    micBtn.classList.add('recording');
    micIcon.textContent = '🔴';
  };
  rec.onend = () => {
    listening = false;
    micBtn.classList.remove('recording');
    micIcon.textContent = '🎙';
  };
  rec.onresult = e => {
    const transcript = e.results[e.results.length-1][0].transcript;
    inputEl.value = transcript;
    if (e.results[e.results.length-1].isFinal) {
      sendMessage(transcript);
    }
  };
  rec.onerror = () => { listening = false; micBtn.classList.remove('recording'); micIcon.textContent = '🎙'; };
})();

// ── Image feed error handling ───────────────────────────────────────
function initFeedImg(imgId, nosigId) {
  const img = q(`#${imgId}`);
  const nos = q(`#${nosigId}`);
  if (!img || !nos) return;
  img.addEventListener('load',  () => { nos.style.display = 'none'; });
  img.addEventListener('error', () => { nos.style.display = 'flex'; img.style.display = 'none'; });
  if (img.complete && img.naturalWidth > 0) nos.style.display = 'none';
}
initFeedImg('img-feed1', 'nosig-1');
initFeedImg('img-depth', 'nosig-6');

// ── 수직 드래그 리사이저 (SLAM맵 ↔ 스트림 영역) ────────────────────
(function initHResizer() {
  const resizer     = document.querySelector('#h-resizer');
  const slamSec     = document.querySelector('.slam-section');
  const streamsSec  = document.querySelector('.streams-section');
  const centerPanel = document.querySelector('.panel-center');
  if (!resizer || !slamSec || !streamsSec || !centerPanel) return;

  // 저장된 높이 복원
  const saved = sessionStorage.getItem('streams-height');
  if (saved) {
    streamsSec.style.flex = `0 0 ${saved}px`;
  }

  let dragging = false;
  let startY   = 0;
  let startH   = 0;

  resizer.addEventListener('mousedown', e => {
    dragging = true;
    startY   = e.clientY;
    startH   = streamsSec.getBoundingClientRect().height;
    resizer.classList.add('dragging');
    document.body.style.cursor     = 'row-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const totalH  = centerPanel.getBoundingClientRect().height;
    const statusH = 38; // nav-status-bar
    const resizerH = 6;
    const minStream = 100;
    const minSlam   = 120;
    const maxStream = totalH - minSlam - statusH - resizerH;
    const delta  = startY - e.clientY;   // 위로 드래그 → 스트림 높아짐
    const newH   = Math.min(maxStream, Math.max(minStream, startH + delta));
    streamsSec.style.flex = `0 0 ${newH}px`;
    sessionStorage.setItem('streams-height', Math.round(newH));
    window.dispatchEvent(new Event('resize'));
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.style.cursor     = '';
    document.body.style.userSelect = '';
    window.dispatchEvent(new Event('resize'));
  });
})();

// ════════════════════════════════════════════════════════════════════════════
// WebRTC  —  참고 코드(robot_monitor.html) 기반 완전 재구현
// 포트:  :59530 WebRTC (main SEG+PPE · depth)
//        :59531 MJPEG CAM PRIMARY
// DataChannel "state" → updateUI(data) / showCapture(data)
// ════════════════════════════════════════════════════════════════════════════
const WEBRTC_BASE = `http://${location.hostname}:59530`;
const CAM_URL     = `http://${location.hostname}:59531/video_feed`;
const LIVING_SET  = new Set(['person','cat','dog','bird','teddy bear','cow','sheep','horse']);

// ── CAM MJPEG 초기화 ──────────────────────────────────────────────────────
(function initCam() {
  // img-feed1, img-cam 두 ID 모두 대응
  const camEl = document.querySelector('#img-feed1') || document.querySelector('#img-cam');
  if (!camEl) return;
  camEl.src    = CAM_URL;
  camEl.onerror = () => {
    camEl.style.display = 'none';
    const ns = document.querySelector('#nosig-1');
    if (ns) ns.style.display = 'flex';
  };
  camEl.onload = () => {
    camEl.style.display = 'block';
    const ns = document.querySelector('#nosig-1');
    if (ns) ns.style.display = 'none';
  };
})();

// ── 상태 변수 ─────────────────────────────────────────────────────────────
let pc         = null;
let connected  = false;
let clientId   = null;
let trackCount = 0;
let dcMsgTotal = 0, dcMsgPrev = 0;
let statsTimer = null, dcRateTimer = null;
const dcLines  = [];

// ── 공개 토글 ─────────────────────────────────────────────────────────────
function toggleWebRTC() { connected ? disconnect() : connect(); }
window.toggleWebRTC = toggleWebRTC;

// ── 상태 표시 ─────────────────────────────────────────────────────────────
function setWrtcStatus(s) {
  const dot = q('#wrtc-dot');
  const lbl = q('#wrtc-lbl');
  const btn = q('#wrtc-btn');
  if (dot) dot.className = 'wrtc-dot ' + s;
  if (lbl) lbl.textContent = s.toUpperCase();
  if (!btn) return;
  if (s === 'connected') {
    connected = true;  btn.textContent = 'DISCONNECT'; btn.disabled = false;
  } else if (s === 'connecting') {
    btn.textContent = '···'; btn.disabled = true;
  } else {
    connected = false; btn.textContent = 'CONNECT';    btn.disabled = false;
  }
}

// ── connect ───────────────────────────────────────────────────────────────
async function connect() {
  setWrtcStatus('connecting');
  clientId   = crypto.randomUUID();
  trackCount = 0;

  pc = new RTCPeerConnection({ iceServers: [] });

  // ── 트랙 수신: e.transceiver.mid 기반 (참고 코드와 동일한 방식)
  // 서버 addTrack 순서: mid=0 → main SEG+PPE, mid=1 → depth
  pc.ontrack = (e) => {
    const mid = e.transceiver ? e.transceiver.mid : String(trackCount);
    const idx = parseInt(mid, 10);
    // 독립 MediaStream으로 감싸서 두 트랙이 서로 간섭 없도록
    const stream = new MediaStream([e.track]);
    console.log(`[WebRTC] ontrack mid=${mid} idx=${idx} track=${e.track.kind}`);

    if (idx === 0) {
      // SEG+PPE 메인 스트림 → #vm
      const vm = q('#vm');
      if (vm) { vm.srcObject = stream; }
      const ns = q('#nsm');
      if (ns) ns.style.display = 'none';
      setBadge('sbadge-2', 'LIVE', true);
    } else if (idx === 1) {
      // Depth map → #vd
      const vd = q('#vd');
      if (vd) { vd.srcObject = stream; }
      const ns = q('#nosig-6');
      if (ns) ns.style.display = 'none';
      setBadge('sbadge-5', 'LIVE', true);
    }

    trackCount++;
    const itrk = q('#itrk');
    if (itrk) itrk.textContent = trackCount;
  };

  // ── DataChannel: label="state" → JSON 파싱 ───────────────────────────
  pc.ondatachannel = (e) => {
    if (e.channel.label !== 'state') return;
    const dc     = e.channel;
    dc.binaryType = 'arraybuffer';

    dc.onopen = () => {
      console.log('[WebRTC] DataChannel open');
      // FACE·DETECT, PPE·HELMET 배지 활성
      setBadge('sbadge-3', 'LIVE', true);
      setBadge('sbadge-4', 'LIVE', true);
      startDcRate();
    };

    dc.onmessage = (ev) => {
      dcMsgTotal++;
      const raw = ev.data instanceof ArrayBuffer
        ? new TextDecoder().decode(ev.data) : ev.data;
      try {
        const data = JSON.parse(raw);
        // face / ppe 캡처 이미지 vs. 일반 상태 데이터 분기
        if (data.type === 'face' || data.type === 'ppe') {
          showCapture(data);
        } else {
          updateUI(data);
        }
      } catch (err) { console.warn('[WebRTC] DC parse error', err); }
    };

    dc.onclose = () => {
      console.log('[WebRTC] DataChannel closed');
      setBadge('sbadge-3', 'DC', false);
      setBadge('sbadge-4', 'DC', false);
    };
  };

  // ── PC 상태 ──────────────────────────────────────────────────────────
  pc.onconnectionstatechange = () => {
    const ipcs = q('#ipcs');
    if (ipcs) ipcs.textContent = pc.connectionState;
    console.log('[WebRTC] connectionState:', pc.connectionState);
    if (['failed', 'closed', 'disconnected'].includes(pc.connectionState))
      setWrtcStatus('error');
  };
  pc.oniceconnectionstatechange = () => {
    const iice = q('#iice');
    if (iice) iice.textContent = pc.iceConnectionState;
    console.log('[WebRTC] iceConnectionState:', pc.iceConnectionState);
  };

  // ── Step1: 서버에서 offer 받기 ───────────────────────────────────────
  let serverOffer;
  try {
    const res = await fetch(`${WEBRTC_BASE}/webrtc/offer?client_id=${clientId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    serverOffer = await res.json();
    const cands = serverOffer.sdp.split('\n').filter(l => l.startsWith('a=candidate'));
    console.log(`[WebRTC] server offer: ${cands.length} candidates`);
    if (cands.length === 0)
      console.error('[WebRTC] !! no ICE candidates in server SDP — check server NIC !!');
  } catch (err) {
    console.error('[WebRTC] get offer failed:', err);
    setWrtcStatus('error'); return;
  }

  // ── Step2: setRemoteDescription ──────────────────────────────────────
  await pc.setRemoteDescription(new RTCSessionDescription(serverOffer));

  // ── Step3: answer 생성 ───────────────────────────────────────────────
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);

  // ── Step4: ICE gathering 완료 대기 (최대 3초) ─────────────────────
  await waitIce(pc, 3000);
  const myCands = pc.localDescription.sdp.split('\n').filter(l => l.startsWith('a=candidate'));
  console.log(`[WebRTC] client answer: ${myCands.length} candidates`);

  // ── Step5: answer → 서버 전송 ─────────────────────────────────────
  try {
    const res2 = await fetch(`${WEBRTC_BASE}/webrtc/answer`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        sdp:       pc.localDescription.sdp,
        type:      pc.localDescription.type,
        client_id: clientId,
      }),
    });
    if (!res2.ok) throw new Error(`HTTP ${res2.status}`);
    console.log('[WebRTC] answer sent OK');
    setWrtcStatus('connected');
    const isrv = q('#isrv');
    if (isrv) isrv.textContent = location.hostname + ':59530';
    startStats();
  } catch (err) {
    console.error('[WebRTC] send answer failed:', err);
    setWrtcStatus('error');
  }
}

// ICE gathering 완료 대기 (참고 코드와 동일)
function waitIce(pc, ms) {
  return new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const tid = setTimeout(resolve, ms);
    pc.addEventListener('icegatheringstatechange', function h() {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(tid);
        pc.removeEventListener('icegatheringstatechange', h);
        resolve();
      }
    });
  });
}

// ── disconnect ────────────────────────────────────────────────────────────
async function disconnect() {
  stopStats();
  if (pc) {
    fetch(`${WEBRTC_BASE}/webrtc/${clientId}`, { method: 'DELETE' }).catch(() => {});
    pc.close(); pc = null;
  }
  // 비디오 srcObject 해제
  const vm = q('#vm'); if (vm) vm.srcObject = null;
  const vd = q('#vd'); if (vd) vd.srcObject = null;

  // 이미지 초기화
  ['img-face', 'img-ppe'].forEach(id => {
    const el = q('#' + id); if (el) el.src = '';
  });

  // no-signal 복원
  ['nsm', 'nosig-6', 'nsig-face', 'nsig-ppe'].forEach(id => {
    const el = q('#' + id); if (el) el.style.display = '';
  });

  // 배지 초기화
  setBadge('sbadge-2', 'IDLE', false);
  setBadge('sbadge-3', 'DC',   false);
  setBadge('sbadge-4', 'DC',   false);
  setBadge('sbadge-5', 'IDLE', false);

  trackCount = 0;
  const itrk = q('#itrk'); if (itrk) itrk.textContent = '0';
  setWrtcStatus('disconnected');
}

// ── showCapture: face / ppe base64 이미지 표시 ────────────────────────────
function showCapture(data) {
  const imgId = data.type === 'face' ? 'img-face'   : 'img-ppe';
  const nsId  = data.type === 'face' ? 'nsig-face'  : 'nsig-ppe';
  const el    = q('#' + imgId);
  const ns    = q('#' + nsId);
  if (el) el.src = 'data:image/jpeg;base64,' + data.b64;
  if (ns) ns.style.display = 'none';

  // DC LOG에 캡처 이벤트 기록
  pushDcLog(
    `${data.type.toUpperCase()} ${data.label || ''} ${data.conf != null ? (data.conf * 100).toFixed(0) + '%' : ''}`
  );
}

// ── updateUI: DataChannel state JSON → 화면 전반 갱신 ────────────────────
function updateUI(s) {
  console.log('data',s)
  // 배터리
  const pct = s.charge ?? 0;
  const bc  = q('#bar-charge');
  if (bc) {
    bc.style.width = Math.min(100, pct) + '%';
    bc.className = 'sbar-fill charge' + (pct < 20 ? ' low' : pct < 50 ? ' mid' : '');
  }
  setTxt('#hb-charge', pct + '%');
  setTxt('#val-charge', pct + '%');
  setTxt('#hb-temp',    s.temp    != null ? s.temp + '°C'   : '—');
  setTxt('#hb-voltage', s.voltage != null ? s.voltage + 'V' : '—');
  setTxt('#hb-objs',    s.cnt_object ?? '—');
  setTxt('#hb-live',    s.cnt_live   ?? '—');

  // left panel bars
  const bt = q('#bar-temp');
  if (bt) bt.style.width = Math.min(100, (s.temp || 0) / 100 * 100) + '%';
  const bv = q('#bar-volt');
  if (bv) bv.style.width = Math.min(100, (s.voltage || 0) / 30 * 100) + '%';
  setTxt('#val-temp', s.temp    != null ? s.temp + '°'    : '—');
  setTxt('#val-volt', s.voltage != null ? s.voltage + 'V' : '—');

  // OD count + canvas + list
  setTxt('#od-cnt', s.cnt_object || 0);
  const boxes = s.boxes ?? [];
  if (typeof renderODCanvas === 'function') renderODCanvas(boxes);
  if (typeof renderODList   === 'function') renderODList(boxes);

  // Human analysis
  const h = s.human ?? {};
  setTxt('#h-gender',   h.gender   || '—');
  setTxt('#h-age',      h.age      || '—');
  setTxt('#h-emotion',  h.emotion  || '—');
  setTxt('#h-depth',    h.depth    != null ? (+h.depth).toFixed(2) + ' m' : '—');
  setTxt('#h-position', h.position || '—');

  // AprilTag
  const tag = s.tag ?? {};
  setTxt('#tag-id',   tag.id   ?? '—');
  setTxt('#tag-dist', tag.dist != null ? (+tag.dist).toFixed(2) + ' m' : '—');

  // robot_data (sysinfo) — DC를 통해서도 수신 가능
  if (s.sysinfo) updateRobotData(s.sysinfo);
  else if (s.batteryPower != null || s.cpuUsage != null) updateRobotData(s);

  // ctrl_info
  if (s.ctrlInfo) updateCtrlInfo(s.ctrlInfo);

  // DC LOG
  pushDcLog(`L:${s.cnt_live ?? 0} O:${s.cnt_object ?? 0} BAT:${s.charge ?? 0}%`);
}

// DC LOG 공통 push
function pushDcLog(msg) {
  const now = new Date().toISOString().substr(11, 8);
  dcLines.unshift(
    `<div style="display:flex;justify-content:space-between;` +
    `font-family:var(--fm);font-size:8px;padding:1px 0;color:var(--tx2)">` +
    `<span style="color:var(--ac);opacity:.6">${now}</span>` +
    `<span>${msg}</span></div>`
  );
  if (dcLines.length > 6) dcLines.pop();
  const db = q('#dc-log-body');
  if (db) db.innerHTML = dcLines.join('');
}

// ── Stats 타이머 ──────────────────────────────────────────────────────────
function startStats() {
  statsTimer = setInterval(async () => {
    if (!pc) return;
    try {
      const stats = await pc.getStats();
      stats.forEach(r => {
        if (r.type === 'candidate-pair' && r.state === 'succeeded' && r.currentRoundTripTime != null) {
          // irtt는 nav-status-bar에서 Vx/Vy/Ω로 재활용 중이므로 별도 ID 사용
          const el = q('#wrtc-rtt');
          if (el) el.textContent = Math.round(r.currentRoundTripTime * 1000) + ' ms';
        }
      });
    } catch { /* getStats 실패 무시 */ }
  }, 1000);
}

function startDcRate() {
  dcRateTimer = setInterval(() => {
    const el = q('#wrtc-dcr');
    if (el) el.textContent = (dcMsgTotal - dcMsgPrev) + '/s';
    dcMsgPrev = dcMsgTotal;
  }, 1000);
}

function stopStats() {
  clearInterval(statsTimer); clearInterval(dcRateTimer);
  statsTimer = dcRateTimer = null;
}

// ── FPS 카운터 (vm 기준) ─────────────────────────────────────────────────
let fpsN = 0, fpsT = performance.now();
(function fpsLoop() {
  requestAnimationFrame(() => {
    fpsN++;
    const now = performance.now();
    if (now - fpsT >= 1000) {
      const el = q('#fps');
      if (el) el.textContent = fpsN + ' fps';
      fpsN = 0; fpsT = now;
    }
    fpsLoop();
  });
})();

// ── 헬퍼 ─────────────────────────────────────────────────────────────────
function setTxt(sel, val) {
  const el = q(sel); if (el) el.textContent = val;
}
function setBadge(id, text, live) {
  const el = q('#' + id);
  if (!el) return;
  el.textContent = text;
  el.className   = 'stream-badge ' + (live ? 'live' : 'idle');
}


// ── Start heartbeat ──────────────────────────────────────────────────
pollHeartbeat();

console.log('%cARCos Dashboard v4.0 ONLINE', 'color:#00ffe7;font-family:monospace;font-size:14px');// ── Robot Data 시각화 (실제 SDK robot_data 기준) ─────────────────────────────
// sysinfo 필드 매핑:
//   batteryPower  → 배터리 잔량 %       (bat_pct)
//   batteryVol    → 배터리 전압 mV      (bat_vol)
//   batteryAmp    → 배터리 전류 mA      (bat_amp)
//   batteryTemp   → 배터리 온도 °C      (bat_temp)
//   cpuUsage      → CPU 사용률 %        (cpu_usage)
//   cpuMemory     → 메모리 사용률 %     (cpu_mem)
//   cpuTemp       → CPU 온도 °C         (cpu_temp)
//   cpuFrequency  → CPU 주파수 MHz      (cpu_freq)
//   motorTempMax  → 모터 최대 온도 °C   (motor_temp_max)
//   motorTempAvg  → 모터 평균 온도 °C   (motor_temp_avg)
//   motorErrCnt   → 에러 모터 수        (motor_err_cnt)
//   motorTemps[]  → 각 모터 온도 배열   (motor_temps=[...])
//   motorErrors[] → 각 모터 에러 배열   (motor_errors=[...])

function updateRobotData(si) {
  if (!si) return;

  // LIVE 배지
  const badge = q('#rd-badge');
  if (badge) badge.style.display = 'inline';

  // ── BATTERY ────────────────────────────────────────────────────
  const pct = si.batteryPower ?? si.battery;
  setVal('rd-bat-pct',  pct  != null ? pct.toFixed(1)  : null);
  setVal('rd-bat-vol',  si.batteryVol  != null ? si.batteryVol.toFixed(0)  : null);
  setVal('rd-bat-amp',  si.batteryAmp  != null ? si.batteryAmp.toFixed(0)  : null);
  setVal('rd-bat-temp', si.batteryTemp != null ? si.batteryTemp.toFixed(1) : null);

  // 배터리 바
  if (pct != null && q('#bar-bat-pct')) {
    q('#bar-bat-pct').style.width = `${Math.min(100, pct)}%`;
    q('#bar-bat-pct').className = 'sbar-fill charge'
      + (pct < 20 ? ' low' : pct < 50 ? ' mid' : '');
  }

  // ── CPU / SYSTEM ────────────────────────────────────────────────
  const cpu = si.cpuUsage ?? si.cpu;
  setVal('rd-cpu-usage', cpu != null ? cpu.toFixed(1) : null);
  setVal('rd-cpu-mem',   si.cpuMemory    != null ? si.cpuMemory.toFixed(1)    : null);
  setVal('rd-cpu-temp',  si.cpuTemp      != null ? si.cpuTemp.toFixed(1)      : null);
  setVal('rd-cpu-freq',  si.cpuFrequency != null ? si.cpuFrequency.toFixed(0) : null);

  if (cpu != null && q('#bar-cpu-usage')) {
    q('#bar-cpu-usage').style.width = `${Math.min(100, cpu)}%`;
    q('#bar-cpu-usage').className = 'sbar-fill temp' + (cpu > 80 ? ' hot' : '');
  }
  if (si.cpuMemory != null && q('#bar-cpu-mem')) {
    q('#bar-cpu-mem').style.width = `${Math.min(100, si.cpuMemory)}%`;
  }

  // ── MOTOR TEMP ──────────────────────────────────────────────────
  const mtMax = si.motorTempMax ?? si.motorTemp;
  setVal('rd-motor-temp-max', mtMax          != null ? mtMax.toFixed(1)          : null);
  setVal('rd-motor-temp-avg', si.motorTempAvg != null ? si.motorTempAvg.toFixed(1) : null);

  if (mtMax != null && q('#bar-motor-temp')) {
    q('#bar-motor-temp').style.width = `${Math.min(100, mtMax / 80 * 100)}%`;
    q('#bar-motor-temp').className = 'sbar-fill temp' + (mtMax > 60 ? ' hot' : '');
  }

  // 에러 배지
  const errCnt = si.motorErrCnt || 0;
  const errBadge = q('#rd-motor-err-badge');
  if (errBadge) {
    errBadge.style.display = errCnt > 0 ? 'inline' : 'none';
    errBadge.textContent = `${errCnt} ERR!`;
  }

  // ── 모터 히트맵 ─────────────────────────────────────────────────
  const temps  = si.motorTemps  || [];
  const errors = si.motorErrors || [];
  const grid   = q('#rd-motor-grid');
  if (grid && temps.length > 0) {
    grid.innerHTML = temps.map((t, i) => {
      const err  = errors[i] && errors[i] !== 0;
      const cls  = err      ? 'err'
                 : t > 70   ? 'crit'
                 : t > 55   ? 'hot'
                 : t > 40   ? 'warm'
                 : 'ok';
      const label = t.toFixed(0);
      const title = `M${i+1}: ${t.toFixed(1)}°C${err ? ' ERR:'+errors[i] : ''}`;
      return `<div class="rd-motor-cell ${cls}" title="${title}">${label}</div>`;
    }).join('');
  }
}

function setVal(id, val) {
  const el = q('#' + id);
  if (el) el.textContent = val != null ? val : '—';
}

// ── ctrl_info 시각화 ─────────────────────────────────────────────────────────
function updateCtrlInfo(ci) {
  if (!ci) return;

  // 네비 상태
  const nsMode = q('#ns-mode');
  if (nsMode) {
    let t = (ci.navState || 'IDLE').toUpperCase();
    if (ci.isPause)       t = 'PAUSED';
    if (ci.isBack)        t = 'BACKING';
    if (ci.isRotate)      t = 'ROTATING';
    if (ci.isClimbStairs) t = 'CLIMBING';
    nsMode.textContent = t;
  }

  // 목표 좌표
  if (ci.targetX != null) {
    const nx = q('#ns-x'); if (nx) nx.textContent = ci.targetX.toFixed(3);
    const ny = q('#ns-y'); if (ny) ny.textContent = ci.targetY.toFixed(3);
    const nh = q('#ns-hdg');
    if (nh) nh.textContent = (ci.targetYaw * 180 / Math.PI).toFixed(1) + '°';
  }

  // 진행률
  const bar = q('#ns-seq-bar');
  if (bar && ci.completionPct != null) bar.style.width = `${Math.min(100, ci.completionPct)}%`;

  const step = q('#ns-seq-step');
  if (step) step.textContent = ci.lastTime > 0
    ? `${(ci.usedTime||0).toFixed(0)}s / ${ci.lastTime.toFixed(0)}s`
    : `${(ci.completionPct||0).toFixed(1)}%`;

  const name = q('#ns-seq-name');
  if (name && ci.targetNode != null && ci.targetNode >= 0)
    name.textContent = `NODE ${ci.targetNode}`;

  // 속도 표시 (irtt 자리에 vx/vy/vyaw 표시)
  const irtt = q('#irtt');
  if (irtt && ci.vx != null)
    irtt.textContent = `${ci.vx.toFixed(2)} / ${ci.vy.toFixed(2)} / ${ci.vyaw.toFixed(2)}`;

  const idcr = q('#idcr');
  if (idcr && ci.completionPct != null)
    idcr.textContent = ci.completionPct.toFixed(1) + '%';
}

function parseRobotDataFromState(s) {
  // heartbeat state나 sysinfo 이벤트에서 robot_data 필드 추출
  if (!s) return null;
  if (s.sysinfo) return s.sysinfo;
  // batteryPower나 cpuUsage 등이 직접 있는 경우
  if (s.batteryPower != null || s.cpuUsage != null || s.motorTempMax != null) return s;
  return null;
}


