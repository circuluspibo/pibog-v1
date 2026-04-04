/**
 * ARCos Fleet Map — Canvas Renderer v3.2
 * ─────────────────────────────────────────────────────────────────
 * Mouse controls:
 *   LMB drag    → pan (all modes) / drag waypoint (select mode on WP)
 *   RMB drag    → orbit (3D-style horizontal rotation)
 *   MMB drag    → pan (always)
 *   Wheel       → zoom
 *   LMB click   → add waypoint (waypoint mode) / select WP / goto (select mode)
 *
 * Rotation: 0/90/180/270° discrete buttons OR RMB-drag free rotation
 * Tilt: 2.5D mode with adjustable depth angle (tiltAngle)
 */
class FleetMap {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx    = this.canvas.getContext('2d');
    this.dpr    = window.devicePixelRatio || 1;

    // View
    this.panX     = 0;
    this.panY     = 0;
    this.zoom     = 50;
    this.minZoom  = 4;
    this.maxZoom  = 400;
    this.rotation = 0;       // radians
    this.tilt     = 0;       // 0 = flat, 1 = full 2.5D
    this.tiltAngle = 58;     // degrees — adjustable via slider

    // Data
    this.waypoints  = {};
    this.robots     = {};
    this.trails     = {};
    this.arrivals   = [];
    this.seqPreview = null;

    // Interaction state
    this.mode       = 'select';
    this.showGrid   = true;
    this.showTrails = true;

    this._lmbDown  = false;   // left button held
    this._rmbDown  = false;   // right button (orbit)
    this._mmbDown  = false;   // middle button (pan)
    this._dragPan  = false;   // LMB is panning (not on a WP)
    this._dragWP   = null;    // WP name being dragged
    this._orbitStart= 0;      // mouseX at RMB down for orbit
    this._rotAtOrbitStart = 0;
    this._lastMX   = 0;
    this._lastMY   = 0;
    this._pendingClick = null;
    this._hoverWP  = null;
    this.selectedWP = null;

    // Callbacks
    this.onMapClick      = null;
    this.onWaypointClick = null;
    this.onWPMoved       = null;
    this.onSeqWPClick    = null;

    this._pal   = ['#00ffe7','#ff6b00','#39ff14','#ffd600','#e040fb','#00b0ff','#ff6d00','#76ff03'];
    this._palIdx = 0;
    this._themeAccent = 'rgba(0,255,231,0.04)';

    this._init();
  }

  _init() {
    this._resize();
    window.addEventListener('resize', () => this._resize());

    const c = this.canvas;
    c.addEventListener('mousedown',   e => this._mdown(e));
    c.addEventListener('mousemove',   e => this._mmove(e));
    c.addEventListener('mouseup',     e => this._mup(e));
    c.addEventListener('mouseleave',  () => this._endDrag());
    c.addEventListener('wheel',       e => this._wheel(e), { passive: false });
    c.addEventListener('contextmenu', e => e.preventDefault());

    // Touch
    let pinch0 = null, tPan = null;
    c.addEventListener('touchstart', e => {
      if (e.touches.length === 1) tPan = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      if (e.touches.length === 2) { pinch0 = this._pinchD(e); tPan = null; }
    }, { passive: true });
    c.addEventListener('touchmove', e => {
      if (e.touches.length === 1 && tPan) {
        const t = e.touches[0];
        this.panX += t.clientX - tPan.x;
        this.panY += t.clientY - tPan.y;
        tPan = { x: t.clientX, y: t.clientY };
      }
      if (e.touches.length === 2 && pinch0) {
        const d = this._pinchD(e);
        const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const my = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        this._applyZoom(d / pinch0, mx, my);
        pinch0 = d;
      }
      e.preventDefault();
    }, { passive: false });
    c.addEventListener('touchend', () => { pinch0 = null; tPan = null; }, { passive: true });

    // Center origin
    requestAnimationFrame(() => {
      const r = c.parentElement.getBoundingClientRect();
      this.panX = r.width  / 2;
      this.panY = r.height / 2;
      this._loop();
    });
  }

  _resize() {
    const el = this.canvas.parentElement;
    const r  = el.getBoundingClientRect();
    this.canvas.width  = r.width  * this.dpr;
    this.canvas.height = r.height * this.dpr;
    this.canvas.style.width  = r.width  + 'px';
    this.canvas.style.height = r.height + 'px';
  }

  _pinchD(e) {
    return Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
  }

  // ── Coordinate transforms ────────────────────────────────────────
  worldToScreen(wx, wy) {
    const cos = Math.cos(this.rotation), sin = Math.sin(this.rotation);
    const rx  =  wx * cos - wy * sin;
    const ry  =  wx * sin + wy * cos;
    let sx = this.panX + rx * this.zoom;
    let sy = this.panY - ry * this.zoom;
    if (this.tilt > 0) {
      const tf = Math.sin(this.tiltAngle * Math.PI / 180) * this.tilt;
      sy = this.panY + (sy - this.panY) * tf;
    }
    return { sx, sy };
  }

  screenToWorld(sx, sy) {
    let tsy = sy;
    if (this.tilt > 0) {
      const tf = Math.sin(this.tiltAngle * Math.PI / 180) * this.tilt;
      if (tf > 0.01) tsy = this.panY + (sy - this.panY) / tf;
    }
    const rx =  (sx - this.panX) / this.zoom;
    const ry = -(tsy - this.panY) / this.zoom;
    const cos = Math.cos(-this.rotation), sin = Math.sin(-this.rotation);
    return { wx: rx * cos - ry * sin, wy: rx * sin + ry * cos };
  }

  // ── Mouse handlers ───────────────────────────────────────────────
  _mpos(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  _hitWP(sx, sy) {
    for (const [name, pose] of Object.entries(this.waypoints)) {
      const p = this.worldToScreen(pose.x, pose.y);
      if (Math.abs(sx - p.sx) < 16 && Math.abs(sy - p.sy) < 16) return name;
    }
    return null;
  }

  _mdown(e) {
    const { x, y } = this._mpos(e);
    this._lastMX = x; this._lastMY = y;

    // Middle button → always pan
    if (e.button === 1) {
      this._mmbDown = true;
      this.canvas.style.cursor = 'all-scroll';
      e.preventDefault(); return;
    }

    // Right button → orbit (3D-style rotation around vertical axis)
    if (e.button === 2) {
      this._rmbDown = true;
      this._orbitStart = x;
      this._rotAtOrbitStart = this.rotation;
      this.canvas.style.cursor = 'ew-resize';
      return;
    }

    // Left button
    this._lmbDown = true;
    const hit = this._hitWP(x, y);

    if (this.mode === 'select' && hit) {
      // Drag waypoint
      this._dragWP  = hit;
      this._dragPan = false;
      this.selectedWP = hit;
      if (this.onWaypointClick) this.onWaypointClick(hit, this.waypoints[hit]);
      this.canvas.style.cursor = 'grabbing';
    } else if (this.mode === 'sequence' && hit) {
      if (this.onSeqWPClick) this.onSeqWPClick(hit);
    } else {
      // Pan
      this._dragPan = true;
      this._dragWP  = null;
      if (!hit) this.selectedWP = null;
      if (this.mode === 'waypoint') this._pendingClick = { x, y };
      this.canvas.style.cursor = 'grabbing';
    }
  }

  _mmove(e) {
    const { x, y } = this._mpos(e);
    const wc = this.screenToWorld(x, y);
    const dx = x - this._lastMX, dy = y - this._lastMY;

    // Coord HUD
    const cd = document.getElementById('map-coord');
    if (cd) cd.textContent = `X: ${wc.wx.toFixed(3)}  Y: ${wc.wy.toFixed(3)}`;
    const zd = document.getElementById('map-zoom-lbl');
    if (zd) zd.textContent = `ZOOM ${(this.zoom/50).toFixed(1)}×`;

    // Hover
    const prev = this._hoverWP;
    this._hoverWP = this._hitWP(x, y);
    if (this._hoverWP !== prev) {
      this.canvas.style.cursor = this._hoverWP
        ? (this.mode === 'select' ? 'grab' : 'pointer')
        : (this._rmbDown ? 'ew-resize' : this._mmbDown ? 'all-scroll' : 'grab');
    }

    // Orbit (RMB) — rotate around vertical axis like 3D software
    if (this._rmbDown) {
      // 1px = ~0.3° rotation
      const delta = (x - this._orbitStart) * 0.005;
      this.rotation = this._rotAtOrbitStart + delta;
      this._updateViewLabel();
      this._syncRotBtns();
    }

    // MMB pan
    if (this._mmbDown) {
      this.panX += dx; this.panY += dy;
    }

    // LMB actions
    if (this._lmbDown) {
      if (this._dragPan) {
        this.panX += dx; this.panY += dy;
        if (dx * dx + dy * dy > 9) this._pendingClick = null;
      } else if (this._dragWP && this.waypoints[this._dragWP]) {
        const wb = this.screenToWorld(x - dx, y - dy);
        const wa = this.screenToWorld(x, y);
        this.waypoints[this._dragWP].x += wa.wx - wb.wx;
        this.waypoints[this._dragWP].y += wa.wy - wb.wy;
      }
    }

    this._lastMX = x; this._lastMY = y;

    // Tooltip
    const tip = document.getElementById('map-tip');
    if (tip) {
      if (this._hoverWP) {
        const wp = this.waypoints[this._hoverWP];
        tip.style.cssText = `display:block;left:${x+18}px;top:${y-8}px`;
        tip.innerHTML = `<b>${this._hoverWP}</b><br>x:${wp.x.toFixed(3)}<br>y:${wp.y.toFixed(3)}`;
      } else { tip.style.display = 'none'; }
    }
  }

  _mup(e) {
    if (e.button === 2) { this._rmbDown = false; this.canvas.style.cursor = 'grab'; return; }
    if (e.button === 1) { this._mmbDown = false; this.canvas.style.cursor = 'grab'; return; }

    if (this._lmbDown) {
      if (this.mode === 'waypoint' && this._pendingClick) {
        const { x, y } = this._pendingClick;
        const w = this.screenToWorld(x, y);
        if (this.onMapClick) this.onMapClick(w.wx, w.wy);
      }
      this._pendingClick = null;
      if (this._dragWP && this.onWPMoved) {
        const wp = this.waypoints[this._dragWP];
        if (wp) this.onWPMoved(this._dragWP, wp.x, wp.y);
      }
    }
    this._endDrag();
  }

  _endDrag() {
    this._lmbDown = false; this._rmbDown = false; this._mmbDown = false;
    this._dragPan = false; this._dragWP = null;
    this.canvas.style.cursor = 'grab';
  }

  _wheel(e) {
    e.preventDefault();
    const { x, y } = this._mpos(e);
    this._applyZoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, x, y);
  }

  _applyZoom(f, cx, cy) {
    const nz = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * f));
    this.panX = cx - (cx - this.panX) * (nz / this.zoom);
    this.panY = cy - (cy - this.panY) * (nz / this.zoom);
    this.zoom = nz;
  }

  _updateViewLabel() {
    const vd = document.getElementById('map-view-lbl');
    if (!vd) return;
    const deg = Math.round(((this.rotation * 180 / Math.PI) % 360 + 360) % 360);
    vd.textContent = `${this.tilt > 0 ? '2.5D' : '2D'} · ${deg}°`;
  }

  _syncRotBtns() {
    const deg = Math.round(((this.rotation * 180 / Math.PI) % 360 + 360) % 360);
    document.querySelectorAll('.rot-btn[data-deg]').forEach(b => {
      const bd = parseInt(b.dataset.deg);
      b.classList.toggle('active', Math.abs(bd - deg) < 5 || (bd === 0 && Math.abs(deg - 360) < 5));
    });
  }

  // ── Render loop ──────────────────────────────────────────────────
  _loop() { this._draw(); requestAnimationFrame(() => this._loop()); }

  _draw() {
    const ctx = this.ctx;
    const W   = this.canvas.width  / this.dpr;
    const H   = this.canvas.height / this.dpr;

    ctx.save();
    ctx.scale(this.dpr, this.dpr);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#010508'; ctx.fillRect(0, 0, W, H);

    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.6);
    bg.addColorStop(0, this._themeAccent); bg.addColorStop(1, 'transparent');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

    if (this.showGrid)   this._drawGrid(ctx, W, H);
    this._drawOriginAxes(ctx);
    if (this.showTrails) this._drawTrails(ctx);
    this._drawSeqPreview(ctx);
    this._drawWaypoints(ctx);
    this._drawArrivals(ctx);
    this._drawRobots(ctx);
    this._drawCrosshair(ctx, W, H);
    this._drawCompass(ctx, W, H);
    ctx.restore();
  }

  _drawGrid(ctx, W, H) {
    const step = this.zoom, big = step * 5;
    ctx.strokeStyle = 'rgba(14,58,82,0.45)'; ctx.lineWidth = 0.5;
    ctx.beginPath(); this._gridLines(ctx, W, H, step); ctx.stroke();
    ctx.strokeStyle = 'rgba(0,255,231,0.07)'; ctx.lineWidth = 1;
    ctx.beginPath(); this._gridLines(ctx, W, H, big); ctx.stroke();
    ctx.fillStyle = 'rgba(0,255,231,0.18)'; ctx.font = '7px Share Tech Mono'; ctx.textAlign = 'left';
    const bsx = ((this.panX % big) + big) % big;
    const bsy = ((this.panY % big) + big) % big;
    for (let x = bsx - big; x < W + big; x += big) {
      const wx = Math.round((x - this.panX) / this.zoom);
      if (!wx) continue; ctx.fillText(`${wx}m`, x + 2, Math.max(11, this.panY - 2));
    }
    for (let y = bsy - big; y < H + big; y += big) {
      const wy = Math.round(-(y - this.panY) / this.zoom);
      if (!wy) continue; ctx.fillText(`${wy}m`, Math.max(2, this.panX + 2), y + 9);
    }
  }

  _gridLines(ctx, W, H, step) {
    const sx = ((this.panX % step) + step) % step;
    const sy = ((this.panY % step) + step) % step;
    for (let x = sx - step; x < W + step; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let y = sy - step; y < H + step; y += step) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  }

  _drawOriginAxes(ctx) {
    const { sx, sy } = this.worldToScreen(0, 0);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ff4040'; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + 24, sy); ctx.stroke();
    ctx.strokeStyle = '#40ff80'; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, sy - 24); ctx.stroke();
    ctx.fillStyle   = '#00ffe7'; ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = '8px Share Tech Mono'; ctx.textAlign = 'left';
    ctx.fillStyle = '#ff4040aa'; ctx.fillText('X', sx + 26, sy + 4);
    ctx.fillStyle = '#40ff80aa'; ctx.fillText('Y', sx + 2, sy - 26);
  }

  _drawTrails(ctx) {
    Object.entries(this.trails).forEach(([id, pts]) => {
      if (pts.length < 2) return;
      const color = this.robots[id]?.color || '#00ffe7';
      ctx.setLineDash([3, 7]); ctx.lineWidth = 1.5;
      ctx.strokeStyle = color + '50'; ctx.beginPath();
      pts.forEach((p, i) => {
        const { sx, sy } = this.worldToScreen(p.x, p.y);
        i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      });
      ctx.stroke(); ctx.setLineDash([]);
    });
  }

  _drawSeqPreview(ctx) {
    if (!this.seqPreview?.length) return;
    ctx.setLineDash([6, 5]); ctx.strokeStyle = 'rgba(255,107,0,0.5)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.seqPreview.forEach((p, i) => {
      const { sx, sy } = this.worldToScreen(p.x, p.y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke(); ctx.setLineDash([]);
    this.seqPreview.forEach((p, i) => {
      const { sx, sy } = this.worldToScreen(p.x, p.y);
      ctx.fillStyle = 'rgba(255,107,0,0.7)'; ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 8px Share Tech Mono'; ctx.textAlign = 'center';
      ctx.fillText(i + 1, sx, sy + 3);
    });
  }

  _drawWaypoints(ctx) {
    const now = Date.now();
    Object.entries(this.waypoints).forEach(([name, pose]) => {
      const { sx, sy } = this.worldToScreen(pose.x, pose.y);
      const isHov = this._hoverWP === name, isSel = this.selectedWP === name;
      const S = 10, col = isSel ? '#ffaa00' : isHov ? '#fff' : '#ff6b00';
      const t = (now % 2400) / 2400;
      ctx.strokeStyle = `rgba(255,107,0,${0.4 * (1 - t)})`; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(sx, sy, S + t * 18, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = col + '28'; ctx.strokeStyle = col; ctx.lineWidth = isSel ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(sx, sy - S); ctx.lineTo(sx + S, sy); ctx.lineTo(sx, sy + S); ctx.lineTo(sx - S, sy);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = col + 'cc'; ctx.font = `${isSel ? 'bold ' : ''}10px Share Tech Mono`;
      ctx.textAlign = 'left'; ctx.fillText(name, sx + S + 5, sy + 4);
    });
  }

  _drawArrivals(ctx) {
    const now = Date.now();
    this.arrivals = this.arrivals.filter(a => now - a.t < 2800);
    this.arrivals.forEach(a => {
      const p = (now - a.t) / 2800;
      const { sx, sy } = this.worldToScreen(a.wx, a.wy);
      const col = a.success ? '#39ff14' : '#ff1744';
      [0, 0.15, 0.3].forEach(off => {
        const rp = Math.max(0, p - off), r = rp * 65, al = (1 - rp) * 0.5;
        if (al <= 0 || r <= 0) return;
        ctx.strokeStyle = col + Math.round(al * 255).toString(16).padStart(2, '0');
        ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.stroke();
      });
      if (p < 0.75) {
        const al = 1 - p / 0.75, fly = p * 42;
        ctx.fillStyle = col + Math.round(al * 255).toString(16).padStart(2, '0');
        ctx.font = `bold ${(1 - p * 0.4) * 22}px Orbitron`; ctx.textAlign = 'center';
        ctx.fillText(a.success ? '✓' : '✗', sx, sy - fly);
      }
    });
  }

  _drawRobots(ctx) {
    const now = Date.now();
    Object.values(this.robots).forEach(r => {
      if (!r.position) return;
      const { x, y, q_z = 0, q_w = 1 } = r.position;
      const { sx, sy } = this.worldToScreen(x, y);
      const yaw = 2 * Math.atan2(q_z, q_w);
      const S = 14, col = r.color || '#00ffe7';
      const isObs = r.obstacle?.active, isNav = r.status === 'navigating';

      if (isNav) {
        const t = (now % 1600) / 1600;
        ctx.strokeStyle = `rgba(255,107,0,${0.4 * (1 - t)})`; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(sx, sy, S + t * 24, 0, Math.PI * 2); ctx.stroke();
      }
      if (isObs) {
        const fl = Math.sin(now / 200) > 0;
        ctx.strokeStyle = fl ? 'rgba(255,23,68,0.9)' : 'rgba(255,23,68,0.3)';
        ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(sx, sy, S + 8, 0, Math.PI * 2); ctx.stroke();
        const prog = r.obstacle?.progress || 0;
        if (prog > 0) {
          ctx.strokeStyle = 'rgba(57,255,20,0.5)'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(sx, sy, S + 12, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2); ctx.stroke();
        }
      }

      ctx.save(); ctx.translate(sx, sy); ctx.rotate(-yaw + Math.PI / 2);
      ctx.fillStyle   = (isObs ? '#ff1744' : col) + '28';
      ctx.strokeStyle = isObs ? '#ff1744' : col; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, -S); ctx.lineTo(S * 0.72, S * 0.6);
      ctx.lineTo(0, S * 0.22); ctx.lineTo(-S * 0.72, S * 0.6);
      ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();

      ctx.strokeStyle = (isObs ? '#ff1744' : col) + '55'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(sx, sy, S + 5, 0, Math.PI * 2); ctx.stroke();

      ctx.fillStyle = isObs ? '#ff1744' : col;
      ctx.font = '9px Orbitron'; ctx.textAlign = 'center';
      ctx.fillText(r.name, sx, sy - S - 8);

      const dotC = { idle: '#4e8ba4', navigating: '#ff6b00', error: '#ff1744', mapping: '#00b4ff', paused: '#888', obstacle: '#ff1744' };
      ctx.fillStyle = dotC[r.status] || '#888';
      ctx.beginPath(); ctx.arc(sx + S - 1, sy - S + 2, 4, 0, Math.PI * 2); ctx.fill();

      if (isObs) {
        ctx.fillStyle = Math.sin(now / 300) > 0 ? '#ff1744' : '#ff174480';
        ctx.font = '13px Arial'; ctx.textAlign = 'center';
        ctx.fillText('⚠', sx, sy + S + 18);
        const cnt = r.obstacle?.eventCount;
        if (cnt > 0) { ctx.fillStyle = '#ffd600'; ctx.font = '8px Share Tech Mono'; ctx.fillText(`×${cnt}`, sx + 12, sy + S + 28); }
      }
    });
  }

  _drawCompass(ctx, W, H) {
    const cx = W - 42, cy = 44, R = 20;
    ctx.strokeStyle = 'rgba(0,255,231,0.13)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2 - this.rotation;
      ctx.strokeStyle = 'rgba(0,255,231,0.2)'; ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * (R - 3), cy + Math.sin(a) * (R - 3));
      ctx.lineTo(cx + Math.cos(a) * (R - 8), cy + Math.sin(a) * (R - 8));
      ctx.stroke();
    }
    const na = -this.rotation - Math.PI / 2;
    const nx = cx + Math.cos(na) * (R - 5), ny = cy + Math.sin(na) * (R - 5);
    ctx.strokeStyle = '#ff4040'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nx, ny); ctx.stroke();
    const sa = na + Math.PI;
    ctx.strokeStyle = 'rgba(0,255,231,0.25)'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(sa) * (R - 5), cy + Math.sin(sa) * (R - 5)); ctx.stroke();
    ctx.fillStyle = '#ff4040'; ctx.font = 'bold 8px Orbitron'; ctx.textAlign = 'center';
    ctx.fillText('N', nx, ny + (na < 0 ? -3 : 10));
    const deg = Math.round(((this.rotation * 180 / Math.PI) % 360 + 360) % 360);
    ctx.fillStyle = 'rgba(0,255,231,0.3)'; ctx.font = '7px Share Tech Mono';
    ctx.fillText(`${deg}°`, cx, cy + R + 10);
    if (this.tilt > 0) { ctx.fillStyle = 'rgba(255,214,0,0.55)'; ctx.fillText(`2.5D/${this.tiltAngle}°`, cx, cy + R + 19); }
    ctx.textAlign = 'left';
  }

  _drawCrosshair(ctx, W, H) {
    ctx.strokeStyle = 'rgba(0,255,231,0.07)'; ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(this._lastMX, 0); ctx.lineTo(this._lastMX, H);
    ctx.moveTo(0, this._lastMY); ctx.lineTo(W, this._lastMY);
    ctx.stroke(); ctx.setLineDash([]);
  }

  // ── Public API ───────────────────────────────────────────────────
  setWaypoints(wp) { this.waypoints = JSON.parse(JSON.stringify(wp)); }

  setRobot(data) {
    if (!this.robots[data.id]) {
      data.color = this._pal[this._palIdx % this._pal.length]; this._palIdx++;
      this.trails[data.id] = [];
    }
    this.robots[data.id] = { ...this.robots[data.id], ...data };
  }

  removeRobot(id) { delete this.robots[id]; delete this.trails[id]; }

  updateRobotPosition(robotId, pos) {
    if (!this.robots[robotId]) return;
    this.robots[robotId].position = pos;
    const trail = this.trails[robotId] || (this.trails[robotId] = []);
    const last  = trail[trail.length - 1];
    if (!last || Math.hypot(pos.x - last.x, pos.y - last.y) > 0.03)
      trail.push({ x: pos.x, y: pos.y, t: Date.now() });
    if (trail.length > 800) trail.shift();
  }

  updateRobotObstacle(robotId, obs) {
    if (this.robots[robotId]) this.robots[robotId].obstacle = obs;
  }

  triggerArrival(wx, wy, success) { this.arrivals.push({ wx, wy, t: Date.now(), success }); }
  clearTrails()     { Object.keys(this.trails).forEach(id => { this.trails[id] = []; }); }
  clearSeqPreview() { this.seqPreview = null; }
  setMode(m)        { this.mode = m; }

  /** Set discrete rotation (0/90/180/270) */
  setRotation(deg) {
    this.rotation = deg * Math.PI / 180;
    this._updateViewLabel(); this._syncRotBtns();
  }

  /** Set tilt angle (degrees 20–80) and re-render */
  setTiltAngle(deg) {
    this.tiltAngle = Math.max(20, Math.min(80, deg));
    this._updateViewLabel();
  }

  toggleTilt() {
    this.tilt = this.tilt > 0 ? 0 : 1;
    this._updateViewLabel();
    return this.tilt > 0;
  }

  showSequencePath(steps, waypoints) {
    this.seqPreview = steps
      .filter(s => s.type === 'goto' && waypoints[s.poseName])
      .map(s => waypoints[s.poseName]);
  }

  fitView() {
    const pts = [
      ...Object.values(this.waypoints).map(p => ({ x: p.x, y: p.y })),
      ...Object.values(this.robots).filter(r => r.position).map(r => r.position)
    ];
    if (!pts.length) {
      this.zoom = 50;
      this.panX = this.canvas.width  / this.dpr / 2;
      this.panY = this.canvas.height / this.dpr / 2;
      return;
    }
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const W = this.canvas.width / this.dpr, H = this.canvas.height / this.dpr;
    const m = 70;
    this.zoom = Math.min(
      (W - m * 2) / (Math.max(...xs) - Math.min(...xs) || 10),
      (H - m * 2) / (Math.max(...ys) - Math.min(...ys) || 10),
      this.maxZoom
    );
    this.panX = W / 2 - ((Math.min(...xs) + Math.max(...xs)) / 2) * this.zoom;
    this.panY = H / 2 + ((Math.min(...ys) + Math.max(...ys)) / 2) * this.zoom;
  }

  setThemeAccent(theme) {
    const m = { red: 'rgba(255,45,85,0.05)', blue: 'rgba(0,196,255,0.04)', default: 'rgba(0,255,231,0.04)' };
    this._themeAccent = m[theme] || m.default;
  }
}

window.FleetMap = FleetMap;
