// Tuxemon Cotton Town map: tile layers (with flips and animations), collision grid and A* pathfinding.

import { loadImage } from "./sprites.js";

const FLIP_H = 0x80000000;
const GID_MASK = 0x1fffffff;

export let map = null;

export async function loadMap() {
  const res = await fetch("assets/town.json");
  if (!res.ok) throw new Error(`assets/town.json: HTTP ${res.status}`);
  const town = await res.json();
  const images = await Promise.all(town.tilesets.map((ts) => loadImage(`assets/${ts.image}`)));
  const tilesets = town.tilesets.map((ts, i) => ({ ...ts, img: images[i] })).sort((a, b) => a.firstgid - b.firstgid);
  const animations = new Map();
  for (const [gid, frames] of Object.entries(town.animations)) {
    animations.set(Number(gid), { frames, total: frames.reduce((s, f) => s + f[1], 0) });
  }
  map = { ...town, tilesets, animations, sourceCache: new Map() };
  return map;
}

function source(gid) {
  let src = map.sourceCache.get(gid);
  if (src) return src;
  let ts = null;
  for (const t of map.tilesets) if (t.firstgid <= gid) ts = t;
  const local = gid - ts.firstgid;
  src = { img: ts.img, sx: (local % ts.columns) * map.tilesize, sy: Math.floor(local / ts.columns) * map.tilesize };
  map.sourceCache.set(gid, src);
  return src;
}

function frameGid(gid, now) {
  const anim = map.animations.get(gid);
  if (!anim) return gid;
  let t = now % anim.total;
  for (const [g, ms] of anim.frames) {
    if (t < ms) return g;
    t -= ms;
  }
  return anim.frames[anim.frames.length - 1][0];
}

// Draws only the tiles visible in the canvas viewport; camX/camY are integer pixel offsets.
export function drawLayers(ctx, layers, camX, camY, now) {
  const ts = map.tilesize;
  const x0 = Math.max(0, Math.floor(camX / ts));
  const y0 = Math.max(0, Math.floor(camY / ts));
  const x1 = Math.min(map.width - 1, Math.floor((camX + ctx.canvas.width - 1) / ts));
  const y1 = Math.min(map.height - 1, Math.floor((camY + ctx.canvas.height - 1) / ts));
  for (const layer of layers) {
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const raw = layer[y * map.width + x];
        if (!raw) continue;
        const gid = frameGid(raw & GID_MASK, now);
        const { img, sx, sy } = source(gid);
        const dx = x * ts - camX;
        const dy = y * ts - camY;
        if (raw & FLIP_H) {
          ctx.save();
          ctx.translate(dx + ts, dy);
          ctx.scale(-1, 1);
          ctx.drawImage(img, sx, sy, ts, ts, 0, 0, ts, ts);
          ctx.restore();
        } else {
          ctx.drawImage(img, sx, sy, ts, ts, dx, dy, ts, ts);
        }
      }
    }
  }
}

export function isBlocked(x, y) {
  if (x < 0 || y < 0 || x >= map.width || y >= map.height) return true;
  return map.blocked[y * map.width + x] === "1";
}

// 4-direction A* with a Manhattan heuristic; returns tiles excluding the start, or null when unreachable.
export function findPath(from, to) {
  if (from.x === to.x && from.y === to.y) return [];
  if (isBlocked(to.x, to.y)) return null;
  const w = map.width;
  const key = (x, y) => y * w + x;
  const h = (x, y) => Math.abs(x - to.x) + Math.abs(y - to.y);
  const g = new Map([[key(from.x, from.y), 0]]);
  const parent = new Map();
  const open = [{ x: from.x, y: from.y, f: h(from.x, from.y) }];
  const closed = new Set();
  while (open.length) {
    let best = 0;
    for (let i = 1; i < open.length; i++) if (open[i].f < open[best].f) best = i;
    const cur = open.splice(best, 1)[0];
    const ck = key(cur.x, cur.y);
    if (closed.has(ck)) continue;
    if (cur.x === to.x && cur.y === to.y) {
      const path = [];
      for (let k = ck; k !== key(from.x, from.y); k = parent.get(k)) path.push({ x: k % w, y: Math.floor(k / w) });
      return path.reverse();
    }
    closed.add(ck);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = cur.x + dx;
      const ny = cur.y + dy;
      if (isBlocked(nx, ny)) continue;
      const nk = key(nx, ny);
      const ng = g.get(ck) + 1;
      if (closed.has(nk) || ng >= (g.get(nk) ?? Infinity)) continue;
      g.set(nk, ng);
      parent.set(nk, ck);
      open.push({ x: nx, y: ny, f: ng + h(nx, ny) });
    }
  }
  return null;
}
