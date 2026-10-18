// Tuxemon overworld sheets (3 cols x 4 rows of 16x32) and portraits (front view: 64x64 at (64,0)).

const ROW = { down: 0, left: 1, right: 2, up: 3 };
const WALK = [0, 1, 2, 1];
const portraitHeights = new Map();

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

export async function loadSprites(names) {
  const sheets = {};
  await Promise.all(names.map(async (n) => {
    sheets[n] = await loadImage(`assets/tuxemon/sprites/${n}.png`);
    const portrait = await loadImage(`assets/tuxemon/portraits/${n}.png`);
    portraitHeights.set(n, portrait.naturalHeight);
  }));
  return sheets;
}

// px/py are tile coordinates (fractional while walking); the 16x32 frame's feet sit on the tile.
export function drawCharacter(ctx, img, px, py, dir, moving, now, camX, camY) {
  const col = moving ? WALK[Math.floor(now / 120) % 4] : 1;
  ctx.drawImage(img, col * 16, ROW[dir] * 32, 16, 32,
    Math.round(px * 16 - camX), Math.round(py * 16 - 16 - camY), 16, 32);
}

function portraitUrl(sprite) {
  return `url("assets/tuxemon/portraits/${sprite}.png")`;
}

// Front view (64x64 crop at (64,0)) rendered at `scale`.
export function frontPortrait(el, sprite, scale) {
  const h = portraitHeights.get(sprite) ?? 64;
  el.style.backgroundImage = portraitUrl(sprite);
  el.style.backgroundSize = `${128 * scale}px ${h * scale}px`;
  el.style.backgroundPosition = `${-64 * scale}px 0`;
  el.style.width = `${64 * scale}px`;
  el.style.height = `${64 * scale}px`;
}
