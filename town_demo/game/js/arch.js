// The Valentine's heart arch over the south gate of Johnson Park, drawn in world pixels (16 px tiles) minus the
// camera. `done` is the number of completed stages: frame, assembly, paint, lights, roses.

const WOOD = "#b07840";
const WHITE = "#f4f0e8";
const OUTLINE = "#282828";
const RED = "#e03050";
const POSTS = [131, 170]; // left x of each 3 px post
const POST_TOP = 436;
const POST_BOTTOM = 464;
const BEAM = { x0: 129, x1: 175, y: 434, h: 4 };
const HEART = [".XX.XX.", "XXXXXXX", "XXXXXXX", ".XXXXX.", "..XXX..", "...X..."];
const HEART_CX = 152;
const HEART_CY = 428;
const LIGHT_Y = 433;
const ROSE_XS = [127, 135, 166, 174];
const ROSE_Y = 461;

export function drawArch(ctx, done, camX, camY, now) {
  if (done === 0) return;
  const rect = (x, y, w, h, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(x - camX, y - camY, w, h);
  };
  const painted = done >= 3;
  const beams = [
    ...POSTS.map((x) => [x, POST_TOP, 3, POST_BOTTOM - POST_TOP]),
    [BEAM.x0, BEAM.y, BEAM.x1 - BEAM.x0, BEAM.h],
  ];
  if (painted) for (const [x, y, w, h] of beams) rect(x - 1, y - 1, w + 2, h + 2, OUTLINE);
  for (const [x, y, w, h] of beams) rect(x, y, w, h, painted ? WHITE : WOOD);
  if (done >= 2) {
    const x0 = HEART_CX - Math.floor(HEART[0].length / 2);
    const y0 = HEART_CY - Math.floor(HEART.length / 2);
    HEART.forEach((row, dy) => {
      for (let dx = 0; dx < row.length; dx++) if (row[dx] === "X") rect(x0 + dx, y0 + dy, 1, 1, painted ? RED : WOOD);
    });
  }
  if (done >= 4) {
    const dots = [];
    for (let x = BEAM.x0 + 1; x < BEAM.x1; x += 4) dots.push([x, LIGHT_Y]);
    for (const px of POSTS) for (let y = POST_TOP + 2; y < POST_BOTTOM; y += 4) dots.push([px + 1, y]);
    const phase = Math.floor(now / 400) % 2;
    dots.forEach(([x, y], i) => rect(x, y, 1, 1, phase === i % 2 ? "#ffe060" : "#806020"));
  }
  if (done >= 5) {
    for (const x of ROSE_XS) {
      rect(x, ROSE_Y, 2, 2, "#e8457a");
      rect(x, ROSE_Y + 2, 1, 1, "#3a8a3a");
    }
  }
}
