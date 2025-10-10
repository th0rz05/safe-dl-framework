const canvas = document.getElementById("board");
const context = canvas.getContext("2d");
const nextCanvas = document.getElementById("next");
const nextContext = nextCanvas.getContext("2d");

context.scale(30, 30);
nextContext.scale(24, 24);

const arena = createMatrix(10, 20);

const colors = {
  0: "#0b1420",
  1: "#4f46e5",
  2: "#22d3ee",
  3: "#f97316",
  4: "#22c55e",
  5: "#a855f7",
  6: "#eab308",
  7: "#ef4444",
};

const scoreEl = document.getElementById("score");
const linesEl = document.getElementById("lines");
const levelEl = document.getElementById("level");
const restartBtn = document.getElementById("restart");

const state = {
  dropCounter: 0,
  dropInterval: 1000,
  lastTime: 0,
  pause: false,
  score: 0,
  lines: 0,
  level: 1,
  queue: [],
};

const player = {
  matrix: null,
  pos: { x: 0, y: 0 },
};

function createMatrix(w, h) {
  const matrix = [];
  while (h--) {
    matrix.push(new Array(w).fill(0));
  }
  return matrix;
}

function createPiece(type) {
  switch (type) {
    case "T":
      return [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
      ];
    case "O":
      return [
        [2, 2],
        [2, 2],
      ];
    case "L":
      return [
        [0, 3, 0],
        [0, 3, 0],
        [0, 3, 3],
      ];
    case "J":
      return [
        [0, 0, 4],
        [0, 0, 4],
        [0, 4, 4],
      ];
    case "I":
      return [
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
      ];
    case "S":
      return [
        [0, 6, 6],
        [6, 6, 0],
        [0, 0, 0],
      ];
    case "Z":
      return [
        [7, 7, 0],
        [0, 7, 7],
        [0, 0, 0],
      ];
    default:
      return [[1]];
  }
}

function merge(arena, player) {
  player.matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value !== 0) {
        arena[y + player.pos.y][x + player.pos.x] = value;
      }
    });
  });
}

function collide(arena, player) {
  const [m, o] = [player.matrix, player.pos];
  for (let y = 0; y < m.length; y++) {
    for (let x = 0; x < m[y].length; x++) {
      if (m[y][x] !== 0 && (arena[y + o.y] && arena[y + o.y][x + o.x]) !== 0) {
        return true;
      }
    }
  }
  return false;
}

function rotate(matrix, dir) {
  for (let y = 0; y < matrix.length; ++y) {
    for (let x = 0; x < y; ++x) {
      [matrix[x][y], matrix[y][x]] = [matrix[y][x], matrix[x][y]];
    }
  }
  if (dir > 0) {
    matrix.forEach((row) => row.reverse());
  } else {
    matrix.reverse();
  }
}

function playerReset() {
  if (state.queue.length <= 3) {
    refillQueue();
  }
  const piece = state.queue.shift();
  player.matrix = createPiece(piece);
  player.pos.y = 0;
  player.pos.x = ((arena[0].length / 2) | 0) - ((player.matrix[0].length / 2) | 0);

  if (collide(arena, player)) {
    arena.forEach((row) => row.fill(0));
    updateScore(true);
  }
  drawNext();
}

function refillQueue() {
  const pieces = ["T", "O", "L", "J", "I", "S", "Z"];
  for (let i = pieces.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pieces[i], pieces[j]] = [pieces[j], pieces[i]];
  }
  state.queue.push(...pieces);
}

function arenaSweep() {
  let rowCount = 1;
  let cleared = 0;
  outer: for (let y = arena.length - 1; y >= 0; --y) {
    for (let x = 0; x < arena[y].length; ++x) {
      if (arena[y][x] === 0) {
        continue outer;
      }
    }
    const row = arena.splice(y, 1)[0].fill(0);
    arena.unshift(row);
    ++y;
    state.score += rowCount * 100;
    cleared += 1;
    rowCount *= 2;
  }

  if (cleared > 0) {
    state.lines += cleared;
    const newLevel = 1 + Math.floor(state.lines / 10);
    if (newLevel !== state.level) {
      state.level = newLevel;
      state.dropInterval = Math.max(120, 1000 - (state.level - 1) * 80);
    }
  }
}

function update(time = 0) {
  if (state.pause) {
    state.lastTime = time;
    requestAnimationFrame(update);
    return;
  }

  const deltaTime = time - state.lastTime;
  state.lastTime = time;
  state.dropCounter += deltaTime;
  if (state.dropCounter > state.dropInterval) {
    playerDrop();
  }
  draw();
  requestAnimationFrame(update);
}

function drawMatrix(matrix, offset, ctx = context) {
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value !== 0) {
        ctx.fillStyle = colors[value];
        ctx.fillRect(x + offset.x, y + offset.y, 1, 1);
        ctx.fillStyle = "rgba(8,12,20,0.35)";
        ctx.fillRect(x + offset.x, y + offset.y, 1, 0.2);
      }
    });
  });
}

function draw() {
  context.fillStyle = colors[0];
  context.fillRect(0, 0, canvas.width, canvas.height);
  drawMatrix(arena, { x: 0, y: 0 });
  drawMatrix(player.matrix, player.pos);
}

function drawNext() {
  nextContext.save();
  nextContext.setTransform(1, 0, 0, 1, 0, 0);
  nextContext.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
  nextContext.restore();

  nextContext.fillStyle = colors[0];
  nextContext.fillRect(0, 0, nextCanvas.width, nextCanvas.height);

  const preview = state.queue[0];
  if (!preview) {
    return;
  }
  const matrix = createPiece(preview);
  const offset = {
    x: Math.floor((4 - matrix[0].length) / 2),
    y: Math.floor((4 - matrix.length) / 2),
  };
  drawMatrix(matrix, offset, nextContext);
}

function updateScore(reset = false) {
  if (reset) {
    state.score = 0;
    state.lines = 0;
    state.level = 1;
    state.dropInterval = 1000;
  }
  scoreEl.textContent = state.score;
  linesEl.textContent = state.lines;
  levelEl.textContent = state.level;
}

function playerDrop() {
  player.pos.y++;
  if (collide(arena, player)) {
    player.pos.y--;
    merge(arena, player);
    arenaSweep();
    updateScore();
    playerReset();
  }
  state.dropCounter = 0;
}

function playerMove(dir) {
  player.pos.x += dir;
  if (collide(arena, player)) {
    player.pos.x -= dir;
  }
}

function playerRotate(dir) {
  const pos = player.pos.x;
  let offset = 1;
  rotate(player.matrix, dir);
  while (collide(arena, player)) {
    player.pos.x += offset;
    offset = -(offset + (offset > 0 ? 1 : -1));
    if (offset > player.matrix[0].length) {
      rotate(player.matrix, -dir);
      player.pos.x = pos;
      return;
    }
  }
}

function hardDrop() {
  while (!collide(arena, player)) {
    player.pos.y++;
  }
  player.pos.y--;
  merge(arena, player);
  arenaSweep();
  updateScore();
  playerReset();
  state.dropCounter = 0;
}

function togglePause() {
  state.pause = !state.pause;
  restartBtn.textContent = state.pause ? "Resume" : "Restart";
}

function restartGame() {
  arena.forEach((row) => row.fill(0));
  state.queue.length = 0;
  refillQueue();
  playerReset();
  updateScore(true);
}

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") {
    playerMove(-1);
  } else if (event.key === "ArrowRight") {
    playerMove(1);
  } else if (event.key === "ArrowDown") {
    playerDrop();
  } else if (event.key === "ArrowUp") {
    playerRotate(1);
  } else if (event.code === "Space") {
    hardDrop();
  } else if (event.key.toLowerCase() === "p") {
    togglePause();
  }
});

restartBtn.addEventListener("click", () => {
  if (state.pause) {
    togglePause();
  }
  restartGame();
});

restartGame();
updateScore();
update();
