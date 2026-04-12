const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const keys = new Set();
window.addEventListener("keydown", (e) => {
  keys.add(e.key.toLowerCase());
  if ([" ", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(e.key.toLowerCase())) e.preventDefault();
  if (game.state === "reward" && ["1", "2", "3"].includes(e.key)) pickReward(Number(e.key) - 1);
  if (game.state === "install" && ["1", "2", "3", "4"].includes(e.key)) installCore(Number(e.key) - 1);
  if (game.state === "title" && e.key.toLowerCase() === "enter") startRun();
  if (game.state === "gameover" && e.key.toLowerCase() === "enter") game.state = "title";
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

const SLOT_KEYS = ["head", "body", "arms", "legs"];
const SLOT_JP = { head: "頭", body: "胴", arms: "腕", legs: "脚" };

const BASE_REWARDS = [
  { name: "疾風脚", slot: "legs", desc: "移動速度 +0.6", apply: (p) => p.stats.speed += 0.6 },
  { name: "連弩腕", slot: "arms", desc: "攻撃速度 +20%", apply: (p) => p.stats.fireRate *= 0.8 },
  { name: "鉄壁胴", slot: "body", desc: "ノックバック軽減", apply: (p) => p.stats.knockbackResist += 0.25 },
  { name: "慧眼頭", slot: "head", desc: "クリティカル率 +8%", apply: (p) => p.stats.crit += 0.08 },
  { name: "重跳脚", slot: "legs", desc: "ジャンプ力 +2", apply: (p) => p.stats.jump += 2 },
  { name: "魔導胴", slot: "body", desc: "最大ショット数 +1", apply: (p) => p.stats.maxShots += 1 },
];

const BOSS_CORES = {
  inferno: {
    name: "爆炎コア",
    color: "#ff6f3c",
    install: {
      arms: "特殊: 火炎貫通弾 (K)",
      legs: "移動: 空中ホバー (K長押し)",
      body: "防御: 被弾時に炎衝撃波",
      head: "支援: 被弾直後の短時間攻撃力UP",
    },
  },
};

const game = {
  state: "title",
  stage: 1,
  areaKills: 0,
  killTarget: 12,
  rewardChoices: [],
  souls: Number(localStorage.getItem("masou_souls") || 0),
  metaAtk: Number(localStorage.getItem("masou_meta_atk") || 0),
  metaSpd: Number(localStorage.getItem("masou_meta_spd") || 0),
  pendingCore: null,
  worldWidth: 2400,
  cameraX: 0,
};

const player = {
  x: 80, y: 400, w: 28, h: 40, vx: 0, vy: 0,
  onGround: false, facing: 1, inv: 0, boostTimer: 0, hoverFuel: 75,
  stats: {}, armors: {}, cores: {},
};

let bullets = [];
let enemyBullets = [];
let enemies = [];
let boss = null;
let elapsed = 0;
let spawnTimer = 0;
let fireTimer = 0;
let specialTimer = 0;

function resetPlayerForRun() {
  player.x = 80; player.y = 400; player.vx = 0; player.vy = 0;
  player.stats = {
    speed: 3.2 + game.metaSpd * 0.15,
    jump: 10,
    fireRate: 0.32,
    damage: 1 + game.metaAtk * 0.1,
    crit: 0.05,
    knockbackResist: 0,
    maxShots: 2,
  };
  player.armors = { head: true, body: true, arms: true, legs: true };
  player.cores = {};
  player.inv = 0;
  player.boostTimer = 0;
  player.hoverFuel = 75;
}

function startRun() {
  game.stage = 1;
  game.areaKills = 0;
  game.killTarget = 10;
  game.pendingCore = null;
  game.worldWidth = 2400;
  resetPlayerForRun();
  resetStage();
  game.state = "play";
}

function resetStage() {
  bullets = [];
  enemyBullets = [];
  enemies = [];
  boss = null;
  elapsed = 0;
  spawnTimer = 0;
  fireTimer = 0;
  specialTimer = 0;
  player.x = 80;
  game.cameraX = 0;
}

function armorCount() { return SLOT_KEYS.filter((s) => player.armors[s]).length; }

function spawnEnemy() {
  const t = game.stage;
  enemies.push({
    x: Math.min(game.worldWidth - 50, game.cameraX + canvas.width + Math.random() * 300),
    y: 420,
    w: 26,
    h: 30,
    hp: 1 + Math.floor(t / 2),
    vx: -(1.1 + t * 0.12),
    cd: 0.7 + Math.random() * 1.2,
  });
}

function beginReward() {
  game.rewardChoices = [...BASE_REWARDS].sort(() => Math.random() - 0.5).slice(0, 3);
  game.state = "reward";
}

function pickReward(idx) {
  const r = game.rewardChoices[idx];
  if (!r) return;
  r.apply(player);
  player.armors[r.slot] = true;
  game.state = "boss";
  spawnBoss();
}

function spawnBoss() {
  boss = {
    x: game.worldWidth - 300,
    y: 360,
    w: 90,
    h: 110,
    hp: 28 + game.stage * 6,
    maxHp: 28 + game.stage * 6,
    vx: -1,
    fire: 1.1,
  };
}

function installCore(idx) {
  const slot = SLOT_KEYS[idx];
  if (!slot || !game.pendingCore) return;
  player.cores[slot] = game.pendingCore;
  game.pendingCore = null;
  game.stage += 1;
  game.killTarget = 10 + game.stage * 3;
  game.areaKills = 0;
  game.worldWidth += 600;
  resetStage();
  game.state = "play";
}

function loseArmor() {
  if (player.inv > 0) return;
  const avail = SLOT_KEYS.filter((s) => player.armors[s]);
  if (!avail.length) {
    gameOver();
    return;
  }
  const s = avail[Math.floor(Math.random() * avail.length)];
  player.armors[s] = false;
  if (player.cores[s] && player.cores[s].name === "爆炎コア" && s === "body") {
    for (const e of enemies) if (Math.abs(e.x - player.x) < 120) e.hp -= 3;
  }
  player.inv = 1.0;
  if (!armorCount()) setTimeout(() => {}, 0);
}

function gameOver() {
  const gain = Math.floor((game.stage - 1) * 8 + game.areaKills * 0.7);
  game.souls += gain;
  localStorage.setItem("masou_souls", game.souls);
  game.lastGain = gain;
  game.state = "gameover";
}

function upgradeMeta(type) {
  const key = type === "atk" ? "metaAtk" : "metaSpd";
  const cost = 10 + game[key] * 8;
  if (game.souls < cost) return;
  game.souls -= cost;
  game[key] += 1;
  localStorage.setItem("masou_souls", game.souls);
  localStorage.setItem(type === "atk" ? "masou_meta_atk" : "masou_meta_spd", game[key]);
}

window.addEventListener("keydown", (e) => {
  if (game.state === "title") {
    if (e.key === "1") upgradeMeta("atk");
    if (e.key === "2") upgradeMeta("spd");
  }
});

function update(dt) {
  if (!["play", "boss"].includes(game.state)) return;
  elapsed += dt;
  if (player.inv > 0) player.inv -= dt;
  if (player.boostTimer > 0) player.boostTimer -= dt;

  const left = keys.has("a") || keys.has("arrowleft");
  const right = keys.has("d") || keys.has("arrowright");
  const jump = keys.has("w") || keys.has(" ") || keys.has("space");
  const shot = keys.has("j") || keys.has("z");
  const special = keys.has("k") || keys.has("x");

  let move = (right ? 1 : 0) - (left ? 1 : 0);
  player.vx = move * player.stats.speed;
  if (move) player.facing = move;

  if (jump && player.onGround) {
    player.vy = -player.stats.jump;
    player.onGround = false;
  }

  if (player.cores.legs?.name === "爆炎コア" && special && !player.onGround && player.hoverFuel > 0) {
    player.vy = Math.min(player.vy, -0.3);
    player.hoverFuel -= 24 * dt;
  } else if (player.onGround) {
    player.hoverFuel = Math.min(75, player.hoverFuel + 32 * dt);
  }

  player.vy += 24 * dt;
  player.x = Math.max(0, Math.min(game.worldWidth - player.w, player.x + player.vx * 60 * dt));
  player.y += player.vy;
  if (player.y > 420) {
    player.y = 420;
    player.vy = 0;
    player.onGround = true;
  }

  game.cameraX = Math.max(0, Math.min(game.worldWidth - canvas.width, player.x - 260));

  fireTimer -= dt;
  const maxShots = Math.floor(player.stats.maxShots);
  if (shot && fireTimer <= 0 && bullets.length < maxShots) {
    const crit = Math.random() < player.stats.crit;
    bullets.push({ x: player.x + player.w / 2, y: player.y + 16, vx: 8 * player.facing, p: crit ? 2.2 : 1.2, pierce: false });
    fireTimer = player.stats.fireRate;
  }

  specialTimer -= dt;
  if (special && specialTimer <= 0 && player.cores.arms?.name === "爆炎コア") {
    bullets.push({ x: player.x + player.w / 2, y: player.y + 16, vx: 10 * player.facing, p: 4.2, pierce: true, flame: true });
    specialTimer = 0.9;
  }

  if (game.state === "play") {
    spawnTimer -= dt;
    if (spawnTimer <= 0) {
      spawnEnemy();
      spawnTimer = Math.max(0.35, 1.2 - game.stage * 0.08);
    }
  }

  for (const b of bullets) b.x += b.vx;
  bullets = bullets.filter((b) => b.x > 0 && b.x < game.worldWidth);

  for (const e of enemies) {
    e.x += e.vx;
    e.cd -= dt;
    if (e.cd <= 0 && Math.abs(e.x - player.x) < 480) {
      enemyBullets.push({ x: e.x + e.w / 2, y: e.y + 12, vx: e.vx * 2.5 });
      e.cd = 1.5 + Math.random() * 1.2;
    }
  }

  for (const eb of enemyBullets) eb.x += eb.vx;
  enemyBullets = enemyBullets.filter((b) => b.x > game.cameraX - 60 && b.x < game.cameraX + canvas.width + 60);

  for (const b of bullets) {
    for (const e of enemies) {
      if (hit(b, e)) {
        e.hp -= (player.boostTimer > 0 ? 1.4 : 1) * player.stats.damage * b.p;
        if (!b.pierce) b.dead = true;
      }
    }
    if (boss && hit(b, boss)) {
      boss.hp -= (player.boostTimer > 0 ? 1.4 : 1) * player.stats.damage * b.p;
      if (!b.pierce) b.dead = true;
    }
  }
  bullets = bullets.filter((b) => !b.dead);

  enemies = enemies.filter((e) => {
    if (e.hp <= 0) {
      game.areaKills += 1;
      return false;
    }
    return e.x > -100;
  });

  if (game.state === "play" && game.areaKills >= game.killTarget) {
    beginReward();
  }

  if (boss) {
    boss.x += boss.vx;
    if (boss.x < game.worldWidth - 600 || boss.x > game.worldWidth - 120) boss.vx *= -1;
    boss.fire -= dt;
    if (boss.fire <= 0) {
      enemyBullets.push({ x: boss.x, y: boss.y + 40, vx: -4 });
      enemyBullets.push({ x: boss.x, y: boss.y + 40, vx: -2.8 });
      boss.fire = 0.55;
    }
    if (boss.hp <= 0) {
      game.pendingCore = BOSS_CORES.inferno;
      boss = null;
      game.state = "install";
    }
  }

  for (const eb of enemyBullets) {
    if (hitPoint(eb.x, eb.y, player)) {
      loseArmor();
      if (player.cores.head?.name === "爆炎コア") player.boostTimer = 2.5;
      eb.dead = true;
    }
  }
  for (const e of enemies) {
    if (hit(player, e)) loseArmor();
  }
  if (boss && hit(player, boss)) loseArmor();

  enemyBullets = enemyBullets.filter((b) => !b.dead);
}

function hit(a, b) {
  return a.x < b.x + b.w && a.x + (a.w || 4) > b.x && a.y < b.y + b.h && a.y + (a.h || 4) > b.y;
}
function hitPoint(x, y, r) {
  return x > r.x && x < r.x + r.w && y > r.y && y < r.y + r.h;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (game.state === "title") return drawTitle();

  drawWorld();
  drawUI();

  if (game.state === "reward") drawReward();
  if (game.state === "install") drawInstall();
  if (game.state === "gameover") drawGameOver();
}

function drawTitle() {
  ctx.fillStyle = "#d9d9ff";
  ctx.font = "bold 56px sans-serif";
  ctx.fillText("魔装遊戯", 360, 150);
  ctx.font = "20px sans-serif";
  ctx.fillStyle = "#fff";
  ctx.fillText("Enterで出撃 / 1:攻撃永続UP / 2:速度永続UP", 230, 220);
  const c1 = 10 + game.metaAtk * 8;
  const c2 = 10 + game.metaSpd * 8;
  ctx.fillText(`魂: ${game.souls}  攻撃Lv:${game.metaAtk}(次${c1})  速度Lv:${game.metaSpd}(次${c2})`, 200, 260);
  ctx.fillStyle = "#b7b7c9";
  ctx.fillText("被弾すると魔装が1部位破壊。全損後の被弾でゲームオーバー。", 200, 320);
}

function drawWorld() {
  const cam = game.cameraX;
  for (let x = -cam % 64; x < canvas.width; x += 64) {
    ctx.fillStyle = "#1c1c37";
    ctx.fillRect(x, 460, 60, 80);
  }

  ctx.fillStyle = player.inv > 0 ? "#ffb0b0" : "#7be0ff";
  ctx.fillRect(player.x - cam, player.y, player.w, player.h);

  for (const e of enemies) {
    ctx.fillStyle = "#f06b6b";
    ctx.fillRect(e.x - cam, e.y, e.w, e.h);
  }
  for (const b of bullets) {
    ctx.fillStyle = b.flame ? "#ff8e3c" : "#fff07e";
    ctx.fillRect(b.x - cam, b.y, 8, 4);
  }
  for (const eb of enemyBullets) {
    ctx.fillStyle = "#ff3c6f";
    ctx.fillRect(eb.x - cam, eb.y, 6, 6);
  }
  if (boss) {
    ctx.fillStyle = "#be4b2d";
    ctx.fillRect(boss.x - cam, boss.y, boss.w, boss.h);
    ctx.fillStyle = "#000";
    ctx.fillRect(230, 16, 500, 16);
    ctx.fillStyle = "#ff6545";
    ctx.fillRect(230, 16, 500 * Math.max(0, boss.hp / boss.maxHp), 16);
  }
}

function drawUI() {
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(10, 10, 190, 118);
  ctx.fillStyle = "#fff";
  ctx.font = "16px sans-serif";
  ctx.fillText(`STAGE ${game.stage}`, 20, 30);
  ctx.fillText(`討伐 ${game.areaKills}/${game.killTarget}`, 20, 50);
  ctx.fillText(`魂 ${game.souls}`, 20, 70);
  ctx.fillText(`Hover ${Math.floor(player.hoverFuel)}`, 20, 90);

  SLOT_KEYS.forEach((s, i) => {
    ctx.fillStyle = player.armors[s] ? "#a1ffaf" : "#5f5f5f";
    ctx.fillText(`${SLOT_JP[s]}:${player.armors[s] ? "装備" : "破損"}`, 20, 112 + i * 18);
  });

  ctx.fillStyle = "#ddd";
  ctx.fillText(`Core[腕:${player.cores.arms?.name || "-"}]`, 660, 30);
  ctx.fillText(`Core[脚:${player.cores.legs?.name || "-"}]`, 660, 48);
}

function drawReward() {
  panel("宝箱報酬: 1〜3で選択", 160, 95, 640, 300);
  game.rewardChoices.forEach((r, i) => {
    ctx.fillStyle = "#fff";
    ctx.fillText(`${i + 1}. ${r.name} [${SLOT_JP[r.slot]}]`, 220, 160 + i * 80);
    ctx.fillStyle = "#cfcff0";
    ctx.fillText(r.desc, 260, 188 + i * 80);
  });
}

function drawInstall() {
  panel("コア・インストール: 1頭 2胴 3腕 4脚", 130, 80, 700, 340);
  ctx.fillStyle = "#fff";
  ctx.fillText(`獲得: ${game.pendingCore?.name}`, 180, 140);
  SLOT_KEYS.forEach((s, i) => {
    ctx.fillStyle = "#ffd5c5";
    ctx.fillText(`${i + 1}. ${SLOT_JP[s]} に装着`, 190, 190 + i * 52);
    ctx.fillStyle = "#fff";
    ctx.fillText(game.pendingCore.install[s], 320, 190 + i * 52);
  });
}

function drawGameOver() {
  panel("GAME OVER", 280, 150, 400, 180);
  ctx.fillStyle = "#fff";
  ctx.fillText(`獲得した魂: +${game.lastGain || 0}`, 340, 230);
  ctx.fillText("Enterでタイトルへ", 350, 265);
}

function panel(title, x, y, w, h) {
  ctx.fillStyle = "rgba(10,10,30,0.92)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#7d88ff";
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "#fff";
  ctx.font = "22px sans-serif";
  ctx.fillText(title, x + 20, y + 38);
  ctx.font = "18px sans-serif";
}

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;
  update(dt);
  draw();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
