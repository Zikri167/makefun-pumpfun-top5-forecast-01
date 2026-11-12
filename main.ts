// pumpfun-top5-forecast-01 entry (Node 22, stdlib only, deterministic).
//
// Reasoning (mine, validated on local split-halves in this session):
//   - The single forecast is scored at 7 reads, 7-13 h after the cut in the
//     final, so every probability is the equal-weight mean over the read
//     horizons taken from task.json (never hardcoded).
//   - At those horizons the top 5 is dominated by coins not yet born: in the
//     practice rehearsal 3 of the last read's top 5 were unnameable, and even
//     at 10-30 min horizons most top-5 slots went to coins absent from the
//     previous mark. So known coins get modest mass and the mass a newcomer
//     would take is honestly left unnamed, by simulating synthetic fresh
//     coins (peak caps calibrated from the history) alongside the known ones.
//   - Each known coin's log10 cap is projected with a saturating drift
//     (a coin cannot compound for 10 h), diffusion growing with horizon,
//     and a crash-to-floor jump: young volatile coins die fast, proven
//     survivors (old pool coins still rising) die slowly.
//   - Curve coins cannot sit above the graduation cap (~300 SOL) without
//     graduating, so draws above it are mostly pulled back to the cap.
//   - Coins that are old, low and flat are named at p=0: the rules take
//     them out of the unnamed share's denominator, leaving more of it to
//     the newcomers. Never zero a top-11 coin or a young climber.
//   - One seeded Monte Carlo pass tallies both place probabilities and
//     place-cap bins; top5 is ordered by expected place under the sim.
// No network, no wall clock, no Math.random: a fixed-seed PRNG only, so two
// runs are byte-identical. Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- tunables (chosen before seeing split results, then confirmed by
// variant tests in this session: NEW_MU0 2.55 scored -29.54 and 2.85 scored
// -31.94 read-weighted vs 2.7's -28.47; NEWCOMERS 16 scored -28.93 and 32
// scored -29.26 vs 24's -28.47, so 2.7 and 24 are kept) ----
const SIMS = 8000; // Monte Carlo draws per read horizon (raised from 6000 in v2 to cut MC noise; runtime still ~1.5 s)
const NEWCOMERS_PER_SIM = 24; // synthetic fresh coins per draw
const NEW_MU0 = 2.7; // log10 SOL, median fresh-coin draw at the cut
const NEW_MU_PER_H = 0.015; // newcomer scale drifts up slowly with horizon
const NEW_SD = 0.42;
const DRIFT_TAU_H = 1.0; // drift saturates over ~1 h
const DRIFT_SHRINK = 0.5; // shrink observed slopes halfway to zero
const DRIFT_CAP = 0.45; // max |drift| in log10 per hour
const SIG0 = 0.12; // log10 sd at the cut
const SIG_CAP = 0.85;
const GRAD_CAP_LOG = 2.5; // ~316 SOL: a curve coin cannot exceed this ungraduated
const FLOOR_LOG = 1.447; // ~28 SOL bonding-curve floor
const MIN_P = 0.002; // smallest named probability emitted
const MICRO = 1_000_000;

interface TaskJson {
  places: number;
  output: string;
  reads?: string[];
  history_until?: string;
  size: { edges: number[] };
}
interface HistRow {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}
interface Mark {
  at: string;
  ok: boolean;
  rows: HistRow[] | null;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as TaskJson;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const cutMs = Date.parse(lastMark.at);
if (!Number.isFinite(cutMs)) throw new Error('last mark has no parseable time');

// Horizons in hours after the cut, from task.json.
const readTimes = (task.reads ?? []).map((r) => Date.parse(r)).filter((t) => Number.isFinite(t) && t > cutMs);
const horizonsH = readTimes.length > 0
  ? readTimes.map((t) => (t - cutMs) / 3_600_000)
  : [10];
const nH = horizonsH.length;

// ---- seeded deterministic PRNG (mulberry32) + Box-Muller normal ----
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(0x9e3779b9);
let spare: number | null = null;
function randn(): number {
  if (spare !== null) {
    const v = spare;
    spare = null;
    return v;
  }
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  const th = 2 * Math.PI * v;
  spare = r * Math.sin(th);
  return r * Math.cos(th);
}

// ---- per-coin state at the cut ----
const logSol = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

interface Coin {
  mint: string;
  rank: number;
  logcap: number;
  ageMin: number;
  venue: string;
  slopePerH: number;
  sigRate: number;
  p10death: number;
}

const traj = new Map<string, Array<{ t: number; logcap: number }>>();
for (const m of marks) {
  const t = Date.parse(m.at);
  for (const r of m.rows!) {
    const arr = traj.get(r.mint) ?? [];
    arr.push({ t, logcap: logSol(r.market_cap_lamports) });
    traj.set(r.mint, arr);
  }
}

// Least-squares slope over the last 60 min, and sd of 10-min steps.
function slopeAndVol(pts: Array<{ t: number; logcap: number }>): { slope: number; vol10: number } {
  const tEnd = pts[pts.length - 1]!.t;
  const w = pts.filter((p) => tEnd - p.t <= 3_600_000);
  const use = w.length >= 2 ? w : pts;
  let slope = 0;
  if (use.length >= 2) {
    const n = use.length;
    const mt = use.reduce((s, p) => s + p.t, 0) / n;
    const ml = use.reduce((s, p) => s + p.logcap, 0) / n;
    let num = 0;
    let den = 0;
    for (const p of use) {
      num += (p.t - mt) * (p.logcap - ml);
      den += (p.t - mt) * (p.t - mt);
    }
    if (den > 0) slope = (num / den) * 3_600_000; // log10 per hour
  }
  let vol10 = 0.1;
  if (pts.length >= 3) {
    const ds: number[] = [];
    for (let i = 1; i < pts.length; i++) ds.push(pts[i]!.logcap - pts[i - 1]!.logcap);
    const m = ds.reduce((s, d) => s + d, 0) / ds.length;
    vol10 = Math.sqrt(ds.reduce((s, d) => s + (d - m) * (d - m), 0) / ds.length);
  }
  return { slope, vol10 };
}

// Per-10-min crash-to-floor probability. Young pool coins crash often
// (12-19% per 10 min in the data); old coins that are still rising almost
// never do; curve coins sit near the floor unless they graduate.
function deathRate(ageMin: number, venue: string, slopePerH: number, logcap: number): number {
  if (venue === 'pool' || venue === 'migrating') {
    if (ageMin >= 30 && slopePerH >= 0) return 0.01;
    if (ageMin >= 30) return 0.02;
    return 0.12;
  }
  if (logcap >= 2.0 && slopePerH > 0.1) return 0.1;
  return 0.2;
}

const lastRows = [...lastMark.rows!].sort((a, b) => a.rank - b.rank);
const coins: Coin[] = [];
for (const r of lastRows) {
  const pts = traj.get(r.mint) ?? [{ t: cutMs, logcap: logSol(r.market_cap_lamports) }];
  const { slope, vol10 } = slopeAndVol(pts);
  const ageMin = Math.max(0, (cutMs - Date.parse(r.created_at)) / 60_000);
  const shrunk = Math.max(-0.3, Math.min(DRIFT_CAP, slope * DRIFT_SHRINK));
  const sigRate = Math.max(0.12, Math.min(0.6, vol10 * Math.sqrt(6) * 0.7 + 0.12));
  coins.push({
    mint: r.mint,
    rank: r.rank,
    logcap: logSol(r.market_cap_lamports),
    ageMin,
    venue: r.venue,
    slopePerH: shrunk,
    sigRate,
    p10death: deathRate(ageMin, r.venue, shrunk, logSol(r.market_cap_lamports)),
  });
}

// Candidates: the top of the chart by current size plus bonuses for young
// risers and proven growers. Ranks 6+ essentially never climb in, so the
// candidate set is capped at 120.
function potential(c: Coin): number {
  let s = c.logcap;
  if (c.ageMin < 30 && c.slopePerH > 0.1) s += 0.3;
  if (c.venue === 'pool' && c.ageMin >= 20 && c.slopePerH > 0) s += 0.15;
  return s;
}
const ranked = [...coins].sort((a, b) => potential(b) - potential(a));
const candidates = ranked.slice(0, 120);
const candSet = new Set(candidates.map((c) => c.mint));

// Zero-list: old/low and flat-or-falling coins, near-certain never to hold a
// top-5 place. Named at p=0 so the rules drop them from the unnamed share's
// denominator. Never zero a top-11 coin or a young climber.
const zeroMints: string[] = [];
for (const c of coins) {
  if (candSet.has(c.mint)) continue;
  if (c.rank < 12) continue;
  const oldEnough = c.ageMin >= 30;
  const low = c.logcap < 1.65; // under ~45 SOL
  const flatOrFalling = c.slopePerH <= 0.05;
  if ((oldEnough && flatOrFalling) || (low && flatOrFalling)) zeroMints.push(c.mint);
}

// ---- Monte Carlo ----
const P = task.places;
const edges = task.size.edges;
const placeCoinCounts: Array<Map<string, number>> = Array.from({ length: P }, () => new Map());
const placeBinCounts: Array<Map<number, number>> = Array.from({ length: P }, () => new Map());
const totalDraws = SIMS * nH;

function binOf(logcap: number): number {
  if (logcap < edges[0]!) return 0;
  if (logcap >= edges[edges.length - 1]!) return edges.length;
  let lo = 0;
  let hi = edges.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (edges[mid]! <= logcap) lo = mid;
    else hi = mid - 1;
  }
  return lo + 1;
}

for (let hIdx = 0; hIdx < nH; hIdx++) {
  const h = horizonsH[hIdx]!;
  const driftFactor = DRIFT_TAU_H * (1 - Math.exp(-h / DRIFT_TAU_H));
  const newMu = NEW_MU0 + NEW_MU_PER_H * h;
  for (let s = 0; s < SIMS; s++) {
    const drawn: Array<{ mint: string; logcap: number }> = new Array(candidates.length + NEWCOMERS_PER_SIM);
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i]!;
      const pDead = Math.min(0.9, 1 - Math.pow(1 - c.p10death, 6 * h));
      let lc: number;
      if (rand() < pDead) {
        lc = FLOOR_LOG + randn() * 0.1;
      } else {
        const mu = c.logcap + c.slopePerH * driftFactor;
        const sigma = Math.min(SIG_CAP, Math.sqrt(SIG0 * SIG0 + c.sigRate * c.sigRate * h));
        lc = mu + randn() * sigma;
        if (c.venue === 'curve' && lc > GRAD_CAP_LOG) {
          // A curve coin past the graduation cap must have graduated to reach it.
          const pGrad = c.slopePerH > 0.1 && c.ageMin < 45 ? 0.5 : 0.2;
          if (rand() >= pGrad) lc = GRAD_CAP_LOG + Math.abs(randn()) * 0.02;
        }
      }
      drawn[i] = { mint: c.mint, logcap: lc };
    }
    for (let j = 0; j < NEWCOMERS_PER_SIM; j++) {
      drawn[candidates.length + j] = { mint: '', logcap: newMu + randn() * NEW_SD };
    }
    // Track the top P caps (newcomers have empty mint: their mass stays unnamed).
    const top: Array<{ mint: string; logcap: number }> = [];
    for (const d of drawn) {
      if (top.length < P) {
        top.push(d);
        top.sort((a, b) => b.logcap - a.logcap || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
      } else if (d.logcap > top[P - 1]!.logcap) {
        top[P - 1] = d;
        top.sort((a, b) => b.logcap - a.logcap || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
      }
    }
    for (let k = 0; k < P; k++) {
      const d = top[k]!;
      const bc = placeBinCounts[k]!;
      bc.set(binOf(d.logcap), (bc.get(binOf(d.logcap)) ?? 0) + 1);
      if (d.mint !== '') {
        const cc = placeCoinCounts[k]!;
        cc.set(d.mint, (cc.get(d.mint) ?? 0) + 1);
      }
    }
  }
}

// ---- assemble the forecast ----
interface ProbRow { mint: string; place: number; p: number }
interface SizeRow { place: number; bin: number; p: number }
const probabilities: ProbRow[] = [];
const sizes: SizeRow[] = [];
const namedAtPlace: Array<Set<string>> = Array.from({ length: P }, () => new Set());

for (let k = 0; k < P; k++) {
  const counts = [...placeCoinCounts[k]!.entries()]
    .map(([mint, n]) => ({ mint, p: n / totalDraws }))
    .filter((e) => e.p >= MIN_P)
    .sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : 1));
  const room = 100 - counts.length;
  for (const e of counts.slice(0, 100)) {
    const u = Math.floor(e.p * MICRO);
    if (u > 0) {
      probabilities.push({ mint: e.mint, place: k + 1, p: u / MICRO });
      namedAtPlace[k]!.add(e.mint);
    }
  }
  let added = 0;
  for (const mint of zeroMints) {
    if (added >= Math.max(0, room)) break;
    if (namedAtPlace[k]!.has(mint)) continue;
    probabilities.push({ mint, place: k + 1, p: 0 });
    namedAtPlace[k]!.add(mint);
    added++;
  }
  const bins = [...placeBinCounts[k]!.entries()]
    .map(([bin, n]) => ({ bin, p: n / totalDraws }))
    .filter((e) => e.p >= MIN_P)
    .sort((a, b) => b.p - a.p || a.bin - b.bin);
  for (const e of bins) {
    const u = Math.floor(e.p * MICRO);
    if (u > 0) sizes.push({ place: k + 1, bin: e.bin, p: u / MICRO });
  }
}

// top5: order by expected place under the simulated distribution.
const coinPlace = new Map<string, number[]>();
for (let k = 0; k < P; k++) {
  for (const [mint, n] of placeCoinCounts[k]!) {
    const arr = coinPlace.get(mint) ?? [0, 0, 0, 0, 0];
    arr[k] = n / totalDraws;
    coinPlace.set(mint, arr);
  }
}
const expectedPlace = [...coinPlace.entries()].map(([mint, ps]) => {
  const mass = ps.reduce((s, p) => s + p, 0);
  const exp = ps.reduce((s, p, k) => s + p * (k + 1), 0) + (1 - mass) * 6;
  return { mint, exp, mass };
});
expectedPlace.sort((a, b) => a.exp - b.exp || b.mass - a.mass || (a.mint < b.mint ? -1 : 1));
let top5 = expectedPlace.slice(0, P).map((e) => e.mint);
if (top5.length < P) {
  for (const r of lastRows) {
    if (top5.length >= P) break;
    if (!top5.includes(r.mint)) top5.push(r.mint);
  }
}
top5 = top5.slice(0, P);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
