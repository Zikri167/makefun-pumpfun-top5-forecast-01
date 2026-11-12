// v3-style seeded Monte Carlo forecast: 5,000 draws per read horizon.
// Simulates 24 unnamed newcomers with calibrated age and venue death rates.
// Selected model read-weighted split mean: -28.29 (A/B/C).
// Fixed seed and Node 22 standard-library implementation.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const input = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const output = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const task = JSON.parse(readFileSync(join(input, 'task.json'), 'utf8'));
const marks = readFileSync(join(input, 'history.ndjson'), 'utf8').trim().split('\n').map((s) => JSON.parse(s)).filter((m) => m.ok && Array.isArray(m.rows));
if (!marks.length) throw new Error('history has no readable marks');
const cut = Date.parse(marks[marks.length - 1].at);
const horizons = task.reads.map((s: string) => (Date.parse(s) - cut) / 3600000).filter((h: number) => h > 0);
const places = task.places;
const floor = 1.447;
const edges: number[] = task.size.edges;
type Point = { t: number; x: number };
type Row = { rank: number; mint: string; market_cap_lamports: string; venue: string; created_at: string; mayhem: boolean };
type Coin = { mint: string; rank: number; x: number; age: number; venue: string; mayhem: boolean; drift: number; vol: number; death: number };
const path = new Map<string, Point[]>();
for (const mark of marks) {
  const t = Date.parse(mark.at);
  for (const row of mark.rows as Row[]) {
    const xs = path.get(row.mint) ?? [];
    xs.push({ t, x: Math.log10(Number(BigInt(row.market_cap_lamports))) - 9 });
    path.set(row.mint, xs);
  }
}
const last = [...marks[marks.length - 1].rows as Row[]].sort((a, b) => a.rank - b.rank);
function features(points: Point[]): { slope: number; vol: number; peak: number } {
  const end = points[points.length - 1].t;
  const recent = points.filter((p) => end - p.t <= 3600000);
  const ps = recent.length > 1 ? recent : points;
  const mt = ps.reduce((s, p) => s + p.t, 0) / ps.length;
  const mx = ps.reduce((s, p) => s + p.x, 0) / ps.length;
  let cov = 0;
  let variance = 0;
  for (const p of ps) { cov += (p.t - mt) * (p.x - mx); variance += (p.t - mt) ** 2; }
  const slope = variance ? cov / variance * 3600000 : 0;
  const diffs = points.slice(1).map((p, i) => p.x - points[i].x);
  const dm = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : 0;
  const vol = diffs.length ? Math.sqrt(diffs.reduce((s, d) => s + (d - dm) ** 2, 0) / diffs.length) : 0.1;
  return { slope, vol, peak: Math.max(...points.map((p) => p.x)) };
}
function deathRate(c: { age: number; venue: string; drift: number; x: number }): number {
  if (c.venue === 'pool' || c.venue === 'migrating') {
    if (c.age >= 30 && c.drift >= 0) return 0.01;
    if (c.age >= 30) return 0.02;
    return 0.12;
  }
  if (c.x >= 2 && c.drift > 0.1) return 0.1;
  return 0.2;
}
const all: Coin[] = last.map((r) => {
  const pts = path.get(r.mint) ?? [{ t: cut, x: Math.log10(Number(BigInt(r.market_cap_lamports))) - 9 }];
  const f = features(pts);
  const age = Math.max(0, (cut - Date.parse(r.created_at)) / 60000);
  const x = pts[pts.length - 1].x;
  const drift = Math.max(-0.3, Math.min(0.45, f.slope * 0.5));
  const c = { mint: r.mint, rank: r.rank, x, age, venue: r.venue, mayhem: r.mayhem, drift, vol: Math.max(0.12, Math.min(0.6, f.vol * Math.sqrt(6) * 0.5 + 0.12)), death: 0 };
  c.death = deathRate(c);
  return c;
});
function potential(c: Coin): number {
  return c.x + (c.age < 30 && c.drift > 0.1 ? 0.3 : 0) + (c.venue === 'pool' && c.age >= 20 && c.drift > 0 ? 0.15 : 0);
}
const candidates = [...all].sort((a, b) => potential(b) - potential(a)).slice(0, 120);
const selected = new Set(candidates.map((c) => c.mint));
const zeros = all.filter((c) => !selected.has(c.mint) && c.rank >= 12 && ((c.age >= 30 || c.x < 1.65) && c.drift <= 0.05)).map((c) => c.mint);
let seed = 0x9e3779b9;
function rand(): number {
  seed = (seed + 0x6d2b79f5) | 0;
  let v = seed;
  v = Math.imul(v ^ (v >>> 15), v | 1);
  v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
  return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
}
let spare: number | undefined;
function normal(): number {
  if (spare !== undefined) { const x = spare; spare = undefined; return x; }
  let a = 0;
  while (a === 0) a = rand();
  const b = rand();
  const r = Math.sqrt(-2 * Math.log(a));
  spare = r * Math.sin(2 * Math.PI * b);
  return r * Math.cos(2 * Math.PI * b);
}
const sims = 5000;
const newcomers = 24;
const coinCounts = Array.from({ length: places }, () => new Map<string, number>());
const binCounts = Array.from({ length: places }, () => new Map<number, number>());
function binOf(x: number): number {
  if (x < edges[0]) return 0;
  if (x >= edges[edges.length - 1]) return edges.length;
  let lo = 0; let hi = edges.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (edges[mid] <= x) lo = mid; else hi = mid - 1; }
  return lo + 1;
}
for (const h of horizons) {
  const saturation = 1 - Math.exp(-h);
  for (let sim = 0; sim < sims; sim++) {
    const draw: Array<{ mint: string; x: number }> = [];
    for (const c of candidates) {
      const dead = Math.min(0.9, 1 - Math.pow(1 - c.death, 6 * h));
      let x: number;
      if (rand() < dead) x = floor + normal() * 0.1;
      else {
        const agePrior = c.age < 30 && c.venue === 'pool' ? 0.04 : c.age >= 30 ? -0.035 : -0.015;
        const drift = 0.5 * c.drift + 0.5 * agePrior;
        const sigma = Math.min(0.85, Math.sqrt(0.12 ** 2 + c.vol ** 2 * h));
        x = c.x + drift * saturation + normal() * sigma;
        if (c.venue === 'curve' && x > 2.5) {
          const graduate = c.drift > 0.1 && c.age < 45 ? 0.5 : 0.2;
          if (rand() >= graduate) x = 2.5 + Math.abs(normal()) * 0.02;
        }
      }
      if (c.mayhem && c.venue !== 'pool') x = -1;
      draw.push({ mint: c.mint, x });
    }
    const mu = 2.7 + 0.015 * h;
    for (let i = 0; i < newcomers; i++) draw.push({ mint: '', x: mu + normal() * 0.42 });
    draw.sort((a, b) => b.x - a.x || a.mint.localeCompare(b.mint));
    for (let k = 0; k < places; k++) {
      const d = draw[k];
      const bc = binCounts[k];
      const bn = binOf(d.x);
      bc.set(bn, (bc.get(bn) ?? 0) + 1);
      if (d.mint) { const cc = coinCounts[k]; cc.set(d.mint, (cc.get(d.mint) ?? 0) + 1); }
    }
  }
}
const denominator = sims * horizons.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
const sizes: Array<{ place: number; bin: number; p: number }> = [];
const named = Array.from({ length: places }, () => new Set<string>());
for (let k = 0; k < places; k++) {
  const ranked = [...coinCounts[k].entries()].map(([mint, n]) => ({ mint, p: n / denominator })).filter((v) => v.p >= 0.002).sort((a, b) => b.p - a.p || a.mint.localeCompare(b.mint));
  for (const v of ranked.slice(0, 100)) { const p = Math.floor(v.p * 1000000) / 1000000; if (p > 0) { probabilities.push({ mint: v.mint, place: k + 1, p }); named[k].add(v.mint); } }
  for (const mint of zeros) { if (probabilities.filter((r) => r.place === k + 1).length >= 100) break; if (!named[k].has(mint)) { probabilities.push({ mint, place: k + 1, p: 0 }); named[k].add(mint); } }
  for (const [bin, n] of binCounts[k]) { const p = Math.floor(n / denominator * 1000000) / 1000000; if (p >= 0.002) sizes.push({ place: k + 1, bin, p }); }
}
const coinPlace = new Map<string, number[]>();
for (let k = 0; k < places; k++) for (const [mint, n] of coinCounts[k]) { const ps = coinPlace.get(mint) ?? [0, 0, 0, 0, 0]; ps[k] = n / denominator; coinPlace.set(mint, ps); }
const expected = [...coinPlace].map(([mint, ps]) => ({ mint, score: ps.reduce((s, p, i) => s + p * (i + 1), 0) + (1 - ps.reduce((s, p) => s + p, 0)) * 6, mass: ps.reduce((s, p) => s + p, 0) })).sort((a, b) => a.score - b.score || b.mass - a.mass || a.mint.localeCompare(b.mint));
const top5 = expected.slice(0, 5).map((x) => x.mint);
for (const r of last) if (top5.length < 5 && !top5.includes(r.mint)) top5.push(r.mint);
mkdirSync(output, { recursive: true });
writeFileSync(join(output, task.output), JSON.stringify({ probabilities, top5, sizes }) + '\n');
