import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

type SnapshotRow = {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue?: string;
  created_at?: string;
  mayhem?: boolean;
};
type Mark = { at: string; ok: boolean; rows: SnapshotRow[] | null };
type Seen = { time: number; logCap: number; row: SnapshotRow };
type Candidate = {
  mint: string;
  logCap: number;
  rank: number;
  ageMinutes: number;
  venue: string;
  mayhem: boolean;
  slope: number;
  sightings: number;
  bestRank: number;
};
type Task = { reads: string[]; places: number; size: { edges: number[] }; output: string };

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as Task;
const marks = readFileSync(join(inputDir, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((line) => line.length > 0)
  .map((line) => JSON.parse(line) as Mark)
  .filter((mark) => mark.ok && Array.isArray(mark.rows));
if (marks.length === 0) throw new Error('no readable snapshots');

const finalMark = marks[marks.length - 1]!;
const cutTime = Date.parse(finalMark.at);
const sightingsByMint = new Map<string, Seen[]>();
for (const mark of marks) {
  const stamp = Date.parse(mark.at);
  for (const row of mark.rows ?? []) {
    const seen = sightingsByMint.get(row.mint) ?? [];
    seen.push({
      time: stamp,
      logCap: Math.log10(Number(BigInt(row.market_cap_lamports))) - 9,
      row,
    });
    sightingsByMint.set(row.mint, seen);
  }
}

const candidates: Candidate[] = [];
for (const [mint, sightings] of sightingsByMint) {
  sightings.sort((a, b) => a.time - b.time);
  const newest = sightings[sightings.length - 1]!;
  const earlier = sightings[Math.max(0, sightings.length - 3)]!;
  const intervals = Math.max(1, (newest.time - earlier.time) / 600000);
  const slope = Math.max(-0.8, Math.min(0.8, (newest.logCap - earlier.logCap) / intervals));
  const created = newest.row.created_at ? Date.parse(newest.row.created_at) : newest.time;
  candidates.push({
    mint,
    logCap: newest.logCap,
    rank: newest.row.rank,
    ageMinutes: Math.max(0, (cutTime - created) / 60000),
    venue: newest.row.venue ?? 'curve',
    mayhem: newest.row.mayhem === true,
    slope,
    sightings: sightings.length,
    bestRank: Math.min(...sightings.map((seen) => seen.row.rank)),
  });
}
const forecastPool = candidates.filter((coin) => coin.rank <= 200 || coin.bestRank <= 10);

let state = 0x51f15e;
function uniform(): number {
  state = (state + 0x6d2b79f5) | 0;
  let value = state;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}
function gaussian(): number {
  const u = Math.max(1e-12, uniform());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
}
function findSizeBin(logCap: number): number {
  const edges = task.size.edges;
  if (logCap < edges[0]!) return 0;
  if (logCap >= edges[edges.length - 1]!) return 81;
  let left = 0;
  let right = edges.length - 1;
  while (left < right) {
    const middle = (left + right) >>> 1;
    if (logCap < edges[middle + 1]!) right = middle;
    else left = middle + 1;
  }
  return left + 1;
}

const trials = 7000;
const placeTally = new Map<string, number>();
const binTally = Array.from({ length: task.places }, () => new Float64Array(82));
for (const readTime of task.reads) {
  const horizonHours = Math.max(0, (Date.parse(readTime) - cutTime) / 3600000);
  for (let trial = 0; trial < trials; trial++) {
    const ranking: Array<{ mint: string | null; logCap: number }> = [];
    for (const coin of forecastPool) {
      const ageHours = coin.ageMinutes / 60;
      const expectedChange = coin.venue === 'pool'
        ? (coin.rank <= 15 && ageHours < 1 ? 0.10 : ageHours < 2 ? 0.025 : -0.035)
        : -0.035;
      const trend = (expectedChange + 0.5 * (coin.slope - expectedChange))
        * (1 - Math.exp(-horizonHours / 1.5));
      const volatility = Math.min(0.9, Math.sqrt(0.01 + 0.14 * horizonHours));
      let futureCap = coin.logCap + trend + volatility * gaussian();
      const baseCrashRisk = coin.rank <= 5
        ? 0.19
        : coin.rank <= 15
          ? 0.14
          : coin.venue === 'pool' ? 0.12 : 0.06;
      const crashRisk = 1 - Math.pow(1 - baseCrashRisk, Math.max(0.1, horizonHours * 6));
      if (uniform() < crashRisk) futureCap -= 0.8 + uniform() * 1.3;
      if (coin.venue === 'curve' && !coin.mayhem) futureCap = Math.min(futureCap, 2.48);
      if (coin.mayhem && coin.venue !== 'pool') futureCap = -1;
      ranking.push({ mint: coin.mint, logCap: futureCap });
    }
    const unseenCount = Math.max(2, Math.round(7 + 3 * horizonHours));
    for (let newcomer = 0; newcomer < unseenCount; newcomer++) {
      ranking.push({
        mint: null,
        logCap: 2.70 + 0.42 * gaussian() + Math.min(0.55, horizonHours * 0.05),
      });
    }
    ranking.sort((a, b) => b.logCap - a.logCap);
    for (let placeIndex = 0; placeIndex < task.places; placeIndex++) {
      const result = ranking[placeIndex]!;
      if (result.mint !== null) {
        const tallyKey = String(placeIndex + 1) + '|' + result.mint;
        placeTally.set(tallyKey, (placeTally.get(tallyKey) ?? 0) + 1);
      }
      binTally[placeIndex]![findSizeBin(result.logCap)]!++;
    }
  }
}

const denominator = trials * task.reads.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= task.places; place++) {
  const ordered = forecastPool
    .map((coin) => ({
      coin,
      probability: (placeTally.get(String(place) + '|' + coin.mint) ?? 0) / denominator,
    }))
    .sort((a, b) => b.probability - a.probability);
  for (const entry of ordered.slice(0, 80)) {
    if (entry.probability >= 0.0001) {
      probabilities.push({ mint: entry.coin.mint, place, p: entry.probability });
    }
  }
}
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let place = 1; place <= task.places; place++) {
  for (let bin = 0; bin < 82; bin++) {
    const probability = binTally[place - 1]![bin]! / denominator;
    if (probability >= 0.0001) sizes.push({ place, bin, p: probability });
  }
}
const combinedChance = new Map<string, number>();
for (const row of probabilities) {
  combinedChance.set(row.mint, (combinedChance.get(row.mint) ?? 0) + row.p);
}
const top5 = [...forecastPool]
  .sort((a, b) => (combinedChance.get(b.mint) ?? 0) - (combinedChance.get(a.mint) ?? 0))
  .slice(0, 5)
  .map((coin) => coin.mint);

mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, task.output), JSON.stringify({ probabilities, top5, sizes }) + '\n');
