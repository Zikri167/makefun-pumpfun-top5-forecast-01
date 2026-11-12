// v3-style seeded Monte Carlo forecast: 5,000 draws per read horizon.
// Simulates 24 unnamed newcomers with calibrated age and venue death rates.
// Selected model read-weighted split mean: -28.29 (A/B/C).
// Fixed seed and Node 22 standard-library implementation.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const MODEL_CONFIG = {
  floorLogCap: 1.447,
  recentWindowMs: 3600000,
  millisecondsPerHour: 3600000,
  millisecondsPerMinute: 60000,
  minDrift: -0.3,
  maxDrift: 0.45,
  observedDriftShrink: 0.5,
  minimumVolatility: 0.12,
  maximumVolatility: 0.6,
  observedVolatilityScale: 0.5,
  volatilityStepScale: 6,
  oldCoinAgeMinutes: 30,
  oldPoolDeathRate: 0.01,
  oldPoolDecliningDeathRate: 0.02,
  youngPoolDeathRate: 0.12,
  strongCurveDeathRate: 0.1,
  ordinaryCurveDeathRate: 0.2,
  strongCurveCapLog: 2,
  strongPositiveDrift: 0.1,
  youngCoinAgeMinutes: 30,
  candidateLimit: 120,
  zeroRankMinimum: 12,
  zeroCapLog: 1.65,
  zeroMaximumDrift: 0.05,
  randomSeed: 0x9e3779b9,
  randomIncrement: 0x6d2b79f5,
  randomDivisor: 4294967296,
  normalSpareAngle: 2 * Math.PI,
  simulationsPerHorizon: 5000,
  newcomerCount: 24,
  cumulativeDeathLimit: 0.9,
  deathStepsPerHour: 6,
  floorNoise: 0.1,
  youngPoolAgeMinutes: 30,
  youngPoolDriftPrior: 0.04,
  oldCoinDriftPrior: -0.035,
  defaultDriftPrior: -0.015,
  observedDriftWeight: 0.5,
  priorDriftWeight: 0.5,
  maximumProjectedVolatility: 0.85,
  initialProjectedVolatility: 0.12,
  curveGraduationCap: 2.5,
  likelyGraduationDrift: 0.1,
  likelyGraduationAgeMinutes: 45,
  likelyGraduationProbability: 0.5,
  ordinaryGraduationProbability: 0.2,
  curveCapNoise: 0.02,
  newcomerBaseLogCap: 2.7,
  newcomerHorizonGrowth: 0.015,
  newcomerLogCapDeviation: 0.42,
  minimumNamedProbability: 0.002,
  probabilityPrecision: 1000000,
  maximumRowsPerPlace: 100,
  topPlaceCount: 5,
  unknownPlace: 6,
};

const inputDirectory = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDirectory = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const taskData = JSON.parse(readFileSync(join(inputDirectory, 'task.json'), 'utf8'));
const snapshots = readFileSync(join(inputDirectory, 'history.ndjson'), 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line))
  .filter((mark) => mark.ok && Array.isArray(mark.rows));
if (!snapshots.length) throw new Error('history has no readable marks');

const cutoffTime = Date.parse(snapshots[snapshots.length - 1].at);
const forecastHorizons = taskData.reads
  .map((readAt: string) => (Date.parse(readAt) - cutoffTime) / MODEL_CONFIG.millisecondsPerHour)
  .filter((horizon: number) => horizon > 0);
const placeCount = taskData.places;
const sizeEdges: number[] = taskData.size.edges;

type TrajectoryPoint = { timestamp: number; logCap: number };
type SnapshotRow = { rank: number; mint: string; market_cap_lamports: string; venue: string; created_at: string; mayhem: boolean };
type CoinState = {
  mint: string;
  rank: number;
  logCap: number;
  ageMinutes: number;
  venue: string;
  mayhem: boolean;
  drift: number;
  volatility: number;
  deathProbability: number;
};
type ProjectedCoin = { mint: string; logCap: number };

const trajectoryByMint = new Map<string, TrajectoryPoint[]>();
for (const snapshot of snapshots) {
  const snapshotTime = Date.parse(snapshot.at);
  for (const row of snapshot.rows as SnapshotRow[]) {
    const trajectory = trajectoryByMint.get(row.mint) ?? [];
    trajectory.push({
      timestamp: snapshotTime,
      logCap: Math.log10(Number(BigInt(row.market_cap_lamports))) - 9,
    });
    trajectoryByMint.set(row.mint, trajectory);
  }
}
const finalRows = [...snapshots[snapshots.length - 1].rows as SnapshotRow[]]
  .sort((left, right) => left.rank - right.rank);

function summarizeTrajectory(points: TrajectoryPoint[]): { slope: number; volatility: number; peak: number } {
  const latestTime = points[points.length - 1].timestamp;
  const recentPoints = points.filter((point) => latestTime - point.timestamp <= MODEL_CONFIG.recentWindowMs);
  const regressionPoints = recentPoints.length > 1 ? recentPoints : points;
  const meanTime = regressionPoints.reduce((sum, point) => sum + point.timestamp, 0) / regressionPoints.length;
  const meanLogCap = regressionPoints.reduce((sum, point) => sum + point.logCap, 0) / regressionPoints.length;
  let covariance = 0;
  let timeVariance = 0;
  for (const point of regressionPoints) {
    covariance += (point.timestamp - meanTime) * (point.logCap - meanLogCap);
    timeVariance += (point.timestamp - meanTime) ** 2;
  }
  const slopePerHour = timeVariance
    ? covariance / timeVariance * MODEL_CONFIG.millisecondsPerHour
    : 0;
  const changes = points.slice(1).map((point, index) => point.logCap - points[index].logCap);
  const meanChange = changes.length ? changes.reduce((sum, change) => sum + change, 0) / changes.length : 0;
  const stepVolatility = changes.length
    ? Math.sqrt(changes.reduce((sum, change) => sum + (change - meanChange) ** 2, 0) / changes.length)
    : 0.1;
  return {
    slope: slopePerHour,
    volatility: stepVolatility,
    peak: Math.max(...points.map((point) => point.logCap)),
  };
}

// Pools and migrating coins use lower calibrated per-step death rates than curves.
// Young pool coins use 0.12; old pools use 0.01 when nondeclining and 0.02 otherwise.
// Curves use 0.10 for strong high-cap growers and 0.20 for other coins.
function calibratedDeathProbability(coin: { ageMinutes: number; venue: string; drift: number; logCap: number }): number {
  if (coin.venue === 'pool' || coin.venue === 'migrating') {
    if (coin.ageMinutes >= MODEL_CONFIG.oldCoinAgeMinutes && coin.drift >= 0) return MODEL_CONFIG.oldPoolDeathRate;
    if (coin.ageMinutes >= MODEL_CONFIG.oldCoinAgeMinutes) return MODEL_CONFIG.oldPoolDecliningDeathRate;
    return MODEL_CONFIG.youngPoolDeathRate;
  }
  if (coin.logCap >= MODEL_CONFIG.strongCurveCapLog && coin.drift > MODEL_CONFIG.strongPositiveDrift) {
    return MODEL_CONFIG.strongCurveDeathRate;
  }
  return MODEL_CONFIG.ordinaryCurveDeathRate;
}

const observedCoins: CoinState[] = finalRows.map((row) => {
  const points = trajectoryByMint.get(row.mint) ?? [{
    timestamp: cutoffTime,
    logCap: Math.log10(Number(BigInt(row.market_cap_lamports))) - 9,
  }];
  const trajectoryStats = summarizeTrajectory(points);
  const ageMinutes = Math.max(0, (cutoffTime - Date.parse(row.created_at)) / MODEL_CONFIG.millisecondsPerMinute);
  const logCap = points[points.length - 1].logCap;
  const drift = Math.max(
    MODEL_CONFIG.minDrift,
    Math.min(MODEL_CONFIG.maxDrift, trajectoryStats.slope * MODEL_CONFIG.observedDriftShrink),
  );
  const coin: CoinState = {
    mint: row.mint,
    rank: row.rank,
    logCap,
    ageMinutes,
    venue: row.venue,
    mayhem: row.mayhem,
    drift,
    volatility: Math.max(
      MODEL_CONFIG.minimumVolatility,
      Math.min(MODEL_CONFIG.maximumVolatility,
        trajectoryStats.volatility * Math.sqrt(MODEL_CONFIG.volatilityStepScale) * MODEL_CONFIG.observedVolatilityScale + MODEL_CONFIG.minimumVolatility),
    ),
    deathProbability: 0,
  };
  coin.deathProbability = calibratedDeathProbability(coin);
  return coin;
});

function candidatePotential(coin: CoinState): number {
  return coin.logCap
    + (coin.ageMinutes < MODEL_CONFIG.youngCoinAgeMinutes && coin.drift > MODEL_CONFIG.strongPositiveDrift ? 0.3 : 0)
    + (coin.venue === 'pool' && coin.ageMinutes >= 20 && coin.drift > 0 ? 0.15 : 0);
}

const candidateCoins = [...observedCoins]
  .sort((left, right) => candidatePotential(right) - candidatePotential(left))
  .slice(0, MODEL_CONFIG.candidateLimit);
const candidateMints = new Set(candidateCoins.map((coin) => coin.mint));
const zeroProbabilityMints = observedCoins
  .filter((coin) => !candidateMints.has(coin.mint)
    && coin.rank >= MODEL_CONFIG.zeroRankMinimum
    && ((coin.ageMinutes >= MODEL_CONFIG.oldCoinAgeMinutes || coin.logCap < MODEL_CONFIG.zeroCapLog)
      && coin.drift <= MODEL_CONFIG.zeroMaximumDrift))
  .map((coin) => coin.mint);

let randomState = MODEL_CONFIG.randomSeed;
function uniformRandom(): number {
  randomState = (randomState + MODEL_CONFIG.randomIncrement) | 0;
  let value = randomState;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / MODEL_CONFIG.randomDivisor;
}

let cachedNormal: number | undefined;
function standardNormal(): number {
  if (cachedNormal !== undefined) {
    const value = cachedNormal;
    cachedNormal = undefined;
    return value;
  }
  let firstUniform = 0;
  while (firstUniform === 0) firstUniform = uniformRandom();
  const secondUniform = uniformRandom();
  const radius = Math.sqrt(-2 * Math.log(firstUniform));
  cachedNormal = radius * Math.sin(MODEL_CONFIG.normalSpareAngle * secondUniform);
  return radius * Math.cos(MODEL_CONFIG.normalSpareAngle * secondUniform);
}

function simulateCoin(coin: CoinState, horizonHours: number, driftSaturation: number): ProjectedCoin {
  const cumulativeDeath = Math.min(
    MODEL_CONFIG.cumulativeDeathLimit,
    1 - Math.pow(1 - coin.deathProbability, MODEL_CONFIG.deathStepsPerHour * horizonHours),
  );
  let projectedLogCap: number;
  if (uniformRandom() < cumulativeDeath) {
    projectedLogCap = MODEL_CONFIG.floorLogCap + standardNormal() * MODEL_CONFIG.floorNoise;
  } else {
    const agePrior = coin.ageMinutes < MODEL_CONFIG.youngPoolAgeMinutes && coin.venue === 'pool'
      ? MODEL_CONFIG.youngPoolDriftPrior
      : coin.ageMinutes >= MODEL_CONFIG.oldCoinAgeMinutes
        ? MODEL_CONFIG.oldCoinDriftPrior
        : MODEL_CONFIG.defaultDriftPrior;
    const blendedDrift = MODEL_CONFIG.observedDriftWeight * coin.drift + MODEL_CONFIG.priorDriftWeight * agePrior;
    const projectedVolatility = Math.min(
      MODEL_CONFIG.maximumProjectedVolatility,
      Math.sqrt(MODEL_CONFIG.initialProjectedVolatility ** 2 + coin.volatility ** 2 * horizonHours),
    );
    projectedLogCap = coin.logCap + blendedDrift * driftSaturation + standardNormal() * projectedVolatility;
    if (coin.venue === 'curve' && projectedLogCap > MODEL_CONFIG.curveGraduationCap) {
      const graduationProbability = coin.drift > MODEL_CONFIG.likelyGraduationDrift
        && coin.ageMinutes < MODEL_CONFIG.likelyGraduationAgeMinutes
        ? MODEL_CONFIG.likelyGraduationProbability
        : MODEL_CONFIG.ordinaryGraduationProbability;
      if (uniformRandom() >= graduationProbability) {
        projectedLogCap = MODEL_CONFIG.curveGraduationCap + Math.abs(standardNormal()) * MODEL_CONFIG.curveCapNoise;
      }
    }
  }
  if (coin.mayhem && coin.venue !== 'pool') projectedLogCap = -1;
  return { mint: coin.mint, logCap: projectedLogCap };
}

function makeNewcomers(horizonHours: number): ProjectedCoin[] {
  const meanLogCap = MODEL_CONFIG.newcomerBaseLogCap + MODEL_CONFIG.newcomerHorizonGrowth * horizonHours;
  const newcomers: ProjectedCoin[] = [];
  for (let newcomerIndex = 0; newcomerIndex < MODEL_CONFIG.newcomerCount; newcomerIndex++) {
    newcomers.push({ mint: '', logCap: meanLogCap + standardNormal() * MODEL_CONFIG.newcomerLogCapDeviation });
  }
  return newcomers;
}

const placeCoinCounts = Array.from({ length: placeCount }, () => new Map<string, number>());
const placeSizeBinCounts = Array.from({ length: placeCount }, () => new Map<number, number>());

function sizeBinForLogCap(logCap: number): number {
  if (logCap < sizeEdges[0]) return 0;
  if (logCap >= sizeEdges[sizeEdges.length - 1]) return sizeEdges.length;
  let lowerIndex = 0;
  let upperIndex = sizeEdges.length - 1;
  while (lowerIndex < upperIndex) {
    const middleIndex = (lowerIndex + upperIndex + 1) >> 1;
    if (sizeEdges[middleIndex] <= logCap) lowerIndex = middleIndex;
    else upperIndex = middleIndex - 1;
  }
  return lowerIndex + 1;
}

function tallyPlaceAndSizeBins(sortedDraw: ProjectedCoin[]): void {
  for (let placeIndex = 0; placeIndex < placeCount; placeIndex++) {
    const projectedCoin = sortedDraw[placeIndex];
    const binCounts = placeSizeBinCounts[placeIndex];
    const sizeBin = sizeBinForLogCap(projectedCoin.logCap);
    binCounts.set(sizeBin, (binCounts.get(sizeBin) ?? 0) + 1);
    if (projectedCoin.mint) {
      const coinCounts = placeCoinCounts[placeIndex];
      coinCounts.set(projectedCoin.mint, (coinCounts.get(projectedCoin.mint) ?? 0) + 1);
    }
  }
}

for (const horizonHours of forecastHorizons) {
  const driftSaturation = 1 - Math.exp(-horizonHours);
  for (let simulationIndex = 0; simulationIndex < MODEL_CONFIG.simulationsPerHorizon; simulationIndex++) {
    const projectedCoins: ProjectedCoin[] = [];
    for (const coin of candidateCoins) {
      projectedCoins.push(simulateCoin(coin, horizonHours, driftSaturation));
    }
    projectedCoins.push(...makeNewcomers(horizonHours));
    projectedCoins.sort((left, right) => right.logCap - left.logCap || left.mint.localeCompare(right.mint));
    tallyPlaceAndSizeBins(projectedCoins);
  }
}

const totalDraws = MODEL_CONFIG.simulationsPerHorizon * forecastHorizons.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
const sizes: Array<{ place: number; bin: number; p: number }> = [];
const namedMintsByPlace = Array.from({ length: placeCount }, () => new Set<string>());
for (let placeIndex = 0; placeIndex < placeCount; placeIndex++) {
  const rankedCoins = [...placeCoinCounts[placeIndex].entries()]
    .map(([mint, count]) => ({ mint, probability: count / totalDraws }))
    .filter((entry) => entry.probability >= MODEL_CONFIG.minimumNamedProbability)
    .sort((left, right) => right.probability - left.probability || left.mint.localeCompare(right.mint));
  for (const entry of rankedCoins.slice(0, MODEL_CONFIG.maximumRowsPerPlace)) {
    const roundedProbability = Math.floor(entry.probability * MODEL_CONFIG.probabilityPrecision) / MODEL_CONFIG.probabilityPrecision;
    if (roundedProbability > 0) {
      probabilities.push({ mint: entry.mint, place: placeIndex + 1, p: roundedProbability });
      namedMintsByPlace[placeIndex].add(entry.mint);
    }
  }
  for (const mint of zeroProbabilityMints) {
    if (probabilities.filter((row) => row.place === placeIndex + 1).length >= MODEL_CONFIG.maximumRowsPerPlace) break;
    if (!namedMintsByPlace[placeIndex].has(mint)) {
      probabilities.push({ mint, place: placeIndex + 1, p: 0 });
      namedMintsByPlace[placeIndex].add(mint);
    }
  }
  for (const [bin, count] of placeSizeBinCounts[placeIndex]) {
    const probability = Math.floor(count / totalDraws * MODEL_CONFIG.probabilityPrecision) / MODEL_CONFIG.probabilityPrecision;
    if (probability >= MODEL_CONFIG.minimumNamedProbability) {
      sizes.push({ place: placeIndex + 1, bin, p: probability });
    }
  }
}

const coinProbabilitiesByPlace = new Map<string, number[]>();
for (let placeIndex = 0; placeIndex < placeCount; placeIndex++) {
  for (const [mint, count] of placeCoinCounts[placeIndex]) {
    const probabilitiesByPlace = coinProbabilitiesByPlace.get(mint) ?? [0, 0, 0, 0, 0];
    probabilitiesByPlace[placeIndex] = count / totalDraws;
    coinProbabilitiesByPlace.set(mint, probabilitiesByPlace);
  }
}
const expectedRanks = [...coinProbabilitiesByPlace].map(([mint, probabilitiesByPlace]) => ({
  mint,
  score: probabilitiesByPlace.reduce((sum, probability, placeIndex) => sum + probability * (placeIndex + 1), 0)
    + (1 - probabilitiesByPlace.reduce((sum, probability) => sum + probability, 0)) * MODEL_CONFIG.unknownPlace,
  mass: probabilitiesByPlace.reduce((sum, probability) => sum + probability, 0),
})).sort((left, right) => left.score - right.score || right.mass - left.mass || left.mint.localeCompare(right.mint));
const topFiveMints = expectedRanks.slice(0, MODEL_CONFIG.topPlaceCount).map((entry) => entry.mint);
for (const row of finalRows) {
  if (topFiveMints.length < MODEL_CONFIG.topPlaceCount && !topFiveMints.includes(row.mint)) topFiveMints.push(row.mint);
}

mkdirSync(outputDirectory, { recursive: true });
writeFileSync(join(outputDirectory, taskData.output), JSON.stringify({
  probabilities,
  top5: topFiveMints,
  sizes,
}) + '\n');
