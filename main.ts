// GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01
//
// Model: dilution + progression.  A trophy's earn rate is earners / players
// tracked.  Both counts grow after launch, but on different clocks: the
// tracked-player base grows as buyers link their accounts, while earners
// grow only as players actually reach that trophy.  So we model the two
// counts separately and divide.
//
//   players P(t): the true tracked-player curve, linearly interpolated
//     between the reads where the published count moved (a jump is a
//     stale-page catch-up: the players it newly counts started during the
//     window since the previous refresh), extrapolated past the last read
//     with a power-law fit on those change points.  When the history is
//     perfectly flat near release the page is stale and a refresh is due:
//     the rules state the first refresh took the count 638 -> 3,868, so
//     a ~6x jump is the prior for that first refresh.
//   earners E_i(t): players mature into a trophy with an exponential delay
//     of mean D_i hours after being tracked:
//       E_i(t) = R_i * (P(t) - A(t)),  A(t) = sum of inflows decayed by
//       exp(-age / D_i).  A is updated recursively on a 10-minute grid, so
//       the whole fit is O(grid * #delays) and shared across trophies.
//     R_i (the eventual share) and D_i are fit per trophy by least squares
//     on the observed rates, with a description/tier prior on D_i that
//     dominates when the history is flat and yields to data when it moves.
//   forecast: rate_i(target) = R_i * C_i(target), C_i = (P - A) / P.
//   spread: logit sd grows with horizon (random-walk-like) plus the
//     per-trophy fit residual and a stale-page term when the history is
//     flat, so the 90% band is honest about refresh shocks.
//
// No network, no randomness, deterministic: the same case always yields
// the same bytes.  Node 22 standard library only.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const D_GRID = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450, 675, 1000, 1500];
const STEP_MIN = 10; // time-grid step, minutes
const RELEASE = Date.parse("2026-11-19T00:00:00Z");

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function logit(pct: number): number {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}
function logistic(l: number): number {
  return 100 / (1 + Math.exp(-l));
}
// Inverse standard normal CDF (Acklam's approximation, ~1e-9 accuracy).
function normInv(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q: number, r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// Characteristic delay (hours) for a player to earn a trophy after being
// tracked, from its description and tier.  Early story beats are fast,
// grinds and completionist trophies are slow.
function priorDelay(t: any): number {
  const d = (t.description || "").toLowerCase();
  const name = (t.name || "").toLowerCase();
  if (t.id === 2 || name.includes("gone fishing")) return 0.5;
  if (d.includes("chapter 1")) return 4;
  if (d.includes("chapter 2")) return 10;
  if (d.includes("chapter 3")) return 20;
  if (d.includes("chapter 4")) return 36;
  if (d.includes("final mission") && t.tier === "gold") return 60;
  if (d.includes("100% completion")) return 800;
  if (t.tier === "platinum") return 700;
  if (d.includes("250 stunts")) return 300;
  if (d.includes("collectible card")) return 250;
  if (d.includes("waink")) return 250;
  if (d.includes("50 challenges")) return 200;
  if (d.includes("strangers")) return 200;
  if (d.includes("gold medal")) return 150;
  if (d.includes("laundering")) return 150;
  if (d.includes("garage properties")) return 150;
  if (d.includes("meditate")) return 120;
  if (d.includes("drone race")) return 120;
  if (d.includes("bank vaults")) return 120;
  if (d.includes("spend $100,000")) return 120;
  if (d.includes("20 different races")) return 100;
  if (d.includes("explore all of leonida")) return 100;
  if (d.includes("discover 100 vehicles")) return 80;
  if (d.includes("register 10 vehicles")) return 60;
  if (d.includes("arcade game")) return 24;
  if (d.includes("fishing challenge")) return 36;
  if (d.startsWith("complete ")) return 48; // named story mission
  if (d.startsWith("complete the final mission for")) return 60;
  if (t.tier === "silver") return 150;
  if (t.tier === "gold") return 200;
  return 72;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: any[];
  history: any[];
}

function forecastCase(c: CaseFile): any {
  const tAt = Date.parse(c.at);
  const tTarget = Date.parse(c.target);
  const horizonDays = Math.max(0, (tTarget - tAt) / 86400000);
  const trophies = c.trophies;
  const n = trophies.length;
  const hist = (c.history || []).slice().sort((a, b) => Date.parse(a.at) - Date.parse(b.at));

  // --- players tracked P(t) -------------------------------------------------
  // A read shows the page as of Exophase's last refresh before it, so a
  // read's rates belong to the last refresh point at or before it -- the
  // last read where the page's content (count or any rate) changed -- not
  // to the read's own clock time.  In a stale stretch every read evaluates
  // at the same refresh point, which is exactly why flat history cannot
  // identify a trophy's delay and the description prior must carry it.
  const refreshPts: { t: number; p: number }[] = [];
  let prevP = -1, prevRates = "";
  for (const h of hist) {
    const p = Number(h.players_tracked);
    const ratesKey = JSON.stringify(h.rates || []);
    if (refreshPts.length === 0 || p !== prevP || ratesKey !== prevRates) {
      refreshPts.push({ t: Date.parse(h.at), p: Number.isFinite(p) && p > 0 ? p : (refreshPts.length ? refreshPts[refreshPts.length - 1].p : 0) });
    }
    prevP = p; prevRates = ratesKey;
  }
  function evalTime(tMs: number): number {
    let t = refreshPts.length ? refreshPts[0].t : tMs;
    for (const rp of refreshPts) { if (rp.t <= tMs) t = rp.t; else break; }
    return t;
  }
  // Change points (refresh points where the published count moved) drive
  // the growth fit; flat stretches are stale publishing, not a flat base.
  const changePts: { tau: number; p: number }[] = [];
  let lastCp = -1;
  for (const rp of refreshPts) {
    if (rp.p !== lastCp) {
      changePts.push({ tau: (rp.t - RELEASE) / 3600000, p: rp.p });
      lastCp = rp.p;
    }
  }
  const pLast = changePts.length ? changePts[changePts.length - 1].p : 0;
  const pFirst = changePts.length ? changePts[0].p : 0;
  const tauLast = changePts.length ? changePts[changePts.length - 1].tau : 0;
  const tauTarget = (tTarget - RELEASE) / 3600000;

  let pTarget: number;
  if (changePts.length >= 2) {
    // Power law in (hours since release + 6): ln P = a + b ln(tau + 6).
    let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
    for (const cp of changePts) {
      const x = Math.log(Math.max(0.5, cp.tau + 6));
      const y = Math.log(cp.p);
      sx += x; sy += y; sxx += x * x; sxy += x * y; m++;
    }
    const den = m * sxx - sx * sx;
    let b = den !== 0 ? (m * sxy - sx * sy) / den : 0;
    const a = (sy - b * sx) / m;
    if (b < 0.03) b = 0; // no measurable growth -> hold the count
    pTarget = b === 0 ? pLast : Math.exp(a + b * Math.log(Math.max(0.5, tauTarget + 6)));
    pTarget = clamp(pTarget, pLast, pLast * 25);
  } else if (changePts.length === 1 && tauLast < 24) {
    // Flat page on launch night: a refresh is due.  The rules state the
    // first refresh took the count 638 -> 3,868 (~6x); use that as the prior.
    pTarget = pFirst * 6.07;
  } else {
    pTarget = pLast * (1 + 0.02 * horizonDays);
  }
  if (!Number.isFinite(pTarget) || pTarget <= 0) pTarget = Math.max(pLast, 1);

  // P on the time grid (10-minute steps).  Q(t) is the *true* tracked-player
  // curve: linear interpolation between the change points, from 0 at the
  // earliest launch -- the game released on rolling midnight, first in New
  // Zealand (UTC+13), 11:00 UTC on 18 Nov, about 13 hours before the first
  // read -- out to the extrapolated target count.  Interpolating, rather
  // than stepping at each read, matters because Exophase publishes with
  // lag: a jump in the published count is a stale-page catch-up, and the
  // players it newly counts started during the window since the previous
  // refresh, not at the refresh instant.  Spreading each jump over that
  // window is what keeps a 10-minute first mission at ~100% while slower
  // trophies dilute.
  const t0 = (hist.length ? Date.parse(hist[0].at) : tAt) - 13 * 3600000;
  const stepMs = STEP_MIN * 60000;
  const nSteps = Math.max(1, Math.ceil((tTarget - t0) / stepMs));
  const qPts: { t: number; p: number }[] = [{ t: t0, p: 0 }];
  for (const cp of changePts) {
    qPts.push({ t: RELEASE + cp.tau * 3600000, p: cp.p });
  }
  qPts.push({ t: tTarget, p: pTarget });
  function qAt(tMs: number): number {
    if (tMs <= qPts[0].t) return qPts[0].p;
    for (let k = 1; k < qPts.length; k++) {
      if (tMs <= qPts[k].t) {
        const a = qPts[k - 1], b = qPts[k];
        const f = (tMs - a.t) / Math.max(1, b.t - a.t);
        return a.p + (b.p - a.p) * clamp(f, 0, 1);
      }
    }
    return qPts[qPts.length - 1].p;
  }
  const pGrid = new Float64Array(nSteps + 1);
  const inflow = new Float64Array(nSteps + 1);
  let prev = 0;
  for (let s = 0; s <= nSteps; s++) {
    const tMs = t0 + s * stepMs;
    const p = qAt(tMs);
    pGrid[s] = p;
    inflow[s] = Math.max(0, p - prev);
    prev = p;
  }

  // --- maturity curve C(t; D) for each delay in the grid --------------------
  // A(t) = unmatured mass, decays exp(-dt/D); C = 1 - A/P.
  const cAtStep: number[][] = D_GRID.map(() => new Float64Array(nSteps + 1) as any);
  for (let di = 0; di < D_GRID.length; di++) {
    const D = D_GRID[di];
    const decay = Math.exp(-(STEP_MIN / 60) / D);
    const mid = Math.exp(-(STEP_MIN / 120) / D);
    let a = 0;
    const arr = cAtStep[di];
    for (let s = 0; s <= nSteps; s++) {
      a = a * decay + inflow[s] * mid;
      const p = pGrid[s];
      arr[s] = p > 0 ? clamp(1 - a / p, 0, 1) : 0;
    }
  }
  function stepIndex(tMs: number): number {
    return clamp(Math.round((tMs - t0) / stepMs), 0, nSteps);
  }
  const targetStep = stepIndex(tTarget);

  // --- per-trophy fit -------------------------------------------------------
  const zVals = (c.levels && c.levels.length === 5 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95])
    .map((lv: number) => normInv(clamp(lv, 0.001, 0.999)));
  const sHorizon = 0.10 + 0.16 * Math.sqrt(horizonDays);
  const distinctP = changePts.length;
  const stale = distinctP <= 2 && tauLast < 48;
  const sStale = stale ? 0.45 : 0;

  const rows: any[] = [];
  for (let i = 0; i < n; i++) {
    const t = trophies[i];
    const dPrior = priorDelay(t);
    // Observations: (C_j(D) is filled per D below), rates per read.
    const obs: { step: number; r: number }[] = [];
    for (const h of hist) {
      const r = h.rates && h.rates[i];
      if (typeof r === "number" && Number.isFinite(r)) {
        // Evaluate at the refresh the read actually shows, not its clock.
        obs.push({ step: stepIndex(evalTime(Date.parse(h.at))), r: clamp(r, 0, 100) });
      }
    }
    let median: number;
    let sFit = 0.05;
    if (obs.length === 0) {
      median = 1.0;
      sFit = 0.8;
    } else {
      const rMax = Math.max(...obs.map((o) => o.r));
      const rMean = obs.reduce((s, o) => s + o.r, 0) / obs.length;
      let bestD = dPrior, bestObj = Infinity, bestR = rMean, bestC: number[] = [];
      for (let di = 0; di < D_GRID.length; di++) {
        const D = D_GRID[di];
        const arr = cAtStep[di];
        const cs = obs.map((o) => arr[o.step]);
        let num = 0, den2 = 0;
        for (let k = 0; k < obs.length; k++) { num += obs[k].r * cs[k]; den2 += cs[k] * cs[k]; }
        let R = den2 > 1e-12 ? num / den2 : rMean;
        R = clamp(R, rMax, 99.99);
        // Objective in logit space -- the scorer's scale -- so a miss from
        // 90% to 100% counts like the 1%-to-2% miss it really is; the
        // prior penalty is normalised so it is worth ~half an observation
        // of typical size rather than scaling with the rate's magnitude.
        let ss = 0;
        for (let k = 0; k < obs.length; k++) {
          const pred = clamp(R * cs[k], 0.01, 99.99);
          const e = logit(obs[k].r) - logit(pred);
          ss += e * e;
        }
        const obj = ss / obs.length + 0.5 * Math.pow(Math.log2(D / dPrior), 2);
        if (obj < bestObj) { bestObj = obj; bestD = D; bestR = R; bestC = cs; }
      }
      // Logit residuals of the winning fit -> spread contribution.
      let ss = 0, cnt = 0;
      for (let k = 0; k < obs.length; k++) {
        const pred = clamp(bestR * bestC[k], 0.01, 99.99);
        const e = logit(obs[k].r) - logit(pred);
        ss += e * e; cnt++;
      }
      sFit = cnt ? clamp(Math.sqrt(ss / cnt), 0.05, 0.6) : 0.05;
      median = clamp(bestR * cAtStep[D_GRID.indexOf(bestD)][targetStep], 0.01, 99.99);
    }
    const s = Math.sqrt(sHorizon * sHorizon + sFit * sFit + sStale * sStale + 0.01);
    const lm = logit(median);
    let q = zVals.map((z: number) => {
      const v = Math.round(clamp(logistic(lm + z * s), 0.01, 99.99) * 10000) / 10000;
      return v;
    });
    // A trophy every tracked player holds is universal by nature -- the
    // first mission, earned within minutes of starting.  The house's own
    // first refresh kept it at 100% while a 6x larger base arrived, so
    // the belief is a point mass at 100% with the model's diluted value
    // as the low tail: median and upper quantiles sit at 99.99.
    const lastRate = obs.length ? obs[obs.length - 1].r : 0;
    if (lastRate >= 99.99) {
      q = [q[0], 99.99, 99.99, 99.99, 99.99];
    }
    // Guarantee non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    rows.push({ id: t.id, q });
  }
  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inDir, f), "utf8")) as CaseFile;
    const out = forecastCase(c);
    const id = f.slice(0, -3);
    writeFileSync(join(outDir, id + ".out"), JSON.stringify(out));
  }
}

main();
