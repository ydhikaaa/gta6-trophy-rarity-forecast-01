// GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01
//
// Round 3 model: cohort maturity on a lognormal required-time kernel,
// an earner-conservation floor, refresh-aware player counts — plus a
// stale-page discount on projected growth, calibrated against what the
// server actually scored in rounds 1 and 2.
//
// Earn rate = earners / players tracked, and the two sides are modelled
// separately.  Each trophy's required playing time is lognormal (median
// D hours of player age, spread sigma): the matured share of a cohort
// of age a is Phi(ln(a / D) / sigma).  The thin early tail is the
// point — almost nobody earns a grind trophy in their first hours, so
// slow trophies dilute almost fully when the base jumps, while fast
// trophies fill in quickly.  Per trophy, D and the eventual share R
// are fit by least squares in logit space (the scorer's scale) over
// the refresh points, with a description/tier prior on D that carries
// flat histories and yields to data when the history moves.  The
// kernel spread sigma is fit once per case over the trophies whose
// histories actually resolve.
//
// Observations are censored by the page's print resolution: a printed
// 0.00 or 0.01 is an interval, not a point, so the fit scores such
// observations with a hinge cost in logit space (free inside the
// printed interval, squared distance outside it).
//
// The floor is model-free: earners never un-earn and the tracked base
// never shrinks, so the target rate cannot fall below
// r_last * P_last / P_target.
//
// What round 3 adds — the stale-page regime.  When the page has sat
// unchanged (one snapshot, or the last snapshot is old), the answer is
// a refresh that lands inside the horizon.  Rounds 1 and 2 both
// projected the kernel's full earner growth over that window and both
// overshot: the server scored round 2 at 0.202 with its medians
// systematically above the refreshed page.  The refresh pairs a fresh
// player count with trophy data from an older crawl wave, so only
// part of the projected growth is in the answering snapshot, while
// the floor (pure dilution, earners frozen) undershoots by the aging
// the pre-refresh base genuinely did.  The round-3 median therefore
// sits between the two: logit(median) = logit(floor) + KEEP * the
// model's logit growth over the floor, with KEEP = 0.35, and the
// stale-regime spread is scaled to 0.45 of round 2's, which
// over-covered once the medians came down.  Both adjustments are
// gated on the stale condition, so a dense, fresh history — the
// final's regime — runs the round-2 model unchanged.
//
// No network, no randomness, deterministic.  Node 22 stdlib only.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const D_GRID = [0.75, 1.5, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072];
const SIGMA_GRID = [0.85, 1.1, 1.4];
const SIGMA_DEFAULT = 1; // SIGMA_GRID[1] = 1.1
const RELEASE = Date.parse("2026-11-19T00:00:00Z");
const T_START = RELEASE - 13 * 3600000; // rolling-midnight launch: NZ first, 11:00 UTC 18 Nov
const STALE_KEEP = 0.35; // share of projected logit growth kept when the page is stale
const STALE_SPREAD = 0.45; // spread scale in the stale regime

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
// Standard normal CDF (Abramowitz-Stegun erf approximation).
function normCdf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + s * y);
}
// Inverse standard normal CDF (Acklam's approximation).
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
// Matured share of a cohort of age aHours for median required time D.
function maturedShare(aHours: number, D: number, sigma: number): number {
  if (aHours <= 0) return 0;
  return normCdf(Math.log(aHours / D) / sigma);
}
// Fit cost of predicting pred for a printed rate r.  Rates of 0.03+ are
// points; below that, print resolution makes them intervals and the
// cost is a hinge in logit space.
function obsCost(r: number, pred: number): number {
  const lp = logit(pred);
  if (r >= 0.03) {
    const e = logit(r) - lp;
    return e * e;
  }
  const hi = logit(r + 0.005);
  if (lp > hi) { const e = lp - hi; return e * e; }
  if (r > 0.005) {
    const lo = logit(r - 0.005);
    if (lp < lo) { const e = lo - lp; return e * e; }
  }
  return 0;
}

// Prior median required time (wall-clock hours of player age) from the
// trophy's description and tier.  Wall-clock, not playtime: a chapter
// a keen player finishes in one evening still takes a day or two of
// calendar time for the median tracked player.
function priorMedianHours(t: any): number {
  const d = (t.description || "").toLowerCase();
  const name = (t.name || "").toLowerCase();
  let h: number;
  if (t.id === 2 || name.includes("gone fishing")) h = 1;
  else if (d.includes("chapter 1")) h = 24;
  else if (d.includes("chapter 2")) h = 60;
  else if (d.includes("chapter 3")) h = 110;
  else if (d.includes("chapter 4")) h = 170;
  else if (d.includes("final mission") && t.tier === "gold") h = 280;
  else if (d.includes("100% completion")) h = 1400;
  else if (t.tier === "platinum") h = 2400;
  else if (d.includes("250 stunts")) h = 600;
  else if (d.includes("collectible card")) h = 500;
  else if (d.includes("waink")) h = 500;
  else if (d.includes("50 challenges")) h = 450;
  else if (d.includes("strangers")) h = 450;
  else if (d.includes("gold medal")) h = 400;
  else if (d.includes("laundering")) h = 400;
  else if (d.includes("garage properties")) h = 400;
  else if (d.includes("meditate")) h = 350;
  else if (d.includes("drone race")) h = 350;
  else if (d.includes("bank vaults")) h = 350;
  else if (d.includes("spend $100,000")) h = 300;
  else if (d.includes("20 different races")) h = 300;
  else if (d.includes("explore all of leonida")) h = 300;
  else if (d.includes("discover 100 vehicles")) h = 200;
  else if (d.includes("register 10 vehicles")) h = 250;
  else if (d.includes("arcade game")) h = 90;
  else if (d.includes("fishing challenge")) h = 60;
  else if (d.includes("each type of activity")) h = 120;
  else if (d.startsWith("complete the final mission for")) h = 320;
  else if (d.startsWith("complete ")) h = 200;
  else if (t.tier === "silver") h = 450;
  else if (t.tier === "gold") h = 700;
  else h = 300;
  // Hidden trophies are not targeted on purpose; incidental discovery
  // runs later.  Story trophies are earned by playing, so they keep
  // their time.
  if (t.secret === true && !d.startsWith("complete")) h *= 1.3;
  return h;
}
// Ceiling on the eventual share R (percent who ever earn the trophy).
function tierCap(t: any): number {
  if (t.tier === "platinum") return 35;
  if (t.tier === "gold") return 70;
  if (t.tier === "silver") return 85;
  return 97;
}
// Middle-of-the-road eventual share: the centre for trophies never
// observed earned, and the weak prior centre for R in fits.
function tierMidShare(t: any): number {
  if (t.tier === "platinum") return 6;
  if (t.tier === "gold") return 20;
  if (t.tier === "silver") return 35;
  return 50;
}
function nearestGridD(h: number): number {
  return D_GRID.reduce((best, d) => (Math.abs(Math.log2(d / h)) < Math.abs(Math.log2(best / h)) ? d : best), D_GRID[0]);
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

  // --- refresh points ---------------------------------------------------
  // A read shows the page as of Exophase's last refresh before it.  A
  // refresh point starts where the page's content changed; its
  // snapshot time sigma is estimated half a refresh gap before the
  // read that first shows it (for a lone point, half the time the page
  // had already sat unchanged at the checkpoint).
  const refresh: { rho: number; sigma: number; p: number; rates: any[] }[] = [];
  {
    let prevP = -1, prevRates = "";
    const raw: { rho: number; p: number; rates: any[] }[] = [];
    for (const h of hist) {
      const p = Number(h.players_tracked);
      const ratesKey = JSON.stringify(h.rates || []);
      if (raw.length === 0 || p !== prevP || ratesKey !== prevRates) {
        raw.push({ rho: Date.parse(h.at), p: Number.isFinite(p) && p > 0 ? p : (raw.length ? raw[raw.length - 1].p : 0), rates: h.rates || [] });
      }
      prevP = p; prevRates = ratesKey;
    }
    for (let k = 0; k < raw.length; k++) {
      let lag: number;
      if (raw.length === 1) lag = clamp((tAt - raw[0].rho) / 2, 0, 8 * 3600000);
      else if (k === 0) lag = 0;
      else lag = clamp((raw[k].rho - raw[k - 1].rho) / 2, 0, 4 * 3600000);
      refresh.push({ rho: raw[k].rho, sigma: raw[k].rho - lag, p: raw[k].p, rates: raw[k].rates });
    }
  }
  const K = refresh.length;
  const pLast = K ? refresh[K - 1].p : 0;

  // --- target player count ----------------------------------------------
  let pTarget: number;
  {
    const pts = refresh.filter((r) => r.p > 0).map((r) => ({ tau: (r.sigma - RELEASE) / 3600000, p: r.p }));
    const tauTarget = (tTarget - RELEASE) / 3600000;
    if (pts.length >= 2) {
      // Power law in (hours since release + 6): ln P = a + b ln(tau + 6).
      let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
      for (const cp of pts) {
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
    } else if (pts.length === 1 && pts[0].tau < 24) {
      // Flat page on launch night: a refresh is due, and the rules
      // state the first refresh took the count 638 -> 3,868 (~6x).
      pTarget = pts[0].p * 6.07;
    } else {
      pTarget = pLast * (1 + 0.02 * horizonDays);
    }
    if (!Number.isFinite(pTarget) || pTarget <= 0) pTarget = Math.max(pLast, 1);
  }

  // --- true player curve on a time grid -----------------------------------
  // Points: (T_START, 0), each (sigma_k, P_k), (target, P_target).
  // Between points a concave ramp front-loads arrivals: a catch-up
  // jump is spread back over the window in which those players
  // actually arrived, not dumped at the refresh instant.
  const curve: { t: number; p: number }[] = [{ t: T_START, p: 0 }];
  for (const r of refresh) if (r.p > 0) curve.push({ t: r.sigma, p: r.p });
  curve.push({ t: tTarget, p: pTarget });
  curve.sort((a, b) => a.t - b.t);
  for (let k = 1; k < curve.length; k++) if (curve[k].p < curve[k - 1].p) curve[k].p = curve[k - 1].p;
  function pAt(tMs: number): number {
    if (tMs <= curve[0].t) return curve[0].p;
    for (let k = 1; k < curve.length; k++) {
      if (tMs <= curve[k].t) {
        const a = curve[k - 1], b = curve[k];
        const u = clamp((tMs - a.t) / Math.max(1, b.t - a.t), 0, 1);
        const s = 1 - (1 - u) * (1 - u);
        return a.p + (b.p - a.p) * s;
      }
    }
    return curve[curve.length - 1].p;
  }
  const spanDays = (tTarget - T_START) / 86400000;
  const stepMs = (spanDays > 3.5 ? 30 : 10) * 60000;
  const nSteps = Math.max(1, Math.ceil((tTarget - T_START) / stepMs));
  const inflow = new Float64Array(nSteps);
  {
    let prev = pAt(T_START);
    for (let s = 0; s < nSteps; s++) {
      const p = pAt(T_START + (s + 1) * stepMs);
      inflow[s] = Math.max(0, p - prev);
      prev = p;
    }
  }
  // Matured-player counts at every snapshot time, plus the target:
  // C[si][di][e] = sum over cohorts of inflow * maturedShare(age).
  const evalTimes: number[] = refresh.map((r) => r.sigma);
  evalTimes.push(tTarget);
  const targetEval = evalTimes.length - 1;
  const C: number[][][] = SIGMA_GRID.map((sigma) =>
    D_GRID.map((D) => {
      const arr = new Float64Array(evalTimes.length);
      for (let e = 0; e < evalTimes.length; e++) {
        const te = evalTimes[e];
        let m = 0;
        for (let s = 0; s < nSteps; s++) {
          const tc = T_START + (s + 0.5) * stepMs;
          if (tc >= te) break;
          if (inflow[s] > 0) m += inflow[s] * maturedShare((te - tc) / 3600000, D, sigma);
        }
        arr[e] = m;
      }
      return arr as any;
    })
  );
  function shareAt(si: number, di: number, e: number): number {
    const p = e === targetEval ? pTarget : Math.max(1, refresh[e].p);
    return clamp(C[si][di][e] / p, 0, 1);
  }

  // --- regime and spread ingredients --------------------------------------
  const zVals = (c.levels && c.levels.length === 5 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95])
    .map((lv: number) => normInv(clamp(lv, 0.001, 0.999)));
  const sHorizon = 0.12 + 0.18 * Math.sqrt(horizonDays);
  const staleHours = K ? Math.max(0, (tAt - refresh[K - 1].sigma) / 3600000) : 0;
  // Stale: the answering page is a refresh that lands inside the
  // horizon, so its count is fresh but its trophy data lags (header).
  const stale = K <= 1 || staleHours >= 1.5;
  let sRefresh: number;
  if (stale) {
    sRefresh = 0.55 * Math.min(1, 0.3 + staleHours / 6);
  } else {
    sRefresh = 0.10 + 0.05 * Math.sqrt(horizonDays);
  }

  // --- per-trophy observations ----------------------------------------------
  interface TrophyFit { median: number; sFit: number; obj: number; voter: boolean }
  const allObs: { e: number; r: number }[][] = [];
  for (let i = 0; i < n; i++) {
    const obs: { e: number; r: number }[] = [];
    for (let k = 0; k < K; k++) {
      const r = refresh[k].rates[i];
      if (typeof r === "number" && Number.isFinite(r)) obs.push({ e: k, r: clamp(r, 0, 100) });
    }
    allObs.push(obs);
  }

  // Fit one trophy under one kernel spread.  A trophy with no usable
  // observations gets a placeholder; the row assembly uses its prior
  // branch for those and ignores this fit.
  function fitTrophy(i: number, si: number): TrophyFit {
    const t = trophies[i];
    const dPrior = priorMedianHours(t);
    const cap = tierCap(t);
    const obs = allObs[i];
    if (obs.length === 0) return { median: 0.01, sFit: 0.8, obj: 0, voter: false };
    const distinct = new Set(obs.map((o) => o.e)).size;
    if (distinct <= 1) {
      // Flat history: one snapshot only, so (R, D) cannot be
      // separated.  D from the prior, R from the snapshot.
      const e0 = obs[0].e;
      const rMean = obs.reduce((s, o) => s + o.r, 0) / obs.length;
      const di = D_GRID.indexOf(nearestGridD(dPrior));
      const cSigma = shareAt(si, di, e0);
      let R = cSigma > 1e-9 ? rMean / cSigma : cap;
      let capHit = false;
      if (R > cap) { R = cap; capHit = true; }
      if (R < rMean) R = rMean;
      return { median: clamp(R * shareAt(si, di, targetEval), 0.01, 99.99), sFit: capHit ? 0.5 : 0.3, obj: 0, voter: false };
    }
    // Moving history: grid-search (D, R) on censored logit residuals,
    // with a soft prior penalty on D and a weak prior on R.
    const rTyp = Math.min(cap, tierMidShare(t) * 1.6);
    let bestObj = Infinity, bestR = 0, bestDi = 0, bestSs = 0;
    for (let di = 0; di < D_GRID.length; di++) {
      const cs = obs.map((o) => shareAt(si, di, o.e));
      const rMax = Math.max(...obs.map((o) => o.r));
      const ladder: number[] = [];
      for (let R = Math.max(rMax, 0.01); ; R *= 1.12) {
        ladder.push(Math.min(R, cap));
        if (R >= cap) break;
      }
      for (const R of ladder) {
        let ss = 0;
        for (let k = 0; k < obs.length; k++) {
          ss += obsCost(obs[k].r, clamp(R * cs[k], 0.01, 99.99));
        }
        const obj = ss / obs.length
          + 0.15 * Math.min(Math.pow(Math.log2(D_GRID[di] / dPrior), 2), 9)
          + 0.05 * Math.pow(Math.log(R / rTyp), 2);
        if (obj < bestObj) { bestObj = obj; bestR = R; bestDi = di; bestSs = ss; }
      }
    }
    const censoredShare = obs.filter((o) => o.r < 0.03).length / obs.length;
    let sFit = clamp(Math.sqrt(bestSs / obs.length), 0.05, 0.7);
    if (censoredShare >= 0.5) sFit = Math.max(sFit, 0.45);
    const voter = obs.some((o) => o.r >= 0.03);
    return { median: clamp(bestR * shareAt(si, bestDi, targetEval), 0.01, 99.99), sFit, obj: bestObj, voter };
  }

  // Case-wide kernel spread: total fit quality over the trophies
  // whose histories actually move and resolve (the voters).
  const fitsBySigma: TrophyFit[][] = SIGMA_GRID.map((_, si) =>
    trophies.map((_, i) => fitTrophy(i, si))
  );
  let siBest = SIGMA_DEFAULT;
  {
    let bestTot = Infinity;
    for (let si = 0; si < SIGMA_GRID.length; si++) {
      let tot = 0, voters = 0;
      for (let i = 0; i < n; i++) {
        if (fitsBySigma[si][i].voter) { tot += fitsBySigma[si][i].obj; voters++; }
      }
      if (voters > 0 && tot < bestTot) { bestTot = tot; siBest = si; }
    }
  }

  // --- assemble rows ----------------------------------------------------------
  const rows: any[] = [];
  for (let i = 0; i < n; i++) {
    const t = trophies[i];
    const obs = allObs[i];
    const lastObs = obs.length ? obs[obs.length - 1] : null;
    const rLast = lastObs ? lastObs.r : null;
    const pAtLast = lastObs ? Math.max(1, refresh[lastObs.e].p) : 0;
    const eLast = rLast !== null ? (rLast / 100) * pAtLast : null; // observed earners
    const floorRate = eLast !== null && eLast > 0 ? (eLast / pTarget) * 100 : 0;

    let median: number;
    let sFit: number;
    if (obs.length === 0) {
      // Never observed: prior share at the prior median time.
      const di = D_GRID.indexOf(nearestGridD(priorMedianHours(t)));
      median = clamp(tierMidShare(t) * shareAt(SIGMA_DEFAULT, di, targetEval), 0.01, 99.99);
      sFit = 0.8;
    } else if (Math.max(...obs.map((o) => o.r)) === 0) {
      // Observed, but nobody has earned it yet at print resolution:
      // the prior carries it, with the censored-observation spread.
      const di = D_GRID.indexOf(nearestGridD(priorMedianHours(t)));
      median = clamp(tierMidShare(t) * shareAt(SIGMA_DEFAULT, di, targetEval), 0.01, 99.99);
      sFit = 0.6;
    } else {
      const fit = fitsBySigma[siBest][i];
      median = fit.median;
      sFit = fit.sFit;
    }
    // Earner-conservation floor on the median.
    if (floorRate > 0) median = clamp(Math.max(median, floorRate), 0.01, 99.99);
    // Stale-page discount: keep only part of the projected growth
    // over the floor (see the header for the calibration evidence).
    if (stale && floorRate > 0 && median > floorRate) {
      const lm = logit(floorRate) + STALE_KEEP * (logit(median) - logit(floorRate));
      median = clamp(logistic(lm), 0.01, 99.99);
    }

    const sCnt = eLast !== null ? Math.min(0.5, 0.6 / Math.sqrt(eLast + 1)) : 0.3;
    let s = Math.sqrt(sHorizon * sHorizon + sFit * sFit + sRefresh * sRefresh + sCnt * sCnt);
    if (stale) s *= STALE_SPREAD;
    const lm = logit(median);
    let q = zVals.map((z: number) => Math.round(clamp(logistic(lm + z * s), 0.01, 99.99) * 10000) / 10000);
    // The 5% quantile cannot fall far below the conserved-earner
    // floor: those earners are already counted, and the base would
    // have to grow far past the extrapolated count to dilute them.
    if (floorRate > 0) q[0] = Math.max(q[0], Math.round(0.7 * floorRate * 10000) / 10000);
    // A trophy every tracked player holds is universal by nature —
    // the house's first refresh kept it at 100% while a 6x larger
    // base arrived, so the median and upper quantiles sit at 99.99
    // and only the low tail carries the diluted value.
    if (rLast !== null && rLast >= 99.99) {
      q = [q[0], 99.99, 99.99, 99.99, 99.99];
    }
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
