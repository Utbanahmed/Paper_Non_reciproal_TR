#!/usr/bin/env python3
"""
Leak localisation in a reservoir-pipe-valve (RPV) system from a valve-closure (water-hammer)
test: PINN with vs without a time-reversal (TR) term in the loss.

System and classical TR method follow the time-reversal defect-detection chapter of
M. Waqar, PhD thesis, HKUST (2022) (Waqar, Louati & Ghidaoui).

Usage (run from this folder; everything goes to ./results/):
    python tr_pinn_leak.py all          # every stage in order, stops at a failed check
    python tr_pinn_leak.py moc          # stages 1-2  (physics, MOC ground truth, checks a-e)
    python tr_pinn_leak.py trpinn       # stage 3     (TR-field PINN, check f)
    python tr_pinn_leak.py sanity       # stage 4     (forward PINN at true leak, check g)
    python tr_pinn_leak.py smoke        # smoke test + wall-time estimate
    python tr_pinn_leak.py sweep        # stage 5     (2x2 ablation)
    python tr_pinn_leak.py report       # figures, summary.csv, REPORT.md from saved runs
"""
import os, sys, time, json, math, csv, copy
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================================
# PARAMETERS (all of them)
# =====================================================================================
# --- physical (dimensional, SI) ---
L = 200.0                 # pipe length [m]
D = 0.0792                # diameter [m]
A_WAVE = 1302.0           # wave speed a [m/s]
F_DW = 0.02               # Darcy-Weisbach friction factor
G = 9.81
H0 = 20.0                 # steady head at the valve [m]
Q0 = 0.4e-3               # steady valve flow [m^3/s]
S_L_FRAC = 0.001          # leak area / pipe area (0.1 %)
CHI_TRUE = 0.4            # leak position x/L (0.6 L from the valve)
R_MULT = 1.0              # friction multiplier (1.0 = physical)
R_STRESS = 0.3            # nondim r used in the friction stress case
TC = 0.025                # valve closure time (10-90 %) [s]
T0 = 0.046                # closure start (tanh centre) [s]
T_REC = 0.40              # record length [s]
FS = 2048.0               # sampling rate [Hz]
LEAK_MODEL = "linear"     # 'linear' (same as PINN) or 'nonlinear' (orifice sqrt(H0+h))

# --- MOC ---
N_MOC = 500               # reaches (Courant 1; leak exactly on a node)
F_ENV_FRAC = 0.01         # f_max: spectrum of dS/dt falls to 1 % of its peak
F_CUT_MULT = 1.5          # derivative filter: zero all content above 1.5 f_max
SNRS = [np.inf, 10.0, 0.0]
NOISE_SEED = 1234

# --- PINN (identical for every method) ---
WIDTH = 64                # hidden width          (spec: 128; see REPORT deviations)
DEPTH = 4                 # hidden layers
N_FF = 64                 # Fourier features (embedding dim = 2*N_FF)
FF_SIGMA = 2.0            # std of Fourier-feature frequencies (cycles per nondim unit)
FF_MODE = "characteristic"  # 'characteristic': features sin/cos(2 pi k (t -+ x)); 'gaussian': isotropic
N_COL = 4000              # uniform collocation points (spec: 20000)
N_COL_LEAK = 1000         # extra points within +-5 eps of chi_hat (spec: 2000)
RESAMPLE_EVERY = 500      # resample uniform collocation
RESAMPLE_LEAK_EVERY = 100 # resample the points around the moving chi_hat
CAUSAL_BINS = 32          # causal weighting (Wang, Sankaran & Perdikaris 2022)
CAUSAL_EPS = 1.0
LR_NET = 1e-3
LR_LEAK = 5e-3
N_ITER = 4000             # forward-PINN Adam iterations (spec: 20000; CPU budget, see REPORT)
N_WARM = 400              # warm-up (leak frozen), same 10 % fraction as the spec (2000/20000)
TR_DECAY_END = 0.7        # lambda_TR cosine-decays to 0 at 70 % of N_ITER
N_ITER_TR = 8000          # TR-field PINN iterations
N_ITER_SANITY = N_ITER
CL_HAT0 = 0.02
TR_TAPER = True           # raised-cosine onset (width tc*) on the re-emitted TR signal (MOC and PINN)
SCAT_SCALE = 0.1          # output scale of the forward net (x Q0*): it learns the scattered field
TR_NET_SCALE = 0.1        # output scale of the TR net correction (x max|dds|)
X_IMG = (0.1, 0.9)        # imaging window
N_XB = 801                # x grid on which the PINN-TR image B(x) is tabulated

# --- experiment design ---
OFFSETS_K = [1, 2, 4, 8]
N_RANDOM_INIT = 4
SMOKE_ITER = 500
SEEDS = [0, 1, 2]
SEEDS_NOISY = [0]
SEEDS_STRESS = [0]
A_MODEL_FACTORS = [0.95, 1.05]
WALL_BUDGET_H = 3.0

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUNS_DIR = os.path.join(RESULTS, "runs")
CACHE = os.path.join(RESULTS, "cache")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WORKERS = 1 if DEVICE.type == "cuda" else max(1, (os.cpu_count() or 1))
DTYPE = torch.float32

for d_ in (RESULTS, RUNS_DIR, CACHE):
    os.makedirs(d_, exist_ok=True)


def set_seeds(s):
    np.random.seed(s)
    torch.manual_seed(s)


set_seeds(0)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# =====================================================================================
# 1. PHYSICS: derived parameters, nondimensionalisation
# =====================================================================================
A_PIPE = math.pi * D ** 2 / 4.0
S_L = S_L_FRAC * A_PIPE
LEAK_DIST = (1.0 - CHI_TRUE) * L              # leak -> valve
V0 = Q0 / A_PIPE
H_L0 = H0 + F_DW * LEAK_DIST / D * V0 ** 2 / (2 * G)   # flow Q0 between leak and valve
Q_L0 = S_L * math.sqrt(2 * G * H_L0)
QM = Q0 + Q_L0
R_DIM = F_DW * QM / (D * A_PIPE) * R_MULT     # linearised friction [1/s]
F_MAX = None                                  # set by excitation_spectrum()


class Model:
    """Nondimensional model for an assumed wave speed a_m (truth: a_m = a)."""

    def __init__(self, a_m=A_WAVE, r=None):
        self.a = a_m
        self.Z = a_m / (G * A_PIPE)
        self.ZL = 2 * H_L0 / Q_L0
        self.cL = self.Z / self.ZL
        self.Q0s = Q0 * self.Z / H0
        self.r = R_DIM * L / a_m if r is None else r
        self.ts = L / a_m                      # time scale [s]
        self.T = T_REC / self.ts
        self.t0 = T0 / self.ts
        self.tc = (TC / 2.2) / self.ts
        self.eps = self.tc / 2.0               # leak mollifier width

    def S(self, t):
        return 0.5 * (1 + np.tanh((t - self.t0) / self.tc))

    def dS(self, t):
        return 0.5 / self.tc / np.cosh((t - self.t0) / self.tc) ** 2

    def fcut(self):                            # nondim cut-off frequency
        return F_CUT_MULT * F_MAX * self.ts

    def cl_to_sl(self, cl):
        QL = 2 * H_L0 * cl / self.Z
        return QL / math.sqrt(2 * G * H_L0)


def excitation_spectrum():
    """f_max: where |FFT(dS/dt)| falls to 1 % of its peak."""
    global F_MAX
    dt = 1.0 / (64 * FS)
    t = np.arange(0, 4.0, dt)                       # long window, pulse fully contained
    tcd = TC / 2.2
    dS = 0.5 / tcd / np.cosh((t - 1.0) / tcd) ** 2
    spec = np.abs(np.fft.rfft(dS)) * dt
    f = np.fft.rfftfreq(len(t), dt)
    k = np.argmax(spec < F_ENV_FRAC * spec.max())
    # linear interpolation between the bracketing bins
    f1, f2, s1, s2 = f[k - 1], f[k], spec[k - 1], spec[k]
    F_MAX = f1 + (F_ENV_FRAC * spec.max() - s1) * (f2 - f1) / (s2 - s1)
    return f, spec


excitation_spectrum()
LAM_MIN = A_WAVE / F_MAX
TOL = LAM_MIN / 2 / L                                 # success threshold (nondim)
TRUE = Model(A_WAVE)


def print_physics():
    m = TRUE
    log("=== 1. PHYSICS ===")
    log(f"A = {A_PIPE:.4e} m^2, s_L = {S_L:.3e} m^2, V0 = {V0:.4f} m/s")
    log(f"H_L0 = {H_L0:.4f} m  (H0 + friction loss leak->valve {H_L0 - H0:.4f} m)")
    log(f"Q_L0 = {Q_L0:.4e} m^3/s ({100 * Q_L0 / Q0:.1f} % of Q0)")
    log(f"Z = {m.Z:.1f} s/m^2, Z_L = {m.ZL:.1f} s/m^2, c_L = Z/Z_L = {m.cL:.4f}")
    log(f"H_J = Z*Q0 = {m.Z * Q0:.3f} m;  Q0* = {m.Q0s:.4f};  leak reflection -cL/(2+cL) = {-m.cL / (2 + m.cL):.4f}")
    log(f"R = {R_DIM:.4e} 1/s, r = R L/a = {m.r:.5f}")
    log(f"L/a = {m.ts:.5f} s; T* = {m.T:.4f}; t0* = {m.t0:.4f}; tc* = {m.tc:.5f}; eps = {m.eps:.5f}")
    log(f"f_max = {F_MAX:.2f} Hz; lambda_min = {LAM_MIN:.2f} m; lambda_min/2 = {LAM_MIN / 2:.2f} m = {TOL:.4f} (nondim)")
    log(f"derivative filter cut-off 1.5 f_max = {F_CUT_MULT * F_MAX:.1f} Hz")


# =====================================================================================
# Spectral differentiation (the single filter used everywhere)
# =====================================================================================
def spec_dt(sig, dt, fcut, axis=-1):
    """Filtered d/dt by FFT: remove the linear trend through the end points (so that the
    periodic extension is continuous), multiply by i*omega, zero |f| > fcut, add the
    trend slope back.  sig sampled uniformly with step dt (nondim)."""
    sig = np.moveaxis(np.asarray(sig, dtype=float), axis, -1)
    n = sig.shape[-1]
    s = np.arange(n) / (n - 1)
    slope = (sig[..., -1:] - sig[..., :1])
    trend = sig[..., :1] + slope * s
    F = np.fft.rfft(sig - trend, axis=-1)
    f = np.fft.rfftfreq(n, dt)
    F = F * (2j * np.pi * f)
    F[..., f > fcut] = 0.0
    out = np.fft.irfft(F, n=n, axis=-1) + slope / ((n - 1) * dt)
    return np.moveaxis(out, -1, axis)


def spec_filter(sig, dt, fcut, axis=-1):
    """Same low-pass (and trend handling) without differentiation."""
    sig = np.moveaxis(np.asarray(sig, dtype=float), axis, -1)
    n = sig.shape[-1]
    s = np.arange(n) / (n - 1)
    trend = sig[..., :1] + (sig[..., -1:] - sig[..., :1]) * s
    F = np.fft.rfft(sig - trend, axis=-1)
    f = np.fft.rfftfreq(n, dt)
    F[..., f > fcut] = 0.0
    return np.moveaxis(np.fft.irfft(F, n=n, axis=-1) + trend, -1, axis)


# =====================================================================================
# 2. MOC solver (nondimensional, Courant 1)
# =====================================================================================
def moc(N, T, r, valve_q, cL=0.0, chi=None, leak_model="linear", Hs=None):
    """h_t + q_x = -cL h delta(x-chi); q_t + h_x + r q = 0; h(0)=0; q(1,t)=valve_q(t).
    Returns t (nt+1,), x (N+1,), h, q (N+1, nt+1); at the leak node q is the downstream flow.
    nt = ceil(T/dt), so the grid covers [0, T]."""
    dx = 1.0 / N
    dt = dx
    nt = int(math.ceil(T / dt - 1e-9))
    t = np.arange(nt + 1) * dt
    x = np.arange(N + 1) * dx
    h = np.zeros((N + 1, nt + 1))
    q = np.zeros((N + 1, nt + 1))
    iL = None
    if cL > 0:
        iL = int(round(chi * N))
        assert abs(iL * dx - chi) < 1e-9, "leak must lie on a node"
    hc = np.zeros(N + 1)
    qc = np.zeros(N + 1)        # downstream (right) flow at every node
    qc[-1] = valve_q(t[0])
    ql_leak = qc[iL] if iL is not None else 0.0   # upstream (left) flow at leak node
    q[:, 0] = qc
    if leak_model == "nonlinear":
        qLs = Q_L0 * TRUE.Z / H0       # nondim steady leak flow
        rho = H0 / H_L0
    for n in range(nt):
        qA = qc[:-1]                    # flow leaving node i-1 to the right
        Cp = hc[:-1] + qA - r * dt * qA          # arrives at nodes 1..N
        qB = qc[1:].copy()                       # flow entering node i+1 from the left
        if iL is not None:
            qB[iL - 1] = ql_leak                 # node iL-1 sees the upstream flow of the leak node
        Cm = hc[1:] - qB + r * dt * qB           # arrives at nodes 0..N-1
        hn = np.empty(N + 1)
        qn = np.empty(N + 1)
        hn[1:N] = 0.5 * (Cp[:-1] + Cm[1:])
        qn[1:N] = 0.5 * (Cp[:-1] - Cm[1:])
        hn[0] = 0.0
        qn[0] = -Cm[0]
        qn[N] = valve_q(t[n + 1])
        hn[N] = Cp[-1] - qn[N]
        if iL is not None:
            cp, cm = Cp[iL - 1], Cm[iL]
            if leak_model == "linear":
                hp = (cp + cm) / (2 + cL)
            else:     # cp + cm = 2h + qLs*(sqrt(1 + rho h) - 1)
                hp = (cp + cm) / (2 + cL)
                for _ in range(30):
                    fval = 2 * hp + qLs * (math.sqrt(1 + rho * hp) - 1) - (cp + cm)
                    fp = 2 + 0.5 * qLs * rho / math.sqrt(1 + rho * hp)
                    hp -= fval / fp
            hn[iL] = hp
            ql_leak = cp - hp
            qn[iL] = hp - cm
        hc, qc = hn, qn
        h[:, n + 1] = hc
        q[:, n + 1] = qc
    return t, x, h, q


def resample(tg, y, tq):
    """Cubic-spline resampling along the last axis (from MOC grid to other times)."""
    return CubicSpline(tg, y, axis=-1, extrapolate=True)(tq)


# =====================================================================================
# Data set for one condition (truth vs model)
# =====================================================================================
class Condition:
    def __init__(self, name, snr=np.inf, a_fac=1.0, r=None, leak_model=LEAK_MODEL):
        self.name, self.snr, self.a_fac, self.r_override = name, snr, a_fac, r
        self.leak_model = leak_model

    def key(self):
        return self.name


def build_data(cond, N=N_MOC, verbose=False):
    """Everything the PINNs need for one condition; cached to npz."""
    path = os.path.join(CACHE, f"data_{cond.name}_N{N}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tru = Model(A_WAVE, r=cond.r_override)
    mod = Model(A_WAVE * cond.a_fac, r=cond.r_override)
    # --- (i) truth with leak: measured head at the valve (true a) ---
    tt, xx, h, q = moc(N, tru.T + 0.05, tru.r, lambda t: -tru.Q0s * tru.S(t), cL=tru.cL,
                       chi=CHI_TRUE, leak_model=cond.leak_model)
    ns = int(math.floor(T_REC * FS + 1e-9)) + 1
    t_dim = np.arange(ns) / FS
    d_clean = resample(tt, h[-1], t_dim / tru.ts)
    # --- (ii) model, leak-free, same excitation (model a) ---
    tm, xm, h0, q0 = moc(N, mod.T, mod.r, lambda t: -mod.Q0s * mod.S(t))
    tk = t_dim / mod.ts                                      # data times in model units
    h01 = resample(tm, h0[-1], tk)
    ds_clean = d_clean - h01
    rng = np.random.default_rng(NOISE_SEED)
    if np.isfinite(cond.snr):
        sig = np.max(np.abs(ds_clean)) * 10 ** (-cond.snr / 20)
        d = d_clean + sig * rng.standard_normal(ns)
    else:
        sig = 0.0
        d = d_clean.copy()
    ds = d - h01
    dtk = tk[1] - tk[0]
    fc = mod.fcut()
    dds = spec_dt(ds, dtk, fc)
    ddds = spec_dt(dds, dtk, fc)                             # for the TR-PINN valve BC derivative
    w0 = spec_dt(h0, tm[1] - tm[0], fc, axis=1)              # incident field derivative (model grid)
    w0_1k = resample(tm, w0[-1], tk)
    # --- TR field (MOC, leak-free model, reversed scattered derivative re-emitted) ---
    dds_spl = CubicSpline(tk, dds, extrapolate=True)
    tb, xb, hb, qb = moc(N, mod.T, mod.r, lambda tau: -dds_spl(mod.T - tau) * taper(tau, mod.tc))
    out = dict(t_moc=tt, x=xx, h=h, q=q, t_dim=t_dim, tk=tk, d=d, d_clean=d_clean, h01=h01,
               ds=ds, ds_clean=ds_clean, dds=dds, ddds=ddds, sig=sig, tm=tm, h0=h0, q0=q0, w0=w0,
               w0_1k=w0_1k, tb=tb, hb=hb, qb=qb, T=mod.T, t0=mod.t0, tc=mod.tc, eps=mod.eps,
               r=mod.r, Q0s=mod.Q0s, cL_true=tru.cL, a_m=mod.a, Z_m=mod.Z, fcut=fc)
    np.savez(path, **out)
    return out


def taper(tau, width):
    """Raised-cosine onset of the re-emitted TR signal (zero IC consistent with the valve BC)."""
    if not TR_TAPER:
        return np.ones_like(np.asarray(tau, dtype=float))
    u = np.clip(np.asarray(tau, dtype=float) / width, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * u))


def hb_reversed(D_, i_x=None):
    """hb(x, T - t_n) on the model forward grid t_n (rows = x nodes)."""
    tm, tb, hb, T = D_["tm"], D_["tb"], D_["hb"], float(D_["T"])
    taus = T - tm
    hbx = hb if i_x is None else hb[i_x:i_x + 1]
    return np.stack([np.interp(taus, tb, row) for row in hbx])


def tr_image(hbr, w0, dt):
    """Normalised zero-lag imaging condition B(x) = -<hb(x,T-t), w0(x,t)> / norms."""
    num = -np.sum(hbr * w0, axis=1) * dt
    den = np.sqrt(np.sum(hbr ** 2, axis=1) * dt) * np.sqrt(np.sum(w0 ** 2, axis=1) * dt) + 1e-30
    return num / den


def classical_image(D_, xs):
    """Undamped matched filter on the first half-period (thesis baseline)."""
    tk, dds, w01 = D_["tk"], D_["dds"], D_["w0"][-1]
    tm = D_["tm"]
    tend = float(D_["t0"]) + 2 - 3 * float(D_["tc"])
    m = tk < tend
    B = np.zeros(len(xs))
    for j, x in enumerate(xs):
        G = np.interp(tk[m] - 2 * (1 - x), tm, w01, left=0.0, right=0.0)
        B[j] = -np.sum(G * dds[m]) / (np.sqrt(np.sum(G ** 2)) + 1e-30)
    nrm = np.sqrt(np.sum(dds[m] ** 2))
    return B / (nrm + 1e-30)


# =====================================================================================
# Stage 1-2 driver: MOC ground truth, checks (a)-(e)
# =====================================================================================
CHECKS = {}


def record_check(name, passed, msg):
    CHECKS[name] = dict(passed=bool(passed), msg=msg)
    log(f"CHECK ({name}) {'PASS' if passed else 'FAIL'}: {msg}")
    with open(os.path.join(RESULTS, "checks.json"), "w") as f:
        json.dump(CHECKS, f, indent=1)


def load_checks():
    p = os.path.join(RESULTS, "checks.json")
    if os.path.exists(p):
        CHECKS.update(json.load(open(p)))


def stage_moc():
    print_physics()
    log("=== 2. MOC ground truth ===")
    base = Condition("base")
    D1 = build_data(base, N_MOC)
    D2 = build_data(base, 2 * N_MOC)
    m = TRUE
    # (a) grid convergence
    rel = np.linalg.norm(D2["d_clean"] - D1["d_clean"]) / np.linalg.norm(D1["d_clean"])
    relmax = np.max(np.abs(D2["d_clean"] - D1["d_clean"])) / np.max(np.abs(D1["d_clean"]))
    relds = np.linalg.norm(D2["ds_clean"] - D1["ds_clean"]) / np.linalg.norm(D1["ds_clean"])
    record_check("a", rel < 0.01 and relmax < 0.01,
                 f"N={N_MOC}->{2 * N_MOC}: rel L2 change of d = {rel:.2e}, rel max = {relmax:.2e}"
                 f" (scattered ds: {relds:.2e})")
    # (b) Joukowsky step and leak reflection
    tk, d, ds = D1["tk"], D1["d_clean"], D1["ds_clean"]
    HJ = m.Q0s
    step = np.interp(m.t0 + 1.0, tk, d)
    t_arr = m.t0 + 2 * (1 - CHI_TRUE)
    plateau = np.mean(ds[(tk > t_arr + 0.3) & (tk < m.t0 + 1.9)])
    half = np.argmax(np.abs(ds) > 0.5 * abs(plateau))
    t_half = np.interp(0.5 * abs(plateau), np.abs(ds[half - 1:half + 1]), tk[half - 1:half + 1])
    amp_spec = -m.cL / (2 + m.cL) * HJ
    amp_2x = -2 * m.cL / (2 + m.cL) * HJ
    ok_step = abs(step - HJ) / HJ < 0.02
    ok_t = abs(t_half - t_arr) < 0.5 * m.tc
    ok_amp = abs(plateau - amp_2x) / abs(amp_2x) < 0.1
    record_check("b", ok_step and ok_t and ok_amp,
                 f"step at valve {step * H0:.3f} m vs H_J {HJ * H0:.3f} m (err {100 * abs(step - HJ) / HJ:.2f} %); "
                 f"leak echo half-amplitude time {t_half:.4f} vs t0*+2(1-chi) = {t_arr:.4f}; "
                 f"echo plateau {plateau * H0:.4f} m = {plateau / HJ:.4f} H_J. Spec formula -cL/(2+cL) H_J = "
                 f"{amp_spec * H0:.4f} m; with the closed-valve doubling 2x(-cL/(2+cL)) H_J = {amp_2x * H0:.4f} m "
                 f"(ratio measured/2x = {plateau / amp_2x:.3f})")
    # plots: x-t field, signals
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    im = ax.pcolormesh(D1["t_moc"], D1["x"], D1["h"] * H0, shading="auto", cmap="RdBu_r",
                       vmin=-12, vmax=12)
    ax.axhline(CHI_TRUE, color="k", ls="--", lw=0.8)
    ax.set_xlabel("t* = t a/L"); ax.set_ylabel("x/L"); ax.set_title("MOC head perturbation h(x,t) [m], with leak")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig01_moc_field.png"), dpi=130)
    plt.close(fig)
    Dn = {s: build_data(Condition(snr_name(s), snr=s), N_MOC) for s in SNRS}
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(tk, D1["d_clean"] * H0, label="d(t) (leak)")
    axs[0].plot(tk, D1["h01"] * H0, "--", label="h0(1,t) (leak-free)")
    axs[0].set_ylabel("head [m]"); axs[0].legend()
    for s, c in zip(SNRS, ["k", "C1", "C2"]):
        axs[1].plot(tk, Dn[s]["ds"] * H0, color=c, lw=0.8 if s == np.inf else 0.5,
                    label=f"ds, SNR={s} dB", zorder=3 if s == np.inf else 1)
        axs[2].plot(tk, Dn[s]["dds"] * H0, color=c, lw=0.8 if s == np.inf else 0.5,
                    label=f"dds, SNR={s} dB", zorder=3 if s == np.inf else 1)
    axs[1].set_ylabel("ds [m]"); axs[2].set_ylabel("d(ds)/dt* [m]"); axs[2].set_xlabel("t*")
    for a_ in axs[1:]:
        a_.legend(fontsize=7); a_.axvline(t_arr, color="grey", ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig02_signals.png"), dpi=130); plt.close(fig)

    log("=== 3. TR field, duality, imaging (MOC) ===")
    iL = int(round(CHI_TRUE * N_MOC))
    dt = D1["tm"][1] - D1["tm"][0]
    lhs = np.sum(D1["dds"] ** 2) * (tk[1] - tk[0])
    hbr = hb_reversed(D1)
    w_leak = spec_dt(D1["h"][iL, :len(D1["tm"])], dt, D1["fcut"])
    rhs = -m.cL * np.sum(hbr[iL] * w_leak) * dt
    rhs_born = -m.cL * np.sum(hbr[iL] * D1["w0"][iL]) * dt
    mis = abs(rhs - lhs) / lhs
    mis_b = abs(rhs_born - lhs) / lhs
    record_check("c", mis < 0.02, f"duality: int dds^2 = {lhs:.5e}; -cL int hb*dh/dt (exact field) = {rhs:.5e} "
                                  f"(mismatch {100 * mis:.2f} %); Born (h0 at leak) = {rhs_born:.5e} "
                                  f"(mismatch {100 * mis_b:.2f} %)")
    xs = D1["x"]
    msk = (xs > X_IMG[0]) & (xs < X_IMG[1])
    B = tr_image(hbr[msk], D1["w0"][msk], dt)
    xB = xs[msk][np.argmax(B)]
    record_check("d", abs(xB - CHI_TRUE) < TOL, f"MOC TR image argmax x = {xB:.4f} (chi = {CHI_TRUE}, "
                                                 f"error {abs(xB - CHI_TRUE) / TOL:.2f} lambda_min/2)")
    xc = np.linspace(X_IMG[0], X_IMG[1], 801)
    Bc = classical_image(D1, xc)
    xcB = xc[np.argmax(Bc)]
    record_check("e", abs(xcB - CHI_TRUE) < TOL, f"classical matched filter argmax x = {xcB:.4f} "
                                                  f"(error {abs(xcB - CHI_TRUE) / TOL:.2f} lambda_min/2)")
    np.savez(os.path.join(CACHE, "images_moc_base.npz"), x=xs[msk], B=B, xc=xc, Bc=Bc)
    # classical image for the noisy data too (for the report)
    cl = {}
    for s in SNRS:
        bc = classical_image(Dn[s], xc)
        cl[snr_name(s)] = float(xc[np.argmax(bc)])
    json.dump(cl, open(os.path.join(CACHE, "classical.json"), "w"))
    return D1


def snr_name(s):
    return "base" if not np.isfinite(s) else f"snr{int(s)}"




# =====================================================================================
# 4. PINNs
# =====================================================================================
TWO_PI = 2 * math.pi


class FFNet(nn.Module):
    """MLP with Fourier-feature embedding, tanh.  Returns the outputs and their exact
    x- and t-derivatives (forward-mode tangents propagated alongside the values)."""

    def __init__(self, seed, n_out=2):
        super().__init__()
        g = torch.Generator().manual_seed(10_000 + seed)
        if FF_MODE == "characteristic":     # k ~ N(0, sigma^2) along the directions t - x and t + x
            k = torch.randn(N_FF, generator=g) * FF_SIGMA * TWO_PI
            sgn = torch.ones(N_FF)
            sgn[: N_FF // 2] = -1.0
            Bff = torch.stack([sgn * k, k])
        else:
            Bff = torch.randn(2, N_FF, generator=g) * FF_SIGMA * TWO_PI
        self.register_buffer("Bff", Bff)
        dims = [2 * N_FF] + [WIDTH] * DEPTH + [n_out]
        self.lins = nn.ModuleList()
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            with torch.no_grad():
                lin.weight.copy_(torch.randn(dims[i + 1], dims[i], generator=g) * std)
                lin.bias.zero_()
            self.lins.append(lin)

    def forward(self, x, t):
        n = x.shape[0]
        Bx, Bt = self.Bff[0], self.Bff[1]
        p = x[:, None] * Bx + t[:, None] * Bt
        s, c = torch.sin(p), torch.cos(p)
        Z = torch.cat([torch.cat([s, c], 1), torch.cat([c * Bx, -s * Bx], 1),
                       torch.cat([c * Bt, -s * Bt], 1)], 0)
        for lin in self.lins[:-1]:
            Y = Z @ lin.weight.T
            a = torch.tanh(Y[:n] + lin.bias)
            ap = (1 - a * a).repeat(2, 1)
            Z = torch.cat([a, ap * Y[n:]], 0)
        lin = self.lins[-1]
        Y = Z @ lin.weight.T
        return Y[:n] + lin.bias, Y[n:2 * n], Y[2 * n:]


def tt(a):
    return torch.as_tensor(np.asarray(a), dtype=DTYPE, device=DEVICE)


class Interp1:
    """Linear interpolation on a uniform grid (torch, differentiable in the query)."""

    def __init__(self, x0, dx, y):
        self.x0, self.dx, self.y = float(x0), float(dx), tt(y)
        self.n = self.y.shape[-1]

    def __call__(self, xq):
        u = (xq - self.x0) / self.dx
        i = torch.clamp(torch.floor(u.detach()).long(), 0, self.n - 2)
        w = u - i
        return self.y[i] * (1 - w) + self.y[i + 1] * w


class Bilinear:
    """Bilinear interpolation of a field on the MOC grid (x rows, t columns)."""

    def __init__(self, x, t, F):
        self.dx, self.dt = float(x[1] - x[0]), float(t[1] - t[0])
        self.F = tt(F)
        self.nx, self.nt = self.F.shape

    def __call__(self, xq, tq):
        u, v = xq / self.dx, tq / self.dt
        i = torch.clamp(torch.floor(u.detach()).long(), 0, self.nx - 2)
        j = torch.clamp(torch.floor(v.detach()).long(), 0, self.nt - 2)
        a, b = u - i, v - j
        F = self.F
        return ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j]
                + (1 - a) * b * F[i, j + 1] + a * b * F[i + 1, j + 1])


def causal_loss(res2, t, T):
    """Causal residual weighting: bins in t, w_i = exp(-eps * sum_{k<i} L_k) (no gradient)."""
    b = torch.clamp((t / T * CAUSAL_BINS).long(), 0, CAUSAL_BINS - 1)
    Lb = torch.zeros(CAUSAL_BINS, device=DEVICE, dtype=DTYPE).index_add_(0, b, res2)
    cnt = torch.bincount(b, minlength=CAUSAL_BINS).clamp(min=1).to(DTYPE)
    Lb = Lb / cnt
    cum = torch.cumsum(Lb.detach(), 0) - Lb.detach()
    w = torch.exp(-CAUSAL_EPS * cum)
    return torch.mean(w * Lb), w.min().item()


def sample_uniform(n, T, gen):
    x = torch.rand(n, generator=gen, device="cpu").to(DEVICE, DTYPE)
    t = torch.rand(n, generator=gen, device="cpu").to(DEVICE, DTYPE) * T
    return x, t


# ---------------------------------------------------------------- TR-field PINN ------
class TRField:
    """hb_tau + qb_x = 0; qb_tau + hb_x + r qb = 0; hb(0)=0; qb(1,tau) = G(tau) = -dds(T - tau) (tapered);
    zero IC.  Output transform = undamped method-of-images (d'Alembert) solution + network correction:
        hb = -G(tau+x-1) + G(tau-x-1) + G(tau+x-3) + x (tau/T) Nh
        qb =  G(tau+x-1) + G(tau-x-1) - G(tau+x-3) + (1-x)(tau/T) Nq
    (exact BCs and IC for tau < 3; the lift solves the r = 0 equations exactly, so the network
    learns the friction correction).  Fields normalised by max|dds|."""

    def __init__(self, D_, seed=0):
        self.T, self.r = float(D_["T"]), float(D_["r"])
        tk, dds = D_["tk"], D_["dds"]
        self.scale = float(np.max(np.abs(dds)))
        tau = np.linspace(-3.5, self.T, 16385)
        g = -CubicSpline(tk, dds, extrapolate=True)(self.T - np.clip(tau, 0, None)) * taper(tau, float(D_["tc"]))
        g[tau < 0] = 0.0
        self.G = Interp1(tau[0], tau[1] - tau[0], g / self.scale)
        self.net = FFNet(seed).to(DEVICE)

    def lift(self, x, tau):
        a, b, c = self.G(tau + x - 1), self.G(tau - x - 1), self.G(tau + x - 3)
        return -a + b + c, a + b - c

    def net_part(self, x, tau):
        N, Nx, Nt = self.net(x, tau)
        T, sc = self.T, TR_NET_SCALE
        Nh, Nq, Nhx, Nqx, Nht, Nqt = N[:, 0], N[:, 1], Nx[:, 0], Nx[:, 1], Nt[:, 0], Nt[:, 1]
        s = tau / T
        h = sc * x * s * Nh
        hx = sc * (s * Nh + x * s * Nhx)
        ht = sc * x * (Nh / T + s * Nht)
        q = sc * (1 - x) * s * Nq
        qx = sc * (-s * Nq + (1 - x) * s * Nqx)
        qt = sc * (1 - x) * (Nq / T + s * Nqt)
        return h, q, hx, ht, qx, qt

    def loss(self, x, tau):
        h, q, hx, ht, qx, qt = self.net_part(x, tau)
        hL, qL = self.lift(x, tau)
        r1 = ht + qx                         # the lift satisfies hb_tau + qb_x = 0 exactly
        r2 = qt + hx + self.r * (q + qL)     # ... and qb_tau + hb_x = 0
        return causal_loss(r1 ** 2 + r2 ** 2, tau, self.T)

    @torch.no_grad()
    def eval_hb(self, x, tau, chunk=200_000):
        out = []
        for i in range(0, len(x), chunk):
            xx, tt_ = x[i:i + chunk], tau[i:i + chunk]
            h, q, *_ = self.net_part(xx, tt_)
            hL, qL = self.lift(xx, tt_)
            out.append(torch.stack([h + hL, q + qL], 1))
        return torch.cat(out).cpu().numpy() * self.scale


def train_tr(D_, n_iter, seed=0, tag=""):
    set_seeds(seed)
    tr = TRField(D_, seed)
    opt = torch.optim.Adam(tr.net.parameters(), lr=LR_NET)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.time()
    for it in range(n_iter):
        if it % RESAMPLE_EVERY == 0:
            x, tau = sample_uniform(N_COL + N_COL_LEAK, tr.T, gen)   # same point budget as forward net
        opt.zero_grad(set_to_none=True)
        L, wmin = tr.loss(x, tau)
        L.backward()
        opt.step()
        if it % 100 == 0 or it == n_iter - 1:
            hist.append((it, L.item(), wmin))
        if it % 2000 == 0:
            log(f"  [TR {tag}] it {it:6d} L {L.item():.3e} wmin {wmin:.3f} ({time.time() - t0:.0f}s)")
    return tr, np.array(hist), time.time() - t0


def tr_pinn_field_on_grid(tr, D_):
    """PINN TR field on the MOC grid (x rows, tau columns), and hb(x, T-t) on the forward grid."""
    x, tb = D_["x"], D_["tb"]
    X, TB = np.meshgrid(x, tb, indexing="ij")
    hq = tr.eval_hb(tt(X.ravel()), tt(TB.ravel()))
    return hq[:, 0].reshape(X.shape), hq[:, 1].reshape(X.shape)


def tr_pinn_image(tr, D_, xs):
    """B(x) from the PINN TR field (evaluated at T - t_n) and w0 (bilinear from the MOC grid)."""
    tm = D_["tm"]
    T = float(D_["T"])
    X, TT = np.meshgrid(xs, tm, indexing="ij")
    hbr = tr.eval_hb(tt(X.ravel()), tt(np.clip(T - TT.ravel(), 0, T)))[:, 0].reshape(X.shape)
    w0i = Bilinear(D_["x"], tm, D_["w0"])
    with torch.no_grad():
        w0 = w0i(tt(X.ravel()), tt(TT.ravel())).cpu().numpy().reshape(X.shape)
    return tr_image(hbr, w0, tm[1] - tm[0])


# ---------------------------------------------------------------- forward PINN -------
class Forward:
    """(x,t) -> (h,q) with the known leak-free model field as lift (hard IC/BCs):
        h = h0(x,t) + x (t/T) Nh,   q = q0(x,t) + (1-x)(t/T) Nq   (x SCAT_SCALE*Q0*)
    h0, q0 satisfy h(0)=0, q(1)=-Q0* S(t), zero IC and the leak-free equations, so the
    network represents the scattered (leak) field.  PDE residuals of the TOTAL field:
        h_t + q_x + cL_hat h delta_eps(x - chi_hat) = (hs_t + qs_x) + cL_hat (h0 + hs) delta_eps
        q_t + h_x + r q                               = qs_t + hs_x + r qs
    Learnable leak: chi_hat = 0.1 + 0.8 sigmoid(z), cL_hat = softplus(v)."""

    def __init__(self, D_, seed, chi0, cl0, fixed_leak=False):
        self.T, self.r, self.Q0s = float(D_["T"]), float(D_["r"]), float(D_["Q0s"])
        self.eps = float(D_["eps"])
        self.sc = SCAT_SCALE * self.Q0s
        self.net = FFNet(seed).to(DEVICE)
        p = (chi0 - 0.1) / 0.8
        self.z = torch.tensor(math.log(p / (1 - p)), dtype=DTYPE, device=DEVICE, requires_grad=not fixed_leak)
        self.v = torch.tensor(math.log(math.expm1(cl0)), dtype=DTYPE, device=DEVICE, requires_grad=not fixed_leak)
        self.tk = tt(D_["tk"])
        self.d = tt(D_["d"])
        self.dds = tt(D_["dds"])
        self.h0i = Bilinear(D_["x"], D_["tm"], D_["h0"])
        self.q0i = Bilinear(D_["x"], D_["tm"], D_["q0"])
        w0i = Bilinear(D_["x"], D_["tm"], D_["w0"])
        one = torch.ones_like(self.tk)
        self.w01 = w0i(one, self.tk)
        self.h01 = self.h0i(one, self.tk)
        self.var_d = float(np.var(D_["d"]))
        self.var_dds = float(np.var(D_["dds"]))

    def chi(self):
        return 0.1 + 0.8 * torch.sigmoid(self.z)

    def cl(self):
        return torch.nn.functional.softplus(self.v)

    def scat(self, x, t):
        N, Nx, Nt = self.net(x, t)
        T, sc = self.T, self.sc
        Nh, Nq, Nhx, Nqx, Nht, Nqt = N[:, 0], N[:, 1], Nx[:, 0], Nx[:, 1], Nt[:, 0], Nt[:, 1]
        s = t / T
        h = sc * x * s * Nh
        hx = sc * (s * Nh + x * s * Nhx)
        ht = sc * x * (Nh / T + s * Nht)
        q = sc * (1 - x) * s * Nq
        qx = sc * (-s * Nq + (1 - x) * s * Nqx)
        qt = sc * (1 - x) * (Nq / T + s * Nqt)
        return h, q, hx, ht, qx, qt

    def pde_loss(self, x, t):
        h, q, hx, ht, qx, qt = self.scat(x, t)
        e = self.eps
        delta = torch.exp(-0.5 * ((x - self.chi()) / e) ** 2) / (math.sqrt(2 * math.pi) * e)
        r1 = ht + qx + self.cl() * (self.h0i(x, t) + h) * delta
        r2 = qt + hx + self.r * q
        return causal_loss(r1 ** 2 + r2 ** 2, t, self.T)

    def valve(self):
        """Total head at the valve and its time derivative (autodiff for the scattered part)."""
        h, q, hx, ht, qx, qt = self.scat(torch.ones_like(self.tk), self.tk)
        return self.h01 + h, self.w01 + ht

    def data_loss(self):
        h, ht = self.valve()
        return (torch.mean((h - self.d) ** 2) / self.var_d
                + torch.mean((ht - self.w01 - self.dds) ** 2) / self.var_dds)

    @torch.no_grad()
    def eval(self, x, t, chunk=200_000):
        out = []
        for i in range(0, len(x), chunk):
            xx, tt_ = x[i:i + chunk], t[i:i + chunk]
            h, q, *_ = self.scat(xx, tt_)
            out.append(torch.stack([h + self.h0i(xx, tt_), q + self.q0i(xx, tt_)], 1))
        return torch.cat(out).cpu().numpy()


def lam_tr(it, n_iter, n_warm):
    if it < n_warm:
        return 0.0
    end = TR_DECAY_END * n_iter
    if it >= end:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (it - n_warm) / (end - n_warm)))


def train_forward(D_, seed, chi0, cl0, use_tr, B_tab=None, n_iter=None, n_warm=None,
                  fixed_leak=False, warm_state=None, tag="", verbose=False):
    """Train one forward PINN.  B_tab = (xs, B) tabulated PINN-TR image (only for TR variants).
    warm_state: a state dict returned at the end of warm-up (shared between V and T variants:
    during warm-up the leak is frozen, so L_TR has no gradient and the two are identical)."""
    n_iter = N_ITER if n_iter is None else n_iter
    n_warm = N_WARM if n_warm is None else n_warm
    set_seeds(seed)
    fw = Forward(D_, seed, chi0, cl0, fixed_leak)
    opt = torch.optim.Adam(fw.net.parameters(), lr=LR_NET)
    opt_l = None if fixed_leak else torch.optim.Adam([fw.z, fw.v], lr=LR_LEAK)
    Bint = Interp1(B_tab[0][0], B_tab[0][1] - B_tab[0][0], B_tab[1]) if B_tab is not None else None
    gen = torch.Generator().manual_seed(seed)
    hist = []
    norm = None
    it0 = 0
    t_start = time.time()
    if warm_state is not None:
        fw.net.load_state_dict(warm_state["net"])
        opt.load_state_dict(warm_state["opt"])
        gen.set_state(warm_state["gen"])
        hist = list(warm_state["hist"])
        it0 = n_warm
        x, t = warm_state["x"], warm_state["t"]
        xl, tl = warm_state["xl"], warm_state["tl"]
    warm_out = None
    for it in range(it0, n_iter):
        if it == n_warm and warm_state is None:
            warm_out = dict(net=copy.deepcopy(fw.net.state_dict()), opt=copy.deepcopy(opt.state_dict()),
                            gen=gen.get_state(), hist=list(hist), x=x, t=t, xl=xl, tl=tl,
                            time=time.time() - t_start)
        if it % RESAMPLE_EVERY == 0:
            x, t = sample_uniform(N_COL, fw.T, gen)
        if it % RESAMPLE_LEAK_EVERY == 0:
            c = fw.chi().item()
            u, tl = sample_uniform(N_COL_LEAK, fw.T, gen)
            xl = torch.clamp(c + (2 * u - 1) * 5 * fw.eps, 0.0, 1.0)
        if it == n_warm:          # loss normalisers (same rule for every method)
            with torch.no_grad():
                Lp, _ = fw.pde_loss(torch.cat([x, xl]), torch.cat([t, tl]))
                Ld = fw.data_loss()
                Lt = (1 - Bint(fw.chi())).item() if Bint is not None else 1.0
            norm = dict(pde=Lp.item(), data=Ld.item(), tr=max(Lt, 1e-3))
        opt.zero_grad(set_to_none=True)
        if opt_l is not None:
            opt_l.zero_grad(set_to_none=True)
        Lp, wmin = fw.pde_loss(torch.cat([x, xl]), torch.cat([t, tl]))
        Ld = fw.data_loss()
        lt = lam_tr(it, n_iter, n_warm) if (use_tr and Bint is not None) else 0.0
        Lt = 1 - Bint(fw.chi()) if Bint is not None else torch.zeros((), device=DEVICE)
        if it < n_warm:
            Ltot = Lp + Ld
        else:
            Ltot = Lp / norm["pde"] + Ld / norm["data"]
            if lt > 0:
                Ltot = Ltot + lt * Lt / norm["tr"]
        Ltot.backward()
        opt.step()
        if opt_l is not None and it >= n_warm:
            opt_l.step()
        if it % 10 == 0 or it == n_iter - 1:
            hist.append((it, fw.chi().item(), fw.cl().item(), Lp.item(), Ld.item(), Lt.item(), lt, wmin))
        if verbose and it % 1000 == 0:
            log(f"  [{tag}] it {it:6d} Lpde {Lp.item():.3e} Ldata {Ld.item():.3e} LTR {Lt.item():.3f} "
                f"chi {fw.chi().item():.4f} cL {fw.cl().item():.4f} wmin {wmin:.2f} ({time.time() - t_start:.0f}s)")
    return fw, np.array(hist), time.time() - t_start, warm_out


# =====================================================================================
# Conditions, TR-PINN stage (check f), sanity stage (check g)
# =====================================================================================
CONDITIONS = {
    "base": Condition("base"),
    "snr10": Condition("snr10", snr=10.0),
    "snr0": Condition("snr0", snr=0.0),
    "r03": Condition("r03", r=R_STRESS),
    "a095": Condition("a095", a_fac=A_MODEL_FACTORS[0]),
    "a105": Condition("a105", a_fac=A_MODEL_FACTORS[1]),
}
COND_INFO = {"base": ("inf", "phys", 1.0), "snr10": ("10", "phys", 1.0), "snr0": ("0", "phys", 1.0),
             "r03": ("inf", "0.3", 1.0), "a095": ("inf", "phys", 0.95), "a105": ("inf", "phys", 1.05)}
XB = np.linspace(X_IMG[0], X_IMG[1], N_XB)


def moc_image(D_):
    xs = D_["x"]
    m = (xs > X_IMG[0]) & (xs < X_IMG[1])
    return xs[m], tr_image(hb_reversed(D_)[m], D_["w0"][m], D_["tm"][1] - D_["tm"][0])


def tr_job(name):
    """Train the TR-field PINN for one condition; save image table and field error."""
    torch.set_num_threads(1)
    out = os.path.join(CACHE, f"tr_{name}.npz")
    if os.path.exists(out):
        return name
    D_ = build_data(CONDITIONS[name])
    tr, hist, wt = train_tr(D_, N_ITER_TR, seed=0, tag=name)
    hbp, qbp = tr_pinn_field_on_grid(tr, D_)
    err = np.linalg.norm(hbp - D_["hb"]) / np.linalg.norm(D_["hb"])
    errq = np.linalg.norm(qbp - D_["qb"]) / np.linalg.norm(D_["qb"])
    B = tr_pinn_image(tr, D_, XB)
    xm, Bm = moc_image(D_)
    torch.save(tr.net.state_dict(), os.path.join(CACHE, f"trnet_{name}.pt"))
    np.savez(out, xs=XB, B=B, xm=xm, Bm=Bm, err=err, errq=errq, hist=hist, wt=wt,
             hb_pinn=hbp[::2, ::2], hb_moc=D_["hb"][::2, ::2], x=D_["x"][::2], tb=D_["tb"][::2])
    return name


def stage_trpinn(names=None):
    log("=== 4a. TR-field PINN (check f) ===")
    names = list(CONDITIONS) if names is None else names
    for n in names:
        build_data(CONDITIONS[n])
    run_pool(tr_job, names)
    ok_all = True
    for n in names:
        z = np.load(os.path.join(CACHE, f"tr_{n}.npz"))
        xa = float(z["xs"][np.argmax(z["B"])])
        xam = float(z["xm"][np.argmax(z["Bm"])])
        ok = float(z["err"]) < 0.10 and abs(xa - CHI_TRUE) < TOL
        record_check("f" if n == "base" else f"f_{n}", ok,
                     f"[{n}] TR-PINN rel L2 (hb) vs MOC = {100 * float(z['err']):.2f} % (qb {100 * float(z['errq']):.2f} %); "
                     f"PINN image argmax {xa:.4f} (error {abs(xa - CHI_TRUE) / TOL:.2f} lambda/2); "
                     f"MOC image argmax {xam:.4f}; training {float(z['wt']):.0f} s")
        ok_all &= ok if n == "base" else True
    plot_tr(names)
    return CHECKS["f"]["passed"] if "f" in CHECKS else ok_all


def tr_init(name):
    z = np.load(os.path.join(CACHE, f"tr_{name}.npz"))
    return float(z["xs"][np.argmax(z["B"])])


def stage_sanity():
    log("=== 5. Sanity: forward PINN with chi, c_L fixed at truth (check g) ===")
    torch.set_num_threads(os.cpu_count() or 1)
    D_ = build_data(CONDITIONS["base"])
    fw, hist, wt, _ = train_forward(D_, 0, CHI_TRUE, float(D_["cL_true"]), False, n_iter=N_ITER_SANITY,
                                    fixed_leak=True, tag="sanity", verbose=True)
    with torch.no_grad():
        h1, ht1 = [a.cpu().numpy() for a in fw.valve()]
    e_head = np.linalg.norm(h1 - D_["d_clean"]) / np.linalg.norm(D_["d_clean"])
    e_sc = np.linalg.norm(ht1 - fw.w01.cpu().numpy() - D_["dds"]) / np.linalg.norm(D_["dds"])
    nt = len(D_["tm"])
    X, TT = np.meshgrid(D_["x"], D_["t_moc"][:nt], indexing="ij")
    hq = fw.eval(tt(X.ravel()), tt(TT.ravel()))
    Hp, Qp = hq[:, 0].reshape(X.shape), hq[:, 1].reshape(X.shape)
    H, Q = D_["h"][:, :nt], D_["q"][:, :nt]
    e_h = np.linalg.norm(Hp - H) / np.linalg.norm(H)
    e_q = np.linalg.norm(Qp - Q) / np.linalg.norm(Q)
    Hs, Hsp = H - D_["h0"], Hp - D_["h0"]
    e_hs = np.linalg.norm(Hsp - Hs) / np.linalg.norm(Hs)
    ok = e_head < 0.02 and e_sc < 0.20 and e_h < 0.10
    record_check("g", ok, f"valve head rel L2 {100 * e_head:.3f} %; scattered derivative vs dds rel L2 {100 * e_sc:.2f} %; "
                          f"field vs MOC rel L2 h {100 * e_h:.2f} %, q {100 * e_q:.2f} % "
                          f"(scattered part h - h0 alone: {100 * e_hs:.1f} %); {N_ITER_SANITY} its, {wt:.0f} s")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for a_, F, ttl in zip(axs, [Hs * H0, Hsp * H0, (Hsp - Hs) * H0],
                          ["MOC scattered h - h0 [m]", "PINN scattered h - h0 [m]", "PINN - MOC [m]"]):
        im = a_.pcolormesh(TT[0], D_["x"], F, shading="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        a_.set_title(ttl); a_.set_xlabel("t*"); a_.axhline(CHI_TRUE, color="k", ls=":", lw=0.6)
    axs[0].set_ylabel("x/L"); fig.colorbar(im, ax=axs)
    fig.savefig(os.path.join(RESULTS, "fig04_sanity_scattered_field.png"), dpi=120); plt.close(fig)
    return ok


# =====================================================================================
# Sweep (2x2 ablation)
# =====================================================================================
def run_pool(fn, jobs):
    if N_WORKERS <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(N_WORKERS, len(jobs)), maxtasksperchild=1) as pool:
        res = []
        for r in pool.imap_unordered(fn, jobs):
            res.append(r)
            log(f"  done {len(res)}/{len(jobs)}: {r if not isinstance(r, dict) else r.get('tag')}")
        return res


def initial_guesses(ks):
    g = []
    for k in ks:
        for sgn in (-1, 1):
            g.append(("k", sgn * k, float(np.clip(CHI_TRUE + sgn * k * TOL, 0.12, 0.88))))
    rng = np.random.default_rng(777)
    for j in range(N_RANDOM_INIT):
        g.append(("rand", j, float(rng.uniform(0.1, 0.9))))
    return g


def plan_jobs(reduced, cfg):
    ks = [1, 4, 8] if reduced else OFFSETS_K
    seeds_base = SEEDS[:2] if reduced else SEEDS
    jobs = []
    plan = [("base", seeds_base)] + [(n, SEEDS_NOISY) for n in ("snr10", "snr0")] + \
           [("r03", SEEDS_STRESS)] + [(n, [0]) for n in ("a095", "a105")]
    for name, seeds in plan:
        for seed in seeds:
            for kind, k, c0 in initial_guesses(ks):
                jobs.append(dict(cond=name, seed=seed, chi0=c0, kind=kind, k=k, pair="0", **cfg))
            jobs.append(dict(cond=name, seed=seed, chi0=None, kind="tr", k=0, pair="1", **cfg))
    return jobs


def run_tag(job, method):
    c0 = "TR" if job["chi0"] is None else f"{job['chi0']:.4f}"
    return f"{job['cond']}_{method}_s{job['seed']}_{job['kind']}{job['k']}_{c0}_n{job['n_iter']}"


def summarise_run(job, method, fw, hist, wt, D_):
    chi_f, cl_f = fw.chi().item(), fw.cl().item()
    mod = Model(float(D_["a_m"]), r=float(D_["r"]))
    sl = mod.cl_to_sl(cl_f)
    err = abs(chi_f - CHI_TRUE) / TOL
    inside = np.abs(hist[:, 1] - CHI_TRUE) < TOL
    it_succ = None
    if inside[-1]:
        out = np.where(~inside)[0]
        it_succ = int(hist[out[-1] + 1, 0]) if len(out) else int(hist[0, 0])
    snr, rr, af = COND_INFO[job["cond"]]
    return dict(tag=run_tag(job, method), method=method, cond=job["cond"], SNR=snr, r=rr, a_model=af,
                seed=job["seed"], init_kind=job["kind"], k=job["k"], chi_hat0=float(hist[0, 1]),
                chi_hat=chi_f, cL_hat=cl_f, s_L_hat=sl, loc_err_lam2=err, size_err_pct=100 * abs(sl - S_L) / S_L,
                success=bool(err < 1), it_to_success=it_succ, wall_s=wt, n_iter=job["n_iter"])


def pair_job(job):
    """V and T variants with the same initial guess and seed; the warm-up is shared."""
    torch.set_num_threads(1)
    global N_ITER, N_WARM
    N_ITER, N_WARM = job["n_iter"], job["n_warm"]
    methods = ("V0", "T0") if job["pair"] == "0" else ("V1", "T1")
    tags = [run_tag(job, m) for m in methods]
    if all(os.path.exists(os.path.join(RUNS_DIR, t + ".json")) for t in tags):
        return dict(tag=tags[0] + " (cached)")
    D_ = build_data(CONDITIONS[job["cond"]])
    zt = np.load(os.path.join(CACHE, f"tr_{job['cond']}.npz"))
    B_tab = (zt["xs"], zt["B"])
    chi0 = float(zt["xs"][np.argmax(zt["B"])]) if job["chi0"] is None else job["chi0"]
    out = []
    fw, hist, wt, warm = train_forward(D_, job["seed"], chi0, CL_HAT0, use_tr=False, B_tab=B_tab)
    out.append((methods[0], fw, hist, wt))
    fw, hist, wt2, _ = train_forward(D_, job["seed"], chi0, CL_HAT0, use_tr=True, B_tab=B_tab, warm_state=warm)
    out.append((methods[1], fw, hist, wt2 + warm["time"]))
    for m, fw_, h_, w_ in out:
        rec = summarise_run(job, m, fw_, h_, w_, D_)
        np.save(os.path.join(RUNS_DIR, rec["tag"] + "_hist.npy"), h_)
        json.dump(rec, open(os.path.join(RUNS_DIR, rec["tag"] + ".json"), "w"))
    return dict(tag=tags[0])


def stage_smoke():
    """Smoke test (N_ITER = SMOKE_ITER) with N_WORKERS pairs in parallel; returns sec/iteration/worker."""
    log(f"=== smoke test: {N_WORKERS} parallel pair jobs, N_ITER = {SMOKE_ITER} ===")
    cfg = dict(n_iter=SMOKE_ITER, n_warm=SMOKE_ITER // 10)
    jobs = [j for j in plan_jobs(True, cfg) if j["cond"] == "base"][:N_WORKERS]
    t0 = time.time()
    run_pool(pair_job, jobs)
    wall = time.time() - t0
    its = 2 * SMOKE_ITER - SMOKE_ITER // 10           # V + T after the shared warm-up
    spi = wall / its
    recs = [json.load(open(os.path.join(RUNS_DIR, run_tag(j, m) + ".json"))) for j in jobs
            for m in ("V0", "T0")]
    for r in recs:
        log(f"  smoke {r['tag']}: chi {r['chi_hat0']:.3f} -> {r['chi_hat']:.3f}, cL {r['cL_hat']:.4f}")
    est = {}
    for red in (False, True):
        n = len(plan_jobs(red, dict(n_iter=N_ITER, n_warm=N_WARM)))
        est[red] = n * (2 * N_ITER - N_WARM) * spi / min(N_WORKERS, n) / 3600
        log(f"  {'reduced' if red else 'full'} design: {n} pair jobs ({2 * n} runs) -> estimated wall time "
            f"{est[red]:.2f} h ({spi:.3f} s per iteration per batch of {N_WORKERS} workers)")
    json.dump(dict(spi=spi, est_full=est[False], est_reduced=est[True]), open(os.path.join(CACHE, "smoke.json"), "w"))
    return est


def stage_sweep():
    sm = json.load(open(os.path.join(CACHE, "smoke.json")))
    reduced = sm["est_full"] > WALL_BUDGET_H
    log(f"=== 6. sweep: {'REDUCED (2 seeds, k in {1,4,8})' if reduced else 'full'} design; "
        f"estimated {sm['est_reduced' if reduced else 'est_full']:.2f} h ===")
    skip = [n for n in CONDITIONS if not CHECKS.get("f" if n == "base" else f"f_{n}", {}).get("passed", False)]
    if skip:
        log(f"  conditions whose TR-PINN failed check (f): {skip} (run anyway, flagged in REPORT)")
    jobs = plan_jobs(reduced, dict(n_iter=N_ITER, n_warm=N_WARM))
    json.dump(dict(reduced=reduced, n_jobs=len(jobs)), open(os.path.join(CACHE, "sweep_plan.json"), "w"))
    t0 = time.time()
    run_pool(pair_job, jobs)
    log(f"  sweep wall time {(time.time() - t0) / 3600:.2f} h")


# =====================================================================================
# 7. Outputs: figures, summary.csv, REPORT.md
# =====================================================================================
METHOD_COLORS = {"V0": "#1f77b4", "V1": "#17becf", "T0": "#d62728", "T1": "#ff7f0e"}
METHODS = ["V0", "V1", "T0", "T1"]


def plot_tr(names):
    show = [n for n in ("base", "r03") if n in names]
    fig, axs = plt.subplots(len(show), 3, figsize=(14, 3.6 * len(show)), squeeze=False)
    for row, n in zip(axs, show):
        z = np.load(os.path.join(CACHE, f"tr_{n}.npz"))
        vm = np.max(np.abs(z["hb_moc"]))
        for a_, F, ttl in zip(row, [z["hb_moc"], z["hb_pinn"], z["hb_pinn"] - z["hb_moc"]],
                              ["MOC", "PINN", "PINN - MOC"]):
            im = a_.pcolormesh(z["tb"], z["x"], F, shading="auto", cmap="RdBu_r", vmin=-vm, vmax=vm)
            a_.set_title(f"[{n}] TR field hb(x, tau): {ttl}"); a_.set_xlabel("tau* = T* - t*")
            a_.axhline(CHI_TRUE, color="k", ls=":", lw=0.6)
        row[0].set_ylabel("x/L"); fig.colorbar(im, ax=row)
    fig.savefig(os.path.join(RESULTS, "fig03_tr_field_moc_vs_pinn.png"), dpi=120); plt.close(fig)
    zc = np.load(os.path.join(CACHE, "images_moc_base.npz"))
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for a_, n in zip(axs.ravel(), names):
        z = np.load(os.path.join(CACHE, f"tr_{n}.npz"))
        a_.plot(z["xm"], z["Bm"], "k", lw=1.5, label="B(x), MOC TR field")
        a_.plot(z["xs"], z["B"], "C3--", lw=1.2, label="B(x), PINN TR field")
        D_ = build_data(CONDITIONS[n])
        a_.plot(zc["xc"], classical_image(D_, zc["xc"]), "C0", lw=1, label="B_cl(x), classical (1st half-period)")
        a_.axvline(CHI_TRUE, color="g", lw=1); a_.axvspan(CHI_TRUE - TOL, CHI_TRUE + TOL, color="g", alpha=0.12)
        a_.set_title(n); a_.set_ylim(-0.3, 1.05); a_.set_xlabel("x/L")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig05_images.png"), dpi=120); plt.close(fig)


def load_runs():
    recs = []
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(RUNS_DIR, f)))
            if r["n_iter"] == N_ITER:
                recs.append(r)
    return recs


def write_csv(recs):
    cols = ["method", "cond", "SNR", "r", "a_model", "seed", "init_kind", "k", "chi_hat0", "chi_hat", "cL_hat",
            "s_L_hat", "loc_err_lam2", "size_err_pct", "success", "it_to_success", "wall_s"]
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([r[c] if not isinstance(r[c], float) else f"{r[c]:.6g}" for c in cols])


def plot_runs(recs):
    conds = [c for c in CONDITIONS if any(r["cond"] == c for r in recs)]
    # chi_hat vs iteration (key figure)
    fig, axs = plt.subplots(len(conds), 1, figsize=(10, 3.2 * len(conds)), squeeze=False)
    for a_, c in zip(axs[:, 0], conds):
        for r in recs:
            if r["cond"] != c:
                continue
            h = np.load(os.path.join(RUNS_DIR, r["tag"] + "_hist.npy"))
            a_.plot(h[:, 0], h[:, 1], color=METHOD_COLORS[r["method"]], lw=0.8, alpha=0.75)
        a_.axhspan(CHI_TRUE - TOL, CHI_TRUE + TOL, color="g", alpha=0.15)
        a_.axhline(CHI_TRUE, color="g", lw=0.8)
        a_.axvline(N_WARM, color="grey", ls=":", lw=0.8)
        a_.axvline(TR_DECAY_END * N_ITER, color="grey", ls="--", lw=0.8)
        a_.set_ylabel("chi_hat"); a_.set_title(f"condition {c}  (dotted: unfreeze; dashed: lambda_TR = 0)")
        a_.set_ylim(0.08, 0.92)
    for m in METHODS:
        axs[0, 0].plot([], [], color=METHOD_COLORS[m], label=m)
    axs[0, 0].legend(ncol=4, fontsize=8)
    axs[-1, 0].set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig06_chi_vs_iteration.png"), dpi=120); plt.close(fig)
    # success rate vs initial offset k (V0/T0), per condition
    fig, axs = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.4), squeeze=False, sharey=True)
    for a_, c in zip(axs[0], conds):
        ks = sorted({r["k"] for r in recs if r["cond"] == c and r["init_kind"] == "k"})
        for m in ("V0", "T0"):
            sr = [np.mean([r["success"] for r in recs if r["cond"] == c and r["method"] == m
                           and r["init_kind"] == "k" and r["k"] == k]) for k in ks]
            a_.plot(ks, sr, "o-", color=METHOD_COLORS[m], label=m)
        for m in ("V1", "T1"):
            v = [r["success"] for r in recs if r["cond"] == c and r["method"] == m]
            if v:
                a_.axhline(np.mean(v), color=METHOD_COLORS[m], ls="--", label=f"{m} (TR init)")
        a_.set_xlabel("signed initial offset k (x lambda_min/2)"); a_.set_title(c); a_.set_ylim(-0.05, 1.05)
    axs[0, 0].set_ylabel("success rate"); axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig07_success_vs_offset.png"), dpi=120); plt.close(fig)
    # final error histogram
    fig, a_ = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 12, 25)
    for m in METHODS:
        e = [min(r["loc_err_lam2"], 11.9) for r in recs if r["method"] == m]
        a_.hist(e, bins=bins, histtype="step", lw=1.5, color=METHOD_COLORS[m], label=f"{m} (n={len(e)})")
    a_.axvline(1, color="g"); a_.set_xlabel("|chi_hat - chi| / (lambda_min/2)  (all conditions)")
    a_.set_ylabel("runs"); a_.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig08_error_histogram.png"), dpi=120); plt.close(fig)
    # loss curves for a representative V0 / T0 pair (base, seed 0, k = +4)
    rep = [r for r in recs if r["cond"] == "base" and r["seed"] == 0 and r["init_kind"] == "k" and r["k"] == 4]
    if rep:
        fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
        for a_, m in zip(axs, ("V0", "T0")):
            r = [x for x in rep if x["method"] == m][0]
            h = np.load(os.path.join(RUNS_DIR, r["tag"] + "_hist.npy"))
            a_.semilogy(h[:, 0], h[:, 3], label="L_pde (causal)")
            a_.semilogy(h[:, 0], h[:, 4], label="L_data")
            a_.semilogy(h[:, 0], h[:, 5], label="L_TR = 1 - B(chi_hat)")
            a_.semilogy(h[:, 0], np.maximum(h[:, 6], 1e-4), "k:", label="lambda_TR")
            ax2 = a_.twinx(); ax2.plot(h[:, 0], h[:, 1], "g", lw=1); ax2.set_ylim(0.08, 0.92)
            ax2.axhspan(CHI_TRUE - TOL, CHI_TRUE + TOL, color="g", alpha=0.1); ax2.set_ylabel("chi_hat", color="g")
            a_.set_title(f"{m}: {r['tag']}", fontsize=8); a_.set_xlabel("iteration")
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig09_loss_curves.png"), dpi=120); plt.close(fig)


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def write_report(recs):
    load_checks()
    conds = [c for c in CONDITIONS if any(r["cond"] == c for r in recs)]
    plan = json.load(open(os.path.join(CACHE, "sweep_plan.json"))) if os.path.exists(
        os.path.join(CACHE, "sweep_plan.json")) else {}
    sm = json.load(open(os.path.join(CACHE, "smoke.json"))) if os.path.exists(
        os.path.join(CACHE, "smoke.json")) else {}
    cl = json.load(open(os.path.join(CACHE, "classical.json"))) if os.path.exists(
        os.path.join(CACHE, "classical.json")) else {}
    L_ = []
    L_.append("# TR-PINN vs vanilla PINN: leak localisation in a reservoir-pipe-valve system\n")
    L_.append("Numerical demonstration generated by `tr_pinn_leak.py` (all numbers below are produced by the script).\n")
    L_.append("System, linearised leak model and classical time-reversal (TR) defect-detection method follow the TR "
              "chapter of M. Waqar, *PhD thesis*, The Hong Kong University of Science and Technology (2022) "
              "(M. Waqar, M. Louati & M. S. Ghidaoui). Causal residual weighting: S. Wang, S. Sankaran & "
              "P. Perdikaris (2022), *Respecting causality is all you need for training physics-informed neural networks*.\n")
    m = TRUE
    L_.append("## Set-up\n")
    L_.append(f"L = {L} m, D = {D} m, a = {A_WAVE} m/s, f = {F_DW}, H0 = {H0} m, Q0 = {Q0} m^3/s; leak s_L = "
              f"{S_L:.3e} m^2 at chi = {CHI_TRUE} L. H_L0 = {H_L0:.3f} m, Q_L0 = {Q_L0:.3e} m^3/s, c_L = {m.cL:.4f}, "
              f"H_J = {m.Z * Q0:.2f} m, r = {m.r:.4f}, T* = {m.T:.3f}, t0* = {m.t0:.4f}, tc* = {m.tc:.4f}. "
              f"f_max = {F_MAX:.1f} Hz, lambda_min = {LAM_MIN:.2f} m, success threshold lambda_min/2 = "
              f"{LAM_MIN / 2:.2f} m = {TOL:.4f} L. Hardware: {DEVICE.type.upper()}, {N_WORKERS} worker(s).\n")
    L_.append("## Checks\n")
    L_.append("| check | result | numbers |\n|---|---|---|")
    order = ["a", "b", "c", "d", "e", "f"] + [k for k in CHECKS if k.startswith("f_")] + ["g"]
    for k in order:
        if k in CHECKS:
            L_.append(f"| ({k}) | {'PASS' if CHECKS[k]['passed'] else 'FAIL'} | {CHECKS[k]['msg']} |")
    L_.append("")
    L_.append("## Deviations from the specification (and why)\n")
    for dv in DEVIATIONS:
        L_.append(f"- {dv}")
    if sm:
        L_.append(f"- Smoke test ({SMOKE_ITER} its): {sm['spi']:.3f} s per iteration per batch of {N_WORKERS} "
                  f"parallel workers; estimated full design {sm['est_full']:.2f} h, reduced design "
                  f"{sm['est_reduced']:.2f} h. " + ("Full design > 3 h, so the sweep used **2 seeds and offsets "
                  "k in {1, 4, 8}** (as instructed)." if plan.get("reduced") else "Full design used."))
    L_.append("")
    L_.append("## Results\n")
    L_.append("Success = |chi_hat - chi| < lambda_min/2. Location error in units of lambda_min/2; size error = "
              "|s_L_hat - s_L| / s_L. Iterations-to-success = first logged iteration after which chi_hat stays "
              "inside the band (successful runs only).\n")
    L_.append("| condition | method | runs | success rate | median loc. err (lambda/2) | median size err (%) | "
              "median its-to-success |\n|---|---|---|---|---|---|---|")
    for c in conds:
        for mth in METHODS:
            rr = [r for r in recs if r["cond"] == c and r["method"] == mth]
            if not rr:
                continue
            sr = np.mean([r["success"] for r in rr])
            L_.append(f"| {c} | {mth} | {len(rr)} | {100 * sr:.0f} % ({sum(r['success'] for r in rr)}/{len(rr)}) | "
                      f"{med([r['loc_err_lam2'] for r in rr]):.2f} | {med([r['size_err_pct'] for r in rr]):.1f} | "
                      f"{med([r['it_to_success'] for r in rr]):.0f} |")
    L_.append("")
    L_.append("Success rate of V0 / T0 by signed initial offset k (base condition, all seeds):\n")
    ks = sorted({r["k"] for r in recs if r["cond"] == "base" and r["init_kind"] == "k"})
    L_.append("| k | " + " | ".join(str(k) for k in ks) + " | random inits |")
    L_.append("|---|" + "---|" * (len(ks) + 1))
    for mth in ("V0", "T0"):
        row = []
        for k in ks:
            v = [r["success"] for r in recs if r["cond"] == "base" and r["method"] == mth and r["init_kind"] == "k"
                 and r["k"] == k]
            row.append(f"{sum(v)}/{len(v)}")
        v = [r["success"] for r in recs if r["cond"] == "base" and r["method"] == mth and r["init_kind"] == "rand"]
        L_.append(f"| {mth} | " + " | ".join(row) + f" | {sum(v)}/{len(v)} |")
    L_.append("")
    L_.append("TR-PINN image argmax (= V1/T1 initial guess) and classical matched-filter argmax per condition:\n")
    L_.append("| condition | PINN-TR argmax | MOC-TR argmax | classical B_cl argmax |\n|---|---|---|---|")
    zc = np.load(os.path.join(CACHE, "images_moc_base.npz"))
    for c in conds:
        z = np.load(os.path.join(CACHE, f"tr_{c}.npz"))
        bcl = cl.get(c)
        if bcl is None:
            bcl = float(zc["xc"][np.argmax(classical_image(build_data(CONDITIONS[c]), zc["xc"]))])
        L_.append(f"| {c} | {float(z['xs'][np.argmax(z['B'])]):.4f} | {float(z['xm'][np.argmax(z['Bm'])]):.4f} | "
                  f"{bcl:.4f} |")
    L_.append(f"\n(true chi = {CHI_TRUE}; band = +-{TOL:.4f})\n")
    L_.append("## Interpretation\n")
    ip = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interpretation.md")
    L_.append(open(ip).read() if os.path.exists(ip) else "_(to be written after inspecting the results)_")
    L_.append("\n## Figures\n")
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith(".png"):
            L_.append(f"- `{f}`")
    open(os.path.join(RESULTS, "REPORT.md"), "w").write("\n".join(L_) + "\n")
    log("REPORT.md written")


def stage_report():
    recs = load_runs()
    write_csv(recs)
    plot_runs(recs)
    write_report(recs)


DEVIATIONS = [
    "**Compute (no GPU; 4 CPU cores).** The spec settings (4x128 net, 20 000 + 2 000 collocation points, 20 000 "
    "Adam iterations) cost ~0.6 s/iteration on this machine (~3.4 h per run, >100 runs). Every forward PINN "
    "therefore uses width 64 (4 hidden layers), 4 000 + 1 000 collocation points and 4 000 Adam iterations "
    "(warm-up 400 = the same 10 %; lambda_TR decays to 0 at 70 %). All four methods use exactly the same "
    "settings; no L-BFGS for any method. Check (g) passes with these settings.",
    "**Output transforms (hard IC/BCs).** With the spec's example lifts (q = -Q0* S(t) x + ..., qb = g(tau) x + ...) "
    "the boundary data are spread uniformly in x and the networks must cancel a pulse that is sharp in t "
    "everywhere in the pipe. Within the budget this failed: TR-PINN rel. L2 error 54-99 % "
    "(6 trials, Gaussian/characteristic Fourier features, causal eps 0-10) and the forward PINN at the true "
    "leak reached a field error of 46 % (check g failed). The lifts were replaced by physically consistent ones: "
    "forward net h = h0 + x(t/T)Nh, q = q0 + (1-x)(t/T)Nq with (h0, q0) the known leak-free MOC field (the method "
    "already assumes this model, cf. w0), so the network represents the scattered field; TR net = closed-form "
    "undamped method-of-images solution of the TR problem + x(tau/T)Nh, (1-x)(tau/T)Nq. Both satisfy the BCs/IC "
    "exactly; the PDE residuals are those of the total field. Consequence: for r ~ 0.004 the TR-PINN only "
    "learns a <1 % friction correction (check f is then easy); the r = 0.3 stress case is the real TR-PINN test.",
    "**Fourier features** are aligned with the characteristic directions (features sin/cos(2 pi k (t -+ x)), "
    "k ~ N(0, 2^2)); same embedding for every net.",
    "**Collocation around chi_hat** (1 000 points within +-5 eps) is resampled every 100 iterations (uniform "
    "points every 500) so that it follows the moving estimate; identical for all methods.",
    "**L_TR** uses B(x) tabulated once on 801 points in (0.1, 0.9) from the frozen TR-PINN field and w0 (linear "
    "interpolation in chi_hat); this is the same function as evaluating B(chi_hat) on the fly. Its normaliser "
    "(value at the end of warm-up) is floored at 1e-3.",
    "**TR re-emission taper.** The re-emitted signal qb(1,tau) = -dds(T*-tau) is multiplied by a raised-cosine "
    "onset of width tc* (MOC and PINN alike) so that the zero IC is consistent with the valve BC; for clean data "
    "dds(T*) is 0.5 % of its peak, but with noise the spectral derivative has an end-of-record edge artefact.",
    "**Spectral derivative**: the straight line through the end points is removed before the FFT (and its slope "
    "added back) so that the periodic extension is continuous; the same filter (zero above 1.5 f_max) everywhere.",
    "**Check (b) amplitude.** The spec's expected echo amplitude -c_L/(2+c_L) H_J is the reflection coefficient of "
    "the travelling wave; at the closed valve (flow prescribed) the returning wave is doubled, so the sensor sees "
    "2 x (-c_L/(2+c_L)) H_J = 6.4 % of H_J (not 3 %). MOC agrees with the doubled value; the check uses it.",
    "**Loss weights** during warm-up: L_pde + L_data (raw); afterwards each term divided by its value at the end "
    "of warm-up (same rule for all methods).",
    "**Iterations-to-success** is the first logged iteration (every 10) after which chi_hat stays in the band.",
    "LEAK_MODEL = 'nonlinear' is implemented in the MOC but was not run (time).",
]


# =====================================================================================
# main
# =====================================================================================
def main(stage):
    load_checks()
    if stage in ("moc", "all"):
        stage_moc()
        if not all(CHECKS[k]["passed"] for k in "abcde"):
            log("STOP: a stage 1-2 check failed"); return
    if stage in ("trpinn", "all"):
        if not stage_trpinn():
            log("STOP: check (f) failed"); return
    if stage in ("sanity", "all"):
        if not stage_sanity():
            log("STOP: check (g) failed"); return
    if stage in ("smoke", "all"):
        stage_smoke()
    if stage in ("sweep", "all"):
        stage_sweep()
    if stage in ("report", "all"):
        stage_report()
    with open(os.path.join(RESULTS, f"log_{stage}.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
