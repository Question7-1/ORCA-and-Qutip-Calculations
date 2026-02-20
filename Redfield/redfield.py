import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import expm as matrix_expm

print("=" * 70)
print("=" * 70)

# ============================================================================
# Constants
# ============================================================================
hbar   = constants.hbar
k_B    = constants.k
c_SI   = constants.c
cm2Hz  = c_SI * 100        # cm^-1 -> Hz (linear)
MHz    = 1e6
GHz    = 1e9
TWO_PI = 2 * np.pi

# ============================================================================
# Physical parameters (from my ORCA calculations)
# ============================================================================
T_bath = 4.2          # K
I      = 5 / 2
dim    = int(2 * I + 1)
T1_exp = 41.39        # s (experimental)

e2qQ_h_MHz = -1858.043213
eta_Q      = 0.940691
A_iso_MHz  = -9.9235

huang_rhys_modes = [
    (23.02, 0.111), (37.61, 0.026), (38.25, 0.025), (41.41, 0.050),
    (53.00, 0.018), (57.20, 0.032), (76.86, 0.013), (80.03, 0.082),
    (95.53, 0.029), (103.09, 0.018), (129.74, 0.110), (140.52, 0.013),
    (168.89, 0.031), (173.45, 0.021), (176.65, 0.021), (192.62, 0.017),
    (196.48, 0.013), (203.15, 0.016), (212.25, 0.029), (257.19, 0.076),
    (298.48, 0.054), (306.59, 0.092), (332.02, 0.409), (406.01, 0.026),
    (498.91, 0.025), (597.70, 0.045), (640.12, 0.014), (679.79, 0.144),
    (703.94, 0.155), (705.43, 0.112), (709.81, 0.044), (710.66, 0.118),
    (794.41, 0.025), (899.14, 0.028), (957.94, 0.016), (1006.58, 0.090),
    (1007.41, 0.111), (1015.95, 0.016), (1096.71, 0.024), (1131.89, 0.021),
]

# Pre-compute mode quantities (rad/s)
omega_modes = np.array([om_cm * cm2Hz * TWO_PI for om_cm, _ in huang_rhys_modes])
S_vals      = np.array([S for _, S in huang_rhys_modes])

# Bose-Einstein occupancies
x_modes = hbar * omega_modes / (k_B * T_bath)
n_modes = np.where(x_modes < 50, 1.0 / np.expm1(x_modes), 0.0)

# ============================================================================
# Hamiltonian
# ============================================================================
Ix = qt.jmat(I, "x")
Iy = qt.jmat(I, "y")
Iz = qt.jmat(I, "z")
Ip = qt.jmat(I, "+")
Im = qt.jmat(I, "-")
Id = qt.qeye(dim)

pf_Q  = (TWO_PI * e2qQ_h_MHz * MHz) / (4 * I * (2 * I - 1))
H_NQC = pf_Q * (3*Iz*Iz - I*(I+1)*Id + eta_Q*(Ip*Ip + Im*Im))
H_HFC = TWO_PI * A_iso_MHz * MHz * Iz
H_0   = H_NQC + H_HFC

print(f"\nH_NQC: e2qQ/h = {e2qQ_h_MHz:.3f} MHz, eta_Q = {eta_Q:.4f}")
print(f"H_HFC: A_iso  = {A_iso_MHz:.4f} MHz")

evals, evecs = H_0.eigenstates()
evals = np.real(evals)

print(f"\nEnergy levels / 2pi (MHz):")
for i, E in enumerate(evals):
    print(f"  |{i}>  E = {E/(TWO_PI*MHz):+.4f} MHz")

dE_45 = abs(evals[5] - evals[4]) / (TWO_PI * MHz)

# Thermal populations
beta      = 1.0 / (k_B * T_bath)
boltzmann = np.exp(-beta * hbar * evals)
boltzmann /= boltzmann.sum()
print(f"\nThermal populations: {np.round(boltzmann, 6)}")
print(f"  (Near-equal: hbar*omega_max / kT = "
      f"{hbar*np.max(np.abs(evals))/(k_B*T_bath)*1000:.2f} milli-units -> classical limit)")

# ============================================================================
# Spectral density
# ============================================================================
def make_spectral_density(gamma_ph):
    """
    J(omega) = sum_k S_k * omega_k * [(n_k+1)*L(omega-omega_k) + n_k*L(omega+omega_k)]
    L(f; fk, gk) = (gk/2pi) / ((f - fk)^2 + (gk/2)^2)
    gamma_ph: shared Lorentzian FWHM in rad/s.
    """
    gk_hz = gamma_ph / TWO_PI  # shared linewidth in Hz

    def sd(omega):
        f     = omega / TWO_PI
        fk    = omega_modes / TWO_PI
        L_em  = gk_hz / ((f - fk)**2 + (gk_hz/2)**2)
        L_abs = gk_hz / ((f + fk)**2 + (gk_hz/2)**2)
        return float(max(np.sum(S_vals * omega_modes * ((n_modes+1)*L_em + n_modes*L_abs)), 0.0))

    return sd

# ============================================================================
# Coupling operators
# ============================================================================
V_axial   = 3*Iz*Iz - I*(I+1)*Id
V_rhombic = Ip*Ip + Im*Im

def make_a_ops(gamma_ph):
    sd = make_spectral_density(gamma_ph)
    return [[V_rhombic, sd], [Ix, sd], [V_axial, sd]]

# ============================================================================
# KMS check at a given gamma_ph
# ============================================================================
def check_kms(gamma_ph, label=""):
    sd = make_spectral_density(gamma_ph)
    print(f"\nKMS check [{label}]:")
    print(f"  {'freq (MHz)':>12}  {'J_emit':>12}  {'J_abs':>12}  "
          f"{'ratio J_abs/J_emit':>18}  {'exp(-hbar*om/kT)':>16}  {'OK?':>5}")
    for f_MHz in [42, 476, 500, 774, 1243]:
        om  = TWO_PI * f_MHz * MHz
        je  = sd(+om)
        ja  = sd(-om)
        ratio    = ja/je if je > 0 else 0
        expected = np.exp(-hbar * om / (k_B * T_bath))
        ok = "YES" if abs(ratio - expected) < 0.01 else "WARN"
        print(f"  {f_MHz:>12}  {je:>12.4e}  {ja:>12.4e}  {ratio:>18.6f}  {expected:>16.6f}  {ok:>5}")

# ============================================================================
# Core function: build R, extract T1 from amplitude-weighted eigenmode
# ============================================================================
pop_idx = [i*dim + i for i in range(dim)]

def compute_T1(gamma_ph, verbose=False):
    """Returns T1 = 1 / |lambda_slowest_nonzero|."""
    sec_cutoff = TWO_PI * 1e6
    R, ekets   = qt.bloch_redfield_tensor(H_0, make_a_ops(gamma_ph), sec_cutoff=sec_cutoff)
    R_mat      = R.full()
    R_pop      = np.real(R_mat[np.ix_(pop_idx, pop_idx)])

    # Diagonalise
    lams, vecs = np.linalg.eig(R_pop)
    lams = np.real(lams)

    # Zero-eigenvalue threshold (stationary state)
    max_rate = np.max(np.abs(lams))
    thresh   = max(max_rate * 1e-6, 1e-30)
    is_zero  = np.abs(lams) < thresh

    nonzero_lams = lams[~is_zero]
    T1 = 1.0 / np.min(np.abs(nonzero_lams)) if len(nonzero_lams) > 0 else np.inf

    # Amplitude decomposition for initial state |5> (informational)
    p0 = np.zeros(dim); p0[-1] = 1.0
    try:
        c = np.linalg.solve(vecs, p0)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(vecs, p0, rcond=None)[0]
    amplitudes = np.real(c * vecs[-1, :])

    if verbose:
        print(f"\nEigenmode decomposition of |5> population:")
        print(f"  {'k':>3}  {'lambda (rad/s)':>16}  {'tau (s)':>12}  "
              f"{'amp in |5>':>12}  {'note':>10}")
        order = np.argsort(np.abs(lams))
        for k in order:
            tau_s  = f"{1/abs(lams[k]):.4e}" if abs(lams[k]) > thresh else "inf"
            note   = "<-- T1 (slowest)" if (abs(lams[k]) > thresh and
                      abs(lams[k]) == np.min(np.abs(lams[np.abs(lams)>thresh]))) else ""
            print(f"  {k:>3}  {lams[k]:>16.4e}  {tau_s:>12}  "
                  f"{amplitudes[k]:>12.4e}  {note}")
        print(f"\n  T1 (slowest eigenvalue) = {T1:.4e} s")

    return T1, R_pop, lams, amplitudes, is_zero, vecs

# ============================================================================
# Main run at gamma_ph = 1 GHz
# ============================================================================
gamma_ph_main = TWO_PI * 1.0 * GHz
print(f"\n{'='*70}")
print(f"Main run: gamma_ph/2pi = {gamma_ph_main/TWO_PI/GHz:.1f} GHz")
print(f"{'='*70}")

check_kms(gamma_ph_main, label="gamma_ph = 1 GHz")

print("\nBuilding Bloch-Redfield tensor ...")
T1_main, R_pop_main, lams_main, amps_main, iz_main, vecs_main = \
    compute_T1(gamma_ph_main, verbose=True)

print(f"\n{'='*50}")
print(f"  T1 (slowest eigenvalue)  = {T1_main:.4e} s")
print(f"  T1 (experimental)        = {T1_exp:.2f} s")
print(f"  Ratio                    = {T1_main/T1_exp:.3f}x")
print(f"{'='*50}")

# Transition rate matrix W — extract from R_pop_main diagonal structure
# R_pop[i,i] = -sum_{j!=i} W[j,i], R_pop[i,j] = W[i,j] for i!=j
W = np.zeros((dim, dim))
for i in range(dim):
    for j in range(dim):
        if i != j:
            W[i,j] = float(max(R_pop_main[i, j], 0.0))

print(f"\nTransition rate matrix W_ij (s^-1):")
print(np.array2string(W, formatter={"float_kind": lambda x: f"{x:9.3e}"}))

# ============================================================================
# gamma_ph SCAN — 1 to 100 GHz (actually executed)
# ============================================================================
print(f"\n{'='*70}")
print(f"gamma_ph scan: 1 to 100 GHz")
print(f"{'='*70}")

gamma_scan_GHz = np.logspace(0, 2, 20)  # 1 to 100 GHz, 20 points
T1_scan = []

for g_GHz in gamma_scan_GHz:
    gph = TWO_PI * g_GHz * GHz
    t1, _, _, _, _, _ = compute_T1(gph, verbose=False)
    T1_scan.append(t1)
    ratio = t1 / T1_exp
    print(f"  gamma_ph/2pi = {g_GHz:6.1f} GHz  ->  T1 = {t1:.3e} s  "
          f"(ratio = {ratio:.3f}x)")

T1_scan = np.array(T1_scan)

# Find crossing with T1_exp
cross_idx = np.where(np.diff(np.sign(T1_scan - T1_exp)))[0]
if len(cross_idx) > 0:
    ci = cross_idx[0]
    g_lo, g_hi = gamma_scan_GHz[ci], gamma_scan_GHz[ci+1]
    t_lo, t_hi = T1_scan[ci], T1_scan[ci+1]
    # log-log interpolation
    g_cross = np.exp(np.interp(np.log(T1_exp),
                               sorted([np.log(t_lo), np.log(t_hi)]),
                               sorted([np.log(g_lo), np.log(g_hi)], reverse=(t_lo < t_hi))))
    print(f"\n  T1 crosses experimental at gamma_ph/2pi ~ {g_cross:.2f} GHz")
else:
    g_cross = None
    print(f"\n  No crossing in 1–100 GHz.")
    print(f"  T1 ~ 1/gamma_ph throughout (far-tail Lorentzian regime).")
    print(f"  Extrapolating: crossing expected at gamma_ph/2pi ~ "
          f"{gamma_scan_GHz[0] * T1_scan[0] / T1_exp:.2f} GHz")

# ============================================================================
# Population dynamics
# ============================================================================
print(f"Printing population dynamics ...")
p0    = np.zeros(dim); p0[-1] = 1.0
t_end = 5.0 * T1_main          # integrate to 5 * T1 so slow tail is visible
tlist = np.linspace(0, t_end, 600)

populations = np.zeros((dim, len(tlist)))
for k, t in enumerate(tlist):
    populations[:, k] = np.real(matrix_expm(R_pop_main * t) @ p0)

# Reconstruct slow-mode overlay for |5> analytically
thresh_main = max(np.max(np.abs(lams_main)) * 1e-6, 1e-30)
try:
    c_main = np.linalg.solve(vecs_main, p0)
except np.linalg.LinAlgError:
    c_main = np.linalg.lstsq(vecs_main, p0, rcond=None)[0]

nonzero_mask = np.abs(lams_main) > thresh_main
idx_slowest  = np.where(nonzero_mask)[0][np.argmin(np.abs(lams_main[nonzero_mask]))]
lam_slow     = lams_main[idx_slowest]
stat_idx     = np.argmin(np.abs(lams_main))
p5_stat      = float(np.real(c_main[stat_idx] * vecs_main[-1, stat_idx]))
A_slow       = float(np.real(c_main[idx_slowest] * vecs_main[-1, idx_slowest]))

t_fine  = np.linspace(0, t_end, 1000)
p5_slow = p5_stat + A_slow * np.exp(lam_slow * t_fine)

# ============================================================================
# PLOTS
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

# --- Panel 1: Energy levels ---
ax    = axes[0]
E_MHz = (evals - evals[0]) / (TWO_PI*MHz)
for i, E in enumerate(E_MHz):
    ax.hlines(E, 0.1, 0.9, colors="steelblue", linewidth=3)
    ax.text(0.93, E, f"|{i}>  {E:+.1f} MHz", fontsize=9, va="center")
ax.annotate("Near-degenerate\n(eta=0.94)", xy=(0.5, E_MHz[4]),
            xytext=(1.6, E_MHz[4]-60), fontsize=7.5, color="tomato",
            arrowprops=dict(arrowstyle="->", color="tomato", lw=0.8))
ax.set_xlim(0, 2.6); ax.set_ylim(-40, E_MHz[-1]+70)
ax.set_ylabel("Energy / 2pi (MHz)", fontsize=11)
ax.set_title("NQC + HFC Energy Levels\nB=0, T=4.2K", fontsize=11, fontweight="bold")
ax.set_xticks([]); ax.grid(True, alpha=0.3, axis="y")

# --- Panel 2: J(omega) ---
ax      = axes[1]
sd_main = make_spectral_density(gamma_ph_main)
f_plot  = np.logspace(6, 13, 600)
J_plot  = np.array([sd_main(TWO_PI*f) for f in f_plot])
ax.loglog(f_plot/GHz, J_plot, color="steelblue", lw=2, label="J(ω)")
labeled = False
for i, E_i in enumerate(evals):
    for j, E_j in enumerate(evals):
        if i < j:
            f_trans = abs(E_j - E_i) / TWO_PI
            if 1e7 < f_trans < 2e9:
                ax.axvline(f_trans/GHz, color="tomato", lw=0.8, alpha=0.7,
                           label="Nuclear transitions" if not labeled else "")
                labeled = True
for om, S_k in zip(omega_modes, S_vals):
    ax.axvline(om/TWO_PI/GHz, color="gray",
               lw=0.3 + 0.8*S_k/S_vals.max(), alpha=0.4, ls="--")
ax.axvspan(0.001, 2, alpha=0.07, color="tomato")
ax.set_xlabel("Frequency (GHz)", fontsize=11)
ax.set_ylabel("J(ω)", fontsize=11)
ax.set_title(f"Spectral Density\nγ_ph/2π = {gamma_ph_main/TWO_PI/GHz:.0f} GHz",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

# --- Panel 3: gamma_ph scan ---
ax = axes[2]
ax.loglog(gamma_scan_GHz, T1_scan, "o-", color="steelblue", lw=2,
          label="T1 (slowest eigenvalue)")
ax.axhline(T1_exp, color="tomato", lw=2, ls="-", label=f"T1 exp = {T1_exp} s")
if g_cross is not None:
    ax.axvline(g_cross, color="green", lw=1.5, ls=":",
               label=f"Crossing ~ {g_cross:.1f} GHz")
ax.set_xlabel("γ_ph / 2π (GHz)", fontsize=11)
ax.set_ylabel("T1 (s)", fontsize=11)
ax.set_title("T1 vs γ_ph Scan\n1–100 GHz", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

# --- Panel 4: Population dynamics ---
ax   = axes[3]
cmap = plt.cm.plasma(np.linspace(0.1, 0.9, dim))
for i in range(dim):
    ax.plot(tlist, populations[i], color=cmap[i], lw=1.8, label=f"|{i}>")
    ax.axhline(boltzmann[i], color=cmap[i], ls=":", lw=0.8, alpha=0.4)
# Overlay: slowest eigenmode for |5> (this IS T1)
ax.plot(t_fine, p5_slow, "k--", lw=2, label=f"T1 slow mode\n({T1_main:.1f} s)")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Population", fontsize=11)
ax.set_title(f"Population Relaxation\n"
             f"γ_ph/2π = {gamma_ph_main/TWO_PI/GHz:.0f} GHz, T1 = {T1_main:.2e} s",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=7.5, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"  gamma_ph/2pi (main run)  = {gamma_ph_main/TWO_PI/GHz:.1f} GHz")
print(f"  T1 (slowest eigenvalue)  = {T1_main:.4e} s")
print(f"  T1 (experimental)        = {T1_exp:.2f} s")
print(f"  Ratio                    = {T1_main/T1_exp:.3f}x")
if g_cross is not None:
    print(f"  Best-fit gamma_ph/2pi    ~ {g_cross:.2f} GHz")
else:
    extrap = gamma_scan_GHz[0] * T1_scan[0] / T1_exp
    print(f"  Extrapolated crossing    ~ {extrap:.2f} GHz (below scan range)")
print("="*70)
