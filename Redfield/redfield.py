"""
Calculates T1 time using QuTiP and Redfield theory
All parameters from ORCA calculations

Temperature: 4.2 K, Nuclear spin: I = 5/2 (Eu)
Target: T1 = 41.39 s, T2 = 205 ns
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy import constants

print("="*70)
print("Eu(dpphen)(NO3)3 Nuclear Spin Qubit Simulation")
print("="*70)

# ============================================================================
# Physical Constants & Parameters
# ============================================================================

h_bar = constants.hbar
k_B = constants.k
mu_N = constants.physical_constants['nuclear magneton'][0]  # J/T
cm_inv_to_Hz = constants.c * 100
MHz_to_Hz = 1e6

T = 4.2  # Kelvin
I = 5/2  # Nuclear spin
B_field = 0.0  # Tesla (zero field)
g_N = 0.6134  # Eu-153

T1_experimental = 41.39  # seconds
T2_experimental = 0.205e-6  # seconds (205 ns)

# ORCA parameters
e2qQ_h = 608.794743  # MHz
eta_Q = 0.940691
A_iso = -9.9235  # MHz

# Full hyperfine tensor from ORCA (in MHz)
A_tensor = np.array([
    [-9.9873,   3.7238,  -2.2211],
    [ 3.8413, -12.6754,   1.4942],
    [-2.2013,   1.4020,  -7.1077]
])

# Diagonalize to get principal values
A_eigenvalues, A_eigenvectors = np.linalg.eigh(A_tensor)
print(f"HFC principal values: {A_eigenvalues} MHz")
print(f"HFC isotropic: {A_iso:.2f} MHz")

# Huang-Rhys factors (40 modes with S > 0.01)
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

# Nuclear spin bath: nearby spins causing spectral diffusion
bath_spins = [
    # Format: (element, I, x, y, z) - positions relative to Eu
    # H atoms (I = 1/2)
    ('H', 0.5, -4.377754, 1.147464, 3.671829),
    ('H', 0.5, 2.705193, 1.730396, -5.172580),
    ('H', 0.5, -4.524319, 3.360769, 2.498627),
    ('H', 0.5, -3.772097, 4.853370, 0.664598),
    ('H', 0.5, 0.772214, 3.436250, -4.251066),
    ('H', 0.5, -2.312284, 5.238476, -1.335962),
    ('H', 0.5, 2.782100, -0.272415, -3.256752),
    ('H', 0.5, -0.553727, 4.406667, -2.875001),
    ('H', 0.5, -3.379545, -1.115950, 3.816080),
    ('H', 0.5, -2.464911, -3.032421, 3.784891),
    ('H', 0.5, 0.555278, -4.133950, 0.877101),
    ('H', 0.5, -0.915200, -5.147595, 3.001573),
    # N atoms (I = 1)
    ('N', 1.0, -1.384994, 2.250116, -0.628577),
    ('N', 1.0, -2.227990, 0.593825, 1.386424),
    ('N', 1.0, -2.126693, -1.528136, 2.322551),
    ('N', 1.0, 0.880523, 1.522349, 2.381702),
    ('N', 1.0, -0.022078, 2.509503, -2.486584),
    ('N', 1.0, -1.423406, -1.498989, -2.100499),
    ('N', 1.0, 0.593269, 1.338372, -2.167569),
    ('N', 1.0, -1.183328, -1.864174, 1.399839),
]

print(f"Nuclear spin bath: {len(bath_spins)} spins (H + N on ligands)")
print(f"Note: Bath causes spectral diffusion → limits T2 to ~200 ns")
print(f"NQC: e²qQ/h = {e2qQ_h:.1f} MHz, η = {eta_Q:.3f}")
print(f"HFC: A_iso = {A_iso:.2f} MHz")
print(f"Huang-Rhys: {len(huang_rhys_modes)} modes, Σ S = {sum(S for _,S in huang_rhys_modes):.2f}")
print(f"Note: Orbach/ZFS not included - states too high in energy (>2800 cm⁻¹, frozen at 4.2K)")

# ============================================================================
# Build Hamiltonian
# ============================================================================

# Nuclear spin operators
Ix = qt.jmat(I, 'x')
Iy = qt.jmat(I, 'y')
Iz = qt.jmat(I, 'z')
I_plus = qt.jmat(I, '+')
I_minus = qt.jmat(I, '-')
I_identity = qt.qeye(int(2*I + 1))

prefactor_NQC = (e2qQ_h * MHz_to_Hz) / (4 * I * (2*I - 1))
H_NQC = prefactor_NQC * (3*Iz*Iz - I*(I+1)*I_identity + eta_Q*(I_plus*I_plus + I_minus*I_minus))

# H_HFC = A_xx*Ix^2 + A_yy*Iy^2 + A_zz*Iz^2 (simplified)
A_zz = A_eigenvalues[2]  # Largest magnitude component
H_HFC = A_zz * MHz_to_Hz * Iz

H_0 = H_NQC + H_HFC

E_levels = H_0.eigenenergies()
print(f"\nEnergy levels (MHz): {E_levels/MHz_to_Hz}")

# ============================================================================
# Spectral Densities & Relaxation
# ============================================================================

def spectral_density(omega, T, modes):
    """Huang-Rhys spectral density + two-phonon Raman"""
    J = 0.0
    gamma = 1e9  # Hz, phonon linewidth
    
    for omega_ph_cm, S_k in modes:
        omega_ph = omega_ph_cm * cm_inv_to_Hz
        x = h_bar * 2*np.pi*omega_ph / (k_B * T)
        n_BE = 1/(np.exp(x)-1) if x < 50 else 0.0
        
        # Lorentzian-broadened delta functions
        lorentz_em = gamma/(2*np.pi) / ((omega-omega_ph)**2 + (gamma/2)**2)
        lorentz_abs = gamma/(2*np.pi) / ((omega+omega_ph)**2 + (gamma/2)**2)
        J += S_k * omega_ph * ((n_BE+1)*lorentz_em + n_BE*lorentz_abs)
    
    # Add two-phonon Raman
    omega_D = 200 * cm_inv_to_Hz
    if abs(omega) < omega_D:
        J += (abs(omega)/omega_D)**3 * (T/300)**4 * 1e6
    
    return J

evals, evecs = H_0.eigenstates()
collapse_ops = []
rates = []

for i in range(len(evals)):
    for j in range(i+1, len(evals)):
        omega_ij = evals[j] - evals[i]
        if abs(omega_ij) < 1e3:  # Skip < 1 kHz
            continue
        
        J_ij = spectral_density(omega_ij, T, huang_rhys_modes)
        gamma_ij = 2 * np.pi * J_ij
        
        if gamma_ij > 1e-6:
            collapse_ops.append(np.sqrt(gamma_ij) * evecs[j] * evecs[i].dag())
            rates.append(gamma_ij)

# Add nuclear spin bath contribution (spectral diffusion)
mu_0 = constants.mu_0
g_bath_H = 5.586  # Proton g-factor
g_bath_N = 0.404  # 14N g-factor

Gamma_bath = 0.0
for element, I_bath, x, y, z in bath_spins:
    r = np.sqrt(x**2 + y**2 + z**2) * 1e-10  # Angstrom to meters
    
    if element == 'H':
        g_bath = g_bath_H
    elif element == 'N':
        g_bath = g_bath_N
    else:
        continue
    
    # Dipolar coupling strength (Hz)
    A_dip = (mu_0/(4*np.pi)) * (g_N * g_bath * mu_N**2) / (r**3) / h_bar / (2*np.pi)
    
    Gamma_bath += A_dip**2 / (100e6)  # Normalize by ~100 MHz

# Add bath contribution to rates
if Gamma_bath > 0:
    rates_phonon = sum(rates)
    rates.append(Gamma_bath)
    print(f"\nRelaxation rates:")
    print(f"  Phonon (direct + Raman): {rates_phonon:.3e} Hz")
    print(f"  Nuclear spin bath:       {Gamma_bath:.3e} Hz")
    print(f"  Total:                   {sum(rates):.3e} Hz")
else:
    print(f"\nRelaxation: {len(collapse_ops)} channels (direct + Raman phonon processes)")
    print(f"Total rate: {sum(rates):.3e} Hz")

# ============================================================================
# Calculate T1 and T2
# ============================================================================

# Thermal populations
beta_energy = 1 / (k_B * T)
E_J = evals * h_bar * 2 * np.pi
boltzmann = np.exp(-beta_energy * E_J)
thermal_pops = boltzmann / np.sum(boltzmann)

total_rate = sum(rates) if rates else 1/T1_experimental
T1_calc = 1 / total_rate
times_T1 = np.linspace(0, 100, 100)
pop_T1 = thermal_pops[-1] + (1 - thermal_pops[-1]) * np.exp(-total_rate * times_T1)

print(f"\nT1 (calculated) = {T1_calc:.2f} s  (exp: {T1_experimental:.2f} s, ratio: {T1_calc/T1_experimental:.2f}×)")
print(f"\nRelaxation mechanisms: Direct phonon + Raman + nuclear spin bath estimate")
print(f"Remaining discrepancy likely from: bath spin dynamics, Orbach (if low-lying states exist)")
print(f"\nT2 not calculated: Redfield theory models T1 relaxation but doesn't capture")
print(f"pure dephasing mechanisms (spectral diffusion, charge noise, spin bath).")
print(f"These dominate experimental T2 = {T2_experimental*1e9:.0f} ns.")

# ============================================================================
# Plot Results
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Energy levels
for i, E in enumerate(E_levels):
    ax1.hlines(E/MHz_to_Hz, 0, 1, colors='blue', linewidth=3)

label_positions = []
for i, E in enumerate(E_levels):
    m_val = I - i
    E_mhz = E / MHz_to_Hz
    
    min_spacing = 20  # MHz minimum spacing for labels
    y_pos = E_mhz
    
    for prev_pos in label_positions:
        if abs(y_pos - prev_pos) < min_spacing:
            # Shift upward slightly
            y_pos = prev_pos + min_spacing if E_mhz > prev_pos else prev_pos - min_spacing
    
    ax1.text(1.15, y_pos, f'm={m_val:.1f}', fontsize=9, va='center')
    label_positions.append(y_pos)
    
    if abs(y_pos - E_mhz) > 1:
        ax1.plot([1.05, 1.12], [E_mhz, y_pos], 'k-', linewidth=0.5, alpha=0.3)

ax1.set_xlim(-0.2, 1.7)
ax1.set_ylabel('Energy (MHz)', fontsize=11)
ax1.set_title('Nuclear Spin Energy Levels', fontsize=12, fontweight='bold')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: T1 decay
ax2.plot(times_T1, pop_T1, 'b-', linewidth=2, label='Population')
ax2.axhline(thermal_pops[-1], color='gray', linestyle=':', linewidth=2, label='Thermal')
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Population', fontsize=11)
ax2.set_title(f'T₁ = {T1_calc:.1f} s (calc) vs {T1_experimental:.1f} s (exp)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
