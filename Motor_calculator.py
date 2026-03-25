#!/usr/bin/env python3
"""
DC / BLDC Hub Motor Performance Calculator
===========================================
Physics-based estimation for permanent-magnet DC and BLDC (DC-equivalent) motors.

Improvements over the original:
  • Full AWG 0-40 table with NEMA/IEC values
  • Correct Ke/Kt derived from Faraday's law:  Ke = N_series × B × L × r
  • Non-circular operating-point solver:  ω = (V − I·R) / Ke,  I = T / Kt
  • End-turn geometry included in mean turn length
  • Parallel winding paths supported
  • Stall detection (stall torque & stall current)
  • Separate copper / core / friction losses in power budget
  • Steinmetz core-loss model (scales with B, frequency, and material)
  • Lumped thermal model tied to motor geometry
  • Ampacity check against AWG wire rating
  • Efficiency = shaft power / (shaft + copper + core losses)
  • Default values for all inputs — just press Enter
"""

import math

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

COPPER_TEMP_COEF = 0.00393   # /°C  (IEC 60228)
GRAVITY          = 9.81       # m/s²

# ══════════════════════════════════════════════════════════════
#  LOOK-UP TABLES
# ══════════════════════════════════════════════════════════════

# AWG → resistance (Ω/m) at 20 °C, solid copper, NEMA/IEC 60228
AWG_OHMS_PER_M = {
    0:  3.224e-4,  2:  5.127e-4,  4:  8.152e-4,  6:  1.296e-3,
    8:  2.061e-3, 10:  3.277e-3, 12:  5.211e-3, 14:  8.286e-3,
   16:  1.318e-2, 18:  2.095e-2, 20:  3.329e-2, 22:  5.296e-2,
   24:  8.422e-2, 26:  1.339e-1, 28:  2.128e-1, 30:  3.385e-1,
   32:  5.385e-1, 34:  8.568e-1, 36:  1.361,    38:  2.164,
   40:  3.441,
}

# AWG → approximate ampacity (A), chassis wiring guide — used as a warning threshold
AWG_AMPACITY = {
    0: 150, 2: 95,  4: 70,  6: 55,  8: 40,  10: 30, 12: 20, 14: 15,
   16: 13,  18: 10, 20: 5,  22: 3,  24: 2,  26: 1,  28: 0.5,
   30: 0.3, 32: 0.2, 34: 0.16, 36: 0.1, 38: 0.07, 40: 0.05,
}

# Steinmetz core-loss factor relative to silicon-steel at 1 T, 50 Hz
# P_core ∝ loss_factor × B^1.8 × (f/50)^1.15  in W/kg
CORE_LOSS_FACTOR = {
    "air":       0.00,
    "iron":      2.80,   # solid iron  (high eddy losses)
    "silicon":   1.00,   # silicon-steel laminations (0.5 mm, standard)
    "ferrite":   0.35,   # ferrite ceramic core
    "amorphous": 0.25,   # amorphous ribbon (low loss, expensive)
}


# ══════════════════════════════════════════════════════════════
#  INPUT HELPERS
# ══════════════════════════════════════════════════════════════

def get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  ✗  Please enter a number.")

def get_pos_float(prompt, default=None):
    while True:
        v = get_float(prompt, default)
        if v > 0:
            return v
        print("  ✗  Must be greater than zero.")

def get_pos_int(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            v = int(raw)
            if v > 0:
                return v
            print("  ✗  Must be a positive whole number.")
        except ValueError:
            print("  ✗  Enter a whole number.")

def get_choice(prompt, valid, default=None):
    while True:
        raw = input(prompt).strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in valid:
            return raw
        print(f"  ✗  Choose from: {', '.join(valid)}")

def closest_awg(n):
    """Return the nearest AWG key present in AWG_OHMS_PER_M."""
    if n in AWG_OHMS_PER_M:
        return n
    return min(AWG_OHMS_PER_M, key=lambda x: abs(x - n))


# ══════════════════════════════════════════════════════════════
#  PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════

def calc_resistance(awg, num_slots, turns, stator_L_m, rotor_r_m, temp_c):
    """
    Total winding resistance at temperature.

    Mean Turn Length (MTL):
        active conductors  = 2 × L_stack      (two slot sides per turn)
        end-turn portion   = π × slot_pitch   (semicircle ≈ coil overhang)
        MTL = 2·L + π·(2π·r / num_slots)

    Temperature correction:  ρ(T) = ρ₂₀ · [1 + α·(T − 20)]

    Returns (R_ohm, awg_key_used)
    """
    awg_key  = closest_awg(awg)
    rho20    = AWG_OHMS_PER_M[awg_key]
    rho_T    = rho20 * (1 + COPPER_TEMP_COEF * (temp_c - 20))
    slot_pitch = (2 * math.pi * rotor_r_m) / num_slots
    mtl        = 2 * stator_L_m + math.pi * slot_pitch
    R          = num_slots * turns * mtl * rho_T
    return R, awg_key


def calc_motor_constants(num_slots, turns, B_gap_T,
                         stator_L_m, rotor_r_m, parallel_paths):
    """
    Back-EMF constant Ke [V·s/rad] and torque constant Kt [N·m/A].

    Derivation (distributed winding, DC equivalent):
        Each conductor contributes  dV = B · L · (r·ω)
        Series conductors in one electrical path:
            N_series = (num_slots × turns) / parallel_paths

        Ke = N_series × B_gap × L_stack × r_rotor

    Ke = Kt in SI units (energy conservation: T·ω = E·I).

    For 3-phase BLDC in trapezoidal drive, these per-phase values
    apply directly to the DC equivalent circuit.
    """
    N_series = (num_slots * turns) / parallel_paths
    ke       = N_series * B_gap_T * stator_L_m * rotor_r_m
    return ke, ke   # Kt = Ke in SI


def calc_operating_point(V_supply, T_load_Nm, ke, kt, R_ohm):
    """
    Steady-state operating point from the two fundamental motor equations:

        V  = Ke·ω  +  I·R     (KVL around the armature circuit)
        T  = Kt·I              (force equation)

    Solving for I and ω given T_load:
        I   = T_load / Kt
        ω   = (V − I·R) / Ke

    Stall occurs when T_load ≥ T_stall = Kt · (V / R).

    Returns: (omega_rad_s, current_A, stall_torque_Nm, no_load_rpm, stalled_bool)
    """
    if ke <= 0 or kt <= 0:
        return 0.0, 0.0, 0.0, 0.0, True

    i_stall  = V_supply / R_ohm
    t_stall  = kt * i_stall
    omega0   = V_supply / ke          # theoretical no-load speed (rad/s)

    if T_load_Nm >= t_stall:
        return 0.0, i_stall, t_stall, omega0, True

    I     = T_load_Nm / kt
    omega = max(0.0, (V_supply - I * R_ohm) / ke)
    return omega, I, t_stall, omega0, False


def calc_core_loss(material, B_T, omega_rad_s, num_magnets, rotor_r_m, stator_L_m):
    """
    Simplified Steinmetz core loss.

    Electrical frequency:  f = (P/2) × (ω / 2π)   where P = num_magnets

    Specific loss model (W/kg):
        P_sp = lf × B^1.8 × (f / 50)^1.15

    Core mass estimated from stator geometry with ~40% iron fill factor.
    """
    lf = CORE_LOSS_FACTOR.get(material, CORE_LOSS_FACTOR["silicon"])
    if lf == 0.0 or omega_rad_s <= 0:
        return 0.0

    f_hz   = (num_magnets / 2.0) * omega_rad_s / (2 * math.pi)
    vol    = math.pi * rotor_r_m**2 * stator_L_m * 0.40   # 40% fill
    mass   = vol * 7650                                     # silicon-steel density
    p_sp   = lf * (B_T ** 1.8) * (f_hz / 50.0) ** 1.15
    return max(0.0, p_sp * mass)


def calc_load_torque(mass_kg, friction_coeff, radius_m, accel_rad_s2):
    """
    T_friction = μ · m · g · r          (Coulomb rolling/bearing friction)
    T_inertia  = J · α   where J = ½·m·r²   (solid-disk approximation)
    T_total    = T_friction + T_inertia
    """
    Tf = friction_coeff * mass_kg * GRAVITY * radius_m
    J  = 0.5 * mass_kg * radius_m**2
    Ti = J * accel_rad_s2
    return Tf + Ti, Tf, Ti


def calc_thermal(p_cu_W, p_core_W, duration_s, rotor_r_m, stator_L_m):
    """
    Lumped single-node thermal model (no forced cooling):
        Q   = (P_cu + P_core) × t
        ΔT  = Q / (m · Cp)

    Motor mass estimated from geometry; Cp ≈ 450 J/(kg·°C) for Cu/Fe composite.
    Returns (delta_T_degC, total_heat_J)
    """
    vol_m3  = math.pi * rotor_r_m**2 * stator_L_m * 0.45
    mass_kg = max(0.05, vol_m3 * 7200)   # averaged Cu+Fe density
    Cp      = 450
    Q_j     = (p_cu_W + p_core_W) * duration_s
    delta_T = Q_j / (mass_kg * Cp)
    return delta_T, Q_j


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_calculation(p):
    """Run all physics modules and return a results dict."""
    r = p["r_cm"] / 100.0
    L = p["L_cm"] / 100.0

    # 1. Winding resistance
    R_ohm, awg_key = calc_resistance(
        p["awg"], p["slots"], p["turns"], L, r, p["temp_c"])

    # 2. Motor constants
    ke, kt = calc_motor_constants(
        p["slots"], p["turns"], p["B"], L, r, p["paths"])

    # 3. Load torque
    T_load, Tf, Ti = calc_load_torque(
        p["mass"], p["mu"], r, p["alpha"])

    # 4. Steady-state operating point
    omega, I, T_stall, omega0, stalled = calc_operating_point(
        p["V"], T_load, ke, kt, R_ohm)

    rpm  = omega  * 60.0 / (2 * math.pi)
    rpm0 = omega0 * 60.0 / (2 * math.pi)

    back_emf = ke * omega
    Vr       = I  * R_ohm
    i_stall  = p["V"] / R_ohm if R_ohm > 0 else 0.0

    # 5. Power budget
    # p_shaft = T_load × ω  (mechanical output delivered to the shaft)
    # This equals p_elec_useful = back_emf × I  (consistent check).
    p_shaft  = T_load * omega           # shaft mechanical output
    p_cu     = I**2 * R_ohm            # copper resistive loss
    p_core   = calc_core_loss(          # iron / core loss
        p["material"], p["B"], omega, p["magnets"], r, L)
    # Total electrical input = shaft output + copper loss + core loss
    p_in     = p_shaft + p_cu + p_core
    # (Note: p_in ≈ V×I when core loss is small, which it normally is.)
    eta = p_shaft / p_in * 100.0 if p_in > 0 else 0.0

    # Friction share of shaft power (informational)
    p_fric_out = Tf * omega

    # 6. Thermal estimate over 5 minutes
    delta_T, Q_j = calc_thermal(p_cu, p_core, 300, r, L)

    # 7. Ampacity check
    amp_max  = AWG_AMPACITY.get(awg_key, None)
    amp_warn = (amp_max is not None and I > amp_max)

    return {
        # Status
        "stalled":     stalled,
        # Speed
        "rpm":         rpm,
        "omega":       omega,
        "rpm0":        rpm0,
        "omega0":      omega0,
        # Current & torque
        "I":           I,
        "i_stall":     i_stall,
        "T_load":      T_load,
        "T_stall":     T_stall,
        "Tf":          Tf,
        "Ti":          Ti,
        # Voltage
        "back_emf":    back_emf,
        "Vr":          Vr,
        # Power
        "p_in":        p_in,
        "p_cu":        p_cu,
        "p_core":      p_core,
        "p_shaft":     p_shaft,
        "p_fric_out":  p_fric_out,
        "eta":         eta,
        # Motor constants
        "R":           R_ohm,
        "ke":          ke,
        "kt":          kt,
        "awg":         awg_key,
        # Thermal
        "dT":          delta_T,
        "Q_kj":        Q_j / 1000.0,
        # Warnings
        "amp_warn":    amp_warn,
        "amp_max":     amp_max,
    }


# ══════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════

def display_results(res, V, T_amb):
    W   = 58
    bar = "═" * W
    thin = "─" * W

    print(f"\n{bar}")
    print("  MOTOR PERFORMANCE RESULTS")
    print(bar)

    if res["stalled"]:
        print("\n  ⚠  MOTOR STALLED — load torque exceeds stall torque")
        print(f"\n     Load torque   : {res['T_load']:.5f} N·m")
        print(f"     Stall torque  : {res['T_stall']:.5f} N·m  ← must be larger")
        print(f"     Stall current : {res['i_stall']:.3f} A  (will overheat rapidly)")
        print(f"\n  Suggestions:")
        print(f"     • Increase supply voltage")
        print(f"     • Use more turns or stronger magnets (higher Kt)")
        print(f"     • Reduce load mass or friction coefficient")
        print(f"\n{bar}")
        return

    print(f"\n  ┌─ Speed {'─'*46}┐")
    print(f"  │  Operating speed   : {res['rpm']:>10.2f} RPM  ({res['omega']:.3f} rad/s)")
    print(f"  │  No-load speed     : {res['rpm0']:>10.2f} RPM  (theoretical ideal)")
    print(f"  │  Speed regulation  : {(1 - res['omega']/res['omega0'])*100:>10.1f} %")
    print(f"  └{'─'*54}┘")

    print(f"\n  ┌─ Torque {'─'*45}┐")
    print(f"  │  Total load torque  : {res['T_load']:>9.5f} N·m")
    print(f"  │  ├ Friction torque  : {res['Tf']:>9.5f} N·m")
    print(f"  │  └ Inertia  torque  : {res['Ti']:>9.5f} N·m")
    print(f"  │  Stall torque (ref) : {res['T_stall']:>9.5f} N·m")
    print(f"  │  Load / stall ratio : {res['T_load']/res['T_stall']*100:>9.1f} %  (< 80% = safe margin)")
    print(f"  └{'─'*54}┘")

    print(f"\n  ┌─ Electrical {'─'*41}┐")
    print(f"  │  Supply voltage     : {V:>9.2f} V")
    print(f"  │  Back EMF           : {res['back_emf']:>9.3f} V  ({res['back_emf']/V*100:.1f}% of supply)")
    print(f"  │  Resistive drop I·R : {res['Vr']:>9.3f} V")
    print(f"  │  Operating current  : {res['I']:>9.4f} A")
    print(f"  │  Stall current      : {res['i_stall']:>9.3f} A  (reference)")
    print(f"  └{'─'*54}┘")

    print(f"\n  ┌─ Power Budget {'─'*39}┐")
    print(f"  │  Total input power        : {res['p_in']:>8.4f} W")
    print(f"  │  ├ Copper loss  (I²·R)    : {res['p_cu']:>8.4f} W")
    print(f"  │  ├ Core loss   (Steinmetz) : {res['p_core']:>8.4f} W")
    print(f"  │  └ Shaft output power     : {res['p_shaft']:>8.4f} W")
    print(f"  │     └ (friction component): {res['p_fric_out']:>8.4f} W")
    print(f"  │")
    print(f"  │  ★ Efficiency             : {res['eta']:>8.2f} %")
    print(f"  └{'─'*54}┘")

    print(f"\n  ┌─ Thermal — 5-min run, no cooling {'─'*19}┐")
    print(f"  │  Total heat generated : {res['Q_kj']:>8.3f} kJ")
    print(f"  │  Temperature rise     : {res['dT']:>8.1f} °C")
    print(f"  │  Estimated winding T  : {T_amb + res['dT']:>8.1f} °C")
    if T_amb + res["dT"] > 130:
        print(f"  │  ⚠  Exceeds typical Class B insulation limit (130 °C)")
    print(f"  └{'─'*54}┘")

    print(f"\n  ┌─ Motor Parameters {'─'*35}┐")
    print(f"  │  Winding resistance  R  : {res['R']:>9.6f} Ω   (AWG {res['awg']})")
    print(f"  │  Back-EMF constant  Ke  : {res['ke']:>9.6f} V·s/rad")
    print(f"  │  Torque constant    Kt  : {res['kt']:>9.6f} N·m/A")
    print(f"  └{'─'*54}┘")

    if res["amp_warn"]:
        print(f"\n  ⚠  AMPACITY WARNING: {res['I']:.4f} A exceeds")
        print(f"     AWG {res['awg']} chassis rating of {res['amp_max']} A.")
        print(f"     Wire insulation may degrade.  Consider thicker wire.")

    print(f"\n{bar}")
    print("  These are estimates from simplified models. Real motors vary due to")
    print("  winding asymmetry, magnetic saturation, harmonics, and manufacturing.")
    print(bar)


# ══════════════════════════════════════════════════════════════
#  EXPLANATION MENU
# ══════════════════════════════════════════════════════════════

PARAM_HELP = [
    ("Air-gap Flux Density  B  (Tesla)",
     "Magnetic field in the air gap between rotor magnets and stator.\n"
     "  NdFeB magnets : 0.6–1.1 T (strongest common type)\n"
     "  SmCo magnets  : 0.7–1.0 T\n"
     "  Ferrite magnets: 0.2–0.4 T\n"
     "  This single value has the most impact on Ke and Kt.\n"
     "  Input the air-gap value, not the remanence (typically 60–80% of Br)."),

    ("Number of Magnets",
     "Total permanent magnets on the rotor — must be even (N/S pairs).\n"
     "  Also sets the electrical frequency: f = (magnets/2) × (RPM/60).\n"
     "  More poles → smoother torque, higher core loss at same speed."),

    ("Stator Slots",
     "Slots cut into the stator iron where coils are wound.\n"
     "  Common: 9, 12, 18, 24, 36. More slots → smoother torque, less cogging.\n"
     "  Slots and magnets must be chosen carefully (slot/pole ratio)."),

    ("Turns per Coil",
     "Wire turns wound around one stator tooth/slot.\n"
     "  More turns → higher Ke (more volts per RPM), lower speed, higher R.\n"
     "  Fewer turns → lower Ke, higher speed, lower R.\n"
     "  Ke scales exactly linearly with turns."),

    ("Wire Gauge (AWG)",
     "American Wire Gauge — smaller number = thicker wire.\n"
     "  Thicker wire → lower resistance, higher current capacity,\n"
     "  but fewer turns fit in the same slot.\n"
     "  Motor windings: typically AWG 14–26.\n"
     "  Any AWG 0–40 is accepted; nearest table value is used."),

    ("Parallel Winding Paths",
     "Number of electrically parallel paths through the winding.\n"
     "  Lap winding: typically 2.  Wave winding: 1.  Concentric: varies.\n"
     "  More paths → lower Ke AND lower R (both scale as 1/paths)."),

    ("Rotor Radius (cm)",
     "Radius of the rotor measured to the magnet surface.\n"
     "  Larger radius → more torque for the same B and turns (Ke ∝ r).\n"
     "  Also determines the hub motor's physical size."),

    ("Stator Stack Length (cm)",
     "Axial length of the stator core lamination stack.\n"
     "  Longer stack → proportionally more torque and power (Ke ∝ L).\n"
     "  Also increases wire length and resistance proportionally."),

    ("Core Material",
     "Material of the stator core:\n"
     "  air       → coreless (Halbach array etc.), zero core loss\n"
     "  iron      → solid iron, highest core loss (legacy designs)\n"
     "  silicon   → silicon-steel laminations, standard motor choice\n"
     "  ferrite   → ferrite ceramic, low loss, brittle\n"
     "  amorphous → amorphous metal, very low loss, costly"),

    ("Supply Voltage (V)",
     "DC voltage from battery or PSU.\n"
     "  No-load speed ≈ V / Ke (proportional to voltage).\n"
     "  Stall torque  ≈ Kt × V / R (also proportional to voltage)."),

    ("Ambient Temperature (°C)",
     "Air temperature around the motor.\n"
     "  Copper resistance rises ~0.39%/°C.  At 80°C vs 20°C,\n"
     "  resistance is ~24% higher → more copper loss, lower speed."),

    ("Load Mass (kg)",
     "Mass of the object the motor drives.\n"
     "  Contributes to both friction torque (via normal force)\n"
     "  and inertia torque (via moment of inertia J = ½mr²)."),

    ("Friction Coefficient μ",
     "Ratio of friction force to normal force (dimensionless).\n"
     "  Greased ball bearing : 0.001–0.005\n"
     "  Dry bushing          : 0.05–0.15\n"
     "  Rubber on concrete   : 0.6–0.8\n"
     "  This is the bearing/contact friction, not aerodynamic drag."),

    ("Angular Acceleration (rad/s²)",
     "Desired ramp-up rate for the rotor.\n"
     "  0 means steady-state (constant speed).\n"
     "  Sets the additional inertia torque T_i = J × α.\n"
     "  1 rad/s² is gentle; >30 rad/s² is aggressive for small motors."),
]

def explanation_menu():
    while True:
        print("\n  ─── Parameter Reference ───────────────────────────────")
        for i, (title, _) in enumerate(PARAM_HELP, 1):
            print(f"  {i:>2}. {title}")
        print("   0. Back")
        ch = input("\n  Choice: ").strip()
        if ch == "0":
            break
        try:
            idx = int(ch) - 1
            if 0 <= idx < len(PARAM_HELP):
                title, body = PARAM_HELP[idx]
                print(f"\n  ── {title} ──")
                for line in body.split("\n"):
                    print(f"  {line}")
            else:
                print("  ✗  Out of range.")
        except ValueError:
            print("  ✗  Invalid choice.")


# ══════════════════════════════════════════════════════════════
#  INPUT COLLECTION
# ══════════════════════════════════════════════════════════════

def collect_inputs():
    print("\n  Press Enter to accept [default] values shown in brackets.\n")

    print("  ── Winding ────────────────────────────────────────────")
    slots = get_pos_int  ("  Stator slots              [12]  : ", 12)
    awg   = get_pos_int  ("  Wire gauge AWG            [22]  : ", 22)
    turns = get_pos_int  ("  Turns per coil            [40]  : ", 40)
    paths = get_pos_int  ("  Parallel winding paths    [ 2]  : ", 2)

    print("\n  ── Magnets & Geometry ─────────────────────────────────")
    B       = get_pos_float("  Air-gap flux density (T)  [0.8] : ", 0.8)
    magnets = get_pos_int  ("  Number of magnets         [14]  : ", 14)
    r_cm    = get_pos_float("  Rotor radius       (cm)   [5.0] : ", 5.0)
    L_cm    = get_pos_float("  Stator stack len   (cm)   [3.5] : ", 3.5)

    valid_mat = list(CORE_LOSS_FACTOR.keys())
    print(f"\n  Core materials: {', '.join(valid_mat)}")
    material = get_choice(
        "  Core material           [silicon]: ",
        valid_mat, "silicon")

    print("\n  ── Operating Conditions ───────────────────────────────")
    V      = get_pos_float("  Supply voltage      (V)   [36]  : ", 36.0)
    temp_c = get_float    ("  Ambient temperature (°C)  [25]  : ", 25.0)

    print("\n  ── Load (press Enter for defaults) ────────────────────")
    mass  = get_pos_float ("  Load mass           (kg)  [2.0] : ", 2.0)
    mu    = get_pos_float ("  Friction coefficient       [0.005]: ", 0.005)
    alpha = get_float     ("  Angular accel  (rad/s²)   [0]   : ", 0.0)

    return {
        "slots": slots, "awg": awg, "turns": turns, "paths": paths,
        "B": B, "magnets": magnets, "r_cm": r_cm, "L_cm": L_cm,
        "material": material,
        "V": V, "temp_c": temp_c,
        "mass": mass, "mu": mu, "alpha": alpha,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

BANNER = """\
╔══════════════════════════════════════════════════════════╗
║   DC / BLDC Hub Motor Performance Calculator             ║
║   Physics-based estimation tool  v2.0                    ║
╚══════════════════════════════════════════════════════════╝

  ★ Outputs are ESTIMATES from simplified electromagnetic models.
  ★ Use for design exploration and feasibility — not final spec.
  ★ Core-loss model: Steinmetz  |  Thermal: lumped single-node
"""

def main():
    print(BANNER)
    last_params = None

    while True:
        print("  ┌─ Main Menu ──────────────────────────────────────────┐")
        print("  │  1. Parameter explanations                           │")
        print("  │  2. Run new calculation                              │")
        if last_params:
            print("  │  3. Rerun with last inputs                           │")
        print("  │  0. Exit                                             │")
        print("  └──────────────────────────────────────────────────────┘")

        choice = input("\n  Choice: ").strip()

        if choice == "0":
            print("\n  Goodbye.\n")
            break
        elif choice == "1":
            explanation_menu()
        elif choice == "2":
            params      = collect_inputs()
            last_params = params
            results     = run_calculation(params)
            display_results(results, params["V"], params["temp_c"])
        elif choice == "3" and last_params:
            results = run_calculation(last_params)
            display_results(results, last_params["V"], last_params["temp_c"])
        else:
            print("  ✗  Invalid choice.")

if __name__ == "__main__":
    main()
