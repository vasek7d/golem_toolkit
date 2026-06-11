# IV correction — variable naming and legacy mapping

Canonical names in **golem_toolkit** and in `active_probe_calib.nc` are **descriptive** (what the quantity physically is). Shorter symbols from the circuit_test notebooks and scripts are kept here as **legacy aliases** for traceability.

## Design rules

1. **NetCDF variables** use `snake_case` English phrases (`open_fixture_admittance_real`).
2. Each exported array carries attrs:
   - `legacy_name` — old symbol (e.g. `Y_open`)
   - `source` — where it was measured or derived (e.g. `resistor_probe_tests/R1_open-circuit`)
   - `units` — SI unit string
3. **Python code** uses the canonical names; `LEGACY_ALIASES` in `iv_calibration.py` maps legacy → canonical when reading old CSVs or notebooks.
4. **circuit_test** analysis scripts keep legacy names internally; only the **export** step writes canonical names to `.nc`.

---

## Calibration quantities (frequency-dependent)

| Canonical name (`.nc` base) | Legacy symbol | Units | Physical meaning | Origin |
|---|---|---|---|---|
| `voltage_divider_transfer` | `H_divider` | 1 | Complex transfer from **divider output** (CH2) to **tip voltage**: `U_true_phasor = U_raw_phasor / voltage_divider_transfer(f)` | [direct_input](circuit_test): `FFT(CH2)/FFT(CH1)` with inter-channel deskew applied in that **calibration measurement** so H is the true divider transfer (equivalent to time-aligning channels before the ratio; see IV_CORRECTION.md) |
| `open_fixture_admittance` | `Y_open` | S | Admittance with probe wiring installed and **no intentional load** (open circuit): `I_par_phasor = open_fixture_admittance(f) × U_true_phasor` | [resistor_probe_tests/R1_open-circuit](circuit_test) |
| `shunt_path_correction` | `K_current` | 1 | Empirical **complex gain** on the shunt-current measurement so the 2 kΩ validation gives the fitted conductance: `shunt_path_correction = G_load_fit / (Y_2k − Y_open)` | Derived in `compute_validation()` from R2 − R1 |
| `loaded_fixture_admittance` | `Y_2k` | S | Total admittance with **2 kΩ load** at probe location | [resistor_probe_tests/R2_2kOhm-load](circuit_test) (stored for validation plots, optional in `.nc`) |
| `probe_load_admittance` | `Y_probe` | S | Load admittance after open subtraction: `loaded_fixture_admittance − open_fixture_admittance` | Derived (R2 − R1) |

Complex arrays are stored as `{name}_real` and `{name}_imag` (where applicable).

### Explicit parasitic arrays (also stored in `.nc`)

| Canonical name | Legacy | Units | Definition |
|---|---|---|---|
| `open_fixture_effective_capacitance` | `C_open` | F | `Im(open_fixture_admittance) / (2πf)` |
| `open_fixture_effective_capacitance_corrected` | `C_open_corrected` | F | `Im(open_fixture_admittance × shunt_path_correction) / (2πf)` |
| `open_fixture_leak_conductance` | `G_open`, `Re(Y_open)` | S | `Re(open_fixture_admittance)` |
| `open_fixture_leak_conductance_corrected` | `G_open_corrected` | S | `Re(open_fixture_admittance × shunt_path_correction)` |

Leakage **current** at tip voltage `U` is `I_leak = leak_conductance × U` (conductance is stored, not current vs frequency).

Reference attrs: `reference_tip_voltage_v`, `reference_leakage_current_a = dc_leak_conductance_s × reference_tip_voltage_v`.

---

## Calibration scalars (DC / metadata)

| Canonical attr | Legacy | Units | Meaning | Origin |
|---|---|---|---|---|
| `scope_current_voltage_deskew_s` | `delta_t_scope_s` | s | CH1–CH2 inter-channel delay from BNC deskew test. Used when **measuring H** (time-align channels in the calibration ratio) and on shots when **processing current**: `I_total_phasor × exp(j2πf·Δt)` synchronizes CH1 with CH2. Not applied again to voltage on shots. | `direct_input/deskewing_BNC` |
| `shunt_resistance_ohm` | `shunt_ohm` | Ω | Shunt for current measurement (47 Ω) | Hardware |
| `divider_zero_offset_v` | `zero_offset_U` | V | Zero-drive offset on divider channel | R0 zero-drive shot |
| `shunt_zero_offset_a` | `zero_offset_I` | A | Zero-drive offset on shunt current | R0 zero-drive shot |
| `dc_leak_conductance_s` | `G_leak_S` | S | Real conductance at DC: leak path in open fixture | `Re(open_fixture_admittance × shunt_path_correction)` extrapolated to 0 Hz |
| `voltage_divider_transfer_dc_real` | `H_divider_dc_real` | 1 | Divider transfer extrapolated to DC | Extrapolation from `voltage_divider_transfer` |
| `voltage_divider_transfer_dc_imag` | `H_divider_dc_imag` | 1 | Imaginary part at DC (often ≈ 0) | Same |
| `shunt_path_correction_dc_real` | `K_current_dc_real` | 1 | Real part of shunt correction at DC | Extrapolation from `shunt_path_correction` |
| `open_fixture_effective_capacitance_f` | `C_open_F` | F | Lowest-frequency effective capacitance (DC attr) |
| `validation_load_resistance_ohm` | `load_fit_resistance_ohm` | Ω | Fitted resistor from low-frequency R2−R1 | `compute_validation()` |
| `validation_load_fit_max_hz` | `load_fit_max_hz` | Hz | Upper frequency of load fit band | `LOAD_FIT_MAX_HZ` in analyze_resistor_probe.py |

---

## Processing inputs and outputs

| Canonical name | Legacy / note | Units | Meaning |
|---|---|---|---|
| `I_raw` | — | A | Measured total current (shunt) |
| `U_raw` | — | V | Measured divider **output** (not yet tip voltage) |
| `U_true` | `V_tip` in some scripts | V | Corrected tip voltage at probe |
| `I_total` | — | A | Corrected total current through shunt |
| `I_parasitic` | — | A | Modeled parasitic current; AC: pure f0 sinusoid from calibration |
| `I_probe` | — | A | `I_total − I_parasitic` — probe/plasma current; AC: f0 from phasors, harmonics/noise from measurement |

### AC processing attrs (on ``process_iv_signals`` output)

| Attr | Meaning |
|---|---|
| `U_true_phasor` | Tip voltage phasor at `f0_hz` after `/ voltage_divider_transfer` |
| `I_probe_phasor` | Probe current phasor at `f0_hz` after parasitic subtraction |
| `scope_current_voltage_deskew_applied_s` | Δt used in `exp(j2πf0Δt)` on the current phasor |
| `voltage_divider_transfer_applied` | Complex **H** at `f0_hz` |
| `shunt_path_correction_applied` | Complex **K** at `f0_hz` |

---

## Python legacy alias table

Used by `load_iv_calibration()` and export tools:

```python
LEGACY_ALIASES = {
    "H_divider": "voltage_divider_transfer",
    "Y_open": "open_fixture_admittance",
    "K_current": "shunt_path_correction",
    "Y_2k": "loaded_fixture_admittance",
    "Y_probe": "probe_load_admittance",
    "G_leak_S": "dc_leak_conductance_s",
    "C_open": "open_fixture_effective_capacitance",
    "C_open_corrected": "open_fixture_effective_capacitance_corrected",
    "G_open": "open_fixture_leak_conductance",
    "G_open_corrected": "open_fixture_leak_conductance_corrected",
    "delta_t_scope_s": "scope_current_voltage_deskew_s",
    "shunt_ohm": "shunt_resistance_ohm",
}
```

---

## Figure filenames (canonical)

| File | Legacy-style name | Content |
|---|---|---|
| `calib_voltage_divider_transfer.png` | `calib_H_divider.png` | `\|transfer\|`, phase vs `freq` |
| `calib_open_fixture_parasitics.png` | `calib_Y_open.png` | C_eff and G_leak, raw vs corrected |
| `calib_shunt_path_correction.png` | `calib_K_current.png` | magnitude, phase, equiv. delay |
| `calib_validation_load_resistance.png` | `calib_validation_resistance.png` | `1/Re(probe_load_admittance)` vs fit |

---

## Formula cheat sheet (canonical names)

**AC mode at drive frequency `f0`:**

```
U_true_phasor = U_raw_phasor / voltage_divider_transfer(f0)
I_total_phasor = I_raw_phasor × exp(j2π f0 Δt_scope) × shunt_path_correction(f0)
I_parasitic_phasor = open_fixture_admittance(f0) × shunt_path_correction(f0) × U_true_phasor
I_probe_phasor = I_total_phasor − I_parasitic_phasor
```

Time traces (AC): `U_true` and `I_total` use `replace_fundamental` so the f0 component matches the phasors above while harmonics/noise stay from the scope; `I_parasitic` is a pure f0 sinusoid; **`I_probe(t) = I_total(t) − I_parasitic(t)`**. See [IV_CORRECTION.md](IV_CORRECTION.md#why-replace-the-f0-part-of-the-waveform) for rationale.

**DC mode:**

```
U_true(t) = (U_raw(t) − divider_zero_offset) / Re(voltage_divider_transfer_dc)
I_total(t) = (I_raw(t) − shunt_zero_offset) × Re(shunt_path_correction_dc)
I_parasitic(t) = dc_leak_conductance × U_true(t)
I_probe(t) = I_total(t) − I_parasitic(t)
```
