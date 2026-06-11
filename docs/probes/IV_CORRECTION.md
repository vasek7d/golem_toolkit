# Active probe IV correction

Correct raw oscilloscope shunt current and active voltage-divider output to tip voltage and probe/plasma current using lab calibration stored in `active_probe_calib.nc`.

See also: [NAMING.md](NAMING.md) (canonical names and legacy symbols), [architecture.mmd](architecture.mmd) (processing flowchart).

## Quick start

```python
from golem_toolkit.loading import load_tek64_csv
from golem_toolkit.probes import load_iv_calibration, load_probe_waveforms, process_iv_signals

calib = load_iv_calibration("active_probe_calib.nc")
shunt = calib.attrs["shunt_resistance_ohm"]

# Load CSV once, then derive probe inputs (no second read)
ds = load_tek64_csv("shot.csv", channels=["CH1", "CH2"])
I_raw, U_raw = load_probe_waveforms(
    ds,
    f0_hz=20_000.0,
    shunt_resistance_ohm=shunt,
    i_channel="CH1",  # shunt voltage → current
    u_channel="CH2",  # divider output
)

# Or pass the path directly (loads internally)
I_raw, U_raw = load_probe_waveforms("shot.csv", f0_hz=20_000.0, shunt_resistance_ohm=shunt)

# Single-tone AC at 20 kHz
ds = process_iv_signals(I_raw, U_raw, calibration=calib, mode="ac", f0_hz=20_000.0)

# Slow DC / Langmuir sweep
ds = process_iv_signals(I_raw, U_raw, calibration=calib, mode="dc")
```

Inputs: `I_raw` (A, shunt), `U_raw` (V, divider output CH2).  
Outputs: `U_true`, `I_total`, `I_parasitic`, `I_probe` on coordinate `t`.  
AC attrs: `U_true_phasor`, `I_probe_phasor`, `f0_hz` (use phasors for conductance at f0).

## What is admittance?

**Admittance** `Y(f) = I(f)/V(f)` (siemens) is the AC generalization of conductance. Complex admittance has:

- **Real part** — leak conductance `G = Re(Y)`
- **Imaginary part** — susceptance; for capacitance `B = ωC`

Open-circuit calibration measures the **fixture admittance** (wiring + stray C, no plasma/load).

## Glossary

### Phasor

A **complex number** describing one sinusoid at frequency `f0`. Multiplying phasors applies gain and phase shift — used for divider correction, deskew, and parasitic subtraction at `f0`.

### FFT complex amplitude

From samples `x[n]` with spacing `Δt`, compute `spectrum = np.fft.rfft(x)`, bin `k = round(f0·N·Δt)`, then:

```
X = 2 * spectrum[k] / N
```

This is a **peak-amplitude cosine** convention (see circuit_test). Reconstruct with `Re(X * exp(j2π f0 t))`.

### rfft

**Real-input FFT** — `numpy.fft.rfft` returns only non-negative frequency bins because real signals have redundant negative-frequency content.

### replace_fundamental

Given a measured waveform `x(t)`, drive frequency `f0`, and a **calibrated** complex amplitude `X` at `f0`:

```
x_out(t) = x(t) − sinusoid(FFT bin of x at f0) + sinusoid(X)
```

Only the fundamental at `f0` is swapped. Harmonics (2f0, 3f0, …) and broadband noise are left unchanged. Used to build `U_true` and `I_total` in AC mode (see below).

## AC algorithm (`mode="ac"`)

1. Subtract zero offsets from calibration → working signals `I_work`, `U_work` (see below).
2. Interpolate `voltage_divider_transfer`, `open_fixture_admittance`, `shunt_path_correction` at `f0`.
3. Form phasors with `fft_complex_amplitude`.
4. `U_true_phasor = U_raw_phasor / voltage_divider_transfer`
5. `I_total_phasor = I_raw_phasor × exp(j2π f0 Δt) × shunt_path_correction`
6. `I_parasitic_phasor = open_fixture_admittance × shunt_path_correction × U_true_phasor`
7. `I_probe_phasor = I_total_phasor − I_parasitic_phasor`
8. Build time traces (see **Why replace the f0 part of the waveform?** below):
   - `U_true` = `replace_fundamental(U_work / |H|, U_true_phasor)`
   - `I_total` = `replace_fundamental(I_work, I_total_phasor)`
   - `I_parasitic` = pure f0 sinusoid from `I_parasitic_phasor`
   - **`I_probe = I_total − I_parasitic`**
   - Scalars **`U_true_phasor`**, **`I_probe_phasor`** stored in dataset attrs (for conductance at f0).

### Constructing `U_true` (AC mode)

**Physical meaning:** `U_true(t)` is the estimated **tip voltage** (voltage at the probe head), reconstructed from the active divider output `U_raw` (scope CH2).

**Calibration:** `voltage_divider_transfer` = **H(f)** (legacy `H_divider`) is measured in the `direct_input` test: CH1 = direct BNC reference, CH2 = divider output. By convention:

```
H(f) = U_divider_phasor / U_reference_phasor   (deskew accounted for in this measurement — see below)
```

So tip voltage at f0 is **`U_true_phasor = U_divider_phasor / H(f)`** — complex division (gain **and** phase).

See **Scope deskew: calibration vs shot processing** below for how **Δt** enters the divider calibration and why current is deskewed when processing shots.

#### Step 1 — phasor at f0

From `U_work` (divider output after zero-offset removal):

```
U_true_phasor = fft_complex_amplitude(U_work, f0) / H(f)
```

This applies the **divider transfer function** only. No `exp(j2πf0Δt)` is applied to voltage during shot processing.

#### Step 2 — time waveform

The code does **not** use `U_work / H` sample-by-sample (that would require H at every harmonic). Instead:

```
u_gain_only(t) = U_work(t) / |H(f0)|          # magnitude scaling only
U_true(t)      = replace_fundamental(u_gain_only, f0, U_true_phasor)
```

| Step | What it does |
|---|---|
| `U_work / \|H\|` | Roughly scales divider output to tip-voltage amplitude; **phase at f0 still wrong** |
| `replace_fundamental` | Replaces the f0 sine in that trace with the **calibrated** `U_true_phasor` (correct gain and phase at f0); **harmonics and noise unchanged** |

So at f0, `U_true` matches calibration; at 2f0, 3f0, … the waveform is `U_work` scaled by `1/|H|` only — we do not yet have a broadband divider model.

#### Comparison with `I_total`

| | Voltage `U_true` | Current `I_total` |
|---|---|---|
| Working signal | `U_work` (CH2) | `I_work` (CH1 / shunt) |
| Pre-scale before `replace_fundamental` | `U_work / \|H\|` | `I_work` (no pre-scale) |
| Calibrated phasor | `FFT(U_work) / H` — divider correction | `FFT(I_work) × exp(j2πf0Δt) × K` — deskew + shunt path |
| Scope deskew **Δt** | Not applied at shot processing | `exp(j2πf0Δt)` synchronizes I with U |
| Shunt correction **K** | — | On current only |

### Scope deskew: calibration vs shot processing

**Where Δt comes from:** `scope_current_voltage_deskew_s` (legacy `delta_t_scope_s`) is measured in `direct_input/deskewing_BNC`: the same tone is fed to CH1 and CH2; phase difference is converted to an inter-channel delay (average of original and reversed cable orientations for pair **CH1–CH2**, typically tens of ns).

**During divider calibration (determining H):** The scope ratio `FFT(CH2)/FFT(CH1)` would include the scope’s CH1–CH2 timing error as well as the divider’s transfer. Deskew is applied **only in that calibration step** so that the stored **H** reflects the actual voltage-divider transfer function.

Conceptually, deskew **aligns the two channels in time before comparing them**. In code we FFT each channel first, then correct the ratio — that is equivalent for a frequency-independent delay Δt (Fourier shift theorem):

```
# time domain (conceptual): align CH2 to CH1, then form ratio
CH2_aligned(t) = CH2(t − Δt)          # or shift CH1 the other way — same idea
H(f)           = FFT(CH2_aligned) / FFT(CH1)

# same as frequency domain (what the code does):
H(f)           = [FFT(CH2) / FFT(CH1)] × exp(−j2π f Δt)
```

The phase factor is applied to **one channel’s phasor** (equivalently to the ratio); it is not a separate physical effect on top of the divider. At a single tone `f0`, pre-FFT shift and post-FFT `exp(±j2πfΔt)` give the same corrected phasor. We use the phasor form because calibration and shot processing already work in the frequency domain at `f0`.

That is a correction to the **characterization measurement**, not a substitute for synchronizing I and U on plasma shots. **H** is what we use as the divider model: gain and phase of the active divider.

**During shot processing:**

- **`U_true`** comes from CH2 (divider output). The phasor uses **`/ H`** — divider correction only — then `replace_fundamental` builds the waveform. Voltage is not multiplied by `exp(±j2πf0Δt)`; its time base is that of CH2.

- **`I_total`** comes from CH1 (shunt). CH1 and CH2 are not sampled at the same instant. To form consistent phasors and IV data, the current must be **deskewed to the same time reference as voltage**:

```
I_total_phasor = FFT(I_work at f0) × exp(+j2π f0 Δt) × K(f0)
```

Here **`exp(+j2πf0Δt)`** is the operational step that **synchronizes I and U** — the same Δt as in the BNC calibration, again equivalent to shifting the CH1 waveform in time before the FFT at `f0`. **K** is separate (shunt-path gain/phase from the 2 kΩ validation).

**Summary**

| Stage | Role of deskew |
|---|---|
| **Measuring H** | Remove scope timing from the divider calibration so **H** is the true divider transfer |
| **`U_true` on shots** | Use **H** for divider correction; no extra Δt factor |
| **`I_total` on shots** | Apply **`exp(+j2πf0Δt)`** so current (CH1) aligns with voltage (CH2) |

For IV plots and time traces, `U_true` is the x-axis voltage. For **`G = Re(I_probe_phasor / U_true_phasor)`** at f0, use the phasors — the harmonic content in the `U_true` waveform does not enter that ratio.

### Working signals: `I_work` and `U_work`

After optional zero-offset subtraction:

```
I_work = I_raw − shunt_zero_offset
U_work = U_raw − divider_zero_offset
```

| Symbol | Physical meaning |
|---|---|
| `I_raw` | Total current through the shunt (A), as passed in (typically CH1 / R_shunt) |
| `U_raw` | Active divider output (V), as passed in (typically CH2) |
| `I_work` | Shunt current with DC offset removed — the measured total current before deskew/K at waveform level |
| `U_work` | Divider output with DC offset removed |

Phasors are computed from `I_work` and `U_work`, then calibration is applied in the frequency domain at `f0`.

### Why replace the f0 part of the waveform?

The calibrated phasors (`U_true_phasor`, `I_total_phasor`) incorporate divider transfer, scope deskew, and shunt-path correction. The **raw waveforms** `U_work(t)` and `I_work(t)` still contain the **uncorrected** fundamental at f0 (wrong gain/phase from scope and shunt path).

If `I_total` were simply `I_work`, then `I_probe = I_total − I_parasitic` would subtract a **calibrated** parasitic model from an **uncorrected** total current at f0 — inconsistent.

`replace_fundamental` fixes only the f0 component:

```
I_total(t) = I_work(t) − [measured f0 sine] + [calibrated f0 sine from I_total_phasor]
             └──────────── harmonics + noise unchanged from scope ────────────┘
```

Same for `U_true`: see [Constructing U_true (AC mode)](IV_CORRECTION.md#constructing-u_true-ac-mode).

**What comes from where:**

| Frequency content | Source |
|---|---|
| Fundamental at f0 in `I_total`, `U_true` | Calibration (phasors) |
| Harmonics, noise in `I_total`, `U_true` | Measurement (`I_work`, `U_work`) |
| `I_parasitic` | Calibration only (pure f0 sinusoid; no harmonic model yet) |
| f0 in `I_probe` | `I_total_phasor − I_parasitic_phasor` (= `I_probe_phasor`) |
| Harmonics, noise in `I_probe` | Passed through via `I_total`; not removed by parasitic subtraction |

**Why not `replace_fundamental(I_work, I_probe_phasor)` for `I_probe`?**  
It is algebraically identical to `I_total − I_parasitic`, but the subtraction form matches the physics and keeps the pipeline clear: build corrected total, subtract modeled parasitic.

**Phasor-only use:** For conductance at f0, `I_probe_phasor` and `U_true_phasor` are sufficient; waveforms are for IV plots and time traces where harmonics and noise should remain visible.


## DC algorithm (`mode="dc"`)

Quasi-static (slow sweep):

1. `U_true = (U_raw − offset) / Re(voltage_divider_transfer_dc)`
2. `I_total = (I_raw − offset) × Re(shunt_path_correction_dc)`
3. `I_parasitic = dc_leak_conductance × U_true`
4. `I_probe = I_total − I_parasitic`

Assumes sweep slow enough that `C·dU/dt` is negligible.

## Calibration file contents

Frequency-dependent arrays on coordinate `freq`:

- `voltage_divider_transfer_real/imag` (legacy `H_divider`)
- `open_fixture_admittance_real/imag` (legacy `Y_open`)
- `shunt_path_correction_real/imag` (legacy `K_current`)
- `open_fixture_effective_capacitance` / `_corrected` (legacy `C_open`)
- `open_fixture_leak_conductance` / `_corrected` (legacy `G_open`)

Scalars include `scope_current_voltage_deskew_s`, offsets, `dc_leak_conductance_s`, DC extrapolated transfer and correction.

Generate from circuit_test:

```bash
python active_voltage_divider_test/export_iv_calibration.py
```

Example file: [examples/active_probe_calib.nc](examples/active_probe_calib.nc).

## Figures

Representative calibration plots are in [figures/](figures/).

## Future work

`mode="broadband"` for arbitrary waveforms after a denser frequency-sweep calibration (not implemented yet).
