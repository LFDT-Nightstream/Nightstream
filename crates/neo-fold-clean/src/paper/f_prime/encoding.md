# F' Encoding Boundary

This note distinguishes two different encodings.

## `enc_inst(h)`

`enc_inst` maps the raw F' output hash `h = state_x_out_digest(...)` into
the public input of the *next* fresh CCS instance.

The raw hash lanes are ordinary Goldilocks field values. They are fine as
computed values. They are not fine as the fresh CCS public input under
SuperNeo `b = 2`, because the public input `x` is part of the committed
assignment `z = [x, w]` and `CcsInstance::from_low_norm_assignment`
requires `‖z‖_∞ < b`.

Therefore `enc_inst(h)` is implemented as canonical 64-bit decomposition
of the four Goldilocks lanes:

```text
h            = [h0, h1, h2, h3]
enc_inst(h)  = bits(h0) || bits(h1) || bits(h2) || bits(h3)     // 256 bits
```

The full CCS public input is `[1 || enc_inst(h)]` (257 entries, all
`{0, 1}`).

`enc_inst(h)` solves only the public-instance boundary.

## `enc(F')`

`enc(F')` is the larger unresolved problem: encode the *private* F'
execution witness `w` so the full assignment

```text
z = [x, w]
```

is low-norm and can be passed to `CcsInstance::from_low_norm_assignment`.

Do not confuse `enc_inst(h)` with `enc(F')`.

- `enc_inst(h)` handles the public input.
- `enc(F')` must eventually handle the private witness.

Poseidon2 itself does not need low norm. Sumcheck round values,
transcript challenges, K-extension lanes, ring-mul intermediates — all
are ordinary field values during F' execution. They become
low-norm-relevant only at the boundary where F' is committed as a
foldable `u_i = CCS(b, L)` instance.

## Current implementation status

The current source-image code (`source_image.rs` + `source_image_circuit.rs`)
backs:

- `enc_inst(prior_x_out)` — input recursive link body
- `enc_inst(current_x_out)` — output recursive link body
- selected u64 boundary counters (`chunk_count_in`, `step_count_in`, `pc`)

It does **not** yet encode the full private F' witness.

The production-looking projection image is currently a **cost prototype**,
not `enc(F')`: shipped compilers do not fill its projection regions, and the
manual shell reserves K-mul slots without emitting the equations that relate
them to verifier inputs and outputs. The deliberately red gate
`folded_f_prime_kmul_slots_must_be_semantically_constrained` pins this gap.
The complete field-native F' R1CS remains the authoritative relation.

## Open design questions for `enc(F')` — now answered quantitatively

The 2026-07-06 inventory (every figure reproduces from the tests named
below and in the "Reproduce" section) answered these:

- *What is the low-norm assignment for an F' execution?* The production
  shell measures **94,330,948 committed bits per recursive step**
  (`system_phase_1_4a_fibonacci_structure::phase_1_4a_production_config_pins_emitter_counts`):
  465 ring-action pairs × 196,992 bits (97 %, dominated by each pair's
  D² partial-product region), 7,100 K-mul slots × 384 bits (2.7M), and
  ~135k for the state-hash trace + boundary + counters.
- *Which values are public `x`?* Plain F′ uses the 257-slot `enc_inst`
  boundary. Nebula F′ appends the current `S_mem.x`, segment-open bit, and
  `D_pre` bits; the following recursive step consumes that suffix together
  with the same claim's `adv` (HyperNova/Nebula one-step delay).
- *Which are derived, never committed?* Canonical-u64 lanes — rows
  substitute `Σ 2^i · z[bit]` directly; already implemented.
- *Do digit encodings help?* No: the measured SignedDigit ladder
  (`perf_ring_action_low_norm_prototype`: 3,079 full-field / 39,853
  SignedDigit / 200,071 U64 cols per pair) is **invalid on production
  wires** — the ring action acts on commitments, which are full-range
  mod q; only ρ is low-norm (a 1.6 % saving). The earlier figures once
  quoted here ("F' ≈ 10M rows post-optimization", "~3.2B bit slots
  naive") are historical estimates superseded by this measurement.

**REGIME DECIDED (Nico, 2026-07-08): the folded regime (Road A).**
F' becomes a foldable low-norm instance each step, with the ring-action
obligations discharged by the projection check (candidate E below).
The terminal-Spartan road (H) remains the compression story for
proof-size/portability later, but is not the induction mechanism.
Integration order: (1) β transcript schedule — **DONE**: native
`pi_rlc` owns it on both prove and verify paths (recompute per-lane
quotients from authoritative inputs → absorb c* and every q_lane →
squeeze β; wire-identity check fails closed if the mixer is not the
ring action), the NIFS.V circuit replays it bit-for-bit and now enforces
the complete product-commitment (`c + adv`) projection identities using
the exact transcript-bound q and β wires, and
`tests/system/rlc_projection.rs` drives a real fold through the
schedule and the projection-trace encoders with zero residual,
(2) complete the authoritative field-native relation: the `c + adv`
product commitment folds through PiCCS/PiRLC/PiDEC and is opened by the
terminal relation; the recursive relation consumes the prior fresh
claim's suffix/`adv`, enforces the delayed `NebulaLane` transition, composes
current `S_mem`, and projection-checks the c/adv, X, and y clients. Shape-only
synthesis covers base, bootstrap-recursive, and steady-recursive execution.
What remains is to make the accumulator-handle binding affordable, then
(3) lower that one fixed-shape relation into bit-backed CCS; do not maintain
the manual shell as a second verifier, (4) make the shipped encoder fill the
lowered witness, then let the terminal-only verifier consume the induction
and flip the two multi-chunk gates. The old 14,040,452-bit shell measurement
is retained only as prototype evidence that projection beats D²; it is not a
completion claim. The first safe field-shape audit at reduced `κ = 1`
measures `29,184 × 27,851` (base), `13,343,973 × 13,374,181`
(bootstrap), and `31,811,965 × 30,083,642` (steady). Overlaying one-hot
branch-private advice removes summed-width duplication, but the final
relation still has a hard **30,083,645-bit lower bound** before actual
field bit widths, already 1.88× the 16M engineering budget. The dominant
surface is the full running-accumulator Poseidon authority handle, not the
projection ring action. Lemma 5 carries an author self-review whose one
novel claim (a Φ(β) = 0 completeness caveat) was **refuted by external
review** and is retained in the note as a correction record — the
honest identity holds identically at roots of Φ, and Φ_81 has no roots
in K at these parameters anyway. The non-author review remains an open
tracked flag, proceeding at Nico's direction; the refuted self-review
finding is itself the argument for keeping that flag.

## Candidates, costed (ring-action term per step)

| # | Candidate | Bits/step | Status |
|---|---|---|---|
| A | U64 status quo | 91.6M (94.3M total) | works; ~1,650× the S_mem app circuit it recurses |
| B | SignedDigit as measured | — | invalid: operands are full-range commitments |
| C | Mixed (ρ = SignedDigit{5}, rest U64) | 90.1M | valid; saves 1.6 % — not a lever |
| D | Digit-decompose c, act on digits | ~260M | valid; strictly worse (14 SignedDigit pairs vs 1 U64 pair, ×2.8) |
| E | Projection check: verify `Σ_i ρ_i·c_i = out (mod Φ)` as `Σ ρ_i(X)c_i(X) = q(X)Φ(X) + out(X)` at a post-commitment `β ∈ K` | The primitive is measured at ~21k vs ~196k bits per pair. The old manual-shell model is 14,040,452 bits, but omits authoritative wiring and the Nebula census and is not a production step measurement. | **Authoritative NIFS.V/F′ integration includes `c + adv`, X/y projection, delayed lane transition, and current `S_mem`**. Remaining: affordable accumulator binding, fixed-shape low-norm compilation, real encoder fill, terminal delayed transition, and terminal induction. Lemma 5's target census is `P=1,275`, batched `J=85`; conservative `J≤1,275`. |
| F | Fewer pairs (arity/κ trades) | linear only | doesn't touch the 197k/pair |
| G | SIS accumulators (C14/L2) | Prototype: 3 fields at κ=1, including one final Poseidon2 digest, measures 10,532 field columns / 10,537 rows / 93,425 nnz before lowering and 661,445 committed bits / 671,981 rows after complete low-norm lowering. The Ajtai core adds only `Dκ` equations but Θ(`Dκ·64N`) coefficients for `N` fields. | Native/circuit parity, tamper rejection, and canonical-bit slot reuse land in `accumulator_sis_circuit` plus the low-norm compiler; not adopted. Full integration still needs a structured seeded-ring matrix representation, otherwise a full accumulator produces multi-billion-entry sparse matrices. Exact seed/transcript domains, production measurement, and the hash-then-FS lemma remain owed. |
| H | Terminal-proof regime (PR5): never commit F' | 0 per step | field-native cost once per chain (~1–3M-constraint relation); sidesteps enc(F') entirely |

Bottom line: E removed the original ring-action wall, and one-hot advice
overlay removes avoidable branch-width duplication. The authoritative
audit nevertheless proves that the current Poseidon accumulator handle
cannot meet the 16M Road A budget even under an optimistic one-bit-per-field
bound at reduced κ. The 14,040,452-bit shell remains prototype evidence,
not a security or completion gate. Road A therefore needs C14/L2 (or an
equally reviewed binding compression) before fixed-point lowering and the
encoder can be completed; simply lifting the budget would hide a much larger
production-κ cost.

The generic-lowering tax is now measured on the real object, not just the
C14 toy: the complete authoritative NIFS.V circuit over an honest two-fold
chain at the small direct-CCS app shape is **5,934,125 field cols /
5,893,265 rows / 45.4M nnz**, and its complete low-norm lowering is
**371,089,193 committed bits / 376,982,457 rows** — 62.5 bits per field
col, 64.0 rows per row, satisfiability-checked on both sides
(`perf_lowered_nifs_v`). Wire-level lowering without selective commitment
is therefore not a road: it is ~26× the shell cost model and ~4× the D²
shell it was meant to replace. Any completion must keep the bulk of the
verifier's wires derived (row-substituted linear forms), committing bits
only at range-checked and hashed boundaries — which is what the one-bit
audit above assumes and what the shell did by hand.

## Reproduce every number

```bash
# Complete authoritative NIFS.V circuit, low-norm lowered (371M bits, 62.5 bits/col):
cargo test -p neo-fold-clean --release --test perf_lowered_nifs_v -- --ignored --nocapture

# Historical manual-shell cost model (not complete enc(F')):
cargo test -p neo-fold-clean --release --test system_phase_1_4a_fibonacci_structure \
  phase_1_4a_production_config_pins_emitter_counts -- --nocapture

# Authoritative recursive F' gadget census after commitment projection:
cargo test -p neo-fold-clean --release --test system_phase_1_3d_step_parity \
  phase_1_3d_kmul_ring_action_coverage_full_step_three_way_parity -- --nocapture

# Encoding ladder (3,079 full-field / 39,853 SignedDigit / 200,071 U64 cols per pair):
cargo test -p neo-fold-clean --release --test perf_ring_action_low_norm_prototype -- --nocapture

# 156,740-bit canonical shipped image (state-hash authority only):
cargo test -p neo-fold-clean --release --test system_fibonacci_f_prime_layout_budget -- --nocapture

# Candidate C14 primitive, including one final Poseidon2 digest:
cargo test -p neo-fold-clean --release --test reductions_accumulator_sis -- --nocapture

# S_mem app-circuit comparison point (55,434 rows / 57,244 cols):
cargo test -p neo-fold-clean --release --test perf_nebula -- --ignored --nocapture \
  nebula_v3_targets_structure_snapshot
```
