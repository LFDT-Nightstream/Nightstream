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
The accumulator handle now reuses the already-computed running-parent CE
digest after native and in-circuit NIFS.V verify strict Pi_DEC consistency
between that parent and every child. Re-hashing all children was duplicate
authority; child-tamper tests still fail after rebuilding the compact handle.
The R2 authoritative-relation and R3 low-norm-compilation milestones are
**DONE**. Five witness-proportional claim/projection/leaf roles use independent
rank-2 seeded SIS/Ajtai maps followed by one independent short rank-1 map and
a domain-separated Poseidon2 digest. Each long map consumes the same 41
centered unit digits that encode its authoritative source fields, rather than
a second 64-bit serialization. The v3 digest envelope binds the role, field
count, and primary rank. `CscWithSeededPhi81` keeps both maps compact through
CCS and SuperNeo evaluation. The selective compiler lowers Poseidon2 S-boxes,
projection evaluations, K multiplication, rejection selection, and centered
PiDEC checks directly instead of committing their R1CS temporaries. Full field
values without an existing canonical decomposition use 41 balanced-ternary
digits in `{-1,0,1}`; canonical-u64 fields retain their shared 64-bit slots.
This is still `w = 1` at the committed-coordinate boundary: radix 3 lives in
the verifier-owned matrix coefficients, while every witness digit has norm at
most one. Private final Poseidon outputs are substituted linearly, five product
pairs share one direct CCS row, and long evaluations use telescoping
accumulators. K dot products use the exact Karatsuba sums `P`, `Q`, and `R`
instead of retaining every per-term K output. The reduced compiler profile
reaches a rectangular three-arm verifier-shape fixed point at **2,486,540
semantic rows / 9,613,188 committed coordinates / 13 matrices / degree 8**.
SplitNc checks FE over the row domain and NC over the assignment domain, so the
selective relation does not need an identity matrix or square row padding.
The active `road_a_reduced_profile_fixed_point_stabilizes_within_budget` test
pins that compiler invariant, and `compile_fixed_point` rejects an oversized
result. R7's Appendix B.2 preflight at `kappa = 18`, `k_rho = 14`, `T = 216`
and maximum v3.1 memory geometry measures **15,730,104 coordinates** on the
first selective census. The verifier-shape fixed point stabilizes at
**2,819,360 semantic rows / 15,612,210 committed coordinates / 13 matrices /
degree 8**, 387,790 coordinates below the unchanged 16M ceiling. The active
`nebula_v3_targets_folded_f_prime_production_preflight` test pins that result,
the two-level map dimensions, and the 65.32-bit conservative maximum-chain
union against the declared 64-bit target (`SEG_MAX=2^16`, `q_H≤2^16`).
R4's shipped encoder
and R5's terminal induction are **DONE**: `NebulaFPrimeChainBuilder` deposits
the fixed relation with serial `K=1`, recursive steps consume the prior claim's
delayed suffix, finalization consumes the trailing claim, and the terminal-only
verifier accepts the final accumulator plus terminal fold without the audit
history. The active
`r4_shipped_encoder_verifies_multistep_memory_chain` test traverses all three
arms over three one-step segments and rejects link, suffix, lane, and history
tampering. Focused delayed-suffix tests cover the absent-`D_pre` interior
encoding without another production-sized fold. The plain shipped
encoder is exercised by `r1cs_stateful_linked_fibonacci_chain_verifies_end_to_end`.
The active R5 gate `multi_chunk_f_prime_chain_must_verify_terminal_only`
additionally rejects a changed pre-final running commitment, so earlier folded
history remains load-bearing without audit replay.
Legacy and generic F' frontends remain terminal-only fail-closed. The old
14,040,452-bit manual shell remains prototype evidence only.
Lemma 5 carries an author self-review whose one
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
| E | Projection check: verify `Σ_i ρ_i·c_i = out (mod Φ)` as `Σ ρ_i(X)c_i(X) = q(X)Φ(X) + out(X)` at a post-commitment `β ∈ K` | The primitive is measured at ~21k vs ~196k bits per pair. The reduced-profile fixed point is 9,613,188 coordinates over 2,486,540 rows; the production fixed point is 15,612,210 coordinates over 2,819,360 rows. The old 14,040,452-bit manual shell remains a non-authoritative reference. | **Implemented end to end** in authoritative NIFS.V/F′ for `c + adv`, X/y projection, delayed lane transition, current `S_mem`, and terminal-only lifecycle induction. Exact Karatsuba K-dot tracing keeps production below the unchanged 16M ceiling. Lemma 5's maximum-geometry census is `P=2,250`, batched `J=150`; conservative `J≤2,250`. |
| F | Fewer pairs (arity/κ trades) | linear only | doesn't touch the 197k/pair |
| G | SIS accumulators (C14/L2) | A role-specific rank-2 map binds the authoritative 41-trit encoding; an independent short rank-1 map compresses its 108-field output before Poseidon2. | **Adopted for five R2 binding roles**, with compact seeded matrices, native/circuit parity, stage-tamper tests, concrete rank-2/rank-1 estimates, and security-note Lemma 6's hash-then-FS reduction. Replacing the carried `D` chains remains deferred. |
| H | Terminal-proof regime (PR5): never commit F' | 0 per step | field-native cost once per chain (~1–3M-constraint relation); sidesteps enc(F') entirely |

Bottom line: E removed the ring-action wall, the verified-parent handle removed
the duplicate child hash chain, and the SIS/selective compiler closes the R2/R3
mechanics. R4-R6 consume that relation through the shipped encoder and
terminal-only memory induction. Both reduced and production fixed points fit
the unchanged 16M ceiling. Generic all-binary lowering and the old manual shell
remain reference paths, not production encodings.

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
# Reduced R2 verifier-shape fixed point (2,486,540 rows x 9,613,188 coordinates):
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  road_a_reduced_profile_fixed_point_stabilizes_within_budget -- --exact --nocapture

# R7 production fixed-point gate (2,819,360 rows x 15,612,210 coordinates):
cargo test -p neo-fold-clean --release --test perf_nebula \
  nebula_v3_targets_folded_f_prime_production_preflight -- --exact --nocapture

# R4 shipped encoder over two multi-step memory segments:
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  r4_shipped_encoder_verifies_multistep_memory_chain -- --exact --nocapture

# R4 plain multi-step encoder through the encoded F' audit relation:
cargo test -p neo-fold-clean --release --test system_r1cs_compiler_stateful \
  r1cs_stateful_linked_fibonacci_chain_verifies_end_to_end -- --exact

# R5 final accumulator + latest fold, with no audit-history authority:
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  multi_chunk_f_prime_chain_must_verify_terminal_only -- --exact --nocapture

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
