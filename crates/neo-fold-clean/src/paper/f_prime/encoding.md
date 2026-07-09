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

## Open design questions for `enc(F')` — now answered quantitatively

The 2026-07-06 inventory (every figure reproduces from the tests named
below and in the "Reproduce" section) answered these:

- *What is the low-norm assignment for an F' execution?* The production
  shell measures **94,330,948 committed bits per recursive step**
  (`system_phase_1_4a_fibonacci_structure::phase_1_4a_production_config_pins_emitter_counts`):
  465 ring-action pairs × 196,992 bits (97 %, dominated by each pair's
  D² partial-product region), 7,100 K-mul slots × 384 bits (2.7M), and
  ~135k for the state-hash trace + boundary + counters.
- *Which values are public `x`?* Only the 257-slot `enc_inst` boundary.
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
Integration order: (1) β transcript schedule on the F' step transcript
(native prover computes β and the quotients; the circuit re-derives β
from the replayed transcript), (2) projection regions replace the D²
ring-action regions in the F' image/structure (flips the
`folded_f_prime_shell_must_adopt_projection_budget` gate), (3) Nebula
lane transition joins the F' state bundle (spec §13 step 9), (4) the
terminal-only verifier consumes the induction (flips the two
multi-chunk gates). Lemma 5 carries an author self-review whose one
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
| E | Projection check — **integrated through Road A Phase B**: verify `Σ_i ρ_i·c_i = out (mod Φ)` as the polynomial identity `Σ ρ_i(X)c_i(X) = q(X)Φ(X) + out(X)` tested at a post-commitment challenge `β ∈ K` (`engine/r1cs_circuit/ring_action.rs::enforce_ring_action_projection_batch`; SZ error ≤ (2D−2)/\|K\| ≈ 2^−121 per identity) | **integrated shell measurement**: production `FPrimeImageLayout` is **14,040,452 committed bits/step** with projection regions (D2 reference: 94,330,948; 6.7x). The marginal primitive remains 330 cols/pair vs 3,078 full-field / ~21k vs ~196k bits bit-backed → **9.3× per pair**. | parity + rejection tests in `tests/system/ring_action_projection.rs`; budget gate `ivc_invariants.rs::folded_f_prime_shell_must_adopt_projection_budget` is **GREEN un-ignored**; semantic-row gate `ivc_invariants.rs::projection_shell_semantic_rows_must_be_enforced` is **GREEN un-ignored** — `frontends/f_prime/projection_structure.rs` emits the beta ladder, evaluation sums, Karatsuba relations, and final identity rows. Still owed: beta transcript binding, real encoder fill from fold traces, terminal induction. **The soundness case is written**: `specs/nebula-superneo-security-note.md` Lemma 5 (§4b, ledger C18; committed) — exact transcript schedule (`q` and `out` absorbed before β), J·(2d−2)/\|K\| bound with P (product pairs, drives cost) distinguished from J (projection identities, drives error): conservative J ≤ 465 → ≈ 2^−112.4/fold until the adoption census proves smaller (known clients would give J = 4κ = 72 → ≈ 2^−115). Composition with Lemma 1, five adoption audit items incl. the census table. Still owed: Lemma 5's non-author review (attack the J census and wire-identity obligation first), then β's transcript wiring and terminal F' integration |
| F | Fewer pairs (arity/κ trades) | linear only | doesn't touch the 197k/pair |
| G | SIS accumulators (C14/L2) | — | helps the absorb (Poseidon) side, already small; not the fold math |
| H | Terminal-proof regime (PR5): never commit F' | 0 per step | field-native cost once per chain (~1–3M-constraint relation); sidesteps enc(F') entirely |

Bottom line: encodings alone cannot move the wall (best honest bit
encoding ≈ 90M); before candidate E the data favored H, and **Nico
decided Road A (folded) with E as its mechanism** — see the regime
decision above. E's integrated shell now measures **14,040,452
committed bits/step** (~245× the S_mem app circuit instead of
~1,650×), green under the budget gate. The K-mul/sumcheck region
(2.7M) is a candidate for the same projection treatment later.

## Reproduce every number

```bash
# 94,330,948-bit production shell (7,100 K-muls, 465 ring pairs):
cargo test -p neo-fold-clean --release --test system_phase_1_4a_fibonacci_structure \
  phase_1_4a_production_config_pins_emitter_counts -- --nocapture

# Encoding ladder (3,079 full-field / 39,853 SignedDigit / 200,071 U64 cols per pair):
cargo test -p neo-fold-clean --release --test perf_ring_action_low_norm_prototype -- --nocapture

# 134,852-bit canonical shipped image (state-hash authority only):
cargo test -p neo-fold-clean --release --test system_fibonacci_f_prime_layout_budget -- --nocapture

# S_mem app-circuit comparison point (55,434 rows / 57,244 cols):
cargo test -p neo-fold-clean --release --test perf_nebula -- --ignored --nocapture \
  nebula_v3_targets_structure_snapshot
```
