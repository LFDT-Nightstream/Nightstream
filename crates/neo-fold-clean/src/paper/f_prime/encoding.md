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

The 2026-07-06 inventory (design note in the local `specs/` working
directory; every figure reproduces from the tests named below) answered
these:

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

What remains open is the *regime decision*, not a measurement: fold F'
(requires new protocol work on the ring-action check to escape the D²
term — e.g., a projection check, which needs its own soundness lemma) or
prove the terminal relation field-native (PR5). Until that decision, do
not extend source-image plumbing to internal F' field values. The
current scope is the right stopping point.
