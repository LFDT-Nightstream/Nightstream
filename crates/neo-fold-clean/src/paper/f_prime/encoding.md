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

## Open design questions for `enc(F')`

These need an answer before `enc(F')` can be implemented:

- What exactly is the low-norm assignment `z = [x, w]` for an F'
  execution?
- Which values are public `x`? Which values are private `w`?
- Which field computations are represented by bits/digits in `w`?
- Which values are merely derived linear combinations and never
  committed directly?
- How does this interact with the cost wall measured earlier (F' ≈ 10M
  rows post-optimization; naive lowering of all field vars was ~3.2B
  bit slots)?

Until those are settled, do not extend source-image plumbing to internal
F' field values. The current scope is the right stopping point.
