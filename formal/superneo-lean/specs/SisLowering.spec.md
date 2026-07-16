# SIS low-norm lowering contract

This component specifies the exact model-level representation of source-field
SIS semantics by canonical centered-unit words. It covers deterministic
composition through two SIS maps and a Poseidon2 envelope, but it does not
assert Module-SIS hardness, collision resistance, or implementation
conformance.

## Fixed encoding

| Parameter | Value | Meaning |
|---|---:|---|
| Field modulus | Goldilocks `q` | Canonical source-field range |
| Digit count | 41 | Width of one SIS word |
| Radix | 3 | Ordinary trit radix |
| Centered alphabet | `{-1, 0, 1}` | Low-norm SIS coefficients |
| Shift | `(3^41 - 1) / 2` | All-ones offset between centered units and trits |

For a canonical source field `x`, the word is the 41-digit ordinary base-three
expansion of

```text
N = (x + shift) mod q,
```

with each trit translated by `d = t - 1`. The source-row predicate requires
the fixed word length, `N < q`, and reconstruction of `x` after removing the
shift.

## Required theorem surface

| Result | Guarantee |
|---|---|
| `decodeNat_encodeNat_of_lt` | Covered unsigned values round-trip through 41 trits |
| `encodeNat_decodeNat_of_length` | Every fixed-width centered-unit word has one unsigned encoding |
| `decodeWord_canonicalWord` | A canonical low-norm word recovers its source field |
| `sourceFieldRows_iff_canonicalWord` | Alphabet, reconstruction, and canonicality select exactly one word |
| `sourceRows_unique_decoding` | Ordered field sequences have one word sequence and decode back exactly |
| `sourceSisRows_iff_lowNormEncoding` | Source SIS semantics equal canonical low-norm semantics |
| `sourceSisRows_unique` | Fixed source fields determine both the message and recomputed binding output |

## Structural SIS and envelope model

The first SIS map consumes the flattened centered-unit words. The short SIS
map consumes the first map's output. The final envelope is a deterministic
function of:

```text
(role, field_count, primary_rank, short_map_output).
```

These functions are semantic parameters at this layer. Equality in the model
means recomputation from the canonical message; it does not supply a binding
or collision-resistance theorem.

## Concrete refinement obligations

A production lowering theorem must establish all of the following:

1. Each field's emitted centered-unit, reconstruction, and borrow rows are
   sound and complete for `SourceFieldRows`.
2. Rust and Lean use the same Goldilocks representatives, 41-digit
   least-significant-first order, shift, and canonical `< q` comparison.
3. Every source-field/digit alias is total, shape-correct, non-overlapping,
   and references the same authoritative digits used by the SIS map.
4. Every source column represented by an alias, binary slot, or decoded
   full-field slot has the same value in every retained row, and no source
   reference escapes the validated substitution map.
5. `SeededPhi81LinearBlock` word starts, word width, row-major padding,
   dimension, rank, seeds, rejection sampling, and `Phi_81` rotations
   instantiate `Pipeline.primaryMap` and `Pipeline.shortMap` exactly.
6. The primary and short maps use the protocol-prescribed independent domains
   and ranks.
7. The Poseidon2 preimage uses the exact v4 domain, role, field count, primary
   rank, and recomputed short-map output in production order.
8. Poseidon2 rows are sound and complete for the concrete envelope function;
   no prover-carried digest or commitment replaces recomputation.
9. Any source-row removal or low-norm materialization preserves every public
   output and all downstream row references.

Only a theorem discharging these obligations may claim Rust/R1CS conformance.
