# Canonical fixed-active PiRLC sampler program

Owner: `Nightstream/Implementation/R1CS/Canonical/PiRlcCanonicalSamplerProgram.lean`

Assurance tier: model-proved canonical encoding. This is not current-production
correspondence and is not a complete `nifsVerify` recipe.

## Selected encoding

The program fixes the 15-coordinate active sampler and emits, in order:

| family | rows | auxiliary columns |
|---|---:|---:|
| 75-call symbolic Poseidon2 transcript | 26,400 | 26,400 |
| 15 canonical-u64 decoders | 16,560 | 15,840 |
| 15 × 64 sampler candidates | 24,000 | 21,120 |
| 15 first-accepted selectors | 38,970 | 36,525 |
| **total** | **105,930** | **99,885** |

The totals are derived from the emitted row list and the concatenated allocation
list:

```lean
PiRlcCanonicalSamplerProgram.rows_length
PiRlcCanonicalSamplerProgram.allocation_length
```

The auxiliary count includes each Poseidon2 call's 344 S-box temporaries and
eight output columns. The earlier 99,285 subtotal omitted those 600 output
columns and is not the cost of this program.

## Physical guarantees

- `allocation_nodup` proves the declared transcript and sampler allocations are
  pairwise duplicate-free.
- `ownership_is_positional` proves the emitted rows are exactly the image of a
  duplicate-free structured receipt list.
- `PiRlcCanonicalU64Placement.laneInput_member_temporaryColumns` binds each u64
  lane read to the exact symbolic-transcript allocation.
- `PiRlcCanonicalSamplerProgramConservation.rows_conservation` proves every
  operand of every emitted row is either caller-owned authoritative input below
  `duplexBase` or a member of the exact declared allocation.
- `rows_complete` constructs one honest satisfying assignment when the exact
  sampler no-shortfall condition holds.
- `PiRlcCanonicalSamplerSound.outputs_eq_firstAccepted` gives the independent
  first-accepted semantic result from satisfaction of the three sampler suffix
  families.

These statements do not claim that every declared column is written. They prove
the exact row count, duplicate-free declared allocation, structured row
receipts, honest satisfiability, and absence of arbitrary column reads.

## Explicit exclusions

This result does not provide:

- source binding to a complete selected NIFS transcript;
- PiCCS, PiDEC, Ajtai, accumulator, or complete `nifsVerify` constraints;
- the application-owned `step` recipe;
- a complete F-prime canonical program or its total cost;
- Rust-program equality, current-row equality, or generated artifacts;
- Fiat–Shamir, collision-resistance, probability, or end-to-end security
  claims.

The current Rust implementation and historical artifacts were not used as
semantic or row authority.

