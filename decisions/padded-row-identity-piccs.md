# One-joint padded-row PiCCS

**Status:** Accepted

## Problem

Nightstream has rectangular CCS matrices. The PiCCS reference and optimized
implementations must use one protocol so that direct cross-checks can detect a
protocol change. A separate row SumCheck and column SumCheck adds a second
point and a column-output lifecycle that SuperNeo does not use.

## SuperNeo

Section 7.3 permits `n_F <= m`, defines the padded identity
`M_1 = [I; 0]`, and runs one joint SumCheck on the `log m` row cube. It fixes
one `alpha`, one `gamma`, the strict centered norm polynomial, the absolute
gamma target, one terminal point, and full ring-valued matrix outputs.

The reviewed paper files used for this decision have these SHA-256 values:

- Section 5: `43d084a85ae746b74485c1132e181000dd8b14218f3d8c0536b4d730e2eaeca5`
- Section 7: `46a68727c2abfb5b856517a831bfa8b6f625bf508ae9dd9694a9b33e2e49fbde`
- Appendix D: `37131dd724623d9599ff222c7f143182e04fc73e8dae69cb954c7cb253dd24cf`

## Decision

Use one power-of-two row cube that covers both the application row count and
the complete committed assignment. Add a virtual identity matrix before the
application matrices. The application constraint polynomial ignores this new
input.

Nightstream pads unused rows with zero rows instead of duplicate rows. This is
valid only when the application constraint polynomial is zero on the all-zero
matrix-output vector. The implementation must reject a structure that does not
have this property.

The Nightstream compiler pads the public-input prefix to complete degree-54
ring slots. It marks the added coordinates as public zeros. This is a relation
parameter restriction, not a change to PiCCS. It prevents a public ring slot
from also containing private witness coefficients.

The protocol has no column challenge, second SumCheck, `s_col`, or `y_zcol`.
PiCCS outputs the full ring evaluation of the virtual identity and every
application matrix. PiRLC and PiDEC operate on all these ring values.

The Rust protocol name is `PaddedRowIdentity`. Fiat--Shamir framing and proof
bytes are a versioned Nightstream transport profile; they do not change the
paper algebra or interactive messages.
