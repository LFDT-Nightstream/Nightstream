## 5. Nightstream v1 profile

### NS-ALGEBRA-PROFILE — Goldilocks and Phi81 selection

The v1 profile MUST use

```text
q = 2^64-2^32+1
K_ext = F_q[U]/(U^2-7)
Phi = X^54+X^27+1
d = 54, b = 2, k_rho = 16, B = 65536.
```

Decision: NSD-DOMAIN-001 and NSD-ENCODING-001.

### NS-SHAPE-LOGICAL — Selected logical shape

The verifier-key relation artifact MUST supply the exact positive logical row
count and the exact full committed assignment width `m` for `z=x||w`, with
`m` a multiple of 54 and at most 16,777,206 fields, or 310,689 complete Phi81
ring columns. The first 270 fields are `x`. The profile has one fresh claim
and 16 running claims. Source order MUST be fresh claim 0 followed by running
claims 0 through 15. A proof-supplied shape MUST NOT select these values.

Decision: NSD-DOMAIN-001 and NSD-CARRIER-001.

### NS-SHAPE-PADDING — Common row-cube injection

The verifier MUST use a 24-variable Boolean row cube. Logical relation rows
and assignment coordinates MUST occupy their zero-based prefixes in
little-endian Boolean index order. Every unused relation row and every cube
position after the assignment prefix MUST be zero.

Decision: NSD-DOMAIN-001 and NSD-DOMAIN-MAP-001.

### NS-SHAPE-IDENTITY — Padded identity matrix

For the artifact-owned assignment width `m`, the protocol MUST prepend the
implicit padded matrix `M_0=[I_m;0]` to the 13 application matrices. A
verifier key with another identity width or matrix count MUST reject.

Decision: NSD-DOMAIN-MAP-001 and NSD-NORM-BINDING-001.

### NS-SHAPE-POLYNOMIAL — Lifted application polynomial

The structure MUST have 14 matrix inputs and total polynomial degree 8. Its
polynomial MUST be `f_v1(u_0,u_1,...,u_13)=f_app(u_1,...,u_13)` and therefore
MUST ignore the new identity input while it preserves every dependency of
`f_app`.

Decision: NSD-DOMAIN-MAP-001 and NSD-PICCS-001.

### NS-PUBLIC-CARRIER — Ring-aligned public input

`x` MUST contain five consecutive ring elements, or 270 base fields. A fresh
input MUST put its 257 logical fields first and MUST set fields 257 through
269 to zero. A carried input owns all 270 fields.

Decision: NSD-CARRIER-001 and NSD-AUTHORITY-001.

### NS-SPLIT-BINARY — Deterministic signed-bit split

For a centered scalar `a` with `abs(a)<2^16`, let `s=-1` when `a<0` and
`s=1` otherwise. Child `h` MUST be `s*bit_h(abs(a))` for `h=0..15`. The
algorithm MUST apply coordinate-wise, encode `-1` as `q-1`, and reject an
out-of-bound prover assignment before it emits children. The verifier MUST
apply the same error rule only to the public parent input; it cannot inspect
private assignment coordinates.

Decision: NSD-SPLIT-001.

### NS-COMMITMENT-PROFILE — Ajtai commitment key

The v1 commitment MUST have `kappa=18` rows and exactly `m/54` message columns
in `R_F`, where `m` comes from the verifier-key relation artifact. It MUST use
`c_a=sum_j A_(a,j)z_j` as a left matrix-vector product. Setup MUST use
`nightstream-ajtai-chacha8-setup-par-v1` from
`src/profile/ajtai-setup-v1.toml`: derive row and fixed-size chunk seeds in
order, then fill matrix entries in row, column, coefficient order. For each
ring entry, read 54 little-endian `u64` values before replacement sampling;
rejected values are replaced in coefficient order. A transpose, blinding
column, commitment randomness, or affine term is not part of v1. The verifier
key MUST bind the 32-byte setup seed, dimensions, setup ID, and 18-ring
commitment encoding through a recomputed Poseidon2 digest.

Decision: NSD-ENCODING-001 and NSD-AUTHORITY-001.
