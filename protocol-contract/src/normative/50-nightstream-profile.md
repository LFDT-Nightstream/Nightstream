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

The verifier-key relation artifact MUST contain exactly 6,377,555 logical
rows. Its full committed carrier for `z=x||w` MUST contain 264,627,486 fields,
or 4,900,509 complete Phi81 ring columns. The first 270 fields are `x`. The
profile has one fresh claim and 16 running claims. Source order MUST be fresh
claim 0 followed by running claims 0 through 15. A proof-supplied shape MUST
NOT select these values.

Decision: NSD-DOMAIN-001 and NSD-CARRIER-001.

### NS-SHAPE-PADDING — Common row-cube injection

The verifier MUST use a 28-variable Boolean row cube. Logical relation rows
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

The v1 commitment MUST have `kappa=22` rows and exactly 4,900,509 message
columns in `R_F`. It MUST use `c_a=sum_j A_(a,j)z_j` as a left matrix-vector
product. Setup MUST use `nightstream-ajtai-chacha20-wide256-v1` with the
verifier-owned seed
`fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702`.
One RFC-8439 ChaCha20 block MUST be indexed by nonce
`row_u32_le || block_u64_le` and counter `lane_u32`. The first 256 output
bits MUST be interpreted as one little-endian integer and reduced modulo the
Goldilocks prime. There is no rejection, retry, fallback, materialized full
key, transpose, blinding column, commitment randomness, or affine term. The
verification key MUST bind the seed, dimensions, setup ID, 256-bit reduction
rule, and 22-ring commitment encoding through the recomputed Poseidon2
verifier context. The selected post-allowance Module-SIS estimate is about
110 bits; this estimate is not a formal theorem.

Decision: NSD-ENCODING-001 and NSD-AUTHORITY-001.
