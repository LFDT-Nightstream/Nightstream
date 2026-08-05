## 6. Nightstream PaddedRowIdentity PiCCS

### NS-PICCS-VARIANT — Joint row-domain protocol

Nightstream MUST use the reviewed joint PiCCS polynomial on the selected
24-variable row cube. In its norm term, `MLE(z_i)` MUST mean the 24-variable
MLE of `z_i` followed by zero padding. It MUST NOT dispatch to a rectangular
FE/NC split.

Decision: NSD-PICCS-001 and NSD-NORM-BINDING-001.

### NS-PICCS-PADDING-EQUIVALENCE — Zero-row refinement

For every application matrix, the padded output MUST equal its logical output
on rows `0..14944218` and zero afterwards. The Structure MUST satisfy
`f_app(0,...,0)=0`. For `M_0`, the constant-term projection of its ring output
MUST equal `z` on coordinates `0..11437037` and zero afterwards. The logical
and padded CCS relations MUST accept the same `(x,w)` pair.

Decision: NSD-DOMAIN-MAP-001 and NSD-NORM-BINDING-001.

### NS-PICCS-COINS — Exact PiCCS challenge family

PiCCS MUST use one `alpha in K_ext^24`, one `gamma in K_ext`, and one
`K_ext` SumCheck challenge per round. `beta_a`, `beta_r`, `beta_m`, and all
other PiCCS batching coins MUST be absent.

Decision: NSD-BATCH-COINS-001 and NSD-TRANSCRIPT-001.

### NS-PICCS-SUMCHECK — Exact round shape

The joint polynomial has `D_Q=9`. The proof MUST contain exactly 24 round
polynomials, each encoded as 10 extension coefficients from degree 0 through
9. The verifier MUST run SN-SUMCHECK-ROUNDS and reject a missing, extra, or
noncanonical coefficient.

Decision: NSD-PICCS-001, NSD-ENCODING-001, and NSD-TRANSCRIPT-001.

### NS-PICCS-TERMINAL — Exact output family

At the final SumCheck point, the proof MUST contain one `R_K` evaluation for
each of 15 sources and 14 matrices, in source-major then matrix-major order.
`F` uses the new outputs for fresh source 0. `N` uses the new `M_0` output for
sources 0 through 14. `E` uses the new outputs for running sources 1 through
14 and `eq(r_new,r_old)`; `T_abs` uses those running sources' input evaluations
at `r_old`. The verifier MUST derive these terminal values and check the
reviewed joint equation.

Decision: NSD-PICCS-001, NSD-ENCODING-001, and NSD-AUTHORITY-001.

### NS-PICCS-NORM-BINDING — Norm bound from M_0 output

For source `i`, the norm terminal MUST read `ct(y_(i,0))`. Because
`M_0=[I;0]`, this value is `MLE(z_i||0)(r_new)` for the same `z_i` that
satisfies `c_i=L(z_i)`. A separate or optional norm opening MUST reject.

Decision: NSD-NORM-BINDING-001 and NSD-AUTHORITY-001.

### NS-PICCS-NO-COLUMN — No column authority branch

The CE relation and proof MUST NOT contain `y_carrier`, `s_col`, `y_zcol`, a
column terminal, a column point, or a column replay. A verifier optimization
MAY recompute a public-carrier evaluation from `x`, but it MUST NOT carry that
value into relation authority.

Decision: NSD-COLUMN-001 and NSD-COLUMN-MAP-001.

### NS-PICCS-CENSUS — Selected algebraic planning count

For `K_fresh=1`, `k=14`, `t=14`, `d=54`, `ell=24`, and `D_f=8`, the profile
MUST derive

```text
D_Q = 9
N_SC = 9*24 = 216
D_SZ = max(24,39,10599) = 10599
N_field = 10815
coordinate-fork numerator = K_fresh+k+1 = 16.
```

These values are component counts, not an end-to-end security level.

Decision: NSD-SECURITY-001 and NSD-DOMAIN-MAP-001.
