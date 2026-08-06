## 1. Reviewed SuperNeo foundations

### SN-FND-FIELD — Base and extension fields

`F` MUST be the prime field of order `q`, and `K_ext` MUST contain `F`.

Source: PAPER-FND-001.

### SN-FND-RING — Cyclotomic rings

For a selected degree-`d` cyclotomic polynomial `Phi`, the rings MUST be
`R_F=F[X]/(Phi)` and `R_K=K_ext[X]/(Phi)`.

Source: PAPER-FND-001 and PAPER-FND-002.

### SN-FND-DIMENSIONS — Logical coefficient dimensions

The logical dimensions MUST satisfy `n_F=d*n_R` and
`n_F,in=d*n_R,in` before a concrete profile adds any padding.

Source: PAPER-FND-002 and PAPER-FND-003.

### SN-NORM-CENTERED — Centered coefficient norm

A field value MUST use its unique centered integer representative. A vector or
ring norm MUST be the largest absolute centered coefficient.

Source: PAPER-NORM-001.

### SN-NORM-BOUNDS — Strict protocol and ambient bounds

`Bound_a(z)` means `norm(z)<a`. The protocol bound MUST be `B=b^k<q/2`.
The ambient extraction bound MUST be `B_amb=floor(q/2)+1` and MUST NOT replace
`B` in a protocol guard or commitment parameter.

Source: PAPER-FND-004, PAPER-NORM-001, ERR-NORM-ROOTS, and ERR-AMBIENT.

### SN-SPLIT-ABSTRACT — Radix decomposition relation

For each `z` with `Bound_(b^k)(z)`, `split_b(z)` MUST return exactly `k`
ordered vectors with

```text
z = sum_(h=0)^(k-1) b^h*z_h
Bound_b(z_h) for every h.
```

Source: PAPER-SPLIT-001.

### SN-EMBED-COEFFICIENTS — Coefficient embedding order

Field vectors MUST map to ring vectors in consecutive `d`-coefficient blocks.
The inverse map, input projection, commitment, and evaluation MUST use the
same order.

Source: PAPER-EMB-001 and PAPER-EMB-002.

### SN-EMBED-MODULE-ACTION — Ring action in linear combinations

An `R_F` coefficient in PiRLC MUST act through the induced `R_F`-module
action. Scalar-only field mixing is not the paper operation.

Source: PAPER-EMB-003.

### SN-REL-STRUCTURE — Verifier-owned CCS structure

A Structure is one immutable pair `s=({M_j}_{j=0}^{t-1},f)`. Each matrix has
shape `m*n_F`. The verifier key MUST own the matrices, dimensions, coefficient
order, and `f`. Define `D_f=deg_tot(f)` and require `D_f<=u`.

Source: PAPER-REL-001, ERR-DEGREE-SEMANTICS, and ERR-DIM-DEGREE.

### SN-REL-CCS — Norm-bounded CCS membership

For `z=x||w`, `CCS(a,L)` holds exactly when `c=L(z)`, `x=L_in(z)`,
`Bound_a(z)`, and
`f(bar(M_0 z),...,bar(M_(t-1) z))` vanishes on the Boolean row cube.

Source: PAPER-REL-002 and ERR-CCS-ZERO-SET.

### SN-REL-CE — Norm-bounded evaluation and shared-point batch

`CE(a,L)` holds exactly when `c=L(z)`, `x=L_in(z)`, `Bound_a(z)`, the point
`r` has the row-cube arity, and
`y_j=MLE(bar(M_j)z)(r)` in `R_K` for every matrix.

`BatchCE_N(a,L)` MUST be the subset of `CE(a,L)^N` in which all components
use one shared Structure and one shared point `r`. Ordinary `CE(a,L)^N`
retains its Cartesian-product meaning and MUST NOT be used as a shared-point
batch relation.

Source: PAPER-REL-003, PAPER-BATCH-001, ERR-LIN-DOMAIN, ERR-CE-TYPES,
ERR-EVALUATION-NOTATION, and ERR-SHARED-POINT.

### SN-GLOBAL-STRONG-SET — Strong-set requirement

The PiRLC challenge set `C subset R_F` MUST have a declared expansion factor
`T`, negligible inverse size, and the strong-set property from Definition 17.

Source: PAPER-COM-001, PAPER-SET-001, and PAPER-SET-002.

### SN-GLOBAL-NORM-GUARD — PiRLC norm-growth guard

The selected parameters MUST satisfy
`(K_fresh+k)*T*(b-1)<B` and `B=b^k<q/2`.

Source: PAPER-COM-001 and PAPER-COM-002.

### SN-GLOBAL-COMMITMENT — Commitment properties

`Commit` MUST be an `R_F`-module homomorphism and satisfy the paper
`(2B,C)`-relaxed-binding requirement.

Source: PAPER-COM-001 through PAPER-COM-003.

### SN-STRONGSET-DIVISOR — Theorem 8 divisor conditions

The profile MUST fix the Theorem 8 divisor `z` and establish
`z|eta`, `q=1 mod z`, and `ord_eta(q)=eta/z`.

Source: PAPER-SET-001.

### SN-STRONGSET-DIFFERENCE — Strong-set difference bound

With `b_inv=q^(1/phi(z))/sqrt(tau(z))`, every difference of two distinct
members of `C` MUST have norm strictly below `b_inv`.

Source: PAPER-SET-001 and PAPER-SET-002.

### SN-STRONGSET-EXPANSION — Derived expansion factor

The expansion factor MUST be derived from
`T<=2*phi(eta)*max_(rho in C) norm(rho)`. A stored constant alone is not the
derivation.

Source: PAPER-SET-002.

### SN-MSIS-PARAMETERS — Exact Module-SIS assumption

The concrete binding assumption MUST name `kappa`, the commitment width in
ring elements, `q`, the matrix-generation rule, and the infinity-norm bound
`8*T*B`. The commitment width is not the CCS row count `m`.

The Appendix D.7 analytic diagnostic MUST use
`2^(2*sqrt(kappa*d*log2(q)*log2(delta)))`. It MUST NOT move either logarithm
outside the square root or present this diagnostic as end-to-end security.

Source: PAPER-MSIS-001, PAPER-COM-003, PAPER-D7-001,
PAPER-CONFLICT-002, and ERR-D7-SQRT-SCOPE.
