## PAPER-FND — fields, rings, dimensions, and bounds

`PAPER-FND-001` Let `F` be the prime field of order `q`. Let `K` be the
lowest-degree extension of `F` with negligible inverse size. Identify `F` as a
subfield of `K`.

`PAPER-FND-002` Let `Phi` be a degree-`d` cyclotomic polynomial. Define
`R_F=F[X]/(Phi)` and `R_K=K[X]/(Phi)`.

`PAPER-FND-003` The dimensions satisfy:

```text
n_F = d*n_R
n_F,in = d*n_R,in
```

The structure has `m` constraints, `t` matrices, and a total-degree ceiling
`u`. A fold has `K` fresh claims and `k` carried claims.

`PAPER-FND-004` The protocol bounds satisfy `B=b^k<q/2`. The universal ambient
bound is `B_amb=floor(q/2)+1`. The ambient bound is used only in extraction.

Source: `SRC-PAPER-04`, lines 9-21.

## PAPER-NORM — centered values and split

`PAPER-NORM-001` A field value uses the centered representative in
`[-floor(q/2),floor(q/2)]`. Field-vector and ring-vector norms are the largest
absolute coefficient. Thus every field value has norm strictly below `B_amb`.

`PAPER-SPLIT-001` If `norm(z)<b^k`, `split_b(z)` returns exactly `k` vectors
with:

```text
z = sum_(h=0)^(k-1) b^h*z_h
norm(z_h) < b for every h.
```

The paper states these properties. It does not select one deterministic digit
algorithm.

Source: `SRC-PAPER-04`, lines 23-41.

## PAPER-COM — commitment assumptions

`PAPER-COM-001` `Commit(pp,-):R_F^n_R -> Cmt` is an `R_F`-module homomorphism.
The fold needs `(2B,C)`-relaxed binding, where `C` is a strong sampling set with
expansion factor `T`. The global guard is:

```text
(K+k)*T*(b-1) < B.
```

`PAPER-COM-002` A strong sampling set requires every difference of two distinct
members to have norm below the low-norm invertibility bound. Its expansion
factor is the maximum norm growth under multiplication by a member of the set.

`PAPER-COM-003` The concrete Ajtai commitment is a random ring-matrix map.
Theorem 2 gives `(B,C)`-relaxed binding from
`MSIS_(m,4*T*B)^(infinity,kappa,q)`. Definition 14 needs `(2B,C)`-relaxed
binding. The concrete Appendix B substitution is therefore
`MSIS_(m,8*T*B)^(infinity,kappa,q)`.

Source: `SRC-PAPER-04`, lines 43-64; `SRC-PAPER-07`, lines 33-43; and
`SRC-PAPER-12`, lines 15-28.

## PAPER-SET — strong sampling sets and low-norm invertibility

`PAPER-SET-001` Theorem 8 gives the low-norm invertibility bound. Select an
integer `z` that divides `eta`, with `q=1 mod z` and `ord_eta(q)=eta/z`. Set
`tau(z)=z` for odd `z` and `tau(z)=z/2` for even `z`. Then:

```text
b_inv = q^(1/phi(z)) / sqrt(tau(z)).
```

Every `a` in `R_F` with `0<norm(a)<b_inv` is invertible in `R_F`.

`PAPER-SET-002` Definition 17 defines a strong sampling set `C subset R_F`.
Each difference of two distinct members must satisfy `norm(a-b)<b_inv`. The
expansion factor is the largest value of `norm(rho*v)/norm(v)` for `rho` in `C`
and `v` in `R_F`. Theorem 9 bounds it:

```text
T <= 2*phi(eta)*max_(rho in C) norm(rho).
```

Source: `SRC-PAPER-12`, lines 17-23.

## PAPER-MSIS — Module-SIS and the Ajtai instantiation

`PAPER-MSIS-001` Definition 16 defines `MSIS_(m,B)^(infinity,kappa,q)`. The
challenger samples `M` uniformly from `R_F^(kappa x m)`. The solver must find a
nonzero `z` in `Z[X]/(Phi)` with `Mz=0 mod q` and `norm(z)<B`. Definition 18
defines the Ajtai scheme: `Setup` samples `M`, and `Commit(pp,z)=Mz`.

The length parameter `m` of the Module-SIS problem is the commitment message
length in ring elements. It is not the CCS constraint count.

Source: `SRC-PAPER-12`, lines 15-28.
