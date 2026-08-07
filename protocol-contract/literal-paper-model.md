# Reviewed SuperNeo paper model

> Generated reading view. Edit `src/paper/`;
> `refresh_derived.py` rebuilds this file.


Status: **non-normative source extraction**. This file records the reviewed
paper snapshot in `sources.md`. It does not add Nightstream behavior.

Paper indices start at one. Contract indices start at zero.

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
## PAPER-RED — interactive reductions and composition

`PAPER-RED-001` Definition 5 defines an interactive reduction as the PPT
algorithms `(G,K,P,V)`. `G(1^lambda,sz)` outputs `pp`. `K(pp,s)` deterministically
outputs `(pk,vk)`. `P` and `V` interact and output the verifier instance `u_2`
and the prover witness `w_2`.

The reduction is a reduction of knowledge when it has all three properties:

1. completeness: an honest run on `(pp,s,u_1,w_1) in R_1` gives equal prover and
   verifier output instances, and its result is in `R_2`;
2. knowledge soundness: for every EPT adversary with success probability at
   least `1/poly(lambda)`, an EPT extractor returns `w_1` with
   `(pp,s,u_1,w_1) in R_1` with probability at least the success probability
   minus a negligible term;
3. public coin: every verifier message is a uniformly random string, and the
   verifier messages contain all verifier randomness.

`PAPER-RED-002` Lemma 2 gives sequential composition. For reductions of
knowledge `Pi_1:R_1->R_2` and `Pi_2:R_2->R_3` that share `G` and `K`, the
composition `Pi_2 o Pi_1:R_1->R_3` is a reduction of knowledge.

`PAPER-RED-003` Definition 9 defines a weak reduction `Pi:R_1->R_2` for
relations with `R_1 subset R'_1` and a function `phi` on the ambient instance
space of `R_1`. The reduction must be complete and public coin. For each EPT
adversary, an EPT extractor must satisfy both conditions:

1. it returns `w_1` with `(pp,s,u_1,w_1) in R'_1` with probability at least the
   adversary success probability minus a negligible term;
2. when the adversary gives two input instances with equal `phi` image, the
   extractor returns two different witnesses with at most negligible
   probability.

`PAPER-RED-004` Definition 10 defines a strong reduction `Pi:R_1->R_2` for
relations with `R_2 subset R'_2` and a function `phi` on the ambient instance
space of `R_2`. The reduction must be complete and public coin, and:

1. for every EPT adversary, two independent runs of the same prover give output
   instances with equal `phi` image with probability one;
2. for every EPT adversary whose relaxed success probability for `R'_2` is at
   least `1/poly(lambda)`, and whose two runs give different output witnesses
   with at most negligible probability, an EPT extractor returns `w_1` with
   `(pp,s,u_1,w_1) in R_1` with probability at least that relaxed success
   probability minus a negligible term.

`PAPER-RED-005` Theorem 6 gives strong-weak composition. Let `R_2 subset R'_2`.
Let `Pi_1:R_1->R_2 (R'_2)` be strong for `phi`, and let
`Pi_2:R_2 (R'_2)->R_3` be weak for the **same** `phi`. Then `Pi_2 o Pi_1` is a
reduction of knowledge.

The proof of Theorem 6 uses the strong condition (i) of `Pi_1` to supply the
`phi`-agreement premise of `Pi_2`, and the extractor-agreement conclusion of
`Pi_2` to supply the witness-agreement premise of `Pi_1`. Neither direction is
optional.

Source: `SRC-PAPER-04`, lines 66-95, and `SRC-PAPER-06`.

## PAPER-FOLDING — folding-scheme definition

`PAPER-RED-006` Definition 19 defines a folding scheme for a relation `R` as a
reduction of knowledge of type `R * R_ACC -> R_ACC`.

Source: `SRC-PAPER-12`, line 32.

## PAPER-FORK — special sets and coordinate-wise extraction

`PAPER-FORK-001` Definition 20 defines the special set `SS(C,ell)`. It contains
each tuple of one base challenge vector and `ell` neighbours, where neighbour
`i` differs from the base vector in coordinate `i` only.

Theorem 10 gives coordinate-wise extraction. For a challenge space `C^ell` and
an adversary with success probability `eps`, an EPT extractor makes at most
`ell+1` queries in expectation. With probability at least `eps-ell/abs(C)`, it
returns `ell+1` accepted pairs whose challenge vectors form a special set. The
theorem also gives the weaker bound `eps-(ell+1)/abs(C)`, and Appendix D.5 uses
that weaker form.

Source: `SRC-PAPER-12`, lines 34-56.

## PAPER-SUMCHECK — public-coin polynomial sum

`PAPER-SUMCHECK-001` For an `ell`-variable polynomial with maximum individual
degree at most `D`, SumCheck checks a claimed Boolean-cube sum `T`. It outputs
a random point `r`, a claimed value `v`, and the terminal claim `v=Q(r)` for
the verifier to check. The paper states that the protocol is public coin, has
zero completeness error, and has soundness error at most
`ell*D/abs(K_ext)` over the extension field. It refers to an external note for
the full round protocol.

Source: `SRC-PAPER-04`, line 97.
## PAPER-EMB — coefficient embedding

`PAPER-EMB-001` Partition a field vector into consecutive blocks of `d`
coefficients. Embed each block into one ring element. The inverse map preserves
this order.

`PAPER-EMB-002` The transformed inner product satisfies
`ct(bar(a)*bar(b))=<a,b>`. The transform extends to vectors and matrices. Thus
`Mz=ct(bar(M)z)`.

`PAPER-EMB-003` Commitment, input projection, and multilinear evaluation
commute with the ring linear combinations used by PiRLC.

Source: `SRC-PAPER-05`.

## PAPER-REL — Structure, CCS, and CE

`PAPER-REL-001` A Structure is `s=({M_j},f)`. Each matrix has shape
`m*n_F`. Define `D_f=deg_tot(f)` and require `D_f<=u`.

`PAPER-REL-002` For `z=x||w`, `CCS(b,L)` holds when:

```text
c = L(z)
norm(z) < b
f(bar(M_1 z),...,bar(M_t z)) vanishes on the Boolean row cube.
```

`PAPER-REL-003` `CE(b,L)` holds for `(s;c,x,r,{y_j};z)` when:

```text
c = L(z)
x = L_in(z)
norm(z) < b
y_j = MLE(bar(M_j)z)(r) for each j.
```

The point `r` has `log m` extension-field coordinates. `L_in` projects the
first `n_R,in` ring columns under coefficient embedding.

Source: `SRC-PAPER-07`, lines 3-29.

## PAPER-BATCH — relation powers and shared evaluation points

`PAPER-BATCH-001` Appendix C defines `R^N` as the ordinary Cartesian product of
`N` relation instances. Since `r` is part of each CE instance, the literal
type `CE(a,L)^N` permits independent points `r_1,...,r_N`. The protocol text,
however, uses one unindexed point in each batch: PiCCS has one `eq(X,r)`, its
outputs share `r'`, PiRLC retains one `r`, and PiDEC copies one `r` to every
child. This is a formal type mismatch in the printed paper.

Source: `SRC-PAPER-02`, lines 13-27; `SRC-PAPER-07`, PiCCS, PiRLC, and PiDEC;
and `SRC-PAPER-12`, line 3.
## PAPER-PICCS — reviewed joint PiCCS

`PAPER-PICCS-001` The paper assumes that `m` is a power of two, `n_F<=m`, and
the first matrix is the canonical injection `M_1=[I_(n_F);0]`. PiCCS reduces
`CCS(b,L)^K * CE(b,L)^k` to `CE(b,L)^(K+k)`.

The paper permits one common row cube when the logical row count and the
assignment width differ. It normalizes an original relation by choosing
`m>=max(m',n_F)`, prepending the padding injection, and using either repeated
constraint rows or zero rows when `f'(0,...,0)=0`. In both cases, PiCCS uses
one `log m`-variable SumCheck. It does not require `m=n_F`.

These are the literal printed relation powers. Under `PAPER-BATCH-001`, they are
too broad because the displayed protocol has one carried point `r` and one
output point `r'` shared by every component.

The verifier samples `alpha in K^(log m)` and `gamma in K`. Let zero-based
indices be `a in [0,K)`, `i in [0,K+k)`, `c in [0,k)`, `j in [0,t)`, and
`l in [0,d)`. Define:

```text
P_b(V) = product_(h=-(b-1))^(b-1) (V-iota_q(h))
I(c,j,l) = c + k*j + k*t*l

F(X) = sum_a gamma^a * f(bar(M_0 z_a)(X),...,bar(M_(t-1) z_a)(X))

NC(X) = sum_i gamma^i * P_b(tilde(z_i)(X))

Eval_local(X) = eq(X,r)
  * sum_(c,j,l) gamma^I(c,j,l)
    * MLE(cf(bar(M_j z_(K+c)))_l)(X)

Q(X) = eq(X,alpha)*(F(X)+gamma^K*NC(X))
       + gamma^(2K+k)*Eval_local(X)

T_abs = sum_(c,j,l) gamma^(2K+k+I(c,j,l))*cf(y_(K+c,j))_l.
```

The strict norm polynomial has degree `2b-1`. The accepted individual
SumCheck degree is `D_Q=max(D_f+1,2b,2)`.

`PAPER-PICCS-002` The verifier runs one SumCheck for
`T_abs=sum_X Q(X)`. At terminal point `r'`, the prover sends each ring value
`y'_(i,j)=MLE(bar(M_j)z_i)(r')`. The verifier checks the terminal `Q` equation
and outputs the `K+k` CE claims at `r'`.

`PAPER-PICCS-003` The three gamma blocks are disjoint:

```text
fresh:   0 .. K-1
norm:    K .. 2K+k-1
carried: 2K+k .. 2K+k+k*t*d-1.
```

`PAPER-PICCS-004` Lemma 7 gives the exact characterization. For arbitrary
vectors, point, and claimed evaluations, the polynomial identity
`T_abs(C)=sum_x Q(x,A,C)` holds if and only if all three items hold:

1. `f(bar(M_1 z_i),...,bar(M_t z_i))` vanishes on the Boolean cube for each
   `i in [K]`;
2. `norm(z_i)<b` for each `i in [K+k]`;
3. `y_(i,j)=MLE(bar(M_j)z_i)(r)` for each carried `i` and each `j in [t]`.

The proof needs the disjoint exponent blocks, linear independence of the powers
of `C`, Lemma 6, and the scalar equivalence
`P_b(a)=0 iff abs(ctr_q(a))<b`. All three items use the **same** vectors `z_i`.

`PAPER-PICCS-005` The joint protocol requires `m` to be a power of two,
`n_F<=m`, and `M_1=[I_(n_F);0]`. The padding-injection hypothesis carries the
norm item of Lemma 7 to the verifier. The terminal opening gives
`ct(y'_(i,1))=MLE(z_i||0)(r')`, and `y'_(i,1)` is a field of the output CE
instance. The norm check therefore reads the same vector that the output CE
relation binds to the commitment `c_i`.

Remark 3 states this rule. Without the padding-injection hypothesis, the paper
gives no other rule that binds the norm-check opening to the committed witness.

`PAPER-PICCS-006` Lemma 3 states that `Pi_CCS` is strong for the function `phi`
that projects the commitments `(c_i)` from the instance. Its relation types are:

```text
R_1  = CCS(b,L)^K * CE(b,L)^k
R_2  = CE(b,L)^(K+k)
R'_2 = CE(B_amb,L)^(K+k).
```

The ambient output relation uses `B_amb`, not `b` and not `B`.

Source: `SRC-PAPER-07`, lines 5-143, and `SRC-PAPER-13`, lines 102-200.

## PAPER-PIRLC — PiRLC

`PAPER-PIRLC-001` The verifier samples one `rho_i in C` per PiCCS output. Under
the ring-module action, it forms:

```text
z = sum_i rho_i*z_i
c = sum_i rho_i*c_i
x = sum_i rho_i*x_i
y_j = sum_i rho_i*y_(i,j).
```

The Structure and point do not change. The output is one `CE(B,L)` claim. The
weak extraction proof charges the exact coordinate-fork loss `(K+k)/abs(C)`;
the appendix keeps `(K+k+1)/abs(C)` as a conservative bound.

`PAPER-PIRLC-002` Lemma 4 states that `Pi_RLC` is weak for the **same** `phi`
as Lemma 3. Its relation types are:

```text
R_1  = CE(b,L)^(K+k)
R'_1 = CE(B_amb,L)^(K+k)
R_2  = CE(B,L).
```

The single input point `r` is shared by all `K+k` input claims.

Thus the displayed PiRLC algorithm is defined on the shared-point subset of
the printed Cartesian-product input relation.

Source: `SRC-PAPER-07`, PiRLC section, and `SRC-PAPER-13`, lines 330-430.

## PAPER-PIDEC — PiDEC

`PAPER-PIDEC-001` The prover computes `z_0,...,z_(k-1)=split_b(z)`, commits to
each child, and evaluates each child at the unchanged point. The verifier
computes `(x_0,...,x_(k-1))=split_b(x)` and checks:

```text
c = sum_h b^h*c_h
y_j = sum_h b^h*y_(h,j) for each j.
```

The split definition gives `x=sum_h b^h*x_h`; the child public inputs are
verifier-derived and are not prover messages.

The output is exactly `k` claims in `CE(b,L)`.

Every child uses the unchanged parent point. The displayed output is therefore
in the shared-point subset of the printed Cartesian-product relation.

Source: `SRC-PAPER-07`, PiDEC section, and `SRC-PAPER-13`, PiDEC proof.

## PAPER-COMP — composition

`PAPER-COMP-001` One fold is PiCCS, then PiRLC, then PiDEC. The strong and weak
reductions must use the same commitment projection. Under the stated
assumptions, their composition is a reduction of knowledge.

`PAPER-COMP-002` Theorem 1 gives the exact composed type. The sequential
composition `Pi_DEC o Pi_RLC o Pi_CCS` is a reduction of knowledge from
`CCS(b,L)^K * CE(b,L)^k` to `CE(b,L)^k`.

The composition uses Theorem 6 for `Pi_RLC o Pi_CCS`, then Lemma 2 for the
`Pi_DEC` stage. Theorem 7 states that `Pi_DEC:CE(B,L)->CE(b,L)^k` is a
reduction of knowledge. `Pi_DEC` is neither strong nor weak in the paper.

`PAPER-COMP-003` With `R=CCS(b,L)^K` and `R_ACC=CE(b,L)^k`, Theorem 1 makes
the composition a folding scheme under Definition 19.

The printed composition types inherit the `PAPER-BATCH-001` shared-point
mismatch. The protocol pipeline itself produces and consumes only batches in
which all component points are equal.

Source: `SRC-PAPER-02`, line 77; `SRC-PAPER-06`; `SRC-PAPER-07`; and
`SRC-PAPER-12`, line 32.
## PAPER-PROFILE — Appendix B.2 Goldilocks profile

`PAPER-PROFILE-001` The paper reports:

```text
q = 2^64-2^32+1              K = F_(q^2)
eta = 81                     Phi = X^54+X^27+1
d = 54                       kappa = 18
n_F = 2^30                   b = 2
k = 14                       K_fresh <= 61
B = 2^14                     T = 216
C coefficients = {-2,-1,0,1,2}
abs(C) about 2^125           MSIS estimate about 129 bits.
```

Source: `SRC-PAPER-11`, lines 13-19.

## PAPER-D7 — analytic MSIS diagnostic

`PAPER-D7-001` Appendix D.7 contains an executable analytic MSIS
diagnostic. The locked code places only `kappa*d` under the square root and
multiplies by `log2(q)*log2(delta)` afterward. The cited analytic expression
places `kappa*d*log2(q)*log2(delta)` under one square root. The diagnostic does
not supply an end-to-end knowledge-soundness estimate.

Source: `SRC-PAPER-13`, lines 617-632.

## PAPER-CONFLICT-DIM — incompatible printed dimensions

`PAPER-CONFLICT-001` The reviewed paper still requires `n_F=d*n_R`, but its
Goldilocks profile selects `d=54` and `n_F=2^30`. Since
`2^30 mod 54=46`, these values do not define an integer `n_R`.

This extraction does not resolve that profile conflict. Nightstream decisions
`NSD-DOMAIN-001` and `NSD-DOMAIN-MAP-001` select the actual ring-aligned
assignment width and one larger padded row cube. This is an instance of the
paper's `n_F<=m` normalization, but it does not treat the printed `n_F=2^30`
value as a valid ring-aligned committed-vector width.

Source: `SRC-PAPER-04`, `SRC-PAPER-07`, and `SRC-PAPER-11`.

## PAPER-CONFLICT-LENGTH — overloaded commitment length

`PAPER-CONFLICT-002` Definition 4 declares `Setup(1^lambda,m)` and
`Commit(pp,z)` for `z` in `R_F^m`. Definition 14 and the reduction Setup
algorithms instead call `Setup(1^lambda,n_R)` and use
`L:R_F^(n_R)->C`. The two length parameters have the same name and different
meanings.

This extraction reads the commitment length as `n_R`, the committed-vector ring width.
A concrete Module-SIS claim must state which length it uses.

Source: `SRC-PAPER-04` and `SRC-PAPER-07`.
