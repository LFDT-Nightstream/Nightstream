# Nightstream SuperNeo v1 normative protocol contract

> Generated reading view. Edit `src/normative/` and
> `src/requirements/`; `refresh_derived.py` rebuilds this file.

> Assembly overrides:
> - `NS-PICCS-VARIANT` replaces `SN-PICCS-IDENTITY`.


Status: **selected implementation specification; production assurance open**.
This contract does not state that the current Rust code or circuit conforms to
the selected target.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.
Contract indices start at zero. `K_ext` is the extension field. `K_fresh` is
the number of fresh claims.
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
## 2. Reviewed reduction framework

### SN-RED-STAGE — Interactive reduction interface

Each reduction stage MUST be one tuple `(G,K,P,V)`, and `K` MUST be
deterministic. All composed stages MUST use the same generator and encoder.

Source: PAPER-RED-001 and PAPER-RED-002.

### SN-RED-KNOWLEDGE — Reduction-of-knowledge conditions

A stage that claims a reduction of knowledge MUST establish completeness,
knowledge soundness, and public coin. All verifier randomness MUST appear in
uniform verifier messages.

Source: PAPER-RED-001 and PAPER-RED-002.

### SN-RED-SEQUENTIAL — Sequential composition boundary

Sequential composition MUST bind the first stage output to the second stage
input under one shared setup and encoder.

Source: PAPER-RED-002 and PAPER-RED-006.

### SN-RED-RELATIONS — Paper strong and weak relation pairs

The corrected composition uses these shared-point relation pairs:

```text
PiCCS: CCS(b,L)^K_fresh * BatchCE_k(b,L)
        -> BatchCE_(K_fresh+k)(b,L)
        ambient BatchCE_(K_fresh+k)(B_amb,L)
PiRLC: BatchCE_(K_fresh+k)(b,L)
        ambient BatchCE_(K_fresh+k)(B_amb,L)
        -> CE(B,L).
```

Source: PAPER-RED-003 through PAPER-RED-005, PAPER-PICCS-006,
PAPER-PIRLC-002, ERR-AMBIENT, and ERR-SHARED-POINT.

### SN-RED-PROJECTION — Shared commitment projection

Both stages MUST use the same function `phi`, equal to the ordered commitment
projection of their instances.

Source: PAPER-RED-003 through PAPER-RED-005.

### SN-RED-STRONG-CONDITIONS — PiCCS strong conditions

PiCCS MUST preserve the same `phi` image across two independent prover runs
with probability one, and its relaxed extractor MUST target the ambient
PiCCS relation from SN-RED-RELATIONS. Under the strong definition's output-
witness agreement premise, extraction MUST recover an input witness with
probability at least relaxed success minus a negligible term.

Source: PAPER-RED-003, PAPER-RED-005, and PAPER-PICCS-006.

### SN-RED-WEAK-CONDITIONS — PiRLC weak conditions

PiRLC MUST have an extractor for its ambient input relation with probability
at least adversary success minus a negligible term. For two input instances
with the same `phi` image, its extracted witnesses MUST agree except with
negligible probability.

Source: PAPER-RED-004, PAPER-RED-005, and PAPER-PIRLC-002.
## 3. Reviewed SumCheck and PiCCS

### SN-SUMCHECK-CLAIM — SumCheck statement

For an `ell`-variable polynomial `Q`, SumCheck MUST verify
`T=sum_(x in {0,1}^ell) Q(x)` and return one verifier-derived point `r` and
one terminal value claim `v=Q(r)`.

Source: PAPER-SUMCHECK-001.

### SN-SUMCHECK-ROUNDS — SumCheck verifier walk

At round `i`, the verifier MUST receive a univariate polynomial `g_i` of the
declared degree, check `current=g_i(0)+g_i(1)`, sample the next public
challenge, and set `current=g_i(challenge_i)`.

Source: PAPER-SUMCHECK-001.

### SN-SUMCHECK-SOUNDNESS — SumCheck loss

For maximum individual degree `D`, the interactive SumCheck soundness charge
MUST be at most `ell*D/abs(K_ext)`.

Source: PAPER-SUMCHECK-001.

### SN-PICCS-POLYNOMIAL — Reviewed joint polynomial

For zero-based `c in [0,k)`, `j in [0,t)`, and `l in [0,d)`, define
`I(c,j,l)=c+k*j+k*t*l` and let `Eval_(i,j,l)(X)` denote
`MLE(cf(bar(M_j z_i))_l)(X)`. The carried source at local index `c` is global
source `K_fresh+c`. Let `CCS_a(X)` denote
`f(bar(M_0 z_a)(X),...,bar(M_(t-1) z_a)(X))`. For
`P_b(V)=product_(h=-(b-1))^(b-1)(V-iota_q(h))`, the joint polynomial MUST be

```text
F(X)  = sum_(a=0)^(K_fresh-1) gamma^a*CCS_a(X)
NC(X) = sum_(i=0)^(K_fresh+k-1) gamma^i*P_b(MLE(z_i)(X))
Eval(X) = eq(X,r_old)*sum_(c,j,l)
  gamma^I(c,j,l)*Eval_(K_fresh+c,j,l)(X)
Q(X) = eq(X,alpha)*(F(X)+gamma^K_fresh*NC(X))
       + gamma^(2*K_fresh+k)*Eval(X).
```

Source: PAPER-PICCS-001, PAPER-PICCS-002, ERR-NORM-ROOTS, and
ERR-ABS-TARGET.

### SN-PICCS-TARGET — Absolute carried target

The initial SumCheck claim MUST be

```text
T_abs = sum_(c,j,l)
  gamma^(2*K_fresh+k+I(c,j,l))*cf(y_(K_fresh+c,j))_l.
```

The unshifted local sum is not the joint SumCheck target.

Source: PAPER-PICCS-001 and ERR-ABS-TARGET.

### SN-PICCS-EXECUTION — One joint SumCheck

The verifier MUST sample `alpha in K_ext^ell` and `gamma in K_ext`, then run
one SumCheck for `T_abs=sum_X Q(X)` with exactly `ell` rounds.

Source: PAPER-PICCS-001 and PAPER-PICCS-002.

### SN-PICCS-OUTPUT — Output evaluations and terminal check

At the SumCheck point `r_new`, the prover MUST supply `y_(i,j)` for every
source and matrix. The verifier MUST derive `F`, `N`, and `E` from those
values and check the exact terminal equation `v=Q(r_new)`. Every output
component MUST use this one `r_new`, so the output relation is
`BatchCE_(K_fresh+k)(b,L)`.

Source: PAPER-PICCS-001, PAPER-PICCS-002, PAPER-PICCS-005, and
ERR-SHARED-POINT.

### SN-PICCS-CHARACTERIZATION — Joint identity meaning

For arbitrary source vectors, an old point, and carried evaluations, the
identity `T_abs(C)=sum_X Q(X,A,C)` MUST hold as a polynomial identity exactly
when the fresh CCS polynomials vanish, every source has `Bound_b`, and every
carried evaluation is correct. All three conditions MUST use the same ordered
source vectors.

Source: PAPER-PICCS-004.

### SN-PICCS-IDENTITY — Identity-output norm binding

When `M_0=I` on the common cube, `ct(y_(i,0))` MUST equal
`MLE(z_i)(r_new)`. This is the paper link between the norm terminal and the
same witness that the commitment binds.

Source: PAPER-PICCS-001, PAPER-PICCS-005, and PAPER-PICCS-006.

### SN-PICCS-DEGREE — Correct joint individual degree

`D_f` MUST mean total degree, and the joint polynomial MUST use
`D_Q=max(D_f+1,2b,2)` as its maximum individual-degree bound.

Source: PAPER-PICCS-001 through PAPER-PICCS-003, ERR-DEGREE-SEMANTICS, and
ERR-SUMCHECK-DEGREE.

### SN-PICCS-LOSSES — Separate algebraic bad events

The reduction MUST charge `D_Q*ell/abs(K_ext)` for false SumCheck acceptance
and `D_SZ/abs(K_ext)` for the independent nonzero gamma/alpha polynomial root
event. One term cannot replace the other.

Source: PAPER-PICCS-002, PAPER-PICCS-003, and ERR-ERROR-BUDGET.

### SN-PICCS-EXTRACTOR-FLOW — Success-gated extraction

The strong extractor MUST run once without conditioning. If that run does not
reach ambient success, it MUST fail. Only after success may it retry for a
second ambient-success witness, and the global witness-disagreement loss MUST
be the reviewed `sqrt(delta)` term.

Source: PAPER-PICCS-003, PAPER-PICCS-006, ERR-STRONG-EXTRACT, and
ERR-EXTRACT-RUNTIME.

### SN-PICCS-EXTRACTOR-TARGET — Ambient extraction relation

The PiCCS extractor target MUST be
`BatchCE_(K_fresh+k)(B_amb,L)`. It MUST NOT assume the tight output relation
`BatchCE_(K_fresh+k)(b,L)`.

Source: PAPER-PICCS-003, PAPER-PICCS-006, ERR-AMBIENT, and
ERR-SHARED-POINT.
## 4. Reviewed PiRLC, PiDEC, and fold composition

### SN-PIRLC-DOMAIN — PiRLC input family

PiRLC MUST take exactly `BatchCE_(K_fresh+k)(b,L)`, the `K_fresh+k` PiCCS
outputs with one shared Structure and one shared evaluation point.

Source: PAPER-PIRLC-001 and ERR-SHARED-POINT.

### SN-PIRLC-EQUATIONS — Ring-linear combination

After all bound inputs exist, the verifier MUST sample one `rho_i in C` per
source and compute `z`, `c`, `x`, and every `y_j` with the same ordered
`R_F`-module linear combination.

Source: PAPER-PIRLC-001 and PAPER-PIRLC-002.

### SN-PIRLC-OUTPUT — PiRLC output bound

The Structure and evaluation point MUST remain unchanged. The result MUST be
one claim in `CE(B,L)`.

Source: PAPER-PIRLC-001.

### SN-PIRLC-FORK-LOSS — Coordinate-fork probability

For `L=K_fresh+k` independent coordinates, the exact coordinate-fork loss
MUST be `L/abs(C)`. A proof MAY use the conservative `(L+1)/abs(C)` bound and
MUST NOT divide by `abs(C)^L`.

Source: PAPER-PIRLC-001, PAPER-FORK-001, and ERR-COORD-FORK.

### SN-PIRLC-FORK-SET — Complete fork shape

The fork extractor MUST produce one base challenge vector and `L` neighbours,
where neighbour `i` changes only coordinate `i`. Its expected query count is
at most `L+1`.

Source: PAPER-FORK-001 and ERR-PIRLC-PROJECTION.

### SN-PIRLC-AGREEMENT — Weak extractor agreement

The PiRLC proof MUST establish the witness-agreement condition from
SN-RED-WEAK-CONDITIONS separately from coordinate forking and relaxed binding.

Source: PAPER-PIRLC-002 and ERR-PIRLC-PROJECTION.

### SN-PIDEC-SPLIT — PiDEC witness decomposition

PiDEC MUST take one `CE(B,L)` claim and apply the selected exact `split_b` to
its witness. The verifier MUST derive the same ordered public-input split.

Source: PAPER-PIDEC-001.

### SN-PIDEC-EQUATIONS — PiDEC recomposition

The verifier MUST derive `(x_0,...,x_(k-1))=split_b(x)` and MUST check

```text
c = sum_h b^h*c_h
y_j = sum_h b^h*y_(h,j) for every j.
```

The derived public split MUST satisfy `x=sum_h b^h*x_h`; it is not a prover
message.

Source: PAPER-PIDEC-001, ERR-PIDEC-EQUATIONS, and ERR-EVALUATION-NOTATION.

### SN-PIDEC-OUTPUT — PiDEC output family

PiDEC MUST enforce the child count, common Structure, common point, and
canonical public split. Its output MUST be exactly `BatchCE_k(b,L)`.

Source: PAPER-PIDEC-001 and ERR-SHARED-POINT.

### SN-COMP-ORDER — Fold stage order

One fold MUST execute PiCCS, then PiRLC, then PiDEC.

Source: PAPER-COMP-001.

### SN-COMP-BINDING — Phase boundary equality

Each phase output MUST bind exactly to the next phase input for every
relation-authoritative field, and the strong and weak stages MUST share the
commitment projection.

Source: PAPER-COMP-001 and PAPER-COMP-002.

### SN-FOLD-TYPE — Composed folding type

The composed fold MUST have type

```text
PiDEC o PiRLC o PiCCS :
  CCS(b,L)^K_fresh * BatchCE_k(b,L) -> BatchCE_k(b,L).
```

The running width `k` MUST be unchanged.

Source: PAPER-RED-006, PAPER-COMP-002, PAPER-COMP-003, and
ERR-SHARED-POINT.

### SN-FOLD-PROOF — Composition proof structure

The proof MUST use strong-weak composition for `PiRLC o PiCCS`, then
sequential composition with PiDEC. PiDEC MUST NOT enter the strong-weak step.

Source: PAPER-RED-006, PAPER-COMP-002, and PAPER-COMP-003.

### SN-SEC-ABSTRACT — Paper security boundary

The abstract reduction MUST include SumCheck soundness, field root bounds,
strong-set extraction, relaxed binding, and the reviewed extractor
corrections. It does not establish Fiat-Shamir, an implementation, a circuit,
a backend proof, or an on-chain verifier.

Source: PAPER-COMP-001, PAPER-COMP-002, ERR-COORD-FORK, and
ERR-STRONG-EXTRACT.
## 5. Nightstream v1 profile

### NS-ALGEBRA-PROFILE — Goldilocks and Phi81 selection

The v1 profile MUST use

```text
q = 2^64-2^32+1
K_ext = F_q[U]/(U^2-7)
Phi = X^54+X^27+1
d = 54, b = 2, k = 14, B = 16384.
```

Decision: NSD-DOMAIN-001 and NSD-ENCODING-001.

### NS-SHAPE-LOGICAL — Selected logical shape

The verifier-key relation artifact MUST supply the exact positive logical row
count and the exact full committed assignment width `m` for `z=x||w`, with
`m` a multiple of 54 and at most 16,777,206 fields, or 310,689 complete Phi81
ring columns. The first 270 fields are `x`. The profile has one fresh claim
and 14 running claims. Source order MUST be fresh claim 0 followed by running
claims 0 through 13. A proof-supplied shape MUST NOT select these values.

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

For a centered scalar `a` with `abs(a)<2^14`, let `s=-1` when `a<0` and
`s=1` otherwise. Child `h` MUST be `s*bit_h(abs(a))` for `h=0..13`. The
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
## 7. Nightstream fold refinement

### NS-PIRLC-PROFILE — Selected PiRLC inputs and challenges

PiRLC MUST take the 15 CE outputs of NS-PICCS-TERMINAL in the same source
order. It MUST sample exactly 15 ring challenges with NS-SAMPLER-CANDIDATES
after every PiCCS output is transcript-bound. Coefficient `j` of challenge
`i` MUST be the accepted signed digit for global source `i`, coefficient `j`,
encoded through `iota_q`.

Decision: NSD-SAMPLER-001 and NSD-TRANSCRIPT-001.

### NS-PIDEC-PROFILE — Selected PiDEC children

PiDEC MUST use NS-SPLIT-BINARY and output exactly 14 ordered CE children. It
MUST derive each 270-field child public input from the public parent and check
the commitment and all 14 ring-evaluation recomposition equations. In a
sequence, fold `j+1` MUST use those children as its 14 ordered running claims
without insertion, removal, reordering, or value change. A bad sequence link
MUST reject.

Decision: NSD-SPLIT-001 and NSD-AUTHORITY-001.

### NS-RED-PADDED-RELATIONS — Reduction relation refinement

The Nightstream PiCCS strong relation MUST be the paper relation under the
zero-row embedding. Its output and ambient relations MUST remain
`BatchCE_15(b,L)` and `BatchCE_15(B_amb,L)`. The commitment projection MUST
remain unchanged and no padding or cache field may enter it.

Decision: NSD-REDUCTION-FRAMEWORK-001 and NSD-NORM-BINDING-001.

### NS-RED-COMPOSITION — Padded fold proof obligation

The end-to-end fold proof MUST first establish that zero-row embedding
preserves the paper PiCCS identities and strong conditions. It MUST then use
the reviewed weak PiRLC and PiDEC composition without an extended carrier
relation or an extra batching lemma.

Decision: NSD-REDUCTION-FRAMEWORK-001, NSD-BATCH-COINS-001, and
NSD-COLUMN-001.
## 8. Nightstream verifier profile

### NS-AUTH-STRUCTURE — Verifier-key authority

The fold verifier key MUST own the selected profile ID, exact Structure, and
Ajtai setup seed and dimensions. A proof-supplied copy is non-authoritative
and MUST match or reject. The terminal backend manifest has separate authority
under NS-DECIDER-PROFILE.

The canonical relation artifact MUST use format
`nightstream/verifier-key-relation`, schema 1, and payload encoding
`rust-ccs-structure-serde-json-v1`. It MUST contain the complete thirteen
application matrices and polynomial. Its validator recomputes the expected
artifact from the live verifier-owned Structure and rejects unless the
complete decoded payload is equal. Matching dimensions or digests alone grant
no authority. This JSON artifact does not replace the canonical sparse
Structure stream used by NS-ENC-STRUCTURE for verifier-key hashing.

Decision: NSD-AUTHORITY-001 and NSD-ENCODING-001.

### NS-AUTH-CLAIM — CE claim authority

A CE claim MUST contain exactly `c`, `x`, `r`, and `y_ring` as defined by
SN-REL-CE. Shape values MUST come from the verifier key. An untyped or extra
relation field MUST reject.

Decision: NSD-AUTHORITY-001 and NSD-COLUMN-001.

### NS-AUTH-DERIVED — Caches and digests

The verifier MUST recompute constant-term caches, the verifier-key digest,
public-carrier evaluations, offsets, lengths, and the final fold
transcript digest from authoritative inputs. A digest or cache MUST NOT grant
authority to its preimage.

Decision: NSD-AUTHORITY-001, NSD-HASH-001, and NSD-PROVENANCE-001.

### NS-ENC-BASE — Canonical base-field bytes

A base-field element MUST be one 8-byte little-endian unsigned integer less
than `q`. Reduction modulo `q`, alternate integer widths, and noncanonical
values MUST reject.

Decision: NSD-ENCODING-001.

### NS-ENC-EXTENSION — Canonical extension encoding

`K_ext` MUST be `F_q[U]/(U^2-7)`. An element MUST encode as two canonical
base fields in `(c0,c1)` order for `c0+c1*U`.

Decision: NSD-ENCODING-001.

### NS-ENC-RING — Canonical ring encoding

An `R_F` or `R_K` value MUST use coefficients of degrees 0 through 53 in
that order. Each `R_K` coefficient MUST use NS-ENC-EXTENSION. No storage
padding or alternate coefficient order is valid.

Decision: NSD-ENCODING-001.

### NS-ENC-CONTAINER — Typed statement and proof containers

The statement and proof MUST use the v1 magic, version, variant, ordered
section IDs, and exact field and byte counts from the profile. Their decoders
use checked integer arithmetic and compare the complete input length before
payload allocation. Unknown, missing, duplicate, reordered, truncated, or
trailing content MUST reject. The statement is exactly 318,832 bytes and
39,848 base fields. The proof is exactly 463,528 bytes, with sections of 480,
22,680, and 34,776 base fields. One statement-proof pair MUST encode one fold.
A bounded sequence MUST contain exactly `fold_count` pairs in increasing
`fold_index` order.

Decision: NSD-ENCODING-001 and NSD-TRANSCRIPT-001.

### NS-ENC-STRUCTURE — Canonical Structure stream

The Structure MUST use `nightstream-sparse-structure-v1`. Its header and the
implicit `M_0` and zero-padding variants MUST use the profile layout. Each
application matrix MUST use strict row-major nonzero triples with valid indices,
nonzero canonical values, and no duplicates. `f_app` MUST use sorted unique
coefficient-and-exponent terms of degree 1 through 8, with maximum degree 8;
thus `f_app(0,...,0)=0`.

Decision: NSD-ENCODING-001 and NSD-AUTHORITY-001.

### NS-ENC-COMMITMENT — Commitment encoding

A commitment MUST encode as 18 `R_F` elements in row order under NS-ENC-RING.

Decision: NSD-ENCODING-001.

### NS-POSEIDON-PARAMETERS — Exact permutation

All protocol-binding Poseidon2 calls MUST use the values in
`src/profile/poseidon2-goldilocks-v1.toml`: Goldilocks width 8, rate 4,
capacity 4, `x^7`, 4 initial full rounds, 22 partial rounds, 4 terminal full
rounds, and the listed constants and matrices. The permutation MUST use the
listed column-vector operation order and match both profile test vectors.

Decision: NSD-HASH-001 and NSD-TRANSCRIPT-001.

### NS-TRANSCRIPT-SPONGE — Field duplex state transition

The transcript MUST start with eight zeros and absorb cursor 0. Absorption MUST
add a field to `state[cursor]`, increment the cursor, and permute and reset it
to 0 after lane 3. A frame MUST absorb `tag`, payload length, then payload. A
direction change MUST use the profile constants: add 1 at the cursor and 2 at
lane 3 before first squeeze, or add 3 at lane 4 and permute before absorption.

Decision: NSD-TRANSCRIPT-001 and NSD-HASH-001.

### NS-TRANSCRIPT-FRAMING — Squeeze and continuation rules

Before a squeeze, the transcript MUST frame the squeeze tag and requested lane
count under the challenge-frame tag. The first block MUST return rate lanes 0
through 3 in order after the direction permutation. Each continuation block
MUST add 4 at lane 4, permute, and return lanes 0 through 3. A caller MUST use
only the requested lane count.

Decision: NSD-TRANSCRIPT-001.

### NS-VERIFIER-KEY-DIGEST — Exact verifier-key prehash

The verifier MUST start a fresh selected field duplex, frame the contract domain
and profile version, then frame the setup code, dimensions, 32 seed-byte lanes,
and canonical Structure stream with their declared counts. It MUST squeeze four
base fields under the verifier-key-digest tag. The main fold transcript MUST
absorb those four fields under the verifier-key tag.

Decision: NSD-ENCODING-001, NSD-HASH-001, and NSD-TRANSCRIPT-001.

### NS-TRANSCRIPT-ORDER — Fold transcript schedule

The transcript order MUST be session, verifier-key digest, statement, PiCCS
input, alpha, gamma, 24 ordered SumCheck round messages and challenges, PiCCS
outputs, indexed PiRLC sampler attempts, derived PiRLC output, PiDEC children,
and fold finalization. Each fold in a bounded sequence MUST start with a fresh
zero-state duplex and use the sequence's single selected verifier key and
profile. The four-field fold transcript digest MUST be a verifier-derived
receipt and MUST NOT enter CE authority or the next fold statement. No
challenge may be sampled before all messages on which it depends are absorbed.

Decision: NSD-TRANSCRIPT-001, NSD-PICCS-001, and NSD-SAMPLER-001.

### NS-CHALLENGE-EXTENSION — Extension-field challenges

An `alpha`, `gamma`, or SumCheck challenge MUST be two consecutive uniform
base-field squeeze lanes interpreted as `(c0,c1)`. Zero is valid. Alpha MUST
contain 24 elements and SumCheck MUST sample one element after each round
message.

Decision: NSD-TRANSCRIPT-001 and NSD-BATCH-COINS-001.

### NS-SAMPLER-CANDIDATES — Uniform strong-set digits

For source index `i`, coefficient index `j`, and attempt `a`, the transcript
MUST absorb those indices under the PiRLC-candidate tag and squeeze one base
field. It MUST accept only `x<q-1` and map `x mod 5` in order to
`[-2,-1,0,1,2]`, then encode that signed digit with `iota_q` as coefficient
`j` of the source's ring challenge.

Decision: NSD-SAMPLER-001 and NSD-TRANSCRIPT-001.

### NS-SAMPLER-REPETITIONS — Bounded sampler loop

The verifier MUST process 15 sources in order, 54 coefficients per source,
and at most three attempts per coefficient. If all three candidates reject,
the whole proof MUST reject.

Decision: NSD-SAMPLER-001.

### NS-SAMPLER-LOSS — Sampler distribution and exhaustion

Accepted digits MUST be exactly uniform on the five-element alphabet. The
security reduction MUST include a per-fold exhaustion bound of at most
`810/q^3` and MUST compose it across the selected fold limit.

Decision: NSD-SAMPLER-001 and NSD-SECURITY-001.

### NS-SECURITY-POLICY — V1 threat and resource limits

The v1 security target MUST be at least 96 classical bits for one proof and
one session per verifier key with at most 64 folds. The resource census MUST
allow at most 262,144 adaptive oracle queries, including the derived maximum
157,313 prescribed tagged squeezes per key. The release theorem MUST be an
expected-polynomial-time proof of knowledge and MUST state the Ajtai setup or
seeded-PRG assumption.

Decision: NSD-SECURITY-001 and NSD-THREAT-MODEL-001.

### NS-DECIDER-PROFILE — Terminal backend interface

The terminal family MUST be `Spartan-with-WHIR-profile-manifest-v1` over the
selected current-circuit R1CS and the nine-field `public_image_v1` tuple. The
verifier key MUST own one versioned concrete backend manifest. An absent,
unknown, or mismatched manifest MUST reject.

Decision: NSD-DECIDER-001, NSD-CIRCUIT-001, NSD-ENCODING-001, and NSD-HASH-001.
## 9. Conformance, circuit, and security evidence

### NS-RUST-EVIDENCE-ORIGIN — Rust execution origin

Rust conformance evidence MUST name the contract and profile hashes, Rust
revision, source-tree and lock hashes, feature set, compiler, target, producer
binary, command, and run attestation. The observed decisions and mutations
MUST originate in that Rust execution.

Decision: NSD-PROVENANCE-001.

### NS-RUST-EVIDENCE-CONTENT — Independent semantic check

Each evidence item MUST contain a rule-indexed ordered trace, complete input,
first rejection rule, and at least one adversarial mutation. An independent
checker MUST recompute the semantic result and validate every bound hash. A
carried acceptance Boolean MUST NOT establish conformance.

Decision: NSD-PROVENANCE-001.

### NS-CIRCUIT-MANIFEST — Current-circuit identity

The current shipping circuit MUST publish a manifest that binds the contract,
profile, transcript, sampler, source build, frontend relation, backend
relation, verifier key, public-input count, and exact rule-to-row map.
The manifest MUST identify the fixed circuit constant for the verifier-key
digest and the evidence that equates it to native recomputation from the
canonical setup and Structure.

Decision: NSD-CIRCUIT-001 and NSD-PROVENANCE-001.

### NS-CIRCUIT-COMPLETENESS — Native-to-circuit direction

For every normative verifier acceptance, the correspondence proof MUST show
that a satisfying circuit witness exists for the same statement and proof.

Decision: NSD-CIRCUIT-001.

### NS-CIRCUIT-SOUNDNESS — Circuit-to-native direction

For every arbitrary satisfying circuit assignment, the correspondence proof
MUST derive normative verifier acceptance or a named cryptographic bad event.
Honest witness-generation tests alone do not close this claim.

Decision: NSD-CIRCUIT-001.

### NS-CIRCUIT-PUBLIC-INPUT — Public-image decoder and statement binding

Every circuit public vector MUST contain the exact nine fields in
`public_image_v1.public_field_order`. The circuit MUST recompute its four digest
fields with the fresh-duplex session, verifier-key, statement, and squeeze
steps in that profile. The five explicit fields MUST match the selected profile
and decoded statement. A noncanonical public alias MUST be unsatisfiable or
reject. Two distinct preimages with one digest are a named Poseidon2 collision
event, not an encoding alias.

Decision: NSD-CIRCUIT-001, NSD-ENCODING-001, and NSD-HASH-001.

### NS-CIRCUIT-LOWERING — Hints and backend lowering

The circuit proof MUST cover hint constraints, lookups, ranges, frontend row
generation, and frontend-to-backend lowering. Every authoritative field MUST
have one owner and every non-plumbing row MUST map to one contract rule.

Decision: NSD-CIRCUIT-001.

### NS-DECIDER-CORRESPONDENCE — Terminal and deployed verifier

The terminal proof and deployed verifier MUST use the same backend manifest,
canonical public image, Poseidon2 transcript, parser, and verifier key. The
reduction MUST start at the deployed acceptance predicate, not a test fixture.

Decision: NSD-DECIDER-001, NSD-ENCODING-001, and NSD-TRANSCRIPT-001.

### NS-SEC-REDUCTION — Named bad-event ledger

The end-to-end theorem MUST bound separate terms for SumCheck, algebraic
mixing roots, padded-identity refinement, coordinate forking, relaxed binding,
strong extraction, Poseidon2 and Fiat-Shamir, sampler exhaustion, encoding,
implementation transfer, circuit soundness, backend proof, and deployed
verification. Each nonzero term MUST have a theorem, substitution, and owner.

Decision: NSD-SECURITY-001, NSD-THREAT-MODEL-001, and
NSD-REDUCTION-FRAMEWORK-001.

### NS-SEC-COMPOSITION — Lifetime and extractor composition

The reduction MUST compose the one-fold bound across at most 64 adaptive
folds and the stated oracle limit. If it uses a union bound, it MUST establish
a uniform conditional bound for every accepted prefix. The proof-of-knowledge
statement MUST give a concrete expected-polynomial-time extraction bound.

Decision: NSD-SECURITY-001 and NSD-THREAT-MODEL-001.

### NS-RELEASE-IMPLEMENTATION — Implementation-ready gate

Implementation work MAY target this contract only when G0, G0B, and G1 are
closed from current evidence. This state means the semantics and design are
fixed; it does not claim that Lean, Rust, circuit, or security evidence is
complete.

Decision: NSD-PROVENANCE-001.

### NS-RELEASE-PRODUCTION — Production release gate

A production release MUST close G2 through G5 and MUST NOT claim an assurance
tier above its weakest required edge. Any implementation difference from this
profile is a release blocker or a new versioned contract decision.

Decision: NSD-PROVENANCE-001, NSD-CIRCUIT-001, NSD-DECIDER-001, and
NSD-SECURITY-001.
