## PAPER-PICCS — reviewed joint PiCCS

`PAPER-PICCS-001` The paper assumes `m=n_F`, `n_F` is a power of two, and the
first matrix is the identity. PiCCS reduces
`CCS(b,L)^K * CE(b,L)^k` to `CE(b,L)^(K+k)`.

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

`PAPER-PICCS-005` The joint protocol states three hypotheses: `m=n_F`, `n_F` is
a power of two, and `M_1=I_(n_F)`. The identity hypothesis carries the norm
item of Lemma 7 to the verifier. Because `M_1=I`, the terminal opening gives
`ct(y'_(i,1))=tilde(z_i)(r')`, and `y'_(i,1)` is a field of the output CE
instance. The norm check therefore reads the same vector that the output CE
relation binds to the commitment `c_i`.

Remark 3 states this simplification. Without the identity hypothesis, the paper
gives no other rule that binds the norm-check opening to the committed witness.

`PAPER-PICCS-006` Lemma 3 states that `Pi_CCS` is strong for the function `phi`
that projects the commitments `(c_i)` from the instance. Its relation types are:

```text
R_1  = CCS(b,L)^K * CE(b,L)^k
R_2  = CE(b,L)^(K+k)
R'_2 = CE(B_amb,L)^(K+k).
```

The ambient output relation uses `B_amb`, not `b` and not `B`.

Source: `SRC-PAPER-07`, lines 45-133, and `SRC-PAPER-13`, lines 84-180.

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
