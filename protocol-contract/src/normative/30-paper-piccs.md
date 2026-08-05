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
values and check the exact terminal equation `v=Q(r_new)`.

Source: PAPER-PICCS-001, PAPER-PICCS-002, and PAPER-PICCS-005.

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

The PiCCS extractor target MUST be `CE(B_amb,L)^(K_fresh+k)`. It MUST NOT
assume the tight output relation `CE(b,L)^(K_fresh+k)`.

Source: PAPER-PICCS-003, PAPER-PICCS-006, and ERR-AMBIENT.
