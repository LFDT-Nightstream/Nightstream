# Neo Summary

**Authors:** Wilson Nguyen (Stanford University) and Srinath Setty (Microsoft Research)

Neo is a lattice-based folding scheme for **CCS (Customizable Constraint Systems)** over **(small) prime fields**, built around (1) a **pay‑per‑bit Ajtai-style matrix commitment** and (2) a **HyperNova-style “CCS → evaluation claims” folding loop**, adapted to the lattice / low‑norm setting. It is designed so that *all foldable sub-arguments* reduce to the same intermediate relation (**ME**) and can be folded together—this is exactly why Neo can also fold **Shout/Twist**-style memory and lookup arguments (they are sum-check based and can be reduced to ME claims).

Below is a “developer-grade” summary of Neo with the key definitions and formulas you need to implement it (and to understand how Twist & Shout fits into the same folding interface).

---

## 1. What Neo is trying to achieve

Neo targets the gap between:

* **Group-based folding** (Nova/HyperNova) which is very fast but not post-quantum, and
* Earlier **lattice-based folding** (e.g., LatticeFold) which has practical downsides, including lack of *pay‑per‑bit* behavior (committing to 1-bit vs 64-bit costs the same), and constraints on ring choices.

Neo’s core contributions are:

1. A **matrix commitment scheme** over prime fields derived from Ajtai’s SIS hash/commitment, engineered to be **pay‑per‑bit** (commitment cost depends on witness bit-width).
2. A **folding scheme for CCS** that works in a “matrix setting” and keeps openings **low‑norm** by design (via decomposition + splitting).
3. A security framework to compose reductions that are *not* individually standard ROKs into a final folding scheme that *is* a **reduction of knowledge** (ROK).

---

## 2. Core multilinear + sum-check algebra (used everywhere)

Neo uses standard multilinear extension + sum-check machinery, but it is important to track two things:

* **Boolean hypercubes** ({0,1}^\ell),
* **Evaluations at random points** (r \in \mathbb{K}^\ell), where (\mathbb{K}) is typically an extension field of (\mathbb{F}_q) to get enough soundness slack when (q) is small.

### 2.1 Multilinear extension (MLE)

Given a vector (f \in \mathbb{F}^{2^\ell}), define its MLE:
[
\widetilde{f}(X_1,\dots,X_\ell)=\sum_{i\in{0,1}^\ell} f_i\cdot \chi_i(X_1,\dots,X_\ell),
]
where
[
\chi_i(X_1,\dots,X_\ell)=\prod_{j=1}^{\ell}\big(i_jX_j+(1-i_j)(1-X_j)\big).
]
Define
[
\hat r:=\bigotimes_{j=1}^{\ell}(r_j,1-r_j)\in \mathbb{K}^{2^\ell},
\quad\text{so}\quad
\widetilde f(r)=\langle f,\hat r\rangle.
]
This “(\hat r)” vector is exactly what shows up in ME claims as the vector you multiply by. 

### 2.2 Equality polynomial

[
\operatorname{eq}(X,Y):=\prod_{i=1}^{\ell}\Big(X_iY_i+(1-X_i)(1-Y_i)\Big).
]
On Boolean points it acts like an equality indicator:
(\operatorname{eq}(x,y)=1 \iff x=y) for (x,y\in{0,1}^\ell).

### 2.3 Zero-sum set (ZS_\ell)

Neo encodes “constraint satisfaction” as a **zero-sum identity** (which is what sum-check natively proves):
[
ZS_\ell:=\left{Q\in\mathbb{F}^{<2}[X_1,\dots,X_\ell];:;\sum_{x\in{0,1}^\ell} Q(x)=0\right}.
]
And indeed:
[
Q\in ZS_\ell \iff \sum_{x\in{0,1}^\ell}Q(x)=0.
]
So “(F \in ZS_\ell)” means “the constraint polynomial (F) sums to zero over the hypercube,” which is what the sum-check protocol will enforce. 

---

## 3. The ring/matrix layer enabling lattice commitments over prime fields

Neo uses a cyclotomic ring **only internally to define the Ajtai/SIS commitment cleanly**, but the *main protocol objects are matrices over the prime field* (\mathbb{F}_q).

### 3.1 Cyclotomic ring and coefficient map

Let (q) be prime and (\Phi_\eta(X)) a cyclotomic polynomial of degree (d). Define:
[
R_q := \mathbb{F}*q[X]/(\Phi*\eta(X)).
]
Let a ring element be (a(X)=\sum_{i=0}^{d-1} a_i X^i). The **coefficient map** is:
[
\operatorname{cf}(a):=(a_0,\dots,a_{d-1})\in\mathbb{F}_q^d,
]
and it extends columnwise to vectors (R_q^m \to \mathbb{F}_q^{d\times m}).

### 3.2 Rotation matrices and the commutative subring (S)

For (a\in R_q), define the **rotation matrix** (\operatorname{rot}(a)\in\mathbb{F}_q^{d\times d}) satisfying:
[
\operatorname{rot}(a)\cdot \operatorname{cf}(b)=\operatorname{cf}(ab)\quad \forall b\in R_q.
]
The set of such rotation matrices is a **commutative subring** (S\subseteq \mathbb{F}_q^{d\times d}) (closed under + and ·).

This (S)-module structure is what makes “random linear combinations” and “folding” behave cleanly in the matrix world.

---

## 4. Low-norm representation: decomposition and splitting

A central problem in lattice commitments is: **binding is only guaranteed for small-norm openings**. Neo therefore represents witness values in a bounded-digit form and has an explicit norm-reduction step in the folding loop.

### 4.1 Digit decomposition (\operatorname{Decomp}_b)

Fix integers (b,d) and set (B=b^d). For a vector (z\in \mathbb{F}^m) with (|z|*\infty<B), define:
[
\operatorname{Decomp}*b(z) := (z*{i,j})*{j\in[d],i\in[m]}\in \mathbb{F}^{d\times m}
]
such that each digit satisfies (z_{i,j}\in[-b+1,,b-1]) and reconstructs:
[
z_i=\sum_{j=1}^{d} b^{j-1}, z_{i,j}.
]
So: **each original coordinate becomes (d) small digits** (bounded by (\approx b)).

### 4.2 Splitting (\operatorname{split}_b) (base-(b) over matrices)

Fix (k) and (B=b^k) and define the scalar matrix (\bar b := b I_d \in \mathbb{F}^{d\times d}).
Given (Z\in\mathbb{F}^{d\times m}) with (|Z|*\infty<B), define:
[
(Z_1,\dots,Z_k)\leftarrow \operatorname{split}*b(Z)
]
with each (|Z_i|*\infty<b) and:
[
Z=\sum*{i=1}^{k} \bar b^{,i-1}, Z_i.
]
This is the **mechanism that resets norm growth** after aggregation. 

---

## 5. Neo’s “pay-per-bit” Ajtai-based matrix commitment

### 5.1 Ajtai commitment over the ring

Setup samples a random matrix (M\in R_q^{\kappa\times m}), and commits to a ring-vector (z\in R_q^m) by:
[
\operatorname{Commit}(\text{pp},z) := Mz \in R_q^\kappa.
]
This is deterministic and binding for short (z), under a Module-SIS assumption. 

### 5.2 The matrix commitment ( \mathcal{L}: \mathbb{F}^{d\times m}\to \mathbb{C})

Neo commits to coefficient matrices (Z\in \mathbb{F}_q^{d\times m}) by mapping them back into the ring:

* parse (M) from public parameters,
* compute (z:=\operatorname{cf}^{-1}(Z)\in R_q^m),
* output commitment
  [
  c := \operatorname{cf}(Mz)\in \mathbb{F}_q^{d\times \kappa}.
  ]
  This is the commitment map (\mathcal{L}(Z)).

### 5.3 Why this is pay-per-bit

If the original witness is a vector (z\in\mathbb{F}*q^m) with (0\le z_i<B=b^d), Neo represents (z_i) by digits ((z*{i,1},\dots,z_{i,d})) and forms a ring element:
[
z_i' := \sum_{j=1}^{d} z_{i,j}, X^{j-1}\in R_q.
]
Then it commits to (z'=(z_1',\dots,z_m')\in R_q^m).
The digit bound (small (|z_{i,j}|)) is what makes the commitment cost “scale with bit-width”.

### 5.4 Binding as “short collision ⇒ MSIS solution”

If (\operatorname{Commit}(\text{pp},Z_1)=\operatorname{Commit}(\text{pp},Z_2)) with (|Z_1|*\infty,|Z_2|*\infty<B), then:
[
M\operatorname{cf}^{-1}(Z_1-Z_2)=0 \in R_q^\kappa,
\quad
|\operatorname{cf}^{-1}(Z_1-Z_2)|_\infty<2B,
]
which is an (\mathrm{MSIS}) witness. This is the binding reduction.

---

## 6. Challenge sets ( \mathcal{C} ) (needed for safe random linear combinations)

Neo cannot sample “arbitrary random scalars” for linear combinations: it needs *structured challenges* so that extraction works and norms don’t blow up.

### 6.1 Strong sampling set (invertible differences)

Let (S\subseteq \mathbb{F}^{d\times d}) be a commutative subring. A subset (\mathcal{C}\subseteq S) is a **strong sampling set** if for any distinct (\rho,\rho'\in\mathcal{C}), the difference ((\rho-\rho')) is invertible in (S).

### 6.2 Expansion factor (controls norm growth)

Define:
[
T := \max_{\substack{v\in\mathbb{F}^d\ \rho\in\mathcal{C}}} \frac{|\rho v|*\infty}{|v|*\infty}.
]
This (T) is the multiplicative factor you pay when you multiply a digit-vector by a challenge. 

There is also an explicit bound for rotation-matrix-derived (\mathcal{C}):
[
\max_{v,\rho}\frac{|\rho v|*\infty}{|v|*\infty}
;\le;
2\phi(\eta)\cdot \max_{\rho'\in \mathcal{C}*R}|\rho'|*\infty,
]
where (\phi) is Euler’s totient function and (\mathcal{C}_R\subseteq R_q) is the corresponding ring challenge set. 

### 6.3 Relaxed binding (the notion used in extraction)

Neo uses **relaxed binding** for commitments: adversaries try to create a “collision under a linear relation” involving challenge differences (\Delta_1,\Delta_2\in(\mathcal{C}-\mathcal{C})):
[
\Delta_1\cdot c=\operatorname{Commit}(\text{pp},Z_1),\quad
\Delta_2\cdot c=\operatorname{Commit}(\text{pp},Z_2),
]
with (|Z_1|*\infty,|Z_2|*\infty<B), but (\Delta_1 Z_2\neq \Delta_2 Z_1).
Neo requires this to be negligible. 

A key bridge lemma is: if the scheme is ((d,m,2TB))-binding then it is ((d,m,B,\mathcal{C}))-relaxed binding. 

---

## 7. The two core relations: MCS and ME

Neo “lifts” CCS into a matrix world and then folds everything through a single evaluation relation.

### 7.1 Structure (CCS data)

A **structure** is:
[
\mathbf{s}:=\left{{M_j\in \mathbb{F}^{n\times m}}_{j\in[t]},;; f\in\mathbb{F}^{<u}[X_1,\dots,X_t]\right}.
]
So: (M_j) are constraint matrices; (f) is the constraint polynomial. 

### 7.2 Matrix constraint system relation ( \mathsf{MCS}(b,\mathcal{L}) )

With witness (w) and public input (x), let (z=x|w) and (Z=\operatorname{Decomp}_b(z)).
Then:

* (c=\mathcal{L}(Z)),
* and the CCS constraint is encoded as a zero-sum condition:
  [
  f(\widetilde{M_1}z,\dots,\widetilde{M_t}z)\in ZS_n.
  ]
  So MCS is “the original statement” (CCS satisfaction + low-norm representability via decomposition + commitment). 

### 7.3 Matrix evaluation relation ( \mathsf{ME}(b,\mathcal{L}) )

ME is the “foldable interface relation.” Its instance includes:
[
(c,; X,; r,; {y_j}_{j\in[t]}),
]
and witness is (Z\in\mathbb{F}^{d\times m}), such that:

1. (c=\mathcal{L}(Z)),
2. (X=\mathcal{L}_x(Z)) (projection to the “public input columns”),
3. (|Z|_\infty<b),
4. for each (j):
   [
   y_j = Z M_j^\top \hat r \in \mathbb{K}^d.
   ]
   This is called a “partial evaluation claim” because you evaluate along the (n)-dimension via (\hat r) but keep the (d)-dimension as a length-(d) vector. 

---

## 8. Folding scheme for CCS: Π_CCS, Π_RLC, Π_DEC

Neo’s folding step is the composition:
[
\Pi := \Pi_{\mathrm{DEC}}\circ \Pi_{\mathrm{RLC}}\circ \Pi_{\mathrm{CCS}}.
]
It consumes (roughly) “one fresh MCS + existing ME accumulator claims” and outputs a refreshed ME accumulator with bounded norm. 

### 8.1 Global reduction parameters (the invariants you must enforce)

Neo fixes:

* (B=b^k < q/2),
* a strong sampling set (\mathcal{C}\subseteq S) with expansion factor (T),
* and requires
  [
  (k+1)T(b-1) < B,\qquad 1/|\mathcal{C}| = \text{negl}(\lambda),
  ]
  and that the commitment is ((d,m,2B,\mathcal{C}))-relaxed binding and (S)-homomorphic. 

These inequalities are not “paper-only”: they are exactly what makes Π_RLC complete and keeps norms below the binding threshold.

---

## 8.2 Π_CCS: reduce CCS satisfaction to evaluation claims via sum-check

**High-level intent.** Π_CCS builds one polynomial (Q) whose sum over the Boolean hypercube encodes:

* the CCS constraints (via a polynomial (F)),
* digit/range constraints enforcing (|Z_i|_\infty<b) (via (NC_i)),
* and “re-randomization” of existing evaluation claims to a fresh random point (via (\mathrm{Eval}_{(i,j)})),

then runs the **sum-check protocol** to reduce that sum claim to one random-point evaluation claim. 

**Key sub-polynomials.**
Using verifier challenges (\alpha,\beta,\gamma), define:

* (F) from the CCS polynomial:
  [
  F := f(\widetilde{M_1z_1},\dots,\widetilde{M_tz_1}),
  ]
* digit/range check for each (Z_i):
  [
  NC_i(\mathbf{X}) := \prod_{j=-(b-1)}^{b-1}\left(\widetilde{Z_i}(\mathbf{X})-j\right),
  ]
* rerandomization terms:
  [
  \mathrm{Eval}*{(i,j)}(\mathbf{X}) := \operatorname{eq}(\mathbf{X},(\alpha,r))\cdot \widetilde{M}*{(i,j)}(\mathbf{X}),
  \quad\text{where } M_{(i,j)}:=Z_i M_j^\top.
  ]
  Then define the sum-check polynomial:
  [
  \begin{aligned}
  Q(\mathbf{X}) :=;& \operatorname{eq}(\mathbf{X},\beta)\Big(F+\sum_{i\in[k]}\gamma^i NC_i(\mathbf{X})\Big)\
  &+\gamma^k\sum_{j=1,i=2}^{t,k}\gamma^{i+(j-1)k-1}\cdot \mathrm{Eval}*{(i,j)}(\mathbf{X}).
  \end{aligned}
  ]
  The claimed sum is:
  [
  T := \gamma^k \sum*{j=1,i=2}^{t,k}\gamma^{i+(j-1)k-1}\cdot \widetilde y_{(i,j)}(\alpha).
  ]
  Sum-check reduces “(\sum_{\mathbf{x}\in{0,1}^{\log(dn)}} Q(\mathbf{x})=T)” to an evaluation claim (v\stackrel{?}{=}Q(\alpha',r')). 

**Why the digit packing shows up in verification.**
The prover sends (y'*{(i,j)} := Z_i M_j^\top \hat r'\in\mathbb{K}^d). The verifier “reconstructs” the scalar value used by (f) by base-(b) packing:
[
m_j := \sum*{\ell\in[d]} b^{\ell-1}\cdot y'_{(1,j),\ell},
\quad
F:=f(m_1,\dots,m_t).
]
Then it checks the derived expression for (v) matches (Q(\alpha',r')) (including the digit/range checks and rerandomization checks). 

---

## 8.3 Π_RLC: compress many ME claims into one ME(B) claim

Input is (k{+}1) ME claims sharing the same evaluation point (r):
[
(s; c_i,X_i,r,{y_{(i,j)}}*{j\in[t]}; Z_i)*{i\in[k+1]} \in \mathrm{ME}(b,\mathcal{L})^{k+1}.
]
Verifier samples (\rho_1,\dots,\rho_{k+1}\leftarrow\mathcal{C}) and computes:
[
c=\sum_i \rho_i c_i,\quad X=\sum_i \rho_i X_i,\quad y_j=\sum_i \rho_i y_{(i,j)}.
]
Prover computes:
[
Z=\sum_i \rho_i Z_i.
]
Output is one claim in (\mathrm{ME}(B,\mathcal{L})). 

**Norm control (completeness):**
[
|Z|*\infty
\le \sum*{i=1}^{k+1}|\rho_i Z_i|*\infty
\le \sum*{i=1}^{k+1} T|Z_i|_\infty
\le (k+1)T(b-1) < B.
]
This is exactly why the parameter inequality matters. 

---

## 8.4 Π_DEC: split the big‑norm ME(B) witness back into k small‑norm ME(b) witnesses

Input is one (\mathrm{ME}(B,\mathcal{L})) with witness (Z).
Prover computes ((Z_1,\dots,Z_k)\leftarrow\mathrm{split}*b(Z)), and sends:
[
c_i:=\mathcal{L}(Z_i),\qquad y*{(i,j)}:=Z_i M_j^\top \hat r.
]
Verifier checks recombination:
[
c \stackrel{?}{=} \sum_{i=1}^{k} \bar b^{,i-1} c_i,
\qquad
y_j \stackrel{?}{=} \sum_{i=1}^{k}\bar b^{,i-1} y_{(i,j)}.
]
Output is (k) claims in (\mathrm{ME}(b,\mathcal{L})^k). 

Π_DEC is directly a reduction of knowledge in Neo’s framework. 

---

## 9. Security architecture (why Neo introduces new definitions)

A normal folding scheme proof wants each reduction to be a standard **reduction of knowledge** (ROK). Neo’s middle reductions ((\Pi_{\mathrm{CCS}}) and (\Pi_{\mathrm{RLC}})) are not standard ROKs *in isolation*, so Neo introduces projection-based relaxations and proves a **composition theorem**.

### 9.1 Standard reduction of knowledge (ROK)

Neo uses the standard “there exists an extractor (\mathcal{E})” definition: if a prover convinces the verifier with probability (\epsilon), the extractor can produce a witness that makes the statement true, with probability (\epsilon - \text{negl}). 

### 9.2 Projection (\phi) and “restricted / relaxed” notions

Neo defines a projection function (\phi:\mathcal{U}\to\mathcal{Z}) and then:

* A reduction is **(\phi)-restricted** if the verifier’s output depends only on (\phi(u_1)) rather than the full input instance (so a malicious prover can’t steer outputs by deviating on the non-(\phi) parts).

* **(\phi)-relaxed knowledge soundness**: extractor is allowed to extract a witness for a *related relaxed relation* (R'_1), and must be “stable” under repeated extraction (otherwise you can break relaxed binding).

* **(\phi)-restricted knowledge soundness**: knowledge soundness only against adversaries that (informally) can’t vary extracted witnesses while holding (\phi(u)) fixed. 

### 9.3 Composition theorem

If:

* (\Pi_1) is complete, public coin, **(\phi)-restricted**, and **(\phi)-restricted knowledge sound**, and
* (\Pi_2) is complete, public coin, and **(\phi)-relaxed knowledge sound**,

then the composition (\Pi_2\circ\Pi_1) is a **standard ROK**. 

Neo then shows:

* (\Pi_{\mathrm{CCS}}) satisfies the (\phi)-restricted properties, and
* (\Pi_{\mathrm{RLC}}) satisfies the (\phi)-relaxed property (tied to strong sampling set invertibility + relaxed binding),
  so (\Pi_{\mathrm{RLC}}\circ\Pi_{\mathrm{CCS}}) is a standard ROK; then composing with (\Pi_{\mathrm{DEC}}) yields the full folding scheme. 

---

## 10. Instantiations (what parameters Neo actually suggests)

Neo provides concrete parameter choices for:

* Mersenne‑61 (q=2^{61}-1),
* Goldilocks (q=2^{64}-2^{32}+1),
* “Almost Goldilocks” (AGL) close to Goldilocks but enabling a power-of-two cyclotomic ring. 

Examples (as given in the instantiation table snippets):

### Almost Goldilocks (AGL)

* (q=(2^{64}-2^{32}+1)-32),
* (\Phi_\eta(X)=X^{64}+1), (d=64),
* (\kappa=13), (m=2^{26}),
* (b=2), (k=11), (B=2^{11}),
* (\mathbb{K}=\mathbb{F}_{q^2}).

### Goldilocks

* (q=2^{64}-2^{32}+1),
* (\Phi_\eta(X)=X^{54}+X^{27}+1), (d=54),
* (\kappa=16), (m=2^{24}),
* (b=2), (k=12), (B=2^{12}),
* (\mathbb{K}=\mathbb{F}_{q^2}).

### Mersenne‑61

* (q=2^{61}-1),
* (\Phi_\eta(X)=X^{54}+X^{27}+1), (d=54),
* (\kappa=16), (m=2^{22}),
* (b=2), (k=12), (B=2^{12}),
* (\mathbb{K}=\mathbb{F}_{q^2}),
* with security figures derived from (|\mathcal{C}|), (|\mathbb{K}|), and MSIS hardness. 

---

## 11. Why this matters for Twist & Shout on top of Neo

Neo explicitly calls out that **Shout (lookups)** and **Twist (read‑write memory)** are sum-check-based memory-checking arguments and that Neo can fold them, yielding a lattice-based instantiation (i.e., plausible post-quantum security) for those arguments by reducing them to the same ME interface Neo already folds. 

From an implementation/interface standpoint:

* Neo’s folding loop is fundamentally “fold ME claims.”
* Any argument (CCS, Shout, Twist, adapters like IDX→OH) that reduces to a batch of ME instances can be injected into the same (\Pi_{\mathrm{RLC}}\rightarrow \Pi_{\mathrm{DEC}}) pipeline without changing those components. (This is exactly the approach in your integration notes.)
* The only hard requirement is that the ME instances you combine in one (\Pi_{\mathrm{RLC}}) call must share the same (r) (since Π_RLC’s input instances share (r) explicitly).
