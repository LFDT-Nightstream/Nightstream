# Twist and Shout Summary

**Authors:** Srinath Setty and Justin Thaler

The paper "Twist and Shout" presents **two sum-check–only memory/lookup arguments**:

* **Shout**: a fast argument for **read-only memory / lookups**. In the paper's read-only memory formulation the table contents are public, and the protocol either:
  (i) assumes the table is *MLE-structured* so the verifier can evaluate (\tilde{\mathrm{Val}}(r)) in (O(\log K)), or
  (ii) for non-MLE-structured tables, has Val committed in advance by an honest party so the verifier can force the needed evaluation opening.
* **Twist**: a fast argument for **read–write memory** (stateful memory), where the memory contents evolve over time, but the prover never commits to the full memory state; instead it commits to **sparse increments**, and the memory state is treated as a **virtual polynomial** recovered inside sum-check.

**(Terminology note)** The paper's "cycle" is a *memory operation* (read or write), not a processor cycle.

Below is a detailed "implementer's summary" with the key objects and formulas.

---

## 1. Core algebraic tools (what everything reduces to)

### 1.1 Multilinear extensions (MLEs)

For a vector/function (f:{0,1}^n \to \mathbb{F}), the multilinear extension (\tilde f:\mathbb{F}^n\to \mathbb{F}) is defined via the multilinear "equality" (indicator) polynomial (\widetilde{\mathrm{eq}}(x,b)):
[
\tilde f(x);=;\sum_{b\in{0,1}^n} f(b)\cdot \widetilde{\mathrm{eq}}(x,b).
]


The equality polynomial is:
[
\widetilde{\mathrm{eq}}(x,b);=;\prod_{i=1}^n \big(x_i b_i + (1-x_i)(1-b_i)\big),
\tag{15}
]
so (\widetilde{\mathrm{eq}}(x,b)=1) if (x=b\in{0,1}^n) and (0) on other Boolean points.

A standard multilinearity fact used constantly is:
[
g(c,x') = (1-c)\cdot g(0,x') + c\cdot g(1,x') \quad \text{for multilinear } g,
\tag{14}
]
which is the basis for “folding” variables during sum-check.

**Implementation note**: You will repeatedly need tables of values (\widetilde{\mathrm{eq}}(r,b)) for all (b\in{0,1}^{\log N}). The paper points to a linear-time method (Lemma 1) for building all these evaluations efficiently.

---

### 1.2 The sum-check protocol (what turns “global constraints” into “one random evaluation”)

Sum-check verifies claims of the form
[
H \stackrel{?}{=} \sum_{b\in{0,1}^{\ell}} g(b),
\tag{16}
]
for an (\ell)-variate polynomial (g) with known degree bounds (d_i) in each variable
(multilinear is the special case (d_i=1) for all (i)).

In round (i), the prover sends a univariate polynomial (s_i(X)) of degree (\le d_i)
and the verifier checks consistency.
In the paper's convention, the prover's message is **(d_i) field elements** (not "coefficients"):
you can view these as evaluations of (s_i) at points ({0,2,3,\dots,d_i}), with (s_i(1)) derived from the
consistency check.

* Round 1: (s_1(0)+s_1(1)=H)
* Round (i>1): (s_i(0)+s_i(1)=s_{i-1}(r_{i-1}))

Then verifier samples a random challenge (r_i\in\mathbb{F}). After (\ell) rounds, the claim reduces to checking (g(r_1,\dots,r_\ell)=s_\ell(r_\ell)).

The paper also recalls the standard “linear-time prover” approach: in each round, maintain an array of partial evaluations and fold it using the new challenge (using multilinearity).

---

### 1.3 The “less-than” polynomial (\widetilde{LT}) (used only in Twist)

Twist needs a multilinear extension (\widetilde{LT}(j',j)) of the Boolean function (LT(j',j)) that outputs (1) iff (\mathrm{int}(j')<\mathrm{int}(j)) (strictly less-than), for (j',j\in{0,1}^{\log T}). The key identity is that it can express a prefix-sum over time via MLEs (see §4 below).

---

## 2. The addressing model: one-hot, and “(d)-dimensional one-hot”

The paper encodes an address (z\in{0,1,\dots,K-1}) as a **one-hot vector** (e_z\in{0,1}^K), where ((e_z)_i=1) iff (i=z).

To reduce costs, it introduces a **(d)-dimensional one-hot encoding**. Assume for simplicity that (K^{1/d}) is an integer. Represent (z) by a tuple ((v_1,\dots,v_d)) where each (v_i\in{0,1}^{K^{1/d}}) is one-hot, and for a decomposed index (k=(k_1,\dots,k_d)\in [K^{1/d}]^d):
[
e_z(k);=;\prod_{i=1}^d v_i(k_i).
]


**Why this matters**: a plain one-hot vector has length (K), while the (d)-dimensional version uses (d\cdot K^{1/d}) indicator entries, which is dramatically smaller when (K) is huge and (d) is chosen appropriately.

In Shout/Twist you do this **per cycle** (j): the read-address indicators become (\mathrm{ra}_1(\cdot,j),\dots,\mathrm{ra}_d(\cdot,j)), and similarly for writes.

---

## 3. The basic problem statements (what Shout and Twist prove)

### 3.1 Read-only memory checking / lookups (Shout)

There is a table (\mathbf{Val}\in\mathbb{F}^K) (think: read-only memory / lookup table). For each cycle (j\in[T]) there is a read address and a returned value. Using one-hot read indicators (\mathbf{ra}(k,j)\in{0,1}), correctness is:
[
\sum_{k\in[K]} \mathbf{ra}(k,j)\cdot \mathbf{Val}(k);=;\mathbf{rv}(j)
\quad\text{for all cycles } j\in[T].
\tag{3}
]


The paper immediately turns this into a single MLE identity checked at a random point (r_{\text{cycle}}):
[
\tilde{\mathbf{rv}}(r_{\text{cycle}});=;\sum_{k\in{0,1}^{\log K}}
\tilde{\mathbf{ra}}(k,r_{\text{cycle}})\cdot \tilde{\mathbf{Val}}(k).
\tag{4}
]


This is the core Shout check when (d=1).

---

### 3.2 Read–write memory checking (Twist)

Now memory evolves due to writes. The paper explains a "naïve" correctness condition for reads (now (\mathbf{Val}(k,j)) depends on time (j)).
(In the base formulation, locations are initialized to 0 and a read returns the most recent prior write to that address (or 0 if none). The paper notes this can be generalized to non-zero initialization by making initial values public or committed.)
[
\sum_{k\in[K]} \mathbf{ra}(k,j)\cdot \mathbf{Val}(k,j) ;=; \mathbf{rv}(j)
\quad\text{for all }j\in[T].
\tag{7}
]


Then it gives the sum-check form (random (r'\in\mathbb{F}^{\log T})):
[
\tilde{\mathrm{rv}}(r') ;=;
\sum_{(k,j)\in{0,1}^{\log K}\times{0,1}^{\log T}}
\widetilde{\mathrm{eq}}(r',j)\cdot \tilde{\mathrm{ra}}(k,j)\cdot \tilde{\mathrm{Val}}(k,j).
\tag{8}
]


But committing to all of (\mathrm{Val}(k,j)) is too expensive. Twist’s key idea is: **never commit to (\mathrm{Val})**; instead commit to sparse **increments** (\mathrm{Inc}) and reconstruct (\mathrm{Val}) virtually.

---

## 4. Shout in detail (read-only memory / lookup argument)

### 4.1 What Shout is trying to give the verifier

Shout is structured as a PIOP where the prover has committed to the address indicators (the one-hot encoding columns), and wants to give the verifier **query access to the virtual polynomial** (\tilde{\mathbf{rv}}) (i.e., the verifier can ask for (\tilde{\mathbf{rv}}(r_{\text{cycle}})) at random points). In the core Shout protocol, this is done by a sum-check instance whose last-round check requires only:

* evaluations of the committed address polynomials at a random point, and
* an evaluation of the table MLE (\tilde{\mathbf{Val}}(r_{\text{address}})), which the verifier can do itself if (\mathbf{Val}) is “structured”.

---

### 4.2 Shout with general (d): the key polynomial identity

With (d)-dimensional one-hot addresses (\tilde{\mathbf{ra}}_i(k_i,j)), the read relation becomes:
[
\tilde{\mathbf{rv}}(r_{\text{cycle}})
;=;
\sum_{k=(k_1,\dots,k_d)\in({0,1}^{\log K/d})^d \atop j\in{0,1}^{\log T}}
\widetilde{\mathrm{eq}}(r_{\text{cycle}},j)\cdot
\Big(\prod_{i=1}^d \tilde{\mathbf{ra}}_i(k_i,j)\Big)\cdot
\tilde{\mathbf{Val}}(k).
\tag{66}
]

This is exactly the “read-checking sum-check” instance Shout runs.

At the end of this sum-check:

* the early (\log K) rounds bind the **address variables** to a random (r_{\text{address}}),
* the last (\log T) rounds bind the **cycle variables** to a random (r'_{\text{cycle}}),
  and the verifier needs (\tilde{\mathbf{ra}}_i(r_{\text{address},i},r'_{\text{cycle}})) plus (\tilde{\mathbf{Val}}(r_{\text{address}})).

---

### 4.3 One-hot correctness (necessary for soundness)

If the “address indicators” are not truly one-hot per cycle, the read-check can be cheated. So Shout is paired with a “one-hot-encoding-checking PIOP” (Figure 8), which checks:

1. **Booleanity** of indicator entries:
   enforce (\mathrm{ra}_i(k,j)\in\{0,1\}) via a zero-check / sum-check of ((\mathrm{ra}_i)^2 - \mathrm{ra}_i).
   (Some instantiations effectively absorb this into the commitment/evaluation layer; the paper sometimes omits its *cost* in that setting.)

2. **Hamming-weight-one**: for each cycle (j), (\sum_{k_i}\mathrm{ra}_i(k_i,j)=1).

3. **(Optional but very useful) address-value oracle** (\tilde{\mathbf{raf}}): it grants query access to the *field element* representation of the address, even though only one-hot indicators were committed. This is done by another sum-check in Figure 8:
   [
   y ;=;
   \sum_{k=(k_1,\dots,k_d),,j}
   \widetilde{\mathrm{eq}}(r'_{\text{cycle}},j)\cdot
   \Big(\sum_{i=1}^{d}\sum_{\ell=0}^{\log(K)/d-1} 2^{i\cdot\log(K)/d+\ell}\cdot k_{i,\ell}\Big)\cdot
   \prod_{i=1}^d \tilde{\mathbf{ra}}_i(k_i,j).
   ]


This sum is essentially “replace (\tilde{\mathbf{Val}}(k))” in the read-check with (\widetilde{\mathrm{int}}(k)) (the integer value of the address). The paper explicitly points this out and notes you can batch it with the read-check using a random linear combination. 

For (d=1), the one-hot check is analyzed as Theorem 2 (Figure 6).
For general (d), the one-hot check is Figure 8 and its bound is in Theorem 3.

---

### 4.4 Batching multiple sum-checks

Shout (and the one-hot checks) require multiple sum-check instances. The paper uses the standard technique: combine multiple claims into one by sampling a random scalar (z) and proving the random linear combination, which increases soundness error only additively (and avoids (t)-fold proof blowup).

---

### 4.5 Soundness statement for Shout

**(Conditional on one-hot correctness.)** Theorem 3: assuming each per-cycle address column is a valid (d)-dimensional one-hot encoding, Figure 7 has soundness error at most:
[
\big((d+2)\log T + 2\log K\big)/|\mathbb{F}|.
]
Figure 8 (the one-hot-check PIOP) has its own soundness error at most:
[
\big(4d\log T + 6\log K\big)/|\mathbb{F}|.
]

---

## 5. Twist in detail (read–write memory)

### 5.1 The “increment” idea (Twist’s main conceptual move)

Twist avoids committing to the entire time-indexed memory state (\mathrm{Val}(k,j)). Instead, it commits to a sparse matrix (\mathrm{Inc}(k,j)) defined by the write semantics:
[
\mathrm{Inc}(k,j) \;:=\; \mathrm{wa}(k,j)\cdot(\mathrm{wv}(j)-\mathrm{Val}(k,j)).
\tag{9}
]
On Boolean indices, this equals the per-cycle delta:
[
\mathrm{Inc}(k,j)=\mathrm{Val}(k,j+1)-\mathrm{Val}(k,j).
]
The paper describes this “remaining issue” and immediately turns it into a sum-check that checks the above at a random point (write-checking sum-check).

Intuition: at time (j), exactly one address is written, so (\mathrm{wa}(k,j)) is one-hot over (k). Therefore only one row entry of (\mathrm{Inc}(\cdot,j)) is nonzero (hence sparse).

Then the memory value is the prefix-sum of increments:
[
\widetilde{\mathrm{Val}}(r_{\text{address}}, r_{\text{cycle}})
;=;
\sum_{j'\in{0,1}^{\log T}}
\widetilde{\mathrm{Inc}}(r_{\text{address}},j')\cdot \widetilde{LT}(j',r_{\text{cycle}}).
\tag{11}
]


This is the **Val-evaluation sum-check**: it lets the verifier obtain (\tilde{\mathrm{Val}}(r_{\text{address}},r_{\text{cycle}})) without ever committing to (\mathrm{Val}). The verifier can compute (\widetilde{LT}) itself in (O(\log T)).

---

### 5.2 The core Twist PIOP (Figure 9): three coupled sum-checks

Figure 9 (core Twist) runs three sum-checks:

#### (A) Read-checking sum-check (same structural shape as Shout, but with time-varying (\mathrm{Val}))

For a verifier-chosen (r'\in\mathbb{F}^{\log T}), prove:
[
\widetilde{\mathrm{rv}}(r') \;=\;
\sum_{k=(k_1,\dots,k_d),\,j}
\widetilde{\mathrm{eq}}(r',j)\cdot
\Big(\prod_{i=1}^d \widetilde{\mathrm{ra}}_i(k_i,j)\Big)\cdot
\widetilde{\mathrm{Val}}(k,j).
\tag{33}
]


This is the direct analogue of Shout’s equation (66), except (\mathrm{Val}) depends on (j).

#### (B) Write-checking sum-check (enforces that committed Inc matches the writes)

In parallel, prove that (\widetilde{\mathrm{Inc}}(r,r')) equals:
[
\sum_{k=(k_1,\dots,k_d),,j}
\widetilde{\mathrm{eq}}(r,k)\cdot \widetilde{\mathrm{eq}}(r',j)\cdot
\Big(\prod_{i=1}^d \widetilde{\mathrm{wa}}_i(k_i,j)\Big)\cdot
\big(\widetilde{\mathrm{wv}}(j)-\widetilde{\mathrm{Val}}(k,j)\big).
\tag{34}
]


A “simpler” (d=1) form of the same check is also shown earlier as equation (12).

#### (C) Val-evaluation sum-check (provides (\widetilde{\mathrm{Val}}(r_{\text{address}},r_{\text{cycle}})))

After (A) and (B) bind a random point ((r_{\text{address}}, r_{\text{cycle}})), the verifier needs (\tilde{\mathrm{Val}}(r_{\text{address}}, r_{\text{cycle}})). This is obtained by:
[
\widetilde{\mathrm{Val}}(r_{\text{address}}, r_{\text{cycle}}) \;=\;
\sum_{j'\in{0,1}^{\log T}}
\widetilde{\mathrm{Inc}}(r_{\text{address}},j')\cdot \widetilde{LT}(j',r_{\text{cycle}}).
]


**End-of-sum-check openings**: The paper explicitly notes that at the end of the read/write sum-checks, the verifier needs evaluations of (\widetilde{\mathrm{wa}}, \widetilde{\mathrm{wv}}, \widetilde{\mathrm{ra}}) from commitments, and (\widetilde{\mathrm{Val}}(r_{\text{address}}, r_{\text{cycle}})) from the Val-eval sum-check. 

---

### 5.3 One-hot correctness in Twist

Twist assumes (for soundness of Figure 9) that read and write addresses are correct (d)-dimensional one-hot encodings (for some (\tilde{\mathrm{raf}}(j)) and (\tilde{\mathrm{waf}}(j))). It states you can enforce this by invoking the Figure 8 one-hot-encoding-checking PIOP for both read and write address polynomials. 

---

### 5.4 Soundness bound for Twist

**(Conditional on one-hot correctness.)** Theorem 4's bound applies assuming read/write addresses are provided as correct (d)-dimensional one-hot encodings (enforce via Figure 8 for both ra and wa):
[
((2d+3)\log T + 3\log K)/|\mathbb{F}|.
]


---

## 6. What makes the prover fast (enough to matter in practice)

A generic sum-check prover over (\ell=\log(K)+\log(T)) variables is (O(2^\ell)=O(KT)), which is too slow. The paper’s key “engineering” point is that *these particular polynomials have special structure*:

* **Address indicators are sparse**: only one address per cycle is active.
* **Memory changes are sparse/local**: from cycle (j) to (j+1), only one location changes (a single write).

For example, for the “naïve read-checking” form (equations (7)/(8)), the paper claims it can implement the prover in
[
O(K + T\log K)
]
time using these properties, versus (O(KT)) naïvely.

It also provides cost summaries. For example, for Shout when (K\gg T), it itemizes the multiplications incurred by read-checking, Booleanity checks, raf-evaluation, etc., totaling (\tilde O(T)) with constants depending on parameters like (d).

---

## 7. Mapping to Neo (implementation notes; not in the paper)

This is the conceptual alignment (not changing Neo’s folding core; just describing what Twist/Shout *emit*).

### 7.1 What must be committed vs what is “virtual”

**Shout (read-only / lookup):**

* **Committed**: address indicator polynomials (\tilde{\mathrm{ra}}_i(\cdot,\cdot)).
* **Virtual**: (\tilde{\mathrm{rv}}(\cdot)) (the “read value” polynomial is defined implicitly by the read-checking identity).
* **Verifier-known**: (\tilde{\mathrm{Val}}(\cdot)) must be evaluable by the verifier at random points (structured/public table).

**Twist (read–write):**

* **Committed**: (\tilde{\mathrm{Inc}}(k,j)), (\tilde{\mathrm{wv}}(j)), and address indicator polynomials (\tilde{\mathrm{ra}}_i,\tilde{\mathrm{wa}}_i).
* **Virtual**: (\tilde{\mathrm{Val}}(k,j)) defined from (\tilde{\mathrm{Inc}}) and (\widetilde{LT}); and (\tilde{\mathrm{rv}}) can also be virtual (query access provided by Twist/Shout).

### 7.2 What each sum-check ultimately requires (=> what becomes Neo “ME claims”)

Every sum-check ends at a random point and requires opening a handful of committed polynomials at that point:

* Shout read-check (eq. 66 / core form eq. (30)): open (\tilde{\mathrm{ra}}_i(r_{\text{address},i},r'_{\text{cycle}})); verifier evaluates (\tilde{\mathrm{Val}}(r_{\text{address}})).

* Twist read-check/write-check (eq. 33/34): open (\tilde{\mathrm{ra}}_i,\tilde{\mathrm{wa}}_i,\tilde{\mathrm{wv}},\tilde{\mathrm{Inc}}) at the random points induced by the sum-check; and obtain (\tilde{\mathrm{Val}}(r_{\text{address}},r_{\text{cycle}})) via Val-eval sum-check. 

* One-hot checking (Figure 8): opens of (\tilde{\mathrm{ra}}_i) at the random points in those sum-checks, plus (optionally) the raf-evaluation output (y=\tilde{\mathrm{raf}}(r_{\text{cycle}})).

In Neo terms: each of these “openings at a random point” is exactly the kind of object you fold (your ME relation is built around commitments + partial evaluations). So architecturally, Twist/Shout can be implemented as reductions that *output a bundle of evaluation claims* consistent with the sum-check transcript.

### 7.3 Address-value oracle (\tilde{\mathrm{raf}}) is already a bridge

Your integration notes discuss “index → one-hot adapters”. The paper’s Figure 8 “raf-evaluation sum-check” is very close in spirit: it lets you recover the **numeric address** polynomial (\tilde{\mathrm{raf}}) from one-hot columns via sum-check (and batch it with Shout).

If your VM naturally represents addresses as integers/bits in CCS, then you will still want an adapter in *the opposite direction* (bit/index → virtual one-hot queries). The paper’s direction is one-hot → integer, but the same sum-check “replace (\tilde{\mathrm{Val}}(k)) by (\widetilde{\mathrm{int}}(k))” trick explains why this is algebraically straightforward.

---

## 8. Minimal “cheat sheet” of the key identities

### Shout (lookup)

* Correctness for all cycles (j):
  [
  \mathrm{rv}(j)=\sum_k \mathrm{ra}(k,j)\cdot \mathrm{Val}(k).
  \tag{3}
  ]

* Random-point MLE form ((d=1)):
  [
  \tilde{\mathrm{rv}}(r_{\text{cycle}})=\sum_k \tilde{\mathrm{ra}}(k,r_{\text{cycle}})\tilde{\mathrm{Val}}(k).
  \tag{4}
  ]

* Random-point MLE form (general (d)):
  [
  \tilde{\mathrm{rv}}(r_{\text{cycle}})
  =\sum_{k,j}\widetilde{\mathrm{eq}}(r_{\text{cycle}},j)\Big(\prod_i \tilde{\mathrm{ra}}_i(k_i,j)\Big)\tilde{\mathrm{Val}}(k).
  \tag{66}
  ]

### Twist (read–write memory)

* Read correctness (naïve):
  [
  \mathrm{rv}(j)=\sum_k \mathrm{ra}(k,j)\mathrm{Val}(k,j).
  \tag{7}
  ]

* Increment definition (write semantics):
  [
  \mathrm{Inc}(k,j):=\mathrm{wa}(k,j)\cdot(\mathrm{wv}(j)-\mathrm{Val}(k,j)).
  \tag{9}
  ]
  On Boolean indices: (\mathrm{Inc}(k,j)=\mathrm{Val}(k,j+1)-\mathrm{Val}(k,j)).

* Virtual memory from prefix sums:
  [
  \tilde{\mathrm{Val}}(r_{\text{address}},r_{\text{cycle}})
  =\sum_{j'}\tilde{\mathrm{Inc}}(r_{\text{address}},j')\cdot \widetilde{LT}(j',r_{\text{cycle}}).
  \tag{11}
  ]

* Core Twist sum-check targets (general (d)):
  [
  \tilde{\mathrm{rv}}(r')=\sum_{k,j}\widetilde{\mathrm{eq}}(r',j)\Big(\prod_i \tilde{\mathrm{ra}}_i(k_i,j)\Big)\tilde{\mathrm{Val}}(k,j),
  \tag{33}
  ]
  [
  \tilde{\mathrm{Inc}}(r,r')=\sum_{k,j}\widetilde{\mathrm{eq}}(r,k)\widetilde{\mathrm{eq}}(r',j)
  \Big(\prod_i \tilde{\mathrm{wa}}_i(k_i,j)\Big)\big(\tilde{\mathrm{wv}}(j)-\tilde{\mathrm{Val}}(k,j)\big).
  \tag{34}
  ]