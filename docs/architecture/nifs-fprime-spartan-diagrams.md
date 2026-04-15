# SuperNeo / Nightstream — Architectural Diagrams
## Paper-grounded reference for Π_CCS, Π_RLC, Π_DEC, F', NIFS, (optionally Spartan)

> Authoritative source: `docs/superneo-paper/` (Theorem 1, §7.3–§7.5, §6, §C).
> Generic compiler: `docs/hypernova-paper/` §6.1–§6.3 (Construction 2).
>
> **Every claim below is either tagged `[paper: §X]` (stated in the paper) or
> `[impl]` (an engineering choice made in this codebase). Do not conflate them.**
>
> If you need to derive a different choice later, change `[impl]` items freely;
> `[paper: §X]` items are load-bearing for soundness and cannot be altered
> without re-proving the composition.

---

## Preamble

### What this document is

A reference diagram set that names, in paper-faithful vocabulary, every object
that flows through one step of the SuperNeo folding scheme and through the
NIVC-style compiler that wraps it into an IVC proof system. The objects are:

- **CCS(b, L)** and **CE(b, L)** — the two norm-bounded relations the scheme operates on.
- **Π_CCS, Π_RLC, Π_DEC** — the three interactive reductions they compose.
- **NIFS** — the non-interactive folding scheme produced by Fiat-Shamir of
  Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS.
- **F'** — the augmented step function that wraps NIFS.V in-circuit per
  HyperNova Construction 2.
- **Spartan** — an OPTIONAL terminal SNARK compression backend. The paper's
  normative statement is an IVC scheme; "compress the last fold with Spartan"
  is a `[impl]` Decider choice. Nightstream takes it.

### What this document is not

- **Not a soundness proof.** See SuperNeo Appendix D.3–D.6 and HyperNova H.3.
- **Not a concrete-efficiency analysis.** See SuperNeo §7.6 and Appendix B.
- **Not a zk component.** Zero-knowledge is a separate concern; see HyperNova §7.

### Notation conventions (read before any diagram)

SuperNeo works simultaneously over a field `F`, its extension `K`, and two
cyclotomic rings `R_F = F[X]/Φ(X)`, `R_K = K[X]/Φ(X)` of degree `d`. Witnesses
live in the field; folding happens in the ring. Implementers must not conflate
these views. We use:

| Symbol              | Lives in              | Meaning                                              |
|---------------------|-----------------------|------------------------------------------------------|
| `z ∈ F^{n_F}`       | field vector          | Raw witness (what CCS constrains)                    |
| `**z** ∈ R_F^{n_R}` | ring vector           | Ring embedding of `z` via coefficients (`n_F = d·n_R`) |
| `bar{M_j}`          | `R_F^{m × n_R}`       | Ring-lifted CCS matrix                               |
| `bar{M_j} · **z**`  | `R_F^m` (as vector, viewed as length-m polynomial over `R_F`) | Ring matrix-vector product |
| `hat{·}(r)`         | evaluation map        | MLE evaluation at point `r ∈ K^{log m}` of a length-m ring-poly |
| `y_{i,j} := hat{bar{M_j} · **z_i**}(r) ∈ R_K` | ring eval | The **evaluation claim** the relation CE carries    |
| `ct(y_{i,j}) ∈ K`   | constant term         | Field value recovered when crossing to K-arithmetic  |
| `cf(y)_ℓ ∈ K`       | ℓ-th coefficient      | Coefficient extractor (used inside sum-check)        |
| `x ∈ F^{n_{F,in}}`  | field public input    | The x-component the outer CCS relation sees          |
| `**x** ∈ R_F^{n_{R,in}}` | ring public input | The x-component viewed via coefficient bijection     |
| `L`                 | `R_F^{n_R} → C`       | Ajtai commitment (homomorphic R_F-module map, §4 Def 4) |
| `L_in`              | `R_F^{n_R} → R_F^{n_{R,in}}` | First-indices projection (trivial R_F-module map) |
| `C ⊂ R_F`           | strong sampling set   | Challenge set for Π_RLC (§C Def 17)                  |
| `T`                 | scalar                | **Norm-expansion factor** of `C` (§C Def 17, Thm 9), |
|                     |                       | i.e. `T ≥ ‖ρ·v‖_∞ / ‖v‖_∞` for all `ρ∈C, v∈R_F`     |
|                     |                       | NOT a hit-rate or density                            |

### Shape-evolution shorthand (memorize this)

```
                Π_CCS              Π_RLC              Π_DEC            enc/enc_inst
  CCS(b,L)^K    ─────▶   CE(b,L)   ─────▶   CE(B,L)   ─────▶  CE(b,L)^k  ─────▶  CCS(b,L)
   × CE(b,L)^k          ^{K+k}              (norm B)          (norm b)           (fresh u_{i+1}
  [paper §7.1]  sum-    [§7.3]     weak     [§7.4]     b-ary   [§7.5]             = (c_{i+1}, x_{i+1})
   fresh+run     check  at r'     RLC                 decomp                      with c_{i+1}
                                                                                  = L([x_{i+1},w_{i+1}]))
```

### Legend

| Symbol              | Meaning                                                |
|---------------------|--------------------------------------------------------|
| `CCS(b, L)`         | Norm-bounded CCS relation (§7.1 Def 12)                |
| `CE(b, L)`          | Norm-bounded CCS evaluation relation (§7.1 Def 13)     |
| `CE(b, L)^k`        | `k`-fold product relation (running accumulator shape)  |
| `K`                 | # fresh CCS instances folded per step (Def 1, Def 14)  |
| `k`                 | # running CE slots / b-ary decomposition depth         |
| `b`, `B = b^k`      | Norm bounds (small / large)                            |
| `C ⊂ R_F`           | Strong sampling set (§C Def 17)                        |
| `T`                 | Norm-expansion factor of `C` (Thm 9)                   |
| `f`, `M_j`          | CCS structure (Def 11)                                 |
| `α, γ, r_·, ρ_i`    | Verifier challenges (all FS-derived in NIFS)           |
| `enc`, `enc_inst`   | NIVC encoders; see §8. `enc_inst` is also how we       |
|                     | low-norm-encode the hash-of-state into public input x. |
| `vk_fs`             | FS verifier key. In pure IVC: scalar. In NIVC with     |
|                     | `ℓ` functions: vector `vk_fs = (vk_fs,1,…,vk_fs,ℓ)`.   |
| `[paper §X]`        | Statement is from the paper at section X               |
| `[impl]`            | Statement is a codebase engineering choice             |
| `NIFS.P / NIFS.V`   | Native folding scheme prover / verifier                |
| `NIFS.V_circuit`    | R1CS mirror of NIFS.V embedded in F'                   |

---

## §1 Parameters and relations

### §1.1 Global parameters `[paper: §7.2 Def 14 + Appendix B]`

From SuperNeo Appendix B, three concrete parameter sets. `K` shows the paper's
allowable range (upper bound drives `(K+k)·T·(b-1) < B`, Def 14).

| Parameter               | Almost Goldilocks        | Goldilocks               | Mersenne 61            |
|-------------------------|--------------------------|--------------------------|------------------------|
| Field order `q`         | `(2^64 − 2^32 + 1) − 32` | `2^64 − 2^32 + 1`        | `2^61 − 1`             |
| Cyclotomic `Φ(X)`       | `X^64 + 1`               | `X^54 + X^27 + 1`        | `X^54 + X^27 + 1`      |
| Ring degree `d`         | `64`                     | `54`                     | `54`                   |
| Norm `b`                | `2`                      | `2`                      | `2`                    |
| Decomp depth `k`        | `13`                     | `14`                     | `14`                   |
| Fresh count `K ∈`       | `[1, 50]`                | `[1, 61]`                | `[1, 61]`              |
| `B = b^k`               | `2^13`                   | `2^14`                   | `2^14`                 |
| `T` (exp. factor of C)  | `128`                    | `216`                    | `216`                  |
| C coefficient set       | `{−1, 0, 1, 2}`          | `{−2, −1, 0, 1, 2}`      | `{−2, −1, 0, 1, 2}`    |
| `\|C\|`                 | `4^64`                   | `5^54`                   | `5^54`                 |
| `(K+k)·T·(b-1) < B`     | `63·128 = 8064 < 8192`   | `75·216 = 16200 < 16384` | same as Goldilocks     |
| MSIS security           | ≈129 bits                | ≈129 bits                | ≈129 bits              |

> **Nightstream choice `[impl]`**: Goldilocks-family, `b = 2`, `k = 14`,
> `B = 2^14`, `|C| = 5^54`, `T = 216`. `K` is a deployment knob bounded
> above by `61`; choice affects per-step prover cost but not soundness
> framework. At the NIVC compiler boundary we specialize to `K = 1` (see §1.3, §8).

### §1.2 The three relations `[paper: §7.1 Def 12, Def 13]`

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Norm-bounded CCS:  CCS(b, L)                              [§7.1 Def 12] │
│                                                                          │
│    (s ; (c ∈ C_cmt, x ∈ F^{n_{F,in}}) ; w ∈ F^{n_F − n_{F,in}})          │
│                                                                          │
│    where  z := [x, w] ∈ F^{n_F}                                          │
│           **z**        := coefficient embedding ∈ R_F^{n_R}              │
│           c   = L(**z**)                    (Ajtai commits FULL z)       │
│           ‖z‖_∞ < b                         (applies to x AND w)         │
│           f(bar{M_1}·**z**, …, bar{M_t}·**z**) = 0 on {0,1}^{log m}      │
│                                                                          │
│  Role:  "FRESH" per-step claim. Proves one execution of a circuit.       │
│                                                                          │
│  ‼  The commitment is to z = [x, w], not just to w. There is NO          │
│     "commit the witness + carry a separate instance remainder" split     │
│     here (that's HyperNova's LCCCS/CCCS shape, not SuperNeo's).          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Norm-bounded CCS Evaluation:  CE(b, L)                    [§7.1 Def 13] │
│                                                                          │
│    (s ; (c, x, r, {y_j}_{j∈[t]}) ; z)                                    │
│                                                                          │
│    where  c    = L(**z**)                   (Ajtai commitment)           │
│           x    = L_in(**z**)                (public-input projection)    │
│           ‖z‖_∞ < b                         (norm bound)                 │
│           ∀j∈[t]: y_j = hat{bar{M_j}·**z**}(r) ∈ R_K  (ring eval at r)   │
│                                                                          │
│  Role:  One RUNNING slot.                                                │
│                                                                          │
│  Type note on x:                                                         │
│    The paper types x ∈ F^{n_{F,in}}. Under the coefficient bijection      │
│    F^{n_{F,in}} ≅ R_F^{n_{R,in}} (n_{F,in} = d·n_{R,in}) we equivalently │
│    view x as **x** ∈ R_F^{n_{R,in}}. Π_RLC does its linear combination   │
│    in the RING view (§5); the FIELD view appears at the CCS boundary.   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Product relation:   CE(b, L)^k                                          │
│                                                                          │
│    ONE relation element is a k-tuple of CE slots:                        │
│                                                                          │
│       (s ; (c_i, x_i, r, {y_{i,j}}_{j∈[t]})_{i∈[k]} ; (z_i)_{i∈[k]})     │
│                                                                          │
│    - Shared structure  s                                                 │
│    - Shared evaluation point  r                                          │
│    - Per-slot commitment, public input, evals, witness                   │
│                                                                          │
│  Role:  The ENTIRE running accumulator is ONE element of this relation,  │
│         NOT k separate CE instances. Π_DEC's output and Π_CCS's next     │
│         input live here.                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why two relations, not one:** CCS captures constraint satisfaction (what
a fresh circuit execution proves); CE captures *ring evaluations of the
ring-lifted witness polynomials at a random point* (what sum-check reduces
CCS to). Folding is cheap in CE-space because CE is linear in `**z**`.

### §1.3 What `K` and `k` mean `[paper: Def 1]`

- `K ∈ ℕ_{≥1}` — **number of fresh `CCS(b,L)` instances the verifier absorbs per step.**
  Nightstream sets `K = 1` `[impl]`. This is also the **compiler-boundary
  specialization**: HyperNova's Construction 2 is stated for NIFS over
  `(R_1, R_2)` with a single fresh `R_2` instance per step, so when we
  instantiate Construction 2 with Π_SuperNeo, the "outer" `K` seen by the
  compiler is always 1. The paper's `K ≤ 61` is Π_SuperNeo's inner
  parallelism budget if a step produced several fresh CCS instances.
- `k ∈ ℕ_{≥1}` — **size of the running accumulator product `CE(b,L)^k`.**
  Forced by `B = b^k`: Π_DEC outputs exactly `k` CE slots (one per b-ary
  digit). Nightstream `k = 14` `[impl]`, matching Goldilocks Appendix B.

### §1.4 Why three stages (vs HyperNova's two) `[paper: §2.4]`

HyperNova's folding scheme is `Π_RLC ∘ Π_CCS`. SuperNeo needs one more
stage because of the lattice setting:

```
 Lattice problem                     │  Fix
─────────────────────────────────────┼──────────────────────────────────────
 Π_RLC in ring yields witness of     │  Π_DEC: b-ary decomposes B → k
 growing norm B = b^k after fold     │  pieces of norm b, ready to feed
 (HyperNova in field: norm doesn't   │  into next step's Π_CCS as the
 matter, Pedersen binding is total)  │  running CE(b,L)^k input.
                                     │
 Π_RLC by itself is only weakly      │  Π_CCS must be strong with a
 knowledge-sound in lattices         │  matching φ; strong-weak composition
                                     │  (Thm 6) then gives a RoK.
                                     │
 Witness of Π_CCS output has to be   │  Norm-check polynomial NC(X̄) is
 norm-bounded, so sum-check needs    │  added to Q(X̄) alongside F(X̄) and
 to enforce this too                 │  Eval(X̄) (signed range; see §4).
```

---

## §2 Module-boundary stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DECIDER / PUBLIC VERIFIER                       [impl]                  │
│     • OPTIONAL compression backend.                                      │
│     • Nightstream choice (§13): run one extra NIFS fold to absorb u_N    │
│       into U_N, then prove the ACCUMULATOR-SATISFIABILITY relation       │
│       "U_final ∈ CE(b,L)^k" with Spartan. Any SNARK for that relation    │
│       works; the paper does not mandate Spartan.                         │
│     • On-chain verifier: one hash-chain check + one SNARK.V.             │
│                                                                          │
│     crates/neo-fold-next/src/rv64im/decider/                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ (final CE(b,L)^k accumulator, Π_N)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  AUGMENTED STEP F' (NIVC compiler)           [paper: HyperNova §6.3]     │
│     • F' = chunk_step ∘ NIFS.V_circuit ∘ hash ∘ enc_inst                 │
│     • Fixed shape ⇒ fixed vk for any compression backend                 │
│     • Public output x := enc_inst( H(vk_fs, i+1, z_0, z_{i+1},           │
│                                      U_{i+1}, pc_{i+1}) )     [impl: H=P2]│
│       (enc_inst is the low-norm encoding mandated by Def 12 on u_{i+1})  │
│     • Fresh instance for next step:                                      │
│         z_{i+1}^{F'} := [x_{i+1}, w_{i+1}]  (R1CS witness of F')         │
│         c_{i+1}      := L( **z_{i+1}^{F'}** )      (Ajtai commits full z)│
│         u_{i+1}      := ( c_{i+1}, x_{i+1} )       ∈ CCS(b,L)            │
│                                                                          │
│     crates/neo-fold-next/src/rv64im/nivc/                                │
│     crates/neo-fold-next/src/rv64im/main_relation_spartan/f_prime.rs     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ (native U_{i+1} == circuit U_{i+1} bit-exact)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  NIFS: FS-compiled Π_SuperNeo                [paper: §2.4, §7, Thm 1]    │
│     Π_SuperNeo  :=  Π_DEC ∘ Π_RLC ∘ Π_CCS                                │
│       Π_CCS    strong, sum-check reduction                               │
│       Π_RLC    weak,   ring-linear combination (independent ρ_i ∈ C)     │
│       Π_DEC    reduction of knowledge, b-ary decomposition               │
│                                                                          │
│     Fiat-Shamir: all verifier messages (α, γ, sum-check challenges,      │
│                  ρ_1..ρ_{K+k}) are squeezed from a transcript; none      │
│                  of them appear in π_fold.                               │
│                                                                          │
│     crates/neo-fold-next/src/rv64im/nifs/                                │
│         ├── pi_ccs.rs                                                    │
│         ├── pi_rlc.rs                                                    │
│         └── pi_dec.rs                                                    │
│                                                                          │
│     circuit mirror: crates/neo-fold-next/src/rv64im/main_relation_circuit│
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ (CCS / CE instances + witnesses)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RELATIONS & PRIMITIVES                       [paper: §4, §7.1]          │
│     • CCS(b,L), CE(b,L), CE(b,L)^k types                                 │
│     • Ajtai ring commitment (homomorphic, (2B, C)-relaxed binding)       │
│     • Coefficient embedding F^{n_F} ↔ R_F^{n_R} (bijection, n_F = d·n_R) │
│     • Ring-lifted matrices bar{M_j} ∈ R_F^{m × n_R}                      │
│     • Ring polynomial evaluation hat{·}(r) : R_F^m → R_K                 │
│     • Poseidon2 transcript (native + circuit)                  [impl]    │
│     • Sum-check engine                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Ownership (Ousterhout-style):**
- Relations & primitives own **"what a valid instance looks like"**; nothing knows about reductions.
- NIFS owns **"how claims fold"**; nothing knows about IVC indexing or F'.
- F' owns **"how one IVC step is encoded"**; nothing knows about deciders.
- Decider / verifier owns **"how we ship a verifier-facing proof"**; it is
  the only one that chooses a compression backend.

---

## §3 Shape evolution through one fold step

This is the most important diagram. It is the native Π_SuperNeo; the
circuit inside F' is a bit-exact mirror of the verifier side.

```
INPUT  to step i+1  [paper: §7.3 Input]:
   U_i  =  one element of CE(b, L)^k          ← running accumulator from step i,
                                                 a k-tuple sharing s and r
   u_i  =  K fresh CCS(b, L) instances        ← from this step's chunk
                                                 (prover pre-committed W_i, w_i).
                                                 Each u_i = (c, x) with c = L(**z**)
                                                 for z = [x, w]. At the NIVC
                                                 compiler boundary, K = 1.

╔═══════════════════════════════════════════════════════════════════════════╗
║  Π_CCS   [paper: §7.3]   STRONG interactive reduction                     ║
║                                                                           ║
║  Input  ∈  CCS(b, L)^K  ×  CE(b, L)^k                                     ║
║  Output ∈  CE(b, L)^{K+k}                                                 ║
║                                                                           ║
║  Mechanics (see §4 for details):                                          ║
║    • Verifier sends (FS-squeezes) challenges α ∈ K^{log m},  γ ∈ K        ║
║    • Prover + verifier run sum-check on                                   ║
║                                                                           ║
║      Q(X̄) = eq(X̄, α) · ( F(X̄) + γ^K · NC(X̄) )  +  γ^{2K+k} · Eval(X̄)     ║
║                                                                           ║
║        F(X̄)     encodes the K CCS constraints                             ║
║        NC(X̄)    encodes the K+k signed-range polys  [SuperNeo-specific]   ║
║        Eval(X̄)  encodes the k prior-step evaluation claims                ║
║    • log m sum-check rounds → point r' ∈ K^{log m}                        ║
║    • Prover sends y'_{i,j} = hat{bar{M_j}·**z_i**}(r') ∈ R_K              ║
║      for all i ∈ [K+k], j ∈ [t]                                           ║
║    • Verifier checks v == eq(r',α)·(F+γ^K·N) + γ^{2K+k}·E                 ║
║                                                                           ║
║  ★ Result: both CCS matrices and the old r-point are gone. All K+k        ║
║    claims now live in CE, at the shared new point r', as ring evals.      ║
╚════════════════════════════════╦══════════════════════════════════════════╝
                                 │   K+k  CE claims at same point r',
                                 │   each witness still has ‖z_i‖_∞ < b
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║  Π_RLC   [paper: §7.4]   WEAK interactive reduction                       ║
║                                                                           ║
║  Input  ∈  CE(b, L)^{K+k}                                                 ║
║  Output ∈  CE(B, L)                                                       ║
║                                                                           ║
║  Mechanics (see §5 for details):                                          ║
║    • Verifier (FS-)samples K+k INDEPENDENT challenges                     ║
║          ρ_1, …, ρ_{K+k}  ←$  C   (strong sampling set; §C Def 17)       ║
║      !! NOT ρ, ρ^2, ρ^3, … — independence keeps norms ≤ T per draw.       ║
║    • Both sides compute (RING view; see §5):                              ║
║          c      := Σ_i ρ_i · c_i                 (commitment side)        ║
║          **x**  := Σ_i ρ_i · **x_i**             (public-input, ring)     ║
║          y_j    := Σ_i ρ_i · y_{i,j}    ∀j∈[t]   (eval side)             ║
║    • Prover also computes                                                 ║
║          **z**  := Σ_i ρ_i · **z_i**             (witness, ring)         ║
║                                                                           ║
║  ★ Output is one CE claim of norm ‖z‖_∞ < B = b^k. Norm grew;             ║
║    Π_DEC fixes it next.                                                   ║
╚════════════════════════════════╦══════════════════════════════════════════╝
                                 │   1 CE claim,  norm B
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║  Π_DEC   [paper: §7.5]   Reduction of Knowledge                           ║
║                                                                           ║
║  Input  ∈  CE(B, L)                                                       ║
║  Output ∈  CE(b, L)^k                                                     ║
║                                                                           ║
║  Mechanics (see §6 for details):                                          ║
║    • Prover b-ary decomposes the field witness z:                         ║
║          (z_1, …, z_k)  ←  split_b(z)                                     ║
║          z = Σ_{i=1..k} b^{i-1} · z_i,   ‖z_i‖_∞ < b                      ║
║    • Prover commits each piece: c_i = L(**z_i**)                          ║
║    • Prover evaluates each piece in ring:                                 ║
║          y_{i,j} := hat{bar{M_j}·**z_i**}(r) ∈ R_K                        ║
║    • Prover sends (c_i, {y_{i,j}}_{j∈[t]})_{i∈[k]} to verifier            ║
║    • Verifier checks the additively-homomorphic sums                      ║
║          c      ?=  Σ_{i=1..k} b^{i-1} · c_i                              ║
║          y_j    ?=  Σ_{i=1..k} b^{i-1} · y_{i,j}       ∀j∈[t]             ║
║                                                                           ║
║  ★ Output: ONE element of CE(b, L)^k (k-tuple at shared r'),              ║
║    each slot with ‖z_i‖_∞ < b. Ready to serve as running input U_{i+1}.   ║
╚════════════════════════════════╦══════════════════════════════════════════╝
                                 │   U_{i+1} ∈ CE(b, L)^k
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║  enc / enc_inst   [paper: HN §6.2 Def 12, §6.3 Construction 2 step 5]    ║
║                                                                           ║
║  F' was just executed: its R1CS run is encoded as a FRESH CCS instance    ║
║  to be folded in step i+2.                                                ║
║                                                                           ║
║  (a) NP-encoding of F'-trace to a CCS witness:                            ║
║        (s_{F'}, x_{F'}, w_{i+1}) ← enc( F', (⊥, y=x_{i+1}), advice )      ║
║        where x_{F'} = x_{i+1}  (public IO of the F' run, already          ║
║                                 low-norm by enc_inst)                     ║
║                                                                           ║
║  (b) SuperNeo commitment to the FULL [x, w] vector:                       ║
║        z_{i+1}^{F'}  :=  [ x_{i+1},  w_{i+1} ] ∈ F^{n_F}                  ║
║        c_{i+1}       :=  L( **z_{i+1}^{F'}** )     (commits the ENTIRE z)║
║        u_{i+1}       :=  ( c_{i+1},  x_{i+1} )     ∈ CCS(b, L) instance   ║
║                                                                           ║
║  (c) Hash-of-state → public input of the NEXT step's F':                  ║
║        h             :=  H( vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1} ) ║
║        x_{i+1}       :=  enc_inst( h )    ∈ F^{n_{F,in}},  ‖x‖_∞ < b      ║
║                                                                           ║
║  Why enc_inst, not raw hash: Def 12 requires ‖z‖_∞ < b on z = [x, w].    ║
║  A raw Poseidon2 digest is NOT low-norm. For b=2, enc_inst bit-           ║
║  decomposes h into ~256 bit-valued field elements (each in {0,1}).       ║
║                                                                           ║
║  ‼  DIFFERENCE FROM HYPERNOVA:                                            ║
║     HyperNova Construction 2 step 6 writes                                ║
║         u_{i+1} ← (Commit(pp, w_{i+1}), u'_{i+1})                         ║
║     i.e. commit the WITNESS only, carry the INSTANCE REMAINDER            ║
║     separately. That is correct for LCCCS/CCCS (HN Construction 1).       ║
║     For a SuperNeo NIFS, the single relation CCS(b,L) commits             ║
║     z = [x, w], so the fresh instance is (c, x) with c = L([x,w]).        ║
╚═══════════════════════════════════════════════════════════════════════════╝

OUTPUT feeding step i+2:
   U_{i+1}  ∈  CE(b, L)^k     (accumulator, shared point r', small norm)
   u_{i+1}  ∈  CCS(b, L)      (fresh, encoding this step's F'-run,
                               shape (c_{i+1}, x_{i+1}) with
                               c_{i+1} = L([x_{i+1}, w_{i+1}]))

INVARIANT (load-bearing for soundness):
   ‖z_i‖_∞ < b        for all i ∈ [k]                       [paper: §7.5 output]
   ‖x_{i+1}‖_∞ < b    (enforced by enc_inst, checked by next NC(X̄))
   ‖c‖                validated implicitly via relaxed binding (Def 4, Thm 2)
```

---

## §4 Π_CCS internals — Strong interactive reduction `[paper: §7.3]`

```
Π_CCS.⟨P, V⟩   ((pk, vk), u_1, w_1)  →  (u_2; w_2)

INPUTS
  (s ; c_i, x_i ; w_i)_{i=1..K}                    ← K fresh CCS instances
      where z_i := [x_i, w_i] ∈ F^{n_F}, c_i = L(**z_i**)
  (s ; c_i, x_i, r, {y_{i,j}}_{j∈[t]} ; z_i)_{i=K+1..K+k}   ← k CE slots of U_i
      where y_{i,j} = hat{bar{M_j}·**z_i**}(r) ∈ R_K

1. V → P :    α  ←$  K^{log m}                     [paper: §7.3 step 1]
              γ  ←$  K
   (In NIFS: α, γ are FS-squeezed from the transcript.)

2. Define  (over K[X_1, …, X_{log m}]):

   F(X̄)    :=  Σ_{i=1..K}  γ^{i-1} ·
                  f( hat{bar{M_1}·**z_i**}(X̄), …, hat{bar{M_t}·**z_i**}(X̄) )
                                                                     [K terms]

   NC(X̄)   :=  Σ_{i=1..K+k} γ^{i-1} · R_b( z̃_i (X̄) )             [K+k terms]
                 where  R_b(T)  :=  ∏_{a = -(b-1)}^{b-1} (T − a)
                 is the SIGNED-RANGE vanishing polynomial of degree 2b−1.
                 For b = 2:  R_2(T) = (T+1)·T·(T−1) = T^3 − T.
                 (Signed range matches SuperNeo Def 3's signed norm;
                 an unsigned ∏_{j=0..b-1}(T−j) would be wrong.)

   Eval(X̄) :=  eq(X̄, r)
               · Σ_{i=K+1..K+k} Σ_{j=1..t} Σ_{ℓ=1..d}
                     γ^{I(i,j,ℓ)} · cf(bar{M_j}·**z_i**)_ℓ (X̄)      [k·t·d]
              where I(i,j,ℓ) = (i−(K+1)) + k(j−1) + kt(ℓ−1)
              and cf(·)_ℓ is the ℓ-th coefficient (field-valued) of
              the ring polynomial bar{M_j}·**z_i**.

   Q(X̄)    :=  eq(X̄, α) · ( F(X̄) + γ^K · NC(X̄) ) + γ^{2K+k} · Eval(X̄)

   Claimed sum:
     T  :=  Σ_{i=K+1..K+k} Σ_{j=1..t} Σ_{ℓ=1..d}  γ^{I(i,j,ℓ)} · cf(y_{i,j})_ℓ

3. P ↔ V :   SumCheck(T; Q)                        [paper: §4 Def 6]
             (log m univariate rounds; final point r' ∈ K^{log m})
             Output claim: "v = Q(r')"

4. P → V :   ∀ i ∈ [K+k], j ∈ [t]:
               y'_{i,j} := hat{bar{M_j}·**z_i**}(r')  ∈ R_K

5. V checks (reconstructing the three summands at r'):

   F_val  :=  Σ_{i=1..K}  γ^{i-1} · f( ct(y'_{i,1}), …, ct(y'_{i,t}) )
   N_val  :=  Σ_{i=1..K+k} γ^{i-1} · R_b( ct(y'_{i,1}) )
                 (same signed-range polynomial as NC above; ct(·)
                  recovers the field constant term)
   E_val  :=  eq(r', r) · Σ γ^{I(i,j,ℓ)} · cf(y'_{i,j})_ℓ

   v  ?=  eq(r', α) · ( F_val + γ^K · N_val ) + γ^{2K+k} · E_val

6. OUTPUT   (s ; c_i, x_i, r', {y'_{i,j}}_{j∈[t]} ; z_i)_{i∈[K+k]}

LEMMA 3   Π_CCS is STRONG (Def 10) for φ = (c_i)_{i∈[K+k]} (commitment vector). [§7.3]
```

**Why the three polynomial terms?**
- `F(X̄)` forces CCS constraint satisfaction on the K fresh instances.
- `NC(X̄)` forces `‖z_i‖_∞ < b` on all K+k witnesses, using the SIGNED
  range polynomial whose zero set is exactly the signed range the paper's
  norm uses (Def 3). This is the **SuperNeo-specific addition** — not in
  HyperNova.
- `Eval(X̄)` re-asserts the k prior-step evaluation claims at the new point.

**Why "strong":** the output `r'` and `(c_i)_{i∈[K+k]}` are deterministic
functions of the transcript, so any accepting prover commits to a unique
output commitment-projection `φ`.

---

## §5 Π_RLC internals — Weak interactive reduction `[paper: §7.4]`

```
Π_RLC.⟨P, V⟩   ((pk, vk), u_1, w_1)  →  (u_2; w_2)

INPUT   (s ; c_i, x_i, r, {y_{i,j}}_{j∈[t]} ; z_i)_{i∈[K+k]}
                                                    ∈ CE(b, L)^{K+k}

1. V :       ρ_1, …, ρ_{K+k}  ←$  C                [paper: §7.4 step 1]

   ⚠  K+k INDEPENDENT samples from the strong sampling set.
      NOT ρ, ρ^2, …, ρ^{K+k−1} — powers-of-ρ would grow in norm
      exponentially and break Ajtai (2B, C)-relaxed binding.

   Sampling: each ρ_i is FS-squeezed directly into C; there is no
   "accept iff in C" rejection test. See §10 for the concrete
   direct sampler (hash-to-C via base-b_C digit decoding).

   None of the ρ_i appear in π_fold; they are recomputed by the verifier.

   V → P :   (ρ_1, …, ρ_{K+k})

2. V + P both compute — RING VIEW throughout:

   c        :=  Σ_{i=1..K+k}  ρ_i · c_i                (commit side; C_cmt) 
   **x**    :=  Σ_{i=1..K+k}  ρ_i · **x_i**            (public-input, R_F^{n_{R,in}})
   y_j      :=  Σ_{i=1..K+k}  ρ_i · y_{i,j}    ∀j∈[t]  (eval side; R_K)

   P only:
   **z**    :=  Σ_{i=1..K+k}  ρ_i · **z_i**            (witness, R_F^{n_R})

   Decoding back to field (at the CCS boundary, if needed):
   x        :=  field-view of **x** via inverse coefficient bijection
                ∈ F^{n_{F,in}}

3. OUTPUT  (s ; c, x, r, {y_j}_{j∈[t]} ; z)  ∈  CE(B, L)

   Norm bound:  ‖z‖_∞ < (K+k) · T · (b−1)  <  B       [paper: Def 14]
                (T = expansion factor of C; linear in per-coord scaling)

LEMMA 4   Π_RLC is WEAK (Def 9) for φ = (c_i)_{i∈[K+k]} (commitment vector). [§7.4]
```

**Why two views for x?** The paper's Def 13 types `x ∈ F^{n_{F,in}}` in
the RELATION, but Def 14's `L_in : R_F^{n_R} → R_F^{n_{R,in}}` is a
RING map. Under the coefficient bijection `F^{n_{F,in}} ≅ R_F^{n_{R,in}}`
(since `n_{F,in} = d · n_{R,in}`) these are the same data seen two ways.
Π_RLC does its linear combination over `R_F` (so `ρ · **x_i**` is
well-typed); the outer CCS instance carries the field-view if convenient.

**Why "weak":** two accepting transcripts with different ρ-vectors
produce two different output witnesses, so the extractor can only
return one candidate — but that candidate is *relaxed-binding-unique*
(Theorem 2), which is exactly the condition the strong-weak composition
theorem requires.

**Why Π_RLC's φ matches Π_CCS's φ (Theorem 6 prereq):** both project the
full commitment vector `(c_i)_{i∈[K+k]}`, which Π_CCS produces and Π_RLC
consumes unchanged.

---

## §6 Π_DEC internals — Reduction of Knowledge `[paper: §7.5]`

```
Π_DEC.⟨P, V⟩   ((pk, vk), u_1, w_1)  →  (u_2; w_2)

INPUT   (s ; c, x, r, {y_j}_{j∈[t]} ; z)   ∈  CE(B, L)

1. P : compute the b-ary decomposition of the field witness.
       (z_1, …, z_k)  ←  split_b(z)                 [paper: §4 Def 3]
       such that  z = Σ_{i=1..k} b^{i-1} · z_i  and  ‖z_i‖_∞ < b

       For each i ∈ [k]:
           c_i       :=  L(**z_i**)                 (commit piece)
           y_{i,j}   :=  hat{bar{M_j}·**z_i**}(r) ∈ R_K   ∀j∈[t]

   P → V :   (c_i, {y_{i,j}}_{j∈[t]})_{i∈[k]}

2. V : compute the x-side decomposition.
       (x_1, …, x_k)  ←  split_b(x)

       Checks (additively homomorphic, linear in # pieces):

       c      ?=  Σ_{i=1..k}  b^{i-1} · c_i                 [paper §7.5]
       y_j    ?=  Σ_{i=1..k}  b^{i-1} · y_{i,j}   ∀j∈[t]

       (The norm-bound ‖z_i‖_∞ < b is NOT checked here; it is enforced
        by the next step's NC(X̄) inside Π_CCS when these claims
        re-enter as the running CE(b,L)^k input.)

3. OUTPUT  (s ; c_i, x_i, r, {y_{i,j}}_{j∈[t]} ; z_i)_{i∈[k]}
                                                   ∈  CE(b, L)^k
           (one k-tuple element of the product relation)

THEOREM 7   Π_DEC is a Reduction of Knowledge (Def 5).   [§7.5]
```

**Extraction intuition:** an accepting Π_DEC execution gives the
verifier `(c_1, …, c_k)` such that `Σ b^{i-1} · c_i = c`. Two
accepting runs with different piece-vectors would produce two distinct
relaxed openings for the same `c` — break Ajtai relaxed-binding (Thm 2).
So the piece-vector is uniquely determined by the accepting transcript,
which is the RoK knowledge-soundness condition.

---

## §7 NIFS composition via strong-weak-RoK glue `[paper: §2.4, §6 Thm 6, Thm 1]`

```
     Π_SuperNeo  :=  Π_DEC  ∘  Π_RLC  ∘  Π_CCS
                     (RoK)     (weak)     (strong)

            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │    Π_CCS     │      │    Π_RLC     │      │    Π_DEC     │
            │   STRONG     │ ───▶ │    WEAK      │ ───▶ │     RoK      │ ──▶
            │              │      │              │      │              │
            │  φ = (c_i)   │      │  φ = (c_i)   │      │              │
            │ projection   │      │ projection   │      │              │
            └──────────────┘      └──────────────┘      └──────────────┘
                    ▲                     ▲                     ▲
                    └─── Lemma 1 ─────────┘                     │
                       (Thm 6: strong ∘ weak = RoK)              │
                    │ output ∈ CE(B, L) │                        │
                    │  Π_RLC ∘ Π_CCS    │                        │
                    │     is a RoK      │◀──── Lemma 2 ──────────┘
                    │                                  (seq. comp. of RoKs)
                    │  CCS(b,L)^K × CE(b,L)^k  →  CE(b,L)^k       │
                    └────────── Theorem 1 ──────────────▶ Π_SuperNeo is a RoK
                                                          [paper: §2.4, Thm 1]

     KEY: φ is the same function (commitment projection) on both
          Π_CCS's output and Π_RLC's input. That identity is what
          makes Theorem 6 applicable.
```

**Fiat-Shamir step**: Π_SuperNeo is interactive. The NIFS is
its non-interactive compilation: verifier challenges `α, γ, ρ_1, …, ρ_{K+k}`
(and the sum-check per-round challenges) are derived by absorbing the
running transcript into a Poseidon2 sponge. FS is standard; see
HyperNova Appendix B.

**What `π_fold` contains (and does NOT contain):**

```
π_fold  :=  (  sum-check messages g_1..g_{log m},
               y'_{i,j} for (i,j) ∈ [K+k]×[t],        ← from Π_CCS
               (c_i, {y_{i,j}}_{j∈[t]})_{i∈[k]}        ← from Π_DEC
            )

π_fold does NOT contain:
  • α, γ              (FS-squeezed)
  • sum-check round challenges r_1..r_{log m}   (FS-squeezed)
  • ρ_1..ρ_{K+k}      (FS-squeezed — listing them would be a soundness bug)
  • the output U_{i+1} itself  (recomputed by the verifier from π_fold)

Caller convention: when HN Construction 2 writes `NIFS.V(vk_fs, U_i, u_i, π)`,
the `π` argument IS exactly π_fold as above — no U_{i+1} is passed.
```

---

## §8 F' circuit structure `[paper: HyperNova §6.3 Construction 2]`

F' is the augmented step function of HyperNova's compiler. HyperNova's
generic compiler is stated for an NIFS over a single pair `(R_1, R_2)` and
a single fresh `R_2`-instance per step (Construction 2, step 4). When we
instantiate this compiler with Π_SuperNeo, we are therefore using the
**`K = 1` specialization** of Π_SuperNeo at the compiler boundary.

Nightstream further specializes to ℓ = 1 (single program, pure IVC) so
`pc` is trivial.

```
F'(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), ω_i, π_fold)  →  x_{i+1}

   where vk_fs is SCALAR (pure IVC, ℓ=1).
   In generic NIVC with ℓ programs: vk_fs = (vk_fs,1, …, vk_fs,ℓ) and the
   circuit selects vk_fs[pc_i] before running NIFS.V.           [HN §6.3 K step 2]

┌──────────────────────────────────────────────────────────────────────────┐
│                      F'  (expressed in R1CS)                             │
│                                                                          │
│  (1) chunk_step_circuit                       (2) NIFS.V_circuit         │
│      (z_i, ω_i) → z_{i+1}                         (reads π_fold as NDT)  │
│                                                                          │
│      This is the per-step program logic.     Internally three phases:    │
│                                                                          │
│                                               a) Π_CCS.V_circuit         │
│                                                  - absorb u_i, U_i       │
│                                                  - squeeze α, γ          │
│                                                  - replay sum-check      │
│                                                    (log m rounds)        │
│                                                  - check v vs            │
│                                                    eq·(F+γ^K·N)+γ^{2K+k}·E│
│                                                                          │
│                                               b) Π_RLC.V_circuit         │
│                                                  - for i∈1..K+k:         │
│                                                      direct sampler      │
│                                                      squeezes ρ_i ∈ C    │
│                                                      (see §10; bounded   │
│                                                      integer-rejection)  │
│                                                  - form c, **x**, {y_j}  │
│                                                    (ring view)           │
│                                                                          │
│                                               c) Π_DEC.V_circuit         │
│                                                  - absorb c_1..c_k       │
│                                                  - check c = Σ b^{i-1}·c_i│
│                                                  - check y_j = Σ …       │
│                                                                          │
│                                               Output: U_{i+1} ∈ CE(b,L)^k│
│                                                                          │
│  (3) (ℓ=1 ⇒ pc_{i+1} trivially = 1)                                      │
│                                                                          │
│  (4) h       :=  H( vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1} )        │
│      x_{i+1} :=  enc_inst( h )                                           │
│                                                                          │
│      [H = Poseidon2]       [impl]                                        │
│      [enc_inst low-norm encoding is MANDATED by HN §6.2 Def 12           │
│       Partial Functions + SuperNeo Def 12 ‖z‖_∞ < b on z = [x, w].       │
│       A raw Poseidon2 digest is NOT admissible as x.]                    │
│                                                                          │
│      For b = 2: enc_inst bit-decomposes h into ~256 bit-valued field     │
│      elements (one per bit of h). Each element ∈ {0, 1}, so ‖x‖_∞ < 2.   │
└──────────────────────────────────────────────────────────────────────────┘

Base case (i = 0)                          [paper: HN §6.3 step 3]:
  - check z_0 == z_i
  - set U_{i+1} := U_⊥   (default instance; see below)

Default instance U_⊥                       [paper: HN §6.2 Def 12]:

    U_⊥  :=  the k-tuple element of CE(b,L)^k whose every slot is:
               (  c = 0,
                  x = 0,                  (zero low-norm public input)
                  r = r_⊥ (fixed),
                  y_j = 0  ∀j ∈ [t],
                  z = 0   (witness)
               )
    z = [x, w] = 0 trivially has ‖z‖_∞ = 0 < b.
    c = L(**0**) = 0 by Ajtai homomorphy.
    ∀j: y_j = hat{bar{M_j}·**0**}(r_⊥) = 0  at any r_⊥.
    So U_⊥ is a satisfying CE(b,L)^k element: this is exactly the
    "default instance" that HN Def 12 (Default instances) requires
    NIVC-compatible NIFS-es to admit.

Inductive case (i ≥ 1)                     [paper: HN §6.3 step 4 — adapted
                                            to SuperNeo's single-relation form]:

  In HyperNova, Construction 2 step 4a parses u_i as (C, u'_i) — a
  witness commitment and an "instance remainder" — because its R_2 relation
  (CCCS) commits the witness w only and carries a separate instance.

  In SuperNeo, there is no such split: the single relation CCS(b,L) has
  u = (c, x) with c = L([x, w]). The step-4a parse therefore becomes:

    - parse u_i as (c_i, x_i)
    - h_prev := H(vk_fs, i, z_0, z_i, U_i, pc_i)
    - check   x_i ?= enc_inst( h_prev )
        (direct comparison of the public-input component to
         enc_inst(hash); there is no separate u'_i object to check)
    - check   1 ≤ pc_i ≤ ℓ                   (ℓ = 1 ⇒ trivial)
    - U_{i+1} := NIFS.V_circuit(U_i, u_i, π_fold)    (CE(b,L)^k output)

  ‼  A note for anyone reading HN §6.3 in parallel:
     HN's `u'_i` equals our `x_i` in the SuperNeo instantiation. The
     `c_i` we parse out is already the Ajtai commitment L([x_i, w_i]),
     which is what NIFS.V_circuit's Π_CCS phase absorbs as "the fresh
     commitment" when forming the running accumulator.
```

**Paper sizing optimization** `[paper: HN §6.1 last paragraph]`:
`F'` outputs only `x_{i+1} = enc_inst(h)`, not the full `U_{i+1}`. The
preimage (everything inside `H`) flows to the next step as
non-deterministic advice. This keeps `|public IO of u_{i+1}|` constant
in `i`.

**Hash-preimage linking across steps:** step `i+1` receives `U_i` as
non-deterministic advice, hashes it into `h_prev`, and checks
`x_i ?= enc_inst(h_prev)`. This ties step `i+1`'s view of `U_i` to
the public input committed in step `i`'s output, which is what closes
the chain end-to-end.

---

## §9 Per-step prover flow — native + circuit

```
Prover state at step i:  Π_i  =  ((U_i, W_i), (u_i, w_i), pc_i)
Prover input:             non-deterministic advice ω_i

┌─ NATIVE (Rust, prover-only) ──────────────┬─ CIRCUIT (R1CS inside F') ───┐
│                                           │                              │
│ 1. pc_{i+1} := φ(z_i, ω_i)                │   (ℓ=1: pc always 1)         │
│                                           │                              │
│ 2. Run NIFS.P (see §4–§6):                │                              │
│      Π_CCS.P → (K+k) CE claims at r'      │                              │
│      Π_RLC.P → 1 CE claim, norm B         │                              │
│      Π_DEC.P → U_{i+1} ∈ CE(b,L)^k        │                              │
│    Outputs:                               │                              │
│      native U_{i+1}                       │                              │
│      π_fold = ( sum-check messages,       │                              │
│                 y'_{i,j} from Π_CCS,      │                              │
│                 (c_i, y_{i,j}) from Π_DEC)│                              │
│                                           │                              │
│      ⚠  NO ρ_i's in π_fold.  NO α/γ.       │                              │
│         Verifier re-squeezes them from FS.│                              │
│                                           │                              │
│ 3. Run F' NATIVELY (replay harness) to    │                              │
│    compute  x_{i+1} = enc_inst(h)  as a   │                              │
│    byte-for-byte mirror of what the       │                              │
│    circuit will.                          │                              │
│                                           │                              │
│ 4.                                        │  F' in R1CS:                 │
│                                           │    chunk_step_circuit →      │
│                                           │    NIFS.V_circuit(π_fold) →  │
│                                           │    circuit U_{i+1}           │
│                                           │    hash h → enc_inst →       │
│                                           │    circuit x_{i+1}           │
│                                           │                              │
│ 5. ASSERT  native U_{i+1}  ==  circuit U_{i+1}  bit-exact  (§15)         │
│                                                                          │
│ 6. Encode this F'-run as a fresh CCS(b,L) instance — SuperNeo form:     │
│      (s_{F'}, x_{F'}, w_{i+1}) ← enc( F', (⊥, x_{i+1}), advice )        │
│      where x_{F'} := x_{i+1}                                             │
│      z_{i+1}^{F'} := [ x_{i+1}, w_{i+1} ]   ∈ F^{n_F}                   │
│      c_{i+1}     := L( **z_{i+1}^{F'}** )    (commits FULL [x, w])      │
│      u_{i+1}     := ( c_{i+1}, x_{i+1} )    ∈ CCS(b, L)                 │
│                                                                          │
│      ‼  NOT  u_{i+1} := (L(w_{i+1}), u'_{i+1}).  That HyperNova-shaped  │
│         form is wrong for SuperNeo — it would commit the witness only   │
│         and break Def 12's "c = L([x, w])" requirement.                 │
│                                                                          │
│ 7. Wrap:                                                                 │
│      Π_{i+1} := ((U_{i+1}, W_{i+1}), (u_{i+1}, w_{i+1}), pc_{i+1})       │
│                                                                          │
│    ⚠  NO per-step SNARK here.                                            │
│    Construction 2's per-step prover ends at step 7; the only SNARK       │
│    (if any) is the terminal one in the Decider (§11, §13).               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## §10 Transcript / Fiat-Shamir flow `[impl]`

> **`[impl]` notice, for the entire section:** the SuperNeo and HyperNova
> papers specify the interactive protocol and the FS model (Appendix B of
> HyperNova). The concrete absorb/squeeze order, the domain-separator tag
> `"SuperNeo-v1"`, the Poseidon2 choice, and the direct-sampler details
> below are **codebase conventions**. Load-bearing only for native↔circuit
> agreement (§15); any of them can be renamed or re-shaped as long as the
> native prover and the circuit verifier agree byte-for-byte.

### §10.1 Direct sampler for `ρ_i ∈ C` (load-bearing for correctness)

The paper fixes `C` (Appendix B) as the set of ring polynomials whose
coefficients lie in a small signed range. For Goldilocks / Mersenne 61:
`C = { Σ_{j=0..53} c_j · X^j : c_j ∈ {−2,−1,0,1,2} }`, so `|C| = 5^54`.
For Almost Goldilocks: `c_j ∈ {−1,0,1,2}`, so `|C| = 4^64`.

The transcript **samples ρ_i directly into C** via base-`b_C` digit decoding
— there is NO ambient-ring squeeze followed by "accept iff in C" test.

```
Procedure  sample_rho(transcript)  →  ρ ∈ C:

   # Parameters (Goldilocks/Mersenne 61):  b_C = 5,  d_C = 54,
   # digit_shift = 2,  cap = b_C^{d_C}  =  5^54

   loop:
     u  ←  squeeze  nbits = ⌈log2(cap)⌉ + 8   integer bits     (safety margin)
     if u < cap:
         break                                              # accept
     else:
         absorb  reject-tag(attempt_count)                  # rebind transcript
         continue

   # Decode u to b_C digits:
   for j in 0..d_C:
       d_j  :=  u mod b_C
       u    :=  u div b_C
       c_j  :=  d_j - digit_shift                           # signed coeff

   return  ρ  :=  Σ_{j=0..d_C-1}  c_j · X^j   ∈  C
```

- The **only** source of retries is the integer-range rejection on
  step `u < cap`. With `nbits ≥ log2(cap) + 8`, single-attempt rejection
  probability ≤ `2^{-8}`; bounded retry count `N_retry = 40` ⇒ abort
  probability `≤ 2^{-320}` (in practice `N_retry` is set in the range
  `[20, 40]` depending on `[impl]` policy).
- `N_retry` is a **fixed R1CS constant** — changing it changes circuit shape,
  so it must be pinned ahead of time.
- The expansion factor `T` (paper: `T = 216` for Goldilocks, `T = 128` for
  Almost Goldilocks) governs norm growth in Π_RLC; it is NOT a hit rate.
  It is used ONLY for the feasibility inequality `(K+k)·T·(b−1) < B`.

### §10.2 Transcript order per step (impl-specific sponge layout)

```
     Step i+1  — single Poseidon2 sponge, shared native ↔ circuit  [impl]

                     T_0 := H( "SuperNeo-v1" || pp_digest )
                                │
   absorb( s_digest )           │
   absorb( U_i, u_i )           │
                                ▼
                          ┌── Π_CCS ───────────────────────────┐
                          │  squeeze α                         │
                          │  squeeze γ                         │
                          │  for round r ∈ 1..log m:           │
                          │    absorb g_r(X)    (SC msg)       │
                          │    squeeze r_r                     │
                          │  (r' := (r_1, …, r_{log m}))       │
                          │  absorb y'_{i,j} for all i,j       │
                          └──────────────┬─────────────────────┘
                                         ▼
                          ┌── Π_RLC ───────────────────────────┐
                          │  for i ∈ 1..K+k:                   │
                          │    ρ_i ← sample_rho(transcript)    │
                          │       (direct sampler, §10.1)      │
                          │  (all ρ_i are independent ∈ C)     │
                          └──────────────┬─────────────────────┘
                                         ▼
                          ┌── Π_DEC ───────────────────────────┐
                          │  absorb ( c_1, …, c_k )            │
                          │  absorb ( y_{i,j} for all i, j )   │
                          │  (no further challenge needed;     │
                          │   homomorphic check is determinstic)│
                          └──────────────┬─────────────────────┘
                                         ▼
                                  T_final → used as seed bound
                                  into h = H(vk_fs, i+1, …),
                                  then x_{i+1} = enc_inst(h).

     CLAUDE.md rule: Poseidon2-only across this entire path. No
     SHA/Blake3 pre-hashes; no cross-family mixing.

     Circuit must replay these absorbs / squeezes in the EXACT order.
     Reordering ⇒ transcript divergence ⇒ soundness break.
```

---

## §11 End-to-end N steps + Decider + Verifier

```
                              PROVER  (offline)

  ω_1       ω_2                ω_3                        ω_N
   │         │                  │                          │
   ▼         ▼                  ▼                          ▼
 ┌────┐   ┌────┐              ┌────┐                     ┌────┐
 │ F' │──▶│ F' │─────────────▶│ F' │────── ... ─────────▶│ F' │
 └────┘   └────┘              └────┘                     └────┘
  Π_1     Π_2                  Π_3                        Π_N
   │       │                    │                          │
   │                                                       │
   │  No per-step SNARK. Per step only:                    │
   │    - fold, F'-native, F'-circuit replay-check,        │
   │    - NP-encode F' into fresh u_{i+1} = (c_{i+1},      │
   │      x_{i+1}) with c_{i+1} = L([x_{i+1}, w_{i+1}]).   │
   │                                                       │
   └───────────────────────────────────────────────────────┘
                                     │
                                     │  Decider runs on (Π_N, x_N)
                                     ▼
                              DECIDER  (§13)          (optional compression)

                                     │  π_decider
                                     ▼
                              VERIFIER  (online)

  ┌──────────────────────────────────────────────────────────────────────┐
  │  V checks (Option B preferred; see §13):                             │
  │    (1)  x_N_final  linkage to (z_0, N, z_N, U_N_final)  via          │
  │         h = Poseidon2(...), x = enc_inst(h)                          │
  │    (2)  SNARK.V( vk_snark, π_decider,                                │
  │                  public = <relation-specific, see §13> )             │
  │                                                                      │
  │  Verifier work = O(1) Poseidon2 + O(1) SNARK.V                       │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## §12 Where the terminal SNARK (Spartan) fits — zoom

Common misconception: "F' is inside Spartan." Reverse it:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Terminal SNARK proof (Decider only; Nightstream = Spartan)    [impl]    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                          F'                                        │  │
│  │   ┌──────────────┐    ┌─────────────────────────────┐              │  │
│  │   │ chunk_step   │    │  NIFS.V_circuit (3 stages)  │              │  │
│  │   │ (RV64IM ops) │    │  Π_CCS.V | Π_RLC.V | Π_DEC.V│              │  │
│  │   └──────────────┘    └─────────────────────────────┘              │  │
│  │           │                           │                            │  │
│  │           └─────────────┬─────────────┘                            │  │
│  │                         ▼                                          │  │
│  │                 hash h  →  enc_inst(h)  =  x                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  SNARK public input = (vk_fs, i, z_0, z_i, x, pc_i)                      │
│                       where x = enc_inst(H(...)).                        │
└──────────────────────────────────────────────────────────────────────────┘

Facts:
- NIFS native prover runs OUTSIDE the SNARK, OUTSIDE F'.
- F' CONTAINS NIFS.V_circuit. F' does NOT re-prove sum-check; it CHECKS it.
- The SNARK proves a Decider-chosen relation over the final state (§13).
- Spartan is ONE valid choice; the paper does not require it. Any SNARK
  whose relation matches F''s R1CS shape works.

Per-step: just fold + encode + commit. No SNARK.          [paper: HN §6.3]
Decider:  one SNARK over the final accumulator.           [§13]
```

---

## §13 Decider — compression backend `[impl — not a paper theorem]`

> Normative spec stops at HN §6.3: the recursive NIVC verifier checks
> `(hash linkage) ∧ (every U_i[j] ∈ R_1) ∧ (u_i ∈ R_2)`. Anything that
> compresses this to one SNARK call is a deployment-level optimization.
> The section below specifies Nightstream's exact choice.

### §13.1 Two coherent variants

```
 After N steps:  Π_N = ((U_N, W_N), (u_N, w_N), pc_N)

 Step 0 (common to both variants) — FINAL FOLD:
    Run ONE extra NIFS.P absorbing the fresh u_N into U_N.

      (U_final, W_final, π_final)  ←  NIFS.P(pk_fs, (U_N, W_N), (u_N, w_N))

      Now U_final ∈ CE(b,L)^k is claimed to be a satisfying element,
      and u_N has been folded into it. There are no fresh claims left.

 Variant A — ACCUMULATOR-SATISFIABILITY relation (Nightstream choice):

    SNARK proves:
       ∃ W_final.  (pp, s, U_final, W_final) ∈ CE(b,L)^k
       [i.e., the final accumulator is satisfying]

    SNARK public input:  (vk_fs, N, z_0, z_N, x_N, pc_N, U_final.instance)
    SNARK private input: W_final

    Verifier:
      (1)  recompute  h_N := H(vk_fs, N, z_0, z_N, U_final.instance, pc_N)
      (2)  check      x_N ?= enc_inst(h_N)
      (3)  SNARK.V(vk_snark, π_decider, public)

 Variant B — CONSTRUCTION-2-V-ACCEPTS relation:

    SNARK proves:
       HN-Construction-2-V( vk_fs, (N, z_0, z_N), Π_final ) = 1
       [i.e., the full recursive verifier predicate accepts]

    SNARK public input:  (vk_fs, N, z_0, z_N, x_N, pc_N)
    SNARK private input: Π_final, W_final

    Verifier:
      (1)  SNARK.V(vk_snark, π_decider, public)
      (2)  (the SNARK relation internally contains the hash-chain check)
```

### §13.2 Nightstream's choice `[impl]`

**Variant A** (accumulator-satisfiability):
- SNARK relation is simpler to express (one CE(b,L)^k satisfaction predicate).
- Hash-chain check stays outside the SNARK, keeping the online-verifier
  work transparent and easy to audit on-chain.
- Matches the decider's ownership boundary: it proves the FINAL
  accumulator; the online verifier proves the LINKAGE.
- `vk_snark` is fixed ahead of time (F' shape is fixed), so it can be
  audited independently of the program being proved.

The online verifier stays `O(1) Poseidon2 + O(1) SNARK.V`.

### §13.3 What was wrong with the earlier "SNARK over the last F' run" phrasing

F' itself only proves that ONE more fold step was correctly performed
(chunk_step + NIFS.V_circuit + hash + enc_inst). It does NOT by itself
prove that the running accumulator it produced is satisfying. To close
the recursive argument, one must EITHER:
- explicitly prove the final accumulator is satisfiable (Variant A), OR
- explicitly prove the full V predicate of Construction 2 accepts (Variant B).

Saying "prove the last F' run" is ambiguous between the two and is
incomplete as a Decider specification.

---

## §14 Module layout (crate-level)

```
crates/neo-fold-next/src/rv64im/
│
├── main_relation_ccs.rs                RELATIONS: CCS(b,L) type
│                                       (instance = (c, x) with c = L([x,w]))
├── main_relation_ce.rs                 RELATIONS: CE(b,L) + CE(b,L)^k types [NEW]
│                                       (with ring-view x projection via L_in)
│
├── main_relation_circuit/              NIFS circuit mirrors
│   ├── transcript.rs                   Poseidon2TranscriptCircuit
│   ├── sumcheck_replay.rs              used by pi_ccs circuit
│   ├── rho_sampling.rs                 direct sampler ρ_i ∈ C (§10.1)
│   ├── pi_ccs.rs                       Π_CCS.V_circuit             [refactored]
│   ├── pi_rlc.rs                       Π_RLC.V_circuit                 [NEW]
│   ├── pi_dec.rs                       Π_DEC.V_circuit                 [NEW]
│   └── dec_split.rs                    b-ary homom. check helper       [NEW]
│
├── nifs/                               NIFS native                       [NEW]
│   ├── mod.rs                          Π_SuperNeo composition
│   ├── pi_ccs.rs                       Π_CCS.P / Π_CCS.V (native)
│   ├── pi_rlc.rs                       Π_RLC.P / Π_RLC.V (native)
│   └── pi_dec.rs                       Π_DEC.P / Π_DEC.V (native)
│
├── main_relation_spartan/              F' + Decider circuits
│   ├── chunk_step_ivc.rs               (legacy; delete post-F' cutover)
│   ├── f_prime.rs                      F' R1CS encoder                  [NEW]
│   └── spartan_step.rs                 terminal Spartan proof            [NEW]
│                                       (relation: Variant A, §13.2)
│
├── nivc/                               F' native                         [NEW]
│   ├── mod.rs                          NIVC.{G, K, P, V}
│   ├── f_prime_native.rs               native F' replay harness
│   ├── hash_of_state.rs                h = H(vk_fs, i, z_0, z_i, U_i, pc_i)
│   ├── enc_inst.rs                     low-norm encoding of h → x
│   └── enc.rs                          NP-encoder:
│                                       enc(F', (⊥, x), advice)
│                                         → (s_{F'}, x, w)
│                                       u = (L([x, w]), x)  — SuperNeo form
│
├── decider/                            Decider                           [NEW]
│   ├── mod.rs                          Decider entry points
│   ├── option_a_accumulator.rs         Variant A: prove U_final ∈ CE(b,L)^k
│   └── verify.rs                       on-chain verifier surface
│
└── main_recursion*.rs                  accumulator + owner payload types
```

Rules enforced by the layout:
- `nifs/` knows CCS / CE / Ajtai. No circuit types, no IVC indexing.
- `main_relation_circuit/pi_*.rs` knows R1CS allocators. No native prover logic.
- `nivc/enc.rs` is the **single source of truth** for the SuperNeo-shaped
  fresh-instance encoding `u = (L([x, w]), x)` — do not inline that
  structure anywhere else.
- `nivc/f_prime_native.rs` is a byte-for-byte native mirror of
  `main_relation_spartan/f_prime.rs`. Its sole purpose is the
  replayability assertion (§15).
- `decider/` is the only module aware of "how we ship a proof".

---

## §15 Replayability invariant (load-bearing)

```
         native NIFS.P                            circuit NIFS.V (in F')
    ┌──────────────────────┐                 ┌──────────────────────────┐
    │                      │                 │                          │
U_i ▶│ Π_CCS.P              │                 │ Π_CCS.V_circuit          │
u_i ▶│   sum-check P rounds │───π_fold──────▶ │   sum-check V replay     │
    │   absorb/squeeze FS  │                 │   absorb/squeeze FS       │
    │                      │                 │                          │
    │ Π_RLC.P              │                 │ Π_RLC.V_circuit          │
    │   form c, **x**,     │                 │   form c, **x**, y        │
    │   y (ring view)      │                 │   (ring view)             │
    │   ρ_i ← direct       │                 │   ρ_i ← direct            │
    │      sampler (§10.1) │                 │      sampler (§10.1)      │
    │                      │                 │                          │
    │ Π_DEC.P              │                 │ Π_DEC.V_circuit          │
    │   b-ary decomp       │                 │   homom. check            │
    │                      │                 │                          │
    └─────────┬────────────┘                 └────────────┬──────────────┘
              ▼                                           ▼
    U_{i+1}^(native)        ═══════ must equal ═══════   U_{i+1}^(circuit)
                            bit-exact, every field


Failure modes that violate the invariant (all = soundness breaks):
  • Different FS domain-separator tags native vs circuit
  • Different iteration order for Σ ρ_i · c_i native vs circuit
  • Different ρ_i sampler parameters (nbits, retry-tag encoding, digit map)
  • Different b-ary limb endianness in Π_DEC
  • Different enc_inst bit-ordering native vs circuit
  • Different coefficient-embedding convention F ↔ R_F (byte/limb order)
  • Any digest used as AUTHORITY (CLAUDE.md) rather than recomputed

Tests (colocated in tests/, not in impl files; FoldingMode::Optimized):
  • native_F'(args) == circuit_F'(args)  on random inputs
  • ‖z_i‖_∞ < b after every step (both paths)
  • ‖x‖_∞ < b after enc_inst (both paths)
  • ρ_i sampler: native/circuit produce identical ρ_i on identical transcript
  • Poseidon2-only along this whole path
```

---

## §16 Summary cheat-sheet

| Concept            | Paper role                             | Native                          | Circuit (R1CS)                        | Proved by        |
|--------------------|----------------------------------------|---------------------------------|---------------------------------------|------------------|
| CCS(b, L)          | Fresh per-step relation; u = (c, x)    | `main_relation_ccs.rs`          | `main_relation_circuit/pi_ccs.rs`     | Π_CCS (folded)   |
|                    | with c = L([x, w])                     |                                 |                                       |                  |
| CE(b, L)           | Single running slot (ring-eval claim)  | `main_relation_ce.rs`           | `main_relation_circuit/pi_rlc.rs`     | Π_RLC, Π_DEC     |
| CE(b, L)^k         | Running accumulator (one k-tuple)      | `main_relation_ce.rs`           | composed in NIFS.V_circuit            | Π_DEC output     |
| Π_CCS (strong)     | Sum-check over Q(X̄) (incl. signed NC)  | `nifs/pi_ccs.rs`                | `main_relation_circuit/pi_ccs.rs`     | inside F'        |
| Π_RLC (weak)       | Indep. ρ_i linear combination (ring)   | `nifs/pi_rlc.rs`                | `main_relation_circuit/pi_rlc.rs`     | inside F'        |
| Π_DEC (RoK)        | b-ary witness decomposition            | `nifs/pi_dec.rs`                | `main_relation_circuit/pi_dec.rs`     | inside F'        |
| Π_SuperNeo (NIFS)  | DEC ∘ RLC ∘ CCS                        | `nifs/mod.rs`                   | composed inside `f_prime.rs`          | inside F'        |
| ρ_i direct sampler | Hash → C (base-b_C decode, §10.1)      | `nifs/pi_rlc.rs` helper         | `main_relation_circuit/rho_sampling.rs`| —               |
| enc_inst           | Low-norm encoding of h into x          | `nivc/enc_inst.rs`              | in `f_prime.rs`                       | R1CS constraints |
| enc (SuperNeo)     | F'-trace → (s_{F'}, x, w);             | `nivc/enc.rs`                   | —                                     | —                |
|                    | u = (L([x, w]), x)                     |                                 |                                       |                  |
| F' (one IVC step)  | augmented step function (HN 6.3)       | `nivc/f_prime_native.rs`        | `main_relation_spartan/f_prime.rs`    | SNARK (1×, opt.) |
| NIVC               | IVC protocol wrapper                   | `nivc/mod.rs`                   | —                                     | —                |
| Decider            | prove U_final ∈ CE(b,L)^k (Variant A)  | `decider/option_a_accumulator.rs`| —                                    | Spartan (1×)     |

**Top-level invariants (never negotiable):**

> For all i, all inputs, all randomness:
>   `native_F'(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), ω_i, π_fold)`
>   `== circuit_F'(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), ω_i, π_fold)`
>
> bit-exact.
>
> AND: `‖x_{i+1}‖_∞ < b` is enforced by `enc_inst`, so that the next step's
> NC(X̄) accepts `z_{i+1} = [x_{i+1}, w_{i+1}]` as norm-bounded.
>
> AND: every fresh instance has shape `u = (c, x)` with `c = L([x, w])` —
> NOT `(L(w), u')`. The commitment is to the FULL z = [x, w].

**Soundness chain at a glance:**

```
  Lemma 3  Π_CCS is STRONG with φ = commitments            [paper §7.3]
  Lemma 4  Π_RLC is WEAK   with φ = commitments            [paper §7.4]
  ────────────────────── Theorem 6 ─────────────────────   [paper §6]
  Lemma 1  Π_RLC ∘ Π_CCS is a RoK                          [paper §2.4]
  Theorem 7  Π_DEC is a RoK                                [paper §7.5]
  ────────────────────── Lemma 2 ──────────────────────    [paper §4]
  Theorem 1  Π_SuperNeo = Π_DEC ∘ Π_RLC ∘ Π_CCS is a RoK
             from  CCS(b,L)^K × CE(b,L)^k  to  CE(b,L)^k   [paper §2.4]
  ────────────────────── HN §6.3 ──────────────────────
  Construction 2  (Π_SuperNeo, F') is a sound IVC scheme
```

---

## §17 Paper cross-reference index

| Paper section                                  | Used in §                  |
|------------------------------------------------|----------------------------|
| SuperNeo §2.1 HyperNova recap                  | §1.4                       |
| SuperNeo §2.3 SuperNeo embedding               | §1.2, §1.4                 |
| SuperNeo §2.4 Interactive reductions framework | §1.4, §7                   |
| SuperNeo §4 Def 1 (K, k, b, B, n_F = d·n_R)    | §1.3, Notation             |
| SuperNeo §4 Def 2 (cf, ct)                     | §4, §5                     |
| SuperNeo §4 Def 3 (split_b, signed norm)       | §4 NC(X̄), §6              |
| SuperNeo §4 Def 4 (ring commitment)            | §1.2, §6 extraction        |
| SuperNeo §4 Def 5 (interactive reductions)     | §7                         |
| SuperNeo §4 Def 6 (sum-check)                  | §4                         |
| SuperNeo §4 Thm 2 (Ajtai properties)           | §6 extraction              |
| SuperNeo §6 Def 9, 10 (weak / strong)          | §4, §5, §7                 |
| SuperNeo §6 Theorem 6 (strong-weak comp.)      | §7                         |
| SuperNeo §7.1 Def 11, 12, 13 (s, CCS, CE)      | §1.2                       |
| SuperNeo §7.2 Def 14 (global params; L_in)     | §1.1, §5                   |
| SuperNeo §7.3 Π_CCS + Lemma 3                  | §4                         |
| SuperNeo §7.4 Π_RLC + Lemma 4                  | §5                         |
| SuperNeo §7.5 Π_DEC + Theorem 7                | §6                         |
| SuperNeo §2.4 Theorem 1 (Π_SuperNeo is RoK)    | §3, §7, §16                |
| SuperNeo §C Def 17 (strong sampling set, T)    | §10.1, Notation            |
| SuperNeo §C Thm 9 (expansion factor bound)     | §10.1                      |
| SuperNeo Appendix B (concrete C, T, b, k)      | §1.1, §10.1                |
| HyperNova §6.1 Overview (hash-of-output trick) | §8 sizing                  |
| HyperNova §6.2 Def 12 (NIVC-compat, default,   | §2 F' box, §8 x=enc_inst,  |
|   enc_inst, partial functions)                 | §8 default U_⊥             |
| HyperNova §6.3 Construction 2 (F', K step)     | §8, §9, §11, §12           |
| HyperNova §6.3 Construction 2 V                | §13 (what Option B proves) |
