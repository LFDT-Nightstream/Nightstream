# STRUCTURE.md (minimal workspace, **Goldilocks-only**)

> Goal: the smallest Rust workspace that implements **Neo-style folding** with **Ajtai matrix commitments (pay-per-bit)**, **one sum-check over** $K=\mathbb F_{q^2}$, **Π\_{\text{CCS}} → Π\_{\text{RLC}} → Π\_{\text{DEC}}** under a single FS transcript, **strong-sampler challenges**, and **Spartan2(+real FRI) compression**. This repo fixes the base field to **Goldilocks** $q=2^{64}-2^{32}+1$; no feature flags or alternate fields.&#x20;

---

## Workspace layout

```
neo/
├─ Cargo.toml               # [workspace], shared profiles (no features)
└─ crates/
   ├─ neo-math/             # F_q (Goldilocks) + K=F_{q^2}; R_q; cf/cf^{-1}; rotation ring S
   ├─ neo-ajtai/            # Ajtai matrix commitment (S-homomorphic) + Decomp_b, split_b
   ├─ neo-challenge/        # strong sampling set C ⊂ S; invertibility & expansion checks
   ├─ neo-ccs/              # CCS frontend; linearized relations MCS/ME types
   ├─ neo-fold/             # single transcript + sum-check; Π_CCS, Π_RLC, Π_DEC
   ├─ neo-spartan-bridge/   # final compression: ME → Spartan2(+real FRI)
   └─ neo-integration/      # (tests only) cross-crate & e2e tests
```

**Dependency graph (acyclic):**

```
neo-math (foundation: Fq, K, rings, S-action)
  ├─→ neo-ajtai (matrix commitment)
  ├─→ neo-challenge (strong sampling)
  ├─→ neo-ccs (CCS frontend)
  ├─→ neo-spartan-bridge (final compression)
  └─→ neo-integration (tests only)

neo-ajtai, neo-challenge, neo-ccs
  └─→ neo-fold (single transcript + sum-check)
       └─→ neo-spartan-bridge
            └─→ neo-integration

(simplified view: neo-math → {ajtai,challenge,ccs} → neo-fold → spartan-bridge → integration)
```

Why this is enough: it mirrors Neo's constructions—Ajtai with **pay-per-bit** embedding and S-homomorphism (§3), one sum-check over an **extension of a small prime field** (§2.1, §1.3), CCS→ME relations (§4.1), RLC with **small-norm challenges** (§3.4, §4.5), DEC for norm control (§4.6), and a final **Spartan(+FRI)** compression (§1.5).&#x20;

---

## Crates (what/why)

### `neo-math/`

* **Field & extension:** fixed **Goldilocks** $q=2^{64}-2^{32}+1$; exports `Fq` and $K=\mathbb F_{q^2}$ for \~128-bit sum-check soundness. No features; no alternates.&#x20;
* **Cyclotomic ring:** $R_q=\mathbb F_q[X]/(\Phi_\eta)$ with **η=81**, $\Phi_\eta=X^{54}+X^{27}+1$, $d=54$. Coefficient maps `cf`/`cf_inv`; rotation matrices `rot(a)` forming $S=\{\text{rot}(a)\}\subset \mathbb F_q^{d\times d}$, with $R_q \cong S$.&#x20;
* **Linalg:** small `Matrix<Fq>`/`Vector<Fq>`; left action by `SMatrix`.
  **Invariant:** `rot(a)*cf(b) == cf(a*b)`; scalar matrices are in `S`.&#x20;

### `neo-ajtai/`

* **Ajtai matrix commitment** $L:\mathbb F_q^{d\times m}\to\mathcal C$, **S-homomorphic** and $(d,m,B)$-binding under MSIS; **always on** (no other backends).&#x20;
* **Pay-per-bit embedding:** `decomp_b(z) -> Z` (digits as coefficients), `split_b(Z)` (norm lowering), bit-sparse ring multiply so cost \~ #set bits.&#x20;
  **Contract:** $\rho_1 L(Z_1)+\rho_2 L(Z_2)=L(\rho_1 Z_1+\rho_2 Z_2)$ for $\rho_i\in S$.&#x20;

### `neo-challenge/`

* **Strong set $C\subset S$:** sample $\rho=\text{rot}(a)$ with short coefficients; check **invertible differences** via Lyubashevsky–Seiler bound and track **expansion factor** $T$. For Goldilocks profile we use coeffs in $[-2,-1,0,1,2]$ giving $T\approx 216$.&#x20;

### `neo-ccs/`

* **Frontend:** load CCS matrices $\{M_j\}$ and polynomial $f$.
* **Relations:** **MCS(b,L)** (commit & CCS check) and **ME(b,L)** (partial evals $y_j=Z M_j^\top r^b$, $||Z||_\infty<b$). These are the interfaces the reductions manipulate.&#x20;

### `neo-fold/`

* **Owns the only FS transcript & the only sum-check (over $K=\mathbb F_{q^2}$).**
* **Π\_{\text{CCS}}:** build batched $Q$ (CCS constraints $F$ + **range/norm** polynomials $NC_i$ + eval ties), run **one** sum-check to $(\alpha',r')$, output $k$ ME(b,L).&#x20;
* **Π\_{\text{RLC}}:** combine $k\!+\!1$ ME(b,L) with $\rho_i\in C$ ⇒ one ME(B,L); ensure $\|Z\|_\infty \le (k+1)T(b-1)<B$.&#x20;
* **Π\_{\text{DEC}}:** split base-$b$ to lower norm B→b; verify recomposition of commitment and evals; return $k$ ME(b,L). (This one is RoK by itself.)&#x20;
* **Security wiring:** Π\_{\text{CCS}} is ϕ-restricted + restricted-KS, Π\_{\text{RLC}} is ϕ-relaxed-KS, Π\_{\text{DEC}} is KS; composition yields KS. **Single transcript** across all.&#x20;

### `neo-spartan-bridge/`

* **Only** translates the final linearized **ME(b,L)** claim to **Spartan2(+real FRI)** over the same small field; folding stays hash-free and small-field native.&#x20;

### `neo-integration/` (tests-only crate)

* Cross-crate/e2e tests (see "Tests" below).

---

## Parameters (fixed **Goldilocks** profile)

* **Field:** $q=2^{64}-2^{32}+1$; **extension:** $K=\mathbb F_{q^2}$ (\~128-bit soundness).
* **Cyclotomic:** $\eta=81$, $\Phi_\eta=X^{54}+X^{27}+1$, $d=54$.
* **Ajtai/MSIS rows:** $\kappa=16$.
* **Norm schedule:** $b=2$, $k=12$, $B=b^k=4096$.
* **Strong set $C$:** coeffs in $[-2,-1,0,1,2]$ ⇒ **$T\approx 216$**; invertibility bound $b_{\text{inv}}\approx 2.5\cdot 10^9$.
* **Sanity:** $(k{+}1)T(b{-}1)=13·216·1=2808<B=4096$.
  All match §6.2's "Goldilocks: $2^{64}-2^{32}+1$" profile.&#x20;

---

## Tests layout

* **Per-crate local tests** (in `tests/` folder within each crate, with `#[cfg(test)]`):

  ```
  crates/neo-ajtai/tests/
    ├─ s_homomorphism.rs      # S-linearity property tests
    ├─ binding_tests.rs       # binding/relaxed-binding harness at B
    └─ decomp_split.rs        # decomp_b/split_b consistency & negative cases
  
  crates/neo-challenge/tests/
    ├─ invertibility.rs       # empirical invertibility checks for sampled pairs
    └─ expansion_bounds.rs    # empirical T and expansion factor validation
  
  crates/neo-fold/tests/
    ├─ pi_ccs_tests.rs        # Π_CCS property tests (Schwartz–Zippel & sum-check soundness)
    ├─ pi_rlc_tests.rs        # Π_RLC norm-growth bounds
    ├─ pi_dec_tests.rs        # Π_DEC recomposition & consistency
    └─ composition.rs         # unit tests for reduction composition
  ```

* **Global/integration tests** (`neo-integration/tests/`):

  ```
  e2e_fold_then_dec.rs        # Π_CCS → Π_RLC → Π_DEC roundtrip on small CCS
  composition_ks.rs           # composition theorem shape (restricted/relaxed KS)
  spartan_bridge_smoke.rs     # ME → Spartan2(+FRI) smoke test
  ```

> **Testing structure:** Each crate has its own `tests/` directory with `#[cfg(test)]` annotated test modules, keeping tests separate from implementation code. Only cross-crate integration tests live in `neo-integration`.

---

## Cargo workspace (no features; Goldilocks only)

**Top `Cargo.toml`:**

```toml
[workspace]
members = [
  "crates/neo-math",
  "crates/neo-ajtai",
  "crates/neo-challenge",
  "crates/neo-ccs",
  "crates/neo-fold",
  "crates/neo-spartan-bridge",
  "crates/neo-integration",
]
resolver = "2"

[profile.dev]
opt-level = 1
incremental = true

[profile.release]
lto = "thin"
codegen-units = 1
panic = "abort"
```

**`crates/neo-math/src/lib.rs` (field fix):**

```rust
pub mod goldilocks;           // Fq implementation (q = 2^64 - 2^32 + 1)
pub use goldilocks::Fq;       // no features; this is the only base field
pub mod ext2;                 // K = F_{q^2} over Fq
pub use ext2::K;
pub mod cyclotomic;           // Φ_η = X^54 + X^27 + 1, d=54
pub mod cf;                   // cf / cf_inv
pub mod rot;                  // rot(a) ∈ S ⊂ F_q^{d×d}
pub type SMatrix = rot::SMatrix;
```

*(If you ever need AGL/M61 later, that would be a branch or a new crate—no features in this repo.)*

---

## Public surfaces (minimal)

```rust
// math & rings
use neo_math::{Fq, K, SMatrix, /* cf, rot, etc. */};

// commitments (Ajtai, always on)
use neo_ajtai::{PP, setup, commit, decomp_b, split_b, Commitment};

// relations
use neo_ccs::{CcsShape, MCSInstance, MCSWitness, MEInstance, MEWitness};

// challenges
use neo_challenge::{StrongSet};

// folding (single transcript)
use neo_fold::{FoldTranscript, pi_ccs, rlc, dec}; // Π_CCS, Π_RLC, Π_DEC

// compression (last mile only)
use neo_spartan_bridge::compress_me_to_spartan;
```

---

## Dataflow (one fold step)

1. **Embed & commit:** $z=x||w$ → $Z=\text{Decomp}_b(z)$ → $c=L(Z)$. (**MCS**)&#x20;
2. **Π\_{\text{CCS}} (one sum-check over $K$):** build $Q$ batching CCS $F$, range polynomials, eval ties; rerandomize to $(\alpha',r')$; output $k$ **ME(b,L)**.&#x20;
3. **Π\_{\text{RLC}}:** sample $\rho_i\in C$; fold $k{+}1\to 1$ to **ME(B,L)**; norm bound uses $T$.&#x20;
4. **Π\_{\text{DEC}}:** split base-$b$, verify recomposition; return $k$ **ME(b,L)**. Iterate; compress once via Spartan2(+FRI).&#x20;

---

## Guardrails (compile-time/structure)

* Exactly **one** sum-check over $K=\mathbb F_{q^2}$ (lives only in `neo-fold`); never over a ring.&#x20;
* **Ajtai always on**; **decomposition/range checks are mandatory** and live inside the same transcript as CCS.&#x20;
* **Strong sampler $C$** with invertible differences and bounded expansion $T$ must be used for Π\_{\text{RLC}}.&#x20;
* **Compression only at the end** via Spartan2(+FRI) over the same small field; no simulated components anywhere.&#x20;
