# Neo Implementation Requirements (Derived from Nguyen & Setty 2025)

> **Legend**
> **MUST** = mandatory for cryptographic correctness / soundness  
> **SHOULD** = strongly recommended for performance or UX  
> **NICE** = optional polish / quality-of-life

This document specifies **normative requirements per crate** for implementing *Neo: Lattice-based folding scheme for CCS over small fields with pay-per-bit commitments*. It reflects our workspace layout and the paper's design: **Ajtai is always on**, there is **one FS transcript**, **one sum-check over an extension field $K=\mathbb F_{q^s}$**, and **simulated FRI is removed** (Spartan2 is the only succinct backend).

> **Note on the extension degree.** Neo runs a single sum‑check over an **extension of a small prime field**. Choose the **minimal degree $s$** that achieves the target soundness given the base field $q$. As noted in the paper (footnote 7), **with a 64‑bit base field, $s=2$ suffices for ~128‑bit soundness**; do not assume $s=2$ in general—compute it from the target.

---

## Global project choices & invariants

* **Ajtai-only commitment** (no feature flags or alternate backends).
* **One sum-check over $K=\mathbb F_{q^s}$**, with $s$ computed from the soundness target (see `neo-params`).
* **Mandatory decomposition & range** inside the folding pipeline; **verify openings**.
* **Strong-sampler challenges** with tracked expansion $T$; enforce $(k+1)T(b-1)<B$.
* **Spartan2 bridge only** for last‑mile compression; **no simulated FRI** anywhere.
* **Testing policy:** unit/property tests **live in each crate** next to the code they verify; **only integration/black-box tests** go in the top‑level `neo-tests` crate.

---

## Crate map (what each crate owns)

| Crate                | What it owns (public surface)                                                                                                                                          | Why its boundary matters                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `neo-params`         | Typed parameter sets; validates $(k+1)T(b-1)<B$; computes minimal extension degree $s$ and fixes $K=\mathbb F_{q^s}$.                                            | Centralizes reduction params & the sum-check soundness target.                            |
| `neo-math`           | **field/** $\mathbb F_q$ (Goldilocks / M61) and $K=\mathbb F_{q^s}$ (conjugation, inverse); **ring/** $R_q=\mathbb F_q[X]/(\Phi_\eta)$, `cf/cf^{-1}`, `rot(a)`, $S\subseteq \mathbb F^{d\times d}$. | Keeps arithmetic small‑field‑native; provides the $S$-action used by commitments and RLC. |
| `neo-ajtai`          | Ajtai **matrix commitment** $L:\mathbb F_q^{d\times m}\to\mathcal C$ (S‑homomorphic); `decomp_b`, `split_b`; pay‑per‑bit multiply; verified openings.               | Enforces "Ajtai always‑on" + verified decomposition/range.                                |
| `neo-challenge`      | Strong sampling set $C=\{\mathrm{rot}(a)\}$; invertibility property; expansion $T$.                                                                                | RLC needs invertible deltas and bounded expansion.                                        |
| `neo-ccs`            | CCS loader; linearized helpers; **MCS/ME** relation types.                                                                                                             | Shapes the exact claims reductions manipulate.                                            |
| `neo-fold`           | **One** FS transcript; **one** sum-check over $K$; reductions $\Pi_{\mathrm{CCS}},\Pi_{\mathrm{RLC}},\Pi_{\mathrm{DEC}}$ + composition.                           | Single sum‑check + three‑reduction pipeline (paper §§4–5).                                |
| `neo-spartan-bridge` | Translate final $ME(b,L)$ to Spartan2 (real FRI backend only).                                                                                                       | Confines succinct compression to last mile.                                               |
| `neo-tests`          | **Integration** tests & cross‑crate benches only.                                                                                                                      | Ensures end‑to‑end correctness across crate boundaries.                                   |

---

## `neo-params`

| Req        | Description                                                                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Provide typed presets (Goldilocks, M61) and **enforce** $(k+1)T(b-1)<B$; reject unsafe params at load.                             |
| **MUST**   | Compute the minimal **extension degree $s$** for the target soundness and set $K=\mathbb F_{q^s}$. Expose $s$ and $|K|$.    |
| **MUST**   | Export $(q,\eta,d,\kappa,m,b,k,B,T,C,s)$ with docstrings tying each to the reductions and the sum‑check.                           |
| **SHOULD** | Ship profile docs noting typical $T$ from the chosen challenge sampler, and why $B=b^k$ is used.                                 |
| **NICE**   | `serde` load/save; human‑readable profile IDs.                                                                                       |

*Tests:* in‑crate unit/property tests validating the inequality and preset integrity; tests that $s$ increases when the target soundness is tightened.

---

## `neo-math` (field/ & ring/)

### field/

| Req        | Description                                                                                                                             |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Implement $\mathbb F_q$ (Goldilocks/M61) and $K=\mathbb F_{q^s}$ including conjugation and inversion; document the $s$ interface. |
| **MUST**   | Constant‑time basic ops; no secret‑dependent branching.                                                                                 |
| **SHOULD** | Roots‑of‑unity/NTT hooks sized for ring ops.                                                                                            |

### ring/

| Req        | Description                                                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Define $R_q=\mathbb F_q[X]/(\Phi_\eta)$ and coefficient maps `cf`/`cf^{-1}`; define $\|a\|_\infty=\|\mathrm{cf}(a)\|_\infty$.                              |
| **MUST**   | Implement `rot(a)` and model $S=\{\mathrm{rot}(a)\}\subseteq \mathbb F^{d\times d}$; expose left‑action on vectors/matrices.                                 |
| **SHOULD** | Efficient (negacyclic/NTT) multiplication for common $d$; pay‑per‑bit column‑add path.                                                                       |
| **NICE**   | Small‑norm samplers for tests/benches.                                                                                                                         |

*Tests:* in‑crate (field correctness, ring isomorphism $R_q\cong S$, rotation identities).

---

## `neo-ajtai`

| Req        | Description                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Implement **Ajtai matrix commitment**: `Setup` $M\!\leftarrow\!R_q^{\kappa\times m}$; `Commit(pp,Z)$ computes $c=\mathrm{cf}\!\left(M\cdot\mathrm{cf}^{-1}(Z)\right)$. |
| **MUST**   | **S‑homomorphism:** $ \rho_1 L(Z_1)+\rho_2 L(Z_2)=L(\rho_1 Z_1+\rho_2 Z_2)$, $\rho_i\in S$.                                                                           |
| **MUST**   | **$(d,m,B)$-binding** and **$(d,m,B,C)$-relaxed binding** under MSIS; surface helper checks and verified openings for decomp/range.                                   |
| **MUST**   | **Pay‑per‑bit embedding** + `decomp_b` and `split_b` with range assertions $(\|Z\|_\infty<b)$.                                                                          |
| **MUST**   | Ajtai **always enabled**; no alternate backends or feature flags.                                                                                                         |
| **SHOULD** | Parameter notes (Goldilocks/M61) and a pointer to estimator scripts (see paper App. B).                                                                                   |
| **NICE**   | API to link selected CCS witness coordinates to Ajtai digits (for applications).                                                                                          |

*Tests:* in‑crate (S‑linearity; binding/relaxed‑binding harnesses; decomp/split identities & negative cases; opening verification).

---

## `neo-challenge`

| Req        | Description                                                                                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Define/sample **strong set** $C=\{\mathrm{rot}(a)\}$ from small‑coeff $C_R\subset R_q$; ensure pairwise differences are invertible in $S$ with failure bounded. |
| **MUST**   | Compute/record **expansion $T$**; export to `neo-params` and `neo-fold`.                                                                                             |
| **MUST**   | Domain‑separated sampling API (transcript‑seeded).                                                                                                                     |
| **SHOULD** | Metrics for observed expansion; failure‑rate tests for invertibility.                                                                                                  |

*Tests:* in‑crate (invertibility property; empirical $T$ bounds).

---

## `neo-ccs`

| Req        | Description                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | CCS satisfiability: verify $f(Mz)=0$ row‑wise; handle public inputs $x\subset z$.                               |
| **MUST**   | Support relaxed CCS with slack $u$ and factor $e$ (defaults $u{=}0,e{=}1$).                                   |
| **MUST**   | Define **MCS/ME** instance/witness types and consistency checks $c=L(Z), X=L_x(Z), y_j=Z M_j^\top r^{\mathrm b}$. |
| **SHOULD** | Import/export helpers to/from common arithmetizations.                                                              |

*Tests:* in‑crate (shape checks; satisfiability on toy instances).

---

## `neo-fold`

| Req        | Description                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Own the **only FS transcript**; public‑coin with label‑scoped domains.                                                                              |
| **MUST**   | Implement **one sum‑check over $K=\mathbb F_{q^s}$** on the composed $Q$ (constraints $F$, range $NC_i$, eval $Eval_{i,j}$).              |
| **MUST**   | Implement $\Pi_{\text{CCS}}$, $\Pi_{\text{RLC}}$ (using `neo-challenge`), and $\Pi_{\text{DEC}}$; enforce $(k+1)T(b-1)<B$.                  |
| **MUST**   | **Compose** the three reductions into the folding step $k\!+\!1\to k$ with restricted/relaxed knowledge‑soundness hooks (per paper §5).           |
| **MUST**   | **No simulated FRI**; no other transcripts here.                                                                                                    |
| **SHOULD** | Serde `ProofArtifact` + timing/size metrics; Schwartz–Zippel property tests at the chosen $|K|$.                                                  |
| **NICE**   | Prover trace toggles for profiling.                                                                                                                 |

*Tests:* in‑crate (unit/property for each reduction; composition sanity).

---

## `neo-spartan-bridge`

| Req        | Description                                                                                                                 |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Translate final $ME(b,L)$ into a **Spartan2** proof (setup/prove/verify) over small fields; maintain binding to public IO. |
| **MUST**   | Keep transcript/IO linkage compatible with `neo-fold`.                                                                      |
| **MUST**   | Use **real** FRI only (as required by Spartan2's PCS); no simulated paths.                                                  |
| **SHOULD** | Report proof size/time.                                                                                                     |

*Tests:* in‑crate (round‑trip on tiny ME instances; IO binding).

---

## `neo-tests` (integration **only**)

| Req        | Description                                                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | End‑to‑end: $\Pi_{\text{DEC}}\circ\Pi_{\text{RLC}}\circ\Pi_{\text{CCS}}$ reduces $k\!+\!1\to k$ with expected norm profile under a preset.  |
| **MUST**   | Global parameter gate: presets pass $(k+1)T(b-1)<B$; fail on tampering.                                                                       |
| **MUST**   | Bridge: fold → Spartan2 verify succeeds; tampering (any of $c,X,r,\{y_j\}$) fails.                                                            |
| **SHOULD** | Cross‑crate benches and CSV/JSON metrics.                                                                                                       |

> **Note:** All **crate‑specific** unit/property tests live **inside their crate** (as described above). `neo-tests` is reserved for integration and black‑box validation across crates.

---

## Security & parameter requirements (global)

| Req        | Description                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **MUST**   | Enforce safe params at load: $(k+1)T(b-1)<B$; strong‑sampler size/invertibility; choose $s$ so that single‑run SZ error $\ll 2^{-128}$ (or target). |
| **MUST**   | Respect restricted/relaxed KS notions and the composition theorem when wiring extractors & transcripts.                              |
| **MUST**   | Constant‑time arithmetic and hashing; avoid secret‑dependent control flow.                                                           |
| **SHOULD** | Provide the paper's estimator/Sage scripts or equivalents to justify MSIS hardness for presets; reflect App. B. relationships (e.g., $\propto 8\,T\,B$). |

---

### Reference

All terminology and reductions follow *Neo: Lattice-based folding scheme for CCS over small fields and pay-per-bit commitments* (Nguyen & Setty, ePrint 2025/294). In particular: one sum‑check over an **extension of a small prime field** (single transcript), Ajtai matrix commitments with pay‑per‑bit decomposition and verified openings, the strong‑sampler/expansion analysis and the guard $(k{+}1)T(b{-}1)<B$, and last‑mile succinctness via a real FRI PCS.