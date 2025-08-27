# Neo Implementation Requirements (Derived from Nguyen & Setty 2025)

> **Legend**
> **MUST** = mandatory for cryptographic correctness / soundness
> **SHOULD** = strongly recommended for performance or UX
> **NICE** = optional polish / quality-of-life

This document specifies **normative requirements per crate** for implementing *Neo: Lattice-based folding scheme for CCS over small fields with pay-per-bit commitments*. It reflects our workspace layout and the paper's design: **Ajtai is always on**, there is **one FS transcript** and **one sum-check over** $K=\mathbb F_{q^2}$, and **simulated FRI is removed** (Spartan2 is the only succinct backend).&#x20;

---

## Global project choices & invariants

* **Ajtai-only commitment** (no feature flags or alternate backends).
* **One sum-check over $K=\mathbb F_{q^2}$** (soundness $\le \ell d/|K|$ at 64-bit $q$ ⇒ \~128-bit).
* **Mandatory decomposition & range** inside the folding pipeline.
* **Strong-sampler challenges** with tracked expansion $T$ and enforced $(k+1)T(b-1)<B$.
* **Spartan2 bridge only** for last-mile compression; **no simulated FRI**.
* **Testing policy:** unit/property tests **live in each crate** next to the code they verify; **only integration/black-box tests** go in the top-level `neo-tests` crate.

---

## Crate map (what each crate owns)

| Crate                | What it owns (public surface)                                                                                                                                          | Why its boundary matters                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `neo-params`         | Typed parameter sets; validates $(k+1)T(b-1)<B$; fixes extension $K=\mathbb F_{q^2}$.                                                                                  | Centralizes reduction params & sum-check soundness target.                                |
| `neo-math`           | **field/** $\mathbb F_q$ (Goldilocks/M61/AGL) and $K=\mathbb F_{q^2}$; **ring/** $R_q=\mathbb F_q[X]/(\Phi_\eta)$, `cf/cf^{-1}`, `rot(a)`, $S\subseteq F^{d\times d}$. | Keeps arithmetic small-field-native; provides the $S$-action used by commitments and RLC. |
| `neo-ajtai`          | Ajtai **matrix commitment** $L:\mathbb F_q^{d\times m}\to\mathcal C$ (S-homomorphic); `decomp_b`, `split_b`; pay-per-bit multiply.                                     | Enforces "Ajtai always-on" + verified decomposition/range.                                |
| `neo-challenge`      | Strong sampling set $C=\{\mathrm{rot}(a)\}$; invertibility bound; expansion $T$.                                                                                       | RLC needs invertible deltas and bounded expansion.                                        |
| `neo-ccs`            | CCS loader; linearized helpers; **MCS/ME** relation types.                                                                                                             | Shapes the exact claims reductions manipulate.                                            |
| `neo-fold`           | **One** FS transcript; **one** sum-check over $K$; reductions $\Pi_{\text{CCS}},\Pi_{\text{RLC}},\Pi_{\text{DEC}}$ + composition.                                      | Single sum-check + three-reduction pipeline as in §4–§5.                                  |
| `neo-spartan-bridge` | Translate final $ME(b,L)$ to Spartan2 (real FRI backend only).                                                                                                         | Confines succinct compression to last mile.                                               |
| `neo-tests`          | **Integration** tests & cross-crate benches only.                                                                                                                      | Ensures end-to-end correctness across crate boundaries.                                   |

---

## `neo-params`

| Req        | Description                                                                                               |   |                                                 |
| ---------- | --------------------------------------------------------------------------------------------------------- | - | ----------------------------------------------- |
| **MUST**   | Provide typed presets (AGL/Goldilocks/M61) and **enforce** $(k+1)T(b-1)<B$; reject unsafe params at load. |   |                                                 |
| **MUST**   | Fix $K=\mathbb F_{q^2}$ for the sum-check; expose (                                                       | K | ) for soundness accounting.                     |
| **MUST**   | Export (q,\eta,d,\kappa,m,b,k,B,T,                                                                        | C | ) with docstrings tying each to the reductions. |
| **SHOULD** | Ship profile docs showing why $B=b^k$ and typical $T$ from chosen $C_R$.                                  |   |                                                 |
| **NICE**   | `serde` load/save; human-readable profile IDs.                                                            |   |                                                 |

*Tests:* in-crate unit/property tests validating the inequality and preset integrity.

---

## `neo-math` (field/ & ring/)

### field/

| Req        | Description                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Implement $\mathbb F_q$ (GL/M61/AGL) with canonical and signed views; add $K=\mathbb F_{q^2}$ (conjugation, inverse). |
| **MUST**   | Constant-time basic ops; no secret-dependent branching.                                                               |
| **SHOULD** | Roots-of-unity/NTT hooks sized for ring ops.                                                                          |

### ring/

| Req        | Description                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Define $R_q=\mathbb F_q[X]/(\Phi_\eta)$ and coefficient maps `cf`/`cf^{-1}`; define $\|a\|_\infty=\|\mathrm{cf}(a)\|_\infty$. |
| **MUST**   | Implement `rot(a)` and model $S=\{\mathrm{rot}(a)\}\subseteq F^{d\times d}$; expose left-action on vectors/matrices.          |
| **SHOULD** | Efficient (negacyclic/NTT) multiplication for common $d$; pay-per-bit column-add path.                                        |
| **NICE**   | Small-norm samplers for tests/benches.                                                                                        |

*Tests:* in-crate (field correctness, ring isomorphism $R_q\cong S$, rotation identities).

---

## `neo-ajtai`

| Req        | Description                                                                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Implement **Ajtai matrix commitment**: `Setup` $M\!\leftarrow\!R_q^{\kappa\times m}$; `Commit(pp,Z)` computes $c=\mathrm{cf}(M\cdot\mathrm{cf}^{-1}(Z))$. |
| **MUST**   | **S-homomorphism:** $ \rho_1 L(Z_1)+\rho_2 L(Z_2)=L(\rho_1 Z_1+\rho_2 Z_2)$, $\rho_i\in S$.                                                               |
| **MUST**   | **(d,m,B)-binding** and **(d,m,B,C)-relaxed binding** under MSIS; surface helper checks.                                                                  |
| **MUST**   | **Pay-per-bit embedding** + `decomp_b` and `split_b` with range assertions ($\|Z\|_\infty<b$).                                                            |
| **MUST**   | Ajtai **always enabled**; no alternate backends or feature flags.                                                                                         |
| **SHOULD** | Parameter notes (AGL/GL/M61) and links to estimator scripts (paper App. B).                                                                               |
| **NICE**   | API to link selected CCS witness coordinates to Ajtai digits (for applications).                                                                          |

*Tests:* in-crate (S-linearity; binding/relaxed-binding harnesses; decomp/split identities & negative cases).

---

## `neo-challenge`

| Req        | Description                                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Define/sample **strong set** $C=\{\mathrm{rot}(a)\}$ from small-coeff $C_R\subset R_q$; ensure pairwise differences invertible. |
| **MUST**   | Compute/record **expansion $T$**; export to `neo-params` and `neo-fold`.                                                        |
| **MUST**   | Domain-separated sampling API (transcript-seeded).                                                                              |
| **SHOULD** | Metrics for observed expansion; failure-rate tests for invertibility.                                                           |

*Tests:* in-crate (invertibility property; empirical $T$ bounds).

---

## `neo-ccs`

| Req        | Description                                                                                                       |
| ---------- | ----------------------------------------------------------------------------------------------------------------- |
| **MUST**   | CCS satisfiability: verify $f(Mz)=0$ row-wise; handle public inputs $x\subset z$.                                 |
| **MUST**   | Support relaxed CCS with slack $u$ and factor $e$ (defaults $u{=}0,e{=}1$).                                       |
| **MUST**   | Define **MCS/ME** instance/witness types and consistency checks $c=L(Z), X=L_x(Z), y_j=Z M_j^\top r^{\mathrm b}$. |
| **SHOULD** | Import/export helpers to/from common arithmetizations.                                                            |

*Tests:* in-crate (shape checks; satisfiability on toy instances).

---

## `neo-fold`

| Req        | Description                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Own the **only FS transcript**; public-coin with label-scoped domains.                                                       |
| **MUST**   | Implement **one sum-check over $K=\mathbb F_{q^2}$** on the composed $Q$ (constraints $F$, range $NC_i$, eval $Eval_{i,j}$). |
| **MUST**   | Implement $\Pi_{\text{CCS}}$, $\Pi_{\text{RLC}}$ (using `neo-challenge`), and $\Pi_{\text{DEC}}$; enforce $(k+1)T(b-1)<B$.   |
| **MUST**   | **Compose** the three reductions into the folding step $k\!+\!1\to k$ with restricted/relaxed KS hooks (per paper §5).       |
| **MUST**   | **No simulated FRI**; no other transcripts here.                                                                             |
| **SHOULD** | Serde `ProofArtifact` + timing/size metrics; Schwartz–Zippel property tests.                                                 |
| **NICE**   | Prover trace toggles for profiling.                                                                                          |

*Tests:* in-crate (unit/property for each reduction; composition sanity).

---

## `neo-spartan-bridge`

| Req        | Description                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| **MUST**   | Translate final $ME(b,L)$ into a **Spartan2** proof (setup/prove/verify) over small fields; maintain binding to public IO. |
| **MUST**   | Keep transcript/IO linkage compatible with `neo-fold`.                                                                     |
| **MUST**   | Use **real** FRI only (as required by Spartan2 PCS); no simulated paths.                                                   |
| **SHOULD** | Report proof size/time.                                                                                                    |

*Tests:* in-crate (round-trip on tiny ME instances; IO binding).

---

## `neo-tests` (integration **only**)

| Req        | Description                                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **MUST**   | End-to-end: $\Pi_{\text{DEC}}\circ\Pi_{\text{RLC}}\circ\Pi_{\text{CCS}}$ reduces $k\!+\!1\to k$ with expected norm profile under a preset. |
| **MUST**   | Global parameter gate: presets pass $(k+1)T(b-1)<B$; fail on tampering.                                                                    |
| **MUST**   | Bridge: fold → Spartan2 verify succeeds; tampering (any of $c,X,r,\{y_j\}$) fails.                                                         |
| **SHOULD** | Cross-crate benches and CSV/JSON metrics.                                                                                                  |

> **Note:** All **crate-specific** unit/property tests live **inside their crate** (as described above). `neo-tests` is reserved for integration and black-box validation across crates.

---

## Security & parameter requirements (global)

| Req        | Description                                                                                             |   |                                      |
| ---------- | ------------------------------------------------------------------------------------------------------- | - | ------------------------------------ |
| **MUST**   | Enforce safe params at load: $(k+1)T(b-1)<B$; strong-sampler size/invertibility; (                      | K | ) large enough for target soundness. |
| **MUST**   | Respect restricted/relaxed KS notions and the composition theorem when wiring extractors & transcripts. |   |                                      |
| **MUST**   | Constant-time arithmetic and hashing; avoid secret-dependent control flow.                              |   |                                      |
| **SHOULD** | Provide the paper's estimator/Sage scripts or equivalents to justify MSIS hardness for presets.         |   |                                      |

---

### Reference

All terminology and reductions follow *Neo: Lattice-based folding scheme for CCS over small fields and pay-per-bit commitments* (Nguyen & Setty, ePrint 2025/294).&#x20;