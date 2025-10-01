# Nightstream Lattice System Architecture (IVC with On‑Demand SNARK Emission)

This document describes the high‑level architecture of Nightstream's lattice SNARK pipeline, with emphasis on **Incrementally Verifiable Computation (IVC)** implemented via a **public‑ρ embedded verifier (EV)** and **on‑demand SNARK emission**. The pipeline separates **fast per‑step accumulation** from **expensive compression** (Spartan2/SuperSpartan), allowing you to emit proofs *every step*, *at checkpoints*, or only *once at the end*.

---

## System Flow (Conceptual)

```
+-----------------------------+
| High-Level Input            |
| (RISC-V / WASM / Circuit)   |
| + Public Inputs x           |
| + Private Witness w         |
+-----------------------------+
            |
            v  (External to Nightstream)
+-----------------------------+
| Arithmetization to CCS/MCS  |
| - Define CCS structure s    |
| - Decompose z -> Z (digits) |
| - Ajtai commit c = L(Z)     |
|   (Π_Mat)                   |
+-----------------------------+
            |
            v  (pair with running ME claims for recursion)
+-----------------------------+         Poseidon2 Transcript (off-circuit)
| Stage 1: CCS Reduction      |<-------- ρ_t = H(step_no, step_X, y_prev, c_z_digest)
| Π_CCS + Sumcheck            |          (publicly recomputable by verifier)
| - Encode Q polynomial       |
| - Sumcheck over hypercube   |
| - Output: k ME claims       |
|   + optional Shout/Twist    |
+-----------------------------+
            |
            v
+-----------------------------+
| Stage 2: Aggregation        |
| Π_RLC                       |
| - Random lin. combo via     |
|   public ρ_t                |
| - 1 high-norm ME            |
+-----------------------------+
            |
            v
+-----------------------------+
| Stage 3: Decomposition      |
| Π_DEC                       |
| - Split high-norm Z to      |
|   k-1 low-norm              |
| - Output: k-1 ME claims     |
+-----------------------------+
            |
            v
+-----------------------------+  (Cheap per-step loop; no SNARK yet)
| Stage 4: IVC Embedded       |
| Verifier (EV, public-ρ)     |
| - Enforce y_next =          |
|     y_prev + ρ_t * y_step   |
| - Update accumulator:       |
|     (y_prev,c_prev,step)->  |
|     (y_next,c_next,step+1)  |
+-----------------------------+
            |
            |     Emission policy:
            |     ┌───────────────────────────────────────────┐
            |     │ EveryStep    -> compress now              │
            |     │ Checkpoint(N)-> compress every N steps    │
            |     │ FinalOnly    -> compress once at the end  │
            |     └───────────────────────────────────────────┘
            v
+-----------------------------+     (On-demand, expensive)
| Final SNARK Layer           |<---- batches of folded instances
| Spartan/SuperSpartan +      |
| FRI-derived MLE PCS         |
| - Proves ME evaluations     |
| - Produces lean proof       |
+-----------------------------+
            |
            v
+-----------------------------+
| Output: Succinct Proof      |
| - Transcript bindings       |
| - MLE evals + FRI openings  |
| - Lean (no embedded VK)     |
+-----------------------------+
```

---

## What’s New (vs. prior doc)

1. **Public‑ρ Embedded Verifier (EV) inside the step**

   * Each step derives a **Fiat–Shamir challenge ρ** *off‑circuit* via a Poseidon2 transcript bound to **public data**: step number, step public input $X_t$, previous accumulator $y_{t}$, and previous commitment digest $c_z^{(t)}$.
   * The EV CCS **enforces** $y_{t+1} = y_{t} + \rho_t \cdot y^{\text{step}}_t$ with **ρ as a public input**, so the verifier can **recompute ρ** and check soundness.
   * This fixes the “folding with itself” pitfall by requiring **real** $y^{\text{step}}$ extracted from the step’s computation.

2. **Decoupled Emission** (performance)

   * You **don’t** SNARK every step unless you want streaming proofs.
   * Choose an **emission policy**:

     * **EveryStep** — streaming verifiability, most expensive
     * **Checkpoint(N)** — amortize compression across N steps
     * **FinalOnly** — one proof at the end (classic Nova‑style)

3. **Augmented CCS Structure (per-step)**

   * Your step CCS is **augmented** with the EV (public‑ρ) constraints (and, where wired, Ajtai opening/lincomb gadgets for commitment evolution).
   * No in‑circuit hash is required; **ρ is public** and recomputed by the verifier.

---

## Core Stages (Detailed)

### Input & Arithmetization

* **CCS/MCS**: Represent the computation as a customizable constraint system.
* **Decomposition**: Decompose the witness $z$ into digit matrix $Z$ (balanced base‑$b$ decomposition, $D$ digits).
* **Ajtai Commitment (Π\_Mat)**: Commit to $Z$ to obtain $c = L(Z)$.

  * Commitment coordinates can evolve with the accumulator (e.g., $c_{t+1} = c_t + \rho_t c_{\text{step}, t}$) via lincomb gadgets.

### Stage 1 — Π\_CCS (with Sumcheck)

* Encode the CCS into a **Q polynomial** and prove its correct evaluation via **sumcheck** over the hypercube.
* Outputs **k ME claims** (matrix extension claims) representing multilinear evaluations.

**Optional memory arguments** (compatible with Nightstream):

* **Shout**: read‑only lookup folding
* **Twist**: read/write memory folding

### Stage 2 — Π\_RLC (Aggregation)

* Combine $k$ claims into **one** using a random linear combination with **public ρ** (the per‑step challenge).
* Reduces verification to a single high‑norm ME claim.

### Stage 3 — Π\_DEC (Decomposition)

* Split the high‑norm object into $(k-1)$ low‑norm parts, yielding **$(k-1)$ ME claims** that feed the next iteration.

### Stage 4 — IVC Embedded Verifier (EV, public‑ρ)

* **Challenge Derivation (off‑circuit)**:
  $\rho_t \leftarrow \text{Poseidon2}( \text{“neo/ivc”},\ t,\ X_t,\ y_t,\ c^{(t)}_z )$
  The verifier recomputes the same $\rho_t$ from the same public data.
* **EV Constraints (in‑circuit)**:
  Enforce $y_{t+1}[i] - y_t[i] - u_t[i] = 0$ and $u_t[i] = \rho_t \cdot y^{\text{step}}_t[i]$ for all $i$.
  Public inputs: $[ \rho_t \ \Vert\ y_t \ \Vert\ y_{t+1} ]$.
* **Accumulator Update**:
  Update $(y_t, c^{(t)}_z, \text{step}) \to (y_{t+1}, c^{(t+1)}_z, \text{step}+1)$.
  (Commitment evolution via opening/lincomb gadgets is supported in the augmentation when wired.)

> **Why public‑ρ?**
> It guarantees Fiat–Shamir soundness: the challenge is a function of **public transcript data** and is **recomputable** by the verifier. It avoids in‑circuit hash complexity and variable‑sharing pitfalls.

### Emission Strategy — On Demand

* The per‑step loop above is **cheap** (no SNARK compression).
* You can emit a compressed proof according to one of the policies:

| Policy        | When we compress     | Cost per step | Latency   | Who can verify mid‑stream? |
| ------------- | -------------------- | ------------- | --------- | -------------------------- |
| EveryStep     | After each step      | Highest       | Immediate | Anyone                     |
| Checkpoint(N) | Every N steps        | \~1/N of ES   | Bounded   | At checkpoints             |
| FinalOnly     | Once after last step | Lowest        | End only  | Only at the end            |

“Compress” here means: fold the collected MCS instances and run **Spartan/SuperSpartan + FRI PCS** once to get a **lean proof**.

### Stage 5 — Final SNARK (Spartan/SuperSpartan + FRI MLE PCS)

* Not merely “compression”: the SNARK **establishes correctness** of the remaining **multilinear evaluation (ME)** claims using a polynomial IOP with a **transparent FRI‑derived PCS** (BaseFold/DeepFold).
* Output is a **lean proof** (no embedded VK); verification uses a cached VK digest registry and checks the **context digest** binding to $(\text{CCS}, x)$.

---

## Accumulator & Transcript (What is bound?)

* **Accumulator state** per step: $(y_t, c^{(t)}_z, \text{step})$
* **ρ derivation** binds to (all *public*):

  * `step`: monotone step counter
  * `step_X`: step’s public input (if any)
  * `y_t`: compact accumulator vector
  * `c_z_digest`: Poseidon2 digest of commitment coordinates
* The **verifier recomputes ρ** from these public values; the EV CCS takes ρ as **public input**.

---

## Augmented CCS (per step)

When we do compress a step (e.g., in streaming mode) or a batch (at checkpoint/final), the **augmented CCS** includes:

1. **Step CCS** (your computation)
2. **EV (public‑ρ)** enforcing $y_{t+1} = y_t + \rho_t y^{\text{step}}_t$ with $ρ_t$ recomputed by the verifier
3. **Ajtai** gadgets (when enabled):

   * Opening constraints for selected commitment coordinates
   * Linear‑combination constraints for commitment evolution: $c_{t+1} = c_t + \rho_t c_{\text{step}}$

All sub‑components share the **same ρ** (public input), ensuring consistency.

---

## Verifier Responsibilities

* **EveryStep**: verify each step’s lean proof. Recompute ρ from transcript data and check the public‑ρ EV.
* **Checkpoint(N)**: verify one proof per chunk; each proof attests N accumulated steps.
* **FinalOnly**: verify one proof attesting the entire chain.

In all cases, the verifier also checks the **context digest** binding to $(\text{CCS}, x)$ to prevent replay across circuits or public inputs.

---

## Security Invariants

* **Fiat–Shamir soundness** for per‑step folding: ρ is **publicly recomputable** from an off‑circuit Poseidon2 transcript bound to step number, $X_t$, $y_t$, and $c^{(t)}_z$.
* **No “folding with itself”**: $y^{\text{step}}_t$ is extracted from **real step outputs**, not placeholders.
* **Context binding**: final lean proof includes a **Poseidon2 context digest** of $(\text{CCS}, x)$, checked before Spartan verification.
* **Transparent PCS**: FRI‑derived PCS underlies the Spartan/SuperSpartan layer.
* **Post‑quantum assumptions**: lattice commitments and hash‑based transcripts.

---

## Memory Handling (Optional)

* **Shout** (read‑only lookup) and **Twist** (read/write memory) can be attached to Stage 1/2 to enforce memory semantics in CCS reduction and aggregation. They are orthogonal to the EV and emission policy.

---

## Practical Notes

* **Performance**: the major cost driver is **SNARK compression** (Spartan2). By using **Checkpoint(N)** or **FinalOnly**, you slash amortized cost per step by ≈N×.
* **Determinism vs. entropy**: debug builds may use fixed RNG seeds for reproducibility; release builds should use CSPRNGs for PP setup.
* **VK handling**: proofs are **lean** (no 51MB embedded VK). Verification uses a VK digest registry and checks digest equality.

---

## Glossary

* **CCS** — Customizable Constraint System
* **MCS** — Matrix Commitment Scheme (Ajtai)
* **ME claim** — Multilinear Evaluation claim
* **EV** — Embedded Verifier for IVC (public‑ρ mode)
* **PCS** — Polynomial Commitment Scheme (FRI‑derived)
* **Checkpoint(N)** — Emit one SNARK proof after N steps

---

This updated architecture captures the **public‑ρ IVC** design and the **on‑demand emission** model: run the **cheap recursive loop** per step, and **compress** only when you need a verifiable artifact (every step, per checkpoint, or once at the end).
