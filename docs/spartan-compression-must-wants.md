# Spartan Compression of Neo + Twist/Shout: MUSTs and WANTS

This document captures the **hard requirements (MUST)** and **nice-to-haves (WANTS)** for compressing a Neo shard proof (folding + Twist/Shout) into a **small proof artifact** **without losing** the security/correctness guarantees of the native verifier.

The intended end-state is:

> Run Neo folding with Twist/Shout sidecars to produce a `ShardProof`, then produce **one compact, shareable proof artifact** that attests:
> 1) the **native shard verifier** would accept that run (including Twist/Shout + output binding + step linking), and
> 2) any deferred **obligations are closed** (i.e., `verify_and_finalize` semantics),
> under a clearly defined public statement.

In practice, this can be either:
- a single Spartan2 proof that *also* verifies a succinct closure verifier in-circuit, or
- one “blob” containing a Spartan2 proof plus a separate succinct **closure proof** (still small; target: low 100s KB total).

For a detailed, “junior-dev executable” implementation path, see `docs/spartan-compression-two-phase-plan.md`.

---

## Terminology (code-aligned)

- **ShardProof / FoldRun**: `neo_fold::shard_proof_types::ShardProof` (also called a “FoldRun” in some contexts).
- **StepProof**: one folding chunk’s artifacts: CCS fold, Route‑A memory sidecar, batched time proof, and optional val-lane folding.
- **Route‑A**: shared-challenge time-domain batching across CCS + memory oracles.
- **Twist/Shout**: memory/checker sidecars whose proofs live in `StepProof.mem`.
- **Val lane**: the `r_val` folding lane used by Twist val-eval checks (must not be dropped).
- **Output binding**: optional proof that final memory matches claimed outputs (often a RAM address/value claim).
- **Obligations (finalization)**: the final `ShardObligations { main, val }` returned by shard verification that must be “closed” to match native `verify_and_finalize` semantics.

---

## MUST (requirements)

### 1) Clear public statement (what the Spartan proof means)

The Spartan proof must verify a *well-defined* statement. At minimum, that statement needs to bind to:

- The **circuit/system context** (Neo parameters and the CCS instance being executed), and
- The **VM/program context** (e.g., ROM/program bytes digest, any fixed table specs), and
- The **claimed outputs** (e.g., final boundary state and/or output binding claim).

If the statement does not include these bindings, the proof is vulnerable to replay across circuits/programs or to “proving the wrong thing”.

### 2) Fiat–Shamir soundness (no prover-chosen challenges)

All protocol challenges that the native verifier treats as transcript-derived must be enforced as such for the Spartan-compressed proof.

This includes (non-exhaustive):

- Π‑CCS challenges `(α, β_a, β_r, γ)`
- Route‑A time-cycle point(s) (e.g., `r_cycle`)
- Route‑A time point `r_time`
- Π‑RLC mixing matrices `ρ`
- Any output-binding challenges (`r_prime` in output binding, etc.)

**Acceptable ways to satisfy this MUST:**

- **In-circuit transcript:** implement the Neo transcript in R1CS and recompute challenges inside the Spartan circuit.
- **Digest + external recomputation:** make the required transcript state/digests public IO, prove inside Spartan that these digests correspond to the exact objects used, and derive challenges outside Spartan from those digests.

**Not acceptable:** allocating challenges as unconstrained witness values.

### 3) Spartan circuit must cover the full native verification logic (not a subset)

To claim “Spartan proves correct folding + Twist/Shout”, the Spartan circuit must imply acceptance by the native verifier for the same inputs. Concretely, it must encode the checks performed by the native verification path used for zkVM-style runs:

- Step linking (chunk chaining), if applicable.
- Route‑A CCS verification:
  - time/row rounds consistency with the batched proof,
  - Ajtai rounds verification,
  - CCS terminal identity check.
- Route‑A memory checks:
  - Shout address pre-time proofs and any skipping logic (`active_mask`),
  - Twist address pre-time proofs,
  - Twist val-eval (including rollover when linking chunks),
  - correct handling of the `r_val` lane and its folded obligations.
- Π‑RLC verification:
  - recompute `ρ` and enforce that RLC outputs match transcript-derived `ρ`.
  - enforce commitment consistency where the native verifier enforces it.
- Π‑DEC verification:
  - enforce parent/children relations (including commitments if they are part of the native verification contract).
- Output binding (if enabled):
  - the sumcheck rounds,
  - the final value checks tying the claim back to the last step’s memory openings.

If any of the above is omitted, the Spartan proof is only a **partial attestation** and cannot be treated as a full compression of the zkVM verification.

### 4) Commitment binding and “what is proven about commitments”

The compressed proof must be explicit about **commitment correctness and obligation finalization**.

To claim “no loss of guarantees”, the verifier must learn more than “linear relations among commitments”. It must learn that any deferred obligations are **closed**:

- **Closure semantics:** the compression layer MUST correspond to `fold_shard_verify_and_finalize` (not just `fold_shard_verify`), i.e., for every final `MeInstance` in `obligations.main` and `obligations.val` there exists a bounded witness `Z` consistent with that instance.
- **What “closed” means (per obligation):** Phase 2 MUST define a single predicate that includes at least:
  - Ajtai commitment correctness/opening: `c = Commit(pp, Z)`,
  - the bound/range constraints required by the binding argument (e.g., digit bounds / ℓ∞ bounds),
  - ME consistency between that same `Z` and the carried instance fields (at minimum `X`, and any `y` / `y_scalars` semantics used by verification).
- **Canonical obligation list:** Phase 2 MUST define the canonical order of obligations (main + val lanes) and bind to it (no reordering/omission).
- **PP identity binding:** the public statement MUST bind the Ajtai PP identity used for openings (prefer a seeded PP identity: `(pp_seed, d, m, κ)`), otherwise “opening correctness” is meaningless.
- **Succinct closure proof (recommended):** because Ajtai `verify_open` is “recompute commit and compare” at large `m`, Phase 2 SHOULD use a transparent, PQ-friendly succinct proof for closure (e.g., sumcheck+FRI/STARK-style), and the compression artifact MUST verify it either:
  - directly (two-proof blob), or
  - in-circuit (single Spartan proof that verifies the closure proof verifier).

### 5) Canonical encoding for transcript absorption / digests

Any object that is hashed/absorbed into the transcript (or digested for context binding) MUST have:

- a unique, canonical serialization,
- unambiguous domain separation,
- fixed endianness and limb/field encoding rules.

Without this, it’s easy to accidentally create multiple encodings that hash to different challenges across implementations.

### 6) Field and extension-field correctness

The Spartan circuit must faithfully model:

- Neo’s base field arithmetic,
- the K-extension arithmetic used in Π‑CCS/Twist/Shout (and the chosen representation as R1CS variables),
- any range/base‑b constraints used by the protocols (e.g., Ajtai digit recompositions).

### 7) “Compression” must actually reduce verifier work

The verification experience after compression must not require the verifier to re-run (or even fully parse) the original large `ShardProof`.

At minimum, the verifier should be able to verify using:

- the compressed proof artifact (Spartan proof, plus any required closure proof),
- the public statement inputs,
- small auxiliary digests/metadata.

If verifying still requires streaming the entire original proof (or the full witness `Z`) to recompute challenges or validate openings, then it is not a meaningful compression layer.

---

## Recommended Path (Phase 1 → Phase 2)

This section is the concrete “what do we build next?” path implied by the MUSTs, optimized for:
- a **small shareable artifact** (target: low 100s KB), and
- **no loss of guarantees** vs native verification (`verify_and_finalize` semantics).

### Phase 1 (already shipping): verifier-equivalent compression up to obligations

Deliverable: a pinned-VK Spartan2 proof that implies the native shard verifier would accept a run, **up to** producing final `ShardObligations { main, val }`, under a replay-resistant statement (params/CCS/VM/output binding/step linking).

This phase is about “verifier equivalence” (transcript, Route‑A, Twist/Shout, output binding), not about proving Ajtai openings.

### Phase 2 (the missing piece): close obligations with a succinct closure proof

Core deliverable: upgrade the shareable artifact to imply **`verify_and_finalize` semantics**:
1) shard verifier acceptance (what Phase 1 proves), and
2) obligation closure: every final obligation ME instance is satisfiable/valid end-to-end.

Recommended architecture for the size goal:
- Ship **one blob containing two proofs**:
  - `SpartanProof`: proves native shard verification acceptance and binds to the statement.
  - `ClosureProof`: a transparent (PQ-friendly) succinct proof that obligations are closed.

This keeps the artifact small without requiring Spartan to verify an entire FRI/STARK verifier inside R1CS (which is usually expensive).

**Important sizing note:** avoid putting the explicit `ShardObligations` list (or per-obligation `MeInstance` objects) in the blob unless you have measured it stays under budget. Instead, treat obligations as *private witness* to the closure proof and bind them via a small `obligations_digest` that the Phase 1 Spartan circuit computes from the same obligations it derives during verification.

### Phase 2 milestones (in order)

1) **Write the closure contract (consensus-critical)**
   - Define the canonical obligations list and ordering (main + val lanes).
   - Define the per-obligation closure predicate:
     - `c = Commit(pp, Z)` (Ajtai opening/correctness),
     - bounds required by binding (digit bounds / ℓ∞ bounds),
     - ME consistency between the same `Z` and the carried instance fields (`X`, and any `y` / `y_scalars` semantics relied on by verification).
   - Define a canonical serialization and `obligations_digest = H(obligations_list)` (the obligations list itself need not be public if it’s committed-to by digest).

2) **Bind context + PP identity**
   - Add a stable `context_digest = H(statement fields that define the run context)`.
   - Bind Ajtai PP identity as a cheap-to-verify public input (prefer seeded PP identity):
     - `(pp_seed, d, m, κ)` (and any version tags), rather than hashing a materialized PP.

3) **Build a native reference finalizer for small tests**
   - Implement a reference “closure checker” (not for production `m=2^24`) that:
     - recomputes `Commit(pp, Z)` from explicit `Z`,
     - recomputes the ME consistency checks from the same `Z`,
     - checks bounds.
   - Use it as the oracle for Phase 2 correctness tests and for adversarial “gap” tests.

4) **Implement `ClosureProofV1` (transparent, PQ-friendly, size-tuned)**
   - Prove the closure predicate for the canonical obligations list, bound to `(context_digest, pp_id, obligations_digest)`.
   - **Batch aggressively** to hit the “low 100s KB” target (e.g., use transcript-derived random linear combinations across obligations and exploit the linearity of Ajtai commitments and many ME constraints).
   - Keep verification cheap and independent of the full `ShardProof`/`Z`.

5) **Ship `BridgeProofV2` (one blob)**
   - Define a single serialized artifact containing:
     - the Phase 1 `SpartanProof`,
     - the Phase 2 `ClosureProofV1`,
     - and (optionally) explicit digests (`context_digest`, `obligations_digest`, `pp_id`) to make verification plumbing simple.
   - Verification procedure:
     1) verify `SpartanProof` against pinned VK + statement,
     2) recompute/validate `context_digest` + `pp_id`,
     3) check that the closure proof is bound to the same `obligations_digest` that Spartan produced,
     4) verify `ClosureProofV1`.

6) **Optional (later): one proof system**
   - If you ultimately want “one proof inside one proof system”, make Spartan verify the `ClosureProofV1` verifier in-circuit.
   - Do this only after you have real size/time numbers for `ClosureProofV1`.

## WANTS (nice-to-haves)

### A) Statement flexibility (deployment knobs)

Support a few statement profiles without changing the core proving system, e.g.:

- **Minimal:** program digest + output claim + final boundary state
- **Debuggable:** include per-step digests / step count / selected intermediate commitments
- **Auditable:** include digests for params/CCS/MCS layouts and memory-table specs

### B) Modular circuit architecture

Separate gadgets/circuit modules roughly by verifier responsibility:

- Transcript/digest layer
- Π‑CCS verifier gadgets
- Route‑A batched-time verifier gadgets
- Shout verifier gadgets
- Twist verifier gadgets (addr + val-eval + rollover)
- Π‑RLC/Π‑DEC verifier gadgets
- Output-binding verifier gadgets

This keeps the “SNARK-of-verifier” approach maintainable and allows incremental rollout.

### C) Efficient handling of large vectors

Avoid materializing large vectors in-circuit when possible:

- hash/digest commitments to large tables and only open what is needed,
- use streaming or incremental hashing patterns,
- use sparse representations where the native protocol already exploits sparsity.

### D) Debug/validation tooling

- Ability to cross-check the Spartan circuit’s computed transcript challenges against the native verifier for the same `ShardProof`.
- “First failing constraint” ergonomics and structured logs for mismatches.

### E) Checkpointing / emission policy compatibility

Support emitting a Spartan proof:

- every step,
- at checkpoints (every N chunks),
- once at the end,

while keeping the public statement stable (or intentionally versioned).

### F) Clean integration with existing finalization/obligation concepts

If Neo retains an obligation-finalization phase (main + val lanes), prefer that the Spartan compression layer cleanly consumes those obligations rather than duplicating logic in multiple places.

### G) Proof-size budget

Target a “low 100s KB” shareable artifact for the fully closed proof (Phase 2). Prefer batching/aggregation techniques inside the closure proof to avoid per-obligation proof-size blowups.

---

## Remaining open questions (Phase 2)

- **Closure contract details:** exact per-obligation predicate (bounds + Ajtai opening + ME-consistency), and a single canonical CCS “ME-consistency” function to avoid drift.
- **Obligations binding format:** whether `obligations_digest` is computed from a canonical obligations list, or derived from existing Phase 1 endpoints (e.g., `acc_final_main_digest`/`acc_final_val_digest`) for a leaner interface.
- **Closure proof choice + params:** pick the concrete transparent proof stack (STARK vs sumcheck+FRI), plus concrete security parameters that hit the “low 100s KB” budget with acceptable prover time.
- **Optional later:** whether to SNARK-wrap the closure verifier in-circuit (single proof system) after Phase 2 size/time benchmarks.
