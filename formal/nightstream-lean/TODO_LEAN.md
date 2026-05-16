# TODO_LEAN

Lean-first roadmap for moving the CHIP-8 path in `neo-fold-prototype` from the
current compatibility frontend to the intended release architecture.

This file is about **boundary-changing** work only. Pure Rust performance work
under an unchanged proof/checker boundary does not belong here.

## Ground Rules

- Production verification stays Rust-only.
  See [docs/assurance-strategy.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/docs/assurance-strategy.md).
- Lean is for theorem ownership, checker ownership, and release gating.
- The long-term compatibility target is near-`1:1` Rust↔Lean equality on
  protocol-binding computations, not just theorem-surface agreement.
- Do not formalize the current compatibility path as if it were the final
  architecture.
- The target is the bridge between:
  - SuperNeo main-lane folding, and
  - Twist/Shout-style obligation production and classification.
- Keep the existing simple CHIP-8 kernel proof as the exact/auditor boundary
  until the release-path bridge is proved and implemented.

## Why This File Exists

The repo already has a strong exact CHIP-8 kernel formalization under
[formal/nightstream-lean](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean),
including:

- exact opening boundary
- transcript schedule
- public-input binding
- Stage 1 / Stage 2 / Stage 3 evidence coverage
- exact authenticated trace closure
- exact kernel soundness and audit digest

That proves the current exact kernel contract.

It does **not** yet prove the final release architecture we actually want to
ship, where the verifier should consume a compressed/folded Rust proof rather
than replaying a compatibility export path.

## Current Repo Facts

These are the reasons a Lean-first release-architecture pass is still needed.

- The active CHIP-8 pipeline is explicitly a compatibility path in
  [crates/neo-fold-prototype/src/pipeline/mod.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/pipeline/mod.rs).
- The bridge still exports compatibility artifacts in
  [crates/neo-fold-prototype/src/bridge/mod.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/bridge/mod.rs),
  including `compatibility_path: true`.
- `BytecodeFetch` is still a transcript-binding facade, not a real Shout proof,
  in
  [crates/neo-fold-prototype/src/families/bytecode_fetch.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/bytecode_fetch.rs).
- `RegisterHistory` is still a transcript-binding facade, not a real Twist
  proof, in
  [crates/neo-fold-prototype/src/families/register_history.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/register_history.rs).
- `RamHistory` is still a transcript-binding facade, not a real Twist proof, in
  [crates/neo-fold-prototype/src/families/ram_history.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/ram_history.rs).
- `InstructionSemanticsLookup` exists in proof/planner enums but does not yet
  have a real proving module. See
  [crates/neo-fold-prototype/src/proof.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/proof.rs)
  and
  [crates/neo-fold-prototype/src/stages/planner.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/stages/planner.rs).
- `time_opening` currently owns manifest/reduction/unification summaries over
  `OpeningClaim`s, not the final compressed PCS opening backend, in
  [crates/neo-fold-prototype/src/time_opening.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/time_opening.rs).
- The Rust proof boundary still advertises future placeholders in
  [crates/neo-fold-prototype/src/proof.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/proof.rs).
- The Nightstream Lean stack already fixes the digest boundary and transcript
  schedule, but it does not yet own one fully executable Goldilocks /
  serialization / Poseidon2 parity lane for the exported protocol-binding hash
  values.

## Non-Goals

- Do not put Lean in the release verifier hot path.
- Do not try to formally mirror every temporary Rust file boundary.
- Do not prove the compatibility pipeline as the permanent architecture.
- Do not assume every projected family automatically becomes a SuperNeo main-lane
  CE claim.
- Do not replace the existing exact CHIP-8 kernel proof before the release-path
  replacement theorem is ready.

## Ordered Work

### 1. Freeze the target release theorem surface

Own this first in the Nightstream bridge package:

- [formal/nightstream-lean/specs/BridgeTypes.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/BridgeTypes.spec.md)
- [formal/nightstream-lean/specs/Projection.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/Projection.spec.md)
- [formal/nightstream-lean/specs/FoldAdmissibility.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/FoldAdmissibility.spec.md)
- [formal/nightstream-lean/specs/MainLaneBridge.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/MainLaneBridge.spec.md)
- [formal/nightstream-lean/specs/ShardComposition.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/ShardComposition.spec.md)

Need:

- one exact typed family-output surface for CHIP-8 extension families
- one exact classification rule:
  - becomes main-lane CE claim
  - stays sidecar checked object
  - stays final opening obligation
- one exact fold-admissibility rule
- one exact shard/session composition target

Why:

- This is the real architecture decision point.
- Without it, Rust will keep drifting by convenience.
- This is the theorem layer the repo README already says to start with.

Done when:

- every planned extension family has a typed emitted-obligation shape
- the bridge says exactly what may fold, what may not, and why

### 2. Formalize the readonly bytecode-fetch family as a real Shout-owned surface

Primary formal owners:

- new or expanded CHIP-8 spec/module surface for bytecode fetch under
  [formal/nightstream-lean/specs/chip8/](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/)
- the corresponding Lean interface/implementation under
  [formal/nightstream-lean/Nightstream/Chip8/Stage1/](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/Nightstream/Chip8/Stage1/)

Need:

- theorem-facing Shout instantiation for ROM fetch correctness
- exact emitted obligation shape
- exact authenticated table source
- exact relation to the public program image

Why:

- the Rust owner is still a transcript binding
- fetch correctness should be proved as a real lookup-family fact, not replayed

Blocks:

- replacing
  [bytecode_fetch.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/bytecode_fetch.rs)
  with a real proof family

### 3. Formalize instruction-semantics lookup as a real family

Primary formal owners:

- new CHIP-8 Stage-1/lookup spec and Lean modules

Need:

- one exact theorem-facing surface for instruction-semantics lookup
- exact emitted obligations per supported CHIP-8 instruction family
- exact table/evaluator ownership and authenticated source

Why:

- this family is planned in Rust but not implemented yet
- instruction semantics should live in lookup families where justified instead
  of bloating the main lane

Blocks:

- adding the missing Rust proving module for `InstructionSemanticsLookup`

### 4. Formalize the register-history family as a real Twist-owned surface

Primary formal owners:

- [Chip8TwistConcreteBinding.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8TwistConcreteBinding.spec.md)
- [Chip8TwistRoleSessions.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8TwistRoleSessions.spec.md)
- [Chip8TwistTraceRoleSessions.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8TwistTraceRoleSessions.spec.md)

Need:

- exact register-side emitted proof object for the release path
- exact session closure object and role partition
- exact relation to non-zero initial register state
- exact output classification into:
  - main-lane admissible claim
  - sidecar checked object
  - final opening obligation

Why:

- the current Rust path replays state and hashes it into the transcript
- release mode needs a real proof family, not replay equivalence

Blocks:

- replacing
  [register_history.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/register_history.rs)

### 5. Formalize the RAM-history family as a real Twist-owned surface

Primary formal owners:

- same Stage-2 / trace owners as register history, plus the RAM timeline owners

Need:

- exact RAM-side emitted proof object
- exact relation to non-zero initial RAM state
- exact handling of burst memory behavior
- exact classification of outputs for downstream folding/opening

Why:

- same reason as register history
- RAM is the highest-risk place to accidentally keep a replay-based verifier

Blocks:

- replacing
  [ram_history.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/families/ram_history.rs)

### 6. Prove the family-to-main-lane bridge

Primary formal owners:

- [formal/nightstream-lean/specs/MainLaneBridge.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/MainLaneBridge.spec.md)
- [formal/nightstream-lean/specs/FoldAdmissibility.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/FoldAdmissibility.spec.md)

Need:

- exact projection theorem from family proof outputs to SuperNeo obligations
- exact proof that a projected family output is admissible for the main lane
- exact proof that non-admissible outputs stay outside the main lane

Why:

- this is the bridge the repo still lacks
- it is the difference between “family proof exists” and “main lane may safely
  fold it”

Blocks:

- landing the real staged bridge in Rust
- safely replacing the compatibility export path

### 7. Define the real staged bridge artifact

Primary formal owners:

- a new or expanded staged-bridge surface in Nightstream bridge specs
- corresponding CHIP-8-facing digest/checker surfaces if the public artifact changes

Need:

- one exact typed bridge output replacing `export_compat_steps`
- exact public view for release proofs
- exact statement of what the backend verifier is allowed to trust from the
  CHIP-8 frontend

Why:

- the current bridge owner still says `compatibility_path: true`
- release mode should not depend on the compatibility export shape

Blocks:

- replacing
  [crates/neo-fold-prototype/src/bridge/mod.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/bridge/mod.rs)
  and the compatibility parts of
  [crates/neo-fold-prototype/src/pipeline/mod.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/pipeline/mod.rs)

### 8. Define the final opening/compression boundary

Primary formal owners:

- [formal/nightstream-lean/specs/PCSOpeningSemantics.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/PCSOpeningSemantics.spec.md)
- CHIP-8 opening-boundary/checker specs if the packaged artifact changes

Need:

- one exact theorem-facing final opening object for the release path
- exact relation between claim-space summaries and real lower-layer opening proof
- exact optional/mandatory status of reduction summaries
- exact checker boundary for the compressed opening artifact

Why:

- current `time_opening` is still a manifest/reduction/unification summary layer
- a fast verifier needs the actual compressed opening backend boundary, not just
  a summary of claims

Blocks:

- upgrading
  [crates/neo-fold-prototype/src/time_opening.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/time_opening.rs)
  and possibly
  [crates/neo-fold-prototype/src/finalize.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/finalize.rs)

### 9. Define the final packaged proof and checker surface

Primary formal owners:

- [Chip8StagedExecutionDigest.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8StagedExecutionDigest.spec.md)
- [Chip8ArtifactAudit.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8ArtifactAudit.spec.md)
- any new release-path digest owner if the current digest is no longer enough

Need:

- one exact Lean-defined release proof digest
- one exact Lean checker input/output contract
- one exact relation between Rust proof packaging and the theorem-facing digest

Why:

- Rust should populate a Lean-defined release artifact, not invent one
- this is the contract for CI, release gating, and differential checks

Blocks:

- changes to
  [crates/neo-fold-prototype/src/proof.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/proof.rs)
  and
  [crates/neo-fold-prototype/src/finalize.rs](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-prototype/src/finalize.rs)

### 10. Define the executable protocol-parity lane

Primary formal owners:

- new or expanded Nightstream transcript / digest / artifact specs
- corresponding executable Lean owners under
  [formal/nightstream-lean/Nightstream/](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/Nightstream/)

Need:

- one executable Lean model of the protocol-binding Goldilocks arithmetic used
  by exported artifact fields
- one exact canonical serialization owner for transcript-bound public objects
- one exact labeled absorb / squeeze transcript owner
- one executable Poseidon2-over-Goldilocks owner for protocol-binding digest and
  challenge fields
- deterministic golden vectors for:
  - `meta_pub` serialization
  - `root0`
  - transcript-derived challenges
  - staged digest / bundle fields that cross the Rust↔Lean boundary

Why:

- maximal Rust↔Lean compatibility requires exact computational parity, not only
  theorem-level existential alignment
- release gating is much stronger if exported protocol-binding values can be
  checked `1:1` against Lean-owned computations
- golden vectors give a stable regression lane for protocol bytes, not just
  semantic shapes

Blocks:

- fixing the final release artifact shape first so the parity lane targets the
  right exported values

### 11. Update soundness accounting for the release path

Primary formal owners:

- [Chip8SoundnessAccounting.spec.md](/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8SoundnessAccounting.spec.md)

Need:

- one explicit composed soundness budget for:
  - SuperNeo main-lane reductions
  - Shout families
  - Twist families
  - final opening/compression
  - Fiat-Shamir composition

Why:

- a new release-path composition means a new proof-composition accounting story
- this must stay theorem-owned, not left to spreadsheet reasoning

## Rust Work That Does Not Need Lean First

These are safe to do in Rust alone as long as they do not change exported proof
or checker boundaries:

- optimize `Π_CCS -> Π_RLC -> Π_DEC` internals
- improve batching/caching/allocation behavior
- speed up trace building
- speed up transcript plumbing
- refactor code structure while preserving the exact public artifact

If a Rust change turns a placeholder, transcript binding, or compatibility object
into a real proof object, it belongs back in the Lean-first list above.

## Execution Order

Use this order. Do not start in the middle unless the boundary above it is
already frozen.

1. Freeze the target release theorem surface.
2. Formalize the four extension families.
3. Prove the family-to-main-lane bridge.
4. Define the real staged bridge artifact.
5. Define the final opening/compression boundary.
6. Define the final packaged proof/checker surface.
7. Define the executable protocol-parity lane.
8. Update soundness accounting.
9. Only then replace the Rust compatibility path.

## Exit Criteria

This roadmap is complete only when all of the following are true:

- the release-path proof artifact is Lean-defined
- Rust populates that artifact exactly
- the Lean checker accepts the Rust-produced artifact
- Lean owns executable protocol-binding computations for exported Goldilocks /
  Poseidon2-derived fields
- deterministic golden vectors cover those protocol-binding exported values
- Rust and Lean agree exactly on those exported values
- no release verifier path depends on compatibility export objects
- bytecode/register/RAM/instruction-semantics families are real proofs, not
  transcript-binding facades
- the family-to-main-lane bridge is theorem-owned
- the final verifier consumes the compressed release artifact instead of replaying
  the exact simple-kernel bridge row by row

Until then, the current exact CHIP-8 kernel proof remains the correct
auditor/debug boundary, and the compatibility Rust path remains an interim path
rather than the final release architecture.
