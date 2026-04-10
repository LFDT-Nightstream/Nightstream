# RV64IM Witness-Backed Side Bridge Specification

STATE: DESIGN TARGET

## Scope

This document specifies the end-state theorem-facing side-bridge relation for
RV64IM in `neo-fold-next`.

It fixes the ownership boundary between:

1. the current bridge theorem owned by `riscv-kernel.md`,
2. the future witness-backed side relation that internalizes that theorem as a
   private witness relation,
3. and the later recursive or succinct compiler layer that proves the verifier
   of that fixed relation.

This document owns:

- the exact theorem that the witness-backed side bridge must preserve,
- the exact public/private split for that relation,
- the stable authoritative projection shared by later compiler layers,
- and the negative rules forbidding digest-consistency replacements.

This document does **not** own:

- the inner RV64IM execution semantics,
- the accepted-artifact replay or audit path,
- the concrete recursive backend choice,
- the concrete compression backend choice,
- or the final exported compressed proof format.

Those remain owned by:

- `riscv-kernel.md`
- `riscv-recursive-proof.md`
- `riscv-recursive-instantiation.md`

## Normative References

This document is constrained by the following local references:

- `./riscv-kernel.md`
  - the current concrete RV64IM bridge theorem, opening theorem, and
    row-local/root-local ownership boundaries
- `./riscv-recursive-proof.md`
  - the normalized theorem-facing exported proof boundary
- `./riscv-recursive-instantiation.md`
  - the concrete backend and migration contract
- `../../docs/superneo-paper/01_1_Introduction.md`
  - standard compilers apply after the folding relation is fixed
  - efficient proof compression using Spartan is a later compiler layer
- `../../docs/superneo-paper/06_6_Strong_and_weak_interactive_reductions.md`
  - strong/weak composition requires a stable shared projection `phi`
- `../../docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
  - the paper-owned relation is the CE/CCS relation plus
    `Π_CCS -> Π_RLC -> Π_DEC`, not a verifier for a variable-length proof blob

This document is normative only where its requirements are implied by, or are a
conservative specialization of, those references plus the repo-owned RV64IM
bridge theorem.

## 1. Design Goal

The exported RV64IM Nightstream verifier shall eventually validate a succinct
proof of a **fixed witness-backed side relation**.

That fixed relation shall own exactly the same semantic bridge theorem
currently enforced by the native RV64IM side-claim, side-opening,
opening-artifact, and side-terminal seams, but it shall own those checks as
**private witness constraints**, not as a public verifier replay over shipped
native witness artifacts.

The final external verifier shall not depend on:

- the native side-terminal witness artifact,
- raw side-claim packaged witnesses,
- raw side-opening packaged witnesses,
- raw kernel opening witness packages,
- or any bridge-local digest bundle exported only because the current hybrid
  path happens to carry it.

## 2. Theorem Target

The witness-backed side bridge owns one exact theorem:

> the carried RV64IM side bridge is valid if and only if there exists a private
> witness that realizes the same accepted-opening identity, row-local bridge
> binding, stage linkage, opening provenance, root-execution binding, and
> handoff equality checks that the current native RV64IM bridge theorem
> requires.

Any conforming witness-backed bridge relation, recursive wrapper, or succinct
outer proof must preserve:

- the same exact accepted opening identity reused across row projection,
  opening provenance, and bridge binding,
- the same canonical authenticated row-to-chunk routing induced by the carried
  fold schedule,
- the same exact `RootEncode(z_j)` image and row-local acceptance binding for
  each theorem-bearing selected row,
- the same exact prepared-step / exported-row equality checks currently
  required by the bridge theorem,
- the same exact Ajtai opening / commitment consistency checks,
- and the same exact handoff equality to the main residual proof boundary.

A later owner may replace native replay with a private-witness relation or a
succinct compiler, but it may not weaken the theorem to digest consistency or
to "self-consistent carried summaries".

## 3. Public Instance

The public instance of the witness-backed side bridge shall contain only the
minimal authoritative bindings required by the theorem and by external verifier
composition.

At minimum, the public instance shall bind:

- the canonical public proof statement or its canonical digest,
- the canonical side-bundle binding required to identify the accepted side
  surfaces consumed by the bridge theorem,
- the canonical opening-artifact binding required by the accepted opening
  theorem boundary,
- the canonical bridge handoff bindings required to connect the side bridge to
  the main residual proof,
- and the canonical verifier-context / root-parameter binding required by the
  published Nightstream verifier statement.

No bridge-local digest bundle is theorem-facing by default.

No field may remain public solely because the current hybrid bridge exports it.

### 3A. Current Side-Bridge Field Classification

The following inventory is exhaustive for the currently carried RV64IM
side-bridge surfaces.

The four semantic classes are:

- `theorem-facing public`
- `recursive-internal public`
- `private witness`
- `audit-only`

Cryptographic proof payload bytes are not assigned one of those four semantic
roles. They are proof objects, not statement/witness/binding fields. This
inventory therefore classifies the semantic fields currently carried alongside
those proof objects.

| Current owner | Current field(s) | End-state class | Required disposition |
| --- | --- | --- | --- |
| `NightstreamStatement` | `public_io_digest`, `verifier_context_digest`, `fold_schedule`, `semantic_step_count`, `chunk_summaries`, `linkage_root`, `proof_binding_root` | theorem-facing public | Remain the canonical published statement boundary. |
| `NightstreamProofBindingInputs` | `main_decider_proof_digest`, `main_residual_proof_digest`, `side_bridge_artifact_digest`, `linkage_artifact_digest` | recursive-internal public | May remain only as internal proof-binding inputs beneath the published statement root. The normalized public statement exposes only `proof_binding_root`. |
| `Rv64imNightstreamProof` | `hybrid_side_bridge_artifact.digest`, `linkage_artifact.digest` | recursive-internal public | These are canonical binding handles only. They shall not force the full native artifacts to remain theorem-facing. |
| `Rv64imNightstreamProof` | `hybrid_side_bridge_artifact.bridge_artifact`, `hybrid_side_bridge_artifact.backend_proof` | audit-only | Transitional helper payloads only. They remain on the current hybrid path until the backend proof owns the full witness-backed bridge theorem. |
| `Rv64imLinkageClaims` | `public_chunk_digests` | theorem-facing public | Already reflected by `chunk_summaries`; any retained copy must remain definitionally tied to the published chunk summaries. |
| `Rv64imLinkageClaims` | `bridge_handoff_digests`, `digest` | recursive-internal public | Remain public only to the recursive/compiled bridge verifier that links side and main residual proofs. |
| `Rv64imSideProofBundle` | `statement_core_digest`, `transcript`, `stage1`, `stage2`, `stage3`, `stage_claim_proof_bridge`, `kernel_opening_bridge`, `kernel_claim_bridge`, `kernel_claim_proof_bridge`, `kernel_export_bridge`, `semantic_rows_digest`, `row_local_ccs_acceptance_digest`, `execution_semantics_refinement_digest`, `family_digest`, `root_execution_digest`, `digest` | private witness | Internalize into the witness-backed side relation. Redundant sub-digests must be recomputed or deleted, not carried as a second authority. |
| `Rv64imKernelOpeningBridge` and nested bridge summaries | all non-proof semantic fields | private witness | Keep only as witness material needed to prove the bridge theorem. Convenience digest mirrors are not theorem-facing. |
| `Rv64imStageClaimProofBridge`, `Rv64imKernelClaimBridge`, `Rv64imKernelClaimProofBridge`, `Rv64imKernelExportSourceBridge` | all fields | private witness | Internalize and progressively delete redundant carried digest mirrors where the fixed relation can recompute them. |
| `Rv64imSideEvalClaimRelationStatement` | `public_statement` | theorem-facing public | This is the same canonical public proof statement already bound by the published verifier. |
| `Rv64imSideEvalClaimRelationStatement` | `side_bundle`, `phase0_opened_objects`, `eval_claim_bundle` | private witness | These Phase 0 theorem surfaces move under the fixed witness-backed side relation. |
| `Rv64imSideEvalClaimRelationWitness` | `claim_witnesses` | private witness | Remains witness-only. |
| `Rv64imSideEvalClaimArtifact` | `statement_digest`, `phase0_opening_targets`, `eval_claim_bundle`, `digest` | audit-only | Transitional compact replay surface only. Remove from the final theorem-facing proof path once the fixed relation owns Phase 0 privately. |
| `Rv64imOpeningArtifact` | `digest` | recursive-internal public | Canonical opening binding handle only. |
| `Rv64imOpeningArtifact` | `phase0_artifact`, `convergence_artifact` | private witness | Internalize beneath the fixed witness-backed side relation. |
| `Rv64imWitnessBackedSideBridgeStatement` | `nightstream_statement`, `public_statement`, `side_bundle_digest`, `opening_artifact_digest`, `bridge_handoff_digests` | theorem-facing public | This is the owned fixed relation statement for the side bridge. Any later compiler layer must target this public boundary or a conservative projection of it. |
| `Rv64imWitnessBackedSideBridgeWitness` | `side_bundle`, `opening_artifact`, `claim_witness`, `opening_witness` | private witness | This is the owned private witness for the fixed side relation. Later compiler layers may encode it differently, but they may not weaken its theorem. |
| `Rv64imWitnessBackedSideBridgeArtifact` | `witness`, `digest` | audit-only | Transitional native artifact for the new owned relation. Its digest is bound to the external witness-backed side-bridge statement, but that statement digest is not carried redundantly inside the artifact. |
| `Rv64imHybridSideBridgeArtifact` | `bridge_artifact`, `backend_proof`, `digest` | audit-only | Transitional hybrid helper only. The backend shell still targets the current digest/handoff adapter, but the native theorem replay now flows through the owned witness-backed side relation. |

No row above may be reclassified implicitly by implementation convenience.
Any deviation from this table must update this document and
`riscv-recursive-instantiation.md` together.

## 4. Private Witness

The private witness shall contain the concrete theorem-bearing objects needed to
realize the bridge theorem.

This includes the objects currently realized through native replay, such as:

- the side-claim witness objects,
- the side-opening witness objects,
- the opening-artifact witness/provenance objects,
- the selected-row authenticated payloads and row-local execution/opening
  objects,
- the prepared-step linkage objects,
- and any lower-level packages needed to prove the owned theorem above from the
  public instance.

If a value is purely derived from other witness or public data, it shall be
recomputed inside the relation rather than carried as an independent authority.

## 5. Stable Projection Requirement

The witness-backed side bridge shall expose one stable authoritative projection
`phi_side` for the purposes of strong/weak composition in the sense of
SuperNeo's interactive-reduction framework.

`phi_side` shall project only the authoritative commitment / public-binding data
that identifies the carried bridge instance. It shall not project convenience
digests that can be recomputed from those authoritative values.

Any later reduction or compiler layer that composes above this bridge must bind
the same `phi_side`.

## 6. Fixed-Relation Requirement

The witness-backed side bridge shall be a verifier for a **fixed relation**,
not a verifier for an arbitrary-length proof object.

Therefore:

- the side relation shall be specified first as a public/private theorem,
- only after that may a recursive or succinct backend compile that relation,
- and the backend shall prove the verifier of that fixed relation rather than
  directly verifying a variable-length Nightstream proof artifact.

This follows the SuperNeo compiler direction:

- first define the relation and reduction structure,
- then apply standard compilers to IVC/PCD,
- then apply proof compression.

## 7. Compiler Requirements

Any later recursive wrapper or succinct backend compiler over this bridge must
satisfy all of the following:

- it compiles a fixed relation with a stable public/private interface,
- it preserves the exact theorem target above,
- it does not expose the private witness as theorem-facing output,
- it does not introduce additional theorem-facing digest inputs merely for
  convenience,
- and it keeps all protocol-binding transcript inputs canonical and derived
  from authoritative relation data.

If the concrete backend cannot support an unbounded number of bridge components
with one fixed shape, then a conforming implementation must either:

- use a fixed maximum shape with canonical padding and selectors, or
- introduce a recursive wrapper above per-shape compiled relations.

Direct verification of an arbitrary-length side proof by one fixed compiled
circuit is not a required property of this owner.

## 8. Negative Rules

The following are forbidden at this theorem boundary:

- treating digests as authority rather than as compression of authoritative
  objects,
- carrying duplicate authorities for the same fact across the public boundary,
- mixing attacker-chosen helper digests into Fiat-Shamir transcript inputs
  without first recomputing or canonically binding them,
- exposing native witness artifacts as part of the final theorem-facing
  verifier API,
- and using accepted-artifact replay or other prover-side oracle objects as the
  canonical public verifier input.

## 9. Concrete Implementation Obligations

The migration is complete only when all of the following are true:

1. the exact RV64IM side-bridge theorem is written as one owned
   public/private relation above the current bridge seams;
2. every currently carried bridge field is classified into exactly one of:
   theorem-facing public, recursive-internal public, private witness, or
   audit-only;
3. all current native side-terminal witness-artifact checks are internalized as
   private witness constraints of that relation;
4. the public verifier no longer depends on shipped native side/opening witness
   artifacts;
5. all remaining free digest/helper fields at that boundary are removed,
   internalized, or recomputed canonically from authoritative inputs;
6. the fixed-shape strategy for the compiler layer is frozen;
7. the concrete recursive/compression backend only compiles the fixed relation,
   not a variable-length proof object;
8. and the accepted-artifact path remains audit/prover-side only, not the
   theorem-facing verifier boundary.

## 10. Consequence For The Current RV64IM Path

The current RV64IM Nightstream verifier path is conforming only as a
transitional hybrid bridge:

- it still ships compact native theorem witness artifacts,
- it still performs native verifier checks beneath the current Spartan shell,
- and it therefore does not yet realize the end-state witness-backed side
  bridge specified here.

Those current hybrid objects should be understood as migration aids and
proof-complete audit surfaces, not as the final theorem-facing design target.
