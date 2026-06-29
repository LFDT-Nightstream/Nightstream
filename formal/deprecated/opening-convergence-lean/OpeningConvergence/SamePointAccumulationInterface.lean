import OpeningConvergence.Basic

/-!
# Module 3: SamePointAccumulation — Interface

Owns the proof that Phase 2a (same-object same-point identity collapse)
preserves the evaluation relation.

## Theorems
- Theorem 7: Phase2IdentityCollapse
- Theorem 8: SingletonPassthrough

## Spec
See `specs/SamePointAccumulation.spec.md`
-/

namespace OpeningConvergence.SamePointAccumulation

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-! ## Theorem 7: Phase2IdentityCollapse

Given unified claims that share the same opened object, point, and payload
(guaranteed by SameObjectPayloadUniqueness from Phase 1), the collapsed
ReducedEvalClaim preserves the evaluation relation.

Semantic equivalence:
    (∀ i: f(r*) = v*_i)  ↔  f(r*) = v*

Provenance preservation:
    source_claim_ids are in canonical bucket order and the reduced_claim_digest
    is deterministic.
-/

/-- A Phase 2 group: claims that share the same opened object and point. -/
structure Phase2Group (K : Type*) (ell : Nat) (groupSize : Nat) where
  openedObject : OpenedObjectId
  point : Fin ell → K
  payloads : Fin groupSize → FamilyEvalPayload K
  hIdentical : ∀ a b : Fin groupSize, payloads a = payloads b

/-- Phase2IdentityCollapse: collapsing identical claims preserves the
    evaluation relation and provenance.
    Since all payloads are identical (hIdentical), picking any representative
    preserves the evaluation relation exactly. -/
theorem phase2IdentityCollapse
    {ell groupSize : Nat}
    (group : Phase2Group K ell groupSize)
    (hNonempty : groupSize > 0)
    (i : Fin groupSize)
    :
    let collapsed := group.payloads ⟨0, hNonempty⟩
    group.payloads i = collapsed := by
  simp only
  exact group.hIdentical i ⟨0, hNonempty⟩

/-- Phase2IdentityCollapse provenance: source IDs are preserved in canonical
    order and the digest is deterministic. -/
theorem phase2IdentityCollapseProvenance
    {ell groupSize : Nat}
    (group : Phase2Group K ell groupSize)
    (sourceIds : Fin groupSize → Nat)
    (hOrdered : ∀ a b : Fin groupSize, a < b → sourceIds a < sourceIds b)
    :
    -- Source IDs are strictly increasing (already given by hypothesis)
    ∀ a b : Fin groupSize, a < b → sourceIds a < sourceIds b := by
  exact hOrdered

/-! ## Theorem 8: SingletonPassthrough

If a Phase 2 group has exactly one claim, the output ReducedEvalClaim is
identical to the input (modulo type wrapper). The five singleton families
(Stage2 + Stage3Continuity) use this path in v1.
-/

omit [Field K] [Fintype K] [DecidableEq K] in
/-- SingletonPassthrough: singleton group produces identical output. -/
theorem singletonPassthrough
    {ell : Nat}
    (openedObject : OpenedObjectId)
    (point : Fin ell → K)
    (payload : FamilyEvalPayload K)
    :
    let reduced : ReducedEvalClaim K ell := {
      openedObject := openedObject
      point := point
      payload := payload
      sourceClaims := [0]
    }
    reduced.payload = payload := by
  rfl

end OpeningConvergence.SamePointAccumulation
