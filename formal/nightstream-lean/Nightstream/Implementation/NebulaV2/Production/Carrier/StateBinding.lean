import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsPublicTranscript

/-!
Contract: complete Poseidon2 state binding for one field-native fresh claim.

The field-native claim has no profile or application-statement sidecar. Its
complete authority is the exact paper-NIFS public frame: running state,
mandatory four-component bundle, CCS public input, and the memory batch bound
inside that CCS input. Candidate identity and relation identity occur in the
verifier-owned transcript prefix.

This module is a semantic transcript theorem. Generated sponge rows are a
separate implementation obligation.

Assurance tier: model-level transcript binding.

Does not own generated rows, Poseidon2 security, external statement parsing,
terminal verification, Rust refinement, candidate selection, or a verifier
key.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionFullClaimStateBinding

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim
open Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId

noncomputable def authoritativeFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : List Nat :=
  ProductPoseidon2.statementIdentifierFields statementId ++
    ProductionProductNifsPublicTranscript.frame degreeBound value

theorem authoritativeFrame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (authoritativeFrame statementId degreeBound value).length =
      366 + (17 + ProductNifsCodec.runningFieldCountFor
        fullShape.rowVariables + 3888 + 540) := by
  rw [authoritativeFrame, List.length_append,
    ProductionProductNifsPublicTranscript.frame_lengthFor contract]
  simp [ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length]

/-- Compatibility value for the fixed-25 reference shape. Production
artifacts must use `authoritativeFrame_lengthFor` at their selected exponent. -/
theorem authoritativeFrame_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (authoritativeFrame statementId degreeBound value).length = 88021 := by
  rw [authoritativeFrame_lengthFor contract.toSelected statementId degreeBound
    value, contract.rowVariables]
  decide

noncomputable def bindingState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : State :=
  ProductionProductNifsPublicTranscript.publicState statementId degreeBound value

private theorem absorbList_append
    (left right : List Nat) (state : State) :
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (left ++ right) state =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants right
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, Poseidon2Duplex.absorbList]
      exact inductionHypothesis _

theorem bindingState_replays_authoritativeFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    bindingState statementId degreeBound value =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (authoritativeFrame statementId degreeBound value)
        ProductPoseidon2.initialState := by
  rw [bindingState, ProductionProductNifsPublicTranscript.publicState,
    ProductPoseidon2.initialStateForStatement, authoritativeFrame,
    absorbList_append]

abbrev CanonicalClaim (candidate : Id) (fullShape : Phi81Relation.Shape) :=
  ProductionProductNifsPublicTranscript.CanonicalValue candidate fullShape

abbrev FullClaimTranscriptCollision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) (degreeBound : Nat) :=
  ProductionProductNifsPublicTranscript.PublicTranscriptCollision candidate
    fullShape statementId degreeBound

/-- Equal compact states recover the exact typed claim or name the only two
semantic hash failures used by this layer. The desired equality is not a
premise. -/
theorem equal_bindingState_recovers_claim_or_named_failure
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {statementId : StatementId} {degreeBound : Nat}
    (left right : CanonicalClaim candidate fullShape)
    (equal : bindingState statementId degreeBound left.value =
      bindingState statementId degreeBound right.value) :
    left.value = right.value ∨
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate ∨
      FullClaimTranscriptCollision candidate fullShape statementId
        degreeBound := by
  rcases
      ProductionProductNifsPublicTranscript.equal_publicState_recovers_authority_or_named_failure
        contract left right equal with direct | failure
  · apply Or.inl
    apply Value.ext
    · exact direct.2.2.1
    · exact direct.2.1
    · exact direct.1
    · exact direct.2.2.2
  · exact Or.inr failure

/-- Candidate separation occurs in the verifier-owned frame before hashing. -/
theorem authoritativeFrames_ne_of_candidate_ne
    {leftCandidate rightCandidate : Id}
    (different : leftCandidate ≠ rightCandidate)
    {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (left : Value leftCandidate fullShape)
    (right : Value rightCandidate fullShape) :
    authoritativeFrame statementId degreeBound left ≠
      authoritativeFrame statementId degreeBound right := by
  intro equal
  have frameEqual :
      ProductionProductNifsPublicTranscript.frame degreeBound left =
        ProductionProductNifsPublicTranscript.frame degreeBound right := by
    exact List.append_cancel_left equal
  exact ProductionProductNifsPublicTranscript.frames_ne_of_candidate_ne
    different left right frameEqual

end Nightstream.Implementation.NebulaV2.ProductionFullClaimStateBinding
