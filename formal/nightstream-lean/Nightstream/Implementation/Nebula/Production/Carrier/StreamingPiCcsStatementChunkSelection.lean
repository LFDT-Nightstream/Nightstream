import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementBindingState

/-!
Contract: select the production PiCCS variable fields from claim-replay
chunks.

Assurance tier: model-level serialization and state-binding refinement.

Owns the fixed radix-four claim-frame windows for the prior point and carried
evaluations. The selector reads the same ordered chunks that the full-claim
Poseidon2 replay reads. It does not create a second source frame.

Does not own generated rows, physical columns, PiCCS challenges, collision
resistance, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementChunkSelection

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

def claimChunkWidth : Nat := 1024

/-- Exact physical chunk locations of the two selected windows. -/
theorem production_chunk_window_geometry :
    pointFrameStart = 0 * claimChunkWidth + 383 /\
      runningPointFieldCount 26 = 52 /\
      evaluationFrameStart 26 = 60 * claimChunkWidth + 987 /\
      evaluationFrameStart 26 + runningEvaluationFieldCount =
        81 * claimChunkWidth + 651 /\
      383 + 52 <= claimChunkWidth /\
      987 + 37 = claimChunkWidth := by
  decide

/-- Select the two PiCCS variable windows from the one ordered claim stream.
The full claim replay remains the authority for `chunks.flatten`. -/
def selectedFieldsFromChunks (chunks : List (List Nat)) : List Nat :=
  ((chunks.flatten.drop pointFrameStart).take
      (runningPointFieldCount 26)) ++
    ((chunks.flatten.drop (evaluationFrameStart 26)).take
      runningEvaluationFieldCount)

/-- Exact claim replay makes the selected chunk fields equal to the direct
authoritative claim-frame fields. -/
theorem selectedFieldsFromChunks_eq_authoritative
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (exactFrame : chunks.flatten =
      authoritativeFrame statementId degreeBound value) :
    selectedFieldsFromChunks chunks =
      selectedAuthoritativeFields statementId degreeBound value := by
  unfold selectedFieldsFromChunks selectedAuthoritativeFields pointWindow
    evaluationWindow
  rw [exactFrame, contract.rowVariablesExact]

@[simp] theorem selectedFieldsFromChunks_length_of_exactFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (exactFrame : chunks.flatten =
      authoritativeFrame statementId degreeBound value) :
    (selectedFieldsFromChunks chunks).length = 21220 := by
  rw [selectedFieldsFromChunks_eq_authoritative contract statementId
    degreeBound value chunks exactFrame]
  exact selectedAuthoritativeFields_length_r26 contract statementId
    degreeBound value

/-- Poseidon2 state computed from the exact selected fields of the claim
chunks. Candidate and shape are explicit because the chunks alone do not
carry their verifier-owned context. -/
noncomputable def selectedStateFromChunks
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (chunks : List (List Nat)) : ProductPoseidon2.State :=
  stateForFields candidate fullShape statementId degreeBound
    (selectedFieldsFromChunks chunks)

/-- Exact full-claim replay gives the exact authoritative variable-field
binding state. -/
theorem selectedStateFromChunks_eq_authoritative
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (exactFrame : chunks.flatten =
      authoritativeFrame statementId degreeBound value) :
    selectedStateFromChunks candidate fullShape statementId degreeBound
        chunks =
      authoritativeState statementId degreeBound value := by
  unfold selectedStateFromChunks authoritativeState
  rw [selectedFieldsFromChunks_eq_authoritative contract statementId
    degreeBound value chunks exactFrame]

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementChunkSelection
