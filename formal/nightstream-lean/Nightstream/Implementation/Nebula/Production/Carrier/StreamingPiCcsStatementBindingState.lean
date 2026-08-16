import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFusedPass
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementBinding

/-!
Contract: Poseidon2 binding for the variable production PiCCS statement.

Assurance tier: model-level transcript binding and cryptographic-reduction
boundary.

Owns a context-bound Poseidon2 state for the prior point and carried
evaluations selected from the authoritative claim frame. It also reduces any
different supplied variable frame with the same state to one named replay
collision.

The statement identifier and production prefix remain verifier-owned. The
direct point and evaluation fields remain the PiCCS algebra authority. The
binding state is checked compression only.

Does not own claim-chunk extraction, generated rows, physical columns,
PiCCS statement absorption, Rust refinement, collision resistance, or
lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId

/-- ASCII `NSPB`: Nightstream statement-PiCCS binding. -/
def domainTag : Nat := 0x4e535042

def frameVersion : Nat := 1

/-- Verifier-owned context for the variable-field binding. The statement
identifier is already present in `initialStateForStatement`. -/
def contextFields
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (degreeBound : Nat) : List Nat :=
  [domainTag, frameVersion] ++
    ProductionProductNifsPublicTranscript.fixedPrefix candidate fullShape
      degreeBound

@[simp] theorem contextFields_length
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (degreeBound : Nat) :
    (contextFields candidate fullShape degreeBound).length = 19 := by
  simp [contextFields,
    ProductionProductNifsPublicTranscript.fixedPrefix_length]

/-- Context-bound starting state used by both claim replay and PiCCS start. -/
noncomputable def contextState
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) (degreeBound : Nat) : State :=
  Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
    (contextFields candidate fullShape degreeBound)
    (ProductPoseidon2.initialStateForStatement statementId)

/-- Recompute the checked compression from direct variable fields. -/
noncomputable def stateForFields
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) (degreeBound : Nat)
    (fields : List Nat) : State :=
  Poseidon2Duplex.absorbSlice ProductPoseidon2.constants fields
    (contextState candidate fullShape statementId degreeBound)

/-- Expected state computed from the exact authoritative claim-frame
selection. -/
noncomputable def authoritativeState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : State :=
  stateForFields candidate fullShape statementId degreeBound
    (selectedAuthoritativeFields statementId degreeBound value)

/-- Exact failure event for a different variable frame that reaches the same
context-bound Poseidon2 state. -/
def VariableReplayCollision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : Prop :=
  exists supplied : List Nat,
    supplied ≠ selectedAuthoritativeFields statementId degreeBound value /\
      stateForFields candidate fullShape statementId degreeBound supplied =
        authoritativeState statementId degreeBound value

/-- Equal checked states recover the exact selected frame or expose the named
Poseidon2 replay collision. -/
theorem equal_state_recovers_fields_or_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (supplied : List Nat)
    (equal :
      stateForFields candidate fullShape statementId degreeBound supplied =
        authoritativeState statementId degreeBound value) :
    supplied = selectedAuthoritativeFields statementId degreeBound value \/
      VariableReplayCollision statementId degreeBound value := by
  by_cases exactFields : supplied =
      selectedAuthoritativeFields statementId degreeBound value
  · exact Or.inl exactFields
  · exact Or.inr ⟨supplied, exactFields, equal⟩

/-- The authoritative binding state is exactly the state recomputed from the
variable fields of the production PiCCS verifier input. -/
theorem authoritativeState_eq_exactVerifierInput
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (degreeBound : Nat)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    authoritativeState statementId degreeBound value =
      stateForFields candidate
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
        statementId degreeBound
        (frameOrderVariableFields
          (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
            statementId config artifact value.recursiveState fresh)) := by
  unfold authoritativeState
  rw [selectedAuthoritativeFields_exactVerifierInput candidate statementId
    degreeBound config artifact value fresh]

/-- A PiCCS-start assignment that replays to the authoritative state uses the
exact direct variable fields, or it exposes the named collision. -/
theorem accepted_fields_match_exactVerifierInput_or_collision
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (degreeBound : Nat)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (supplied : List Nat)
    (equal :
      stateForFields candidate
          (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
          statementId degreeBound supplied =
        authoritativeState statementId degreeBound value) :
    supplied =
        frameOrderVariableFields
          (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
            statementId config artifact value.recursiveState fresh) \/
      VariableReplayCollision statementId degreeBound value := by
  rcases equal_state_recovers_fields_or_collision statementId degreeBound value
      supplied equal with exactFields | collision
  · left
    exact exactFields.trans
      (selectedAuthoritativeFields_exactVerifierInput candidate statementId
        degreeBound config artifact value fresh)
  · exact Or.inr collision

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState
