import Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
import Nightstream.Implementation.NebulaV2.ProductNifsPublicAbsorptionRowsFor

/-!
Contract: exact production-profile public-field placement for PiCCS rows.

Owns the bridge from the exponent-selected physical row expressions to the
candidate-specific paper NIFS public-input serialization. Satisfying the
complete public-prefix rows then fixes the decoded post-public-input Poseidon2
state to the exact production successor state.

`Placement` is a generated-manifest and implementation-refinement boundary.
It contains only equality between decoded physical fields and independently
defined paper inputs. It does not contain a challenge, a verifier result,
PiCCS acceptance, NIFS acceptance, or the desired execution conclusion.

Does not own the remaining PiCCS placement, the concrete production NIFS key,
PiRLC, PiDEC, Rust refinement, cryptographic transcript security, or terminal
verification.

Emits constraints: no; it proves the meaning of existing transcript rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsPublicAbsorptionRowsFor
open Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- The exact physical public-input expressions for one production PiCCS
call. The length is authority-bearing and profile fixed. -/
def publicFieldCount (fullShape : Phi81Relation.Shape) : Nat :=
  17 + ProductNifsCodec.runningFieldCountFor fullShape.rowVariables +
    3888 + 540

structure PublicWires (fullShape : Phi81Relation.Shape) where
  fields : List LinComb
  fields_length : fields.length = publicFieldCount fullShape

abbrev PrefixInput (fullShape : Phi81Relation.Shape) :=
  ProductNifsPublicAbsorptionRowsFor.Input (publicFieldCount fullShape)

/-- Replace only the public-input expressions of a complete PiCCS row input.
All statement, proof, output, and transcript-allocation data remain unchanged.
-/
def installPublicWires {fullShape : Phi81Relation.Shape}
    (input : PrefixInput fullShape) (wires : PublicWires fullShape) :
    PrefixInput fullShape where
  statementId := input.statementId
  fields := wires.fields
  fields_length := wires.fields_length
  transcriptBase := input.transcriptBase

@[simp] theorem installPublicWires_fields
    {fullShape : Phi81Relation.Shape}
    (input : PrefixInput fullShape) (wires : PublicWires fullShape) :
    (installPublicWires input wires).fields = wires.fields := rfl

@[simp] theorem installPublicWires_statementId
    {fullShape : Phi81Relation.Shape}
    (input : PrefixInput fullShape) (wires : PublicWires fullShape) :
    (installPublicWires input wires).statementId = input.statementId := rfl

/-- Exact generated placement of the candidate-specific paper inputs.

This equality must be proved by the generated column map and by the concrete
implementation refinement. It is not a protocol soundness assumption. -/
structure Placement
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (_contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat)
    (running : ProductionFieldNativeFullClaim.Running fullShape)
    (fresh : ProductionFieldNativeFullClaim.Fresh fullShape)
    (wires : PublicWires fullShape) (assignment : Nat -> Nat) : Prop where
  exactFields :
    fieldValues assignment wires.fields =
      publicNifsFields candidate degreeBound running fresh

/-- Value-level successor state selected by the production paper NIFS key. -/
noncomputable def successorPublicState
    (statementId : ProductPoseidon2.StatementId)
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat)
    (running : ProductionFieldNativeFullClaim.Running fullShape)
    (fresh : ProductionFieldNativeFullClaim.Fresh fullShape) :
    ProductPoseidon2.State :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (publicNifsFields candidate degreeBound running fresh)
    (ProductPoseidon2.initialStateForStatement statementId)

/-- Complete PiCCS row satisfaction fixes the public-input prefix to the exact
candidate-specific successor state. No state or verifier result is supplied
as an assumption. -/
theorem rows_imply_successor_public_state
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat)
    (running : ProductionFieldNativeFullClaim.Running fullShape)
    (fresh : ProductionFieldNativeFullClaim.Fresh fullShape)
    (input : PrefixInput fullShape) (wires : PublicWires fullShape)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate contract degreeBound running fresh
      wires assignment)
    (satisfied : Satisfies
      (ProductNifsPublicAbsorptionRowsFor.rows
        (installPublicWires input wires)) assignment) :
    decodedBuilder assignment
        (ProductNifsPublicAbsorptionRowsFor.absorbPublicInput
          (installPublicWires input wires)) =
      successorPublicState input.statementId candidate degreeBound
        running fresh := by
  calc
    decodedBuilder assignment
        (ProductNifsPublicAbsorptionRowsFor.absorbPublicInput
          (installPublicWires input wires)) =
        ProductNifsPublicAbsorptionRowsFor.valueAbsorbPublic assignment
          (installPublicWires input wires) :=
      ProductNifsPublicAbsorptionRowsFor.rows_semantics assignment
        (installPublicWires input wires) residues one satisfied
    _ = successorPublicState input.statementId candidate degreeBound
        running fresh := by
      unfold ProductNifsPublicAbsorptionRowsFor.valueAbsorbPublic
        successorPublicState
      rw [installPublicWires_fields, placement.exactFields]
      rw [installPublicWires_statementId]

/-- The same row theorem stated for a canonical complete production claim. -/
theorem rows_imply_value_public_state
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (input : PrefixInput fullShape) (wires : PublicWires fullShape)
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate contract degreeBound
      value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue contract value)
      wires assignment)
    (satisfied : Satisfies
      (ProductNifsPublicAbsorptionRowsFor.rows
        (installPublicWires input wires)) assignment) :
    decodedBuilder assignment
        (ProductNifsPublicAbsorptionRowsFor.absorbPublicInput
          (installPublicWires input wires)) =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (frame degreeBound value)
        (ProductPoseidon2.initialStateForStatement input.statementId) := by
  calc
    decodedBuilder assignment
        (absorbPublicInput (installPublicWires input wires)) =
        successorPublicState input.statementId candidate degreeBound
          value.recursiveState
          (ProductionFieldNativeFullClaim.freshOfValue contract value) :=
      rows_imply_successor_public_state candidate contract degreeBound
        value.recursiveState
        (ProductionFieldNativeFullClaim.freshOfValue contract value)
        input wires assignment residues one placement satisfied
    _ = Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (frame degreeBound value)
        (ProductPoseidon2.initialStateForStatement input.statementId) := by
      unfold successorPublicState
      rw [publicNifsFields_of_value contract]

/-- One physical field list cannot be placed as two different production
profiles. Profile separation is deterministic and occurs before Poseidon2. -/
theorem no_cross_candidate_dual_placement
    {leftCandidate rightCandidate : Id}
    (different : leftCandidate ≠ rightCandidate)
    {fullShape : Phi81Relation.Shape}
    (leftContract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (rightContract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {degreeBound : Nat}
    {leftRunning : ProductionFieldNativeFullClaim.Running fullShape}
    {leftFresh : ProductionFieldNativeFullClaim.Fresh fullShape}
    {rightRunning : ProductionFieldNativeFullClaim.Running fullShape}
    {rightFresh : ProductionFieldNativeFullClaim.Fresh fullShape}
    {wires : PublicWires fullShape} {assignment : Nat -> Nat}
    (leftPlacement : Placement leftCandidate leftContract degreeBound
      leftRunning leftFresh wires assignment)
    (rightPlacement : Placement rightCandidate rightContract degreeBound
      rightRunning rightFresh wires assignment) : False := by
  apply publicNifsFields_ne_of_candidate_ne different
    leftRunning leftFresh rightRunning rightFresh
  exact leftPlacement.exactFields.symm.trans rightPlacement.exactFields

end Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge
