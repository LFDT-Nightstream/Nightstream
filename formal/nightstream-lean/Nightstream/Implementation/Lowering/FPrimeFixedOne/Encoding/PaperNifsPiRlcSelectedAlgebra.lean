import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcParentBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.CommitmentBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicInputBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.EvaluationBridge

/-!
Contract: interpret the exact public `Pi_RLC` coefficient equations through
the concrete Phi81 algebra selected by both `Pi_RLC` and `Pi_DEC`.

The quotient occurrence proves one list-level Phi81 equation per public role.
This module decodes those lists into the independently defined typed
commitment, complete 270-coordinate public input, and matrix-evaluation
carriers.  No caller supplies an algebra law or an accepted parent.

Owns:
- exact decoding of every commitment role into the 18-row Ajtai carrier;
- exact decoding of all five public-input rings into the 270-coordinate
  carrier;
- exact pairing of both limbs of every matrix evaluation; and
- the three concrete combination equations implied by occurrence exactness.

Does not own:
- source or challenge authority;
- the shared point;
- strict-`Pi_DEC` sidecars;
- quotient-row satisfaction or its bad-root branch;
- construction of a complete paper key; or
- Rust/artifact correspondence.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSelectedAlgebra

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge

/-! ## Typed decoding -/

/-- Decode the eighteen public commitment coefficient blocks. -/
def decodeCommitment {matrixCount : Nat}
    (opening : ProjectionOpening matrixCount) :
    PiRLCAlgebra.Commitment.Value 18 :=
  decodeCommitmentRings opening.commitment

/-- Decode the five public coefficient blocks into the complete 270-coordinate
public input selected by `dimensions`. -/
def decodePublicInput (dimensions : Dimensions)
    {matrixCount : Nat} (opening : ProjectionOpening matrixCount) :
    Phi81Relation.PublicInput dimensions.shape :=
  decodeXRings (dimensions := dimensions) opening.x

/-- Pair both limbs of every matrix-indexed evaluation. -/
def decodeEvaluations {matrixCount : Nat}
    (opening : ProjectionOpening matrixCount) : Array Phi81Relation.Evaluation :=
  decodeYRingRings opening.yRing

/-! ## Exact occurrence equations in the selected algebra -/

/-- Occurrence exactness determines the verifier-owned typed commitment fold.
The Ajtai key is intentionally absent: this equation combines public
commitments and is the shared input to either key's commitment checker. -/
theorem exact_output_commitment
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : KPiRlcSemanticBinding.ProjectionColumns
      params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact : (columns.occurrence valid base).Exact assignment) :
    decodeCommitment
        (decodeOpening assignment columns.source.output) =
      PiRLCAlgebra.Commitment.combineCommitments
        (fun index =>
          ringOfList
            (values assignment (columns.source.challenges index)))
        (fun index =>
          decodeCommitment
            (decodeOpening assignment (columns.source.inputs index))) := by
  have output :
      (decodeOpening assignment columns.source.output).commitment =
        fun lane =>
          phi81Combine
            (fun index =>
              values assignment (columns.source.challenges index))
            (fun index =>
              (decodeOpening assignment
                (columns.source.inputs index)).commitment lane) := by
    funext lane
    exact KPiRlcSemanticBinding.exact_output_at
      columns valid base assignment exact (.commitment lane)
  unfold decodeCommitment
  rw [output]
  exact decodeCommitmentRings_phi81Combine _ _

/-- Occurrence exactness determines the typed five-ring public-input fold.
This is the complete 270-coordinate carrier; no 257-coordinate projection
appears in the statement. -/
theorem exact_output_publicInput
    (dimensions : Dimensions)
    {params : GlobalParams} {arity : BatchArity params}
    (columns : KPiRlcSemanticBinding.ProjectionColumns
      params arity dimensions.shape.matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact : (columns.occurrence valid base).Exact assignment) :
    decodePublicInput dimensions
        (decodeOpening assignment columns.source.output) =
      PiRLCAlgebra.PublicInput.combinePublicInputs
        (fun index =>
          ringOfList
            (values assignment (columns.source.challenges index)))
        (fun index =>
          decodePublicInput dimensions
            (decodeOpening assignment (columns.source.inputs index))) := by
  have output :
      (decodeOpening assignment columns.source.output).x =
        fun block =>
          phi81Combine
            (fun index =>
              values assignment (columns.source.challenges index))
            (fun index =>
              (decodeOpening assignment
                (columns.source.inputs index)).x block) := by
    funext block
    exact KPiRlcSemanticBinding.exact_output_at
      columns valid base assignment exact (.x block)
  unfold decodePublicInput
  rw [output]
  exact decodeXRings_phi81Combine _ _

/-- Occurrence exactness determines the typed matrix-evaluation fold at the
same concrete relation shape. -/
theorem exact_output_evaluations
    (dimensions : Dimensions)
    {params : GlobalParams} {arity : BatchArity params}
    (columns : KPiRlcSemanticBinding.ProjectionColumns
      params arity dimensions.shape.matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact : (columns.occurrence valid base).Exact assignment) :
    decodeEvaluations
        (decodeOpening assignment columns.source.output) =
      PiRLCFinite.combineEvaluations (shape := dimensions.shape)
        (fun index =>
          ringOfList
            (values assignment (columns.source.challenges index)))
        (fun index =>
          decodeEvaluations
            (decodeOpening assignment (columns.source.inputs index))) := by
  have output :
      (decodeOpening assignment columns.source.output).yRing =
        fun row limb =>
          phi81Combine
            (fun index =>
              values assignment (columns.source.challenges index))
            (fun index =>
              (decodeOpening assignment
                (columns.source.inputs index)).yRing row limb) := by
    funext row limb
    exact KPiRlcSemanticBinding.exact_output_at
      columns valid base assignment exact (.yRing row limb)
  unfold decodeEvaluations
  rw [output]
  exact decodeYRingRings_phi81Combine _ _

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSelectedAlgebra
