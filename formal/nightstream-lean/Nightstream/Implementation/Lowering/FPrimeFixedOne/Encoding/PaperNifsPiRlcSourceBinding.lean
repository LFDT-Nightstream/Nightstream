import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types

/-!
Contract: bind every public input coefficient of the canonical `Pi_RLC`
quotient occurrence to the decoded running, fresh, and NIFS-proof operands.

Commitment and public-input coordinates are selected from the fresh/running
codecs.  Matrix-evaluation coordinates are selected from the proof's complete
`Pi_CCS` output.  All numeric columns are locations in the one call-local
column map; no value is copied and no source equation is supplied by a caller.

The carrier-coordinate profile contains only serialization functions and
codec projection laws.  It contains no verifier acceptance, quotient
identity, source-authority proposition, or named-event branch.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
open Nightstream.SuperNeo.Folding.Nifs

abbrev K := Nightstream.SuperNeo.Concrete.K

/-! ## Semantic serialization profile -/

/-- Exact base-field coordinate projections of the two generic public
carriers used by the paper key.  The fixed dimensions are protocol-owned:
eighteen commitment rings and five public-input rings, each of Phi81 width. -/
structure CarrierCoordinates
    (Commitment : Type)
    (PublicInput : Type) where
  commitment :
    Commitment → Fin 18 → Fin Concrete.ringDegree → Field
  publicInput :
    PublicInput → Fin 5 → Fin Concrete.ringDegree → Field

/-- Codec-coordinate ownership for every input coefficient consumed by the
public quotient occurrence. -/
structure Views
    {Commitment : Type}
    {PublicInput : Type}
    (shape : Shape)
    (degreeBound : Nat)
    (coordinates : CarrierCoordinates Commitment PublicInput)
    (runningCodec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape))
    (freshCodec :
      Codec (PaperNonInteractive.Fresh Commitment PublicInput shape))
    (proofCodec :
      Codec (PaperNonInteractive.Proof K Commitment shape degreeBound)) where
  freshCommitment :
    ∀ source lane coefficient,
      FView freshCodec (fun fresh =>
        coordinates.commitment (fresh.commitments source) lane coefficient)
  freshPublicInput :
    ∀ source block coefficient,
      FView freshCodec (fun fresh =>
        coordinates.publicInput (fresh.publicInputs source) block coefficient)
  runningCommitment :
    ∀ source lane coefficient,
      FView runningCodec (fun running =>
        coordinates.commitment (running.commitments source) lane coefficient)
  runningPublicInput :
    ∀ source block coefficient,
      FView runningCodec (fun running =>
        coordinates.publicInput (running.publicInputs source) block coefficient)
  fullOutput :
    ∀ source matrix coefficient,
      KView proofCodec (fun proof =>
        proof.piCcsOutput.coordinate source matrix coefficient)

/-! ## One global physical placement -/

/-- Selected operand codecs together with the three call-frame width
certificates.  This record contains representation facts only. -/
structure FrameViews
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {Commitment : Type}
    {PublicInput : Type}
    (shape : Shape)
    (coordinates : CarrierCoordinates Commitment PublicInput)
    (degreeBound : Nat) where
  runningCodec :
    Codec (PaperNonInteractive.Running K Commitment PublicInput shape)
  freshCodec :
    Codec (PaperNonInteractive.Fresh Commitment PublicInput shape)
  proofCodec :
    Codec (PaperNonInteractive.Proof K Commitment shape degreeBound)
  views :
    Views shape degreeBound coordinates runningCodec freshCodec proofCodec
  runningWidthsAgree :
    runningCodec.width = runningRef.port.layout.owners.length
  freshWidthsAgree :
    freshCodec.width = freshRef.port.layout.owners.length
  proofWidthsAgree :
    proofCodec.width = proofRef.port.layout.owners.length
  coefficientWidthAgree :
    shape.coefficientCount = Concrete.ringDegree

/-- Every selected codec coordinate located in the sole numeric namespace of
the surrounding call frame. -/
structure Placement
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {Commitment : Type}
    {PublicInput : Type}
    (shape : Shape)
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound) where
  freshCommitment :
    ∀ source lane coefficient,
      FLocation (columnMap frame)
        (((profile.views.freshCommitment source lane coefficient).column
          (freshOperand frame.operands) profile.freshWidthsAgree))
  freshPublicInput :
    ∀ source block coefficient,
      FLocation (columnMap frame)
        (((profile.views.freshPublicInput source block coefficient).column
          (freshOperand frame.operands) profile.freshWidthsAgree))
  runningCommitment :
    ∀ source lane coefficient,
      FLocation (columnMap frame)
        (((profile.views.runningCommitment source lane coefficient).column
          (runningOperand frame.operands) profile.runningWidthsAgree))
  runningPublicInput :
    ∀ source block coefficient,
      FLocation (columnMap frame)
        (((profile.views.runningPublicInput source block coefficient).column
          (runningOperand frame.operands) profile.runningWidthsAgree))
  fullOutput :
    ∀ source matrix coefficient,
      KLocation (columnMap frame)
        (((profile.views.fullOutput source matrix coefficient).columns
          (proofOperand frame.operands) profile.proofWidthsAgree))

/-- The call frame itself determines the complete placement. -/
def fromFrame
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {Commitment : Type}
    {PublicInput : Type}
    (shape : Shape)
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound) :
    Placement frame shape profile where
  freshCommitment source lane coefficient :=
    let view := profile.views.freshCommitment source lane coefficient
    let typed :=
      view.column (freshOperand frame.operands) profile.freshWidthsAgree
    fLocation frame typed
      (freshOperand_mem frame
        (view.column_mem
          (freshOperand frame.operands) profile.freshWidthsAgree))
  freshPublicInput source block coefficient :=
    let view := profile.views.freshPublicInput source block coefficient
    let typed :=
      view.column (freshOperand frame.operands) profile.freshWidthsAgree
    fLocation frame typed
      (freshOperand_mem frame
        (view.column_mem
          (freshOperand frame.operands) profile.freshWidthsAgree))
  runningCommitment source lane coefficient :=
    let view := profile.views.runningCommitment source lane coefficient
    let typed :=
      view.column (runningOperand frame.operands) profile.runningWidthsAgree
    fLocation frame typed
      (runningOperand_mem frame
        (view.column_mem
          (runningOperand frame.operands) profile.runningWidthsAgree))
  runningPublicInput source block coefficient :=
    let view := profile.views.runningPublicInput source block coefficient
    let typed :=
      view.column (runningOperand frame.operands) profile.runningWidthsAgree
    fLocation frame typed
      (runningOperand_mem frame
        (view.column_mem
          (runningOperand frame.operands) profile.runningWidthsAgree))
  fullOutput source matrix coefficient :=
    let view := profile.views.fullOutput source matrix coefficient
    let typed :=
      view.columns (proofOperand frame.operands) profile.proofWidthsAgree
    kLocation frame typed
      (proofOperand_mem frame
        (view.c0_mem
          (proofOperand frame.operands) profile.proofWidthsAgree))
      (proofOperand_mem frame
        (view.c1_mem
          (proofOperand frame.operands) profile.proofWidthsAgree))

/-! ## Numeric input columns -/

def kNumericLimb
    (columns : Nightstream.Implementation.R1CS.ProjectionProgram.KColumns)
    (limb : Fin 2) : Nat :=
  if limb.val = 0 then columns.c0 else columns.c1

def kValueLimb (value : K) (limb : Fin 2) : Field :=
  if limb.val = 0 then value.c0 else value.c1

theorem fLocation_paper_value_eq
    {columnMap : Nat → ColumnId}
    {typed : FColumnId}
    (location : FLocation columnMap typed)
    (assignment : ColumnId → Field) :
    FPrimeFullHistoryNifsPaper.residue
        (numericAssignment columnMap assignment location.numeric) =
      typed.value assignment := by
  exact location.numeric_value_eq assignment

theorem kLocation_paper_limb_value_eq
    {columnMap : Nat → ColumnId}
    {typed : KColumnIds}
    (location : KLocation columnMap typed)
    (assignment : ColumnId → Field)
    (limb : Fin 2) :
    FPrimeFullHistoryNifsPaper.residue
        (numericAssignment columnMap assignment
          (kNumericLimb location.numeric limb)) =
      kValueLimb (typed.value assignment) limb := by
  have whole := location.numeric_value_eq assignment
  by_cases low : limb.val = 0
  · have component := congrArg (fun value : K => value.c0) whole
    simpa [kNumericLimb, kValueLimb, low,
      FPrimeFullHistoryNifsPaper.residue, NumericRowBridge.residue,
      KConcreteFixedPhaseBridge.ofProjection,
      ProjectionProgram.KColumns.value,
      ProjectionProgram.baseAt] using component
  · have component := congrArg (fun value : K => value.c1) whole
    simpa [kNumericLimb, kValueLimb, low,
      FPrimeFullHistoryNifsPaper.residue, NumericRowBridge.residue,
      KConcreteFixedPhaseBridge.ofProjection,
      ProjectionProgram.KColumns.value,
      ProjectionProgram.baseAt] using component

def Placement.freshColumns
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (source : Fin shape.freshCount) :
    PiRlc.ProjectionColumns shape.matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      (placement.freshCommitment source lane coefficient).numeric
  x block :=
    List.ofFn fun coefficient =>
      (placement.freshPublicInput source block coefficient).numeric
  yRing matrix limb :=
    List.ofFn fun coefficient =>
      kNumericLimb
        (placement.fullOutput (freshSourceIndex source) matrix coefficient).numeric
        limb

def Placement.runningColumns
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (source : Fin shape.runningCount) :
    PiRlc.ProjectionColumns shape.matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      (placement.runningCommitment source lane coefficient).numeric
  x block :=
    List.ofFn fun coefficient =>
      (placement.runningPublicInput source block coefficient).numeric
  yRing matrix limb :=
    List.ofFn fun coefficient =>
      kNumericLimb
        (placement.fullOutput (runningSourceIndex source) matrix coefficient).numeric
        limb

@[simp] theorem Placement.freshColumns_width
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (source : Fin shape.freshCount)
    (role : PublicRole shape.matrixCount) :
    ((placement.freshColumns source).at role).length =
      Concrete.ringDegree := by
  cases role <;>
    simp [PiRlc.ProjectionColumns.at, Placement.freshColumns,
      profile.coefficientWidthAgree]

@[simp] theorem Placement.runningColumns_width
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (source : Fin shape.runningCount)
    (role : PublicRole shape.matrixCount) :
    ((placement.runningColumns source).at role).length =
      Concrete.ringDegree := by
  cases role <;>
    simp [PiRlc.ProjectionColumns.at, Placement.runningColumns,
      profile.coefficientWidthAgree]

/-- Exact `K+k` source order selected by the paper key. -/
def Placement.inputColumns
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {Scalar : Type}
    {State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (index : Fin key.arity.total) :
    PiRlc.ProjectionColumns shape.matrixCount :=
  Fin.addCases placement.freshColumns placement.runningColumns
    (Fin.cast key.total_eq_sourceCount index)

@[simp] theorem Placement.inputColumns_width
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment : Type}
    {PublicInput : Type}
    {Scalar : Type}
    {State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (index : Fin key.arity.total)
    (role : PublicRole shape.matrixCount) :
    ((placement.inputColumns key index).at role).length =
      Concrete.ringDegree := by
  unfold Placement.inputColumns
  let combined :
      Fin (shape.freshCount + shape.runningCount) →
        PiRlc.ProjectionColumns shape.matrixCount :=
    Fin.addCases placement.freshColumns placement.runningColumns
  have widthBySource :
      ∀ source, ((combined source).at role).length =
        Concrete.ringDegree := fun source =>
    Fin.addCases
      (motive := fun source =>
        ((combined source).at role).length = Concrete.ringDegree)
      (fun fresh => by
        simpa [combined] using placement.freshColumns_width fresh role)
      (fun running => by
        simpa [combined] using placement.runningColumns_width running role)
      source
  exact widthBySource (Fin.cast key.total_eq_sourceCount index)

/-! ## Authoritative semantic openings -/

/-- Fresh public source in the exact coefficient order consumed by the
canonical public quotient occurrence. -/
def FrameViews.freshOpening
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (source : Fin shape.freshCount) :
    PiRlc.ProjectionOpening shape.matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      coordinates.commitment (fresh.commitments source) lane coefficient
  x block :=
    List.ofFn fun coefficient =>
      coordinates.publicInput (fresh.publicInputs source) block coefficient
  yRing matrix limb :=
    List.ofFn fun coefficient =>
      kValueLimb
        (proof.piCcsOutput.coordinate
          (freshSourceIndex source) matrix coefficient)
        limb

/-- Running public source in the same coefficient order.  Unlike a fresh
source, no running coordinate is completed or truncated here. -/
def FrameViews.runningOpening
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (source : Fin shape.runningCount) :
    PiRlc.ProjectionOpening shape.matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      coordinates.commitment (running.commitments source) lane coefficient
  x block :=
    List.ofFn fun coefficient =>
      coordinates.publicInput (running.publicInputs source) block coefficient
  yRing matrix limb :=
    List.ofFn fun coefficient =>
      kValueLimb
        (proof.piCcsOutput.coordinate
          (runningSourceIndex source) matrix coefficient)
        limb

/-- Exact `K+k` semantic source order selected by the paper key. -/
def FrameViews.inputOpening
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput Scalar State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (index : Fin key.arity.total) :
    PiRlc.ProjectionOpening shape.matrixCount :=
  Fin.addCases
    (profile.freshOpening fresh proof)
    (profile.runningOpening running proof)
    (Fin.cast key.total_eq_sourceCount index)

/-- Coordinate projection of the verifier-computed public `Pi_CCS` output.
Commitment and public-input values come from the output itself; its evaluation
family is the complete output carried by the decoded proof and copied by
`Key.piCcsOutputs`. -/
def piCcsOutputProjection
    {Commitment PublicInput Scalar State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (coordinates : CarrierCoordinates Commitment PublicInput)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (index : Fin key.arity.total) :
    PiRlc.ProjectionOpening shape.matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      coordinates.commitment
        (key.piCcsOutputs running fresh proof index).commitment
        lane coefficient
  x block :=
    List.ofFn fun coefficient =>
      coordinates.publicInput
        (key.piCcsOutputs running fresh proof index).publicInput
        block coefficient
  yRing matrix limb :=
    List.ofFn fun coefficient =>
      kValueLimb
        (proof.piCcsOutput.coordinate
          (Fin.cast key.total_eq_sourceCount index) matrix coefficient)
        limb

private theorem projectionOpening_eq
    {matrixCount : Nat}
    (left right : PiRlc.ProjectionOpening matrixCount)
    (commitment : left.commitment = right.commitment)
    (x : left.x = right.x)
    (yRing : left.yRing = right.yRing) :
    left = right := by
  cases left
  cases right
  simp only at commitment x yRing
  cases commitment
  cases x
  cases yRing
  rfl

/-- The source order used by the physical carrier is exactly the public
`Pi_CCS` output order computed by the selected paper key. -/
theorem FrameViews.inputOpening_eq_piCcsOutputProjection
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput Scalar State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    (profile : FrameViews frame shape coordinates degreeBound)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (index : Fin key.arity.total) :
    profile.inputOpening key running fresh proof index =
      piCcsOutputProjection coordinates key running fresh proof index := by
  let source := Fin.cast key.total_eq_sourceCount index
  let openingBySource :
      Fin (shape.freshCount + shape.runningCount) →
        PiRlc.ProjectionOpening shape.matrixCount :=
    Fin.addCases
      (profile.freshOpening fresh proof)
      (profile.runningOpening running proof)
  let semanticBySource :
      Fin (shape.freshCount + shape.runningCount) →
        PiRlc.ProjectionOpening shape.matrixCount :=
    fun source => {
      commitment := fun lane =>
        List.ofFn fun coefficient =>
          coordinates.commitment
            ((key.statement running fresh).commitments source)
            lane coefficient
      x := fun block =>
        List.ofFn fun coefficient =>
          coordinates.publicInput
            ((key.statement running fresh).publicInputs source)
            block coefficient
      yRing := fun matrix limb =>
        List.ofFn fun coefficient =>
          kValueLimb
            (proof.piCcsOutput.coordinate source matrix coefficient)
            limb
    }
  have bySource :
      ∀ source : Fin (shape.freshCount + shape.runningCount),
        openingBySource source = semanticBySource source := fun source =>
    Fin.addCases
      (motive := fun source =>
        openingBySource source = semanticBySource source)
      (fun freshSource => by
        have sourceEq :
            freshSourceIndex freshSource =
              Fin.castAdd shape.runningCount freshSource :=
          Fin.eq_of_val_eq rfl
        simp [openingBySource, semanticBySource,
          FrameViews.freshOpening, PaperNonInteractive.Key.statement,
          sourceEq])
      (fun runningSource => by
        have sourceEq :
            runningSourceIndex runningSource =
              Fin.natAdd shape.freshCount runningSource :=
          Fin.eq_of_val_eq rfl
        simp [openingBySource, semanticBySource,
          FrameViews.runningOpening, PaperNonInteractive.Key.statement,
          sourceEq])
      source
  unfold FrameViews.inputOpening piCcsOutputProjection
  change openingBySource source = semanticBySource source
  exact bySource source

/-- Decoding the fresh and proof operands binds every physical fresh-source
coefficient to the corresponding semantic paper source. -/
theorem Placement.decoded_freshColumns_eq
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (assignment : ColumnId → Field)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (decodedFresh :
      profile.freshCodec.decode
          ((freshOperand frame.operands).values assignment) =
        some fresh)
    (decodedProof :
      profile.proofCodec.decode
          ((proofOperand frame.operands).values assignment) =
        some proof)
    (source : Fin shape.freshCount) :
    PiRlc.decodeOpening
        (numericAssignment (columnMap frame) assignment)
        (placement.freshColumns source) =
      profile.freshOpening fresh proof source := by
  apply projectionOpening_eq
  · funext lane
    simp only [PiRlc.decodeOpening, Placement.freshColumns,
      FrameViews.freshOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location := placement.freshCommitment source lane coefficient
    let view := profile.views.freshCommitment source lane coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            location.numeric) =
          (view.column
            (freshOperand frame.operands)
            profile.freshWidthsAgree).value assignment :=
        fLocation_paper_value_eq location assignment
      _ = coordinates.commitment (fresh.commitments source) lane coefficient :=
        view.value_eq_of_decodes
          (freshOperand frame.operands) profile.freshWidthsAgree
          assignment fresh decodedFresh
  · funext block
    simp only [PiRlc.decodeOpening, Placement.freshColumns,
      FrameViews.freshOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location := placement.freshPublicInput source block coefficient
    let view := profile.views.freshPublicInput source block coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            location.numeric) =
          (view.column
            (freshOperand frame.operands)
            profile.freshWidthsAgree).value assignment :=
        fLocation_paper_value_eq location assignment
      _ = coordinates.publicInput
            (fresh.publicInputs source) block coefficient :=
        view.value_eq_of_decodes
          (freshOperand frame.operands) profile.freshWidthsAgree
          assignment fresh decodedFresh
  · funext matrix limb
    simp only [PiRlc.decodeOpening, Placement.freshColumns,
      FrameViews.freshOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location :=
      placement.fullOutput (freshSourceIndex source) matrix coefficient
    let view :=
      profile.views.fullOutput (freshSourceIndex source) matrix coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            (kNumericLimb location.numeric limb)) =
          kValueLimb
            ((view.columns
              (proofOperand frame.operands)
              profile.proofWidthsAgree).value assignment)
            limb :=
        kLocation_paper_limb_value_eq location assignment limb
      _ = kValueLimb
            (proof.piCcsOutput.coordinate
              (freshSourceIndex source) matrix coefficient)
            limb :=
        congrArg (fun value => kValueLimb value limb)
          (view.value_eq_of_decodes
            (proofOperand frame.operands) profile.proofWidthsAgree
            assignment proof decodedProof)

/-- Decoding the running and proof operands binds every physical
running-source coefficient.  In particular, this path never passes through
the fresh completion codec. -/
theorem Placement.decoded_runningColumns_eq
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput : Type}
    {shape : Shape}
    {degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (assignment : ColumnId → Field)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (decodedRunning :
      profile.runningCodec.decode
          ((runningOperand frame.operands).values assignment) =
        some running)
    (decodedProof :
      profile.proofCodec.decode
          ((proofOperand frame.operands).values assignment) =
        some proof)
    (source : Fin shape.runningCount) :
    PiRlc.decodeOpening
        (numericAssignment (columnMap frame) assignment)
        (placement.runningColumns source) =
      profile.runningOpening running proof source := by
  apply projectionOpening_eq
  · funext lane
    simp only [PiRlc.decodeOpening, Placement.runningColumns,
      FrameViews.runningOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location := placement.runningCommitment source lane coefficient
    let view := profile.views.runningCommitment source lane coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            location.numeric) =
          (view.column
            (runningOperand frame.operands)
            profile.runningWidthsAgree).value assignment :=
        fLocation_paper_value_eq location assignment
      _ = coordinates.commitment
            (running.commitments source) lane coefficient :=
        view.value_eq_of_decodes
          (runningOperand frame.operands) profile.runningWidthsAgree
          assignment running decodedRunning
  · funext block
    simp only [PiRlc.decodeOpening, Placement.runningColumns,
      FrameViews.runningOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location := placement.runningPublicInput source block coefficient
    let view := profile.views.runningPublicInput source block coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            location.numeric) =
          (view.column
            (runningOperand frame.operands)
            profile.runningWidthsAgree).value assignment :=
        fLocation_paper_value_eq location assignment
      _ = coordinates.publicInput
            (running.publicInputs source) block coefficient :=
        view.value_eq_of_decodes
          (runningOperand frame.operands) profile.runningWidthsAgree
          assignment running decodedRunning
  · funext matrix limb
    simp only [PiRlc.decodeOpening, Placement.runningColumns,
      FrameViews.runningOpening, FPrimeFullHistoryNifsPaper.values,
      List.map_ofFn]
    apply congrArg List.ofFn
    funext coefficient
    let location :=
      placement.fullOutput (runningSourceIndex source) matrix coefficient
    let view :=
      profile.views.fullOutput (runningSourceIndex source) matrix coefficient
    calc
      FPrimeFullHistoryNifsPaper.residue
          (numericAssignment (columnMap frame) assignment
            (kNumericLimb location.numeric limb)) =
          kValueLimb
            ((view.columns
              (proofOperand frame.operands)
              profile.proofWidthsAgree).value assignment)
            limb :=
        kLocation_paper_limb_value_eq location assignment limb
      _ = kValueLimb
            (proof.piCcsOutput.coordinate
              (runningSourceIndex source) matrix coefficient)
            limb :=
        congrArg (fun value => kValueLimb value limb)
          (view.value_eq_of_decodes
            (proofOperand frame.operands) profile.proofWidthsAgree
            assignment proof decodedProof)

/-- **Complete physical-to-semantic source binding for the selected public
`Pi_RLC` input batch.**

The physical source at every `K+k` index decodes to the paper source at the
same index.  The only hypotheses are whole-operand decoding facts; no source
opening or equality is supplied independently. -/
theorem Placement.decoded_inputColumns_eq
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))}
    {Commitment PublicInput Scalar State : Type}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {coordinates : CarrierCoordinates Commitment PublicInput}
    {profile : FrameViews frame shape coordinates degreeBound}
    (placement : Placement frame shape profile)
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    (assignment : ColumnId → Field)
    (running :
      PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : PaperNonInteractive.Proof K Commitment shape degreeBound)
    (decodedRunning :
      profile.runningCodec.decode
          ((runningOperand frame.operands).values assignment) =
        some running)
    (decodedFresh :
      profile.freshCodec.decode
          ((freshOperand frame.operands).values assignment) =
        some fresh)
    (decodedProof :
      profile.proofCodec.decode
          ((proofOperand frame.operands).values assignment) =
        some proof)
    (index : Fin key.arity.total) :
    PiRlc.decodeOpening
        (numericAssignment (columnMap frame) assignment)
        (placement.inputColumns key index) =
      profile.inputOpening key running fresh proof index := by
  unfold Placement.inputColumns FrameViews.inputOpening
  let physical :
      Fin (shape.freshCount + shape.runningCount) →
        PiRlc.ProjectionColumns shape.matrixCount :=
    Fin.addCases placement.freshColumns placement.runningColumns
  let semantic :
      Fin (shape.freshCount + shape.runningCount) →
        PiRlc.ProjectionOpening shape.matrixCount :=
    Fin.addCases
      (profile.freshOpening fresh proof)
      (profile.runningOpening running proof)
  have bound :
      ∀ source,
        PiRlc.decodeOpening
            (numericAssignment (columnMap frame) assignment)
            (physical source) =
          semantic source := fun source =>
    Fin.addCases
      (motive := fun source =>
        PiRlc.decodeOpening
            (numericAssignment (columnMap frame) assignment)
            (physical source) =
          semantic source)
      (fun freshSource => by
        simpa [physical, semantic] using
          placement.decoded_freshColumns_eq assignment fresh proof
            decodedFresh decodedProof freshSource)
      (fun runningSource => by
        simpa [physical, semantic] using
          placement.decoded_runningColumns_eq assignment running proof
            decodedRunning decodedProof runningSource)
      source
  exact bound (Fin.cast key.total_eq_sourceCount index)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding
