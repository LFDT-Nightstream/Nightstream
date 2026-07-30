import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.R1CS.Canonical.KPointEquality

/-!
Contract: locate selected-NIFS public carrier views in the sole typed
call-frame column namespace and decode them from authoritative bundles.

This representation bridge is generic over the fixed-one parameter package.
It allocates no columns, emits no rows, and carries no semantic equation.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap

universe u

section Frame

variable {parameters : Parameters}
variable
    (family : Family (typeSystem parameters))
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

/-- The sole output bundle has the running codec's exact width. -/
theorem output_widthsAgree
    (callFrame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (family.codecFor (.data .running)).width =
      (Ports.committedRunning parameters).layout.owners.length := by
  exact
    (CallFrame.outputWidthsAgree callFrame)
      (Ports.committedRunning parameters)
      List.mem_cons_self

/-! ## Physical locations -/

def runningFLocation
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value) :
    FLocation (columnMap frame)
      (view.column (runningOperand frame.operands)
        (running_widthsAgree frame)) := by
  apply fLocation frame
  exact runningOperand_mem frame
    (view.column_mem (runningOperand frame.operands)
      (running_widthsAgree frame))

def runningKLocation
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value) :
    KLocation (columnMap frame)
      (view.columns (runningOperand frame.operands)
        (running_widthsAgree frame)) := by
  apply kLocation frame
  · exact runningOperand_mem frame
      (view.c0_mem (runningOperand frame.operands)
        (running_widthsAgree frame))
  · exact runningOperand_mem frame
      (view.c1_mem (runningOperand frame.operands)
        (running_widthsAgree frame))

def freshFLocation
    {value : parameters.Fresh → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .fresh)) value) :
    FLocation (columnMap frame)
      (view.column (freshOperand frame.operands)
        (fresh_widthsAgree frame)) := by
  apply fLocation frame
  exact freshOperand_mem frame
    (view.column_mem (freshOperand frame.operands)
      (fresh_widthsAgree frame))

def proofFLocation
    {value : parameters.NifsProof → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .nifsProof)) value) :
    FLocation (columnMap frame)
      (view.column (proofOperand frame.operands)
        (proof_widthsAgree frame)) := by
  apply fLocation frame
  exact proofOperand_mem frame
    (view.column_mem (proofOperand frame.operands)
      (proof_widthsAgree frame))

def proofKLocation
    {value : parameters.NifsProof → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value) :
    KLocation (columnMap frame)
      (view.columns (proofOperand frame.operands)
        (proof_widthsAgree frame)) := by
  apply kLocation frame
  · exact proofOperand_mem frame
      (view.c0_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))
  · exact proofOperand_mem frame
      (view.c1_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))

def outputFLocation
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value) :
    FLocation (columnMap frame)
      (view.column (unaryOutput frame.outputs)
        (output_widthsAgree family frame)) := by
  apply fLocation frame
  apply output_mem frame
  simpa only [unaryOutput_ids] using
    view.column_mem (unaryOutput frame.outputs)
      (output_widthsAgree family frame)

def outputKLocation
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value) :
    KLocation (columnMap frame)
      (view.columns (unaryOutput frame.outputs)
        (output_widthsAgree family frame)) := by
  apply kLocation frame
  · apply output_mem frame
    simpa only [unaryOutput_ids] using
      view.c0_mem (unaryOutput frame.outputs)
        (output_widthsAgree family frame)
  · apply output_mem frame
    simpa only [unaryOutput_ids] using
      view.c1_mem (unaryOutput frame.outputs)
        (output_widthsAgree family frame)

/-! ## Visible-prefix bounds -/

theorem runningFLocation_numeric_lt
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value) :
    (runningFLocation family frame view).numeric <
      temporaryBase frame := by
  unfold runningFLocation
  apply fLocation_numeric_lt_temporaryBase
  exact runningOperand_mem_visible frame
    (view.column_mem (runningOperand frame.operands)
      (running_widthsAgree frame))

theorem runningKLocation_numeric_lt
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value) :
    (runningKLocation family frame view).numeric.c0 <
        temporaryBase frame
      ∧ (runningKLocation family frame view).numeric.c1 <
        temporaryBase frame := by
  unfold runningKLocation
  apply kLocation_numeric_lt_temporaryBase
  · exact runningOperand_mem_visible frame
      (view.c0_mem (runningOperand frame.operands)
        (running_widthsAgree frame))
  · exact runningOperand_mem_visible frame
      (view.c1_mem (runningOperand frame.operands)
        (running_widthsAgree frame))

theorem freshFLocation_numeric_lt
    {value : parameters.Fresh → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .fresh)) value) :
    (freshFLocation family frame view).numeric <
      temporaryBase frame := by
  unfold freshFLocation
  apply fLocation_numeric_lt_temporaryBase
  exact freshOperand_mem_visible frame
    (view.column_mem (freshOperand frame.operands)
      (fresh_widthsAgree frame))

theorem proofFLocation_numeric_lt
    {value : parameters.NifsProof → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .nifsProof)) value) :
    (proofFLocation family frame view).numeric <
      temporaryBase frame := by
  unfold proofFLocation
  apply fLocation_numeric_lt_temporaryBase
  exact proofOperand_mem_visible frame
    (view.column_mem (proofOperand frame.operands)
      (proof_widthsAgree frame))

theorem proofKLocation_numeric_lt
    {value : parameters.NifsProof → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value) :
    (proofKLocation family frame view).numeric.c0 <
        temporaryBase frame
      ∧ (proofKLocation family frame view).numeric.c1 <
        temporaryBase frame := by
  unfold proofKLocation
  apply kLocation_numeric_lt_temporaryBase
  · exact proofOperand_mem_visible frame
      (view.c0_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))
  · exact proofOperand_mem_visible frame
      (view.c1_mem (proofOperand frame.operands)
        (proof_widthsAgree frame))

theorem outputFLocation_numeric_lt
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value) :
    (outputFLocation family frame view).numeric <
      temporaryBase frame := by
  unfold outputFLocation
  apply fLocation_numeric_lt_temporaryBase
  apply output_mem_visible frame
  simpa only [unaryOutput_ids] using
    view.column_mem (unaryOutput frame.outputs)
      (output_widthsAgree family frame)

theorem outputKLocation_numeric_lt
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value) :
    (outputKLocation family frame view).numeric.c0 <
        temporaryBase frame
      ∧ (outputKLocation family frame view).numeric.c1 <
        temporaryBase frame := by
  unfold outputKLocation
  apply kLocation_numeric_lt_temporaryBase
  · apply output_mem_visible frame
    simpa only [unaryOutput_ids] using
      view.c0_mem (unaryOutput frame.outputs)
        (output_widthsAgree family frame)
  · apply output_mem_visible frame
    simpa only [unaryOutput_ids] using
      view.c1_mem (unaryOutput frame.outputs)
        (output_widthsAgree family frame)

/-! ## Decoding equations -/

theorem runningF_decoded
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value)
    (assignment : ColumnId → Field)
    (running : parameters.Running)
    (decoded :
      (runningOperand frame.operands).Decodes family (.data .running)
        assignment running) :
    residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment (columnMap frame) assignment)
          (runningFLocation family frame view).carried) =
      value running := by
  calc
    residue
          (Nightstream.Implementation.R1CS.lcEval
            (numericAssignment (columnMap frame) assignment)
            (runningFLocation family frame view).carried) =
        (view.column (runningOperand frame.operands)
          (running_widthsAgree frame)).value assignment :=
      (runningFLocation family frame view).carried_value_eq assignment
    _ = value running :=
      view.value_eq_of_bundle_decodes family (.data .running)
        (runningOperand frame.operands) (running_widthsAgree frame)
        assignment running decoded

theorem runningK_decoded
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value)
    (assignment : ColumnId → Field)
    (running : parameters.Running)
    (decoded :
      (runningOperand frame.operands).Decodes family (.data .running)
        assignment running) :
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (runningKLocation family frame view).carried =
      value running := by
  calc
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (runningKLocation family frame view).carried =
        (view.columns (runningOperand frame.operands)
          (running_widthsAgree frame)).value assignment :=
      (runningKLocation family frame view).decodeCarried_eq assignment
    _ = value running :=
      view.value_eq_of_bundle_decodes family (.data .running)
        (runningOperand frame.operands) (running_widthsAgree frame)
        assignment running decoded

theorem freshF_decoded
    {value : parameters.Fresh → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .fresh)) value)
    (assignment : ColumnId → Field)
    (fresh : parameters.Fresh)
    (decoded :
      (freshOperand frame.operands).Decodes family (.data .fresh)
        assignment fresh) :
    residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment (columnMap frame) assignment)
          (freshFLocation family frame view).carried) =
      value fresh := by
  calc
    residue
          (Nightstream.Implementation.R1CS.lcEval
            (numericAssignment (columnMap frame) assignment)
            (freshFLocation family frame view).carried) =
        (view.column (freshOperand frame.operands)
          (fresh_widthsAgree frame)).value assignment :=
      (freshFLocation family frame view).carried_value_eq assignment
    _ = value fresh :=
      view.value_eq_of_bundle_decodes family (.data .fresh)
        (freshOperand frame.operands) (fresh_widthsAgree frame)
        assignment fresh decoded

theorem proofF_decoded
    {value : parameters.NifsProof → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .nifsProof)) value)
    (assignment : ColumnId → Field)
    (proof : parameters.NifsProof)
    (decoded :
      (proofOperand frame.operands).Decodes family (.data .nifsProof)
        assignment proof) :
    residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment (columnMap frame) assignment)
          (proofFLocation family frame view).carried) =
      value proof := by
  calc
    residue
          (Nightstream.Implementation.R1CS.lcEval
            (numericAssignment (columnMap frame) assignment)
            (proofFLocation family frame view).carried) =
        (view.column (proofOperand frame.operands)
          (proof_widthsAgree frame)).value assignment :=
      (proofFLocation family frame view).carried_value_eq assignment
    _ = value proof :=
      view.value_eq_of_bundle_decodes family (.data .nifsProof)
        (proofOperand frame.operands) (proof_widthsAgree frame)
        assignment proof decoded

theorem proofK_decoded
    {value : parameters.NifsProof → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .nifsProof)) value)
    (assignment : ColumnId → Field)
    (proof : parameters.NifsProof)
    (decoded :
      (proofOperand frame.operands).Decodes family (.data .nifsProof)
        assignment proof) :
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (proofKLocation family frame view).carried =
      value proof := by
  calc
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (proofKLocation family frame view).carried =
        (view.columns (proofOperand frame.operands)
          (proof_widthsAgree frame)).value assignment :=
      (proofKLocation family frame view).decodeCarried_eq assignment
    _ = value proof :=
      view.value_eq_of_bundle_decodes family (.data .nifsProof)
        (proofOperand frame.operands) (proof_widthsAgree frame)
        assignment proof decoded

theorem outputF_decoded
    {value : parameters.Running → Field}
    (view :
      PaperNifsCodecProjection.FView
        (family.codecFor (.data .running)) value)
    (assignment : ColumnId → Field)
    (output : parameters.Running)
    (decoded :
      (unaryOutput frame.outputs).Decodes family (.data .running)
        assignment output) :
    residue
        (Nightstream.Implementation.R1CS.lcEval
          (numericAssignment (columnMap frame) assignment)
          (outputFLocation family frame view).carried) =
      value output := by
  calc
    residue
          (Nightstream.Implementation.R1CS.lcEval
            (numericAssignment (columnMap frame) assignment)
            (outputFLocation family frame view).carried) =
        (view.column (unaryOutput frame.outputs)
          (output_widthsAgree family frame)).value assignment :=
      (outputFLocation family frame view).carried_value_eq assignment
    _ = value output :=
      view.value_eq_of_bundle_decodes family (.data .running)
        (unaryOutput frame.outputs) (output_widthsAgree family frame)
        assignment output decoded

theorem outputK_decoded
    {value : parameters.Running → Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        (family.codecFor (.data .running)) value)
    (assignment : ColumnId → Field)
    (output : parameters.Running)
    (decoded :
      (unaryOutput frame.outputs).Decodes family (.data .running)
        assignment output) :
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
        (numericAssignment (columnMap frame) assignment)
        (outputKLocation family frame view).carried =
      value output := by
  calc
    Nightstream.Implementation.R1CS.Canonical.KPointEquality.decoded
          (numericAssignment (columnMap frame) assignment)
          (outputKLocation family frame view).carried =
        (view.columns (unaryOutput frame.outputs)
          (output_widthsAgree family frame)).value assignment :=
      (outputKLocation family frame view).decodeCarried_eq assignment
    _ = value output :=
      view.value_eq_of_bundle_decodes family (.data .running)
        (unaryOutput frame.outputs) (output_widthsAgree family frame)
        assignment output decoded

end Frame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
