import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame

/-!
Contract: erase setup-dependent semantic payloads from every generic carrier
location used by the current benchmark NIFS rows.

Assurance tier: model-level.

Owns: running-input, fresh-input, and running-output numeric-location
stability from exact namespace, bundle, and codec-index equality.

Does not own: selection of codec indices, emitted rows, semantic refinement,
Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4CarrierLocations

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.R1CS.ProjectionProgram

private theorem kColumns_eq
    (left right : KColumns)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- A running-input base-field location is fixed by the complete namespace,
the running bundle, and its codec index. -/
theorem runningFNumeric_eq_of_ids_and_index
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue : leftParameters.Running → Field}
    {rightValue : rightParameters.Running → Field}
    (leftView :
      PaperNifsCodecProjection.FView
        (leftFamily.codecFor (.data .running)) leftValue)
    (rightView :
      PaperNifsCodecProjection.FView
        (rightFamily.codecFor (.data .running)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (runningIdsEqual :
      (PaperNifsCallFrame.runningOperand leftFrame.operands).ids =
        (PaperNifsCallFrame.runningOperand rightFrame.operands).ids)
    (indexEqual : leftView.index.val = rightView.index.val) :
    (ConcreteNifsCarrierFrame.runningFLocation
      leftFamily leftFrame leftView).numeric =
    (ConcreteNifsCarrierFrame.runningFLocation
      rightFamily rightFrame rightView).numeric := by
  unfold ConcreteNifsCarrierFrame.runningFLocation
    PaperNifsGlobalColumnMap.fLocation
  apply PaperNifsGlobalColumnMap.locate_source_congr
  · exact orderedEqual
  · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
      (PaperNifsCallFrame.runningOperand leftFrame.operands)
      (PaperNifsCallFrame.runningOperand rightFrame.operands)
      (PaperNifsCallFrame.running_widthsAgree leftFrame)
      (PaperNifsCallFrame.running_widthsAgree rightFrame)
      leftView.index rightView.index runningIdsEqual indexEqual

/-- A fresh-input base-field location is fixed by the complete namespace,
the fresh bundle, and its codec index. -/
theorem freshFNumeric_eq_of_ids_and_index
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue : leftParameters.Fresh → Field}
    {rightValue : rightParameters.Fresh → Field}
    (leftView :
      PaperNifsCodecProjection.FView
        (leftFamily.codecFor (.data .fresh)) leftValue)
    (rightView :
      PaperNifsCodecProjection.FView
        (rightFamily.codecFor (.data .fresh)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (freshIdsEqual :
      (PaperNifsCallFrame.freshOperand leftFrame.operands).ids =
        (PaperNifsCallFrame.freshOperand rightFrame.operands).ids)
    (indexEqual : leftView.index.val = rightView.index.val) :
    (ConcreteNifsCarrierFrame.freshFLocation
      leftFamily leftFrame leftView).numeric =
    (ConcreteNifsCarrierFrame.freshFLocation
      rightFamily rightFrame rightView).numeric := by
  unfold ConcreteNifsCarrierFrame.freshFLocation
    PaperNifsGlobalColumnMap.fLocation
  apply PaperNifsGlobalColumnMap.locate_source_congr
  · exact orderedEqual
  · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
      (PaperNifsCallFrame.freshOperand leftFrame.operands)
      (PaperNifsCallFrame.freshOperand rightFrame.operands)
      (PaperNifsCallFrame.fresh_widthsAgree leftFrame)
      (PaperNifsCallFrame.fresh_widthsAgree rightFrame)
      leftView.index rightView.index freshIdsEqual indexEqual

/-- A running-output base-field location is fixed by the complete namespace,
the output bundle, and its codec index. -/
theorem outputFNumeric_eq_of_ids_and_index
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue : leftParameters.Running → Field}
    {rightValue : rightParameters.Running → Field}
    (leftView :
      PaperNifsCodecProjection.FView
        (leftFamily.codecFor (.data .running)) leftValue)
    (rightView :
      PaperNifsCodecProjection.FView
        (rightFamily.codecFor (.data .running)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (outputIdsEqual :
      (unaryOutput leftFrame.outputs).ids =
        (unaryOutput rightFrame.outputs).ids)
    (indexEqual : leftView.index.val = rightView.index.val) :
    (ConcreteNifsCarrierFrame.outputFLocation
      leftFamily leftFrame leftView).numeric =
    (ConcreteNifsCarrierFrame.outputFLocation
      rightFamily rightFrame rightView).numeric := by
  unfold ConcreteNifsCarrierFrame.outputFLocation
    PaperNifsGlobalColumnMap.fLocation
  apply PaperNifsGlobalColumnMap.locate_source_congr
  · exact orderedEqual
  · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
      (unaryOutput leftFrame.outputs)
      (unaryOutput rightFrame.outputs)
      (ConcreteNifsCarrierFrame.output_widthsAgree leftFamily leftFrame)
      (ConcreteNifsCarrierFrame.output_widthsAgree rightFamily rightFrame)
      leftView.index rightView.index outputIdsEqual indexEqual

/-- A running-output extension-field location is fixed by the complete
namespace, the output bundle, and its two codec indices. -/
theorem outputKNumeric_eq_of_ids_and_indices
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftValue :
      leftParameters.Running → Nightstream.SuperNeo.Concrete.K}
    {rightValue :
      rightParameters.Running → Nightstream.SuperNeo.Concrete.K}
    (leftView :
      PaperNifsCodecProjection.KView
        (leftFamily.codecFor (.data .running)) leftValue)
    (rightView :
      PaperNifsCodecProjection.KView
        (rightFamily.codecFor (.data .running)) rightValue)
    (orderedEqual :
      PaperNifsGlobalColumnMap.orderedIds leftFrame =
        PaperNifsGlobalColumnMap.orderedIds rightFrame)
    (outputIdsEqual :
      (unaryOutput leftFrame.outputs).ids =
        (unaryOutput rightFrame.outputs).ids)
    (c0IndexEqual : leftView.c0Index.val = rightView.c0Index.val)
    (c1IndexEqual : leftView.c1Index.val = rightView.c1Index.val) :
    (ConcreteNifsCarrierFrame.outputKLocation
      leftFamily leftFrame leftView).numeric =
    (ConcreteNifsCarrierFrame.outputKLocation
      rightFamily rightFrame rightView).numeric := by
  unfold ConcreteNifsCarrierFrame.outputKLocation
    PaperNifsGlobalColumnMap.kLocation
  apply kColumns_eq
  · apply PaperNifsGlobalColumnMap.locate_source_congr
    · exact orderedEqual
    · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
        (unaryOutput leftFrame.outputs)
        (unaryOutput rightFrame.outputs)
        (ConcreteNifsCarrierFrame.output_widthsAgree leftFamily leftFrame)
        (ConcreteNifsCarrierFrame.output_widthsAgree rightFamily rightFrame)
        leftView.c0Index rightView.c0Index outputIdsEqual c0IndexEqual
  · apply PaperNifsGlobalColumnMap.locate_source_congr
    · exact orderedEqual
    · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
        (unaryOutput leftFrame.outputs)
        (unaryOutput rightFrame.outputs)
        (ConcreteNifsCarrierFrame.output_widthsAgree leftFamily leftFrame)
        (ConcreteNifsCarrierFrame.output_widthsAgree rightFamily rightFrame)
        leftView.c1Index rightView.c1Index outputIdsEqual c1IndexEqual

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4CarrierLocations
