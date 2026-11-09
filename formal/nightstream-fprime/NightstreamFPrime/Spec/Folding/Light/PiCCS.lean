import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

/-! SuperNeo v1.2 Section 7.3 with Cyclo (ePrint 2026/359) contribution 2: the norm terms
cover the fresh sources only. A running source enters the norm batch as zero, while its
evaluation claims keep its own assignment, so the `PaperJoint` identities apply unchanged. -/

namespace NightstreamFPrime.Spec.Folding.Light.PiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

universe uExtension

variable {Extension : Type uExtension} {shape : Shape} {columns : Nat}

def normAssignments (data : UnifiedInputs Extension shape columns) :
    Fin shape.sourceCount -> PaperLinearAlgebra.Assignment F columns :=
  fun source =>
    if source.val < shape.freshCount then data.assignments source else fun _ => 0

def normView (data : UnifiedInputs Extension shape columns) :
    UnifiedInputs Extension shape columns :=
  { data with assignments := normAssignments data }

def independentInputs (data : UnifiedInputs Extension shape columns) :
    ConcreteJointData.IndependentInputs Extension shape columns :=
  { data.toIndependentInputs with norm := (normView data).normBatch }

def protocolData (baseOps : InterpolationOps F) (lift : F -> Extension)
    (data : UnifiedInputs Extension shape columns) :
    ProtocolPolynomial.Data Extension shape :=
  { ProtocolDataRefinement.toProtocolData baseOps lift data with
    sourceAssignments :=
      (ProtocolDataRefinement.toProtocolData baseOps lift (normView data)).sourceAssignments }

theorem protocolData_toJointData_eq
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (data : UnifiedInputs Extension shape columns) :
    (protocolData baseOps lift data).toJointData extensionOps =
      (independentInputs data).toJointData baseOps lift := by
  apply SignedJointIdentity.JointData.ext
  · funext source
    exact ProtocolDataRefinement.ccsTable_eq baseOps extensionOps lift laws data source
  · funext source
    exact ProtocolDataRefinement.normTable_eq baseOps extensionOps lift laws
      (normView data) source
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

def FreshNormBounded (data : UnifiedInputs Extension shape columns) : Prop :=
  ∀ source : Fin shape.sourceCount, source.val < shape.freshCount ->
    ∀ column, centeredMagnitude (data.assignments source column) < 2

def SemanticTruth
    (data : UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop :=
  data.freshBatch.AllConstraintsSatisfied baseOps ∧
    FreshNormBounded data ∧
    PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift data.padData ∧
    MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift data.matrixData

theorem normView_bounded_iff (data : UnifiedInputs Extension shape columns) :
    (normView data).AllAssignmentsStrictNormBounded ↔ FreshNormBounded data := by
  constructor
  · intro bounded source isFresh column
    simpa [normView, normAssignments, isFresh] using bounded source column
  · intro bounded source column
    by_cases isFresh : source.val < shape.freshCount
    · simpa [normView, normAssignments, isFresh] using bounded source isFresh column
    · simp [normView, normAssignments, isFresh, centeredMagnitude]

theorem independentInputs_semanticTruth_iff
    (data : UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) :
    (independentInputs data).SemanticTruth baseOps extensionOps lift ↔
      SemanticTruth data baseOps extensionOps lift := by
  unfold ConcreteJointData.IndependentInputs.SemanticTruth SemanticTruth
  rw [← normView_bounded_iff,
    ← (normView data).normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded]
  exact Iff.rfl

theorem jointTableTruth
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F -> Extension)
    (laws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (data : UnifiedInputs Extension shape columns)
    (truth : SemanticTruth data baseOps extensionOps lift) :
    (TableResidualData.toTableObligations extensionOps
      (SignedCoefficientObject.toTableResidualData extensionOps
        ((protocolData baseOps lift data).toJointData extensionOps))).AllHold := by
  rw [protocolData_toJointData_eq baseOps extensionOps lift laws data]
  exact (ConcreteJointData.jointTableTruth_iff_semanticTruth baseOps baseZero noZeroDivisors
    extensionOps extensionLaws lift laws.toZeroReflectingLift (independentInputs data)).mpr
      ((independentInputs_semanticTruth_iff data baseOps extensionOps lift).mpr truth)

end NightstreamFPrime.Spec.Folding.Light.PiCCS
