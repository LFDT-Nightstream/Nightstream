import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckInitial

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

universe uExtension

def liftTable
    {Base : Type}
    {Extension : Type uExtension}
    (lift : Base -> Extension) :
    {variables : Nat} -> BooleanTable Base variables ->
      BooleanTable Extension variables
  | 0, .leaf value => .leaf (lift value)
  | _ + 1, .branch low high =>
      .branch (liftTable lift low) (liftTable lift high)

theorem liftTable_entries_eq
    {Base : Type}
    {Extension : Type uExtension}
    (lift : Base -> Extension)
    {variables : Nat}
    (table : BooleanTable Base variables) :
    (liftTable lift table).entries = table.entries.map lift := by
  induction table with
  | leaf => rfl
  | branch low high lowInduction highInduction =>
      simp [liftTable, BooleanTable.entries, lowInduction, highInduction,
        List.map_append]

structure ZeroReflectingLift
    (baseOps : InterpolationOps F)
    {Extension : Type uExtension}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop where
  zero_iff : forall value,
    lift value = extensionOps.zero <-> value = baseOps.zero

theorem liftTable_allEntriesZero_iff
    (baseOps : InterpolationOps F)
    {Extension : Type uExtension}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (liftLaw : ZeroReflectingLift baseOps extensionOps lift)
    {variables : Nat}
    (table : BooleanTable F variables) :
    (liftTable lift table).AllEntriesZero extensionOps <->
      table.AllEntriesZero baseOps := by
  unfold BooleanTable.AllEntriesZero
  rw [liftTable_entries_eq]
  constructor
  · intro liftedZero value member
    apply (liftLaw.zero_iff value).mp
    exact liftedZero (lift value) (List.mem_map.mpr ⟨value, member, rfl⟩)
  · intro baseZero value member
    rcases List.mem_map.mp member with ⟨baseValue, baseMember, rfl⟩
    exact (liftLaw.zero_iff baseValue).mpr (baseZero baseValue baseMember)

structure IndependentInputs
    (Extension : Type uExtension)
    (shape : Shape)
    (columns : Nat) where
  ccs : CCSResidualTable.FreshBatch F shape columns
  norm : NormResidualTable.SourceBatch shape
  priorPoint : CubePoint Extension shape.cubeVariables
  runningAssignments : Fin shape.runningCount ->
    PaperLinearAlgebra.Assignment F columns
  padCoefficientMatrices : Fin shape.coefficientCount ->
    PaperLinearAlgebra.BooleanMatrix F shape.cubeVariables columns
  matrixCoefficientMatrices :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      PaperLinearAlgebra.BooleanMatrix F shape.cubeVariables columns
  claimedPadCoefficient : PadCoordinate shape -> Extension
  claimedMatrixCoefficient : MatrixCoordinate shape -> Extension

def IndependentInputs.padData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : IndependentInputs Extension shape columns) :
    PadEvaluationResidual.EvaluationData F Extension shape columns where
  priorPoint := data.priorPoint
  assignments := data.runningAssignments
  coefficientMatrices := data.padCoefficientMatrices
  claimedCoefficient := data.claimedPadCoefficient

def IndependentInputs.matrixData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : IndependentInputs Extension shape columns) :
    MatrixEvaluationResidual.EvaluationData F Extension shape columns where
  priorPoint := data.priorPoint
  assignments := data.runningAssignments
  coefficientMatrices := data.matrixCoefficientMatrices
  claimedCoefficient := data.claimedMatrixCoefficient

def IndependentInputs.toJointData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : IndependentInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (lift : F -> Extension) :
    SignedJointIdentity.JointData Extension shape where
  ccs := fun source =>
    liftTable lift (data.ccs.residualTables baseOps source)
  norm := fun source =>
    liftTable lift (data.norm.residualTables source)
  priorPoint := data.priorPoint
  padImage := fun coordinate =>
    PadEvaluationResidual.imageTable baseOps lift data.padData coordinate
  matrixImage := fun coordinate =>
    MatrixEvaluationResidual.imageTable baseOps lift data.matrixData coordinate
  claimedPadCoefficient := data.claimedPadCoefficient
  claimedMatrixCoefficient := data.claimedMatrixCoefficient

def IndependentInputs.SemanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : IndependentInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop :=
  data.ccs.AllConstraintsSatisfied baseOps /\
    data.norm.AllStrictNormBounded /\
    PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift data.padData /\
    MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift
      data.matrixData

private theorem mem_canonicalPadCoordinates
    {shape : Shape}
    (coordinate : PadCoordinate shape) :
    coordinate ∈ canonicalPadCoordinates shape := by
  rcases coordinate with ⟨running, coefficient⟩
  simp [canonicalPadCoordinates, canonicalFinIndices]

private theorem mem_canonicalMatrixCoordinates
    {shape : Shape}
    (coordinate : MatrixCoordinate shape) :
    coordinate ∈ canonicalMatrixCoordinates shape := by
  rcases coordinate with ⟨running, matrix, coefficient⟩
  simp [canonicalMatrixCoordinates, canonicalFinIndices]

theorem tableObligations_allHold_iff_pointwise
    {Field : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) :
    (data.toTableObligations ops).AllHold <->
      (forall coordinate, data.padEvaluation coordinate = ops.zero) /\
      (forall coordinate, data.matrixEvaluation coordinate = ops.zero) /\
      (forall source, (data.ccs source).AllEntriesZero ops) /\
      forall source, (data.norm source).AllEntriesZero ops := by
  unfold Obligations.AllHold TableResidualData.toTableObligations
  constructor
  · rintro ⟨pad, matrix, ccs, norm⟩
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro coordinate
      exact pad (data.padEvaluation coordinate = ops.zero)
        (List.mem_map.mpr
          ⟨coordinate, mem_canonicalPadCoordinates coordinate, rfl⟩)
    · intro coordinate
      exact matrix (data.matrixEvaluation coordinate = ops.zero)
        (List.mem_map.mpr
          ⟨coordinate, mem_canonicalMatrixCoordinates coordinate, rfl⟩)
    · intro source
      exact ccs ((data.ccs source).AllEntriesZero ops)
        (List.mem_map.mpr
          ⟨source, mem_canonicalFinIndices source, rfl⟩)
    · intro source
      exact norm ((data.norm source).AllEntriesZero ops)
        (List.mem_map.mpr
          ⟨source, mem_canonicalFinIndices source, rfl⟩)
  · rintro ⟨pad, matrix, ccs, norm⟩
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro obligation member
      rcases List.mem_map.mp member with ⟨coordinate, _, rfl⟩
      exact pad coordinate
    · intro obligation member
      rcases List.mem_map.mp member with ⟨coordinate, _, rfl⟩
      exact matrix coordinate
    · intro obligation member
      rcases List.mem_map.mp member with ⟨source, _, rfl⟩
      exact ccs source
    · intro obligation member
      rcases List.mem_map.mp member with ⟨source, _, rfl⟩
      exact norm source

theorem jointTableTruth_iff_semanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F -> Extension)
    (liftLaw : ZeroReflectingLift baseOps extensionOps lift)
    (data : IndependentInputs Extension shape columns) :
    (TableResidualData.toTableObligations extensionOps
        (SignedCoefficientObject.toTableResidualData extensionOps
          (data.toJointData baseOps lift))).AllHold <->
      data.SemanticTruth baseOps extensionOps lift := by
  rw [tableObligations_allHold_iff_pointwise]
  constructor
  · rintro ⟨padZero, matrixZero, ccsZero, normZero⟩
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro source
      apply (CCSResidualTable.residualTable_allEntriesZero_iff_constraintSatisfied
        baseOps data.ccs.system (data.ccs.assignments source)).mp
      apply (liftTable_allEntriesZero_iff baseOps extensionOps lift liftLaw
        (data.ccs.residualTables baseOps source)).mp
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData]
        using ccsZero source
    · intro source
      apply (NormResidualTable.residualTable_allEntriesZero_iff_strictNormBounded
        baseOps baseZero noZeroDivisors (data.norm.assignments source)).mp
      apply (liftTable_allEntriesZero_iff baseOps extensionOps lift liftLaw
        (data.norm.residualTables source)).mp
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData]
        using normZero source
    · intro coordinate
      apply (PadEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
        baseOps extensionOps extensionLaws lift data.padData coordinate).mp
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        IndependentInputs.padData, PadEvaluationResidual.residual,
        PadEvaluationResidual.computedCoefficient]
        using padZero coordinate
    · intro coordinate
      apply (MatrixEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
        baseOps extensionOps extensionLaws lift data.matrixData coordinate).mp
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        IndependentInputs.matrixData, MatrixEvaluationResidual.residual,
        MatrixEvaluationResidual.computedCoefficient]
        using matrixZero coordinate
  · rintro ⟨ccsTruth, normTruth, padTruth, matrixTruth⟩
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro coordinate
      have residualZero :=
        (PadEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
          baseOps extensionOps extensionLaws lift data.padData coordinate).mpr
            (padTruth coordinate)
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        IndependentInputs.padData, PadEvaluationResidual.residual,
        PadEvaluationResidual.computedCoefficient]
        using residualZero
    · intro coordinate
      have residualZero :=
        (MatrixEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
          baseOps extensionOps extensionLaws lift data.matrixData coordinate).mpr
            (matrixTruth coordinate)
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        IndependentInputs.matrixData, MatrixEvaluationResidual.residual,
        MatrixEvaluationResidual.computedCoefficient]
        using residualZero
    · intro source
      have baseTableZero :
          (data.ccs.residualTables baseOps source).AllEntriesZero baseOps :=
        (CCSResidualTable.residualTable_allEntriesZero_iff_constraintSatisfied
          baseOps data.ccs.system (data.ccs.assignments source)).mpr
            (ccsTruth source)
      have liftedZero :=
        (liftTable_allEntriesZero_iff baseOps extensionOps lift liftLaw
          (data.ccs.residualTables baseOps source)).mpr baseTableZero
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData]
        using liftedZero
    · intro source
      have baseTableZero :
          (data.norm.residualTables source).AllEntriesZero baseOps :=
        (NormResidualTable.residualTable_allEntriesZero_iff_strictNormBounded
          baseOps baseZero noZeroDivisors (data.norm.assignments source)).mpr
            (normTruth source)
      have liftedZero :=
        (liftTable_allEntriesZero_iff baseOps extensionOps lift liftLaw
          (data.norm.residualTables source)).mpr baseTableZero
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData]
        using liftedZero

theorem coefficientTruth_iff_semanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaw : ZeroReflectingLift baseOps extensionOps lift)
    (data : IndependentInputs Extension shape columns) :
    SignedCoefficientObject.CoefficientTruth extensionOps
        (data.toJointData baseOps lift) <->
      data.SemanticTruth baseOps extensionOps lift := by
  rw [SignedCoefficientObject.coefficientTruth_iff_tableObligations
    extensionOps extensionZeroLaws]
  exact jointTableTruth_iff_semanticTruth baseOps baseZero noZeroDivisors
    extensionOps extensionLaws lift liftLaw data

theorem checkJoint_implies_semanticTruth_or_badEvent
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaw : ZeroReflectingLift baseOps extensionOps lift)
    (data : IndependentInputs Extension shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (maxDegree challengeSetSize : Nat)
    (roundPoint : CubePoint Extension shape.cubeVariables)
    (certificate : SumCheck.Finite.Certificate Extension)
    (checked : SumCheckInitial.checkJoint extensionOps
      (data.toJointData baseOps lift) alpha gamma maxDegree
      roundPoint certificate = true) :
    data.SemanticTruth baseOps extensionOps lift \/
      SignedCoefficientObject.MixingRoot extensionOps
        (data.toJointData baseOps lift) alpha gamma \/
      exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance extensionOps
            (data.toJointData baseOps lift) alpha gamma maxDegree
            challengeSetSize roundPoint.coordinates
            (SumCheckTruthPath.verifierTerminal extensionOps
              (data.toJointData baseOps lift) alpha gamma
              roundPoint.coordinates)
            certificate
            (SumCheckInitial.canonicalExpected extensionOps
              (data.toJointData baseOps lift) alpha gamma
              roundPoint.coordinates))
          round := by
  rcases SumCheckInitial.checkJoint_implies_tableObligations_or_mixingRoot_or_badChallenge
      extensionOps extensionLaws extensionZeroLaws
      (data.toJointData baseOps lift) alpha gamma maxDegree
      challengeSetSize roundPoint certificate checked with truth | bad
  · exact Or.inl <|
      (jointTableTruth_iff_semanticTruth baseOps baseZero noZeroDivisors
        extensionOps extensionLaws lift liftLaw data).mp truth
  · exact Or.inr bad

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData
