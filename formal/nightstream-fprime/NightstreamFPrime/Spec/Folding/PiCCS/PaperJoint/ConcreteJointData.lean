import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormResidualTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckInitial

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ConcreteJointData.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Independent construction of the paper-level joint `Pi_CCS` data.

Protocol: SuperNeo v1.1 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: semantic residual construction before alpha/gamma compression.
Constraint family: CCS, strict norm, Pad evaluation, and matrix evaluation.

Owns: one explicit bundle of independently defined CCS matrices/assignments,
typed strict-norm assignments, and separate Pad/matrix image data; zero-reflecting
placement of base residual tables into the extension carrier; construction of
the sole `SignedJointIdentity.JointData`; and exact equivalence between its
table truth and the four independent semantic obligation families.

Does not own: proof that coefficient-expanded carried matrices implement the
same production ring matrices as the CCS structure, external row/bit order,
the concrete base-to-extension homomorphism, target-convention approval,
expected-round degree bounds, Fiat--Shamir, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: callers provide mathematical matrices, assignments,
claimed prior evaluations, and a base-to-extension function. The Pad and
matrix families share one prior point and one running-assignment family by
construction. Callers do not
provide residual tables, a `JointData`, a semantic truth proposition, an
evaluator, or a per-family equivalence. `IndependentInputs` deliberately does
not claim that its CCS, norm, and running assignment families are the same
paper witnesses;
only `UnifiedSources.UnifiedInputs` may establish that connectivity and
project into this reusable family layer. The only lift premise used for zero
truth is explicit zero reflection; stronger homomorphism and production-layout
claims remain separate refinement obligations.

| Protocol | Phase | Constructed family | Exact theorem |
|---|---|---|---|
| `Pi_CCS` | CCS | lifted explicit sparse-CCS residual tables | zero iff every fresh CCS assignment satisfies the explicit structure |
| `Pi_CCS` | norm | lifted canonical cubic tables | zero iff every typed source satisfies strict centered norm `< 2` |
| `Pi_CCS` | Pad evaluation | verifier-owned Pad coefficient-image tables | residual zero iff every `Eval_K` claim is derived correctly |
| `Pi_CCS` | matrix evaluation | all CCS coefficient-image tables | residual zero iff every `Eval_A` claim is derived correctly |
| `Pi_CCS` | joint assembly | `toJointData` | no caller-selected residual table enters `Q` |
| `Pi_CCS` | semantic closure | `jointTableTruth_iff_semanticTruth` | joint table truth iff CCS, norm, Pad, and matrix truth |
| `Pi_CCS` | executable reduction | `checkJoint_implies_semanticTruth_or_badEvent` | accepted canonical SumCheck yields semantic truth or a named mixing/round event |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

universe uExtension

/-- Lift every leaf without changing the typed Boolean-domain shape or order. -/
def liftTable
    {Base : Type}
    {Extension : Type uExtension}
    (lift : Base -> Extension) :
    {variables : Nat} -> BooleanTable Base variables ->
      BooleanTable Extension variables
  | 0, .leaf value => .leaf (lift value)
  | _ + 1, .branch low high =>
      .branch (liftTable lift low) (liftTable lift high)

/-- Lifting preserves the sole canonical low/high leaf order. -/
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

/-- A lift may be used in semantic residual placement only when it preserves
and reflects zero. No additive or multiplicative homomorphism is hidden here. -/
structure ZeroReflectingLift
    (baseOps : InterpolationOps F)
    {Extension : Type uExtension}
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop where
  zero_iff : forall value,
    lift value = extensionOps.zero <-> value = baseOps.zero

/-- Zero reflection transports exact leaf truth across the carrier boundary. -/
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

/-- Independently assembled family inputs used by the residual lemmas.

This is not a paper-safe public witness model because the CCS, norm, and
carried assignments can differ. `UnifiedSources.UnifiedInputs` is the
authoritative source owner and derives this structure internally. -/
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

/-- Exact v1.1 Pad evaluation view. Its point and assignments come from the
single owners in `IndependentInputs`. -/
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

/-- Exact v1.1 CCS-matrix evaluation view. It shares the same point and
assignments as `padData` by construction. -/
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

/-- The sole joint-data constructor from independent mathematical sources. -/
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

/-- The independent, pre-compression mathematical truth owned by this bundle. -/
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

private theorem mem_canonicalFinIndices
    {count : Nat}
    (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  simp [canonicalFinIndices]

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

/-- Expand the proposition-list serialization back to its typed family owners.
This theorem is generic accounting; it does not establish any leaf semantics. -/
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

/-- Exact semantic closure for the independently constructed joint tables. -/
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

/-- Unsampled joint coefficient truth is no longer conditional on a
caller-assembled `JointData`: it is exactly the four independent semantic
families for `IndependentInputs`. -/
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

/-- The strongest current one-joint executable consequence over independently
constructed semantic data. Degree/root probability and production layout
refinement remain named later boundaries. -/
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
