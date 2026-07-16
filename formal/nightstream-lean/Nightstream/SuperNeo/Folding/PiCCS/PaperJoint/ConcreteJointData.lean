import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial

/-!
Independent construction of the paper-level joint `Pi_CCS` data.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: semantic residual construction before alpha/gamma compression.
Constraint family: CCS, strict norm, and carried prior-evaluation obligations.

Owns: one explicit bundle of independently defined CCS matrices/assignments,
typed strict-norm assignments, and carried matrix-image data; zero-reflecting
placement of base residual tables into the extension carrier; construction of
the sole `SignedJointIdentity.JointData`; and exact equivalence between its
table truth and the three independent semantic obligation families.

Does not own: proof that coefficient-expanded carried matrices implement the
same production ring matrices as the CCS structure, external row/bit order,
the concrete base-to-extension homomorphism, target-convention approval,
expected-round degree bounds, SplitNc, Fiat--Shamir, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: callers provide mathematical matrices, assignments,
claimed prior evaluations, and a base-to-extension function. They do not
provide residual tables, a `JointData`, a semantic truth proposition, an
evaluator, or a per-family equivalence. `IndependentInputs` deliberately does
not claim that its three assignment families are the same paper witnesses;
only `UnifiedSources.UnifiedInputs` may establish that connectivity and
project into this reusable family layer. The only lift premise used for zero
truth is explicit zero reflection; stronger homomorphism and production-layout
claims remain separate refinement obligations.

| Protocol | Phase | Constructed family | Exact theorem |
|---|---|---|---|
| `Pi_CCS` | CCS | lifted explicit sparse-CCS residual tables | zero iff every fresh CCS assignment satisfies the explicit structure |
| `Pi_CCS` | norm | lifted canonical cubic tables | zero iff every typed source satisfies strict centered norm `< 2` |
| `Pi_CCS` | carried evaluation | explicit coefficient-matrix image tables | residual zero iff every claimed prior evaluation is derived correctly |
| `Pi_CCS` | joint assembly | `toJointData` | no caller-selected residual table enters `Q` |
| `Pi_CCS` | semantic closure | `jointTableTruth_iff_semanticTruth` | joint table truth iff CCS and norm and carried truth |
| `Pi_CCS` | executable reduction | `checkJoint_implies_semanticTruth_or_badEvent` | accepted canonical SumCheck yields semantic truth or a named mixing/round event |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck

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
  carried :
    CarriedEvaluationResidual.EvaluationData F Extension shape columns

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
  priorPoint := data.carried.priorPoint
  carriedImage := fun coordinate =>
    CarriedEvaluationResidual.imageTable baseOps lift data.carried coordinate
  claimedCoefficient := data.carried.claimedCoefficient

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
    CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps lift
      data.carried

private theorem mem_canonicalFinIndices
    {count : Nat}
    (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  simp [canonicalFinIndices]

private theorem mem_canonicalCarriedCoordinates
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    coordinate ∈ canonicalCarriedCoordinates shape := by
  rcases coordinate with ⟨running, matrix, coefficient⟩
  simp [canonicalCarriedCoordinates, canonicalFinIndices]

/-- Expand the proposition-list serialization back to its typed family owners.
This theorem is generic accounting; it does not establish any leaf semantics. -/
theorem tableObligations_allHold_iff_pointwise
    {Field : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) :
    (data.toTableObligations ops).AllHold <->
      (forall source, (data.ccs source).AllEntriesZero ops) /\
      (forall source, (data.norm source).AllEntriesZero ops) /\
      forall coordinate, data.carriedEvaluation coordinate = ops.zero := by
  unfold Obligations.AllHold TableResidualData.toTableObligations
  constructor
  · rintro ⟨ccs, norm, carried⟩
    refine ⟨?_, ?_, ?_⟩
    · intro source
      exact ccs ((data.ccs source).AllEntriesZero ops)
        (List.mem_map.mpr
          ⟨source, mem_canonicalFinIndices source, rfl⟩)
    · intro source
      exact norm ((data.norm source).AllEntriesZero ops)
        (List.mem_map.mpr
          ⟨source, mem_canonicalFinIndices source, rfl⟩)
    · intro coordinate
      exact carried (data.carriedEvaluation coordinate = ops.zero)
        (List.mem_map.mpr
          ⟨coordinate, mem_canonicalCarriedCoordinates coordinate, rfl⟩)
  · rintro ⟨ccs, norm, carried⟩
    refine ⟨?_, ?_, ?_⟩
    · intro obligation member
      rcases List.mem_map.mp member with ⟨source, _, rfl⟩
      exact ccs source
    · intro obligation member
      rcases List.mem_map.mp member with ⟨source, _, rfl⟩
      exact norm source
    · intro obligation member
      rcases List.mem_map.mp member with ⟨coordinate, _, rfl⟩
      exact carried coordinate

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
  · rintro ⟨ccsZero, normZero, carriedZero⟩
    refine ⟨?_, ?_, ?_⟩
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
      apply (CarriedEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
        baseOps extensionOps extensionLaws lift data.carried coordinate).mp
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        CarriedEvaluationResidual.residual,
        CarriedEvaluationResidual.computedCoefficient]
        using carriedZero coordinate
  · rintro ⟨ccsTruth, normTruth, carriedTruth⟩
    refine ⟨?_, ?_, ?_⟩
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
    · intro coordinate
      have residualZero :=
        (CarriedEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds
          baseOps extensionOps extensionLaws lift data.carried coordinate).mpr
            (carriedTruth coordinate)
      simpa [SignedCoefficientObject.toTableResidualData, IndependentInputs.toJointData,
        CarriedEvaluationResidual.residual,
        CarriedEvaluationResidual.computedCoefficient]
        using residualZero

/-- Unsampled joint coefficient truth is no longer conditional on a
caller-assembled `JointData`: it is exactly the three independent semantic
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
constructed semantic data. Degree/root probability and production SplitNc
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData
