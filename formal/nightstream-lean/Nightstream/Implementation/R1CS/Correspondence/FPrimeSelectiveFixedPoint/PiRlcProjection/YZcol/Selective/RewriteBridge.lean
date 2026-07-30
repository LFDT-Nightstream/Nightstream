import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard0
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard1
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard3
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard4

/-!
Executable coefficient bridge for the selectively emitted fixed-point
`y_zcol` projection rows.

Owns: fail-closed composition of the sharded all-port coefficient checks,
the retained-row coefficient check, explicit product-sum step semantics, and
transport from active selected-row satisfaction.

Does not own: source-schedule grouping, trace-eliminated assignments,
projection soundness, selector construction, protocol authority, security
events, or permission to remove rows.

Emits constraints: no.

Assurance tier: artifact-checked coefficient equality for this bounded slice,
followed by derived field semantics.

| Child path | Mathematical obligation | Authority class | Artifact owner | Lean owner |
|---|---|---|---|---|
| `coefficients.shards` | every rewrite port equals its independently expanded source form | checked | generated matrix rows + source provenance | `rewriteCoefficientsExact` |
| `coefficients.retained` | each retained A/B/C port equals its source form | checked | generated matrix rows + source provenance | `retainedCoefficientsExact` |
| `semantics.rewrite` | an active satisfied compact row implies its explicit recurrence | derived | decoded selected rows | `rewritePairStepHolds` |
| `semantics.derived_witness` | each derived column obeys the exact zero-base Rust witness recurrence | checked + derived | generated derived registry | `derivedStepHolds_witnessRecurrence` |
| `semantics.retained` | an active satisfied retained row implies exact A·B=C | derived | decoded selected rows | `retainedPairHolds` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

set_option maxRecDepth 100000 in
theorem rewritePairRowsExact :
    rewritePairs.map Prod.fst = Materialized.Artifact.rewriteRows := by
  unfold rewritePairs
  apply List.map_fst_zip
  native_decide

set_option maxRecDepth 100000 in
theorem rewritePairStepsExact :
    rewritePairs.map Prod.snd = decodedRewriteSteps := by
  unfold rewritePairs
  apply List.map_snd_zip
  native_decide

set_option maxRecDepth 100000 in
theorem retainedPairRowsExact :
    retainedPairs.map Prod.fst = Materialized.Artifact.retainedRows := by
  unfold retainedPairs
  apply List.map_fst_zip
  native_decide

set_option maxRecDepth 100000 in
theorem retainedPairStepsExact :
    retainedPairs.map Prod.snd = decodedRetainedSteps := by
  unfold retainedPairs
  apply List.map_snd_zip
  native_decide

/- Artifact fact: independently expanded source/compiler forms equal every
decoded compact port stream. Each branch below comes from one bounded,
proof-free coefficient certificate. -/
theorem rewriteCoefficientsExact :
    ∀ pair ∈ rewritePairs,
      RewriteCoefficientsMatch pair.1 pair.2 := by
  intro pair member
  rw [← Coefficients.rewriteCoefficientChunksExact] at member
  simp only [List.mem_append] at member
  rcases member with member | member
  · exact Coefficients.Shard0.coefficientsExact pair member
  rcases member with member | member
  · exact Coefficients.Shard1.coefficientsExact pair member
  rcases member with member | member
  · exact Coefficients.Shard2.coefficientsExact pair member
  rcases member with member | member
  · exact Coefficients.Shard3.coefficientsExact pair member
  · exact Coefficients.Shard4.coefficientsExact pair member

private def retainedCoefficientCheck : Bool :=
  Coefficients.coefficientMatchShapesCheck
    Coefficients.retainedCoefficientData

set_option maxRecDepth 100000 in
private theorem retainedCoefficientCheck_true :
    retainedCoefficientCheck = true := by
  native_decide

theorem retainedCoefficientsExact :
    ∀ pair ∈ retainedPairs,
      RetainedCoefficientsMatch pair.1 pair.2 := by
  intro pair member
  have shapeMember :
      Coefficients.retainedPairCoefficientShape pair ∈
        Coefficients.retainedCoefficientData :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have allChecked :
      Coefficients.retainedCoefficientData.all
          Coefficients.coefficientMatchShapeCheck = true := by
    simpa only [retainedCoefficientCheck,
      Coefficients.coefficientMatchShapesCheck] using
      retainedCoefficientCheck_true
  exact Coefficients.retainedCoefficientsMatch_of_shape_check_true
    ((List.all_eq_true.mp allChecked) _ shapeMember)

def sourceValue (assignment : Nat → Nat)
    (linear : DecodedSourceLinearCombination) : F :=
  fieldResidue
    (lcEval (compilerAssignment assignment) linear.programTerms)

def derivedValue (assignment : Nat → Nat) (slot : DecodedDerivedSlot) : F :=
  fieldResidue
    (lcEval assignment (slotExpansionTerms slot.start slot.width))

def outputValue (assignment : Nat → Nat) : DecodedRewriteOutput → F
  | .source value => sourceValue assignment value
  | .derivedProductSum slot => derivedValue assignment slot

def previousValue (assignment : Nat → Nat) :
    Option DecodedDerivedSlot → F
  | none => 0
  | some slot => derivedValue assignment slot

def factorValue (assignment : Nat → Nat)
    (factor : DecodedProductFactor) : F :=
  fieldResidue factor.coefficient *
    sourceValue assignment factor.left *
    sourceValue assignment factor.right

def factorValueAt (assignment : Nat → Nat)
    (factors : List DecodedProductFactor) (index : Nat) : F :=
  match factors[index]? with
  | none => 0
  | some factor => factorValue assignment factor

/-- All five evaluation-pair positions. `factorCapacity` proves this fixed
sum covers every decoded factor and absent tail positions contribute zero. -/
def factorSum (assignment : Nat → Nat)
    (factors : List DecodedProductFactor) : F :=
  factorValueAt assignment factors 0 +
    factorValueAt assignment factors 1 +
    factorValueAt assignment factors 2 +
    factorValueAt assignment factors 3 +
    factorValueAt assignment factors 4

/-- Explicit compiler-rewrite recurrence, independent of compact row
satisfaction. -/
def StepHolds (assignment : Nat → Nat) (step : DecodedRewriteStep) : Prop :=
  outputValue assignment step.output =
    sourceValue assignment step.base +
      previousValue assignment step.previous +
      factorSum assignment step.factors

/-- A derived-output semantic step is an exact entry of the Rust witness
registry and obeys that entry's zero-base field recurrence. -/
theorem derivedStepHolds_witnessRecurrence
    {assignment : Nat → Nat} {step : DecodedRewriteStep}
    (member : step ∈ decodedRewriteSteps) {slot : DecodedDerivedSlot}
    (outputEq : step.output = .derivedProductSum slot)
    (holds : StepHolds assignment step) :
    decodedDerivedRecurrencePayload step slot ∈
        Materialized.Checked.derivedProductSums.map rawDerivedRecurrence ∧
      derivedValue assignment slot =
        previousValue assignment step.previous +
          factorSum assignment step.factors := by
  have registered := decodedDerivedRecurrenceRegistered step member
  have registeredPayload :
      decodedDerivedRecurrencePayload step slot ∈
        Materialized.Checked.derivedProductSums.map rawDerivedRecurrence := by
    simpa [decodedDerivedRecurrence, outputEq] using registered
  have baseTerms := decodedDerivedOutputBaseZero step member
  simp only [outputEq] at baseTerms
  have baseZero : sourceValue assignment step.base = 0 := by
    simp [sourceValue, baseTerms, lcEval, fieldResidue]
  unfold StepHolds outputValue at holds
  rw [outputEq, baseZero, Fin.zero_add] at holds
  exact ⟨registeredPayload, holds⟩

/-- Integrated refinement boundary for the actual derived witness program:
the exported registry is the emitted derived recurrence stream, and every
decoded step satisfies that stream's field equation. -/
def WitnessRewriteProgramHolds (assignment : Nat → Nat) : Prop :=
  decodedRewriteSteps.filterMap decodedDerivedRecurrence =
      Materialized.Checked.derivedProductSums.map rawDerivedRecurrence ∧
    ∀ step ∈ decodedRewriteSteps, StepHolds assignment step

theorem witnessRewriteProgramHolds {assignment : Nat → Nat}
    (steps : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    WitnessRewriteProgramHolds assignment :=
  ⟨decodedDerivedRecurrenceRegistryExact, steps⟩

/-- Exact retained source A/B/C equation under the reconstructed source
assignment. -/
def RetainedHolds (assignment : Nat → Nat)
    (step : DecodedRetainedStep) : Prop :=
  sourceValue assignment step.a * sourceValue assignment step.b =
    sourceValue assignment step.c

theorem evalSourceLinearForm (assignment : Nat → Nat)
    (linear : DecodedSourceLinearCombination) :
    evalLinearForm assignment (sourceLinearForm linear) =
      sourceValue assignment linear := by
  exact evalSubstituteLinearTerms assignment compilerLinearForms
    (compilerAssignment assignment) (evalCompilerLinearForm assignment)
    linear.programTerms

theorem evalDerivedLinearForm (assignment : Nat → Nat)
    (slot : DecodedDerivedSlot) :
    evalLinearForm assignment (derivedLinearForm slot) =
      derivedValue assignment slot := by
  exact evalNatTermsLinearForm assignment _

theorem evalOutputLinearForm (assignment : Nat → Nat)
    (output : DecodedRewriteOutput) :
    evalLinearForm assignment (outputLinearForm output) =
      outputValue assignment output := by
  cases output with
  | source value => exact evalSourceLinearForm assignment value
  | derivedProductSum slot => exact evalDerivedLinearForm assignment slot

theorem evalPreviousLinearForm (assignment : Nat → Nat)
    (previous : Option DecodedDerivedSlot) :
    evalLinearForm assignment (previousLinearForm previous) =
      previousValue assignment previous := by
  cases previous with
  | none => rfl
  | some slot => exact evalDerivedLinearForm assignment slot

private theorem negOne_mul (value : F) : (-1 : F) * value = -value := by
  calc
    (-1 : F) * value = -(1 * value) := Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

theorem evalNegateLinearForm (assignment : Nat → Nat) (form : LinearForm) :
    evalLinearForm assignment (negateLinearForm form) =
      -evalLinearForm assignment form := by
  unfold evalLinearForm negateLinearForm
  rw [Materialized.LinearForm.eval_scale, negOne_mul]

theorem evalRewriteCLinearForm (assignment : Nat → Nat)
    (step : DecodedRewriteStep) :
    evalLinearForm assignment (rewriteCLinearForm step) =
      outputValue assignment step.output +
        (-sourceValue assignment step.base +
          -previousValue assignment step.previous) := by
  unfold evalLinearForm rewriteCLinearForm
  rw [Materialized.LinearForm.eval_append,
    Materialized.LinearForm.eval_append]
  change
    (evalLinearForm assignment (outputLinearForm step.output) +
        evalLinearForm assignment
          (negateLinearForm (sourceLinearForm step.base))) +
      evalLinearForm assignment
        (negateLinearForm (previousLinearForm step.previous)) = _
  rw [evalOutputLinearForm, evalNegateLinearForm,
    evalSourceLinearForm, evalNegateLinearForm, evalPreviousLinearForm]
  exact Lean.Grind.Fin.add_assoc _ _ _

theorem evalFactorLeftLinearForm (assignment : Nat → Nat)
    (factor : DecodedProductFactor) :
    evalLinearForm assignment (factorLeftLinearForm factor) =
      fieldResidue factor.coefficient * sourceValue assignment factor.left := by
  unfold evalLinearForm factorLeftLinearForm
  rw [Materialized.LinearForm.eval_scale]
  change
    fieldResidue factor.coefficient *
        evalLinearForm assignment (sourceLinearForm factor.left) = _
  rw [evalSourceLinearForm]

theorem evalFactorRightLinearForm (assignment : Nat → Nat)
    (factor : DecodedProductFactor) :
    evalLinearForm assignment (factorRightLinearForm factor) =
      sourceValue assignment factor.right := by
  exact evalSourceLinearForm assignment factor.right

theorem evalFactorPairAt (assignment : Nat → Nat)
    (factors : List DecodedProductFactor) (index : Nat) :
    evalLinearForm assignment (factorLeftLinearFormAt factors index) *
        evalLinearForm assignment (factorRightLinearFormAt factors index) =
      factorValueAt assignment factors index := by
  cases factorAt : factors[index]? with
  | none =>
      simp only [factorLeftLinearFormAt, factorRightLinearFormAt,
        factorValueAt, factorAt]
      change (0 : F) * 0 = 0
      exact Fin.zero_mul 0
  | some factor =>
      simp only [factorLeftLinearFormAt, factorRightLinearFormAt,
        factorValueAt, factorAt]
      rw [evalFactorLeftLinearForm, evalFactorRightLinearForm]
      rfl

theorem evalSteadySelectorLinearForm {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    evalLinearForm assignment steadySelectorLinearForm = 1 := by
  rw [steadySelectorLinearForm, evalNatTermsLinearForm]
  apply Fin.ext
  simp [lcEval, selectorOne, fieldResidue, modulus_eq]
  native_decide

def rewritePoint (assignment : Nat → Nat)
    (step : DecodedRewriteStep) : Fin 13 → F :=
  fun port => evalLinearForm assignment (rewritePortLinearForm step port)

def retainedPoint (assignment : Nat → Nat)
    (step : DecodedRetainedStep) : Fin 13 → F :=
  fun port => evalLinearForm assignment (retainedPortLinearForm step port)

theorem rewriteRowPoint_eq (assignment : Nat → Nat)
    (row : DecodedRow) (step : DecodedRewriteStep)
    (matching : RewriteCoefficientsMatch row step) :
    rowPoint row (fieldAssignment assignment) = rewritePoint assignment step := by
  funext port
  calc
    rowPoint row (fieldAssignment assignment) port =
        Materialized.LinearForm.eval
          (fun column => fieldResidue (assignment column))
          (Materialized.LinearForm.portTerms (row.port port)) := by
      simpa [rowPoint, fieldAssignment] using
        Materialized.LinearForm.action_eq_eval (row.port port)
          (fun column => fieldResidue (assignment column))
    _ = Materialized.LinearForm.eval
          (fun column => fieldResidue (assignment column))
          (rewritePortLinearForm step port) :=
      Materialized.LinearForm.eval_eq_of_equivalent (matching port) _
    _ = rewritePoint assignment step port := rfl

theorem retainedRowPoint_eq (assignment : Nat → Nat)
    (row : DecodedRow) (step : DecodedRetainedStep)
    (matching : RetainedCoefficientsMatch row step) :
    rowPoint row (fieldAssignment assignment) = retainedPoint assignment step := by
  funext port
  calc
    rowPoint row (fieldAssignment assignment) port =
        Materialized.LinearForm.eval
          (fun column => fieldResidue (assignment column))
          (Materialized.LinearForm.portTerms (row.port port)) := by
      simpa [rowPoint, fieldAssignment] using
        Materialized.LinearForm.action_eq_eval (row.port port)
          (fun column => fieldResidue (assignment column))
    _ = Materialized.LinearForm.eval
          (fun column => fieldResidue (assignment column))
          (retainedPortLinearForm step port) :=
      Materialized.LinearForm.eval_eq_of_equivalent (matching port) _
    _ = retainedPoint assignment step port := rfl

private theorem rewritePoint_generalSelector
    (assignment : Nat → Nat) (step : DecodedRewriteStep) :
    rewritePoint assignment step Role.generalSelector.index = 0 := by
  rfl

private theorem rewritePoint_evalSelector {assignment : Nat → Nat}
    (step : DecodedRewriteStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    rewritePoint assignment step Role.evalSelector.index = 1 := by
  exact evalSteadySelectorLinearForm selectorOne

private theorem retainedPoint_generalSelector {assignment : Nat → Nat}
    (step : DecodedRetainedStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    retainedPoint assignment step Role.generalSelector.index = 1 := by
  exact evalSteadySelectorLinearForm selectorOne

private theorem retainedPoint_evalSelector
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.evalSelector.index = 0 := by
  rfl

private theorem retainedPoint_bit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.bit.index = 0 := by
  rfl

private theorem retainedPoint_sboxInput
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.sboxInput.index = 0 := by
  rfl

private theorem retainedPoint_centeredUnit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.centeredUnit.index = 0 := by
  rfl

private theorem retainedPoint_canonicalDigit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalDigit.index = 0 := by
  rfl

private theorem retainedPoint_canonicalBorrow
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalBorrow.index = 0 := by
  rfl

private theorem retainedPoint_canonicalNextBorrow
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalNextBorrow.index = 0 := by
  rfl

private theorem retainedPoint_canonicalBoundDigit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalBoundDigit.index = 0 := by
  rfl

private theorem retainedPoint_evalTailRight
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.evalTailRight.index = 0 := by
  rfl

private theorem evaluateRewritePoint {assignment : Nat → Nat}
    (step : DecodedRewriteStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    evaluate (rewritePoint assignment step) =
      -(outputValue assignment step.output +
          (-sourceValue assignment step.base +
            -previousValue assignment step.previous)) +
        factorSum assignment step.factors := by
  have canonicalZero :
      canonicalResidual (rewritePoint assignment step) = 0 :=
    canonicalResidual_zero_of_generalSelector_zero _
      (rewritePoint_generalSelector assignment step)
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    rewritePoint_generalSelector, rewritePoint_evalSelector step selectorOne,
    Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero, Fin.one_mul,
    Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]
  change
    ((((-(evalLinearForm assignment (rewriteCLinearForm step)) +
          evalLinearForm assignment (factorLeftLinearFormAt step.factors 0) *
            evalLinearForm assignment (factorRightLinearFormAt step.factors 0)) +
        evalLinearForm assignment (factorLeftLinearFormAt step.factors 1) *
          evalLinearForm assignment (factorRightLinearFormAt step.factors 1)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 2) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 2)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 3) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 3)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 4) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 4) = _
  rw [evalRewriteCLinearForm,
    evalFactorPairAt assignment step.factors 0,
    evalFactorPairAt assignment step.factors 1,
    evalFactorPairAt assignment step.factors 2,
    evalFactorPairAt assignment step.factors 3,
    evalFactorPairAt assignment step.factors 4]
  simp only [factorSum, Lean.Grind.Fin.add_assoc]

private theorem evaluateRetainedPoint {assignment : Nat → Nat}
    (step : DecodedRetainedStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    evaluate (retainedPoint assignment step) =
      sourceValue assignment step.a * sourceValue assignment step.b +
        -sourceValue assignment step.c := by
  have canonicalZero :
      canonicalResidual (retainedPoint assignment step) = 0 :=
    canonicalResidual_zero_of_classPorts_zero _
      (retainedPoint_canonicalDigit assignment step)
      (retainedPoint_canonicalBorrow assignment step)
      (retainedPoint_canonicalNextBorrow assignment step)
      (retainedPoint_canonicalBoundDigit assignment step)
      (retainedPoint_evalTailRight assignment step)
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    retainedPoint_generalSelector step selectorOne,
    retainedPoint_evalSelector, retainedPoint_bit, retainedPoint_sboxInput,
    retainedPoint_centeredUnit, retainedPoint_canonicalDigit,
    retainedPoint_canonicalBorrow, retainedPoint_canonicalNextBorrow,
    retainedPoint_canonicalBoundDigit, retainedPoint_evalTailRight,
    Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero, Fin.one_mul,
    Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]
  change
    evalLinearForm assignment (sourceLinearForm step.a) *
          evalLinearForm assignment (sourceLinearForm step.b) +
        -evalLinearForm assignment (sourceLinearForm step.c) = _
  rw [evalSourceLinearForm, evalSourceLinearForm, evalSourceLinearForm]

theorem rewritePairStepHolds {assignment : Nat → Nat}
    (rowsSatisfied : RowsSatisfied Materialized.Artifact.rewriteRows assignment)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    {pair : DecodedRow × DecodedRewriteStep}
    (pairMember : pair ∈ rewritePairs) :
    StepHolds assignment pair.2 := by
  have rowMember : pair.1 ∈ Materialized.Artifact.rewriteRows := by
    rw [← rewritePairRowsExact]
    exact List.mem_map.mpr ⟨pair, pairMember, rfl⟩
  have matching := rewriteCoefficientsExact pair pairMember
  have rowZero : residual pair.1 (fieldAssignment assignment) = 0 := by
    rw [residual_fieldAssignment_eq_natResidual]
    exact rowsSatisfied pair.1 rowMember
  have pointZero : evaluate (rewritePoint assignment pair.2) = 0 := by
    rw [← rewriteRowPoint_eq assignment pair.1 pair.2 matching]
    exact rowZero
  rw [evaluateRewritePoint pair.2 selectorOne] at pointZero
  unfold StepHolds
  grind

theorem retainedPairHolds {assignment : Nat → Nat}
    (rowsSatisfied : RowsSatisfied Materialized.Artifact.retainedRows assignment)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    {pair : DecodedRow × DecodedRetainedStep}
    (pairMember : pair ∈ retainedPairs) :
    RetainedHolds assignment pair.2 := by
  have rowMember : pair.1 ∈ Materialized.Artifact.retainedRows := by
    rw [← retainedPairRowsExact]
    exact List.mem_map.mpr ⟨pair, pairMember, rfl⟩
  have matching := retainedCoefficientsExact pair pairMember
  have rowZero : residual pair.1 (fieldAssignment assignment) = 0 := by
    rw [residual_fieldAssignment_eq_natResidual]
    exact rowsSatisfied pair.1 rowMember
  have pointZero : evaluate (retainedPoint assignment pair.2) = 0 := by
    rw [← retainedRowPoint_eq assignment pair.1 pair.2 matching]
    exact rowZero
  rw [evaluateRetainedPoint pair.2 selectorOne] at pointZero
  unfold RetainedHolds
  exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
    sourceValue assignment pair.2.a * sourceValue assignment pair.2.b -
        sourceValue assignment pair.2.c = 0 ↔
      sourceValue assignment pair.2.a * sourceValue assignment pair.2.b =
        sourceValue assignment pair.2.c).mp (by
          simpa [Fin.sub_eq_add_neg] using pointZero)

/-- Every decoded rewrite step is justified by its exact selected compact row
once the steady selector is active. -/
theorem allRewriteStepsHold {assignment : Nat → Nat}
    (rowsSatisfied : RowsSatisfied Materialized.Artifact.rewriteRows assignment)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    ∀ step ∈ decodedRewriteSteps, StepHolds assignment step := by
  intro step stepMember
  have pairStepMember : step ∈ rewritePairs.map Prod.snd := by
    rw [rewritePairStepsExact]
    exact stepMember
  rcases List.mem_map.mp pairStepMember with ⟨pair, pairMember, equal⟩
  subst step
  exact rewritePairStepHolds rowsSatisfied selectorOne pairMember

/-- Every decoded retained source row satisfies its exact A/B/C equation once
the steady selector is active. -/
theorem allRetainedStepsHold {assignment : Nat → Nat}
    (rowsSatisfied : RowsSatisfied Materialized.Artifact.retainedRows assignment)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    ∀ step ∈ decodedRetainedSteps, RetainedHolds assignment step := by
  intro step stepMember
  have pairStepMember : step ∈ retainedPairs.map Prod.snd := by
    rw [retainedPairStepsExact]
    exact stepMember
  rcases List.mem_map.mp pairStepMember with ⟨pair, pairMember, equal⟩
  subst step
  exact retainedPairHolds rowsSatisfied selectorOne pairMember

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
