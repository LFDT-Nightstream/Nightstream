import Nightstream.Implementation.R1CS.Canonical.KMulHonest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafCertificate
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTraceRefinement

/-!
Contract: exact refinement from the decoded production PiRLC Poseidon2 leaf
to the independent Lean Poseidon2 reference permutation.

Owns: the source-to-compact column map, two bounded 43-step schedule
certificates, field-to-residue evaluation, and the complete eight-lane call
result from all 86 typed S-box equations.

Does not own: absolute production-column placement, retained-row inclusion,
call-family coverage, transcript placement, replay-chain placement, lifecycle
semantics, or Poseidon2 collision security.

Emits constraints: no.

Assurance tier: artifact-checked call refinement.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

/-- Physical compact-trace column owned by one decoded source column. -/
def sourceColumnIndex : SourceColumn → Nat
  | .externalA lane => 1 + lane.val
  | .externalB lane => 5 + lane.val
  | .local offset => 9 + offset.val

/-- The decoded source assignment in compact Rust column space. Columns outside
the constant, input, and 600-source-column image fail closed to zero. -/
def sourcePhysical (source : SourceAssignment) (column : Nat) : Nat :=
  if zero : column = 0 then
    1
  else if externalA : column < 5 then
    (source.externalA ⟨column - 1, by omega⟩).val
  else if externalB : column < 9 then
    (source.externalB ⟨column - 5, by omega⟩).val
  else if localBound : column < 609 then
    (source.localValue ⟨column - 9, by omega⟩).val
  else
    0

@[simp] theorem sourcePhysical_zero (source : SourceAssignment) :
    sourcePhysical source 0 = 1 := by
  simp [sourcePhysical]

theorem sourcePhysical_canonical (source : SourceAssignment) :
    ∀ column, sourcePhysical source column < goldilocksP := by
  intro column
  unfold sourcePhysical
  split
  · simp [goldilocksP]
  · split
    · simpa [goldilocksP, goldilocksModulus] using
        (source.externalA _).isLt
    · split
      · simpa [goldilocksP, goldilocksModulus] using
          (source.externalB _).isLt
      · split
        · simpa [goldilocksP, goldilocksModulus] using
            (source.localValue _).isLt
        · simp [goldilocksP]

theorem sourcePhysical_sourceColumn (source : SourceAssignment)
    (column : SourceColumn) :
    sourcePhysical source (sourceColumnIndex column) =
      (sourceValue source column).val := by
  cases column with
  | externalA lane =>
      have notZero : 1 + lane.val ≠ 0 := by omega
      have bounded : 1 + lane.val < 5 := by omega
      simp only [sourceColumnIndex, sourcePhysical, notZero, ↓reduceDIte,
        bounded, sourceValue]
      congr 1
      apply Fin.ext
      simp
  | externalB lane =>
      have notZero : 5 + lane.val ≠ 0 := by omega
      have notA : ¬5 + lane.val < 5 := by omega
      have bounded : 5 + lane.val < 9 := by omega
      simp only [sourceColumnIndex, sourcePhysical, notZero, ↓reduceDIte,
        notA, bounded, sourceValue]
      congr 1
      apply Fin.ext
      simp
  | «local» offset =>
      have notZero : 9 + offset.val ≠ 0 := by omega
      have notA : ¬9 + offset.val < 5 := by omega
      have notB : ¬9 + offset.val < 9 := by omega
      have bounded : 9 + offset.val < 609 := by omega
      simp only [sourceColumnIndex, sourcePhysical, notZero, ↓reduceDIte,
        notA, notB, bounded, sourceValue]
      congr 1
      apply Fin.ext
      simp

/-- Exact term order used by the generated compact trace: source terms first,
then the constant-one term, including a zero coefficient when present. -/
def sourceTerms (value : SourceLinearCombination) : List (Nat × Nat) :=
  (value.terms.map fun term =>
    (sourceColumnIndex term.column, term.coefficient.val)) ++
      [(0, value.constant.val)]

/-- Field view of the executable sparse-row evaluator. -/
private def fieldEval (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) : F :=
  ⟨lcEval assignment terms, by
    unfold lcEval
    simpa [goldilocksP, goldilocksModulus] using
      Nat.mod_lt
        (terms.foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) 0)
        (by decide : 0 < goldilocksP)⟩

private theorem fieldEval_cons (assignment : Nat → Nat)
    (term : Nat × Nat) (rest : List (Nat × Nat)) :
    fieldEval assignment (term :: rest) =
      (⟨term.2 % goldilocksP, Nat.mod_lt _ (by decide)⟩ : F) *
          ⟨assignment term.1 % goldilocksP,
            Nat.mod_lt _ (by decide)⟩ +
        fieldEval assignment rest := by
  apply Fin.ext
  simp only [fieldEval, Fin.val_add, Fin.val_mul]
  simp [lcEval_eq_rawSum, rawSum_cons, Nat.add_mod, Nat.mul_mod,
    goldilocksP, goldilocksModulus]

private theorem fieldEval_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    fieldEval assignment (left ++ right) =
      fieldEval assignment left + fieldEval assignment right := by
  apply Fin.ext
  simp only [fieldEval, Fin.val_add]
  rw [lcEval_eq_rawSum, rawSum_append, Nat.add_mod]
  rw [← lcEval_eq_rawSum, ← lcEval_eq_rawSum]
  simp [goldilocksP, goldilocksModulus]

private theorem fieldEval_mapped_terms (source : SourceAssignment) :
    ∀ terms : List SourceTerm,
      fieldEval (sourcePhysical source)
          (terms.map fun term =>
            (sourceColumnIndex term.column, term.coefficient.val)) =
        sum (terms.map fun term =>
          term.coefficient * sourceValue source term.column) := by
  intro terms
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [fieldEval_cons, sourcePhysical_sourceColumn,
        inductionHypothesis]
      congr 2 <;> apply Fin.ext <;>
        simp [goldilocksP, goldilocksModulus,
          Nat.mod_eq_of_lt term.coefficient.isLt,
          Nat.mod_eq_of_lt (sourceValue source term.column).isLt]

private theorem fieldEval_constant (source : SourceAssignment)
    (constant : F) :
    fieldEval (sourcePhysical source) [(0, constant.val)] = constant := by
  apply Fin.ext
  simp [fieldEval, lcEval, sourcePhysical_zero, goldilocksP,
    goldilocksModulus, Nat.mod_eq_of_lt constant.isLt]

/-- The decoded field action and the compact sparse-row evaluator are the same
value. The proof is structural in the source term list. -/
theorem lcEval_sourceTerms (source : SourceAssignment)
    (value : SourceLinearCombination) :
    lcEval (sourcePhysical source) (sourceTerms value) =
      (sourceAction value source).val := by
  have equality :
      fieldEval (sourcePhysical source) (sourceTerms value) =
        sourceAction value source := by
    unfold sourceTerms sourceAction
    rw [fieldEval_append, fieldEval_mapped_terms,
      fieldEval_constant]
    apply Fin.ext
    simp only [Fin.val_add, Nat.add_comm]
  exact congrArg Fin.val equality

/-- All physical columns referenced by one sparse form are below `bound`. -/
def TermsBelow (bound : Nat) (terms : List (Nat × Nat)) : Prop :=
  ∀ term ∈ terms, term.1 < bound

instance (bound : Nat) (terms : List (Nat × Nat)) :
    Decidable (TermsBelow bound terms) := by
  unfold TermsBelow
  infer_instance

def traceIndex (step : Step) : Fin sboxCount :=
  ⟨step.rowOffset.val, by
    simpa only [sboxCount_eq] using step.rowOffset.isLt⟩

/-- Exact call-level comparison for one decoded generated step. -/
structure StepTraceMatches (step : Step) : Prop where
  inputExact : sourceTerms step.input = traceSboxInput (traceIndex step)
  inputPrivate : TermsBelow 601 (sourceTerms step.input)
  outputConstant : step.output.constant = 0
  outputTerms :
    step.output.terms.map (fun term =>
      (sourceColumnIndex term.column, term.coefficient.val)) =
        [(traceSboxOutput (traceIndex step), 1)]
  outputPrivate : traceSboxOutput (traceIndex step) < 601

instance (step : Step) : Decidable (StepTraceMatches step) :=
  if inputExact : sourceTerms step.input =
      traceSboxInput (traceIndex step) then
    if inputPrivate : TermsBelow 601 (sourceTerms step.input) then
      if outputConstant : step.output.constant = 0 then
        if outputTerms :
            step.output.terms.map (fun term =>
              (sourceColumnIndex term.column, term.coefficient.val)) =
                [(traceSboxOutput (traceIndex step), 1)] then
          if outputPrivate : traceSboxOutput (traceIndex step) < 601 then
            isTrue
              ⟨inputExact, inputPrivate, outputConstant, outputTerms,
                outputPrivate⟩
          else
            isFalse (fun matched => outputPrivate matched.outputPrivate)
        else
          isFalse (fun matched => outputTerms matched.outputTerms)
      else
        isFalse (fun matched => outputConstant matched.outputConstant)
    else
      isFalse (fun matched => inputPrivate matched.inputPrivate)
  else
    isFalse (fun matched => inputExact matched.inputExact)

def stepTraceCheck (step : Step) : Bool :=
  decide (StepTraceMatches step)

/-- Each reduction is bounded by one 43-step generated leaf. -/
theorem decoded_head_trace_checked :
    decodedStepHead.all stepTraceCheck = true := by
  rfl

theorem decoded_tail_trace_checked :
    decodedStepTail.all stepTraceCheck = true := by
  rfl

private theorem trace_matches_of_all_checked :
    ∀ steps : List Step, steps.all stepTraceCheck = true →
      ∀ step ∈ steps, StepTraceMatches step
  | [], _ => by simp
  | head :: tail, checked => by
      simp only [List.all_cons, Bool.and_eq_true] at checked
      intro step member
      rcases List.mem_cons.mp member with rfl | tailMember
      · apply of_decide_eq_true
        simpa only [stepTraceCheck] using checked.1
      · exact trace_matches_of_all_checked tail checked.2 step tailMember

theorem decoded_steps_trace_match :
    ∀ step ∈ decodedSteps, StepTraceMatches step := by
  intro step member
  rw [decodedSteps, List.mem_append] at member
  rcases member with headMember | tailMember
  · exact trace_matches_of_all_checked decodedStepHead
      decoded_head_trace_checked step headMember
  · exact trace_matches_of_all_checked decodedStepTail
      decoded_tail_trace_checked step tailMember

/-- The two bounded leaves also certify that every compact S-box index occurs
once, in order. -/
theorem decoded_head_offsets_complete :
    decodedStepHead.map (fun step => step.rowOffset) =
      (List.finRange 86).take 43 := by
  rfl

theorem decoded_tail_offsets_complete :
    decodedStepTail.map (fun step => step.rowOffset) =
      (List.finRange 86).drop 43 := by
  rfl

theorem decoded_offsets_complete :
    decodedSteps.map (fun step => step.rowOffset) = List.finRange 86 := by
  unfold decodedSteps
  rw [List.map_append, decoded_head_offsets_complete,
    decoded_tail_offsets_complete, List.take_append_drop]

theorem decoded_step_for_index (index : Fin sboxCount) :
    ∃ step ∈ decodedSteps, traceIndex step = index := by
  let offset : Fin 86 :=
    ⟨index.val, by simpa only [sboxCount_eq] using index.isLt⟩
  have offsetMember :
      offset ∈ decodedSteps.map (fun step => step.rowOffset) := by
    rw [decoded_offsets_complete]
    simp
  rcases List.mem_map.mp offsetMember with
    ⟨step, stepMember, stepOffset⟩
  refine ⟨step, stepMember, ?_⟩
  apply Fin.ext
  exact congrArg Fin.val stepOffset

private theorem step_output_value
    (source : SourceAssignment) (step : Step)
    (matchProof : StepTraceMatches step) :
    (sourceAction step.output source).val =
      sourcePhysical source (traceSboxOutput (traceIndex step)) := by
  rw [← lcEval_sourceTerms]
  unfold sourceTerms
  rw [matchProof.outputTerms, matchProof.outputConstant]
  simp only [Fin.val_zero]
  have canonical := sourcePhysical_canonical source
    (traceSboxOutput (traceIndex step))
  simp [lcEval, sourcePhysical_zero, Nat.mod_eq_of_lt canonical]

private theorem field_seventh_value (value : F) :
    (value * value * value * value * value * value * value).val =
      sbox7 value.val := by
  have regrouped :
      value * value * value * value * value * value * value =
        value * ((value * value) * ((value * value) * (value * value))) := by
    simp only [Fin.mul_assoc]
  simpa only [sbox7, Fin.val_mul, goldilocksP, goldilocksModulus] using
    congrArg Fin.val regrouped

private theorem matched_step_sbox
    (source : SourceAssignment) (step : Step)
    (matchProof : StepTraceMatches step)
    (holds : StepSboxHolds source step) :
    sourcePhysical source (traceSboxOutput (traceIndex step)) =
      sbox7
        (lcEval (sourcePhysical source)
          (traceSboxInput (traceIndex step))) := by
  have values := congrArg Fin.val holds
  rw [field_seventh_value] at values
  calc
    sourcePhysical source (traceSboxOutput (traceIndex step)) =
        (sourceAction step.output source).val :=
      (step_output_value source step matchProof).symm
    _ = sbox7 (sourceAction step.input source).val := values.symm
    _ = sbox7
        (lcEval (sourcePhysical source) (sourceTerms step.input)) := by
      rw [lcEval_sourceTerms]
    _ = sbox7
        (lcEval (sourcePhysical source)
          (traceSboxInput (traceIndex step))) := by
      rw [matchProof.inputExact]

/-- Output-column owner for the eight compact terminal values. -/
def outputLane? (column : Nat) : Option (Fin width) :=
  if bounded : 601 ≤ column ∧ column < 609 then
    some ⟨column - 601, by simp only [width]; omega⟩
  else
    none

/-- Proof-only completion of the compact trace. The eight final columns are
defined from the final linear forms; all authoritative input and S-box columns
remain the decoded source assignment. -/
def callPhysical (source : SourceAssignment) (column : Nat) : Nat :=
  match outputLane? column with
  | some lane => lcEval (sourcePhysical source) (traceFinalForm lane)
  | none => sourcePhysical source column

theorem outputLane_none_of_lt {column : Nat} (below : column < 601) :
    outputLane? column = none := by
  simp [outputLane?, below]

theorem callPhysical_eq_source_of_lt (source : SourceAssignment)
    {column : Nat} (below : column < 601) :
    callPhysical source column = sourcePhysical source column := by
  simp [callPhysical, outputLane_none_of_lt below]

theorem outputLane_traceOutput (lane : Fin width) :
    outputLane? (traceOutputColumn lane) = some lane := by
  fin_cases lane <;> rfl

theorem callPhysical_traceOutput (source : SourceAssignment)
    (lane : Fin width) :
    callPhysical source (traceOutputColumn lane) =
      lcEval (sourcePhysical source) (traceFinalForm lane) := by
  simp [callPhysical, outputLane_traceOutput]

theorem callPhysical_canonical (source : SourceAssignment) :
    ∀ column, callPhysical source column < goldilocksP := by
  intro column
  unfold callPhysical
  split
  · unfold lcEval
    exact Nat.mod_lt _ (by decide)
  · exact sourcePhysical_canonical source column

private theorem finalFormBelow601 (lane : Fin width) :
    TermsBelow 601 (traceFinalForm lane) := by
  fin_cases lane <;> decide

private theorem lcEval_call_eq_source
    (source : SourceAssignment) (terms : List (Nat × Nat))
    (privateTerms : TermsBelow 601 terms) :
    lcEval (callPhysical source) terms =
      lcEval (sourcePhysical source) terms := by
  apply KMulHonest.lcEval_congr
  intro column mentioned
  rcases List.mem_map.mp mentioned with ⟨term, termMember, termColumn⟩
  subst column
  exact callPhysical_eq_source_of_lt source
    (privateTerms term termMember)

/-- The 86 decoded S-box equations complete the exact compact Poseidon2 trace
on one source assignment. -/
theorem step_sboxes_imply_trace_holds
    (source : SourceAssignment)
    (stepSboxes : ∀ step ∈ decodedSteps, StepSboxHolds source step) :
    TraceHolds (callPhysical source) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [callPhysical_eq_source_of_lt source (by omega)]
    exact sourcePhysical_zero source
  · intro index
    rcases decoded_step_for_index index with
      ⟨step, stepMember, stepIndex⟩
    have matchProof := decoded_steps_trace_match step stepMember
    have baseEquation :=
      matched_step_sbox source step matchProof
        (stepSboxes step stepMember)
    have inputPrivate :
        TermsBelow 601 (traceSboxInput (traceIndex step)) := by
      rw [← matchProof.inputExact]
      exact matchProof.inputPrivate
    rw [← stepIndex]
    rw [callPhysical_eq_source_of_lt source matchProof.outputPrivate,
      lcEval_call_eq_source source _ inputPrivate]
    exact baseEquation
  · intro lane
    rw [callPhysical_traceOutput]
    exact (lcEval_call_eq_source source _ (finalFormBelow601 lane)).symm

def sourceInput (source : SourceAssignment) (lane : Fin width) : F :=
  if first : lane.val < 4 then
    source.externalA ⟨lane.val, first⟩
  else
    source.externalB ⟨lane.val - 4, by
      have laneBound : lane.val < 8 := by
        simpa only [width] using lane.isLt
      omega⟩

theorem sourcePhysical_traceInput (source : SourceAssignment)
    (lane : Fin width) :
    sourcePhysical source (traceInputColumn lane) =
      (sourceInput source lane).val := by
  fin_cases lane <;> rfl

/-- Headline call theorem: all 86 typed generated S-box equations force all
eight outputs of the independently specified production Poseidon2
permutation. -/
theorem step_sboxes_compute_reference
    (source : SourceAssignment)
    (stepSboxes : ∀ step ∈ decodedSteps, StepSboxHolds source step)
    (lane : Fin width) :
    lcEval (sourcePhysical source) (traceFinalForm lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane => (sourceInput source inputLane).val) lane := by
  have computes := trace_computes_reference
    (callPhysical source) (callPhysical_canonical source)
    (step_sboxes_imply_trace_holds source stepSboxes) lane
  rw [callPhysical_traceOutput] at computes
  apply computes.trans
  congr 2
  funext inputLane
  rw [callPhysical_eq_source_of_lt source (by
    fin_cases inputLane <;> decide)]
  exact sourcePhysical_traceInput source inputLane

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
