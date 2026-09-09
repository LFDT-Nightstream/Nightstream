import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAgreement
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw

/-!
The actual coordinate extractor returns a list. Its checked length and indexed
values become the PiCCS witness; no existentially chosen fork computes it.
List-length and dispatch work are charged here. The costed PiCCS access
primitive separately charges each assignment-coordinate access.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOutput

open scoped BigOperators
attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open PaperNonInteractive PaperStrongInterface
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateOracle PiRLC.CoordinateForkLaw
open PiRLC.CoordinateCheckedCalls PiRLC.CoordinateTerminalLaw

variable {Extension Commitment PublicInput Scalar State : Type*}
  {shape : Shape} {columns blockCount width : Nat}
  (key : Key Extension Commitment PublicInput Scalar State shape columns blockCount width)

def witnessOfList (values : List (PaperLinearAlgebra.Assignment F columns)) :
    Option (OutputWitness shape columns) :=
  if length : values.length = key.arity.total then
    some (outputWitnessOfAssignments key fun index =>
      values[index.val]'(by rw [length]; exact index.isLt))
  else none

theorem witnessOfList_ofFn (assignments : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns) :
    witnessOfList key (List.ofFn assignments) = some (outputWitnessOfAssignments key assignments) := by
  simp only [witnessOfList, List.length_ofFn, ↓reduceDIte, List.getElem_ofFn]

/-- One step per length-list node including nil, then one length branch. -/
def decodeResult (values : List (PaperLinearAlgebra.Assignment F columns)) :
    PiRLC.PaperForkExtractionWork.Result (Option (OutputWitness shape columns)) :=
  ⟨witnessOfList key values, values.length + 2⟩

/-- Option dispatch is charged on both branches. -/
def decode : Option (List (PaperLinearAlgebra.Assignment F columns)) →
    PiRLC.PaperForkExtractionWork.Result (Option (OutputWitness shape columns))
  | none => ⟨none, 1⟩
  | some values => ⟨(decodeResult key values).value, (decodeResult key values).work + 1⟩

variable [DecidableEq Scalar] [Fintype (Challenge key.piRlcAlgebra)]
  [Nonempty (Challenge key.piRlcAlgebra)] [Fintype (PaperLinearAlgebra.Assignment F columns)]

def endpointWitness
    (program : Primitives Scalar (PaperLinearAlgebra.Assignment F columns))
    (endpoint : PaperWeakLaw.Endpoint (Fin key.arity.total) (Challenge key.piRlcAlgebra)
      (PaperLinearAlgebra.Assignment F columns)) : Option (OutputWitness shape columns) :=
  (decode key (PaperWeakLaw.terminalValue program endpoint)).value

variable (running : Running Extension Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)
  (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
  (probe : Probe Extension shape)
  (oracle : Oracle (Fin key.arity.total) (Challenge key.piRlcAlgebra)
    (PaperLinearAlgebra.Assignment F columns))
  (checker : Response (PaperLinearAlgebra.Assignment F columns) Scalar key.params key.arity → CheckResult)
  (checkSpec : ∀ response, (checker response).accepted = true ↔
    response.Success key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatchForProbe key running fresh probe))
  (program : Primitives Scalar (PaperLinearAlgebra.Assignment F columns))
  (correct : Correct laws.ring laws.assignmentModule program)

include strongSet checkSpec correct in
/-- A positive sampled endpoint preserves successful return, supplies the
exact relaxed output witness, and has the fixed-length decoding work bound. -/
theorem positive_endpoint
    (endpoint : PaperWeakLaw.Endpoint (Fin key.arity.total) (Challenge key.piRlcAlgebra)
      (PaperLinearAlgebra.Assignment F columns))
    (positive : 0 < (PaperWeakLaw.law oracle
      (oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted)) endpoint).toReal) :
    (endpointWitness key program endpoint).isSome =
      (PaperWeakLaw.terminalValue program endpoint).isSome ∧
    (∀ witness, endpointWitness key program endpoint = some witness →
      AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
        (key.statement running fresh) probe witness) ∧
    (decode key (PaperWeakLaw.terminalValue program endpoint)).work ≤ key.arity.total + 3 := by
  cases endpoint with
  | none => simp [endpointWitness, PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult, decode]
  | some endpoint =>
      rcases endpoint with ⟨vector, initial, outputs⟩
      have endpointPositive := (PaperWeakLaw.law_some_positive_iff oracle
        (oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted))
        vector initial outputs).mp positive
      cases returned : (PiRLC.CoordinateTerminalProgram.finish program vector initial outputs).value with
      | none =>
          simp [endpointWitness, PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult,
            returned, decode]
      | some values =>
          obtain ⟨assignments, listEq, _ambient, witnessValid⟩ :=
            PaperWeakAgreement.positive_return_opens_probe key running fresh laws strongSet
              probe oracle checker checkSpec program correct vector initial outputs values
              endpointPositive returned
          have decoded : endpointWitness key program (some (vector, initial, outputs)) =
              some (outputWitnessOfAssignments key assignments) := by
            simp only [endpointWitness, PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult,
              returned, decode, decodeResult, listEq, witnessOfList_ofFn]
          refine ⟨?_, ?_, ?_⟩
          · simp [decoded, PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult, returned]
          · intro witness equal
            have same := Option.some.inj (decoded.symm.trans equal)
            exact same ▸ witnessValid
          · simp only [PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult, returned,
              decode, decodeResult, listEq, List.length_ofFn]
            omega

include strongSet checkSpec correct in
/-- Decoding preserves the probability of a valid returned witness. -/
theorem valid_mean_eq_returningProbability :
    (∑ endpoint,
      (PaperWeakLaw.law oracle
        (oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted)) endpoint).toReal *
      (if ∃ witness, endpointWitness key program endpoint = some witness ∧
        AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
          (key.statement running fresh) probe witness then (1 : ℝ) else 0)) =
      returningProbability oracle
        (oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted))
        (fun vector initial outputs =>
          (PiRLC.CoordinateTerminalProgram.finish program vector initial outputs).value.isSome) := by
  classical
  let verify := oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted)
  have equal : (∑ endpoint, (PaperWeakLaw.law oracle verify endpoint).toReal *
      (if ∃ witness, endpointWitness key program endpoint = some witness ∧
        AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
          (key.statement running fresh) probe witness then (1 : ℝ) else 0)) =
      ∑ endpoint, (PaperWeakLaw.law oracle verify endpoint).toReal *
        (if (PaperWeakLaw.terminalValue program endpoint).isSome then (1 : ℝ) else 0) := by
    apply Finset.sum_congr rfl
    intro endpoint _
    by_cases positive : 0 < (PaperWeakLaw.law oracle verify endpoint).toReal
    · obtain ⟨same, valid, _work⟩ := positive_endpoint key running fresh laws strongSet
        probe oracle checker checkSpec program correct endpoint positive
      have present : (∃ witness, endpointWitness key program endpoint = some witness ∧
          AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
            (key.statement running fresh) probe witness) ↔
          (endpointWitness key program endpoint).isSome = true := by
        constructor
        · rintro ⟨witness, returned, _⟩
          simp only [returned, Option.isSome_some]
        · intro returned
          cases result : endpointWitness key program endpoint with
          | none => simp [result] at returned
          | some witness => exact ⟨witness, rfl, valid witness result⟩
      simp only [present, same]
    · have zero : (PaperWeakLaw.law oracle verify endpoint).toReal = 0 :=
        le_antisymm (le_of_not_gt positive) ENNReal.toReal_nonneg
      simp only [zero, zero_mul]
  rw [equal, PaperWeakLaw.law_mean]
  simp only [PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult,
    Option.isSome_none, Bool.false_eq_true, ↓reduceIte, mul_zero, zero_add,
    mul_ite, mul_one, returningProbability]
  rfl

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOutput
