import Mathlib.Data.List.GetD
import NightstreamFPrime.Gadgets.Sampling.First54

/-!
Owns the semantic bridge from the fixed first-54 circuit trace to the generic
`FirstAccepted` sampler. It does not define a second candidate order or a
second acceptance rule.
-/

namespace NightstreamFPrime.Gadgets.Sampling.First54

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Sampling

def semanticVerifier (interface : Interface) (offset : Nat) (env : Env) :
    FirstAccepted.Verifier (Fin candidateCount) F where
  accepts := fun candidate =>
    decide ((interface.accepted offset candidate).eval env = 1)
  symbol := fun candidate => (interface.symbol offset candidate).eval env

def candidateStream : FirstAccepted.CandidateStream (Fin candidateCount) :=
  candidateIndex

def semanticCandidates (count : Nat) : List (Fin candidateCount) :=
  FirstAccepted.streamPrefix candidateStream count

def semanticAcceptedSymbols (interface : Interface) (offset : Nat)
    (env : Env) (count : Nat) : List F :=
  FirstAccepted.acceptedSymbols (semanticVerifier interface offset env)
    (semanticCandidates count)

def semanticAcceptedCount (interface : Interface) (offset : Nat)
    (env : Env) (count : Nat) : Nat :=
  FirstAccepted.acceptedCount (semanticVerifier interface offset env)
    (semanticCandidates count)

def acceptedValue (interface : Interface) (offset : Nat) (env : Env)
    (round : Nat) : F :=
  (interface.accepted offset (candidateIndex round)).eval env

def symbolValue (interface : Interface) (offset : Nat) (env : Env)
    (round : Nat) : F :=
  (interface.symbol offset (candidateIndex round)).eval env

def acceptedAt (interface : Interface) (offset : Nat) (env : Env)
    (round : Nat) : Bool :=
  (semanticVerifier interface offset env).accepts (candidateIndex round)

theorem semanticCandidates_succ (count : Nat) :
    semanticCandidates (count + 1) =
      semanticCandidates count ++ [candidateIndex count] := by
  simp [semanticCandidates, FirstAccepted.streamPrefix, candidateStream,
    List.range_succ]

theorem semanticAcceptedSymbols_length (interface : Interface) (offset : Nat)
    (env : Env) (count : Nat) :
    (semanticAcceptedSymbols interface offset env count).length =
      semanticAcceptedCount interface offset env count := by
  simp [semanticAcceptedSymbols, semanticAcceptedCount,
    FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCount]

private theorem acceptedSymbols_singleton
    {Candidate Symbol : Type} (verifier : FirstAccepted.Verifier Candidate Symbol)
    (candidate : Candidate) :
    FirstAccepted.acceptedSymbols verifier [candidate] =
      if verifier.accepts candidate then [verifier.symbol candidate] else [] := by
  cases accepted : verifier.accepts candidate <;>
    simp [FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates,
      accepted]

theorem semanticAcceptedSymbols_succ (interface : Interface) (offset : Nat)
    (env : Env) (count : Nat) :
    semanticAcceptedSymbols interface offset env (count + 1) =
      if acceptedAt interface offset env count then
        semanticAcceptedSymbols interface offset env count ++
          [symbolValue interface offset env count]
      else
        semanticAcceptedSymbols interface offset env count := by
  unfold semanticAcceptedSymbols
  rw [semanticCandidates_succ, FirstAccepted.acceptedSymbols_append,
    acceptedSymbols_singleton]
  change semanticAcceptedSymbols interface offset env count ++
      (if acceptedAt interface offset env count then
        [symbolValue interface offset env count] else []) =
    if acceptedAt interface offset env count then
      semanticAcceptedSymbols interface offset env count ++
        [symbolValue interface offset env count]
    else semanticAcceptedSymbols interface offset env count
  cases accepted : acceptedAt interface offset env count <;> simp [accepted]

theorem semanticAcceptedCount_succ (interface : Interface) (offset : Nat)
    (env : Env) (count : Nat) :
    semanticAcceptedCount interface offset env (count + 1) =
      semanticAcceptedCount interface offset env count +
        if acceptedAt interface offset env count then 1 else 0 := by
  rw [← semanticAcceptedSymbols_length interface offset env (count + 1),
    semanticAcceptedSymbols_succ]
  cases accepted : acceptedAt interface offset env count <;>
    simp [accepted, semanticAcceptedSymbols_length]

theorem acceptedValue_eq_indicator (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (round : Nat) :
    acceptedValue interface offset env round =
      if acceptedAt interface offset env round then 1 else 0 := by
  by_cases one : acceptedValue interface offset env round = 1
  · have one' :
        (interface.accepted offset (candidateIndex round)).eval env = 1 := by
      simpa only [acceptedValue] using one
    have accepted : acceptedAt interface offset env round = true := by
      simp [acceptedAt, semanticVerifier, one']
    simp [accepted, one]
  · have zero : acceptedValue interface offset env round = 0 :=
      Or.resolve_right
        (assumptions.acceptedBoolean (candidateIndex round)) one
    have notOne :
        ¬ (interface.accepted offset (candidateIndex round)).eval env = 1 := by
      intro equality
      apply one
      simpa only [acceptedValue] using equality
    have rejected : acceptedAt interface offset env round = false := by
      simp [acceptedAt, semanticVerifier, notOne]
    simp [rejected, zero]

def oneHotPosition (count : Nat) (slot : Fin First54Step.slotCount) : F :=
  if slot.val = min count First54Step.fullSlot then 1 else 0

private theorem update_oneHotPosition (accepted : F) (flag : Bool)
    (acceptedEq : accepted = if flag then 1 else 0)
    (count : Nat) (slot : Fin First54Step.slotCount) :
    First54Step.update accepted (oneHotPosition count) slot =
      oneHotPosition (count + if flag then 1 else 0) slot := by
  cases flag with
  | false =>
      simp only [Bool.false_eq_true, if_false] at acceptedEq ⊢
      subst accepted
      simp [First54Step.update, oneHotPosition]
  | true =>
      simp only [if_true] at acceptedEq ⊢
      subst accepted
      unfold oneHotPosition First54Step.update
      have fullValue : First54Step.fullSlot = 54 := rfl
      have slotCountValue : First54Step.slotCount = 55 := rfl
      by_cases saturated : First54Step.fullSlot ≤ count
      · rw [Nat.min_eq_right saturated,
          Nat.min_eq_right (by omega)]
        split
        · simp_all [First54Step.fullSlot]
        · split
          · simp_all [First54Step.previousSlot, First54Step.fullSlot]
          · simp_all [First54Step.previousSlot, First54Step.fullSlot]
            intro impossible
            have bounded := slot.isLt
            simp [First54Step.slotCount] at bounded
            omega
      · have countLt : count < First54Step.fullSlot := by omega
        rw [Nat.min_eq_left countLt.le,
          Nat.min_eq_left (by omega : count + 1 ≤ First54Step.fullSlot)]
        split
        · simp_all [First54Step.fullSlot]
        · split
          · by_cases previousActive : count = First54Step.fullSlot - 1
            · subst count
              simp_all [First54Step.previousSlot, First54Step.fullSlot]
            · simp_all [First54Step.previousSlot, First54Step.fullSlot]
              have not54 : (54 : Nat) ≠ count := by omega
              have not53 : (53 : Nat) ≠ count := by omega
              simp [not54, not53]
          · by_cases previousActive :
                (First54Step.previousSlot slot (by omega)).val = count
            · have target : slot.val = count + 1 := by
                simp [First54Step.previousSlot] at previousActive
                omega
              simp [previousActive, target]
            · have target : slot.val ≠ count + 1 := by
                intro target
                apply previousActive
                simp [First54Step.previousSlot]
                omega
              simp [previousActive, target]

def PositionTrace (interface : Interface) (offset : Nat) (env : Env)
    (count : Nat) : Prop :=
  ∀ slot,
    (priorPosition offset count slot).eval env =
      oneHotPosition (semanticAcceptedCount interface offset env count) slot

theorem positionTrace_zero (interface : Interface) (offset : Nat) (env : Env) :
    PositionTrace interface offset env 0 := by
  intro slot
  by_cases first : slot.val = 0 <;>
    simp [PositionTrace, priorPosition, initialPosition, oneHotPosition,
      semanticAcceptedCount, semanticCandidates, FirstAccepted.streamPrefix,
      FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates, first,
      Expr.eval_const] <;> rfl

private theorem positionTrace_succ (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (positionSpecs : ∀ round : Fin candidateCount,
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) env)
    (count : Nat)
    (countLt : count < candidateCount)
    (trace : PositionTrace interface offset env count) :
    PositionTrace interface offset env (count + 1) := by
  intro slot
  let round : Fin candidateCount := ⟨count, countLt⟩
  have step := positionSpecs round slot
  have priorEq :
      (fun current => (priorPosition offset count current).eval env) =
        oneHotPosition (semanticAcceptedCount interface offset env count) := by
    funext current
    exact trace current
  calc
    (priorPosition offset (count + 1) slot).eval env =
        (First54Step.output (positionOffset offset count) slot).eval env := by
      rfl
    _ = First54Step.update (acceptedValue interface offset env count)
        (fun current => (priorPosition offset count current).eval env) slot := by
      simpa [round, positionInterface, acceptedValue] using step
    _ = First54Step.update (acceptedValue interface offset env count)
        (oneHotPosition (semanticAcceptedCount interface offset env count))
          slot := by rw [priorEq]
    _ = oneHotPosition
        (semanticAcceptedCount interface offset env count +
          if acceptedAt interface offset env count then 1 else 0) slot :=
      update_oneHotPosition (acceptedValue interface offset env count)
        (acceptedAt interface offset env count)
        (acceptedValue_eq_indicator interface offset env assumptions count)
        (semanticAcceptedCount interface offset env count) slot
    _ = oneHotPosition
        (semanticAcceptedCount interface offset env (count + 1)) slot := by
      rw [semanticAcceptedCount_succ]

theorem positionTrace_of_specs (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (positionSpecs : ∀ round : Fin candidateCount,
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) env)
    (count : Nat)
    (bounded : count ≤ candidateCount) :
    PositionTrace interface offset env count := by
  induction count with
  | zero => exact positionTrace_zero interface offset env
  | succ count inductionHypothesis =>
      apply positionTrace_succ interface offset env assumptions positionSpecs
        count (by omega)
      exact inductionHypothesis (by omega)

theorem positionTrace (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) (count : Nat)
    (bounded : count ≤ candidateCount) :
    PositionTrace interface offset env count :=
  positionTrace_of_specs interface offset env assumptions
    specification.position count bounded

theorem finalFull_eq_oneHot (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    (finalFull offset).eval env =
      oneHotPosition
        (semanticAcceptedCount interface offset env candidateCount) fullSlot := by
  have trace := positionTrace interface offset env assumptions specification
    candidateCount (Nat.le_refl _)
  simpa [finalFull, priorPosition, candidateCount] using trace fullSlot

theorem enough_of_spec (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    outputCount ≤ semanticAcceptedCount interface offset env candidateCount := by
  have finalEq := finalFull_eq_oneHot interface offset env assumptions specification
  have active : oneHotPosition
      (semanticAcceptedCount interface offset env candidateCount) fullSlot = 1 := by
    rw [← finalEq]
    exact specification.full
  have selected : fullSlot.val = min
      (semanticAcceptedCount interface offset env candidateCount)
        First54Step.fullSlot := by
    by_contra inactive
    have zero : oneHotPosition
        (semanticAcceptedCount interface offset env candidateCount) fullSlot = 0 := by
      simp [oneHotPosition, inactive]
    have contradiction : (0 : F) = 1 := zero.symm.trans active
    exact (by decide : (0 : F) ≠ 1) contradiction
  have minimum : min
      (semanticAcceptedCount interface offset env candidateCount) 54 = 54 := by
    simpa [fullSlot, First54Step.fullSlot] using selected.symm
  change 54 ≤ semanticAcceptedCount interface offset env candidateCount
  by_contra short
  have countLt :
      semanticAcceptedCount interface offset env candidateCount < 54 := by omega
  rw [Nat.min_eq_left countLt.le] at minimum
  omega

private theorem updateOutput_getD (accepted symbol : F) (flag : Bool)
    (acceptedEq : accepted = if flag then 1 else 0)
    (symbols : List F) (slot : Fin First54ValueStep.outputCount) :
    First54ValueStep.update accepted symbol
        (oneHotPosition symbols.length)
        (fun current => symbols.getD current.val 0) slot =
      (if flag then symbols ++ [symbol] else symbols).getD slot.val 0 := by
  cases flag with
  | false =>
      simp only [Bool.false_eq_true, if_false] at acceptedEq ⊢
      subst accepted
      simp [First54ValueStep.update]
  | true =>
      simp only [if_true] at acceptedEq ⊢
      subst accepted
      unfold First54ValueStep.update
      have slotBound : slot.val < 54 := by
        simpa [First54ValueStep.outputCount] using slot.isLt
      by_cases within : slot.val < symbols.length
      · have inactive :
            (First54ValueStep.positionSlot slot).val ≠
              min symbols.length First54Step.fullSlot := by
          by_cases saturated : First54Step.fullSlot ≤ symbols.length
          · rw [Nat.min_eq_right saturated]
            simp [First54ValueStep.positionSlot, First54Step.fullSlot]
            omega
          · have lengthLt : symbols.length < First54Step.fullSlot := by omega
            rw [Nat.min_eq_left lengthLt.le]
            simp [First54ValueStep.positionSlot]
            omega
        rw [List.getD_append symbols [symbol] 0 slot.val within]
        simp [First54ValueStep.update, oneHotPosition, inactive]
      · have beyond : symbols.length ≤ slot.val := by omega
        have lengthLt : symbols.length < First54Step.fullSlot := by
          simp [First54Step.fullSlot]
          omega
        by_cases next : slot.val = symbols.length
        · simp only
          rw [List.getD_eq_default symbols 0 beyond]
          rw [List.getD_append_right symbols [symbol] 0 slot.val beyond]
          simp [First54ValueStep.update, oneHotPosition,
            Nat.min_eq_left lengthLt.le, First54ValueStep.positionSlot, next]
        · have after : symbols.length < slot.val := by omega
          simp only
          rw [List.getD_eq_default symbols 0 beyond]
          rw [List.getD_append_right symbols [symbol] 0 slot.val beyond]
          have singletonDefault :
              [symbol].getD (slot.val - symbols.length) 0 = 0 := by
            apply List.getD_eq_default
            simp
            omega
          rw [singletonDefault]
          simp [First54ValueStep.update, oneHotPosition,
            Nat.min_eq_left lengthLt.le, First54ValueStep.positionSlot, next]

def ValueTrace (interface : Interface) (offset : Nat) (env : Env)
    (count : Nat) : Prop :=
  ∀ slot,
    (priorOutput offset count slot).eval env =
      (semanticAcceptedSymbols interface offset env count).getD slot.val 0

theorem valueTrace_zero (interface : Interface) (offset : Nat) (env : Env) :
    ValueTrace interface offset env 0 := by
  intro slot
  simp [ValueTrace, priorOutput, semanticAcceptedSymbols,
    semanticCandidates, FirstAccepted.streamPrefix,
    FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates] <;> rfl

private theorem valueTrace_succ (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (positionSpecs : ∀ round : Fin candidateCount,
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) env)
    (valueSpecs : ∀ round : Fin candidateCount,
      First54ValueStep.SpecHolds (valueInterface interface offset round.val)
        (valueOffset offset round.val) env)
    (count : Nat)
    (countLt : count < candidateCount)
    (trace : ValueTrace interface offset env count) :
    ValueTrace interface offset env (count + 1) := by
  intro slot
  let round : Fin candidateCount := ⟨count, countLt⟩
  let symbols := semanticAcceptedSymbols interface offset env count
  have step := valueSpecs round slot
  have positionAtCount := positionTrace_of_specs interface offset env assumptions
    positionSpecs count (by omega)
  have lengthEq : symbols.length =
      semanticAcceptedCount interface offset env count :=
    semanticAcceptedSymbols_length interface offset env count
  have positionEq :
      (fun current => (priorPosition offset count current).eval env) =
        oneHotPosition symbols.length := by
    funext current
    rw [positionAtCount current, lengthEq]
  have outputEq :
      (fun current => (priorOutput offset count current).eval env) =
        fun current => symbols.getD current.val 0 := by
    funext current
    exact trace current
  calc
    (priorOutput offset (count + 1) slot).eval env =
        (First54ValueStep.output (valueOffset offset count) slot).eval env := by
      rfl
    _ = First54ValueStep.update (acceptedValue interface offset env count)
        (symbolValue interface offset env count)
        (fun current => (priorPosition offset count current).eval env)
        (fun current => (priorOutput offset count current).eval env) slot := by
      simpa [round, valueInterface, acceptedValue, symbolValue] using step
    _ = First54ValueStep.update (acceptedValue interface offset env count)
        (symbolValue interface offset env count)
        (oneHotPosition symbols.length)
        (fun current => symbols.getD current.val 0) slot := by
      rw [positionEq, outputEq]
    _ = (if acceptedAt interface offset env count then
          symbols ++ [symbolValue interface offset env count]
        else symbols).getD slot.val 0 :=
      updateOutput_getD (acceptedValue interface offset env count)
        (symbolValue interface offset env count)
        (acceptedAt interface offset env count)
        (acceptedValue_eq_indicator interface offset env assumptions count)
        symbols slot
    _ = (semanticAcceptedSymbols interface offset env (count + 1)).getD
        slot.val 0 := by
      rw [semanticAcceptedSymbols_succ]

theorem valueTrace_of_specs (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (positionSpecs : ∀ round : Fin candidateCount,
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) env)
    (valueSpecs : ∀ round : Fin candidateCount,
      First54ValueStep.SpecHolds (valueInterface interface offset round.val)
        (valueOffset offset round.val) env)
    (count : Nat)
    (bounded : count ≤ candidateCount) :
    ValueTrace interface offset env count := by
  induction count with
  | zero => exact valueTrace_zero interface offset env
  | succ count inductionHypothesis =>
      apply valueTrace_succ interface offset env assumptions positionSpecs
        valueSpecs
        count (by omega)
      exact inductionHypothesis (by omega)

theorem valueTrace (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) (count : Nat)
    (bounded : count ≤ candidateCount) :
    ValueTrace interface offset env count :=
  valueTrace_of_specs interface offset env assumptions specification.position
    specification.value count bounded

theorem outputValue_eq_getD (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env)
    (slot : Fin outputCount) :
    (output offset slot).eval env =
      (semanticAcceptedSymbols interface offset env candidateCount).getD
        slot.val 0 := by
  have trace := valueTrace interface offset env assumptions specification
    candidateCount (Nat.le_refl _)
  simpa [output, priorOutput, candidateCount, outputCount] using trace slot

def evalOutput (env : Env) (offset : Nat) : List F :=
  List.ofFn fun slot : Fin outputCount => (output offset slot).eval env

@[simp] theorem evalOutput_length (env : Env) (offset : Nat) :
    (evalOutput env offset).length = outputCount := by
  simp [evalOutput]

theorem evalOutput_eq_firstAccepted (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    evalOutput env offset =
      FirstAccepted.firstAccepted (semanticVerifier interface offset env)
        outputCount (semanticCandidates candidateCount) := by
  let symbols := semanticAcceptedSymbols interface offset env candidateCount
  have enough := enough_of_spec interface offset env assumptions specification
  have symbolsLength : symbols.length =
      semanticAcceptedCount interface offset env candidateCount :=
    semanticAcceptedSymbols_length interface offset env candidateCount
  have enoughSymbols : outputCount ≤ symbols.length := by
    rw [symbolsLength]
    exact enough
  have enoughGeneric : FirstAccepted.Enough
      (semanticVerifier interface offset env) outputCount
        (semanticCandidates candidateCount) := by
    simpa [FirstAccepted.Enough, semanticAcceptedCount] using enough
  apply List.ext_get
  · rw [evalOutput_length]
    exact (FirstAccepted.firstAccepted_length_of_enough enoughGeneric).symm
  · intro index leftLt rightLt
    let slot : Fin outputCount := ⟨index, by
      simpa [evalOutput] using leftLt⟩
    have withinSymbols : index < symbols.length :=
      lt_of_lt_of_le slot.isLt enoughSymbols
    have value := outputValue_eq_getD interface offset env assumptions
      specification slot
    rw [List.getD_eq_get symbols 0 ⟨index, withinSymbols⟩] at value
    simpa [evalOutput, FirstAccepted.firstAccepted, semanticAcceptedSymbols,
      symbols, slot, List.get_eq_getElem] using value

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    FirstAccepted.boundedSample (semanticVerifier interface offset env)
        outputCount (semanticCandidates candidateCount) =
      some (evalOutput env offset) := by
  apply FirstAccepted.boundedSample_eq_some_iff.mpr
  refine ⟨?_, evalOutput_eq_firstAccepted interface offset env assumptions
    specification⟩
  simpa [FirstAccepted.Enough, semanticAcceptedCount] using
    enough_of_spec interface offset env assumptions specification

theorem rows_imply_boundedSample (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    FirstAccepted.boundedSample (semanticVerifier interface offset env)
        outputCount (semanticCandidates candidateCount) =
      some (evalOutput env offset) := by
  exact parentCoverage interface offset env assumptions
    (soundness interface env offset assumptions rows)

private theorem acceptedAt_eq_of_agree_below (interface : Interface)
    (offset : Nat) (initial current : Env)
    (assumptions : Assumptions interface offset initial)
    (agreeBelow : ∀ index, index < offset → current index = initial index)
    (round : Nat) :
    acceptedAt interface offset current round =
      acceptedAt interface offset initial round := by
  have evaluation := Expr.eval_eq_of_agree_below
    (interface.accepted offset (candidateIndex round)) offset current initial
      (assumptions.acceptedBelow (candidateIndex round)) agreeBelow
  simp [acceptedAt, semanticVerifier, evaluation]

private theorem semanticAcceptedCount_eq_of_agree_below
    (interface : Interface) (offset : Nat) (initial current : Env)
    (assumptions : Assumptions interface offset initial)
    (agreeBelow : ∀ index, index < offset → current index = initial index)
    (count : Nat) :
    semanticAcceptedCount interface offset current count =
      semanticAcceptedCount interface offset initial count := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      rw [semanticAcceptedCount_succ, semanticAcceptedCount_succ,
        inductionHypothesis,
        acceptedAt_eq_of_agree_below interface offset initial current
          assumptions agreeBelow count]

private theorem completedRoundSpecs (interface : Interface) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (completed : Sequence.Prefix env offset)
    (operationsEq : completed.operations =
      roundOpsPrefix interface offset candidateCount) :
    (∀ round : Fin candidateCount,
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) completed.current) ∧
    (∀ round : Fin candidateCount,
      First54ValueStep.SpecHolds (valueInterface interface offset round.val)
        (valueOffset offset round.val) completed.current) := by
  have currentAssumptions := assumptionsAtPrefix interface env offset assumptions
    completed
  have parentRows := holdsFlat_implies_holds completed.current
    completed.operations completed.rows
  constructor
  · intro round
    have member : positionOp interface offset round.val ∈
        completed.operations := by
      rw [operationsEq]
      apply List.mem_flatMap.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, by simp [roundOps]⟩
    have callHolds := parentRows _ member
    change First54Step.Assumptions
        (positionInterface interface offset round.val)
          (positionOffset offset round.val) completed.current →
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) completed.current at callHolds
    exact callHolds (positionAssumptions interface offset round.val
      completed.current currentAssumptions)
  · intro round
    have member : valueOp interface offset round.val ∈
        completed.operations := by
      rw [operationsEq]
      apply List.mem_flatMap.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, by simp [roundOps]⟩
    have callHolds := parentRows _ member
    change First54ValueStep.Assumptions
        (valueInterface interface offset round.val)
          (valueOffset offset round.val) completed.current →
      First54ValueStep.SpecHolds (valueInterface interface offset round.val)
        (valueOffset offset round.val) completed.current at callHolds
    exact callHolds (valueAssumptions interface offset round.val
      completed.current currentAssumptions)

theorem complete_of_enough (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (enough : outputCount ≤
      semanticAcceptedCount interface offset env candidateCount) :
    ∃ completed,
      AgreesOutside env completed offset logicalPrivateCount ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases completeRounds interface env offset candidateCount assumptions
      (Nat.le_refl _) with ⟨completedRounds, operationsEq⟩
  have currentAssumptions := assumptionsAtPrefix interface env offset assumptions
    completedRounds
  have agreeBelow : ∀ index, index < offset →
      completedRounds.current index = env index := by
    intro index below
    exact completedRounds.agrees index (Or.inl below)
  have countEq := semanticAcceptedCount_eq_of_agree_below interface offset env
    completedRounds.current assumptions agreeBelow candidateCount
  have currentEnough : outputCount ≤ semanticAcceptedCount interface offset
      completedRounds.current candidateCount := by
    rw [countEq]
    exact enough
  rcases completedRoundSpecs interface env offset assumptions completedRounds
      operationsEq with ⟨positionSpecs, valueSpecs⟩
  have trace := positionTrace_of_specs interface offset completedRounds.current
    currentAssumptions positionSpecs candidateCount (Nat.le_refl _)
  have minimum : min
      (semanticAcceptedCount interface offset completedRounds.current
        candidateCount) First54Step.fullSlot = First54Step.fullSlot :=
    Nat.min_eq_right (by simpa [outputCount, First54Step.fullSlot] using
      currentEnough)
  have selected : fullSlot.val = min
      (semanticAcceptedCount interface offset completedRounds.current
        candidateCount) First54Step.fullSlot := by
    simpa [fullSlot] using minimum.symm
  have full : (finalFull offset).eval completedRounds.current = 1 := by
    calc
      (finalFull offset).eval completedRounds.current =
          (priorPosition offset candidateCount fullSlot).eval
            completedRounds.current := by
        rfl
      _ = oneHotPosition
          (semanticAcceptedCount interface offset completedRounds.current
            candidateCount) fullSlot := trace fullSlot
      _ = 1 := by
        unfold oneHotPosition
        rw [if_pos selected]
  let specification : SpecHolds interface offset completedRounds.current :=
    ⟨positionSpecs, valueSpecs, full⟩
  rcases completeness interface completedRounds.current offset
      currentAssumptions specification with
    ⟨completed, completedAgrees, completedRows⟩
  have roundsLength : localLength completedRounds.operations =
      logicalPrivateCount := by
    rw [operationsEq]
    have total := localLength_eq interface offset
    change localLength
      (roundOpsPrefix interface offset candidateCount ++
        [finalAssertion offset]) = logicalPrivateCount at total
    simpa [Sequence.localLength_append, finalAssertion, Op.localLength] using total
  have roundsAgrees : AgreesOutside env completedRounds.current offset
      logicalPrivateCount := by
    have agreement := completedRounds.agrees
    rw [roundsLength] at agreement
    exact agreement
  have finalAgrees : AgreesOutside completedRounds.current completed offset
      logicalPrivateCount := by
    rw [localLength_eq] at completedAgrees
    exact completedAgrees
  refine ⟨completed, ?_, completedRows⟩
  intro index outside
  rw [finalAgrees index outside, roundsAgrees index outside]

def RelationHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  FirstAccepted.boundedSample (semanticVerifier interface offset env)
      outputCount (semanticCandidates candidateCount) =
    some (evalOutput env offset)

/-- The canonical high-level selector circuit. Its relation is exactly the
generic bounded sampler, and its operations are the same fixed logical rows. -/
def semanticCircuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := RelationHolds interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := by
    intro env offset assumptions rows
    exact rows_imply_boundedSample interface offset env assumptions rows
  completeness := by
    intro env offset assumptions relation
    have enough := (FirstAccepted.boundedSample_eq_some_iff.mp relation).1
    have enoughCount : outputCount ≤
        semanticAcceptedCount interface offset env candidateCount := by
      simpa [FirstAccepted.Enough, semanticAcceptedCount] using enough
    rcases complete_of_enough interface env offset assumptions enoughCount with
      ⟨completed, agrees, rows⟩
    refine ⟨completed, ?_, rows⟩
    rw [localLength_eq]
    exact agrees

end NightstreamFPrime.Gadgets.Sampling.First54
