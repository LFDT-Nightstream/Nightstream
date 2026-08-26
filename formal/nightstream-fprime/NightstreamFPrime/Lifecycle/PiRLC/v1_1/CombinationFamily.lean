import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep
import NightstreamFPrime.Lifecycle.Types

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1.
Obligation: compose the generic Phi81 accumulation child for all 17 PiRLC
sources in exact `K + k` order for one public value family.

The parent owns only source ordering, offsets, and accumulator wiring. It adds
no row between children. Family-specific wrappers select the block and cell
counts for commitment, public input, `Eval_K`, and `Eval_A`.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

abbrev sourceCount : Nat := productionShape.sourceCount

theorem sourceCount_eq : sourceCount = 17 := by
  rw [sourceCount, productionShape_sourceCount]
  rfl

@[simp] theorem castSource_eq_mk (source : Fin 17) :
    Fin.cast sourceCount_eq.symm source =
      ⟨source.val, by
        rw [sourceCount_eq]
        exact source.isLt⟩ := by
  apply Fin.ext
  rfl

def stepSize (blockCount cellCount : Nat) : Nat :=
  CombinationStep.privateCount blockCount cellCount

def logicalPrivateCount (blockCount cellCount : Nat) : Nat :=
  sourceCount * stepSize blockCount cellCount

def logicalRowCount (blockCount cellCount : Nat) : Nat :=
  logicalPrivateCount blockCount cellCount

structure Interface (blockCount cellCount : Nat) where
  challenge : Nat → Fin sourceCount → Fin ringDegree → Expr
  input : Nat → Fin sourceCount → Fin blockCount →
    Fin ringDegree → Fin cellCount → Expr

def stepOffset (offset source : Nat) (blockCount cellCount : Nat) : Nat :=
  offset + source * stepSize blockCount cellCount

def challengeAt {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (parentOffset source : Nat)
    (lane : Fin ringDegree) : Expr :=
  if sourceLt : source < sourceCount then
    interface.challenge parentOffset ⟨source, sourceLt⟩ lane
  else 0

def inputAt {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (parentOffset source : Nat)
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) : Expr :=
  if sourceLt : source < sourceCount then
    interface.input parentOffset ⟨source, sourceLt⟩ block lane cell
  else 0

def priorAt {blockCount cellCount : Nat} [NeZero cellCount]
    (parentOffset source : Nat) (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) : Expr :=
  if source = 0 then 0
  else
    CombinationStep.output
      (stepOffset parentOffset (source - 1) blockCount cellCount)
      (CombinationStep.indexOf block lane cell)

def stepInterface {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (parentOffset source : Nat) : CombinationStep.Interface blockCount cellCount where
  challenge := fun _ => challengeAt interface parentOffset source
  prior := fun _ => priorAt parentOffset source
  value := fun _ => inputAt interface parentOffset source

def stepCircuit {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (parentOffset source : Nat) : FormalCircuit :=
  CombinationStep.circuit (stepInterface interface parentOffset source)

def stepName (source : Nat) : String :=
  "pirlc.v1_1.combination.source_" ++ toString source

def stepOp {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (parentOffset source : Nat) : Op :=
  Sequence.childOp (stepName source)
    (stepCircuit interface parentOffset source)
    (stepOffset parentOffset source blockCount cellCount)

def opsPrefix {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (offset count : Nat) : List Op :=
  (List.range count).map (stepOp interface offset)

def opsAt {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) : List Op :=
  opsPrefix interface offset sourceCount

def main {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) : Circuit Unit := fun offset =>
  ((), offset + logicalPrivateCount blockCount cellCount,
    opsAt interface offset)

@[simp] theorem main_ops {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

@[simp] theorem opsAt_eq {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    opsAt interface offset =
      (List.range sourceCount).map (stepOp interface offset) := by
  rfl

def finalSource : Fin sourceCount :=
  ⟨sourceCount - 1, by rw [sourceCount_eq]; decide⟩

def output {blockCount cellCount : Nat} [NeZero cellCount]
    (_interface : Interface blockCount cellCount) (offset : Nat)
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) : Expr :=
  CombinationStep.output
    (stepOffset offset finalSource.val blockCount cellCount)
    (CombinationStep.indexOf block lane cell)

@[simp] private theorem stepOp_localLength
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (offset source : Nat) :
    (stepOp interface offset source).localLength =
      stepSize blockCount cellCount := by
  rw [stepOp, Sequence.childOp_localLength]
  exact CombinationStep.localLength_eq (stepInterface interface offset source)
    (stepOffset offset source blockCount cellCount)

@[simp] private theorem opsPrefix_localLength
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (offset count : Nat) :
    localLength (opsPrefix interface offset count) =
      count * stepSize blockCount cellCount := by
  simp [opsPrefix, localLength, Function.comp_def]

@[simp] private theorem stepOp_rowCount
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount)
    (offset source : Nat) :
    (stepOp interface offset source).rowCount =
      stepSize blockCount cellCount := by
  rfl

theorem localLength_eq
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      logicalPrivateCount blockCount cellCount := by
  change localLength (opsAt interface offset) = _
  simp [opsAt, logicalPrivateCount]

theorem flatConstraints_length
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount blockCount cellCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt interface offset) = _
  simp [opsAt, opsPrefix, rowCount, logicalRowCount, logicalPrivateCount,
    Function.comp_def]

structure Assumptions {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (_env : Env) : Prop where
  challengeBelow : ∀ source lane,
    (interface.challenge offset source lane).VarsBelow offset
  inputBelow : ∀ source block lane cell,
    (interface.input offset source block lane cell).VarsBelow offset

private theorem childAssumptions
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset source : Nat)
    (sourceLt : source < sourceCount) (env : Env)
    (assumptions : Assumptions interface offset env) :
    CombinationStep.Assumptions (stepInterface interface offset source)
      (stepOffset offset source blockCount cellCount) env := by
  constructor
  · intro lane
    change (challengeAt interface offset source lane).VarsBelow
      (stepOffset offset source blockCount cellCount)
    rw [challengeAt, dif_pos sourceLt]
    apply Expr.VarsBelow.mono _
      (assumptions.challengeBelow ⟨source, sourceLt⟩ lane)
    simp [stepOffset]
  · intro block lane cell
    unfold stepInterface priorAt
    by_cases first : source = 0
    · simp [first, Expr.VarsBelow]
    · simp only [if_neg first, CombinationStep.output, Expr.VarsBelow]
      have indexLt := (CombinationStep.indexOf block lane cell).isLt
      change
        stepOffset offset (source - 1) blockCount cellCount +
            (CombinationStep.indexOf block lane cell).val <
          stepOffset offset source blockCount cellCount
      calc
        stepOffset offset (source - 1) blockCount cellCount +
              (CombinationStep.indexOf block lane cell).val <
            stepOffset offset (source - 1) blockCount cellCount +
              stepSize blockCount cellCount :=
          Nat.add_lt_add_left indexLt _
        _ = stepOffset offset source blockCount cellCount := by
          unfold stepOffset
          rw [Nat.add_assoc]
          congr 1
          calc
            (source - 1) * stepSize blockCount cellCount +
                stepSize blockCount cellCount =
              (source - 1) * stepSize blockCount cellCount +
                1 * stepSize blockCount cellCount := by simp
            _ = ((source - 1) + 1) * stepSize blockCount cellCount := by
              rw [Nat.add_mul]
            _ = source * stepSize blockCount cellCount := by
              rw [Nat.sub_add_cancel
                (Nat.one_le_iff_ne_zero.mpr first)]
  · intro block lane cell
    change (inputAt interface offset source block lane cell).VarsBelow
      (stepOffset offset source blockCount cellCount)
    rw [inputAt, dif_pos sourceLt]
    apply Expr.VarsBelow.mono _
      (assumptions.inputBelow ⟨source, sourceLt⟩ block lane cell)
    simp [stepOffset]

private theorem childScope
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset source : Nat)
    (sourceLt : source < sourceCount) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (stepCircuit interface offset source).main
          (stepOffset offset source blockCount cellCount)),
      expression.VarsBelow
        (stepOffset offset source blockCount cellCount +
          localLength
            (Circuit.ops (stepCircuit interface offset source).main
              (stepOffset offset source blockCount cellCount))) := by
  change ∀ expression ∈ flatConstraints
      (CombinationStep.operations (stepInterface interface offset source)
        (stepOffset offset source blockCount cellCount)),
    expression.VarsBelow
      (stepOffset offset source blockCount cellCount +
        localLength
          (CombinationStep.operations (stepInterface interface offset source)
            (stepOffset offset source blockCount cellCount)))
  have scope := CombinationStep.flatConstraints_varsBelow
    (stepInterface interface offset source)
    (stepOffset offset source blockCount cellCount) env
    (childAssumptions interface offset source sourceLt env assumptions)
  rw [CombinationStep.localLength_eq]
  exact scope

theorem flatConstraints_varsBelow
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + logicalPrivateCount blockCount cellCount) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  intro expression member
  rcases List.mem_flatMap.mp member with
    ⟨operation, operationMember, expressionMember⟩
  rcases List.mem_map.mp operationMember with
    ⟨source, sourceMember, rfl⟩
  have sourceLt := List.mem_range.mp sourceMember
  apply Expr.VarsBelow.mono expression
    (childScope interface offset source sourceLt env assumptions expression (by
      simpa [stepOp, Sequence.childOp] using expressionMember))
  have lengthEq : localLength
      (Circuit.ops (stepCircuit interface offset source).main
        (stepOffset offset source blockCount cellCount)) =
        stepSize blockCount cellCount := by
    exact CombinationStep.localLength_eq (stepInterface interface offset source)
      (stepOffset offset source blockCount cellCount)
  rw [lengthEq]
  have sourceBound : source + 1 ≤ sourceCount :=
    Nat.succ_le_iff.mpr sourceLt
  have scaled := Nat.mul_le_mul_right (stepSize blockCount cellCount) sourceBound
  change
    offset + source * stepSize blockCount cellCount +
        stepSize blockCount cellCount ≤
      offset + sourceCount * stepSize blockCount cellCount
  calc
    offset + source * stepSize blockCount cellCount +
          stepSize blockCount cellCount =
        offset + (source + 1) * stepSize blockCount cellCount := by
      rw [Nat.add_assoc, Nat.add_mul]
      simp
    _ ≤ offset + sourceCount * stepSize blockCount cellCount :=
      Nat.add_le_add_left scaled offset

def evalOutputAt {blockCount cellCount : Nat} [NeZero cellCount]
    (env : Env) (offset : Nat) (source : Fin sourceCount)
    (block : Fin blockCount) (cell : Fin cellCount) : RingF :=
  fun lane =>
    (CombinationStep.output
      (stepOffset offset source.val blockCount cellCount)
      (CombinationStep.indexOf block lane cell)).eval env

def evalOutput {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (block : Fin blockCount) (cell : Fin cellCount) : RingF :=
  fun lane => (output interface offset block lane cell).eval env

def challengeValue {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (source : Fin sourceCount) : RingF :=
  fun lane => (interface.challenge offset source lane).eval env

def inputValue {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (source : Fin sourceCount) (block : Fin blockCount)
    (cell : Fin cellCount) : RingF :=
  fun lane => (interface.input offset source block lane cell).eval env

def term {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (source : Fin sourceCount) (block : Fin blockCount)
    (cell : Fin cellCount) : RingF :=
  ringFMul (challengeValue interface offset env source)
    (inputValue interface offset env source block cell)

def rightCombination : {count : Nat} →
    (Fin count → RingF) → RingF
  | 0, _ => ringFZero
  | _ + 1, terms =>
      ringFAdd (terms 0) (rightCombination fun index => terms index.succ)

@[simp] theorem rightCombination_zero (terms : Fin 0 → RingF) :
    rightCombination terms = ringFZero := by
  rfl

@[simp] theorem rightCombination_succ {count : Nat}
    (terms : Fin (count + 1) → RingF) :
    rightCombination terms =
      ringFAdd (terms 0)
        (rightCombination fun index : Fin count => terms index.succ) := by
  rfl

def orderedCombination {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (block : Fin blockCount) (cell : Fin cellCount) : RingF :=
  rightCombination (count := 17) fun source =>
    term interface offset env (Fin.cast sourceCount_eq.symm source) block cell

def accumulated {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (block : Fin blockCount) (cell : Fin cellCount) :
    Nat → RingF
  | 0 => ringFZero
  | count + 1 =>
      if countLt : count < sourceCount then
        ringFAdd (accumulated interface offset env block cell count)
          (term interface offset env ⟨count, countLt⟩ block cell)
      else accumulated interface offset env block cell count

def PrefixHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  ∀ source : Fin sourceCount,
    CombinationStep.SpecHolds
      (stepInterface interface offset source.val)
      (stepOffset offset source.val blockCount cellCount) env

def RelationHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  ∀ block cell,
    evalOutput interface offset env block cell =
      accumulated interface offset env block cell sourceCount

def CanonicalHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  ∀ block cell,
    evalOutput interface offset env block cell =
      orderedCombination interface offset env block cell

/-- The sequential child wiring is the paper's head-first `K + k` sum. -/
theorem accumulated_eq_orderedCombination
    {blockCount cellCount : Nat}
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (block : Fin blockCount) (cell : Fin cellCount) :
    accumulated interface offset env block cell sourceCount =
      orderedCombination interface offset env block cell := by
  funext lane
  simp [sourceCount_eq, accumulated, orderedCombination, ringFAdd, ringFZero]
  abel

private theorem childSpecs
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    PrefixHolds interface offset env := by
  intro source
  have member : stepOp interface offset source.val ∈
      opsAt interface offset := by
    apply List.mem_map.mpr
    exact ⟨source.val, List.mem_range.mpr source.isLt, rfl⟩
  have call := rows _ member
  change CombinationStep.Assumptions
      (stepInterface interface offset source.val)
        (stepOffset offset source.val blockCount cellCount) env →
    CombinationStep.SpecHolds
      (stepInterface interface offset source.val)
        (stepOffset offset source.val blockCount cellCount) env at call
  exact call (childAssumptions interface offset source.val source.isLt env assumptions)

private theorem outputAt_eq_accumulatedNat
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (prefixHolds : PrefixHolds interface offset env)
    (block : Fin blockCount) (cell : Fin cellCount) :
    ∀ count (countLt : count < sourceCount),
      evalOutputAt env offset ⟨count, countLt⟩ block cell =
        accumulated interface offset env block cell (count + 1) := by
  intro count
  induction count with
  | zero =>
      intro countLt
      funext lane
      have step := prefixHolds ⟨0, countLt⟩
        (CombinationStep.indexOf block lane cell)
      have normalized :
          evalOutputAt env offset ⟨0, countLt⟩ block cell lane =
            (0 : Expr).eval env +
              term interface offset env ⟨0, countLt⟩ block cell lane := by
        simpa [evalOutputAt, term, challengeValue, inputValue,
          stepInterface, priorAt, challengeAt, inputAt, countLt] using step
      rw [show (0 : Expr).eval env = (0 : F) by rfl, Fin.zero_add] at normalized
      simpa [accumulated, countLt, ringFAdd, ringFZero] using normalized
  | succ count inductionHypothesis =>
      intro countLt
      have previousLt : count < sourceCount := by omega
      have previous := inductionHypothesis previousLt
      funext lane
      have step := prefixHolds ⟨count + 1, countLt⟩
        (CombinationStep.indexOf block lane cell)
      have normalized :
          evalOutputAt env offset ⟨count + 1, countLt⟩ block cell lane =
            evalOutputAt env offset ⟨count, previousLt⟩ block cell lane +
              term interface offset env ⟨count + 1, countLt⟩ block cell lane := by
        simpa [evalOutputAt, term, challengeValue, inputValue,
          stepInterface, priorAt, challengeAt, inputAt, countLt,
          previousLt] using step
      rw [normalized, congrFun previous lane]
      simp [accumulated, countLt, ringFAdd]

theorem parentCoverage
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (prefixHolds : PrefixHolds interface offset env) :
    RelationHolds interface offset env := by
  intro block cell
  have final := outputAt_eq_accumulatedNat interface offset env prefixHolds
    block cell finalSource.val finalSource.isLt
  simpa [evalOutput, output, evalOutputAt, finalSource, sourceCount_eq] using final

theorem rows_imply_relation
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    RelationHolds interface offset env :=
  parentCoverage interface offset env
    (childSpecs interface offset env assumptions rows)

theorem relation_implies_canonical
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (relation : RelationHolds interface offset env) :
    CanonicalHolds interface offset env := by
  intro block cell
  calc
    evalOutput interface offset env block cell =
        accumulated interface offset env block cell sourceCount :=
      relation block cell
    _ = orderedCombination interface offset env block cell :=
      accumulated_eq_orderedCombination interface offset env block cell

theorem soundness
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    CanonicalHolds interface offset env :=
  relation_implies_canonical interface offset env
    (rows_imply_relation interface offset env assumptions rows)

private theorem completePrefix
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (count : Nat) (bounded : count ≤ sourceCount) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsPrefix interface offset count := by
  induction count with
  | zero =>
      exact ⟨Sequence.empty env offset, rfl⟩
  | succ count inductionHypothesis =>
      have countLt : count < sourceCount := by omega
      rcases inductionHypothesis (by omega) with ⟨before, beforeOperations⟩
      have startEq : offset + localLength before.operations =
          stepOffset offset count blockCount cellCount := by
        rw [beforeOperations, opsPrefix_localLength]
        rfl
      have currentAssumptions :
          Assumptions interface offset before.current := {
        challengeBelow := assumptions.challengeBelow
        inputBelow := assumptions.inputBelow
      }
      have childAssumptionsNow := childAssumptions interface offset count
        countLt before.current currentAssumptions
      rcases CombinationStep.complete
          (stepInterface interface offset count) before.current
          (stepOffset offset count blockCount cellCount)
          childAssumptionsNow with
        ⟨childEnv, childAgrees, childRows⟩
      rcases Sequence.appendBuiltAt before (stepName count)
          (stepCircuit interface offset count)
          (stepOffset offset count blockCount cellCount) startEq
          (childScope interface offset count countLt before.current
            currentAssumptions)
          childEnv childAgrees childRows with
        ⟨completed, operationsEq, _, _, _⟩
      refine ⟨completed, ?_⟩
      rw [operationsEq, beforeOperations]
      simp [opsPrefix, List.range_succ, stepOp]

theorem complete
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases completePrefix interface offset env assumptions sourceCount
      (Nat.le_refl _) with ⟨completed, operationsEq⟩
  refine ⟨completed.current, ?_, ?_⟩
  · have agrees := completed.agrees
    rw [operationsEq] at agrees
    change AgreesOutside env completed.current offset
      (localLength (opsAt interface offset))
    simpa [opsAt] using agrees
  · change holdsFlat completed.current
      (opsPrefix interface offset sourceCount)
    rw [← operationsEq]
    exact completed.rows

theorem completeness
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (_specification : CanonicalHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  complete interface offset env assumptions

def circuit {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := CanonicalHolds interface
  privateCount := fun _ => logicalPrivateCount blockCount cellCount
  rowCount := fun _ => logicalRowCount blockCount cellCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := by
    intro env offset assumptions rows
    exact soundness interface offset env assumptions rows
  completeness := by
    intro env offset assumptions specification
    exact completeness interface offset env assumptions specification

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily
