import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler
import NightstreamFPrime.Lifecycle.Types

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1.
Obligation: derive all 17 PiRLC challenges in exact `K + k` order from one
post-PiCCS transcript state.

The parent threads each child-owned outgoing state into the next opaque scalar
sampler. It exports centered `RingF` challenges as zero-row views of each
selector output. It adds no copy, boundary, or assertion row.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

abbrev EState := Sampler.EState
abbrev State := Sampler.State
abbrev sourceCount : Nat := productionShape.sourceCount

theorem sourceCount_eq : sourceCount = 17 := by
  rw [sourceCount, productionShape_sourceCount]
  rfl

def logicalPrivateCount : Nat := sourceCount * Sampler.logicalPrivateCount
def logicalRowCount : Nat := sourceCount * Sampler.logicalRowCount

theorem logicalPrivateCount_eq : logicalPrivateCount = 263568 := by
  rw [logicalPrivateCount, sourceCount_eq]
  norm_num [Sampler.logicalPrivateCount]

theorem logicalRowCount_eq : logicalRowCount = 265217 := by
  rw [logicalRowCount, sourceCount_eq]
  norm_num [Sampler.logicalRowCount]

structure Interface where
  initialState : Nat → EState

def sourceOffset (offset source : Nat) : Nat :=
  offset + source * Sampler.logicalPrivateCount

def stateAtExpr (interface : Interface) (offset : Nat) : Nat → EState
  | 0 => interface.initialState offset
  | source + 1 =>
      Sampler.outputState
        { initialState := fun _ => stateAtExpr interface offset source }
        source (sourceOffset offset source)

def childInterface (interface : Interface) (offset source : Nat) :
    Sampler.Interface where
  initialState := fun _ => stateAtExpr interface offset source

@[simp] theorem stateAtExpr_zero (interface : Interface) (offset : Nat) :
    stateAtExpr interface offset 0 = interface.initialState offset := by
  rfl

@[simp] theorem stateAtExpr_succ (interface : Interface) (offset source : Nat) :
    stateAtExpr interface offset (source + 1) =
      Sampler.outputState (childInterface interface offset source) source
        (sourceOffset offset source) := by
  rfl

def challengeExpr (_interface : Interface) (offset : Nat)
    (source : Fin sourceCount) : Fin ringDegree → Expr :=
  Sampler.outputChallenge (sourceOffset offset source.val)

def evalChallenge (_interface : Interface) (offset : Nat) (env : Env)
    (source : Fin sourceCount) : RingF :=
  Sampler.evalOutputChallenge env (sourceOffset offset source.val)

def evalChallenges (interface : Interface) (offset : Nat) (env : Env) :
    Fin sourceCount → RingF :=
  evalChallenge interface offset env

def finalStateExpr (interface : Interface) (offset : Nat) : EState :=
  stateAtExpr interface offset sourceCount

def evalStateAt (interface : Interface) (offset : Nat) (env : Env)
    (source : Nat) : State :=
  Sampler.evalState env (stateAtExpr interface offset source)

def evalInitialState (interface : Interface) (offset : Nat) (env : Env) : State :=
  evalStateAt interface offset env 0

def evalFinalState (interface : Interface) (offset : Nat) (env : Env) : State :=
  evalStateAt interface offset env sourceCount

def childName (source : Nat) : String :=
  "pirlc.v1_1.sampler_chain.source_" ++ toString source

def childOp (interface : Interface) (offset source : Nat) : Op :=
  Sequence.childOp (childName source)
    (Sampler.circuit (childInterface interface offset source) source)
    (sourceOffset offset source)

def opsPrefix (interface : Interface) (offset count : Nat) : List Op :=
  (List.range count).map (childOp interface offset)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  opsPrefix interface offset sourceCount

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + logicalPrivateCount, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

@[simp] theorem opsAt_eq (interface : Interface) (offset : Nat) :
    opsAt interface offset =
      (List.range sourceCount).map (childOp interface offset) := by
  rfl

@[simp] private theorem childOp_localLength (interface : Interface)
    (offset source : Nat) :
    (childOp interface offset source).localLength =
      Sampler.logicalPrivateCount := by
  rw [childOp, Sequence.childOp_localLength]
  exact Sampler.localLength_eq (childInterface interface offset source) source
    (sourceOffset offset source)

@[simp] private theorem childOp_rowCount (interface : Interface)
    (offset source : Nat) :
    (childOp interface offset source).rowCount = Sampler.logicalRowCount := by
  rfl

@[simp] private theorem opsPrefix_localLength (interface : Interface)
    (offset count : Nat) :
    localLength (opsPrefix interface offset count) =
      count * Sampler.logicalPrivateCount := by
  simp [opsPrefix, localLength, Function.comp_def]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = logicalPrivateCount := by
  change localLength (opsAt interface offset) = logicalPrivateCount
  simp [opsAt, logicalPrivateCount]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt interface offset) = logicalRowCount
  simp [opsAt, opsPrefix, rowCount, logicalRowCount, Function.comp_def]

structure Assumptions (interface : Interface) (offset : Nat)
    (_env : Env) : Prop where
  initialBelow : ∀ lane, (interface.initialState offset lane).VarsBelow offset

theorem stateAtExpr_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ count, count ≤ sourceCount → ∀ lane,
      (stateAtExpr interface offset count lane).VarsBelow
        (sourceOffset offset count) := by
  intro count bounded
  induction count with
  | zero =>
      intro lane
      simpa [sourceOffset] using assumptions.initialBelow lane
  | succ count inductionHypothesis =>
      have countLt : count < sourceCount := by omega
      have childAssumptions : Sampler.Assumptions
          (childInterface interface offset count) (sourceOffset offset count) env := by
        intro lane
        exact inductionHypothesis (Nat.le_of_lt countLt) lane
      intro lane
      have scope := Sampler.outputState_varsBelow
        (childInterface interface offset count) count (sourceOffset offset count)
        env childAssumptions lane
      simpa [stateAtExpr_succ, sourceOffset, Nat.succ_mul,
        Nat.add_assoc] using scope

theorem childAssumptions (interface : Interface) (offset source : Nat)
    (sourceLt : source < sourceCount) (env : Env)
    (assumptions : Assumptions interface offset env) :
    Sampler.Assumptions (childInterface interface offset source)
      (sourceOffset offset source) env := by
  intro lane
  exact stateAtExpr_varsBelow interface offset env assumptions source
    (Nat.le_of_lt sourceLt) lane

theorem challengeExpr_varsBelow (interface : Interface) (offset : Nat)
    (source : Fin sourceCount) (lane : Fin ringDegree) :
    (challengeExpr interface offset source lane).VarsBelow
      (sourceOffset offset (source.val + 1)) := by
  simpa [challengeExpr, sourceOffset, Nat.succ_mul, Nat.add_assoc] using
    Sampler.outputChallenge_varsBelow (sourceOffset offset source.val) lane

theorem challengeExpr_eval (interface : Interface) (offset : Nat) (env : Env)
    (source : Fin sourceCount) (lane : Fin ringDegree) :
    (challengeExpr interface offset source lane).eval env =
      evalChallenge interface offset env source lane := by
  calc
    (challengeExpr interface offset source lane).eval env =
        (Sampler.outputWord (sourceOffset offset source.val) lane).eval env - 2 :=
      Expr.eval_sub env _ _
    _ = evalChallenge interface offset env source lane := by
      exact (Sampler.evalOutputChallenge_apply env
        (sourceOffset offset source.val) lane).symm

private theorem childScope (interface : Interface) (offset source : Nat)
    (sourceLt : source < sourceCount) (env : Env)
    (assumptions : Assumptions interface offset env)
    (internal : Sampler.SpecHolds (childInterface interface offset source)
      source (sourceOffset offset source) env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops
          (Sampler.circuit (childInterface interface offset source) source).main
          (sourceOffset offset source)),
      expression.VarsBelow
        (sourceOffset offset source + Sampler.logicalPrivateCount) := by
  exact Sampler.flatConstraints_varsBelow
    (childInterface interface offset source) source (sourceOffset offset source)
    env (childAssumptions interface offset source sourceLt env assumptions) internal

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (internal : ∀ source : Fin sourceCount,
      Sampler.SpecHolds (childInterface interface offset source.val) source.val
        (sourceOffset offset source.val) env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow (offset + logicalPrivateCount) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  intro expression member
  rcases List.mem_flatMap.mp member with
    ⟨operation, operationMember, expressionMember⟩
  rcases List.mem_map.mp operationMember with ⟨source, sourceMember, rfl⟩
  have sourceLt := List.mem_range.mp sourceMember
  apply Expr.VarsBelow.mono expression
    (childScope interface offset source sourceLt env assumptions
      (internal ⟨source, sourceLt⟩) expression (by
        simpa [childOp, Sequence.childOp] using expressionMember))
  have scaled := Nat.mul_le_mul_right Sampler.logicalPrivateCount
    (Nat.succ_le_iff.mpr sourceLt)
  simpa [sourceOffset, logicalPrivateCount, Nat.succ_mul, Nat.add_assoc] using
    Nat.add_le_add_left scaled offset

/-- Completed chain rows provide the internal scalar specifications needed
by the scalar scope proof. -/
theorem flatConstraints_varsBelow_of_rows (interface : Interface)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holdsFlat env (Circuit.ops (main interface) offset)) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow (offset + logicalPrivateCount) := by
  apply flatConstraints_varsBelow interface offset env assumptions
  intro source
  have member : childOp interface offset source.val ∈ opsAt interface offset := by
    apply List.mem_map.mpr
    exact ⟨source.val, List.mem_range.mpr source.isLt, rfl⟩
  have childRows : holdsFlat env
      (Circuit.ops
        (Sampler.circuit (childInterface interface offset source.val)
          source.val).main (sourceOffset offset source.val)) := by
    intro expression expressionMember
    apply rows expression
    change expression ∈ flatConstraints (opsAt interface offset)
    apply List.mem_flatMap.mpr
    refine ⟨childOp interface offset source.val, member, ?_⟩
    simpa [childOp, Sequence.childOp, Op.flatConstraints] using expressionMember
  exact Sampler.soundness (childInterface interface offset source.val)
    source.val env (sourceOffset offset source.val)
    (childAssumptions interface offset source.val source.isLt env assumptions)
    (holdsFlat_implies_holds env _ childRows)

def ChildHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ source : Fin sourceCount,
    Sampler.RelationHolds (childInterface interface offset source.val) source.val
      (sourceOffset offset source.val) env

structure RelationHolds (interface : Interface) (offset : Nat)
    (env : Env) : Prop where
  child : ChildHolds interface offset env
  response :
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
        (evalInitialState interface offset env) sourceCount =
      some (evalChallenges interface offset env)
  finalState :
    evalFinalState interface offset env =
      stateAt NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
        (evalInitialState interface offset env) sourceCount

private theorem childHolds_of_rows (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    ChildHolds interface offset env := by
  intro source
  have member : childOp interface offset source.val ∈ opsAt interface offset := by
    apply List.mem_map.mpr
    exact ⟨source.val, List.mem_range.mpr source.isLt, rfl⟩
  have call := rows _ member
  change Sampler.Assumptions (childInterface interface offset source.val)
      (sourceOffset offset source.val) env →
    Sampler.RelationHolds (childInterface interface offset source.val) source.val
      (sourceOffset offset source.val) env at call
  exact call (childAssumptions interface offset source.val source.isLt env assumptions)

theorem evalStateAt_eq_stateAt (interface : Interface) (offset : Nat)
    (env : Env) (children : ChildHolds interface offset env) :
    ∀ count, count ≤ sourceCount →
      evalStateAt interface offset env count =
        stateAt NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
          (evalInitialState interface offset env) count := by
  intro count bounded
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      have countLt : count < sourceCount := by omega
      rcases children ⟨count, countLt⟩ with
        ⟨coefficients, success, outputEq, stateEq⟩
      calc
        evalStateAt interface offset env (count + 1) =
            Sampler.evalState env
              (Sampler.outputState (childInterface interface offset count) count
                (sourceOffset offset count)) := by rfl
        _ = (Sampler.productionSource (childInterface interface offset count)
              count (sourceOffset offset count) env).nextState := stateEq
        _ = (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification.source
              (evalStateAt interface offset env count) count).nextState := by rfl
        _ = (sourceAt
              NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
              (evalInitialState interface offset env) count).nextState := by
          rw [inductionHypothesis (Nat.le_of_lt countLt)]
          rfl
        _ = stateAt
              NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
              (evalInitialState interface offset env) (count + 1) := by rfl

theorem sampleRingChallenge_eq (interface : Interface) (offset : Nat)
    (env : Env) (children : ChildHolds interface offset env)
    (source : Fin sourceCount) :
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.sampleRingChallenge
        (evalInitialState interface offset env) source.val =
      some (evalChallenge interface offset env source) := by
  rcases Sampler.relation_implies_outputChallenge
      (childInterface interface offset source.val) source.val
      (sourceOffset offset source.val) env (children source) with
    ⟨coefficients, success, challengeEq, _⟩
  have stateEq := evalStateAt_eq_stateAt interface offset env children source.val
    (Nat.le_of_lt source.isLt)
  have successAt :
      FirstAccepted.boundedSample verifier coefficientCount
          (FirstAccepted.streamPrefix
            (sourceAt
              NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
              (evalInitialState interface offset env) source.val).stream
            candidateBound) = some coefficients := by
    have directSuccess :
        FirstAccepted.boundedSample verifier coefficientCount
            (FirstAccepted.streamPrefix
              (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification.source
                (evalStateAt interface offset env source.val) source.val).stream
              candidateBound) = some coefficients := by
      simpa only [Sampler.productionCandidates, Sampler.productionSource,
        Sampler.evalInitialState, childInterface, evalStateAt] using success
    unfold sourceAt
    rw [← stateEq]
    exact directSuccess
  have scalarEq :
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.sampleScalar
          (evalInitialState interface offset env) source.val =
        some
          (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.scalarOfList
            coefficients) := by
    unfold NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.sampleScalar
    simp only [successAt, Option.map_some]
  unfold NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.sampleRingChallenge
  rw [scalarEq]
  simp only [Option.map_some, Option.some.injEq]
  exact challengeEq.symm

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (children : ChildHolds interface offset env) :
    RelationHolds interface offset env := by
  refine ⟨children, ?_, ?_⟩
  · exact
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges_eq_some_of_pointwise
        (evalInitialState interface offset env)
        (evalChallenges interface offset env)
        (sampleRingChallenge_eq interface offset env children)
  · exact evalStateAt_eq_stateAt interface offset env children sourceCount
      (Nat.le_refl _)

theorem soundness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    RelationHolds interface offset env :=
  parentCoverage interface offset env
    (childHolds_of_rows interface offset env assumptions rows)

private theorem evalStateAt_eq_of_agree_below (interface : Interface)
    (offset count : Nat) (before after : Env)
    (scope : ∀ lane, (stateAtExpr interface offset count lane).VarsBelow
      (sourceOffset offset count))
    (agrees : ∀ index, index < sourceOffset offset count →
      after index = before index) :
    evalStateAt interface offset after count =
      evalStateAt interface offset before count := by
  unfold evalStateAt Sampler.evalState
  apply congrArg List.ofFn
  funext lane
  exact (stateAtExpr interface offset count lane).eval_eq_of_agree_below
    (sourceOffset offset count) after before (scope lane) agrees

private theorem completePrefix (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (children : ChildHolds interface offset env)
    (count : Nat) (bounded : count ≤ sourceCount) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsPrefix interface offset count ∧
      evalStateAt interface offset completed.current count =
        evalStateAt interface offset env count := by
  induction count with
  | zero =>
      exact ⟨Sequence.empty env offset, rfl, rfl⟩
  | succ count inductionHypothesis =>
      have countLt : count < sourceCount := by omega
      rcases inductionHypothesis (Nat.le_of_lt countLt) with
        ⟨before, beforeOperations, statePreserved⟩
      have startEq : offset + localLength before.operations =
          sourceOffset offset count := by
        rw [beforeOperations, opsPrefix_localLength]
        rfl
      have currentAssumptions : Assumptions interface offset before.current :=
        ⟨assumptions.initialBelow⟩
      have childAssumptionsNow := childAssumptions interface offset count countLt
        before.current currentAssumptions
      rcases children ⟨count, countLt⟩ with
        ⟨coefficients, originalSuccess, originalOutput, originalState⟩
      have currentInitialEq : Sampler.evalInitialState
          (childInterface interface offset count) (sourceOffset offset count)
          before.current =
        Sampler.evalInitialState (childInterface interface offset count)
          (sourceOffset offset count) env := by
        simpa [Sampler.evalInitialState, childInterface, evalStateAt] using
          statePreserved
      have currentSuccess : Sampler.SamplingSucceeds
          (childInterface interface offset count) count
          (sourceOffset offset count) before.current := by
        refine ⟨coefficients, ?_⟩
        simpa [Sampler.productionCandidates, Sampler.productionSource,
          currentInitialEq] using originalSuccess
      rcases Sampler.complete_of_success
          (childInterface interface offset count) count before.current
          (sourceOffset offset count) childAssumptionsNow currentSuccess with
        ⟨childEnv, childAgrees, childRows⟩
      have builtParentAssumptions : Assumptions interface offset childEnv :=
        ⟨assumptions.initialBelow⟩
      have builtAssumptions := childAssumptions interface offset count countLt
        childEnv builtParentAssumptions
      have childRowsHolds := holdsFlat_implies_holds childEnv
        (Circuit.ops
          (Sampler.circuit (childInterface interface offset count) count).main
          (sourceOffset offset count)) childRows
      have internal := Sampler.soundness (childInterface interface offset count)
        count childEnv (sourceOffset offset count) builtAssumptions childRowsHolds
      have exactScope := childScope interface offset count countLt childEnv
        builtParentAssumptions internal
      have scope : ∀ expression ∈ flatConstraints
          (Circuit.ops
            (Sampler.circuit (childInterface interface offset count) count).main
            (sourceOffset offset count)),
        expression.VarsBelow
          (sourceOffset offset count +
            localLength (Circuit.ops
              (Sampler.circuit (childInterface interface offset count) count).main
              (sourceOffset offset count))) := by
        have childLength : localLength (Circuit.ops
            (Sampler.circuit (childInterface interface offset count) count).main
            (sourceOffset offset count)) = Sampler.logicalPrivateCount := by
          change localLength (Circuit.ops
            (Sampler.main (childInterface interface offset count) count)
              (sourceOffset offset count)) = Sampler.logicalPrivateCount
          exact Sampler.localLength_eq (childInterface interface offset count)
            count (sourceOffset offset count)
        simpa only [childLength] using exactScope
      rcases Sequence.appendBuiltAt before (childName count)
          (Sampler.circuit (childInterface interface offset count) count)
          (sourceOffset offset count)
          startEq scope childEnv childAgrees childRows with
        ⟨completed, completedOperations, _, preserves, completedChildRows⟩
      have operationsEq : completed.operations =
          opsPrefix interface offset (count + 1) := by
        rw [completedOperations, beforeOperations]
        simp [opsPrefix, List.range_succ, childOp]
      have completedAssumptions : Assumptions interface offset completed.current :=
        ⟨assumptions.initialBelow⟩
      have completedChildAssumptions := childAssumptions interface offset count
        countLt completed.current completedAssumptions
      have completedRelation := Sampler.rows_imply_relation
        (childInterface interface offset count) count (sourceOffset offset count)
        completed.current completedChildAssumptions
        (holdsFlat_implies_holds completed.current
          (Circuit.ops
            (Sampler.circuit (childInterface interface offset count) count).main
            (sourceOffset offset count)) completedChildRows)
      rcases completedRelation with
        ⟨_, _, _, completedState⟩
      have priorScope := stateAtExpr_varsBelow interface offset before.current
        currentAssumptions count (Nat.le_of_lt countLt)
      have completedPrior :
          evalStateAt interface offset completed.current count =
            evalStateAt interface offset before.current count := by
        apply evalStateAt_eq_of_agree_below interface offset count
          before.current completed.current priorScope
        intro index below
        exact preserves.values index (by simpa [startEq] using below)
      have initialEq : Sampler.evalInitialState
          (childInterface interface offset count) (sourceOffset offset count)
          completed.current =
        Sampler.evalInitialState (childInterface interface offset count)
          (sourceOffset offset count) env := by
        calc
          _ = evalStateAt interface offset completed.current count := by rfl
          _ = evalStateAt interface offset before.current count := completedPrior
          _ = evalStateAt interface offset env count := statePreserved
          _ = _ := by rfl
      refine ⟨completed, operationsEq, ?_⟩
      calc
        evalStateAt interface offset completed.current (count + 1) =
            Sampler.evalState completed.current
              (Sampler.outputState (childInterface interface offset count) count
                (sourceOffset offset count)) := by rfl
        _ = (Sampler.productionSource (childInterface interface offset count)
              count (sourceOffset offset count) completed.current).nextState :=
          completedState
        _ = (Sampler.productionSource (childInterface interface offset count)
              count (sourceOffset offset count) env).nextState := by
          unfold Sampler.productionSource
          rw [initialEq]
        _ = Sampler.evalState env
              (Sampler.outputState (childInterface interface offset count) count
                (sourceOffset offset count)) := originalState.symm
        _ = evalStateAt interface offset env (count + 1) := by rfl

theorem completeness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (relation : RelationHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases completePrefix interface offset env assumptions relation.child sourceCount
      (Nat.le_refl _) with ⟨completed, operationsEq, _⟩
  refine ⟨completed.current, ?_, ?_⟩
  · have agrees := completed.agrees
    rw [operationsEq] at agrees
    change AgreesOutside env completed.current offset
      (localLength (opsAt interface offset))
    simpa [opsAt] using agrees
  · change holdsFlat completed.current (opsPrefix interface offset sourceCount)
    rw [← operationsEq]
    exact completed.rows

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := RelationHolds interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := by
    intro env offset assumptions rows
    exact soundness interface offset env assumptions rows
  completeness := by
    intro env offset assumptions relation
    exact completeness interface offset env assumptions relation

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain
