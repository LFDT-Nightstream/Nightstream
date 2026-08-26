import NightstreamFPrime.Gadgets.Sampling.First54.Semantics
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption

/-!
Owns one complete PiRLC scalar sampler.

The parent enters the verifier-owned scalar domain, executes all eight digest
windows, and then calls the exact first-54 selector over all 64 candidates.
It consumes every window even when 54 candidates appear early.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

def digestRoundCount : Nat := 8
def entryPrivateCount : Nat := 592
def logicalPrivateCount : Nat := 15504
def logicalRowCount : Nat := 15601

abbrev State := NightstreamFPrime.Lifecycle.Transcript.State
abbrev EState := NightstreamFPrime.Gadgets.Poseidon2.Layer.EState

structure Interface where
  initialState : Nat → EState

def entryInterface (interface : Interface) : TranscriptAbsorption.Interface where
  initialState := interface.initialState

def entryOffset (offset : Nat) : Nat := offset

def windowBase (offset : Nat) : Nat := offset + entryPrivateCount

def windowOffset (offset round : Nat) : Nat :=
  windowBase offset + round * DigestWindow.logicalPrivateCount

def selectorOffset (offset : Nat) : Nat :=
  windowBase offset + digestRoundCount * DigestWindow.logicalPrivateCount

def finalOffset (offset : Nat) : Nat := offset + logicalPrivateCount

def windowInitialState (interface : Interface) (coordinate offset round : Nat) :
    EState :=
  match round with
  | 0 => TranscriptAbsorption.output (entryInterface interface) coordinate
      (entryOffset offset)
  | previous + 1 =>
      DigestWindow.output
        { initialState := fun _ =>
            windowInitialState interface coordinate offset previous }
        (windowOffset offset previous)

def windowInterface (interface : Interface) (coordinate offset round : Nat) :
    DigestWindow.Interface where
  initialState := fun _ => windowInitialState interface coordinate offset round

def candidateRound (candidate : Fin First54.candidateCount) :
    Fin digestRoundCount :=
  ⟨candidate.val / chunksPerDigest, by
    have bounded := candidate.isLt
    simp only [First54.candidateCount, chunksPerDigest, digestRoundCount] at bounded ⊢
    omega⟩

def candidatePosition (candidate : Fin First54.candidateCount) :
    Fin chunksPerDigest :=
  ⟨candidate.val % chunksPerDigest, Nat.mod_lt _ (by decide)⟩

def selectorInterface (interface : Interface) (coordinate offset : Nat) :
    First54.Interface where
  accepted := fun _ candidate =>
    1 - DigestWindow.reject
      (windowOffset offset (candidateRound candidate).val)
        (candidatePosition candidate)
  symbol := fun _ candidate =>
    DigestWindow.remainder
      (windowOffset offset (candidateRound candidate).val)
        (candidatePosition candidate)

def entryCircuit (interface : Interface) (coordinate : Nat) : FormalCircuit :=
  TranscriptAbsorption.circuit (entryInterface interface) coordinate

def windowCircuit (interface : Interface) (coordinate offset round : Nat) :
    FormalCircuit :=
  DigestWindow.circuit (windowInterface interface coordinate offset round)

def selectorCircuit (interface : Interface) (coordinate offset : Nat) :
    FormalCircuit :=
  First54.semanticCircuit (selectorInterface interface coordinate offset)

def entryName : String := "pirlc.v1_1.sampler.enter_scalar"

def windowName (round : Nat) : String :=
  "pirlc.v1_1.sampler.digest_window_" ++ toString round

def selectorName : String := "pirlc.v1_1.sampler.first_54"

def entryOp (interface : Interface) (coordinate offset : Nat) : Op :=
  Sequence.childOp entryName (entryCircuit interface coordinate)
    (entryOffset offset)

def windowOp (interface : Interface) (coordinate offset round : Nat) : Op :=
  Sequence.childOp (windowName round)
    (windowCircuit interface coordinate offset round) (windowOffset offset round)

def windowOps (interface : Interface) (coordinate offset : Nat) : List Op :=
  (List.range digestRoundCount).map (windowOp interface coordinate offset)

def windowOpsPrefix (interface : Interface) (coordinate offset count : Nat) :
    List Op :=
  (List.range count).map (windowOp interface coordinate offset)

def selectorOp (interface : Interface) (coordinate offset : Nat) : Op :=
  Sequence.childOp selectorName (selectorCircuit interface coordinate offset)
    (selectorOffset offset)

def opsAt (interface : Interface) (coordinate offset : Nat) : List Op :=
  [entryOp interface coordinate offset] ++
    windowOps interface coordinate offset ++
    [selectorOp interface coordinate offset]

def main (interface : Interface) (coordinate : Nat) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt interface coordinate offset)

def outputState (interface : Interface) (coordinate offset : Nat) : EState :=
  DigestWindow.output
    (windowInterface interface coordinate offset (digestRoundCount - 1))
      (windowOffset offset (digestRoundCount - 1))

def outputCoefficients (env : Env) (offset : Nat) : List F :=
  First54.evalOutput env (selectorOffset offset)

theorem outputCount_eq_ringDegree : First54.outputCount = ringDegree := by
  rfl

def outputSlot (position : Fin ringDegree) : Fin First54.outputCount :=
  Fin.cast outputCount_eq_ringDegree.symm position

/-- One selected five-symbol residue, before centered embedding. -/
def outputWord (offset : Nat) (position : Fin ringDegree) : Expr :=
  First54.output (selectorOffset offset) (outputSlot position)

/-- The sampled PiRLC challenge as a zero-row centered view of the selector
output. -/
def outputChallenge (offset : Nat) : Fin ringDegree → Expr :=
  fun position => outputWord offset position - 2

def evalOutputChallenge (env : Env) (offset : Nat) : RingF :=
  fun position => (outputChallenge offset position).eval env

theorem evalOutputChallenge_apply (env : Env) (offset : Nat)
    (position : Fin ringDegree) :
    evalOutputChallenge env offset position =
      (outputWord offset position).eval env - 2 := by
  exact Expr.eval_sub env (outputWord offset position) 2

theorem outputChallenge_varsBelow (offset : Nat) (position : Fin ringDegree) :
    (outputChallenge offset position).VarsBelow
      (offset + logicalPrivateCount) := by
  unfold outputChallenge outputWord First54.output First54ValueStep.output
  apply Expr.VarsBelow.sub
  · simp only [Expr.VarsBelow]
    have positionLt := position.isLt
    simp [outputSlot, First54.valueOffset, First54.positionOffset,
      First54.candidateCount, First54.roundPrivateCount,
      First54Step.slotCount, First54ValueStep.outputCount,
      selectorOffset, windowBase, digestRoundCount, entryPrivateCount,
      DigestWindow.logicalPrivateCount, logicalPrivateCount,
      ringDegree] at positionLt ⊢
    omega
  · trivial

@[simp] private theorem entryOp_localLength (interface : Interface)
    (coordinate offset : Nat) :
    (entryOp interface coordinate offset).localLength = entryPrivateCount := by
  rw [entryOp, Sequence.childOp_localLength]
  exact TranscriptAbsorption.localLength_eq (entryInterface interface)
    coordinate (entryOffset offset)

@[simp] private theorem windowOp_localLength (interface : Interface)
    (coordinate offset round : Nat) :
    (windowOp interface coordinate offset round).localLength =
      DigestWindow.logicalPrivateCount := by
  rw [windowOp, Sequence.childOp_localLength]
  exact DigestWindow.localLength_eq
    (windowInterface interface coordinate offset round) (windowOffset offset round)

@[simp] private theorem selectorOp_localLength (interface : Interface)
    (coordinate offset : Nat) :
    (selectorOp interface coordinate offset).localLength =
      First54.logicalPrivateCount := by
  rw [selectorOp, Sequence.childOp_localLength]
  exact First54.localLength_eq (selectorInterface interface coordinate offset)
    (selectorOffset offset)

@[simp] private theorem windowOpsPrefix_localLength (interface : Interface)
    (coordinate offset count : Nat) :
    localLength (windowOpsPrefix interface coordinate offset count) =
      count * DigestWindow.logicalPrivateCount := by
  simp [windowOpsPrefix, localLength, Function.comp_def]

@[simp] private theorem entryOp_rowCount (interface : Interface)
    (coordinate offset : Nat) :
    (entryOp interface coordinate offset).rowCount = entryPrivateCount := by
  change (flatConstraints (Circuit.ops
    (TranscriptAbsorption.circuit (entryInterface interface) coordinate).main
      (entryOffset offset))).length = entryPrivateCount
  exact TranscriptAbsorption.flatConstraints_length (entryInterface interface)
    coordinate (entryOffset offset)

@[simp] private theorem windowOp_rowCount (interface : Interface)
    (coordinate offset round : Nat) :
    (windowOp interface coordinate offset round).rowCount =
      DigestWindow.logicalRowCount := by
  rfl

@[simp] private theorem selectorOp_rowCount (interface : Interface)
    (coordinate offset : Nat) :
    (selectorOp interface coordinate offset).rowCount =
      First54.logicalRowCount := by
  rfl

theorem localLength_eq (interface : Interface) (coordinate offset : Nat) :
    localLength (Circuit.ops (main interface coordinate) offset) =
      logicalPrivateCount := by
  change localLength (opsAt interface coordinate offset) = logicalPrivateCount
  simp [opsAt, windowOps, localLength, digestRoundCount, entryPrivateCount,
    DigestWindow.logicalPrivateCount, First54.logicalPrivateCount,
    First54.candidateCount, First54.roundPrivateCount,
    First54Step.slotCount, First54ValueStep.outputCount,
    logicalPrivateCount, Function.comp_def]

theorem flatConstraints_length (interface : Interface)
    (coordinate offset : Nat) :
    (flatConstraints (Circuit.ops (main interface coordinate) offset)).length =
      logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt interface coordinate offset) = logicalRowCount
  simp [opsAt, windowOps, rowCount, digestRoundCount, entryPrivateCount,
    DigestWindow.logicalRowCount, First54.logicalRowCount,
    First54.logicalPrivateCount, First54.candidateCount,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount, logicalRowCount, Function.comp_def]

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

structure PrefixHolds (interface : Interface) (coordinate offset : Nat)
    (env : Env) : Prop where
  entry : TranscriptAbsorption.SpecHolds (entryInterface interface) coordinate
    (entryOffset offset) env
  window : ∀ round : Fin digestRoundCount,
    DigestWindow.SpecHolds
      (windowInterface interface coordinate offset round.val)
        (windowOffset offset round.val) env

structure SpecHolds (interface : Interface) (coordinate offset : Nat)
    (env : Env) extends PrefixHolds interface coordinate offset env where
  selector : First54.RelationHolds
    (selectorInterface interface coordinate offset) (selectorOffset offset) env

private theorem entryAssumptions (interface : Interface) (coordinate offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    TranscriptAbsorption.Assumptions (entryInterface interface)
      (entryOffset offset) env := by
  simpa [entryInterface, entryOffset] using assumptions

private theorem windowAssumptionsNat (interface : Interface)
    (coordinate offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env)
    (round : Nat) (bounded : round < digestRoundCount) :
    DigestWindow.Assumptions
      (windowInterface interface coordinate offset round)
        (windowOffset offset round) env := by
  induction round with
  | zero =>
      intro lane
      have scope := TranscriptAbsorption.output_varsBelow
        (entryInterface interface) coordinate (entryOffset offset) env
          (entryAssumptions interface coordinate offset assumptions) lane
      rw [TranscriptAbsorption.localLength_eq] at scope
      simpa [windowInterface, windowInitialState, windowOffset, windowBase,
        entryOffset, entryPrivateCount] using scope
  | succ previous inductionHypothesis =>
      have previousAssumptions := inductionHypothesis (by omega)
      intro lane
      have scope := DigestWindow.output_varsBelow
        (windowInterface interface coordinate offset previous)
          (windowOffset offset previous) previousAssumptions lane
      apply Expr.VarsBelow.mono _ scope
      simp [windowInterface, windowInitialState, windowOffset, windowBase,
        DigestWindow.logicalPrivateCount]
      omega

theorem windowAssumptions (interface : Interface)
    (coordinate offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env)
    (round : Fin digestRoundCount) :
    DigestWindow.Assumptions
      (windowInterface interface coordinate offset round.val)
        (windowOffset offset round.val) env :=
  windowAssumptionsNat interface coordinate offset assumptions round.val round.isLt

theorem outputState_varsBelow (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (outputState interface coordinate offset lane).VarsBelow
      (offset + logicalPrivateCount) := by
  intro lane
  have scope := DigestWindow.output_varsBelow
    (windowInterface interface coordinate offset (digestRoundCount - 1))
    (windowOffset offset (digestRoundCount - 1))
    (windowAssumptionsNat interface coordinate offset assumptions
      (digestRoundCount - 1) (by decide)) lane
  apply Expr.VarsBelow.mono _ scope
  simp [outputState, windowOffset, windowBase, digestRoundCount,
    entryPrivateCount, DigestWindow.logicalPrivateCount, logicalPrivateCount]

private theorem rejectBelowWindow (offset : Nat)
    (position : Fin chunksPerDigest) :
    (DigestWindow.reject offset position).VarsBelow
      (offset + DigestWindow.logicalPrivateCount) := by
  have bounded := position.isLt
  simp only [chunksPerDigest] at bounded
  simp [DigestWindow.reject, DigestWindow.laneOffset, DigestWindow.laneOf,
    DigestWindow.partOf, DigestLane.reject, DigestLane.decoderOffset,
    Candidate16Five.rejectExpr,
    CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount,
    DigestLane.logicalPrivateCount, DigestWindow.logicalPrivateCount,
    Expr.VarsBelow]
  omega

private theorem remainderBelowWindow (offset : Nat)
    (position : Fin chunksPerDigest) :
    (DigestWindow.remainder offset position).VarsBelow
      (offset + DigestWindow.logicalPrivateCount) := by
  have bounded := position.isLt
  simp only [chunksPerDigest] at bounded
  simp [DigestWindow.remainder, DigestWindow.laneOffset, DigestWindow.laneOf,
    DigestWindow.partOf, DigestLane.remainder, DigestLane.decoderOffset,
    Candidate16Five.remainderExpr, CanonicalU64.auxiliaryCount,
    Candidate16Five.auxiliaryCount, DigestLane.logicalPrivateCount,
    DigestWindow.logicalPrivateCount, Expr.VarsBelow]
  omega

theorem selectorAssumptions (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (windows : ∀ round : Fin digestRoundCount,
      DigestWindow.SpecHolds
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) env) :
    First54.Assumptions (selectorInterface interface coordinate offset)
      (selectorOffset offset) env := by
  constructor
  · intro candidate
    apply Expr.VarsBelow.sub
    · trivial
    · apply Expr.VarsBelow.mono _
        (rejectBelowWindow
          (windowOffset offset (candidateRound candidate).val)
            (candidatePosition candidate))
      have bounded := (candidateRound candidate).isLt
      simp [windowOffset, selectorOffset, windowBase, digestRoundCount,
        DigestWindow.logicalPrivateCount] at bounded ⊢
      omega
  · intro candidate
    apply Expr.VarsBelow.mono _
      (remainderBelowWindow
        (windowOffset offset (candidateRound candidate).val)
          (candidatePosition candidate))
    have bounded := (candidateRound candidate).isLt
    simp [windowOffset, selectorOffset, windowBase, digestRoundCount,
      DigestWindow.logicalPrivateCount] at bounded ⊢
    omega
  · intro candidate
    let round := candidateRound candidate
    let position := candidatePosition candidate
    have decoder := (windows round).lane (DigestWindow.laneOf position)
    have rejectEq := (decoder.decoder (DigestWindow.partOf position)).reject_eq
    have rejectValue :
        (DigestWindow.reject (windowOffset offset round.val) position).eval env =
          if ((DigestWindow.candidate (windowOffset offset round.val) position).eval
              env).val = rejectionBucket then 1 else 0 := by
      simpa [DigestWindow.reject, DigestWindow.candidate, DigestLane.reject,
        DigestLane.candidate] using rejectEq
    change
      ((1 - DigestWindow.reject (windowOffset offset round.val) position).eval env =
          0 ∨
        (1 - DigestWindow.reject (windowOffset offset round.val) position).eval env =
          1)
    rw [Expr.eval_sub, rejectValue]
    by_cases rejected :
        ((DigestWindow.candidate (windowOffset offset round.val) position).eval
          env).val = rejectionBucket
    · left
      rw [if_pos rejected]
      rfl
    · right
      rw [if_neg rejected]
      rfl

private theorem entryScope (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (entryCircuit interface coordinate).main
          (entryOffset offset)),
      expression.VarsBelow
        (entryOffset offset + localLength
          (Circuit.ops (entryCircuit interface coordinate).main
            (entryOffset offset))) := by
  exact TranscriptAbsorption.flatConstraints_varsBelow
    (entryInterface interface) coordinate (entryOffset offset) env
      (entryAssumptions interface coordinate offset assumptions)

private theorem windowScope (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (round : Nat) (bounded : round < digestRoundCount) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (windowCircuit interface coordinate offset round).main
          (windowOffset offset round)),
      expression.VarsBelow
        (windowOffset offset round + localLength
          (Circuit.ops
            (windowCircuit interface coordinate offset round).main
              (windowOffset offset round))) := by
  exact DigestWindow.flatConstraints_varsBelow
    (windowInterface interface coordinate offset round)
      (windowOffset offset round)
      (windowAssumptionsNat interface coordinate offset assumptions round bounded)

private theorem selectorScope (interface : Interface) (coordinate offset : Nat)
    (env : Env)
    (windows : ∀ round : Fin digestRoundCount,
      DigestWindow.SpecHolds
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (selectorCircuit interface coordinate offset).main
          (selectorOffset offset)),
      expression.VarsBelow
        (selectorOffset offset + localLength
          (Circuit.ops (selectorCircuit interface coordinate offset).main
            (selectorOffset offset))) := by
  exact First54.flatConstraints_varsBelow
    (selectorInterface interface coordinate offset) (selectorOffset offset) env
      (selectorAssumptions interface coordinate offset env windows)

private theorem completeEntry (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = [entryOp interface coordinate offset] := by
  let empty := Sequence.empty env offset
  rcases TranscriptAbsorption.complete (entryInterface interface) coordinate env
      (entryOffset offset)
      (entryAssumptions interface coordinate offset assumptions) with
    ⟨entryEnv, entryAgrees, entryRows⟩
  rcases Sequence.appendBuiltAt empty entryName
      (entryCircuit interface coordinate) (entryOffset offset) (by
        simp [empty, Sequence.empty, entryOffset, localLength])
      (entryScope interface coordinate offset env assumptions)
      entryEnv entryAgrees entryRows with
    ⟨completed, operationsEq, _, _, _⟩
  refine ⟨completed, ?_⟩
  simpa [empty, entryOp] using operationsEq

private theorem completeWindows (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (entryCompleted : Sequence.Prefix env offset)
    (entryOperations : entryCompleted.operations =
      [entryOp interface coordinate offset])
    (count : Nat) (bounded : count ≤ digestRoundCount) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = [entryOp interface coordinate offset] ++
        windowOpsPrefix interface coordinate offset count := by
  induction count with
  | zero =>
      refine ⟨entryCompleted, ?_⟩
      simpa [windowOpsPrefix] using entryOperations
  | succ count inductionHypothesis =>
      have countLt : count < digestRoundCount := by omega
      rcases inductionHypothesis (by omega) with ⟨before, beforeOperations⟩
      have startEq : offset + localLength before.operations =
          windowOffset offset count := by
        rw [beforeOperations, Sequence.localLength_append,
          windowOpsPrefix_localLength]
        simp [windowOffset, windowBase]
        omega
      have currentAssumptions :
          Assumptions interface offset before.current := assumptions
      have childAssumptions := windowAssumptionsNat interface coordinate offset
        currentAssumptions count countLt
      rcases DigestWindow.complete
          (windowInterface interface coordinate offset count) before.current
          (windowOffset offset count) childAssumptions with
        ⟨windowEnv, windowAgrees, windowRows⟩
      rcases Sequence.appendBuiltAt before (windowName count)
          (windowCircuit interface coordinate offset count)
          (windowOffset offset count) startEq
          (windowScope interface coordinate offset before.current
            currentAssumptions count countLt)
          windowEnv windowAgrees windowRows with
        ⟨completed, operationsEq, _, _, _⟩
      refine ⟨completed, ?_⟩
      rw [operationsEq, beforeOperations]
      simp [windowOpsPrefix, List.range_succ, windowOp]

private theorem completedPrefixHolds (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (completed : Sequence.Prefix env offset)
    (operationsEq : completed.operations =
      [entryOp interface coordinate offset] ++
        windowOps interface coordinate offset) :
    PrefixHolds interface coordinate offset completed.current := by
  have currentAssumptions :
      Assumptions interface offset completed.current := assumptions
  have parentRows := holdsFlat_implies_holds completed.current
    completed.operations completed.rows
  constructor
  · have member : entryOp interface coordinate offset ∈
        completed.operations := by
      rw [operationsEq]
      simp
    have callHolds := parentRows _ member
    change TranscriptAbsorption.Assumptions (entryInterface interface)
        (entryOffset offset) completed.current →
      TranscriptAbsorption.SpecHolds (entryInterface interface) coordinate
        (entryOffset offset) completed.current at callHolds
    exact callHolds
      (entryAssumptions interface coordinate offset currentAssumptions)
  · intro round
    have member : windowOp interface coordinate offset round.val ∈
        completed.operations := by
      rw [operationsEq]
      apply List.mem_append_right [entryOp interface coordinate offset]
      apply List.mem_map.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, rfl⟩
    have callHolds := parentRows _ member
    change DigestWindow.Assumptions
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) completed.current →
      DigestWindow.SpecHolds
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) completed.current at callHolds
    exact callHolds
      (windowAssumptions interface coordinate offset currentAssumptions round)

theorem flatConstraints_varsBelow (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface coordinate offset env) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (main interface coordinate) offset),
      expression.VarsBelow (offset + logicalPrivateCount) := by
  change ∀ expression ∈ flatConstraints
      (opsAt interface coordinate offset),
    expression.VarsBelow (offset + logicalPrivateCount)
  intro expression member
  rcases List.mem_flatMap.mp member with
    ⟨operation, operationMember, expressionMember⟩
  rcases List.mem_append.mp operationMember with prefixMember | selectorMember
  · rcases List.mem_append.mp prefixMember with entryMember | windowMember
    · simp only [List.mem_singleton] at entryMember
      subst operation
      apply Expr.VarsBelow.mono expression
        (entryScope interface coordinate offset env assumptions expression (by
          simpa [entryOp, Sequence.childOp] using expressionMember))
      have lengthEq : localLength
          (Circuit.ops (entryCircuit interface coordinate).main
            (entryOffset offset)) = entryPrivateCount := by
        simpa [entryCircuit] using TranscriptAbsorption.localLength_eq
          (entryInterface interface) coordinate (entryOffset offset)
      rw [lengthEq]
      simp [entryOffset, entryPrivateCount, logicalPrivateCount]
    · rcases List.mem_map.mp windowMember with
        ⟨round, roundMember, rfl⟩
      have bounded := List.mem_range.mp roundMember
      apply Expr.VarsBelow.mono expression
        (windowScope interface coordinate offset env assumptions round bounded
          expression (by
            simpa [windowOp, Sequence.childOp] using expressionMember))
      have lengthEq : localLength
          (Circuit.ops
            (windowCircuit interface coordinate offset round).main
              (windowOffset offset round)) =
          DigestWindow.logicalPrivateCount := by
        simpa [windowCircuit] using DigestWindow.localLength_eq
          (windowInterface interface coordinate offset round)
            (windowOffset offset round)
      rw [lengthEq]
      simp [windowOffset, windowBase, digestRoundCount,
        entryPrivateCount, DigestWindow.logicalPrivateCount,
        logicalPrivateCount] at bounded ⊢
      omega
  · simp only [List.mem_singleton] at selectorMember
    subst operation
    apply Expr.VarsBelow.mono expression
      (selectorScope interface coordinate offset env
        specification.toPrefixHolds.window expression (by
          simpa [selectorOp, Sequence.childOp] using expressionMember))
    have lengthEq : localLength
        (Circuit.ops (selectorCircuit interface coordinate offset).main
          (selectorOffset offset)) = First54.logicalPrivateCount := by
      simpa [selectorCircuit, First54.semanticCircuit] using
        First54.localLength_eq (selectorInterface interface coordinate offset)
          (selectorOffset offset)
    rw [lengthEq]
    simp [selectorOffset, windowBase, digestRoundCount,
      entryPrivateCount, DigestWindow.logicalPrivateCount,
      First54.logicalPrivateCount, First54.candidateCount,
      First54.roundPrivateCount, First54Step.slotCount,
      First54ValueStep.outputCount, logicalPrivateCount]

theorem soundness (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface coordinate) offset)) :
    SpecHolds interface coordinate offset env := by
  change holds env (opsAt interface coordinate offset) at rows
  have entryMember : entryOp interface coordinate offset ∈
      opsAt interface coordinate offset := by simp [opsAt]
  have entryCall := rows _ entryMember
  change TranscriptAbsorption.Assumptions (entryInterface interface)
      (entryOffset offset) env →
    TranscriptAbsorption.SpecHolds (entryInterface interface) coordinate
      (entryOffset offset) env at entryCall
  have entrySpec := entryCall
    (entryAssumptions interface coordinate offset assumptions)
  have windowSpecs : ∀ round : Fin digestRoundCount,
      DigestWindow.SpecHolds
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) env := by
    intro round
    have windowMember : windowOp interface coordinate offset round.val ∈
        opsAt interface coordinate offset := by
      apply List.mem_append_left [selectorOp interface coordinate offset]
      apply List.mem_append_right [entryOp interface coordinate offset]
      apply List.mem_map.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, rfl⟩
    have windowCall := rows _ windowMember
    change DigestWindow.Assumptions
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) env →
      DigestWindow.SpecHolds
        (windowInterface interface coordinate offset round.val)
          (windowOffset offset round.val) env at windowCall
    exact windowCall (windowAssumptions interface coordinate offset assumptions round)
  have selectorMember : selectorOp interface coordinate offset ∈
      opsAt interface coordinate offset := by simp [opsAt]
  have selectorCall := rows _ selectorMember
  change First54.Assumptions (selectorInterface interface coordinate offset)
      (selectorOffset offset) env →
    First54.RelationHolds (selectorInterface interface coordinate offset)
      (selectorOffset offset) env at selectorCall
  exact ⟨⟨entrySpec, windowSpecs⟩,
    selectorCall (selectorAssumptions interface coordinate offset env windowSpecs)⟩

def evalState (env : Env) (state : EState) : State :=
  List.ofFn (NightstreamFPrime.Gadgets.Poseidon2.Layer.evalState env state)

def evalInitialState (interface : Interface) (offset : Nat) (env : Env) : State :=
  evalState env (interface.initialState offset)

private theorem evalInitialState_eq_of_agree_below (interface : Interface)
    (offset : Nat) (initial current : Env)
    (assumptions : Assumptions interface offset initial)
    (agrees : ∀ index, index < offset → current index = initial index) :
    evalInitialState interface offset current =
      evalInitialState interface offset initial := by
  unfold evalInitialState evalState
  apply congrArg List.ofFn
  funext lane
  exact (interface.initialState offset lane).eval_eq_of_agree_below offset
    current initial (assumptions lane) agrees

def enteredState (interface : Interface) (coordinate offset : Nat)
    (env : Env) : State :=
  NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine.enterScalar
    (evalInitialState interface offset env) coordinate

theorem windowState_eq_stateBeforeBlock (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (round : Nat) (bounded : round ≤ digestRoundCount) :
    evalState env (windowInitialState interface coordinate offset round) =
      ProductionSchedule.stateBeforeBlock
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
        (enteredState interface coordinate offset env) coordinate round := by
  induction round with
  | zero =>
      exact specification.entry.symm
  | succ previous inductionHypothesis =>
      have previousBound : previous < digestRoundCount := by omega
      let previousRound : Fin digestRoundCount := ⟨previous, previousBound⟩
      have coverage := DigestWindow.parentCoverage
        (windowInterface interface coordinate offset previous)
          (windowOffset offset previous) env
          (specification.window previousRound) (coordinate + previous)
      have outputEq := congrArg Prod.fst coverage
      have priorEq := inductionHypothesis (by omega)
      change evalState env
          (DigestWindow.output
            (windowInterface interface coordinate offset previous)
              (windowOffset offset previous)) =
        (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestBlock
          (evalState env (windowInitialState interface coordinate offset previous))
            (coordinate + previous)).1 at outputEq
      rw [priorEq] at outputEq
      simpa [windowInitialState, evalState,
        ProductionSchedule.stateBeforeBlock_succ,
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine,
        previousRound] using outputEq

theorem candidate_eq_sourceStream (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (candidate : Fin First54.candidateCount) :
    DigestWindow.evalChunk env
        (windowOffset offset (candidateRound candidate).val)
          (candidatePosition candidate) =
      (ProductionSchedule.source
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
          (evalInitialState interface offset env) coordinate).stream candidate.val := by
  let round := candidateRound candidate
  let position := candidatePosition candidate
  have coverage := DigestWindow.parentCoverage
    (windowInterface interface coordinate offset round.val)
      (windowOffset offset round.val) env (specification.window round)
        (coordinate + round.val)
  have chunksEq := congrFun (congrArg Prod.snd coverage) position
  have stateEq := windowState_eq_stateBeforeBlock interface coordinate offset env
    specification round.val (Nat.le_of_lt round.isLt)
  change DigestWindow.evalChunk env (windowOffset offset round.val) position =
    (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestBlock
      (evalState env (windowInitialState interface coordinate offset round.val))
        (coordinate + round.val)).2 position at chunksEq
  rw [stateEq] at chunksEq
  simpa [ProductionSchedule.source, ProductionSchedule.candidateStream,
    ProductionSchedule.chunksAt,
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine,
    round, position, candidateRound, candidatePosition] using chunksEq

theorem outputState_eq_nextState (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env) :
    evalState env (outputState interface coordinate offset) =
      (ProductionSchedule.source
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
          (evalInitialState interface offset env) coordinate).nextState := by
  have final := windowState_eq_stateBeforeBlock interface coordinate offset env
    specification digestRoundCount (Nat.le_refl _)
  simpa [outputState, windowInitialState, ProductionSchedule.source,
    digestRoundCount] using final

def evalCandidate (env : Env) (offset : Nat)
    (candidate : Fin First54.candidateCount) : Chunk :=
  DigestWindow.evalChunk env
    (windowOffset offset (candidateRound candidate).val)
      (candidatePosition candidate)

def coefficientWord (coefficient : Coefficient) : F :=
  ⟨coefficient.val, lt_trans coefficient.isLt (by
    norm_num [alphabetSize, goldilocksModulus])⟩

@[simp] theorem coefficientWord_val (coefficient : Coefficient) :
    (coefficientWord coefficient).val = coefficient.val := by
  rfl

theorem coefficientWord_sub_two_eq_embedCoefficient
    (coefficient : Coefficient) :
    coefficientWord coefficient - (2 : F) =
      Phi81StrongSet.embedCoefficient coefficient := by
  revert coefficient
  decide

private theorem candidateExprVal_eq_evalCandidate (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (candidate : Fin First54.candidateCount) :
    ((DigestWindow.candidate
      (windowOffset offset (candidateRound candidate).val)
        (candidatePosition candidate)).eval env).val =
      (evalCandidate env offset candidate).val := by
  let round := candidateRound candidate
  let position := candidatePosition candidate
  have candidateEq := DigestWindow.candidateValue_eq_digestChunks
    (windowInterface interface coordinate offset round.val)
      (windowOffset offset round.val) env (specification.window round) position
  have chunkEq := DigestWindow.evalChunk_eq_digestChunks
    (windowInterface interface coordinate offset round.val)
      (windowOffset offset round.val) env (specification.window round) position
  calc
    ((DigestWindow.candidate (windowOffset offset round.val) position).eval
        env).val =
      (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
        (DigestWindow.evalState env
          ((windowInterface interface coordinate offset round.val).initialState
            (windowOffset offset round.val))) position).val := candidateEq
    _ = (DigestWindow.evalChunk env (windowOffset offset round.val) position).val :=
      congrArg Fin.val chunkEq.symm
    _ = (evalCandidate env offset candidate).val := by
      rfl

theorem selector_accepts_eq_production (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (candidate : Fin First54.candidateCount) :
    (First54.semanticVerifier (selectorInterface interface coordinate offset)
      (selectorOffset offset) env).accepts candidate =
        verifier.accepts (evalCandidate env offset candidate) := by
  let round := candidateRound candidate
  let position := candidatePosition candidate
  have decoder := (specification.window round).lane (DigestWindow.laneOf position)
  have rejectEq := (decoder.decoder (DigestWindow.partOf position)).reject_eq
  have candidateEq := candidateExprVal_eq_evalCandidate interface coordinate
    offset env specification candidate
  have decoderCandidateEq :
      (((DigestLane.decoderInterface
        (DigestWindow.laneOffset (windowOffset offset round.val)
          (DigestWindow.laneOf position))
          (DigestWindow.partOf position)).candidate
        (DigestLane.decoderOffset
          (DigestWindow.laneOffset (windowOffset offset round.val)
            (DigestWindow.laneOf position))
          (DigestWindow.partOf position))).eval env).val =
        (evalCandidate env offset candidate).val := by
    simpa [DigestWindow.candidate, DigestLane.candidate, round, position] using
      candidateEq
  have rejectValue :
      (DigestWindow.reject (windowOffset offset round.val) position).eval env =
        if (evalCandidate env offset candidate).val = rejectionBucket then 1 else 0 := by
    simpa [DigestWindow.reject, DigestLane.reject, decoderCandidateEq] using
      rejectEq
  rw [Bool.eq_iff_iff, accepts_eq_true_iff_ne_rejectionBucket]
  by_cases rejected : (evalCandidate env offset candidate).val = rejectionBucket
  · simp [First54.semanticVerifier, selectorInterface, Expr.eval_sub,
      rejectValue, rejected, round, position]
    change ¬(1 : F) - 1 = 1
    norm_num [goldilocksModulus]
  · simp [First54.semanticVerifier, selectorInterface, Expr.eval_sub,
      rejectValue, rejected, round, position]
    change (1 : F) = 1
    rfl

theorem selector_symbol_eq_productionWord (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (candidate : Fin First54.candidateCount) :
    (First54.semanticVerifier (selectorInterface interface coordinate offset)
      (selectorOffset offset) env).symbol candidate =
        coefficientWord (verifier.symbol (evalCandidate env offset candidate)) := by
  let round := candidateRound candidate
  let position := candidatePosition candidate
  have decoder := (specification.window round).lane (DigestWindow.laneOf position)
  have remainderEq := (decoder.decoder (DigestWindow.partOf position)).remainder_eq
  have candidateEq := candidateExprVal_eq_evalCandidate interface coordinate
    offset env specification candidate
  have decoderCandidateEq :
      (((DigestLane.decoderInterface
        (DigestWindow.laneOffset (windowOffset offset round.val)
          (DigestWindow.laneOf position))
          (DigestWindow.partOf position)).candidate
        (DigestLane.decoderOffset
          (DigestWindow.laneOffset (windowOffset offset round.val)
            (DigestWindow.laneOf position))
          (DigestWindow.partOf position))).eval env).val =
        (evalCandidate env offset candidate).val := by
    simpa [DigestWindow.candidate, DigestLane.candidate, round, position] using
      candidateEq
  apply Fin.ext
  change
    ((Candidate16Five.remainderExpr
      (DigestLane.decoderOffset
        (DigestWindow.laneOffset (windowOffset offset round.val)
          (DigestWindow.laneOf position))
        (DigestWindow.partOf position))).eval env).val =
      (evalCandidate env offset candidate).val % alphabetSize
  rw [remainderEq, decoderCandidateEq]
  rfl

theorem candidatePrefix_eq_sourcePrefix (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env) :
    (First54.semanticCandidates First54.candidateCount).map
        (evalCandidate env offset) =
      FirstAccepted.streamPrefix
        (ProductionSchedule.source
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
            (evalInitialState interface offset env) coordinate).stream
        candidateBound := by
  unfold First54.semanticCandidates First54.candidateStream
    FirstAccepted.streamPrefix
  simp only [First54.candidateCount, candidateBound, List.map_map]
  apply List.map_congr_left
  intro index member
  have indexLt : index < 64 := List.mem_range.mp member
  have point := candidate_eq_sourceStream interface coordinate offset env
    specification (First54.candidateIndex index)
  simpa [Function.comp_def, First54.candidateIndex, First54.candidateCount,
    Nat.mod_eq_of_lt indexLt] using point

private theorem acceptedSymbols_eq_productionMap (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env)
    (candidates : List (Fin First54.candidateCount)) :
    FirstAccepted.acceptedSymbols
        (First54.semanticVerifier (selectorInterface interface coordinate offset)
          (selectorOffset offset) env) candidates =
      (FirstAccepted.acceptedSymbols verifier
        (candidates.map (evalCandidate env offset))).map coefficientWord := by
  induction candidates with
  | nil => rfl
  | cons candidate tail inductionHypothesis =>
      have acceptsEq := selector_accepts_eq_production interface coordinate
        offset env specification candidate
      have symbolEq := selector_symbol_eq_productionWord interface coordinate
        offset env specification candidate
      have tailEq :
          List.map
              (First54.semanticVerifier
                (selectorInterface interface coordinate offset)
                  (selectorOffset offset) env).symbol
              (List.filter
                (First54.semanticVerifier
                  (selectorInterface interface coordinate offset)
                    (selectorOffset offset) env).accepts tail) =
            List.map (coefficientWord ∘ verifier.symbol)
              (List.filter verifier.accepts
                (tail.map (evalCandidate env offset))) := by
        simpa [FirstAccepted.acceptedSymbols,
          FirstAccepted.acceptedCandidates, List.map_map,
          Function.comp_def] using inductionHypothesis
      cases accepted :
          (First54.semanticVerifier
            (selectorInterface interface coordinate offset)
              (selectorOffset offset) env).accepts candidate with
      | false =>
          have productionAccepted :
              verifier.accepts (evalCandidate env offset candidate) = false := by
            rw [← acceptsEq]
            exact accepted
          simp [FirstAccepted.acceptedSymbols,
            FirstAccepted.acceptedCandidates, accepted, productionAccepted,
            tailEq, List.map_map, Function.comp_def]
      | true =>
          have productionAccepted :
              verifier.accepts (evalCandidate env offset candidate) = true := by
            rw [← acceptsEq]
            exact accepted
          simp [FirstAccepted.acceptedSymbols,
            FirstAccepted.acceptedCandidates, accepted, productionAccepted,
            symbolEq, tailEq, List.map_map, Function.comp_def]

theorem acceptedSymbols_eq_productionWords (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env) :
    First54.semanticAcceptedSymbols
        (selectorInterface interface coordinate offset) (selectorOffset offset)
          env First54.candidateCount =
      (FirstAccepted.acceptedSymbols verifier
        (FirstAccepted.streamPrefix
          (ProductionSchedule.source
            NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
              (evalInitialState interface offset env) coordinate).stream
          candidateBound)).map coefficientWord := by
  unfold First54.semanticAcceptedSymbols
  rw [← candidatePrefix_eq_sourcePrefix interface coordinate offset env
    specification]
  exact acceptedSymbols_eq_productionMap interface coordinate offset env
    specification (First54.semanticCandidates First54.candidateCount)

theorem acceptedCount_eq_production (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env) :
    First54.semanticAcceptedCount
        (selectorInterface interface coordinate offset) (selectorOffset offset)
          env First54.candidateCount =
      FirstAccepted.acceptedCount verifier
        (FirstAccepted.streamPrefix
          (ProductionSchedule.source
            NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
              (evalInitialState interface offset env) coordinate).stream
          candidateBound) := by
  have lengths := congrArg List.length
    (acceptedSymbols_eq_productionWords interface coordinate offset env
      specification)
  simpa [First54.semanticAcceptedCount, First54.semanticAcceptedSymbols,
    FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCount] using lengths

def productionSource (interface : Interface) (coordinate offset : Nat)
    (env : Env) :=
  ProductionSchedule.source
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
      (evalInitialState interface offset env) coordinate

def productionCandidates (interface : Interface) (coordinate offset : Nat)
    (env : Env) : List Chunk :=
  FirstAccepted.streamPrefix
    (productionSource interface coordinate offset env).stream candidateBound

theorem firstAccepted_eq_productionWords (interface : Interface)
    (coordinate offset : Nat) (env : Env)
    (specification : PrefixHolds interface coordinate offset env) :
    FirstAccepted.firstAccepted
        (First54.semanticVerifier (selectorInterface interface coordinate offset)
          (selectorOffset offset) env)
        First54.outputCount
        (First54.semanticCandidates First54.candidateCount) =
      (FirstAccepted.firstAccepted verifier coefficientCount
        (productionCandidates interface coordinate offset env)).map
          coefficientWord := by
  unfold FirstAccepted.firstAccepted productionCandidates productionSource
  have symbolsEq :
      FirstAccepted.acceptedSymbols
          (First54.semanticVerifier
            (selectorInterface interface coordinate offset)
              (selectorOffset offset) env)
          (First54.semanticCandidates First54.candidateCount) =
        (FirstAccepted.acceptedSymbols verifier
          (FirstAccepted.streamPrefix
            (ProductionSchedule.source
              NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.machine
                (evalInitialState interface offset env) coordinate).stream
            candidateBound)).map coefficientWord := by
    simpa [First54.semanticAcceptedSymbols] using
      acceptedSymbols_eq_productionWords interface coordinate offset env
        specification
  rw [symbolsEq]
  simp [First54.outputCount, First54ValueStep.outputCount, coefficientCount,
    List.map_take]

def RelationHolds (interface : Interface) (coordinate offset : Nat)
    (env : Env) : Prop :=
  ∃ coefficients,
    FirstAccepted.boundedSample verifier coefficientCount
        (productionCandidates interface coordinate offset env) =
      some coefficients ∧
    outputCoefficients env offset = coefficients.map coefficientWord ∧
    evalState env (outputState interface coordinate offset) =
      (productionSource interface coordinate offset env).nextState

/-- The input-only condition needed to construct one sampler witness. -/
def SamplingSucceeds (interface : Interface) (coordinate offset : Nat)
    (env : Env) : Prop :=
  ∃ coefficients,
    FirstAccepted.boundedSample verifier coefficientCount
        (productionCandidates interface coordinate offset env) =
      some coefficients

/-- The zero-row centered view is the exact ring challenge assembled from the
same accepted coefficient list as the transcript sampler. -/
theorem relation_implies_outputChallenge
    (interface : Interface) (coordinate offset : Nat) (env : Env)
    (relation : RelationHolds interface coordinate offset env) :
    ∃ coefficients,
      FirstAccepted.boundedSample verifier coefficientCount
          (productionCandidates interface coordinate offset env) =
        some coefficients ∧
      evalOutputChallenge env offset =
        Phi81StrongSet.embedScalar
          (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.scalarOfList
            coefficients) ∧
      evalState env (outputState interface coordinate offset) =
        (productionSource interface coordinate offset env).nextState := by
  rcases relation with ⟨coefficients, success, outputEq, stateEq⟩
  refine ⟨coefficients, success, ?_, stateEq⟩
  funext position
  have coefficientLength : coefficients.length = coefficientCount :=
    FirstAccepted.bounded_success_length success
  have positionLt : position.val < coefficients.length := by
    rw [coefficientLength]
    simpa [coefficientCount, ringDegree] using position.isLt
  have outputPositionLt : position.val < First54.outputCount := by
    simpa [First54.outputCount, First54ValueStep.outputCount, ringDegree] using
      position.isLt
  have wordEq :
      (outputWord offset position).eval env =
        coefficientWord (coefficients.getD position.val ⟨2, by decide⟩) := by
    have selected := congrArg
      (fun values : List F => values.getD position.val 0) outputEq
    simpa [outputCoefficients, First54.evalOutput, outputWord, outputSlot,
      List.getD_eq_getElem?_getD, positionLt, outputPositionLt] using selected
  calc
    evalOutputChallenge env offset position =
        coefficientWord (coefficients.getD position.val ⟨2, by decide⟩) - 2 := by
      exact (evalOutputChallenge_apply env offset position).trans
        (congrArg (fun value : F => value - 2) wordEq)
    _ = Phi81StrongSet.embedCoefficient
        (coefficients.getD position.val ⟨2, by decide⟩) :=
      coefficientWord_sub_two_eq_embedCoefficient _
    _ = Phi81StrongSet.embedScalar
        (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.scalarOfList
          coefficients) position := by
      rfl

theorem parentCoverage (interface : Interface) (coordinate offset : Nat)
    (env : Env) (specification : SpecHolds interface coordinate offset env) :
    RelationHolds interface coordinate offset env := by
  let coefficients := FirstAccepted.firstAccepted verifier coefficientCount
    (productionCandidates interface coordinate offset env)
  have selectorSuccess :
      FirstAccepted.boundedSample
          (First54.semanticVerifier
            (selectorInterface interface coordinate offset)
              (selectorOffset offset) env)
          First54.outputCount
          (First54.semanticCandidates First54.candidateCount) =
        some (outputCoefficients env offset) := by
    exact specification.selector
  have selectorEnough :=
    (FirstAccepted.boundedSample_eq_some_iff.mp selectorSuccess).1
  have selectorEnoughCount :
      First54.outputCount ≤
        First54.semanticAcceptedCount
          (selectorInterface interface coordinate offset)
            (selectorOffset offset) env First54.candidateCount := by
    simpa [FirstAccepted.Enough, First54.semanticAcceptedCount] using
      selectorEnough
  have countEq :
      First54.semanticAcceptedCount
          (selectorInterface interface coordinate offset)
            (selectorOffset offset) env First54.candidateCount =
        FirstAccepted.acceptedCount verifier
          (productionCandidates interface coordinate offset env) := by
    simpa [productionCandidates, productionSource] using
      acceptedCount_eq_production interface coordinate offset env
        specification.toPrefixHolds
  have productionEnough : FirstAccepted.Enough verifier coefficientCount
      (productionCandidates interface coordinate offset env) := by
    unfold FirstAccepted.Enough
    rw [← countEq]
    simpa [First54.outputCount, First54ValueStep.outputCount,
      coefficientCount] using selectorEnoughCount
  have productionSuccess :
      FirstAccepted.boundedSample verifier coefficientCount
          (productionCandidates interface coordinate offset env) =
        some coefficients := by
    apply FirstAccepted.boundedSample_eq_some_iff.mpr
    exact ⟨productionEnough, rfl⟩
  have outputEq : outputCoefficients env offset =
      coefficients.map coefficientWord := by
    calc
      outputCoefficients env offset =
          FirstAccepted.firstAccepted
            (First54.semanticVerifier
              (selectorInterface interface coordinate offset)
                (selectorOffset offset) env)
            First54.outputCount
            (First54.semanticCandidates First54.candidateCount) :=
        FirstAccepted.bounded_success_exact selectorSuccess
      _ = coefficients.map coefficientWord := by
        simpa [coefficients] using
          firstAccepted_eq_productionWords interface coordinate offset env
            specification.toPrefixHolds
  have stateEq : evalState env (outputState interface coordinate offset) =
      (productionSource interface coordinate offset env).nextState := by
    simpa [productionSource] using
      outputState_eq_nextState interface coordinate offset env
        specification.toPrefixHolds
  exact ⟨coefficients, productionSuccess, outputEq, stateEq⟩

theorem rows_imply_relation (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface coordinate) offset)) :
    RelationHolds interface coordinate offset env :=
  parentCoverage interface coordinate offset env
    (soundness interface coordinate env offset assumptions rows)

set_option maxRecDepth 100000 in -- fixed-size: one scalar, eight digest windows, and 64 selector rounds
theorem complete_of_success (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (success : SamplingSucceeds interface coordinate offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface coordinate) offset)) ∧
      holdsFlat completed
        (Circuit.ops (main interface coordinate) offset) := by
  rcases completeEntry interface coordinate offset env assumptions with
    ⟨entryCompleted, entryOperations⟩
  rcases completeWindows interface coordinate offset env assumptions
      entryCompleted entryOperations digestRoundCount (Nat.le_refl _) with
    ⟨windowsCompleted, prefixOperations⟩
  have windowOperations : windowsCompleted.operations =
      [entryOp interface coordinate offset] ++
        windowOps interface coordinate offset := by
    simpa [windowOps, windowOpsPrefix] using prefixOperations
  have prefixSpecification := completedPrefixHolds interface coordinate offset
    env assumptions windowsCompleted windowOperations
  have agreesBelow : ∀ index, index < offset →
      windowsCompleted.current index = env index := by
    intro index below
    exact windowsCompleted.agrees index (Or.inl below)
  have initialStateEq :
      evalInitialState interface offset windowsCompleted.current =
        evalInitialState interface offset env :=
    evalInitialState_eq_of_agree_below interface offset env
      windowsCompleted.current assumptions agreesBelow
  rcases success with ⟨coefficients, productionSuccess⟩
  have currentProductionSuccess :
      FirstAccepted.boundedSample verifier coefficientCount
          (productionCandidates interface coordinate offset
            windowsCompleted.current) =
        some coefficients := by
    simpa [productionCandidates, productionSource, initialStateEq] using
      productionSuccess
  have productionEnough :=
    (FirstAccepted.boundedSample_eq_some_iff.mp currentProductionSuccess).1
  have productionEnoughCount : coefficientCount ≤
      FirstAccepted.acceptedCount verifier
        (productionCandidates interface coordinate offset
          windowsCompleted.current) := by
    simpa [FirstAccepted.Enough] using productionEnough
  have countEq := acceptedCount_eq_production interface coordinate offset
    windowsCompleted.current prefixSpecification
  have selectorEnough : First54.outputCount ≤
      First54.semanticAcceptedCount
        (selectorInterface interface coordinate offset) (selectorOffset offset)
          windowsCompleted.current First54.candidateCount := by
    rw [countEq]
    simpa [First54.outputCount, First54ValueStep.outputCount,
      coefficientCount] using productionEnoughCount
  have selectorChildAssumptions := selectorAssumptions interface coordinate
    offset windowsCompleted.current prefixSpecification.window
  rcases First54.complete_of_enough
      (selectorInterface interface coordinate offset) windowsCompleted.current
      (selectorOffset offset) selectorChildAssumptions selectorEnough with
    ⟨selectorEnv, selectorAgrees, selectorRows⟩
  have selectorStart : offset + localLength windowsCompleted.operations =
      selectorOffset offset := by
    rw [windowOperations, Sequence.localLength_append]
    have windowLength : localLength (windowOps interface coordinate offset) =
        digestRoundCount * DigestWindow.logicalPrivateCount := by
      simpa [windowOps, windowOpsPrefix] using
        windowOpsPrefix_localLength interface coordinate offset digestRoundCount
    rw [windowLength]
    simp [selectorOffset, windowBase]
    omega
  have selectorAgreesExact :
      AgreesOutside windowsCompleted.current selectorEnv (selectorOffset offset)
        (localLength
          (Circuit.ops (selectorCircuit interface coordinate offset).main
            (selectorOffset offset))) := by
    change AgreesOutside windowsCompleted.current selectorEnv
      (selectorOffset offset)
        (localLength
          (Circuit.ops
            (First54.main (selectorInterface interface coordinate offset))
              (selectorOffset offset)))
    rw [First54.localLength_eq]
    exact selectorAgrees
  rcases Sequence.appendBuiltAt windowsCompleted selectorName
      (selectorCircuit interface coordinate offset) (selectorOffset offset)
      selectorStart
      (selectorScope interface coordinate offset windowsCompleted.current
        prefixSpecification.window)
      selectorEnv selectorAgreesExact selectorRows with
    ⟨completed, completedOperations, _, _, _⟩
  have operationsEq : completed.operations =
      opsAt interface coordinate offset := by
    rw [completedOperations, windowOperations]
    rfl
  refine ⟨completed.current, ?_, ?_⟩
  · have agrees := completed.agrees
    rw [operationsEq] at agrees
    change AgreesOutside env completed.current offset
      (localLength (opsAt interface coordinate offset))
    exact agrees
  · change holdsFlat completed.current (opsAt interface coordinate offset)
    rw [← operationsEq]
    exact completed.rows

theorem completeness (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (relation : RelationHolds interface coordinate offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface coordinate) offset)) ∧
      holdsFlat completed
        (Circuit.ops (main interface coordinate) offset) := by
  apply complete_of_success interface coordinate env offset assumptions
  rcases relation with ⟨coefficients, success, _, _⟩
  exact ⟨coefficients, success⟩

def circuit (interface : Interface) (coordinate : Nat) : FormalCircuit where
  main := main interface coordinate
  assumptions := Assumptions interface
  spec := RelationHolds interface coordinate
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface coordinate
  rowCount_eq := flatConstraints_length interface coordinate
  soundness := by
    intro env offset assumptions rows
    exact rows_imply_relation interface coordinate offset env assumptions rows
  completeness := completeness interface coordinate

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler
