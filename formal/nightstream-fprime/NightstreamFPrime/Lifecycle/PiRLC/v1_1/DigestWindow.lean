import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane
import NightstreamFPrime.Lifecycle.Transcript

/-!
Owns one complete PiRLC sampler digest block.

The parent calls four exact lane children in rate-lane order and one
verifier-owned Poseidon2 permutation. It exposes eight candidates in
lane-low/high order and proves that candidates and successor state are the
two projections of `Lifecycle.Transcript.PiRlcSampler.digestBlock`.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

def logicalPrivateCount : Nat := 992
def logicalRowCount : Nat := 1004

abbrev State := NightstreamFPrime.Lifecycle.Transcript.State
abbrev EState := Layer.EState
abbrev Chunk := ProductionAlphabet.Chunk

structure Interface where
  initialState : Nat → EState

def rateLane (lane : Fin 4) : Fin 8 :=
  ⟨lane.val, lt_trans lane.isLt (by decide)⟩

def laneInterface (interface : Interface) (parentOffset : Nat)
    (lane : Fin 4) : DigestLane.Interface where
  source := fun _ => interface.initialState parentOffset (rateLane lane)

def laneOffset (offset : Nat) (lane : Fin 4) : Nat :=
  offset + lane.val * DigestLane.logicalPrivateCount

def permutationOffset (offset : Nat) : Nat :=
  offset + 4 * DigestLane.logicalPrivateCount

def finalOffset (offset : Nat) : Nat :=
  offset + logicalPrivateCount

def laneCircuit (interface : Interface) (parentOffset : Nat)
    (lane : Fin 4) : FormalCircuit :=
  DigestLane.circuit (laneInterface interface parentOffset lane)

def permutationInterface (interface : Interface) (parentOffset : Nat) :
    Permutation.Owned.Interface where
  initialState := fun _ => interface.initialState parentOffset

def permutationCircuit (interface : Interface) (parentOffset : Nat) :
  FormalCircuit :=
  Permutation.Owned.circuit (permutationInterface interface parentOffset)

def laneName (lane : Fin 4) : String :=
  "pirlc.v1_1.digest_window.lane_" ++ toString lane.val

def permutationName : String :=
  "pirlc.v1_1.digest_window.poseidon2_permutation"

def laneOp (interface : Interface) (offset : Nat) (lane : Fin 4) : Op :=
  Sequence.childOp (laneName lane) (laneCircuit interface offset lane)
    (laneOffset offset lane)

def permutationOp (interface : Interface) (offset : Nat) : Op :=
  Sequence.childOp permutationName (permutationCircuit interface offset)
    (permutationOffset offset)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [laneOp interface offset 0, laneOp interface offset 1,
    laneOp interface offset 2, laneOp interface offset 3,
    permutationOp interface offset]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt interface offset)

def output (interface : Interface) (offset : Nat) : EState :=
  Permutation.Owned.output (permutationInterface interface offset)
    (permutationOffset offset)

def laneOf (position : Fin chunksPerDigest) : Fin 4 :=
  ⟨position.val / 2, by
    have bounded := position.isLt
    simp only [chunksPerDigest] at bounded
    omega⟩

def partOf (position : Fin chunksPerDigest) : Fin 2 :=
  ⟨position.val % 2, Nat.mod_lt _ (by decide)⟩

def candidate (offset : Nat) (position : Fin chunksPerDigest) : Expr :=
  DigestLane.candidate (laneOffset offset (laneOf position)) (partOf position)

def remainder (offset : Nat) (position : Fin chunksPerDigest) : Expr :=
  DigestLane.remainder (laneOffset offset (laneOf position)) (partOf position)

def reject (offset : Nat) (position : Fin chunksPerDigest) : Expr :=
  DigestLane.reject (laneOffset offset (laneOf position)) (partOf position)

def centeredCoefficient (offset : Nat)
    (position : Fin chunksPerDigest) : Expr :=
  DigestLane.centeredCoefficient
    (laneOffset offset (laneOf position)) (partOf position)

def evalState (env : Env) (state : EState) : State :=
  List.ofFn (Layer.evalState env state)

private theorem ofFn_getD {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (lane : Fin count) (fallback : Alpha) :
    (List.ofFn values).getD lane.val fallback = values lane := by
  rw [List.getD_eq_get (List.ofFn values) fallback
    ⟨lane.val, by simp⟩]
  simp

def evalChunk (env : Env) (offset : Nat)
    (position : Fin chunksPerDigest) : Chunk :=
  ⟨((candidate offset position).eval env).val % chunkModulus,
    Nat.mod_lt _ (by decide)⟩

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  lane : ∀ lane, DigestLane.SpecHolds (laneInterface interface offset lane)
    (laneOffset offset lane) env
  permutation : Permutation.Owned.SpecHolds
    (permutationInterface interface offset) (permutationOffset offset) env

theorem laneAssumptions (interface : Interface) (offset : Nat)
    (lane : Fin 4) {env : Env}
    (assumptions : Assumptions interface offset env) :
    DigestLane.Assumptions (laneInterface interface offset lane)
      (laneOffset offset lane) env := by
  apply Expr.VarsBelow.mono _ (assumptions (rateLane lane))
  simp [laneOffset]

private theorem permutationAssumptions (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    Permutation.Owned.Assumptions (permutationInterface interface offset)
      (permutationOffset offset) env := by
  intro lane
  apply Expr.VarsBelow.mono _ (assumptions lane)
  simp [permutationOffset]

private theorem laneChildLength (interface : Interface) (offset : Nat)
    (lane : Fin 4) :
    localLength (Circuit.ops (laneCircuit interface offset lane).main
      (laneOffset offset lane)) = DigestLane.logicalPrivateCount := by
  change localLength (Circuit.ops
    (DigestLane.circuit (laneInterface interface offset lane)).main
      (laneOffset offset lane)) = _
  exact DigestLane.localLength_eq _ _

private theorem permutationChildLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (permutationCircuit interface offset).main
      (permutationOffset offset)) = 592 := by
  change localLength (Permutation.Owned.operations
    (permutationInterface interface offset) (permutationOffset offset)) = 592
  exact Permutation.Owned.localLength_eq _ _

@[simp] private theorem laneOp_localLength (interface : Interface) (offset : Nat)
    (lane : Fin 4) :
    (laneOp interface offset lane).localLength =
      DigestLane.logicalPrivateCount := by
  rw [laneOp, Sequence.childOp_localLength]
  exact laneChildLength interface offset lane

@[simp] private theorem permutationOp_localLength (interface : Interface)
    (offset : Nat) :
    (permutationOp interface offset).localLength = 592 := by
  rw [permutationOp, Sequence.childOp_localLength]
  exact permutationChildLength interface offset

@[simp] private theorem laneOp_rowCount (interface : Interface) (offset : Nat)
    (lane : Fin 4) :
    (laneOp interface offset lane).rowCount = DigestLane.logicalRowCount := by
  rfl

@[simp] private theorem permutationOp_rowCount (interface : Interface)
    (offset : Nat) :
    (permutationOp interface offset).rowCount = 592 := by
  rfl

private theorem laneScope (interface : Interface) (offset : Nat)
    (lane : Fin 4) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (laneCircuit interface offset lane).main
        (laneOffset offset lane)),
      expression.VarsBelow
        (laneOffset offset lane + localLength
          (Circuit.ops (laneCircuit interface offset lane).main
            (laneOffset offset lane))) := by
  exact DigestLane.flatConstraints_varsBelow
    (laneInterface interface offset lane) (laneOffset offset lane)
    (laneAssumptions interface offset lane assumptions)

set_option maxRecDepth 100000 in -- fixed-size: one four-lane digest window, not artifact data
private theorem permutationScope (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (permutationCircuit interface offset).main
        (permutationOffset offset)),
      expression.VarsBelow
        (permutationOffset offset + localLength
          (Circuit.ops (permutationCircuit interface offset).main
            (permutationOffset offset))) := by
  exact Permutation.Owned.flatConstraints_varsBelow
    (permutationInterface interface offset) (permutationOffset offset)
    (permutationAssumptions interface offset assumptions)

private theorem laneCall_sound (interface : Interface) (offset : Nat)
    (lane : Fin 4) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) :
    DigestLane.SpecHolds (laneInterface interface offset lane)
      (laneOffset offset lane) env := by
  have member : laneOp interface offset lane ∈ opsAt interface offset := by
    fin_cases lane <;> simp [opsAt]
  have callHolds := rows (laneOp interface offset lane) member
  change DigestLane.Assumptions (laneInterface interface offset lane)
      (laneOffset offset lane) env →
    DigestLane.SpecHolds (laneInterface interface offset lane)
      (laneOffset offset lane) env at callHolds
  exact callHolds (laneAssumptions interface offset lane assumptions)

private theorem permutationCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) :
    Permutation.Owned.SpecHolds (permutationInterface interface offset)
      (permutationOffset offset) env := by
  have callHolds := rows (permutationOp interface offset) (by simp [opsAt])
  change Permutation.Owned.Assumptions (permutationInterface interface offset)
      (permutationOffset offset) env →
    Permutation.Owned.SpecHolds (permutationInterface interface offset)
      (permutationOffset offset) env at callHolds
  exact callHolds (permutationAssumptions interface offset assumptions)

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  change holds env (opsAt interface offset) at rows
  exact ⟨fun lane => laneCall_sound interface offset lane env assumptions rows,
    permutationCall_sound interface offset env assumptions rows⟩

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = logicalPrivateCount := by
  change localLength (opsAt interface offset) = logicalPrivateCount
  simp only [opsAt, localLength, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero, laneOp_localLength, permutationOp_localLength,
    DigestLane.logicalPrivateCount, logicalPrivateCount]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount := by
  change (flatConstraints (opsAt interface offset)).length = logicalRowCount
  rw [flatConstraints_length_eq_rowCount]
  simp only [opsAt, rowCount, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero, laneOp_rowCount, permutationOp_rowCount]
  change DigestLane.logicalRowCount + DigestLane.logicalRowCount +
      DigestLane.logicalRowCount + DigestLane.logicalRowCount + 592 =
    logicalRowCount
  decide

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (laneCircuit interface offset 0).main
        (laneOffset offset 0)) ++
      flatConstraints (Circuit.ops (laneCircuit interface offset 1).main
        (laneOffset offset 1)) ++
      flatConstraints (Circuit.ops (laneCircuit interface offset 2).main
        (laneOffset offset 2)) ++
      flatConstraints (Circuit.ops (laneCircuit interface offset 3).main
        (laneOffset offset 3)) ++
      flatConstraints (Circuit.ops (permutationCircuit interface offset).main
        (permutationOffset offset)) := by
  simp only [opsAt, flatConstraints, List.flatMap_cons, List.flatMap_nil,
    laneOp, permutationOp, Sequence.childOp, Op.flatConstraints,
    FormalCircuit.asSubcircuit_constraints, List.append_nil]
  simp only [List.append_assoc]

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt]
  have lengthEq : localLength (opsAt interface offset) = logicalPrivateCount := by
    simpa using localLength_eq interface offset
  rw [lengthEq]
  intro expression member
  rcases List.mem_append.mp member with beforePermutation | permutationMember
  · rcases List.mem_append.mp beforePermutation with beforeLane3 | lane3Member
    · rcases List.mem_append.mp beforeLane3 with beforeLane2 | lane2Member
      · rcases List.mem_append.mp beforeLane2 with lane0Member | lane1Member
        · exact Expr.VarsBelow.mono _
            (laneScope interface offset 0 assumptions expression lane0Member)
            (by rw [laneChildLength]; norm_num [laneOffset,
              DigestLane.logicalPrivateCount, logicalPrivateCount])
        · exact Expr.VarsBelow.mono _
            (laneScope interface offset 1 assumptions expression lane1Member)
            (by rw [laneChildLength]; norm_num [laneOffset,
              DigestLane.logicalPrivateCount, logicalPrivateCount])
      · exact Expr.VarsBelow.mono _
          (laneScope interface offset 2 assumptions expression lane2Member)
          (by rw [laneChildLength]; norm_num [laneOffset,
            DigestLane.logicalPrivateCount, logicalPrivateCount])
    · exact Expr.VarsBelow.mono _
        (laneScope interface offset 3 assumptions expression lane3Member)
        (by rw [laneChildLength]; norm_num [laneOffset,
          DigestLane.logicalPrivateCount, logicalPrivateCount])
  · exact Expr.VarsBelow.mono _
      (permutationScope interface offset assumptions expression permutationMember)
      (by rw [permutationChildLength]; norm_num [permutationOffset,
        DigestLane.logicalPrivateCount, logicalPrivateCount])

theorem output_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ lane, (output interface offset lane).VarsBelow
      (offset + logicalPrivateCount) := by
  have outputScope := Permutation.Owned.output_varsBelow
    (permutationInterface interface offset) (permutationOffset offset)
      (permutationAssumptions interface offset assumptions)
  intro lane
  simpa [output, permutationOffset, logicalPrivateCount] using outputScope lane

private theorem appendLane
    {initial : Env} {offset : Nat}
    (before : Sequence.Prefix initial offset)
    (interface : Interface) (lane : Fin 4)
    (startEq : offset + localLength before.operations = laneOffset offset lane)
    (assumptions : Assumptions interface offset before.current) :
    ∃ after : Sequence.Prefix initial offset,
      after.operations = before.operations ++ [laneOp interface offset lane] ∧
      offset + localLength after.operations =
        laneOffset offset lane + DigestLane.logicalPrivateCount ∧
      Sequence.PreservesPrefix before after := by
  have childAssumptions := laneAssumptions interface offset lane assumptions
  rcases DigestLane.complete (laneInterface interface offset lane) before.current
      (laneOffset offset lane) childAssumptions with
    ⟨laneEnv, laneAgrees, laneRows⟩
  rcases Sequence.appendBuiltAt before (laneName lane)
      (laneCircuit interface offset lane) (laneOffset offset lane)
      startEq (laneScope interface offset lane assumptions)
      laneEnv laneAgrees laneRows with
    ⟨after, operationsEq, endEq, preserves, _⟩
  refine ⟨after, ?_, ?_, preserves⟩
  · exact operationsEq
  · rw [endEq, laneChildLength]

set_option maxRecDepth 100000 in -- fixed-size: one four-lane digest window, not artifact data
theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let empty := Sequence.empty env offset
  rcases appendLane empty interface 0 (by
      dsimp [empty, Sequence.empty]
      rfl) assumptions with
    ⟨after0, ops0, end0, _⟩
  have prefix0 : after0.operations = [laneOp interface offset 0] := by
    simpa [empty] using ops0
  rcases appendLane after0 interface 1 (by
      rw [end0]
      norm_num [laneOffset, DigestLane.logicalPrivateCount]) assumptions with
    ⟨after1, ops1, end1, _⟩
  have prefix1 : after1.operations =
      [laneOp interface offset 0, laneOp interface offset 1] := by
    rw [ops1, prefix0]
    rfl
  rcases appendLane after1 interface 2 (by
      rw [end1]
      norm_num [laneOffset, DigestLane.logicalPrivateCount]) assumptions with
    ⟨after2, ops2, end2, _⟩
  have prefix2 : after2.operations =
      [laneOp interface offset 0, laneOp interface offset 1,
        laneOp interface offset 2] := by
    rw [ops2, prefix1]
    rfl
  rcases appendLane after2 interface 3 (by
      rw [end2]
      norm_num [laneOffset, DigestLane.logicalPrivateCount]) assumptions with
    ⟨after3, ops3, end3, _⟩
  have prefix3 : after3.operations =
      [laneOp interface offset 0, laneOp interface offset 1,
        laneOp interface offset 2, laneOp interface offset 3] := by
    rw [ops3, prefix2]
    rfl
  have permutationStart : offset + localLength after3.operations =
      permutationOffset offset := by
    rw [end3]
    norm_num [laneOffset, permutationOffset, DigestLane.logicalPrivateCount]
  have permutationAssumption :
      Permutation.Owned.Assumptions (permutationInterface interface offset)
        (permutationOffset offset) after3.current :=
    permutationAssumptions interface offset (env := after3.current) assumptions
  rcases Permutation.Owned.complete (permutationInterface interface offset)
      after3.current (permutationOffset offset) permutationAssumption with
    ⟨permutationEnv, permutationAgrees, permutationRows⟩
  rcases Sequence.appendBuiltAt after3 permutationName
      (permutationCircuit interface offset) (permutationOffset offset)
      permutationStart (permutationScope interface offset assumptions)
      permutationEnv permutationAgrees permutationRows with
    ⟨completed, completedOps, _, _, _⟩
  have operationsEq : completed.operations = opsAt interface offset := by
    rw [completedOps, prefix3]
    rfl
  refine ⟨completed.current, ?_, ?_⟩
  · have agrees := completed.agrees
    rw [operationsEq] at agrees
    have lengthEq : localLength (opsAt interface offset) =
        logicalPrivateCount := by
      simpa using localLength_eq interface offset
    rw [lengthEq] at agrees
    exact agrees
  · simpa [operationsEq] using completed.rows

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  complete interface env offset assumptions

/-- Each exposed value is the exact concrete digest chunk at that position. -/
theorem candidateValue_eq_digestChunks (interface : Interface) (offset : Nat)
    (env : Env) (specification : SpecHolds interface offset env)
    (position : Fin chunksPerDigest) :
    ((candidate offset position).eval env).val =
      (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
        (evalState env (interface.initialState offset)) position).val := by
  have extracted := DigestLane.candidateValue_eq
    (laneInterface interface offset (laneOf position))
    (laneOffset offset (laneOf position)) env
    (specification.lane (laneOf position)) (partOf position)
  have stateLane :
      (evalState env (interface.initialState offset)).getD
          (position.val / 2) 0 =
        (interface.initialState offset (rateLane (laneOf position))).eval env := by
    simpa [evalState, rateLane, laneOf] using
      (ofFn_getD (Layer.evalState env (interface.initialState offset))
        (rateLane (laneOf position)) 0)
  calc
    ((candidate offset position).eval env).val =
        ((interface.initialState offset
          (rateLane (laneOf position))).eval env).val /
            2 ^ (16 * (partOf position).val) %
              2 ^ Candidate16Five.candidateBitCount := extracted
    _ = (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
          (evalState env (interface.initialState offset)) position).val := by
      unfold NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
      change ((interface.initialState offset
          (rateLane (laneOf position))).eval env).val /
            2 ^ (16 * (partOf position).val) %
              2 ^ Candidate16Five.candidateBitCount =
        ((evalState env (interface.initialState offset)).getD
            (position.val / 2) 0).val /
          2 ^ (16 * (position.val % 2)) % chunkModulus
      rw [stateLane]
      norm_num [partOf, Candidate16Five.candidateBitCount, chunkModulus]

theorem evalChunk_eq_digestChunks (interface : Interface) (offset : Nat)
    (env : Env) (specification : SpecHolds interface offset env)
    (position : Fin chunksPerDigest) :
    evalChunk env offset position =
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
        (evalState env (interface.initialState offset)) position := by
  apply Fin.ext
  change ((candidate offset position).eval env).val % chunkModulus = _
  rw [candidateValue_eq_digestChunks interface offset env specification position]
  exact Nat.mod_eq_of_lt
    (NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestChunks
      (evalState env (interface.initialState offset)) position).isLt

/-- Candidate vector and successor state are exactly the two projections of
the verifier-owned concrete digest-block function. -/
theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) (counter : Nat) :
    (evalState env (output interface offset),
        fun position => evalChunk env offset position) =
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestBlock
        (evalState env (interface.initialState offset)) counter := by
  apply Prod.ext
  · exact specification.permutation
  · funext position
    exact evalChunk_eq_digestChunks interface offset env specification position

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow
