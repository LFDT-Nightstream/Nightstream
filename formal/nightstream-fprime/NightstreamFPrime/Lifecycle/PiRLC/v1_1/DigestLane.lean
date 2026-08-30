import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Gadgets.Sampling.Candidate16Five

/-!
Owns one rate-lane decomposition for the PiRLC sampler.

The caller supplies one transcript-state lane. The lane circuit calls one
canonical-u64 child and two exact 16-bit candidate decoders in low/high order.
It adds no boundary row and does not own Poseidon2 or first-accepted selection.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling

def logicalPrivateCount : Nat := 100
def logicalRowCount : Nat := 103

structure Interface where
  source : Nat → Expr

def canonicalInterface (interface : Interface) (parentOffset : Nat) :
    CanonicalU64.Interface where
  source := fun _ => interface.source parentOffset

def canonicalOffset (offset : Nat) : Nat := offset

def lowPart : Fin 2 := ⟨0, by decide⟩
def highPart : Fin 2 := ⟨1, by decide⟩

def decoderOffset (offset : Nat) (part : Fin 2) : Nat :=
  offset + CanonicalU64.auxiliaryCount +
    part.val * Candidate16Five.auxiliaryCount

def finalOffset (offset : Nat) : Nat :=
  offset + logicalPrivateCount

def decoderInterface (offset : Nat) (part : Fin 2) :
    Candidate16Five.Interface :=
  Candidate16Five.canonicalWindowInterface (canonicalOffset offset) part

def canonicalCircuit (interface : Interface) (parentOffset : Nat) :
    FormalCircuit :=
  CanonicalU64.circuit (canonicalInterface interface parentOffset)

def decoderCircuit (parentOffset : Nat) (part : Fin 2) : FormalCircuit :=
  Candidate16Five.circuit (decoderInterface parentOffset part)

def canonicalName : String := "pirlc.v1_1.digest_lane.canonical_u64"
def lowName : String := "pirlc.v1_1.digest_lane.candidate_low"
def highName : String := "pirlc.v1_1.digest_lane.candidate_high"

def canonicalOp (interface : Interface) (offset : Nat) : Op :=
  Sequence.childOp canonicalName (canonicalCircuit interface offset)
    (canonicalOffset offset)

def lowOp (offset : Nat) : Op :=
  Sequence.childOp lowName (decoderCircuit offset lowPart)
    (decoderOffset offset lowPart)

def highOp (offset : Nat) : Op :=
  Sequence.childOp highName (decoderCircuit offset highPart)
    (decoderOffset offset highPart)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [canonicalOp interface offset, lowOp offset, highOp offset]

private theorem canonicalChildLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (canonicalCircuit interface offset).main
      (canonicalOffset offset)) = CanonicalU64.auxiliaryCount := by
  change localLength (CanonicalU64.operations
    (canonicalInterface interface offset) (canonicalOffset offset)) = _
  exact CanonicalU64.localLength_eq _ _

private theorem decoderChildLength (offset : Nat) (part : Fin 2) :
    localLength (Circuit.ops (decoderCircuit offset part).main
      (decoderOffset offset part)) = Candidate16Five.auxiliaryCount := by
  change localLength (Candidate16Five.operations
    (decoderInterface offset part) (decoderOffset offset part)) = _
  exact Candidate16Five.localLength_eq _ _

@[simp] private theorem canonicalOp_localLength (interface : Interface)
    (offset : Nat) :
    (canonicalOp interface offset).localLength =
      CanonicalU64.auxiliaryCount := by
  rw [canonicalOp, Sequence.childOp_localLength]
  exact canonicalChildLength interface offset

@[simp] private theorem lowOp_localLength (offset : Nat) :
    (lowOp offset).localLength = Candidate16Five.auxiliaryCount := by
  rw [lowOp, Sequence.childOp_localLength]
  exact decoderChildLength offset lowPart

@[simp] private theorem highOp_localLength (offset : Nat) :
    (highOp offset).localLength = Candidate16Five.auxiliaryCount := by
  rw [highOp, Sequence.childOp_localLength]
  exact decoderChildLength offset highPart

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt interface offset)

def candidate (offset : Nat) (part : Fin 2) : Expr :=
  (decoderInterface offset part).candidate (decoderOffset offset part)

def remainder (offset : Nat) (part : Fin 2) : Expr :=
  Candidate16Five.remainderExpr (decoderOffset offset part)

def reject (offset : Nat) (part : Fin 2) : Expr :=
  Candidate16Five.rejectExpr (decoderOffset offset part)

def centeredCoefficient (offset : Nat) (part : Fin 2) : Expr :=
  Candidate16Five.centeredCoefficientExpr (decoderOffset offset part)

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.source offset).VarsBelow offset

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  canonical : CanonicalU64.SpecHolds
    (canonicalInterface interface offset) (canonicalOffset offset) env
  low : Candidate16Five.SpecHolds (decoderInterface offset lowPart)
    (decoderOffset offset lowPart) env
  high : Candidate16Five.SpecHolds (decoderInterface offset highPart)
    (decoderOffset offset highPart) env

theorem SpecHolds.decoder
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env)
    (part : Fin 2) :
    Candidate16Five.SpecHolds (decoderInterface offset part)
      (decoderOffset offset part) env := by
  fin_cases part
  · exact specification.low
  · exact specification.high

private theorem canonicalAssumptions (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    CanonicalU64.Assumptions (canonicalInterface interface offset)
      (canonicalOffset offset) env := by
  simpa [Assumptions, canonicalInterface, canonicalOffset] using assumptions

private theorem decoderWordEnd (offset : Nat) (part : Fin 2) :
    canonicalOffset offset + CanonicalU64.auxiliaryCount ≤
      decoderOffset offset part := by
  simp [canonicalOffset, decoderOffset]

private theorem decoderCandidateBelow (offset : Nat) (part : Fin 2) :
    ((decoderInterface offset part).candidate
      (decoderOffset offset part)).VarsBelow (decoderOffset offset part) := by
  apply CanonicalU64.weightedExpr_varsBelow
  have partBound := part.isLt
  simp only [decoderOffset, canonicalOffset, Candidate16Five.candidateBitCount,
    CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount] at *
  omega

private theorem decoderBitsBelow (offset : Nat) (part : Fin 2) :
    ∀ index, index < Candidate16Five.candidateBitCount →
      ((decoderInterface offset part).candidateBit
        (decoderOffset offset part) index).VarsBelow
          (decoderOffset offset part) := by
  intro index bounded
  apply CanonicalU64.bitExpr_varsBelow
  have partBound := part.isLt
  simp only [decoderOffset, canonicalOffset, Candidate16Five.candidateBitCount,
    CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount] at *
  omega

private theorem canonicalScope (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (canonicalCircuit interface offset).main
        (canonicalOffset offset)),
      expression.VarsBelow
        (canonicalOffset offset + localLength
          (Circuit.ops (canonicalCircuit interface offset).main
            (canonicalOffset offset))) := by
  have scope := CanonicalU64.flatConstraints_varsBelow
    (canonicalInterface interface offset) (canonicalOffset offset)
    (canonicalAssumptions interface offset assumptions)
  change ∀ expression ∈ flatConstraints
      (CanonicalU64.operations (canonicalInterface interface offset)
        (canonicalOffset offset)),
    expression.VarsBelow
      (canonicalOffset offset + localLength
        (CanonicalU64.operations (canonicalInterface interface offset)
          (canonicalOffset offset)))
  rw [CanonicalU64.localLength_eq]
  exact scope

private theorem decoderScope (offset : Nat) (part : Fin 2) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (decoderCircuit offset part).main
        (decoderOffset offset part)),
      expression.VarsBelow
        (decoderOffset offset part + localLength
          (Circuit.ops (decoderCircuit offset part).main
            (decoderOffset offset part))) := by
  have scope := Candidate16Five.flatConstraints_varsBelow
    (decoderInterface offset part) (decoderOffset offset part)
    (decoderCandidateBelow offset part) (decoderBitsBelow offset part)
  change ∀ expression ∈ flatConstraints
      (Candidate16Five.operations (decoderInterface offset part)
        (decoderOffset offset part)),
    expression.VarsBelow
      (decoderOffset offset part + localLength
        (Candidate16Five.operations (decoderInterface offset part)
          (decoderOffset offset part)))
  rw [Candidate16Five.localLength_eq]
  exact scope

private theorem canonicalCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) :
    CanonicalU64.SpecHolds (canonicalInterface interface offset)
      (canonicalOffset offset) env := by
  have callHolds := rows (canonicalOp interface offset) (by simp [opsAt])
  change CanonicalU64.Assumptions (canonicalInterface interface offset)
      (canonicalOffset offset) env →
    CanonicalU64.SpecHolds (canonicalInterface interface offset)
      (canonicalOffset offset) env at callHolds
  exact callHolds (canonicalAssumptions interface offset assumptions)

private theorem lowCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (canonical : CanonicalU64.SpecHolds
      (canonicalInterface interface offset) (canonicalOffset offset) env) :
    Candidate16Five.SpecHolds (decoderInterface offset lowPart)
      (decoderOffset offset lowPart) env := by
  have callHolds := rows (lowOp offset) (by simp [opsAt])
  change Candidate16Five.Assumptions (decoderInterface offset lowPart)
      (decoderOffset offset lowPart) env →
    Candidate16Five.SpecHolds (decoderInterface offset lowPart)
      (decoderOffset offset lowPart) env at callHolds
  exact callHolds (Candidate16Five.canonicalWindowAssumptions
    (canonicalOffset offset) (decoderOffset offset lowPart) lowPart env
    (decoderWordEnd offset lowPart) canonical)

private theorem highCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (canonical : CanonicalU64.SpecHolds
      (canonicalInterface interface offset) (canonicalOffset offset) env) :
    Candidate16Five.SpecHolds (decoderInterface offset highPart)
      (decoderOffset offset highPart) env := by
  have callHolds := rows (highOp offset) (by simp [opsAt])
  change Candidate16Five.Assumptions (decoderInterface offset highPart)
      (decoderOffset offset highPart) env →
    Candidate16Five.SpecHolds (decoderInterface offset highPart)
      (decoderOffset offset highPart) env at callHolds
  exact callHolds (Candidate16Five.canonicalWindowAssumptions
    (canonicalOffset offset) (decoderOffset offset highPart) highPart env
    (decoderWordEnd offset highPart) canonical)

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  change holds env (opsAt interface offset) at rows
  have canonical := canonicalCall_sound interface offset env assumptions rows
  exact ⟨canonical,
    lowCall_sound interface offset env rows canonical,
    highCall_sound interface offset env rows canonical⟩

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = logicalPrivateCount := by
  change localLength (opsAt interface offset) = logicalPrivateCount
  simp only [opsAt, localLength, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero]
  rw [canonicalOp_localLength, lowOp_localLength, highOp_localLength]
  simp only [CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount,
    logicalPrivateCount]

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (canonicalCircuit interface offset).main
        (canonicalOffset offset)) ++
      flatConstraints (Circuit.ops (decoderCircuit offset lowPart).main
        (decoderOffset offset lowPart)) ++
      flatConstraints (Circuit.ops (decoderCircuit offset highPart).main
        (decoderOffset offset highPart)) := by
  simp only [opsAt, flatConstraints, List.flatMap_cons, List.flatMap_nil,
    canonicalOp, lowOp, highOp, Sequence.childOp, Op.flatConstraints,
    FormalCircuit.asSubcircuit_constraints, List.append_nil]
  rw [List.append_assoc]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount := by
  change (flatConstraints (opsAt interface offset)).length = logicalRowCount
  rw [flatConstraints_opsAt, List.length_append, List.length_append]
  change CanonicalU64.exactRowCount + Candidate16Five.exactRowCount +
      Candidate16Five.exactRowCount = logicalRowCount
  decide

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
  rcases List.mem_append.mp member with previous | highMember
  · rcases List.mem_append.mp previous with canonicalMember | lowMember
    · exact Expr.VarsBelow.mono _
        (canonicalScope interface offset assumptions expression canonicalMember)
        (by
          rw [canonicalChildLength]
          simp [canonicalOffset, logicalPrivateCount,
            CanonicalU64.auxiliaryCount])
    · exact Expr.VarsBelow.mono _
        (decoderScope offset lowPart expression lowMember)
        (by
          rw [decoderChildLength]
          simp [decoderOffset, lowPart, logicalPrivateCount,
            CanonicalU64.auxiliaryCount,
            Candidate16Five.auxiliaryCount])
  · exact Expr.VarsBelow.mono _
      (decoderScope offset highPart expression highMember)
      (by
        rw [decoderChildLength]
        simp [decoderOffset, highPart, logicalPrivateCount,
          CanonicalU64.auxiliaryCount,
          Candidate16Five.auxiliaryCount])

private theorem decoderCandidate_varsSatisfy (offset : Nat) (part : Fin 2)
    (allowed : Nat → Prop)
    (localSupported : ∀ index, index < logicalPrivateCount →
      allowed (offset + index)) :
    ((decoderInterface offset part).candidate (decoderOffset offset part)
      ).VarsSatisfy allowed := by
  unfold decoderInterface Candidate16Five.canonicalWindowInterface
  apply CanonicalU64.weightedExpr_varsSatisfy
  intro index bounded
  have partLt := part.isLt
  simpa [canonicalOffset, Nat.add_assoc] using
    localSupported (16 * part.val + index) (by
      norm_num [Candidate16Five.candidateBitCount, logicalPrivateCount] at bounded partLt ⊢
      omega)

private theorem decoderBits_varsSatisfy (offset : Nat) (part : Fin 2)
    (allowed : Nat → Prop)
    (localSupported : ∀ index, index < logicalPrivateCount →
      allowed (offset + index)) :
    ∀ index, index < Candidate16Five.candidateBitCount →
      ((decoderInterface offset part).candidateBit
        (decoderOffset offset part) index).VarsSatisfy allowed := by
  intro index bounded
  unfold decoderInterface Candidate16Five.canonicalWindowInterface
  apply CanonicalU64.bitExpr_varsSatisfy
  have partLt := part.isLt
  exact localSupported (16 * part.val + index) (by
    norm_num [Candidate16Five.candidateBitCount, logicalPrivateCount] at bounded partLt ⊢
    omega)

private theorem canonicalScope_varsSatisfy (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (sourceSupported : (interface.source offset).VarsSatisfy allowed)
    (localSupported : ∀ index, index < logicalPrivateCount →
      allowed (offset + index)) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (canonicalCircuit interface offset).main
          (canonicalOffset offset)),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints
      (CanonicalU64.operations (canonicalInterface interface offset)
        (canonicalOffset offset)), expression.VarsSatisfy allowed
  apply CanonicalU64.flatConstraints_varsSatisfy
  · exact sourceSupported
  · intro index bounded
    exact localSupported index (by
      norm_num [CanonicalU64.auxiliaryCount, logicalPrivateCount] at bounded ⊢
      omega)

private theorem decoderScope_varsSatisfy (offset : Nat) (part : Fin 2)
    (allowed : Nat → Prop)
    (localSupported : ∀ index, index < logicalPrivateCount →
      allowed (offset + index)) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (decoderCircuit offset part).main
          (decoderOffset offset part)),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints
      (Candidate16Five.operations (decoderInterface offset part)
        (decoderOffset offset part)), expression.VarsSatisfy allowed
  apply Candidate16Five.flatConstraints_varsSatisfy
  · exact decoderCandidate_varsSatisfy offset part allowed localSupported
  · exact decoderBits_varsSatisfy offset part allowed localSupported
  · intro index bounded
    have partLt := part.isLt
    unfold decoderOffset
    simpa [Nat.add_assoc] using localSupported
      (CanonicalU64.auxiliaryCount +
        part.val * Candidate16Five.auxiliaryCount + index) (by
          norm_num [CanonicalU64.auxiliaryCount,
            Candidate16Five.auxiliaryCount, logicalPrivateCount] at bounded partLt ⊢
          omega)

/-- Every digest-lane constraint reads only the caller-supported Poseidon2
lane or one of the exact 100 lane-local columns. -/
theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (sourceSupported : (interface.source offset).VarsSatisfy allowed)
    (localSupported : ∀ index, index < logicalPrivateCount →
      allowed (offset + index)) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  rw [flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with previous | highMember
  · rcases List.mem_append.mp previous with canonicalMember | lowMember
    · exact canonicalScope_varsSatisfy interface offset allowed
        sourceSupported localSupported expression canonicalMember
    · exact decoderScope_varsSatisfy offset lowPart allowed localSupported
        expression lowMember
  · exact decoderScope_varsSatisfy offset highPart allowed localSupported
      expression highMember

private theorem prefixCanonicalSpec
    (interface : Interface) (offset : Nat) (initial : Env)
    (completedPrefix : Sequence.Prefix initial offset)
    (assumptions : Assumptions interface offset completedPrefix.current)
    (member : canonicalOp interface offset ∈ completedPrefix.operations) :
    CanonicalU64.SpecHolds (canonicalInterface interface offset)
      (canonicalOffset offset) completedPrefix.current := by
  have parentRows := holdsFlat_implies_holds completedPrefix.current
    completedPrefix.operations completedPrefix.rows
  have callHolds := parentRows (canonicalOp interface offset) member
  change CanonicalU64.Assumptions (canonicalInterface interface offset)
      (canonicalOffset offset) completedPrefix.current →
    CanonicalU64.SpecHolds (canonicalInterface interface offset)
      (canonicalOffset offset) completedPrefix.current at callHolds
  exact callHolds (canonicalAssumptions interface offset assumptions)

theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let empty := Sequence.empty env offset
  rcases CanonicalU64.complete (canonicalInterface interface offset) env
      (canonicalOffset offset) (canonicalAssumptions interface offset assumptions) with
    ⟨canonicalEnv, canonicalAgrees, canonicalRows⟩
  rcases Sequence.appendBuiltAt empty canonicalName
      (canonicalCircuit interface offset) (canonicalOffset offset)
      (by
        dsimp [empty, Sequence.empty]
        rfl)
      (canonicalScope interface offset assumptions)
      canonicalEnv canonicalAgrees canonicalRows with
    ⟨afterCanonical, canonicalOps, canonicalEnd, _, _⟩
  have canonicalPrefix : afterCanonical.operations =
      [canonicalOp interface offset] := by
    simpa [empty, canonicalOp] using canonicalOps
  have sourceAssumptionsCanonical :
      Assumptions interface offset afterCanonical.current := assumptions
  have canonicalSpec := prefixCanonicalSpec interface offset env afterCanonical
    sourceAssumptionsCanonical (by simp [canonicalPrefix])
  have lowAssumptions := Candidate16Five.canonicalWindowAssumptions
    (canonicalOffset offset) (decoderOffset offset lowPart) lowPart
    afterCanonical.current (decoderWordEnd offset lowPart) canonicalSpec
  rcases Candidate16Five.complete (decoderInterface offset lowPart)
      afterCanonical.current (decoderOffset offset lowPart) lowAssumptions with
    ⟨lowEnv, lowAgrees, lowRows⟩
  have lowStart : offset + localLength afterCanonical.operations =
      decoderOffset offset lowPart := by
    rw [canonicalEnd, canonicalChildLength]
    simp [canonicalOffset, decoderOffset, lowPart,
      CanonicalU64.auxiliaryCount]
  rcases Sequence.appendBuiltAt afterCanonical lowName
      (decoderCircuit offset lowPart) (decoderOffset offset lowPart)
      lowStart (decoderScope offset lowPart) lowEnv lowAgrees lowRows with
    ⟨afterLow, lowOps, lowEnd, _, _⟩
  have lowPrefix : afterLow.operations =
      [canonicalOp interface offset, lowOp offset] := by
    rw [lowOps, canonicalPrefix]
    rfl
  have sourceAssumptionsLow : Assumptions interface offset afterLow.current :=
    assumptions
  have canonicalSpecLow := prefixCanonicalSpec interface offset env afterLow
    sourceAssumptionsLow (by simp [lowPrefix])
  have highAssumptions := Candidate16Five.canonicalWindowAssumptions
    (canonicalOffset offset) (decoderOffset offset highPart) highPart
    afterLow.current (decoderWordEnd offset highPart) canonicalSpecLow
  rcases Candidate16Five.complete (decoderInterface offset highPart)
      afterLow.current (decoderOffset offset highPart) highAssumptions with
    ⟨highEnv, highAgrees, highRows⟩
  have highStart : offset + localLength afterLow.operations =
      decoderOffset offset highPart := by
    rw [lowEnd, decoderChildLength]
    simp [decoderOffset, lowPart, highPart,
      CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount]
  rcases Sequence.appendBuiltAt afterLow highName
      (decoderCircuit offset highPart) (decoderOffset offset highPart)
      highStart (decoderScope offset highPart) highEnv highAgrees highRows with
    ⟨completed, highOps, _, _, _⟩
  have completedOps : completed.operations = opsAt interface offset := by
    rw [highOps, lowPrefix]
    rfl
  refine ⟨completed.current, ?_, ?_⟩
  · rw [localLength_eq]
    simpa [completedOps] using completed.agrees
  · simpa [completedOps] using completed.rows

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  complete interface env offset assumptions

/-- The candidate expression is exactly the selected 16-bit source window. -/
theorem candidateValue_eq (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) (part : Fin 2) :
    ((candidate offset part).eval env).val =
      ((interface.source offset).eval env).val /
        2 ^ (16 * part.val) % 2 ^ Candidate16Five.candidateBitCount := by
  have bridge := Candidate16Five.canonicalWindowAssumptions
    (canonicalOffset offset) (decoderOffset offset part) part env
    (decoderWordEnd offset part) specification.canonical
  have candidateEq := bridge.2.2.1
  have window := CanonicalU64.windowValue_eq
    (canonicalInterface interface offset) env (canonicalOffset offset)
    (16 * part.val) Candidate16Five.candidateBitCount
    specification.canonical (by
      have partBound := part.isLt
      simp only [Candidate16Five.candidateBitCount, CanonicalU64.bitCount]
      omega)
  calc
    ((candidate offset part).eval env).val =
        Candidate16Five.candidateValue (decoderInterface offset part) env
          (decoderOffset offset part) := candidateEq
    _ = CanonicalU64.weightedValue env (canonicalOffset offset)
          (16 * part.val) Candidate16Five.candidateBitCount := by
        rfl
    _ = ((interface.source offset).eval env).val /
          2 ^ (16 * part.val) % 2 ^ Candidate16Five.candidateBitCount := window

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

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane
