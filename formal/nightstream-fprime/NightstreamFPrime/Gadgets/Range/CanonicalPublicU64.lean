import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Range.CanonicalU64

/-!
Owns the binding between one canonical Goldilocks word and 64 caller-owned
little-endian public bits.

The canonical-u64 child owns all decomposition witnesses and range rows. This
wrapper adds only equality rows from those child bits to the caller bits.
-/

namespace NightstreamFPrime.Gadgets.Range.CanonicalPublicU64

open NightstreamFPrime.Circuit

def bitCount : Nat := CanonicalU64.bitCount
def privateCount : Nat := CanonicalU64.auxiliaryCount
def rowCount : Nat := CanonicalU64.exactRowCount + bitCount

structure Interface where
  source : Nat → Expr
  bit : Nat → Nat → Expr

def childInterface (interface : Interface) (parentOffset : Nat) :
    CanonicalU64.Interface where
  source := fun _ => interface.source parentOffset

def childCircuit (interface : Interface) (parentOffset : Nat) : FormalCircuit :=
  CanonicalU64.circuit (childInterface interface parentOffset)

def childName : String := "range.canonical_public_u64.word"

def childOp (interface : Interface) (offset : Nat) : Op :=
  Sequence.childOp childName (childCircuit interface offset) offset

def bindingConstraint (interface : Interface) (offset index : Nat) : Expr :=
  interface.bit offset index - CanonicalU64.bitExpr offset index

def bindingOps (interface : Interface) (offset : Nat) : List Op :=
  (List.range bitCount).map fun index =>
    .assertZero (bindingConstraint interface offset index)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  childOp interface offset :: bindingOps interface offset

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + privateCount, opsAt interface offset)

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (interface.source offset).VarsBelow offset ∧
    ∀ index, index < bitCount →
      (interface.bit offset index).VarsBelow offset

/-- Each public cell is the canonical little-endian bit of `source`. -/
def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ index, index < bitCount →
    ((interface.bit offset index).eval env).val =
      ((interface.source offset).eval env).val / 2 ^ index % 2

private theorem childLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (childCircuit interface offset).main offset) =
      privateCount := by
  change localLength
      (CanonicalU64.operations (childInterface interface offset) offset) = _
  exact CanonicalU64.localLength_eq _ _

@[simp] private theorem childOp_localLength
    (interface : Interface) (offset : Nat) :
    (childOp interface offset).localLength = privateCount := by
  rw [childOp, Sequence.childOp_localLength]
  exact childLength interface offset

private theorem bindingOps_localLength (interface : Interface) (offset : Nat) :
    localLength (bindingOps interface offset) = 0 := by
  unfold bindingOps localLength
  change (List.map (fun _ => 0) (List.range bitCount)).sum = 0
  simp

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = privateCount := by
  change localLength (opsAt interface offset) = privateCount
  change (childOp interface offset).localLength +
      localLength (bindingOps interface offset) = privateCount
  rw [childOp_localLength, bindingOps_localLength, Nat.add_zero]

private theorem flatConstraints_bindingOps
    (interface : Interface) (offset : Nat) :
    flatConstraints (bindingOps interface offset) =
      (List.range bitCount).map (bindingConstraint interface offset) := by
  unfold flatConstraints bindingOps
  generalize List.range bitCount = indices
  induction indices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp [Op.flatConstraints, inductionHypothesis]

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (childCircuit interface offset).main offset) ++
        (List.range bitCount).map (bindingConstraint interface offset) := by
  change (childOp interface offset).flatConstraints ++
      flatConstraints (bindingOps interface offset) = _
  rw [flatConstraints_bindingOps]
  rfl

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length = rowCount := by
  change (flatConstraints (opsAt interface offset)).length = rowCount
  rw [flatConstraints_opsAt, List.length_append]
  change CanonicalU64.exactRowCount + bitCount = rowCount
  rfl

private theorem childAssumptions (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    CanonicalU64.Assumptions (childInterface interface offset) offset env := by
  exact assumptions.1

private theorem childScope (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (childCircuit interface offset).main offset),
      expression.VarsBelow (offset + privateCount) := by
  change ∀ expression ∈ flatConstraints
      (CanonicalU64.operations (childInterface interface offset) offset),
    expression.VarsBelow (offset + privateCount)
  exact CanonicalU64.flatConstraints_varsBelow
    (childInterface interface offset) offset assumptions.1

theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  rw [localLength_eq]
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + privateCount)
  rw [flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with childMember | bindingMember
  · exact childScope interface offset assumptions expression childMember
  · rcases List.mem_map.mp bindingMember with ⟨index, indexMember, rfl⟩
    apply Expr.VarsBelow.sub
    · exact Expr.VarsBelow.mono _
        (assumptions.2 index (List.mem_range.mp indexMember)) (by omega)
    · apply CanonicalU64.bitExpr_varsBelow
      have bounded := List.mem_range.mp indexMember
      simp only [bitCount, privateCount, CanonicalU64.bitCount,
        CanonicalU64.auxiliaryCount] at bounded ⊢
      omega

private theorem childCall_sound
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) :
    CanonicalU64.SpecHolds (childInterface interface offset) offset env := by
  have callHolds := rows (childOp interface offset) (by simp [opsAt])
  change CanonicalU64.Assumptions (childInterface interface offset) offset env →
    CanonicalU64.SpecHolds (childInterface interface offset) offset env at callHolds
  exact callHolds (childAssumptions interface offset assumptions)

private theorem canonicalBitValue
    (interface : Interface) (env : Env) (offset index : Nat)
    (specification : CanonicalU64.SpecHolds
      (childInterface interface offset) offset env)
    (bounded : index < bitCount) :
    ((CanonicalU64.bitExpr offset index).eval env).val =
      ((interface.source offset).eval env).val / 2 ^ index % 2 := by
  have window := CanonicalU64.windowValue_eq
    (childInterface interface offset) env offset index 1 specification (by
      simpa [bitCount, CanonicalU64.bitCount] using Nat.succ_le_of_lt bounded)
  simpa [CanonicalU64.weightedValue, CanonicalU64.bitValue,
    childInterface] using window

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  change holds env (opsAt interface offset) at rows
  have canonical := childCall_sound interface env offset assumptions rows
  intro index bounded
  have operationMember :
      .assertZero (bindingConstraint interface offset index) ∈
        opsAt interface offset := by
    apply List.mem_cons_of_mem
    exact List.mem_map.mpr
      ⟨index, List.mem_range.mpr bounded, rfl⟩
  have binding := rows
    (.assertZero (bindingConstraint interface offset index)) operationMember
  have boundEqual :
      (interface.bit offset index).eval env =
        (CanonicalU64.bitExpr offset index).eval env := by
    exact sub_eq_zero.mp (by simpa [bindingConstraint] using binding)
  rw [boundEqual]
  exact canonicalBitValue interface env offset index canonical bounded

private theorem completedAgreesBelow
    {before after : Env} {offset count : Nat}
    (agrees : AgreesOutside before after offset count) :
    ∀ index, index < offset → after index = before index := by
  intro index below
  exact agrees index (Or.inl below)

theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases CanonicalU64.complete (childInterface interface offset) env offset
      (childAssumptions interface offset assumptions) with
    ⟨completed, agrees, childRows⟩
  have agreesPrivate :
      AgreesOutside env completed offset privateCount := by
    rw [← childLength interface offset]
    exact agrees
  have agreesBelow := completedAgreesBelow agrees
  have sourcePreserved :
      (interface.source offset).eval completed =
        (interface.source offset).eval env :=
    (interface.source offset).eval_eq_of_agree_below offset completed env
      assumptions.1 agreesBelow
  have childSpecification := CanonicalU64.soundness
    (childInterface interface offset) completed offset
    (childAssumptions interface offset assumptions)
    (holdsFlat_implies_holds completed _ childRows)
  refine ⟨completed, ?_, ?_⟩
  · rw [localLength_eq]
    exact agreesPrivate
  · change holdsFlat completed (opsAt interface offset)
    unfold holdsFlat
    rw [flatConstraints_opsAt, constraintsHold_append]
    refine ⟨childRows, ?_⟩
    intro expression member
    rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    have bounded := List.mem_range.mp indexMember
    have publicPreserved :
        (interface.bit offset index).eval completed =
          (interface.bit offset index).eval env :=
      (interface.bit offset index).eval_eq_of_agree_below offset completed env
        (assumptions.2 index bounded) agreesBelow
    have canonical := canonicalBitValue interface completed offset index
      childSpecification bounded
    rw [bindingConstraint, Expr.eval_sub]
    apply sub_eq_zero.mpr
    apply Fin.eq_of_val_eq
    calc
      ((interface.bit offset index).eval completed).val =
          ((interface.bit offset index).eval env).val :=
        congrArg Fin.val publicPreserved
      _ = ((interface.source offset).eval env).val / 2 ^ index % 2 :=
        specification index bounded
      _ = ((interface.source offset).eval completed).val / 2 ^ index % 2 := by
        rw [sourcePreserved]
      _ = ((CanonicalU64.bitExpr offset index).eval completed).val :=
        canonical.symm

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  complete interface env offset assumptions specification

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun _ => privateCount
  rowCount := fun _ => rowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Gadgets.Range.CanonicalPublicU64
