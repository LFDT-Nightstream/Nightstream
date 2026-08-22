import NightstreamFPrime.Gadgets.Multilinear.PointEquality
import NightstreamFPrime.Gadgets.Polynomial.Horner

/-!
Obligation: Multiply one canonical multilinear point-equality value by one
constant-first extension-field Horner evaluation.

Inputs:
- two symbolic points of one fixed dimension;
- one extension-field Horner point and coefficient list;
- declared point-equality, weighted-sum, and product wires.

Outputs:
- the declared product.

Constraint groups:
- C1: one opaque `PointEquality` child;
- C2: one opaque `Horner` child;
- C3: two extension-component product equations.

This is the sole repeated implementation used by PiCCS `Eval_K` and
`Eval_A`. It owns no protocol coordinate order or gamma shift.
-/

namespace NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

structure Interface (variableCount : Nat) where
  left : Nat → Fin variableCount → KExpr
  right : Nat → Fin variableCount → KExpr
  hornerPoint : Nat → KExpr
  coefficients : Nat → List KExpr
  pointEquality : Nat → KExpr
  weightedSum : Nat → KExpr
  expected : Nat → KExpr

def pointInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) :
    PointEquality.Interface variableCount where
  left := fun _ => interface.left parentOffset
  right := fun _ => interface.right parentOffset
  expected := fun _ => interface.pointEquality parentOffset

def hornerInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) :
    Horner.Interface where
  point := fun _ => interface.hornerPoint parentOffset
  coefficients := fun _ => interface.coefficients parentOffset
  expected := fun _ => interface.weightedSum parentOffset

def pointCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) : FormalCircuit :=
  PointEquality.circuit (pointInterfaceAt interface parentOffset)

def pointLength {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  localLength (Circuit.ops (pointCircuitAt interface offset).main offset)

def hornerOffset {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  offset + pointLength interface offset

def hornerCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) : FormalCircuit :=
  Horner.circuit (hornerInterfaceAt interface parentOffset)

def hornerLength {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  localLength (Circuit.ops (hornerCircuitAt interface offset).main
    (hornerOffset interface offset))

def finalOffset {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  hornerOffset interface offset + hornerLength interface offset

def pointName : String := "multilinear.point_weighted_horner.point_equality"
def hornerName : String := "multilinear.point_weighted_horner.horner"

def pointOp {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Op :=
  .subcircuit ((pointCircuitAt interface offset).asSubcircuit pointName offset)

def hornerOp {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Op :=
  .subcircuit ((hornerCircuitAt interface offset).asSubcircuit hornerName
    (hornerOffset interface offset))

def productExpr {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : KExpr :=
  KExpr.mul (interface.pointEquality offset) (interface.weightedSum offset)

def productAssertions {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) : List Expr :=
  KExpr.equalities (interface.expected offset) (productExpr interface offset)

def opsAt {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : List Op :=
  [pointOp interface offset, hornerOp interface offset] ++
    (productAssertions interface offset).map Op.assertZero

def main {variableCount : Nat} (interface : Interface variableCount) :
    Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

structure Assumptions {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : Prop where
  point : PointEquality.Assumptions (pointInterfaceAt interface offset) offset env
  hornerExternal : Horner.Assumptions (hornerInterfaceAt interface offset)
    offset env
  expectedBelow : (interface.expected offset).VarsBelow offset

structure SpecHolds {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) (env : Env) : Prop where
  point : PointEquality.SpecHolds (pointInterfaceAt interface offset) offset env
  horner : Horner.SpecHolds (hornerInterfaceAt interface offset) offset env
  product : (interface.expected offset).eval env =
    (productExpr interface offset).eval env

/-- Reindexing an interface preserves the complete semantic contract when
all seven externally owned fields are equal at the two offsets. -/
theorem specHolds_at_iff_of_fields_eq {variableCount : Nat}
    (interface : Interface variableCount) (leftOffset rightOffset : Nat)
    (env : Env)
    (leftEq : interface.left leftOffset = interface.left rightOffset)
    (rightEq : interface.right leftOffset = interface.right rightOffset)
    (hornerPointEq : interface.hornerPoint leftOffset =
      interface.hornerPoint rightOffset)
    (coefficientsEq : interface.coefficients leftOffset =
      interface.coefficients rightOffset)
    (pointEqualityEq : interface.pointEquality leftOffset =
      interface.pointEquality rightOffset)
    (weightedSumEq : interface.weightedSum leftOffset =
      interface.weightedSum rightOffset)
    (expectedEq : interface.expected leftOffset =
      interface.expected rightOffset) :
    SpecHolds interface leftOffset env ↔
      SpecHolds interface rightOffset env := by
  constructor
  · rintro ⟨point, horner, product⟩
    refine ⟨?_, ?_, ?_⟩
    · simpa [PointEquality.SpecHolds, pointInterfaceAt, leftEq, rightEq,
        pointEqualityEq] using point
    · simpa [Horner.SpecHolds, hornerInterfaceAt, hornerPointEq,
        coefficientsEq, weightedSumEq] using horner
    · simpa [productExpr, expectedEq, pointEqualityEq, weightedSumEq]
        using product
  · rintro ⟨point, horner, product⟩
    refine ⟨?_, ?_, ?_⟩
    · simpa [PointEquality.SpecHolds, pointInterfaceAt, leftEq, rightEq,
        pointEqualityEq] using point
    · simpa [Horner.SpecHolds, hornerInterfaceAt, hornerPointEq,
        coefficientsEq, weightedSumEq] using horner
    · simpa [productExpr, expectedEq, pointEqualityEq, weightedSumEq]
        using product

theorem pointLength_eq_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    pointLength interface offset = 4 * variableCount - 2 := by
  unfold pointLength pointCircuitAt
  exact PointEquality.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset positive

theorem pointProgramLength_eq_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    (PointEquality.program (pointInterfaceAt interface offset) offset
      ).recipes.length = 4 * variableCount - 2 :=
  PointEquality.program_recipes_length_of_positive
    (pointInterfaceAt interface offset) offset positive

theorem hornerLength_eq {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    hornerLength interface offset =
      2 * ((interface.coefficients offset).length - 1) := by
  unfold hornerLength hornerCircuitAt
  rw [Horner.localLength_eq]
  rfl

theorem hornerAssumptionsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (external : Horner.Assumptions (hornerInterfaceAt interface offset)
      offset env) :
    Horner.Assumptions (hornerInterfaceAt interface offset)
      (hornerOffset interface offset) env := by
  rcases external with ⟨pointBelow, coefficientsBelow, expectedBelow⟩
  have offsetLe : offset ≤ hornerOffset interface offset := by
    unfold hornerOffset
    omega
  exact ⟨
    ((hornerInterfaceAt interface offset).point offset).varsBelow_mono
      pointBelow offsetLe,
    fun coefficient member =>
      coefficient.varsBelow_mono (coefficientsBelow coefficient member) offsetLe,
    ((hornerInterfaceAt interface offset).expected offset).varsBelow_mono
      expectedBelow offsetLe⟩

theorem hornerSpecAt_iff {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) :
    Horner.SpecHolds (hornerInterfaceAt interface offset)
        (hornerOffset interface offset) env ↔
      Horner.SpecHolds (hornerInterfaceAt interface offset) offset env := by
  unfold Horner.SpecHolds hornerInterfaceAt
  rfl

private theorem pointCall_sound {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    PointEquality.SpecHolds (pointInterfaceAt interface offset) offset env := by
  have callHolds := rows (pointOp interface offset) (by simp [opsAt])
  change (pointCircuitAt interface offset).assumptions offset env →
    (pointCircuitAt interface offset).spec offset env at callHolds
  exact callHolds assumptions.point

private theorem hornerCall_sound {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Horner.SpecHolds (hornerInterfaceAt interface offset) offset env := by
  have callHolds := rows (hornerOp interface offset) (by simp [opsAt])
  change (hornerCircuitAt interface offset).assumptions
      (hornerOffset interface offset) env →
    (hornerCircuitAt interface offset).spec
      (hornerOffset interface offset) env at callHolds
  have childSpec := callHolds
    (hornerAssumptionsAt interface offset env assumptions.hornerExternal)
  exact (hornerSpecAt_iff interface offset env).mp childSpec

private theorem productRows_sound {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset)) :
    (interface.expected offset).eval env = (productExpr interface offset).eval env := by
  apply (KExpr.equalities_hold_iff env (interface.expected offset)
    (productExpr interface offset)).mp
  intro expression member
  exact rows (Op.assertZero expression) (by
    simp [opsAt, productAssertions, member])

theorem soundness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  exact ⟨
    pointCall_sound interface offset env rows assumptions,
    hornerCall_sound interface offset env rows assumptions,
    productRows_sound interface offset env rows⟩

/-- The assembled specification is stable when every shared external wire is
unchanged. -/
theorem specHolds_of_agree_below {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have pointSpec := PointEquality.specHolds_of_agree_below
    (pointInterfaceAt interface offset) offset before after assumptions.point
      agrees specification.point
  have hornerSpec := Horner.specHolds_of_agree_below
    (hornerInterfaceAt interface offset) offset before after
      assumptions.hornerExternal agrees specification.horner
  have expectedEq : (interface.expected offset).eval after =
      (interface.expected offset).eval before :=
    (interface.expected offset).eval_eq_of_agree_below offset after before
      assumptions.expectedBelow agrees
  have pointEq : (interface.pointEquality offset).eval after =
      (interface.pointEquality offset).eval before :=
    (interface.pointEquality offset).eval_eq_of_agree_below offset after before
      assumptions.point.2 agrees
  have weightedEq : (interface.weightedSum offset).eval after =
      (interface.weightedSum offset).eval before :=
    (interface.weightedSum offset).eval_eq_of_agree_below offset after before
      assumptions.hornerExternal.2.2 agrees
  refine ⟨pointSpec, hornerSpec, ?_⟩
  unfold productExpr
  rw [expectedEq, KExpr.eval_mul, pointEq, weightedEq]
  exact specification.product

private theorem flatConstraints_opsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset) ++
      flatConstraints (Circuit.ops (hornerCircuitAt interface offset).main
        (hornerOffset interface offset)) ++
      productAssertions interface offset := by
  simp [opsAt, pointOp, hornerOp, FormalCircuit.asSubcircuit,
    flatConstraints, Op.flatConstraints, productAssertions, KExpr.equalities]

private theorem pointOp_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (pointOp interface offset).localLength = pointLength interface offset := by
  unfold pointOp pointLength
  rfl

private theorem hornerOp_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (hornerOp interface offset).localLength = hornerLength interface offset := by
  unfold hornerOp hornerLength
  rfl

theorem opsAt_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    localLength (opsAt interface offset) =
      pointLength interface offset + hornerLength interface offset := by
  rw [show opsAt interface offset =
      [pointOp interface offset, hornerOp interface offset,
        Op.assertZero ((interface.expected offset).c0 -
          (productExpr interface offset).c0),
        Op.assertZero ((interface.expected offset).c1 -
          (productExpr interface offset).c1)] by
    rfl]
  simp only [localLength, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.add_zero]
  rw [pointOp_localLength, hornerOp_localLength]
  simp only [Op.localLength, Nat.add_zero]

private theorem pointRows_preserved {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env middle after : Env) (assumptions : Assumptions interface offset env)
    (positive : 0 < variableCount)
    (pointRows : holdsFlat middle
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (hornerAgrees : AgreesOutside middle after
      (hornerOffset interface offset) (hornerLength interface offset)) :
    holdsFlat after (Circuit.ops (pointCircuitAt interface offset).main offset) := by
  unfold holdsFlat at pointRows ⊢
  apply constraintsHold_of_agree_below middle after _
    (hornerOffset interface offset)
  · have scope := PointEquality.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
    intro expression member
    have below := scope expression member
    have boundEq : offset +
        (PointEquality.program (pointInterfaceAt interface offset) offset
          ).recipes.length = hornerOffset interface offset := by
      rw [pointProgramLength_eq_of_positive interface offset positive]
      unfold hornerOffset
      rw [pointLength_eq_of_positive interface offset positive]
    simpa [boundEq] using below
  · intro index below
    exact hornerAgrees index (Or.inl below)
  · exact pointRows

private theorem productRows_complete {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env after : Env) (assumptions : Assumptions interface offset env)
    (agrees : AgreesOutside env after offset
      (pointLength interface offset + hornerLength interface offset))
    (productSpec : (interface.expected offset).eval env =
      (productExpr interface offset).eval env) :
    ConstraintsHold after (productAssertions interface offset) := by
  have agreesBelow : ∀ index, index < offset → after index = env index :=
    fun index below => agrees index (Or.inl below)
  have expectedEq : (interface.expected offset).eval after =
      (interface.expected offset).eval env :=
    (interface.expected offset).eval_eq_of_agree_below offset after env
      assumptions.expectedBelow agreesBelow
  have equalityEq : (interface.pointEquality offset).eval after =
      (interface.pointEquality offset).eval env :=
    (interface.pointEquality offset).eval_eq_of_agree_below offset after env
      assumptions.point.2 agreesBelow
  have sumEq : (interface.weightedSum offset).eval after =
      (interface.weightedSum offset).eval env :=
    (interface.weightedSum offset).eval_eq_of_agree_below offset after env
      assumptions.hornerExternal.2.2 agreesBelow
  apply (KExpr.equalities_hold_iff after (interface.expected offset)
    (productExpr interface offset)).mpr
  unfold productExpr
  rw [expectedEq, KExpr.eval_mul, equalityEq, sumEq]
  exact productSpec

theorem completeness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat) (positive : 0 < variableCount)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases PointEquality.completeness (pointInterfaceAt interface offset) env
      offset assumptions.point specification.point with
    ⟨middle, pointAgrees, pointRows⟩
  have hornerSpecMiddleAtOffset := Horner.specHolds_of_agree_below
    (hornerInterfaceAt interface offset) offset env middle
    assumptions.hornerExternal
    (fun index below => pointAgrees index (Or.inl below))
    specification.horner
  have hornerSpecMiddle : Horner.SpecHolds
      (hornerInterfaceAt interface offset) (hornerOffset interface offset)
      middle :=
    (hornerSpecAt_iff interface offset middle).mpr
      hornerSpecMiddleAtOffset
  rcases Horner.completeness (hornerInterfaceAt interface offset) middle
      (hornerOffset interface offset)
      (hornerAssumptionsAt interface offset middle assumptions.hornerExternal)
      hornerSpecMiddle with
    ⟨completed, hornerAgrees, hornerRows⟩
  have combinedAgrees : AgreesOutside env completed offset
      (pointLength interface offset + hornerLength interface offset) := by
    simpa [hornerOffset] using pointAgrees.append hornerAgrees
  have pointRowsAfter := pointRows_preserved interface offset env middle
    completed assumptions positive pointRows hornerAgrees
  have productRows := productRows_complete interface offset env completed
    assumptions combinedAgrees specification.product
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset
      (localLength (opsAt interface offset))
    rw [opsAt_localLength]
    exact combinedAgrees
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨(constraintsHold_append completed _ _).mpr
        ⟨pointRowsAfter, hornerRows⟩, productRows⟩

def circuit {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := fun env offset assumptions specification =>
    completeness interface env offset positive assumptions specification

/-- Every flattened row reads only external wires or this assembler's
completed private interval. -/
theorem flatConstraints_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface positive).main offset),
      expression.VarsBelow
        (offset + localLength
          (Circuit.ops (circuit interface positive).main offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
  intro expression member
  rcases List.mem_append.mp member with childrenMember | productMember
  · rcases List.mem_append.mp childrenMember with pointMember | hornerMember
    · have below := PointEquality.flatConstraints_varsBelow
        (pointInterfaceAt interface offset) offset env assumptions.point
          expression pointMember
      apply Expr.VarsBelow.mono expression below
      rw [PointEquality.program_recipes_length_of_positive
        (pointInterfaceAt interface offset) offset positive,
        pointLength_eq_of_positive interface offset positive]
      omega
    · have below := Horner.flatConstraints_varsBelow
        (hornerInterfaceAt interface offset) (hornerOffset interface offset)
          env (hornerAssumptionsAt interface offset env
            assumptions.hornerExternal) expression hornerMember
      apply Expr.VarsBelow.mono expression below
      have lengthEq :
          (Horner.program (hornerInterfaceAt interface offset)
            (hornerOffset interface offset)).recipes.length =
            hornerLength interface offset := by
        unfold Horner.program
        rw [Horner.compile_recipes_length, hornerLength_eq]
        rfl
      rw [lengthEq]
      unfold hornerOffset
      omega
  · exact KExpr.equalities_varsBelow (interface.expected offset)
      (productExpr interface offset)
      (offset + (pointLength interface offset + hornerLength interface offset))
      ((interface.expected offset).varsBelow_mono
        assumptions.expectedBelow (by omega))
      (by
        unfold productExpr
        exact KExpr.mul_varsBelow _ _ _
          ((interface.pointEquality offset).varsBelow_mono
            assumptions.point.2 (by omega))
          ((interface.weightedSum offset).varsBelow_mono
            assumptions.hornerExternal.2.2 (by omega)))
      expression productMember

theorem localLength_eq {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    localLength (Circuit.ops (circuit interface positive).main offset) =
      (4 * variableCount - 2) +
        2 * ((interface.coefficients offset).length - 1) := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength, pointLength_eq_of_positive interface offset positive,
    hornerLength_eq]

theorem operations_length {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    (Circuit.ops (circuit interface positive).main offset).length = 4 := by
  change (opsAt interface offset).length = 4
  simp [opsAt, productAssertions, KExpr.equalities]

theorem flatConstraints_length {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface positive).main offset)).length =
      4 * variableCount +
        (2 * ((interface.coefficients offset).length - 1) + 2) + 2 := by
  change (flatConstraints (opsAt interface offset)).length = _
  have pointFlatLength :
      (flatConstraints (Circuit.ops (pointCircuitAt interface offset).main
        offset)).length = 4 * variableCount := by
    unfold pointCircuitAt
    exact PointEquality.flatConstraints_length_of_positive
      (pointInterfaceAt interface offset) offset positive
  have hornerFlatLength :
      (flatConstraints (Circuit.ops (hornerCircuitAt interface offset).main
        (hornerOffset interface offset))).length =
        2 * ((interface.coefficients offset).length - 1) + 2 := by
    unfold hornerCircuitAt
    rw [Horner.flatConstraints_length]
    rfl
  rw [flatConstraints_opsAt, List.length_append, List.length_append,
    pointFlatLength, hornerFlatLength]
  simp [productAssertions, KExpr.equalities]

/-! ## Child-owned intermediate and output variant -/

namespace Owned

/-!
Obligation: Compute `eq(left,right) * Horner(point, coefficients)` while each
arithmetic child owns its output.

This assembler has no point-equality, Horner-result, or product input wire.
It adds no row at either child boundary. Parents see only this specification,
the exported `output`, and the proved footprint.
-/

structure Interface (variableCount : Nat) where
  left : Nat → Fin variableCount → KExpr
  right : Nat → Fin variableCount → KExpr
  hornerPoint : Nat → KExpr
  coefficients : Nat → List KExpr

def pointInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) :
    PointEquality.Owned.Interface variableCount where
  left := fun _ => interface.left parentOffset
  right := fun _ => interface.right parentOffset

def hornerInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) :
    Horner.Owned.Interface where
  point := fun _ => interface.hornerPoint parentOffset
  coefficients := fun _ => interface.coefficients parentOffset

def pointCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) : FormalCircuit :=
  PointEquality.Owned.circuit (pointInterfaceAt interface parentOffset)

def pointLength {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  localLength (Circuit.ops (pointCircuitAt interface offset).main offset)

def hornerOffset {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  offset + pointLength interface offset

def hornerCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (parentOffset : Nat) : FormalCircuit :=
  Horner.Owned.circuit (hornerInterfaceAt interface parentOffset)

def hornerLength {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  localLength (Circuit.ops (hornerCircuitAt interface offset).main
    (hornerOffset interface offset))

def finalOffset {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Nat :=
  hornerOffset interface offset + hornerLength interface offset

def pointOutput {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : KExpr :=
  PointEquality.Owned.output (pointInterfaceAt interface offset) offset

def weightedSum {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : KExpr :=
  Horner.Owned.output (hornerInterfaceAt interface offset)
    (hornerOffset interface offset)

def output {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : KExpr :=
  KExpr.mul (pointOutput interface offset) (weightedSum interface offset)

def pointName : String :=
  "multilinear.point_weighted_horner.owned.point_equality"

def hornerName : String :=
  "multilinear.point_weighted_horner.owned.horner"

def pointOp {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Op :=
  .subcircuit ((pointCircuitAt interface offset).asSubcircuit pointName offset)

def hornerOp {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : Op :=
  .subcircuit ((hornerCircuitAt interface offset).asSubcircuit hornerName
    (hornerOffset interface offset))

def opsAt {variableCount : Nat} (interface : Interface variableCount)
    (offset : Nat) : List Op :=
  [pointOp interface offset, hornerOp interface offset]

def main {variableCount : Nat} (interface : Interface variableCount) :
    Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

structure Assumptions {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) : Prop where
  point : PointEquality.Owned.Assumptions
    (pointInterfaceAt interface offset) offset env
  hornerExternal : Horner.Owned.Assumptions
    (hornerInterfaceAt interface offset) offset env

structure SpecHolds {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env) : Prop where
  point : PointEquality.Owned.SpecHolds
    (pointInterfaceAt interface offset) offset env
  horner : Horner.Owned.SpecHolds (hornerInterfaceAt interface offset)
    (hornerOffset interface offset) env

theorem pointLength_eq_of_positive {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    pointLength interface offset = 4 * variableCount - 2 := by
  unfold pointLength pointCircuitAt
  exact PointEquality.Owned.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset positive

theorem hornerLength_eq {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    hornerLength interface offset =
      2 * ((interface.coefficients offset).length - 1) := by
  unfold hornerLength hornerCircuitAt
  rw [Horner.Owned.localLength_eq]
  rfl

theorem hornerAssumptionsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) (env : Env)
    (external : Horner.Owned.Assumptions
      (hornerInterfaceAt interface offset) offset env) :
    Horner.Owned.Assumptions (hornerInterfaceAt interface offset)
      (hornerOffset interface offset) env := by
  have offsetLe : offset ≤ hornerOffset interface offset := by
    unfold hornerOffset
    omega
  exact ⟨
    ((hornerInterfaceAt interface offset).point offset).varsBelow_mono
      external.1 offsetLe,
    fun coefficient member => coefficient.varsBelow_mono
      (external.2 coefficient member) offsetLe⟩

private theorem pointCall_sound {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env := by
  have callHolds := rows (pointOp interface offset) (by simp [opsAt])
  change (pointCircuitAt interface offset).assumptions offset env →
    (pointCircuitAt interface offset).spec offset env at callHolds
  exact callHolds assumptions.point

private theorem hornerCall_sound {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Horner.Owned.SpecHolds (hornerInterfaceAt interface offset)
      (hornerOffset interface offset) env := by
  have callHolds := rows (hornerOp interface offset) (by simp [opsAt])
  change (hornerCircuitAt interface offset).assumptions
      (hornerOffset interface offset) env →
    (hornerCircuitAt interface offset).spec
      (hornerOffset interface offset) env at callHolds
  exact callHolds
    (hornerAssumptionsAt interface offset env assumptions.hornerExternal)

theorem soundness {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  exact ⟨pointCall_sound interface offset env rows assumptions,
    hornerCall_sound interface offset env rows assumptions⟩

private theorem flatConstraints_opsAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset) ++
      flatConstraints (Circuit.ops (hornerCircuitAt interface offset).main
        (hornerOffset interface offset)) := by
  simp [opsAt, pointOp, hornerOp, FormalCircuit.asSubcircuit,
    flatConstraints, Op.flatConstraints]

private theorem pointOp_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (pointOp interface offset).localLength = pointLength interface offset := by
  rfl

private theorem hornerOp_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    (hornerOp interface offset).localLength = hornerLength interface offset := by
  rfl

theorem opsAt_localLength {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :
    localLength (opsAt interface offset) =
      pointLength interface offset + hornerLength interface offset := by
  simp [opsAt, localLength, pointOp_localLength, hornerOp_localLength]

private theorem pointRows_preserved {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (env middle after : Env) (assumptions : Assumptions interface offset env)
    (positive : 0 < variableCount)
    (pointRows : holdsFlat middle
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (hornerAgrees : AgreesOutside middle after
      (hornerOffset interface offset) (hornerLength interface offset)) :
    holdsFlat after
      (Circuit.ops (pointCircuitAt interface offset).main offset) := by
  unfold holdsFlat at pointRows ⊢
  apply constraintsHold_of_agree_below middle after _
    (hornerOffset interface offset)
  · have scope := PointEquality.Owned.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
    intro expression member
    have below := scope expression member
    have boundEq : offset +
        (PointEquality.Owned.program
          (pointInterfaceAt interface offset) offset).recipes.length =
        hornerOffset interface offset := by
      unfold hornerOffset
      rw [pointLength_eq_of_positive interface offset positive,
        PointEquality.Owned.program_recipes_length_of_positive
          (pointInterfaceAt interface offset) offset positive]
    simpa [boundEq] using below
  · intro index below
    exact hornerAgrees index (Or.inl below)
  · exact pointRows

/-- Honest child builders compose with no semantic premise. -/
theorem build {variableCount : Nat} (interface : Interface variableCount)
    (env : Env) (offset : Nat) (positive : 0 < variableCount)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases PointEquality.Owned.build (pointInterfaceAt interface offset) env
      offset assumptions.point with
    ⟨middle, pointAgrees, pointRows⟩
  rcases Horner.Owned.build (hornerInterfaceAt interface offset) middle
      (hornerOffset interface offset)
      (hornerAssumptionsAt interface offset middle
        assumptions.hornerExternal) with
    ⟨completed, hornerAgrees, hornerRows⟩
  have combinedAgrees : AgreesOutside env completed offset
      (pointLength interface offset + hornerLength interface offset) := by
    simpa [hornerOffset] using pointAgrees.append hornerAgrees
  have pointRowsAfter := pointRows_preserved interface offset env middle
    completed assumptions positive pointRows hornerAgrees
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset
      (localLength (opsAt interface offset))
    rw [opsAt_localLength]
    exact combinedAgrees
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨pointRowsAfter, hornerRows⟩

def circuit {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := fun env offset assumptions _specification =>
    build interface env offset positive assumptions

theorem completeness {variableCount : Nat}
    (interface : Interface variableCount) (env : Env) (offset : Nat)
    (positive : 0 < variableCount)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface positive).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit interface positive).main offset) :=
  build interface env offset positive assumptions

theorem flatConstraints_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface positive).main offset),
      expression.VarsBelow
        (offset + localLength
          (Circuit.ops (circuit interface positive).main offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
  intro expression member
  rcases List.mem_append.mp member with pointMember | hornerMember
  · have below := PointEquality.Owned.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
        expression pointMember
    apply Expr.VarsBelow.mono expression below
    rw [PointEquality.Owned.program_recipes_length_of_positive
      (pointInterfaceAt interface offset) offset positive,
      pointLength_eq_of_positive interface offset positive]
    omega
  · have below := Horner.Owned.flatConstraints_varsBelow
      (hornerInterfaceAt interface offset) (hornerOffset interface offset)
        env (hornerAssumptionsAt interface offset env
          assumptions.hornerExternal) expression hornerMember
    apply Expr.VarsBelow.mono expression below
    change hornerOffset interface offset + hornerLength interface offset ≤
      offset + (pointLength interface offset + hornerLength interface offset)
    unfold hornerOffset
    omega

theorem localLength_eq {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    localLength (Circuit.ops (circuit interface positive).main offset) =
      (4 * variableCount - 2) +
        2 * ((interface.coefficients offset).length - 1) := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength, pointLength_eq_of_positive interface offset positive,
    hornerLength_eq]

/-- The owned product result lies inside the assembler's declared symbolic
interval. -/
theorem output_varsBelow {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    (output interface offset).VarsBelow
      (offset + localLength
        (Circuit.ops (circuit interface positive).main offset)) := by
  unfold output pointOutput weightedSum
  apply KExpr.mul_varsBelow
  · apply KExpr.varsBelow_mono _
      (PointEquality.Owned.output_varsBelow
        (pointInterfaceAt interface offset) offset env assumptions.point)
    rw [localLength_eq interface positive,
      PointEquality.Owned.localLength_eq_of_positive _ _ positive]
    omega
  · apply KExpr.varsBelow_mono _
      (Horner.Owned.output_varsBelow
        (hornerInterfaceAt interface offset) (hornerOffset interface offset)
          env (hornerAssumptionsAt interface offset env
            assumptions.hornerExternal))
    rw [localLength_eq interface positive]
    unfold hornerOffset
    rw [pointLength_eq_of_positive interface offset positive,
      Horner.Owned.localLength_eq]
    simp only [hornerInterfaceAt]
    omega

theorem operations_length {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    (Circuit.ops (circuit interface positive).main offset).length = 2 := by
  rfl

theorem flatConstraints_length {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit interface positive).main offset)).length =
      (4 * variableCount - 2) +
        2 * ((interface.coefficients offset).length - 1) := by
  change (flatConstraints (opsAt interface offset)).length = _
  rw [flatConstraints_opsAt, List.length_append]
  have pointLengthEq := PointEquality.Owned.flatConstraints_length_of_positive
    (pointInterfaceAt interface offset) offset positive
  have hornerLengthEq := Horner.Owned.flatConstraints_length
    (hornerInterfaceAt interface offset) (hornerOffset interface offset)
  unfold pointCircuitAt hornerCircuitAt
  rw [pointLengthEq, hornerLengthEq]
  rfl

end Owned

end NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner
