import SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.BooleanPairRows

/-!
Owns: specialization of the nonresidue pair/odd-tail schedule to arbitrary
field residuals, one-product R1CS residuals `A * B - C`, and centered-unit
residuals `d^3 - d`.

Does not own: protocol-family selection, Rust row emission, generated matrix
geometry, selector allocation, stage boundaries, global gate minimality, or
authorization to remove production constraints.

Emits constraints: no. This file proves model-level residual equations and
reuses the deterministic structural schedule from `BooleanPairRows`.

Authority boundary: the selector equivalences assume a verifier-fixed value
one. A production consumer must separately prove exact residual expressions,
column ownership, stage/family-local pairing, odd-tail placement, and matrix
conformance before deleting any row.

Degree boundary: degree statements count the declared residual degree,
squaring by the pair equation, and one linear selector factor. They do not
formalize a production polynomial AST or prove a global algebraic lower bound.

Assurance tier: model-level.

| Family | Leaf obligation | Pair equation | Odd tail | Selector-gated degree | Production row removal? |
|---|---|---|---|---:|---|
| arbitrary residual | `r = 0` | `r1^2 - 7*r2^2 = 0` | `r = 0` | `2*deg(r)+1` / `deg(r)+1` | no |
| one-product R1CS | `A*B-C = 0` | packed adjacent equations | ordinary product residual | at most 5 | no |
| centered unit | `d^3-d = 0`, exactly `{-1,0,1}` | packed adjacent digits | ordinary centered residual | at most 7 | no |

| Theorem family | Mathematical guarantee | Concrete assumptions |
|---|---|---|
| `residualPairHolds_iff` | an arbitrary packed pair vanishes iff both residuals vanish | `KExt.w_not_square` through `BooleanPairRows.quadraticZeroPair_iff` |
| `oneProductPairHolds_iff` | one packed row is exactly two `A*B=C` equations | Goldilocks field |
| `centeredUnitPairHolds_iff` | one packed row is exactly two centered-unit memberships | Goldilocks field |
| `familySchedule_order_exact` / `familySchedule_shape_counts` | deterministic adjacent pairs, exact order, one odd tail | reused structural schedule |
| `*_pairRow_is_necessary` / `*_oddTailRow_is_necessary` | deleting either row family admits an invalid equation | explicit family witness |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies

/-! A local structural view of the existing schedule keeps this module's
formulas readable without introducing a second row datatype. -/
namespace BooleanRows

abbrev Row (Coordinate : Type) := BooleanPairRows.Row Coordinate

namespace Row

abbrev coordinates {Coordinate : Type} (row : BooleanRows.Row Coordinate) :
    List Coordinate :=
  BooleanPairRows.Row.coordinates row

@[simp] theorem coordinates_pair {Coordinate : Type}
    (left right : Coordinate) :
    coordinates (BooleanPairRows.Row.pair left right) = [left, right] := by
  rfl

@[simp] theorem coordinates_tail {Coordinate : Type}
    (coordinate : Coordinate) :
    coordinates (BooleanPairRows.Row.tail coordinate) = [coordinate] := by
  rfl

end Row

abbrev schedule {Coordinate : Type} (coordinates : List Coordinate) :
    List (Row Coordinate) :=
  BooleanPairRows.schedule coordinates

abbrev scheduledCoordinates {Coordinate : Type}
    (rows : List (Row Coordinate)) : List Coordinate :=
  BooleanPairRows.scheduledCoordinates rows

abbrev pairRowCount {Coordinate : Type}
    (rows : List (Row Coordinate)) : Nat :=
  BooleanPairRows.pairRowCount rows

abbrev tailRowCount {Coordinate : Type}
    (rows : List (Row Coordinate)) : Nat :=
  BooleanPairRows.tailRowCount rows

abbrev ceilHalf (count : Nat) : Nat :=
  BooleanPairRows.ceilHalf count

theorem quadraticZeroPair_iff {left right : F} :
    BooleanPairRows.QuadraticZeroPair left right ↔
      left = 0 ∧ right = 0 :=
  BooleanPairRows.quadraticZeroPair_iff

theorem scheduledCoordinates_exact {Coordinate : Type}
    (coordinates : List Coordinate) :
    scheduledCoordinates (schedule coordinates) = coordinates :=
  BooleanPairRows.scheduledCoordinates_exact coordinates

theorem schedule_shape_counts {Coordinate : Type}
    (coordinates : List Coordinate) :
    pairRowCount (schedule coordinates) = coordinates.length / 2 ∧
      tailRowCount (schedule coordinates) = coordinates.length % 2 ∧
      (schedule coordinates).length = ceilHalf coordinates.length :=
  BooleanPairRows.schedule_shape_counts coordinates

end BooleanRows

/-! ## Arbitrary residual kernel -/

/-- One nonresidue equation over two arbitrary Goldilocks residuals. -/
def ResidualPairHolds (leftResidual rightResidual : F) : Prop :=
  leftResidual * leftResidual -
    KExt.w * (rightResidual * rightResidual) = 0

/-- The arbitrary residual equation is exact because seven is a quadratic
nonresidue in Goldilocks. -/
theorem residualPairHolds_iff {leftResidual rightResidual : F} :
    ResidualPairHolds leftResidual rightResidual ↔
      leftResidual = 0 ∧ rightResidual = 0 := by
  exact BooleanRows.quadraticZeroPair_iff

/-- One linear selector multiplying the complete arbitrary residual pair. -/
def SelectorGatedResidualPairHolds
    (selector leftResidual rightResidual : F) : Prop :=
  selector *
    (leftResidual * leftResidual -
      KExt.w * (rightResidual * rightResidual)) = 0

theorem selectorGatedResidualPairHolds_iff
    {selector leftResidual rightResidual : F}
    (fixed : selector = 1) :
    SelectorGatedResidualPairHolds selector leftResidual rightResidual ↔
      ResidualPairHolds leftResidual rightResidual := by
  subst selector
  simp [SelectorGatedResidualPairHolds, ResidualPairHolds]

/-! ## Residual-parametric schedule -/

/-- Polynomial residual represented by one structural pair/odd-tail row. -/
def rowResidual {Coordinate : Type}
    (residual : Coordinate → F) : BooleanRows.Row Coordinate → F
  | .pair left right =>
      let leftResidual := residual left
      let rightResidual := residual right
      leftResidual * leftResidual -
        KExt.w * (rightResidual * rightResidual)
  | .tail coordinate => residual coordinate

def RowHolds {Coordinate : Type}
    (residual : Coordinate → F) (row : BooleanRows.Row Coordinate) : Prop :=
  rowResidual residual row = 0

def SelectorGatedRowHolds {Coordinate : Type}
    (selector : F) (residual : Coordinate → F)
    (row : BooleanRows.Row Coordinate) : Prop :=
  selector * rowResidual residual row = 0

/-- Symbolic degree of a structural row after one linear selector factor. -/
def selectorGatedDegree {Coordinate : Type}
    (residualDegree : Nat) : BooleanRows.Row Coordinate → Nat
  | .pair _ _ => 2 * residualDegree + 1
  | .tail _ => residualDegree + 1

theorem rowHolds_iff_residualsZero
    {Coordinate : Type} (residual : Coordinate → F)
    (row : BooleanRows.Row Coordinate) :
    RowHolds residual row ↔
      ∀ coordinate ∈ row.coordinates, residual coordinate = 0 := by
  cases row with
  | pair left right =>
      change ResidualPairHolds (residual left) (residual right) ↔
        ∀ coordinate ∈ [left, right], residual coordinate = 0
      rw [residualPairHolds_iff]
      simp
  | tail coordinate =>
      simp [RowHolds, rowResidual]

theorem selectorGatedRowHolds_iff
    {Coordinate : Type} {selector : F}
    (fixed : selector = 1) (residual : Coordinate → F)
    (row : BooleanRows.Row Coordinate) :
    SelectorGatedRowHolds selector residual row ↔
      RowHolds residual row := by
  subst selector
  simp [SelectorGatedRowHolds, RowHolds]

def RowsHold {Coordinate : Type}
    (residual : Coordinate → F)
    (rows : List (BooleanRows.Row Coordinate)) : Prop :=
  ∀ row ∈ rows, RowHolds residual row

def CoordinatesResidualZero {Coordinate : Type}
    (residual : Coordinate → F) (coordinates : List Coordinate) : Prop :=
  ∀ coordinate ∈ coordinates, residual coordinate = 0

def ScheduleHolds {Coordinate : Type}
    (residual : Coordinate → F) (coordinates : List Coordinate) : Prop :=
  RowsHold residual (BooleanRows.schedule coordinates)

def SelectorGatedScheduleHolds {Coordinate : Type}
    (selector : F) (residual : Coordinate → F)
    (coordinates : List Coordinate) : Prop :=
  ∀ row ∈ BooleanRows.schedule coordinates,
    SelectorGatedRowHolds selector residual row

theorem rowsHold_iff_scheduledResidualsZero
    {Coordinate : Type} (residual : Coordinate → F)
    (rows : List (BooleanRows.Row Coordinate)) :
    RowsHold residual rows ↔
      CoordinatesResidualZero residual
        (BooleanRows.scheduledCoordinates rows) := by
  constructor
  · intro rowsHold coordinate coordinateMember
    rcases List.mem_flatMap.mp coordinateMember with
      ⟨row, rowMember, memberInRow⟩
    exact (rowHolds_iff_residualsZero residual row).mp
      (rowsHold row rowMember) coordinate memberInRow
  · intro coordinatesZero row rowMember
    apply (rowHolds_iff_residualsZero residual row).mpr
    intro coordinate memberInRow
    apply coordinatesZero coordinate
    exact List.mem_flatMap.mpr ⟨row, rowMember, memberInRow⟩

theorem scheduleHolds_iff_coordinatesResidualZero
    {Coordinate : Type} (residual : Coordinate → F)
    (coordinates : List Coordinate) :
    ScheduleHolds residual coordinates ↔
      CoordinatesResidualZero residual coordinates := by
  rw [ScheduleHolds, rowsHold_iff_scheduledResidualsZero,
    BooleanRows.scheduledCoordinates_exact]

theorem selectorGatedScheduleHolds_iff
    {Coordinate : Type} {selector : F}
    (fixed : selector = 1) (residual : Coordinate → F)
    (coordinates : List Coordinate) :
    SelectorGatedScheduleHolds selector residual coordinates ↔
      CoordinatesResidualZero residual coordinates := by
  rw [← scheduleHolds_iff_coordinatesResidualZero residual coordinates]
  constructor
  · intro gated row rowMember
    exact (selectorGatedRowHolds_iff fixed residual row).mp
      (gated row rowMember)
  · intro ungated row rowMember
    exact (selectorGatedRowHolds_iff fixed residual row).mpr
      (ungated row rowMember)

/-- The reused structural schedule recovers the family input order exactly. -/
theorem familySchedule_order_exact {Coordinate : Type}
    (coordinates : List Coordinate) :
    BooleanRows.scheduledCoordinates (BooleanRows.schedule coordinates) =
      coordinates :=
  BooleanRows.scheduledCoordinates_exact coordinates

/-- Exact pair, odd-tail, and total-row census shared by both residual
families. -/
theorem familySchedule_shape_counts {Coordinate : Type}
    (coordinates : List Coordinate) :
    BooleanRows.pairRowCount (BooleanRows.schedule coordinates) =
        coordinates.length / 2 ∧
      BooleanRows.tailRowCount (BooleanRows.schedule coordinates) =
        coordinates.length % 2 ∧
      (BooleanRows.schedule coordinates).length =
        BooleanRows.ceilHalf coordinates.length :=
  BooleanRows.schedule_shape_counts coordinates

/-! ## One-product R1CS residuals -/

/-- Inputs to one ordinary R1CS product equation `A * B = C`. -/
structure OneProductInput where
  a : F
  b : F
  c : F
deriving DecidableEq, Repr

def oneProductResidual (input : OneProductInput) : F :=
  input.a * input.b - input.c

def OneProductHolds (input : OneProductInput) : Prop :=
  oneProductResidual input = 0

theorem oneProductHolds_iff (input : OneProductInput) :
    OneProductHolds input ↔ input.a * input.b = input.c := by
  simp [OneProductHolds, oneProductResidual, sub_eq_zero]

def OneProductPairHolds (left right : OneProductInput) : Prop :=
  ResidualPairHolds (oneProductResidual left) (oneProductResidual right)

theorem oneProductPairHolds_iff (left right : OneProductInput) :
    OneProductPairHolds left right ↔
      OneProductHolds left ∧ OneProductHolds right := by
  exact residualPairHolds_iff

def SelectorGatedOneProductPairHolds
    (selector : F) (left right : OneProductInput) : Prop :=
  SelectorGatedResidualPairHolds selector
    (oneProductResidual left) (oneProductResidual right)

theorem selectorGatedOneProductPairHolds_iff
    {selector : F} (fixed : selector = 1)
    (left right : OneProductInput) :
    SelectorGatedOneProductPairHolds selector left right ↔
      OneProductHolds left ∧ OneProductHolds right := by
  rw [SelectorGatedOneProductPairHolds,
    selectorGatedResidualPairHolds_iff fixed]
  exact oneProductPairHolds_iff left right

theorem oneProductScheduleHolds_iff
    {Coordinate : Type} (assignment : Coordinate → OneProductInput)
    (coordinates : List Coordinate) :
    ScheduleHolds (fun coordinate =>
        oneProductResidual (assignment coordinate)) coordinates ↔
      ∀ coordinate ∈ coordinates,
        OneProductHolds (assignment coordinate) := by
  rw [scheduleHolds_iff_coordinatesResidualZero]
  rfl

theorem selectorGatedOneProductScheduleHolds_iff
    {Coordinate : Type} {selector : F} (fixed : selector = 1)
    (assignment : Coordinate → OneProductInput)
    (coordinates : List Coordinate) :
    SelectorGatedScheduleHolds selector (fun coordinate =>
        oneProductResidual (assignment coordinate)) coordinates ↔
      ∀ coordinate ∈ coordinates,
        OneProductHolds (assignment coordinate) := by
  rw [selectorGatedScheduleHolds_iff fixed]
  rfl

/-- Maximum total degree of the evaluated `A*B-C` residual. -/
def oneProductResidualDegree : Nat := 2

@[simp] theorem oneProduct_pair_selectorGatedDegree
    {Coordinate : Type} (left right : Coordinate) :
    selectorGatedDegree oneProductResidualDegree (.pair left right) = 5 := by
  rfl

@[simp] theorem oneProduct_tail_selectorGatedDegree
    {Coordinate : Type} (coordinate : Coordinate) :
    selectorGatedDegree oneProductResidualDegree (.tail coordinate) = 3 := by
  rfl

theorem oneProduct_selectorGatedDegree_le_five
    {Coordinate : Type} (row : BooleanRows.Row Coordinate) :
    selectorGatedDegree oneProductResidualDegree row ≤ 5 := by
  cases row <;> simp

/-! ## Centered-unit residuals -/

/-- Cubic residual whose roots are exactly the centered field units. -/
def centeredUnitResidual (digit : F) : F :=
  digit * digit * digit - digit

def IsCenteredUnit (digit : F) : Prop :=
  digit = -1 ∨ digit = 0 ∨ digit = 1

instance (digit : F) : Decidable (IsCenteredUnit digit) := by
  unfold IsCenteredUnit
  infer_instance

theorem centeredUnitResidual_eq_zero_iff {digit : F} :
    centeredUnitResidual digit = 0 ↔ IsCenteredUnit digit := by
  have factorization :
      centeredUnitResidual digit =
        digit * (digit - 1) * (digit + 1) := by
    simp [centeredUnitResidual]
    ring
  rw [factorization]
  constructor
  · intro zero
    rcases mul_eq_zero.mp zero with zeroOrOne | negative
    · rcases mul_eq_zero.mp zeroOrOne with zero | one
      · exact Or.inr (Or.inl zero)
      · exact Or.inr (Or.inr (sub_eq_zero.mp one))
    · exact Or.inl (add_eq_zero_iff_eq_neg.mp negative)
  · intro centered
    rcases centered with negative | zero | one
    · subst digit
      simp
    · subst digit
      simp
    · subst digit
      simp

def CenteredUnitPairHolds (left right : F) : Prop :=
  ResidualPairHolds
    (centeredUnitResidual left) (centeredUnitResidual right)

theorem centeredUnitPairHolds_iff (left right : F) :
    CenteredUnitPairHolds left right ↔
      IsCenteredUnit left ∧ IsCenteredUnit right := by
  rw [CenteredUnitPairHolds, residualPairHolds_iff,
    centeredUnitResidual_eq_zero_iff,
    centeredUnitResidual_eq_zero_iff]

def SelectorGatedCenteredUnitPairHolds
    (selector left right : F) : Prop :=
  SelectorGatedResidualPairHolds selector
    (centeredUnitResidual left) (centeredUnitResidual right)

theorem selectorGatedCenteredUnitPairHolds_iff
    {selector : F} (fixed : selector = 1) (left right : F) :
    SelectorGatedCenteredUnitPairHolds selector left right ↔
      IsCenteredUnit left ∧ IsCenteredUnit right := by
  rw [SelectorGatedCenteredUnitPairHolds,
    selectorGatedResidualPairHolds_iff fixed]
  exact centeredUnitPairHolds_iff left right

theorem centeredUnitScheduleHolds_iff
    {Coordinate : Type} (assignment : Coordinate → F)
    (coordinates : List Coordinate) :
    ScheduleHolds (fun coordinate =>
        centeredUnitResidual (assignment coordinate)) coordinates ↔
      ∀ coordinate ∈ coordinates,
        IsCenteredUnit (assignment coordinate) := by
  rw [scheduleHolds_iff_coordinatesResidualZero]
  constructor <;> intro accepted coordinate coordinateMember
  · exact centeredUnitResidual_eq_zero_iff.mp
      (accepted coordinate coordinateMember)
  · exact centeredUnitResidual_eq_zero_iff.mpr
      (accepted coordinate coordinateMember)

theorem selectorGatedCenteredUnitScheduleHolds_iff
    {Coordinate : Type} {selector : F} (fixed : selector = 1)
    (assignment : Coordinate → F) (coordinates : List Coordinate) :
    SelectorGatedScheduleHolds selector (fun coordinate =>
        centeredUnitResidual (assignment coordinate)) coordinates ↔
      ∀ coordinate ∈ coordinates,
        IsCenteredUnit (assignment coordinate) := by
  rw [selectorGatedScheduleHolds_iff fixed]
  constructor <;> intro accepted coordinate coordinateMember
  · exact centeredUnitResidual_eq_zero_iff.mp
      (accepted coordinate coordinateMember)
  · exact centeredUnitResidual_eq_zero_iff.mpr
      (accepted coordinate coordinateMember)

/-- Total degree of `d^3-d`. -/
def centeredUnitResidualDegree : Nat := 3

@[simp] theorem centeredUnit_pair_selectorGatedDegree
    {Coordinate : Type} (left right : Coordinate) :
    selectorGatedDegree centeredUnitResidualDegree (.pair left right) = 7 := by
  rfl

@[simp] theorem centeredUnit_tail_selectorGatedDegree
    {Coordinate : Type} (coordinate : Coordinate) :
    selectorGatedDegree centeredUnitResidualDegree (.tail coordinate) = 4 := by
  rfl

theorem centeredUnit_selectorGatedDegree_le_seven
    {Coordinate : Type} (row : BooleanRows.Row Coordinate) :
    selectorGatedDegree centeredUnitResidualDegree row ≤ 7 := by
  cases row <;> simp

/-! ## Row-family deletion witnesses -/

def pairDeletionAssignment {Value : Type}
    (invalid valid : Value) : Nat → Value
  | 0 => invalid
  | _ => valid

def oddTailDeletionAssignment {Value : Type}
    (invalid valid : Value) : Nat → Value
  | 2 => invalid
  | _ => valid

private theorem pairRow_deletion_witness
    {Value : Type} (residual : Value → F)
    (invalid valid : Value)
    (invalidNonzero : residual invalid ≠ 0) :
    BooleanRows.schedule ([0, 1] : List Nat) = [.pair 0 1] ∧
      RowsHold (fun coordinate =>
        residual (pairDeletionAssignment invalid valid coordinate))
        ((BooleanRows.schedule ([0, 1] : List Nat)).eraseIdx 0) ∧
      ¬ CoordinatesResidualZero (fun coordinate =>
        residual (pairDeletionAssignment invalid valid coordinate)) [0, 1] ∧
      ¬ RowHolds (fun coordinate =>
        residual (pairDeletionAssignment invalid valid coordinate))
        (.pair 0 1) := by
  constructor
  · rfl
  constructor
  · simp [RowsHold, BooleanPairRows.schedule]
  constructor
  · intro allZero
    apply invalidNonzero
    exact allZero 0 (by simp)
  · intro packed
    apply invalidNonzero
    exact (rowHolds_iff_residualsZero _ (.pair 0 1)).mp packed 0
      (by simp)

private theorem oddTailRow_deletion_witness
    {Value : Type} (residual : Value → F)
    (invalid valid : Value)
    (invalidNonzero : residual invalid ≠ 0)
    (validZero : residual valid = 0) :
    BooleanRows.schedule ([0, 1, 2] : List Nat) =
        [.pair 0 1, .tail 2] ∧
      RowsHold (fun coordinate =>
        residual (oddTailDeletionAssignment invalid valid coordinate))
        ((BooleanRows.schedule ([0, 1, 2] : List Nat)).eraseIdx 1) ∧
      ¬ CoordinatesResidualZero (fun coordinate =>
        residual (oddTailDeletionAssignment invalid valid coordinate))
        [0, 1, 2] ∧
      ¬ RowHolds (fun coordinate =>
        residual (oddTailDeletionAssignment invalid valid coordinate))
        (.tail 2) := by
  constructor
  · rfl
  constructor
  · intro row rowMember
    have rowEq : row = .pair 0 1 := by
      simpa [BooleanPairRows.schedule] using rowMember
    subst row
    apply (rowHolds_iff_residualsZero _ (.pair 0 1)).mpr
    intro coordinate coordinateMember
    simp [BooleanRows.Row.coordinates] at coordinateMember
    rcases coordinateMember with rfl | rfl <;>
      simpa [oddTailDeletionAssignment] using validZero
  constructor
  · intro allZero
    apply invalidNonzero
    exact allZero 2 (by simp)
  · intro tail
    apply invalidNonzero
    exact (rowHolds_iff_residualsZero _ (.tail 2)).mp tail 2
      (by simp)

def validOneProductInput : OneProductInput :=
  ⟨0, 0, 0⟩

def invalidOneProductInput : OneProductInput :=
  ⟨1, 1, 0⟩

private theorem validOneProductResidual_zero :
    oneProductResidual validOneProductInput = 0 := by
  norm_num [validOneProductInput, oneProductResidual]

private theorem invalidOneProductResidual_nonzero :
    oneProductResidual invalidOneProductInput ≠ 0 := by
  norm_num [invalidOneProductInput, oneProductResidual]

theorem oneProduct_pairRow_is_necessary :
    BooleanRows.schedule ([0, 1] : List Nat) = [.pair 0 1] ∧
      RowsHold (fun coordinate => oneProductResidual
        (pairDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate))
        ((BooleanRows.schedule ([0, 1] : List Nat)).eraseIdx 0) ∧
      ¬ CoordinatesResidualZero (fun coordinate => oneProductResidual
        (pairDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate)) [0, 1] ∧
      ¬ RowHolds (fun coordinate => oneProductResidual
        (pairDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate)) (.pair 0 1) :=
  pairRow_deletion_witness oneProductResidual
    invalidOneProductInput validOneProductInput
    invalidOneProductResidual_nonzero

theorem oneProduct_oddTailRow_is_necessary :
    BooleanRows.schedule ([0, 1, 2] : List Nat) =
        [.pair 0 1, .tail 2] ∧
      RowsHold (fun coordinate => oneProductResidual
        (oddTailDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate))
        ((BooleanRows.schedule ([0, 1, 2] : List Nat)).eraseIdx 1) ∧
      ¬ CoordinatesResidualZero (fun coordinate => oneProductResidual
        (oddTailDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate)) [0, 1, 2] ∧
      ¬ RowHolds (fun coordinate => oneProductResidual
        (oddTailDeletionAssignment invalidOneProductInput
          validOneProductInput coordinate)) (.tail 2) :=
  oddTailRow_deletion_witness oneProductResidual
    invalidOneProductInput validOneProductInput
    invalidOneProductResidual_nonzero validOneProductResidual_zero

private theorem centeredZeroResidual_zero :
    centeredUnitResidual (0 : F) = 0 := by
  norm_num [centeredUnitResidual]

private theorem centeredTwoResidual_nonzero :
    centeredUnitResidual (2 : F) ≠ 0 := by
  native_decide

theorem centeredUnit_pairRow_is_necessary :
    BooleanRows.schedule ([0, 1] : List Nat) = [.pair 0 1] ∧
      RowsHold (fun coordinate => centeredUnitResidual
        (pairDeletionAssignment (2 : F) 0 coordinate))
        ((BooleanRows.schedule ([0, 1] : List Nat)).eraseIdx 0) ∧
      ¬ CoordinatesResidualZero (fun coordinate => centeredUnitResidual
        (pairDeletionAssignment (2 : F) 0 coordinate)) [0, 1] ∧
      ¬ RowHolds (fun coordinate => centeredUnitResidual
        (pairDeletionAssignment (2 : F) 0 coordinate)) (.pair 0 1) :=
  pairRow_deletion_witness centeredUnitResidual (2 : F) 0
    centeredTwoResidual_nonzero

theorem centeredUnit_oddTailRow_is_necessary :
    BooleanRows.schedule ([0, 1, 2] : List Nat) =
        [.pair 0 1, .tail 2] ∧
      RowsHold (fun coordinate => centeredUnitResidual
        (oddTailDeletionAssignment (2 : F) 0 coordinate))
        ((BooleanRows.schedule ([0, 1, 2] : List Nat)).eraseIdx 1) ∧
      ¬ CoordinatesResidualZero (fun coordinate => centeredUnitResidual
        (oddTailDeletionAssignment (2 : F) 0 coordinate)) [0, 1, 2] ∧
      ¬ RowHolds (fun coordinate => centeredUnitResidual
        (oddTailDeletionAssignment (2 : F) 0 coordinate)) (.tail 2) :=
  oddTailRow_deletion_witness centeredUnitResidual (2 : F) 0
    centeredTwoResidual_nonzero centeredZeroResidual_zero

end SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies
