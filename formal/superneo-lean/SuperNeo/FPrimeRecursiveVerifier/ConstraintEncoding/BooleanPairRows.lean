import Mathlib.Tactic
import SuperNeo.Primitives.ExtensionField

/-!
Owns: the deterministic pair/odd-tail schedule for a finite ordered sequence
of Boolean coordinates, its exact Goldilocks semantics, structural census,
degree budget, and per-row-family necessity witnesses.

Does not own: any protocol phase's coordinate selection, Rust row emission,
generated matrices, selector columns, inactive materialization, column
freshness, or authorization to remove a production row.

Emits constraints: no. This file specifies and proves a reusable encoding.

Authority boundary: coordinate identifiers and their order are caller-owned.
The equivalence uses the proved fact that seven is a Goldilocks quadratic
nonresidue. A production consumer must still prove that its generated rows and
matrix columns instantiate this exact schedule under a verifier-fixed
selector.

Assurance tier: model-level.

| Predicate/theorem | Mathematical obligation | Guarantee | Assumptions | Permits Rust row removal? |
|---|---|---|---|---|
| `isBoolean_iff_zero_or_one` | Boolean semantics | a zero bit residual is exactly a field value in `{0,1}` | Goldilocks field | no |
| `quadraticZeroPair_iff` | nonresidue zero packing | one pair row vanishes iff both residuals vanish | `KExt.w_not_square` | no |
| `scheduledCoordinates_exact` | schedule ownership | flattening row coordinates recovers the input in order | deterministic recursion | no |
| `schedule_shape_counts` | exact cost | `n/2` pair rows, `n%2` tails, `ceil(n/2)` total | list length | no |
| `scheduleHolds_iff_coordinatesBoolean` | Boolean membership | scheduled rows accept iff every named coordinate is Boolean | exact pair/tail semantics | no |
| `selectorGatedScheduleHolds_iff` | active selector semantics | a verifier-fixed one selector preserves the exact equivalence | selector equals one | no |
| `schedule_selectorGatedDegree_le_five` | CCS degree budget | every scheduled row has gated degree at most five | one linear selector factor | no |
| `pairRow_is_necessary` | pair-row minimality | deleting a pair row admits a non-Boolean pair | concrete witness | no |
| `oddTailRow_is_necessary` | tail-row minimality | deleting the odd tail admits a non-Boolean tail | concrete witness | no |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.BooleanPairRows

/-- Residual of ordinary field Boolean membership, `x * (x - 1)`. -/
def bitResidual (value : F) : F :=
  value * (value - 1)

/-- One field coordinate is Boolean exactly when its bit residual vanishes. -/
def IsBoolean (value : F) : Prop :=
  bitResidual value = 0

instance (value : F) : Decidable (IsBoolean value) := by
  unfold IsBoolean
  infer_instance

/-- The residual definition denotes the two Boolean field values, not merely
an opaque zero-test. -/
theorem isBoolean_iff_zero_or_one {value : F} :
    IsBoolean value ↔ value = 0 ∨ value = 1 := by
  constructor
  · intro boolean
    rcases mul_eq_zero.mp boolean with zero | one
    · exact Or.inl zero
    · exact Or.inr (sub_eq_zero.mp one)
  · rintro (rfl | rfl) <;> simp [IsBoolean, bitResidual]

/-- One nonresidue equation packing two independent zero residuals. -/
def QuadraticZeroPair (left right : F) : Prop :=
  left * left - KExt.w * (right * right) = 0

/-- Seven's nonresiduosity prevents cancellation between the two residuals. -/
theorem quadraticZeroPair_iff {left right : F} :
    QuadraticZeroPair left right ↔ left = 0 ∧ right = 0 := by
  constructor
  · intro packed
    by_cases rightZero : right = 0
    · subst right
      simp only [QuadraticZeroPair, mul_zero, sub_zero] at packed
      exact ⟨mul_self_eq_zero.mp packed, rfl⟩
    · exfalso
      apply KExt.w_not_square (left / right)
      have equation : left * left = KExt.w * (right * right) :=
        sub_eq_zero.mp packed
      field_simp [rightZero]
      simpa [pow_two, mul_comm, mul_left_comm, mul_assoc] using equation
  · rintro ⟨rfl, rfl⟩
    simp [QuadraticZeroPair]

/-- One deterministic Boolean-membership row over coordinate identifiers. -/
inductive Row (Coordinate : Type) where
  | pair (left right : Coordinate)
  | tail (coordinate : Coordinate)
deriving DecidableEq, Repr

namespace Row

/-- Coordinates consumed by a row, in source order. -/
def coordinates {Coordinate : Type} : Row Coordinate → List Coordinate
  | .pair left right => [left, right]
  | .tail coordinate => [coordinate]

/-- Rename the coordinates of one scheduled row. -/
def map {Coordinate Target : Type} (rename : Coordinate → Target) :
    Row Coordinate → Row Target
  | .pair left right => .pair (rename left) (rename right)
  | .tail coordinate => .tail (rename coordinate)

/-- Polynomial residual represented by one row. -/
def residual {Coordinate : Type}
    (assignment : Coordinate → F) : Row Coordinate → F
  | .pair left right =>
      let leftResidual := bitResidual (assignment left)
      let rightResidual := bitResidual (assignment right)
      leftResidual * leftResidual - KExt.w * (rightResidual * rightResidual)
  | .tail coordinate => bitResidual (assignment coordinate)

/-- Ungated satisfaction of one scheduled row. -/
def Holds {Coordinate : Type}
    (assignment : Coordinate → F) (row : Row Coordinate) : Prop :=
  row.residual assignment = 0

/-- A linear selector multiplies the complete row residual. -/
def SelectorGatedHolds {Coordinate : Type}
    (selector : F) (assignment : Coordinate → F)
    (row : Row Coordinate) : Prop :=
  selector * row.residual assignment = 0

/-- The pair row has degree four; the ordinary odd tail has degree two. -/
def ungatedDegree {Coordinate : Type} : Row Coordinate → Nat
  | .pair _ _ => 4
  | .tail _ => 2

/-- One linear selector factor raises the row degree by one. -/
def selectorGatedDegree {Coordinate : Type} (row : Row Coordinate) : Nat :=
  row.ungatedDegree + 1

theorem holds_iff_coordinatesBoolean
    {Coordinate : Type} (assignment : Coordinate → F)
    (row : Row Coordinate) :
    row.Holds assignment ↔
      ∀ coordinate ∈ row.coordinates, IsBoolean (assignment coordinate) := by
  cases row with
  | pair left right =>
      change QuadraticZeroPair
          (bitResidual (assignment left)) (bitResidual (assignment right)) ↔
        ∀ coordinate ∈ [left, right], IsBoolean (assignment coordinate)
      rw [quadraticZeroPair_iff]
      simp [IsBoolean]
  | tail coordinate =>
      simp [Holds, residual, coordinates, IsBoolean]

theorem selectorGatedHolds_iff
    {Coordinate : Type} {selector : F}
    (fixed : selector = 1) (assignment : Coordinate → F)
    (row : Row Coordinate) :
    row.SelectorGatedHolds selector assignment ↔ row.Holds assignment := by
  subst selector
  simp [SelectorGatedHolds, Holds]

@[simp] theorem pair_selectorGatedDegree
    {Coordinate : Type} (left right : Coordinate) :
    selectorGatedDegree (.pair left right) = 5 := by
  rfl

@[simp] theorem tail_selectorGatedDegree
    {Coordinate : Type} (coordinate : Coordinate) :
    selectorGatedDegree (.tail coordinate) = 3 := by
  rfl

theorem selectorGatedDegree_le_five
    {Coordinate : Type} (row : Row Coordinate) :
    row.selectorGatedDegree ≤ 5 := by
  cases row <;> simp [selectorGatedDegree, ungatedDegree]

end Row

/-- Pair adjacent coordinates from left to right and retain one ordinary row
for an odd final coordinate. -/
def schedule {Coordinate : Type} : List Coordinate → List (Row Coordinate)
  | [] => []
  | [coordinate] => [.tail coordinate]
  | left :: right :: rest => .pair left right :: schedule rest

/-- Coordinate stream represented by a row schedule. -/
def scheduledCoordinates {Coordinate : Type}
    (rows : List (Row Coordinate)) : List Coordinate :=
  rows.flatMap Row.coordinates

/-- Number of pair rows in a row list. -/
def pairRowCount {Coordinate : Type} (rows : List (Row Coordinate)) : Nat :=
  (rows.filter fun row => match row with
    | .pair _ _ => true
    | .tail _ => false).length

/-- Number of ordinary odd-tail rows in a row list. -/
def tailRowCount {Coordinate : Type} (rows : List (Row Coordinate)) : Nat :=
  (rows.filter fun row => match row with
    | .pair _ _ => false
    | .tail _ => true).length

@[simp] theorem pairRowCount_pair_cons
    {Coordinate : Type} (left right : Coordinate)
    (rows : List (Row Coordinate)) :
    pairRowCount (.pair left right :: rows) = 1 + pairRowCount rows := by
  simp [pairRowCount, Nat.add_comm]

@[simp] theorem pairRowCount_tail_cons
    {Coordinate : Type} (coordinate : Coordinate)
    (rows : List (Row Coordinate)) :
    pairRowCount (.tail coordinate :: rows) = pairRowCount rows := by
  simp [pairRowCount]

@[simp] theorem tailRowCount_pair_cons
    {Coordinate : Type} (left right : Coordinate)
    (rows : List (Row Coordinate)) :
    tailRowCount (.pair left right :: rows) = tailRowCount rows := by
  simp [tailRowCount]

@[simp] theorem tailRowCount_tail_cons
    {Coordinate : Type} (coordinate : Coordinate)
    (rows : List (Row Coordinate)) :
    tailRowCount (.tail coordinate :: rows) = 1 + tailRowCount rows := by
  simp [tailRowCount, Nat.add_comm]

/-- Natural-number ceiling of division by two. -/
def ceilHalf (count : Nat) : Nat :=
  (count + 1) / 2

/-- Row satisfaction independent of how rows were produced. -/
def RowsHold {Coordinate : Type}
    (assignment : Coordinate → F) (rows : List (Row Coordinate)) : Prop :=
  ∀ row ∈ rows, row.Holds assignment

/-- Pointwise Boolean meaning of an ordered coordinate sequence. -/
def CoordinatesBoolean {Coordinate : Type}
    (assignment : Coordinate → F) (coordinates : List Coordinate) : Prop :=
  ∀ coordinate ∈ coordinates, IsBoolean (assignment coordinate)

/-- Satisfaction of the deterministic schedule for one coordinate sequence. -/
def ScheduleHolds {Coordinate : Type}
    (assignment : Coordinate → F) (coordinates : List Coordinate) : Prop :=
  RowsHold assignment (schedule coordinates)

/-- Active satisfaction after multiplying each row by one shared selector. -/
def SelectorGatedScheduleHolds {Coordinate : Type}
    (selector : F) (assignment : Coordinate → F)
    (coordinates : List Coordinate) : Prop :=
  ∀ row ∈ schedule coordinates, row.SelectorGatedHolds selector assignment

/-- Flattening the schedule recovers the input coordinates in exact order. -/
theorem scheduledCoordinates_exact {Coordinate : Type} :
    ∀ coordinates : List Coordinate,
      scheduledCoordinates (schedule coordinates) = coordinates
  | [] => by rfl
  | [coordinate] => by rfl
  | left :: right :: rest => by
      change left :: right :: scheduledCoordinates (schedule rest) =
        left :: right :: rest
      rw [scheduledCoordinates_exact rest]

theorem scheduledCoordinates_mem_iff
    {Coordinate : Type} (coordinates : List Coordinate)
    (coordinate : Coordinate) :
    coordinate ∈ scheduledCoordinates (schedule coordinates) ↔
      coordinate ∈ coordinates := by
  rw [scheduledCoordinates_exact]

/-- Pairing neither introduces nor hides duplicate coordinate identifiers. -/
theorem scheduledCoordinates_nodup_iff
    {Coordinate : Type} (coordinates : List Coordinate) :
    (scheduledCoordinates (schedule coordinates)).Nodup ↔
      coordinates.Nodup := by
  rw [scheduledCoordinates_exact]

/-- Renaming coordinates commutes with construction of the schedule. -/
theorem schedule_map
    {Coordinate Target : Type} (rename : Coordinate → Target) :
    ∀ coordinates : List Coordinate,
      schedule (coordinates.map rename) =
        (schedule coordinates).map (Row.map rename)
  | [] => by rfl
  | [coordinate] => by rfl
  | left :: right :: rest => by
      simp [schedule, Row.map, schedule_map rename rest]

theorem schedule_pairRowCount {Coordinate : Type} :
    ∀ coordinates : List Coordinate,
      pairRowCount (schedule coordinates) = coordinates.length / 2
  | [] => by simp [schedule, pairRowCount]
  | [coordinate] => by simp [schedule, pairRowCount]
  | left :: right :: rest => by
      simp only [schedule, pairRowCount_pair_cons, List.length_cons]
      rw [schedule_pairRowCount rest]
      omega

theorem schedule_tailRowCount {Coordinate : Type} :
    ∀ coordinates : List Coordinate,
      tailRowCount (schedule coordinates) = coordinates.length % 2
  | [] => by rfl
  | [coordinate] => by rfl
  | left :: right :: rest => by
      simp only [schedule, tailRowCount_pair_cons, List.length_cons]
      rw [schedule_tailRowCount rest]
      omega

theorem schedule_length {Coordinate : Type} :
    ∀ coordinates : List Coordinate,
      (schedule coordinates).length = ceilHalf coordinates.length
  | [] => by simp [schedule, ceilHalf]
  | [coordinate] => by simp [schedule, ceilHalf]
  | left :: right :: rest => by
      simp only [schedule, List.length_cons]
      rw [schedule_length rest]
      simp only [ceilHalf]
      omega

/-- Exact pair/tail/total census, including the unique odd tail. -/
theorem schedule_shape_counts
    {Coordinate : Type} (coordinates : List Coordinate) :
    pairRowCount (schedule coordinates) = coordinates.length / 2 ∧
      tailRowCount (schedule coordinates) = coordinates.length % 2 ∧
      (schedule coordinates).length = ceilHalf coordinates.length := by
  exact ⟨schedule_pairRowCount coordinates,
    schedule_tailRowCount coordinates, schedule_length coordinates⟩

/-- Any row list is equivalent to Booleanity of its flattened coordinates. -/
theorem rowsHold_iff_scheduledCoordinatesBoolean
    {Coordinate : Type} (assignment : Coordinate → F)
    (rows : List (Row Coordinate)) :
    RowsHold assignment rows ↔
      CoordinatesBoolean assignment (scheduledCoordinates rows) := by
  constructor
  · intro rowsHold coordinate coordinateMember
    rcases List.mem_flatMap.mp coordinateMember with
      ⟨row, rowMember, memberInRow⟩
    exact (Row.holds_iff_coordinatesBoolean assignment row).mp
      (rowsHold row rowMember) coordinate memberInRow
  · intro coordinatesBoolean row rowMember
    apply (Row.holds_iff_coordinatesBoolean assignment row).mpr
    intro coordinate memberInRow
    apply coordinatesBoolean coordinate
    exact List.mem_flatMap.mpr ⟨row, rowMember, memberInRow⟩

/-- The deterministic packed schedule accepts exactly pointwise Boolean
assignments, for every finite ordered coordinate list. -/
theorem scheduleHolds_iff_coordinatesBoolean
    {Coordinate : Type} (assignment : Coordinate → F)
    (coordinates : List Coordinate) :
    ScheduleHolds assignment coordinates ↔
      CoordinatesBoolean assignment coordinates := by
  rw [ScheduleHolds, rowsHold_iff_scheduledCoordinatesBoolean,
    scheduledCoordinates_exact]

theorem schedule_sound
    {Coordinate : Type} {assignment : Coordinate → F}
    {coordinates : List Coordinate}
    (holds : ScheduleHolds assignment coordinates) :
    CoordinatesBoolean assignment coordinates :=
  (scheduleHolds_iff_coordinatesBoolean assignment coordinates).mp holds

theorem schedule_complete
    {Coordinate : Type} {assignment : Coordinate → F}
    {coordinates : List Coordinate}
    (boolean : CoordinatesBoolean assignment coordinates) :
    ScheduleHolds assignment coordinates :=
  (scheduleHolds_iff_coordinatesBoolean assignment coordinates).mpr boolean

/-- Verifier-fixed selector value one preserves the exact active semantics. -/
theorem selectorGatedScheduleHolds_iff
    {Coordinate : Type} {selector : F}
    (fixed : selector = 1) (assignment : Coordinate → F)
    (coordinates : List Coordinate) :
    SelectorGatedScheduleHolds selector assignment coordinates ↔
      CoordinatesBoolean assignment coordinates := by
  rw [← scheduleHolds_iff_coordinatesBoolean assignment coordinates]
  constructor
  · intro gated row rowMember
    exact (Row.selectorGatedHolds_iff fixed assignment row).mp
      (gated row rowMember)
  · intro ungated row rowMember
    exact (Row.selectorGatedHolds_iff fixed assignment row).mpr
      (ungated row rowMember)

/-- Every row in every schedule fits the degree-five selector-gated budget. -/
theorem schedule_selectorGatedDegree_le_five
    {Coordinate : Type} (coordinates : List Coordinate) :
    ∀ row ∈ schedule coordinates, row.selectorGatedDegree ≤ 5 := by
  intro row _
  exact Row.selectorGatedDegree_le_five row

/-! ## Necessity witnesses -/

private def pairNecessityAssignment : Nat → F
  | 0 => 2
  | _ => 0

private def oddTailNecessityAssignment : Nat → F
  | 2 => 2
  | _ => 0

private theorem fieldTwo_not_boolean : ¬ IsBoolean (2 : F) := by
  native_decide

/-- If the sole pair row is deleted, a non-Boolean first coordinate is
accepted by the remaining empty schedule. The deleted row itself rejects. -/
theorem pairRow_is_necessary :
    schedule ([0, 1] : List Nat) = [.pair 0 1] ∧
      RowsHold pairNecessityAssignment
        ((schedule ([0, 1] : List Nat)).eraseIdx 0) ∧
      ¬ CoordinatesBoolean pairNecessityAssignment [0, 1] ∧
      ¬ (Row.pair 0 1).Holds pairNecessityAssignment := by
  constructor
  · rfl
  constructor
  · simp [RowsHold, schedule]
  constructor
  · intro boolean
    apply fieldTwo_not_boolean
    exact boolean 0 (by simp)
  · intro holds
    apply fieldTwo_not_boolean
    exact (Row.holds_iff_coordinatesBoolean pairNecessityAssignment
      (.pair 0 1)).mp holds 0 (by simp [Row.coordinates])

/-- If the odd-tail row is deleted, the preceding valid pair still holds
while a non-Boolean trailing coordinate is admitted. The tail row rejects. -/
theorem oddTailRow_is_necessary :
    schedule ([0, 1, 2] : List Nat) = [.pair 0 1, .tail 2] ∧
      RowsHold oddTailNecessityAssignment
        ((schedule ([0, 1, 2] : List Nat)).eraseIdx 1) ∧
      ¬ CoordinatesBoolean oddTailNecessityAssignment [0, 1, 2] ∧
      ¬ (Row.tail 2).Holds oddTailNecessityAssignment := by
  constructor
  · rfl
  constructor
  · intro row member
    have rowEq : row = .pair 0 1 := by
      simpa [schedule] using member
    subst row
    apply (Row.holds_iff_coordinatesBoolean oddTailNecessityAssignment
      (.pair 0 1)).mpr
    intro coordinate coordinateMember
    simp [Row.coordinates] at coordinateMember
    rcases coordinateMember with rfl | rfl <;>
      simp [oddTailNecessityAssignment, IsBoolean, bitResidual]
  constructor
  · intro boolean
    apply fieldTwo_not_boolean
    exact boolean 2 (by simp)
  · intro holds
    apply fieldTwo_not_boolean
    exact (Row.holds_iff_coordinatesBoolean oddTailNecessityAssignment
      (.tail 2)).mp holds 2 (by simp [Row.coordinates])

end SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.BooleanPairRows
