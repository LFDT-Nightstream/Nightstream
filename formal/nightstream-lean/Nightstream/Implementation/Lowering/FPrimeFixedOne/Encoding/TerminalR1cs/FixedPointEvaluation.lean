import Nightstream.Implementation.R1CS.Correspondence.TerminalR1cs.Atoms
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-!
Contract: Lean-derived dense linear rows for Phi81 matrix evaluations at one
fixed point.

Assurance tier: model-level.

Owns: canonical basis decomposition of the complete assignment, exact
linearization of every application-matrix/lane evaluation at a verifier-fixed
point, two physical rows per claimed `K` value, support, ownership, soundness,
honest completeness, and cost.

Does not own: the separate padded-identity evaluation, a run-time evaluation
point, witness or claim allocation, transcript derivation, Ajtai commitments,
norm checks, fresh CCS satisfaction, terminal composition, Rust, or artifacts.
A uniform terminal relation must not compile a witness-supplied point into
these coefficients.

Emits constraints: `2 * shape.matrixCount * ringDegree` dense linear rows and
no auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FixedPointEvaluation

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS.TerminalR1cs
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Phi81Relation.EvaluationHomomorphism

/-! ## Canonical finite basis decomposition -/

/-- One complete-carrier unit basis assignment. -/
def basisAssignment {shape : Phi81Relation.Shape}
    (selected : Fin shape.carrierWidth) :
    Phi81Relation.Assignment shape :=
  oneHotAssignment ConcreteCarrier.baseOps selected 1

/-- Head-first sum matching the recursive order used by `combineAssignments`. -/
def headSum :
    {count : Nat} → (Fin count → F) → F
  | 0, _ => 0
  | _ + 1, term =>
      term 0 + headSum (fun index => term index.succ)

private theorem headSum_zero :
    ∀ {count : Nat},
      headSum (fun _ : Fin count => (0 : F)) = 0
  | 0 => rfl
  | _ + 1 => by
      simp only [headSum, headSum_zero]
      exact Fin.zero_add 0

private theorem headSum_select :
    ∀ {count : Nat} (selected : Fin count) (value : F),
      headSum (fun index => if index = selected then value else 0) = value
  | 0, selected, value => Fin.elim0 selected
  | count + 1, selected, value => by
      refine Fin.cases ?_ (fun tail => ?_) selected
      · simp only [headSum, if_pos]
        have tailZero :
            (fun index : Fin count =>
              if index.succ = (0 : Fin (count + 1)) then value else 0) =
              (fun _ : Fin count => 0) := by
          funext index
          rw [if_neg (Fin.succ_ne_zero index)]
        rw [tailZero, headSum_zero, Fin.add_zero]
      · simp only [headSum,
          if_neg (Fin.succ_ne_zero tail).symm]
        have tailTerm :
            (fun index : Fin _ =>
              if index.succ = tail.succ then value else 0) =
              (fun index => if index = tail then value else 0) := by
          funext index
          simp only [Fin.succ_inj]
        rw [tailTerm, headSum_select tail value, Fin.zero_add]

private theorem headSum_congr
    {count : Nat}
    {left right : Fin count → F}
    (equal : ∀ index, left index = right index) :
    headSum left = headSum right := by
  rw [show left = right by funext index; exact equal index]

private theorem combineAssignments_apply
    {shape : Phi81Relation.Shape} :
    ∀ {count : Nat}
      (weights : Fin count → F)
      (assignments :
        Fin count → Phi81Relation.Assignment shape)
      (coordinate : Fin shape.carrierWidth),
      BaseLinear.combineAssignments weights assignments coordinate =
        headSum fun index =>
          weights index * assignments index coordinate
  | 0, weights, assignments, coordinate => rfl
  | _ + 1, weights, assignments, coordinate => by
      simp only [BaseLinear.combineAssignments,
        BaseLinear.assignmentAdd, BaseLinear.assignmentScale,
        BaseLinear.Raw.assignmentAdd, BaseLinear.Raw.assignmentScale,
        headSum]
      rw [combineAssignments_apply
        (fun index => weights index.succ)
        (fun index => assignments index.succ)
        coordinate]

/-- Every complete assignment is the exact finite combination of its unit
basis assignments. -/
theorem combine_basis_eq_assignment
    {shape : Phi81Relation.Shape}
    (assignment : Phi81Relation.Assignment shape) :
    BaseLinear.combineAssignments
        (fun coordinate => assignment coordinate)
        (fun coordinate => basisAssignment coordinate) =
      assignment := by
  funext coordinate
  rw [combineAssignments_apply]
  have termEquality :
      (fun index : Fin shape.carrierWidth =>
        assignment index * basisAssignment index coordinate) =
      (fun index =>
        if index = coordinate then assignment coordinate else 0) := by
    funext index
    by_cases equal : index = coordinate
    · subst index
      simp [basisAssignment, oneHotAssignment, Fin.mul_one]
    · simp [basisAssignment, oneHotAssignment, ConcreteCarrier.baseOps,
        equal, Ne.symm equal, Fin.mul_zero]
  rw [termEquality, headSum_select]

/-! ## Exact dense coefficient rows -/

/-- Physical columns read by all evaluation checks. Claimed `K` values are
caller-owned terminal inputs; this slice allocates no claim column. -/
structure Frame (shape : Phi81Relation.Shape) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  witness : Fin shape.carrierWidth → ColumnId
  claimLow :
    Fin shape.matrixCount → Fin ringDegree → ColumnId
  claimHigh :
    Fin shape.matrixCount → Fin ringDegree → ColumnId

/-- The semantic image of one unit basis coordinate. -/
def basisValue {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (coordinate : Fin shape.carrierWidth) : K :=
  Phi81Relation.matrixEvaluation system
    (basisAssignment coordinate) point matrix lane

/-- Exact low-coordinate coefficient list in canonical witness order. -/
def lowCombination {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : LinearCombination :=
  List.ofFn fun coordinate =>
    ⟨frame.witness coordinate,
      (basisValue system point matrix lane coordinate).c0⟩

/-- Exact high-coordinate coefficient list in canonical witness order. -/
def highCombination {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : LinearCombination :=
  List.ofFn fun coordinate =>
    ⟨frame.witness coordinate,
      (basisValue system point matrix lane coordinate).c1⟩

private theorem eval_ofFn_eq_headSum
    {count : Nat}
    (term : Fin count → Term)
    (assignment : ColumnId → F) :
    LinearCombination.eval assignment (List.ofFn term) =
      headSum fun index =>
        (term index).coefficient * assignment (term index).column := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [LinearCombination.eval_cons, headSum]
      rw [inductionHypothesis]

private theorem combineEvaluations_lane
    {count : Nat}
    (weights : Fin count → F)
    (values : Fin count → Phi81Relation.Evaluation)
    (lane : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights values lane) =
      ⟨headSum fun index => weights index * (values index lane).c0,
       headSum fun index => weights index * (values index lane).c1⟩ := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations,
        BaseLinear.evaluationAdd, BaseLinear.evaluationScale,
        K.mul, K.embed, K.add, headSum]
      rw [inductionHypothesis]
      simp only [Fin.mul_zero, Fin.zero_mul, Fin.add_zero]

private theorem canonical_matrix_value
    {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (assignment : Phi81Relation.Assignment shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    Phi81Relation.matrixEvaluation system assignment point matrix lane =
      ⟨headSum fun coordinate =>
          assignment coordinate *
            (basisValue system point matrix lane coordinate).c0,
       headSum fun coordinate =>
          assignment coordinate *
            (basisValue system point matrix lane coordinate).c1⟩ := by
  have combined :=
    BaseLinear.matrixEvaluation_combine system
      (fun coordinate => assignment coordinate)
      (fun coordinate => basisAssignment coordinate)
      point matrix
  rw [combine_basis_eq_assignment assignment] at combined
  have laneCombined := congrFun combined lane
  rw [combineEvaluations_lane] at laneCombined
  exact laneCombined

/-- Dense low row computes the exact low coordinate of one claimed value. -/
theorem lowCombination_eval {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (assignment : ColumnId → F) :
    (lowCombination frame system point matrix lane).eval assignment =
      (Phi81Relation.matrixEvaluation system
        (fun coordinate => assignment (frame.witness coordinate))
        point matrix lane).c0 := by
  rw [lowCombination, eval_ofFn_eq_headSum]
  rw [headSum_congr (fun coordinate =>
    Fin.mul_comm
      (basisValue system point matrix lane coordinate).c0
      (assignment (frame.witness coordinate)))]
  exact congrArg K.c0
    (canonical_matrix_value system
      (fun coordinate => assignment (frame.witness coordinate))
      point matrix lane).symm

/-- Dense high row computes the exact high coordinate of one claimed value. -/
theorem highCombination_eval {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (assignment : ColumnId → F) :
    (highCombination frame system point matrix lane).eval assignment =
      (Phi81Relation.matrixEvaluation system
        (fun coordinate => assignment (frame.witness coordinate))
        point matrix lane).c1 := by
  rw [highCombination, eval_ofFn_eq_headSum]
  rw [headSum_congr (fun coordinate =>
    Fin.mul_comm
      (basisValue system point matrix lane coordinate).c1
      (assignment (frame.witness coordinate)))]
  exact congrArg K.c1
    (canonical_matrix_value system
      (fun coordinate => assignment (frame.witness coordinate))
      point matrix lane).symm

/-- Flattened matrix/lane index for one of the two coordinate rows. -/
def valueIndex {shape : Phi81Relation.Shape}
    (position :
      Fin (2 * (shape.matrixCount * ringDegree))) : Nat :=
  position.val / 2

def matrixAt {shape : Phi81Relation.Shape}
    (position :
      Fin (2 * (shape.matrixCount * ringDegree))) :
    Fin shape.matrixCount :=
  ⟨valueIndex position / ringDegree, by
    have below := position.isLt
    simp only [valueIndex, ringDegree] at *
    omega⟩

def laneAt {shape : Phi81Relation.Shape}
    (position :
      Fin (2 * (shape.matrixCount * ringDegree))) :
    Fin ringDegree :=
  ⟨valueIndex position % ringDegree, by
    simp [ringDegree]
    omega⟩

/-- One positional row; even positions bind low coordinates and odd positions
bind high coordinates. -/
def rowAt {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (position :
      Fin (2 * (shape.matrixCount * ringDegree))) : OwnedRow :=
  if position.val % 2 = 0 then
    Atoms.linearCheckOwnedRow frame.owner
      (frame.firstOrdinal + position.val) frame.one
      (lowCombination frame system point (matrixAt position) (laneAt position))
      (Goldilocks.singleton
        (frame.claimLow (matrixAt position) (laneAt position)) 1)
  else
    Atoms.linearCheckOwnedRow frame.owner
      (frame.firstOrdinal + position.val) frame.one
      (highCombination frame system point (matrixAt position) (laneAt position))
      (Goldilocks.singleton
        (frame.claimHigh (matrixAt position) (laneAt position)) 1)

def rows {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape) : List OwnedRow :=
  List.ofFn (rowAt frame system point)

def columns {shape : Phi81Relation.Shape}
    (_frame : Frame shape) : List OwnedColumn :=
  []

@[simp] theorem rows_length {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape) :
    (rows frame system point).length =
      2 * (shape.matrixCount * ringDegree) := by
  simp [rows]

@[simp] theorem columns_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (columns frame).length = 0 :=
  rfl

theorem columnIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    ((columns frame).map fun column => column.id).Nodup := by
  simp [columns]

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {count : Nat}
      (function : Fin count → alpha),
      Function.Injective function →
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

theorem rowIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape) :
    ((rows frame system point).map fun owned => owned.id).Nodup := by
  rw [rows, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  have ordinalEqual :=
    congrArg (fun id : RowId => id.ordinal) equal
  simp only [Function.comp_apply, rowAt,
    Atoms.linearCheckOwnedRow] at ordinalEqual
  split at ordinalEqual <;> split at ordinalEqual <;>
    exact Nat.add_left_cancel ordinalEqual

theorem rows_owned {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (owned : OwnedRow)
    (member : owned ∈ rows frame system point) :
    owned.id.owner = frame.owner := by
  rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
  unfold rowAt Atoms.linearCheckOwnedRow
  split <;> rfl

private theorem lowCombination_supported
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (column : ColumnId)
    (mentioned :
      column ∈
        (lowCombination frame system point matrix lane).map
          fun term => term.column) :
    ∃ coordinate, column = frame.witness coordinate := by
  rcases List.mem_map.mp mentioned with
    ⟨term, termMember, rfl⟩
  rcases List.mem_ofFn.mp termMember with ⟨coordinate, rfl⟩
  exact ⟨coordinate, rfl⟩

private theorem highCombination_supported
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (column : ColumnId)
    (mentioned :
      column ∈
        (highCombination frame system point matrix lane).map
          fun term => term.column) :
    ∃ coordinate, column = frame.witness coordinate := by
  rcases List.mem_map.mp mentioned with
    ⟨term, termMember, rfl⟩
  rcases List.mem_ofFn.mp termMember with ⟨coordinate, rfl⟩
  exact ⟨coordinate, rfl⟩

/-- A fixed-point evaluation row mentions only the constant wire, a complete
assignment coordinate, or the selected claimed evaluation coordinate. -/
theorem rowAt_supported
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (position : Fin (2 * (shape.matrixCount * ringDegree)))
    (column : ColumnId)
    (mentioned :
      column ∈ (rowAt frame system point position).columnIds) :
    column = frame.one ∨
      (∃ coordinate, column = frame.witness coordinate) ∨
      column = frame.claimLow (matrixAt position) (laneAt position) ∨
      column = frame.claimHigh (matrixAt position) (laneAt position) := by
  by_cases even : position.val % 2 = 0
  · simp only [rowAt, even, if_pos, Atoms.linearCheckOwnedRow,
      Atoms.linearCheckRow, OwnedRow.columnIds, Row.columnIds]
      at mentioned
    rw [List.map_append, List.map_append] at mentioned
    rcases List.mem_append.mp mentioned with beforeClaim | claim
    · rcases List.mem_append.mp beforeClaim with combination | one
      · exact Or.inr (Or.inl
          (lowCombination_supported frame system point _ _ column
            combination))
      · exact Or.inl (by simpa [Goldilocks.singleton] using one)
    · exact Or.inr (Or.inr (Or.inl
        (by simpa [Goldilocks.singleton] using claim)))
  · simp only [rowAt, even, if_neg, Atoms.linearCheckOwnedRow,
      Atoms.linearCheckRow, OwnedRow.columnIds, Row.columnIds]
      at mentioned
    rw [List.map_append, List.map_append] at mentioned
    rcases List.mem_append.mp mentioned with beforeClaim | claim
    · rcases List.mem_append.mp beforeClaim with combination | one
      · exact Or.inr (Or.inl
          (highCombination_supported frame system point _ _ column
            combination))
      · exact Or.inl (by simpa [Goldilocks.singleton] using one)
    · exact Or.inr (Or.inr (Or.inr
        (by simpa [Goldilocks.singleton] using claim)))

private theorem satisfies_ofFn_iff :
    ∀ {count : Nat}
      (function : Fin count → OwnedRow)
      (assignment : ColumnId → F),
      Satisfies (List.ofFn function) assignment ↔
        ∀ position, (function position).row.Holds assignment
  | 0, function, assignment => by
      simp
  | _ + 1, function, assignment => by
      rw [List.ofFn_succ, satisfies_cons,
        satisfies_ofFn_iff (fun index => function index.succ) assignment]
      constructor
      · rintro ⟨head, tail⟩ position
        exact Fin.cases head tail position
      · intro every
        exact ⟨every 0, fun index => every index.succ⟩

private theorem satisfies_rows_iff {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (assignment : ColumnId → F) :
    Satisfies (rows frame system point) assignment ↔
      ∀ position,
        (rowAt frame system point position).row.Holds assignment :=
  satisfies_ofFn_iff (rowAt frame system point) assignment

private def lowPosition {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    Fin (2 * (shape.matrixCount * ringDegree)) :=
  ⟨2 * (matrix.val * ringDegree + lane.val), by
    have matrixBelow := matrix.isLt
    have laneBelow := lane.isLt
    simp only [ringDegree] at laneBelow ⊢
    omega⟩

private def highPosition {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    Fin (2 * (shape.matrixCount * ringDegree)) :=
  ⟨2 * (matrix.val * ringDegree + lane.val) + 1, by
    have matrixBelow := matrix.isLt
    have laneBelow := lane.isLt
    simp only [ringDegree] at laneBelow ⊢
    omega⟩

@[simp] private theorem matrixAt_lowPosition
    {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    matrixAt (lowPosition matrix lane) = matrix := by
  apply Fin.ext
  simp [matrixAt, valueIndex, lowPosition, ringDegree]
  omega

@[simp] private theorem laneAt_lowPosition
    {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    laneAt (lowPosition matrix lane) = lane := by
  apply Fin.ext
  simp [laneAt, valueIndex, lowPosition, ringDegree]

@[simp] private theorem matrixAt_highPosition
    {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    matrixAt (highPosition matrix lane) = matrix := by
  apply Fin.ext
  simp [matrixAt, valueIndex, highPosition, ringDegree]
  omega

@[simp] private theorem laneAt_highPosition
    {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    laneAt (highPosition matrix lane) = lane := by
  apply Fin.ext
  simp [laneAt, valueIndex, highPosition, ringDegree]
  omega

@[simp] private theorem rowAt_lowPosition
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    (rowAt frame system point (lowPosition matrix lane)).row =
      Atoms.linearCheckRow frame.one
        (lowCombination frame system point matrix lane)
        (Goldilocks.singleton (frame.claimLow matrix lane) 1) := by
  unfold rowAt
  rw [if_pos (by simp [lowPosition])]
  simp only [matrixAt_lowPosition, laneAt_lowPosition,
    Atoms.linearCheckOwnedRow]

@[simp] private theorem rowAt_highPosition
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    (rowAt frame system point (highPosition matrix lane)).row =
      Atoms.linearCheckRow frame.one
        (highCombination frame system point matrix lane)
        (Goldilocks.singleton (frame.claimHigh matrix lane) 1) := by
  unfold rowAt
  rw [if_neg (by simp [highPosition])]
  simp only [matrixAt_highPosition, laneAt_highPosition,
    Atoms.linearCheckOwnedRow]

/-- Satisfying rows bind every claimed matrix/lane value to the exact
authoritative evaluation. -/
theorem rows_sound {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows frame system point) assignment) :
    ∀ matrix lane,
      K.mk
        (assignment (frame.claimLow matrix lane))
        (assignment (frame.claimHigh matrix lane)) =
      Phi81Relation.matrixEvaluation system
        (fun coordinate => assignment (frame.witness coordinate))
        point matrix lane := by
  intro matrix lane
  have every := (satisfies_rows_iff frame system point assignment).mp satisfied
  have lowHolds := every (lowPosition matrix lane)
  have highHolds := every (highPosition matrix lane)
  rw [rowAt_lowPosition] at lowHolds
  rw [rowAt_highPosition] at highHolds
  have lowEqual :=
    (Atoms.linearCheckRow_iff assignment frame.one
      (lowCombination frame system point matrix lane)
      (Goldilocks.singleton (frame.claimLow matrix lane) 1)
      constantOne).mp lowHolds
  have highEqual :=
    (Atoms.linearCheckRow_iff assignment frame.one
      (highCombination frame system point matrix lane)
      (Goldilocks.singleton (frame.claimHigh matrix lane) 1)
      constantOne).mp highHolds
  change K.mk _ _ = K.mk _ _
  congr 1
  · simpa [Goldilocks.singleton, LinearCombination.eval,
      Fin.one_mul, Fin.add_zero, lowCombination_eval]
      using lowEqual.symm
  · simpa [Goldilocks.singleton, LinearCombination.eval,
      Fin.one_mul, Fin.add_zero, highCombination_eval]
      using highEqual.symm

/-- Canonical claimed values satisfy every dense evaluation row. -/
theorem rows_honest {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (claims :
      ∀ matrix lane,
        K.mk
          (assignment (frame.claimLow matrix lane))
          (assignment (frame.claimHigh matrix lane)) =
        Phi81Relation.matrixEvaluation system
          (fun coordinate => assignment (frame.witness coordinate))
          point matrix lane) :
    Satisfies (rows frame system point) assignment := by
  apply (satisfies_rows_iff frame system point assignment).mpr
  intro position
  have claim := claims (matrixAt position) (laneAt position)
  have lowClaim := congrArg K.c0 claim
  have highClaim := congrArg K.c1 claim
  by_cases even : position.val % 2 = 0
  · simp only [rowAt, even, if_pos, Atoms.linearCheckOwnedRow]
    apply
      (Atoms.linearCheckRow_iff assignment frame.one
        (lowCombination frame system point
          (matrixAt position) (laneAt position))
        (Goldilocks.singleton
          (frame.claimLow (matrixAt position) (laneAt position)) 1)
        constantOne).mpr
    simpa [Goldilocks.singleton,
      LinearCombination.eval, Fin.one_mul, Fin.add_zero,
      lowCombination_eval] using lowClaim.symm
  · simp only [rowAt, even, Atoms.linearCheckOwnedRow]
    apply
      (Atoms.linearCheckRow_iff assignment frame.one
        (highCombination frame system point
          (matrixAt position) (laneAt position))
        (Goldilocks.singleton
          (frame.claimHigh (matrixAt position) (laneAt position)) 1)
        constantOne).mpr
    simpa [Goldilocks.singleton,
      LinearCombination.eval, Fin.one_mul, Fin.add_zero,
      highCombination_eval] using highClaim.symm

def cost (shape : Phi81Relation.Shape) : Cost :=
  ⟨2 * (shape.matrixCount * ringDegree), 0, 0, 0⟩

@[simp] theorem cost_rows (shape : Phi81Relation.Shape) :
    (cost shape).recurringRows =
      2 * (shape.matrixCount * ringDegree) :=
  rfl

@[simp] theorem cost_auxiliary (shape : Phi81Relation.Shape) :
    (cost shape).auxiliaryColumns = 0 :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FixedPointEvaluation
