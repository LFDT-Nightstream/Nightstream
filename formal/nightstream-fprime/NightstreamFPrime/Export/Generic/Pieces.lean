import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Lifecycle.Nebula.RowExprs
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.CombinationStep

/-! The pieces of a circuit's flat constraints, for the generic exporter of the bounded-depth step
circuits (CW-5). A block of constraints that is the canonical Poseidon2 permutation
(`PilotData.canonicalConstraints`) under a substitution is one permutation piece. The package
exports it as an invocation of today's permutation template, which Rust packs into 150 CCS rows.
A block that is one `CombinationStep` block of the Π_RLC combinations (`ringTemplate`, one or two
cells) is one ring-product piece, which `PieceRows` checks with 128 rows per cell. Every other
constraint is its own piece and is lowered to R1CS rows. `split` makes a block piece only after a
prefix check, so the pieces give back the constraints (`flatMap_split`), and recognition changes
only the row count. -/

namespace NightstreamFPrime.Export.Generic.Pieces

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

abbrev subst := NightstreamFPrime.Lifecycle.Nebula.Expr.subst

theorem fieldValue_val (value : F) : fieldValue value.val = value := by
  apply Fin.ext
  simp [fieldValue, Spec.Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

@[simp] theorem fieldValue_zero : fieldValue 0 = (0 : F) := rfl
@[simp] theorem fieldValue_one : fieldValue 1 = (1 : F) := rfl

theorem toR1CS_eval (combination : SparseCombination) (env : Env) :
    combination.toR1CS.eval env = combination.eval env := by
  simp [SparseCombination.toR1CS, SparseCombination.eval, R1CS.LinearCombination.eval,
    List.map_map, Function.comp_def]

/-! ### Linear lanes -/

def scale (factor : F) (combination : SparseCombination) : SparseCombination :=
  ⟨(factor * fieldValue combination.constant).val,
    combination.terms.map fun term =>
      ⟨term.column, (factor * fieldValue term.coefficient).val⟩⟩

def plus (left right : SparseCombination) : SparseCombination :=
  ⟨(fieldValue left.constant + fieldValue right.constant).val, left.terms ++ right.terms⟩

def constant? : Expr → Option F
  | .const value => some value
  | _ => none

/-- A linear expression as a sparse combination, or `none`. -/
def linear? : Expr → Option SparseCombination
  | .var column => some ⟨0, [⟨column, 1⟩]⟩
  | .const value => some ⟨value.val, []⟩
  | .add left right => (linear? left).bind fun leftCombination =>
      (linear? right).map (plus leftCombination)
  | .mul left right =>
      match constant? left, constant? right with
      | some factor, _ => (linear? right).map (scale factor)
      | none, some factor => (linear? left).map (scale factor)
      | none, none => none

theorem eval_scale (factor : F) (combination : SparseCombination) (env : Env) :
    (scale factor combination).eval env = factor * combination.eval env := by
  rcases combination with ⟨constant, terms⟩
  simp only [scale, SparseCombination.eval, fieldValue_val, List.map_map, Function.comp_def]
  rw [mul_add]
  congr 1
  induction terms with
  | nil => simp
  | cons term rest ih =>
      simp only [List.map_cons, List.sum_cons] at ih ⊢
      rw [ih, mul_add, mul_assoc]

theorem eval_plus (left right : SparseCombination) (env : Env) :
    (plus left right).eval env = left.eval env + right.eval env := by
  simp only [plus, SparseCombination.eval, fieldValue_val, List.map_append, List.sum_append]
  exact add_add_add_comm _ _ _ _

theorem constant?_eq_some {expression : Expr} {value : F}
    (h : constant? expression = some value) : expression = .const value := by
  cases expression <;> simp_all [constant?]

theorem linear?_eval (env : Env) (expression : Expr) :
    ∀ combination, linear? expression = some combination →
      combination.eval env = expression.eval env := by
  induction expression with
  | var column =>
      intro combination h
      simp only [linear?, Option.some.injEq] at h
      subst h
      simp [SparseCombination.eval]
  | const value =>
      intro combination h
      simp only [linear?, Option.some.injEq] at h
      subst h
      simp [SparseCombination.eval, fieldValue_val]
  | add left right ihLeft ihRight =>
      intro combination h
      simp only [linear?, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
      obtain ⟨leftCombination, leftEq, rightCombination, rightEq, rfl⟩ := h
      rw [eval_plus, ihLeft _ leftEq, ihRight _ rightEq]
      rfl
  | mul left right ihLeft ihRight =>
      intro combination h
      unfold linear? at h
      cases hl : constant? left with
      | some factor =>
          rw [hl] at h
          simp only [Option.map_eq_some_iff] at h
          obtain ⟨inner, innerEq, rfl⟩ := h
          rw [eval_scale, ihRight _ innerEq, constant?_eq_some hl]
          rfl
      | none =>
          cases hr : constant? right with
          | some factor =>
              rw [hl, hr] at h
              simp only [Option.map_eq_some_iff] at h
              obtain ⟨inner, innerEq, rfl⟩ := h
              rw [eval_scale, ihLeft _ innerEq, constant?_eq_some hr, Expr.eval_mul, mul_comm]
              rfl
          | none =>
              rw [hl, hr] at h
              simp at h

/-! ### Matching a block against a template -/

/-- First-order match of a template expression against a circuit expression. Template input
`j < inputs` binds to a circuit subterm; template local `inputs + k` must be column `start + k`. -/
def matchExpr (inputs start : Nat) :
    Expr → Expr → Array (Option Expr) → Option (Array (Option Expr))
  | .var column, target, binding =>
      if column < inputs then
        match binding[column]? with
        | some (some bound) => if bound == target then some binding else none
        | _ => some (binding.modify column fun _ => some target)
      else if target == .var (start + (column - inputs)) then some binding else none
  | .const value, .const other, binding => if value == other then some binding else none
  | .add left right, .add otherLeft otherRight, binding =>
      (matchExpr inputs start left otherLeft binding).bind (matchExpr inputs start right otherRight)
  | .mul left right, .mul otherLeft otherRight, binding =>
      (matchExpr inputs start left otherLeft binding).bind (matchExpr inputs start right otherRight)
  | _, _, _ => none

def matchAll (inputs start : Nat) :
    List Expr → List Expr → Array (Option Expr) → Option (Array (Option Expr))
  | [], _, binding => some binding
  | pattern :: patterns, target :: targets, binding =>
      (matchExpr inputs start pattern target binding).bind (matchAll inputs start patterns targets)
  | _ :: _, [], _ => none

/-- The column of the first local of a block: each template starts with `var inputs - _`. -/
def blockStart : Expr → Option Nat
  | .add (.var column) _ => some column
  | _ => none

def lanesOf (inputs : Nat) (binding : Array (Option Expr)) : List Expr :=
  (List.range inputs).map fun lane => (binding.getD lane none).getD 0

/-- Template input `j` becomes lane `j`; template local `inputs + k` becomes column `start + k`. -/
def laneSubst (inputs start : Nat) (lanes : List Expr) (column : Nat) : Expr :=
  if column < inputs then lanes.getD column 0 else .var (start + (column - inputs))

theorem laneSubst_input {inputs start column : Nat} (lanes : List Expr) (inside : column < inputs) :
    laneSubst inputs start lanes column = lanes.getD column 0 := by
  simp [laneSubst, inside]

theorem laneSubst_local (inputs start index : Nat) (lanes : List Expr) :
    laneSubst inputs start lanes (inputs + index) = .var (start + index) := by
  simp [laneSubst]

def LinearLanes (lanes : List Expr) : Prop :=
  lanes.all (fun lane => (linear? lane).isSome) = true

instance (lanes : List Expr) : Decidable (LinearLanes lanes) := by
  unfold LinearLanes
  infer_instance

/-- `some (start, lanes)` when `constraints` starts with `template` with inputs `lanes` and locals
from `start`, and every lane is linear. -/
def matchTemplate (inputs : Nat) (template constraints : List Expr) : Option (Nat × List Expr) :=
  (constraints.head?.bind blockStart).bind fun start =>
    (matchAll inputs start template constraints (Array.replicate inputs none)).bind fun binding =>
      let lanes := lanesOf inputs binding
      if (template.map (subst (laneSubst inputs start lanes))).isPrefixOf constraints ∧
          LinearLanes lanes then
        some (start, lanes)
      else none

theorem matchTemplate_spec {inputs : Nat} {template constraints : List Expr} {start : Nat}
    {lanes : List Expr} (h : matchTemplate inputs template constraints = some (start, lanes)) :
    template.map (subst (laneSubst inputs start lanes)) <+: constraints ∧ LinearLanes lanes := by
  unfold matchTemplate at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨found, _, binding, _, h⟩ := h
  split at h
  · rename_i checks
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨List.isPrefixOf_iff_prefix.mp checks.1, checks.2⟩
  · simp at h

/-! ### The Π_RLC ring-product template -/

/-- Template inputs of one `CombinationStep` block of `cells` ring products that share a
challenge: the challenge lanes, then the values, then the priors, each value and prior at
`lane * cells + cell`. -/
def ringInputs (cells : Nat) : Nat := ringDegree + 2 * (ringDegree * cells)

def ringValue (cells : Nat) (lane : Fin ringDegree) (cell : Nat) : Nat :=
  ringDegree + (lane.val * cells + cell)

def ringPrior (cells : Nat) (lane : Fin ringDegree) (cell : Nat) : Nat :=
  ringDegree + ringDegree * cells + (lane.val * cells + cell)

def ringLane {cells : Nat} (index : Fin (ringDegree * cells)) : Fin ringDegree :=
  ⟨index.val / cells, Nat.div_lt_of_lt_mul (lt_of_lt_of_eq index.isLt (Nat.mul_comm _ _))⟩

/-- Constraint `lane * cells + cell` of a block: the output minus the prior plus the challenge
times the value, as `CombinationStep.recipe` with `recipeConstraints`. -/
def ringConstraint (cells : Nat) (index : Fin (ringDegree * cells)) : Expr :=
  Expr.var (ringInputs cells + index.val) -
    (Expr.var (ringPrior cells (ringLane index) (index.val % cells)) +
      Lifecycle.PiRLC.v1_2.CombinationStep.mulExpr (fun source => .var source.val)
        (fun source => .var (ringValue cells source (index.val % cells))) (ringLane index))

def ringTemplate (cells : Nat) : List Expr := List.ofFn (ringConstraint cells)

@[simp] theorem ringTemplate_length (cells : Nat) :
    (ringTemplate cells).length = ringDegree * cells := by
  simp [ringTemplate]

/-- The one-cell template (commitment and public-input combinations), built once. -/
def ringTemplateOne : List Expr := ringTemplate 1

/-- The two-cell template (`RingKCombination`, the K-valued evaluation combinations). -/
def ringTemplateTwo : List Expr := ringTemplate 2

/-! ### Pieces -/

inductive Piece
  | constraint (expression : Expr)
  | permutation (start : Nat) (lanes : List Expr)
  | ringProduct (cells start : Nat) (lanes : List Expr)

def Piece.constraints (canonical : List Expr) : Piece → List Expr
  | .constraint expression => [expression]
  | .permutation start lanes => canonical.map (subst (laneSubst 16 start lanes))
  | .ringProduct cells start lanes =>
      (ringTemplate cells).map (subst (laneSubst (ringInputs cells) start lanes))

def Piece.Valid : Piece → Prop
  | .constraint _ => True
  | .permutation _ lanes => LinearLanes lanes
  | .ringProduct _ _ lanes => LinearLanes lanes

/-- The number of constraints of a piece, without building them. -/
def Piece.length (canonical : List Expr) : Piece → Nat
  | .constraint _ => 1
  | .permutation .. => canonical.length
  | .ringProduct cells .. => ringDegree * cells

theorem Piece.length_eq (canonical : List Expr) (piece : Piece) :
    (piece.constraints canonical).length = piece.length canonical := by
  cases piece <;> simp [Piece.constraints, Piece.length]

/-- The block piece at the head of `constraints`: a permutation, or a ring-product block of one
or two cells. -/
def recognize (canonical constraints : List Expr) : Option Piece :=
  match matchTemplate 16 canonical constraints with
  | some (start, lanes) => some (.permutation start lanes)
  | none =>
      match matchTemplate (ringInputs 1) ringTemplateOne constraints with
      | some (start, lanes) => some (.ringProduct 1 start lanes)
      | none =>
          match matchTemplate (ringInputs 2) ringTemplateTwo constraints with
          | some (start, lanes) => some (.ringProduct 2 start lanes)
          | none => none

theorem recognize_spec {canonical constraints : List Expr} {piece : Piece}
    (h : recognize canonical constraints = some piece) :
    piece.constraints canonical <+: constraints ∧ piece.Valid := by
  unfold recognize at h
  split at h
  · rename_i start lanes found
    cases h
    exact matchTemplate_spec found
  · split at h
    · rename_i start lanes found
      cases h
      exact matchTemplate_spec found
    · split at h
      · rename_i start lanes found
        cases h
        exact matchTemplate_spec found
      · cases h

/-- Splits constraints into pieces, tail recursive with `pieces` reversed. Each step uses one
unit of fuel and consumes at least one constraint while `canonical` is not empty. -/
def splitAux (canonical : List Expr) : Nat → List Expr → List Piece → List Piece
  | 0, constraints, pieces => pieces.reverse ++ constraints.map .constraint
  | _ + 1, [], pieces => pieces.reverse
  | fuel + 1, constraint :: rest, pieces =>
      match recognize canonical (constraint :: rest) with
      | some piece =>
          splitAux canonical fuel ((constraint :: rest).drop (piece.length canonical))
            (piece :: pieces)
      | none => splitAux canonical fuel rest (.constraint constraint :: pieces)

def split (canonical constraints : List Expr) : List Piece :=
  splitAux canonical constraints.length constraints []

theorem flatMap_splitAux (canonical : List Expr) :
    ∀ (fuel : Nat) (constraints : List Expr) (pieces : List Piece),
      (splitAux canonical fuel constraints pieces).flatMap (Piece.constraints canonical) =
        pieces.reverse.flatMap (Piece.constraints canonical) ++ constraints := by
  intro fuel
  induction fuel with
  | zero =>
      intro constraints pieces
      simp only [splitAux, List.flatMap_append, List.flatMap_map]
      congr 1
      induction constraints with
      | nil => rfl
      | cons constraint rest ih => simp [List.flatMap_cons, Piece.constraints]
  | succ fuel ih =>
      intro constraints pieces
      cases constraints with
      | nil => simp [splitAux]
      | cons constraint rest =>
          simp only [splitAux]
          split
          · rename_i piece found
            rw [ih]
            have whole : piece.constraints canonical =
                (constraint :: rest).take (piece.length canonical) := by
              rw [← Piece.length_eq]
              exact List.prefix_iff_eq_take.mp (recognize_spec found).1
            simp only [List.reverse_cons, List.flatMap_append, List.flatMap_cons,
              List.flatMap_nil, List.append_nil, List.append_assoc]
            rw [whole, List.take_append_drop]
          · rw [ih]
            simp [List.reverse_cons, List.flatMap_append, Piece.constraints]

theorem flatMap_split (canonical constraints : List Expr) :
    (split canonical constraints).flatMap (Piece.constraints canonical) = constraints := by
  simp [split, flatMap_splitAux]

theorem valid_splitAux (canonical : List Expr) :
    ∀ (fuel : Nat) (constraints : List Expr) (pieces : List Piece),
      (∀ piece ∈ pieces, piece.Valid) →
        ∀ piece ∈ splitAux canonical fuel constraints pieces, piece.Valid := by
  intro fuel
  induction fuel with
  | zero =>
      intro constraints pieces valid piece member
      simp only [splitAux, List.mem_append, List.mem_reverse, List.mem_map] at member
      rcases member with member | ⟨_, _, rfl⟩
      · exact valid piece member
      · trivial
  | succ fuel ih =>
      intro constraints pieces valid
      cases constraints with
      | nil =>
          intro piece member
          simp only [splitAux, List.mem_reverse] at member
          exact valid piece member
      | cons constraint rest =>
          simp only [splitAux]
          split
          · rename_i found _ found'
            apply ih
            intro piece member
            simp only [List.mem_cons] at member
            rcases member with rfl | member
            · exact (recognize_spec found').2
            · exact valid piece member
          · apply ih
            intro piece member
            simp only [List.mem_cons] at member
            rcases member with rfl | member
            · trivial
            · exact valid piece member

theorem valid_split (canonical constraints : List Expr) :
    ∀ piece ∈ split canonical constraints, piece.Valid :=
  valid_splitAux canonical _ constraints [] (by simp)

/-! ### Soundness -/

/-- The invocation of the permutation template that a permutation piece becomes. -/
def invocation (phase rowStart start : Nat) (lanes : List Expr) : PermutationInvocation :=
  ⟨phase, rowStart, start, lanes.map fun lane => (linear? lane).getD zeroSparseCombination⟩

theorem lane_eval (lanes : List Expr) (linear : LinearLanes lanes) (env : Env)
    (column : Nat) :
    (lanes.getD column 0).eval env =
      ((lanes.map fun lane => (linear? lane).getD zeroSparseCombination).getD column
        zeroSparseCombination).toR1CS.eval env := by
  rw [toR1CS_eval]
  by_cases inside : column < lanes.length
  · rw [List.getD_eq_getElem _ _ inside,
      List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map]
    have isSome := List.all_eq_true.mp linear (lanes[column]) (List.getElem_mem inside)
    obtain ⟨combination, found⟩ := Option.isSome_iff_exists.mp isSome
    rw [found, Option.getD_some, linear?_eval env _ combination found]
  · rw [List.getD_eq_default _ _ (by omega),
      List.getD_eq_default _ _ (by simpa using inside)]
    simp [SparseCombination.eval, zeroSparseCombination]

/-- The rows of a valid permutation piece's invocation give its constraints. -/
theorem permutation_sound (phase rowStart start : Nat) (lanes : List Expr)
    (valid : (Piece.permutation start lanes).Valid) (env : Env)
    (rows : PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation phase rowStart start lanes) env) :
    ConstraintsHold env
      ((Piece.permutation start lanes).constraints (PilotData.canonicalConstraints ())) := by
  have canonical := Pilot.canonicalPermutationInvocation_implies_constraints _ env rows
  have agree : (fun column => (laneSubst 16 start lanes column).eval env) =
      Pilot.canonicalInvocationEnv (invocation phase rowStart start lanes) env := by
    funext column
    by_cases small : column < 16
    · have input := Pilot.canonicalInvocationEnv_input
        (invocation phase rowStart start lanes) env ⟨column, small⟩
      simp only at input
      rw [input]
      simp only [laneSubst, small, ite_true]
      exact lane_eval lanes valid env column
    · obtain ⟨index, rfl⟩ : ∃ index, column = 16 + index := ⟨column - 16, by omega⟩
      rw [Pilot.canonicalInvocationEnv_local]
      simp [laneSubst, invocation]
  intro expression member
  simp only [Piece.constraints, List.mem_map] at member
  obtain ⟨pattern, patternMember, rfl⟩ := member
  rw [Lifecycle.Nebula.Expr.eval_subst, agree]
  exact canonical pattern patternMember

def genericConstraints : List Piece → List Expr
  | [] => []
  | .constraint expression :: rest => expression :: genericConstraints rest
  | .permutation _ _ :: rest => genericConstraints rest
  | .ringProduct _ _ _ :: rest => genericConstraints rest

theorem mem_genericConstraints {expression : Expr} :
    ∀ {pieces : List Piece}, Piece.constraint expression ∈ pieces →
      expression ∈ genericConstraints pieces
  | [], member => by simp at member
  | .constraint other :: rest, member => by
      simp only [List.mem_cons, Piece.constraint.injEq] at member
      show expression ∈ other :: genericConstraints rest
      rcases member with rfl | member
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (mem_genericConstraints member)
  | .permutation _ _ :: rest, member => by
      simp only [List.mem_cons, reduceCtorEq, false_or] at member
      show expression ∈ genericConstraints rest
      exact mem_genericConstraints member
  | .ringProduct _ _ _ :: rest, member => by
      simp only [List.mem_cons, reduceCtorEq, false_or] at member
      show expression ∈ genericConstraints rest
      exact mem_genericConstraints member

/-- The constraints hold when every block piece's constraints hold and the other constraints
hold. -/
theorem split_sound (canonical constraints : List Expr) (env : Env)
    (permutations : ∀ start lanes,
      Piece.permutation start lanes ∈ split canonical constraints →
      ConstraintsHold env ((Piece.permutation start lanes).constraints canonical))
    (ringProducts : ∀ cells start lanes,
      Piece.ringProduct cells start lanes ∈ split canonical constraints →
      ConstraintsHold env ((Piece.ringProduct cells start lanes).constraints canonical))
    (generic : ConstraintsHold env (genericConstraints (split canonical constraints))) :
    ConstraintsHold env constraints := by
  intro expression member
  rw [← flatMap_split canonical constraints, List.mem_flatMap] at member
  obtain ⟨piece, pieceMember, inPiece⟩ := member
  cases piece with
  | constraint other =>
      simp only [Piece.constraints, List.mem_singleton] at inPiece
      subst inPiece
      exact generic expression (mem_genericConstraints pieceMember)
  | permutation start lanes =>
      exact permutations start lanes pieceMember expression inPiece
  | ringProduct cells start lanes =>
      exact ringProducts cells start lanes pieceMember expression inPiece

end NightstreamFPrime.Export.Generic.Pieces
