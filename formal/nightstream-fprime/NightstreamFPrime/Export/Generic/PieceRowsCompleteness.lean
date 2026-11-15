import NightstreamFPrime.Export.Generic.PieceRows
import NightstreamFPrime.Layout.R1CS.Completeness

/-! Completeness of the lowered pieces (CW-10.2), the converse of `PieceRows.lowerPiecesAux_sound`.
An environment in which every piece's constraints hold gets the fresh columns of the lowering:
the lowered constraints' columns by `R1CS.executeConstraints`, and each ring product's quotients
by `QuotientProduct.quotient` (`executePieces`). The result agrees with the environment outside
the fresh columns, and every lowered row holds in it (`executePieces_spec`).

The proof needs every piece to read only columns below the first fresh column (`pieceBelow`).
That is a computed check, as `CircuitRelation.packable` is. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.PieceRowsCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Spec.RingRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-! ### The scope check -/

/-- `expression` reads only columns below `bound`, as a computed check. -/
def exprBelow (bound : Nat) : Expr → Bool
  | .var column => decide (column < bound)
  | .const _ => true
  | .add left right => exprBelow bound left && exprBelow bound right
  | .mul left right => exprBelow bound left && exprBelow bound right

theorem exprBelow_iff (bound : Nat) (expression : Expr) :
    exprBelow bound expression = true ↔ expression.VarsBelow bound := by
  induction expression with
  | var column => simp [exprBelow, Expr.VarsBelow]
  | const value => simp [exprBelow, Expr.VarsBelow]
  | add left right leftIH rightIH =>
      simp only [exprBelow, Bool.and_eq_true, leftIH, rightIH, Expr.VarsBelow]
  | mul left right leftIH rightIH =>
      simp only [exprBelow, Bool.and_eq_true, leftIH, rightIH, Expr.VarsBelow]

/-- A piece reads only columns below `bound`: a constraint by its variables, a block or a ring
product by its lanes and its locals. -/
def pieceBelow (bound : Nat) : Piece → Bool
  | .constraint expression => exprBelow bound expression
  | .permutation start lanes => lanes.all (exprBelow bound) && decide (start + 1096 ≤ bound)
  | .ringProduct cells start lanes =>
      lanes.all (exprBelow bound) && decide (start + ringDegree * cells ≤ bound)

theorem lanes_below {bound : Nat} {lanes : List Expr} (below : lanes.all (exprBelow bound) = true)
    (column : Nat) : (lanes.getD column 0).VarsBelow bound := by
  by_cases inside : column < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact (exprBelow_iff bound _).mp (List.all_eq_true.mp below _ (List.getElem_mem inside))
  · rw [List.getD_eq_default _ _ (by omega)]
    trivial

/-! ### Ring products -/

/-- The challenge of a ring product, from its first `ringDegree` lanes. -/
def challengeOf (env : Env) (lanes : List Expr) : RingF :=
  fun lane => (lanes.getD lane.val 0).eval env

def valueOf (env : Env) (lanes : List Expr) (cells cell : Nat) : RingF :=
  fun lane => (lanes.getD (ringValue cells lane cell) 0).eval env

def priorOf (env : Env) (lanes : List Expr) (cells cell : Nat) : RingF :=
  fun lane => (lanes.getD (ringPrior cells lane cell) 0).eval env

def outputOf (env : Env) (cells start cell : Nat) : RingF :=
  fun lane => env (outputColumn cells start lane cell)

/-- The constraints of a ring product give each cell's output as the prior plus the challenge
times the value. -/
theorem output_of_constraints (canonical : List Expr) {cells start : Nat} {lanes : List Expr}
    {env : Env}
    (holds : ConstraintsHold env ((Piece.ringProduct cells start lanes).constraints canonical))
    {cell : Nat} (inside : cell < cells) :
    outputOf env cells start cell =
      ringFAdd (priorOf env lanes cells cell)
        (ringFMul (challengeOf env lanes) (valueOf env lanes cells cell)) := by
  funext lane
  let index : Fin (ringDegree * cells) := ⟨lane.val * cells + cell, slot_lt lane inside⟩
  have laneEq : ringLane index = lane := by
    apply Fin.ext
    simp only [ringLane, index]
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt inside, zero_add]
  have cellEq : index.val % cells = cell := by
    simp only [index]
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt inside]
  let lanesEnv : Env := fun column => (laneSubst (ringInputs cells) start lanes column).eval env
  have input : ∀ column, column < ringInputs cells →
      lanesEnv column = (lanes.getD column 0).eval env := by
    intro column below
    simp only [lanesEnv, laneSubst_input lanes below]
  have challenge : CombinationStep.evalRing lanesEnv (fun source => Expr.var source.val) =
      challengeOf env lanes := by
    funext current
    have := current.isLt
    exact input _ (by simp only [ringInputs]; omega)
  have value : CombinationStep.evalRing lanesEnv
        (fun source => Expr.var (ringValue cells source (index.val % cells))) =
      valueOf env lanes cells cell := by
    funext current
    have := slot_lt current inside
    rw [cellEq]
    exact input _ (by simp only [ringValue, ringInputs]; omega)
  have prior : lanesEnv (ringPrior cells (ringLane index) (index.val % cells)) =
      priorOf env lanes cells cell lane := by
    have := slot_lt lane inside
    rw [laneEq, cellEq]
    exact input _ (by simp only [ringPrior, ringInputs]; omega)
  have output : lanesEnv (ringInputs cells + index.val) =
      env (outputColumn cells start (ringLane index) (index.val % cells)) := by
    simp only [lanesEnv, laneSubst_local, Expr.eval_var, outputColumn, ringLane,
      Nat.div_add_mod']
  have constraint := holds _ (List.mem_map.mpr ⟨_, List.mem_ofFn.mpr ⟨index, rfl⟩, rfl⟩)
  rw [Lifecycle.Nebula.Expr.eval_subst] at constraint
  change (ringConstraint cells index).eval lanesEnv = 0 at constraint
  simp only [ringConstraint, Expr.eval_sub, Expr.eval_hadd, Expr.eval_var,
    CombinationStep.mulExpr_eval] at constraint
  rw [challenge, value, prior, output, laneEq, cellEq] at constraint
  simp only [outputOf, ringFAdd]
  exact sub_eq_zero.mp constraint

/-- The quotient columns of a ring product from `quotient`: `QuotientProduct.quotient` of each
cell's challenge and value. -/
noncomputable def fillQuotients (env : Env) (cells quotient : Nat) (lanes : List Expr) : Env :=
  fun column =>
    if inside : quotient ≤ column ∧ column < quotient + ringDegree * cells then
      QuotientProduct.quotient (challengeOf env lanes)
        (valueOf env lanes cells ((column - quotient) / ringDegree))
        ⟨(column - quotient) % ringDegree, Nat.mod_lt _ (by decide)⟩
    else env column

theorem fillQuotients_outside {env : Env} {cells quotient : Nat} {lanes : List Expr} {column : Nat}
    (outside : column < quotient ∨ quotient + ringDegree * cells ≤ column) :
    fillQuotients env cells quotient lanes column = env column := by
  unfold fillQuotients
  rw [dite_eq_right (by omega)]

theorem fillQuotients_quotient {env : Env} {cells quotient cell : Nat} {lanes : List Expr}
    (inside : cell < cells) (lane : Fin ringDegree) :
    fillQuotients env cells quotient lanes (quotientColumn quotient lane cell) =
      QuotientProduct.quotient (challengeOf env lanes) (valueOf env lanes cells cell) lane := by
  have room : cell * ringDegree + lane.val < ringDegree * cells := by
    have := lane.isLt
    have : (cell + 1) * ringDegree ≤ cells * ringDegree := Nat.mul_le_mul_right _ inside
    rw [Nat.succ_mul] at this
    rw [Nat.mul_comm ringDegree cells]
    omega
  unfold fillQuotients quotientColumn
  rw [dite_eq_left (by omega)]
  have offset : quotient + (cell * ringDegree + lane.val) - quotient = cell * ringDegree + lane.val :=
    by omega
  have cellEq : (cell * ringDegree + lane.val) / ringDegree = cell := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide), Nat.div_eq_of_lt lane.isLt, zero_add]
  have laneEq : (cell * ringDegree + lane.val) % ringDegree = lane.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt lane.isLt]
  simp only [offset, cellEq, laneEq]

/-- The rows of a ring product hold in any environment that keeps the piece's lanes and outputs
from `env` and holds the quotients of `env`. -/
theorem ringRows_complete {cells start quotient : Nat} {lanes : List Expr}
    (linear : LinearLanes lanes) {env after : Env}
    (outputs : ∀ cell < cells,
      outputOf env cells start cell =
        ringFAdd (priorOf env lanes cells cell)
          (ringFMul (challengeOf env lanes) (valueOf env lanes cells cell)))
    (lanesKept : ∀ column, (lanes.getD column 0).eval after = (lanes.getD column 0).eval env)
    (outputsKept : ∀ cell < cells, ∀ lane : Fin ringDegree,
      after (outputColumn cells start lane cell) = env (outputColumn cells start lane cell))
    (quotients : ∀ cell < cells, ∀ lane : Fin ringDegree,
      after (quotientColumn quotient lane cell) =
        QuotientProduct.quotient (challengeOf env lanes) (valueOf env lanes cells cell) lane) :
    R1CS.RowsHold after (ringRows cells start quotient lanes) := by
  intro row member
  obtain ⟨cell, cellMember, nodeMember⟩ := List.mem_flatMap.mp member
  obtain ⟨node, rfl⟩ := List.mem_ofFn.mp nodeMember
  have inside := List.mem_range.mp cellMember
  have challenge : (fun lane : Fin ringDegree => (lanes.getD lane.val 0).eval after) =
      challengeOf env lanes := funext fun lane => lanesKept _
  have value : (fun lane : Fin ringDegree => (lanes.getD (ringValue cells lane cell) 0).eval after) =
      valueOf env lanes cells cell := funext fun lane => lanesKept _
  have prior : (fun lane : Fin ringDegree => (lanes.getD (ringPrior cells lane cell) 0).eval after) =
      priorOf env lanes cells cell := funext fun lane => lanesKept _
  have output : (fun lane : Fin ringDegree => after (outputColumn cells start lane cell)) =
      ringFAdd (priorOf env lanes cells cell)
        (ringFMul (challengeOf env lanes) (valueOf env lanes cells cell)) :=
    (funext fun lane => outputsKept cell inside lane).trans (outputs cell inside)
  have quotientEq : (fun lane : Fin ringDegree => after (quotientColumn quotient lane cell)) =
      QuotientProduct.quotient (challengeOf env lanes) (valueOf env lanes cells cell) :=
    funext fun lane => quotients cell inside lane
  simp only [R1CS.Row.Holds, ringRow, R1CS.LinearCombination.eval_add,
    R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_ofVar, evaluation_eval,
    laneCombination_eval linear]
  rw [challenge, value, prior, output, quotientEq,
    QuotientProduct.complete_add (challengeOf env lanes) (valueOf env lanes cells cell)
      (priorOf env lanes cells cell) node, sub_eq_add_neg, neg_one_mul, add_assoc]

/-! ### Lowering a list of pieces -/

/-- The fresh columns of one piece's lowering, from `next`. -/
noncomputable def executePiece (canonical : List Expr) (env : Env) (piece : Piece) (next : Nat) :
    Env :=
  match piece with
  | .ringProduct cells _ lanes => fillQuotients env cells next lanes
  | piece => R1CS.executeConstraints env (piece.constraints canonical) next

/-- The fresh columns of the pieces' lowering, piece by piece. -/
noncomputable def executePieces (canonical : List Expr) : Env → List Piece → Nat → Env
  | env, [], _ => env
  | env, piece :: rest, next =>
      executePieces canonical (executePiece canonical env piece next) rest
        (lowerPiece canonical piece next).2

/-- The lowering in order, without the reversed accumulator of `lowerPiecesAux`. -/
def lowerList (canonical : List Expr) : List Piece → Nat → List R1CS.Row × Nat
  | [], next => ([], next)
  | piece :: rest, next =>
      ((lowerPiece canonical piece next).1 ++
          (lowerList canonical rest (lowerPiece canonical piece next).2).1,
        (lowerList canonical rest (lowerPiece canonical piece next).2).2)

theorem lowerPiecesAux_eq (canonical : List Expr) :
    ∀ (pieces : List Piece) (next : Nat) (reversed : List R1CS.Row),
      lowerPiecesAux canonical pieces next reversed =
        (reversed.reverse ++ (lowerList canonical pieces next).1,
          (lowerList canonical pieces next).2)
  | [], next, reversed => by simp [lowerPiecesAux, lowerList]
  | piece :: rest, next, reversed => by
      simp only [lowerPiecesAux, lowerList]
      rw [lowerPiecesAux_eq canonical rest]
      simp [List.reverseAux_eq]

theorem lowerPieces_eq (canonical : List Expr) (pieces : List Piece) (next : Nat) :
    lowerPieces canonical pieces next = lowerList canonical pieces next := by
  simp [lowerPieces, lowerPiecesAux_eq]

theorem lowerPiece_next_le (canonical : List Expr) (piece : Piece) (next : Nat) :
    next ≤ (lowerPiece canonical piece next).2 := by
  cases piece <;> simp [lowerPiece]

theorem lowerList_next_le (canonical : List Expr) :
    ∀ (pieces : List Piece) (next : Nat), next ≤ (lowerList canonical pieces next).2
  | [], _ => le_refl _
  | piece :: rest, next =>
      le_trans (lowerPiece_next_le canonical piece next)
        (lowerList_next_le canonical rest _)

theorem executePiece_agrees (canonical : List Expr) (env : Env) (piece : Piece) (next column : Nat)
    (outside : column < next ∨ (lowerPiece canonical piece next).2 ≤ column) :
    executePiece canonical env piece next column = env column := by
  cases piece with
  | ringProduct cells start lanes => exact fillQuotients_outside outside
  | constraint expression =>
      exact R1CS.executeConstraints_agreesOutside env _ next column
        (by simpa [lowerPiece] using outside)
  | permutation start lanes =>
      exact R1CS.executeConstraints_agreesOutside env _ next column
        (by simpa [lowerPiece] using outside)

/-- The lowering writes no column below its first fresh column. -/
theorem executePieces_below (canonical : List Expr) :
    ∀ (env : Env) (pieces : List Piece) (next column : Nat), column < next →
      executePieces canonical env pieces next column = env column
  | _, [], _, _, _ => rfl
  | env, piece :: rest, next, column, below => by
      simp only [executePieces]
      rw [executePieces_below canonical _ rest _ column
        (lt_of_lt_of_le below (lowerPiece_next_le canonical piece next))]
      exact executePiece_agrees canonical env piece next column (Or.inl below)

/-- What a piece needs for its lowering to be complete: it is valid, reads only columns below
`bound`, is not a Poseidon2 block, and its constraints hold in `base`. -/
structure Ready (canonical : List Expr) (bound : Nat) (base : Env) (piece : Piece) : Prop where
  valid : piece.Valid
  below : pieceBelow bound piece = true
  notBlock : ∀ start lanes, piece ≠ .permutation start lanes
  holds : ConstraintsHold base (piece.constraints canonical)

/-- The lowering's fresh columns complete the pieces: the result agrees with `env` outside the
fresh columns, and every lowered row holds in each environment that agrees with it below the last
fresh column. -/
theorem executePieces_spec (canonical : List Expr) {bound : Nat} {base : Env} :
    ∀ (pieces : List Piece) (env : Env) (next : Nat),
      (∀ piece ∈ pieces, Ready canonical bound base piece) → bound ≤ next →
      (∀ column, column < bound → env column = base column) →
      (∀ column, column < next ∨ (lowerList canonical pieces next).2 ≤ column →
          executePieces canonical env pieces next column = env column) ∧
        ∀ target : Env, (∀ column, column < (lowerList canonical pieces next).2 →
            target column = executePieces canonical env pieces next column) →
          R1CS.RowsHold target (lowerList canonical pieces next).1
  | [], env, next, _, _, _ => ⟨fun _ _ => rfl, fun _ _ _ member => by simp [lowerList] at member⟩
  | piece :: rest, env, next, ready, boundLe, agree => by
      have current := ready piece List.mem_cons_self
      let env₁ := executePiece canonical env piece next
      let next₁ := (lowerPiece canonical piece next).2
      have nextLe := lowerPiece_next_le canonical piece next
      have firstAgree : ∀ column, column < next ∨ next₁ ≤ column → env₁ column = env column :=
        executePiece_agrees canonical env piece next
      have agree₁ : ∀ column, column < bound → env₁ column = base column := fun column below =>
        (firstAgree column (Or.inl (by omega))).trans (agree column below)
      obtain ⟨tailAgree, tailRows⟩ := executePieces_spec canonical rest env₁ next₁
        (fun other member => ready other (List.mem_cons_of_mem _ member)) (by omega) agree₁
      have endLe := lowerList_next_le canonical rest next₁
      let after := executePieces canonical env₁ rest next₁
      have afterAgree : ∀ column, column < next₁ ∨ (lowerList canonical rest next₁).2 ≤ column →
          after column = env₁ column := tailAgree
      refine ⟨fun column outside => ?_, fun target same => ?_⟩
      · change after column = env column
        rcases outside with below | beyond
        · rw [afterAgree column (Or.inl (by omega))]
          exact firstAgree column (Or.inl below)
        · rw [afterAgree column (Or.inr beyond)]
          exact firstAgree column (Or.inr (le_trans endLe beyond))
      · change ∀ column, column < (lowerList canonical rest next₁).2 → target column = after column
          at same
        have targetFirst : ∀ column, column < next₁ → target column = env₁ column :=
          fun column below => (same column (by omega)).trans (afterAgree column (Or.inl below))
        have targetBase : ∀ column, column < bound → target column = base column :=
          fun column below => (targetFirst column (by omega)).trans (agree₁ column below)
        change R1CS.RowsHold target
          ((lowerPiece canonical piece next).1 ++ (lowerList canonical rest next₁).1)
        rw [R1CS.rowsHold_append]
        refine ⟨?_, tailRows target same⟩
        cases piece with
        | permutation start lanes => exact absurd rfl (current.notBlock start lanes)
        | constraint expression =>
            have scope : expression.VarsBelow bound := (exprBelow_iff bound _).mp current.below
            have scopeNext : ∀ other ∈ [expression], other.VarsBelow next := by
              intro other member
              rw [List.mem_singleton.mp member]
              exact Expr.VarsBelow.mono expression scope boundLe
            have logical : ConstraintsHold env [expression] :=
              constraintsHold_of_agree_below base env [expression] bound
                (fun other member => by rw [List.mem_singleton.mp member]; exact scope)
                agree current.holds
            have rows := R1CS.executeConstraints_holds_rows env [expression] next scopeNext logical
            have rowsBelow := R1CS.lowerConstraints_rows_varsBelow [expression] next scopeNext
            have nextEq : next₁ = next + R1CS.totalFreshCount [expression] := by
              simp [next₁, lowerPiece, Piece.constraints]
            exact R1CS.rowsHold_of_agree_below _ next₁ env₁ target
              (by rw [nextEq]; exact rowsBelow) targetFirst rows
        | ringProduct cells start lanes =>
            have below := current.below
            simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at below
            obtain ⟨lanesBelow, startBelow⟩ := below
            have valid : LinearLanes lanes := current.valid
            have holds := current.holds
            have lanesBase : ∀ (other : Env), (∀ column, column < bound → other column = base column) →
                ∀ column, (lanes.getD column 0).eval other = (lanes.getD column 0).eval base :=
              fun other agreeing column =>
                Expr.eval_eq_of_agree_below _ bound other base (lanes_below lanesBelow column)
                  agreeing
            have challengeEq : challengeOf env lanes = challengeOf base lanes :=
              funext fun lane => lanesBase env agree _
            have valueEq : ∀ cell, valueOf env lanes cells cell = valueOf base lanes cells cell :=
              fun cell => funext fun lane => lanesBase env agree _
            have outputBelow : ∀ cell < cells, ∀ lane : Fin ringDegree,
                outputColumn cells start lane cell < bound := by
              intro cell inside lane
              have := slot_lt lane inside
              simp only [outputColumn]
              omega
            apply ringRows_complete valid (env := base)
              (fun cell inside => output_of_constraints canonical holds inside)
              (lanesBase target targetBase)
              (fun cell inside lane => targetBase _ (outputBelow cell inside lane))
            intro cell inside lane
            have room : cell * ringDegree + lane.val < ringDegree * cells := by
              have := lane.isLt
              have : (cell + 1) * ringDegree ≤ cells * ringDegree := Nat.mul_le_mul_right _ inside
              rw [Nat.succ_mul] at this
              rw [Nat.mul_comm ringDegree cells]
              omega
            rw [targetFirst _ (by simp only [next₁, lowerPiece, quotientColumn]; omega)]
            change fillQuotients env cells next lanes (quotientColumn next lane cell) = _
            rw [fillQuotients_quotient inside, challengeEq, valueEq]

end NightstreamFPrime.Export.Generic.PieceRowsCompleteness
