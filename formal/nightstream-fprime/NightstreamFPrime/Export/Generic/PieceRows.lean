import NightstreamFPrime.Export.Generic.Pieces
import NightstreamFPrime.Spec.Phi81Relation.QuotientProduct

/-! The R1CS rows of a circuit's pieces (CW-5.7). A ring-product piece is one `CombinationStep`
block of the Π_RLC combinations: for each cell, `output = prior + challenge * value` in the ring.
It becomes 108 rows per cell, row `r` checking at node `r`

  `challenge(r) * value(r) = output(r) - prior(r) + Φ81(r) * quotient(r)`,

where each term is the evaluation of a ring element and the quotient is fresh, as
`Phi81ProductPlan` (SuperNeo v1.2 Section 7.4). `QuotientProduct.sound_add` turns the rows back
into the ring identity, which is the piece's constraints (`ringRows_sound`). A permutation piece
and every other constraint are lowered with `R1CS.lowerConstraints`. `pieces_sound` gives the
circuit's constraints from the rows of `split`. -/

namespace NightstreamFPrime.Export.Generic.PieceRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-! ### Linear lanes as R1CS combinations -/

/-- A lane as an R1CS combination; zero when the lane is not linear. -/
def laneCombination (lane : Expr) : R1CS.LinearCombination :=
  ((linear? lane).getD zeroSparseCombination).toR1CS

theorem linear_getD {lanes : List Expr} (linear : LinearLanes lanes) (column : Nat) :
    (linear? (lanes.getD column 0)).isSome := by
  by_cases inside : column < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact List.all_eq_true.mp linear _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

theorem laneCombination_eval {lanes : List Expr} (linear : LinearLanes lanes) (column : Nat)
    (env : Env) :
    (laneCombination (lanes.getD column 0)).eval env = (lanes.getD column 0).eval env := by
  obtain ⟨combination, found⟩ := Option.isSome_iff_exists.mp (linear_getD linear column)
  rw [laneCombination, found, Option.getD_some, toR1CS_eval, linear?_eval env _ combination found]

/-! ### Evaluations of ring elements -/

def sumCombinations : List R1CS.LinearCombination → R1CS.LinearCombination
  | [] => .zero
  | combination :: rest => .add combination (sumCombinations rest)

theorem sumCombinations_eval (env : Env) :
    ∀ combinations : List R1CS.LinearCombination,
      (sumCombinations combinations).eval env = (combinations.map fun c => c.eval env).sum
  | [] => by simp [sumCombinations]
  | combination :: rest => by
      simp [sumCombinations, sumCombinations_eval env rest]

/-- `Σ_lane point^lane · lanes(lane)`: the evaluation at `point` of the ring element whose lanes
are `lanes`. -/
def evaluation (point : F) (lanes : Fin ringDegree → R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  sumCombinations (List.ofFn fun lane => .scale (point ^ lane.val) (lanes lane))

theorem evaluation_eval (env : Env) (point : F)
    (lanes : Fin ringDegree → R1CS.LinearCombination) :
    (evaluation point lanes).eval env =
      QuotientProduct.evaluate (fun lane => (lanes lane).eval env) point := by
  simp only [evaluation, sumCombinations_eval, List.map_ofFn, Function.comp_def,
    R1CS.LinearCombination.eval_scale, QuotientProduct.evaluate, mul_comm]

/-! ### The rows of a ring-product piece -/

/-- The column of output `lane` of `cell` in a block of `cells` ring products from `start`, as
`CombinationStep.indexOf` within one block. -/
def outputColumn (cells start : Nat) (lane : Fin ringDegree) (cell : Nat) : Nat :=
  start + (lane.val * cells + cell)

/-- The fresh column of quotient `lane` of `cell`, from `quotient`. -/
def quotientColumn (quotient : Nat) (lane : Fin ringDegree) (cell : Nat) : Nat :=
  quotient + (cell * ringDegree + lane.val)

/-- The row of `cell` at `node`: `challenge * value = output - prior + Φ81 * quotient`, each side
evaluated at the node. -/
def ringRow (cells start quotient : Nat) (lanes : List Expr) (cell : Nat) (node : Fin 108) :
    R1CS.Row :=
  let point := QuotientProduct.node node
  { a := evaluation point fun lane => laneCombination (lanes.getD lane.val 0)
    b := evaluation point fun lane => laneCombination (lanes.getD (ringValue cells lane cell) 0)
    c := .add (evaluation point fun lane => .ofVar (outputColumn cells start lane cell))
      (.add (.scale (-1) (evaluation point fun lane =>
          laneCombination (lanes.getD (ringPrior cells lane cell) 0)))
        (.scale (QuotientProduct.modulusValue point)
          (evaluation point fun lane => .ofVar (quotientColumn quotient lane cell)))) }

/-- 108 rows per cell, with the quotients of the block in `ringDegree * cells` fresh columns
from `quotient`. -/
def ringRows (cells start quotient : Nat) (lanes : List Expr) : List R1CS.Row :=
  (List.range cells).flatMap fun cell => List.ofFn (ringRow cells start quotient lanes cell)

theorem ringRow_mem {cells start quotient cell : Nat} {lanes : List Expr} (inside : cell < cells)
    (node : Fin 108) :
    ringRow cells start quotient lanes cell node ∈ ringRows cells start quotient lanes :=
  List.mem_flatMap.mpr ⟨cell, List.mem_range.mpr inside, List.mem_ofFn.mpr ⟨node, rfl⟩⟩

/-- The 108 equations, as the rows state them, give the ring identity. -/
theorem ring_of_equations (left right prior output quotient : RingF)
    (equations : ∀ index : Fin 108,
      QuotientProduct.evaluate left (QuotientProduct.node index) *
          QuotientProduct.evaluate right (QuotientProduct.node index) =
        QuotientProduct.evaluate output (QuotientProduct.node index) +
          (-1 * QuotientProduct.evaluate prior (QuotientProduct.node index) +
            QuotientProduct.modulusValue (QuotientProduct.node index) *
              QuotientProduct.evaluate quotient (QuotientProduct.node index))) :
    output = ringFAdd prior (ringFMul left right) := by
  apply QuotientProduct.sound_add left right prior output quotient
  intro index
  have equation : QuotientProduct.evaluate left (QuotientProduct.node index) *
        QuotientProduct.evaluate right (QuotientProduct.node index) =
      QuotientProduct.evaluate output (QuotientProduct.node index) -
          QuotientProduct.evaluate prior (QuotientProduct.node index) +
        QuotientProduct.modulusValue (QuotientProduct.node index) *
          QuotientProduct.evaluate quotient (QuotientProduct.node index) := by
    rw [equations index, sub_eq_add_neg, neg_one_mul, add_assoc]
  exact equation

/-- The rows of one cell give that cell's ring product. -/
theorem ringRows_product {cells start quotient cell : Nat} {lanes : List Expr}
    (linear : LinearLanes lanes) {env : Env}
    (rows : R1CS.RowsHold env (ringRows cells start quotient lanes)) (inside : cell < cells) :
    (fun lane : Fin ringDegree => env (outputColumn cells start lane cell)) =
      ringFAdd (fun lane => (lanes.getD (ringPrior cells lane cell) 0).eval env)
        (ringFMul (fun lane => (lanes.getD lane.val 0).eval env)
          (fun lane => (lanes.getD (ringValue cells lane cell) 0).eval env)) := by
  apply ring_of_equations _ _ _ _ fun lane => env (quotientColumn quotient lane cell)
  intro node
  have holds := rows _ (ringRow_mem inside node)
  simp only [R1CS.Row.Holds, ringRow, R1CS.LinearCombination.eval_add,
    R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_ofVar, evaluation_eval,
    laneCombination_eval linear] at holds
  exact holds

/-- A lane slot `lane * cells + cell` of a block lies below `ringDegree * cells`. -/
theorem slot_lt {cells cell : Nat} (lane : Fin ringDegree) (inside : cell < cells) :
    lane.val * cells + cell < ringDegree * cells :=
  calc lane.val * cells + cell < lane.val * cells + cells := by omega
    _ = (lane.val + 1) * cells := by rw [Nat.succ_mul]
    _ ≤ ringDegree * cells := Nat.mul_le_mul_right _ lane.isLt

/-- The rows of a valid ring-product piece give its constraints. -/
theorem ringRows_sound (canonical : List Expr) (cells start quotient : Nat) (lanes : List Expr)
    (linear : LinearLanes lanes) (env : Env)
    (rows : R1CS.RowsHold env (ringRows cells start quotient lanes)) :
    ConstraintsHold env ((Piece.ringProduct cells start lanes).constraints canonical) := by
  intro expression member
  simp only [Piece.constraints, ringTemplate, List.mem_map, List.mem_ofFn] at member
  obtain ⟨_, ⟨index, rfl⟩, rfl⟩ := member
  have positive : 0 < cells := by
    have below := index.isLt
    rcases Nat.eq_zero_or_pos cells with zero | positive
    · simp [zero] at below
    · exact positive
  have cellLt := Nat.mod_lt index.val positive
  let lanesEnv : Env := fun column => (laneSubst (ringInputs cells) start lanes column).eval env
  have input : ∀ column, column < ringInputs cells →
      lanesEnv column = (lanes.getD column 0).eval env := by
    intro column below
    simp only [lanesEnv, laneSubst_input lanes below]
  have challenge : CombinationStep.evalRing lanesEnv (fun source => Expr.var source.val) =
      fun lane => (lanes.getD lane.val 0).eval env := by
    funext lane
    have := lane.isLt
    exact input _ (by simp only [ringInputs]; omega)
  have value : CombinationStep.evalRing lanesEnv
        (fun source => Expr.var (ringValue cells source (index.val % cells))) =
      fun lane => (lanes.getD (ringValue cells lane (index.val % cells)) 0).eval env := by
    funext lane
    have := slot_lt lane cellLt
    exact input _ (by simp only [ringValue, ringInputs]; omega)
  have prior : lanesEnv (ringPrior cells (ringLane index) (index.val % cells)) =
      (lanes.getD (ringPrior cells (ringLane index) (index.val % cells)) 0).eval env := by
    have := slot_lt (ringLane index) cellLt
    exact input _ (by simp only [ringPrior, ringInputs]; omega)
  have output : lanesEnv (ringInputs cells + index.val) =
      env (outputColumn cells start (ringLane index) (index.val % cells)) := by
    simp only [lanesEnv, laneSubst_local, Expr.eval_var, outputColumn, ringLane,
      Nat.div_add_mod']
  have product := congrFun (ringRows_product linear rows cellLt) (ringLane index)
  rw [Lifecycle.Nebula.Expr.eval_subst]
  change (ringConstraint cells index).eval lanesEnv = 0
  simp only [ringConstraint, Expr.eval_sub, Expr.eval_hadd, Expr.eval_var,
    CombinationStep.mulExpr_eval]
  rw [challenge, value, prior, output, product]
  simp only [ringFAdd, sub_self]

/-! ### Lowering pieces -/

/-- A piece's rows and the next fresh column: ring rows for a ring-product piece, lowered
constraints otherwise. -/
def lowerPiece (canonical : List Expr) (piece : Piece) (next : Nat) : List R1CS.Row × Nat :=
  match piece with
  | .ringProduct cells start lanes => (ringRows cells start next lanes, next + ringDegree * cells)
  | piece =>
      let lowered := R1CS.lowerConstraints (piece.constraints canonical) next
      (lowered.rows, lowered.next)

theorem lowerPiece_sound (canonical : List Expr) (piece : Piece) (valid : piece.Valid)
    (next : Nat) (env : Env) (rows : R1CS.RowsHold env (lowerPiece canonical piece next).1) :
    ConstraintsHold env (piece.constraints canonical) := by
  cases piece with
  | ringProduct cells start lanes => exact ringRows_sound canonical cells start next lanes valid env rows
  | constraint expression => exact R1CS.lowerConstraints_sound env _ next rows
  | permutation start lanes => exact R1CS.lowerConstraints_sound env _ next rows

/-- Lowers pieces in order, tail recursive with the rows reversed. -/
def lowerPiecesAux (canonical : List Expr) :
    List Piece → Nat → List R1CS.Row → List R1CS.Row × Nat
  | [], next, rowsReversed => (rowsReversed.reverse, next)
  | piece :: rest, next, rowsReversed =>
      let lowered := lowerPiece canonical piece next
      lowerPiecesAux canonical rest lowered.2 (lowered.1.reverseAux rowsReversed)

def lowerPieces (canonical : List Expr) (pieces : List Piece) (next : Nat) :
    List R1CS.Row × Nat :=
  lowerPiecesAux canonical pieces next []

theorem lowerPiecesAux_sound (canonical : List Expr) (env : Env) :
    ∀ (pieces : List Piece) (next : Nat) (rowsReversed : List R1CS.Row),
      (∀ piece ∈ pieces, piece.Valid) →
      R1CS.RowsHold env (lowerPiecesAux canonical pieces next rowsReversed).1 →
      R1CS.RowsHold env rowsReversed ∧
        ∀ piece ∈ pieces, ConstraintsHold env (piece.constraints canonical)
  | [], next, rowsReversed, _, rows =>
      ⟨fun row member => rows row (by simpa [lowerPiecesAux] using member), by simp⟩
  | piece :: rest, next, rowsReversed, valid, rows => by
      obtain ⟨accumulated, restHold⟩ := lowerPiecesAux_sound canonical env rest _ _
        (fun other member => valid other (List.mem_cons_of_mem _ member)) rows
      refine ⟨fun row member => accumulated row (by simp [List.reverseAux_eq, member]), ?_⟩
      intro current member
      rcases List.mem_cons.mp member with rfl | member
      · exact lowerPiece_sound canonical current (valid _ List.mem_cons_self) next env
          fun row inPiece => accumulated row (by simp [List.reverseAux_eq, inPiece])
      · exact restHold current member

/-- The rows of the pieces of `constraints` give the constraints. -/
theorem pieces_sound (canonical constraints : List Expr) (next : Nat) (env : Env)
    (rows : R1CS.RowsHold env (lowerPieces canonical (split canonical constraints) next).1) :
    ConstraintsHold env constraints := by
  have holds := (lowerPiecesAux_sound canonical env _ next [] (valid_split canonical constraints)
    rows).2
  intro expression member
  rw [← flatMap_split canonical constraints, List.mem_flatMap] at member
  obtain ⟨piece, pieceMember, inPiece⟩ := member
  exact holds piece pieceMember expression inPiece

end NightstreamFPrime.Export.Generic.PieceRows
