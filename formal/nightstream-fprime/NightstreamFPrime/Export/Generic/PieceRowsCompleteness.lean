import NightstreamFPrime.Export.Generic.PieceRows
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Layout.R1CS.Support

/-! Completeness of the pieces' R1CS rows (CW-5.7), the converse of `PieceRows.pieces_sound`.
A ring-product piece whose constraints hold gives, cell by cell, the ring identity
`output = prior + challenge * value`; its 108 rows per cell then hold when its fresh columns carry
the honest quotient of `challenge * value` by Φ81 (`ringFill`, `QuotientProduct.complete_add`,
SuperNeo v1.2 Section 7.4). Every other piece is lowered by `R1CS.lowerConstraints`, whose fresh
columns `R1CS.executeConstraints` fills. `fillPieces` fills the pieces in their lowering's order.
When every piece reads only columns below the first fresh column, the filled environment agrees
with the honest one off the fresh columns and the lowered rows hold (`lowerPieces_complete`). -/

namespace NightstreamFPrime.Export.Generic.PieceRowsCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-! ### The variables a substituted template reads -/

/-- A substituted expression that reads below `bound` reads, through each template variable, a
wire below `bound`. -/
theorem subst_varsBelow (wires : Nat → Expr) (bound : Nat) :
    ∀ expression : Expr, (subst wires expression).VarsBelow bound →
      expression.VarsSatisfy fun column => (wires column).VarsBelow bound
  | .var _, below => below
  | .const _, _ => trivial
  | .add left right, below =>
      ⟨subst_varsBelow wires bound left below.1, subst_varsBelow wires bound right below.2⟩
  | .mul left right, below =>
      ⟨subst_varsBelow wires bound left below.1, subst_varsBelow wires bound right below.2⟩

private theorem rawFold_terms {allowed : Nat → Prop} (challenge value : Fin ringDegree → Expr)
    (degree : Nat) :
    ∀ (sources : List Nat) (initial : Expr),
      (sources.foldl (fun accumulated source =>
          if source ≤ degree ∧ degree - source < ringDegree then
            accumulated + CombinationStep.exprCoeff challenge source *
              CombinationStep.exprCoeff value (degree - source)
          else accumulated) initial).VarsSatisfy allowed →
        initial.VarsSatisfy allowed ∧ ∀ source ∈ sources, source ≤ degree →
          degree - source < ringDegree →
            (CombinationStep.exprCoeff challenge source).VarsSatisfy allowed ∧
              (CombinationStep.exprCoeff value (degree - source)).VarsSatisfy allowed
  | [], _, support => ⟨support, fun _ member => by simp at member⟩
  | source :: rest, initial, support => by
      rw [List.foldl_cons] at support
      obtain ⟨step, later⟩ := rawFold_terms challenge value degree rest _ support
      by_cases inside : source ≤ degree ∧ degree - source < ringDegree
      · rw [ite_eq_left inside] at step
        refine ⟨step.1, fun current member low high => ?_⟩
        rcases List.mem_cons.mp member with rfl | member
        · exact step.2
        · exact later current member low high
      · rw [ite_eq_right inside] at step
        refine ⟨step, fun current member low high => ?_⟩
        rcases List.mem_cons.mp member with rfl | member
        · exact absurd ⟨low, high⟩ inside
        · exact later current member low high

/-- Lane 53 of a ring product reads every challenge lane and every value lane. -/
private theorem mulExpr_last {allowed : Nat → Prop} (challenge value : Fin ringDegree → Expr)
    (support : (CombinationStep.mulExpr challenge value ⟨53, by decide⟩).VarsSatisfy allowed)
    (source : Fin ringDegree) :
    (challenge source).VarsSatisfy allowed ∧
      (value ⟨53 - source.val, by have := source.isLt; simp only [ringDegree] at *; omega⟩).VarsSatisfy
        allowed := by
  have raw : (CombinationStep.rawExpr challenge value 53).VarsSatisfy allowed := by
    unfold CombinationStep.mulExpr at support
    exact support.1.1
  have bound := source.isLt
  simp only [ringDegree] at bound
  obtain ⟨_, terms⟩ := rawFold_terms challenge value 53 _ _ raw
  have found := terms source.val (List.mem_range.mpr (by simp only [ringDegree]; omega))
    (by omega) (by simp only [ringDegree]; omega)
  have challengeEq : CombinationStep.exprCoeff challenge source.val = challenge source := by
    simp [CombinationStep.exprCoeff, source.isLt]
  have valueEq : CombinationStep.exprCoeff value (53 - source.val) =
      value ⟨53 - source.val, by simp only [ringDegree]; omega⟩ := by
    simp only [CombinationStep.exprCoeff]
    rw [dite_eq_left (by simp only [ringDegree]; omega)]
  rw [challengeEq, valueEq] at found
  exact found

/-- Constraint `lane * cells + cell` of a ring-product block. -/
def ringIndex (cells cell : Nat) (inside : cell < cells) (lane : Fin ringDegree) :
    Fin (ringDegree * cells) :=
  ⟨lane.val * cells + cell, slot_lt lane inside⟩

theorem ringLane_ringIndex {cells cell : Nat} (inside : cell < cells) (lane : Fin ringDegree) :
    ringLane (ringIndex cells cell inside lane) = lane := by
  apply Fin.ext
  show (lane.val * cells + cell) / cells = lane.val
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt inside, Nat.zero_add]

theorem ringIndex_mod {cells cell : Nat} (inside : cell < cells) (lane : Fin ringDegree) :
    (ringIndex cells cell inside lane).val % cells = cell := by
  show (lane.val * cells + cell) % cells = cell
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt inside]

theorem ringIndex_mem (canonical : List Expr) {cells start : Nat} {lanes : List Expr} {cell : Nat}
    (inside : cell < cells) (lane : Fin ringDegree) :
    subst (laneSubst (ringInputs cells) start lanes) (ringConstraint cells (ringIndex cells cell inside lane)) ∈
      (Piece.ringProduct cells start lanes).constraints canonical :=
  List.mem_map_of_mem (List.mem_ofFn.mpr ⟨_, rfl⟩)

theorem ringValue_lt {cells cell : Nat} (inside : cell < cells) (lane : Fin ringDegree) :
    ringValue cells lane cell < ringInputs cells := by
  have := slot_lt lane inside
  simp only [ringValue, ringInputs]
  omega

theorem ringPrior_lt {cells cell : Nat} (inside : cell < cells) (lane : Fin ringDegree) :
    ringPrior cells lane cell < ringInputs cells := by
  have := slot_lt lane inside
  simp only [ringPrior, ringInputs]
  omega

/-- A ring-product piece whose constraints read below `bound` reads below `bound` in each lane
that its rows read, and its outputs lie below `bound`. -/
theorem ringInputs_below {canonical : List Expr} {cells start bound : Nat} {lanes : List Expr}
    (below : ∀ expression ∈ (Piece.ringProduct cells start lanes).constraints canonical,
      expression.VarsBelow bound)
    {cell : Nat} (inside : cell < cells) :
    (∀ lane : Fin ringDegree, (lanes.getD lane.val 0).VarsBelow bound) ∧
      (∀ lane : Fin ringDegree, (lanes.getD (ringValue cells lane cell) 0).VarsBelow bound) ∧
      (∀ lane : Fin ringDegree, (lanes.getD (ringPrior cells lane cell) 0).VarsBelow bound) ∧
      ∀ lane : Fin ringDegree, outputColumn cells start lane cell < bound := by
  have support := fun lane : Fin ringDegree => subst_varsBelow _ bound _
    (below _ (ringIndex_mem canonical (start := start) (lanes := lanes) inside lane))
  have last := support ⟨53, by decide⟩
  unfold ringConstraint at last
  rw [ringLane_ringIndex inside] at last
  have inputBelow : ∀ column, column < ringInputs cells →
      (laneSubst (ringInputs cells) start lanes column).VarsBelow bound →
        (lanes.getD column 0).VarsBelow bound := by
    intro column inputs holds
    rwa [laneSubst_input lanes inputs] at holds
  refine ⟨fun lane => ?_, fun lane => ?_, fun lane => ?_, fun lane => ?_⟩
  · have found := (mulExpr_last _ _ last.2.2.2 lane).1
    exact inputBelow _ (by have := lane.isLt; simp only [ringInputs]; omega) found
  · have bound' := lane.isLt
    simp only [ringDegree] at bound'
    have found := (mulExpr_last _ _ last.2.2.2 ⟨53 - lane.val, by simp only [ringDegree]; omega⟩).2
    have same : (⟨53 - (53 - lane.val), by simp only [ringDegree]; omega⟩ : Fin ringDegree) = lane :=
      Fin.ext (by simp only; omega)
    rw [same, ringIndex_mod inside] at found
    exact inputBelow _ (ringValue_lt inside lane) found
  · have found := (support lane)
    unfold ringConstraint at found
    rw [ringLane_ringIndex inside, ringIndex_mod inside] at found
    exact inputBelow _ (ringPrior_lt inside lane) found.2.2.1
  · have found := (support lane).1
    change (laneSubst (ringInputs cells) start lanes
      (ringInputs cells + (ringIndex cells cell inside lane).val)).VarsBelow bound at found
    rw [laneSubst_local] at found
    exact found

/-! ### The honest quotients -/

/-- The challenge, and a cell's value and prior, that the lanes of a ring piece read in `env`. -/
def challengeOf (lanes : List Expr) (env : Env) : RingF :=
  fun lane => (lanes.getD lane.val 0).eval env

def valueOf (cells : Nat) (lanes : List Expr) (env : Env) (cell : Nat) : RingF :=
  fun lane => (lanes.getD (ringValue cells lane cell) 0).eval env

def priorOf (cells : Nat) (lanes : List Expr) (env : Env) (cell : Nat) : RingF :=
  fun lane => (lanes.getD (ringPrior cells lane cell) 0).eval env

/-- `env` with the honest quotients of a ring piece in its fresh columns from `quotient`: column
`quotientColumn quotient lane cell` holds lane `lane` of the quotient of the cell's
`challenge * value` by Φ81. -/
def ringFill (cells quotient : Nat) (lanes : List Expr) (env : Env) : Env := fun column =>
  if quotient ≤ column ∧ column < quotient + ringDegree * cells then
    QuotientProduct.quotientCoeff (challengeOf lanes env)
      (valueOf cells lanes env ((column - quotient) / ringDegree))
      ⟨(column - quotient) % ringDegree, Nat.mod_lt _ (by decide)⟩
  else env column

theorem ringFill_outside {cells quotient : Nat} {lanes : List Expr} {env : Env} {column : Nat}
    (outside : column < quotient ∨ quotient + ringDegree * cells ≤ column) :
    ringFill cells quotient lanes env column = env column := by
  unfold ringFill
  rw [ite_eq_right (by omega)]

theorem ringFill_quotient {cells quotient cell : Nat} {lanes : List Expr} {env : Env}
    (inside : cell < cells) (lane : Fin ringDegree) :
    ringFill cells quotient lanes env (quotientColumn quotient lane cell) =
      QuotientProduct.quotient (challengeOf lanes env) (valueOf cells lanes env cell) lane := by
  have laneBound := lane.isLt
  have below : cell * ringDegree + lane.val < ringDegree * cells := by
    have := slot_lt lane inside
    calc cell * ringDegree + lane.val < cell * ringDegree + ringDegree := by omega
      _ = (cell + 1) * ringDegree := by rw [Nat.succ_mul]
      _ ≤ cells * ringDegree := Nat.mul_le_mul_right _ inside
      _ = ringDegree * cells := Nat.mul_comm _ _
  have cellEq : (cell * ringDegree + lane.val) / ringDegree = cell := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide), Nat.div_eq_of_lt laneBound,
      Nat.zero_add]
  have laneEq : (cell * ringDegree + lane.val) % ringDegree = lane.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt laneBound]
  unfold ringFill quotientColumn
  rw [ite_eq_left (by omega)]
  simp only [Nat.add_sub_cancel_left, cellEq]
  rw [← QuotientProduct.quotientCoeff_eq_quotient]
  congr 1
  exact Fin.ext laneEq

/-- A ring piece whose constraints hold gives each cell's ring identity. -/
theorem ring_identity (canonical : List Expr) {cells start : Nat} {lanes : List Expr} {env : Env}
    (holds : ConstraintsHold env ((Piece.ringProduct cells start lanes).constraints canonical))
    {cell : Nat} (inside : cell < cells) :
    (fun lane : Fin ringDegree => env (outputColumn cells start lane cell)) =
      ringFAdd (priorOf cells lanes env cell)
        (ringFMul (challengeOf lanes env) (valueOf cells lanes env cell)) := by
  funext lane
  let lanesEnv : Env := fun column => (laneSubst (ringInputs cells) start lanes column).eval env
  have zero := holds _ (ringIndex_mem canonical (start := start) (lanes := lanes) inside lane)
  rw [Lifecycle.Nebula.Expr.eval_subst] at zero
  change (ringConstraint cells (ringIndex cells cell inside lane)).eval lanesEnv = 0 at zero
  have input : ∀ column, column < ringInputs cells →
      lanesEnv column = (lanes.getD column 0).eval env := by
    intro column below
    simp only [lanesEnv, laneSubst_input lanes below]
  have challenge : CombinationStep.evalRing lanesEnv (fun source => Expr.var source.val) =
      challengeOf lanes env := by
    funext source
    have := source.isLt
    exact input _ (by simp only [ringInputs]; omega)
  have value : CombinationStep.evalRing lanesEnv
        (fun source => Expr.var (ringValue cells source cell)) = valueOf cells lanes env cell := by
    funext source
    exact input _ (ringValue_lt inside source)
  have prior : lanesEnv (ringPrior cells lane cell) = priorOf cells lanes env cell lane :=
    input _ (ringPrior_lt inside lane)
  have output : lanesEnv (ringInputs cells + (ringIndex cells cell inside lane).val) =
      env (outputColumn cells start lane cell) := by
    simp only [lanesEnv, laneSubst_local, Expr.eval_var, outputColumn, ringIndex]
  simp only [ringConstraint, Expr.eval_sub, Expr.eval_hadd, Expr.eval_var,
    CombinationStep.mulExpr_eval] at zero
  rw [ringLane_ringIndex inside, ringIndex_mod inside, challenge, value, prior, output] at zero
  exact sub_eq_zero.mp zero

/-- Ring-row completeness, the converse of `PieceRows.ringRows_sound`: when a ring piece's
constraints hold and read below its fresh columns from `quotient`, its rows hold once those
columns carry the honest quotients. -/
theorem ringRows_complete (canonical : List Expr) (cells start quotient : Nat)
    (lanes : List Expr) (linear : LinearLanes lanes) (env : Env)
    (below : ∀ expression ∈ (Piece.ringProduct cells start lanes).constraints canonical,
      expression.VarsBelow quotient)
    (holds : ConstraintsHold env ((Piece.ringProduct cells start lanes).constraints canonical)) :
    R1CS.RowsHold (ringFill cells quotient lanes env) (ringRows cells start quotient lanes) := by
  intro row member
  obtain ⟨cell, cellMember, rowMember⟩ := List.mem_flatMap.mp member
  have inside := List.mem_range.mp cellMember
  obtain ⟨node, rfl⟩ := List.mem_ofFn.mp rowMember
  obtain ⟨challengeBelow, valueBelow, priorBelow, outputBelow⟩ := ringInputs_below below inside
  let filled := ringFill cells quotient lanes env
  have agree : ∀ column, column < quotient → filled column = env column :=
    fun column below => ringFill_outside (Or.inl below)
  have challenge : (fun lane : Fin ringDegree => (lanes.getD lane.val 0).eval filled) =
      challengeOf lanes env :=
    funext fun lane => (lanes.getD lane.val 0).eval_eq_of_agree_below quotient _ _
      (challengeBelow lane) agree
  have value : (fun lane : Fin ringDegree => (lanes.getD (ringValue cells lane cell) 0).eval filled) =
      valueOf cells lanes env cell :=
    funext fun lane => (lanes.getD (ringValue cells lane cell) 0).eval_eq_of_agree_below quotient
      _ _ (valueBelow lane) agree
  have prior : (fun lane : Fin ringDegree => (lanes.getD (ringPrior cells lane cell) 0).eval filled) =
      priorOf cells lanes env cell :=
    funext fun lane => (lanes.getD (ringPrior cells lane cell) 0).eval_eq_of_agree_below quotient
      _ _ (priorBelow lane) agree
  have output : (fun lane : Fin ringDegree => filled (outputColumn cells start lane cell)) =
      ringFAdd (priorOf cells lanes env cell)
        (ringFMul (challengeOf lanes env) (valueOf cells lanes env cell)) :=
    (funext fun lane => agree _ (outputBelow lane)).trans (ring_identity canonical holds inside)
  have quotientEq : (fun lane : Fin ringDegree => filled (quotientColumn quotient lane cell)) =
      QuotientProduct.quotient (challengeOf lanes env) (valueOf cells lanes env cell) :=
    funext fun lane => ringFill_quotient inside lane
  simp only [R1CS.Row.Holds, ringRow, R1CS.LinearCombination.eval_add,
    R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_ofVar, evaluation_eval,
    laneCombination_eval linear]
  rw [challenge, value, prior, output, quotientEq,
    QuotientProduct.complete_add (challengeOf lanes env) (valueOf cells lanes env cell)
      (priorOf cells lanes env cell) node, sub_eq_add_neg, neg_one_mul, add_assoc]

/-! ### The columns the rows read -/

theorem linear?_varsSatisfy {allowed : Nat → Prop} :
    ∀ (expression : Expr) (combination : SparseCombination),
      linear? expression = some combination → expression.VarsSatisfy allowed →
        ∀ term ∈ combination.terms, allowed term.column
  | .var column, combination, found, support => by
      simp only [linear?, Option.some.injEq] at found
      subst found
      intro term member
      simp only [List.mem_singleton] at member
      subst member
      exact support
  | .const _, combination, found, _ => by
      simp only [linear?, Option.some.injEq] at found
      subst found
      simp
  | .add left right, combination, found, support => by
      simp only [linear?, Option.bind_eq_some_iff, Option.map_eq_some_iff] at found
      obtain ⟨leftCombination, leftFound, rightCombination, rightFound, rfl⟩ := found
      intro term member
      simp only [plus, List.mem_append] at member
      rcases member with member | member
      · exact linear?_varsSatisfy left leftCombination leftFound support.1 term member
      · exact linear?_varsSatisfy right rightCombination rightFound support.2 term member
  | .mul left right, combination, found, support => by
      unfold linear? at found
      cases leftConstant : constant? left with
      | some factor =>
          rw [leftConstant] at found
          simp only [Option.map_eq_some_iff] at found
          obtain ⟨inner, innerFound, rfl⟩ := found
          intro term member
          simp only [scale, List.mem_map] at member
          obtain ⟨original, originalMember, rfl⟩ := member
          exact linear?_varsSatisfy right inner innerFound support.2 original originalMember
      | none =>
          cases rightConstant : constant? right with
          | some factor =>
              rw [leftConstant, rightConstant] at found
              simp only [Option.map_eq_some_iff] at found
              obtain ⟨inner, innerFound, rfl⟩ := found
              intro term member
              simp only [scale, List.mem_map] at member
              obtain ⟨original, originalMember, rfl⟩ := member
              exact linear?_varsSatisfy left inner innerFound support.1 original originalMember
          | none =>
              rw [leftConstant, rightConstant] at found
              simp at found

theorem laneCombination_varsSatisfy {allowed : Nat → Prop} (lane : Expr)
    (support : lane.VarsSatisfy allowed) : (laneCombination lane).VarsSatisfy allowed := by
  intro term member
  unfold laneCombination at member
  cases found : linear? lane with
  | none =>
      rw [found] at member
      simp [zeroSparseCombination, SparseCombination.toR1CS] at member
  | some combination =>
      rw [found, Option.getD_some] at member
      simp only [SparseCombination.toR1CS, List.mem_map] at member
      obtain ⟨original, originalMember, rfl⟩ := member
      exact linear?_varsSatisfy lane combination found support original originalMember

theorem sumCombinations_varsSatisfy {allowed : Nat → Prop} :
    ∀ combinations : List R1CS.LinearCombination,
      (∀ combination ∈ combinations, combination.VarsSatisfy allowed) →
        (sumCombinations combinations).VarsSatisfy allowed
  | [], _ => fun _ member => by simp [sumCombinations, R1CS.LinearCombination.zero] at member
  | combination :: rest, support =>
      R1CS.LinearCombination.VarsSatisfy.add _ _ _ (support _ List.mem_cons_self)
        (sumCombinations_varsSatisfy rest fun other member =>
          support other (List.mem_cons_of_mem _ member))

theorem evaluation_varsSatisfy {allowed : Nat → Prop} (point : F)
    (lanes : Fin ringDegree → R1CS.LinearCombination)
    (support : ∀ lane, (lanes lane).VarsSatisfy allowed) :
    (evaluation point lanes).VarsSatisfy allowed := by
  apply sumCombinations_varsSatisfy
  intro combination member
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
  exact R1CS.LinearCombination.VarsSatisfy.scale _ _ _ (support lane)

/-- The rows of a ring piece read only its lanes' columns, its outputs and its quotients. -/
theorem ringRows_varsSatisfy {allowed : Nat → Prop} (cells start quotient : Nat)
    (lanes : List Expr)
    (challenge : ∀ lane : Fin ringDegree, (lanes.getD lane.val 0).VarsSatisfy allowed)
    (values : ∀ cell, cell < cells → ∀ lane : Fin ringDegree,
      (lanes.getD (ringValue cells lane cell) 0).VarsSatisfy allowed)
    (priors : ∀ cell, cell < cells → ∀ lane : Fin ringDegree,
      (lanes.getD (ringPrior cells lane cell) 0).VarsSatisfy allowed)
    (outputs : ∀ cell, cell < cells → ∀ lane : Fin ringDegree,
      allowed (outputColumn cells start lane cell))
    (quotients : ∀ cell, cell < cells → ∀ lane : Fin ringDegree,
      allowed (quotientColumn quotient lane cell)) :
    ∀ row ∈ ringRows cells start quotient lanes, row.VarsSatisfy allowed := by
  intro row member
  obtain ⟨cell, cellMember, rowMember⟩ := List.mem_flatMap.mp member
  have inside := List.mem_range.mp cellMember
  obtain ⟨node, rfl⟩ := List.mem_ofFn.mp rowMember
  let point := QuotientProduct.node node
  have challengeRow := evaluation_varsSatisfy point
    (fun lane => laneCombination (lanes.getD lane.val 0))
    fun lane => laneCombination_varsSatisfy _ (challenge lane)
  have valueRow := evaluation_varsSatisfy point
    (fun lane => laneCombination (lanes.getD (ringValue cells lane cell) 0))
    fun lane => laneCombination_varsSatisfy _ (values cell inside lane)
  have priorRow := evaluation_varsSatisfy point
    (fun lane => laneCombination (lanes.getD (ringPrior cells lane cell) 0))
    fun lane => laneCombination_varsSatisfy _ (priors cell inside lane)
  have outputRow := evaluation_varsSatisfy point
    (fun lane => R1CS.LinearCombination.ofVar (outputColumn cells start lane cell))
    fun lane => R1CS.LinearCombination.VarsSatisfy.ofVar _ _ (outputs cell inside lane)
  have quotientRow := evaluation_varsSatisfy point
    (fun lane => R1CS.LinearCombination.ofVar (quotientColumn quotient lane cell))
    fun lane => R1CS.LinearCombination.VarsSatisfy.ofVar _ _ (quotients cell inside lane)
  exact ⟨challengeRow, valueRow, R1CS.LinearCombination.VarsSatisfy.add _ _ _ outputRow
    (R1CS.LinearCombination.VarsSatisfy.add _ _ _
      (R1CS.LinearCombination.VarsSatisfy.scale _ _ _ priorRow)
      (R1CS.LinearCombination.VarsSatisfy.scale _ _ _ quotientRow))⟩

theorem quotientColumn_lt {quotient cell cells : Nat} (inside : cell < cells)
    (lane : Fin ringDegree) :
    quotientColumn quotient lane cell < quotient + ringDegree * cells := by
  have laneBound := lane.isLt
  have : cell * ringDegree + lane.val < ringDegree * cells := by
    calc cell * ringDegree + lane.val < cell * ringDegree + ringDegree := by omega
      _ = (cell + 1) * ringDegree := by rw [Nat.succ_mul]
      _ ≤ cells * ringDegree := Nat.mul_le_mul_right _ inside
      _ = ringDegree * cells := Nat.mul_comm _ _
  simp only [quotientColumn]
  omega

/-! ### Filling the pieces -/

/-- The honest values of a piece's fresh columns from `next`: the quotients of a ring piece, the
lowering's products (`R1CS.executeConstraints`) of any other piece. -/
def fillPiece (canonical : List Expr) (env : Env) (piece : Piece) (next : Nat) : Env :=
  match piece with
  | .ringProduct cells _ lanes => ringFill cells next lanes env
  | piece => R1CS.executeConstraints env (piece.constraints canonical) next

/-- The honest values of the pieces' fresh columns, piece by piece, as `PieceRows.lowerPieces`
places them. -/
def fillPieces (canonical : List Expr) : Env → List Piece → Nat → Env
  | env, [], _ => env
  | env, piece :: rest, next =>
      fillPieces canonical (fillPiece canonical env piece next) rest
        (lowerPiece canonical piece next).2

theorem lowerPiece_next (canonical : List Expr) (piece : Piece) (next : Nat) :
    next ≤ (lowerPiece canonical piece next).2 := by
  cases piece with
  | ringProduct cells start lanes => exact Nat.le_add_right _ _
  | constraint expression =>
      show next ≤ (R1CS.lowerConstraints _ next).next
      rw [R1CS.lowerConstraints_next]
      exact Nat.le_add_right _ _
  | permutation start lanes =>
      show next ≤ (R1CS.lowerConstraints _ next).next
      rw [R1CS.lowerConstraints_next]
      exact Nat.le_add_right _ _

theorem fillPiece_outside (canonical : List Expr) (env : Env) (piece : Piece) (next : Nat)
    {column : Nat} (outside : column < next ∨ (lowerPiece canonical piece next).2 ≤ column) :
    fillPiece canonical env piece next column = env column := by
  cases piece with
  | ringProduct cells start lanes => exact ringFill_outside outside
  | constraint expression =>
      apply R1CS.executeConstraints_agreesOutside
      simpa [lowerPiece] using outside
  | permutation start lanes =>
      apply R1CS.executeConstraints_agreesOutside
      simpa [lowerPiece] using outside

theorem lowerPiece_complete (canonical : List Expr) (piece : Piece) (valid : piece.Valid)
    (next : Nat) (env : Env)
    (below : ∀ expression ∈ piece.constraints canonical, expression.VarsBelow next)
    (holds : ConstraintsHold env (piece.constraints canonical)) :
    R1CS.RowsHold (fillPiece canonical env piece next) (lowerPiece canonical piece next).1 := by
  cases piece with
  | ringProduct cells start lanes =>
      exact ringRows_complete canonical cells start next lanes valid env below holds
  | constraint expression =>
      exact R1CS.executeConstraints_holds_rows env _ next below holds
  | permutation start lanes =>
      exact R1CS.executeConstraints_holds_rows env _ next below holds

/-- The rows of a piece read below its lowering's next fresh column. -/
theorem lowerPiece_rows_below (canonical : List Expr) (piece : Piece) (next : Nat)
    (below : ∀ expression ∈ piece.constraints canonical, expression.VarsBelow next) :
    ∀ row ∈ (lowerPiece canonical piece next).1,
      row.VarsBelow (lowerPiece canonical piece next).2 := by
  cases piece with
  | ringProduct cells start lanes =>
      intro row member
      have inputs := fun cell (inside : cell < cells) => ringInputs_below below inside
      have scope := ringRows_varsSatisfy (allowed := fun column => column < next + ringDegree * cells)
        cells start next lanes
        (fun lane => by
          rcases Nat.eq_zero_or_pos cells with zero | positive
          · subst zero
            have empty : row ∈ ringRows 0 start next lanes := member
            simp only [ringRows, List.range_zero, List.flatMap_nil, List.not_mem_nil] at empty
          · exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr
              (Expr.VarsBelow.mono _ ((inputs 0 positive).1 lane) (Nat.le_add_right _ _)))
        (fun cell inside lane => (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr
          (Expr.VarsBelow.mono _ ((inputs cell inside).2.1 lane) (Nat.le_add_right _ _)))
        (fun cell inside lane => (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr
          (Expr.VarsBelow.mono _ ((inputs cell inside).2.2.1 lane) (Nat.le_add_right _ _)))
        (fun cell inside lane => by
          have := (inputs cell inside).2.2.2 lane
          omega)
        (fun cell inside lane => quotientColumn_lt inside lane) row member
      exact scope
  | constraint expression =>
      intro row member
      have scope := R1CS.lowerConstraints_rows_varsBelow _ next below row member
      simpa [lowerPiece, R1CS.lowerConstraints_next] using scope
  | permutation start lanes =>
      intro row member
      have scope := R1CS.lowerConstraints_rows_varsBelow _ next below row member
      simpa [lowerPiece, R1CS.lowerConstraints_next] using scope

theorem lowerPiecesAux_next (canonical : List Expr) :
    ∀ (pieces : List Piece) (next : Nat) (first second : List R1CS.Row),
      (lowerPiecesAux canonical pieces next first).2 =
        (lowerPiecesAux canonical pieces next second).2
  | [], _, _, _ => rfl
  | piece :: rest, next, first, second => by
      simp only [lowerPiecesAux]
      exact lowerPiecesAux_next canonical rest _ _ _

theorem lowerPiecesAux_le (canonical : List Expr) :
    ∀ (pieces : List Piece) (next : Nat) (rowsReversed : List R1CS.Row),
      next ≤ (lowerPiecesAux canonical pieces next rowsReversed).2
  | [], _, _ => le_refl _
  | piece :: rest, next, rowsReversed => by
      simp only [lowerPiecesAux]
      exact le_trans (lowerPiece_next canonical piece next)
        (lowerPiecesAux_le canonical rest _ _)

/-- The filled environment agrees with `env` off the fresh columns. -/
theorem fillPieces_outside (canonical : List Expr) :
    ∀ (pieces : List Piece) (env : Env) (next : Nat) {column : Nat},
      column < next ∨ (lowerPieces canonical pieces next).2 ≤ column →
        fillPieces canonical env pieces next column = env column
  | [], _, _, _, _ => rfl
  | piece :: rest, env, next, column, outside => by
      have restNext : (lowerPieces canonical (piece :: rest) next).2 =
          (lowerPieces canonical rest (lowerPiece canonical piece next).2).2 := by
        simp only [lowerPieces, lowerPiecesAux]
        exact lowerPiecesAux_next canonical rest _ _ _
      have pieceNext := lowerPiece_next canonical piece next
      have restLe := lowerPiecesAux_le canonical rest (lowerPiece canonical piece next).2 []
      rw [restNext] at outside
      simp only [fillPieces]
      rcases outside with low | high
      · rw [fillPieces_outside canonical rest _ _ (Or.inl (by omega))]
        exact fillPiece_outside canonical env piece next (Or.inl low)
      · rw [fillPieces_outside canonical rest _ _ (Or.inr high)]
        exact fillPiece_outside canonical env piece next (Or.inr (by
          simp only [lowerPieces] at high
          omega))

theorem lowerPiecesAux_complete (canonical : List Expr) :
    ∀ (pieces : List Piece) (env : Env) (bound next : Nat) (rowsReversed : List R1CS.Row),
      bound ≤ next → (∀ piece ∈ pieces, piece.Valid) →
      (∀ piece ∈ pieces, ∀ expression ∈ piece.constraints canonical,
        expression.VarsBelow bound) →
      (∀ piece ∈ pieces, ConstraintsHold env (piece.constraints canonical)) →
      (∀ row ∈ rowsReversed, row.VarsBelow next) → R1CS.RowsHold env rowsReversed →
      R1CS.RowsHold (fillPieces canonical env pieces next)
        (lowerPiecesAux canonical pieces next rowsReversed).1
  | [], env, _, _, rowsReversed, _, _, _, _, _, rows => by
      intro row member
      exact rows row (by simpa [lowerPiecesAux] using member)
  | piece :: rest, env, bound, next, rowsReversed, le, valid, below, holds, rowsBelow, rows => by
      let filled := fillPiece canonical env piece next
      have agree : ∀ column, column < next → filled column = env column :=
        fun column low => fillPiece_outside canonical env piece next (Or.inl low)
      have pieceBelow : ∀ expression ∈ piece.constraints canonical, expression.VarsBelow next :=
        fun expression member => Expr.VarsBelow.mono _
          (below piece List.mem_cons_self expression member) le
      have pieceRows := lowerPiece_complete canonical piece (valid _ List.mem_cons_self) next env
        pieceBelow (holds _ List.mem_cons_self)
      have pieceRowsBelow := lowerPiece_rows_below canonical piece next pieceBelow
      have oldRows : R1CS.RowsHold filled rowsReversed :=
        R1CS.rowsHold_of_agree_below rowsReversed next env filled rowsBelow agree rows
      simp only [lowerPiecesAux]
      apply lowerPiecesAux_complete canonical rest filled bound
      · exact le_trans le (lowerPiece_next canonical piece next)
      · exact fun other member => valid other (List.mem_cons_of_mem _ member)
      · exact fun other member => below other (List.mem_cons_of_mem _ member)
      · intro other member
        exact constraintsHold_of_agree_below env filled _ bound
          (below other (List.mem_cons_of_mem _ member))
          (fun column low => agree column (by omega)) (holds other (List.mem_cons_of_mem _ member))
      · intro row member
        rw [List.reverseAux_eq, List.mem_append, List.mem_reverse] at member
        rcases member with member | member
        · exact pieceRowsBelow row member
        · exact R1CS.Row.VarsBelow.mono row (rowsBelow row member)
            (lowerPiece_next canonical piece next)
      · intro row member
        rw [List.reverseAux_eq, List.mem_append, List.mem_reverse] at member
        rcases member with member | member
        · exact pieceRows row member
        · exact oldRows row member

/-- Lowering completeness, the converse of `PieceRows.lowerPiecesAux_sound`: when every piece's
constraints hold and read below the first fresh column `next`, the lowered rows hold in the filled
environment, which agrees with `env` off the fresh columns. -/
theorem lowerPieces_complete (canonical : List Expr) (pieces : List Piece) (next : Nat)
    (env : Env) (valid : ∀ piece ∈ pieces, piece.Valid)
    (below : ∀ piece ∈ pieces, ∀ expression ∈ piece.constraints canonical,
      expression.VarsBelow next)
    (holds : ∀ piece ∈ pieces, ConstraintsHold env (piece.constraints canonical)) :
    R1CS.RowsHold (fillPieces canonical env pieces next) (lowerPieces canonical pieces next).1 ∧
      ∀ column, column < next ∨ (lowerPieces canonical pieces next).2 ≤ column →
        fillPieces canonical env pieces next column = env column :=
  ⟨lowerPiecesAux_complete canonical pieces env next next [] (le_refl _) valid below holds
      (fun _ member => by simp at member) (fun _ member => by simp at member),
    fun _ outside => fillPieces_outside canonical pieces env next outside⟩

theorem lowerPiecesAux_rows_below (canonical : List Expr) :
    ∀ (pieces : List Piece) (bound next : Nat) (rowsReversed : List R1CS.Row),
      bound ≤ next →
      (∀ piece ∈ pieces, ∀ expression ∈ piece.constraints canonical,
        expression.VarsBelow bound) →
      (∀ row ∈ rowsReversed, row.VarsBelow next) →
      ∀ row ∈ (lowerPiecesAux canonical pieces next rowsReversed).1,
        row.VarsBelow (lowerPiecesAux canonical pieces next rowsReversed).2
  | [], _, _, rowsReversed, _, _, rowsBelow => by
      intro row member
      exact rowsBelow row (by simpa [lowerPiecesAux] using member)
  | piece :: rest, bound, next, rowsReversed, le, below, rowsBelow => by
      have pieceBelow : ∀ expression ∈ piece.constraints canonical, expression.VarsBelow next :=
        fun expression member => Expr.VarsBelow.mono _
          (below piece List.mem_cons_self expression member) le
      simp only [lowerPiecesAux]
      apply lowerPiecesAux_rows_below canonical rest bound
      · exact le_trans le (lowerPiece_next canonical piece next)
      · exact fun other member => below other (List.mem_cons_of_mem _ member)
      · intro row member
        rw [List.reverseAux_eq, List.mem_append, List.mem_reverse] at member
        rcases member with member | member
        · exact lowerPiece_rows_below canonical piece next pieceBelow row member
        · exact R1CS.Row.VarsBelow.mono row (rowsBelow row member)
            (lowerPiece_next canonical piece next)

/-- The lowered rows read below the last fresh column. -/
theorem lowerPieces_rows_below (canonical : List Expr) (pieces : List Piece) (next : Nat)
    (below : ∀ piece ∈ pieces, ∀ expression ∈ piece.constraints canonical,
      expression.VarsBelow next) :
    ∀ row ∈ (lowerPieces canonical pieces next).1,
      row.VarsBelow (lowerPieces canonical pieces next).2 :=
  lowerPiecesAux_rows_below canonical pieces next next [] (le_refl _) below
    (fun _ member => by simp at member)

end NightstreamFPrime.Export.Generic.PieceRowsCompleteness
