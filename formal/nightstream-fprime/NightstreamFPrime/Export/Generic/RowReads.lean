import NightstreamFPrime.Export.Generic.PieceRows
import NightstreamFPrime.Export.Generic.SlotLayout
import NightstreamFPrime.Layout.R1CS.Support

/-! The columns that a row reads (SPEC.md section 4.4). A step package gives a slot only to a
column that a row of its relation reads (`StepPackage.runsOf`), so completeness needs the columns
that each row and each expression read: a row reads the columns of its terms
(`SlotLayout.rowColumns`), an expression its variables (`SlotLayout.exprColumns`), and the rows of
a ring piece only the columns of its lanes, its outputs and its quotients
(`ringRows_varsSatisfy`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.RowReads

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Spec.RingRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-! ### Expressions and rows -/

theorem exprColumns_varsSatisfy :
    ∀ expression : Expr, expression.VarsSatisfy (· ∈ exprColumns expression)
  | .var column => List.mem_singleton_self column
  | .const _ => trivial
  | .add left right =>
      ⟨(exprColumns_varsSatisfy left).mono _ fun _ member => List.mem_append_left _ member,
        (exprColumns_varsSatisfy right).mono _ fun _ member => List.mem_append_right _ member⟩
  | .mul left right =>
      ⟨(exprColumns_varsSatisfy left).mono _ fun _ member => List.mem_append_left _ member,
        (exprColumns_varsSatisfy right).mono _ fun _ member => List.mem_append_right _ member⟩

theorem exprColumns_of_varsSatisfy {allowed : Nat → Prop} :
    ∀ expression : Expr, expression.VarsSatisfy allowed →
      ∀ column ∈ exprColumns expression, allowed column
  | .var column, support, other, member => by
      simp only [exprColumns, List.mem_singleton] at member
      subst member
      exact support
  | .const _, _, _, member => by simp [exprColumns] at member
  | .add left right, support, column, member => by
      simp only [exprColumns, List.mem_append] at member
      rcases member with inLeft | inRight
      · exact exprColumns_of_varsSatisfy left support.1 column inLeft
      · exact exprColumns_of_varsSatisfy right support.2 column inRight
  | .mul left right, support, column, member => by
      simp only [exprColumns, List.mem_append] at member
      rcases member with inLeft | inRight
      · exact exprColumns_of_varsSatisfy left support.1 column inLeft
      · exact exprColumns_of_varsSatisfy right support.2 column inRight

/-- An expression reads the same value in two environments that agree on its columns. -/
theorem eval_of_columns {left right : Env} (expression : Expr)
    (agree : ∀ column ∈ exprColumns expression, left column = right column) :
    expression.eval left = expression.eval right :=
  Expr.eval_eq_of_agree_satisfy expression _ left right (exprColumns_varsSatisfy expression) agree

theorem rowColumns_varsSatisfy (row : R1CS.Row) : row.VarsSatisfy (· ∈ rowColumns row) := by
  refine ⟨fun term member => ?_, fun term member => ?_, fun term member => ?_⟩ <;>
    exact List.mem_map_of_mem (by simp [member])

theorem rowColumns_of_varsSatisfy {row : R1CS.Row} {allowed : Nat → Prop}
    (scope : row.VarsSatisfy allowed) : ∀ column ∈ rowColumns row, allowed column := by
  intro column member
  obtain ⟨term, termMember, rfl⟩ := List.mem_map.mp member
  rcases List.mem_append.mp termMember with inFirst | inC
  · rcases List.mem_append.mp inFirst with inA | inB
    · exact scope.1 term inA
    · exact scope.2.1 term inB
  · exact scope.2.2 term inC

/-- A lane of `lanes`, or the zero lane past them, reads only the lanes' columns. -/
theorem getD_varsSatisfy (lanes : List Expr) (index : Nat) :
    (lanes.getD index 0).VarsSatisfy (· ∈ lanes.flatMap exprColumns) := by
  by_cases inside : index < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact (exprColumns_varsSatisfy _).mono _ fun column member =>
      List.mem_flatMap.mpr ⟨_, List.getElem_mem inside, member⟩
  · rw [List.getD_eq_default _ _ (by omega)]
    trivial

/-! ### The rows of a ring piece -/

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

end NightstreamFPrime.Export.Generic.RowReads
