import Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership
import Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded
import Nightstream.Implementation.R1CS.Canonical.KStrictNormOwnership
import Nightstream.Implementation.R1CS.Canonical.KPointEquality
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.R1CS.Canonical.KHornerSupport

/-!
Contract: reusable finite column bounds for composed quadratic-extension
programs.

The primitive gadgets expose exact row conservation, but endpoint programs
need a common statement saying that source combinations below a boundary and
explicit multiplication frames below that boundary imply every emitted row
is below it. This module owns that composition only. It emits no constraints
and assigns no protocol meaning to the columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

def CarriedBelow (value : Carried) (boundary : Nat) : Prop :=
  BelowBase value.low boundary ∧ BelowBase value.high boundary

def RowsBelow (rows : List Row) (boundary : Nat) : Prop :=
  ∀ row, row ∈ rows → ∀ column,
    Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column →
      column < boundary

theorem carried_mono
    {value : Carried} {base boundary : Nat}
    (below : CarriedBelow value base) (ordered : base ≤ boundary) :
    CarriedBelow value boundary :=
  ⟨fun column mentioned =>
      Nat.lt_of_lt_of_le (below.1 column mentioned) ordered,
    fun column mentioned =>
      Nat.lt_of_lt_of_le (below.2 column mentioned) ordered⟩

theorem zero_below (boundary : Nat) :
    CarriedBelow KLinear.zeroCarried boundary := by
  constructor <;> intro column mentioned <;>
    simp [KLinear.zeroCarried, BelowBase, Mentions] at mentioned

theorem constant_below
    (value : Nightstream.SuperNeo.Concrete.K)
    (boundary : Nat) (positive : 0 < boundary) :
    CarriedBelow (KLinear.constantCarried value) boundary := by
  constructor <;> intro column mentioned
  all_goals
    have same : column = 0 := by
      simpa [KLinear.constantCarried, Mentions] using mentioned
    omega

theorem one_below (boundary : Nat) (positive : 0 < boundary) :
    CarriedBelow KLinear.oneCarried boundary := by
  constructor
  · intro column mentioned
    have same : column = 0 := by
      simpa [KLinear.oneCarried, Mentions] using mentioned
    omega
  · intro column mentioned
    simp [KLinear.oneCarried, Mentions] at mentioned

theorem add_below
    {left right : Carried} {boundary : Nat}
    (leftBelow : CarriedBelow left boundary)
    (rightBelow : CarriedBelow right boundary) :
    CarriedBelow (KLinear.addCarried left right) boundary := by
  constructor
  · intro column mentioned
    simp only [KLinear.addCarried, BelowBase, Mentions, List.map_append,
      List.mem_append] at mentioned
    rcases mentioned with inLeft | inRight
    · exact leftBelow.1 column inLeft
    · exact rightBelow.1 column inRight
  · intro column mentioned
    simp only [KLinear.addCarried, BelowBase, Mentions, List.map_append,
      List.mem_append] at mentioned
    rcases mentioned with inLeft | inRight
    · exact leftBelow.2 column inLeft
    · exact rightBelow.2 column inRight

theorem sub_below
    {left right : Carried} {boundary : Nat}
    (leftBelow : CarriedBelow left boundary)
    (rightBelow : CarriedBelow right boundary) :
    CarriedBelow (KLinear.subCarried left right) boundary :=
  add_below leftBelow (by
    constructor
    · intro column mentioned
      exact rightBelow.1 column (by
        simpa [KLinear.scaleCarried, LinearSubstitution.scaleTerms,
          Mentions, Function.comp_apply] using mentioned)
    · intro column mentioned
      exact rightBelow.2 column (by
        simpa [KLinear.scaleCarried, LinearSubstitution.scaleTerms,
          Mentions, Function.comp_apply] using mentioned))

theorem to_boolean_carriedBelow
    {value : Carried} {boundary : Nat}
    (below : CarriedBelow value boundary) :
    KBooleanMleSupport.CarriedBelow value boundary :=
  fun column mentioned =>
    mentioned.elim (below.1 column) (below.2 column)

theorem from_boolean_carriedBelow
    {value : Carried} {boundary : Nat}
    (below : KBooleanMleSupport.CarriedBelow value boundary) :
    CarriedBelow value boundary :=
  ⟨fun column mentioned => below column (Or.inl mentioned),
    fun column mentioned => below column (Or.inr mentioned)⟩

theorem tabulate_below
    {variables boundary : Nat}
    (values : BooleanVertex variables → Carried)
    (below : ∀ vertex, CarriedBelow (values vertex) boundary) :
    KBooleanMleSupport.TableBelowBase
      (BooleanTable.tabulate values) boundary := by
  induction variables with
  | zero =>
      exact to_boolean_carriedBelow (below BooleanVertex.nil)
  | succ variables inductionHypothesis =>
      constructor
      · exact inductionHypothesis
          (fun tail => values (.cons false tail))
          (fun tail => below (.cons false tail))
      · exact inductionHypothesis
          (fun tail => values (.cons true tail))
          (fun tail => below (.cons true tail))

theorem paddedTable_below
    {variables boundary : Nat}
    (values : Fin ringDegree → Carried)
    (below : ∀ lane, CarriedBelow (values lane) boundary) :
    KBooleanMleSupport.TableBelowBase
      (KBooleanMleCarriedPadded.carriedTable
        (variables := variables) values)
      boundary := by
  unfold KBooleanMleCarriedPadded.carriedTable
  apply tabulate_below
  intro vertex
  split
  next bounded =>
    exact below ⟨NumericBooleanDomain.index vertex, bounded⟩
  next =>
    exact zero_below boundary

theorem coordinates_below_ofFn
    {variables boundary : Nat}
    (values : Fin variables → Carried)
    (below : ∀ index, CarriedBelow (values index) boundary) :
    KBooleanMleSupport.CoordinatesBelowBase
      (List.ofFn values) boundary := by
  intro value member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact to_boolean_carriedBelow (below index)

theorem boolean_rows_below
    {variables base boundary : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried)
    (tableBelow : KBooleanMleSupport.TableBelowBase table base)
    (coordinatesBelow :
      KBooleanMleSupport.CoordinatesBelowBase coordinates base)
    (endBelow :
      base + 3 * KBooleanMle.frameCount variables ≤ boundary) :
    RowsBelow
      (KBooleanMle.rows (KFrames.frameAt base)
        table coordinates 0)
      boundary := by
  intro row member column mentioned
  exact Nat.lt_of_lt_of_le
    (KBooleanMleSupport.rows_below base table coordinates 0
      tableBelow coordinatesBelow row member column mentioned)
    (by simpa using endBelow)

theorem boolean_output_below
    {variables base boundary : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried)
    (tableBelow : KBooleanMleSupport.TableBelowBase table base)
    (coordinatesBelow :
      KBooleanMleSupport.CoordinatesBelowBase coordinates base)
    (endBelow :
      base + 3 * KBooleanMle.frameCount variables ≤ boundary) :
    CarriedBelow
      (KBooleanMle.carried (KFrames.frameAt base)
        table coordinates 0)
      boundary :=
  from_boolean_carriedBelow
    (KBooleanMleSupport.carriedBelow_mono
      (KBooleanMleSupport.carried_below base
        table coordinates 0 tableBelow coordinatesBelow)
      (by simpa using endBelow))

theorem frame_output_below
    (base step boundary : Nat)
    (endBelow : base + 3 * (step + 1) ≤ boundary) :
    CarriedBelow
      (KMulChain.frameOutput (KFrames.frameAt base step)) boundary := by
  constructor <;> intro column mentioned
  · simp only [KMulChain.frameOutput, outLow, BelowBase, Mentions,
      List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil,
      or_false] at mentioned
    rcases mentioned with rfl | rfl
    all_goals
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega
  · simp only [KMulChain.frameOutput, outHigh, BelowBase, Mentions,
      List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil,
      or_false] at mentioned
    rcases mentioned with rfl | rfl | rfl
    all_goals
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega

theorem strictNorm_rows_below
    (input : KStrictNorm.Input) (boundary : Nat)
    (positive : 0 < boundary)
    (valueBelow : CarriedBelow input.value boundary)
    (framesEnd : input.frameBase + 6 ≤ boundary) :
    RowsBelow (KStrictNorm.rows input) boundary := by
  intro row member column mentioned
  rcases KStrictNormOwnership.rows_conservation
      input row member column mentioned with
    rfl | source | allocated
  · exact positive
  · rcases source with low | high
    · exact valueBelow.1 column low
    · exact valueBelow.2 column high
  · unfold KStrictNormOwnership.Allocated KStrictNorm.columns at allocated
    rw [KFrames.frameColumns_mem_iff] at allocated
    omega

theorem strictNorm_output_below
    (input : KStrictNorm.Input) (boundary : Nat)
    (framesEnd : input.frameBase + 6 ≤ boundary) :
    CarriedBelow (KStrictNorm.output input) boundary := by
  unfold KStrictNorm.output KStrictNorm.secondFrame
  exact frame_output_below input.frameBase 1 boundary (by
    simpa using framesEnd)

theorem equality_rows_below
    (left right : Carried) (boundary : Nat) (positive : 0 < boundary)
    (leftBelow : CarriedBelow left boundary)
    (rightBelow : CarriedBelow right boundary) :
    RowsBelow (KEquality.rows left right) boundary := by
  intro row member column mentioned
  rcases KEquality.rows_conservation left right row member column mentioned with
    rfl | inLeftLow | inLeftHigh | inRightLow | inRightHigh
  · exact positive
  · exact leftBelow.1 column inLeftLow
  · exact leftBelow.2 column inLeftHigh
  · exact rightBelow.1 column inRightLow
  · exact rightBelow.2 column inRightHigh

theorem mul_rows_below
    (left right : Carried) (base step boundary : Nat)
    (leftBelow : CarriedBelow left boundary)
    (rightBelow : CarriedBelow right boundary)
    (frameEnd :
      base + 3 * (step + 1) ≤ boundary) :
    RowsBelow
      (KMul.rows left right (KFrames.frameAt base step)) boundary := by
  intro row member column mentioned
  rcases KMulOwnership.rows_conservation left right
      (KFrames.frameAt base step) row member column mentioned with
    operand | frameColumn
  · rcases operand with inLeftLow | inLeftHigh | inRightLow | inRightHigh
    · exact leftBelow.1 column inLeftLow
    · exact leftBelow.2 column inLeftHigh
    · exact rightBelow.1 column inRightLow
    · exact rightBelow.2 column inRightHigh
  · rcases frameColumn with rfl | rfl | rfl
    all_goals
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega

theorem mulChain_rows_below
    (initial : Carried) (factors : List Carried)
    (base step boundary : Nat)
    (initialBelow : CarriedBelow initial boundary)
    (factorsBelow : ∀ factor ∈ factors, CarriedBelow factor boundary)
    (framesEnd : base + 3 * (step + factors.length) ≤ boundary) :
    RowsBelow
      (KMulChain.rows initial (KFrames.frameAt base) factors step)
      boundary := by
  intro row member column mentioned
  rcases KMulChainOwnership.rows_conservation
      (KFrames.frameAt base) initial factors step
      row member column mentioned with
    inInitial | ⟨factor, factorMember, inFactor⟩
      | ⟨later, lower, upper, inFrame⟩
  · rcases inInitial with low | high
    · exact initialBelow.1 column low
    · exact initialBelow.2 column high
  · rcases inFactor with low | high
    · exact (factorsBelow factor factorMember).1 column low
    · exact (factorsBelow factor factorMember).2 column high
  · rcases inFrame with rfl | rfl | rfl
    all_goals
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega

theorem mulChain_output_below
    (initial : Carried) (factors : List Carried)
    (base step boundary : Nat)
    (initialBelow : CarriedBelow initial boundary)
    (framesEnd : base + 3 * (step + factors.length) ≤ boundary) :
    CarriedBelow
      (KMulChain.productCarried initial (KFrames.frameAt base) factors step)
      boundary := by
  cases factors with
  | nil => simpa [KMulChain.productCarried] using initialBelow
  | cons factor rest =>
      unfold KMulChain.productCarried
      induction rest generalizing step with
      | nil =>
          simpa [KMulChain.productCarried] using
            frame_output_below base step boundary (by
              simp only [List.length_cons, List.length_nil] at framesEnd
              omega)
      | cons next tail inductionHypothesis =>
          rw [KMulChain.productCarried]
          apply inductionHypothesis (step := step + 1)
          · simp only [List.length_cons] at framesEnd ⊢
            omega

private theorem sparse_expandedFactors_below
    {matrixCount boundary : Nat}
    (point : Fin matrixCount → Carried)
    (monomial :
      CCSResidualTable.Monomial Nightstream.SuperNeo.Concrete.K matrixCount)
    (pointBelow : ∀ index, CarriedBelow (point index) boundary) :
    ∀ factor ∈ KSparsePolynomial.expandedFactors point monomial,
      CarriedBelow factor boundary := by
  intro factor member
  rcases List.mem_flatMap.mp member with
    ⟨index, _, inReplicate⟩
  have same : factor = point index :=
    List.eq_of_mem_replicate inReplicate
  subst factor
  exact pointBelow index

private theorem sparse_termsRows_below
    {matrixCount boundary : Nat}
    (point : Fin matrixCount → Carried)
    (positive : 0 < boundary)
    (pointBelow : ∀ index, CarriedBelow (point index) boundary) :
    ∀ (terms :
        List
          (CCSResidualTable.Monomial
            Nightstream.SuperNeo.Concrete.K matrixCount))
      (base offset : Nat),
      base + 3 * (offset + KSparsePolynomial.totalDegreeSum terms) ≤
          boundary →
      RowsBelow
        (KSparsePolynomial.termsRows point terms base offset) boundary
  | [], base, offset, _ => by
      intro row member
      simp [KSparsePolynomial.termsRows] at member
  | monomial :: rest, base, offset, endBelow => by
      intro row member column mentioned
      rcases List.mem_append.mp member with inHead | inTail
      · apply mulChain_rows_below
          (KLinear.constantCarried monomial.coefficient)
          (KSparsePolynomial.expandedFactors point monomial)
          (base + 3 * offset) 0 boundary
        · exact constant_below monomial.coefficient boundary positive
        · exact sparse_expandedFactors_below
            point monomial pointBelow
        · rw [KSparsePolynomial.expandedFactors_length]
          simp only [KSparsePolynomial.totalDegreeSum, List.map_cons,
            List.sum_cons] at endBelow
          omega
        · exact inHead
        · exact mentioned
      · apply sparse_termsRows_below point positive pointBelow rest
          base (offset + monomial.totalDegree)
        · unfold KSparsePolynomial.totalDegreeSum at endBelow ⊢
          simp only [List.map_cons, List.sum_cons] at endBelow
          omega
        · exact inTail
        · exact mentioned

private theorem sparse_termsOutputs_below
    {matrixCount boundary : Nat}
    (point : Fin matrixCount → Carried)
    (positive : 0 < boundary)
    (pointBelow : ∀ index, CarriedBelow (point index) boundary) :
    ∀ (terms :
        List
          (CCSResidualTable.Monomial
            Nightstream.SuperNeo.Concrete.K matrixCount))
      (base offset : Nat),
      base + 3 * (offset + KSparsePolynomial.totalDegreeSum terms) ≤
          boundary →
      ∀ output ∈
        KSparsePolynomial.termOutputs point terms base offset,
        CarriedBelow output boundary
  | [], base, offset, _, output, member => by
      simp [KSparsePolynomial.termOutputs] at member
  | monomial :: rest, base, offset, endBelow, output, member => by
      simp only [KSparsePolynomial.termOutputs, List.mem_cons] at member
      rcases member with rfl | inRest
      · apply mulChain_output_below
          (KLinear.constantCarried monomial.coefficient)
          (KSparsePolynomial.expandedFactors point monomial)
          (base + 3 * offset) 0 boundary
        · exact constant_below monomial.coefficient boundary positive
        · rw [KSparsePolynomial.expandedFactors_length]
          simp only [KSparsePolynomial.totalDegreeSum, List.map_cons,
            List.sum_cons] at endBelow
          omega
      · apply sparse_termsOutputs_below point positive pointBelow rest
          base (offset + monomial.totalDegree)
        · unfold KSparsePolynomial.totalDegreeSum at endBelow ⊢
          simp only [List.map_cons, List.sum_cons] at endBelow
          omega
        · exact inRest

private theorem sparse_sumCarried_below
    {boundary : Nat} :
    ∀ outputs : List Carried,
      (∀ output ∈ outputs, CarriedBelow output boundary) →
      CarriedBelow (KSparsePolynomial.sumCarried outputs) boundary
  | [], _ => zero_below boundary
  | output :: rest, below => by
      apply add_below
      · exact below output List.mem_cons_self
      · apply sparse_sumCarried_below rest
        intro value member
        exact below value (List.mem_cons_of_mem output member)

theorem sparsePolynomial_rows_below
    {matrixCount boundary : Nat}
    (input : KSparsePolynomial.Input matrixCount)
    (positive : 0 < boundary)
    (pointBelow : ∀ index, CarriedBelow (input.point index) boundary)
    (framesEnd :
      input.frameBase +
          3 * KSparsePolynomial.totalDegreeSum input.polynomial.terms ≤
        boundary) :
    RowsBelow (KSparsePolynomial.rows input) boundary := by
  unfold KSparsePolynomial.rows
  apply sparse_termsRows_below input.point positive pointBelow
  simpa using framesEnd

theorem sparsePolynomial_output_below
    {matrixCount boundary : Nat}
    (input : KSparsePolynomial.Input matrixCount)
    (positive : 0 < boundary)
    (pointBelow : ∀ index, CarriedBelow (input.point index) boundary)
    (framesEnd :
      input.frameBase +
          3 * KSparsePolynomial.totalDegreeSum input.polynomial.terms ≤
        boundary) :
    CarriedBelow (KSparsePolynomial.output input) boundary := by
  unfold KSparsePolynomial.output
  apply sparse_sumCarried_below
  intro output member
  exact sparse_termsOutputs_below input.point positive pointBelow
    input.polynomial.terms input.frameBase 0
    (by simpa using framesEnd) output member

private theorem point_intercept_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.intercept input index) boundary := by
  unfold KPointEquality.intercept KLinear.oneMinus
  exact sub_below (one_below boundary positive) (rightBelow index)

private theorem point_slope_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.slope input index) boundary := by
  unfold KPointEquality.slope
  exact sub_below (rightBelow index)
    (point_intercept_below input positive rightBelow index)

private theorem point_factor_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (framesEnd : input.frameBase + 3 * variables ≤ boundary)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.factor input index) boundary := by
  unfold KPointEquality.factor KPointEquality.factorProduct
  apply add_below (point_intercept_below input positive rightBelow index)
  exact frame_output_below input.frameBase index.val boundary (by
    omega)

private theorem point_factors_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (framesEnd : input.frameBase + 3 * variables ≤ boundary) :
    ∀ factor ∈ KPointEquality.factors input,
      CarriedBelow factor boundary := by
  intro factor member
  rcases List.mem_map.mp member with ⟨index, _, rfl⟩
  exact point_factor_below input positive rightBelow framesEnd index

theorem pointEquality_rows_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (leftBelow : ∀ index, CarriedBelow (input.left index) boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (framesEnd :
      input.frameBase + 3 * variables + 3 * (variables - 1) ≤ boundary) :
    RowsBelow (KPointEquality.rows input) boundary := by
  intro row member column mentioned
  rcases List.mem_append.mp member with inFactors | inProduct
  · rcases List.mem_flatMap.mp inFactors with
      ⟨index, _, inRow⟩
    exact mul_rows_below
      (input.left index) (KPointEquality.slope input index)
      input.frameBase index.val boundary
      (leftBelow index)
      (point_slope_below input positive rightBelow index)
      (by omega)
      row inRow column mentioned
  · unfold KPointEquality.productRows at inProduct
    split at inProduct
    next empty =>
      simp at inProduct
    next first rest equal =>
      apply mulChain_rows_below first rest
        (KPointEquality.productBase input) 0 boundary
      · exact point_factors_below input positive rightBelow
          (by omega) first (by
            rw [equal]
            exact List.mem_cons_self)
      · intro factor factorMember
        exact point_factors_below input positive rightBelow
          (by omega) factor (by
            rw [equal]
            exact List.mem_cons_of_mem first factorMember)
      · have sized := KPointEquality.factors_length input
        rw [equal] at sized
        simp only [List.length_cons] at sized
        unfold KPointEquality.productBase
        omega
      · exact inProduct
      · exact mentioned

theorem pointEquality_output_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow : ∀ index, CarriedBelow (input.right index) boundary)
    (framesEnd :
      input.frameBase + 3 * variables + 3 * (variables - 1) ≤ boundary) :
    CarriedBelow (KPointEquality.equalityCarried input) boundary := by
  unfold KPointEquality.equalityCarried
  split
  next empty =>
    exact one_below boundary positive
  next first rest equal =>
    apply mulChain_output_below first rest
      (KPointEquality.productBase input) 0 boundary
    · exact point_factors_below input positive rightBelow
        (by omega) first (by
          rw [equal]
          exact List.mem_cons_self)
    · have sized := KPointEquality.factors_length input
      rw [equal] at sized
      simp only [List.length_cons] at sized
      unfold KPointEquality.productBase
      omega

theorem horner_rows_below
    (beta : Carried) (coefficients : List Carried)
    (base step boundary : Nat)
    (betaBelow : CarriedBelow beta boundary)
    (coefficientsBelow :
      ∀ coefficient ∈ coefficients, CarriedBelow coefficient boundary)
    (framesEnd :
      base + 3 * (step + (coefficients.length - 1)) ≤ boundary) :
    RowsBelow
      (KHorner.hornerRows beta (KFrames.frameAt base) coefficients step)
      boundary := by
  intro row member column mentioned
  rcases KHornerSupport.hornerRows_mentions beta (KFrames.frameAt base)
      coefficients step row member column mentioned with
    inBeta | ⟨coefficient, coefficientMember, inCoefficient⟩
      | ⟨later, lower, upper, inFrame⟩
  · rcases inBeta with low | high
    · exact betaBelow.1 column low
    · exact betaBelow.2 column high
  · rcases inCoefficient with low | high
    · exact (coefficientsBelow coefficient coefficientMember).1 column low
    · exact (coefficientsBelow coefficient coefficientMember).2 column high
  · rcases inFrame with rfl | rfl | rfl
    all_goals
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame]
      omega

theorem horner_output_below
    (beta : Carried) (coefficients : List Carried)
    (base step boundary : Nat)
    (coefficientsBelow :
      ∀ coefficient ∈ coefficients, CarriedBelow coefficient boundary)
    (framesEnd :
      base + 3 * (step + (coefficients.length - 1)) ≤ boundary) :
    CarriedBelow
      (KHorner.hornerCarried beta (KFrames.frameAt base) coefficients step)
      boundary := by
  constructor
  · intro column mentioned
    rcases KHornerSupport.hornerCarried_mentions beta (KFrames.frameAt base)
        coefficients step column (Or.inl mentioned) with
      ⟨coefficient, coefficientMember, inCoefficient⟩
        | ⟨later, lower, upper, inFrame⟩
    · rcases inCoefficient with low | high
      · exact (coefficientsBelow coefficient coefficientMember).1 column low
      · exact (coefficientsBelow coefficient coefficientMember).2 column high
    · rcases inFrame with rfl | rfl | rfl
      all_goals
        simp only [KFrames.frameAt, KFrames.frameColumn,
          KFrames.columnsPerFrame]
        omega
  · intro column mentioned
    rcases KHornerSupport.hornerCarried_mentions beta (KFrames.frameAt base)
        coefficients step column (Or.inr mentioned) with
      ⟨coefficient, coefficientMember, inCoefficient⟩
        | ⟨later, lower, upper, inFrame⟩
    · rcases inCoefficient with low | high
      · exact (coefficientsBelow coefficient coefficientMember).1 column low
      · exact (coefficientsBelow coefficient coefficientMember).2 column high
    · rcases inFrame with rfl | rfl | rfl
      all_goals
        simp only [KFrames.frameAt, KFrames.frameColumn,
          KFrames.columnsPerFrame]
        omega

end Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
