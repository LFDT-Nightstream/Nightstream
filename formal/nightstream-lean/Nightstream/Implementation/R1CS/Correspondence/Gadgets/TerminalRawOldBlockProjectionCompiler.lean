import Nightstream.Implementation.R1CS.Core.Projection.Interpretation

/-!
Exact symbolic compiler for the direct terminal raw-witness projection.

This leaf models the row schedule emitted by the terminal repair, without
materialising a production-sized certificate in Lean:

* a compact prefix chi tensor, with one five-row `KMulTrace` per live parent
  at every point coordinate;
* two base-field multiplication rows for every live packed coordinate, after
  radix recomposition of the fourteen ordered raw witness cells; and
* two terminal linear equality rows for every active Phi81 lane.

The compiler is indexed by ordinary `Fin`/`List` data.  Its theorems consume
actual `RowHolds` facts for the rows below, never an `Accepted` proposition.
Generated artifacts own only the fixed profile, column map, and row-at-index
identity.  This file does not identify a raw column with a prover-carried
`CeClaim.y_zcol`, a digest, or a commitment opening.
-/

namespace Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩
local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm⟩

private theorem k_ext {left right : K}
    (c0 : left.c0 = right.c0) (c1 : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp_all

def K.sub (left right : K) : K where
  c0 := left.c0 - right.c0
  c1 := left.c1 - right.c1

private theorem fmul_neg_right (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := fmul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := congrArg Neg.neg (fmul_comm _ _)

theorem K.sub_mul_self (left point : K) :
    K.sub left (K.mul left point) =
      K.mul left (K.sub K.one point) := by
  rcases left with ⟨left0, left1⟩
  rcases point with ⟨point0, point1⟩
  simp only [K.sub, K.mul, K.one, K.mk.injEq]
  constructor <;>
    simp only [Fin.sub_eq_add_neg, fmul_add, Fin.mul_one, Fin.mul_zero,
      Fin.add_zero, Fin.zero_mul, Fin.zero_add,
      Lean.Grind.Fin.neg_mul, fmul_neg_right,
      Lean.Grind.AddCommGroup.neg_add] <;>
    ac_rfl

/-- The root of the tensor is the constant extension-field one. -/
def tensorRoot : KTerms where
  c0 := [(0, 1)]
  c1 := []

/-- Linear terms for one verifier-owned point coordinate. -/
def pointTerms (point : KColumns) : KTerms :=
  KTerms.ofColumns point

/-- Linear terms for `1 - point`.  This allocation-free representation is
the exact one used by `klc_one_minus`. -/
def oneMinusPointTerms (point : KColumns) : KTerms where
  c0 := [(0, 1), (point.c0, goldilocksP - 1)]
  c1 := [(point.c1, goldilocksP - 1)]

/-- One compact-prefix tensor round.  `multiplications[i]` owns the sole
five-row K multiplication emitted for live parent `i`. -/
structure TensorLevel where
  multiplicationCount : Nat
  trace : Fin multiplicationCount -> KMulTrace

private def emptyTerms : KTerms := ⟨[], []⟩

private def dummyTrace : KMulTrace where
  left := emptyTerms
  right := emptyTerms
  sumLeft := []
  sumRight := []
  productC0 := 0
  productC1 := 0
  productSum := 0
  output := default

def tensorTraceAt (level : TensorLevel) (parent : Nat) : KMulTrace :=
  if inRange : parent < level.multiplicationCount then
    level.trace ⟨parent, inRange⟩
  else
    dummyTrace

/-- A high child exists exactly when its little-endian tensor index remains
inside the live block prefix. -/
def highLive (blockCount round parent : Nat) : Bool :=
  decide (parent + 2 ^ round < blockCount)

/-- When both children are live, the multiplication computes the high child
and the low child is the allocation-free LC `parent - high`.  Otherwise the
multiplication computes the only live low child using `1 - point`. -/
def lowTerms (blockCount round : Nat) (parents : List KTerms)
    (level : TensorLevel) : List KTerms :=
  (List.range parents.length).map fun parent =>
    let input := parents.getD parent tensorRoot
    let multiplication := tensorTraceAt level parent
    if highLive blockCount round parent then
      { c0 := input.c0 ++
          [(multiplication.output.c0, goldilocksP - 1)]
        c1 := input.c1 ++
          [(multiplication.output.c1, goldilocksP - 1)] }
    else
      KTerms.ofColumns multiplication.output

def highTerms (blockCount round : Nat) (parents : List KTerms)
    (level : TensorLevel) : List KTerms :=
  (List.range parents.length).filterMap fun parent =>
    if highLive blockCount round parent then
      some (KTerms.ofColumns
        (tensorTraceAt level parent).output)
    else none

/-- Exact output order of `build_chi_tensor`: all live lows, then all live
highs. -/
def nextTensorTerms (blockCount round : Nat) (parents : List KTerms)
    (level : TensorLevel) : List KTerms :=
  lowTerms blockCount round parents level ++
    highTerms blockCount round parents level

def tensorTermsFrom (blockCount : Nat) : Nat -> List KTerms ->
    List TensorLevel -> List KTerms
  | _, parents, [] => parents
  | round, parents, level :: tail =>
      tensorTermsFrom blockCount (round + 1)
        (nextTensorTerms blockCount round parents level) tail

def tensorTerms (blockCount : Nat) (levels : List TensorLevel) :
    List KTerms :=
  tensorTermsFrom blockCount 0 [tensorRoot] levels

/-- Complete column-parametric row layout.  Raw witness columns are child
major and logical-coordinate minor. -/
structure Layout where
  radix : Nat
  childCount : Nat
  activeLanes : Nat
  logicalWidth : Nat
  blockVariables : Nat
  oldBlock : Fin blockVariables -> KColumns
  parent : Fin activeLanes -> KColumns
  childWitnessFirst : Fin childCount -> Nat
  productFirst : Nat
  tensorLevels : List TensorLevel

def blockCount (layout : Layout) : Nat :=
  (layout.logicalWidth + layout.activeLanes - 1) / layout.activeLanes

/-- Value of a verifier point coordinate, with a total fallback used only to
keep the compact semantic recursion proof-free.  A valid tensor schedule
never takes the fallback branch. -/
def oldBlockValue (layout : Layout) (assignment : Nat -> Nat)
    (round : Nat) : K :=
  if within : round < layout.blockVariables then
    (layout.oldBlock ⟨round, within⟩).value assignment
  else
    K.zero

/-- One semantic compact-prefix round.  The output order is the production
order: every low child first, followed by the live high children. -/
def nextTensorValues (blockCount round : Nat) (point : K)
    (parents : List K) : List K :=
  (List.range parents.length).map (fun parent =>
      K.mul (parents.getD parent K.one) (K.sub K.one point)) ++
    (List.range parents.length).filterMap (fun parent =>
      if highLive blockCount round parent then
        some (K.mul (parents.getD parent K.one) point)
      else
        none)

/-- Proof-sized semantic execution of the compact-prefix tensor schedule.
The `TensorLevel` values contribute only the number of scheduled rounds;
their exact columns and operands are checked separately by
`TensorScheduleValid`. -/
def tensorValuesFrom (layout : Layout) (assignment : Nat -> Nat) :
    Nat -> List K -> List TensorLevel -> List K
  | _, parents, [] => parents
  | round, parents, _ :: tail =>
      tensorValuesFrom layout assignment (round + 1)
        (nextTensorValues (blockCount layout) round
          (oldBlockValue layout assignment round) parents) tail

def tensorValues (layout : Layout) (assignment : Nat -> Nat) : List K :=
  tensorValuesFrom layout assignment 0 [K.one] layout.tensorLevels

/-- Syntactic shape of one compact-prefix round.  These equalities identify
the exact operands of every multiplication; they do not mention an
assignment or the values of any columns. -/
structure TensorLevelValid (layout : Layout) (round : Nat)
    (parents : List KTerms) (level : TensorLevel) : Prop where
  roundWithin : round < layout.blockVariables
  parentWidth : parents.length = Nat.min (blockCount layout) (2 ^ round)
  multiplicationWidth : level.multiplicationCount = parents.length
  operands : forall parent : Fin parents.length,
    let trace := level.trace
      ⟨parent.val, by rw [multiplicationWidth]; exact parent.isLt⟩
    trace.left = parents.get parent /\
      trace.right =
        if highLive (blockCount layout) round parent.val then
          pointTerms (layout.oldBlock ⟨round, roundWithin⟩)
        else
          oneMinusPointTerms (layout.oldBlock ⟨round, roundWithin⟩)

/-- Recursive, proof-sized certificate that the levels are precisely the
Rust compact-prefix tensor schedule. -/
def TensorScheduleValidFrom (layout : Layout) :
    Nat -> List KTerms -> List TensorLevel -> Prop
  | round, parents, [] =>
      round = layout.blockVariables /\
        parents.length = blockCount layout
  | round, parents, level :: tail =>
      TensorLevelValid layout round parents level /\
        TensorScheduleValidFrom layout (round + 1)
          (nextTensorTerms (blockCount layout) round parents level) tail

def TensorScheduleValid (layout : Layout) : Prop :=
  TensorScheduleValidFrom layout 0 [tensorRoot] layout.tensorLevels

def radixCoefficient (layout : Layout) (child : Nat) : Nat :=
  layout.radix ^ child % goldilocksP

/-- Child-major, lane-major raw `WitnessMat` column map.  Logical
coordinates are block-major (`block * activeLanes + lane`), whereas
`FinalWitnessWires::entry` stores one child's matrix row-major as
`lane * blockCount + block`. -/
def rawWitnessColumn (layout : Layout) (child : Fin layout.childCount)
    (coordinate : Fin layout.logicalWidth) : Nat :=
  layout.childWitnessFirst child +
    (coordinate.val % layout.activeLanes) * blockCount layout +
      coordinate.val / layout.activeLanes

/-- Two consecutive base-product outputs per live logical coordinate. -/
def productColumns (layout : Layout)
    (coordinate : Fin layout.logicalWidth) : KColumns :=
  let ordinal :=
    (coordinate.val % layout.activeLanes) * blockCount layout +
      coordinate.val / layout.activeLanes
  { c0 := layout.productFirst + 2 * ordinal
    c1 := layout.productFirst + 2 * ordinal + 1 }

def rawTerms (layout : Layout) (coordinate : Fin layout.logicalWidth) :
    List (Nat × Nat) :=
  List.ofFn fun child : Fin layout.childCount =>
    (rawWitnessColumn layout child coordinate,
      radixCoefficient layout child.val)

def coordinateBlock (layout : Layout)
    (coordinate : Fin layout.logicalWidth) : Nat :=
  coordinate.val / layout.activeLanes

def coordinateChiTerms (layout : Layout)
    (coordinate : Fin layout.logicalWidth) : KTerms :=
  (tensorTerms (blockCount layout) layout.tensorLevels).getD
    (coordinateBlock layout coordinate) tensorRoot

/-- Two base multiplication rows for one live logical coordinate. -/
def coordinateDefinitions (layout : Layout)
    (coordinate : Fin layout.logicalWidth) : List Definition :=
  let chi := coordinateChiTerms layout coordinate
  let raw := rawTerms layout coordinate
  let output := productColumns layout coordinate
  [⟨output.c0, .product raw chi.c0⟩,
   ⟨output.c1, .product raw chi.c1⟩]

def laneCoordinates (layout : Layout) (lane : Fin layout.activeLanes) :
    List (Fin layout.logicalWidth) :=
  (List.range (blockCount layout)).filterMap fun block =>
    let coordinate := block * layout.activeLanes + lane.val
    if inRange : coordinate < layout.logicalWidth then
      some ⟨coordinate, inRange⟩
    else none

def laneTerms (layout : Layout) (limb : KColumns -> Nat)
    (lane : Fin layout.activeLanes) : List (Nat × Nat) :=
  (laneCoordinates layout lane).map fun coordinate =>
    (limb (productColumns layout coordinate), 1)

def terminalRowsFor (layout : Layout)
    (lane : Fin layout.activeLanes) : List Row :=
  [builderLinearRow (layout.parent lane).c0
      (laneTerms layout KColumns.c0 lane),
   builderLinearRow (layout.parent lane).c1
      (laneTerms layout KColumns.c1 lane)]

/-- One exact row of a tensor multiplication. -/
structure TensorRowIndex (layout : Layout) where
  level : Fin layout.tensorLevels.length
  multiplication : Fin
    (layout.tensorLevels.get level).multiplicationCount
  definition : Fin 5

/-- Compact index of all physical rows.  This representation stays small
even when the production profile has more than twenty-five million rows. -/
inductive RowIndex (layout : Layout) where
  | tensor (index : TensorRowIndex layout)
  | coordinate (coordinate : Fin layout.logicalWidth) (limb : Fin 2)
  | terminal (lane : Fin layout.activeLanes) (limb : Fin 2)

def tensorTrace {layout : Layout} (index : TensorRowIndex layout) : KMulTrace :=
  (layout.tensorLevels.get index.level).trace index.multiplication

def expectedRow {layout : Layout} : RowIndex layout -> Row
  | .tensor index =>
      ((tensorTrace index).definitions.get
        ⟨index.definition.val, by
          simp [KMulTrace.definitions]⟩).builderRow
  | .coordinate coordinate limb =>
      ((coordinateDefinitions layout coordinate).get
        ⟨limb.val, by simp [coordinateDefinitions]⟩).builderRow
  | .terminal lane limb =>
      (terminalRowsFor layout lane).get
        ⟨limb.val, by simp [terminalRowsFor]⟩

/-- Indexed satisfaction of the actual compiler rows.  No production-sized
`List Row` is constructed or normalized. -/
def RowsSatisfied (layout : Layout) (assignment : Nat -> Nat) : Prop :=
  forall index : RowIndex layout, RowHolds assignment (expectedRow index)

/-- Tensor-only satisfaction, used by compiler variants that share the compact
prefix schedule but own a different terminal family. -/
def TensorRowsSatisfied (layout : Layout) (assignment : Nat -> Nat) : Prop :=
  forall index : TensorRowIndex layout,
    RowHolds assignment (expectedRow (.tensor index))

/-- Coordinate-product satisfaction, independent of how a compiler consumes
the lane-major product sums. -/
def CoordinateRowsSatisfied (layout : Layout)
    (assignment : Nat -> Nat) : Prop :=
  forall coordinate : Fin layout.logicalWidth, forall limb : Fin 2,
    RowHolds assignment (expectedRow (.coordinate coordinate limb))

theorem RowsSatisfied.tensor
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfies : RowsSatisfied layout assignment) :
    TensorRowsSatisfied layout assignment :=
  fun index => satisfies (.tensor index)

theorem RowsSatisfied.coordinate
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfies : RowsSatisfied layout assignment) :
    CoordinateRowsSatisfied layout assignment :=
  fun coordinate limb => satisfies (.coordinate coordinate limb)

/-- Pure layout facts.  In particular, this structure contains neither an
assignment nor a semantic acceptance claim. -/
structure ShapeValid (layout : Layout) : Prop where
  positiveLanes : 0 < layout.activeLanes
  levelCount : layout.tensorLevels.length = layout.blockVariables
  tensorSchedule : TensorScheduleValid layout
  finalTensorWidth :
    (tensorTerms (blockCount layout) layout.tensorLevels).length =
      blockCount layout
  tensorDefinitionCanonical :
    forall level, level ∈ layout.tensorLevels ->
      forall multiplication : Fin level.multiplicationCount,
        let trace := level.trace multiplication
        forall definition, definition ∈ trace.definitions ->
          definition.Canonical
  rawCoefficientsCanonical :
    forall child, child < layout.childCount ->
      0 < radixCoefficient layout child /\
        radixCoefficient layout child < goldilocksP
  tensorTraceShape :
    forall level, level ∈ layout.tensorLevels ->
      forall multiplication : Fin level.multiplicationCount,
        (level.trace multiplication).SumLayoutValid

theorem rawTerms_canonical
    {layout : Layout} (valid : ShapeValid layout)
    (coordinate : Fin layout.logicalWidth) :
    CanonicalTerms (rawTerms layout coordinate) := by
  intro term member
  rcases List.mem_ofFn.mp member with ⟨child, rfl⟩
  exact valid.rawCoefficientsCanonical child.val child.isLt

theorem laneTerms_canonical
    (layout : Layout) (limb : KColumns -> Nat)
    (lane : Fin layout.activeLanes) :
    CanonicalTerms (laneTerms layout limb lane) := by
  intro term member
  rcases List.mem_map.mp member with ⟨coordinate, _, rfl⟩
  change 0 < 1 /\ 1 < goldilocksP
  exact ⟨by decide, by decide⟩

/-- Every tensor multiplication in a satisfying row program has its exact
Goldilocks-quadratic meaning.  Structural identification of its left/right
terms with a particular compact-prefix round remains a layout theorem, not an
acceptance premise. -/
theorem tensor_multiplication_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment)
    (index : TensorRowIndex layout) :
    (tensorTrace index).output.value assignment =
      K.mul ((tensorTrace index).left.value assignment)
        ((tensorTrace index).right.value assignment) := by
  let level := layout.tensorLevels.get index.level
  let trace := level.trace index.multiplication
  have levelMember : level ∈ layout.tensorLevels :=
    List.get_mem _ index.level
  apply KMulTrace.sound trace assignment
      (valid.tensorTraceShape level levelMember index.multiplication)
  intro definition definitionMember
  rcases List.mem_iff_getElem.mp definitionMember with
    ⟨definitionIndex, definitionLt, definitionEq⟩
  have definitionLtFive : definitionIndex < 5 := by
    simpa [KMulTrace.definitions] using definitionLt
  let rowIndex : TensorRowIndex layout :=
    { level := index.level
      multiplication := index.multiplication
      definition := ⟨definitionIndex, definitionLtFive⟩ }
  let definitionAt := trace.definitions.get
    ⟨definitionIndex, definitionLt⟩
  have rowHolds := satisfies rowIndex
  have canonicalDefinition : definitionAt.Canonical :=
    valid.tensorDefinitionCanonical level levelMember index.multiplication
      definitionAt (List.get_mem _ _)
  have holdsAt : Definition.Holds assignment definitionAt := by
    apply builderDefinition_sound canonical one definitionAt
      canonicalDefinition
    simpa [expectedRow, tensorTrace, rowIndex, definitionAt, level, trace]
      using rowHolds
  rw [← definitionEq]
  exact holdsAt

/-- The two product rows bind one derived K value to the radix-weighted raw
child LC times the corresponding chi term. -/
theorem coordinate_product_sound_of_coordinateRows
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : CoordinateRowsSatisfied layout assignment)
    (coordinate : Fin layout.logicalWidth) :
    (productColumns layout coordinate).value assignment =
      K.mul
        (K.ofBase (residue (lcEval assignment
          (rawTerms layout coordinate))))
        ((coordinateChiTerms layout coordinate).value assignment) := by
  let chi := coordinateChiTerms layout coordinate
  let raw := rawTerms layout coordinate
  let output := productColumns layout coordinate
  have c0 : Definition.Holds assignment
      ⟨output.c0, .product raw chi.c0⟩ := by
    apply builderDefinition_sound canonical one _ (by trivial)
    simpa [expectedRow, coordinateDefinitions, chi, raw, output] using
      satisfies coordinate ⟨0, by decide⟩
  have c1 : Definition.Holds assignment
      ⟨output.c1, .product raw chi.c1⟩ := by
    apply builderDefinition_sound canonical one _ (by trivial)
    simpa [expectedRow, coordinateDefinitions, chi, raw, output] using
      satisfies coordinate ⟨1, by decide⟩
  simp only [Definition.Holds, Rhs.eval] at c0 c1
  change output.value assignment =
    K.mul (K.ofBase (residue (lcEval assignment raw)))
      (chi.value assignment)
  simp only [KColumns.value, KTerms.value, K.ofBase, K.mul, K.mk.injEq,
    Fin.zero_mul, Fin.mul_zero, Fin.add_zero]
  constructor
  · apply Fin.ext
    simp only [baseAt, residue, Fin.val_mul]
    rw [c0]
    simp [Nat.mul_mod]
  · apply Fin.ext
    simp only [baseAt, residue, Fin.val_mul]
    rw [c1]
    simp [Nat.mul_mod]

theorem coordinate_product_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (coordinate : Fin layout.logicalWidth) :
    (productColumns layout coordinate).value assignment =
      K.mul
        (K.ofBase (residue (lcEval assignment
          (rawTerms layout coordinate))))
        ((coordinateChiTerms layout coordinate).value assignment) := by
  exact coordinate_product_sound_of_coordinateRows valid canonical one
    satisfies.coordinate coordinate

/-- Value of the exact terminal accumulation LCs. -/
def projectedLane (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin layout.activeLanes) : K where
  c0 := residue (lcEval assignment (laneTerms layout KColumns.c0 lane))
  c1 := residue (lcEval assignment (laneTerms layout KColumns.c1 lane))

private theorem rawLcEval_append (assignment : Nat -> Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem residue_lcEval_append (assignment : Nat -> Nat)
    (left right : List (Nat × Nat)) :
    residue (lcEval assignment (left ++ right)) =
      residue (lcEval assignment left) +
        residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue, Fin.val_add]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_append]
  simp [Nat.add_mod]

private theorem residue_lcEval_negativeColumn
    (assignment : Nat -> Nat) (column : Nat) :
    residue (lcEval assignment [(column, goldilocksP - 1)]) =
      -(baseAt assignment column) := by
  have negOne : residue (goldilocksP - 1) = (-1 : F) := by
    apply Fin.ext
    rfl
  calc
    residue (lcEval assignment [(column, goldilocksP - 1)]) =
        residue (goldilocksP - 1) * baseAt assignment column := by
      apply Fin.ext
      simp only [residue, baseAt, Fin.val_mul, lcEval, List.foldl,
        Nat.zero_add]
      simpa only [Nat.mod_mod] using
        (Nat.mul_mod (goldilocksP - 1) (assignment column) goldilocksP)
    _ = (-1 : F) * baseAt assignment column := by rw [negOne]
    _ = -(baseAt assignment column) := by
      calc
        (-1 : F) * baseAt assignment column =
            -(1 * baseAt assignment column) :=
          Lean.Grind.Fin.neg_mul 1 _
        _ = -(baseAt assignment column) := by rw [Fin.one_mul]

theorem oneMinusPointTerms_value
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (point : KColumns) :
    (oneMinusPointTerms point).value assignment =
      K.sub K.one (point.value assignment) := by
  apply k_ext
  · change residue (lcEval assignment
        ([(0, 1)] ++ [(point.c0, goldilocksP - 1)])) = _
    rw [residue_lcEval_append, residue_lcEval_negativeColumn]
    have root : residue (lcEval assignment [(0, 1)]) = (1 : F) := by
      apply Fin.ext
      simp only [lcEval, List.foldl, one, Nat.mul_one, Nat.zero_add,
        residue]
      decide
    rw [root]
    simp [K.sub, K.one, KColumns.value, Fin.sub_eq_add_neg]
  · change residue (lcEval assignment
        [(point.c1, goldilocksP - 1)]) = _
    rw [residue_lcEval_negativeColumn]
    simp [K.sub, K.one, KColumns.value, Fin.sub_eq_add_neg]

theorem differenceTerms_value
    (assignment : Nat -> Nat) (left : KTerms) (right : KColumns) :
    ({ c0 := left.c0 ++ [(right.c0, goldilocksP - 1)]
       c1 := left.c1 ++ [(right.c1, goldilocksP - 1)] } : KTerms).value
        assignment =
      K.sub (left.value assignment) (right.value assignment) := by
  apply k_ext
  · change residue (lcEval assignment
        (left.c0 ++ [(right.c0, goldilocksP - 1)])) = _
    rw [residue_lcEval_append, residue_lcEval_negativeColumn]
    simp [K.sub, KTerms.value, KColumns.value, Fin.sub_eq_add_neg]
  · change residue (lcEval assignment
        (left.c1 ++ [(right.c1, goldilocksP - 1)])) = _
    rw [residue_lcEval_append, residue_lcEval_negativeColumn]
    simp [K.sub, KTerms.value, KColumns.value, Fin.sub_eq_add_neg]

theorem tensorRoot_value (assignment : Nat -> Nat)
    (one : assignment 0 = 1) :
    tensorRoot.value assignment = K.one := by
  apply k_ext
  · apply Fin.ext
    simp only [tensorRoot, KTerms.value, K.one, lcEval, List.foldl,
      one, Nat.mul_one, Nat.zero_add, residue]
    decide
  · apply Fin.ext
    simp [tensorRoot, KTerms.value, K.one, lcEval, residue]

/-- Member-facing form of `tensor_multiplication_sound`, convenient for the
structural induction over a suffix of the compact schedule. -/
theorem tensor_multiplication_sound_of_mem
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment)
    (level : TensorLevel) (levelMember : level ∈ layout.tensorLevels)
    (multiplication : Fin level.multiplicationCount) :
    (level.trace multiplication).output.value assignment =
      K.mul ((level.trace multiplication).left.value assignment)
        ((level.trace multiplication).right.value assignment) := by
  rcases List.mem_iff_getElem.mp levelMember with
    ⟨levelIndex, levelIndexLt, rfl⟩
  let index : TensorRowIndex layout :=
    { level := ⟨levelIndex, levelIndexLt⟩
      multiplication := multiplication
      definition := ⟨0, by decide⟩ }
  exact tensor_multiplication_sound valid canonical one satisfies index

private theorem termsValue_getD
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (parents : List KTerms) (index : Nat) :
    (parents.getD index tensorRoot).value assignment =
      (parents.map fun terms => terms.value assignment).getD index K.one := by
  by_cases inRange : index < parents.length
  · have mappedInRange :
        index < (parents.map fun terms => terms.value assignment).length := by
      simpa using inRange
    simp [List.getD, inRange, mappedInRange]
  · have mappedOutOfRange :
        ¬index < (parents.map fun terms => terms.value assignment).length := by
      simpa using inRange
    simp [List.getD, inRange, mappedOutOfRange,
      tensorRoot_value assignment one]

/-- Semantic value of the unique multiplication owned by one live parent in
one compact-prefix round. -/
theorem tensor_trace_output_value
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment)
    {round : Nat} {parents : List KTerms} {level : TensorLevel}
    (levelValid : TensorLevelValid layout round parents level)
    (levelMember : level ∈ layout.tensorLevels)
    (parent : Fin parents.length) :
    let multiplication : Fin level.multiplicationCount :=
      ⟨parent.val, by
        rw [levelValid.multiplicationWidth]
        exact parent.isLt⟩
    let trace := level.trace multiplication
    trace.output.value assignment =
      if highLive (blockCount layout) round parent.val then
        K.mul ((parents.get parent).value assignment)
          (oldBlockValue layout assignment round)
      else
        K.mul ((parents.get parent).value assignment)
          (K.sub K.one (oldBlockValue layout assignment round)) := by
  let multiplication : Fin level.multiplicationCount :=
    ⟨parent.val, by
      rw [levelValid.multiplicationWidth]
      exact parent.isLt⟩
  let trace := level.trace multiplication
  change trace.output.value assignment =
    if highLive (blockCount layout) round parent.val then
      K.mul ((parents.get parent).value assignment)
        (oldBlockValue layout assignment round)
    else
      K.mul ((parents.get parent).value assignment)
        (K.sub K.one (oldBlockValue layout assignment round))
  have operands := levelValid.operands parent
  have leftOperand : trace.left = parents.get parent := by
    simpa [trace, multiplication] using operands.1
  have multiplicationSound :
      trace.output.value assignment =
        K.mul (trace.left.value assignment) (trace.right.value assignment) := by
    exact tensor_multiplication_sound_of_mem valid canonical one satisfies
      level levelMember multiplication
  have pointEq :
      oldBlockValue layout assignment round =
        (layout.oldBlock ⟨round, levelValid.roundWithin⟩).value assignment := by
    simp [oldBlockValue, levelValid.roundWithin]
  by_cases live : highLive (blockCount layout) round parent.val = true
  · have rightOperand :
        trace.right =
          pointTerms (layout.oldBlock ⟨round, levelValid.roundWithin⟩) := by
      simpa [trace, multiplication, live] using operands.2
    rw [multiplicationSound, leftOperand, rightOperand]
    simp [live, pointTerms, pointEq, KTerms.ofColumns_value]
  · have rightOperand :
        trace.right =
          oneMinusPointTerms
            (layout.oldBlock ⟨round, levelValid.roundWithin⟩) := by
      simpa [trace, multiplication, live] using operands.2
    rw [multiplicationSound, leftOperand, rightOperand,
      oneMinusPointTerms_value assignment one, pointEq]
    simp [live]

private theorem filterMap_congr_mem
    {Input Output : Type} (indices : List Input)
    (left right : Input -> Option Output)
    (equal : forall index, index ∈ indices -> left index = right index) :
    indices.filterMap left = indices.filterMap right := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      simp only [List.filterMap_cons]
      rw [equal index (by simp)]
      rw [inductionHypothesis (fun current member =>
        equal current (by simp [member]))]

/-- One structurally valid tensor level maps the symbolic terms to the exact
semantic low/high compact-prefix values. -/
theorem nextTensorTerms_values
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment)
    {round : Nat} {parents : List KTerms} {level : TensorLevel}
    (levelValid : TensorLevelValid layout round parents level)
    (levelMember : level ∈ layout.tensorLevels) :
    (nextTensorTerms (blockCount layout) round parents level).map
        (fun terms => terms.value assignment) =
      nextTensorValues (blockCount layout) round
        (oldBlockValue layout assignment round)
        (parents.map fun terms => terms.value assignment) := by
  unfold nextTensorTerms nextTensorValues
  rw [List.map_append]
  congr 1
  · simp only [lowTerms, List.map_map, List.length_map,
      Function.comp_apply]
    apply List.map_congr_left
    intro parent parentMember
    have parentLt : parent < parents.length := List.mem_range.mp parentMember
    let parentFin : Fin parents.length := ⟨parent, parentLt⟩
    let multiplication : Fin level.multiplicationCount :=
      ⟨parent, by
        rw [levelValid.multiplicationWidth]
        exact parentLt⟩
    let trace := level.trace multiplication
    have traceAt : tensorTraceAt level parent = trace := by
      simp [tensorTraceAt, trace, multiplication,
        levelValid.multiplicationWidth, parentLt]
    have outputValue :
        trace.output.value assignment =
          if highLive (blockCount layout) round parent then
            K.mul ((parents.get parentFin).value assignment)
              (oldBlockValue layout assignment round)
          else
            K.mul ((parents.get parentFin).value assignment)
              (K.sub K.one (oldBlockValue layout assignment round)) := by
      simpa [trace, multiplication, parentFin] using
        tensor_trace_output_value valid canonical one satisfies levelValid
          levelMember parentFin
    have parentValue :
        (parents.getD parent tensorRoot).value assignment =
          (parents.map fun terms => terms.value assignment).getD parent K.one :=
      termsValue_getD assignment one parents parent
    have parentGetValue :
        (parents.get parentFin).value assignment =
          (parents.map fun terms => terms.value assignment).getD parent K.one := by
      rw [← parentValue]
      simp [List.getD, parentLt, parentFin]
    by_cases live : highLive (blockCount layout) round parent = true
    · simp only [Function.comp_apply, live, if_true]
      rw [traceAt, differenceTerms_value, outputValue, parentValue,
        parentGetValue]
      simp only [live, if_true]
      exact K.sub_mul_self _ _
    · have dead : highLive (blockCount layout) round parent = false :=
        Bool.eq_false_iff.mpr live
      have outputValue' :
          trace.output.value assignment =
            K.mul ((parents.get parentFin).value assignment)
              (K.sub K.one (oldBlockValue layout assignment round)) := by
        simpa [dead] using outputValue
      simp only [Function.comp_apply]
      simp only [dead, Bool.false_eq_true, if_false]
      rw [traceAt, KTerms.ofColumns_value, outputValue', parentGetValue]
  · unfold highTerms
    rw [List.map_filterMap]
    simp only [List.length_map]
    apply filterMap_congr_mem (List.range parents.length)
    intro parent parentMember
    have parentLt : parent < parents.length := List.mem_range.mp parentMember
    let parentFin : Fin parents.length := ⟨parent, parentLt⟩
    let multiplication : Fin level.multiplicationCount :=
      ⟨parent, by
        rw [levelValid.multiplicationWidth]
        exact parentLt⟩
    let trace := level.trace multiplication
    have traceAt : tensorTraceAt level parent = trace := by
      simp [tensorTraceAt, trace, multiplication,
        levelValid.multiplicationWidth, parentLt]
    have outputValue :
        trace.output.value assignment =
          if highLive (blockCount layout) round parent then
            K.mul ((parents.get parentFin).value assignment)
              (oldBlockValue layout assignment round)
          else
            K.mul ((parents.get parentFin).value assignment)
              (K.sub K.one (oldBlockValue layout assignment round)) := by
      simpa [trace, multiplication, parentFin] using
        tensor_trace_output_value valid canonical one satisfies levelValid
          levelMember parentFin
    have parentValue :
        (parents.getD parent tensorRoot).value assignment =
          (parents.map fun terms => terms.value assignment).getD parent K.one :=
      termsValue_getD assignment one parents parent
    have parentGetValue :
        (parents.get parentFin).value assignment =
          (parents.map fun terms => terms.value assignment).getD parent K.one := by
      rw [← parentValue]
      simp [List.getD, parentLt, parentFin]
    by_cases live : highLive (blockCount layout) round parent = true
    · simp only [live, if_true, Option.map_some]
      rw [traceAt, KTerms.ofColumns_value, outputValue]
      simp only [live, if_true]
      rw [parentGetValue]
    · have dead : highLive (blockCount layout) round parent = false :=
        Bool.eq_false_iff.mpr live
      simp [dead]

private theorem tensorTermsFrom_values
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment) :
    forall {round : Nat} {parents : List KTerms} {levels : List TensorLevel},
      TensorScheduleValidFrom layout round parents levels ->
      (forall level, level ∈ levels -> level ∈ layout.tensorLevels) ->
      (tensorTermsFrom (blockCount layout) round parents levels).map
          (fun terms => terms.value assignment) =
        tensorValuesFrom layout assignment round
          (parents.map fun terms => terms.value assignment) levels
  | _, _, [], _, _ => rfl
  | round, parents, level :: tail, schedule, contained => by
      rcases schedule with ⟨levelValid, tailSchedule⟩
      have levelMember : level ∈ layout.tensorLevels :=
        contained level (by simp)
      have tailContained :
          forall current, current ∈ tail -> current ∈ layout.tensorLevels := by
        intro current member
        exact contained current (by simp [member])
      rw [tensorTermsFrom, tensorValuesFrom]
      rw [tensorTermsFrom_values valid canonical one satisfies tailSchedule
        tailContained]
      rw [nextTensorTerms_values valid canonical one satisfies levelValid
        levelMember]

/-- All satisfying tensor rows evaluate the compiler's compact symbolic
tensor to its proof-sized semantic execution. -/
theorem tensorTerms_values_eq_tensorValues
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment) :
    (tensorTerms (blockCount layout) layout.tensorLevels).map
        (fun terms => terms.value assignment) =
      tensorValues layout assignment := by
  simpa [tensorTerms, tensorValues, tensorRoot_value assignment one] using
    (tensorTermsFrom_values valid canonical one satisfies
      valid.tensorSchedule
      (fun level member => member))

/-- The chi term selected for one logical coordinate is the corresponding
entry of the semantic compact tensor forced by the actual tensor rows. -/
theorem coordinateChiTerms_value_eq_tensorValue_of_tensorRows
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : TensorRowsSatisfied layout assignment)
    (coordinate : Fin layout.logicalWidth) :
    (coordinateChiTerms layout coordinate).value assignment =
      (tensorValues layout assignment).getD
        (coordinateBlock layout coordinate) K.one := by
  unfold coordinateChiTerms
  calc
    ((tensorTerms (blockCount layout) layout.tensorLevels).getD
        (coordinateBlock layout coordinate) tensorRoot).value assignment =
      ((tensorTerms (blockCount layout) layout.tensorLevels).map
        (fun terms => terms.value assignment)).getD
          (coordinateBlock layout coordinate) K.one :=
        termsValue_getD assignment one _ _
    _ = (tensorValues layout assignment).getD
          (coordinateBlock layout coordinate) K.one := by
      rw [tensorTerms_values_eq_tensorValues valid canonical one satisfies]

theorem coordinateChiTerms_value_eq_tensorValue
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (coordinate : Fin layout.logicalWidth) :
    (coordinateChiTerms layout coordinate).value assignment =
      (tensorValues layout assignment).getD
        (coordinateBlock layout coordinate) K.one := by
  exact coordinateChiTerms_value_eq_tensorValue_of_tensorRows valid canonical
    one satisfies.tensor coordinate

private theorem residue_lcEval_unitMap
    (assignment : Nat -> Nat) {Index : Type}
    (indices : List Index) (column : Index -> Nat) :
    residue (lcEval assignment
      (indices.map fun index => (column index, 1))) =
      indices.foldr
        (fun index suffix => baseAt assignment (column index) + suffix) 0 := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      change residue (lcEval assignment
          ([(column index, 1)] ++
            tail.map fun current => (column current, 1))) = _
      rw [residue_lcEval_append, inductionHypothesis]
      simp [lcEval, baseAt, residue]

private theorem foldr_kAdd_c0 {Index : Type} (indices : List Index)
    (value : Index -> K) :
    (indices.foldr (fun index suffix => K.add (value index) suffix)
      K.zero).c0 =
      indices.foldr (fun index suffix => (value index).c0 + suffix) 0 := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      change (value index).c0 +
          (tail.foldr (fun current suffix => K.add (value current) suffix)
            K.zero).c0 =
        (value index).c0 +
          tail.foldr (fun current suffix => (value current).c0 + suffix) 0
      rw [inductionHypothesis]

private theorem foldr_kAdd_c1 {Index : Type} (indices : List Index)
    (value : Index -> K) :
    (indices.foldr (fun index suffix => K.add (value index) suffix)
      K.zero).c1 =
      indices.foldr (fun index suffix => (value index).c1 + suffix) 0 := by
  induction indices with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      change (value index).c1 +
          (tail.foldr (fun current suffix => K.add (value current) suffix)
            K.zero).c1 =
        (value index).c1 +
          tail.foldr (fun current suffix => (value current).c1 + suffix) 0
      rw [inductionHypothesis]

/-- Direct semantic projection of the authoritative raw input columns using
the chi values forced by the tensor rows. -/
def decodedRawProjection (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin layout.activeLanes) : K :=
  (laneCoordinates layout lane).foldr
    (fun coordinate suffix =>
      K.add
        (K.mul
          (K.ofBase (residue (lcEval assignment
            (rawTerms layout coordinate))))
          ((coordinateChiTerms layout coordinate).value assignment))
        suffix)
    K.zero

theorem projectedLane_eq_productFold
    (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin layout.activeLanes) :
    projectedLane layout assignment lane =
      (laneCoordinates layout lane).foldr
        (fun coordinate suffix =>
          K.add ((productColumns layout coordinate).value assignment) suffix)
        K.zero := by
  apply k_ext
  · change residue (lcEval assignment
        ((laneCoordinates layout lane).map fun coordinate =>
          ((productColumns layout coordinate).c0, 1))) = _
    rw [foldr_kAdd_c0]
    exact residue_lcEval_unitMap assignment
      (laneCoordinates layout lane)
      (fun coordinate => (productColumns layout coordinate).c0)
  · change residue (lcEval assignment
        ((laneCoordinates layout lane).map fun coordinate =>
          ((productColumns layout coordinate).c1, 1))) = _
    rw [foldr_kAdd_c1]
    exact residue_lcEval_unitMap assignment
      (laneCoordinates layout lane)
      (fun coordinate => (productColumns layout coordinate).c1)

/-- Exact row soundness for the terminal equality family. -/
theorem rows_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment) :
    forall lane, (layout.parent lane).value assignment =
      projectedLane layout assignment lane := by
  intro lane
  have row0 : RowHolds assignment
      (builderLinearRow (layout.parent lane).c0
        (laneTerms layout KColumns.c0 lane)) := by
    simpa [expectedRow, terminalRowsFor] using
      satisfies (.terminal lane ⟨0, by decide⟩)
  have row1 : RowHolds assignment
      (builderLinearRow (layout.parent lane).c1
        (laneTerms layout KColumns.c1 lane)) := by
    simpa [expectedRow, terminalRowsFor] using
      satisfies (.terminal lane ⟨1, by decide⟩)
  have c0 := builderLinearRow_sound canonical one _ _
    (laneTerms_canonical layout KColumns.c0 lane) row0
  have c1 := builderLinearRow_sound canonical one _ _
    (laneTerms_canonical layout KColumns.c1 lane) row1
  simp only [KColumns.value, projectedLane, K.mk.injEq]
  exact ⟨congrArg residue c0, congrArg residue c1⟩

/-- Composition of the exact tensor/product/terminal row families up to the
decoded raw-column projection.  No product column remains in the conclusion. -/
theorem rows_imply_decodedRawProjection
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment) :
    forall lane, (layout.parent lane).value assignment =
      decodedRawProjection layout assignment lane := by
  intro lane
  rw [rows_sound valid canonical one satisfies lane,
    projectedLane_eq_productFold]
  unfold decodedRawProjection
  let coordinates := laneCoordinates layout lane
  change coordinates.foldr
      (fun coordinate suffix =>
        K.add ((productColumns layout coordinate).value assignment) suffix)
      K.zero =
    coordinates.foldr
      (fun coordinate suffix =>
        K.add
          (K.mul
            (K.ofBase (residue (lcEval assignment
              (rawTerms layout coordinate))))
            ((coordinateChiTerms layout coordinate).value assignment))
          suffix)
      K.zero
  induction coordinates with
  | nil => rfl
  | cons coordinate tail inductionHypothesis =>
      simp only [List.foldr_cons]
      rw [coordinate_product_sound valid canonical one satisfies coordinate,
        inductionHypothesis]

/-- The ten non-Phi81 lanes are verifier-computed zero and own no rows. -/
def decodedPaddedProjection (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin 64) : K :=
  if active : lane.val < layout.activeLanes then
    decodedRawProjection layout assignment ⟨lane.val, active⟩
  else
    K.zero

theorem decodedPaddedProjection_padding_zero
    (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin 64) (padding : layout.activeLanes <= lane.val) :
    decodedPaddedProjection layout assignment lane = K.zero := by
  simp [decodedPaddedProjection, Nat.not_lt.mpr padding]

/-- Local honest-completeness theorem.  Its premises are the explicit SSA
equations and terminal equalities, not row satisfaction or semantic
acceptance.  A generated fixture constructs these equations by executing the
same definitions. -/
theorem rows_complete
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (tensorHolds : forall index : TensorRowIndex layout,
      Definition.Holds assignment
        ((tensorTrace index).definitions.get
          ⟨index.definition.val, by
            simp [KMulTrace.definitions]⟩))
    (coordinateHolds : forall coordinate,
      Definition.Holds assignment
          ((coordinateDefinitions layout coordinate).get
            ⟨0, by simp [coordinateDefinitions]⟩) /\
        Definition.Holds assignment
          ((coordinateDefinitions layout coordinate).get
            ⟨1, by simp [coordinateDefinitions]⟩))
    (terminalHolds : forall lane,
      assignment (layout.parent lane).c0 =
          lcEval assignment (laneTerms layout KColumns.c0 lane) /\
        assignment (layout.parent lane).c1 =
          lcEval assignment (laneTerms layout KColumns.c1 lane)) :
    RowsSatisfied layout assignment := by
  intro index
  cases index with
  | tensor tensorIndex =>
      let level := layout.tensorLevels.get tensorIndex.level
      let trace := tensorTrace tensorIndex
      let definition := trace.definitions.get
        ⟨tensorIndex.definition.val, by
          simp [KMulTrace.definitions]⟩
      apply builderDefinition_complete canonical one definition
      · apply valid.tensorDefinitionCanonical level
          (List.get_mem _ tensorIndex.level) tensorIndex.multiplication definition
        exact List.get_mem _ _
      · exact tensorHolds tensorIndex
  | coordinate coordinate limb =>
      have cases : limb = 0 ∨ limb = 1 := by omega
      rcases cases with rfl | rfl
      · apply builderDefinition_complete canonical one _ (by trivial)
        exact (coordinateHolds coordinate).1
      · apply builderDefinition_complete canonical one _ (by trivial)
        exact (coordinateHolds coordinate).2
  | terminal lane limb =>
      have cases : limb = 0 ∨ limb = 1 := by omega
      rcases cases with rfl | rfl
      · exact builderLinearRow_complete one _ _
          (laneTerms_canonical layout KColumns.c0 lane)
          (terminalHolds lane).1
      · exact builderLinearRow_complete one _ _
          (laneTerms_canonical layout KColumns.c1 lane)
          (terminalHolds lane).2

/-- Exact conceptual cardinality without constructing the row list. -/
def tensorMultiplicationCount (layout : Layout) : Nat :=
  layout.tensorLevels.foldl
    (fun count level => count + level.multiplicationCount) 0

def rowCount (layout : Layout) : Nat :=
  5 * tensorMultiplicationCount layout +
    2 * layout.logicalWidth + 2 * layout.activeLanes

/-- Artifact-facing physical identity.  The generated owner supplies a
bijective index into the conceptual rows, so every physical row has exactly
one owner without ever constructing a 25-million-element `List Row`. -/
structure ArtifactContract (layout : Layout)
    (artifactRow : Fin (rowCount layout) -> Row) where
  profileRadix : layout.radix = 2
  profileChildren : layout.childCount = 14
  profileActiveLanes : layout.activeLanes = 54
  profilePaddingLanes : 64 - layout.activeLanes = 10
  profileLogicalWidth : layout.logicalWidth = 11437038
  profileBlockVariables : layout.blockVariables = 19
  profileBlockCount : blockCount layout = 211797
  profileTensorMultiplications : tensorMultiplicationCount layout = 473940
  profileRows : rowCount layout = 25243884
  shape : ShapeValid layout
  physicalIndex : RowIndex layout -> Fin (rowCount layout)
  physicalIndex_injective : Function.Injective physicalIndex
  physicalIndex_surjective : Function.Surjective physicalIndex
  rowAt : forall index : RowIndex layout,
    artifactRow (physicalIndex index) = expectedRow index

def ArtifactRowsSatisfied
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (_contract : ArtifactContract layout artifactRow)
    (assignment : Nat -> Nat) : Prop :=
  forall index, RowHolds assignment (artifactRow index)

theorem ArtifactContract.rowsSatisfied
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    {assignment : Nat -> Nat}
    (satisfies : ArtifactRowsSatisfied contract assignment) :
    RowsSatisfied layout assignment := by
  intro index
  rw [← contract.rowAt index]
  exact satisfies (contract.physicalIndex index)

theorem artifact_rows_sound
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : ArtifactRowsSatisfied contract assignment) :
    forall lane, (layout.parent lane).value assignment =
      decodedRawProjection layout assignment lane := by
  exact rows_imply_decodedRawProjection contract.shape canonical one
    (contract.rowsSatisfied satisfies)

end Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
