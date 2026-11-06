import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Lifecycle.XOut

/-! The ordinary part of a circuit's logical relation for the generic exporter (CW-5.2): each R1CS
row becomes one row of a `ProductionRelation.Plan` through `OrdinarySourcePlan`, read through a
`Layout` of the source columns among the logical columns. `CircuitRelation` builds a step circuit's relation from
these rows and the packed Poseidon2 blocks.

The logical assignment's first `ringDegree * publicRingColumns` columns are the public input
`x` (`Phi81Relation.projectPublicInput`), so column 0 is the `encHash` marker 1 and serves as the
constant column. Four extra constraints read the output digest back from `x`
(`decodeHashWord`, as `RecursivePublicOutputPlan`). -/

namespace NightstreamFPrime.Export.Generic.LogicalPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### Column bounds -/

private theorem le_foldl_max {α : Type} (measure : α → Nat) :
    ∀ (values : List α) (initial : Nat),
      initial ≤ values.foldl (fun bound value => max bound (measure value)) initial
  | [], _ => le_refl _
  | value :: rest, initial =>
      le_trans (le_max_left initial (measure value)) (le_foldl_max measure rest _)

private theorem measure_le_foldl_max {α : Type} (measure : α → Nat) :
    ∀ (values : List α) (initial : Nat) (value : α), value ∈ values →
      measure value ≤ values.foldl (fun bound value => max bound (measure value)) initial
  | [], _, _, member => by simp at member
  | head :: rest, initial, value, member => by
      rcases List.mem_cons.mp member with rfl | member
      · exact le_trans (le_max_right initial (measure value)) (le_foldl_max measure rest _)
      · exact measure_le_foldl_max measure rest _ value member

/-- The largest column a linear combination reads. -/
def combinationMax (combination : R1CS.LinearCombination) : Nat :=
  combination.terms.foldl (fun bound term => max bound term.1) 0

def rowMax (row : R1CS.Row) : Nat :=
  max (combinationMax row.a) (max (combinationMax row.b) (combinationMax row.c))

/-- The largest column the rows read. -/
def rowsMax (rows : List R1CS.Row) : Nat :=
  rows.foldl (fun bound row => max bound (rowMax row)) 0

theorem column_le_combinationMax {combination : R1CS.LinearCombination} {term : Nat × F}
    (member : term ∈ combination.terms) : term.1 ≤ combinationMax combination :=
  measure_le_foldl_max Prod.fst combination.terms 0 term member

theorem rowMax_le_rowsMax {rows : List R1CS.Row} {row : R1CS.Row} (member : row ∈ rows) :
    rowMax row ≤ rowsMax rows :=
  measure_le_foldl_max rowMax rows 0 row member

/-- The rows fit the cube and read only columns below `width`. -/
structure Fits (rows : List R1CS.Row) (width : Nat) : Prop where
  rowCount : rows.length ≤ 2 ^ cubeVariables
  columns : rowsMax rows < width

theorem Fits.positive {rows : List R1CS.Row} {width : Nat} (fits : Fits rows width) :
    0 < width := by
  have := fits.columns
  omega

theorem Fits.rowBounded {rows : List R1CS.Row} {width : Nat} (fits : Fits rows width)
    {row : R1CS.Row} (member : row ∈ rows) : SourceCompiler.RowBounded width row := by
  have rowBound := rowMax_le_rowsMax member
  have below := fits.columns
  refine ⟨?_, ?_, ?_⟩ <;> intro term termMember
  · have := column_le_combinationMax termMember
    have : combinationMax row.a ≤ rowMax row := le_max_left _ _
    omega
  · have := column_le_combinationMax termMember
    have : combinationMax row.b ≤ rowMax row :=
      le_trans (le_max_left _ _) (le_max_right _ _)
    omega
  · have := column_le_combinationMax termMember
    have : combinationMax row.c ≤ rowMax row :=
      le_trans (le_max_right _ _) (le_max_right _ _)
    omega

/-! ### The plan -/

def program (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width) :
    OrdinarySourcePlan.Program width where
  rowCount := rows.length
  rowCount_le := fits.rowCount
  row := fun index => rows[index]
  bounded := fun index => fits.rowBounded (List.getElem_mem index.isLt)

/-- Where the source columns live among the logical columns (CW-5.11). The source map gives each
source column's form; the public source columns `0` to `269` are the logical public columns, and
logical column `0`, the `encHash` marker, is the constant column. The concrete layout gives every
other live source column a low-norm slot (`Layout/LowNormAssignment`), because SuperNeo's Π_CCS
checks that each fresh witness coordinate lies in `{-1, 0, 1}`. -/
def publicCount : Nat := ringDegree * PaperAlgebra.publicRingColumns

structure Layout (width logicalWidth : Nat) where
  map : SourceCompiler.SourceMap width logicalWidth
  publicFits : publicCount ≤ logicalWidth
  publicForm : ∀ column : Fin width, ∀ bound : column.val < publicCount,
    map.form column = SparseForm.singleton ⟨column.val, by omega⟩ 1

/-- Logical column 0, the `encHash` marker, is the constant column. -/
def Layout.one {width logicalWidth : Nat} (layout : Layout width logicalWidth) :
    Fin logicalWidth :=
  ⟨0, Nat.lt_of_lt_of_le (by decide) layout.publicFits⟩

def inputs (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width) {logicalWidth : Nat}
    (layout : Layout width logicalWidth) : (program rows width fits).Inputs logicalWidth :=
  ⟨layout.one, fun _ => layout.map⟩

/-- The logical relation's plan: one plan row per R1CS row, read through the layout. -/
def plan (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width) {logicalWidth : Nat}
    (layout : Layout width logicalWidth) : Plan logicalWidth :=
  ((program rows width fits).compile (inputs rows width fits layout)).toPlan

/-- The source environment of a logical assignment: column `c` reads its form. -/
def sourceEnv {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  fun column =>
    if bounded : column < width then (layout.map.form ⟨column, bounded⟩).eval assignment else 0

theorem sourceEnv_public {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (column : Nat)
    (bound : column < publicCount) (inside : column < width) :
    sourceEnv layout assignment column =
      assignment ⟨column, Nat.lt_of_lt_of_le bound layout.publicFits⟩ := by
  simp only [sourceEnv, inside, dite_true, layout.publicForm ⟨column, inside⟩ bound,
    SparseForm.singleton_eval, one_mul]

/-- A source column's form, and the empty form beyond the source width. -/
def Layout.form {width logicalWidth : Nat} (layout : Layout width logicalWidth) (column : Nat) :
    SparseForm logicalWidth :=
  if bounded : column < width then layout.map.form ⟨column, bounded⟩ else .empty

theorem Layout.form_eval {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (column : Nat) :
    (layout.form column).eval assignment = sourceEnv layout assignment column := by
  unfold Layout.form sourceEnv
  split
  · rfl
  · exact SparseForm.empty_eval assignment

/-- Sparse terms over source columns as a form over the logical columns. -/
def Layout.terms {width logicalWidth : Nat} (layout : Layout width logicalWidth) :
    List (Nat × F) → SparseForm logicalWidth
  | [] => .empty
  | (column, coefficient) :: rest =>
      .add (.scale coefficient (layout.form column)) (layout.terms rest)

/-- A linear combination over source columns as a form; its constant reads the constant column. -/
def Layout.combination {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (combination : R1CS.LinearCombination) : SparseForm logicalWidth :=
  .add (.singleton layout.one combination.constant) (layout.terms combination.terms)

theorem Layout.terms_eval {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) :
    ∀ terms : List (Nat × F), (layout.terms terms).eval assignment =
      (terms.map fun term => term.2 * sourceEnv layout assignment term.1).sum
  | [] => by simp [Layout.terms, SparseForm.empty_eval]
  | (column, coefficient) :: rest => by
      simp only [Layout.terms, SparseForm.add_eval, SparseForm.scale_eval, Layout.form_eval,
        Layout.terms_eval layout assignment rest, List.map_cons, List.sum_cons]

theorem Layout.combination_eval {width logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment layout.one = 1)
    (combination : R1CS.LinearCombination) :
    (layout.combination combination).eval assignment =
      combination.eval (sourceEnv layout assignment) := by
  simp only [Layout.combination, SparseForm.add_eval, SparseForm.singleton_eval, one, mul_one,
    Layout.terms_eval]
  rfl

theorem rowsZero_iff_rowsHold (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width)
    {logicalWidth : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment layout.one = 1) :
    (plan rows width fits layout).RowsZero assignment ↔
      R1CS.RowsHold (sourceEnv layout assignment) rows := by
  have preserves : ∀ index,
      OrdinarySourcePlan.SourceMap.PreservesRow ((inputs rows width fits layout).sourceMap index)
        assignment (sourceEnv layout assignment) ((program rows width fits).row index)
        ((program rows width fits).bounded index) := by
    intro index
    have bounded := (program rows width fits).bounded index
    refine ⟨?_, ?_, ?_⟩ <;> intro term member
    · simp [inputs, sourceEnv, bounded.1 term member]
    · simp [inputs, sourceEnv, bounded.2.1 term member]
    · simp [inputs, sourceEnv, bounded.2.2 term member]
  rw [plan, OrdinarySourcePlan.Program.rowsZero_iff _ _ assignment (sourceEnv layout assignment) one
    preserves]
  constructor
  · intro holds row member
    obtain ⟨index, bound, rfl⟩ := List.getElem_of_mem member
    exact holds ⟨index, bound⟩
  · intro holds index
    exact holds _ (List.getElem_mem index.isLt)

/-! ### Reading the output digest back from the public input -/

/-- The public cell of bit `bit` of digest word `word` (`digestBitIndexNat`). -/
def bitColumn (word : Fin 4) (bit : Nat) : Nat := 1 + word.val * 64 + bit % 64

/-- `Σ_bit 2^bit · x[bitColumn word bit]`, the expression form of `decodeHashWord`. -/
def decodeExpr (word : Fin 4) : Expr :=
  (List.range 64).foldl (fun value bit =>
    value + Expr.const (Poseidon2.ofNat (2 ^ bit)) * Expr.var (bitColumn word bit)) 0

/-- The digest word in column `digestColumn` is the word that `x` encodes. -/
def decodeConstraint (digestColumn : Nat) (word : Fin 4) : Expr :=
  Expr.var digestColumn - decodeExpr word

def decodeConstraints (digestColumn : Fin 4 → Nat) : List Expr :=
  List.ofFn fun word => decodeConstraint (digestColumn word) word

private theorem decodeFold_eval (env : Env) (word : Fin 4) :
    ∀ (bits : List Nat) (initial : Expr),
      (bits.foldl (fun value bit =>
          value + Expr.const (Poseidon2.ofNat (2 ^ bit)) * Expr.var (bitColumn word bit))
        initial).eval env =
      bits.foldl (fun value bit =>
          value + Poseidon2.ofNat (2 ^ bit) * env (bitColumn word bit)) (initial.eval env)
  | [], _ => rfl
  | bit :: rest, initial => by
      simp only [List.foldl_cons]
      rw [decodeFold_eval env word rest]
      rfl

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The public input that an environment's first columns carry. -/
def publicInput (env : Env) :
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun column => env column.val

theorem decodeExpr_eval (env : Env) (word : Fin 4) :
    (decodeExpr word).eval env =
      decodeHashWord (publicFits := publicFits) (publicInput env) word := by
  unfold decodeExpr decodeHashWord
  rw [decodeFold_eval]
  rfl

/-- When `x = encHash digest`, the decode constraints give each digest word. -/
theorem digest_of_decode (env : Env) (digestColumn : Fin 4 → Nat) (digest : Digest)
    (encoded : publicInput (publicFits := publicFits) env = encHash digest)
    (holds : ConstraintsHold env (decodeConstraints digestColumn)) (word : Fin 4) :
    env (digestColumn word) = digest.getD word.val 0 := by
  have zero := holds (decodeConstraint (digestColumn word) word)
    (List.mem_ofFn.mpr ⟨word, rfl⟩)
  rw [decodeConstraint, Expr.eval_sub, Expr.eval_var, decodeExpr_eval (publicFits := publicFits),
    encoded, decodeHashWord_encHash] at zero
  exact sub_eq_zero.mp zero

/-! ### The constant column -/

theorem encoded_one {width : Nat} (layout : Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest) (inside : 0 < width)
    (encoded : publicInput (publicFits := publicFits) (sourceEnv layout assignment) =
      encHash digest) :
    assignment layout.one = 1 := by
  have marker := congrFun encoded encHashMarkerIndex
  rw [encHash_marker] at marker
  simpa [publicInput, encHashMarkerIndex, Layout.one,
    sourceEnv_public layout assignment 0 (by decide) inside] using marker

end NightstreamFPrime.Export.Generic.LogicalPlan
