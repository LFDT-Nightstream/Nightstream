import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Lifecycle.XOut

/-! The ordinary part of a circuit's logical relation for the generic exporter (CW-5.2): each R1CS
row becomes one row of a `ProductionRelation.Plan` through `OrdinarySourcePlan`, over the identity
map from source columns to logical columns. `CircuitRelation` builds a step circuit's relation from
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

def identityMap (width : Nat) : SourceCompiler.SourceMap width width :=
  ⟨fun column => SparseForm.singleton column 1⟩

def oneColumn {rows : List R1CS.Row} {width : Nat} (fits : Fits rows width) : Fin width :=
  ⟨0, fits.positive⟩

def inputs (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width) :
    (program rows width fits).Inputs width :=
  ⟨oneColumn fits, fun _ => identityMap width⟩

/-- The logical relation's plan: one plan row per R1CS row, columns unchanged. -/
def plan (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width) : Plan width :=
  ((program rows width fits).compile (inputs rows width fits)).toPlan

/-- The source environment of a logical assignment: column `c` reads `assignment c`. -/
def sourceEnv {width : Nat} (assignment : Assignment F width) : Env :=
  fun column => if bounded : column < width then assignment ⟨column, bounded⟩ else 0

theorem rowsZero_iff_rowsHold (rows : List R1CS.Row) (width : Nat) (fits : Fits rows width)
    (assignment : Assignment F width) (one : assignment (oneColumn fits) = 1) :
    (plan rows width fits).RowsZero assignment ↔ R1CS.RowsHold (sourceEnv assignment) rows := by
  have preserves : ∀ index,
      OrdinarySourcePlan.SourceMap.PreservesRow ((inputs rows width fits).sourceMap index)
        assignment (sourceEnv assignment) ((program rows width fits).row index)
        ((program rows width fits).bounded index) := by
    intro index
    have bounded := (program rows width fits).bounded index
    refine ⟨?_, ?_, ?_⟩ <;> intro term member
    · simp [inputs, identityMap, sourceEnv, SparseForm.singleton_eval, bounded.1 term member]
    · simp [inputs, identityMap, sourceEnv, SparseForm.singleton_eval, bounded.2.1 term member]
    · simp [inputs, identityMap, sourceEnv, SparseForm.singleton_eval, bounded.2.2 term member]
  rw [plan, OrdinarySourcePlan.Program.rowsZero_iff _ _ assignment (sourceEnv assignment) one
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

theorem encoded_one (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : publicInput (publicFits := publicFits) (sourceEnv assignment) = encHash digest)
    {rows : List R1CS.Row} (fits : Fits rows logicalWidth) :
    assignment (oneColumn fits) = 1 := by
  have marker := congrFun encoded encHashMarkerIndex
  rw [encHash_marker] at marker
  simpa [publicInput, sourceEnv, encHashMarkerIndex, oneColumn, fits.positive] using marker

end NightstreamFPrime.Export.Generic.LogicalPlan
