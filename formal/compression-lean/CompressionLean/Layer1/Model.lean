import CompressionLean.Layer1.RingRows
import CompressionLean.Layer1.Lookups

/-!
Owns the data of the layer-1 relation (`crates/neo-spartan/src/lib.rs`,
`ring.rs`, `matrix.rs`) and the relation itself.

- `Shape`: the verifier's sizes. The witness cube has the lane variables low
  (`laneVars`, `D ≤ 2^laneVars` used lanes) and the block variables high.
- `Setup`: the honest setup: key rows, runs (row, matrix, slot, coefficient)
  and slots (block, lane, class). The verifier holds its root; under Merkle
  binding the opened values are these.
- `Statement`: the CE(B) claim: commitment rows, public blocks, `χ_r`, Eval_K,
  Eval_A, as `F` coordinates (`re`, `im` for the quadratic field).
- `Relation`: every cell in `[-H, H]` and the `κ + 2 + 2·matrices + public`
  ring rows over `F[X]/Φ`, with the weights of `ring.rs`.
-/

namespace CompressionLean.Layer1

open Polynomial

/-- The verifier's sizes. -/
structure Shape where
  /-- Ring degree `D` (54) and the lane variables (6, so `D ≤ 64`). -/
  D : ℕ
  laneVars : ℕ
  blockVars : ℕ
  /-- Used blocks: the witness has `D·blocks` coordinates. -/
  blocks : ℕ
  kappa : ℕ
  matrices : ℕ
  publicBlocks : ℕ
  normBound : ℕ
  runVars : ℕ
  rowVars : ℕ
  slotVars : ℕ
  matrixVars : ℕ
  setupVars : ℕ
  queries : ℕ

namespace Shape

variable (s : Shape)

def cubeVars : ℕ := s.laneVars + s.blockVars

/-- Rows: commitment, Eval_K (re, im), Eval_A (re, im per matrix), public blocks. -/
def rows : ℕ := s.kappa + 2 + 2 * s.matrices + s.publicBlocks

def evalKRow (part : ℕ) : ℕ := s.kappa + part
def evalARow (j part : ℕ) : ℕ := s.kappa + 2 + 2 * j + part
def publicRow (j : ℕ) : ℕ := s.kappa + 2 + 2 * s.matrices + j

end Shape

/-- The honest setup of one circuit: key rows on the cube and the matrix runs. -/
structure Setup (F : Type) where
  /-- Key row `i` at cube index `2^laneVars·b + l`; zero off the used lanes and blocks. -/
  key : ℕ → ℕ → F
  runRow : ℕ → ℕ
  runMatrix : ℕ → ℕ
  runSlot : ℕ → ℕ
  runCoef : ℕ → F
  /-- Runs `k < runs`; padding runs after them. -/
  runs : ℕ
  slotBlock : ℕ → ℕ
  slotLane : ℕ → ℕ
  slotLength : ℕ → ℕ
  slotRatio : ℕ → F
  slots : ℕ

/-- The CE(B) claim. A quadratic-field value is its `(re, im)` pair. -/
structure Statement (F : Type) where
  commitment : ℕ → ℕ → F
  publicInput : ℕ → ℕ → F
  /-- `χ_r(i)` at every index (rows for the lookup, coordinates for Eval_K). -/
  chi : ℕ → F × F
  evalK : ℕ → F × F
  evalA : ℕ → ℕ → F × F

/-- One P0 candidate: the witness cube and the early matrix columns. -/
structure P0Tables (F : Type) where
  z : ℕ → F
  e : ℕ → F × F
  /-- `U_j(σ)`: matrix `j`, slot `σ`. -/
  U : ℕ → ℕ → F × F
  blockCopy : ℕ → F
  mult : ℕ → F

/-- One P1 candidate: the two folded setup groups and the block weights `Ω^A`. -/
structure P1Tables (K : Type) where
  foldKey : ℕ → K
  foldRun : ℕ → K
  omegaA : ℕ → K

/-- A commitment is a list of at most `L` candidates (WHIR's list decoding). -/
instance candidates_nonempty {α : Type} (L : ℕ) : Nonempty {l : List α // l.length ≤ L} :=
  ⟨⟨[], by simp⟩⟩

variable {F : Type} [Field F]

/-- `Σ_{p < D} v(p)·X^p`. -/
noncomputable def coeffPoly (D : ℕ) (v : ℕ → F) : F[X] :=
  ∑ p ∈ Finset.range D, C (v p) * X ^ p

/-- `bar(w)` as a ring element: coefficient `p` is `Σ_l bar(p, l)·w(l)`. -/
noncomputable def barPoly (D : ℕ) (bar : ℕ → ℕ → F) (w : ℕ → F) : F[X] :=
  coeffPoly D fun p => ∑ l ∈ Finset.range D, bar p l * w l

/-- The Eval_A weight `(M_jᵀ χ_r)_c` from the runs: `Σ_k coef·ratio^{c - start}·χ_r(row)`. -/
noncomputable def evalAWeight (S : Setup F) (st : Statement F) (D j c : ℕ) : F × F :=
  ∑ k ∈ (Finset.range S.runs).filter (fun k => S.runMatrix k = j ∧
      D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
      c < D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k)),
    (S.runCoef k * S.slotRatio (S.runSlot k) ^
        (c - (D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k)))) • st.chi (S.runRow k)

/-- The true slot sums `U_j(σ) = Σ_{k : matrix j, slot σ} coef_k·χ_r(row_k)` over `2^runVars` runs. -/
noncomputable def trueU (s : Shape) (S : Setup F) (st : Statement F) (j σ : ℕ) : F × F :=
  ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter (fun k => S.runMatrix k = j ∧ S.runSlot k = σ),
    S.runCoef k • st.chi (S.runRow k)

/-- The setup is well formed: padding runs are zero, runs fit their slots and blocks. -/
structure Setup.Fits (S : Setup F) (s : Shape) : Prop where
  runs_le : S.runs ≤ 2 ^ s.runVars
  padding : ∀ k, S.runs ≤ k → S.runCoef k = 0 ∧ S.runRow k = 0 ∧ S.runSlot k = 0 ∧ S.runMatrix k = 0
  row_lt : ∀ k, S.runRow k < 2 ^ s.rowVars
  slot_lt : ∀ k, S.runSlot k < 2 ^ s.slotVars
  matrix_lt : ∀ k, S.runMatrix k < 2 ^ s.matrixVars
  slots_le : S.slots ≤ 2 ^ s.slotVars
  lane_lt : ∀ σ, S.slotLane σ < s.D
  length_le : ∀ σ, S.slotLength σ ≤ s.D
  inside : ∀ σ, s.D * S.slotBlock σ + S.slotLane σ + S.slotLength σ ≤ s.D * s.blocks
  key_zero : ∀ i b l, (s.blocks ≤ b ∨ s.D ≤ l) → S.key i (2 ^ s.laneVars * b + l) = 0

/-- The witness block `b` as a ring element. -/
noncomputable def witnessBlock (s : Shape) (z : ℕ → F) (b : ℕ) : F[X] :=
  coeffPoly s.D fun l => z (2 ^ s.laneVars * b + l)

/-- The ring rows of the relation, in `ring.rs` order. -/
noncomputable def relationRows (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) :
    RingRows F where
  rows := s.rows
  blocks := s.blocks
  weight ρ b :=
    if ρ < s.kappa then coeffPoly s.D fun l => S.key ρ (2 ^ s.laneVars * b + l)
    else if ρ < s.kappa + 2 then
      barPoly s.D bar fun l =>
        if ρ = s.evalKRow 0 then (st.chi (s.D * b + l)).1 else (st.chi (s.D * b + l)).2
    else if ρ < s.kappa + 2 + 2 * s.matrices then
      barPoly s.D bar fun l =>
        if (ρ - s.kappa - 2) % 2 = 0 then (evalAWeight S st s.D ((ρ - s.kappa - 2) / 2) (s.D * b + l)).1
        else (evalAWeight S st s.D ((ρ - s.kappa - 2) / 2) (s.D * b + l)).2
    else if b = ρ - (s.kappa + 2 + 2 * s.matrices) then 1 else 0
  target ρ :=
    if ρ < s.kappa then coeffPoly s.D (st.commitment ρ)
    else if ρ < s.kappa + 2 then
      coeffPoly s.D fun l => if ρ = s.evalKRow 0 then (st.evalK l).1 else (st.evalK l).2
    else if ρ < s.kappa + 2 + 2 * s.matrices then
      coeffPoly s.D fun l =>
        if (ρ - s.kappa - 2) % 2 = 0 then (st.evalA ((ρ - s.kappa - 2) / 2) l).1
        else (st.evalA ((ρ - s.kappa - 2) / 2) l).2
    else coeffPoly s.D (st.publicInput (ρ - (s.kappa + 2 + 2 * s.matrices)))

/-- CE(B) in ring-row form: the norm conjunct and every linear conjunct. -/
def Relation (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) (z : ℕ → F) :
    Prop :=
  (∀ x, x < 2 ^ s.cubeVars → ∃ t, t < 2 * s.normBound + 1 ∧ z x = slotValue s.normBound t) ∧
    (relationRows s bar S st).Holds Φ (witnessBlock s z)

end CompressionLean.Layer1
