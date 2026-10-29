import CompressionLean.Layer1.RingRows
import CompressionLean.Layer1.Lookups
import Mathlib.Algebra.QuadraticAlgebra.Defs

namespace CompressionLean.Layer1

open Polynomial

structure Shape where
  D : ℕ
  laneVars : ℕ
  blockVars : ℕ
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

def rows : ℕ := s.kappa + 2 + 2 * s.matrices + s.publicBlocks

def evalKRow (part : ℕ) : ℕ := s.kappa + part
def evalARow (j part : ℕ) : ℕ := s.kappa + 2 + 2 * j + part
def publicRow (j : ℕ) : ℕ := s.kappa + 2 + 2 * s.matrices + j

end Shape

structure Setup (F : Type) where
  key : ℕ → ℕ → F
  runRow : ℕ → ℕ
  runMatrix : ℕ → ℕ
  runSlot : ℕ → ℕ
  runCoef : ℕ → F
  runs : ℕ
  slotBlock : ℕ → ℕ
  slotLane : ℕ → ℕ
  slotLength : ℕ → ℕ
  slotRatio : ℕ → F
  slots : ℕ

abbrev Quad (F : Type) [Field F] := QuadraticAlgebra F 7 0

structure Statement (F : Type) where
  commitment : ℕ → ℕ → F
  publicInput : ℕ → ℕ → F
  r : List (F × F)
  evalK : ℕ → F × F
  evalA : ℕ → ℕ → F × F

def Statement.point {F : Type} [Field F] (st : Statement F) : List (Quad F) := st.r.map fun c => ⟨c.1, c.2⟩

def Statement.chi {F : Type} [Field F] (st : Statement F) (i : ℕ) : F × F :=
  ((eqAt st.point i).re, (eqAt st.point i).im)

structure P0Tables (F : Type) where
  z : ℕ → F
  e : ℕ → F × F
  U : ℕ → ℕ → F × F
  blockCopy : ℕ → F
  mult : ℕ → F

structure P1Tables (K : Type) where
  foldKey : ℕ → K
  foldRun : ℕ → K
  omegaA : ℕ → K

instance candidates_nonempty {α : Type} (L : ℕ) : Nonempty {l : List α // l.length ≤ L} :=
  ⟨⟨[], by simp⟩⟩

variable {F : Type} [Field F]

noncomputable def coeffPoly (D : ℕ) (v : ℕ → F) : F[X] :=
  ∑ p ∈ Finset.range D, C (v p) * X ^ p

noncomputable def barPoly (D : ℕ) (bar : ℕ → ℕ → F) (w : ℕ → F) : F[X] :=
  coeffPoly D fun p => ∑ l ∈ Finset.range D, bar p l * w l

noncomputable def evalAWeight (S : Setup F) (st : Statement F) (D j c : ℕ) : F × F :=
  ∑ k ∈ (Finset.range S.runs).filter (fun k => S.runMatrix k = j ∧
      D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
      c < D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k)),
    (S.runCoef k * S.slotRatio (S.runSlot k) ^
        (c - (D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k)))) • st.chi (S.runRow k)

noncomputable def trueU (s : Shape) (S : Setup F) (st : Statement F) (j σ : ℕ) : F × F :=
  ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter (fun k => S.runMatrix k = j ∧ S.runSlot k = σ),
    S.runCoef k • st.chi (S.runRow k)

structure Setup.Fits (S : Setup F) (s : Shape) : Prop where
  runs_le : S.runs ≤ 2 ^ s.runVars
  padding : ∀ k, S.runs ≤ k → S.runCoef k = 0 ∧ S.runRow k = 0 ∧ S.runSlot k = 0 ∧ S.runMatrix k = 0
  row_lt : ∀ k, S.runRow k < 2 ^ s.rowVars
  slot_lt : ∀ k, S.runSlot k < 2 ^ s.slotVars
  matrix_lt : ∀ k, S.runMatrix k < 2 ^ s.matrixVars
  slots_le : S.slots ≤ 2 ^ s.slotVars
  run_slot : ∀ k, k < S.runs → S.runSlot k < S.slots
  run_matrix : ∀ k, k < S.runs → S.runMatrix k < s.matrices
  lane_lt : ∀ σ, S.slotLane σ < s.D
  length_le : ∀ σ, S.slotLength σ ≤ s.D
  inside : ∀ σ, s.D * S.slotBlock σ + S.slotLane σ + S.slotLength σ ≤ s.D * s.blocks
  key_zero : ∀ i b l, (s.blocks ≤ b ∨ s.D ≤ l) → S.key i (2 ^ s.laneVars * b + l) = 0

noncomputable def witnessBlock (s : Shape) (z : ℕ → F) (b : ℕ) : F[X] :=
  coeffPoly s.D fun l => z (2 ^ s.laneVars * b + l)

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

def Relation (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) (z : ℕ → F) :
    Prop :=
  (∀ x, x < 2 ^ s.cubeVars → ∃ t, t < 2 * s.normBound + 1 ∧ z x = slotValue s.normBound t) ∧
    (relationRows s bar S st).Holds Φ (witnessBlock s z)

end CompressionLean.Layer1
