import CompressionLean.Ports.Shrink.Evaluate
import CompressionLean.Ports.Shrink.Rounds
import CompressionLean.Game.SumCheck
import CompressionLean.Lib.IdentityTest

/-! The statement words are the shape run's public words; Rust checks that they equal the
program's statement (`mod.rs::verify`). -/

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

structure Transcript (I F K : Type) where
  inst : I
  cands : List (ℕ → F)
  tau : List K
  rx : List K
  outerLast : K
  values : Fin 8 → K
  rho : K
  ry : List K
  innerLast : K
  opened : K

noncomputable def tauBad (C : Ccs F) (cands : List (ℕ → F)) (point : List K) : Finset K :=
  Finset.univ.biUnion fun k : Fin cands.length =>
    foldBad (constraintTable C.entries (C.z (cands.get k))) C.rowVars point

def trueValues (C : Ccs F) (w : ℕ → F) (rx : List K) (j : Fin 8) : K :=
  mle rx fun r => ι (rowValue C.entries (C.z w) j r)

noncomputable def rhoBad (C : Ccs F) (cands : List (ℕ → F)) (rx : List K) (values : Fin 8 → K) :
    Finset K :=
  Finset.univ.biUnion fun k : Fin cands.length => zeros (batchPoly values (trueValues C (cands.get k) rx))

def zValue (C : Ccs F) (ry : List K) (opened : K) : K :=
  (1 - ry.getD C.cellVars 0) * opened + ry.getD C.cellVars 0 * C.publicValue (ry.take C.cellVars)

/-- Assumption: idealized WHIR opening; the WHIR error is a named term of the final bound, not proved here. -/
def Opens (C : Ccs F) (cands : List (ℕ → F)) (ry : List K) (opened : K) : Prop :=
  ∃ w ∈ cands, opened = mle (ry.take C.cellVars) fun i => ι (w i)

def Final (C : Ccs F) (shapeOk : Prop) (cands : List (ℕ → F)) (ry : List K) (matrices last opened : K) :
    Prop :=
  shapeOk ∧ Opens C cands ry opened ∧ last = matrices * zValue C ry opened

open Classical in
noncomputable def programGame {I : Type} (i : I) (C : Ccs F) (shapeOk : Prop)
    (matrices : List K → List K → K → K) (L : ℕ) : Game (Option (Transcript I F K)) :=
  .recv {l : List (ℕ → F) // l.length ≤ L} fun cands =>
  Game.bind (Game.drawPoint (tauBad C cands.1) C.rowVars []) fun tau =>
  Game.bind (Game.sumcheck Finset.univ 7
      (fun k : Fin cands.1.length =>
        combRound C.rowVars (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb)
      (fun _ => ∅) C.rowVars [] 0) fun outer =>
  .recv (Fin 8 → K) fun values =>
  if outer.2 = eqEval tau outer.1 * constraint values then
    .draw K (rhoBad C cands.1 outer.1 values) fun rho =>
    Game.bind (Game.sumcheck Finset.univ 1
        (fun k : Fin cands.1.length =>
          combRound (C.cellVars + 1) (innerTables C.entries (C.z (cands.1.get k)) outer.1 rho) productComb)
        (fun _ => ∅) (C.cellVars + 1) [] (batch rho values)) fun inner =>
    .recv K fun opened =>
    if Final C shapeOk cands.1 inner.1 (matrices outer.1 inner.1 rho) inner.2 opened then
      .pure (some ⟨i, cands.1, tau, outer.1, outer.2, values, rho, inner.1, inner.2, opened⟩)
    else .pure none
  else .pure none

noncomputable def shrinkGame {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) : Game (Option (Transcript I F K)) :=
  .recv I fun i => programGame i ((prog i).ccs T s) ((prog i).counts = s) (verifierMatrices T s (prog i)) L

end CompressionLean.Shrink
