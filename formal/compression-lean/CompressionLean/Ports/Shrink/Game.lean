import CompressionLean.Ports.Shrink.Evaluate
import CompressionLean.Ports.Shrink.Rounds
import CompressionLean.Game.SumCheck
import CompressionLean.Lib.IdentityTest

/-!
Owns the shrink verifier (`verify`, `mod.rs:233-288`) as a public-coin game
against an adaptive prover, in the model of `Game/Game.lean`.

The prover names the program first. Rust absorbs the program's key and
statement before the commitment (`mod.rs:117-126`), so a prover may choose
them. Then, in transcript order:

1. the commitment: a list of at most `L` candidates for `w` (WHIR's list);
2. `τ`: `m = rowVars` draws (`mod.rs:246-248`);
3. the outer sum-check: `m` rounds of degree 8 from the claim `0`
   (`mod.rs:249`);
4. the claimed values `v_j = M̃_j·z(rx)` and the outer check
   `last = eq(τ, rx)·P(v)` (`mod.rs:250-253`);
5. `ρ` (`mod.rs:254-255`);
6. the inner sum-check: `n + 1` rounds of degree 2 from `Σ_j ρ^j·v_j`, with
   `n = cellVars` (`mod.rs:256-257`);
7. the opened value of `w̃` at `ry[..n]` (`mod.rs:258-261`);
8. the last checks (`Final`): the shape run has the shape's counts
   (`mod.rs:263-274`), the opening holds, and `last = matrices·z̃(ry)` with
   `z̃(ry) = (1 - ry_n)·w̃(ry[..n]) + ry_n·x̃(ry[..n])` (`mod.rs:275-286`).
   `matrices` is the verifier's value of `Σ_j ρ^j·M̃_j(rx, ry)`: the shape
   run's total plus `block_part` (`verifierMatrices`, `Evaluate.lean`).

The opening is idealized (`Opens`): the verifier accepts it when one
candidate agrees with the opened value. The WHIR error that this hides is a
named term of the budget (`Budget.lean`). The proof's shape check
(`mod.rs:236-242`) is in the message types. The statement words are the
shape run's public words; Rust checks that they equal the program's
statement (`mod.rs:272`).

Each draw names its bad set from earlier messages only, as a union over the
candidates.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

/-- One shrink transcript: the program, the candidates, the challenges and
the prover's values. -/
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

/-- The `τ` values that would zero a false candidate's zero-check: per
candidate, at most one per coordinate. -/
noncomputable def tauBad (C : Ccs F) (cands : List (ℕ → F)) (point : List K) : Finset K :=
  Finset.univ.biUnion fun k : Fin cands.length =>
    foldBad (constraintTable C.entries (C.z (cands.get k))) C.rowVars point

/-- A candidate's true values `M̃_j·z(rx)`. -/
def trueValues (C : Ccs F) (w : ℕ → F) (rx : List K) (j : Fin 8) : K :=
  mle rx fun r => ι (rowValue C.entries (C.z w) j r)

/-- The `ρ` values that would hide wrong claimed values: per candidate, the
roots of `batchPoly`. -/
noncomputable def rhoBad (C : Ccs F) (cands : List (ℕ → F)) (rx : List K) (values : Fin 8 → K) :
    Finset K :=
  Finset.univ.biUnion fun k : Fin cands.length => zeros (batchPoly values (trueValues C (cands.get k) rx))

/-- `z̃(ry)` from the opened `w̃(ry[..n])` and the verifier's `x̃(ry[..n])`
(`mod.rs:276-283`). -/
def zValue (C : Ccs F) (ry : List K) (opened : K) : K :=
  (1 - ry.getD C.cellVars 0) * opened + ry.getD C.cellVars 0 * C.publicValue (ry.take C.cellVars)

/-- The idealized WHIR opening (`mod.rs:258-261`): one candidate agrees with
the opened value at `ry[..n]`. -/
def Opens (C : Ccs F) (cands : List (ℕ → F)) (ry : List K) (opened : K) : Prop :=
  ∃ w ∈ cands, opened = mle (ry.take C.cellVars) fun i => ι (w i)

/-- The verifier's last checks (`mod.rs:258-286`); `matrices` is its value
of `Σ_j ρ^j·M̃_j(rx, ry)` (`mod.rs:275`). -/
def Final (C : Ccs F) (shapeOk : Prop) (cands : List (ℕ → F)) (ry : List K) (matrices last opened : K) :
    Prop :=
  shapeOk ∧ Opens C cands ry opened ∧ last = matrices * zValue C ry opened

open Classical in
/-- The verifier on one program's CCS `C`; `shapeOk` is its shape-run check
and `matrices rx ry ρ` its value of `Σ_j ρ^j·M̃_j(rx, ry)`. -/
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

/-- The shrink verifier for a family of programs: the prover names program
`i`, whose shape run `prog i` fixes the CCS under the layout of the shape
`s` (`Shrink::new`, `mod.rs:101-111`) and the verifier's matrix value. -/
noncomputable def shrinkGame {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) : Game (Option (Transcript I F K)) :=
  .recv I fun i => programGame i ((prog i).ccs T s) ((prog i).counts = s) (verifierMatrices T s (prog i)) L

end CompressionLean.Shrink
