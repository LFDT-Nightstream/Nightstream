import NightstreamFPrime.Spec.Compression.Verifier.Bad
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Game.Restoration
import Mathlib.Algebra.MvPolynomial.CommRing

/-! The verifier of Rust `neo_compress::verify` as a public-coin game (SPEC.md section 6).

The prover names the claims, commits `W`, receives `τ`, commits `V`, receives the challenges, runs
the first sum-check, sends `W(ρ)`, the values of the columns of `V` and the key part, receives the
column point, runs the key sum-check and sends the value of the key polynomial (`game`).

What this model assumes:

- The Akita openings are ideal: an opening is accepted only when the committed table, which the
  prover named before the later challenges, has the opened value at the point (`Accepts`, the last
  three conditions). Akita's own knowledge error is not in the bound.
- The key polynomial is the honest key of `Params.key` (the verification key is the honest Akita
  commitment of it; Rust pins the bytes of the key).
- The matrix part of the weights is abstract: `VerifierData.matrixPart` equals the matrix part that the
  honest weights give (`MatrixSpec`).

`game_avoids`: on each path that avoids the bad sets, an accepted proof gives `Relation` for the
committed `W`. `game_densityLe`: each draw has density at most `error`. `soundness`: a prover that
makes `q` oracle queries in the state-restoration experiment names an accepted proof whose `W` does
not satisfy `Relation` with probability at most `q · error`. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

/-- What the verifier holds besides `Params`: the matrix part of the weights (Rust `matrix_part`,
abstract in this unit), the variables of the column point (Rust `LOOKUP_VARS`) and the zeros of
the opening point of `V` (Rust `inverse_point`). -/
structure VerifierData (P : Params F K) where
  matrixPart : Statement P → Challenges P → List K → K
  columnVars : ℕ
  zeros : ℕ

/-- `matrixPart` computes the matrix part of the honest weights. -/
def MatrixSpec {P : Params F K} (vf : VerifierData P) : Prop :=
  ∀ st ch ρ, ρ.length = P.witnessVars → vf.matrixPart st ch ρ = matrixTrue P st ch ρ

/-- The values after the first sum-check (Rust `Proof`): `W(ρ)`, the column values of `V`, and
the key part. -/
abbrev Values (K : Type) := K × (ℕ → K) × K

section Game

variable {P : Params F K} (vf : VerifierData P)

/-- The tables of the key sum-check: the weights `T` and `F = Σ_i ω^i A_i`. -/
def keyTables (ch : Challenges P) (ρ : List K) : Fin 2 → ℕ → K :=
  ![Algebra.weightTable P.lanes (laneForm P ch ρ) (uses P ch ρ),
    folded P.lanes P.rows (fun i j h => ι (P.key i j h)) ch.omega]

/-- The checks of Rust `verify` after the key sum-check, with the ideal Akita openings. -/
def Accepts (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) (ch : Challenges P)
    (ρ : List K) (value : K) (vals : Values K) (col : Fin vf.columnVars → K) (kp : List K)
    (kfinal kv : K) : Prop :=
  Shaped P st ∧
  (∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
    τ k ≠ tableValue (P.shape c).bound v) ∧
  (∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0) ∧
  kfinal = rowScale P.rowVars ch.omega * kv *
    weightAt P.lanes P.laneVars (laneForm P ch ρ) (uses P ch ρ) kp ∧
  value = vals.1 * (vals.2.2 + vf.matrixPart st ch ρ + closedPart P st τ ch ρ -
      checkAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * vals.2.1 k)) +
    checkAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * ι (τ k) * vals.2.1 k) +
    sumAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * vals.2.1 k) ∧
  kv = mle (kp ++ rowPoint P.rowVars ch.omega)
    (keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h)) ∧
  vals.1 = mle ρ (fun x => ι (W x)) ∧
  mle (List.ofFn col) vals.2.1 =
    mle (ρ.take P.spanVars ++ List.ofFn col ++ List.replicate vf.zeros 0) (fun x => ι (V x))

/-- After the key sum-check. -/
noncomputable def afterKey (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) (ρ : List K) (value : K) (vals : Values K)
    (col : Fin vf.columnVars → K) : Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (kp, kfinal) => .recv K fun kv => by
      classical
      exact .pure (if Accepts vf st W V τ ch ρ value vals col kp kfinal kv then some (st, W)
        else none)

/-- After the first sum-check. -/
noncomputable def afterMain (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) : Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (ρ, value) =>
      .recv (Values K) fun vals =>
        .draw (Fin vf.columnVars → K) (columnBad vf.columnVars vals.2.1 (columnAt P V ρ))
          fun col => bind (sumcheck 2 (combRound (P.laneVars + P.keyBlockVars) (keyTables ch ρ)
            productComb) (P.laneVars + P.keyBlockVars) [] vals.2.2)
            (afterKey vf st W V τ ch ρ value vals col)

/-- The verifier against an adaptive prover. -/
noncomputable def game : Game (Option (Statement P × (ℕ → F))) :=
  .recv (Statement P) fun st =>
    .recv (ℕ → F) fun W =>
      .draw (Fin P.lookups → F) (tauBad P W) fun τ =>
        .recv (ℕ → F) fun V =>
          .draw (Challenges P) (challengeBad P st W V τ) fun ch =>
            bind (sumcheck 3 (combRound P.witnessVars (mainTables P st W V τ ch) mainComb)
              P.witnessVars [] (claimedSum P st ch)) (afterMain vf st W V τ ch)

end Game

omit [Fintype K] [DecidableEq K] in
theorem mainComb_degree : (mainComb : MvPolynomial (Fin 6) K).totalDegree ≤ 3 := by
  unfold mainComb
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le ((MvPolynomial.totalDegree_add _ _).trans
    (max_le ?_ ?_)) ?_)
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X]
    have : (MvPolynomial.X 1 - MvPolynomial.X 2 * MvPolynomial.X 3 :
        MvPolynomial (Fin 6) K).totalDegree ≤ 2 :=
      (MvPolynomial.totalDegree_sub _ _).trans (max_le (by rw [MvPolynomial.totalDegree_X]; norm_num)
        ((MvPolynomial.totalDegree_mul _ _).trans (by
          rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X])))
    omega
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X]
    omega
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X]
    omega

end NightstreamFPrime.Spec.Compression.Verifier
