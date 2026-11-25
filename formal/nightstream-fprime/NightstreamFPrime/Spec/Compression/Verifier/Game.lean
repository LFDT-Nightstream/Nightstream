import NightstreamFPrime.Spec.Compression.Verifier.Bad
import NightstreamFPrime.Spec.Compression.Verifier.Weights
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Game.Restoration

/-! The verifier of Rust `ns_compress::verify` as a public-coin game (SPEC.md section 6).

The prover names the claims, commits `W`, receives `τ`, commits `V`, receives the challenges, runs
the first sum-check, sends `W(ρ)`, the shared inverse value and the key part, runs the key sum-check,
and sends the value of the key polynomial (`game`).

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
abstract in this unit) and the zeros of the opening point of `V` (Rust `inverse_point`). -/
structure VerifierData (P : Params F K) where
  matrixPart : Statement P → Challenges P → List K → K
  zeros : ℕ

/-- `matrixPart` computes the matrix part of the honest weights. -/
def MatrixSpec {P : Params F K} (vf : VerifierData P) : Prop :=
  ∀ st ch ρ, ρ.length = P.witnessVars → vf.matrixPart st ch ρ = matrixTrue P st ch ρ

/-- The values after the first sum-check (Rust `Proof`): `W(ρ)`, the value of the shared inverse, and
the key part. -/
abbrev Values (K : Type) := K × K × K

section Game

variable {P : Params F K} (vf : VerifierData P)

/-- The tables of the key sum-check: the weights `T` and `F = Σ_i ω^i A_i`. -/
def keyTables (ch : Challenges P) (ρ : List K) : Fin 2 → ℕ → K :=
  ![Algebra.weightTable P.lanes (laneForm P ch ρ) (uses P ch ρ),
    folded P.lanes P.rows (fun i j h => ι (P.key i j h)) ch.omega]

/-- The checks of Rust `verify` after the key sum-check, with the ideal Akita openings. -/
def Accepts (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) (ch : Challenges P)
    (ρ : List K) (value : K) (vals : Values K) (kp : List K)
    (kfinal kv : K) : Prop :=
  Shaped P st ∧
  (∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
    τ k ≠ tableValue (P.shape c).bound v) ∧
  (∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0) ∧
  kfinal = rowScale P.rowVars ch.omega * kv *
    weightAt P.lanes P.laneVars (laneForm P ch ρ) (uses P ch ρ) kp ∧
  value = mainSummand (fun k => ι (τ k)) ch.psi vals.1
    (vals.2.2 + vf.matrixPart st ch ρ + closedPart P st τ ch ρ)
    (checkAt P ch ρ) (lowAt P ρ * vals.2.1) (sumAt P ch ρ) ∧
  kv = mle (kp ++ rowPoint P.rowVars ch.omega)
    (keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h)) ∧
  vals.1 = mle ρ (fun x => ι (W x)) ∧
  vals.2.1 = mle (ρ.take P.spanVars ++ List.replicate vf.zeros 0) (fun x => ι (V x))

/-- After the key sum-check. -/
noncomputable def afterKey (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) (ρ : List K) (value : K) (vals : Values K) :
    Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (kp, kfinal) => .recv K fun kv => by
      classical
      exact .pure (if Accepts vf st W V τ ch ρ value vals kp kfinal kv then some (st, W)
        else none)

/-- After the first sum-check. -/
noncomputable def afterMain (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) : Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (ρ, value) =>
      .recv (Values K) fun vals =>
        bind (sumcheck 2 (combRound (P.laneVars + P.keyBlockVars) (keyTables ch ρ)
          productComb) (P.laneVars + P.keyBlockVars) [] vals.2.2)
          (afterKey vf st W V τ ch ρ value vals)

/-- The verifier after the prover names the claims `st` and the table `W`. -/
noncomputable def afterTables (st : Statement P) (W : ℕ → F) :
    Game (Option (Statement P × (ℕ → F))) :=
  .draw (Fin P.lookups → F) (tauBad P W) fun τ =>
    .recv (ℕ → F) fun V =>
      .draw (Challenges P) (challengeBad P st W V τ) fun ch =>
        bind (sumcheck (P.lookups + 2) (combRound P.witnessVars (mainTables P st W V τ ch) (mainComb P τ ch))
          P.witnessVars [] (claimedSum P st ch)) (afterMain vf st W V τ ch)

/-- The verifier against an adaptive prover. -/
noncomputable def game : Game (Option (Statement P × (ℕ → F))) :=
  .recv (Statement P) fun st => .recv (ℕ → F) fun W => afterTables vf st W

end Game

end NightstreamFPrime.Spec.Compression.Verifier
