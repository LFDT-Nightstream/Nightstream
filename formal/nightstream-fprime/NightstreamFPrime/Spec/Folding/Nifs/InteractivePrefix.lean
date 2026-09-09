import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution

/-!
The PiCCS part of the adversary in SuperNeo B.1 returns its public message
and continuation state. Relaying it to the weak extractor preserves every
issued round and the public output. It does not supply an intermediate witness.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.InteractivePrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)

/-- One fixed-private-tape PiCCS execution, with the captured suffix state. -/
structure Prover (State : Type*) (shape : Shape) (width : Nat) where
  rounds : CubePoint K shape.cubeVariables → K → Strategy width
  output : CubePoint K shape.cubeVariables → K → List K →
    Option (FullOutputCoordinates.FullOutput K shape × State)

variable {State : Type*} {shape : Shape} {columns width : Nat}

/-- Execute only the public PiCCS prefix. Aborted rounds remain an abort. -/
def run (prover : Prover State shape width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : Option (Probe K shape × State) :=
  match issued (prover.rounds alpha gamma) [] point.coordinates with
  | none => none
  | some rounds =>
      (prover.output alpha gamma point.coordinates).map fun output =>
        ({ coins := { alpha, gamma, roundPoint := point }
           response := {
             rounds := FixedPhase.RawCertificate.encode { rounds }
             fullOutput := output.1 } }, output.2)

/-- The relayed prover uses the identical round strategy. Only after its
public output does it invoke the suffix and obtain an extracted witness. -/
def relay (prover : Prover State shape width)
    (suffix : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Option (OutputWitness shape columns)) :
    CausalExecution.Prover shape columns width where
  rounds := prover.rounds
  output := fun alpha gamma coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      (prover.output alpha gamma coordinates).bind fun output =>
        (suffix { alpha, gamma, roundPoint := ⟨coordinates, dimension⟩ }
          output.1 output.2).map fun witness => (output.1, witness)
    else none

theorem run_coins (prover : Prover State shape width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) (receipt : Probe K shape × State)
    (returned : run prover alpha gamma point = some receipt) :
    receipt.1.coins = { alpha, gamma, roundPoint := point } := by
  cases rounds : issued (prover.rounds alpha gamma) [] point.coordinates with
  | none => simp [run, rounds] at returned
  | some messages =>
      cases output : prover.output alpha gamma point.coordinates with
      | none => simp [run, rounds, output] at returned
      | some result =>
          simp only [run, rounds, output, Option.map_some, Option.some.injEq] at returned
          subst receipt
          rfl

/-- Equality of the actual returned values in the two B.1 experiments.
Rounds execute once. The suffix receives the same coins, public output,
and captured state, and supplies the only intermediate witness. -/
theorem run_relay (prover : Prover State shape width)
    (suffix : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Option (OutputWitness shape columns))
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    CausalExecution.run (relay prover suffix) alpha gamma point =
      (run prover alpha gamma point).bind fun receipt =>
        (suffix receipt.1.coins receipt.1.response.fullOutput receipt.2).map
          fun witness => (receipt.1, witness) := by
  cases point with
  | mk coordinates dimension =>
      cases rounds : issued (prover.rounds alpha gamma) [] coordinates with
      | none => simp [CausalExecution.run, relay, run, rounds]
      | some messages =>
          cases output : prover.output alpha gamma coordinates with
          | none => simp [CausalExecution.run, relay, run, rounds, output, dimension]
          | some result =>
              simp [CausalExecution.run, relay, run, rounds, output, dimension,
                Option.map_map, Function.comp_def]

/-- Apply the public PiCCS verifier before resuming the suffix. This model
reconstructs the issued transcript from a fixed private tape. An actual
costed implementation can retain it from the first execution. -/
def checked (prover : Prover State shape width) (check : Probe K shape → Bool) :
    Prover State shape width where
  rounds := prover.rounds
  output := fun alpha gamma coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      match run prover alpha gamma ⟨coordinates, dimension⟩ with
      | none => none
      | some receipt =>
          if check receipt.1 then some (receipt.1.response.fullOutput, receipt.2) else none
    else none

/-- Public rejection aborts before any weak-extractor call. -/
theorem run_checked (prover : Prover State shape width) (check : Probe K shape → Bool)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    run (checked prover check) alpha gamma point =
      (run prover alpha gamma point).filter (fun receipt => check receipt.1) := by
  cases point with
  | mk coordinates dimension =>
      cases rounds : issued (prover.rounds alpha gamma) [] coordinates with
      | none => simp [run, checked, rounds]
      | some messages =>
          cases output : prover.output alpha gamma coordinates with
          | none => simp [run, checked, rounds, output, dimension]
          | some result =>
              simp only [run, checked, rounds, dimension, ↓reduceDIte, output, Option.map_some]
              split <;> simp_all [Option.filter]

theorem checked_returns (prover : Prover State shape width) (check : Probe K shape → Bool)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) (receipt : Probe K shape × State)
    (returned : run (checked prover check) alpha gamma point = some receipt) :
    run prover alpha gamma point = some receipt ∧ check receipt.1 = true := by
  rw [run_checked] at returned
  simpa only [Option.filter_eq_some_iff] using returned

end NightstreamFPrime.Spec.Folding.Nifs.InteractivePrefix
