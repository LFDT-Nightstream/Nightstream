import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal

/-!
One PiCCS prover with its private coins fixed. Round messages use only the
already sampled alpha/gamma and the past round prefix. The final full-output
message and witness may use the complete challenge stream. Either stage can
abort. The existing Probe and raw fixed-width certificate remain authoritative.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)

/-- Causal round messages, followed by the complete output message and witness. -/
structure Prover (shape : Shape) (columns width : Nat) where
  rounds : CubePoint K shape.cubeVariables → K → Strategy width
  output : CubePoint K shape.cubeVariables → K → List K →
    Option (FullOutputCoordinates.FullOutput K shape × OutputWitness shape columns)

/-- Execute the actual prefix-only rounds before the final response. -/
def run {shape : Shape} {columns width : Nat} (prover : Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (roundPoint : CubePoint K shape.cubeVariables) :
    Option (Probe K shape × OutputWitness shape columns) :=
  match issued (prover.rounds alpha gamma) [] roundPoint.coordinates with
  | none => none
  | some rounds =>
      (prover.output alpha gamma roundPoint.coordinates).map fun output =>
        ({ coins := { alpha, gamma, roundPoint }
           response := {
             rounds := FixedPhase.RawCertificate.encode { rounds }
             fullOutput := output.1 } }, output.2)

/-- Every returned probe has the actual sampled coins and issued raw messages. -/
theorem run_implies_receipt {shape : Shape} {columns width : Nat}
    (prover : Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (roundPoint : CubePoint K shape.cubeVariables)
    (probe : Probe K shape) (witness : OutputWitness shape columns)
    (returned : run prover alpha gamma roundPoint = some (probe, witness)) :
    probe.coins.alpha = alpha ∧ probe.coins.gamma = gamma ∧
      probe.coins.roundPoint = roundPoint ∧
      GoldilocksCausal.IssuedProbe (prover.rounds alpha gamma) probe := by
  cases execution : issued (prover.rounds alpha gamma) [] roundPoint.coordinates with
  | none =>
      simp only [run, execution] at returned
      cases returned
  | some rounds =>
      cases response : prover.output alpha gamma roundPoint.coordinates with
      | none =>
          simp only [run, execution, response, Option.map_none] at returned
          cases returned
      | some output =>
          have equal :
              (({ coins := { alpha, gamma, roundPoint }
                  response := {
                    rounds := FixedPhase.RawCertificate.encode { rounds }
                    fullOutput := output.1 } } : Probe K shape), output.2) =
                (probe, witness) := by
            exact Option.some.inj (by
              simpa only [run, execution, response, Option.map_some] using returned)
          have probeEqual := congrArg Prod.fst equal
          dsimp only at probeEqual
          subst probe
          refine ⟨rfl, rfl, rfl, ⟨{ rounds }, ?_, execution⟩⟩
          exact FixedPhase.RawCertificate.decode_encode _

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
