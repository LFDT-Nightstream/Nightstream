import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)

structure Prover (shape : Shape) (columns width : Nat) where
  rounds : CubePoint K shape.cubeVariables → K → Strategy width
  output : CubePoint K shape.cubeVariables → K → List K →
    Option (FullOutputCoordinates.FullOutput K shape × OutputWitness shape columns)

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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
