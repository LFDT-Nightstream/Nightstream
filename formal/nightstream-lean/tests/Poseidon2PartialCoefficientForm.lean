import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm

namespace Nightstream.Tests.Poseidon2PartialCoefficientForm

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm

example (lane : Fin width) :
    (coefficientForm canonicalLayout partialRounds lane).length =
      width + partialRounds :=
  coefficientForm_length canonicalLayout partialRounds lane

example (assignment : Nat → Nat) (round : Nat)
    (roundBound : round ≤ partialRounds) (lane : Fin width) :
    lcEval assignment (coefficientForm canonicalLayout round lane) =
      lcEval assignment (partialState canonicalLayout round lane) :=
  lcEval_coefficientForm canonicalLayout round roundBound lane assignment

end Nightstream.Tests.Poseidon2PartialCoefficientForm
