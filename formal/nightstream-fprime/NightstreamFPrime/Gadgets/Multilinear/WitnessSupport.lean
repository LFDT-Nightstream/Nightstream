import NightstreamFPrime.Gadgets.Polynomial.WitnessSupport
import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner

namespace NightstreamFPrime.Gadgets.Multilinear

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Polynomial

theorem PointEquality.Owned.witnessesFromConstraints {variableCount : Nat}
    (interface : PointEquality.Owned.Interface variableCount) (offset : Nat) :
    WitnessesFromConstraints
      (Circuit.ops (PointEquality.Owned.circuit interface).main offset) :=
  WitnessesFromConstraints.arithmetic _ _

theorem PointWeightedHorner.Owned.witnessesFromConstraints {variableCount : Nat}
    (interface : PointWeightedHorner.Owned.Interface variableCount)
    (positive : 0 < variableCount) (offset : Nat) :
    WitnessesFromConstraints
      (Circuit.ops (PointWeightedHorner.Owned.circuit interface positive).main offset) := by
  change WitnessesFromConstraints
    ([PointWeightedHorner.Owned.pointOp interface offset] ++
      [PointWeightedHorner.Owned.hornerOp interface offset])
  apply WitnessesFromConstraints.append
  · exact WitnessesFromConstraints.call _ _ _
      (PointEquality.Owned.witnessesFromConstraints _ _)
  · exact WitnessesFromConstraints.call _ _ _
      (Horner.Owned.witnessesFromConstraints _ _)

end NightstreamFPrime.Gadgets.Multilinear
