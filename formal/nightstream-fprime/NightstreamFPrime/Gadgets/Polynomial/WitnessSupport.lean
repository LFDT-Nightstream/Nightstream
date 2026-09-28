import NightstreamFPrime.Circuit.WitnessConstraintSupport
import NightstreamFPrime.Gadgets.Polynomial.Power
import NightstreamFPrime.Gadgets.Polynomial.Sparse

namespace NightstreamFPrime.Gadgets.Polynomial

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem Horner.Owned.witnessesFromConstraints (interface : Horner.Owned.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (Horner.Owned.circuit interface).main offset) := by
  rw [Horner.Owned.circuit_ops]
  exact WitnessesFromConstraints.arithmetic _ _

theorem Power.witnessesFromConstraints (exponent : Nat) (interface : Power.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (Power.circuit exponent interface).main offset) :=
  Horner.Owned.witnessesFromConstraints _ _

theorem Sparse.Owned.witnessesFromConstraints {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial K matrixCount)
    (interface : Sparse.Owned.Interface matrixCount) (offset : Nat) :
    WitnessesFromConstraints
      (Circuit.ops (Sparse.Owned.circuit polynomial interface).main offset) :=
  WitnessesFromConstraints.arithmetic _ _

end NightstreamFPrime.Gadgets.Polynomial
