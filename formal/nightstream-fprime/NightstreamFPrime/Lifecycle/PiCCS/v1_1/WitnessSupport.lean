import NightstreamFPrime.Gadgets.Multilinear.WitnessSupport
import NightstreamFPrime.Gadgets.SumCheck.WitnessSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-! Each arithmetic child's actual recipe reads occur in its constraints. -/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.SumCheck

theorem InitialClaim.witnessesFromConstraints (interface : InitialClaim.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (InitialClaim.circuit interface).main offset) :=
  Horner.Owned.witnessesFromConstraints _ _

theorem SumcheckChain.witnessesFromConstraints {degree : Nat}
    (interface : SumcheckChain.Interface degree) (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (SumcheckChain.circuit interface).main offset) :=
  FixedChain.Owned.witnessesFromConstraints _ _

theorem EvalKTerminal.witnessesFromConstraints (interface : EvalKTerminal.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (EvalKTerminal.circuit interface).main offset) :=
  PointWeightedHorner.Owned.witnessesFromConstraints _ _ _

theorem EvalATerminal.witnessesFromConstraints (interface : EvalATerminal.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (EvalATerminal.circuit interface).main offset) :=
  PointWeightedHorner.Owned.witnessesFromConstraints _ _ _

theorem NormTerminal.witnessesFromConstraints (interface : NormTerminal.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (NormTerminal.circuit interface).main offset) :=
  Horner.Owned.witnessesFromConstraints _ _

theorem FinalIdentity.witnessesFromConstraints (interface : FinalIdentity.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (FinalIdentity.circuit interface).main offset) := by
  change WitnessesFromConstraints
    (([FinalIdentity.pointOp interface offset] ++
      ([FinalIdentity.matrixPowerOp interface offset] ++
        [FinalIdentity.constraintPowerOp interface offset])) ++
      (FinalIdentity.terminalAssertions interface offset).map Op.assertZero)
  apply WitnessesFromConstraints.append
  · apply WitnessesFromConstraints.append
    · exact WitnessesFromConstraints.call _ _ _
        (PointEquality.Owned.witnessesFromConstraints _ _)
    · apply WitnessesFromConstraints.append
      · exact WitnessesFromConstraints.call _ _ _ (Power.witnessesFromConstraints _ _ _)
      · exact WitnessesFromConstraints.call _ _ _ (Power.witnessesFromConstraints _ _ _)
  · exact WitnessesFromConstraints.assertions _

end NightstreamFPrime.Lifecycle.PiCCS.v1_1
