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
by
  rw [SumcheckChain.circuit_ops]
  change WitnessesFromConstraints
    ([Op.witness (WitnessBatch.arithmetic offset
      (CompactChain.program (SumcheckChain.coreInterface interface offset) offset).recipes)] ++
      (CompactChain.program (SumcheckChain.coreInterface interface offset) offset).checks.map Op.assertZero)
  exact WitnessesFromConstraints.append _ _
    (WitnessesFromConstraints.arithmetic _ _)
    (WitnessesFromConstraints.assertions _)


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

theorem GammaPowers.witnessesFromConstraints (gamma : Quadratic.KExpr) (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (GammaPowers.circuit gamma).main offset) :=
  WitnessesFromConstraints.arithmetic _ _

theorem FinalIdentity.witnessesFromConstraints (interface : FinalIdentity.Interface)
    (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (FinalIdentity.circuit interface).main offset) := by
  change WitnessesFromConstraints
    (([FinalIdentity.pointOp interface offset] ++
      [FinalIdentity.gammaOp interface offset]) ++
      (FinalIdentity.terminalAssertions interface offset).map Op.assertZero)
  apply WitnessesFromConstraints.append
  · exact WitnessesFromConstraints.append _ _
      (WitnessesFromConstraints.call _ _ _ (PointEquality.Owned.witnessesFromConstraints _ _))
      (WitnessesFromConstraints.call _ _ _ (GammaPowers.witnessesFromConstraints _ _))
  · exact WitnessesFromConstraints.assertions _

end NightstreamFPrime.Lifecycle.PiCCS.v1_1
