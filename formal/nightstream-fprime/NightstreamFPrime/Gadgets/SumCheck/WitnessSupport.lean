import NightstreamFPrime.Circuit.WitnessConstraintSupport
import NightstreamFPrime.Gadgets.SumCheck.FixedChain

namespace NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned

open NightstreamFPrime.Circuit

theorem witnessesFromConstraints {degree roundCount : Nat}
    (interface : Interface degree roundCount) (offset : Nat) :
    WitnessesFromConstraints (Circuit.ops (circuit interface).main offset) :=
  WitnessesFromConstraints.assertions _

end NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned
