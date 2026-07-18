import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority

namespace Tests.FPrimeConcretePhi81ParentOnlyAuthority

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ParentOnlyAuthority

example : Substitution.leftSlot ≠ Substitution.rightSlot :=
  Substitution.slots_ne

example :
    Substitution.leftRunning.toPaper ≠ Substitution.rightRunning.toPaper :=
  Substitution.paperRunning_ne

example
    {Digest : Type}
    (handle : Substitution.Statement -> Digest) :
    ¬ Substitution.ParentOnlyAccumulatorBinds handle :=
  Substitution.no_parentOnlyAccumulator_binds handle

end Tests.FPrimeConcretePhi81ParentOnlyAuthority
