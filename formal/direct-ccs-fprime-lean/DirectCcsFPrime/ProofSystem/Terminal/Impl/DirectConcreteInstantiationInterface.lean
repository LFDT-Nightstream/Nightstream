import DirectCcsFPrime.ProofSystem.Terminal.Impl.DirectConcreteInstantiation

/-!
Typed interface for concrete direct CCS terminal instantiation.

Spec: `specs/ProofSystem/Terminal/Impl/DirectConcreteInstantiation.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectConcreteInstantiationInterface

abbrev ConcreteCEData :=
  DirectConcreteInstantiation.ConcreteCEData

abbrev ConcreteCEData.commitMap :=
  @DirectConcreteInstantiation.ConcreteCEData.commitMap

abbrev ConcreteCEData.ce :=
  @DirectConcreteInstantiation.ConcreteCEData.ce

abbrev ConcreteCEData.ajtaiBackedCommitMap :=
  @DirectConcreteInstantiation.ConcreteCEData.ajtaiBackedCommitMap

abbrev terminal_soundness_of_concrete_ce_and_msis :=
  @DirectConcreteInstantiation.terminal_soundness_of_concrete_ce_and_msis

end DirectConcreteInstantiationInterface

end DirectCcsFPrime
