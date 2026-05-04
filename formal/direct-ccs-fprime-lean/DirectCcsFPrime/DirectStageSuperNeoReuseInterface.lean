import DirectCcsFPrime.DirectStageSuperNeoReuse

/-!
Typed interface for reusing theorem-native SuperNeo stage contexts.

Spec: `specs/DirectStageSuperNeoReuse.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectStageSuperNeoReuseInterface

abbrev reusedStageAuthority_of_ceRelation :=
  @DirectStageSuperNeoReuse.reusedStageAuthority_of_ceRelation

abbrev reusedStageAuthority_of_section71Context :=
  DirectStageSuperNeoReuse.reusedStageAuthority_of_section71Context

abbrev Section71ContextualStageComputations :=
  DirectStageSuperNeoReuse.Section71ContextualStageComputations

abbrev Section71ContextualStageComputations_toContextualReused :=
  @DirectStageSuperNeoReuse.Section71ContextualStageComputations.toContextualReused

abbrev piCCSStrong_of_compute :=
  @DirectStageSuperNeoReuse.Section71ContextualStageComputations.piCCSStrong_of_compute

abbrev piRLCWeak_of_compute :=
  @DirectStageSuperNeoReuse.Section71ContextualStageComputations.piRLCWeak_of_compute

abbrev piDECKnowledge_of_piCCS_compute :=
  @DirectStageSuperNeoReuse.Section71ContextualStageComputations.piDECKnowledge_of_piCCS_compute

abbrev piDECKnowledge_of_piRLC_compute :=
  @DirectStageSuperNeoReuse.Section71ContextualStageComputations.piDECKnowledge_of_piRLC_compute

end DirectStageSuperNeoReuseInterface

end DirectCcsFPrime
