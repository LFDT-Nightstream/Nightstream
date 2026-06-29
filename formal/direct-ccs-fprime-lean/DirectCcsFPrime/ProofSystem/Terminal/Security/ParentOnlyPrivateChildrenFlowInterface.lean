import DirectCcsFPrime.ProofSystem.Terminal.Security.ParentOnlyPrivateChildrenFlow

/-!
Typed interface for the parent-only private-child flow.

Spec: `specs/ProofSystem/Terminal/Security/ParentOnlyPrivateChildrenFlow.spec.md`
-/

namespace DirectCcsFPrime

namespace ParentOnlyPrivateChildrenFlowInterface

abbrev private_children_flow_of_parent_only_step :=
  @ParentOnlyPrivateChildrenFlow.private_children_flow_of_parent_only_step

abbrev same_private_child_inputs_without_public_child_hashes :=
  @ParentOnlyPrivateChildrenFlow.same_private_child_inputs_without_public_child_hashes

abbrev same_next_parent_source_without_public_child_hashes :=
  @ParentOnlyPrivateChildrenFlow.same_next_parent_source_without_public_child_hashes

end ParentOnlyPrivateChildrenFlowInterface

end DirectCcsFPrime
