import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary.Generated.ShiftedTernaryArtifact
import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary.Generated.ShiftedTernarySharedSlotsArtifact

/-!
Stable facade for generated shifted-ternary evidence.

Owns: the public import boundary for the source R1CS artifact and its exact
gadget-native shared-slot instantiation.

Does not own: semantics, soundness, or row-removal arguments.

Emits constraints: no.

Authority boundary: both children are generated evidence; handwritten
correspondence modules must prove what their rows mean.

| Child | Evidence | Semantic owner |
|---|---|---|
| `ShiftedTernaryArtifact` | Exact source rows and witnesses | `Ownership.ShiftedTernary` |
| `ShiftedTernarySharedSlotsArtifact` | Exact target aliases, row roles, and retained CCS rows | `Correspondence.ShiftedTernary.SharedSlots` |
-/
