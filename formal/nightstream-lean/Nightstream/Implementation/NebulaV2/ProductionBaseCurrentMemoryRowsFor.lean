import Nightstream.Implementation.NebulaV2.BaseManifestStateAuthority
import Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows

/-!
Contract: fixed base-relation ownership of the first produced memory batch.

The selected base artifact must contain the complete candidate-specific
checked-memory row program. Its first boundary shares the outgoing base carry
columns, and every boundary shares the verifier-owned compact-chain header
columns. A satisfying base assignment therefore derives the memory batch that
is committed into claim zero. No memory result is accepted from the prover or
from a higher-level semantic witness.

`Authority` is a verifier-key/generator certificate. A generated artifact must
construct it and bind it to the verifier-key identity. This module does not
claim that a generated artifact or deployed verifier exists.

Assurance tier: base relation composition.

Emits constraints: no new rows. It certifies rows already in the base
manifest.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-- Static evidence that one checked-memory batch belongs to the selected
base relation and uses its authority-bearing outgoing carry. -/
structure Authority
    (candidate : Id)
    {widths : FullClaimEnvelope.CompilerWidths}
    (artifact : BaseManifestSchema.Artifact widths) where
  layout : ProductionMemoryCheckedBatchRows.Layout candidate
  valid : layout.Valid
  firstBoundaryCarryExact :
    (layout.boundaries 0).carry =
      artifact.layouts.outgoingMemoryCarry.carry
  boundaryHeadersExact : forall index role lane,
    (layout.boundaries index).carry.headerColumn role lane =
      artifact.layouts.outgoingMemoryCarry.carry.headerColumn role lane
  rowsIncluded :
    R1CS.rowsIncluded (ProductionMemoryCheckedBatchRows.rows layout)
      artifact.programRows = true

namespace Authority

/-- Satisfaction of the selected base relation implies satisfaction of the
entire current-memory batch. -/
theorem satisfied
    {candidate : Id}
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    (authority : Authority candidate artifact)
    {assignment : Nat -> Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (ProductionMemoryCheckedBatchRows.rows authority.layout)
      assignment := by
  intro row member
  exact satisfies row
    (rowsIncluded_sound authority.rowsIncluded row member)

/-- The headers placed in the base outgoing carry also occupy every boundary
of the fixed checked-memory batch. -/
theorem headersPlaced
    {candidate : Id}
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    (authority : Authority candidate artifact)
    {assignment : Nat -> Nat}
    (call : BaseManifestStateAuthority.Call artifact assignment)
    (headers : ChainHeaders Digest.Value)
    (headersExact : call.headers = headers) :
    ProductionMemoryCheckedBatchRows.HeadersPlaced authority.layout
      assignment headers := by
  subst headers
  intro index role lane
  rw [authority.boundaryHeadersExact index role lane]
  exact call.outgoingHeadersPlaced role lane

/-- The first memory batch is reconstructed from the same satisfying
assignment as the selected base relation. -/
@[irreducible] noncomputable def result
    {candidate : Id}
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    (authority : Authority candidate artifact)
    {assignment : Nat -> Nat}
    (call : BaseManifestStateAuthority.Call artifact assignment)
    (headers : ChainHeaders Digest.Value)
    (headersExact : call.headers = headers)
    (satisfies : Satisfies artifact.programRows assignment) :
    ProductionMemoryCheckedBatchRows.Result authority.layout assignment
      headers :=
  ProductionMemoryCheckedBatchRows.derive authority.valid headers
    call.canonicalAssignment call.one
    (authority.headersPlaced call headers headersExact)
    (authority.satisfied satisfies)

/-- The row-derived first boundary and the base outgoing value use the same
field columns. -/
theorem outgoingValue_eq_firstBoundary
    {candidate : Id}
    {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    (authority : Authority candidate artifact)
    {assignment : Nat -> Nat}
    (call : BaseManifestStateAuthority.Call artifact assignment)
    (headers : ChainHeaders Digest.Value)
    (headersExact : call.headers = headers)
    (satisfies : Satisfies artifact.programRows assignment) :
    call.outgoingValue =
      (authority.result call headers headersExact satisfies).boundary 0 := by
  apply MemoryCarryCodec.Value.fieldValue_injective
  funext tag
  calc
    call.outgoingValue.fieldValue tag =
        assignment
          (artifact.layouts.outgoingMemoryCarry.carry.fieldColumn tag) :=
      (call.outgoingCarryColumnsMatch satisfies).placed tag |>.symm
    _ = assignment
        ((authority.layout.boundaries 0).carry.fieldColumn tag) := by
      rw [authority.firstBoundaryCarryExact]
    _ = ((authority.result call headers headersExact satisfies).boundary 0).fieldValue tag :=
      ((authority.result call headers headersExact satisfies).boundaryParsed 0).placed tag

end Authority

end Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor
