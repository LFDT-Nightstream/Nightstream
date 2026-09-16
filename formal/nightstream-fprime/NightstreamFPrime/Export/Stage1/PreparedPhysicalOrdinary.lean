import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.PreparedPhysicalSources

/-!
Assemble the canonical ordinary-row payloads from already prepared sources.
The source equalities occur only in erased fields. Packet batches remain in
PreparedWitnessGroups for generation. This module does not collect tasks or
claim coverage of the other selected package row categories.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalOrdinary

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open PreparedPhysicalSources

private def packetBlock (source : OrdinaryRowPlan.Block)
    {build : Unit → PiCCSPackets.Packet} (group : PreparedWitnessGroup build)
    (rowsEqual : (build ()).rows = source.rows Data.logicalWidth Data.publicFits) :
    PreparedRowBlock source :=
  { toRowPayload := group.toRowPayload
    rows_eq := group.rows_eq.trans (congrArg
      (fun rows => (Rows.witnessInstructionsTR rows, Rows.assertionRowsTR rows))
      rowsEqual) }

private def piCcs (statement : PreparedRowBlock .statementBinding)
    (groups : PreparedWitnessGroups) :
    PreparedRowBlocks (OrdinaryRowPlan.piCcsBlocks ()) :=
  let initialClaim := packetBlock .initialClaim groups.initialClaim
    (PiCCSPackets.initialClaim_rows Data.logicalWidth Data.publicFits)
  let sumcheck := packetBlock .sumcheck groups.sumcheck
    (PiCCSPackets.sumcheck_rows Data.logicalWidth Data.publicFits)
  let evalK := packetBlock .evalK groups.evalK
    (PiCCSPackets.evalK_rows Data.logicalWidth Data.publicFits)
  let evalA := packetBlock .evalA groups.evalA
    (PiCCSPackets.evalA_rows Data.logicalWidth Data.publicFits)
  let ccs := packetBlock .ccs groups.ccs
    (PiCCSPackets.ccs_rows Data.logicalWidth Data.publicFits)
  let norm := packetBlock .norm groups.norm
    (PiCCSPackets.norm_rows Data.logicalWidth Data.publicFits)
  let finalIdentity := packetBlock .finalIdentity groups.finalIdentity
    (PiCCSPackets.finalIdentity_rows Data.logicalWidth Data.publicFits)
  { blocks := [statement.toRowPayload, initialClaim.toRowPayload,
      sumcheck.toRowPayload, evalK.toRowPayload, evalA.toRowPayload,
      ccs.toRowPayload, norm.toRowPayload, finalIdentity.toRowPayload]
    rows_eq := by
      simp only [OrdinaryRowPlan.piCcsBlocks, List.map_cons, List.map_nil]
      rw [statement.rows_eq, initialClaim.rows_eq, sumcheck.rows_eq,
        evalK.rows_eq, evalA.rows_eq, ccs.rows_eq, norm.rows_eq,
        finalIdentity.rows_eq] }

private def singleton {source : OrdinaryRowPlan.Block}
    (prepared : PreparedRowBlock source) : PreparedRowBlocks [source] :=
  { blocks := [prepared.toRowPayload]
    rows_eq := congrArg (fun value => [value]) prepared.rows_eq }

/-- Assemble only payload references, in the canonical ordinary block order. -/
def assemble (statement : PreparedRowBlock .statementBinding)
    (groups : PreparedWitnessGroups)
    (piRlc : PreparedRowBlocks (OrdinaryRowPlan.piRlcBlocks ()))
    (piDec : PreparedRowBlock (OrdinaryRowPlan.piDecBlock ()))
    (running : PreparedRowBlock (OrdinaryRowPlan.runningTransitionBlock ())) :
    PreparedRowBlocks (OrdinaryRowPlan.canonicalBlocks ()) :=
  (((piCcs statement groups).append piRlc).append (singleton piDec)).append
    (singleton running)

private theorem payloadFor {sources : List OrdinaryRowPlan.Block}
    (prepared : PreparedRowBlocks sources) {source : OrdinaryRowPlan.Block}
    (member : source ∈ sources) :
    ∃ payload ∈ prepared.blocks,
      (payload.witnessInstructions, payload.assertionRows) =
        (Rows.witnessInstructionsTR (source.rows Data.logicalWidth Data.publicFits),
          Rows.assertionRowsTR (source.rows Data.logicalWidth Data.publicFits)) := by
  apply List.mem_map.mp
  rw [prepared.rows_eq]
  exact List.mem_map.mpr ⟨source, member, rfl⟩

/-- Checking all assembled payloads covers both classified arithmetic fields.
No source-row list is materialized by this proposition or its proof. -/
theorem holds (prepared : PreparedRowBlocks (OrdinaryRowPlan.canonicalBlocks ()))
    (env : Env)
    (instructions : ∀ block ∈ prepared.blocks,
      ∀ instruction ∈ block.witnessInstructions, instruction.Holds env)
    (assertions : ∀ block ∈ prepared.blocks,
      ∀ row ∈ block.assertionRows, row.Holds env) :
    (∀ instruction ∈ Rows.witnessInstructionsTR (Data.arithmeticRows ()),
      instruction.Holds env) ∧
    (∀ row ∈ Rows.assertionRowsTR (Data.arithmeticRows ()), row.Holds env) := by
  constructor
  · intro instruction member
    rw [← OrdinaryRowPlan.canonicalWitnessInstructions_expand] at member
    rcases List.mem_flatMap.mp member with ⟨source, sourceMember, instructionMember⟩
    obtain ⟨payload, payloadMember, equal⟩ := payloadFor prepared sourceMember
    apply instructions payload payloadMember instruction
    have same := congrArg Prod.fst equal
    change payload.witnessInstructions =
      Rows.witnessInstructionsTR (source.rows Data.logicalWidth Data.publicFits) at same
    rw [same]
    exact instructionMember
  · intro row member
    rw [← OrdinaryRowPlan.canonicalAssertionRows_expand] at member
    rcases List.mem_flatMap.mp member with ⟨source, sourceMember, rowMember⟩
    obtain ⟨payload, payloadMember, equal⟩ := payloadFor prepared sourceMember
    apply assertions payload payloadMember row
    have same := congrArg Prod.snd equal
    change payload.assertionRows =
      Rows.assertionRowsTR (source.rows Data.logicalWidth Data.publicFits) at same
    rw [same]
    exact rowMember

end NightstreamFPrime.Export.Stage1.PreparedPhysicalOrdinary
