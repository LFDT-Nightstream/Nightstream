import NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Export.Stage1.PiCCSPackets

/-!
Prepared physical sources retain only classified rows, runtime witness batches,
and permutation blocks. Source indices and their equalities are erased. The
records do not retain source packets or unclassified source rows. This module
owns data provenance, not final-plan coverage or row-checker correctness.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalSources

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

structure RowPayload where
  witnessInstructions : List WitnessInstruction
  assertionRows : List SparseRow

structure ClassifiedRows (source : List Rows.CompiledRow) extends RowPayload where
  rows_eq : (witnessInstructions, assertionRows) =
    (Rows.witnessInstructionsTR source, Rows.assertionRowsTR source)

/-- Classify once; the unclassified source is not part of the result. -/
def prepareRows (source : List Rows.CompiledRow) : ClassifiedRows source :=
  let classified := Rows.classifyRowsTR source
  { witnessInstructions := classified.1
    assertionRows := classified.2
    rows_eq := Rows.classifyRowsTR_eq source }

abbrev PreparedRowBlock (source : OrdinaryRowPlan.Block) :=
  ClassifiedRows (source.rows Data.logicalWidth Data.publicFits)

def prepareRowBlock (source : OrdinaryRowPlan.Block) : PreparedRowBlock source :=
  prepareRows (source.rows Data.logicalWidth Data.publicFits)

/-- Every payload is tied to its source block in the original order. -/
structure PreparedRowBlocks (sources : List OrdinaryRowPlan.Block) where
  blocks : List RowPayload
  rows_eq :
    blocks.map (fun block => (block.witnessInstructions, block.assertionRows)) =
      sources.map (fun source =>
        (Rows.witnessInstructionsTR (source.rows Data.logicalWidth Data.publicFits),
          Rows.assertionRowsTR (source.rows Data.logicalWidth Data.publicFits)))

def prepareRowBlocks :
    (sources : List OrdinaryRowPlan.Block) → PreparedRowBlocks sources
  | [] => { blocks := [], rows_eq := rfl }
  | source :: rest =>
      let prepared := prepareRowBlock source
      let tail := prepareRowBlocks rest
      { blocks := prepared.toRowPayload :: tail.blocks
        rows_eq := congrArg₂ List.cons prepared.rows_eq tail.rows_eq }

def PreparedRowBlocks.append {left right : List OrdinaryRowPlan.Block}
    (first : PreparedRowBlocks left) (second : PreparedRowBlocks right) :
    PreparedRowBlocks (left ++ right) :=
  { blocks := first.blocks ++ second.blocks
    rows_eq := by
      simp only [List.map_append, first.rows_eq, second.rows_eq] }

abbrev PreparedPiRlcSource (source : Nat) :=
  PreparedRowBlocks (OrdinaryRowPlan.piRlcSourceBlocks source)

/-- Includes all digest-lane blocks and the final selector block, in order. -/
def preparePiRlcSource (source : Nat) : PreparedPiRlcSource source :=
  prepareRowBlocks (OrdinaryRowPlan.piRlcSourceBlocks source)

structure PreparedWitnessGroup (build : Unit → PiCCSPackets.Packet)
    extends RowPayload where
  batches : List WitnessBatch
  batches_eq : batches = (build ()).batches
  rows_eq : (witnessInstructions, assertionRows) =
    (Rows.witnessInstructionsTR (build ()).rows,
      Rows.assertionRowsTR (build ()).rows)

/-- Build one packet once, then reuse the common rows classifier. -/
def prepareWitnessGroup (build : Unit → PiCCSPackets.Packet) :
    PreparedWitnessGroup build :=
  let packet := build ()
  let rows := prepareRows packet.rows
  { toRowPayload := rows.toRowPayload
    batches := packet.batches
    batches_eq := rfl
    rows_eq := rows.rows_eq }

abbrev InitialClaimGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.initialClaim Data.logicalWidth Data.publicFits)

abbrev SumcheckGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.sumcheck Data.logicalWidth Data.publicFits)

abbrev EvalKGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.evalK Data.logicalWidth Data.publicFits)

abbrev EvalAGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.evalA Data.logicalWidth Data.publicFits)

abbrev CcsGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.ccs Data.logicalWidth Data.publicFits)

abbrev NormGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.norm Data.logicalWidth Data.publicFits)

abbrev FinalIdentityGroup := PreparedWitnessGroup
  (fun _ => PiCCSPackets.finalIdentity Data.logicalWidth Data.publicFits)

/-- Keep the seven exact source indices after collecting task results. -/
structure PreparedWitnessGroups where
  initialClaim : InitialClaimGroup
  sumcheck : SumcheckGroup
  evalK : EvalKGroup
  evalA : EvalAGroup
  ccs : CcsGroup
  norm : NormGroup
  finalIdentity : FinalIdentityGroup

/-- The emitter's task boundaries can carry these proofs without an IO axiom. -/
structure PreparedWitnessTasks where
  initialClaim : Task (Except IO.Error InitialClaimGroup)
  sumcheck : Task (Except IO.Error SumcheckGroup)
  evalK : Task (Except IO.Error EvalKGroup)
  evalA : Task (Except IO.Error EvalAGroup)
  ccs : Task (Except IO.Error CcsGroup)
  norm : Task (Except IO.Error NormGroup)
  finalIdentity : Task (Except IO.Error FinalIdentityGroup)

abbrev PreparedRowTask (source : OrdinaryRowPlan.Block) :=
  Task (Except IO.Error (PreparedRowBlock source))

abbrev PreparedPiRlcSourceTask (source : Nat) :=
  Task (Except IO.Error (PreparedPiRlcSource source))

structure PreparedPermutationBlocks where
  blocks : List PermutationPlan.Block
  blocks_eq : blocks = PermutationPlan.canonicalBlocks ()

def preparePermutationBlocks (_unit : Unit) : PreparedPermutationBlocks :=
  { blocks := PermutationPlan.canonicalBlocks ()
    blocks_eq := rfl }

end NightstreamFPrime.Export.Stage1.PreparedPhysicalSources
