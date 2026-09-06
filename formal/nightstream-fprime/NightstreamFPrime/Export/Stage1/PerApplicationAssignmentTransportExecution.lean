import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExpressions
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportProducts

/-!
Owns the executable meaning of the serialized per-application assignment
transport. The interpreter reads the compact product recipes, expression
lists, block metadata, and affine source runs. It performs point lookup and
does not materialize the complete logical assignment.

The public executor first checks exact structural equality with the canonical
plan. This fail-closed check consumes the opcode, selector, count, and fixed
recipe fields that are validation metadata rather than value inputs.

`PerApplicationCanonicalAssignment.RawValues.schedule` remains the semantic
assignment authority. The final theorem proves that this serialized-plan
interpreter is pointwise equal to that assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationAssignmentBlocks
open PerApplicationAssignmentPlan
open PerApplicationCanonicalAssignment

abbrev Program := Lifecycle.Stage1.Application.Program

abbrev BaseValues (program : Program) :=
  PerApplicationAssignmentTransportProducts.BaseValues program

/-- Derive the two retained product suffixes from the serialized recipes. -/
def rawValues (program : Program) (plan : Plan) (base : BaseValues program) :
    RawValues program where
  base := base
  groupValue := fun invocation group =>
    PerApplicationAssignmentTransportProducts.phi81GroupValue
      plan.phi81 program base invocation group.val
  products := fun candidate =>
    PerApplicationAssignmentTransportProducts.first54ProductValue
      plan.first54 program base candidate.val

/-- Raw values derived from the exact transport carried by schema 6. -/
def canonicalRawValues (program : Program) (base : BaseValues program) :
    RawValues program :=
  rawValues program (PerApplicationAssignmentTransport.canonical program) base

/-- The schema-6 Phi81 recipe computes the existing honest retained group. -/
theorem canonicalRawValues_groupValue_eq_honestGroupValue
    (program : Program) (base : BaseValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 33) :
    (canonicalRawValues program base).groupValue invocation group =
      PiRLCProductPlan.honestGroupValue
        (PiRLCRetainedInputs.productInputs
          (PerApplicationCanonicalEncodes.retainedGeometry program))
        (canonicalRawValues program base).assignment invocation group := by
  change
    PerApplicationAssignmentTransportProducts.phi81GroupValue
        phi81GroupRecipe program base invocation group.val = _
  exact
    PerApplicationAssignmentTransportProducts.canonical_phi81GroupValue_eq_honestGroupValue
      (canonicalRawValues program base) invocation group

/-- The schema-6 First54 recipe computes the existing honest product. -/
theorem canonicalRawValues_products_eq_honestProducts
    (program : Program) (base : BaseValues program)
    (candidate : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    (canonicalRawValues program base).products candidate =
      PiRLCFirst54DirectPlan.honestProducts program base candidate := by
  change
    PerApplicationAssignmentTransportProducts.first54ProductValue
        first54ProductRecipe program base candidate.val = _
  exact
    PerApplicationAssignmentTransportProducts.canonical_first54ProductValue_eq_honestProducts
      (canonicalRawValues program base) candidate

/-- Direct point lookup in a serialized affine-run stream. -/
private def runSource : List AffineRuns.Run → Nat → Nat
  | [], _ => 0
  | run :: rest, slot =>
      if slot < run.count then
        run.first + run.step * slot
      else
        runSource rest (slot - run.count)

private theorem affineValues_getD_of_lt (first step count slot : Nat)
    (inside : slot < count) :
    (AffineRuns.values first step count).getD slot 0 =
      first + step * slot := by
  induction count generalizing first slot with
  | zero => omega
  | succ count inductionHypothesis =>
      cases slot with
      | zero => simp [AffineRuns.values]
      | succ slot =>
          simp only [AffineRuns.values, List.getD_cons_succ]
          rw [inductionHypothesis (first := first + step)
            (slot := slot) (by omega)]
          ring

/-- Direct run lookup is the pointwise meaning of affine-run expansion. -/
private theorem runSource_eq_expand_getD
    (runs : List AffineRuns.Run) (slot : Nat) :
    runSource runs slot = (AffineRuns.expand runs).getD slot 0 := by
  induction runs generalizing slot with
  | nil => rfl
  | cons run rest inductionHypothesis =>
      change
        (if slot < run.count then run.first + run.step * slot
          else runSource rest (slot - run.count)) =
        (run.expand ++ AffineRuns.expand rest).getD slot 0
      by_cases inside : slot < run.count
      · rw [if_pos inside, List.getD_append]
        · simpa [AffineRuns.Run.expand] using
            (affineValues_getD_of_lt run.first run.step run.count slot inside).symm
        · simpa using inside
      · have after : run.expand.length ≤ slot := by
          simpa using Nat.le_of_not_gt inside
        rw [if_neg inside,
          List.getD_append_right _ _ _ _ after]
        simpa using inductionHypothesis (slot := slot - run.count)

/-- Canonical run lookup selects the exact Lean-owned source index. -/
private theorem canonical_runSource (program : Program) (kind : BlockKind)
    (slot : Nat)
    (bound :
      slot < (PerApplicationAssignmentBlocks.entry program kind).block.slotCount) :
    runSource
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind).sourceRuns
        slot =
      PerApplicationAssignmentBlocks.sourceIndex program kind
        ⟨slot, bound⟩ := by
  rw [runSource_eq_expand_getD]
  change
    (AffineRuns.expand
      (PerApplicationAssignmentBlocks.sourceRunsFor program kind)).getD
        slot 0 = _
  rw [PerApplicationAssignmentBlocks.sourceRuns_expand]
  unfold PerApplicationAssignmentBlocks.sourceIndices
  exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD _ ⟨slot, bound⟩ 0

/-- Evaluate one serialized expression against the retained-source view. -/
private def expressionValue {program : Program} (expressions : List Expr)
    (raw : RawValues program) (index : Nat) : F :=
  (expressions.getD index 0).eval
    (SourceCompiler.sourceEnv raw.retainedSource)

/-- Total value view of one serialized block source domain. -/
private def domainValue (program : Program) (plan : Plan)
    (raw : RawValues program) : SourceDomain → Nat → F
  | .retained => SourceCompiler.sourceEnv raw.retainedSource
  | .piCcsPayload => expressionValue plan.payloadExpressions raw
  | .physicalBase => SourceCompiler.sourceEnv raw.base

/-- On the canonical plan, each domain/index pair reads the scalar selected
by the existing opcode interpreter. -/
private theorem canonical_domain_source (program : Program)
    (raw : RawValues program) (kind : BlockKind) (slot : Nat)
    (entryBound :
      slot < (PerApplicationAssignmentBlocks.entry program kind).block.slotCount)
    (rawBound : slot < (kind.expand raw).block.slotCount) :
    domainValue program (PerApplicationAssignmentTransport.canonical program)
        raw (PerApplicationAssignmentBlocks.sourceDomainOf kind)
        (PerApplicationAssignmentBlocks.sourceIndex program kind
          ⟨slot, entryBound⟩) =
      (kind.expand raw).source
        ((kind.expand raw).block.source ⟨slot, rawBound⟩) := by
  cases kind <;>
    simp [domainValue, expressionValue,
      PerApplicationAssignmentTransport.canonical_payloadExpressions,
      PerApplicationAssignmentBlocks.sourceDomainOf,
      PerApplicationAssignmentBlocks.sourceIndex,
      PerApplicationAssignmentBlocks.entry,
      PerApplicationAssignmentBlocks.zeroRaw,
      PerApplicationAssignmentPlan.BlockKind.expand,
      PerApplicationAssignmentPlan.BlockKind.template,
      PerApplicationCanonicalAssignment.Canonical.ofBlock,
      CanonicalBlockAssignment.ofBlock,
      PerApplicationCanonicalAssignment.RawValues.payloadSource,
      PerApplicationCanonicalAssignment.RawValues.applicationSource,
      PiCCSPoseidonPreservation.sourceAssignment,
      PiCCSActionPayloadBlock.block,
      FieldSuffixBlock.block, FieldSuffixBlock.derivedColumn]
  · have payloadBound : slot < PiCCSActionPayloadBlock.payloadCount := by
      simpa [PerApplicationAssignmentPlan.BlockKind.expand,
        PerApplicationAssignmentPlan.BlockKind.template,
        PerApplicationCanonicalAssignment.Canonical.ofBlock,
        CanonicalBlockAssignment.ofBlock,
        PiCCSActionPayloadBlock.block, FieldSuffixBlock.block] using rawBound
    let index : Fin PiCCSActionPayloadBlock.payloadCount :=
      ⟨slot, payloadBound⟩
    calc
      ((PerApplicationAssignmentTransport.payloadExpressions program).getD
          slot 0).eval (SourceCompiler.sourceEnv raw.retainedSource) =
        PiCCSActionPayloadBlock.payloadValue program raw.retainedSource
          index := by
        simpa [index] using
          (PerApplicationAssignmentTransportExpressions.payloadExpression_eval
            program raw index)
      _ = PiCCSActionPayloadBlock.sourceAssignment program raw.retainedSource
          ⟨PiCCSActionPayloadBlock.prefixSourceWidth program + slot, by
            unfold PiCCSActionPayloadBlock.sourceWidth
              FieldSuffixBlock.sourceWidth
            omega⟩ := by
        symm
        simpa [index, PiCCSActionPayloadBlock.payloadColumn,
          FieldSuffixBlock.derivedColumn] using
            (PiCCSActionPayloadBlock.sourceAssignment_payload program
              raw.retainedSource index)
/-- One serialized block slot. Invalid run coverage fails closed. -/
private def blockSlotValue (program : Program) (plan : Plan)
    (raw : RawValues program) (block : BlockPlan)
    (slot : Fin block.slotCount) : F :=
  if _covered :
      (block.sourceRuns.map AffineRuns.Run.count).sum = block.slotCount then
    domainValue program plan raw block.sourceDomain
      (runSource block.sourceRuns slot.val)
  else
    0

/-- An identity-source block lets the compact plan supply one value per slot
without inventing another retained source layout. -/
private def identityBlock (block : BlockPlan) :
    LowNormBlock.Block block.slotCount where
  kind := block.slotKind
  slotCount := block.slotCount
  source := fun slot => slot

/-- Interpret one serialized block without expanding its slots. -/
private def transportBlockValue (program : Program) (plan : Plan)
    (raw : RawValues program) (block : BlockPlan) :
    CanonicalBlockAssignment.BlockValue :=
  CanonicalBlockAssignment.ofBlock (identityBlock block)
    (blockSlotValue program plan raw block)

@[simp] private theorem transportBlockValue_block_kind
    (program : Program) (plan : Plan) (raw : RawValues program)
    (block : BlockPlan) :
    (transportBlockValue program plan raw block).block.kind = block.slotKind := by
  rfl

@[simp] private theorem transportBlockValue_block_slotCount
    (program : Program) (plan : Plan) (raw : RawValues program)
    (block : BlockPlan) :
    (transportBlockValue program plan raw block).block.slotCount =
      block.slotCount := by
  rfl

/-- Two function-valued blocks with the same visible geometry and selected
slot values have the same direct coordinate lookup. -/
private theorem blockValue_coordinateAt_eq
    (left right : CanonicalBlockAssignment.BlockValue)
    (kindEq : left.block.kind = right.block.kind)
    (slotCountEq : left.block.slotCount = right.block.slotCount)
    (sourceEq : ∀ (slot : Nat)
      (leftBound : slot < left.block.slotCount)
      (rightBound : slot < right.block.slotCount),
      left.source (left.block.source ⟨slot, leftBound⟩) =
        right.source (right.block.source ⟨slot, rightBound⟩))
    (index : Nat) : left.coordinateAt index = right.coordinateAt index := by
  have countEq : left.coordinateCount = right.coordinateCount := by
    unfold CanonicalBlockAssignment.BlockValue.coordinateCount
      LowNormBlock.Block.coordinateCount
    rw [slotCountEq, kindEq]
  unfold CanonicalBlockAssignment.BlockValue.coordinateAt
  by_cases inside : index < left.coordinateCount
  · have rightInside : index < right.coordinateCount := by
      rw [← countEq]
      exact inside
    rw [dif_pos inside, dif_pos rightInside]
    dsimp only
    let leftSlot : Fin left.block.slotCount :=
      ⟨index / left.block.kind.width, by
        apply (Nat.div_lt_iff_lt_mul (by
          cases left.block.kind <;>
            norm_num [LowNormSlot.Kind.width, BalancedTernary.width])).2
        simpa [CanonicalBlockAssignment.BlockValue.coordinateCount,
          LowNormBlock.Block.coordinateCount] using inside⟩
    let rightSlot : Fin right.block.slotCount :=
      ⟨index / right.block.kind.width, by
        apply (Nat.div_lt_iff_lt_mul (by
          cases right.block.kind <;>
            norm_num [LowNormSlot.Kind.width, BalancedTernary.width])).2
        simpa [CanonicalBlockAssignment.BlockValue.coordinateCount,
          LowNormBlock.Block.coordinateCount] using rightInside⟩
    let leftCoordinate : Fin left.block.kind.width :=
      ⟨index % left.block.kind.width, Nat.mod_lt _ (by
        cases left.block.kind <;>
          norm_num [LowNormSlot.Kind.width, BalancedTernary.width])⟩
    let rightCoordinate : Fin right.block.kind.width :=
      ⟨index % right.block.kind.width, Nat.mod_lt _ (by
        cases right.block.kind <;>
          norm_num [LowNormSlot.Kind.width, BalancedTernary.width])⟩
    change LowNormSlot.coordinate left.block.kind
        (left.source (left.block.source leftSlot)) leftCoordinate =
      LowNormSlot.coordinate right.block.kind
        (right.source (right.block.source rightSlot)) rightCoordinate
    have slotEq : leftSlot.val = rightSlot.val := by
      simp [leftSlot, rightSlot, kindEq]
    have rightBound : leftSlot.val < right.block.slotCount := by
      rw [← slotCountEq]
      exact leftSlot.isLt
    have leftSlotEq :
        (⟨leftSlot.val, leftSlot.isLt⟩ : Fin left.block.slotCount) =
          leftSlot := by
      apply Fin.ext
      rfl
    have rightSlotEq :
        (⟨leftSlot.val, rightBound⟩ : Fin right.block.slotCount) =
          rightSlot := by
      apply Fin.ext
      exact slotEq
    rw [← leftSlotEq,
      sourceEq leftSlot.val leftSlot.isLt rightBound, rightSlotEq]
    have coordinateEq :
        Fin.cast (congrArg LowNormSlot.Kind.width kindEq) leftCoordinate =
          rightCoordinate := by
      apply Fin.ext
      simp [leftCoordinate, rightCoordinate, kindEq]
    rw [← coordinateEq]
    exact (LowNormSlot.coordinate_cast kindEq _ leftCoordinate).symm
  · have rightOutside : ¬ index < right.coordinateCount := by
      rw [← countEq]
      exact inside
    rw [dif_neg inside, dif_neg rightOutside]

private theorem canonicalBlock_kind (program : Program)
    (raw : RawValues program) (kind : BlockKind) :
    (transportBlockValue program
        (PerApplicationAssignmentTransport.canonical program) raw
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)).block.kind =
      (kind.expand raw).block.kind := by
  change
    (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind).slotKind = _
  rw [PerApplicationAssignmentBlocks.BlockPlan.ofKind_slotKind]
  exact (PerApplicationAssignmentBlocks.entry_geometry_eq_expand
    program raw kind).1

private theorem canonicalBlock_slotCount (program : Program)
    (raw : RawValues program) (kind : BlockKind) :
    (transportBlockValue program
        (PerApplicationAssignmentTransport.canonical program) raw
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)).block.slotCount =
      (kind.expand raw).block.slotCount := by
  change
    (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind).slotCount = _
  rw [PerApplicationAssignmentBlocks.BlockPlan.ofKind_slotCount]
  exact (PerApplicationAssignmentBlocks.entry_geometry_eq_expand
    program raw kind).2

/-- One canonical serialized block has the same coordinate lookup as its
Lean-owned opcode expansion. -/
private theorem canonicalBlock_coordinateAt (program : Program)
    (raw : RawValues program) (kind : BlockKind) (index : Nat) :
    (transportBlockValue program
        (PerApplicationAssignmentTransport.canonical program) raw
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)).coordinateAt
          index =
      (kind.expand raw).coordinateAt index := by
  apply blockValue_coordinateAt_eq
  · exact canonicalBlock_kind program raw kind
  · exact canonicalBlock_slotCount program raw kind
  · intro slot leftBound rightBound
    change slot <
      (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind).slotCount at leftBound
    have entryBound : slot <
        (PerApplicationAssignmentBlocks.entry program kind).block.slotCount := by
      simpa only [PerApplicationAssignmentBlocks.BlockPlan.ofKind_slotCount] using
        leftBound
    change
      blockSlotValue program
          (PerApplicationAssignmentTransport.canonical program) raw
          (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)
          ⟨slot, leftBound⟩ =
        (kind.expand raw).source
          ((kind.expand raw).block.source ⟨slot, rightBound⟩)
    unfold blockSlotValue
    rw [dif_pos
      (PerApplicationAssignmentBlocks.BlockPlan.ofKind_sourceRuns_count
        program kind)]
    rw [canonical_runSource program kind slot entryBound]
    exact
      canonical_domain_source program raw kind slot entryBound rightBound

private theorem canonicalBlock_coordinateCount (program : Program)
    (raw : RawValues program) (kind : BlockKind) :
    (transportBlockValue program
        (PerApplicationAssignmentTransport.canonical program) raw
        (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)).coordinateCount =
      (kind.expand raw).coordinateCount := by
  unfold CanonicalBlockAssignment.BlockValue.coordinateCount
    LowNormBlock.Block.coordinateCount
  rw [canonicalBlock_slotCount program raw kind,
    canonicalBlock_kind program raw kind]

/-- Interpret the serialized block order. The list has 38 function-valued
entries; it contains no expanded slot or coordinate list. -/
private def transportSchedule (program : Program) (plan : Plan)
    (raw : RawValues program) : CanonicalBlockAssignment.Schedule :=
  plan.blocks.map (transportBlockValue program plan raw)

private theorem canonicalBlocks_coordinateAt_aux (program : Program)
    (raw : RawValues program) (kinds : List BlockKind) (index : Nat) :
    CanonicalBlockAssignment.coordinateAt
        ((kinds.map
          (PerApplicationAssignmentBlocks.BlockPlan.ofKind program)).map
            (transportBlockValue program
              (PerApplicationAssignmentTransport.canonical program) raw))
        index =
      CanonicalBlockAssignment.coordinateAt
        (kinds.map (BlockKind.expand raw)) index := by
  induction kinds generalizing index with
  | nil => rfl
  | cons kind kinds inductionHypothesis =>
      simp only [List.map_cons]
      unfold CanonicalBlockAssignment.coordinateAt
      have countEq := canonicalBlock_coordinateCount program raw kind
      by_cases inside : index <
          (transportBlockValue program
            (PerApplicationAssignmentTransport.canonical program) raw
            (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind)).coordinateCount
      · have rightInside : index < (kind.expand raw).coordinateCount := by
          rw [← countEq]
          exact inside
        rw [dif_pos inside, dif_pos rightInside]
        exact canonicalBlock_coordinateAt program raw kind index
      · have rightOutside : ¬ index < (kind.expand raw).coordinateCount := by
          rw [← countEq]
          exact inside
        rw [dif_neg inside, dif_neg rightOutside, countEq]
        exact inductionHypothesis _

/-- The canonical serialized block program and the opcode plan have the same
pointwise coordinate stream. -/
private theorem canonicalBlocks_coordinateAt (program : Program)
    (raw : RawValues program) (index : Nat) :
    CanonicalBlockAssignment.coordinateAt
        (transportSchedule program
          (PerApplicationAssignmentTransport.canonical program) raw) index =
      CanonicalBlockAssignment.coordinateAt
        (PerApplicationAssignmentPlan.expand raw) index := by
  change
    CanonicalBlockAssignment.coordinateAt
        ((PerApplicationAssignmentBlocks.canonical program).map
          (transportBlockValue program
            (PerApplicationAssignmentTransport.canonical program) raw)) index = _
  unfold PerApplicationAssignmentBlocks.canonical
    PerApplicationAssignmentPlan.expand
  exact canonicalBlocks_coordinateAt_aux program raw
    PerApplicationAssignmentPlan.canonicalKinds index

/-- Evaluate the serialized four-word output-digest program. -/
private def outputDigest {program : Program} (plan : Plan)
    (raw : RawValues program) : Digest :=
  List.ofFn fun lane : Fin PilotProduction.digestWords =>
    expressionValue plan.outputDigestExpressions raw lane.val

private theorem canonical_outputDigest (program : Program)
    (raw : RawValues program) :
    outputDigest (PerApplicationAssignmentTransport.canonical program) raw =
      raw.outputDigest := by
  unfold outputDigest
  calc
    List.ofFn (fun lane : Fin PilotProduction.digestWords =>
        expressionValue
          (PerApplicationAssignmentTransport.canonical program).outputDigestExpressions
          raw lane.val) =
      List.ofFn (fun lane : Fin PilotProduction.digestWords =>
        raw.outputDigest.getD lane.val 0) := by
      apply congrArg List.ofFn
      funext lane
      simpa [expressionValue,
        PerApplicationAssignmentTransport.canonical] using
        (PerApplicationAssignmentTransportExpressions.outputDigestExpression_eval
          program raw lane)
    _ = raw.outputDigest := by
      unfold PerApplicationCanonicalAssignment.RawValues.outputDigest
      apply congrArg List.ofFn
      funext lane
      exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD _ lane 0

/-- Execute the already-validated serialized assignment transport by point
lookup. -/
private def executeUnchecked (program : Program) (plan : Plan)
    (base : BaseValues program) :
    Assignment F (PerApplicationFixedPoint.logicalWidth program) :=
  let raw := rawValues program plan base
  CanonicalBlockAssignment.assignment
    (encodedHashCells (outputDigest plan raw))
    (transportSchedule program plan raw)

private theorem canonical_execute_eq_plan_execute (program : Program)
    (base : BaseValues program) :
    executeUnchecked program
        (PerApplicationAssignmentTransport.canonical program) base =
      PerApplicationAssignmentPlan.execute (canonicalRawValues program base) := by
  funext column
  let raw := canonicalRawValues program base
  change
    CanonicalBlockAssignment.assignment
        (encodedHashCells
          (outputDigest
            (PerApplicationAssignmentTransport.canonical program) raw))
        (transportSchedule program
          (PerApplicationAssignmentTransport.canonical program) raw) column =
      CanonicalBlockAssignment.assignment
        (encodedHashCells raw.outputDigest)
        (PerApplicationAssignmentPlan.expand raw) column
  rw [canonical_outputDigest program raw]
  unfold CanonicalBlockAssignment.assignment
  by_cases publicRegion :
      column.val < ProductionAssignment.publicWidth
  · rw [dif_pos publicRegion, dif_pos publicRegion]
  · rw [dif_neg publicRegion, dif_neg publicRegion]
    exact canonicalBlocks_coordinateAt program raw _

/-- Execute only the exact schema-6 plan. Structural equality checks every
serialized field; a mutation fails closed. -/
def execute (program : Program) (plan : Plan) (base : BaseValues program) :
    Option (Assignment F (PerApplicationFixedPoint.logicalWidth program)) :=
  if plan = PerApplicationAssignmentTransport.canonical program then
    some (executeUnchecked program plan base)
  else
    none

/-- The real schema-6 interpreter returns the exact canonical logical
assignment. -/
theorem canonical_execute_eq_assignment (program : Program)
    (base : BaseValues program) :
    execute program (PerApplicationAssignmentTransport.canonical program) base =
      some (canonicalRawValues program base).assignment := by
  unfold execute
  rw [if_pos rfl, canonical_execute_eq_plan_execute,
    PerApplicationAssignmentPlan.execute_eq_assignment]

/-- The accepted schema-6 output agrees with the canonical assignment at
every logical column. -/
theorem canonical_execute_eq_assignment_pointwise (program : Program)
    (base : BaseValues program)
    (column : Fin (PerApplicationFixedPoint.logicalWidth program)) :
    (execute program (PerApplicationAssignmentTransport.canonical program) base).map
        (fun assignment => assignment column) =
      some ((canonicalRawValues program base).assignment column) := by
  simpa using congrArg (Option.map (fun assignment => assignment column))
    (canonical_execute_eq_assignment program base)

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution
