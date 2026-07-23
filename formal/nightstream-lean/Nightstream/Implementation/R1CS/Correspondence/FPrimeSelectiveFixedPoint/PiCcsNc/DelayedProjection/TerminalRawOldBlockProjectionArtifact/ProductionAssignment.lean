import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionPlacement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DelayedBlockLane

/-!
Transparent assignment for the generated production raw-old-block rows.

The source assignment is constructed from verifier-owned pending state, the
exact ordered raw `WitnessMat` family, and canonical internal witness values.
The physical assignment follows the generated Rust emitter inverse.  Pulling
it through the generated column map is the assignment consumed by the exact
compiler rows.  No column equality is a caller premise.

Owns: transparent construction of canonical and physical assignments from the
outgoing pending state, the ordered fourteen packed `WitnessMat` values, and
compiler-derived internal witness columns, including the generated
final-round scale interval.

Does not own: satisfaction of the projection rows, terminal CE, commitment
binding, transcript generation, trace composition, costs, or row-removal
authority.

Emits constraints: no; owns production assignment decoding only.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.execution.assignment.pending` | old-block and parent columns read the exact outgoing pending value | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.execution.assignment.children` | child-major canonical columns read the exact ordered packed witness matrices | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.execution.assignment.final_point` | the omitted nineteenth point coordinate reads pending old-block coordinate 18 | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.execution.assignment.final_scale` | all 270 generated final-scale definition columns pull back to the compiler internal witness | derived |
| `f_prime.pi_ccs_nc.delayed.execution.assignment.physical` | physical assignment lookup pulls back to the canonical source through the generated inverse | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private abbrev activeSemanticShape := ProductionDomain.semanticShape
private abbrev SemanticK := Nightstream.SuperNeo.Concrete.K

private theorem productionChildCountEq :
    14 = productionGlobalParams.k := by
  rfl

private theorem productionCarrierWidthEq :
    211797 * 54 = activeSemanticShape.carrierWidth := by
  rw [ProductionDomain.semanticShape_carrierWidth]

/-- Select one canonical base-field limb from a concrete extension value. -/
def concreteLimb (value : SemanticK) (limb : Nat) : Nat :=
  if limb = 0 then value.c0.val else value.c1.val

/-- Decode one old-block scalar from its canonical two-limb offset.  The
fixed source range guarantees that the total `getD` fallback is unreachable
in every compiler read theorem below. -/
def pendingOldScalar
    (pending : ProductionDelayedBlockLane) (offset : Nat) : Nat :=
  concreteLimb
    (pending.oldBlock.coordinates.getD (offset / 2)
      Nightstream.SuperNeo.Concrete.K.zero)
    (offset % 2)

/-- Decode one active parent scalar from its canonical two-limb offset. -/
def pendingParentScalar
    (pending : ProductionDelayedBlockLane) (offset : Nat) : Nat :=
  concreteLimb
    (pending.parentYZcol
      ⟨offset / 2 % 54, Nat.mod_lt _ (by decide)⟩)
    (offset % 2)

private def normalizedRawOffset (offset : Nat) :
    Fin (14 * (54 * 211797)) :=
  ⟨offset % (14 * (54 * 211797)),
    Nat.mod_lt _ (by decide)⟩

private def rawChildAndCell (offset : Nat) :
    Fin 14 × Fin (54 * 211797) :=
  finToPair (by decide : 0 < 54 * 211797) (normalizedRawOffset offset)

private def rawLaneAndBlock (offset : Nat) : Fin 54 × Fin 211797 :=
  finToPair (by decide : 0 < 211797) (rawChildAndCell offset).2

/-- Child decoded from the Rust child-major raw witness interval. -/
def rawChild (offset : Nat) : Fin productionGlobalParams.k :=
  Fin.cast productionChildCountEq (rawChildAndCell offset).1

/-- Semantic block-major assignment coordinate decoded from Rust's
lane-major/block-minor `FinalWitnessWires` interval. -/
def rawCoordinate (offset : Nat) : Fin activeSemanticShape.carrierWidth :=
  Fin.cast productionCarrierWidthEq
    (pairToFin (rawLaneAndBlock offset).2 (rawLaneAndBlock offset).1)

/-- Authoritative raw scalar.  This reads only the ordered full
`WitnessMat` family; no `CeClaim.y_zcol` sidecar or digest occurs. -/
def rawWitnessScalar
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (offset : Nat) : Nat :=
  (PackedWitness.unpack (finalWitnesses (rawChild offset))
    (rawCoordinate offset)).val

/-- Canonical compiler-column assignment before runtime placement. -/
def sourceAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) : Nat :=
  if column = constantOneColumn then 1
  else if column < parentFirstColumn then
    pendingOldScalar pending (column - oldBlockFirstColumn)
  else if column < witnessFamilyFirstColumn then
    pendingParentScalar pending (column - parentFirstColumn)
  else if column < tensorFirstColumn then
    rawWitnessScalar finalWitnesses (column - witnessFamilyFirstColumn)
  else
    (internalWitness column).val

/-- Actual production assignment indexed by physical Rust columns. -/
def physicalAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) : Nat :=
  match emitterColumnInverse productionEmitterLayout column with
  | some sourceColumn =>
      sourceAssignment pending finalWitnesses internalWitness sourceColumn
  | none => 0

/-- Compiler view of the physical production assignment. -/
def canonicalAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) : Nat -> Nat :=
  pullAssignment
    (physicalAssignment pending finalWitnesses internalWitness)
    (emitterColumnMap productionEmitterLayout)

/-! ## Canonicality and fixed source reads -/

private theorem concreteLimb_lt
    (value : SemanticK) (limb : Nat) :
    concreteLimb value limb < goldilocksP := by
  unfold concreteLimb
  split
  · simpa [goldilocksP, goldilocksModulus] using value.c0.isLt
  · simpa [goldilocksP, goldilocksModulus] using value.c1.isLt

private theorem pendingOldScalar_lt
    (pending : ProductionDelayedBlockLane) (offset : Nat) :
    pendingOldScalar pending offset < goldilocksP :=
  concreteLimb_lt _ _

private theorem pendingParentScalar_lt
    (pending : ProductionDelayedBlockLane) (offset : Nat) :
    pendingParentScalar pending offset < goldilocksP :=
  concreteLimb_lt _ _

private theorem rawWitnessScalar_lt
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (offset : Nat) :
    rawWitnessScalar finalWitnesses offset < goldilocksP := by
  unfold rawWitnessScalar goldilocksP goldilocksModulus
  exact (PackedWitness.unpack (finalWitnesses (rawChild offset))
    (rawCoordinate offset)).isLt

theorem sourceAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) :
    sourceAssignment pending finalWitnesses internalWitness column <
      goldilocksP := by
  unfold sourceAssignment
  split
  · decide
  split
  · exact pendingOldScalar_lt _ _
  split
  · exact pendingParentScalar_lt _ _
  split
  · exact rawWitnessScalar_lt _ _
  · exact (internalWitness column).isLt

theorem physicalAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) :
    physicalAssignment pending finalWitnesses internalWitness column <
      goldilocksP := by
  unfold physicalAssignment
  split
  · exact sourceAssignment_lt _ _ _ _
  · decide

theorem canonicalAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) :
    canonicalAssignment pending finalWitnesses internalWitness column <
      goldilocksP :=
  physicalAssignment_lt _ _ _ _

/-- Every compiler column used by the generated program reads the exact
source value selected before runtime placement. -/
theorem canonicalAssignment_eq_source
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) (columnInRange : column < canonicalColumnCount) :
    canonicalAssignment pending finalWitnesses internalWitness column =
      sourceAssignment pending finalWitnesses internalWitness column := by
  unfold canonicalAssignment pullAssignment physicalAssignment
  rw [productionEmitterValid.columnRoundTrip column columnInRange]

/-- Every compiler-owned derived column, including the final-scale interval,
reads the supplied internal witness value before physical placement. -/
theorem sourceAssignment_internal
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) (derived : tensorFirstColumn <= column) :
    sourceAssignment pending finalWitnesses internalWitness column =
      (internalWitness column).val := by
  have notConstant : column ≠ constantOneColumn := by
    simp only [tensorFirstColumn, constantOneColumn] at derived ⊢
    omega
  have notOldBlock : ¬ column < parentFirstColumn := by
    simp only [tensorFirstColumn, parentFirstColumn] at derived ⊢
    omega
  have notParent : ¬ column < witnessFamilyFirstColumn := by
    simp only [tensorFirstColumn, witnessFamilyFirstColumn] at derived ⊢
    omega
  have notRaw : ¬ column < tensorFirstColumn := Nat.not_lt.mpr derived
  simp [sourceAssignment, notConstant, notOldBlock, notParent, notRaw]

/-- Pulling a compiler-owned derived column through the generated physical
map preserves its exact internal witness value. -/
theorem canonicalAssignment_internal
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (column : Nat) (derived : tensorFirstColumn <= column)
    (inRange : column < canonicalColumnCount) :
    canonicalAssignment pending finalWitnesses internalWitness column =
      (internalWitness column).val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    column inRange]
  exact sourceAssignment_internal pending finalWitnesses internalWitness
    column derived

/-- The 270 final-scale columns are exactly 54 consecutive five-column
Karatsuba traces.  This theorem covers the full interval without constructing
a 270-element certificate. -/
theorem canonicalAssignment_finalScaleColumn
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin 54) (definition : Fin 5) :
    canonicalAssignment pending finalWitnesses internalWitness
        (finalScaleFirstColumn + 5 * lane.val + definition.val) =
      (internalWitness
        (finalScaleFirstColumn + 5 * lane.val + definition.val)).val := by
  apply canonicalAssignment_internal
  · simp only [tensorFirstColumn, finalScaleFirstColumn]
    omega
  · have laneLt := lane.isLt
    have definitionLt := definition.isLt
    simp only [canonicalColumnCount, finalScaleFirstColumn]
    omega

private theorem finalScale_definition_output
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Nightstream.Implementation.R1CS.Program.Definition)
    (member : definition ∈
      (productionFactoredLayout.scale lane).definitions) :
    ∃ offset : Fin 5,
      definition.output =
        finalScaleFirstColumn + 5 * lane.val + offset.val := by
  change definition ∈ (finalScaleTrace lane.val).definitions at member
  simp only [KMulTrace.definitions, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl
  · exact ⟨0, rfl⟩
  · exact ⟨1, rfl⟩
  · exact ⟨2, rfl⟩
  · exact ⟨3, rfl⟩
  · exact ⟨4, rfl⟩

/-- Every generated final-scale SSA definition reads its output from the
transparent internal witness through the physical emitter inverse. -/
theorem canonicalAssignment_finalScaleDefinition
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Nightstream.Implementation.R1CS.Program.Definition)
    (member : definition ∈
      (productionFactoredLayout.scale lane).definitions) :
    canonicalAssignment pending finalWitnesses internalWitness
        definition.output =
      (internalWitness definition.output).val := by
  rcases finalScale_definition_output lane definition member with
    ⟨offset, output⟩
  rw [output]
  exact canonicalAssignment_finalScaleColumn pending finalWitnesses
    internalWitness lane offset

@[simp] theorem canonicalAssignment_constantOne
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    canonicalAssignment pending finalWitnesses internalWitness 0 = 1 := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness 0
    (by decide)]
  rfl

theorem sourceAssignment_oldBlock_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (round : Fin productionLayout.blockVariables) :
    sourceAssignment pending finalWitnesses internalWitness
        (productionLayout.oldBlock round).c0 =
      (pending.oldBlock.coordinates.getD round.val
        Nightstream.SuperNeo.Concrete.K.zero).c0.val := by
  have roundLt18 := round.isLt
  change round.val < 18 at roundLt18
  have roundLt : round.val < 19 := by omega
  have notConstant : 1 + 2 * round.val ≠ 0 := by omega
  have inOldBlock : 1 + 2 * round.val < 39 := by omega
  have offsetEq : 1 + 2 * round.val - 1 = 2 * round.val := by omega
  have quotientEq : (2 * round.val) / 2 = round.val := by omega
  have remainderEq : (2 * round.val) % 2 = 0 := by omega
  simp [sourceAssignment, productionLayout, oldBlockColumns,
    oldBlockColumnsNat, kColumnsAt, pendingOldScalar, concreteLimb,
    constantOneColumn, oldBlockFirstColumn, parentFirstColumn,
    notConstant, inOldBlock, offsetEq, quotientEq, remainderEq]

theorem sourceAssignment_oldBlock_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (round : Fin productionLayout.blockVariables) :
    sourceAssignment pending finalWitnesses internalWitness
        (productionLayout.oldBlock round).c1 =
      (pending.oldBlock.coordinates.getD round.val
        Nightstream.SuperNeo.Concrete.K.zero).c1.val := by
  have roundLt18 := round.isLt
  change round.val < 18 at roundLt18
  have roundLt : round.val < 19 := by omega
  have notConstant : 1 + 2 * round.val + 1 ≠ 0 := by omega
  have inOldBlock : 1 + 2 * round.val + 1 < 39 := by omega
  have offsetEq : 1 + 2 * round.val + 1 - 1 = 2 * round.val + 1 := by omega
  have quotientEq : (2 * round.val + 1) / 2 = round.val := by omega
  have remainderEq : (2 * round.val + 1) % 2 = 1 := by omega
  simp [sourceAssignment, productionLayout, oldBlockColumns,
    oldBlockColumnsNat, kColumnsAt, pendingOldScalar, concreteLimb,
    constantOneColumn, oldBlockFirstColumn, parentFirstColumn,
    notConstant, inOldBlock, offsetEq, quotientEq, remainderEq]

/-- The factorized compiler omits old-block coordinate 18 from the prefix
tensor, but its final-scale right operand still reads that exact
verifier-owned pending coordinate. -/
theorem sourceAssignment_factoredPoint_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    sourceAssignment pending finalWitnesses internalWitness
        (oldBlockColumnsNat 18).c0 =
      (pending.oldBlock.coordinates.getD 18
        Nightstream.SuperNeo.Concrete.K.zero).c0.val := by
  simp [sourceAssignment, oldBlockColumnsNat, kColumnsAt,
    pendingOldScalar, concreteLimb, constantOneColumn,
    oldBlockFirstColumn, parentFirstColumn]

theorem sourceAssignment_factoredPoint_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    sourceAssignment pending finalWitnesses internalWitness
        (oldBlockColumnsNat 18).c1 =
      (pending.oldBlock.coordinates.getD 18
        Nightstream.SuperNeo.Concrete.K.zero).c1.val := by
  simp [sourceAssignment, oldBlockColumnsNat, kColumnsAt,
    pendingOldScalar, concreteLimb, constantOneColumn,
    oldBlockFirstColumn, parentFirstColumn]

theorem canonicalAssignment_factoredPoint_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    canonicalAssignment pending finalWitnesses internalWitness
        productionFactoredLayout.factor.finalPoint.c0 =
      (pending.oldBlock.coordinates.getD 18
        Nightstream.SuperNeo.Concrete.K.zero).c0.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness]
  · simpa [productionFinalPointColumn] using
      sourceAssignment_factoredPoint_c0 pending finalWitnesses internalWitness
  · change 37 < canonicalColumnCount
    decide

theorem canonicalAssignment_factoredPoint_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    canonicalAssignment pending finalWitnesses internalWitness
        productionFactoredLayout.factor.finalPoint.c1 =
      (pending.oldBlock.coordinates.getD 18
        Nightstream.SuperNeo.Concrete.K.zero).c1.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness]
  · simpa [productionFinalPointColumn] using
      sourceAssignment_factoredPoint_c1 pending finalWitnesses internalWitness
  · change 38 < canonicalColumnCount
    decide

theorem sourceAssignment_parent_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionLayout.activeLanes) :
    sourceAssignment pending finalWitnesses internalWitness
        (productionLayout.parent lane).c0 =
      (pending.parentYZcol lane).c0.val := by
  have laneLt : lane.val < 54 := lane.isLt
  have notConstant : 39 + 2 * lane.val ≠ 0 := by omega
  have notOldBlock : ¬ 39 + 2 * lane.val < 39 := by omega
  have inParent : 39 + 2 * lane.val < 147 := by omega
  have offsetEq : 39 + 2 * lane.val - 39 = 2 * lane.val := by omega
  have quotientEq : (2 * lane.val) / 2 = lane.val := by omega
  have remainderEq : (2 * lane.val) % 2 = 0 := by omega
  have laneMod : lane.val % 54 = lane.val := Nat.mod_eq_of_lt laneLt
  simp [sourceAssignment, productionLayout, parentColumns,
    parentColumnsNat, kColumnsAt, pendingParentScalar, concreteLimb,
    constantOneColumn, parentFirstColumn, witnessFamilyFirstColumn,
    notConstant, notOldBlock, inParent, offsetEq, quotientEq,
    remainderEq, laneMod]

theorem sourceAssignment_parent_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionLayout.activeLanes) :
    sourceAssignment pending finalWitnesses internalWitness
        (productionLayout.parent lane).c1 =
      (pending.parentYZcol lane).c1.val := by
  have laneLt : lane.val < 54 := lane.isLt
  have notConstant : 39 + 2 * lane.val + 1 ≠ 0 := by omega
  have notOldBlock : ¬ 39 + 2 * lane.val + 1 < 39 := by omega
  have inParent : 39 + 2 * lane.val + 1 < 147 := by omega
  have offsetEq : 39 + 2 * lane.val + 1 - 39 =
      2 * lane.val + 1 := by omega
  have quotientEq : (2 * lane.val + 1) / 2 = lane.val := by omega
  have remainderEq : (2 * lane.val + 1) % 2 = 1 := by omega
  have laneMod : lane.val % 54 = lane.val := Nat.mod_eq_of_lt laneLt
  simp [sourceAssignment, productionLayout, parentColumns,
    parentColumnsNat, kColumnsAt, pendingParentScalar, concreteLimb,
    constantOneColumn, parentFirstColumn, witnessFamilyFirstColumn,
    notConstant, notOldBlock, inParent, offsetEq, quotientEq,
    remainderEq, laneMod]

/-! ## Raw `WitnessMat` storage inverse -/

private theorem productionCoordinateRectangle :
    productionLayout.logicalWidth = 211797 * 54 := by
  rfl

private def compilerChild
    (child : Fin productionLayout.childCount) : Fin 14 :=
  Fin.cast productionChildCount child

private def compilerCoordinate
    (coordinate : Fin productionLayout.logicalWidth) :
    Fin (211797 * 54) :=
  Fin.cast productionCoordinateRectangle coordinate

private def compilerBlockLane
    (coordinate : Fin productionLayout.logicalWidth) :
    Fin 211797 × Fin 54 :=
  finToPair (by decide : 0 < 54) (compilerCoordinate coordinate)

private def compilerRawCell
    (coordinate : Fin productionLayout.logicalWidth) :
    Fin (54 * 211797) :=
  pairToFin (compilerBlockLane coordinate).2
    (compilerBlockLane coordinate).1

private def compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    Fin (14 * (54 * 211797)) :=
  pairToFin (compilerChild child) (compilerRawCell coordinate)

private theorem compilerRawColumn_eq_storage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawWitnessColumn productionLayout child coordinate =
      witnessFamilyFirstColumn +
        (compilerRawStorage child coordinate).val := by
  change
    147 + child.val * 11437038 +
          (coordinate.val % 54) * 211797 + coordinate.val / 54 =
      147 +
        (child.val * (54 * 211797) +
          ((coordinate.val % 54) * 211797 + coordinate.val / 54))
  omega

private theorem compilerRawOffset_eq_storage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawWitnessColumn productionLayout child coordinate -
        witnessFamilyFirstColumn =
      (compilerRawStorage child coordinate).val := by
  rw [compilerRawColumn_eq_storage]
  simp [witnessFamilyFirstColumn]

private theorem normalizedRawOffset_compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    normalizedRawOffset (compilerRawStorage child coordinate).val =
      compilerRawStorage child coordinate := by
  apply Fin.ext
  simp [normalizedRawOffset,
    Nat.mod_eq_of_lt (compilerRawStorage child coordinate).isLt]

private theorem rawChildAndCell_compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawChildAndCell (compilerRawStorage child coordinate).val =
      (compilerChild child, compilerRawCell coordinate) := by
  unfold rawChildAndCell
  rw [normalizedRawOffset_compilerRawStorage]
  exact finToPair_pairToFin (by decide : 0 < 54 * 211797)
    (compilerChild child) (compilerRawCell coordinate)

private theorem rawLaneAndBlock_compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawLaneAndBlock (compilerRawStorage child coordinate).val =
      ((compilerBlockLane coordinate).2,
       (compilerBlockLane coordinate).1) := by
  unfold rawLaneAndBlock
  rw [rawChildAndCell_compilerRawStorage]
  exact finToPair_pairToFin (by decide : 0 < 211797)
    (compilerBlockLane coordinate).2
    (compilerBlockLane coordinate).1

private theorem rawChild_compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawChild (compilerRawStorage child coordinate).val = child := by
  unfold rawChild
  rw [rawChildAndCell_compilerRawStorage]
  apply Fin.ext
  rfl

private theorem rawCoordinate_compilerRawStorage
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawCoordinate (compilerRawStorage child coordinate).val = coordinate := by
  unfold rawCoordinate
  rw [rawLaneAndBlock_compilerRawStorage]
  have reencoded := pairToFin_finToPair
    (by decide : 0 < 54) (compilerCoordinate coordinate)
  apply Fin.ext
  exact congrArg Fin.val reencoded

/-- The canonical raw compiler column reads the exact ordered full
`WitnessMat` assignment coordinate. -/
theorem sourceAssignment_rawWitness
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    sourceAssignment pending finalWitnesses internalWitness
        (rawWitnessColumn productionLayout child coordinate) =
      (PackedWitness.unpack (finalWitnesses child) coordinate).val := by
  have columnEq : rawWitnessColumn productionLayout child coordinate =
      witnessFamilyFirstColumn +
        (compilerRawStorage child coordinate).val :=
    compilerRawColumn_eq_storage child coordinate
  have notConstant :
      rawWitnessColumn productionLayout child coordinate ≠
        constantOneColumn := by
    rw [columnEq]
    simp [witnessFamilyFirstColumn, constantOneColumn]
  have notOldBlock :
      ¬ rawWitnessColumn productionLayout child coordinate <
        parentFirstColumn := by
    rw [columnEq]
    simp only [witnessFamilyFirstColumn, parentFirstColumn]
    omega
  have notParent :
      ¬ rawWitnessColumn productionLayout child coordinate <
        witnessFamilyFirstColumn := by
    rw [columnEq]
    omega
  have inRaw : rawWitnessColumn productionLayout child coordinate <
      tensorFirstColumn := by
    rw [columnEq]
    have storageLt := (compilerRawStorage child coordinate).isLt
    simp only [witnessFamilyFirstColumn, tensorFirstColumn] at storageLt ⊢
    omega
  have offsetEq : rawWitnessColumn productionLayout child coordinate -
      witnessFamilyFirstColumn =
        (compilerRawStorage child coordinate).val :=
    compilerRawOffset_eq_storage child coordinate
  rw [sourceAssignment]
  simp only [notConstant, if_false, notOldBlock, notParent, inRaw,
    if_true, offsetEq, rawWitnessScalar,
    rawChild_compilerRawStorage, rawCoordinate_compilerRawStorage]

theorem canonicalAssignment_oldBlock_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (round : Fin productionLayout.blockVariables) :
    canonicalAssignment pending finalWitnesses internalWitness
        (productionLayout.oldBlock round).c0 =
      (pending.oldBlock.coordinates.getD round.val
        Nightstream.SuperNeo.Concrete.K.zero).c0.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    (productionLayout.oldBlock round).c0]
  · exact sourceAssignment_oldBlock_c0 pending finalWitnesses
      internalWitness round
  · change 1 + 2 * round.val < canonicalColumnCount
    have roundLt18 := round.isLt
    change round.val < 18 at roundLt18
    have roundLt : round.val < 19 := by omega
    simp only [canonicalColumnCount]
    omega

theorem canonicalAssignment_oldBlock_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (round : Fin productionLayout.blockVariables) :
    canonicalAssignment pending finalWitnesses internalWitness
        (productionLayout.oldBlock round).c1 =
      (pending.oldBlock.coordinates.getD round.val
        Nightstream.SuperNeo.Concrete.K.zero).c1.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    (productionLayout.oldBlock round).c1]
  · exact sourceAssignment_oldBlock_c1 pending finalWitnesses
      internalWitness round
  · change 1 + 2 * round.val + 1 < canonicalColumnCount
    have roundLt18 := round.isLt
    change round.val < 18 at roundLt18
    have roundLt : round.val < 19 := by omega
    simp only [canonicalColumnCount]
    omega

theorem canonicalAssignment_parent_c0
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionLayout.activeLanes) :
    canonicalAssignment pending finalWitnesses internalWitness
        (productionLayout.parent lane).c0 =
      (pending.parentYZcol lane).c0.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    (productionLayout.parent lane).c0]
  · exact sourceAssignment_parent_c0 pending finalWitnesses internalWitness lane
  · change 39 + 2 * lane.val < canonicalColumnCount
    have laneLt : lane.val < 54 := lane.isLt
    simp only [canonicalColumnCount]
    omega

theorem canonicalAssignment_parent_c1
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionLayout.activeLanes) :
    canonicalAssignment pending finalWitnesses internalWitness
        (productionLayout.parent lane).c1 =
      (pending.parentYZcol lane).c1.val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    (productionLayout.parent lane).c1]
  · exact sourceAssignment_parent_c1 pending finalWitnesses internalWitness lane
  · change 39 + 2 * lane.val + 1 < canonicalColumnCount
    have laneLt : lane.val < 54 := lane.isLt
    simp only [canonicalColumnCount]
    omega

theorem productionRawWitnessColumn_lt_canonical
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    rawWitnessColumn productionLayout child coordinate <
      canonicalColumnCount := by
  rw [compilerRawColumn_eq_storage]
  have storageLt := (compilerRawStorage child coordinate).isLt
  simp only [witnessFamilyFirstColumn, canonicalColumnCount] at storageLt ⊢
  omega

theorem canonicalAssignment_rawWitness
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (child : Fin productionLayout.childCount)
    (coordinate : Fin productionLayout.logicalWidth) :
    canonicalAssignment pending finalWitnesses internalWitness
        (rawWitnessColumn productionLayout child coordinate) =
      (PackedWitness.unpack (finalWitnesses child) coordinate).val := by
  rw [canonicalAssignment_eq_source pending finalWitnesses internalWitness
    (rawWitnessColumn productionLayout child coordinate)
    (productionRawWitnessColumn_lt_canonical child coordinate)]
  exact sourceAssignment_rawWitness pending finalWitnesses internalWitness
    child coordinate

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
