import Nightstream.Implementation.R1CS.Core.SequentialProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionAssignment

/-!
List-free SSA program for the optimized production raw-old-block projection.

The Rust emitter allocates every derived column consecutively after the raw
`WitnessMat` interval.  This leaf decodes that interval arithmetically into
the exact generated definition families:

* 262,143 five-column compact-tensor multiplications;
* 11,437,038 two-column raw-coordinate products; and
* 54 five-column final-scale multiplications.

No production-sized list or executable certificate is constructed.  The
reference proof below is structural: tensor terms refer only to earlier
rounds, coordinate products refer only to source/tensor columns, and final
scale traces refer only to source/product columns or an earlier column in the
same five-column trace.

Owns: deterministic materialization order for all 24,185,061 derived
production columns.

Does not own: source assignment values, row satisfaction, terminal parent
rows, physical placement, semantic projection authority, or security events.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.raw_old_block.derived.tensor` | decode the 262,143 five-definition tensor traces in allocation order | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.derived.coordinate` | decode the 11,437,038 two-definition raw-coordinate products in allocation order | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.derived.scale` | decode the 54 five-definition final-scale traces in allocation order | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.derived.program` | prove every generated definition reads only source or earlier derived columns | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private def generatedBlockCount : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount

private def generatedActiveLanes : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.activeLanes

private abbrev generatedTensorTrace :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace
private abbrev generatedRawTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
private abbrev generatedChiTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.chiTerms
private abbrev generatedPointTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.pointTerms
private abbrev generatedOneMinusPointTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms
private abbrev generatedLaneTerms :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms
private abbrev generatedFinalScaleTrace :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace

@[simp] private theorem generatedBlockCount_exact :
    generatedBlockCount = 211797 := by
  rfl

@[simp] private theorem generatedActiveLanes_exact :
    generatedActiveLanes = 54 := by
  rfl

/-- Number of consecutive derived columns in each generated family. -/
def productionTensorDefinitionCount : Nat := productFirstColumn - tensorFirstColumn
def productionCoordinateDefinitionCount : Nat :=
  finalScaleFirstColumn - productFirstColumn
def productionScaleDefinitionCount : Nat :=
  canonicalColumnCount - finalScaleFirstColumn
def productionDerivedDefinitionCount : Nat :=
  canonicalColumnCount - tensorFirstColumn

@[simp] theorem productionTensorDefinitionCount_exact :
    productionTensorDefinitionCount = 262143 * 5 := by
  rfl

@[simp] theorem productionCoordinateDefinitionCount_exact :
    productionCoordinateDefinitionCount = 11437038 * 2 := by
  rfl

@[simp] theorem productionScaleDefinitionCount_exact :
    productionScaleDefinitionCount = 54 * 5 := by
  rfl

@[simp] theorem productionDerivedDefinitionCount_exact :
    productionDerivedDefinitionCount = 24185061 := by
  rfl

theorem productionDerivedDefinitionCount_partition :
    productionDerivedDefinitionCount =
      productionTensorDefinitionCount +
        productionCoordinateDefinitionCount +
          productionScaleDefinitionCount := by
  rfl

private theorem productionProductFirstColumn_eq :
    productFirstColumn =
      tensorFirstColumn + productionTensorDefinitionCount := by
  rfl

private theorem productionFinalScaleFirstColumn_eq :
    finalScaleFirstColumn =
      tensorFirstColumn + productionTensorDefinitionCount +
        productionCoordinateDefinitionCount := by
  rfl

private theorem productionTensorColumns_before_products :
    tensorFirstColumn < productFirstColumn := by
  decide

private theorem productionTensorColumns_before_scale :
    tensorFirstColumn <= finalScaleFirstColumn := by
  decide

private def traceDefinition
    (trace : KMulTrace) (definition : Fin 5) : Definition :=
  trace.definitions.get
    ⟨definition.val, by simpa [KMulTrace.definitions] using definition.isLt⟩

private def tensorDefinitionNumber (index : Nat) : Fin 5 :=
  ⟨index % 5, Nat.mod_lt _ (by decide)⟩

/-- Tensor definition selected by the generated round-major owner decoder. -/
def productionTensorDefinitionAt (index : Nat) : Definition :=
  let ordinal := index / 5
  let owner := tensorOwner ordinal
  traceDefinition (generatedTensorTrace owner.1 owner.2)
    (tensorDefinitionNumber index)

/-- Coordinate definition selected by the generated
lane-major/block-minor/limb order. -/
def productionCoordinateDefinitionAt (index : Nat) : Definition :=
  let ordinal := index / 2
  let lane := ordinal / generatedBlockCount
  let block := ordinal % generatedBlockCount
  let limb := index % 2
  let chi := generatedChiTerms block
  let selected := if limb = 0 then chi.c0 else chi.c1
  ⟨productColumn lane block limb,
    .product (generatedRawTerms lane block) selected⟩

/-- Final-scale definition selected by the generated lane-major trace order. -/
def productionScaleDefinitionAt (index : Nat) : Definition :=
  let lane := index / 5
  traceDefinition (generatedFinalScaleTrace lane) (tensorDefinitionNumber index)

/-- Exact generated definition at one derived-column offset.  The branches
are the three contiguous Rust allocation intervals, not stage labels or row
counts inferred from a digest. -/
def productionDefinitionAt (index : Nat) : Definition :=
  if index < productionTensorDefinitionCount then
    productionTensorDefinitionAt index
  else if index <
      productionTensorDefinitionCount + productionCoordinateDefinitionCount then
    productionCoordinateDefinitionAt
      (index - productionTensorDefinitionCount)
  else
    productionScaleDefinitionAt
      (index - productionTensorDefinitionCount -
        productionCoordinateDefinitionCount)

/-! ## Exact generated selection -/

theorem productionDefinitionAt_tensor
    (index : Nat) (inFamily : index < productionTensorDefinitionCount) :
    productionDefinitionAt index = productionTensorDefinitionAt index := by
  unfold productionDefinitionAt
  rw [if_pos inFamily]

theorem productionDefinitionAt_coordinate
    (index : Nat)
    (afterTensor : productionTensorDefinitionCount <= index)
    (beforeScale :
      index < productionTensorDefinitionCount +
        productionCoordinateDefinitionCount) :
    productionDefinitionAt index =
      productionCoordinateDefinitionAt
        (index - productionTensorDefinitionCount) := by
  unfold productionDefinitionAt
  rw [if_neg (Nat.not_lt.mpr afterTensor), if_pos beforeScale]

theorem productionDefinitionAt_scale
    (index : Nat)
    (afterCoordinate :
      productionTensorDefinitionCount +
        productionCoordinateDefinitionCount <= index) :
    productionDefinitionAt index =
      productionScaleDefinitionAt
        (index - productionTensorDefinitionCount -
          productionCoordinateDefinitionCount) := by
  have afterTensor : productionTensorDefinitionCount <= index := by omega
  unfold productionDefinitionAt
  rw [if_neg (Nat.not_lt.mpr afterTensor),
    if_neg (Nat.not_lt.mpr afterCoordinate)]

theorem productionTensorDefinitionAt_generated
    (index : Nat) :
    productionTensorDefinitionAt index =
      let ordinal := index / 5
      let owner := tensorOwner ordinal
      traceDefinition (generatedTensorTrace owner.1 owner.2)
        ⟨index % 5, Nat.mod_lt _ (by decide)⟩ := by
  rfl

theorem productionCoordinateDefinitionAt_generated
    (index : Nat) :
    productionCoordinateDefinitionAt index =
      let ordinal := index / 2
      let lane := ordinal / generatedBlockCount
      let block := ordinal % generatedBlockCount
      let limb := index % 2
      let chi := generatedChiTerms block
      let selected := if limb = 0 then chi.c0 else chi.c1
      ⟨productColumn lane block limb,
        .product (generatedRawTerms lane block) selected⟩ := by
  rfl

theorem productionScaleDefinitionAt_generated
    (index : Nat) :
    productionScaleDefinitionAt index =
      traceDefinition (generatedFinalScaleTrace (index / 5))
        ⟨index % 5, Nat.mod_lt _ (by decide)⟩ := by
  rfl

/-! ## Five-column trace order -/

private theorem fin5_cases {predicate : Fin 5 -> Prop}
    (case0 : predicate 0) (case1 : predicate 1)
    (case2 : predicate 2) (case3 : predicate 3)
    (case4 : predicate 4) : forall index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro impossible
  exact Fin.elim0 impossible

private def TermsBelow (terms : List (Nat × Nat)) (bound : Nat) : Prop :=
  forall term, term ∈ terms -> term.1 < bound

private def KTermsBelow (terms : KTerms) (bound : Nat) : Prop :=
  TermsBelow terms.c0 bound ∧ TermsBelow terms.c1 bound

private def KColumnsBelow (columns : KColumns) (bound : Nat) : Prop :=
  columns.c0 < bound ∧ columns.c1 < bound

private theorem TermsBelow.mono
    {terms : List (Nat × Nat)} {small large : Nat}
    (below : TermsBelow terms small) (le : small <= large) :
    TermsBelow terms large := by
  intro term member
  exact Nat.lt_of_lt_of_le (below term member) le

private theorem TermsBelow.append
    {left right : List (Nat × Nat)} {bound : Nat}
    (leftBelow : TermsBelow left bound)
    (rightBelow : TermsBelow right bound) :
    TermsBelow (left ++ right) bound := by
  intro term member
  rcases List.mem_append.mp member with member | member
  · exact leftBelow term member
  · exact rightBelow term member

private theorem KTermsBelow.mono
    {terms : KTerms} {small large : Nat}
    (below : KTermsBelow terms small) (le : small <= large) :
    KTermsBelow terms large :=
  ⟨below.1.mono le, below.2.mono le⟩

private theorem KTermsBelow.ofColumns
    {columns : KColumns} {bound : Nat}
    (below : KColumnsBelow columns bound) :
    KTermsBelow (KTerms.ofColumns columns) bound := by
  constructor <;> intro term member <;>
    simp [KTerms.ofColumns] at member <;> subst term
  · exact below.1
  · exact below.2

private theorem KTermsBelow.subtractOutput
    {terms : KTerms} {output : KColumns} {bound : Nat}
    (termsBelow : KTermsBelow terms bound)
    (outputBelow : KColumnsBelow output bound) :
    KTermsBelow (subtractOutput terms output) bound := by
  constructor <;> intro term member
  · change term ∈
        terms.c0 ++ [(output.c0, goldilocksP - 1)] at member
    rcases List.mem_append.mp member with member | member
    · exact termsBelow.1 term member
    · simp only [List.mem_singleton] at member
      subst term
      exact outputBelow.1
  · change term ∈
        terms.c1 ++ [(output.c1, goldilocksP - 1)] at member
    rcases List.mem_append.mp member with member | member
    · exact termsBelow.2 term member
    · simp only [List.mem_singleton] at member
      subst term
      exact outputBelow.2

private theorem tensorRoundMulStart_succ (round : Nat) :
    tensorRoundMulStart (round + 1) =
      tensorRoundMulStart round + tensorRoundMulCount round := by
  unfold tensorRoundMulStart
  rw [List.range_succ, List.foldl_append]
  simp

private theorem tensorRoundMulCount_positive (round : Nat) :
    0 < tensorRoundMulCount round := by
  unfold tensorRoundMulCount
  rw [Nat.lt_min]
  constructor
  · decide
  · exact Nat.pow_pos (by decide)

private theorem tensorRoundHighCount_le_mulCount (round : Nat) :
    tensorRoundHighCount round <= tensorRoundMulCount round := by
  unfold tensorRoundHighCount tensorRoundMulCount
  rw [Nat.le_min]
  constructor
  · exact Nat.le_trans (Nat.min_le_left _ _) (Nat.sub_le _ _)
  · exact Nat.min_le_right _ _

private theorem tensorPriorBound_lt_next (round : Nat) :
    tensorMulFirstColumn round 0 < tensorMulFirstColumn (round + 1) 0 := by
  rw [show tensorMulFirstColumn (round + 1) 0 =
      tensorFirstColumn +
        5 * (tensorRoundMulStart round + tensorRoundMulCount round) by
    simp [tensorMulFirstColumn, tensorMulOrdinal,
      tensorRoundMulStart_succ]]
  simp only [tensorMulFirstColumn, tensorMulOrdinal, Nat.add_zero]
  have positive := tensorRoundMulCount_positive round
  omega

private theorem tensorOutput_below_next
    (round parent : Nat) (parentInRange :
      parent < tensorRoundMulCount round) :
    KColumnsBelow (tensorOutputColumns round parent)
      (tensorMulFirstColumn (round + 1) 0) := by
  rw [show tensorMulFirstColumn (round + 1) 0 =
      tensorFirstColumn +
        5 * (tensorRoundMulStart round + tensorRoundMulCount round) by
    simp [tensorMulFirstColumn, tensorMulOrdinal,
      tensorRoundMulStart_succ]]
  simp only [tensorOutputColumns, kColumnsAt, tensorMulFirstColumn,
    tensorMulOrdinal, KColumnsBelow]
  omega

/-- Every symbolic tensor term at the start of a round refers strictly below
that round's first generated multiplication column. -/
private theorem tensorTermsAt_below_round :
    forall (round index : Nat),
      KTermsBelow (tensorTermsAt round index)
        (tensorMulFirstColumn round 0) := by
  intro round
  induction round with
  | zero =>
      intro index
      unfold tensorTermsAt
      split
      · simp [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorRoot,
          KTermsBelow, TermsBelow,
          tensorMulFirstColumn, tensorMulOrdinal, tensorRoundMulStart,
          constantOneColumn, tensorFirstColumn]
      · simp [emptyKTerms, KTermsBelow, TermsBelow]
  | succ round inductionHypothesis =>
      intro index
      simp only [tensorTermsAt]
      split
      · rename_i indexInRange
        split
        · exact KTermsBelow.subtractOutput
            ((inductionHypothesis index).mono
              (Nat.le_of_lt (tensorPriorBound_lt_next round)))
            (tensorOutput_below_next round index indexInRange)
        · exact KTermsBelow.ofColumns
            (tensorOutput_below_next round index indexInRange)
      · rename_i indexOutside
        split
        · rename_i parentHigh
          have parentInRange :
              index - tensorRoundMulCount round <
                tensorRoundMulCount round :=
            Nat.lt_of_lt_of_le parentHigh
              (tensorRoundHighCount_le_mulCount round)
          exact KTermsBelow.ofColumns
            (tensorOutput_below_next round
              (index - tensorRoundMulCount round) parentInRange)
        · simp [emptyKTerms, KTermsBelow, TermsBelow]

private theorem generatedPointTerms_below_tensor
    (round : Nat) (roundInRange : round < 19) :
    KTermsBelow (generatedPointTerms round) tensorFirstColumn := by
  have c0 : (oldBlockColumnsNat round).c0 < tensorFirstColumn := by
    simp [oldBlockColumnsNat, kColumnsAt, oldBlockFirstColumn,
      tensorFirstColumn]
    omega
  have c1 : (oldBlockColumnsNat round).c1 < tensorFirstColumn := by
    simp [oldBlockColumnsNat, kColumnsAt, oldBlockFirstColumn,
      tensorFirstColumn]
    omega
  exact KTermsBelow.ofColumns ⟨c0, c1⟩

private theorem generatedOneMinusTerms_below_tensor
    (round : Nat) (roundInRange : round < 19) :
    KTermsBelow (generatedOneMinusPointTerms round) tensorFirstColumn := by
  constructor <;> intro term member
  · simp only [generatedOneMinusPointTerms,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms,
      List.mem_cons, List.mem_singleton, List.not_mem_nil] at member
    rcases member with rfl | member
    · simp [constantOneColumn, tensorFirstColumn]
    · rcases member with rfl | impossible
      · simp [oldBlockColumnsNat, kColumnsAt, oldBlockFirstColumn,
        tensorFirstColumn]
        omega
      · exact False.elim impossible
  · simp only [generatedOneMinusPointTerms,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.oneMinusPointTerms,
      List.mem_cons, List.mem_singleton, List.not_mem_nil] at member
    rcases member with rfl | impossible
    · simp [oldBlockColumnsNat, kColumnsAt, oldBlockFirstColumn,
        tensorFirstColumn]
      omega
    · exact False.elim impossible

private structure TraceSequentialAt (trace : KMulTrace) (first : Nat) : Prop where
  leftBelow : KTermsBelow trace.left first
  rightBelow : KTermsBelow trace.right first
  sumLeftBelow : TermsBelow trace.sumLeft first
  sumRightBelow : TermsBelow trace.sumRight first
  productC0 : trace.productC0 = first
  productC1 : trace.productC1 = first + 1
  productSum : trace.productSum = first + 2
  outputC0 : trace.output.c0 = first + 3
  outputC1 : trace.output.c1 = first + 4

private theorem traceDefinition_output
    {trace : KMulTrace} {first : Nat}
    (sequential : TraceSequentialAt trace first) :
    forall definition : Fin 5,
      (traceDefinition trace definition).output = first + definition.val := by
  apply fin5_cases
  · change trace.productC0 = first
    exact sequential.productC0
  · change trace.productC1 = first + 1
    exact sequential.productC1
  · change trace.productSum = first + 2
    exact sequential.productSum
  · change trace.output.c0 = first + 3
    exact sequential.outputC0
  · change trace.output.c1 = first + 4
    exact sequential.outputC1

private theorem traceDefinition_references_before
    {trace : KMulTrace} {first : Nat}
    (sequential : TraceSequentialAt trace first)
    (definition : Fin 5) (column : Nat)
    (member : column ∈ (traceDefinition trace definition).rhs.refs) :
    column < first + definition.val := by
  revert definition
  apply fin5_cases <;> intro member
  · change column ∈
        trace.left.c0.map Prod.fst ++ trace.right.c0.map Prod.fst at member
    rcases List.mem_append.mp member with member | member
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact sequential.leftBelow.1 term termMember
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact sequential.rightBelow.1 term termMember
  · change column ∈
        trace.left.c1.map Prod.fst ++ trace.right.c1.map Prod.fst at member
    rcases List.mem_append.mp member with member | member
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact Nat.lt_trans (sequential.leftBelow.2 term termMember) (by omega)
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact Nat.lt_trans (sequential.rightBelow.2 term termMember) (by omega)
  · change column ∈
        trace.sumLeft.map Prod.fst ++ trace.sumRight.map Prod.fst at member
    rcases List.mem_append.mp member with member | member
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact Nat.lt_trans (sequential.sumLeftBelow term termMember) (by omega)
    · rcases List.mem_map.mp member with ⟨term, termMember, termEq⟩
      subst column
      exact Nat.lt_trans (sequential.sumRightBelow term termMember) (by omega)
  · change column ∈ [trace.productC0, trace.productC1] at member
    simp only [List.mem_cons, List.mem_singleton] at member
    rcases member with rfl | member
    · rw [sequential.productC0]
      omega
    · rcases member with rfl | impossible
      · rw [sequential.productC1]
        omega
      · simp at impossible
  · change column ∈
        [trace.productSum, trace.productC0, trace.productC1] at member
    simp only [List.mem_cons, List.mem_singleton] at member
    rcases member with rfl | member
    · rw [sequential.productSum]
      omega
    · rcases member with rfl | member
      · rw [sequential.productC0]
        omega
      · rcases member with rfl | impossible
        · rw [sequential.productC1]
          omega
        · simp at impossible

private theorem generatedTensorTrace_sequential
    (ordinal : Nat) (inRange : ordinal < 262143) :
    let owner := tensorOwner ordinal
    TraceSequentialAt (generatedTensorTrace owner.1 owner.2)
      (tensorFirstColumn + 5 * ordinal) := by
  let owner := tensorOwner ordinal
  have valid : owner.1 < 18 ∧
      owner.2 < tensorRoundMulCount owner.1 ∧
      tensorMulOrdinal owner.1 owner.2 = ordinal :=
    productionTensorOwner_valid ordinal inRange
  have ordinalEq : tensorRoundMulStart owner.1 + owner.2 = ordinal := by
    simpa only [tensorMulOrdinal] using valid.2.2
  have leftBelowAtRound :=
    tensorTermsAt_below_round owner.1 owner.2
  have roundStartLe :
      tensorMulFirstColumn owner.1 0 <=
        tensorFirstColumn + 5 * ordinal := by
    simp only [tensorMulFirstColumn, tensorMulOrdinal, Nat.add_zero]
    omega
  have leftBelow : KTermsBelow (tensorTermsAt owner.1 owner.2)
      (tensorFirstColumn + 5 * ordinal) :=
    leftBelowAtRound.mono roundStartLe
  have rightBelowTensor :
      KTermsBelow
        (if owner.2 < tensorRoundHighCount owner.1 then
          generatedPointTerms owner.1 else generatedOneMinusPointTerms owner.1)
        tensorFirstColumn := by
    have roundLt19 : owner.1 < 19 := by omega
    split
    · exact generatedPointTerms_below_tensor owner.1 roundLt19
    · exact generatedOneMinusTerms_below_tensor owner.1 roundLt19
  have tensorLeFirst : tensorFirstColumn <= tensorFirstColumn + 5 * ordinal := by
    omega
  have rightBelow := rightBelowTensor.mono tensorLeFirst
  have firstEq : tensorMulFirstColumn owner.1 owner.2 =
      tensorFirstColumn + 5 * ordinal := by
    simp only [tensorMulFirstColumn, tensorMulOrdinal, ordinalEq]
  refine
    { leftBelow := leftBelow
      rightBelow := rightBelow
      sumLeftBelow := leftBelow.1.append leftBelow.2
      sumRightBelow := rightBelow.1.append rightBelow.2
      productC0 := by
        change tensorMulFirstColumn owner.1 owner.2 =
          tensorFirstColumn + 5 * ordinal
        exact firstEq
      productC1 := by
        change tensorMulFirstColumn owner.1 owner.2 + 1 =
          tensorFirstColumn + 5 * ordinal + 1
        rw [firstEq]
      productSum := by
        change tensorMulFirstColumn owner.1 owner.2 + 2 =
          tensorFirstColumn + 5 * ordinal + 2
        rw [firstEq]
      outputC0 := by
        change tensorMulFirstColumn owner.1 owner.2 + 3 =
          tensorFirstColumn + 5 * ordinal + 3
        rw [firstEq]
      outputC1 := by
        change tensorMulFirstColumn owner.1 owner.2 + 3 + 1 =
          tensorFirstColumn + 5 * ordinal + 4
        rw [firstEq] }

private theorem productionTensorDefinitionAt_output
    (index : Nat) (inRange : index < productionTensorDefinitionCount) :
    (productionTensorDefinitionAt index).output = tensorFirstColumn + index := by
  have ordinalInRange : index / 5 < 262143 := by
    simpa [productionTensorDefinitionCount] using
      (Nat.div_lt_iff_lt_mul (by decide : 0 < 5)).2 inRange
  let ordinal := index / 5
  have sequential := generatedTensorTrace_sequential ordinal ordinalInRange
  have output := traceDefinition_output sequential (tensorDefinitionNumber index)
  simp only [productionTensorDefinitionAt, ordinal]
  rw [output]
  simp only [tensorDefinitionNumber]
  omega

private theorem productionTensorDefinitionAt_references_before
    (index : Nat) (inRange : index < productionTensorDefinitionCount)
    (column : Nat)
    (member : column ∈ (productionTensorDefinitionAt index).rhs.refs) :
    column < tensorFirstColumn + index := by
  have ordinalInRange : index / 5 < 262143 := by
    simpa [productionTensorDefinitionCount] using
      (Nat.div_lt_iff_lt_mul (by decide : 0 < 5)).2 inRange
  let ordinal := index / 5
  have sequential := generatedTensorTrace_sequential ordinal ordinalInRange
  have before := traceDefinition_references_before sequential
    (tensorDefinitionNumber index) column (by
      simpa [productionTensorDefinitionAt, ordinal] using member)
  simp only [tensorDefinitionNumber] at before
  omega

/-! ## Coordinate-family order and references -/

private theorem generatedRawTerms_below_tensor
    (lane block : Nat) (laneInRange : lane < 54)
    (blockInRange : block < generatedBlockCount) :
    TermsBelow (generatedRawTerms lane block) tensorFirstColumn := by
  intro term member
  simp only [generatedRawTerms,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms,
    List.mem_map, List.mem_range] at member
  rcases member with ⟨child, childInRange, rfl⟩
  change child < 14 at childInRange
  change block < 211797 at blockInRange
  change 147 + child * (54 * 211797) + (lane * 211797 + block) <
    160118679
  omega

private theorem generatedChiTerms_below_product (block : Nat) :
    KTermsBelow (generatedChiTerms block) productFirstColumn := by
  have below := tensorTermsAt_below_round tensorVariables block
  simpa [generatedChiTerms,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.chiTerms,
    tensorVariables, tensorMulFirstColumn, tensorMulOrdinal,
    tensorRoundMulStart, tensorRoundMulCount, productFirstColumn,
    tensorFirstColumn, generatedBlockCount] using below

private theorem productionCoordinateDefinitionAt_output
    (index : Nat) (inRange : index < productionCoordinateDefinitionCount) :
    (productionCoordinateDefinitionAt index).output =
      productFirstColumn + index := by
  simp only [productionCoordinateDefinitionAt, productColumn, witnessOffset]
  change productFirstColumn +
      2 * ((index / 2 / 211797) * 211797 + index / 2 % 211797) +
        index % 2 = productFirstColumn + index
  have blockSplit := Nat.div_add_mod (index / 2) 211797
  have limbSplit := Nat.div_add_mod index 2
  omega

private theorem productionCoordinateDefinitionAt_references_before
    (index : Nat) (inRange : index < productionCoordinateDefinitionCount)
    (column : Nat)
    (member :
      column ∈ (productionCoordinateDefinitionAt index).rhs.refs) :
    column < productFirstColumn + index := by
  let ordinal := index / 2
  let lane := ordinal / generatedBlockCount
  let block := ordinal % generatedBlockCount
  have ordinalInRange : ordinal < 11437038 := by
    exact (Nat.div_lt_iff_lt_mul (by decide : 0 < 2)).2 (by
      simpa [productionCoordinateDefinitionCount] using inRange)
  have laneInRange : lane < 54 := by
    change (index / 2) / 211797 < 54
    rw [Nat.div_lt_iff_lt_mul (by decide : 0 < 211797)]
    exact ordinalInRange
  have blockInRange : block < generatedBlockCount :=
    Nat.mod_lt _ (by decide)
  have rawBelow := generatedRawTerms_below_tensor lane block
    laneInRange blockInRange
  have chiBelow := generatedChiTerms_below_product block
  simp only [productionCoordinateDefinitionAt, Rhs.refs,
    List.mem_append, List.mem_map] at member
  rcases member with ⟨term, termMember, rfl⟩ | ⟨term, termMember, rfl⟩
  · exact Nat.lt_of_lt_of_le
      (Nat.lt_trans (rawBelow term termMember)
        productionTensorColumns_before_products)
      (Nat.le_add_right _ _)
  · split at termMember
    · exact Nat.lt_of_lt_of_le (chiBelow.1 term termMember) (by omega)
    · exact Nat.lt_of_lt_of_le (chiBelow.2 term termMember) (by omega)

/-! ## Final-scale order and references -/

private theorem generatedLaneTerms_below_scale
    (lane limb : Nat) (laneInRange : lane < 54)
    (limbInRange : limb < 2) :
    TermsBelow
      (generatedLaneTerms lane limb)
      finalScaleFirstColumn := by
  intro term member
  simp only [generatedLaneTerms,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.laneTerms,
    List.mem_map, List.mem_range] at member
  rcases member with ⟨block, blockInRange, rfl⟩
  change block < 211797 at blockInRange
  change 161429394 + 2 * (lane * 211797 + block) + limb < 184303470
  omega

private theorem generatedFinalScaleTrace_sequential
    (lane : Nat) (laneInRange : lane < 54) :
    TraceSequentialAt (generatedFinalScaleTrace lane)
      (finalScaleFirstColumn + 5 * lane) := by
  have left0 := generatedLaneTerms_below_scale lane 0 laneInRange (by decide)
  have left1 := generatedLaneTerms_below_scale lane 1 laneInRange (by decide)
  have rightTensor := generatedOneMinusTerms_below_tensor 18 (by decide)
  have tensorLeScale : tensorFirstColumn <= finalScaleFirstColumn + 5 * lane := by
    exact Nat.le_trans productionTensorColumns_before_scale (Nat.le_add_right _ _)
  have right := rightTensor.mono tensorLeScale
  have scaleBaseLe :
      finalScaleFirstColumn <= finalScaleFirstColumn + 5 * lane :=
    Nat.le_add_right _ _
  have left0At := TermsBelow.mono left0 scaleBaseLe
  have left1At := TermsBelow.mono left1 scaleBaseLe
  refine
    { leftBelow := by
        constructor
        · simpa only [generatedFinalScaleTrace,
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
            using left0At
        · simpa only [generatedFinalScaleTrace,
            Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
            using left1At
      rightBelow := by
        simpa only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
          using right
      sumLeftBelow := by
        simpa only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
          using TermsBelow.append left0At left1At
      sumRightBelow := by
        simpa only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
          using right.1.append right.2
      productC0 := by
        simp only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
      productC1 := by
        simp only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
      productSum := by
        simp only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace]
      outputC0 := by
        simp only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace,
          kColumnsAt]
      outputC1 := by
        simp only [generatedFinalScaleTrace,
          Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace,
          kColumnsAt] }

private theorem productionScaleDefinitionAt_output
    (index : Nat) (inRange : index < productionScaleDefinitionCount) :
    (productionScaleDefinitionAt index).output =
      finalScaleFirstColumn + index := by
  have laneInRange : index / 5 < 54 := by
    exact (Nat.div_lt_iff_lt_mul (by decide : 0 < 5)).2 (by
      simpa [productionScaleDefinitionCount] using inRange)
  have sequential := generatedFinalScaleTrace_sequential (index / 5)
    laneInRange
  have output := traceDefinition_output sequential (tensorDefinitionNumber index)
  simp only [productionScaleDefinitionAt]
  rw [output]
  simp only [tensorDefinitionNumber]
  omega

private theorem productionScaleDefinitionAt_references_before
    (index : Nat) (inRange : index < productionScaleDefinitionCount)
    (column : Nat)
    (member : column ∈ (productionScaleDefinitionAt index).rhs.refs) :
    column < finalScaleFirstColumn + index := by
  have laneInRange : index / 5 < 54 := by
    exact (Nat.div_lt_iff_lt_mul (by decide : 0 < 5)).2 (by
      simpa [productionScaleDefinitionCount] using inRange)
  have sequential := generatedFinalScaleTrace_sequential (index / 5)
    laneInRange
  have before := traceDefinition_references_before sequential
    (tensorDefinitionNumber index) column (by
      simpa [productionScaleDefinitionAt] using member)
  simp only [tensorDefinitionNumber] at before
  omega

/-! ## Consecutive production SSA block -/

theorem productionDefinitionAt_output
    (index : Nat) (inProgram : index < productionDerivedDefinitionCount) :
    (productionDefinitionAt index).output = tensorFirstColumn + index := by
  by_cases inTensor : index < productionTensorDefinitionCount
  · rw [productionDefinitionAt_tensor index inTensor,
      productionTensorDefinitionAt_output index inTensor]
  · by_cases inCoordinate :
        index < productionTensorDefinitionCount +
          productionCoordinateDefinitionCount
    · have afterTensor : productionTensorDefinitionCount <= index :=
        Nat.le_of_not_gt inTensor
      rw [productionDefinitionAt_coordinate index afterTensor inCoordinate]
      have offsetInRange :
          index - productionTensorDefinitionCount <
            productionCoordinateDefinitionCount := by omega
      rw [productionCoordinateDefinitionAt_output _ offsetInRange]
      rw [productionProductFirstColumn_eq]
      omega
    · have afterCoordinate :
          productionTensorDefinitionCount +
            productionCoordinateDefinitionCount <= index :=
        Nat.le_of_not_gt inCoordinate
      rw [productionDefinitionAt_scale index afterCoordinate]
      have offsetInRange :
          index - productionTensorDefinitionCount -
              productionCoordinateDefinitionCount <
            productionScaleDefinitionCount := by
        rw [productionDerivedDefinitionCount_partition] at inProgram
        omega
      rw [productionScaleDefinitionAt_output _ offsetInRange]
      rw [productionFinalScaleFirstColumn_eq]
      omega

theorem productionDefinitionAt_references_before
    (index : Nat) (inProgram : index < productionDerivedDefinitionCount)
    (column : Nat) (member : column ∈ (productionDefinitionAt index).rhs.refs) :
    column < tensorFirstColumn + index := by
  by_cases inTensor : index < productionTensorDefinitionCount
  · rw [productionDefinitionAt_tensor index inTensor] at member
    exact productionTensorDefinitionAt_references_before index inTensor
      column member
  · by_cases inCoordinate :
        index < productionTensorDefinitionCount +
          productionCoordinateDefinitionCount
    · have afterTensor : productionTensorDefinitionCount <= index :=
        Nat.le_of_not_gt inTensor
      rw [productionDefinitionAt_coordinate index afterTensor inCoordinate] at member
      have offsetInRange :
          index - productionTensorDefinitionCount <
            productionCoordinateDefinitionCount := by omega
      have before := productionCoordinateDefinitionAt_references_before
        (index - productionTensorDefinitionCount) offsetInRange column member
      rw [productionProductFirstColumn_eq] at before
      omega
    · have afterCoordinate :
          productionTensorDefinitionCount +
            productionCoordinateDefinitionCount <= index :=
        Nat.le_of_not_gt inCoordinate
      rw [productionDefinitionAt_scale index afterCoordinate] at member
      have offsetInRange :
          index - productionTensorDefinitionCount -
              productionCoordinateDefinitionCount <
            productionScaleDefinitionCount := by
        rw [productionDerivedDefinitionCount_partition] at inProgram
        omega
      have before := productionScaleDefinitionAt_references_before
        (index - productionTensorDefinitionCount -
          productionCoordinateDefinitionCount) offsetInRange column member
      rw [productionFinalScaleFirstColumn_eq] at before
      omega

/-- The complete optimized production derived-column program.  Its count is
definitionally `canonicalColumnCount - tensorFirstColumn`, and its output and
dependency fields are proved from generated compiler formulas. -/
def productionDerivedProgram :
    SequentialProgram tensorFirstColumn
      (canonicalColumnCount - tensorFirstColumn) where
  definitionAt := productionDefinitionAt
  output_eq := by
    intro index inProgram
    exact productionDefinitionAt_output index inProgram
  references_before := by
    intro index inProgram column member
    exact productionDefinitionAt_references_before index inProgram column member

@[simp] theorem productionDerivedProgram_definitionAt (index : Nat) :
    productionDerivedProgram.definitionAt index = productionDefinitionAt index :=
  rfl

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
