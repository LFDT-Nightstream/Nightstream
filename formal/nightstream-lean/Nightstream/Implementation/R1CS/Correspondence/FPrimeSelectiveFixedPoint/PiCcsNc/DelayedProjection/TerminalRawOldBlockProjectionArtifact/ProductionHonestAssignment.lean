import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionDerivedProgram

/-!
List-free honest assignment for the optimized production raw-old-block rows.

The source contains only verifier-owned pending state and the fourteen ordered
raw `WitnessMat` matrices.  `productionDerivedProgram` deterministically
materializes every derived tensor, coordinate-product, and final-scale column.
The resulting values are then used as the internal witness of the generated
physical/canonical assignment map.

This leaf does not take row satisfaction, projection acceptance, a child
sidecar, or a digest as an input.  The only semantic premise used by terminal
completeness is equality between each parent value and its generated
final-scale output.

Owns: honest source construction, list-free materialization, agreement with
the production canonical assignment, and the completeness reduction from
typed compiler definitions to artifact rows.

Does not own: semantic authority of the terminal equality, commitment
binding, transcript generation, or security reductions.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.raw_old_block.honest.source` | construct non-derived columns from outgoing pending state and fourteen ordered raw `WitnessMat` matrices | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.honest.materialize` | execute the list-free derived program and prove canonical field bounds | computed |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.honest.bridge` | identify every typed tensor, coordinate, and scale definition with its generated program index | derived |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.honest.rows` | construct satisfaction of all factored artifact rows from the semantic terminal equality | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

private abbrev HonestMatrices :=
  Fin productionGlobalParams.k ->
    PackedWitness.Matrix ProductionDomain.semanticShape

private def zeroInternalWitness : Nat -> ProjectionProgram.F :=
  fun _ => 0

/-- Honest non-derived columns.  The dummy values above `tensorFirstColumn`
are never read by the sequential program before those columns are overwritten. -/
def honestSourceAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) : Nat -> Nat :=
  sourceAssignment pending finalWitnesses zeroInternalWitness

/-- Deterministic list-free execution of all 24,185,061 derived definitions. -/
def honestMaterializedAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) : Nat -> Nat :=
  SequentialProgram.materialize productionDerivedProgram
    (honestSourceAssignment pending finalWitnesses)

theorem honestSourceAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) (column : Nat) :
    honestSourceAssignment pending finalWitnesses column < goldilocksP := by
  exact sourceAssignment_lt pending finalWitnesses zeroInternalWitness column

theorem honestMaterializedAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) (column : Nat) :
    honestMaterializedAssignment pending finalWitnesses column < goldilocksP := by
  exact SequentialProgram.materialize_canonical productionDerivedProgram
    (honestSourceAssignment pending finalWitnesses)
    (honestSourceAssignment_lt pending finalWitnesses) column

/-- Canonical field-valued internal witness computed by the SSA interpreter. -/
def honestInternalWitness
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (column : Nat) : ProjectionProgram.F :=
  ⟨honestMaterializedAssignment pending finalWitnesses column,
    honestMaterializedAssignment_lt pending finalWitnesses column⟩

/-- The assignment after the actual generated physical-column map is pulled
back to canonical compiler columns. -/
def honestCanonicalAssignment
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) : Nat -> Nat :=
  canonicalAssignment pending finalWitnesses
    (honestInternalWitness pending finalWitnesses)

theorem honestCanonicalAssignment_lt
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) (column : Nat) :
    honestCanonicalAssignment pending finalWitnesses column < goldilocksP := by
  exact canonicalAssignment_lt pending finalWitnesses
    (honestInternalWitness pending finalWitnesses) column

theorem honestMaterializedAssignment_source_below
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) {column : Nat}
    (below : column < tensorFirstColumn) :
    honestMaterializedAssignment pending finalWitnesses column =
      honestSourceAssignment pending finalWitnesses column := by
  exact SequentialProgram.materialize_source_below productionDerivedProgram
    (honestSourceAssignment pending finalWitnesses) below

private theorem sourceAssignment_honestInternal_below
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) {column : Nat}
    (below : column < tensorFirstColumn) :
    sourceAssignment pending finalWitnesses
        (honestInternalWitness pending finalWitnesses) column =
      honestSourceAssignment pending finalWitnesses column := by
  simp [honestSourceAssignment, zeroInternalWitness, sourceAssignment, below]

/-- The physical emitter inverse and the deterministic SSA materializer agree
on every canonical column used by the generated program. -/
theorem honestCanonicalAssignment_eq_materialized
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (column : Nat) (inRange : column < canonicalColumnCount) :
    honestCanonicalAssignment pending finalWitnesses column =
      honestMaterializedAssignment pending finalWitnesses column := by
  rw [show honestCanonicalAssignment pending finalWitnesses column =
      sourceAssignment pending finalWitnesses
        (honestInternalWitness pending finalWitnesses) column by
    exact canonicalAssignment_eq_source pending finalWitnesses
      (honestInternalWitness pending finalWitnesses) column inRange]
  by_cases below : column < tensorFirstColumn
  · rw [sourceAssignment_honestInternal_below pending finalWitnesses below]
    exact (honestMaterializedAssignment_source_below pending finalWitnesses
      below).symm
  · have derived : tensorFirstColumn <= column := Nat.le_of_not_gt below
    rw [sourceAssignment_internal pending finalWitnesses
      (honestInternalWitness pending finalWitnesses) column derived]
    rfl

/-! ## Generated-definition satisfaction -/

private theorem lcEval_congr_on_terms
    {left right : Nat -> Nat} (terms : List (Nat × Nat))
    (agreement : forall term, term ∈ terms -> left term.1 = right term.1) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgreement : forall initial,
      terms.foldl (fun acc term => acc + term.2 * left term.1) initial =
        terms.foldl (fun acc term => acc + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head (by simp)]
        exact inductionHypothesis (fun term member =>
          agreement term (by simp [member])) _
  rw [foldAgreement 0]

private theorem rhsEval_congr_on_refs
    {left right : Nat -> Nat} (rhs : Rhs)
    (agreement : forall column, column ∈ rhs.refs ->
      left column = right column) :
    rhs.eval left = rhs.eval right := by
  cases rhs with
  | linear terms =>
      apply lcEval_congr_on_terms terms
      intro term member
      apply agreement term.1
      exact List.mem_map.mpr ⟨term, member, rfl⟩
  | product lhs rhs =>
      simp only [Rhs.eval]
      rw [lcEval_congr_on_terms lhs (by
        intro term member
        apply agreement term.1
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]
      rw [lcEval_congr_on_terms rhs (by
        intro term member
        apply agreement term.1
        apply List.mem_append_right
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]

theorem honestMaterializedAssignment_definition_holds
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (index : Nat) (inProgram : index < productionDerivedDefinitionCount) :
    Definition.Holds
      (honestMaterializedAssignment pending finalWitnesses)
      (productionDefinitionAt index) := by
  exact SequentialProgram.materialize_definition_holds
    productionDerivedProgram (honestSourceAssignment pending finalWitnesses)
    index inProgram

/-- Every generated SSA definition also holds in the canonical compiler view.
This is a consequence of full canonical-column agreement, not a satisfaction
premise. -/
theorem honestCanonicalAssignment_definition_holds
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (index : Nat) (inProgram : index < productionDerivedDefinitionCount) :
    Definition.Holds
      (honestCanonicalAssignment pending finalWitnesses)
      (productionDefinitionAt index) := by
  have outputEq := productionDefinitionAt_output index inProgram
  have outputInRange :
      (productionDefinitionAt index).output < canonicalColumnCount := by
    rw [outputEq]
    unfold productionDerivedDefinitionCount at inProgram
    have firstLe : tensorFirstColumn <= canonicalColumnCount := by decide
    omega
  have referenceInRange : forall column,
      column ∈ (productionDefinitionAt index).rhs.refs ->
        column < canonicalColumnCount := by
    intro column member
    exact Nat.lt_trans
      (productionDefinitionAt_references_before index inProgram column member)
      (by simpa [outputEq] using outputInRange)
  have rhsAgreement := rhsEval_congr_on_refs
    (productionDefinitionAt index).rhs (by
      intro column member
      exact honestCanonicalAssignment_eq_materialized pending finalWitnesses
        column (referenceInRange column member))
  have materializedHolds :=
    honestMaterializedAssignment_definition_holds pending finalWitnesses
      index inProgram
  unfold Definition.Holds at materializedHolds ⊢
  rw [honestCanonicalAssignment_eq_materialized pending finalWitnesses
    (productionDefinitionAt index).output outputInRange]
  rw [rhsAgreement]
  exact materializedHolds

/-! ## Semantic terminal equality -/

/-- The only semantic premise of honest row completeness: each authoritative
parent value equals the final-scale output computed by the SSA program. -/
def SemanticTerminalParentEquality
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices) : Prop :=
  forall lane : Fin productionFactoredLayout.base.activeLanes,
    (productionFactoredLayout.base.parent lane).value
        (honestCanonicalAssignment pending finalWitnesses) =
      (productionFactoredLayout.scale lane).output.value
        (honestCanonicalAssignment pending finalWitnesses)

private theorem semanticTerminalParentEquality_terminalHolds
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (terminalEquality :
      SemanticTerminalParentEquality pending finalWitnesses) :
    forall lane,
      honestCanonicalAssignment pending finalWitnesses
          (productionFactoredLayout.base.parent lane).c0 =
          lcEval (honestCanonicalAssignment pending finalWitnesses)
            (terminalTerms productionFactoredLayout lane KColumns.c0) /\
        honestCanonicalAssignment pending finalWitnesses
          (productionFactoredLayout.base.parent lane).c1 =
          lcEval (honestCanonicalAssignment pending finalWitnesses)
            (terminalTerms productionFactoredLayout lane KColumns.c1) := by
  intro lane
  have equality := terminalEquality lane
  rw [productionFinalScaleTrace] at equality
  have c0Equality := congrArg (fun value => value.c0.val) equality
  have c1Equality := congrArg (fun value => value.c1.val) equality
  have parent0Canonical := honestCanonicalAssignment_lt pending finalWitnesses
    (productionFactoredLayout.base.parent lane).c0
  have parent1Canonical := honestCanonicalAssignment_lt pending finalWitnesses
    (productionFactoredLayout.base.parent lane).c1
  have output0Canonical := honestCanonicalAssignment_lt pending finalWitnesses
    (finalScaleTrace lane.val).output.c0
  have output1Canonical := honestCanonicalAssignment_lt pending finalWitnesses
    (finalScaleTrace lane.val).output.c1
  simp only [KColumns.value, baseAt, residue] at c0Equality c1Equality
  rw [Nat.mod_eq_of_lt parent0Canonical,
    Nat.mod_eq_of_lt output0Canonical] at c0Equality
  rw [Nat.mod_eq_of_lt parent1Canonical,
    Nat.mod_eq_of_lt output1Canonical] at c1Equality
  constructor
  · calc
      honestCanonicalAssignment pending finalWitnesses
          (productionFactoredLayout.base.parent lane).c0 =
          honestCanonicalAssignment pending finalWitnesses
            (finalScaleTrace lane.val).output.c0 := c0Equality
      _ = lcEval (honestCanonicalAssignment pending finalWitnesses)
            (terminalTerms productionFactoredLayout lane KColumns.c0) := by
        simp [terminalTerms, lcEval,
          Nat.mod_eq_of_lt output0Canonical]
  · calc
      honestCanonicalAssignment pending finalWitnesses
          (productionFactoredLayout.base.parent lane).c1 =
          honestCanonicalAssignment pending finalWitnesses
            (finalScaleTrace lane.val).output.c1 := c1Equality
      _ = lcEval (honestCanonicalAssignment pending finalWitnesses)
            (terminalTerms productionFactoredLayout lane KColumns.c1) := by
        simp [terminalTerms, lcEval,
          Nat.mod_eq_of_lt output1Canonical]

/-! ## Exact remaining syntactic bridge

The generated SSA program and typed compiler use different compact indices.
The structure below states only their syntactic equality.  It contains no
assignment, acceptance predicate, or satisfaction proposition.  Once a
kernel theorem constructs this value, the final completeness theorem below
has only `SemanticTerminalParentEquality` as a semantic premise.
-/

private theorem honestTensorSlotOrdinal
    (level : Fin productionLayout.tensorLevels.length)
    (multiplication : Fin
      (productionLayout.tensorLevels.get level).multiplicationCount) :
    (TensorSlot.at level multiplication).toFin.val =
      tensorMulOrdinal level.val multiplication.val := by
  rw [TensorSlot.toFin_at_val]
  change
    (productionTensorLevels.take level.val).foldl
        (fun count current => count + current.multiplicationCount) 0 +
      multiplication.val =
    tensorRoundMulStart level.val + multiplication.val
  congr 1
  unfold productionTensorLevels tensorRoundMulStart
  rw [← List.map_take]
  change
    List.foldl (fun count current => count + current.multiplicationCount) 0
        (List.map productionTensorLevel
          (List.take level.val (List.range 18))) =
      List.foldl (fun count prior => count + tensorRoundMulCount prior) 0
        (List.range level.val)
  have levelLt : level.val < 18 := by
    have value := level.isLt
    change level.val < 18 at value
    exact value
  rw [List.take_range, Nat.min_eq_left (Nat.le_of_lt levelLt)]
  rw [List.foldl_map]
  rfl

private theorem honestPhysicalIndex_tensor
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    (productionPhysicalIndex (.tensor index)).val =
      tensorPhysicalRow index.level.val index.multiplication.val
        index.definition.val := by
  rcases index with ⟨level, multiplication, definition⟩
  have ordinalEq := honestTensorSlotOrdinal level multiplication
  change
    (TensorSlot.at level multiplication).toFin.val * 5 + definition.val =
      tensorPhysicalRow level.val multiplication.val definition.val
  unfold tensorPhysicalRow
  simpa [Nat.mul_comm] using
    congrArg (fun value => value * 5 + definition.val) ordinalEq

private theorem honestPhysicalIndex_coordinate
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    (productionPhysicalIndex (.coordinate coordinate limb)).val =
      productPhysicalRow (coordinate.val % 54) (coordinate.val / 54)
        limb.val := by
  change
    tensorSlotCount productionLayout.tensorLevels * 5 +
        (((coordinate.val % productionLayout.activeLanes) *
          blockCount productionLayout +
          coordinate.val / productionLayout.activeLanes) * 2 + limb.val) =
      productPhysicalRow (coordinate.val % 54) (coordinate.val / 54)
        limb.val
  rw [show tensorSlotCount productionLayout.tensorLevels = 262143 by
    rw [← tensorMultiplicationCount_eq_slotCount]
    exact productionTensorMultiplicationCount]
  rw [productionActiveLanes, productionBlockCount]
  unfold productPhysicalRow witnessOffset productRowFirst
  simp only [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
  omega

private theorem honestPhysicalIndex_scale
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    (productionPhysicalIndex (.scale lane definition)).val =
      finalScalePhysicalRow lane.val definition.val := by
  change
    tensorSlotCount productionLayout.tensorLevels * 5 +
        2 * productionLayout.logicalWidth +
        (lane.val * 5 + definition.val) =
      finalScalePhysicalRow lane.val definition.val
  rw [show tensorSlotCount productionLayout.tensorLevels = 262143 by
    rw [← tensorMultiplicationCount_eq_slotCount]
    exact productionTensorMultiplicationCount]
  rw [productionLogicalWidth]
  unfold finalScalePhysicalRow finalScaleRowFirst
  omega

private theorem honestTensorOwnerAt
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    tensorOwner
        (tensorMulOrdinal index.level.val index.multiplication.val) =
      (index.level.val, index.multiplication.val) := by
  have ownerEquality := productionOwner_tensor index
  change ownerAtNat (productionPhysicalIndex (.tensor index)).val = _
    at ownerEquality
  rw [honestPhysicalIndex_tensor index] at ownerEquality
  have ordinalLt :
      tensorMulOrdinal index.level.val index.multiplication.val < 262143 := by
    calc
      tensorMulOrdinal index.level.val index.multiplication.val =
          (TensorSlot.at index.level index.multiplication).toFin.val :=
        (honestTensorSlotOrdinal index.level index.multiplication).symm
      _ < tensorSlotCount productionLayout.tensorLevels :=
        (TensorSlot.at index.level index.multiplication).toFin.isLt
      _ = 262143 := by
        rw [← tensorMultiplicationCount_eq_slotCount]
        exact productionTensorMultiplicationCount
  have definitionLt : index.definition.val < 5 := index.definition.isLt
  have rowLt :
      tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val < tensorRows := by
    unfold tensorPhysicalRow tensorRows
    omega
  have rowDiv :
      tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val / 5 =
        tensorMulOrdinal index.level.val index.multiplication.val := by
    simp [tensorPhysicalRow]
    omega
  have rowMod :
      tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val % 5 = index.definition.val := by
    simp [tensorPhysicalRow, Nat.mod_eq_of_lt definitionLt]
  change ownerAtNat
      (tensorPhysicalRow index.level.val index.multiplication.val
        index.definition.val) = _
      at ownerEquality
  unfold ownerAtNat at ownerEquality
  rw [if_pos rowLt, rowDiv, rowMod] at ownerEquality
  have roundEq :
      (tensorOwner
        (tensorMulOrdinal index.level.val index.multiplication.val)).1 =
          index.level.val := by
    simpa using congrArg (fun owner => match owner with
      | RowOwner.tensor round _ _ => round
      | _ => 0) ownerEquality
  have parentEq :
      (tensorOwner
        (tensorMulOrdinal index.level.val index.multiplication.val)).2 =
          index.multiplication.val := by
    simpa using congrArg (fun owner => match owner with
      | RowOwner.tensor _ parent _ => parent
      | _ => 0) ownerEquality
  exact Prod.ext roundEq parentEq

private theorem honestFin18_cases {predicate : Fin 18 -> Prop}
    (case0 : predicate 0) (case1 : predicate 1)
    (case2 : predicate 2) (case3 : predicate 3)
    (case4 : predicate 4) (case5 : predicate 5)
    (case6 : predicate 6) (case7 : predicate 7)
    (case8 : predicate 8) (case9 : predicate 9)
    (case10 : predicate 10) (case11 : predicate 11)
    (case12 : predicate 12) (case13 : predicate 13)
    (case14 : predicate 14) (case15 : predicate 15)
    (case16 : predicate 16) (case17 : predicate 17) :
    forall index, predicate index := by
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
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  refine Fin.cases case13 ?_ index
  intro index
  refine Fin.cases case14 ?_ index
  intro index
  refine Fin.cases case15 ?_ index
  intro index
  refine Fin.cases case16 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case17

private theorem honestTensorTraceEq
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace
        index.level.val index.multiplication.val =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
        index := by
  rcases index with ⟨level, multiplication, definition⟩
  revert multiplication
  refine honestFin18_cases (predicate := fun current =>
      forall multiplication : Fin
          (productionLayout.tensorLevels.get current).multiplicationCount,
        Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace
            current.val multiplication.val =
          Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
            (layout := productionLayout)
            ⟨current, multiplication, definition⟩)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ level <;>
    intro multiplication <;> rfl

private theorem honestMapRange_eq_ofFn
    {Item : Type} (count : Nat) (value : Nat -> Item) :
    (List.range count).map value =
      List.ofFn fun index : Fin count => value index.val := by
  apply List.ext_get
  · simp
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_map, List.getElem_range,
      List.getElem_ofFn]

private theorem honestProductionRawTerms
    (coordinate : Fin productionLayout.logicalWidth) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
        (coordinate.val % 54) (coordinate.val / 54) =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
        productionLayout coordinate := by
  unfold
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
  rw [honestMapRange_eq_ofFn]
  apply congrArg List.ofFn
  funext child
  apply Prod.ext
  · unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawWitnessColumn
    rw [productionChildWitnessFirst, productionActiveLanes,
      productionBlockCount]
    simp [childWitnessColumn, childWitnessFirst, childWitnessFirstNat,
      witnessOffset,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount,
      Nat.add_assoc]
  · unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.radixCoefficient
    rw [productionRadix]
    rfl

private theorem honestProductionChiTerms
    (coordinate : Fin productionLayout.logicalWidth) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.chiTerms
        (coordinate.val / 54) =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms
        productionLayout coordinate := by
  change
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTermsAt
        18 (coordinate.val / 54) = _
  exact (productionCoordinateChiTerms coordinate).symm

private theorem honestProductionProductColumns
    (coordinate : Fin productionLayout.logicalWidth) :
    productColumns productionLayout coordinate =
      { c0 := productColumn (coordinate.val % 54) (coordinate.val / 54) 0
        c1 := productColumn (coordinate.val % 54) (coordinate.val / 54) 1 } := by
  have coordinateLt := coordinate.isLt
  change coordinate.val < 11437038 at coordinateLt
  have blockLt : coordinate.val / 54 < 211797 :=
    (Nat.div_lt_iff_lt_mul (by decide : 0 < 54)).2 coordinateLt
  unfold productColumns productColumn witnessOffset
  rw [productionActiveLanes, productionBlockCount, productionProductFirst]
  rfl

private theorem honestTensorIndexInProgram
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    (productionPhysicalIndex (.tensor index)).val <
      productionDerivedDefinitionCount := by
  rw [honestPhysicalIndex_tensor, productionDerivedDefinitionCount_exact]
  have ordinalLt :
      tensorMulOrdinal index.level.val index.multiplication.val < 262143 := by
    calc
      tensorMulOrdinal index.level.val index.multiplication.val =
          (TensorSlot.at index.level index.multiplication).toFin.val :=
        (honestTensorSlotOrdinal index.level index.multiplication).symm
      _ < tensorSlotCount productionLayout.tensorLevels :=
        (TensorSlot.at index.level index.multiplication).toFin.isLt
      _ = 262143 := by
        rw [← tensorMultiplicationCount_eq_slotCount]
        exact productionTensorMultiplicationCount
  have definitionLt := index.definition.isLt
  unfold tensorPhysicalRow
  omega

private theorem honestCoordinateIndexInProgram
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    (productionPhysicalIndex (.coordinate coordinate limb)).val <
      productionDerivedDefinitionCount := by
  rw [honestPhysicalIndex_coordinate, productionDerivedDefinitionCount_exact]
  have coordinateLt := coordinate.isLt
  change coordinate.val < 11437038 at coordinateLt
  have laneLt : coordinate.val % 54 < 54 := Nat.mod_lt _ (by decide)
  have blockLt : coordinate.val / 54 < 211797 :=
    (Nat.div_lt_iff_lt_mul (by decide : 0 < 54)).2 coordinateLt
  have limbLt := limb.isLt
  unfold productPhysicalRow productRowFirst witnessOffset
  simp only [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
  omega

private theorem honestScaleIndexInProgram
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    (productionPhysicalIndex (.scale lane definition)).val <
      productionDerivedDefinitionCount := by
  rw [honestPhysicalIndex_scale, productionDerivedDefinitionCount_exact]
  have laneLt := lane.isLt
  change lane.val < 54 at laneLt
  have definitionLt := definition.isLt
  unfold finalScalePhysicalRow finalScaleRowFirst
  omega

private theorem honestTensorDefinition
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    productionDefinitionAt (productionPhysicalIndex (.tensor index)).val =
      (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
        index).definitions.get
        ⟨index.definition.val, by simp [KMulTrace.definitions]⟩ := by
  have inFamily :
      (productionPhysicalIndex (.tensor index)).val <
        productionTensorDefinitionCount := by
    rw [honestPhysicalIndex_tensor, productionTensorDefinitionCount_exact]
    have ordinalLt :
        tensorMulOrdinal index.level.val index.multiplication.val < 262143 := by
      calc
        tensorMulOrdinal index.level.val index.multiplication.val =
            (TensorSlot.at index.level index.multiplication).toFin.val :=
          (honestTensorSlotOrdinal index.level index.multiplication).symm
        _ < tensorSlotCount productionLayout.tensorLevels :=
          (TensorSlot.at index.level index.multiplication).toFin.isLt
        _ = 262143 := by
          rw [← tensorMultiplicationCount_eq_slotCount]
          exact productionTensorMultiplicationCount
    have definitionLt := index.definition.isLt
    unfold tensorPhysicalRow
    omega
  rw [productionDefinitionAt_tensor _ inFamily]
  rw [honestPhysicalIndex_tensor]
  have definitionLt := index.definition.isLt
  have rowDiv :
      tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val / 5 =
        tensorMulOrdinal index.level.val index.multiplication.val := by
    simp [tensorPhysicalRow]
    omega
  have rowMod :
      tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val % 5 = index.definition.val := by
    simp [tensorPhysicalRow, Nat.mod_eq_of_lt definitionLt]
  change
    (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace
      (tensorOwner
        (tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val / 5)).1
      (tensorOwner
        (tensorPhysicalRow index.level.val index.multiplication.val
          index.definition.val / 5)).2).definitions.get
        ⟨tensorPhysicalRow index.level.val index.multiplication.val
            index.definition.val % 5,
          Nat.mod_lt _ (by decide)⟩ = _
  rw [rowDiv]
  rw [honestTensorOwnerAt index, honestTensorTraceEq index]
  congr 1
  exact Fin.ext rowMod

private theorem honestCoordinateDefinition
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    productionDefinitionAt
        (productionPhysicalIndex (.coordinate coordinate limb)).val =
      (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
        productionFactoredLayout.base coordinate).get
        ⟨limb.val, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩ := by
  change productionDefinitionAt
      (productionPhysicalIndex (.coordinate coordinate limb)).val =
    (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
      productionLayout coordinate).get
      ⟨limb.val, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩
  have physicalEq := honestPhysicalIndex_coordinate coordinate limb
  let ordinal :=
    (coordinate.val % 54) * 211797 + coordinate.val / 54
  have coordinateLt := coordinate.isLt
  change coordinate.val < 11437038 at coordinateLt
  have blockLt : coordinate.val / 54 < 211797 :=
    (Nat.div_lt_iff_lt_mul (by decide : 0 < 54)).2 coordinateLt
  have limbLt := limb.isLt
  have afterTensor :
      productionTensorDefinitionCount <=
        (productionPhysicalIndex (.coordinate coordinate limb)).val := by
    rw [physicalEq, productionTensorDefinitionCount_exact]
    unfold productPhysicalRow productRowFirst witnessOffset
    omega
  have beforeScale :
      (productionPhysicalIndex (.coordinate coordinate limb)).val <
        productionTensorDefinitionCount +
          productionCoordinateDefinitionCount := by
    rw [physicalEq, productionTensorDefinitionCount_exact,
      productionCoordinateDefinitionCount_exact]
    unfold productPhysicalRow productRowFirst witnessOffset
    simp only [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
    omega
  rw [productionDefinitionAt_coordinate _ afterTensor beforeScale]
  have offsetEq :
      (productionPhysicalIndex (.coordinate coordinate limb)).val -
          productionTensorDefinitionCount =
        2 * ordinal + limb.val := by
    rw [physicalEq, productionTensorDefinitionCount_exact]
    unfold productPhysicalRow productRowFirst witnessOffset
    simp only [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
    omega
  rw [offsetEq]
  unfold productionCoordinateDefinitionAt
  change
    (let decodedOrdinal := (2 * ordinal + limb.val) / 2
     let decodedLane := decodedOrdinal / 211797
     let decodedBlock := decodedOrdinal % 211797
     let decodedLimb := (2 * ordinal + limb.val) % 2
     let chi :=
       Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.chiTerms
         decodedBlock
     let selected := if decodedLimb = 0 then chi.c0 else chi.c1
     Definition.mk (productColumn decodedLane decodedBlock decodedLimb)
       (.product
         (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
           decodedLane decodedBlock)
         selected)) = _
  have productDiv : (2 * ordinal + limb.val) / 2 = ordinal := by omega
  have productMod : (2 * ordinal + limb.val) % 2 = limb.val := by omega
  have laneEq : ordinal / 211797 = coordinate.val % 54 := by
    dsimp [ordinal]
    rw [Nat.mul_comm (coordinate.val % 54) 211797,
      Nat.mul_add_div (by decide : 0 < 211797),
      Nat.div_eq_of_lt blockLt, Nat.add_zero]
  have blockEq : ordinal % 211797 = coordinate.val / 54 := by
    dsimp [ordinal]
    exact Nat.mul_add_mod_of_lt blockLt
  simp only [productDiv, productMod, laneEq, blockEq]
  have rawEq := honestProductionRawTerms coordinate
  have chiEq := honestProductionChiTerms coordinate
  have outputEq := honestProductionProductColumns coordinate
  refine Fin.cases ?_ (fun tail => ?_) limb
  · simp only [Fin.val_zero, if_pos]
    unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
    rw [rawEq, chiEq, outputEq]
    rfl
  · have tailZero : tail = 0 := Fin.ext (by omega)
    subst tail
    simp only [Fin.val_succ, Fin.val_zero, Nat.zero_add, OfNat.ofNat,
      Nat.reduceEqDiff, if_false]
    unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
    rw [rawEq, chiEq, outputEq]
    rfl

private theorem honestScaleDefinition
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    productionDefinitionAt
        (productionPhysicalIndex (.scale lane definition)).val =
      (productionFactoredLayout.scale lane).definitions.get
        ⟨definition.val, by simp [KMulTrace.definitions]⟩ := by
  have physicalEq := honestPhysicalIndex_scale lane definition
  have afterCoordinate :
      productionTensorDefinitionCount +
          productionCoordinateDefinitionCount <=
        (productionPhysicalIndex (.scale lane definition)).val := by
    rw [physicalEq, productionTensorDefinitionCount_exact,
      productionCoordinateDefinitionCount_exact]
    unfold finalScalePhysicalRow finalScaleRowFirst
    omega
  rw [productionDefinitionAt_scale _ afterCoordinate]
  have offsetEq :
      (productionPhysicalIndex (.scale lane definition)).val -
          productionTensorDefinitionCount -
          productionCoordinateDefinitionCount =
        5 * lane.val + definition.val := by
    rw [physicalEq, productionTensorDefinitionCount_exact,
      productionCoordinateDefinitionCount_exact]
    unfold finalScalePhysicalRow finalScaleRowFirst
    omega
  rw [offsetEq]
  change
    (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.finalScaleTrace
      ((5 * lane.val + definition.val) / 5)).definitions.get
        ⟨(5 * lane.val + definition.val) % 5,
          Nat.mod_lt _ (by decide)⟩ = _
  have definitionLt := definition.isLt
  have scaleDiv : (5 * lane.val + definition.val) / 5 = lane.val := by
    rw [Nat.mul_add_div (by decide : 0 < 5),
      Nat.div_eq_of_lt definitionLt, Nat.add_zero]
  have scaleMod : (5 * lane.val + definition.val) % 5 = definition.val := by
    simpa [Nat.mod_eq_of_lt definitionLt] using
      Nat.mul_add_mod_self_right lane.val 5 definition.val
  rw [scaleDiv]
  change
    (finalScaleTrace lane.val).definitions.get
        ⟨(5 * lane.val + definition.val) % 5,
          Nat.mod_lt _ (by decide)⟩ =
      (finalScaleTrace lane.val).definitions.get
        ⟨definition.val, by simp [KMulTrace.definitions]⟩
  have scaleIndexEq :
      (⟨(5 * lane.val + definition.val) % 5,
        by
          rw [scaleMod]
          simpa [KMulTrace.definitions] using definition.isLt⟩ :
        Fin (finalScaleTrace lane.val).definitions.length) =
      ⟨definition.val, by simp [KMulTrace.definitions]⟩ := by
    exact Fin.ext scaleMod
  exact congrArg
    (fun index => (finalScaleTrace lane.val).definitions.get index)
    scaleIndexEq

private structure TypedDefinitionBridge : Prop where
  tensorIndexInProgram :
    forall index : PrefixTensorRowIndex productionFactoredLayout.base,
      (productionPhysicalIndex (.tensor index)).val <
        productionDerivedDefinitionCount
  tensorDefinition :
    forall index : PrefixTensorRowIndex productionFactoredLayout.base,
      productionDefinitionAt (productionPhysicalIndex (.tensor index)).val =
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
          index).definitions.get
          ⟨index.definition.val, by simp [KMulTrace.definitions]⟩
  coordinateIndexInProgram :
    forall coordinate : Fin productionFactoredLayout.base.logicalWidth,
      forall limb : Fin 2,
        (productionPhysicalIndex (.coordinate coordinate limb)).val <
          productionDerivedDefinitionCount
  coordinateDefinition :
    forall coordinate : Fin productionFactoredLayout.base.logicalWidth,
      forall limb : Fin 2,
        productionDefinitionAt
            (productionPhysicalIndex (.coordinate coordinate limb)).val =
          (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
            productionFactoredLayout.base coordinate).get
            ⟨limb.val, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩
  scaleIndexInProgram :
    forall lane : Fin productionFactoredLayout.base.activeLanes,
      forall definition : Fin 5,
        (productionPhysicalIndex (.scale lane definition)).val <
          productionDerivedDefinitionCount
  scaleDefinition :
    forall lane : Fin productionFactoredLayout.base.activeLanes,
      forall definition : Fin 5,
        productionDefinitionAt
            (productionPhysicalIndex (.scale lane definition)).val =
          (productionFactoredLayout.scale lane).definitions.get
            ⟨definition.val, by simp [KMulTrace.definitions]⟩

private theorem productionTypedDefinitionBridge : TypedDefinitionBridge where
  tensorIndexInProgram := honestTensorIndexInProgram
  tensorDefinition := honestTensorDefinition
  coordinateIndexInProgram := honestCoordinateIndexInProgram
  coordinateDefinition := honestCoordinateDefinition
  scaleIndexInProgram := honestScaleIndexInProgram
  scaleDefinition := honestScaleDefinition

/-- Completeness after the purely syntactic index bridge.  No premise here is
row satisfaction, semantic projection acceptance, or prover-carried
authority. -/
private theorem honestArtifactRowsSatisfied_of_typedDefinitionBridge
    (bridge : TypedDefinitionBridge)
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (terminalEquality :
      SemanticTerminalParentEquality pending finalWitnesses) :
    ArtifactRowsSatisfied productionArtifactContract
      (honestCanonicalAssignment pending finalWitnesses) := by
  let assignment := honestCanonicalAssignment pending finalWitnesses
  have tensorHolds :
      forall index : PrefixTensorRowIndex productionFactoredLayout.base,
        Definition.Holds assignment
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
            index).definitions.get
            ⟨index.definition.val, by simp [KMulTrace.definitions]⟩) := by
    intro index
    rw [← bridge.tensorDefinition index]
    exact honestCanonicalAssignment_definition_holds pending finalWitnesses
      (productionPhysicalIndex (.tensor index)).val
      (bridge.tensorIndexInProgram index)
  have coordinateHolds : forall coordinate,
      Definition.Holds assignment
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
            productionFactoredLayout.base coordinate).get
            ⟨0, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩) /\
        Definition.Holds assignment
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
            productionFactoredLayout.base coordinate).get
            ⟨1, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩) := by
    intro coordinate
    constructor
    · rw [← bridge.coordinateDefinition coordinate ⟨0, by decide⟩]
      exact honestCanonicalAssignment_definition_holds pending finalWitnesses
        (productionPhysicalIndex (.coordinate coordinate ⟨0, by decide⟩)).val
        (bridge.coordinateIndexInProgram coordinate ⟨0, by decide⟩)
    · rw [← bridge.coordinateDefinition coordinate ⟨1, by decide⟩]
      exact honestCanonicalAssignment_definition_holds pending finalWitnesses
        (productionPhysicalIndex (.coordinate coordinate ⟨1, by decide⟩)).val
        (bridge.coordinateIndexInProgram coordinate ⟨1, by decide⟩)
  have scaleHolds : forall lane definition,
      definition ∈ (productionFactoredLayout.scale lane).definitions ->
        Definition.Holds assignment definition := by
    intro lane definition member
    rcases List.mem_iff_getElem.mp member with
      ⟨definitionIndex, definitionLt, definitionEq⟩
    have definitionLtFive : definitionIndex < 5 := by
      simpa [KMulTrace.definitions] using definitionLt
    let typedIndex : Fin 5 := ⟨definitionIndex, definitionLtFive⟩
    have holds := honestCanonicalAssignment_definition_holds
      pending finalWitnesses
      (productionPhysicalIndex (.scale lane typedIndex)).val
      (bridge.scaleIndexInProgram lane typedIndex)
    rw [bridge.scaleDefinition lane typedIndex] at holds
    simpa [typedIndex, ← definitionEq] using holds
  have rows :=
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.rows_complete
      productionShape
    (honestCanonicalAssignment_lt pending finalWitnesses)
    (canonicalAssignment_constantOne pending finalWitnesses
      (honestInternalWitness pending finalWitnesses))
    tensorHolds coordinateHolds scaleHolds
    (semanticTerminalParentEquality_terminalHolds pending finalWitnesses
      terminalEquality)
  exact productionArtifactContract.artifactRowsSatisfied_of_rowsSatisfied rows

/-- Honest completeness of the optimized artifact rows.  The only semantic
premise is the parent-versus-final-scale equality; all tensor, coordinate,
and scale equations are constructed by the list-free SSA interpreter, and
their typed compiler indices are discharged by
`productionTypedDefinitionBridge`. -/
theorem honestArtifactRowsSatisfied
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses : HonestMatrices)
    (terminalEquality :
      SemanticTerminalParentEquality pending finalWitnesses) :
    ArtifactRowsSatisfied productionArtifactContract
      (honestCanonicalAssignment pending finalWitnesses) :=
  honestArtifactRowsSatisfied_of_typedDefinitionBridge
    productionTypedDefinitionBridge pending finalWitnesses terminalEquality

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
