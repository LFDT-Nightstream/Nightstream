import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Layout

/-!
Kernel derivation of the physical-emitter column inverse.

Rust constructs `EmitterLayout` through
`RawOldBlockProjectionColumnMap::new`, whose exact shape, uniqueness, and
interval checks are exported as `emitterColumnMapValid`.  This leaf proves
that the generated Rust-order inverse recovers every canonical program
column.  The inverse is therefore not a caller premise or an implementation
failure event.

Owns: the kernel derivation that a checked emitter layout has disjoint typed
column intervals and that its generated inverse recovers every canonical
column read or written by the projection program.

Does not own: the fixed emitter values, physical row placement, assignment
values, row satisfaction, semantic projection equality, or commitment
authority.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.execution.columns.intervals` | old-block, parent, child, tensor, product, and final-scale column regions are in range and pairwise compatible | derived from checked layout |
| `f_prime.pi_ccs_nc.delayed.execution.columns.inverse` | generated physical-to-canonical lookup recovers every program-owned canonical column | derived |
| `f_prime.pi_ccs_nc.delayed.execution.columns.pull` | pulling a physical assignment through the emitter map agrees on every owned source | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private abbrev planBlockVariables : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockVariables

private abbrev planActiveLanes : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.activeLanes

private abbrev planChildCount : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.childCount

private def kScalars (columns : List KColumns) : List Nat :=
  columns.flatMap fun current => [current.c0, current.c1]

private def IntervalsDisjointProp
    (left right : ColumnInterval) : Prop :=
  left.stop <= right.start ∨ right.stop <= left.start

private def IntervalContainsProp
    (interval : ColumnInterval) (column : Nat) : Prop :=
  interval.start <= column ∧ column < interval.stop

private theorem allDistinct_eq_true_iff (values : List Nat) :
    allDistinct values = true <-> values.Nodup := by
  induction values with
  | nil => simp [allDistinct]
  | cons head tail inductionHypothesis =>
      simp [allDistinct, inductionHypothesis]

private theorem intervalsDisjoint_eq_true_iff
    (left right : ColumnInterval) :
    intervalsDisjoint left right = true <->
      IntervalsDisjointProp left right := by
  simp [intervalsDisjoint, IntervalsDisjointProp]

private theorem intervalsPairwiseDisjoint_eq_true_iff
    (intervals : List ColumnInterval) :
    intervalsPairwiseDisjoint intervals = true <->
      intervals.Pairwise IntervalsDisjointProp := by
  induction intervals with
  | nil => simp [intervalsPairwiseDisjoint]
  | cons head tail inductionHypothesis =>
      simp [intervalsPairwiseDisjoint, inductionHypothesis,
        intervalsDisjoint_eq_true_iff]

private theorem scalarsOutsideIntervals_eq_true_iff
    (emitter : EmitterLayout) :
    scalarsOutsideIntervals emitter = true <->
      forall scalar, scalar ∈ emitterScalarColumns emitter ->
        forall interval, interval ∈ emitterIntervals emitter ->
          ¬ IntervalContainsProp interval scalar := by
  simp [scalarsOutsideIntervals, intervalContains,
    IntervalContainsProp]

private theorem findKColumnAux_eq_none_of_not_mem
    (columns : List KColumns) (start column : Nat)
    (notMember : column ∉ kScalars columns) :
    findKColumnAux columns start column = none := by
  induction columns generalizing start with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have notC0 : column ≠ head.c0 := by
        intro equal
        apply notMember
        simp [kScalars, equal]
      have notC1 : column ≠ head.c1 := by
        intro equal
        apply notMember
        simp [kScalars, equal]
      have notTail : column ∉ kScalars tail := by
        intro member
        apply notMember
        unfold kScalars
        exact List.mem_append_right _ member
      simp [findKColumnAux, notC0, notC1,
        inductionHypothesis (start := start + 1) notTail]

private theorem kScalar_c0_mem (columns : List KColumns)
    (index : Fin columns.length) :
    (columns.get index).c0 ∈ kScalars columns := by
  unfold kScalars
  exact List.mem_flatMap.mpr
    ⟨columns.get index, List.get_mem columns index, by simp⟩

private theorem kScalar_c1_mem (columns : List KColumns)
    (index : Fin columns.length) :
    (columns.get index).c1 ∈ kScalars columns := by
  unfold kScalars
  exact List.mem_flatMap.mpr
    ⟨columns.get index, List.get_mem columns index, by simp⟩

private theorem findKColumnAux_c0
    (columns : List KColumns) (start : Nat)
    (distinct : (kScalars columns).Nodup)
    (index : Fin columns.length) :
    findKColumnAux columns start (columns.get index).c0 =
      some (start + index.val, 0) := by
  induction columns generalizing start with
  | nil => exact Fin.elim0 index
  | cons head tail inductionHypothesis =>
      have expanded :
          (head.c0 :: head.c1 :: kScalars tail).Nodup := by
        simpa [kScalars] using distinct
      have headC0Not : head.c0 ∉ head.c1 :: kScalars tail :=
        (List.nodup_cons.mp expanded).1
      have restDistinct : (head.c1 :: kScalars tail).Nodup :=
        (List.nodup_cons.mp expanded).2
      have headC1Not : head.c1 ∉ kScalars tail :=
        (List.nodup_cons.mp restDistinct).1
      have tailDistinct : (kScalars tail).Nodup :=
        (List.nodup_cons.mp restDistinct).2
      refine Fin.cases ?_ (fun next => ?_) index
      · simp [findKColumnAux]
      · have targetMember := kScalar_c0_mem tail next
        have notHeadC0 : (tail.get next).c0 ≠ head.c0 := by
          intro equal
          have member : head.c0 ∈ kScalars tail := by
            rw [← equal]
            exact targetMember
          exact headC0Not (List.mem_cons_of_mem _ member)
        have notHeadC1 : (tail.get next).c0 ≠ head.c1 := by
          intro equal
          have member : head.c1 ∈ kScalars tail := by
            rw [← equal]
            exact targetMember
          exact headC1Not member
        rw [show (head :: tail).get next.succ = tail.get next by rfl]
        simp only [findKColumnAux]
        rw [if_neg notHeadC0, if_neg notHeadC1,
          inductionHypothesis (start := start + 1) tailDistinct next]
        simp [Nat.add_comm, Nat.add_left_comm]

private theorem findKColumnAux_c1
    (columns : List KColumns) (start : Nat)
    (distinct : (kScalars columns).Nodup)
    (index : Fin columns.length) :
    findKColumnAux columns start (columns.get index).c1 =
      some (start + index.val, 1) := by
  induction columns generalizing start with
  | nil => exact Fin.elim0 index
  | cons head tail inductionHypothesis =>
      have expanded :
          (head.c0 :: head.c1 :: kScalars tail).Nodup := by
        simpa [kScalars] using distinct
      have headC0Not : head.c0 ∉ head.c1 :: kScalars tail :=
        (List.nodup_cons.mp expanded).1
      have restDistinct : (head.c1 :: kScalars tail).Nodup :=
        (List.nodup_cons.mp expanded).2
      have headC1Not : head.c1 ∉ kScalars tail :=
        (List.nodup_cons.mp restDistinct).1
      have tailDistinct : (kScalars tail).Nodup :=
        (List.nodup_cons.mp restDistinct).2
      refine Fin.cases ?_ (fun next => ?_) index
      · have c1NeC0 : head.c1 ≠ head.c0 := by
          intro equal
          exact headC0Not (by simp [equal])
        simp [findKColumnAux, c1NeC0]
      · have targetMember := kScalar_c1_mem tail next
        have notHeadC0 : (tail.get next).c1 ≠ head.c0 := by
          intro equal
          have member : head.c0 ∈ kScalars tail := by
            rw [← equal]
            exact targetMember
          exact headC0Not (List.mem_cons_of_mem _ member)
        have notHeadC1 : (tail.get next).c1 ≠ head.c1 := by
          intro equal
          have member : head.c1 ∈ kScalars tail := by
            rw [← equal]
            exact targetMember
          exact headC1Not member
        rw [show (head :: tail).get next.succ = tail.get next by rfl]
        simp only [findKColumnAux]
        rw [if_neg notHeadC0, if_neg notHeadC1,
          inductionHypothesis (start := start + 1) tailDistinct next]
        simp [Nat.add_comm, Nat.add_left_comm]

private theorem findKColumnAux_selected
    (columns : List KColumns) (start : Nat)
    (distinct : (kScalars columns).Nodup)
    (index : Fin columns.length) (limb : Fin 2) :
    findKColumnAux columns start
        (selectLimb (columns.get index) limb.val) =
      some (start + index.val, limb.val) := by
  refine Fin.cases ?_ (fun next => ?_) limb
  · simpa [selectLimb] using findKColumnAux_c0 columns start distinct index
  · exact Fin.cases
      (by simpa [selectLimb] using
        findKColumnAux_c1 columns start distinct index)
      (fun impossible => Fin.elim0 impossible)
      next

private structure ValidFacts (emitter : EmitterLayout) : Prop where
  shape : emitterShapePinned emitter = true
  scalars : allDistinct (emitterScalarColumns emitter) = true
  intervals : intervalsPairwiseDisjoint (emitterIntervals emitter) = true
  outside : scalarsOutsideIntervals emitter = true

private theorem EmitterLayoutValid.facts
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    ValidFacts emitter := by
  have checked := valid.checked
  simp only [emitterColumnMapValid, Bool.and_eq_true] at checked
  exact ⟨checked.1.1.1, checked.1.1.2, checked.1.2, checked.2⟩

private theorem EmitterLayoutValid.oldBlock_length
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    emitter.oldBlock.length = planBlockVariables := by
  have shape := valid.facts.shape
  simp only [emitterShapePinned, Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.1.1.1.1.2

private theorem EmitterLayoutValid.parent_length
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    emitter.parent.length = planActiveLanes := by
  have shape := valid.facts.shape
  simp only [emitterShapePinned, Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.1.1.1.2

private theorem EmitterLayoutValid.finalWitnessFirst_length
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    emitter.finalWitnessFirst.length = planChildCount := by
  have shape := valid.facts.shape
  simp only [emitterShapePinned, Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.1.1.2

private theorem EmitterLayoutValid.productFirst_eq
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    emitter.productFirst = emitter.tensorFirst + tensorRows := by
  have shape := valid.facts.shape
  simp only [emitterShapePinned, Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.1.2

private theorem EmitterLayoutValid.finalScaleFirst_eq
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    emitter.finalScaleFirst = emitter.productFirst + productRows := by
  have shape := valid.facts.shape
  simp only [emitterShapePinned, Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.2

private theorem EmitterLayoutValid.scalarColumns_nodup
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    (emitterScalarColumns emitter).Nodup :=
  (allDistinct_eq_true_iff _).mp valid.facts.scalars

private theorem EmitterLayoutValid.intervals_pairwise
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    (emitterIntervals emitter).Pairwise IntervalsDisjointProp :=
  (intervalsPairwiseDisjoint_eq_true_iff _).mp valid.facts.intervals

private theorem EmitterLayoutValid.scalars_outside
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    forall scalar, scalar ∈ emitterScalarColumns emitter ->
      forall interval, interval ∈ emitterIntervals emitter ->
        ¬ IntervalContainsProp interval scalar :=
  (scalarsOutsideIntervals_eq_true_iff _).mp valid.facts.outside

private def witnessInterval (first : Nat) : ColumnInterval :=
  { start := first, stop := first + witnessEntriesPerChild }

private theorem findWitnessIntervalAux_selected
    (firsts : List Nat) (start : Nat)
    (pairwise : (firsts.map witnessInterval).Pairwise IntervalsDisjointProp)
    (index : Fin firsts.length) (offset : Nat)
    (offsetInRange : offset < witnessEntriesPerChild) :
    findWitnessIntervalAux firsts start (firsts.get index + offset) =
      some (start + index.val, offset) := by
  induction firsts generalizing start with
  | nil => exact Fin.elim0 index
  | cons first tail inductionHypothesis =>
      have expanded :
          (witnessInterval first :: tail.map witnessInterval).Pairwise
            IntervalsDisjointProp := by
        simpa [witnessInterval] using pairwise
      have headDisjoint := (List.pairwise_cons.mp expanded).1
      have tailPairwise := (List.pairwise_cons.mp expanded).2
      refine Fin.cases ?_ (fun next => ?_) index
      · change findWitnessIntervalAux (first :: tail) start
          (first + offset) = some (start, offset)
        simp only [findWitnessIntervalAux]
        rw [if_pos (by omega)]
        simp
      · rw [show (first :: tail).get next.succ = tail.get next by rfl]
        have intervalMember : witnessInterval (tail.get next) ∈
            tail.map witnessInterval :=
          List.mem_map.mpr ⟨tail.get next, List.get_mem tail next, rfl⟩
        have disjoint := headDisjoint _ intervalMember
        have notInHead : ¬
            (first ≤ tail.get next + offset ∧
              tail.get next + offset < first + witnessEntriesPerChild) := by
          intro inHead
          rcases disjoint with before | after
          · simp only [witnessInterval] at before
            omega
          · simp only [witnessInterval] at after
            omega
        simp only [findWitnessIntervalAux]
        rw [if_neg notInHead,
          inductionHypothesis (start := start + 1) tailPairwise next]
        simp [Nat.add_comm, Nat.add_left_comm]

private structure ScalarLayoutFacts (emitter : EmitterLayout) : Prop where
  zeroNot : constantOneColumn ∉
    kScalars emitter.oldBlock ++ kScalars emitter.parent
  oldNodup : (kScalars emitter.oldBlock).Nodup
  parentNodup : (kScalars emitter.parent).Nodup
  oldParentDistinct : forall oldColumn,
    oldColumn ∈ kScalars emitter.oldBlock ->
      forall parentColumn, parentColumn ∈ kScalars emitter.parent ->
        oldColumn ≠ parentColumn

private theorem EmitterLayoutValid.scalarLayoutFacts
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    ScalarLayoutFacts emitter := by
  have normalized :
      (constantOneColumn ::
        (kScalars emitter.oldBlock ++ kScalars emitter.parent)).Nodup := by
    simpa [emitterScalarColumns, kScalars, List.append_assoc] using
      valid.scalarColumns_nodup
  have zeroNot := (List.nodup_cons.mp normalized).1
  have remaining := (List.nodup_cons.mp normalized).2
  have split := (List.nodup_append.mp remaining)
  exact ⟨zeroNot, split.1, split.2.1, split.2.2⟩

private def tensorInterval (emitter : EmitterLayout) : ColumnInterval :=
  { start := emitter.tensorFirst, stop := emitter.productFirst }

private def productInterval (emitter : EmitterLayout) : ColumnInterval :=
  { start := emitter.productFirst,
    stop := emitter.finalScaleFirst }

private def finalScaleInterval (emitter : EmitterLayout) : ColumnInterval :=
  { start := emitter.finalScaleFirst,
    stop := emitter.finalScaleFirst + finalScaleRows }

private theorem tensorInterval_mem (emitter : EmitterLayout) :
    tensorInterval emitter ∈ emitterIntervals emitter := by
  simp [emitterIntervals, tensorInterval]

private theorem productInterval_mem (emitter : EmitterLayout) :
    productInterval emitter ∈ emitterIntervals emitter := by
  simp [emitterIntervals, productInterval]

private theorem finalScaleInterval_mem (emitter : EmitterLayout) :
    finalScaleInterval emitter ∈ emitterIntervals emitter := by
  simp [emitterIntervals, finalScaleInterval]

private theorem childInterval_mem (emitter : EmitterLayout)
    (index : Fin emitter.finalWitnessFirst.length) :
    witnessInterval (emitter.finalWitnessFirst.get index) ∈
      childIntervals emitter := by
  simp only [childIntervals, witnessInterval]
  exact List.mem_map.mpr
    ⟨emitter.finalWitnessFirst.get index,
      List.get_mem emitter.finalWitnessFirst index, rfl⟩

private theorem EmitterLayoutValid.childIntervals_pairwise
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter) :
    (emitter.finalWitnessFirst.map witnessInterval).Pairwise
      IntervalsDisjointProp := by
  have full := valid.intervals_pairwise
  have childPrefix :
      (childIntervals emitter).Pairwise IntervalsDisjointProp := by
    simpa [emitterIntervals] using (List.pairwise_append.mp full).1
  simpa [childIntervals, witnessInterval] using childPrefix

private theorem EmitterLayoutValid.childInterval_disjoint_tensor
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.finalWitnessFirst.length) :
    IntervalsDisjointProp
      (witnessInterval (emitter.finalWitnessFirst.get index))
      (tensorInterval emitter) := by
  have full := valid.intervals_pairwise
  have cross := (List.pairwise_append.mp full).2.2
  exact cross _ (childInterval_mem emitter index) _ (by
    simp [tensorInterval])

private theorem EmitterLayoutValid.childInterval_disjoint_product
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.finalWitnessFirst.length) :
    IntervalsDisjointProp
      (witnessInterval (emitter.finalWitnessFirst.get index))
      (productInterval emitter) := by
  have full := valid.intervals_pairwise
  have cross := (List.pairwise_append.mp full).2.2
  exact cross _ (childInterval_mem emitter index) _ (by
    simp [productInterval])

private theorem EmitterLayoutValid.childInterval_disjoint_finalScale
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.finalWitnessFirst.length) :
    IntervalsDisjointProp
      (witnessInterval (emitter.finalWitnessFirst.get index))
      (finalScaleInterval emitter) := by
  have full := valid.intervals_pairwise
  have cross := (List.pairwise_append.mp full).2.2
  exact cross _ (childInterval_mem emitter index) _ (by
    simp [finalScaleInterval])

private theorem intervalsDisjoint_excludes_right
    {left right : ColumnInterval} {column : Nat}
    (disjoint : IntervalsDisjointProp left right)
    (inLeft : IntervalContainsProp left column) :
    ¬ IntervalContainsProp right column := by
  intro inRight
  rcases disjoint with before | after <;>
    rcases inLeft with ⟨leftStart, leftStop⟩ <;>
    rcases inRight with ⟨rightStart, rightStop⟩ <;>
    omega

private theorem getD_eq_get {alpha : Type} (values : List alpha)
    (index : Fin values.length) (fallback : alpha) :
    values.getD index.val fallback = values.get index := by
  rw [List.get_eq_getElem]
  exact (List.getElem_eq_getD fallback).symm

private theorem selectedKScalar_mem
    (columns : List KColumns) (index : Fin columns.length)
    (limb : Fin 2) :
    selectLimb (columns.get index) limb.val ∈ kScalars columns := by
  refine Fin.cases ?_ (fun next => ?_) limb
  · simpa [selectLimb] using kScalar_c0_mem columns index
  · exact Fin.cases
      (by simpa [selectLimb] using kScalar_c1_mem columns index)
      (fun impossible => Fin.elim0 impossible)
      next

private theorem EmitterLayoutValid.oldSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.oldBlock.length) (limb : Fin 2) :
    emitterColumnInverse emitter
        (selectLimb (emitter.oldBlock.get index) limb.val) =
      some (oldBlockFirstColumn + 2 * index.val + limb.val) := by
  let target := selectLimb (emitter.oldBlock.get index) limb.val
  have targetOld : target ∈ kScalars emitter.oldBlock :=
    selectedKScalar_mem emitter.oldBlock index limb
  have targetScalar : target ∈ emitterScalarColumns emitter := by
    simp only [emitterScalarColumns, List.mem_append, List.mem_singleton]
    exact Or.inl (Or.inr targetOld)
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    apply valid.scalarLayoutFacts.zeroNot
    apply List.mem_append_left
    rw [← equal]
    exact targetOld
  have outsideTensor : ¬
      (emitter.tensorFirst ≤ target ∧ target < emitter.productFirst) := by
    simpa [tensorInterval, IntervalContainsProp] using
      valid.scalars_outside target targetScalar
        (tensorInterval emitter) (tensorInterval_mem emitter)
  have outsideProduct : ¬
      (emitter.productFirst ≤ target ∧
        target < emitter.productFirst + productRows) := by
    simpa [productInterval, IntervalContainsProp,
      valid.finalScaleFirst_eq] using
      valid.scalars_outside target targetScalar
        (productInterval emitter) (productInterval_mem emitter)
  have outsideFinalScale : ¬
      (emitter.finalScaleFirst ≤ target ∧
        target < emitter.finalScaleFirst + finalScaleRows) := by
    simpa [finalScaleInterval, IntervalContainsProp] using
      valid.scalars_outside target targetScalar
        (finalScaleInterval emitter) (finalScaleInterval_mem emitter)
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_neg outsideTensor, if_neg outsideProduct,
    if_neg outsideFinalScale,
    findKColumnAux_selected emitter.oldBlock 0
      valid.scalarLayoutFacts.oldNodup index limb]
  simp

private theorem EmitterLayoutValid.parentSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.parent.length) (limb : Fin 2) :
    emitterColumnInverse emitter
        (selectLimb (emitter.parent.get index) limb.val) =
      some (parentFirstColumn + 2 * index.val + limb.val) := by
  let target := selectLimb (emitter.parent.get index) limb.val
  have targetParent : target ∈ kScalars emitter.parent :=
    selectedKScalar_mem emitter.parent index limb
  have targetScalar : target ∈ emitterScalarColumns emitter := by
    simp only [emitterScalarColumns, List.mem_append, List.mem_singleton]
    exact Or.inr targetParent
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    apply valid.scalarLayoutFacts.zeroNot
    apply List.mem_append_right
    rw [← equal]
    exact targetParent
  have targetNotOld : target ∉ kScalars emitter.oldBlock := by
    intro targetOld
    exact valid.scalarLayoutFacts.oldParentDistinct target targetOld
      target targetParent rfl
  have outsideTensor : ¬
      (emitter.tensorFirst ≤ target ∧ target < emitter.productFirst) := by
    simpa [tensorInterval, IntervalContainsProp] using
      valid.scalars_outside target targetScalar
        (tensorInterval emitter) (tensorInterval_mem emitter)
  have outsideProduct : ¬
      (emitter.productFirst ≤ target ∧
        target < emitter.productFirst + productRows) := by
    simpa [productInterval, IntervalContainsProp,
      valid.finalScaleFirst_eq] using
      valid.scalars_outside target targetScalar
        (productInterval emitter) (productInterval_mem emitter)
  have outsideFinalScale : ¬
      (emitter.finalScaleFirst ≤ target ∧
        target < emitter.finalScaleFirst + finalScaleRows) := by
    simpa [finalScaleInterval, IntervalContainsProp] using
      valid.scalars_outside target targetScalar
        (finalScaleInterval emitter) (finalScaleInterval_mem emitter)
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_neg outsideTensor, if_neg outsideProduct,
    if_neg outsideFinalScale,
    findKColumnAux_eq_none_of_not_mem emitter.oldBlock 0 target
      targetNotOld,
    findKColumnAux_selected emitter.parent 0
      valid.scalarLayoutFacts.parentNodup index limb]
  simp

private theorem constantOne_mem_scalarColumns (emitter : EmitterLayout) :
    constantOneColumn ∈ emitterScalarColumns emitter := by
  simp [emitterScalarColumns]

private theorem EmitterLayoutValid.witnessSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (index : Fin emitter.finalWitnessFirst.length) (offset : Nat)
    (offsetInRange : offset < witnessEntriesPerChild) :
    emitterColumnInverse emitter
        (emitter.finalWitnessFirst.get index + offset) =
      some (witnessFamilyFirstColumn +
        index.val * witnessEntriesPerChild + offset) := by
  let target := emitter.finalWitnessFirst.get index + offset
  let selectedInterval :=
    witnessInterval (emitter.finalWitnessFirst.get index)
  have inSelected : IntervalContainsProp selectedInterval target := by
    simp only [selectedInterval, witnessInterval, IntervalContainsProp,
      target]
    omega
  have selectedMem : selectedInterval ∈ emitterIntervals emitter := by
    apply List.mem_append_left
    simpa [selectedInterval] using childInterval_mem emitter index
  have targetNotScalar : target ∉ emitterScalarColumns emitter := by
    intro targetScalar
    exact valid.scalars_outside target targetScalar selectedInterval
      selectedMem inSelected
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    apply targetNotScalar
    rw [equal]
    exact constantOne_mem_scalarColumns emitter
  have targetNotOld : target ∉ kScalars emitter.oldBlock := by
    intro targetOld
    apply targetNotScalar
    simp only [emitterScalarColumns, List.mem_append, List.mem_singleton]
    exact Or.inl (Or.inr targetOld)
  have targetNotParent : target ∉ kScalars emitter.parent := by
    intro targetParent
    apply targetNotScalar
    simp only [emitterScalarColumns, List.mem_append, List.mem_singleton]
    exact Or.inr targetParent
  have outsideTensor : ¬
      (emitter.tensorFirst ≤ target ∧ target < emitter.productFirst) := by
    simpa [selectedInterval, tensorInterval, IntervalContainsProp] using
      intervalsDisjoint_excludes_right
        (valid.childInterval_disjoint_tensor index) inSelected
  have outsideProduct : ¬
      (emitter.productFirst ≤ target ∧
        target < emitter.productFirst + productRows) := by
    simpa [selectedInterval, productInterval, IntervalContainsProp,
      valid.finalScaleFirst_eq] using
      intervalsDisjoint_excludes_right
        (valid.childInterval_disjoint_product index) inSelected
  have outsideFinalScale : ¬
      (emitter.finalScaleFirst ≤ target ∧
        target < emitter.finalScaleFirst + finalScaleRows) := by
    simpa [selectedInterval, finalScaleInterval,
      IntervalContainsProp] using
      intervalsDisjoint_excludes_right
        (valid.childInterval_disjoint_finalScale index) inSelected
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_neg outsideTensor, if_neg outsideProduct,
    if_neg outsideFinalScale,
    findKColumnAux_eq_none_of_not_mem emitter.oldBlock 0 target
      targetNotOld,
    findKColumnAux_eq_none_of_not_mem emitter.parent 0 target
      targetNotParent,
    findWitnessIntervalAux_selected emitter.finalWitnessFirst 0
      valid.childIntervals_pairwise index offset offsetInRange]
  simp

private theorem EmitterLayoutValid.tensorSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (offset : Nat) (offsetInRange : offset < tensorRows) :
    emitterColumnInverse emitter (emitter.tensorFirst + offset) =
      some (tensorFirstColumn + offset) := by
  let target := emitter.tensorFirst + offset
  have inTensor : emitter.tensorFirst ≤ target ∧
      target < emitter.productFirst := by
    rw [valid.productFirst_eq]
    simp only [target]
    omega
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    have outside := valid.scalars_outside constantOneColumn
      (constantOne_mem_scalarColumns emitter) (tensorInterval emitter)
      (tensorInterval_mem emitter)
    apply outside
    simpa [tensorInterval, IntervalContainsProp, equal] using inTensor
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_pos inTensor]
  simp

private theorem EmitterLayoutValid.productSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (offset : Nat) (offsetInRange : offset < productRows) :
    emitterColumnInverse emitter (emitter.productFirst + offset) =
      some (productFirstColumn + offset) := by
  let target := emitter.productFirst + offset
  have outsideTensor : ¬
      (emitter.tensorFirst ≤ target ∧ target < emitter.productFirst) := by
    simp only [target]
    omega
  have inProduct : emitter.productFirst ≤ target ∧
      target < emitter.productFirst + productRows := by
    simp only [target]
    omega
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    have outside := valid.scalars_outside constantOneColumn
      (constantOne_mem_scalarColumns emitter) (productInterval emitter)
      (productInterval_mem emitter)
    apply outside
    simpa [productInterval, IntervalContainsProp, equal,
      valid.finalScaleFirst_eq] using inProduct
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_neg outsideTensor, if_pos inProduct]
  simp

private theorem EmitterLayoutValid.finalScaleSelected_inverse
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (offset : Nat) (offsetInRange : offset < finalScaleRows) :
    emitterColumnInverse emitter (emitter.finalScaleFirst + offset) =
      some (finalScaleFirstColumn + offset) := by
  let target := emitter.finalScaleFirst + offset
  have outsideTensor : ¬
      (emitter.tensorFirst ≤ target ∧ target < emitter.productFirst) := by
    simp only [target]
    rw [valid.finalScaleFirst_eq]
    omega
  have outsideProduct : ¬
      (emitter.productFirst ≤ target ∧
        target < emitter.productFirst + productRows) := by
    simp only [target]
    rw [valid.finalScaleFirst_eq]
    omega
  have inFinalScale : emitter.finalScaleFirst ≤ target ∧
      target < emitter.finalScaleFirst + finalScaleRows := by
    simp only [target]
    omega
  have targetNeZero : target ≠ constantOneColumn := by
    intro equal
    have outside := valid.scalars_outside constantOneColumn
      (constantOne_mem_scalarColumns emitter) (finalScaleInterval emitter)
      (finalScaleInterval_mem emitter)
    apply outside
    simpa [finalScaleInterval, IntervalContainsProp, equal] using
      inFinalScale
  simp only [emitterColumnInverse]
  rw [if_neg targetNeZero, if_neg outsideTensor, if_neg outsideProduct,
    if_pos inFinalScale]
  simp

private theorem planBlockVariables_eq : planBlockVariables = 19 := by
  rfl

private theorem planActiveLanes_eq : planActiveLanes = 54 := by
  rfl

private theorem planChildCount_eq : planChildCount = 14 := by
  rfl

private theorem witnessEntriesPerChild_eq :
    witnessEntriesPerChild = 11437038 := by
  rfl

private theorem tensorRows_eq : tensorRows = 1310715 := by
  rfl

private theorem productRows_eq : productRows = 22874076 := by
  rfl

private theorem finalScaleRows_eq : finalScaleRows = 270 := by
  rfl

/-- The Rust-order inverse recovers every canonical program column.

This theorem is derived solely from the generated Boolean layout certificate.
It is not a caller-supplied refinement premise. -/
theorem EmitterLayoutValid.columnRoundTrip
    {emitter : EmitterLayout} (valid : EmitterLayoutValid emitter)
    (column : Nat) (columnInRange : column < canonicalColumnCount) :
    emitterColumnInverse emitter (emitterColumnMap emitter column) =
      some column := by
  by_cases isConstant : column = constantOneColumn
  · subst column
    simp [emitterColumnMap, emitterColumnInverse, constantOneColumn]
  by_cases inOldBlock : column < parentFirstColumn
  · have lower : oldBlockFirstColumn ≤ column := by
      simp only [constantOneColumn, oldBlockFirstColumn] at isConstant ⊢
      omega
    let offset := column - oldBlockFirstColumn
    have offsetInRange : offset < 2 * planBlockVariables := by
      rw [planBlockVariables_eq]
      simp only [offset, oldBlockFirstColumn, parentFirstColumn] at *
      omega
    have offsetNumeric : offset < 38 := by
      simpa [planBlockVariables_eq] using offsetInRange
    have indexInRange : offset / 2 < emitter.oldBlock.length := by
      rw [valid.oldBlock_length, planBlockVariables_eq]
      omega
    let index : Fin emitter.oldBlock.length :=
      ⟨offset / 2, indexInRange⟩
    let limb : Fin 2 :=
      ⟨offset % 2, Nat.mod_lt _ (by decide)⟩
    have mapped : emitterColumnMap emitter column =
        selectLimb (emitter.oldBlock.get index) limb.val := by
      simp only [emitterColumnMap, if_neg isConstant, if_pos inOldBlock]
      change selectLimb (emitter.oldBlock.getD index.val default) limb.val =
        selectLimb (emitter.oldBlock.get index) limb.val
      rw [getD_eq_get]
    rw [mapped, valid.oldSelected_inverse index limb]
    have modDiv := Nat.mod_add_div offset 2
    have columnReconstruct : oldBlockFirstColumn + offset = column := by
      simp only [offset]
      omega
    have canonicalReconstruct :
        oldBlockFirstColumn + 2 * index.val + limb.val = column := by
      simp only [index, limb]
      omega
    exact congrArg some canonicalReconstruct
  by_cases inParent : column < witnessFamilyFirstColumn
  · have lower : parentFirstColumn ≤ column := by omega
    let offset := column - parentFirstColumn
    have offsetInRange : offset < 2 * planActiveLanes := by
      rw [planActiveLanes_eq]
      simp only [offset, parentFirstColumn, witnessFamilyFirstColumn] at *
      omega
    have offsetNumeric : offset < 108 := by
      simpa [planActiveLanes_eq] using offsetInRange
    have indexInRange : offset / 2 < emitter.parent.length := by
      rw [valid.parent_length, planActiveLanes_eq]
      omega
    let index : Fin emitter.parent.length :=
      ⟨offset / 2, indexInRange⟩
    let limb : Fin 2 :=
      ⟨offset % 2, Nat.mod_lt _ (by decide)⟩
    have mapped : emitterColumnMap emitter column =
        selectLimb (emitter.parent.get index) limb.val := by
      simp only [emitterColumnMap, if_neg isConstant, if_neg inOldBlock,
        if_pos inParent]
      change selectLimb (emitter.parent.getD index.val default) limb.val =
        selectLimb (emitter.parent.get index) limb.val
      rw [getD_eq_get]
    rw [mapped, valid.parentSelected_inverse index limb]
    have modDiv := Nat.mod_add_div offset 2
    have columnReconstruct : parentFirstColumn + offset = column := by
      simp only [offset]
      omega
    have canonicalReconstruct :
        parentFirstColumn + 2 * index.val + limb.val = column := by
      simp only [index, limb]
      omega
    exact congrArg some canonicalReconstruct
  by_cases inWitness : column < tensorFirstColumn
  · have lower : witnessFamilyFirstColumn ≤ column := by omega
    let offset := column - witnessFamilyFirstColumn
    have offsetInRange : offset <
        planChildCount * witnessEntriesPerChild := by
      rw [planChildCount_eq, witnessEntriesPerChild_eq]
      simp only [offset, witnessFamilyFirstColumn, tensorFirstColumn] at *
      omega
    have offsetNumeric : offset < 14 * 11437038 := by
      simpa [planChildCount_eq, witnessEntriesPerChild_eq] using offsetInRange
    have entriesPositive : 0 < witnessEntriesPerChild := by
      rw [witnessEntriesPerChild_eq]
      decide
    have indexInRange : offset / witnessEntriesPerChild <
        emitter.finalWitnessFirst.length := by
      rw [valid.finalWitnessFirst_length, planChildCount_eq,
        witnessEntriesPerChild_eq]
      omega
    let index : Fin emitter.finalWitnessFirst.length :=
      ⟨offset / witnessEntriesPerChild, indexInRange⟩
    let within := offset % witnessEntriesPerChild
    have withinInRange : within < witnessEntriesPerChild :=
      Nat.mod_lt _ entriesPositive
    have mapped : emitterColumnMap emitter column =
        emitter.finalWitnessFirst.get index + within := by
      simp only [emitterColumnMap, if_neg isConstant, if_neg inOldBlock,
        if_neg inParent, if_pos inWitness]
      change emitter.finalWitnessFirst.getD index.val 0 + within =
        emitter.finalWitnessFirst.get index + within
      rw [getD_eq_get]
    rw [mapped, valid.witnessSelected_inverse index within withinInRange]
    have modDiv := Nat.mod_add_div offset witnessEntriesPerChild
    have columnReconstruct :
        witnessFamilyFirstColumn + offset = column := by
      simp only [offset]
      omega
    have canonicalReconstruct :
        witnessFamilyFirstColumn +
          index.val * witnessEntriesPerChild + within = column := by
      simp only [index, within]
      rw [Nat.mul_comm (offset / witnessEntriesPerChild)
        witnessEntriesPerChild]
      omega
    exact congrArg some canonicalReconstruct
  by_cases inTensor : column < productFirstColumn
  · have lower : tensorFirstColumn ≤ column := by omega
    let offset := column - tensorFirstColumn
    have offsetInRange : offset < tensorRows := by
      rw [tensorRows_eq]
      simp only [offset, tensorFirstColumn, productFirstColumn] at *
      omega
    have mapped : emitterColumnMap emitter column =
        emitter.tensorFirst + offset := by
      simp only [emitterColumnMap, if_neg isConstant, if_neg inOldBlock,
        if_neg inParent, if_neg inWitness, if_pos inTensor]
      change emitter.tensorFirst + column - tensorFirstColumn =
        emitter.tensorFirst + offset
      rw [Nat.add_sub_assoc lower]
    rw [mapped, valid.tensorSelected_inverse offset offsetInRange]
    have canonicalReconstruct : tensorFirstColumn + offset = column := by
      simp only [offset]
      omega
    exact congrArg some canonicalReconstruct
  by_cases inProduct : column < finalScaleFirstColumn
  · have lower : productFirstColumn ≤ column := by omega
    let offset := column - productFirstColumn
    have offsetInRange : offset < productRows := by
      rw [productRows_eq]
      simp only [offset, productFirstColumn, finalScaleFirstColumn] at *
      omega
    have mapped : emitterColumnMap emitter column =
        emitter.productFirst + offset := by
      simp only [emitterColumnMap, if_neg isConstant, if_neg inOldBlock,
        if_neg inParent, if_neg inWitness, if_neg inTensor,
        if_pos inProduct]
      change emitter.productFirst + column - productFirstColumn =
        emitter.productFirst + offset
      rw [Nat.add_sub_assoc lower]
    rw [mapped, valid.productSelected_inverse offset offsetInRange]
    have canonicalReconstruct : productFirstColumn + offset = column := by
      simp only [offset]
      omega
    exact congrArg some canonicalReconstruct
  · have lower : finalScaleFirstColumn ≤ column := by omega
    let offset := column - finalScaleFirstColumn
    have offsetInRange : offset < finalScaleRows := by
      rw [finalScaleRows_eq]
      simp only [offset, finalScaleFirstColumn, canonicalColumnCount] at *
      omega
    have mapped : emitterColumnMap emitter column =
        emitter.finalScaleFirst + offset := by
      simp only [emitterColumnMap, if_neg isConstant, if_neg inOldBlock,
        if_neg inParent, if_neg inWitness, if_neg inTensor,
        if_neg inProduct]
      change emitter.finalScaleFirst + column - finalScaleFirstColumn =
        emitter.finalScaleFirst + offset
      rw [Nat.add_sub_assoc lower]
    rw [mapped, valid.finalScaleSelected_inverse offset offsetInRange]
    have canonicalReconstruct : finalScaleFirstColumn + offset = column := by
      simp only [offset]
      omega
    exact congrArg some canonicalReconstruct

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
