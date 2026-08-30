import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.Stage1.SpartanBounds

/-!
Owns indexed access to the exact canonical PiRLC sampler ordinary rows for the
direct 14-matrix compiler. The source list is the established 32 digest-lane
lowerings and one fail-closed selector assertion per scalar source.

This module does not retain source values or construct final matrix rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Exact sampler ordinary rows after the canonical Spartan permutation. -/
def sourceRows : List R1CS.Row :=
  (PiRLCSamplerOrdinaryRows.rows (logicalWidth := logicalWidth)
    (publicFits := publicFits)).map Rows.CompiledRow.toR1CS

@[simp] theorem sourceRows_length :
    (sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length = 220881 := by
  rw [sourceRows, List.length_map]
  exact PiRLCSamplerOrdinaryRows.rows_length

def poseidonSource (source round : Nat) (lane : Fin 4) : Nat :=
  match round with
  | 0 => PiRLCStarts.samplerSourceLogicalStart source + 584 + lane.val
  | previous + 1 =>
      DigestWindow.permutationOffset
          (Sampler.windowOffset
            (SamplerChain.sourceOffset PiRLCStarts.samplerLogicalStart source)
            previous) +
        584 + lane.val

theorem fastLaneSource_eq_var (source round : Nat) (lane : Fin 4) :
    PiRLCSamplerOrdinaryRows.fastLaneSource
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane = Expr.var (poseidonSource source round lane) := by
  unfold PiRLCSamplerOrdinaryRows.fastLaneSource
    PiRLCSamplerOrdinaryRows.fastWindowInitialState
    PiRLCSamplerProjection.fastProductionWindowInitialState poseidonSource
  cases round with
  | zero =>
      rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
      rfl
  | succ previous => rfl

def selectorSource (source : Nat) : Nat :=
  First54.positionOffset (PiRLCStarts.selectorLogicalStart source)
      (First54.candidateCount - 1) + First54.fullSlot.val

theorem selectorFinalConstraint_source (source : Nat) :
    PiRLCSamplerOrdinaryRows.selectorFinalConstraint source =
      Expr.var (selectorSource source) - 1 := by
  rfl

/-- Exact pre-Spartan source values used by the sampler ordinary remainder. -/
inductive Source : Nat → Prop where
  | poseidon (source round : Nat) (lane : Fin 4)
      (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
      (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount) :
      Source (poseidonSource source round lane)
  | logical (source round lane position : Nat)
      (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
      (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount)
      (laneLt : lane < 4) (positionLt : position < 100) :
      Source (PiRLCStarts.digestLaneLogicalStart source round lane + position)
  | fresh (source round lane position : Nat)
      (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
      (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount)
      (laneLt : lane < 4) (positionLt : position < 303) :
      Source (PiRLCStarts.digestLaneFreshStart source round lane + position)
  | selector (source : Nat)
      (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount) :
      Source (selectorSource source)

/-- Exact support after the canonical Spartan column permutation. -/
def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ Spartan.sourceToSpartan source = column

private theorem laneConstraints_varsSatisfy
    (source round : Nat) (lane : Fin 4)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
    (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount) :
    ∀ expression ∈ PiRLCSamplerOrdinaryRows.laneConstraints
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane,
      expression.VarsSatisfy Source := by
  rw [PiRLCSamplerOrdinaryRows.laneConstraints_eq_fromCircuit]
  apply DigestLane.flatConstraints_varsSatisfy
  · rw [← PiRLCSamplerOrdinaryRows.fastLaneSource_eq]
    rw [fastLaneSource_eq_var]
    exact Source.poseidon source round lane sourceLt roundLt
  · intro index bounded
    exact Source.logical source round lane.val index sourceLt roundLt lane.isLt
      (by simpa [DigestLane.logicalPrivateCount] using bounded)

private theorem selectorConstraint_varsSatisfy (source : Nat)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount) :
    (PiRLCSamplerOrdinaryRows.selectorFinalConstraint source).VarsSatisfy
      Source := by
  rw [selectorFinalConstraint_source]
  exact Expr.VarsSatisfy.sub _ _ Source (Source.selector source sourceLt) trivial

private theorem laneLowered_varsBelow
    (source round : Nat) (lane : Fin 4)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
    (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount) :
    ∀ row ∈ R1CS.lowerConstraints
        (PiRLCSamplerOrdinaryRows.laneConstraints
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source round lane)
        (PiRLCStarts.digestLaneFreshStart source round lane.val) |>.rows,
      row.VarsBelow Spartan.SourceColumnCount := by
  have scope : ∀ expression ∈ PiRLCSamplerOrdinaryRows.laneConstraints
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane,
      expression.VarsBelow
        (PiRLCStarts.digestLaneFreshStart source round lane.val) := by
    intro expression member
    apply Expr.VarsBelow.mono expression
      (PiRLCSamplerOrdinaryRows.laneConstraints_varsBelow
        source round lane roundLt expression member)
    have laneLt := lane.isLt
    norm_num [PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
      PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
      PiRLCStarts.phaseFreshStart, PiRLCStarts.digestLaneLogicalStart,
      PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
      Formal.samplerOffset, Formal.logicalPrivateCount,
      DigestLane.logicalPrivateCount] at laneLt ⊢
    omega
  have lowered := R1CS.lowerConstraints_rows_varsBelow
    (PiRLCSamplerOrdinaryRows.laneConstraints
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
    (PiRLCStarts.digestLaneFreshStart source round lane.val) scope
  have freshCount : R1CS.totalFreshCount
      (PiRLCSamplerOrdinaryRows.laneConstraints
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane) = 303 := by
    rw [PiRLCSamplerOrdinaryRows.laneConstraints_eq_fromCircuit]
    exact NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (PiRLCStarts.digestLaneLogicalStart source round lane.val)
      (PiRLCSamplerOrdinaryRows.laneInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
  rw [freshCount] at lowered
  intro row member
  apply R1CS.Row.VarsBelow.mono row (lowered row member)
  have laneLt := lane.isLt
  simp only [PiRLCSamplerOrdinaryRows.sourceCount] at sourceLt
  simp only [PiRLCSamplerOrdinaryRows.digestRoundCount] at roundLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
    PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
    PiRLCStarts.phaseFreshStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Formal.logicalPrivateCount] at sourceLt roundLt laneLt ⊢
  omega

private theorem laneLowered_varsSatisfy
    (source round : Nat) (lane : Fin 4)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
    (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount) :
    ∀ row ∈ R1CS.lowerConstraints
        (PiRLCSamplerOrdinaryRows.laneConstraints
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source round lane)
        (PiRLCStarts.digestLaneFreshStart source round lane.val) |>.rows,
      row.VarsSatisfy Source := by
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy
    (PiRLCSamplerOrdinaryRows.laneConstraints
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
    (PiRLCStarts.digestLaneFreshStart source round lane.val) Source
    (laneConstraints_varsSatisfy source round lane sourceLt roundLt)
  have freshCount : R1CS.totalFreshCount
      (PiRLCSamplerOrdinaryRows.laneConstraints
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane) = 303 := by
    rw [PiRLCSamplerOrdinaryRows.laneConstraints_eq_fromCircuit]
    exact NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (PiRLCStarts.digestLaneLogicalStart source round lane.val)
      (PiRLCSamplerOrdinaryRows.laneInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
  rw [freshCount] at lowered
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro column support
  rcases support with sourceSupport | freshSupport
  · exact sourceSupport
  · let position := column -
        PiRLCStarts.digestLaneFreshStart source round lane.val
    have positionLt : position < 303 := by
      dsimp [position]
      omega
    have columnEq :
        PiRLCStarts.digestLaneFreshStart source round lane.val + position =
          column := by
      dsimp [position]
      omega
    rw [← columnEq]
    exact Source.fresh source round lane.val position sourceLt roundLt lane.isLt
      positionLt

private theorem selectorLowered_varsBelow (source : Nat)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount) :
    ∀ row ∈ R1CS.lowerConstraints
        [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source]
        (PiRLCStarts.selectorFreshStart source + 34047) |>.rows,
      row.VarsBelow Spartan.SourceColumnCount := by
  let start := PiRLCStarts.selectorFreshStart source + 34047
  have scope : ∀ expression ∈
      [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source],
      expression.VarsBelow start := by
    intro expression member
    simp only [List.mem_singleton] at member
    subst expression
    unfold PiRLCSamplerOrdinaryRows.selectorFinalConstraint
    apply Expr.VarsBelow.sub
    · simp only [First54.finalFull, First54.positionOffset,
        First54Step.output, Expr.VarsBelow]
      norm_num [start, PiRLCStarts.selectorFreshStart,
        PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart, PiRLCStarts.selectorLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
        PiRLCStarts.phaseLogicalStart, Formal.logicalPrivateCount,
        First54.candidateCount, First54.roundPrivateCount,
        First54.fullSlot, First54Step.slotCount, First54Step.fullSlot,
        First54ValueStep.outputCount, Formal.samplerOffset,
        PiRLCInputs.phaseOffset]
      omega
    · trivial
  have lowered := R1CS.lowerConstraints_rows_varsBelow
    [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source] start scope
  have noFresh : R1CS.totalFreshCount
      [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source] = 0 := by
    rfl
  rw [noFresh, Nat.add_zero] at lowered
  intro row member
  apply R1CS.Row.VarsBelow.mono row (lowered row member)
  simp only [PiRLCSamplerOrdinaryRows.sourceCount] at sourceLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [start, PiRLCStarts.selectorFreshStart,
    PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
    PiRLCStarts.phaseFreshStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    Formal.logicalPrivateCount] at sourceLt ⊢
  omega

private theorem selectorLowered_varsSatisfy (source : Nat)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount) :
    ∀ row ∈ R1CS.lowerConstraints
        [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source]
        (PiRLCStarts.selectorFreshStart source + 34047) |>.rows,
      row.VarsSatisfy Source := by
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy
    [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source]
    (PiRLCStarts.selectorFreshStart source + 34047) Source (by
      intro expression member
      simp only [List.mem_singleton] at member
      subst expression
      exact selectorConstraint_varsSatisfy source sourceLt)
  have noFresh : R1CS.totalFreshCount
      [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source] = 0 := by
    rfl
  rw [noFresh, Nat.add_zero] at lowered
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro _ support
  rcases support with sourceSupport | freshSupport
  · exact sourceSupport
  · omega

/-- Every canonical sampler ordinary row is confined to the exact Spartan
source domain. -/
theorem sourceRows_varsBelow :
    ∀ row ∈ sourceRows (logicalWidth := logicalWidth)
        (publicFits := publicFits),
      row.VarsBelow Spartan.spartanColumnCount := by
  intro row member
  rcases List.mem_map.mp member with ⟨compiled, compiledMember, rfl⟩
  unfold PiRLCSamplerOrdinaryRows.rows at compiledMember
  rcases List.mem_flatMap.mp compiledMember with
    ⟨source, sourceMember, sourceRowMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  unfold PiRLCSamplerOrdinaryRows.sourceRows at sourceRowMember
  rcases List.mem_append.mp sourceRowMember with windowMember | selectorMember
  · rcases List.mem_flatMap.mp windowMember with
      ⟨round, roundMember, windowRowMember⟩
    have roundLt := List.mem_range.mp roundMember
    unfold PiRLCSamplerOrdinaryRows.windowRows at windowRowMember
    rcases List.mem_flatMap.mp windowRowMember with
      ⟨lane, _laneMember, laneRowMember⟩
    have mappedMember : Rows.CompiledRow.toR1CS compiled ∈
        (PiRLCSamplerOrdinaryRows.laneRows
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source round lane).map Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, laneRowMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.laneRows_toR1CS] at mappedMember
    exact Spartan.remapRows_varsBelow _
      (laneLowered_varsBelow source round lane sourceLt roundLt)
      _ mappedMember
  · have mappedMember : Rows.CompiledRow.toR1CS compiled ∈
        (PiRLCSamplerOrdinaryRows.selectorFinalRows source).map
          Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, selectorMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.selectorFinalRows_toR1CS] at mappedMember
    exact Spartan.remapRows_varsBelow _
      (selectorLowered_varsBelow source sourceLt) _ mappedMember

/-- Every sampler ordinary row uses only the exact retained source set after
the canonical Spartan permutation. -/
theorem sourceRows_varsSatisfy :
    ∀ row ∈ sourceRows (logicalWidth := logicalWidth)
        (publicFits := publicFits),
      row.VarsSatisfy Target := by
  intro row member
  rcases List.mem_map.mp member with ⟨compiled, compiledMember, rfl⟩
  unfold PiRLCSamplerOrdinaryRows.rows at compiledMember
  rcases List.mem_flatMap.mp compiledMember with
    ⟨source, sourceMember, sourceRowMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  unfold PiRLCSamplerOrdinaryRows.sourceRows at sourceRowMember
  rcases List.mem_append.mp sourceRowMember with windowMember | selectorMember
  · rcases List.mem_flatMap.mp windowMember with
      ⟨round, roundMember, windowRowMember⟩
    have roundLt := List.mem_range.mp roundMember
    unfold PiRLCSamplerOrdinaryRows.windowRows at windowRowMember
    rcases List.mem_flatMap.mp windowRowMember with
      ⟨lane, _laneMember, laneRowMember⟩
    have mappedMember : Rows.CompiledRow.toR1CS compiled ∈
        (PiRLCSamplerOrdinaryRows.laneRows
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source round lane).map Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, laneRowMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.laneRows_toR1CS] at mappedMember
    exact Spartan.remapRows_varsSatisfy Source Target _
      (laneLowered_varsSatisfy source round lane sourceLt roundLt)
      (fun column support => ⟨column, support, rfl⟩) _ mappedMember
  · have mappedMember : Rows.CompiledRow.toR1CS compiled ∈
        (PiRLCSamplerOrdinaryRows.selectorFinalRows source).map
          Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, selectorMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.selectorFinalRows_toR1CS] at mappedMember
    exact Spartan.remapRows_varsSatisfy Source Target _
      (selectorLowered_varsSatisfy source sourceLt)
      (fun column support => ⟨column, support, rfl⟩) _ mappedMember

theorem sourceRows_rowCount_le :
    (sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [sourceRows_length]
  norm_num [Lifecycle.cubeVariables]

def sourceListIndex (index : Fin 220881) :
    Fin (sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length :=
  Fin.cast sourceRows_length.symm index

def programRow (index : Fin 220881) : R1CS.Row :=
  (sourceRows (logicalWidth := logicalWidth)
    (publicFits := publicFits)).get (sourceListIndex index)

private theorem ofFn_cast_get {Alpha : Type} (rows : List Alpha) {count : Nat}
    (lengthEq : rows.length = count) :
    List.ofFn (fun index : Fin count =>
      rows.get (Fin.cast lengthEq.symm index)) = rows := by
  subst count
  simpa using List.ofFn_get rows

theorem programRows_eq :
    List.ofFn (programRow (logicalWidth := logicalWidth)
      (publicFits := publicFits)) =
      sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  unfold programRow sourceListIndex
  exact ofFn_cast_get _ sourceRows_length

structure SupportedProgram (rows : List R1CS.Row) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ Lifecycle.cubeVariables
  row : Fin rowCount → R1CS.Row
  exactRows : List.ofFn row = rows
  bounded : ∀ index, (row index).VarsBelow Spartan.spartanColumnCount

def SupportedProgram.toProgram {rows : List R1CS.Row}
    (source : SupportedProgram rows) :
    OrdinarySourcePlan.Program Spartan.spartanColumnCount where
  rowCount := source.rowCount
  rowCount_le := source.rowCount_le
  row := source.row
  bounded := source.bounded

def supportedProgram : SupportedProgram
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits)) where
  rowCount := 220881
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := programRow (logicalWidth := logicalWidth) (publicFits := publicFits)
  exactRows := programRows_eq
  bounded := by
    intro index
    unfold programRow
    exact sourceRows_varsBelow (logicalWidth := logicalWidth)
      (publicFits := publicFits) _
      (List.get_mem _ (sourceListIndex (logicalWidth := logicalWidth)
        (publicFits := publicFits) index))

def program : OrdinarySourcePlan.Program Spartan.spartanColumnCount :=
  (supportedProgram (logicalWidth := logicalWidth)
    (publicFits := publicFits)).toProgram

@[simp] theorem program_rowCount :
    (program (logicalWidth := logicalWidth)
      (publicFits := publicFits)).rowCount = 220881 := by
  rfl

theorem programRow_bounded (index : Fin 220881) :
    (programRow (logicalWidth := logicalWidth)
      (publicFits := publicFits) index).VarsBelow
        Spartan.spartanColumnCount := by
  unfold programRow
  exact sourceRows_varsBelow (logicalWidth := logicalWidth)
    (publicFits := publicFits) _
    (List.get_mem _ (sourceListIndex (logicalWidth := logicalWidth)
      (publicFits := publicFits) index))

private theorem holds_iff_rowsHold_ofFn {count : Nat}
    (rowAt : Fin count → R1CS.Row) (env : Env) :
    (∀ index, (rowAt index).Holds env) ↔
      R1CS.RowsHold env (List.ofFn rowAt) := by
  unfold R1CS.RowsHold
  exact List.forall_mem_ofFn_iff.symm

private theorem predicate_iff_of_eq {Alpha : Type} (predicate : Alpha → Prop)
    {left right : Alpha} (equal : left = right) :
    predicate left ↔ predicate right := by
  cases equal
  rfl

/-- Indexed canonical sampler rows hold exactly when the complete Lean-lowered
row list holds in package order. -/
theorem programRows_hold_iff_rowsHold (env : Env) :
    (∀ index : Fin 220881,
      (programRow (logicalWidth := logicalWidth)
        (publicFits := publicFits) index).Holds env) ↔
      R1CS.RowsHold env
        (sourceRows (logicalWidth := logicalWidth)
          (publicFits := publicFits)) := by
  exact (holds_iff_rowsHold_ofFn
    (programRow (logicalWidth := logicalWidth) (publicFits := publicFits))
    env).trans
      (predicate_iff_of_eq (R1CS.RowsHold env)
        (programRows_eq (logicalWidth := logicalWidth)
          (publicFits := publicFits)))

private theorem supportedHolds_iff_rowsHold {rows : List R1CS.Row}
    (source : SupportedProgram rows) (env : Env) :
    source.toProgram.Holds env ↔ R1CS.RowsHold env rows := by
  exact (holds_iff_rowsHold_ofFn source.row env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) source.exactRows)

theorem program_holds_iff_rowsHold (env : Env) :
    (program (logicalWidth := logicalWidth)
      (publicFits := publicFits)).Holds env ↔
      R1CS.RowsHold env
        (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
  exact supportedHolds_iff_rowsHold
    (supportedProgram (logicalWidth := logicalWidth) (publicFits := publicFits)) env

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource
