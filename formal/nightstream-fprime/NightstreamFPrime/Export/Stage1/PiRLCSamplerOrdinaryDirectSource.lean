import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.Stage1.SpartanBounds
import NightstreamFPrime.Layout.Stage1.SpartanValues

/-! Indexed access and exact read support for the sampler's ordinary rows.
Only the four transcript lanes, checked core, lowering values, and coefficient
words are read. Temporary witness helpers do not enter the retained assignment. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1 NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def sourceRows : List R1CS.Row :=
  (PiRLCSamplerOrdinaryRows.rows (logicalWidth := logicalWidth)
    (publicFits := publicFits)).map Rows.CompiledRow.toR1CS

@[simp] theorem sourceRows_length :
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits)).length = 14943 := by
  rw [sourceRows, List.length_map]
  exact PiRLCSamplerOrdinaryRows.rows_length

def poseidonSource (source : Nat) (lane : Fin 4) : Nat :=
  PiRLCStarts.samplerSourceLogicalStart source + 1080 + lane.val

def coreStart (source : Nat) : Nat :=
  WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source)

theorem rangeSource_eq_var (source : Nat) (lane : Fin 4) (offset : Nat) :
    (PiRLCSamplerOrdinaryRows.rangeInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) source).source lane offset =
      Expr.var (poseidonSource source lane) := by
  unfold PiRLCSamplerOrdinaryRows.rangeInterface PiRLCSamplerInvocations.fastAdvanceState
  rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
  rfl

inductive Source : Nat → Prop where
  | poseidon (source : Nat) (lane : Fin 4) (sourceLt : source < 17) :
      Source (poseidonSource source lane)
  | logical (source position : Nat) (sourceLt : source < 17) (positionLt : position < 617) :
      Source (coreStart source + position)
  | fresh (source position : Nat) (sourceLt : source < 17) (positionLt : position < 144) :
      Source (PiRLCStarts.rangeFreshStart source + position)
  | word (source position : Nat) (sourceLt : source < 17) (positionLt : position < 54) :
      Source (PiRLCStarts.challengeWordStart source + position)

def Target (column : Nat) : Prop :=
  ∃ source, Source source ∧ Spartan.sourceToSpartan source = column

theorem Source.bounded {column : Nat} (supported : Source column) :
    column < Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq]
  cases supported <;>
    simp only [poseidonSource, coreStart, WideReduction.Program.coreOffset,
      PiRLCStarts.rangeLogicalStart, PiRLCStarts.rangeFreshStart,
      PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
      PiRLCStarts.phaseFreshStart, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.challengeWordStart, PiRLCStarts.samplerLogicalStart,
      SamplerChain.sourceOffset, Sampler.rangeOffset, Sampler.wordsOffset,
      Sampler.advanceOffset, Sampler.counts.1, WideReduction.Program.privateCount_eq,
      Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq] at *
  all_goals norm_num [WideReduction.HintProgram.helperCount_eq, Formal.logicalPrivateCount_eq] at *
  all_goals omega

private theorem rangeConstraints_supported (source : Nat) (sourceLt : source < 17) :
    ∀ expression ∈ PiRLCSamplerOrdinaryRows.rangeConstraints
        (logicalWidth := logicalWidth) (publicFits := publicFits) source,
      expression.VarsSatisfy Source := by
  apply WideReduction.Program.flatConstraints_varsSatisfy
  · intro lane
    rw [rangeSource_eq_var]
    exact Source.poseidon source lane sourceLt
  · intro column lower upper
    change coreStart source ≤ column at lower
    change column < coreStart source + 617 at upper
    have positionLt : column - coreStart source < 617 := by omega
    have eq : coreStart source + (column - coreStart source) = column := by omega
    rw [← eq]
    exact Source.logical source _ sourceLt positionLt

private theorem wordConstraints_supported (source : Nat) (sourceLt : source < 17) :
    ∀ expression ∈ PiRLCSamplerOrdinaryRows.wordConstraints source,
      expression.VarsSatisfy Source := by
  apply SamplerWords.flatConstraints_varsSatisfy
  · intro column lower upper
    change coreStart source ≤ column at lower
    change column < coreStart source + 617 at upper
    have positionLt : column - coreStart source < 617 := by omega
    have eq : coreStart source + (column - coreStart source) = column := by omega
    rw [← eq]
    exact Source.logical source _ sourceLt positionLt
  · intro position bound
    exact Source.word source position sourceLt bound

private theorem range_fresh (source : Nat) :
    R1CS.totalFreshCount (PiRLCSamplerOrdinaryRows.rangeConstraints
      (logicalWidth := logicalWidth) (publicFits := publicFits) source) = 144 := by
  have total := PiRLCSamplerOrdinaryRows.rangeRows_length
    (logicalWidth := logicalWidth) (publicFits := publicFits) source
  rw [PiRLCSamplerOrdinaryRows.rangeRows, PiCCSArithmetic.compilePacket_length,
    R1CS.totalRowCount_eq_fresh_add_length] at total
  have length : (PiRLCSamplerOrdinaryRows.rangeConstraints
      (logicalWidth := logicalWidth) (publicFits := publicFits) source).length = 681 :=
    WideReduction.Program.rowCount_eq _ _
  rw [length] at total
  omega

private theorem rangeLowered_supported (source : Nat) (sourceLt : source < 17) :
    ∀ row ∈ (R1CS.lowerConstraints (PiRLCSamplerOrdinaryRows.rangeConstraints
        (logicalWidth := logicalWidth) (publicFits := publicFits) source)
        (PiRLCStarts.rangeFreshStart source)).rows,
      row.VarsSatisfy Source := by
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy _
    (PiRLCStarts.rangeFreshStart source) Source
    (rangeConstraints_supported (logicalWidth := logicalWidth) (publicFits := publicFits) source sourceLt)
  rw [range_fresh] at lowered
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro column support
  rcases support with supported | fresh
  · exact supported
  · have positionLt : column - PiRLCStarts.rangeFreshStart source < 144 := by omega
    have eq : PiRLCStarts.rangeFreshStart source +
        (column - PiRLCStarts.rangeFreshStart source) = column := by omega
    rw [← eq]
    exact Source.fresh source _ sourceLt positionLt

private theorem wordLowered_supported (source : Nat) (sourceLt : source < 17) :
    ∀ row ∈ (R1CS.lowerConstraints (PiRLCSamplerOrdinaryRows.wordConstraints source)
        (PiRLCStarts.rangeFreshStart source + 144)).rows,
      row.VarsSatisfy Source := by
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy _
    (PiRLCStarts.rangeFreshStart source + 144) Source
    (wordConstraints_supported source sourceLt)
  have fresh := Layout.PiRLC.v1_1.Sampler.words_fresh
    (PiRLCSamplerInvocations.sourceLogicalStart source)
  change R1CS.totalFreshCount (PiRLCSamplerOrdinaryRows.wordConstraints source) = 0 at fresh
  rw [fresh] at lowered
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro column support
  rcases support with supported | fresh
  · exact supported
  · omega

/-- No row reads the temporary quotient-construction helpers. -/
theorem sourceRows_varsSatisfy :
    ∀ row ∈ sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits),
      row.VarsSatisfy Target := by
  intro row member
  obtain ⟨compiled, compiledMember, rfl⟩ := List.mem_map.mp member
  obtain ⟨source, sourceMember, sourceRowMember⟩ := List.mem_flatMap.mp compiledMember
  have sourceLt : source < 17 := List.mem_range.mp sourceMember
  rcases List.mem_append.mp sourceRowMember with rangeMember | wordMember
  · have mapped : compiled.toR1CS ∈ (PiRLCSamplerOrdinaryRows.rangeRows
        (logicalWidth := logicalWidth) (publicFits := publicFits) source).map Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, rangeMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.rangeRows_toR1CS] at mapped
    exact Spartan.remapRows_varsSatisfy Source Target _
      (rangeLowered_supported source sourceLt)
      (fun column support => ⟨column, support, rfl⟩) _ mapped
  · have mapped : compiled.toR1CS ∈ (PiRLCSamplerOrdinaryRows.wordRows source).map Rows.CompiledRow.toR1CS :=
      List.mem_map.mpr ⟨compiled, wordMember, rfl⟩
    rw [PiRLCSamplerOrdinaryRows.wordRows_toR1CS] at mapped
    exact Spartan.remapRows_varsSatisfy Source Target _
      (wordLowered_supported source sourceLt)
      (fun column support => ⟨column, support, rfl⟩) _ mapped

theorem sourceRows_varsBelow :
    ∀ row ∈ sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits),
      row.VarsBelow Spartan.spartanColumnCount := by
  intro row member
  change row.VarsSatisfy (fun column => column < Spartan.spartanColumnCount)
  apply R1CS.Row.VarsSatisfy.mono row (sourceRows_varsSatisfy row member)
  intro column support
  rcases support with ⟨source, supported, rfl⟩
  exact Spartan.sourceToSpartan_lt source supported.bounded

theorem sourceRows_rowCount_le :
    (sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [sourceRows_length]
  norm_num [Lifecycle.cubeVariables]

def sourceListIndex (index : Fin 14943) :
    Fin (sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length :=
  Fin.cast sourceRows_length.symm index

def programRow (index : Fin 14943) : R1CS.Row :=
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
  rowCount := 14943
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
      (publicFits := publicFits)).rowCount = 14943 := by
  rfl

theorem programRow_bounded (index : Fin 14943) :
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
    (∀ index : Fin 14943,
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
