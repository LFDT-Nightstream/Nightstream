import NightstreamFPrime.Export.Stage1.PerApplicationPackage
import NightstreamFPrime.Export.Stage1.PiCCSInvocationSchedule

/-!
Owns physical row preservation for the generic per-application package.

`ShiftCompatible` is a proof object, not a cryptographic assumption. Its three
fields state exact R1CS row equalities for the hash, explicit permutation, and
compact invocation encodings after the constant/public suffix moves. A later
production theorem must construct this object from the canonical generators.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationPreservation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1.PerApplicationPackage

def RefWithin (inputCount localCount : Nat) : ColumnRef → Prop
  | .input index => index < inputCount
  | .local index => index < localCount

def CombinationWithin (inputCount localCount : Nat)
    (combination : TemplateCombination) : Prop :=
  ∀ term ∈ combination.terms, RefWithin inputCount localCount term.column

def TemplateRowWithin (inputCount localCount : Nat)
    (row : TemplateRow) : Prop :=
  CombinationWithin inputCount localCount row.a ∧
    CombinationWithin inputCount localCount row.b ∧
      CombinationWithin inputCount localCount row.c

def CompactTemplateRowWithin (inputCount localCount : Nat)
    (row : CompactTemplateRow) : Prop :=
  CombinationWithin inputCount localCount row.a ∧
    CombinationWithin inputCount localCount row.b ∧
      CombinationWithin inputCount localCount row.c

@[simp] theorem mapRowColumns_eq_renameRow (column : Nat → Nat)
    (row : Layout.R1CS.Row) :
    mapRowColumns column row = CompactRows.renameRow column row := by
  rfl

theorem instantiateRow_mapColumns_of_within
    (finalPackage sourcePackage : CircuitPackage)
    (finalChain sourceChain : HashChain) (invocation : Nat)
    (column : Nat → Nat) (inputCount localCount : Nat) (row : TemplateRow)
    (within : TemplateRowWithin inputCount localCount row)
    (agree : ∀ reference, RefWithin inputCount localCount reference →
      instantiateColumn finalPackage finalChain invocation reference =
        mapCombinationColumns column
          (instantiateColumn sourcePackage sourceChain invocation reference)) :
    instantiateRow finalPackage finalChain invocation row =
      CompactRows.renameRow column
        (instantiateRow sourcePackage sourceChain invocation row) := by
  cases row with
  | mk outputLocal a b c =>
      unfold TemplateRowWithin at within
      unfold instantiateRow
      rw [show CompactRows.renameRow column
          { a := instantiateCombination sourcePackage sourceChain invocation a
            b := instantiateCombination sourcePackage sourceChain invocation b
            c := instantiateCombination sourcePackage sourceChain invocation c } =
          mapRowColumns column
          { a := instantiateCombination sourcePackage sourceChain invocation a
            b := instantiateCombination sourcePackage sourceChain invocation b
            c := instantiateCombination sourcePackage sourceChain invocation c }
        by rfl]
      unfold mapRowColumns
      congr 1
      · exact instantiateCombination_mapColumns finalPackage sourcePackage
          finalChain sourceChain invocation column a
          (fun term member => agree term.column (within.1 term member))
      · exact instantiateCombination_mapColumns finalPackage sourcePackage
          finalChain sourceChain invocation column b
          (fun term member => agree term.column (within.2.1 term member))
      · exact instantiateCombination_mapColumns finalPackage sourcePackage
          finalChain sourceChain invocation column c
          (fun term member => agree term.column (within.2.2 term member))

theorem instantiateInvocationRow_mapColumns_of_within
    (finalInvocation sourceInvocation : PermutationInvocation)
    (column : Nat → Nat) (inputCount localCount : Nat) (row : TemplateRow)
    (within : TemplateRowWithin inputCount localCount row)
    (agree : ∀ reference, RefWithin inputCount localCount reference →
      instantiateInvocationColumn finalInvocation reference =
        mapCombinationColumns column
          (instantiateInvocationColumn sourceInvocation reference)) :
    instantiateInvocationRow finalInvocation row =
      CompactRows.renameRow column
        (instantiateInvocationRow sourceInvocation row) := by
  cases row with
  | mk outputLocal a b c =>
      unfold TemplateRowWithin at within
      unfold instantiateInvocationRow
      rw [show CompactRows.renameRow column
          { a := instantiateInvocationCombination sourceInvocation a
            b := instantiateInvocationCombination sourceInvocation b
            c := instantiateInvocationCombination sourceInvocation c } =
          mapRowColumns column
          { a := instantiateInvocationCombination sourceInvocation a
            b := instantiateInvocationCombination sourceInvocation b
            c := instantiateInvocationCombination sourceInvocation c }
        by rfl]
      unfold mapRowColumns
      congr 1
      · exact instantiateInvocationCombination_mapColumns finalInvocation
          sourceInvocation column a
          (fun term member => agree term.column (within.1 term member))
      · exact instantiateInvocationCombination_mapColumns finalInvocation
          sourceInvocation column b
          (fun term member => agree term.column (within.2.1 term member))
      · exact instantiateInvocationCombination_mapColumns finalInvocation
          sourceInvocation column c
          (fun term member => agree term.column (within.2.2 term member))

theorem instantiateCompactRow_mapColumns_of_within
    (finalInvocation sourceInvocation : CompactRowInvocation)
    (column : Nat → Nat) (inputCount localCount : Nat)
    (row : CompactTemplateRow)
    (within : CompactTemplateRowWithin inputCount localCount row)
    (agree : ∀ reference, RefWithin inputCount localCount reference →
      instantiateCompactColumn finalInvocation reference =
        column (instantiateCompactColumn sourceInvocation reference)) :
    instantiateCompactRow finalInvocation row =
      CompactRows.renameRow column
        (instantiateCompactRow sourceInvocation row) := by
  cases row with
  | mk outputLocal a b c =>
      unfold CompactTemplateRowWithin at within
      unfold instantiateCompactRow
      rw [show CompactRows.renameRow column
          { a := instantiateCompactCombination sourceInvocation a
            b := instantiateCompactCombination sourceInvocation b
            c := instantiateCompactCombination sourceInvocation c } =
          mapRowColumns column
          { a := instantiateCompactCombination sourceInvocation a
            b := instantiateCompactCombination sourceInvocation b
            c := instantiateCompactCombination sourceInvocation c }
        by rfl]
      unfold mapRowColumns
      congr 1
      · exact instantiateCompactCombination_mapColumns finalInvocation
          sourceInvocation column a
          (fun term member => agree term.column (within.1 term member))
      · exact instantiateCompactCombination_mapColumns finalInvocation
          sourceInvocation column b
          (fun term member => agree term.column (within.2.1 term member))
      · exact instantiateCompactCombination_mapColumns finalInvocation
          sourceInvocation column c
          (fun term member => agree term.column (within.2.2 term member))

private theorem abstractColumn_refWithin (inputCount localCount column : Nat)
    (bound : column < inputCount + localCount) :
    RefWithin inputCount localCount
      (CompactRows.abstractColumn inputCount column) := by
  by_cases input : column < inputCount
  · simp [RefWithin, CompactRows.abstractColumn, input]
  · simp [RefWithin, CompactRows.abstractColumn, input]
    omega

private theorem abstractCombination_within
    (inputCount localCount : Nat)
    (combination : Layout.R1CS.LinearCombination)
    (scope : combination.VarsBelow (inputCount + localCount)) :
    CombinationWithin inputCount localCount
      (CompactRows.abstractCombination inputCount combination) := by
  intro term member
  unfold CompactRows.abstractCombination at member
  simp only [List.mem_map] at member
  rcases member with ⟨sourceTerm, sourceMember, rfl⟩
  exact abstractColumn_refWithin inputCount localCount sourceTerm.1
    (scope sourceTerm sourceMember)

private theorem abstractRow_within (inputCount localCount : Nat)
    (row : Layout.R1CS.Row)
    (scope : row.VarsBelow (inputCount + localCount)) :
    CompactTemplateRowWithin inputCount localCount
      (CompactRows.abstractRow inputCount row) := by
  exact ⟨abstractCombination_within inputCount localCount row.a scope.1,
    abstractCombination_within inputCount localCount row.b scope.2.1,
    abstractCombination_within inputCount localCount row.c scope.2.2⟩

theorem compactTemplate_rowWithin (inputCount outputInput : Nat)
    (outputRecipe : Expr) (row : CompactTemplateRow)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (member : row ∈
      (CompactRows.compactTemplate inputCount outputInput outputRecipe).rows) :
    CompactTemplateRowWithin inputCount
      (Layout.R1CS.mulCount (Expr.var outputInput - outputRecipe)) row := by
  unfold CompactRows.compactTemplate at member
  simp only [List.mem_map] at member
  rcases member with ⟨sourceRow, sourceMember, rfl⟩
  exact abstractRow_within inputCount
    (Layout.R1CS.mulCount (Expr.var outputInput - outputRecipe)) sourceRow
    (Layout.R1CS.lowerGenericConstraint_rows_varsBelow
      (Expr.var outputInput - outputRecipe) inputCount scope sourceRow
      sourceMember)

theorem compactConstraintTemplate_rowWithin (inputCount outputInput : Nat)
    (outputRecipe : Expr) (row : CompactTemplateRow)
    (scope : (Expr.var outputInput - outputRecipe).VarsBelow inputCount)
    (member : row ∈
      (CompactRows.compactConstraintTemplate inputCount outputInput
        outputRecipe).rows) :
    CompactTemplateRowWithin inputCount
      (Layout.R1CS.constraintFreshCount
        (Expr.var outputInput - outputRecipe)) row := by
  unfold CompactRows.compactConstraintTemplate at member
  simp only [List.mem_map] at member
  rcases member with ⟨sourceRow, sourceMember, rfl⟩
  exact abstractRow_within inputCount
    (Layout.R1CS.constraintFreshCount
      (Expr.var outputInput - outputRecipe)) sourceRow
    (Layout.R1CS.lowerConstraint_rows_varsBelow
      (Expr.var outputInput - outputRecipe) inputCount scope sourceRow
      sourceMember)

private theorem pilotColumnRef_refWithin (column : Nat) (bound : column < 600) :
    RefWithin 8 592 (PilotData.columnRef column) := by
  by_cases input : column < 8
  · simp [RefWithin, PilotData.columnRef, input]
  · simp [RefWithin, PilotData.columnRef, input]
    omega

private theorem pilotTemplateCombination_within
    (combination : Layout.R1CS.LinearCombination)
    (scope : combination.VarsBelow 600) :
    CombinationWithin 8 592 (PilotData.templateCombination combination) := by
  intro term member
  unfold PilotData.templateCombination at member
  simp only [List.mem_map] at member
  rcases member with ⟨sourceTerm, sourceMember, rfl⟩
  exact pilotColumnRef_refWithin sourceTerm.1 (scope sourceTerm sourceMember)

private theorem pilotTemplateRow_within (output : Nat)
    (row : Layout.R1CS.Row) (scope : row.VarsBelow 600) :
    TemplateRowWithin 8 592
      { outputLocal := output
        a := PilotData.templateCombination row.a
        b := PilotData.templateCombination row.b
        c := PilotData.templateCombination row.c } := by
  exact ⟨pilotTemplateCombination_within row.a scope.1,
    pilotTemplateCombination_within row.b scope.2.1,
    pilotTemplateCombination_within row.c scope.2.2⟩

private theorem pilotTemplateRowsFrom_within (output : Nat)
    (rows : List Layout.R1CS.Row)
    (scope : ∀ row ∈ rows, row.VarsBelow 600) :
    ∀ row ∈ PilotData.templateRowsFrom output rows,
      TemplateRowWithin 8 592 row := by
  induction rows generalizing output with
  | nil => simp [PilotData.templateRowsFrom]
  | cons head rest inductionHypothesis =>
      intro row member
      simp only [PilotData.templateRowsFrom, List.mem_cons] at member
      rcases member with rfl | member
      · exact pilotTemplateRow_within output head (scope head (by simp))
      · exact inductionHypothesis (output + 1)
          (fun current currentMember => scope current (by simp [currentMember]))
          row member

private theorem canonicalConstraints_scope :
    ∀ expression ∈ PilotData.canonicalConstraints (),
      expression.VarsBelow 600 := by
  intro expression member
  have scope := recipeConstraints_varsBelow_of_causal 8
    (PilotData.canonicalRecipes ())
    (NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile_schedule_causal
      8 PilotData.canonicalState (by
        intro lane
        exact lane.isLt)) expression member
  have recipeLength : (PilotData.canonicalRecipes ()).length = 592 := by
    exact NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile_schedule_recipe_count
      8 PilotData.canonicalState
  rw [recipeLength] at scope
  norm_num at scope
  exact scope

set_option maxRecDepth 100000 in -- fixed-size: 592 canonical permutation rows
private theorem canonicalRows_scope :
    ∀ row ∈ PilotData.canonicalRows (), row.VarsBelow 600 := by
  have scope := Layout.R1CS.lowerConstraints_rows_varsBelow
    (PilotData.canonicalConstraints ()) 600 canonicalConstraints_scope
  have noFresh : Layout.R1CS.totalFreshCount
      (PilotData.canonicalConstraints ()) = 0 := by
    rfl
  simpa [PilotData.canonicalRows, noFresh] using scope

theorem canonicalPermutationTemplate_rowWithin (row : TemplateRow)
    (member : row ∈ basePackage.permutation.rows) :
    TemplateRowWithin 8 592 row := by
  change row ∈ PilotData.templateRows () at member
  exact pilotTemplateRowsFrom_within 0 (PilotData.canonicalRows ())
    canonicalRows_scope row member

theorem shiftColumn_add_of_private
    (program : Lifecycle.Stage1.Application.Program) (start offset : Nat)
    (bound : start + offset < basePackage.layout.constantColumn) :
    shiftColumn program (start + offset) =
      shiftColumn program start + offset := by
  rw [shiftColumn_private program (start + offset) bound,
    shiftColumn_private program start (by omega)]

theorem shiftColumn_add_of_suffix
    (program : Lifecycle.Stage1.Application.Program) (start offset : Nat)
    (bound : basePackage.layout.constantColumn ≤ start) :
    shiftColumn program (start + offset) =
      shiftColumn program start + offset := by
  rw [shiftColumn_constantOrPublic program start bound,
    shiftColumn_constantOrPublic program (start + offset) (by omega)]
  omega

theorem shiftedInvocationInputCombination
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : PermutationInvocation) (lane : Nat) :
    invocationInputCombination
        (shiftPermutationInvocation program invocation) lane =
      shiftSparseCombination program
        (invocationInputCombination invocation lane) := by
  unfold invocationInputCombination shiftPermutationInvocation
  simpa [zeroSparseCombination, shiftSparseCombination]
    using (List.getD_map (n := lane) invocation.inputs
      zeroSparseCombination (shiftSparseCombination program))

theorem shiftedInvocationColumn
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : PermutationInvocation)
    (witnessBound : invocation.witnessStart + 592 ≤
      basePackage.layout.constantColumn)
    (reference : ColumnRef) (within : RefWithin 8 592 reference) :
    instantiateInvocationColumn
        (shiftPermutationInvocation program invocation) reference =
      mapCombinationColumns (shiftColumn program)
        (instantiateInvocationColumn invocation reference) := by
  cases reference with
  | input lane =>
      simp only [instantiateInvocationColumn]
      rw [shiftedInvocationInputCombination]
      exact shiftSparseCombination_toR1CS program
        (invocationInputCombination invocation lane)
  | «local» index =>
      simp only [RefWithin] at within
      simp only [instantiateInvocationColumn, shiftPermutationInvocation]
      rw [mapCombinationColumns_ofVar,
        shiftColumn_private program invocation.witnessStart (by omega),
        shiftColumn_private program (invocation.witnessStart + index) (by
          omega)]

private theorem scheduleWithin_member_witnessBound
    {bound ceiling : Nat} {invocations : List PermutationInvocation}
    (schedule : Invocations.ScheduleWithin bound ceiling invocations) :
    ∀ invocation ∈ invocations, invocation.witnessStart + 592 ≤ ceiling := by
  induction invocations generalizing bound with
  | nil => simp
  | cons head rest inductionHypothesis =>
      rcases schedule with
        ⟨_startsAfter, headBound, _inputs, _stableInputs, restSchedule⟩
      intro invocation member
      rcases List.mem_cons.mp member with rfl | member
      · exact headBound
      · exact inductionHypothesis restSchedule invocation member

private theorem piCcsInvocation_witnessBound
    (invocation : PermutationInvocation)
    (member : invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits) :
    invocation.witnessStart + 592 ≤ basePackage.layout.constantColumn := by
  let layoutWitness : ProductionKey.LogicalRelation
      Data.logicalWidth Data.publicFits :=
    { matrices := fun _ _ _ => 0
      cubeFits := by
        norm_num [Data.logicalWidth,
          NightstreamFPrime.Export.Stage1.VerifierContext.candidateLogicalWidth,
          NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth,
          NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount,
          NightstreamFPrime.Spec.ringDegree,
          Lifecycle.cubeVariables] }
  have schedule := PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits layoutWitness
  have bound := scheduleWithin_member_witnessBound schedule.1 invocation member
  exact Nat.le_trans bound (by
    change PiCCSInvocations.invocationCeiling ≤
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn
    rw [NightstreamFPrime.Layout.Stage1.Spartan.constantColumn_eq_private]
    exact PiCCSInvocations.invocationCeiling_le_private)

private theorem samplerEntryInvocation_witnessStart
    (source : Nat) (invocation : PermutationInvocation)
    (member : invocation ∈
      PiRLCSamplerInvocations.entryInvocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source) :
    invocation.witnessStart =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (PiRLCSamplerInvocations.sourceLogicalStart source) := by
  simp [PiRLCSamplerInvocations.entryInvocations,
    PiRLCSamplerInvocations.entryTrace, Invocations.compileActions,
    Invocations.compileBlocks,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.actions,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords,
    NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks,
    NightstreamFPrime.Spec.Poseidon2.rate] at member
  subst invocation
  rfl

private theorem samplerInvocation_witnessBound
    (invocation : PermutationInvocation)
    (member : invocation ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)) :
    invocation.witnessStart + 592 ≤ basePackage.layout.constantColumn := by
  unfold PiRLCSamplerInvocations.invocations at member
  rcases List.mem_flatMap.mp member with
    ⟨source, sourceMember, sourceInvocationMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  unfold PiRLCSamplerInvocations.sourceInvocations at sourceInvocationMember
  rcases List.mem_append.mp sourceInvocationMember with
      entryMember | windowMember
  · rw [samplerEntryInvocation_witnessStart source invocation entryMember]
    have sourceLocal : NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiRLCSamplerInvocations.sourceLogicalStart source := by
      unfold PiRLCSamplerInvocations.sourceLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
      omega
    change NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (PiRLCSamplerInvocations.sourceLogicalStart source) + 592 ≤
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn
    unfold NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
    rw [if_neg (by
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
        at sourceLocal ⊢
      omega), if_neg (by
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
        at sourceLocal ⊢
      omega), if_neg (by omega)]
    norm_num [PiRLCSamplerInvocations.sourceCount,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn,
      PiRLCSamplerInvocations.sourceLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
      at sourceLt ⊢
    omega
  · unfold PiRLCSamplerInvocations.windowInvocations at windowMember
    rcases List.mem_map.mp windowMember with
      ⟨round, roundMember, rfl⟩
    have roundLt := List.mem_range.mp roundMember
    have windowLocal :
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
          NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
            source round := by
      unfold NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
      omega
    change NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
          source round) + 592 ≤
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn
    unfold NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
    rw [if_neg (by
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
        at windowLocal ⊢
      omega), if_neg (by
      norm_num [NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
        at windowLocal ⊢
      omega), if_neg (by omega)]
    norm_num [PiRLCSamplerInvocations.sourceCount,
      PiRLCSamplerInvocations.digestRoundCount,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
      at sourceLt roundLt ⊢
    omega

theorem canonicalPermutationInvocation_witnessBound
    (invocation : PermutationInvocation)
    (member : invocation ∈ basePackage.permutationInvocations) :
    invocation.witnessStart + 592 ≤ basePackage.layout.constantColumn := by
  change invocation ∈ Data.permutationInvocations () at member
  rw [Data.permutationInvocations_eq, List.mem_append] at member
  rcases member with member | member
  · exact piCcsInvocation_witnessBound invocation member
  · exact samplerInvocation_witnessBound invocation member

structure HashChainPrivate (chain : HashChain) : Prop where
  inputEnd : chain.inputStart + chain.inputLength ≤
    basePackage.layout.constantColumn
  witnessEnd : chain.witnessStart + chain.witnessLength ≤
    basePackage.layout.constantColumn
  witnessLength : chain.witnessLength = (chain.absorbCount + 1) * 592

theorem canonicalHashChain_private (chain : HashChain)
    (member : chain ∈ basePackage.hashChains) : HashChainPrivate chain := by
  rw [show basePackage.hashChains = [Data.priorChain, Data.outputChain] by
    exact Data.circuitPackage_hashChains] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · constructor <;>
      norm_num [basePackage, Data.priorChain, Data.liftPilotChain,
        Data.circuitPackage_layout, Data.physicalLayout,
        PilotData.priorChain, NightstreamFPrime.Layout.PilotValues.stateHashWords,
        NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
        NightstreamFPrime.Layout.PilotValues.absorbCount,
        NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
        PilotData.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.constantColumn,
        NightstreamFPrime.Spec.Poseidon2.rate]
  · constructor <;>
      norm_num [basePackage, Data.outputChain, Data.liftPilotChain,
        Data.circuitPackage_layout, Data.physicalLayout,
        PilotData.outputChain, NightstreamFPrime.Layout.PilotValues.stateHashWords,
        NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
        NightstreamFPrime.Layout.PilotValues.secondPrivateStart,
        NightstreamFPrime.Layout.PilotValues.absorbCount,
        NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
        PilotData.outputWitnessStart,
        NightstreamFPrime.Layout.PilotValues.outputWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.PilotValues.priorCanonicalPrivateCount,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.constantColumn,
        NightstreamFPrime.Spec.Poseidon2.rate]

/-- Every canonical hash-chain invocation owns one complete 592-column local
interval below the private boundary. -/
theorem canonicalHashInvocation_witnessBound (chain : HashChain)
    (member : chain ∈ basePackage.hashChains)
    (invocation : Fin (chain.absorbCount + 1)) :
    chain.witnessStart + invocation.val * 592 + 592 ≤
      basePackage.layout.constantColumn := by
  have privateLayout := canonicalHashChain_private chain member
  have invocationSucc : invocation.val + 1 ≤ chain.absorbCount + 1 := by
    omega
  have scaled := Nat.mul_le_mul_right 592 invocationSucc
  calc
    chain.witnessStart + invocation.val * 592 + 592 =
        chain.witnessStart + (invocation.val + 1) * 592 := by ring
    _ ≤ chain.witnessStart + (chain.absorbCount + 1) * 592 :=
      Nat.add_le_add_left scaled _
    _ = chain.witnessStart + chain.witnessLength := by
      rw [privateLayout.witnessLength]
    _ ≤ basePackage.layout.constantColumn := privateLayout.witnessEnd

private theorem shiftedHashInvocationInput
    (program : Lifecycle.Stage1.Application.Program) (chain : HashChain)
    (layout : HashChainPrivate chain) (invocation lane : Nat)
    (invocationBound : invocation ≤ chain.absorbCount) (laneBound : lane < 8) :
    invocationInput (package program) (shiftHashChain program chain)
        invocation lane =
      mapCombinationColumns (shiftColumn program)
        (invocationInput basePackage chain invocation lane) := by
  have baseLocalCount : basePackage.permutation.localColumnCount = 592 := by
    rfl
  have finalLocalCount :
      (package program).permutation.localColumnCount = 592 := by
    rfl
  have baseOutputStart : basePackage.permutation.outputLocalStart = 584 := by
    rfl
  have finalOutputStart :
      (package program).permutation.outputLocalStart = 584 := by
    rfl
  have baseRate : basePackage.poseidon.rate = 4 := by
    rfl
  have finalRate : (package program).poseidon.rate = 4 := by
    rfl
  unfold invocationInput
  simp only [shiftHashChain]
  rw [baseLocalCount, finalLocalCount, baseOutputStart, finalOutputStart,
    baseRate, finalRate]
  by_cases isFirst : invocation = 0
  · subst invocation
    simp only [if_pos rfl]
    by_cases absorbing : 0 < chain.absorbCount
    · simp only [absorbing, if_pos]
      by_cases present : lane < 4 ∧ 0 * 4 + lane < chain.inputLength
      · have inputEnd := layout.inputEnd
        have inputShift := shiftColumn_add_of_private program chain.inputStart
          (0 * 4 + lane) (by omega)
        norm_num at present inputShift
        simp [present.1, present.2, inputShift]
      · norm_num at present
        have absent : ¬(lane < 4 ∧ lane < chain.inputLength) := by
          intro both
          exact (Nat.not_lt.mpr (present both.1)) both.2
        simp [absent]
    · simp only [absorbing, if_neg]
      by_cases zeroLane : lane = 0
      · simp [zeroLane]
      · simp [zeroLane]
  · have witnessEnd := layout.witnessEnd
    have previousShift :
        shiftColumn program
            (chain.witnessStart + (invocation - 1) * 592 + 584 + lane) =
          shiftColumn program chain.witnessStart +
            (invocation - 1) * 592 + 584 + lane := by
      have offsetBound :
          (invocation - 1) * 592 + 584 + lane < chain.witnessLength := by
        rw [layout.witnessLength]
        omega
      have shifted := shiftColumn_add_of_private program chain.witnessStart
        ((invocation - 1) * 592 + 584 + lane) (by omega)
      calc
        shiftColumn program
            (chain.witnessStart + (invocation - 1) * 592 + 584 + lane) =
            shiftColumn program
              (chain.witnessStart +
                ((invocation - 1) * 592 + 584 + lane)) := by
              congr 1
              omega
        _ = shiftColumn program chain.witnessStart +
              ((invocation - 1) * 592 + 584 + lane) := shifted
        _ = shiftColumn program chain.witnessStart +
              (invocation - 1) * 592 + 584 + lane := by omega
    simp only [isFirst, if_false]
    by_cases absorbing : invocation < chain.absorbCount
    · simp only [absorbing, if_pos]
      by_cases present : lane < 4 ∧ invocation * 4 + lane < chain.inputLength
      · have inputEnd := layout.inputEnd
        have inputShift := shiftColumn_add_of_private program chain.inputStart
          (invocation * 4 + lane) (by omega)
        simp [present, previousShift, inputShift]
      · simp [present, previousShift]
    · simp only [absorbing, if_neg]
      by_cases zeroLane : lane = 0
      · subst lane
        simp at previousShift
        simp [previousShift]
      · simp [zeroLane, previousShift]

private theorem shiftedHashColumn
    (program : Lifecycle.Stage1.Application.Program) (chain : HashChain)
    (layout : HashChainPrivate chain) (invocation : Nat)
    (invocationBound : invocation ≤ chain.absorbCount)
    (reference : ColumnRef) (within : RefWithin 8 592 reference) :
    instantiateColumn (package program) (shiftHashChain program chain)
        invocation reference =
      mapCombinationColumns (shiftColumn program)
        (instantiateColumn basePackage chain invocation reference) := by
  cases reference with
  | input lane =>
      simp only [RefWithin] at within
      simp only [instantiateColumn]
      exact shiftedHashInvocationInput program chain layout invocation lane
        invocationBound within
  | «local» index =>
      simp only [RefWithin] at within
      unfold instantiateColumn invocationLocalStart shiftHashChain
      rw [mapCombinationColumns_ofVar]
      rw [show (package program).permutation.localColumnCount = 592 by rfl,
        show basePackage.permutation.localColumnCount = 592 by rfl]
      change Layout.R1CS.LinearCombination.ofVar
          (shiftColumn program chain.witnessStart + invocation * 592 + index) =
        Layout.R1CS.LinearCombination.ofVar
          (shiftColumn program
            (chain.witnessStart + invocation * 592 + index))
      have offsetBound : invocation * 592 + index < chain.witnessLength := by
        rw [layout.witnessLength]
        omega
      have witnessEnd := layout.witnessEnd
      have shifted := shiftColumn_add_of_private program chain.witnessStart
        (invocation * 592 + index) (by omega)
      apply congrArg Layout.R1CS.LinearCombination.ofVar
      calc
        shiftColumn program chain.witnessStart + invocation * 592 + index =
            shiftColumn program chain.witnessStart +
              (invocation * 592 + index) := by omega
        _ = shiftColumn program
              (chain.witnessStart + (invocation * 592 + index)) := shifted.symm
        _ = shiftColumn program
              (chain.witnessStart + invocation * 592 + index) := by
                congr 1
                omega

def CompactRangeCompatible
    (program : Lifecycle.Stage1.Application.Program)
    (range : CompactInputRange) : Prop :=
  ∀ offset, offset < range.inputCount →
    shiftColumn program (range.columnStart + offset * range.columnStride) =
      shiftColumn program range.columnStart + offset * range.columnStride

theorem shiftedCompactInputColumn
    (program : Lifecycle.Stage1.Application.Program)
    (ranges : List CompactInputRange)
    (compatible : ∀ range ∈ ranges, CompactRangeCompatible program range)
    (input : Nat) :
    compactInputColumn (ranges.map (shiftCompactInputRange program)) input =
      shiftColumn program (compactInputColumn ranges input) := by
  induction ranges with
  | nil =>
      have zeroPrivate : 0 < basePackage.layout.constantColumn := by
        norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
          NightstreamFPrime.Layout.Stage1.Spartan.constantColumn]
      simp [compactInputColumn, shiftColumn, zeroPrivate]
  | cons range ranges inductionHypothesis =>
      by_cases selected :
          range.inputStart ≤ input ∧
            input < range.inputStart + range.inputCount
      · have offsetBound : input - range.inputStart < range.inputCount := by
          omega
        have shifted := compatible range (by simp) (input - range.inputStart)
          offsetBound
        simp [compactInputColumn, shiftCompactInputRange, selected, shifted]
      · have tailCompatible : ∀ current ∈ ranges,
            CompactRangeCompatible program current := by
          intro current member
          exact compatible current (by simp [member])
        simpa [compactInputColumn, shiftCompactInputRange, selected] using
          inductionHypothesis tailCompatible

structure CompactInvocationPrivate
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : CompactRowInvocation) (localCount : Nat) : Prop where
  localEnd : invocation.localStart + localCount ≤
    basePackage.layout.constantColumn
  inputRanges : ∀ range ∈ invocation.inputRanges,
    CompactRangeCompatible program range

theorem shiftedCompactColumn
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : CompactRowInvocation)
    (inputCount localCount : Nat)
    (layout : CompactInvocationPrivate program invocation localCount)
    (reference : ColumnRef)
    (within : RefWithin inputCount localCount reference) :
    instantiateCompactColumn
        (shiftCompactRowInvocation program invocation) reference =
      shiftColumn program (instantiateCompactColumn invocation reference) := by
  cases reference with
  | input input =>
      simp only [RefWithin] at within
      simp only [instantiateCompactColumn, shiftCompactRowInvocation]
      exact shiftedCompactInputColumn program invocation.inputRanges
        layout.inputRanges input
  | «local» index =>
      simp only [RefWithin] at within
      unfold instantiateCompactColumn shiftCompactRowInvocation
      have localEnd := layout.localEnd
      rw [shiftColumn_private program invocation.localStart (by omega),
        shiftColumn_private program (invocation.localStart + index) (by
          omega)]

/-- Exact row-equality obligations for the three template invocation forms.
Ordinary sparse rows are already covered by
`packageRows_imply_baseOrdinaryRows`. -/
structure ShiftCompatible
    (program : Lifecycle.Stage1.Application.Program) : Prop where
  hashRows : ∀ chain ∈ basePackage.hashChains,
    ∀ invocation, invocation ≤ chain.absorbCount →
      ∀ row ∈ basePackage.permutation.rows,
        instantiateRow (package program)
            (shiftHashChain program chain) invocation row =
          CompactRows.renameRow (shiftColumn program)
            (instantiateRow basePackage chain invocation row)
  permutationRows : ∀ invocation ∈
      basePackage.permutationInvocations,
    ∀ row ∈ basePackage.permutation.rows,
      instantiateInvocationRow
          (shiftPermutationInvocation program invocation) row =
        CompactRows.renameRow (shiftColumn program)
          (instantiateInvocationRow invocation row)
  compactRows : ∀ invocation ∈
      basePackage.compactRowInvocations,
    ∀ template,
      basePackage.compactRowTemplates[
          invocation.templateIndex]? = some template →
        ∀ row ∈ template.rows,
          instantiateCompactRow
              (shiftCompactRowInvocation program invocation)
              row =
            CompactRows.renameRow (shiftColumn program)
              (instantiateCompactRow invocation row)

theorem canonicalHashRows
    (program : Lifecycle.Stage1.Application.Program) :
    ∀ chain ∈ basePackage.hashChains,
      ∀ invocation, invocation ≤ chain.absorbCount →
        ∀ row ∈ basePackage.permutation.rows,
          instantiateRow (package program)
              (shiftHashChain program chain) invocation row =
            CompactRows.renameRow (shiftColumn program)
              (instantiateRow basePackage chain invocation row) := by
  intro chain chainMember invocation invocationBound row rowMember
  exact instantiateRow_mapColumns_of_within
    (package program) basePackage (shiftHashChain program chain) chain
    invocation (shiftColumn program) 8 592 row
    (canonicalPermutationTemplate_rowWithin row rowMember)
    (shiftedHashColumn program chain
      (canonicalHashChain_private chain chainMember) invocation
      invocationBound)

theorem canonicalPermutationRows
    (program : Lifecycle.Stage1.Application.Program) :
    ∀ invocation ∈ basePackage.permutationInvocations,
      ∀ row ∈ basePackage.permutation.rows,
        instantiateInvocationRow
            (shiftPermutationInvocation program invocation) row =
          CompactRows.renameRow (shiftColumn program)
            (instantiateInvocationRow invocation row) := by
  intro invocation invocationMember row rowMember
  exact instantiateInvocationRow_mapColumns_of_within
    (shiftPermutationInvocation program invocation) invocation
    (shiftColumn program) 8 592 row
    (canonicalPermutationTemplate_rowWithin row rowMember)
    (shiftedInvocationColumn program invocation
      (canonicalPermutationInvocation_witnessBound invocation invocationMember))

/-- Satisfaction of every row in the final per-application package implies
satisfaction of the complete validated prefix under the exact column
pullback. -/
theorem packageRows_imply_basePackage
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (compatible : ShiftCompatible program)
    (holds : (package program).RowsHold env) :
    basePackage.RowsHold (baseEnv program env) := by
  have ordinary :=
    packageRows_imply_baseOrdinaryRows program env holds
  refine ⟨?_, ?_, ?_, ordinary.1, ordinary.2⟩
  · intro chain chainMember invocation invocationBound row rowMember
    have shiftedMember : shiftHashChain program chain ∈
        (package program).hashChains := by
      rw [package_hashChains]
      exact List.mem_map_of_mem chainMember
    have finalRow :
        (instantiateRow (package program)
          (shiftHashChain program chain) invocation row
        ).Holds env := by
      exact holds.1 _ shiftedMember invocation invocationBound row (by
        rw [package_permutation]
        exact rowMember)
    rw [compatible.hashRows chain chainMember invocation invocationBound row
      rowMember] at finalRow
    exact (CompactRows.renameRow_holds
      (shiftColumn program)
      (instantiateRow basePackage chain invocation row) env).mp
        finalRow
  · intro invocation invocationMember row rowMember
    have shiftedMember :
        shiftPermutationInvocation program invocation ∈
          (package program).permutationInvocations := by
      rw [package_permutationInvocations]
      exact List.mem_map_of_mem invocationMember
    have finalRow :
        (instantiateInvocationRow
          (shiftPermutationInvocation program invocation) row
        ).Holds env := by
      exact holds.2.1 _ shiftedMember row (by
        rw [package_permutation]
        exact rowMember)
    rw [compatible.permutationRows invocation invocationMember row rowMember]
      at finalRow
    exact (CompactRows.renameRow_holds
      (shiftColumn program)
      (instantiateInvocationRow invocation row) env).mp finalRow
  · intro invocation invocationMember
    have shiftedMember :
        shiftCompactRowInvocation program invocation ∈
          (package program).compactRowInvocations := by
      rw [package_compactRowInvocations]
      exact List.mem_map_of_mem invocationMember
    have finalRows := holds.2.2.1 _ shiftedMember
    unfold CompactRowInvocationHolds at finalRows ⊢
    rw [package_compactRowTemplates] at finalRows
    cases templateEquation :
        basePackage.compactRowTemplates[
          invocation.templateIndex]? with
    | none =>
        simpa [shiftCompactRowInvocation, templateEquation]
          using finalRows
    | some template =>
        simp only
        unfold Layout.R1CS.RowsHold
        intro instantiatedRow instantiatedMember
        obtain ⟨row, rowMember, rfl⟩ := List.mem_map.mp instantiatedMember
        have shiftedRows :
            Layout.R1CS.RowsHold env
              (template.rows.map (instantiateCompactRow
                (shiftCompactRowInvocation program invocation))) := by
          simpa [shiftCompactRowInvocation, templateEquation]
            using finalRows
        have finalRow :
            (instantiateCompactRow
              (shiftCompactRowInvocation program invocation)
              row).Holds env := by
          exact shiftedRows _ (List.mem_map_of_mem rowMember)
        rw [compatible.compactRows invocation invocationMember template
          templateEquation row rowMember] at finalRow
        exact (CompactRows.renameRow_holds
          (shiftColumn program)
          (instantiateCompactRow invocation row) env).mp finalRow

end NightstreamFPrime.Export.Stage1.PerApplicationPreservation
