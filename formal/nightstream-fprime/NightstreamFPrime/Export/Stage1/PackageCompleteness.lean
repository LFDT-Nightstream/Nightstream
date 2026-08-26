import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Export.Stage1.PiCCSCompleteness

/-!
Owns the package-level constructive assembler for the exact production
PiCCS v1_1 phase. It composes the closed transcript and arithmetic proof
packets and adds no row or alternate verifier path.
-/

namespace NightstreamFPrime.Export.Stage1.PackageCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSCompleteness

theorem preOutputIntervalEnd_eq :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.statementWitnessStart +
        (PiCCSInvocations.invocationCeiling -
          NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            PiCCSInvocations.statementWitnessStart) =
      PiCCSInvocations.invocationCeiling := by
  have startLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        PiCCSInvocations.statementWitnessStart := by
    unfold PiCCSInvocations.statementWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
  have startStrict : PiCCSInvocations.statementWitnessStart <
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase := by
    unfold PiCCSInvocations.statementWitnessStart
    rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  have mapped :=
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
      PiCCSInvocations.statementWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase startLocal
      startStrict).le
  change NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.statementWitnessStart +
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.statementWitnessStart) =
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
  omega

theorem pullback_after_preOutput_agreesBelow
    (before after : Env)
    (agrees : AgreesOutside before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.statementWitnessStart)
      (PiCCSInvocations.invocationCeiling -
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.statementWitnessStart)) :
    ∀ column, column < NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset →
      NightstreamFPrime.Layout.Stage1.Spartan.pullback after column =
        NightstreamFPrime.Layout.Stage1.Spartan.pullback before column := by
  intro column below
  unfold NightstreamFPrime.Layout.Stage1.Spartan.pullback
  apply agrees
  have belowStatement : column < PiCCSInvocations.statementWitnessStart := by
    simpa [PiCCSInvocations.statementWitnessStart,
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart] using
      below
  rcases
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_before_piCcsLocal
        column PiCCSInvocations.statementWitnessStart (by
          unfold PiCCSInvocations.statementWitnessStart
          rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
          norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
        belowStatement with mappedBefore | mappedPublic
  · exact Or.inl mappedBefore
  · exact Or.inr (by
      rw [preOutputIntervalEnd_eq]
      exact Nat.le_trans PiCCSInvocations.invocationCeiling_le_private
        mappedPublic.le)

structure PiCCSRowsHold (env : Env) : Prop where
  invocations : ∀ invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits,
    PermutationInvocationHolds (Data.circuitPackage ()) invocation env
  arithmetic : R1CS.RowsHold env
    ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits).map
      Rows.CompiledRow.toR1CS)

/-- Exact phase-local row packets assemble into the authoritative Stage 1
package predicate. Every premise is indexed by a canonical `Data` list. -/
theorem rowsHold_of_packets
    (env : Env)
    (pilotChains : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      HashChainHolds (Data.circuitPackage ()) chain env)
    (piCcsInvocations : ∀ invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits,
      PermutationInvocationHolds (Data.circuitPackage ()) invocation env)
    (piRlcInvocations : ∀ invocation ∈
      PiRLCSamplerInvocations.invocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits),
      PermutationInvocationHolds (Data.circuitPackage ()) invocation env)
    (first54Invocations : ∀ invocation ∈
      PiRLCFirst54Invocations.invocations,
      CompactRowInvocationHolds (Data.circuitPackage ()) invocation env)
    (combinationInvocations : ∀ invocation ∈
      PiRLCCombinationInvocations.invocations,
      CompactRowInvocationHolds (Data.circuitPackage ()) invocation env)
    (pilotAssertions : ∀ row ∈
      Data.liftPilotRows (PilotData.assertionRows ()), row.Holds env)
    (piCcsArithmetic : R1CS.RowsHold env
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth
        Data.publicFits).map Rows.CompiledRow.toR1CS))
    (piRlcArithmetic : R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS)) :
    (Data.circuitPackage ()).RowsHold env := by
  have ordinary :=
    NightstreamFPrime.Export.Stage1.Package.phaseArithmeticRows_imply_packageOrdinary
      env piCcsArithmetic piRlcArithmetic
  refine ⟨?_, ?_, ?_, ordinary.1, ?_⟩
  · intro chain member
    rw [Data.circuitPackage_hashChains] at member
    exact pilotChains chain member
  · intro invocation member
    rw [Data.circuitPackage_permutationInvocations,
      Data.components_permutationInvocations,
      Data.permutationInvocations_eq, List.mem_append] at member
    rcases member with member | member
    · exact piCcsInvocations invocation member
    · exact piRlcInvocations invocation member
  · intro invocation member
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq, List.mem_append] at member
    rcases member with member | member
    · exact first54Invocations invocation member
    · exact combinationInvocations invocation member
  · intro row member
    rw [Data.circuitPackage_assertionRows] at member
    unfold Data.Components.assertionRows at member
    rw [List.mem_append] at member
    rcases member with member | member
    · exact pilotAssertions row member
    · exact ordinary.2 row member

private theorem agreesOutside_widen
    {before after : Env} {start length innerStart innerLength : Nat}
    (inner : AgreesOutside before after innerStart innerLength)
    (starts : start ≤ innerStart)
    (ends : innerStart + innerLength ≤ start + length) :
    AgreesOutside before after start length := by
  intro index outside
  apply inner index
  rcases outside with beforeStart | afterEnd
  · exact Or.inl (lt_of_lt_of_le beforeStart starts)
  · exact Or.inr (Nat.le_trans ends afterEnd)

private theorem agreesOutside_trans
    {before middle after : Env} {start length : Nat}
    (left : AgreesOutside before middle start length)
    (right : AgreesOutside middle after start length) :
    AgreesOutside before after start length := by
  intro index outside
  exact (right index outside).trans (left index outside)

private theorem piCcsPrivateEnd_eq :
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart +
        (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
          NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) =
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount := by
  norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
    NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount]

theorem complete_piCcsRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (phase : Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) template) :
    ∃ completed,
      AgreesOutside env completed
          NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart
          (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
            NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) ∧
        PiCCSRowsHold completed := by
  rcases complete_preOutputInvocations Data.logicalWidth Data.publicFits
      relation env with
    ⟨afterPre, preAgrees, _preExact, preHolds⟩
  have transcripts := preOutputInvocations_imply_specs Data.logicalWidth
    Data.publicFits relation afterPre preHolds
  let parent := PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits
  let initialAssumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation parent
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
        Data.logicalWidth Data.publicFits)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  have acceptedAfterPre := Formal.CompletenessSupport.accepted_of_agree_below
    relation ajtai parent
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterPre) template
    initialAssumptions.external (fun index below =>
      (pullback_after_preOutput_agreesBelow env afterPre preAgrees index
        below).symm) phase.accepted
  rcases complete_arithmeticLogical relation ajtai template afterPre
      acceptedAfterPre transcripts with
    ⟨logical, logicalOperations, logicalEnd⟩
  let afterLogical :=
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval afterPre
      logical.current PiCCSArithmetic.initialClaimLogicalStart
      arithmeticLogicalLength
  have logicalPullback :
      NightstreamFPrime.Layout.Stage1.Spartan.pullback afterLogical =
        logical.current := by
    exact copyArithmetic_pullback_eq afterPre logical logicalEnd
  have preHoldsAfterLogical := preOutputHolds_after_copyArithmetic relation
    afterPre logical.current preHolds
  rcases complete_outputInvocations relation afterLogical with
    ⟨afterOutput, outputAgrees, _outputExact, outputHolds⟩
  have preHoldsAfterOutput := preOutputHolds_after_output relation afterLogical
    afterOutput outputAgrees preHoldsAfterLogical
  have logicalHolds := arithmeticLogicalHolds_after_output relation afterLogical
    afterOutput logical logicalOperations logicalEnd logicalPullback outputAgrees
  have phaseBeforeInitial :
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ≤
        PiCCSArithmetic.initialClaimLogicalStart := by
    unfold PiCCSArithmetic.initialClaimLogicalStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
    omega
  have phaseBeforeOutput :
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ≤
        PiCCSInvocations.outputWitnessStart := by
    exact Nat.le_trans phaseBeforeInitial (by
      rw [← arithmeticLogicalEnd_eq]
      omega)
  have stateAgrees : ∀ index,
      index < NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset →
        NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput index =
          NightstreamFPrime.Layout.Stage1.Spartan.pullback env index := by
    intro index below
    calc
      NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput index =
          NightstreamFPrime.Layout.Stage1.Spartan.pullback afterLogical index :=
        pullback_after_output_agreesBelow afterLogical afterOutput outputAgrees
          index (lt_of_lt_of_le below phaseBeforeOutput)
      _ = logical.current index := congrFun logicalPullback index
      _ = NightstreamFPrime.Layout.Stage1.Spartan.pullback afterPre index :=
        logical.agrees index (Or.inl
          (lt_of_lt_of_le below phaseBeforeInitial))
      _ = NightstreamFPrime.Layout.Stage1.Spartan.pullback env index :=
        pullback_after_preOutput_agreesBelow env afterPre preAgrees index below
  have initialStateAssumptions : StateBinding.Assumptions
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface Data.logicalWidth
          Data.publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    exact initialAssumptions.statementBinding
  have initialStateSpec : StateBinding.SpecHolds
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface Data.logicalWidth
          Data.publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    exact phase.stateBinding
  have stateAtOutput : StateBinding.SpecHolds
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface Data.logicalWidth
          Data.publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput) := by
    exact StateBinding.specHolds_of_agree_below
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface Data.logicalWidth
          Data.publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
      initialStateAssumptions stateAgrees initialStateSpec
  have stateLogicalHolds : ConstraintsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
      (PiCCSArithmetic.statementBindingConstraints Data.logicalWidth
        Data.publicFits) :=
    statementBindingConstraints_hold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
      stateAtOutput
  have outputLeFresh : PiCCSInvocations.outputWitnessStart ≤
      PiCCSArithmetic.initialClaimFreshStart := by
    unfold PiCCSArithmetic.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart
    rw [← PiCCSInvocations.outputEnd_eq_logicalFreshBase Data.logicalWidth
      Data.publicFits]
    omega
  have packetScope : ∀ expression ∈
      packetConstraints Data.logicalWidth Data.publicFits,
      expression.VarsBelow PiCCSArithmetic.initialClaimFreshStart := by
    intro expression member
    have operationMember : expression ∈ flatConstraints logical.operations := by
      rw [logicalOperations, arithmeticLogicalOps_constraints relation]
      exact member
    exact Expr.VarsBelow.mono expression
      (by
        have below := logical.scope expression operationMember
        rw [logicalEnd] at below
        exact below)
      outputLeFresh
  have stateAssumptionsAtOutput : StateBinding.Assumptions
      (Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface Data.logicalWidth
          Data.publicFits)).state
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput) :=
    Formal.CompletenessSupport.stateBindingAssumptionsAt
      initialStateAssumptions
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
  have stateScope := statementBindingConstraints_varsBelow
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
    stateAssumptionsAtOutput
  have phaseLeFresh :
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ≤
        PiCCSArithmetic.initialClaimFreshStart :=
    Nat.le_trans phaseBeforeOutput outputLeFresh
  have emittedScope : ∀ expression ∈
      emittedConstraints Data.logicalWidth Data.publicFits,
      expression.VarsBelow PiCCSArithmetic.initialClaimFreshStart := by
    intro expression member
    rw [emittedConstraints, List.mem_append] at member
    rcases member with stateMember | packetMember
    · exact Expr.VarsBelow.mono expression
        (stateScope expression stateMember) phaseLeFresh
    · exact packetScope expression packetMember
  have emittedHolds : ConstraintsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback afterOutput)
      (emittedConstraints Data.logicalWidth Data.publicFits) := by
    intro expression member
    rw [emittedConstraints, List.mem_append] at member
    rcases member with stateMember | packetMember
    · exact stateLogicalHolds expression stateMember
    · exact logicalHolds expression packetMember
  rcases complete_arithmeticRows relation afterOutput emittedScope emittedHolds with
    ⟨completed, physicalAgrees, arithmeticHolds⟩
  have allBefore := (PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits relation).2
  have stableInputs := schedule_stableInputs
    (PiCCSInvocations.invocations_scheduleWithin Data.logicalWidth
      Data.publicFits relation).1
  have physicalEnd : PiCCSInvocations.invocationCeiling + 685348 ≤
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount := by
    rw [PiCCSInvocations.invocationCeiling_eq,
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount_eq]
    norm_num
  have allHoldsBefore : ∀ invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        afterOutput := by
    intro invocation member
    rw [PiCCSInvocations.invocations, List.mem_append] at member
    rcases member with preMember | outputMember
    · exact preHoldsAfterOutput invocation preMember
    · exact outputHolds invocation outputMember
  have allHoldsAfter : ∀ invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        completed := by
    intro invocation member
    apply NightstreamFPrime.Export.Pilot.permutationInvocationHolds_of_agreesOutside
      invocation afterOutput completed PiCCSInvocations.invocationCeiling
        685348
    · intro lane term termMember
      rcases stableInputs invocation member lane term termMember with
        inputBefore | inputPublic
      · have endBefore := allBefore invocation member
        have startBefore : invocation.witnessStart ≤
            PiCCSInvocations.invocationCeiling := by
          omega
        exact Or.inl (lt_of_lt_of_le inputBefore startBefore)
      · exact Or.inr (Nat.le_trans physicalEnd inputPublic)
    · intro index below
      exact Or.inl (lt_of_lt_of_le (by omega) (allBefore invocation member))
    · exact physicalAgrees
    · exact allHoldsBefore invocation member
  have broadStart :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          PiCCSInvocations.statementWitnessStart =
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart := by
    rfl
  have preBroad : AgreesOutside env afterPre
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) := by
    apply agreesOutside_widen preAgrees
    · rw [broadStart]
    · rw [preOutputIntervalEnd_eq, piCcsPrivateEnd_eq]
      exact PiCCSInvocations.invocationCeiling_le_private
  have logicalAgrees :=
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval_agreesOutside
      afterPre logical.current PiCCSArithmetic.initialClaimLogicalStart
      arithmeticLogicalLength
  have logicalBroad : AgreesOutside afterPre afterLogical
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) := by
    apply agreesOutside_widen logicalAgrees
    · rw [← broadStart]
      exact
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
          PiCCSInvocations.statementWitnessStart
          PiCCSArithmetic.initialClaimLogicalStart (by
            unfold PiCCSInvocations.statementWitnessStart
            rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
            norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
          (by
            unfold PiCCSInvocations.statementWitnessStart
              PiCCSArithmetic.initialClaimLogicalStart
              NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart
              NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart
              NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart
              NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart
            omega)).le
    · rw [arithmeticMappedEnd_eq, piCcsPrivateEnd_eq]
      exact Nat.le_trans arithmeticMappedEnd_le_invocationCeiling
        PiCCSInvocations.invocationCeiling_le_private
  have outputBroad : AgreesOutside afterLogical afterOutput
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) := by
    apply agreesOutside_widen outputAgrees
    · rw [← broadStart]
      exact
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
          PiCCSInvocations.statementWitnessStart
          PiCCSInvocations.outputWitnessStart (by
            unfold PiCCSInvocations.statementWitnessStart
            rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
            norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
          (by
            unfold PiCCSInvocations.statementWitnessStart
              PiCCSInvocations.outputWitnessStart
            rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq,
              NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingWitnessStart_eq]
            norm_num)).le
    · rw [outputIntervalEnd_eq, piCcsPrivateEnd_eq]
      exact PiCCSInvocations.invocationCeiling_le_private
  have physicalBroad : AgreesOutside afterOutput completed
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount -
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart) := by
    apply agreesOutside_widen physicalAgrees
    · rw [← broadStart]
      exact
        (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt_of_piCcsLocal
          PiCCSInvocations.statementWitnessStart
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase (by
            unfold PiCCSInvocations.statementWitnessStart
            rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
            norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset])
          (by
            unfold PiCCSInvocations.statementWitnessStart
            rw [NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq]
            norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq])).le
    · rw [piCcsPrivateEnd_eq]
      exact physicalEnd
  have totalBroad := agreesOutside_trans
    (agreesOutside_trans (agreesOutside_trans preBroad logicalBroad)
      outputBroad) physicalBroad
  refine ⟨completed, totalBroad, ⟨?_, ?_⟩⟩
  · intro invocation member
    have held := allHoldsAfter invocation member
    unfold PermutationInvocationHolds at held ⊢
    rw [Data.circuitPackage_permutation]
    simpa [PilotData.circuitPackage] using held
  · exact arithmeticHolds

end NightstreamFPrime.Export.Stage1.PackageCompleteness
