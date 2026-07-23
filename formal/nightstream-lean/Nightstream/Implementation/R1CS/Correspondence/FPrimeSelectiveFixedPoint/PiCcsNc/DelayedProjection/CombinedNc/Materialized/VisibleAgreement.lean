import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PhysicalAgreement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSoundness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveObligations
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleDependency

/-!
Ordered agreement for the 1,689 visible combined-NC outputs.

Owns: strong-induction transport from exact selected rewrite obligations to
agreement between deterministic source reconstruction and compiler execution
on the 748 physical outputs and 941 rewrite-terminal pivots.

Does not own: retained-check transfer, source-row satisfaction, protocol
acceptance, transcript authority, raw-child authority, commitment binding,
costs, or permission to remove rows.

This leaf contains no executable certificate.  Every dependency is supplied
by `VisibleDependency` as a literal compiler input or visible output strictly
below the current target.  Source inputs use deterministic reconstruction;
visible dependencies use the strong-induction hypothesis.

Assurance tier: artifact-checked for the fixed generated production profile
after this leaf and its focused parent validate.
-/

/-!
Emits constraints: none; this module proves agreement of existing visible columns.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.visible_agreement` | Show visible compiler columns equal their authoritative source-assignment values. | direct dataflow |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleAgreement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open SelectiveCompilerBridge
open RewriteChain
open RewriteBlockSemantics
open RewriteBlockSoundness

private abbrev sourceView (assignment : Nat → Nat) : Nat → Nat :=
  PhysicalAgreement.reconstructedAssignment assignment

private abbrev compilerView (assignment : Nat → Nat) : Nat → Nat :=
  SourceAssignment.compilerAssignment assignment

private def VisibleColumn (column : Nat) : Prop :=
  column ∈ SourceDisposition.physicalDefinitionOutputs ∨
    column ∈ SourceDisposition.terminalPivotColumns

/-! ## Strictly-earlier dependency transport -/

private theorem allowedEarlier_agrees
    {assignment : Nat → Nat} {target column : Nat}
    (inductionHypothesis : ∀ earlier, earlier < target →
      VisibleColumn earlier →
      sourceView assignment earlier = compilerView assignment earlier)
    (allowed : VisibleDependency.AllowedEarlier target column) :
    sourceView assignment column = compilerView assignment column := by
  rcases allowed with ⟨sourceOrPhysical, earlier⟩
  rcases sourceOrPhysical with compilerInput | physicalOutput
  · rcases
        SourceDisposition.physicalInputColumns_subset_sourceInputOrPivots
          column compilerInput with sourceInput | pivot
    · exact PhysicalAgreement.inputAgreement assignment column sourceInput
    · exact inductionHypothesis column earlier (Or.inr pivot)
  · exact inductionHypothesis column earlier (Or.inl physicalOutput)

/-! ## Physical compiler definitions -/

private theorem physicalOutput_agrees
    {assignment : Nat → Nat} (constantOne : assignment 0 = 1)
    {target : Nat}
    (targetMember :
      target ∈ SourceDisposition.physicalDefinitionOutputs)
    (inductionHypothesis : ∀ earlier, earlier < target →
      VisibleColumn earlier →
      sourceView assignment earlier = compilerView assignment earlier) :
    sourceView assignment target = compilerView assignment target := by
  rcases List.mem_map.mp targetMember with
    ⟨definition, definitionMember, targetExact⟩
  subst target
  rcases SourceDisposition.physicalDefinitions_refine_source definition
      definitionMember with
    ⟨source, sourceMember, sourceOutput, rowEquivalent,
      physicalCanonical⟩
  have sourceProjectionMember : source ∈
      definitions StageProgram.instructions := by
    rw [← SourceExecution.sourceDefinitions_eq_stageProjection]
    exact sourceMember
  have sourceHolds := SourceExecution.reconstruct_definitionsHold
    (compilerView assignment) source sourceProjectionMember
  have sourceCanonical :=
    StageProgram.definitions_canonical source sourceProjectionMember
  have sourceBuilderHolds :
      RowHolds (sourceView assignment) source.builderRow := by
    exact Program.builderDefinition_complete
      (PhysicalAgreement.reconstructed_canonical assignment)
      (PhysicalAgreement.reconstructed_constantOne constantOne)
      source sourceCanonical sourceHolds
  have physicalBuilderHolds :
      RowHolds (sourceView assignment) definition.builderRow :=
    ProjectionIndexedRows.rowHolds_of_permutationEquivalent rowEquivalent
      sourceBuilderHolds
  have reconstructedPhysicalHolds :
      Definition.Holds (sourceView assignment) definition := by
    exact Program.builderDefinition_sound
      (PhysicalAgreement.reconstructed_canonical assignment)
      (PhysicalAgreement.reconstructed_constantOne constantOne)
      definition physicalCanonical physicalBuilderHolds
  have compilerHolds :=
    CompilerExecution.compilerAssignment_definitionsHold assignment
      definition (by
        rw [← CompilerExecution.compilerDefinitionPhases_exact]
        exact List.mem_append_right _ definitionMember)
  apply AssignmentAgreement.definitionOutput_eq_of_holds
    (known := definition.rhs.refs)
  · intro column reference
    exact allowedEarlier_agrees inductionHypothesis
      (VisibleDependency.physicalReference_allowedEarlier definitionMember
        reference)
  · intro column reference
    exact reference
  · exact reconstructedPhysicalHolds
  · exact compilerHolds

/-! ## Batch-target recovery -/

private theorem mapM_some_member {Alpha Beta : Type}
    (decode : Alpha → Option Beta) :
    ∀ {inputs : List Alpha} {outputs : List Beta},
      inputs.mapM decode = some outputs →
      ∀ output ∈ outputs,
        ∃ input ∈ inputs, decode input = some output := by
  intro inputs
  induction inputs with
  | nil =>
      intro outputs decoded
      simp at decoded
      subst outputs
      simp
  | cons head tail inductionHypothesis =>
      intro outputs decoded
      cases headResult : decode head with
      | none => simp [headResult] at decoded
      | some decodedHead =>
          cases tailResult : tail.mapM decode with
          | none => simp [headResult, tailResult] at decoded
          | some decodedTail =>
              simp [headResult, tailResult] at decoded
              subst outputs
              intro output member
              simp only [List.mem_cons] at member
              rcases member with rfl | tailMember
              · exact ⟨head, by simp, headResult⟩
              · rcases inductionHypothesis tailResult output tailMember with
                  ⟨input, inputMember, inputDecodes⟩
                exact ⟨input, by simp [inputMember], inputDecodes⟩

private theorem batchTarget_member
    {batch : RewriteBatchIndex.Batch} {target : Nat}
    (member : target ∈ (batchTargetColumns? batch).getD []) :
    ∃ raws chain,
      rawStepsFor? batch = some raws ∧
      chain ∈ partitionChains raws ∧
      rawChainTarget? chain = some target := by
  cases rawsResult : rawStepsFor? batch with
  | none => simp [batchTargetColumns?, batchChains?, rawsResult] at member
  | some raws =>
      cases targetsResult :
          (partitionChains raws).mapM rawChainTarget? with
      | none =>
          simp [batchTargetColumns?, batchChains?, rawsResult,
            targetsResult] at member
      | some targets =>
          have targetMember : target ∈ targets := by
            simpa [batchTargetColumns?, batchChains?, rawsResult,
              targetsResult] using member
          rcases mapM_some_member rawChainTarget? targetsResult target
              targetMember with ⟨chain, chainMember, targetExact⟩
          exact ⟨raws, chain, rfl, chainMember, targetExact⟩

private theorem generatedTarget_member
    {target : Nat}
    (member : target ∈ SourceDisposition.terminalPivotColumns) :
    ∃ batch ∈ RewriteBatchIndex.allBatches,
      target ∈ (batchTargetColumns? batch).getD [] := by
  have generatedMember : target ∈ generatedChainTargetColumns := by
    have schedule := generatedTargetSchedule_exact
    unfold TargetScheduleMatches at schedule
    rw [schedule]
    exact member
  unfold generatedChainTargetColumns at generatedMember
  exact List.mem_flatMap.mp generatedMember

private theorem chainRaw_generated
    {batch : RewriteBatchIndex.Batch} {raws chain : List RawRewriteStep}
    (rawsExact : rawStepsFor? batch = some raws)
    (chainMember : chain ∈ partitionChains raws) :
    ∀ raw ∈ chain, raw ∈ Provenance.rewriteSteps := by
  intro raw rawMember
  have generated := generatedRawMember_of_rawStepsFor rawsExact raw (by
    rw [← partitionChains_flatten raws]
    exact List.mem_flatten.mpr ⟨chain, chainMember, rawMember⟩)
  exact generated

/-! ## Decoded recurrence transport -/

private structure ChainCompilerFacts
    (assignment : Nat → Nat) (target : Nat)
    (raws : List RawRewriteStep)
    (decoded :
      List (DecodedRewriteStep Metadata.sourceRelationColumns)) : Prop where
  compilerHolds : ∀ step ∈ decoded,
    RewriteStepHolds (compilerView assignment)
      (SourceAssignment.derivedValue assignment) step
  contributionsAgree : ∀ step ∈ decoded,
    contribution (sourceView assignment) step =
      contribution (compilerView assignment) step

private theorem decodedSteps_compilerFacts
    {assignment : Nat → Nat} {target : Nat}
    {raws : List RawRewriteStep}
    {decoded :
      List (DecodedRewriteStep Metadata.sourceRelationColumns)}
    (decodes : DecodedSteps raws decoded)
    (generated : ∀ raw ∈ raws, raw ∈ Provenance.rewriteSteps)
    (dependencies : ∀ raw ∈ raws,
      VisibleDependency.StepDependencies target raw)
    (rewriteObligations :
      SelectiveObligations.GeneratedRewriteObligationsHold assignment)
    (earlierAgreement : ∀ column,
      VisibleDependency.AllowedEarlier target column →
        sourceView assignment column = compilerView assignment column) :
    ChainCompilerFacts assignment target raws decoded := by
  induction decodes with
  | nil =>
      exact ⟨by simp, by simp⟩
  | @cons raw decodedStep raws decodedSteps headDecodes tailDecodes
      inductionHypothesis =>
      have rawGenerated := generated raw (by simp)
      rcases rewriteObligations raw rawGenerated with
        ⟨obligationStep, obligationDecodes, obligationHolds⟩
      have decodedExact : obligationStep = decodedStep := by
        exact Option.some.inj (obligationDecodes.symm.trans headDecodes)
      subst obligationStep
      have headAgreement : AgreeOn (sourceView assignment)
          (compilerView assignment) (rawContributionReferences raw) := by
        intro column reference
        exact earlierAgreement column
          (dependencies raw (by simp) column reference)
      have typedReferences := contributionReferencesOnly_of_raw
        headDecodes (fun _column reference => reference)
      have headContribution :=
        AssignmentAgreement.contribution_eq_of_agreeOn
          headAgreement typedReferences
      have tailFacts := inductionHypothesis
        (fun candidate member => generated candidate (by simp [member]))
        (fun candidate member => dependencies candidate (by simp [member]))
      constructor
      · intro step member
        simp only [List.mem_cons] at member
        rcases member with rfl | member
        · exact obligationHolds
        · exact tailFacts.compilerHolds step member
      · intro step member
        simp only [List.mem_cons] at member
        rcases member with rfl | member
        · exact headContribution
        · exact tailFacts.contributionsAgree step member

private theorem decodedSteps_unique
    {raws : List RawRewriteStep}
    {left right :
      List (DecodedRewriteStep Metadata.sourceRelationColumns)}
    (leftDecodes : DecodedSteps raws left)
    (rightDecodes : DecodedSteps raws right) : left = right := by
  induction leftDecodes generalizing right with
  | nil =>
      cases rightDecodes
      rfl
  | @cons raw leftHead raws leftTail headDecodes tailDecodes
      inductionHypothesis =>
      cases rightDecodes with
      | @cons _ rightHead _ rightTail rightHeadDecodes rightTailDecodes =>
          have headExact : leftHead = rightHead :=
            Option.some.inj (headDecodes.symm.trans rightHeadDecodes)
          have tailExact := inductionHypothesis rightTailDecodes
          rw [headExact, tailExact]

private theorem sourceChain_output_unique
    {previous : Option Nat}
    {steps : List (DecodedRewriteStep Metadata.sourceRelationColumns)}
    {left right : DecodedLinearCombination Metadata.sourceRelationColumns}
    (leftChain : SourceChain previous steps left)
    (rightChain : SourceChain previous steps right) : left = right := by
  induction leftChain generalizing right with
  | @terminal previous step left previousExact leftOutput =>
      cases rightChain with
      | @terminal _ _ right rightPrevious rightOutput =>
          exact DecodedRewriteOutput.source.inj
            (leftOutput.symm.trans rightOutput)
      | @derived _ _ compilerIndex rest right rightPrevious rightOutput tail =>
          cases tail
  | @derived previous step compilerIndex rest left previousExact leftOutput
      tail inductionHypothesis =>
      cases rightChain with
      | @terminal _ _ right rightPrevious rightOutput =>
          cases tail
      | @derived _ _ rightIndex _ right rightPrevious rightOutput rightTail =>
          have indexExact : compilerIndex = rightIndex :=
            DecodedRewriteOutput.derivedProductSum.inj
              (leftOutput.symm.trans rightOutput)
          subst rightIndex
          exact inductionHypothesis rightTail

/-! ## Terminal output transport -/

private theorem rawTriangularAt_tail_of_target
    {rawOutput : RawLinearCombination} {target : Nat}
    (targetExact : rawTargetColumn? rawOutput = some target) :
    RawTriangularAt
      ((rawOutput.terms.drop 1).map fun term => term.column)
      rawOutput target := by
  unfold rawTargetColumn? at targetExact
  split at targetExact
  next constantZero =>
    cases termsExact : rawOutput.terms with
    | nil => simp [termsExact] at targetExact
    | cons head tail =>
        rcases head with ⟨column, coefficient⟩
        cases coefficient with
        | zero => simp [termsExact] at targetExact
        | succ coefficient =>
            cases coefficient with
            | zero =>
                simp [termsExact] at targetExact
                subst column
                refine ⟨constantZero, tail, termsExact, ?_⟩
                intro term member
                simp only [termsExact, List.drop_cons, List.drop_zero,
                  List.mem_map]
                exact ⟨term, member, rfl⟩
            | succ coefficient => simp [termsExact] at targetExact
  next constantNonzero => simp at targetExact

private theorem target_agrees_of_typedChain
    {assignment : Nat → Nat} {target : Nat}
    {chain : List RawRewriteStep} {data : TypedSourceChainData}
    (typed : TypedSourceChain chain data)
    (dependencies : ∃ rawOutput dependencyTarget,
      VisibleDependency.PivotChainDependencies chain rawOutput
        dependencyTarget)
    (targetExact : rawChainTarget? chain = some target)
    (rewriteObligations :
      SelectiveObligations.GeneratedRewriteObligationsHold assignment)
    (generated : ∀ raw ∈ chain, raw ∈ Provenance.rewriteSteps)
    (earlierAgreement : ∀ column,
      VisibleDependency.AllowedEarlier target column →
        sourceView assignment column = compilerView assignment column)
    (sourceEquation :
      ∀ (decoded : List
          (DecodedRewriteStep Metadata.sourceRelationColumns))
        (output : DecodedLinearCombination Metadata.sourceRelationColumns),
        DecodedSteps chain decoded →
        SourceChain none decoded output →
        linearCombinationValue output (sourceView assignment) =
          contributionSum (sourceView assignment) decoded) :
    sourceView assignment target = compilerView assignment target := by
  rcases data with ⟨decoded, output, decodedRawOutput⟩
  rcases typed with
    ⟨decodes, sourceChain, decodedRawOutputExact, outputDecodes⟩
  rcases dependencies with ⟨rawOutput, dependencyTarget, dependencies⟩
  rcases dependencies with
    ⟨rawOutputExact,
      dependencyTargetExact, stepsEarlier, tailEarlier, _tailLength⟩
  have dependencyTarget_eq : dependencyTarget = target := by
    rw [targetExact] at dependencyTargetExact
    exact (Option.some.inj dependencyTargetExact).symm
  subst dependencyTarget
  have rawOutput_eq : decodedRawOutput = rawOutput := by
    rw [rawOutputExact] at decodedRawOutputExact
    exact (Option.some.inj decodedRawOutputExact).symm
  subst decodedRawOutput
  have rawTarget : rawTargetColumn? rawOutput = some target := by
    simpa [rawChainTarget?, rawOutputExact] using targetExact
  have compilerFacts := decodedSteps_compilerFacts decodes generated
    stepsEarlier rewriteObligations earlierAgreement
  have contributionSums :
      contributionSum (sourceView assignment) decoded =
        contributionSum (compilerView assignment) decoded :=
    RewriteChain.contributionSum_congr
      compilerFacts.contributionsAgree
  have compilerValue :=
    RewriteChain.sourceValue_eq_previous_add_contributions sourceChain
      compilerFacts.compilerHolds
  have valueEqual :
      linearCombinationValue output (sourceView assignment) =
        linearCombinationValue output (compilerView assignment) := by
    calc
      linearCombinationValue output (sourceView assignment) =
          contributionSum (sourceView assignment) decoded :=
        sourceEquation decoded output decodes sourceChain
      _ = contributionSum (compilerView assignment) decoded :=
        contributionSums
      _ = linearCombinationValue output (compilerView assignment) := by
        simpa [rewritePreviousValue] using compilerValue.symm
  have triangular := typedTriangular_of_decode outputDecodes
    (rawTriangularAt_tail_of_target rawTarget)
  apply typedTriangular_target_eq_of_value_eq triangular
  · intro column member
    rcases List.mem_map.mp member with ⟨term, termMember, rfl⟩
    exact earlierAgreement term.column (tailEarlier term termMember)
  · exact PhysicalAgreement.reconstructed_canonical assignment target
  · exact SourceAssignment.compilerAssignmentCanonical assignment target
  · exact valueEqual

/-! ## Five-definition source owners -/

private theorem smallChain_sourceEquation
    {assignment : Nat → Nat} {batch : RewriteBatchIndex.Batch}
    {raws chain : List RawRewriteStep}
    (witness : SmallBatchWitness batch)
    (rawsExact : rawStepsFor? batch = some raws)
    (chainMember : chain ∈ partitionChains raws) :
    ∀ (decoded : List
        (DecodedRewriteStep Metadata.sourceRelationColumns))
      (output : DecodedLinearCombination Metadata.sourceRelationColumns),
      DecodedSteps chain decoded →
      SourceChain none decoded output →
      linearCombinationValue output (sourceView assignment) =
        contributionSum (sourceView assignment) decoded := by
  rcases witness with
    ⟨definitions, witnessRaws, definitionsExact, witnessRawsExact,
      _sourceLength, _definitionCount, _compactExact, _sourceRangesExact,
      _rawCount, decodedMatches, chainsValid, _chainsTriangular⟩
  have witnessRaws_eq : witnessRaws = raws :=
    Option.some.inj (witnessRawsExact.symm.trans rawsExact)
  subst witnessRaws
  have definitionsHold : DefinitionsHold (sourceView assignment)
      definitions := by
    intro definition definitionMember
    exact SourceExecution.reconstruct_definitionsHold
      (compilerView assignment) definition
      (generatedDefinitionMember_of_sourceDefinitionsForBatch
        definitionsExact definition definitionMember)
  intro decoded output decodes sourceChain
  cases chainExact : chain with
  | nil =>
      have invalid := chainsValid [] (by simpa [chainExact] using chainMember)
      contradiction
  | cons raw rest =>
      have rawMember : raw ∈ raws := by
        rw [← partitionChains_flatten raws]
        exact List.mem_flatten.mpr
          ⟨chain, chainMember, by simp [chainExact]⟩
      rcases decodedMatches raw rawMember with
        ⟨matchedStep, matchedOutput, matchedDecodes,
          matchedOutputExact, exactMatch⟩
      subst chain
      cases decodes with
      | @cons _ decodedStep _ decodedTail stepDecodes tailDecodes =>
          have stepExact : decodedStep = matchedStep :=
            Option.some.inj (stepDecodes.symm.trans matchedDecodes)
          subst decodedStep
          cases sourceChain with
          | @terminal _ _ _ previousExact sourceOutputExact =>
              cases tailDecodes
              have outputExact : output = matchedOutput := by
                exact DecodedRewriteOutput.source.inj
                  (sourceOutputExact.symm.trans matchedOutputExact)
              subst output
              exact
                RewriteSourceSemantics.ChainAgreement.exactChainMatch_implies_sourceValue_eq_contributions
                  exactMatch definitionsHold
          | @derived _ _ compilerIndex remaining _ previousExact
              derivedOutputExact tail =>
              rw [matchedOutputExact] at derivedOutputExact
              contradiction

/-! ## Large dot-product source owners -/

private theorem outputDotOwner_trace
    {offset : Nat} {traces : List TerminalProgram.OutputTrace}
    {owner : DotOwnerKey}
    (member : owner ∈ outputDotOwnersFrom offset traces) :
    ∃ trace ∈ traces, owner.trace = trace.evaluation := by
  induction traces generalizing offset with
  | nil => simp [outputDotOwnersFrom] at member
  | cons trace traces inductionHypothesis =>
      simp only [outputDotOwnersFrom, List.mem_cons] at member
      rcases member with ownerExact | tailMember
      · subst owner
        exact ⟨trace, by simp, rfl⟩
      · rcases inductionHypothesis tailMember with
          ⟨found, foundMember, traceExact⟩
        exact ⟨found, by simp [foundMember], traceExact⟩

private theorem outputTraceEvaluationDefinition_mem_terminal
    {trace : TerminalProgram.OutputTrace} {definition : Definition}
    (traceMember : trace ∈ TerminalProgram.outputTraces)
    (definitionMember : definition ∈ trace.evaluation.definitions) :
    definition ∈ TerminalProgram.definitions := by
  have traceDefinition : definition ∈ trace.definitions := by
    simp [TerminalProgram.OutputTrace.definitions, definitionMember]
  have outputDefinition : definition ∈ TerminalProgram.outputDefinitions := by
    unfold TerminalProgram.outputDefinitions
    exact List.mem_flatMap.mpr ⟨trace, traceMember, traceDefinition⟩
  simp [TerminalProgram.definitions, TerminalProgram.prefixDefinitions,
    TerminalProgram.laneComputationDefinitions, outputDefinition]

private theorem dotOwnerDefinition_mem_terminal
    {owner : DotOwnerKey} (ownerMember : owner ∈ dotOwners)
    {definition : Definition}
    (definitionMember : definition ∈ owner.trace.definitions) :
    definition ∈ TerminalProgram.definitions := by
  unfold dotOwners at ownerMember
  rcases List.mem_append.mp ownerMember with outputOwner | finalOwner
  · rcases outputDotOwner_trace outputOwner with
      ⟨trace, traceMember, traceExact⟩
    rw [traceExact] at definitionMember
    exact outputTraceEvaluationDefinition_mem_terminal traceMember
      definitionMember
  · have ownerCases : owner = ordinaryDotOwner ∨ owner = runningDotOwner := by
      simpa only [List.mem_cons, List.mem_singleton, List.not_mem_nil,
        or_false] using finalOwner
    rcases ownerCases with ownerExact | ownerExact
    · subst owner
      change definition ∈ TerminalProgram.ordinarySum.definitions at definitionMember
      simp [TerminalProgram.definitions, TerminalProgram.prefixDefinitions,
        TerminalProgram.laneComputationDefinitions, definitionMember]
    · subst owner
      change definition ∈ TerminalProgram.runningSum.definitions at definitionMember
      simp [TerminalProgram.definitions, TerminalProgram.prefixDefinitions,
        TerminalProgram.suffixDefinitions, TerminalProgram.delayedDefinitions,
        TerminalProgram.delayedPreSelectorDefinitions, definitionMember]

private theorem definition_mem_of_define_mem
    {definition : Definition} {instructions : List Instruction}
    (member : Instruction.define definition ∈ instructions) :
    definition ∈ CheckedProgram.definitions instructions := by
  exact List.mem_filterMap.mpr ⟨.define definition, member, rfl⟩

private theorem terminalDefinition_mem_sourceProgram
    {definition : Definition}
    (member : definition ∈ TerminalProgram.definitions) :
    definition ∈ definitions StageProgram.instructions := by
  apply definition_mem_of_define_mem
  simp [StageProgram.instructions, StageProgram.terminalInstructions, member]

private theorem dotOwnerDefinitionsHold
    {assignment : Nat → Nat} {owner : DotOwnerKey}
    (ownerMember : owner ∈ dotOwners) :
    DefinitionsHold (sourceView assignment) owner.trace.definitions := by
  intro definition member
  exact SourceExecution.reconstruct_definitionsHold
    (compilerView assignment) definition
    (terminalDefinition_mem_sourceProgram
      (dotOwnerDefinition_mem_terminal ownerMember member))

private theorem largeChain_sourceEquation
    {assignment : Nat → Nat} {owner : DotOwnerKey}
    {raws chain : List RawRewriteStep} {data : LargeFactorFoldData}
    (witness : LargeFactorFoldWitness owner raws (sourceView assignment) data)
    (definitionsHold : DefinitionsHold (sourceView assignment)
      owner.trace.definitions)
    (chainMember : chain ∈ partitionChains raws) :
    ∀ (decoded : List
        (DecodedRewriteStep Metadata.sourceRelationColumns))
      (output : DecodedLinearCombination Metadata.sourceRelationColumns),
      DecodedSteps chain decoded →
      SourceChain none decoded output →
      linearCombinationValue output (sourceView assignment) =
        contributionSum (sourceView assignment) decoded := by
  have equations := witness.sourceContributionEquations definitionsHold
  rw [witness.partitionExact] at chainMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at chainMember
  rcases chainMember with chainExact | chainExact | chainExact
  · subst chain
    intro decoded output decodes sourceChain
    have decodedExact : decoded = data.qDecoded :=
      decodedSteps_unique decodes witness.qDecodes
    subst decoded
    have outputExact : output = data.qOutput :=
      sourceChain_output_unique sourceChain witness.qSourceChain
    subst output
    exact equations.1
  · subst chain
    intro decoded output decodes sourceChain
    have decodedExact : decoded = data.c0Decoded :=
      decodedSteps_unique decodes witness.c0Decodes
    subst decoded
    have outputExact : output = data.c0Output :=
      sourceChain_output_unique sourceChain witness.c0SourceChain
    subst output
    exact equations.2.1
  · subst chain
    intro decoded output decodes sourceChain
    have decodedExact : decoded = data.sumDecoded :=
      decodedSteps_unique decodes witness.sumDecodes
    subst decoded
    have outputExact : output = data.sumOutput :=
      sourceChain_output_unique sourceChain witness.sumSourceChain
    subst output
    exact equations.2.2

/-! ## Rewrite-terminal pivots -/

private theorem pivotOutput_agrees
    {assignment : Nat → Nat} {target : Nat}
    (targetMember : target ∈ SourceDisposition.terminalPivotColumns)
    (rewriteObligations :
      SelectiveObligations.GeneratedRewriteObligationsHold assignment)
    (inductionHypothesis : ∀ earlier, earlier < target →
      VisibleColumn earlier →
      sourceView assignment earlier = compilerView assignment earlier) :
    sourceView assignment target = compilerView assignment target := by
  rcases generatedTarget_member targetMember with
    ⟨batch, batchMember, localTargetMember⟩
  rcases batchTarget_member localTargetMember with
    ⟨raws, chain, rawsExact, chainMember, targetExact⟩
  have chainsExact : batchChains? batch = some (partitionChains raws) := by
    simp [batchChains?, rawsExact]
  have dependencies :=
    VisibleDependency.generatedPivotChain_dependencies batchMember
      chainsExact chainMember
  have generated := chainRaw_generated rawsExact chainMember
  have earlierAgreement : ∀ column,
      VisibleDependency.AllowedEarlier target column →
        sourceView assignment column = compilerView assignment column := by
    intro column allowed
    exact allowedEarlier_agrees inductionHypothesis allowed
  cases generatedBatchWitness batchMember with
  | small smallWitness =>
      have sourceEquation := smallChain_sourceEquation
        (assignment := assignment) smallWitness rawsExact chainMember
      rcases smallWitness with
        ⟨definitions, witnessRaws, definitionsExact, witnessRawsExact,
          sourceLength, definitionCount, compactExact, sourceRangesExact,
          rawCount, decodedMatches, chainsValid, chainsTriangular⟩
      have witnessRaws_eq : witnessRaws = raws :=
        Option.some.inj (witnessRawsExact.symm.trans rawsExact)
      subst witnessRaws
      rcases typedSourceChain_of_generated generated
          (chainsValid chain chainMember) with ⟨typedData, typed⟩
      exact target_agrees_of_typedChain typed dependencies targetExact
        rewriteObligations generated earlierAgreement sourceEquation
  | large largeWitness =>
      have factorResult := largeWitness.factorFolds (sourceView assignment)
      rcases largeWitness with
        ⟨witnessOwner, witnessRaws, ownerExact, ownerMember,
          sourceRangeExact, witnessRawsExact, compactExact,
          sourceRangesExact, chainCount, chainLengths, targetColumns,
          chainsValid, chainsTriangular, contributionsExact⟩
      have witnessRaws_eq : witnessRaws = raws :=
        Option.some.inj (witnessRawsExact.symm.trans rawsExact)
      subst witnessRaws
      rcases factorResult with
        ⟨owner, factorRaws, factorData, factorOwnerMember,
          factorSourceRange, factorRawsExact, factorWitness⟩
      have factorRaws_eq : factorRaws = raws :=
        Option.some.inj (factorRawsExact.symm.trans rawsExact)
      subst factorRaws
      have sourceEquation := largeChain_sourceEquation factorWitness
        (dotOwnerDefinitionsHold factorOwnerMember) chainMember
      rcases typedSourceChain_of_generated generated
          (chainsValid chain chainMember) with ⟨typedData, typed⟩
      exact target_agrees_of_typedChain typed dependencies targetExact
        rewriteObligations generated earlierAgreement sourceEquation

/-! ## Public visible agreement -/

/-- Exact generated rewrite obligations plus constant-one enforcement imply
agreement on every physical compiler output and rewrite-terminal pivot. -/
theorem visibleOutputs_agree_of_rewriteObligations
    {assignment : Nat → Nat}
    (rewriteObligations :
      SelectiveObligations.GeneratedRewriteObligationsHold assignment)
    (constantOne : assignment 0 = 1) :
    AgreeOn (sourceView assignment) (compilerView assignment)
      (SourceDisposition.physicalDefinitionOutputs ++
        SourceDisposition.terminalPivotColumns) := by
  have pointwise : ∀ target, VisibleColumn target →
      sourceView assignment target = compilerView assignment target := by
    intro target
    induction target using Nat.strongRecOn with
    | ind target inductionHypothesis =>
        intro visible
        rcases visible with physical | pivot
        · exact physicalOutput_agrees constantOne physical
            inductionHypothesis
        · exact pivotOutput_agrees pivot rewriteObligations
            inductionHypothesis
  intro target member
  exact pointwise target (List.mem_append.mp member)

/-- Literal selected-row satisfaction supplies the rewrite obligations used
by the ordered agreement proof.  Selector-one and constant-one remain
explicit production boundaries. -/
theorem selectedRows_imply_visibleOutputs_agree
    {assignment : Nat → Nat}
    (satisfies :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1) :
    AgreeOn (sourceView assignment) (compilerView assignment)
      (SourceDisposition.physicalDefinitionOutputs ++
        SourceDisposition.terminalPivotColumns) := by
  have obligations :=
    SelectiveObligations.generatedEmittedRowsSatisfy_implies_generatedObligations
      satisfies selectorOne
  exact visibleOutputs_agree_of_rewriteObligations obligations.1 constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleAgreement
