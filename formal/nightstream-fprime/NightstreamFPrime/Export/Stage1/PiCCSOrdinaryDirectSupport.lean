import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSource
import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHornerSupport
import NightstreamFPrime.Gadgets.Polynomial.SparseSupport
import NightstreamFPrime.Layout.R1CS.Support
import NightstreamFPrime.Layout.Stage1.PiCCSInputSupport
import NightstreamFPrime.Layout.Stage1.PiCCSTranscriptSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.StatementBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.SumcheckChain

/-!
Owns retained-source support for the complete canonical PiCCS ordinary-row
packet. The emitted constraint list and R1CS lowering remain the established
Lean authority. This module adds no row, column, or retained coordinate.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupport
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem expression_external_source {expression : Expr}
    (support : expression.VarsSatisfy External) :
    expression.VarsSatisfy Source :=
  support.mono expression external_source

private theorem k_external_source {value : KExpr}
    (support : Horner.KSupported value External) :
    Horner.KSupported value Source :=
  support.mono (fun _ supported => external_source _ supported)

private theorem externalInputsSource
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    Formal.ExternalInputsSupported
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset Source := by
  have support := PiCCSOrdinarySourceSupport.externalInputsSupported
    logicalWidth publicFits
  refine {
    priorStateFixed := ?_
    outputStateFixed := ?_
    priorStateContext := ?_
    outputStateContext := ?_
    expectedContext := ?_
    runningPoint := ?_
    runningCommitment := ?_
    runningPublicInput := ?_
    runningEval_K := ?_
    runningEval_A := ?_
    freshCommitment := ?_
    freshPublicInput := ?_
    roundCoefficient := ?_
    outputEval_K := ?_
    outputEval_A := ?_ }
  · intro word member
    exact expression_external_source (support.priorStateFixed word member)
  · intro word member
    exact expression_external_source (support.outputStateFixed word member)
  · intro lane
    exact expression_external_source (support.priorStateContext lane)
  · intro lane
    exact expression_external_source (support.outputStateContext lane)
  · intro lane
    exact expression_external_source (support.expectedContext lane)
  · intro coordinate
    exact k_external_source (support.runningPoint coordinate)
  · intro source row coefficient
    exact expression_external_source
      (support.runningCommitment source row coefficient)
  · intro source column
    exact expression_external_source (support.runningPublicInput source column)
  · intro source coefficient
    exact k_external_source (support.runningEval_K source coefficient)
  · intro source matrix coefficient
    exact k_external_source (support.runningEval_A source matrix coefficient)
  · intro source row coefficient
    exact expression_external_source
      (support.freshCommitment source row coefficient)
  · intro source column
    exact expression_external_source (support.freshPublicInput source column)
  · intro roundIndex coefficient
    exact k_external_source (support.roundCoefficient roundIndex coefficient)
  · intro source coefficient
    exact k_external_source (support.outputEval_K source coefficient)
  · intro source matrix coefficient
    exact k_external_source (support.outputEval_A source matrix coefficient)

private theorem sharedInputsSource
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    Formal.ExternalInputsSupported
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset Source := by
  simpa [PiCCSArithmetic.sharedInterface, PiCCSInvocations.sharedInterface,
    PiCCSArithmetic.parentInterface, PiCCSInvocations.parentInterface,
    Formal.atOffset] using externalInputsSource

private theorem fixedLocal_source (start length : Nat)
    (initialLe : PiCCSArithmetic.initialClaimLogicalStart ≤ start)
    (finishLe : start + length ≤ PiCCSStarts.outputBindingWitnessStart) :
    ∀ index, start ≤ index → index < start + length → Source index := by
  intro index lower upper
  exact local_source index (Nat.le_trans initialLe lower)
    (Nat.lt_of_lt_of_le upper finishLe)

private theorem initialProgramLength
    (interface : InitialClaim.Interface) (offset : Nat) :
    (InitialClaim.program interface offset).recipes.length = 25918 := by
  unfold InitialClaim.program Horner.Owned.program
  rw [Horner.compile_recipes_length]
  change 2 * ((InitialClaim.coefficientExprs interface offset).length - 1) =
    25918
  rw [InitialClaim.coefficientExprs_length]

private theorem normProgramLength
    (interface : NormTerminal.Interface) (offset : Nat) :
    (Horner.Owned.program (NormTerminal.ownedInterface interface) offset
      ).recipes.length = 32 := by
  unfold Horner.Owned.program
  rw [Horner.compile_recipes_length]
  change 2 * ((NormTerminal.coefficientExprs interface offset).length - 1) = 32
  rw [NormTerminal.coefficientExprs_length]

private theorem initial_le_initial :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.initialClaimLogicalStart :=
  Nat.le_refl _

private theorem initial_finish_le :
    PiCCSArithmetic.initialClaimLogicalStart + 25918 ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.initialClaimLogicalStart,
    PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem initial_le_evalK :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.evalKLogicalStart := by
  norm_num [PiCCSArithmetic.evalKLogicalStart, PiCCSStarts.evalKLogicalStart,
    PiCCSStarts.sumcheckLogicalStart, PiCCSStarts.initialClaimLogicalStart,
    PiCCSArithmetic.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem evalK_finish_le :
    PiCCSArithmetic.evalKLogicalStart + 1836 ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.evalKLogicalStart, PiCCSStarts.evalKLogicalStart,
    PiCCSStarts.sumcheckLogicalStart, PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem initial_le_evalA :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.evalALogicalStart := by
  norm_num [PiCCSArithmetic.evalALogicalStart, PiCCSStarts.evalALogicalStart,
    PiCCSStarts.evalKLogicalStart, PiCCSStarts.sumcheckLogicalStart,
    PiCCSStarts.initialClaimLogicalStart, PiCCSArithmetic.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem evalA_finish_le :
    PiCCSArithmetic.evalALogicalStart + 24300 ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.evalALogicalStart, PiCCSStarts.evalALogicalStart,
    PiCCSStarts.evalKLogicalStart, PiCCSStarts.sumcheckLogicalStart,
    PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem initial_le_ccs :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.ccsLogicalStart := by
  exact Nat.le_trans initial_le_evalA (by
    norm_num [PiCCSArithmetic.ccsLogicalStart, PiCCSStarts.ccsLogicalStart,
      PiCCSArithmetic.evalALogicalStart, PiCCSStarts.evalALogicalStart])

private theorem ccs_finish_le :
    PiCCSArithmetic.ccsLogicalStart + 2 ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.ccsLogicalStart, PiCCSStarts.ccsLogicalStart,
    PiCCSStarts.evalALogicalStart, PiCCSStarts.evalKLogicalStart,
    PiCCSStarts.sumcheckLogicalStart, PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem initial_le_norm :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.normLogicalStart := by
  exact Nat.le_trans initial_le_ccs (by
    norm_num [PiCCSArithmetic.normLogicalStart, PiCCSStarts.normLogicalStart,
      PiCCSArithmetic.ccsLogicalStart, PiCCSStarts.ccsLogicalStart])

private theorem norm_finish_le :
    PiCCSArithmetic.normLogicalStart + 32 ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.normLogicalStart, PiCCSStarts.normLogicalStart,
    PiCCSStarts.ccsLogicalStart, PiCCSStarts.evalALogicalStart,
    PiCCSStarts.evalKLogicalStart, PiCCSStarts.sumcheckLogicalStart,
    PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

private theorem initial_le_final :
    PiCCSArithmetic.initialClaimLogicalStart ≤
      PiCCSArithmetic.finalIdentityLogicalStart := by
  exact Nat.le_trans initial_le_norm (by
    norm_num [PiCCSArithmetic.finalIdentityLogicalStart,
      PiCCSStarts.finalIdentityLogicalStart,
      PiCCSArithmetic.normLogicalStart, PiCCSStarts.normLogicalStart])

private theorem final_finish_le :
    PiCCSArithmetic.finalIdentityLogicalStart + FinalIdentity.privateCount ≤
      PiCCSStarts.outputBindingWitnessStart := by
  rw [PiCCSStarts.outputBindingWitnessStart_eq]
  norm_num [PiCCSArithmetic.finalIdentityLogicalStart,
    PiCCSStarts.finalIdentityLogicalStart, FinalIdentity.privateCount,
    PiCCSStarts.normLogicalStart, PiCCSStarts.ccsLogicalStart,
    PiCCSStarts.evalALogicalStart, PiCCSStarts.evalKLogicalStart,
    PiCCSStarts.sumcheckLogicalStart, PiCCSStarts.initialClaimLogicalStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq]

/-- Every expression in the canonical eight-child ordinary packet uses only
the selected pre-Spartan retained source families. -/
theorem emittedConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    ∀ expression ∈ PiCCSCompleteness.emittedConstraints
        logicalWidth publicFits,
      expression.VarsSatisfy Source := by
  let shared := PiCCSArithmetic.sharedInterface
    logicalWidth publicFits
  let initialInterface := Formal.initialClaimInterface shared
  let sumcheckInterface := Formal.sumcheckInterface shared
  let evalKInterface := Formal.evalKInterface shared
  let evalAInterface := Formal.evalAInterface shared
  let ccsInterface := Formal.ccsRowInterface shared
  let normInterface := Formal.normRowInterface shared
  let finalInterface := Formal.finalIdentityRowInterface shared
  have inputs := sharedInputsSource (logicalWidth := logicalWidth)
    (publicFits := publicFits)
  have transcript := PiCCSOrdinarySourceSupport.transcriptValuesSupported
    logicalWidth publicFits
  have initialStartEq : Formal.initialClaimStart shared =
      PiCCSArithmetic.initialClaimLogicalStart := by
    dsimp [shared, PiCCSArithmetic.sharedInterface,
      PiCCSInvocations.sharedInterface]
    rw [Formal.initialClaimStart_atOffset]
    exact (PiCCSArithmetic.initialClaimLogicalStart_matches
      logicalWidth publicFits).symm
  have sumcheckStartEq : Formal.sumcheckStart shared =
      PiCCSArithmetic.sumcheckLogicalStart := by
    dsimp [shared, PiCCSArithmetic.sharedInterface,
      PiCCSInvocations.sharedInterface]
    rw [Formal.sumcheckStart_atOffset]
    exact (PiCCSArithmetic.sumcheckLogicalStart_matches
      logicalWidth publicFits).symm
  have evalKStartEq : Formal.evalKStart shared =
      PiCCSArithmetic.evalKLogicalStart := by
    dsimp [shared, PiCCSArithmetic.sharedInterface,
      PiCCSInvocations.sharedInterface]
    rw [Formal.evalKStart_atOffset]
    exact (PiCCSArithmetic.evalKLogicalStart_matches
      logicalWidth publicFits).symm
  have evalAStartEq : Formal.evalAStart shared =
      PiCCSArithmetic.evalALogicalStart := by
    dsimp [shared, PiCCSArithmetic.sharedInterface,
      PiCCSInvocations.sharedInterface]
    rw [Formal.evalAStart_atOffset]
    exact (PiCCSArithmetic.evalALogicalStart_matches
      logicalWidth publicFits).symm
  have ccsStartEq : Formal.ccsStart shared =
      PiCCSArithmetic.ccsLogicalStart := by
    dsimp [shared, PiCCSArithmetic.sharedInterface,
      PiCCSInvocations.sharedInterface]
    rw [Formal.ccsStart_atOffset]
    exact (PiCCSArithmetic.ccsLogicalStart_matches
      logicalWidth publicFits).symm
  have normStartEq : Formal.normStart shared =
      PiCCSArithmetic.normLogicalStart := by
    unfold Formal.normStart
    rw [ccsStartEq]
    rfl
  have initialLocal : ∀ index,
      PiCCSArithmetic.initialClaimLogicalStart ≤ index →
      index < PiCCSArithmetic.initialClaimLogicalStart +
        (InitialClaim.program initialInterface
          PiCCSArithmetic.initialClaimLogicalStart).recipes.length →
      Source index := by
    rw [initialProgramLength]
    exact fixedLocal_source _ _ initial_le_initial initial_finish_le
  have gammaSupport : Horner.KSupported
      (initialInterface.gamma PiCCSArithmetic.initialClaimLogicalStart) Source := by
    simpa [initialInterface, Formal.initialClaimInterface] using transcript.gamma
  have initialEvalKSupport : ∀ coordinate,
      Horner.KSupported
        (initialInterface.eval_K PiCCSArithmetic.initialClaimLogicalStart
          coordinate) Source := by
    intro coordinate
    simpa [initialInterface, Formal.initialClaimInterface] using
      inputs.runningEval_K coordinate.running coordinate.coefficient
  have initialEvalASupport : ∀ coordinate,
      Horner.KSupported
        (initialInterface.eval_A PiCCSArithmetic.initialClaimLogicalStart
          coordinate) Source := by
    intro coordinate
    simpa [initialInterface, Formal.initialClaimInterface] using
      inputs.runningEval_A coordinate.running coordinate.matrix
        coordinate.coefficient
  have initialRows := InitialClaim.flatConstraints_varsSatisfy
    initialInterface PiCCSArithmetic.initialClaimLogicalStart Source
    gammaSupport initialEvalKSupport initialEvalASupport initialLocal
  have initialOutput := InitialClaim.output_varsSatisfy
    initialInterface PiCCSArithmetic.initialClaimLogicalStart Source
    gammaSupport initialEvalKSupport initialEvalASupport initialLocal
  have sumInitialSupport : SumcheckChain.KSupported
      (sumcheckInterface.initial PiCCSArithmetic.sumcheckLogicalStart) Source := by
    unfold sumcheckInterface Formal.sumcheckInterface Formal.initialClaimOutput
    rw [initialStartEq]
    exact initialOutput
  have sumRoundSupport : ∀ roundIndex,
      SumcheckChain.RoundSupported
        (sumcheckInterface.round PiCCSArithmetic.sumcheckLogicalStart roundIndex)
        Source := by
    intro roundIndex
    constructor
    · intro coefficient
      simpa [sumcheckInterface, Formal.sumcheckInterface,
        Formal.roundTranscriptRound, RoundTranscript.round,
        RoundTranscript.Message.asRound,
        Formal.roundTranscriptInterface] using
        inputs.roundCoefficient roundIndex coefficient
    · simpa [sumcheckInterface, Formal.sumcheckInterface,
        Formal.roundTranscriptRound, RoundTranscript.round,
        RoundTranscript.Message.asRound] using transcript.roundPoint roundIndex
  have sumcheckRows := SumcheckChain.flatConstraints_varsSatisfy
    sumcheckInterface PiCCSArithmetic.sumcheckLogicalStart Source
    sumInitialSupport sumRoundSupport
  have sumcheckOutput := SumcheckChain.output_varsSatisfy
    sumcheckInterface PiCCSArithmetic.sumcheckLogicalStart Source
    sumInitialSupport sumRoundSupport
  have evalKLocal : ∀ index,
      PiCCSArithmetic.evalKLogicalStart ≤ index →
      index < PiCCSArithmetic.evalKLogicalStart + localLength
        (Circuit.ops (EvalKTerminal.circuit evalKInterface).main
          PiCCSArithmetic.evalKLogicalStart) → Source index := by
    intro index lower upper
    rw [EvalKTerminal.localLength_eq] at upper
    apply fixedLocal_source PiCCSArithmetic.evalKLogicalStart 1836
      initial_le_evalK evalK_finish_le index lower upper
  have evalKRoundSupport : ∀ coordinate,
      Horner.KSupported
        (evalKInterface.roundPoint PiCCSArithmetic.evalKLogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalKInterface, Formal.evalKInterface] using
      transcript.roundPoint coordinate
  have evalKPriorSupport : ∀ coordinate,
      Horner.KSupported
        (evalKInterface.priorPoint PiCCSArithmetic.evalKLogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalKInterface, Formal.evalKInterface] using
      inputs.runningPoint coordinate
  have evalKOutputInputSupport : ∀ coordinate,
      Horner.KSupported
        (evalKInterface.outputEval_K PiCCSArithmetic.evalKLogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalKInterface, Formal.evalKInterface] using
      inputs.outputEval_K (UnifiedSources.runningSourceIndex coordinate.running)
        coordinate.coefficient
  have evalKRows := EvalKTerminal.flatConstraints_varsSatisfy evalKInterface
    PiCCSArithmetic.evalKLogicalStart Source evalKRoundSupport evalKPriorSupport
    (by simpa [evalKInterface, Formal.evalKInterface] using transcript.gamma)
    evalKOutputInputSupport evalKLocal
  have evalKOutput := EvalKTerminal.output_varsSatisfy evalKInterface
    PiCCSArithmetic.evalKLogicalStart Source evalKRoundSupport evalKPriorSupport
    (by simpa [evalKInterface, Formal.evalKInterface] using transcript.gamma)
    evalKOutputInputSupport evalKLocal
  have evalALocal : ∀ index,
      PiCCSArithmetic.evalALogicalStart ≤ index →
      index < PiCCSArithmetic.evalALogicalStart + localLength
        (Circuit.ops (EvalATerminal.circuit evalAInterface).main
          PiCCSArithmetic.evalALogicalStart) → Source index := by
    intro index lower upper
    rw [EvalATerminal.localLength_eq] at upper
    apply fixedLocal_source PiCCSArithmetic.evalALogicalStart 24300
      initial_le_evalA evalA_finish_le index lower upper
  have evalARoundSupport : ∀ coordinate,
      Horner.KSupported
        (evalAInterface.roundPoint PiCCSArithmetic.evalALogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalAInterface, Formal.evalAInterface] using
      transcript.roundPoint coordinate
  have evalAPriorSupport : ∀ coordinate,
      Horner.KSupported
        (evalAInterface.priorPoint PiCCSArithmetic.evalALogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalAInterface, Formal.evalAInterface] using
      inputs.runningPoint coordinate
  have evalAOutputInputSupport : ∀ coordinate,
      Horner.KSupported
        (evalAInterface.outputEval_A PiCCSArithmetic.evalALogicalStart coordinate)
        Source := by
    intro coordinate
    simpa [evalAInterface, Formal.evalAInterface] using
      inputs.outputEval_A (UnifiedSources.runningSourceIndex coordinate.running)
        coordinate.matrix coordinate.coefficient
  have evalARows := EvalATerminal.flatConstraints_varsSatisfy evalAInterface
    PiCCSArithmetic.evalALogicalStart Source evalARoundSupport evalAPriorSupport
    (by simpa [evalAInterface, Formal.evalAInterface] using transcript.gamma)
    evalAOutputInputSupport evalALocal
  have evalAOutput := EvalATerminal.output_varsSatisfy evalAInterface
    PiCCSArithmetic.evalALogicalStart Source evalARoundSupport evalAPriorSupport
    (by simpa [evalAInterface, Formal.evalAInterface] using transcript.gamma)
    evalAOutputInputSupport evalALocal
  have ccsLocal : ∀ index,
      PiCCSArithmetic.ccsLogicalStart ≤ index →
      index < PiCCSArithmetic.ccsLogicalStart + localLength
        (Circuit.ops (Sparse.Owned.circuit Formal.ccsRowPolynomial
          (CcsTerminal.sparseInterface ccsInterface)).main
          PiCCSArithmetic.ccsLogicalStart) → Source index := by
    intro index lower upper
    rw [Sparse.Owned.localLength_eq] at upper
    apply fixedLocal_source PiCCSArithmetic.ccsLogicalStart 2
      initial_le_ccs ccs_finish_le index lower upper
  have ccsPointSupport : ∀ matrix,
      Horner.KSupported
        ((CcsTerminal.sparseInterface ccsInterface).point
          PiCCSArithmetic.ccsLogicalStart matrix) Source := by
    intro matrix
    simpa [ccsInterface, Formal.ccsRowInterface,
      CcsTerminal.sparseInterface] using
      inputs.outputEval_A (UnifiedSources.freshSourceIndex Formal.freshIndex)
        matrix
        Formal.rowConstantCoefficient
  have ccsRows := Sparse.Owned.flatConstraints_varsSatisfy
    Formal.ccsRowPolynomial (CcsTerminal.sparseInterface ccsInterface)
    PiCCSArithmetic.ccsLogicalStart Source ccsPointSupport ccsLocal
  have ccsOutput := Sparse.Owned.output_varsSatisfy
    Formal.ccsRowPolynomial (CcsTerminal.sparseInterface ccsInterface)
    PiCCSArithmetic.ccsLogicalStart Source ccsLocal
  have normLocal : ∀ index,
      PiCCSArithmetic.normLogicalStart ≤ index →
      index < PiCCSArithmetic.normLogicalStart +
        (Horner.Owned.program (NormTerminal.ownedInterface normInterface)
          PiCCSArithmetic.normLogicalStart).recipes.length → Source index := by
    rw [normProgramLength]
    exact fixedLocal_source _ _ initial_le_norm norm_finish_le
  have normSourceSupport : ∀ source,
      Horner.KSupported
        (normInterface.sourceAssignment PiCCSArithmetic.normLogicalStart source)
        Source := by
    intro source
    simpa [normInterface, Formal.normRowInterface] using
      inputs.outputEval_K source Formal.rowConstantCoefficient
  have normRows := NormTerminal.flatConstraints_varsSatisfy normInterface
    PiCCSArithmetic.normLogicalStart Source
    (by simpa [normInterface, Formal.normRowInterface] using transcript.gamma)
    normSourceSupport normLocal
  have normOutput := NormTerminal.output_varsSatisfy normInterface
    PiCCSArithmetic.normLogicalStart Source
    (by simpa [normInterface, Formal.normRowInterface] using transcript.gamma)
    normSourceSupport normLocal
  have finalLocal : ∀ index,
      PiCCSArithmetic.finalIdentityLogicalStart ≤ index →
      index < PiCCSArithmetic.finalIdentityLogicalStart +
        FinalIdentity.privateCount → Source index := by
    exact fixedLocal_source _ _ initial_le_final final_finish_le
  have finalRows := FinalIdentity.flatConstraints_varsSatisfy finalInterface
    PiCCSArithmetic.finalIdentityLogicalStart Source
    (by intro coordinate; simpa [finalInterface,
      Formal.finalIdentityRowInterface] using transcript.roundPoint coordinate)
    (by intro coordinate; simpa [finalInterface,
      Formal.finalIdentityRowInterface] using transcript.alpha coordinate)
    (by simpa [finalInterface, Formal.finalIdentityRowInterface] using
      transcript.gamma)
    (by
      unfold finalInterface Formal.finalIdentityRowInterface Formal.evalKOutput
      rw [evalKStartEq]
      exact evalKOutput)
    (by
      unfold finalInterface Formal.finalIdentityRowInterface Formal.evalAOutput
      rw [evalAStartEq]
      exact evalAOutput)
    (by
      unfold finalInterface Formal.finalIdentityRowInterface Formal.ccsRowOutput
      rw [ccsStartEq]
      exact ccsOutput)
    (by
      unfold finalInterface Formal.finalIdentityRowInterface Formal.normRowOutput
      rw [normStartEq]
      exact normOutput)
    (by
      unfold finalInterface Formal.finalIdentityRowInterface Formal.sumcheckOutput
      rw [sumcheckStartEq]
      exact sumcheckOutput)
    finalLocal
  intro expression member
  rw [PiCCSCompleteness.emittedConstraints, List.mem_append] at member
  rcases member with statementMember | packetMember
  · apply Formal.statementBindingConstraints_varsSatisfy
      (PiCCSArithmetic.parentInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset PiCCSArithmetic.statementBindingLogicalStart
      Source (externalInputsSource (logicalWidth := logicalWidth)
        (publicFits := publicFits)) expression
    simpa [PiCCSArithmetic.statementBindingConstraints,
      NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints] using
      statementMember
  · rw [PiCCSCompleteness.packetConstraints] at packetMember
    simp only [List.mem_append] at packetMember
    rcases packetMember with initialMember | sumcheckMember | evalKMember |
        evalAMember | ccsMember | normMember | finalMember
    · exact initialRows expression (by
        simpa [PiCCSArithmetic.initialClaimConstraints,
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
          Formal.initialClaimCircuit] using initialMember)
    · exact sumcheckRows expression (by
        simpa [PiCCSArithmetic.sumcheckConstraints,
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
          Formal.sumcheckCircuit] using sumcheckMember)
    · exact evalKRows expression (by
        simpa [PiCCSArithmetic.evalKConstraints,
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
          Formal.evalKCircuit] using evalKMember)
    · exact evalARows expression (by
        simpa [PiCCSArithmetic.evalAConstraints,
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
          Formal.evalACircuit] using evalAMember)
    · exact ccsRows expression (by
        simpa [PiCCSArithmetic.ccsConstraints, PiCCSArithmetic.mainConstraints,
          Formal.ccsRowMain, ccsInterface] using ccsMember)
    · exact normRows expression (by
        simpa [PiCCSArithmetic.normConstraints, PiCCSArithmetic.mainConstraints,
          Formal.normRowMain, normInterface,
          NormTerminal.circuit_ops_eq_ownedOps] using normMember)
    · exact finalRows expression (by
        simpa [PiCCSArithmetic.finalIdentityConstraints,
          PiCCSArithmetic.mainConstraints, Formal.finalIdentityRowMain,
          finalInterface] using finalMember)

/-- Exact R1CS lowering and Spartan remapping preserve the retained PiCCS
source set. Every generic multiplication column is included in the declared
fresh retained interval. -/
theorem sourceRows_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ PiCCSOrdinaryDirectSource.sourceRows logicalWidth publicFits,
      row.VarsSatisfy Target := by
  let constraints := PiCCSCompleteness.emittedConstraints
    logicalWidth publicFits
  let freshStart := PiCCSArithmetic.initialClaimFreshStart
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy constraints freshStart
    Source (by simpa [constraints] using emittedConstraints_varsSatisfy)
  have endEq :
      freshStart + R1CS.totalFreshCount constraints =
        PiRLCInputs.phaseOffset := by
    rw [show R1CS.totalFreshCount constraints = 731605 by
      simpa [constraints] using
        PiCCSCompleteness.emittedConstraints_totalFreshCount relation]
    unfold freshStart PiCCSArithmetic.initialClaimFreshStart
      PiCCSStarts.initialClaimFreshStart PiCCSStarts.roundTranscriptFreshStart
      PiCCSStarts.challengeFreshStart PiCCSStarts.statementAbsorptionFreshStart
      PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase
    rw [PiCCSInputs.phaseOffset_eq]
    norm_num [PiRLCInputs.phaseOffset]
  have loweredSource : ∀ row ∈
      (R1CS.lowerConstraints constraints freshStart).rows,
      row.VarsSatisfy Source := by
    intro row member
    apply (lowered row member).mono row
    intro column support
    rcases support with source | ⟨lower, upper⟩
    · exact source
    · apply fresh_source column lower
      rw [endEq] at upper
      exact upper
  rw [PiCCSOrdinaryDirectSource.sourceRows,
    PiCCSCompleteness.arithmeticRows_toR1CS_eq relation]
  apply Spartan.remapRows_varsSatisfy Source Target _ loweredSource
  intro column support
  exact source_target column support

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSupport
