import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Export.Stage1.PiCCSCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCCombinationCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Completeness

/-!
Owns the package-level constructive assemblers for the exact production
PiCCS, PiRLC, and PiDEC v1_1 phase rows. It adds no row or alternate verifier
path.
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

structure PiRLCRowsHold (env : Env) : Prop where
  permutations : ∀ invocation ∈
      PiRLCSamplerInvocations.invocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits),
    PermutationInvocationHolds (Data.circuitPackage ()) invocation env
  first54 : ∀ invocation ∈ PiRLCFirst54Invocations.invocations,
    CompactRowInvocationHolds (Data.circuitPackage ()) invocation env
  combinations : ∀ invocation ∈ PiRLCCombinationInvocations.invocations,
    CompactRowInvocationHolds (Data.circuitPackage ()) invocation env
  arithmetic : R1CS.RowsHold env
    ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS)

structure PiDECRowsHold (env : Env) : Prop where
  arithmetic : R1CS.RowsHold env
    ((PiDECArithmetic.canonicalPlan
      Data.logicalWidth Data.publicFits).rows.map Rows.CompiledRow.toR1CS)

private theorem piCcsArithmeticLogicalEnds :
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.initialClaimLogicalStart + 25918 ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.sumcheckLogicalStart ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.evalKLogicalStart + 1828 ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.evalALogicalStart + 24292 ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.ccsLogicalStart + 2 ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.normLogicalStart + 32 ≤
        PiCCSArithmetic.initialClaimFreshStart ∧
      PiCCSArithmetic.finalIdentityLogicalStart + 27750 ≤
        PiCCSArithmetic.initialClaimFreshStart := by
  norm_num [PiCCSArithmetic.initialClaimFreshStart,
    PiCCSArithmetic.initialClaimLogicalStart,
    PiCCSArithmetic.sumcheckLogicalStart,
    PiCCSArithmetic.evalKLogicalStart,
    PiCCSArithmetic.evalALogicalStart,
    PiCCSArithmetic.ccsLogicalStart,
    PiCCSArithmetic.normLogicalStart,
    PiCCSArithmetic.finalIdentityLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalALogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.normLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptFreshStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeFreshStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionFreshStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputColumnCount]

/-- Every source row lowered into the ordinary PiCCS packet reads only the
completed PiCCS logical prefix. This proof is structural in the eight opaque
child circuits and does not traverse the emitted rows. -/
theorem piCcsEmittedConstraints_varsBelow
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env) :
    ∀ expression ∈ emittedConstraints Data.logicalWidth Data.publicFits,
      expression.VarsBelow PiCCSArithmetic.initialClaimFreshStart := by
  let parent := PiCCSInvocations.parentInterface Data.logicalWidth
    Data.publicFits
  let assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation parent
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
        Data.logicalWidth Data.publicFits) env
  have initialStartEq : PiCCSArithmetic.initialClaimLogicalStart =
      Formal.initialClaimOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.initialClaimLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have sumcheckStartEq : PiCCSArithmetic.sumcheckLogicalStart =
      Formal.sumcheckOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.sumcheckLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have evalKStartEq : PiCCSArithmetic.evalKLogicalStart =
      Formal.evalKOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.evalKLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have evalAStartEq : PiCCSArithmetic.evalALogicalStart =
      Formal.evalAOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.evalALogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have ccsStartEq : PiCCSArithmetic.ccsLogicalStart =
      Formal.ccsOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.ccsLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have normStartEq : PiCCSArithmetic.normLogicalStart =
      Formal.normRowOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.normLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  have finalStartEq : PiCCSArithmetic.finalIdentityLogicalStart =
      Formal.finalIdentityRowOffset parent
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
    dsimp [parent]
    simpa [PiCCSArithmetic.parentInterface] using
      (PiCCSArithmetic.finalIdentityLogicalStart_matches Data.logicalWidth
        Data.publicFits)
  intro expression member
  rw [emittedConstraints, List.mem_append] at member
  rcases member with statementMember | packetMember
  · exact Expr.VarsBelow.mono expression
      (statementBindingConstraints_varsBelow env assumptions.statementBinding
        expression statementMember) piCcsArithmeticLogicalEnds.1
  · rw [packetConstraints] at packetMember
    simp only [List.mem_append] at packetMember
    rcases packetMember with initialMember | sumcheckMember | evalKMember |
        evalAMember | ccsMember | normMember | finalMember
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.initialClaimCircuit (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.initialClaimOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.initialClaimConstraints_eq]
        exact initialMember
      have below := InitialClaim.flatConstraints_varsBelow
        (Formal.initialClaimInterface (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.initialClaimOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        env assumptions.initialClaim expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.initialClaimCircuit] using childMember)
      rw [InitialClaim.localLength_eq, ← initialStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.sumcheckCircuit (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.sumcheckOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.sumcheckConstraints_eq]
        exact sumcheckMember
      have childAssumptions : SumcheckChain.Assumptions
          (Formal.sumcheckInterface (Formal.atOffset parent
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
          (Formal.sumcheckOffset parent
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
          (fun _ => 0) := by
        simpa using assumptions.sumcheck
      have below := SumcheckChain.flatConstraints_varsBelow
        (Formal.sumcheckInterface (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.sumcheckOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        childAssumptions expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.sumcheckCircuit] using childMember)
      rw [← sumcheckStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.evalKCircuit (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.evalKOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.evalKConstraints_eq]
        exact evalKMember
      have below := EvalKTerminal.flatConstraints_varsBelow
        (Formal.evalKInterface (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.evalKOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        env assumptions.eval_K expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.evalKCircuit] using childMember)
      rw [EvalKTerminal.localLength_eq, ← evalKStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.evalACircuit (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.evalAOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.evalAConstraints_eq]
        exact evalAMember
      have below := EvalATerminal.flatConstraints_varsBelow
        (Formal.evalAInterface (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.evalAOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        env assumptions.eval_A expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.evalACircuit] using childMember)
      rw [EvalATerminal.localLength_eq, ← evalAStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.2.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.ccsCircuit relation (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.ccsOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.ccsConstraints_eq relation]
        exact ccsMember
      have childAssumptions : CcsTerminal.Assumptions relation
          (Formal.ccsInterface relation (Formal.atOffset parent
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
          (Formal.ccsOffset parent
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
          (fun _ => 0) := by
        simpa using assumptions.ccs
      have below := CcsTerminal.flatConstraints_varsBelow relation
        (Formal.ccsInterface relation (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.ccsOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        childAssumptions expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.ccsCircuit] using childMember)
      norm_num [CcsTerminal.privateCount] at below
      rw [← ccsStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.2.2.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.normCircuit relation (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.normOffset relation parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.normConstraints_eq relation]
        exact normMember
      have below := NormTerminal.flatConstraints_varsBelow
        (Formal.normInterface relation (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.normOffset relation parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        env assumptions.norm expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.normCircuit] using childMember)
      rw [NormTerminal.localLength_eq,
        Formal.normOffset_eq_normRowOffset,
        ← normStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.2.2.2.2.1
    · have childMember : expression ∈
          NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
            (Formal.finalIdentityCircuit relation (Formal.atOffset parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
            (Formal.finalIdentityOffset relation parent
              NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
        rw [PiCCSCompleteness.finalIdentityConstraints_eq relation]
        exact finalMember
      have below := FinalIdentity.flatConstraints_varsBelow
        (Formal.finalIdentityInterface relation (Formal.atOffset parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset))
        (Formal.finalIdentityOffset relation parent
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
        env assumptions.finalIdentity expression (by
          simpa [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
            Formal.finalIdentityCircuit] using childMember)
      rw [FinalIdentity.localLength_eq,
        Formal.finalIdentityOffset_eq_finalIdentityRowOffset,
        ← finalStartEq] at below
      exact Expr.VarsBelow.mono expression below
        piCcsArithmeticLogicalEnds.2.2.2.2.2.2.2

def phaseSuffixStart : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
    NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset

def phaseSuffixLength : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount - phaseSuffixStart

theorem phaseSuffixEnd_eq :
    phaseSuffixStart + phaseSuffixLength =
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount := by
  norm_num [phaseSuffixStart, phaseSuffixLength,
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
    NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
    NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
    NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
    NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
    NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]

private theorem pullback_agreesBelow_piRlc
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength) :
    ∀ index, index < NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset →
      NightstreamFPrime.Layout.Stage1.Spartan.pullback after index =
        NightstreamFPrime.Layout.Stage1.Spartan.pullback before index := by
  intro index below
  unfold NightstreamFPrime.Layout.Stage1.Spartan.pullback
  apply agrees
  rcases NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_before_piCcsLocal
      index NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset (by
        norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
          NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]) below with
    mappedBefore | mappedPublic
  · exact Or.inl mappedBefore
  · apply Or.inr
    rw [phaseSuffixEnd_eq]
    exact mappedPublic.le

/-- PiRLC completion cannot change any exact Lean-lowered PiCCS arithmetic
row. Source rows end at the PiRLC source offset; remapping sends earlier
private columns below the write and public columns above its exact end. -/
theorem piCcsArithmeticRows_of_piRlcAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength)
    (holds : R1CS.RowsHold before
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits).map
        Rows.CompiledRow.toR1CS)) :
    R1CS.RowsHold after
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits).map
        Rows.CompiledRow.toR1CS) := by
  let sourceRows :=
    (R1CS.lowerConstraints
      (emittedConstraints Data.logicalWidth Data.publicFits)
      PiCCSArithmetic.initialClaimFreshStart).rows
  have sourceHolds : R1CS.RowsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback before) sourceRows := by
    apply (NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold before
      sourceRows).mp
    rw [← PiCCSCompleteness.arithmeticRows_toR1CS_eq relation]
    exact holds
  have sourceScope : ∀ row ∈ sourceRows,
      row.VarsBelow NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset := by
    have loweredScope := R1CS.lowerConstraints_rows_varsBelow
      (emittedConstraints Data.logicalWidth Data.publicFits)
      PiCCSArithmetic.initialClaimFreshStart
      (piCcsEmittedConstraints_varsBelow relation
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback before))
    rw [PiCCSCompleteness.emittedConstraints_totalFreshCount relation] at loweredScope
    have endEq : PiCCSArithmetic.initialClaimFreshStart + 700767 =
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset := by
      unfold PiCCSArithmetic.initialClaimFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart
        NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
      rw [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
      norm_num [NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
    rw [endEq] at loweredScope
    exact loweredScope
  have sourceAfter : R1CS.RowsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback after) sourceRows :=
    R1CS.rowsHold_of_agree_below sourceRows
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback before)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback after) sourceScope
      (pullback_agreesBelow_piRlc before after agrees) sourceHolds
  rw [PiCCSCompleteness.arithmeticRows_toR1CS_eq relation]
  exact (NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold after
    sourceRows).mpr sourceAfter

private theorem liftPilotColumn_piRlcOutside (column : Nat) :
    NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn column <
        phaseSuffixStart ∨
      phaseSuffixStart + phaseSuffixLength ≤
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn column := by
  unfold NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
  split
  all_goals try split
  all_goals
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
      phaseSuffixStart, phaseSuffixLength] at * <;> omega

private theorem liftPilotCombination_eval_of_piRlcAgreesOutside
    (combination : SparseCombination) (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength) :
    (Data.liftPilotCombination combination).eval after =
      (Data.liftPilotCombination combination).eval before := by
  unfold Data.liftPilotCombination Data.liftPilotTerm SparseCombination.eval
  rw [List.map_map, List.map_map]
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro term member
  simp only [Function.comp_apply]
  rw [agrees _ (liftPilotColumn_piRlcOutside term.column)]

/-- The PiRLC-and-later write interval is disjoint from every input and
target column of the lifted Pilot witness program. -/
theorem pilotWitnessInstructions_of_piRlcAgreesOutside
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength)
    (holds : ∀ instruction ∈
      Data.liftPilotInstructions (PilotData.witnessInstructions ()),
      instruction.Holds before) :
    ∀ instruction ∈
      Data.liftPilotInstructions (PilotData.witnessInstructions ()),
      instruction.Holds after := by
  intro instruction member
  rw [Data.liftPilotInstructions, List.mem_map] at member
  rcases member with ⟨pilotInstruction, pilotMember, rfl⟩
  have beforeHolds := holds (Data.liftPilotInstruction pilotInstruction) (by
    rw [Data.liftPilotInstructions, List.mem_map]
    exact ⟨pilotInstruction, pilotMember, rfl⟩)
  unfold Data.liftPilotInstruction WitnessInstruction.Holds at beforeHolds ⊢
  rw [liftPilotCombination_eval_of_piRlcAgreesOutside
      pilotInstruction.a before after agrees,
    liftPilotCombination_eval_of_piRlcAgreesOutside
      pilotInstruction.b before after agrees,
    agrees _ (liftPilotColumn_piRlcOutside pilotInstruction.target)]
  exact beforeHolds

/-- The PiRLC write interval is disjoint from every lifted Pilot assertion
column, including the relocated public suffix. -/
theorem pilotAssertionRows_of_piRlcAgreesOutside
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength)
    (holds : ∀ row ∈ Data.liftPilotRows (PilotData.assertionRows ()),
      row.Holds before) :
    ∀ row ∈ Data.liftPilotRows (PilotData.assertionRows ()),
      row.Holds after := by
  intro row member
  rw [Data.liftPilotRows, List.mem_map] at member
  rcases member with ⟨pilotRow, _pilotMember, rfl⟩
  have beforeHolds := holds (Data.liftPilotRow pilotRow) (by
    rw [Data.liftPilotRows, List.mem_map]
    exact ⟨pilotRow, _pilotMember, rfl⟩)
  unfold Data.liftPilotRow SparseRow.Holds at beforeHolds ⊢
  rw [liftPilotCombination_eval_of_piRlcAgreesOutside pilotRow.a before after
      agrees,
    liftPilotCombination_eval_of_piRlcAgreesOutside pilotRow.b before after
      agrees,
    liftPilotCombination_eval_of_piRlcAgreesOutside pilotRow.c before after
      agrees]
  exact beforeHolds

private theorem pilotHashInvocationInput_varsBelow
    (chain : HashChain) (invocation : Nat) (lane : Fin 8) (bound : Nat)
    (invocationBound : invocation ≤ chain.absorbCount)
    (inputBound : chain.inputStart + chain.inputLength ≤ bound)
    (witnessBound : chain.witnessStart +
      (chain.absorbCount + 1) * 592 ≤ bound) :
    (invocationInput (PilotData.circuitPackage ()) chain invocation lane.val
      ).VarsBelow bound := by
  have laneBound := lane.isLt
  by_cases invocationZero : invocation = 0
  · subst invocation
    by_cases absorbing : 0 < chain.absorbCount
    · by_cases inputPresent :
          lane.val < 4 ∧ lane.val < chain.inputLength
      · have inputBelow : chain.inputStart + lane.val < bound := by
          omega
        simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          Spec.Poseidon2.rate,
          absorbing, inputPresent, R1CS.LinearCombination.VarsBelow,
          R1CS.LinearCombination.zero, R1CS.LinearCombination.ofVar,
          R1CS.LinearCombination.add, inputBelow]
      · simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          Spec.Poseidon2.rate,
          absorbing, inputPresent, R1CS.LinearCombination.VarsBelow,
          R1CS.LinearCombination.zero]
    · by_cases zeroLane : lane.val = 0
      · simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          absorbing, zeroLane, R1CS.LinearCombination.VarsBelow,
          R1CS.LinearCombination.zero, R1CS.LinearCombination.one,
          R1CS.LinearCombination.add]
      · simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          absorbing, zeroLane, R1CS.LinearCombination.VarsBelow,
          R1CS.LinearCombination.zero]
  · have previousBelow :
        chain.witnessStart + (invocation - 1) * 592 + 584 + lane.val <
          bound := by
      omega
    by_cases absorbing : invocation < chain.absorbCount
    · by_cases inputPresent :
          lane.val < 4 ∧
            invocation * 4 + lane.val < chain.inputLength
      · have inputBelow :
            chain.inputStart + (invocation * 4 + lane.val) < bound := by
          omega
        simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          Spec.Poseidon2.rate,
          invocationZero, absorbing, inputPresent,
          R1CS.LinearCombination.VarsBelow, R1CS.LinearCombination.ofVar,
          R1CS.LinearCombination.add, previousBelow, inputBelow]
      · simp [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          Spec.Poseidon2.rate,
          invocationZero, absorbing, inputPresent,
          R1CS.LinearCombination.VarsBelow, R1CS.LinearCombination.ofVar,
          previousBelow]
    · by_cases zeroLane : lane.val = 0
      · simpa [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          invocationZero, absorbing, zeroLane,
          R1CS.LinearCombination.VarsBelow, R1CS.LinearCombination.ofVar,
          R1CS.LinearCombination.one, R1CS.LinearCombination.add] using
            previousBelow
      · simpa [invocationInput, PilotData.circuitPackage,
          PilotData.poseidonSchedule, PilotData.permutationTemplate,
          invocationZero, absorbing, zeroLane,
          R1CS.LinearCombination.VarsBelow, R1CS.LinearCombination.ofVar] using
            previousBelow

private theorem pilotCanonicalConstraints_varsBelow :
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

private theorem pilotHashChainHolds_of_agree_below
    (chain : HashChain) (before after : Env) (bound : Nat)
    (inputBound : chain.inputStart + chain.inputLength ≤ bound)
    (witnessBound : chain.witnessStart +
      (chain.absorbCount + 1) * 592 ≤ bound)
    (agrees : ∀ index, index < bound → after index = before index)
    (holds : HashChainHolds (PilotData.circuitPackage ()) chain before) :
    HashChainHolds (PilotData.circuitPackage ()) chain after := by
  intro invocation invocationBound
  let mapped (current : Env) : Env := fun column =>
    (PilotData.columnRef column).eval (fun reference =>
      (instantiateColumn (PilotData.circuitPackage ()) chain invocation
        reference).eval current)
  have mappedAgrees : ∀ column, column < 600 →
      mapped after column = mapped before column := by
    intro column columnBound
    by_cases input : column < 8
    · unfold mapped
      rw [show PilotData.columnRef column = .input column by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateColumn]
      exact R1CS.LinearCombination.eval_eq_of_agree_below
        (invocationInput (PilotData.circuitPackage ()) chain invocation column)
        bound after before
        (pilotHashInvocationInput_varsBelow chain invocation
          ⟨column, input⟩ bound invocationBound inputBound witnessBound)
        agrees
    · unfold mapped
      rw [show PilotData.columnRef column = .local (column - 8) by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateColumn,
        R1CS.LinearCombination.eval_ofVar]
      apply agrees
      unfold invocationLocalStart
      norm_num [PilotData.circuitPackage, PilotData.permutationTemplate] at witnessBound ⊢
      omega
  have beforeRows :=
    (NightstreamFPrime.Export.Pilot.canonicalTemplateInvocation_iff chain
      invocation before).mp (holds invocation invocationBound)
  change R1CS.RowsHold (mapped before) (PilotData.canonicalRows ()) at beforeRows
  have beforeLogical : ConstraintsHold (mapped before)
      (PilotData.canonicalConstraints ()) := by
    unfold PilotData.canonicalRows at beforeRows
    exact R1CS.lowerConstraints_sound (mapped before)
      (PilotData.canonicalConstraints ()) 600 beforeRows
  have afterLogical : ConstraintsHold (mapped after)
      (PilotData.canonicalConstraints ()) :=
    constraintsHold_of_agree_below (mapped before) (mapped after)
      (PilotData.canonicalConstraints ()) 600
      pilotCanonicalConstraints_varsBelow mappedAgrees beforeLogical
  apply (NightstreamFPrime.Export.Pilot.canonicalTemplateInvocation_iff chain
    invocation after).mpr
  change R1CS.RowsHold (mapped after) (PilotData.canonicalRows ())
  unfold PilotData.canonicalRows
  apply R1CS.lowerConstraints_complete_of_noFresh
  · apply R1CS.recipeConstraints_noFresh
    apply NightstreamFPrime.Layout.Poseidon2.compile_schedule_direct
    intro lane
    exact R1CS.isAffine_var _
  · exact afterLogical

private theorem stage1HashChainHolds_iff_pilotTemplate
    (chain : HashChain) (env : Env) :
    HashChainHolds (Data.circuitPackage ()) chain env ↔
      HashChainHolds (PilotData.circuitPackage ()) chain env := by
  constructor
  · intro holds invocation invocationBound row member
    have stageMember : row ∈ (Data.circuitPackage ()).permutation.rows := by
      rw [Data.circuitPackage_permutation]
      exact member
    have held := holds invocation invocationBound row stageMember
    change (instantiateRow (PilotData.circuitPackage ()) chain invocation row
      ).Holds env
    exact held
  · intro holds invocation invocationBound row member
    have pilotMember := member
    rw [Data.circuitPackage_permutation] at pilotMember
    have held := holds invocation invocationBound row pilotMember
    change (instantiateRow (Data.circuitPackage ()) chain invocation row).Holds env
    exact held

/-- Both lifted Pilot hash chains are entirely outside the PiRLC private
write interval, so their exact canonical Poseidon2 rows remain true. -/
theorem pilotHashChains_of_piRlcAgreesOutside
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength)
    (holds : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      HashChainHolds (Data.circuitPackage ()) chain before) :
    ∀ chain ∈ [Data.priorChain, Data.outputChain],
      HashChainHolds (Data.circuitPackage ()) chain after := by
  intro chain member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · apply (stage1HashChainHolds_iff_pilotTemplate Data.priorChain after).mpr
    apply pilotHashChainHolds_of_agree_below Data.priorChain before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
    · norm_num [Data.priorChain, Data.liftPilotChain, PilotData.priorChain,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
    · norm_num [Data.priorChain, Data.liftPilotChain, PilotData.priorChain,
        PilotData.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
        NightstreamFPrime.Layout.PilotValues.absorbCount,
        NightstreamFPrime.Spec.Poseidon2.rate,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
    · intro index below
      exact agrees index (Or.inl below)
    · exact (stage1HashChainHolds_iff_pilotTemplate Data.priorChain before).mp
        (holds Data.priorChain (by simp))
  · apply (stage1HashChainHolds_iff_pilotTemplate Data.outputChain after).mpr
    apply pilotHashChainHolds_of_agree_below Data.outputChain before after
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
    · norm_num [Data.outputChain, Data.liftPilotChain, PilotData.outputChain,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
    · norm_num [Data.outputChain, Data.liftPilotChain, PilotData.outputChain,
        PilotData.outputWitnessStart,
        NightstreamFPrime.Layout.PilotValues.outputWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
        NightstreamFPrime.Layout.PilotValues.absorbCount,
        NightstreamFPrime.Spec.Poseidon2.rate,
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
        NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
        NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
    · intro index below
      exact agrees index (Or.inl below)
    · exact (stage1HashChainHolds_iff_pilotTemplate Data.outputChain before).mp
        (holds Data.outputChain (by simp))

/-- The exact PiCCS permutation schedule and ordinary rows are disjoint from
the later PiRLC private interval. Public inputs lie at or above its exact end. -/
theorem piCcsRows_of_piRlcAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (before after : Env)
    (agrees : AgreesOutside before after
      phaseSuffixStart phaseSuffixLength)
    (holds : PiCCSRowsHold before) : PiCCSRowsHold after := by
  have allBefore := (PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits relation).2
  have stableInputs := schedule_stableInputs
    (PiCCSInvocations.invocations_scheduleWithin Data.logicalWidth
      Data.publicFits relation).1
  have invocationCeilingBefore : PiCCSInvocations.invocationCeiling ≤
      phaseSuffixStart := by
    rw [PiCCSInvocations.invocationCeiling_eq]
    norm_num [
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
      phaseSuffixStart]
  refine ⟨?_, piCcsArithmeticRows_of_piRlcAgreesOutside relation before after
    agrees holds.arithmetic⟩
  intro invocation member
  have held := holds.invocations invocation member
  have heldPilot : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation before := by
    unfold PermutationInvocationHolds at held ⊢
    rw [Data.circuitPackage_permutation] at held
    exact held
  have preserved :=
    NightstreamFPrime.Export.Pilot.permutationInvocationHolds_of_agreesOutside
      invocation before after phaseSuffixStart phaseSuffixLength
      (by
        intro lane term termMember
        rcases stableInputs invocation member lane term termMember with
          inputBefore | inputPublic
        · apply Or.inl
          exact lt_of_lt_of_le inputBefore
            (Nat.le_trans (by
              have scheduled := allBefore invocation member
              omega) invocationCeilingBefore)
        · apply Or.inr
          rw [phaseSuffixEnd_eq]
          exact inputPublic)
      (by
        intro index below
        apply Or.inl
        have scheduled := allBefore invocation member
        exact lt_of_lt_of_le (by omega) invocationCeilingBefore)
      agrees heldPilot
  unfold PermutationInvocationHolds at preserved ⊢
  rw [Data.circuitPackage_permutation]
  exact preserved

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
    (pilotInstructions : ∀ instruction ∈
      Data.liftPilotInstructions (PilotData.witnessInstructions ()),
      instruction.Holds env)
    (pilotAssertions : ∀ row ∈
      Data.liftPilotRows (PilotData.assertionRows ()), row.Holds env)
    (piCcsArithmetic : R1CS.RowsHold env
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth
        Data.publicFits).map Rows.CompiledRow.toR1CS))
    (piRlcArithmetic : R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS))
    (piDecArithmetic : R1CS.RowsHold env
      ((PiDECArithmetic.canonicalPlan
        Data.logicalWidth Data.publicFits).rows.map
          Rows.CompiledRow.toR1CS)) :
    (Data.circuitPackage ()).RowsHold env := by
  have ordinary :=
    NightstreamFPrime.Export.Stage1.Package.phaseArithmeticRows_imply_packageOrdinary
      env pilotInstructions piCcsArithmetic piRlcArithmetic piDecArithmetic
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

/-- The five exact remapped PiRLC physical packets construct every canonical
PiRLC package component. -/
theorem piRlcRowsHold_of_packets
    (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    PiRLCRowsHold env := by
  exact ⟨PiRLCSamplerCompleteness.remappedPacket_implies_permutationInvocations
      env packets,
    PiRLCFirst54Completeness.remappedPacket_implies_first54Invocations
      env packets,
    PiRLCCombinationCompleteness.remappedPackets_imply_packageCombinationInvocations
      env packets,
    PiRLCSamplerCompleteness.remappedPacket_implies_ordinaryRows env packets⟩

/-- The pilot and all three exact phase-row packets assemble to the one
canonical circuit-package predicate in one final environment. -/
theorem rowsHold_of_phaseRows
    (env : Env)
    (pilotChains : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      HashChainHolds (Data.circuitPackage ()) chain env)
    (pilotInstructions : ∀ instruction ∈
      Data.liftPilotInstructions (PilotData.witnessInstructions ()),
      instruction.Holds env)
    (pilotAssertions : ∀ row ∈
      Data.liftPilotRows (PilotData.assertionRows ()), row.Holds env)
    (piCcs : PiCCSRowsHold env) (piRlc : PiRLCRowsHold env)
    (piDec : PiDECRowsHold env) :
    (Data.circuitPackage ()).RowsHold env := by
  exact rowsHold_of_packets env pilotChains piCcs.invocations
    piRlc.permutations piRlc.first54 piRlc.combinations pilotInstructions
    pilotAssertions
    piCcs.arithmetic piRlc.arithmetic piDec.arithmetic

/-- A valid semantic production PiRLC phase constructs all canonical PiRLC
package components in one final-column environment. -/
theorem complete_piRlcRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        PiRLCPackageCompleteness.phaseInterface
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai PiRLCPackageCompleteness.phaseInterface
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset) 8353953 ∧
        PiRLCRowsHold completed := by
  rcases PiRLCPackageCompleteness.completePackets relation ajtai env assumptions
      phase with ⟨completed, agrees, packets⟩
  exact ⟨completed, agrees, piRlcRowsHold_of_packets completed packets⟩

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
  have physicalEnd : PiCCSInvocations.invocationCeiling + 700767 ≤
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
        700767
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
