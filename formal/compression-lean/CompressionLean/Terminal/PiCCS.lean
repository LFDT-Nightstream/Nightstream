import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Lifecycle.VerifierContext
import CompressionLean.Terminal.Rows

/-!
Owns the PiCCS edge of the terminal package: its rows imply the exact
SuperNeo v1_1 PiCCS phase and bind both state preimages to the selected
verifier context.

The proofs are F′'s `Export/Stage1/Package` proofs with the terminal package
in place of the Stage 1 package. Only the membership steps differ.
-/

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem piCcsTranscriptSpecs
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    PiCCSInvocations.TranscriptSpecs Data.logicalWidth Data.publicFits env := by
  apply PiCCSInvocations.invocations_imply_transcriptSpecs Data.logicalWidth
    Data.publicFits relation env
  intro invocation member
  apply permutationInvocation_holds env holds invocation
  rw [Data.permutationInvocations_eq]
  exact List.mem_append_left _ member

/-- The terminal rows cover all twelve children of the F′ PiCCS circuit. -/
theorem piCcsSpecHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds relation
      (PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let parent := PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits
  have assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation parent
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
        Data.logicalWidth Data.publicFits)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  have transcripts := piCcsTranscriptSpecs relation env holds
  have packets := PiCCSArithmetic.arithmeticRows_imply_packetHolds
    Data.logicalWidth Data.publicFits env (piCcsArithmeticRows_hold env holds)
  have arithmetic := PiCCSArithmetic.packetHolds_imply_arithmeticSpecs
    Data.logicalWidth Data.publicFits relation env assumptions packets
  exact {
    statementBinding := arithmetic.statementBinding_parent
    statementAbsorption := transcripts.statementAbsorption_parent
    challenge := transcripts.challengeDerivation_parent
    roundTranscript := transcripts.roundTranscript_parent
    initialClaim := arithmetic.initialClaim_parent
    sumcheck := arithmetic.sumcheck_parent
    eval_K := arithmetic.evalK_parent
    eval_A := arithmetic.evalA_parent
    ccs := arithmetic.ccs_parent
    norm := arithmetic.norm_parent
    finalIdentity := arithmetic.finalIdentity_parent
    outputBinding := transcripts.outputBinding_parent relation }

/-- The terminal rows imply the exact SuperNeo v1_1 PiCCS phase. -/
theorem piCcsPhaseHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) template := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds
    relation ajtai
  exact piCcsSpecHolds relation env holds

/-- Both state preimages carry the context digest that the verifier selects
for its public context columns. -/
theorem selectedVerifierContext
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (authority : VerifierContext.Authority)
    (env : Env)
    (selected : Package.SelectedVerifierContext authority env)
    (holds : (circuitPackage ()).RowsHold env) :
    (∀ lane : Fin 4,
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (VerifierContext.digest authority).getD lane.val 0) ∧
    (∀ lane : Fin 4,
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (VerifierContext.digest authority).getD lane.val 0) := by
  have specification := piCcsSpecHolds relation env holds
  constructor
  · intro lane
    have context := specification.statementBinding.state.priorContext lane
    change
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContext lane).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) at context
    rw [context]
    change env (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart +
        lane.val)) = _
    rw [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_expectedContext]
    exact selected lane
  · intro lane
    have context := specification.statementBinding.state.outputContext lane
    change
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContext lane).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) at context
    rw [context]
    change env (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart +
        lane.val)) = _
    rw [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_expectedContext]
    exact selected lane

end CompressionLean.Terminal
