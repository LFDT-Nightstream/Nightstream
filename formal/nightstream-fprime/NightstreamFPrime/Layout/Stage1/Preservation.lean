import NightstreamFPrime.Layout.Stage1.CompactPullback
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition
import NightstreamFPrime.Layout.Stage1.PiCCSTranscriptRelocation
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
import NightstreamFPrime.Layout.Stage1.PiRLCGeneratedRelocation
import NightstreamFPrime.Layout.Stage1.PiRLCOutputRelocation
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBindingSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.PhaseTransport
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.GeneratedSupport
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerGeneratedSupport

/-!
Owns physical preservation for the complete Stage 1 row order.

The first theorem family is structural: one final row predicate decomposes
into the exact validated prefix, selected application, and NextPreimage
blocks, then into every phase-local physical predicate. No emitted row list is
evaluated in the kernel.
-/

namespace NightstreamFPrime.Layout.Stage1.Preservation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) : Prop :=
  R1CS.RowsHold env (Lowering.physicalRows relation program)

def ApplicationPhysicalHolds
    (program : Lifecycle.Stage1.Application.Program) (env : Env) : Prop :=
  R1CS.RowsHold env (Lowering.applicationRows program)

def NextPreimagePhysicalHolds (env : Env) : Prop :=
  R1CS.RowsHold env Lowering.nextPreimageRows

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PhysicalHolds relation program env ↔
      (R1CS.RowsHold env
          (Lowering.shiftRows program (Spartan.remappedRows relation)) ∧
        ApplicationPhysicalHolds program env) ∧
      NextPreimagePhysicalHolds env := by
  unfold PhysicalHolds Lowering.physicalRows ApplicationPhysicalHolds
    NextPreimagePhysicalHolds
  rw [R1CS.rowsHold_append, R1CS.rowsHold_append]

abbrev sourceEnv := CompactPullback.sourceEnv

def piCcsDelta (program : Lifecycle.Stage1.Application.Program) : Nat :=
  AssemblerInputs.piCcsOffset program - PilotPiCCS.piCcsOffset

theorem piCcsOffset_add_delta
    (program : Lifecycle.Stage1.Application.Program) :
    PilotPiCCS.piCcsOffset + piCcsDelta program =
      AssemblerInputs.piCcsOffset program := by
  unfold piCcsDelta AssemblerInputs.piCcsOffset
    AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    AssemblerInputs.rootOffset AssemblerInputs.applicationLocalStart
    AssemblerInputs.applicationWitnessStart PilotPiCCS.piCcsOffset
  rw [PiCCSInputs.phaseOffset_eq, Spartan.sourceColumnCount_eq]
  omega

private theorem sourceEnv_eq_compactEnv_of_piCcsSource
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat)
    (support : PiCCSOrdinarySourceSupport.Source index) :
    sourceEnv program env index = CompactPullback.compactEnv program env index := by
  symm
  exact CompactPullback.compactEnv_source program env index
    (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount support)

private theorem compactEnv_shiftedPiCcsLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat)
    (support : SupportRange.Extend (fun _ => False)
      PilotPiCCS.piCcsOffset (PilotPiCCS.piCcsOffset + 4581414) index) :
    CompactPullback.compactEnv program env (index + piCcsDelta program) =
      sourceEnv program env index := by
  rcases support with impossible | ⟨lower, upper⟩
  · contradiction
  · let relative := index - PilotPiCCS.piCcsOffset
    have relativeLt : relative < 4581414 := by
      dsimp only [relative]
      omega
    have sourceIndex : PilotPiCCS.piCcsOffset + relative = index := by
      dsimp only [relative]
      omega
    have targetIndex : AssemblerInputs.piCcsOffset program + relative =
        index + piCcsDelta program := by
      rw [← piCcsOffset_add_delta program]
      omega
    rw [← targetIndex,
      CompactPullback.compactEnv_piCcsLocal program env relative relativeLt,
      sourceIndex]

private theorem compactPiCcsExternalAgreement
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (support : PiCCSOrdinarySourceSupport.External index) :
    sourceEnv program env index = CompactPullback.compactEnv program env index :=
  sourceEnv_eq_compactEnv_of_piCcsSource program env index
    (PiCCSOrdinarySourceSupport.external_source index support)

private theorem compactPiCcsStateBinding
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (binding : Lifecycle.PiCCS.v1_1.StateBinding.SpecHolds
      (Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface
        (Lifecycle.PiCCS.v1_1.Formal.atOffset
          (PilotPiCCS.interface (publicFits := publicFits))
          PilotPiCCS.piCcsOffset)).state PilotPiCCS.piCcsOffset
      (sourceEnv program env)) :
    Lifecycle.PiCCS.v1_1.StateBinding.SpecHolds
      (Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface
        (Lifecycle.PiCCS.v1_1.Formal.atOffset
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program))).state
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) := by
  have baseBinding : Lifecycle.PiCCS.v1_1.StateBinding.SpecHolds
      (Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface
        (Lifecycle.PiCCS.v1_1.Formal.atOffset
          (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)).state PiCCSInputs.phaseOffset
      (sourceEnv program env) := by
    simpa [PilotPiCCS.interface, PilotPiCCS.piCcsOffset] using binding
  have transported :=
    Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.stateBinding_of_agree_satisfy
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      PiCCSOrdinarySourceSupport.External (sourceEnv program env)
      (CompactPullback.compactEnv program env)
      (PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits)
      (compactPiCcsExternalAgreement program env) baseBinding
  rcases transported with
    ⟨priorCanonical, outputCanonical, priorContext, outputContext⟩
  exact ⟨priorCanonical, outputCanonical, priorContext, outputContext⟩

private theorem compactCubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem compactPiCcsRunningPoint_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      PilotPiCCS.piCcsOffset (sourceEnv program env)).point =
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).point := by
  change Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
      (PiCCSInputs.runningExpr logicalWidth publicFits).point
      (sourceEnv program env) =
    Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
      (PiCCSInputs.runningExpr logicalWidth publicFits).point
      (CompactPullback.compactEnv program env)
  apply compactCubePoint_ext
  unfold Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
  change List.ofFn (fun coordinate =>
      ((PiCCSInputs.runningExpr logicalWidth publicFits).point coordinate).eval
        (sourceEnv program env)) =
    List.ofFn (fun coordinate =>
      ((PiCCSInputs.runningExpr logicalWidth publicFits).point coordinate).eval
        (CompactPullback.compactEnv program env))
  apply congrArg List.ofFn
  funext coordinate
  have support :=
    (PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
      ).runningPoint coordinate
  exact congrArg₂ K.mk
    (Expr.eval_eq_of_agree_satisfy
      ((PiCCSInputs.runningExpr logicalWidth publicFits).point coordinate).c0
        PiCCSOrdinarySourceSupport.External
        (sourceEnv program env) (CompactPullback.compactEnv program env)
        support.1 (compactPiCcsExternalAgreement program env))
    (Expr.eval_eq_of_agree_satisfy
      ((PiCCSInputs.runningExpr logicalWidth publicFits).point coordinate).c1
        PiCCSOrdinarySourceSupport.External
        (sourceEnv program env) (CompactPullback.compactEnv program env)
        support.2 (compactPiCcsExternalAgreement program env))

private theorem compactPiCcsExpr_eval_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (expression : Expr)
    (support : expression.VarsSatisfy PiCCSOrdinarySourceSupport.External) :
    expression.eval (sourceEnv program env) =
      expression.eval (CompactPullback.compactEnv program env) :=
  expression.eval_eq_of_agree_satisfy PiCCSOrdinarySourceSupport.External
    (sourceEnv program env) (CompactPullback.compactEnv program env) support
    (compactPiCcsExternalAgreement program env)

private theorem compactPiCcsKExpr_eval_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (value : KExpr)
    (support : value.c0.VarsSatisfy PiCCSOrdinarySourceSupport.External ∧
      value.c1.VarsSatisfy PiCCSOrdinarySourceSupport.External) :
    value.eval (sourceEnv program env) =
      value.eval (CompactPullback.compactEnv program env) := by
  exact congrArg₂ K.mk
    (compactPiCcsExpr_eval_eq program env value.c0 support.1)
    (compactPiCcsExpr_eval_eq program env value.c1 support.2)

private theorem compactEvaluationFamily_ext
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem compactFullOutput_ext
    (left right : FullOutputCoordinates.FullOutput K productionShape)
    (pad : left.padCoordinate = right.padCoordinate)
    (matrix : left.matrixCoordinate = right.matrixCoordinate) : left = right := by
  cases left
  cases right
  simp_all

private theorem compactPiCcsRunningCommitments_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env)).commitments =
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).commitments := by
  funext source row coefficient
  change ((PiCCSInputs.runningExpr logicalWidth publicFits).commitment
      source row coefficient).eval (sourceEnv program env) =
    ((PiCCSInputs.runningExpr logicalWidth publicFits).commitment
      source row coefficient).eval (CompactPullback.compactEnv program env)
  exact compactPiCcsExpr_eval_eq program env _
    ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
      ).runningCommitment source row coefficient)

private theorem compactPiCcsRunningPublicInputs_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env)).publicInputs =
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).publicInputs := by
  funext source column
  change ((PiCCSInputs.runningExpr logicalWidth publicFits).publicInput
      source column).eval (sourceEnv program env) =
    ((PiCCSInputs.runningExpr logicalWidth publicFits).publicInput
      source column).eval (CompactPullback.compactEnv program env)
  exact compactPiCcsExpr_eval_eq program env _
    ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
      ).runningPublicInput source column)

private theorem compactPiCcsRunningEvaluations_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env)).evaluations =
    (Lifecycle.PiCCS.v1_1.Formal.evalRunning
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).evaluations := by
  funext source
  apply compactEvaluationFamily_ext
  · funext coefficient
    change (((PiCCSInputs.runningExpr logicalWidth publicFits).evaluation source
      ).eval_K coefficient).eval (sourceEnv program env) =
      (((PiCCSInputs.runningExpr logicalWidth publicFits).evaluation source
        ).eval_K coefficient).eval (CompactPullback.compactEnv program env)
    exact compactPiCcsKExpr_eval_eq program env _
      ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
        ).runningEval_K source coefficient)
  · funext matrix coefficient
    change (((PiCCSInputs.runningExpr logicalWidth publicFits).evaluation source
      ).eval_A matrix coefficient).eval (sourceEnv program env) =
      (((PiCCSInputs.runningExpr logicalWidth publicFits).evaluation source
        ).eval_A matrix coefficient).eval
          (CompactPullback.compactEnv program env)
    exact compactPiCcsKExpr_eval_eq program env _
      ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
        ).runningEval_A source matrix coefficient)

private theorem compactPiCcsFreshCommitments_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalFresh
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env)).commitments =
    (Lifecycle.PiCCS.v1_1.Formal.evalFresh
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).commitments := by
  funext source row coefficient
  change ((PiCCSInputs.freshExpr logicalWidth publicFits).commitment
      source row coefficient).eval (sourceEnv program env) =
    ((PiCCSInputs.freshExpr logicalWidth publicFits).commitment
      source row coefficient).eval (CompactPullback.compactEnv program env)
  exact compactPiCcsExpr_eval_eq program env _
    ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
      ).freshCommitment source row coefficient)

private theorem compactPiCcsFreshPublicInputs_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalFresh
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env)).publicInputs =
    (Lifecycle.PiCCS.v1_1.Formal.evalFresh
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env)).publicInputs := by
  funext source column
  change ((PiCCSInputs.freshExpr logicalWidth publicFits).publicInput
      source column).eval (sourceEnv program env) =
    ((PiCCSInputs.freshExpr logicalWidth publicFits).publicInput
      source column).eval (CompactPullback.compactEnv program env)
  exact compactPiCcsExpr_eval_eq program env _
    ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
      ).freshPublicInput source column)

private theorem compactPiCcsProofRounds_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env) template).piCcsRounds =
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) template).piCcsRounds := by
  funext roundIndex
  change (PiCCSInputs.roundMessage roundIndex).semanticPolynomial
      (sourceEnv program env) =
    (PiCCSInputs.roundMessage roundIndex).semanticPolynomial
      (CompactPullback.compactEnv program env)
  exact Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.messagePolynomial_eq_of_agree_satisfy
      (PiCCSInputs.roundMessage roundIndex)
      PiCCSOrdinarySourceSupport.External (sourceEnv program env)
      (CompactPullback.compactEnv program env)
      (fun coefficient =>
        (PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
          ).roundCoefficient roundIndex coefficient)
      (compactPiCcsExternalAgreement program env)

private theorem compactPiCcsProofOutput_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env) template).piCcsOutput =
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) template).piCcsOutput := by
  apply compactFullOutput_ext
  · funext source coefficient
    change (PiCCSInputs.outputEval_K source coefficient).eval
        (sourceEnv program env) =
      (PiCCSInputs.outputEval_K source coefficient).eval
        (CompactPullback.compactEnv program env)
    exact compactPiCcsKExpr_eval_eq program env _
      ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
        ).outputEval_K source coefficient)
  · funext source matrix coefficient
    change (PiCCSInputs.outputEval_A source matrix coefficient).eval
        (sourceEnv program env) =
      (PiCCSInputs.outputEval_A source matrix coefficient).eval
        (CompactPullback.compactEnv program env)
    exact compactPiCcsKExpr_eval_eq program env _
      ((PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
        ).outputEval_A source matrix coefficient)

private theorem compactPiCcsProofCommitments_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env) template).piDecCommitments =
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) template).piDecCommitments := by
  change template.piDecCommitments = template.piDecCommitments
  rfl

private theorem compactPiCcsProofEvaluations_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) (env : Env) :
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (sourceEnv program env) template).piDecEvaluations =
    (Lifecycle.PiCCS.v1_1.Formal.evalProof relation
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) template).piDecEvaluations := by
  change template.piDecEvaluations = template.piDecEvaluations
  rfl

def piRlcDelta (program : Lifecycle.Stage1.Application.Program) : Nat :=
  AssemblerInputs.piRlcOffset program - PilotPiCCSPiRLC.piRlcOffset

theorem piRlcOffset_add_delta
    (program : Lifecycle.Stage1.Application.Program) :
    PilotPiCCSPiRLC.piRlcOffset + piRlcDelta program =
      AssemblerInputs.piRlcOffset program := by
  unfold piRlcDelta AssemblerInputs.piRlcOffset
    AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
    AssemblerInputs.priorOffset AssemblerInputs.rootOffset
    AssemblerInputs.applicationLocalStart
    AssemblerInputs.applicationWitnessStart
  rw [PilotPiCCSPiRLC.piRlcOffset_eq, Spartan.sourceColumnCount_eq]
  omega

private theorem compactEnv_shiftedPiRlcLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat)
    (support : SupportRange.Extend (fun _ => False)
      PilotPiCCSPiRLC.piRlcOffset
      (PilotPiCCSPiRLC.piRlcOffset + 315894) index) :
    CompactPullback.compactEnv program env (index + piRlcDelta program) =
      sourceEnv program env index := by
  rcases support with impossible | ⟨lower, upper⟩
  · contradiction
  · let relative := index - PilotPiCCSPiRLC.piRlcOffset
    have relativeLt : relative < 315894 := by
      dsimp only [relative]
      omega
    have sourceIndex : PilotPiCCSPiRLC.piRlcOffset + relative = index := by
      dsimp only [relative]
      omega
    have targetIndex : AssemblerInputs.piRlcOffset program + relative =
        index + piRlcDelta program := by
      rw [← piRlcOffset_add_delta program]
      omega
    rw [← targetIndex,
      CompactPullback.compactEnv_piRlcLocal program env relative relativeLt,
      sourceIndex]

private theorem compactPiCcsOutputSupport
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Duplex.Formal.StateSupported
      (Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState relation
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset)
      (SupportRange.Extend (fun _ => False) PilotPiCCS.piCcsOffset
        (PilotPiCCS.piCcsOffset + 4581414)) := by
  intro lane
  unfold Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState
  rw [congrFun (PiCCSTranscriptRelocation.outputFinalState_direct
    (Lifecycle.PiCCS.v1_1.Formal.outputBindingInterface
      (Lifecycle.PiCCS.v1_1.Formal.atOffset
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset))
    (Lifecycle.PiCCS.v1_1.Formal.outputBindingOffset relation
      (PilotPiCCS.interface (publicFits := publicFits))
      PilotPiCCS.piCcsOffset)) lane]
  simp only [Permutation.scheduleOutput, Permutation.freshState,
    Expr.VarsSatisfy]
  apply Or.inr
  constructor
  · rw [Lifecycle.PiCCS.v1_1.Formal.outputBindingOffset_eq_outputBindingRowOffset]
    unfold Lifecycle.PiCCS.v1_1.Formal.outputBindingRowOffset
      Lifecycle.PiCCS.v1_1.Formal.finalIdentityRowOffset
      Lifecycle.PiCCS.v1_1.Formal.normRowOffset
      Lifecycle.PiCCS.v1_1.Formal.ccsRowOffset
      Lifecycle.PiCCS.v1_1.Formal.evalARowOffset
      Lifecycle.PiCCS.v1_1.Formal.evalKRowOffset
      Lifecycle.PiCCS.v1_1.Formal.sumcheckRowOffset
      Lifecycle.PiCCS.v1_1.Formal.initialClaimRowOffset
    omega
  · have finish :=
      Lifecycle.PiCCS.v1_1.Formal.finalRowOffset_eq_add_of_degreeBound_eq_nine
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset (ProductionKey.degreeBound_eq relation)
    unfold Lifecycle.PiCCS.v1_1.Formal.finalRowOffset at finish
    rw [← Lifecycle.PiCCS.v1_1.Formal.outputBindingOffset_eq_outputBindingRowOffset
      relation (PilotPiCCS.interface (publicFits := publicFits))
      PilotPiCCS.piCcsOffset] at finish
    have laneBound := lane.isLt
    change lane.val < 8 at laneBound
    omega

private theorem compactPiCcsOutgoingState_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiCCS.v1_1.StatementAbsorption.evalState
        (sourceEnv program env)
        (Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState relation
          (PilotPiCCS.interface (publicFits := publicFits))
          PilotPiCCS.piCcsOffset) =
      Lifecycle.PiCCS.v1_1.StatementAbsorption.evalState
        (CompactPullback.compactEnv program env)
        (Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState relation
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)) := by
  unfold Lifecycle.PiCCS.v1_1.StatementAbsorption.evalState Layer.evalState
  apply congrArg List.ofFn
  funext lane
  rw [← piCcsOffset_add_delta program,
    congrFun (PiCCSTranscriptRelocation.formalOutputFinalState_shift relation
      (PilotPiCCS.interface (publicFits := publicFits))
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      PilotPiCCS.piCcsOffset (piCcsDelta program)) lane]
  exact (state_eval_eq_of_shift_agreement (piCcsDelta program)
    (Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState relation
      (PilotPiCCS.interface (publicFits := publicFits))
      PilotPiCCS.piCcsOffset)
    (SupportRange.Extend (fun _ => False) PilotPiCCS.piCcsOffset
      (PilotPiCCS.piCcsOffset + 4581414))
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiCcsOutputSupport relation)
    (compactEnv_shiftedPiCcsLocal program env) lane).symm

private theorem compactPiCcsRunning_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiCCS.v1_1.Formal.evalRunning
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset (sourceEnv program env) =
      Lifecycle.PiCCS.v1_1.Formal.evalRunning
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program)
        (CompactPullback.compactEnv program env) := by
  exact Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.running_ext _ _
    (compactPiCcsRunningPoint_eq program env)
    (compactPiCcsRunningCommitments_eq program env)
    (compactPiCcsRunningPublicInputs_eq program env)
    (compactPiCcsRunningEvaluations_eq program env)

private theorem compactPiCcsFresh_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiCCS.v1_1.Formal.evalFresh
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset (sourceEnv program env) =
      Lifecycle.PiCCS.v1_1.Formal.evalFresh
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program)
        (CompactPullback.compactEnv program env) := by
  exact Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.fresh_ext _ _
    (compactPiCcsFreshCommitments_eq program env)
    (compactPiCcsFreshPublicInputs_eq program env)

private theorem compactPiCcsProof_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) (env : Env) :
    Lifecycle.PiCCS.v1_1.Formal.evalProof relation
        (PilotPiCCS.interface (publicFits := publicFits))
        PilotPiCCS.piCcsOffset (sourceEnv program env) template =
      Lifecycle.PiCCS.v1_1.Formal.evalProof relation
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program)
        (CompactPullback.compactEnv program env) template := by
  exact Lifecycle.PiCCS.v1_1.Formal.PhaseTransport.proof_ext _ _
    (compactPiCcsProofRounds_eq relation program template env)
    (compactPiCcsProofOutput_eq relation program template env)
    (compactPiCcsProofCommitments_eq relation program template env)
    (compactPiCcsProofEvaluations_eq relation program template env)

theorem compactPiCcsPhaseHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (phase : Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PilotPiCCS.interface (publicFits := publicFits))
      PilotPiCCS.piCcsOffset (sourceEnv program env) template) :
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      (AssemblerInputs.piCcsOffset program)
      (CompactPullback.compactEnv program env) template := by
  have roundPointEq :=
    PiCCSTranscriptRelocation.evalRoundPoint_eq_of_shift_agreement
      (PilotPiCCS.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      (AssemblerInputs.piCcsInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      PilotPiCCS.piCcsOffset (piCcsDelta program) (sourceEnv program env)
      (CompactPullback.compactEnv program env)
      (compactEnv_shiftedPiCcsLocal program env)
  rw [piCcsOffset_add_delta program] at roundPointEq
  have runningEq := compactPiCcsRunning_eq
    (logicalWidth := logicalWidth) (publicFits := publicFits) program env
  have freshEq := compactPiCcsFresh_eq
    (logicalWidth := logicalWidth) (publicFits := publicFits) program env
  have proofEq := compactPiCcsProof_eq relation program template env
  have inputsEq :
      ((Lifecycle.PiCCS.v1_1.Formal.evalRunning
          (PilotPiCCS.interface (publicFits := publicFits))
          PilotPiCCS.piCcsOffset (sourceEnv program env),
        Lifecycle.PiCCS.v1_1.Formal.evalFresh
          (PilotPiCCS.interface (publicFits := publicFits))
          PilotPiCCS.piCcsOffset (sourceEnv program env)),
        Lifecycle.PiCCS.v1_1.Formal.evalProof relation
          (PilotPiCCS.interface (publicFits := publicFits))
          PilotPiCCS.piCcsOffset (sourceEnv program env) template) =
      ((Lifecycle.PiCCS.v1_1.Formal.evalRunning
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)
          (CompactPullback.compactEnv program env),
        Lifecycle.PiCCS.v1_1.Formal.evalFresh
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)
          (CompactPullback.compactEnv program env)),
        Lifecycle.PiCCS.v1_1.Formal.evalProof relation
          (AssemblerInputs.piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (AssemblerInputs.piCcsOffset program)
          (CompactPullback.compactEnv program env) template) :=
    congrArg₂ Prod.mk (congrArg₂ Prod.mk runningEq freshEq) proofEq
  have acceptedEq := congrArg (fun values =>
    NightstreamFPrime.Spec.Folding.PiCCS.Accepted
      (ProductionKey.key relation ajtai) values.1.1 values.1.2 values.2) inputsEq
  have executionEq := congrArg (fun values =>
    (ProductionKey.key relation ajtai).piCcsExecution
      values.1.1 values.1.2 values.2) inputsEq
  exact {
    stateBinding := compactPiCcsStateBinding program env phase.stateBinding
    accepted := Eq.mp acceptedEq phase.accepted
    roundPoint := roundPointEq.trans (phase.roundPoint.trans
      (congrArg (fun execution => execution.coins.roundPoint) executionEq))
    outgoingState :=
      (compactPiCcsOutgoingState_eq relation program env).symm.trans
        (phase.outgoingState.trans
          (congrArg (fun execution => execution.outgoingState) executionEq)) }

private theorem compactPilotPriorCanonical
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (specification : Lifecycle.PriorStateHash.SpecHolds
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (sourceEnv program env)) :
    Lifecycle.PriorStateHash.SpecHolds PilotProduction.priorInterface
      PilotProduction.witnessOffset (CompactPullback.compactEnv program env) := by
  apply Lifecycle.PriorStateHash.specHolds_of_agree_below
    PilotProduction.priorInterface PilotProduction.witnessOffset
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (PilotProduction.assumptions (sourceEnv program env)).1
  · intro index bounded
    apply CompactPullback.compactEnv_source
    rw [PilotProduction.witnessOffset_eq] at bounded
    rw [Spartan.sourceColumnCount_eq]
    omega
  · exact specification

private theorem compactPilotOutputCanonical
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (specification : Lifecycle.OutputHash.SpecHolds
      PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset) (sourceEnv program env)) :
    Lifecycle.OutputHash.SpecHolds PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset)
      (CompactPullback.compactEnv program env) := by
  apply Lifecycle.OutputHash.specHolds_of_agree_below
    PilotProduction.outputInterface
    (Lifecycle.Pilot.outputOffset PilotProduction.interface
      PilotProduction.witnessOffset)
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (PilotProduction.assumptions (sourceEnv program env)).2
  · intro index bounded
    apply CompactPullback.compactEnv_source
    rw [← PilotProduction.lifecycleOutputOffset_matches] at bounded
    change index < 7410524 at bounded
    rw [Spartan.sourceColumnCount_eq]
    omega
  · exact specification

theorem compactPilotPrior
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (specification : Lifecycle.PriorStateHash.SpecHolds
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (sourceEnv program env)) :
    Lifecycle.PriorStateHash.SpecHolds PilotProduction.priorInterface
      (AssemblerInputs.priorOffset program)
      (CompactPullback.compactEnv program env) := by
  have canonical := compactPilotPriorCanonical program env specification
  unfold Lifecycle.PriorStateHash.SpecHolds at canonical ⊢
  simpa [PilotProduction.priorInterface, PilotProduction.makePriorInterface,
    PilotProduction.priorPreimage, PilotProduction.priorPublicInput] using
    canonical

theorem compactPilotOutput
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (specification : Lifecycle.OutputHash.SpecHolds
      PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset) (sourceEnv program env)) :
    Lifecycle.OutputHash.SpecHolds PilotProduction.outputInterface
      (AssemblerInputs.outputHashOffset program)
      (CompactPullback.compactEnv program env) := by
  have canonical := compactPilotOutputCanonical program env specification
  unfold Lifecycle.OutputHash.SpecHolds
    NightstreamFPrime.Gadgets.Poseidon2.Formal.SpecHolds
    Lifecycle.OutputHash.hashInterface at canonical ⊢
  simpa [PilotProduction.outputInterface, PilotProduction.makeOutputInterface,
    PilotProduction.outputPreimage, PilotProduction.outputDigest] using canonical

private theorem compactPiCcsRoundPointCoordinate_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (coordinate : Fin productionShape.cubeVariables) :
    (Lifecycle.PiCCS.v1_1.Formal.roundPoint
      (Lifecycle.PiCCS.v1_1.Formal.atOffset
        (PilotPiCCS.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCS.piCcsOffset)
      PilotPiCCS.piCcsOffset coordinate).eval (sourceEnv program env) =
    (Lifecycle.PiCCS.v1_1.Formal.roundPoint
      (Lifecycle.PiCCS.v1_1.Formal.atOffset
        (AssemblerInputs.piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program)
        (AssemblerInputs.piCcsOffset program))
      (AssemblerInputs.piCcsOffset program) coordinate).eval
        (CompactPullback.compactEnv program env) := by
  have shift := (PiCCSTranscriptRelocation.formalRoundOutputs_shift
    (PilotPiCCS.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (AssemblerInputs.piCcsInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) program)
    PilotPiCCS.piCcsOffset (piCcsDelta program)).1 coordinate
  rw [piCcsOffset_add_delta program] at shift
  rw [shift]
  exact (quadratic_eval_eq_of_shift_agreement (piCcsDelta program) _ _
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    ((PiCCSTranscriptRelocation.formalRoundOutputs_localSupport
      (PilotPiCCS.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PilotPiCCS.piCcsOffset).1 coordinate)
    (compactEnv_shiftedPiCcsLocal program env)).symm

private theorem compactPiRlcPoint_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.InputBinding.evalPoint
        ((PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits)).point
          PilotPiCCSPiRLC.piRlcOffset)
        (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.InputBinding.evalPoint
        ((AssemblerInputs.piRlcInterface relation program).point
          (AssemblerInputs.piRlcOffset program))
        (CompactPullback.compactEnv program env) := by
  apply compactCubePoint_ext
  change List.ofFn _ = List.ofFn _
  apply congrArg List.ofFn
  funext coordinate
  simpa [Lifecycle.PiRLC.v1_1.InputBinding.evalPoint,
    Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint,
    PiRLCInputs.interface, PiRLCInputs.piCcsInterface,
    AssemblerInputs.piRlcInterface, AssemblerInputs.piCcsRoundPoint,
    Lifecycle.PiCCS.v1_1.Formal.atOffset] using
      compactPiCcsRoundPointCoordinate_eq program env coordinate

private theorem sourceEnv_eq_compactEnv_belowPiRlc
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    ∀ index, index < PiRLCInputs.phaseOffset →
      sourceEnv program env index = CompactPullback.compactEnv program env index := by
  intro index bounded
  symm
  apply CompactPullback.compactEnv_source
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiRLCInputs.phaseOffset] at bounded ⊢
  omega

private theorem compactPiRlcInputs_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.Semantics.evalInputs relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.Semantics.evalInputs relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  funext source
  let sourceIndex := Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source
  change Lifecycle.PiRLC.v1_1.InputBinding.evalInput relation
      (PiRLCInputs.sourceInput sourceIndex)
      ((PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits)).point
        PilotPiCCSPiRLC.piRlcOffset) (sourceEnv program env) =
    Lifecycle.PiRLC.v1_1.InputBinding.evalInput relation
      (PiRLCInputs.sourceInput sourceIndex)
      ((AssemblerInputs.piRlcInterface relation program).point
        (AssemblerInputs.piRlcOffset program))
      (CompactPullback.compactEnv program env)
  exact PiRLCInputBounds.sourceInput_eval_eq_of_point_and_agree_below
    relation sourceIndex _ _ _ _ (compactPiRlcPoint_eq relation program env)
    (sourceEnv_eq_compactEnv_belowPiRlc program env)

private theorem compactPiRlcInitialState_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.SamplerChain.evalInitialState
        (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
          (Lifecycle.PiRLC.v1_1.Formal.atOffset
            (PiRLCInputs.interface
              (logicalWidth := logicalWidth) (publicFits := publicFits))
            PilotPiCCSPiRLC.piRlcOffset))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.SamplerChain.evalInitialState
        (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
          (Lifecycle.PiRLC.v1_1.Formal.atOffset
            (AssemblerInputs.piRlcInterface relation program)
            (AssemblerInputs.piRlcOffset program)))
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  change Lifecycle.PiRLC.v1_1.Sampler.evalState (sourceEnv program env)
      (PiRLCInputs.piCcsOutputState
        (logicalWidth := logicalWidth) (publicFits := publicFits)) =
    Lifecycle.PiRLC.v1_1.Sampler.evalState
      (CompactPullback.compactEnv program env)
      (AssemblerInputs.piCcsOutputState relation program)
  rw [PiRLCInputs.piCcsOutputState_eq_parent relation]
  exact compactPiCcsOutgoingState_eq relation program env

private theorem compactPiRlcOutputSlot_val
    (position : Fin ringDegree) :
    (Lifecycle.PiRLC.v1_1.Sampler.outputSlot position).val = position.val := by
  rfl

private theorem compactPiRlcOutputWordSupport
    (source : Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount)
    (position : Fin ringDegree) :
    (Lifecycle.PiRLC.v1_1.Sampler.outputWord
      (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        PilotPiCCSPiRLC.piRlcOffset source.val) position).VarsSatisfy
      (SupportRange.Extend (fun _ => False) PilotPiCCSPiRLC.piRlcOffset
        (PilotPiCCSPiRLC.piRlcOffset + 315894)) := by
  simp only [Lifecycle.PiRLC.v1_1.Sampler.outputWord,
    First54.output, First54ValueStep.output, Expr.VarsSatisfy]
  apply Or.inr
  constructor
  · unfold Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
      Lifecycle.PiRLC.v1_1.Sampler.selectorOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowBase
      First54.valueOffset First54.positionOffset
    omega
  · have sourceBound := source.isLt
    have positionBound := position.isLt
    have slotValue := compactPiRlcOutputSlot_val position
    change source.val < 17 at sourceBound
    change position.val < 54 at positionBound
    unfold Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
      Lifecycle.PiRLC.v1_1.Sampler.selectorOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowBase
      First54.valueOffset First54.positionOffset
    norm_num [Lifecycle.PiRLC.v1_1.Sampler.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.Sampler.digestRoundCount,
      Lifecycle.PiRLC.v1_1.DigestWindow.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.DigestLane.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.Sampler.entryPrivateCount,
      First54.candidateCount, First54.roundPrivateCount,
      First54Step.slotCount, First54ValueStep.outputCount,
      First54.outputCount] at ⊢
    omega

private theorem compactPiRlcOutputStateSupport
    (source : Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount) :
    Duplex.Formal.StateSupported
      (Lifecycle.PiRLC.v1_1.Sampler.outputState
        (Lifecycle.PiRLC.v1_1.SamplerChain.childInterface
          (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
            (Lifecycle.PiRLC.v1_1.Formal.atOffset
              (PiRLCInputs.interface
                (logicalWidth := logicalWidth) (publicFits := publicFits))
              PilotPiCCSPiRLC.piRlcOffset))
          PilotPiCCSPiRLC.piRlcOffset source.val)
        source.val
        (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
          PilotPiCCSPiRLC.piRlcOffset source.val))
      (SupportRange.Extend (fun _ => False) PilotPiCCSPiRLC.piRlcOffset
        (PilotPiCCSPiRLC.piRlcOffset + 315894)) := by
  intro lane
  simp only [Lifecycle.PiRLC.v1_1.Sampler.outputState,
    Lifecycle.PiRLC.v1_1.DigestWindow.output,
    Permutation.Owned.output, Permutation.scheduleOutput,
    Permutation.freshState, Expr.VarsSatisfy]
  apply Or.inr
  constructor
  · unfold Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowBase
      Lifecycle.PiRLC.v1_1.DigestWindow.permutationOffset
    omega
  · have sourceBound := source.isLt
    have laneBound := lane.isLt
    change source.val < 17 at sourceBound
    change lane.val < 8 at laneBound
    unfold Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowOffset
      Lifecycle.PiRLC.v1_1.Sampler.windowBase
      Lifecycle.PiRLC.v1_1.DigestWindow.permutationOffset
    norm_num [Lifecycle.PiRLC.v1_1.Sampler.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.Sampler.digestRoundCount,
      Lifecycle.PiRLC.v1_1.DigestWindow.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.DigestLane.logicalPrivateCount,
      Lifecycle.PiRLC.v1_1.Sampler.entryPrivateCount]
    omega

private theorem compactPiRlcSourceOffset_shift
    (program : Lifecycle.Stage1.Application.Program) (source : Nat) :
    Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        (AssemblerInputs.piRlcOffset program) source =
      Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        PilotPiCCSPiRLC.piRlcOffset source + piRlcDelta program := by
  rw [← piRlcOffset_add_delta program]
  unfold Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
  omega

private theorem compactPiRlcOutputWord_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (source : Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount)
    (position : Fin ringDegree) :
    (Lifecycle.PiRLC.v1_1.Sampler.outputWord
      (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        PilotPiCCSPiRLC.piRlcOffset source.val) position).eval
        (sourceEnv program env) =
      (Lifecycle.PiRLC.v1_1.Sampler.outputWord
        (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
          (AssemblerInputs.piRlcOffset program) source.val) position).eval
        (CompactPullback.compactEnv program env) := by
  rw [compactPiRlcSourceOffset_shift program source.val]
  rw [PiRLCGeneratedRelocation.samplerOutputWord_shift]
  exact (expression_eval_eq_of_shift_agreement (piRlcDelta program)
    (Lifecycle.PiRLC.v1_1.Sampler.outputWord
      (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        PilotPiCCSPiRLC.piRlcOffset source.val) position)
    (SupportRange.Extend (fun _ => False) PilotPiCCSPiRLC.piRlcOffset
      (PilotPiCCSPiRLC.piRlcOffset + 315894))
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiRlcOutputWordSupport source position)
    (compactEnv_shiftedPiRlcLocal program env)).symm

private theorem compactPiRlcOutputCoefficients_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (source : Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount) :
    Lifecycle.PiRLC.v1_1.Sampler.outputCoefficients (sourceEnv program env)
        (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
          PilotPiCCSPiRLC.piRlcOffset source.val) =
      Lifecycle.PiRLC.v1_1.Sampler.outputCoefficients
        (CompactPullback.compactEnv program env)
        (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
          (AssemblerInputs.piRlcOffset program) source.val) := by
  unfold Lifecycle.PiRLC.v1_1.Sampler.outputCoefficients First54.evalOutput
  apply congrArg List.ofFn
  funext slot
  let position : Fin ringDegree :=
    Fin.cast Lifecycle.PiRLC.v1_1.Sampler.outputCount_eq_ringDegree slot
  have wordEq := compactPiRlcOutputWord_eq program env source position
  simpa [Lifecycle.PiRLC.v1_1.Sampler.outputWord,
    Lifecycle.PiRLC.v1_1.Sampler.outputSlot, position] using wordEq

private theorem compactPiRlcChallenges_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.SamplerChain.evalChallenges
        (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
          (Lifecycle.PiRLC.v1_1.Formal.atOffset
            (PiRLCInputs.interface
              (logicalWidth := logicalWidth) (publicFits := publicFits))
            PilotPiCCSPiRLC.piRlcOffset))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.SamplerChain.evalChallenges
        (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
          (Lifecycle.PiRLC.v1_1.Formal.atOffset
            (AssemblerInputs.piRlcInterface relation program)
            (AssemblerInputs.piRlcOffset program)))
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  apply Lifecycle.PiRLC.v1_1.SamplerChain.evalChallenges_eq_of_outputWord_eq
  intro source position
  exact compactPiRlcOutputWord_eq program env source position

private theorem compactPiRlcOutputState_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (source : Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount) :
    Lifecycle.PiRLC.v1_1.Sampler.evalState (sourceEnv program env)
        (Lifecycle.PiRLC.v1_1.Sampler.outputState
          (Lifecycle.PiRLC.v1_1.SamplerChain.childInterface
            (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
              (Lifecycle.PiRLC.v1_1.Formal.atOffset
                (PiRLCInputs.interface
                  (logicalWidth := logicalWidth) (publicFits := publicFits))
                PilotPiCCSPiRLC.piRlcOffset))
            PilotPiCCSPiRLC.piRlcOffset source.val)
          source.val
          (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
            PilotPiCCSPiRLC.piRlcOffset source.val)) =
      Lifecycle.PiRLC.v1_1.Sampler.evalState
        (CompactPullback.compactEnv program env)
        (Lifecycle.PiRLC.v1_1.Sampler.outputState
          (Lifecycle.PiRLC.v1_1.SamplerChain.childInterface
            (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
              (Lifecycle.PiRLC.v1_1.Formal.atOffset
                (AssemblerInputs.piRlcInterface relation program)
                (AssemblerInputs.piRlcOffset program)))
            (AssemblerInputs.piRlcOffset program) source.val)
          source.val
          (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
            (AssemblerInputs.piRlcOffset program) source.val)) := by
  rw [compactPiRlcSourceOffset_shift program source.val]
  rw [PiRLCGeneratedRelocation.samplerOutputState_shift]
  unfold Lifecycle.PiRLC.v1_1.Sampler.evalState
  apply congrArg List.ofFn
  funext lane
  exact (state_eval_eq_of_shift_agreement (piRlcDelta program)
    (Lifecycle.PiRLC.v1_1.Sampler.outputState
      (Lifecycle.PiRLC.v1_1.SamplerChain.childInterface
        (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
          (Lifecycle.PiRLC.v1_1.Formal.atOffset
            (PiRLCInputs.interface
              (logicalWidth := logicalWidth) (publicFits := publicFits))
            PilotPiCCSPiRLC.piRlcOffset))
        PilotPiCCSPiRLC.piRlcOffset source.val)
      source.val
      (Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
        PilotPiCCSPiRLC.piRlcOffset source.val))
    (SupportRange.Extend (fun _ => False) PilotPiCCSPiRLC.piRlcOffset
      (PilotPiCCSPiRLC.piRlcOffset + 315894))
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiRlcOutputStateSupport source)
    (compactEnv_shiftedPiRlcLocal program env) lane).symm

private theorem compactPiRlcSampler
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (sampler : Lifecycle.PiRLC.v1_1.SamplerChain.RelationHolds
      (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
        (Lifecycle.PiRLC.v1_1.Formal.atOffset
          (PiRLCInputs.interface
            (logicalWidth := logicalWidth) (publicFits := publicFits))
          PilotPiCCSPiRLC.piRlcOffset))
      PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env)) :
    Lifecycle.PiRLC.v1_1.SamplerChain.RelationHolds
      (Lifecycle.PiRLC.v1_1.Formal.samplerInterface
        (Lifecycle.PiRLC.v1_1.Formal.atOffset
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program)))
      (AssemblerInputs.piRlcOffset program)
      (CompactPullback.compactEnv program env) := by
  apply Lifecycle.PiRLC.v1_1.SamplerChain.RelationHolds.of_cross_eval_eq
  · intro count countBound
    cases count with
    | zero => exact compactPiRlcInitialState_eq relation program env
    | succ source =>
        let sourceIndex :
            Fin Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount :=
          ⟨source, by omega⟩
        simpa [Lifecycle.PiRLC.v1_1.SamplerChain.evalStateAt, sourceIndex] using
          compactPiRlcOutputState_eq relation program env sourceIndex
  · intro source
    exact compactPiRlcOutputCoefficients_eq program env source
  · intro source
    exact compactPiRlcOutputState_eq relation program env source
  · exact compactPiRlcChallenges_eq relation program env
  · exact sampler

private theorem compactPiRlcSemanticChallenges_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.Semantics.evalChallenges
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.Semantics.evalChallenges
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  funext source
  exact congrFun (compactPiRlcChallenges_eq relation program env)
    (Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source)

private theorem compactPiRlcOutput_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  have relocated := PiRLCOutputRelocation.evalOutput_eq_of_shift_agreement
    relation
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (AssemblerInputs.piRlcInterface relation program)
    PilotPiCCSPiRLC.piRlcOffset (piRlcDelta program)
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (by
      rw [piRlcOffset_add_delta program]
      exact compactPiRlcPoint_eq relation program env)
    (by
      simpa [Lifecycle.PiRLC.v1_1.Formal.logicalPrivateCount] using
        compactEnv_shiftedPiRlcLocal program env)
  rw [piRlcOffset_add_delta program] at relocated
  exact relocated

private theorem compactPiRlcAttempt_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiRLC.v1_1.Semantics.attempt relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) =
      Lifecycle.PiRLC.v1_1.Semantics.attempt relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) := by
  exact Lifecycle.PiRLC.v1_1.Semantics.attempt_eq_of_cross_components
    relation
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (AssemblerInputs.piRlcInterface relation program)
    PilotPiCCSPiRLC.piRlcOffset (AssemblerInputs.piRlcOffset program)
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiRlcInputs_eq relation program env)
    (compactPiRlcSemanticChallenges_eq relation program env)
    (compactPiRlcOutput_eq relation program env)

theorem compactPiRlcPhaseHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (phase : Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env)) :
    Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program)
      (CompactPullback.compactEnv program env) := by
  exact Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds.of_cross_attempt_eq
    relation ajtai
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (AssemblerInputs.piRlcInterface relation program)
    PilotPiCCSPiRLC.piRlcOffset (AssemblerInputs.piRlcOffset program)
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiRlcSampler relation program env phase.sampler)
    (compactPiRlcAttempt_eq relation program env) phase

private theorem sourceEnv_eq_compactEnv_belowPiDec
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    ∀ index, index < PiDECInputs.phaseOffset →
      sourceEnv program env index = CompactPullback.compactEnv program env index := by
  intro index bounded
  symm
  apply CompactPullback.compactEnv_source
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
    PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
    PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
    PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild]
    at bounded ⊢
  omega

private theorem compactPiDecParent_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
      (PiDECInputs.interface logicalWidth publicFits)
      PilotPiCCSPiRLCPiDEC.piDecOffset (sourceEnv program env)).parent =
    (Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program)
      (CompactPullback.compactEnv program env)).parent := by
  calc
    _ = Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env) := by rfl
    _ = Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program)
        (CompactPullback.compactEnv program env) :=
      compactPiRlcOutput_eq relation program env
    _ = _ := (AssemblerInputs.piDecParent_eval_eq_piRlcOutput relation program
      (CompactPullback.compactEnv program env)).symm

theorem compactPiDecOutput_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.PiDEC.v1_1.Semantics.output relation
        (PiDECInputs.interface logicalWidth publicFits)
        PilotPiCCSPiRLCPiDEC.piDecOffset (sourceEnv program env) =
      Lifecycle.PiDEC.v1_1.Semantics.output relation
        (AssemblerInputs.piDecInterface relation program)
        (AssemblerInputs.piDecOffset program)
        (CompactPullback.compactEnv program env) := by
  apply Lifecycle.PiDEC.v1_1.Semantics.output_eq_of_cross_components
  · exact congrArg (fun value => value.point)
      (compactPiDecParent_eq relation program env)
  · intro child row lane
    change (PiDECInputs.childCommitment child row lane).eval
        (sourceEnv program env) =
      (PiDECInputs.childCommitment child row lane).eval
        (CompactPullback.compactEnv program env)
    exact Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      (sourceEnv program env) (CompactPullback.compactEnv program env)
      ((PiDECInputs.inputsBelow relation).messageCommitment child row lane)
      (sourceEnv_eq_compactEnv_belowPiDec program env)
  · intro child coordinate
    change (PiDECInputs.childPublicInput child
        (Fin.cast
          (Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq
            logicalWidth publicFits) coordinate)).eval (sourceEnv program env) =
      (PiDECInputs.childPublicInput child
        (Fin.cast
          (Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq
            logicalWidth publicFits) coordinate)).eval
        (CompactPullback.compactEnv program env)
    exact Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      (sourceEnv program env) (CompactPullback.compactEnv program env)
      ((PiDECInputs.inputsBelow relation).digit child coordinate)
      (sourceEnv_eq_compactEnv_belowPiDec program env)
  · intro child
    apply compactEvaluationFamily_ext
    · funext coefficient
      exact KExpr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
        (sourceEnv program env) (CompactPullback.compactEnv program env)
        ((PiDECInputs.inputsBelow relation).messageEval_K child coefficient)
        (sourceEnv_eq_compactEnv_belowPiDec program env)
    · funext matrix coefficient
      exact KExpr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
        (sourceEnv program env) (CompactPullback.compactEnv program env)
        ((PiDECInputs.inputsBelow relation).messageEval_A child matrix
          coefficient)
        (sourceEnv_eq_compactEnv_belowPiDec program env)

theorem compactPiDecPhaseHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (phase : Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiDECInputs.interface logicalWidth publicFits)
      PilotPiCCSPiRLCPiDEC.piDecOffset (sourceEnv program env)) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piDecInterface relation program)
      (AssemblerInputs.piDecOffset program)
      (CompactPullback.compactEnv program env) := by
  exact Lifecycle.PiDEC.v1_1.Semantics.phaseHolds_of_cross_parent_output_eq
    relation ajtai (PiDECInputs.interface logicalWidth publicFits)
    (AssemblerInputs.piDecInterface relation program)
    PilotPiCCSPiRLCPiDEC.piDecOffset (AssemblerInputs.piDecOffset program)
    (sourceEnv program env) (CompactPullback.compactEnv program env)
    (compactPiDecParent_eq relation program env)
    (compactPiDecOutput_eq relation program env) phase

structure ChildPhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) : Prop where
  pilot : Pilot.PhysicalHolds PilotProduction.interface
    PilotProduction.witnessOffset (sourceEnv program env)
  piCcs : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
    (PilotPiCCS.interface (publicFits := publicFits))
    PilotPiCCS.piCcsOffset (sourceEnv program env)
  piRlc : NightstreamFPrime.Layout.PiRLC.v1_1.PhysicalHolds relation
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env)
  piDec : NightstreamFPrime.Layout.PiDEC.v1_1.PhysicalHolds relation
    (PiDECInputs.interface logicalWidth publicFits)
    PilotPiCCSPiRLCPiDEC.piDecOffset (sourceEnv program env)
  running : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits
    (sourceEnv program env)
  application : ApplicationPhysicalHolds program env
  nextPreimage : NextPreimagePhysicalHolds env

theorem physical_implies_children
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (physical : PhysicalHolds relation program env) :
    ChildPhysicalHolds relation program env := by
  rcases (physicalHolds_iff relation program env).mp physical with
    ⟨⟨shiftedPrefix, application⟩, nextPreimage⟩
  have spartanPrefix : R1CS.RowsHold (Lowering.basePullback program env)
      (Spartan.remappedRows relation) :=
    (Lowering.shiftRows_hold program env (Spartan.remappedRows relation)).mp
      shiftedPrefix
  have sourcePrefix :
      PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation
        (sourceEnv program env) := by
    exact (Spartan.remappedRows_hold relation
      (Lowering.basePullback program env)).mp spartanPrefix
  rcases (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation
      (sourceEnv program env)).mp sourcePrefix with ⟨throughPiDec, running⟩
  rcases (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation
      (sourceEnv program env)).mp throughPiDec with ⟨throughPiRlc, piDec⟩
  rcases (PilotPiCCSPiRLC.physicalHolds_iff relation
      (sourceEnv program env)).mp throughPiRlc with ⟨throughPiCcs, piRlc⟩
  rcases (PilotPiCCS.physicalHolds_iff relation
      (sourceEnv program env)).mp throughPiCcs with ⟨pilot, piCcs⟩
  exact ⟨pilot, piCcs, piRlc, piDec, running, application, nextPreimage⟩

structure ChildSpecs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) : Prop where
  pilot : Lifecycle.Pilot.SpecHolds PilotProduction.interface
    PilotProduction.witnessOffset (sourceEnv program env)
  piCcs : Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
    (PilotPiCCS.interface (publicFits := publicFits))
    PilotPiCCS.piCcsOffset (sourceEnv program env) template
  piRlc : Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PilotPiCCSPiRLC.piRlcOffset (sourceEnv program env)
  piDec : Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
    (PiDECInputs.interface logicalWidth publicFits)
    PilotPiCCSPiRLCPiDEC.piDecOffset (sourceEnv program env)
  running : Lifecycle.Stage1.RunningTransition.SpecHolds
    (RunningTransitionInputs.interface logicalWidth publicFits)
    RunningTransitionInputs.phaseOffset (sourceEnv program env)
  application : Lifecycle.Stage1.Application.Holds program.step
    (ApplicationInputs.interface program) (ApplicationInputs.localStart program)
    env
  nextPreimage : Lifecycle.Stage1.NextPreimage.SpecHolds
    NextPreimageInputs.spartanInterface Lowering.nextPreimagePrivateStart env

theorem physical_implies_childSpecs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (physical : PhysicalHolds relation program env) :
    ChildSpecs relation ajtai program template env := by
  have children := physical_implies_children relation program env physical
  let source := sourceEnv program env
  have piCcsAssumptions : Lifecycle.PiCCS.v1_1.Formal.Assumptions relation
      (PilotPiCCS.interface (publicFits := publicFits))
      PilotPiCCS.piCcsOffset source :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
      (PilotPiCCS.interface (publicFits := publicFits)) PilotPiCCS.piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits) source
  have piRlcAssumptions := PiRLCInputBounds.assumptions relation source
  have piDecAssumptions := PiDECInputs.assumptions relation source
  have applicationAssumptions := program.assumptions
    (ApplicationInputs.interface program) (ApplicationInputs.localStart program)
    env (ApplicationInputs.externalBelow program)
  have applicationFlat : holdsFlat env (Lowering.applicationOperations program) := by
    exact R1CS.LoweringPlan.sound (Lowering.applicationPlan program) env
      children.application
  have nextPreimageFlat : holdsFlat env Lowering.nextPreimageOperations := by
    exact R1CS.LoweringPlan.sound Lowering.nextPreimagePlan env
      children.nextPreimage
  refine {
    pilot := Pilot.physical_implies_spec PilotProduction.interface
      PilotProduction.witnessOffset source (PilotProduction.assumptions source)
      children.pilot
    piCcs := NightstreamFPrime.Layout.PiCCS.v1_1.physical_implies_phaseHolds
      relation ajtai (PilotPiCCS.interface (publicFits := publicFits)) template
      PilotPiCCS.piCcsOffset source piCcsAssumptions children.piCcs
    piRlc := NightstreamFPrime.Layout.PiRLC.v1_1.physical_implies_phaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PilotPiCCSPiRLC.piRlcOffset source piRlcAssumptions children.piRlc
    piDec := NightstreamFPrime.Layout.PiDEC.v1_1.physical_implies_phaseHolds
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PilotPiCCSPiRLCPiDEC.piDecOffset source piDecAssumptions children.piDec
    running := RunningTransitionLayout.physical_implies_specHolds relation source
      children.running
    application := program.soundness (ApplicationInputs.interface program)
      (ApplicationInputs.localStart program) env applicationAssumptions
      (holdsFlat_implies_holds env _ applicationFlat)
    nextPreimage := Lifecycle.Stage1.NextPreimage.soundness
      NextPreimageInputs.spartanInterface env Lowering.nextPreimagePrivateStart
      (holdsFlat_implies_holds env _ nextPreimageFlat) }

private theorem compactApplicationInput_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.Application.inputState
        (AssemblerInputs.applicationInterface program)
        (AssemblerInputs.applicationOffset program)
        (CompactPullback.compactEnv program env) =
      Lifecycle.Stage1.Application.inputState
        (ApplicationInputs.interface program)
        (ApplicationInputs.localStart program) env := by
  unfold Lifecycle.Stage1.Application.inputState
  apply congrArg List.ofFn
  funext index
  simp [AssemblerInputs.applicationInterface, ApplicationInputs.interface,
    CompactPullback.compactEnv_applicationInput]

private theorem compactApplicationWitness_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.Application.witnessValue
        (AssemblerInputs.applicationInterface program)
        (AssemblerInputs.applicationOffset program)
        (CompactPullback.compactEnv program env) =
      Lifecycle.Stage1.Application.witnessValue
        (ApplicationInputs.interface program)
        (ApplicationInputs.localStart program) env := by
  unfold Lifecycle.Stage1.Application.witnessValue
  apply congrArg List.ofFn
  funext index
  simp [AssemblerInputs.applicationInterface, ApplicationInputs.interface,
    CompactPullback.compactEnv_applicationWitness]

private theorem compactApplicationOutput_eq
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.Application.outputState
        (AssemblerInputs.applicationInterface program)
        (AssemblerInputs.applicationOffset program)
        (CompactPullback.compactEnv program env) =
      Lifecycle.Stage1.Application.outputState
        (ApplicationInputs.interface program)
        (ApplicationInputs.localStart program) env := by
  unfold Lifecycle.Stage1.Application.outputState
  apply congrArg List.ofFn
  funext index
  simp [AssemblerInputs.applicationInterface, ApplicationInputs.interface,
    CompactPullback.compactEnv_applicationOutput]

/-- The selected physical application rows imply the exact application field
of the compact seven-child logical circuit. -/
theorem physical_implies_compactApplication
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (physical : PhysicalHolds relation program env) :
    (program.circuit (AssemblerInputs.applicationInterface program)).spec
      (AssemblerInputs.applicationOffset program)
      (CompactPullback.compactEnv program env) := by
  have children := physical_implies_childSpecs relation ajtai program template
    env physical
  apply (program.spec_iff (AssemblerInputs.applicationInterface program)
    (AssemblerInputs.applicationOffset program)
    (CompactPullback.compactEnv program env)).mpr
  have physicalApplication := children.application
  unfold Lifecycle.Stage1.Application.Holds at physicalApplication ⊢
  rw [compactApplicationInput_eq, compactApplicationWitness_eq,
    compactApplicationOutput_eq]
  exact physicalApplication

end NightstreamFPrime.Layout.Stage1.Preservation
