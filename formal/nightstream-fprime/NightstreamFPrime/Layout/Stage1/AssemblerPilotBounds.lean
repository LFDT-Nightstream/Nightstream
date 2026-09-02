import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Layout.Stage1.AssemblerInputs

/-!
Owns the compact source bounds for the two pilot children in the Stage 1
logical assembler. The proofs use the direct source-variable definitions and
add no row, semantic predicate, or physical placement.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerPilotBounds

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle

theorem rootOffset_eq
    (program : Lifecycle.Stage1.Application.Program) :
    AssemblerInputs.rootOffset program =
      29336724 + program.witnessWordCount := by
  unfold AssemblerInputs.rootOffset AssemblerInputs.applicationLocalStart
    AssemblerInputs.applicationWitnessStart
  rw [Spartan.sourceColumnCount_eq]

private theorem pilotSourceEnd_le_root
    (program : Lifecycle.Stage1.Application.Program) :
    PilotProduction.outputDigestStart + PilotProduction.digestWords ≤
      AssemblerInputs.rootOffset program := by
  rw [rootOffset_eq]
  have nonnegative : 0 ≤ program.witnessWordCount := Nat.zero_le _
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq,
    PilotProduction.digestWords,
    Lifecycle.PriorStateHash.publicWidth,
    Lifecycle.PaperAlgebra.publicRingColumns, Spec.ringDegree] at nonnegative ⊢
  omega

/-- The compact prior-state child reads only source expressions before the
compact root. -/
def priorAssumptions
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    PriorStateHash.Assumptions PilotProduction.priorInterface
      (AssemblerInputs.priorOffset program) env := by
  refine ⟨?_, ?_⟩
  · intro expression member
    rw [PilotProduction.priorInterface_preimage_apply] at member
    unfold PilotProduction.priorPreimage at member
    apply PilotProduction.variableExprs_below
      PilotProduction.priorPreimageStart PilotProduction.stateHashWords
      (AssemblerInputs.priorOffset program) ?_ expression member
    unfold AssemblerInputs.priorOffset
    exact Nat.le_trans (by
      unfold PilotProduction.outputDigestStart
        PilotProduction.outputPreimageStart
        PilotProduction.priorPublicInputStart
      omega) (pilotSourceEnd_le_root program)
  · intro column
    rw [PilotProduction.priorInterface_publicInput_apply]
    unfold PilotProduction.priorPublicInput AssemblerInputs.priorOffset
    simp only [Expr.VarsBelow]
    have columnBound := column.isLt
    rw [rootOffset_eq]
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, PriorStateHash.publicWidth,
      Spec.ringDegree, Lifecycle.PaperAlgebra.publicRingColumns]
      at columnBound ⊢
    omega

/-- The output-hash preimage remains in the caller-owned source prefix. -/
theorem outputPreimageBelowRoot
    (program : Lifecycle.Stage1.Application.Program) :
    ∀ expression ∈ PilotProduction.outputInterface.preimage
        (AssemblerInputs.outputHashOffset program),
      expression.VarsBelow (AssemblerInputs.rootOffset program) := by
  intro expression member
  rw [PilotProduction.outputInterface_preimage_apply] at member
  unfold PilotProduction.outputPreimage at member
  apply PilotProduction.variableExprs_below
    PilotProduction.outputPreimageStart PilotProduction.stateHashWords
    (AssemblerInputs.rootOffset program) ?_ expression member
  exact Nat.le_trans (by
    unfold PilotProduction.outputDigestStart
    omega) (pilotSourceEnd_le_root program)

/-- The expected output digest remains in the caller-owned source prefix. -/
theorem outputDigestBelowRoot
    (program : Lifecycle.Stage1.Application.Program) (lane : Fin 4) :
    (PilotProduction.outputInterface.digest
      (AssemblerInputs.outputHashOffset program) lane).VarsBelow
        (AssemblerInputs.rootOffset program) := by
  unfold PilotProduction.outputInterface PilotProduction.makeOutputInterface
    PilotProduction.outputDigest
  simp only [Expr.VarsBelow]
  have laneBound := lane.isLt
  rw [rootOffset_eq]
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq, PilotProduction.digestWords,
    PriorStateHash.publicWidth, Spec.ringDegree,
    Lifecycle.PaperAlgebra.publicRingColumns] at laneBound ⊢
  omega

/-- Exact compact-root support used to transport output-hash semantics after
the prior child completes. -/
def outputSupport
    (program : Lifecycle.Stage1.Application.Program) :
    (∀ expression ∈ PilotProduction.outputInterface.preimage
        (AssemblerInputs.outputHashOffset program),
      expression.VarsSatisfy
        (fun index => index < AssemblerInputs.rootOffset program)) ∧
      ∀ lane, (PilotProduction.outputInterface.digest
        (AssemblerInputs.outputHashOffset program) lane).VarsSatisfy
          (fun index => index < AssemblerInputs.rootOffset program) := by
  refine ⟨?_, ?_⟩
  · intro expression member
    exact (Expr.varsSatisfy_lt_iff_varsBelow expression _).2
      (outputPreimageBelowRoot program expression member)
  · intro lane
    exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).2
      (outputDigestBelowRoot program lane)

/-- The compact output-hash child has its exact causal assumptions at the
later logical allocation. -/
def outputAssumptions
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    OutputHash.Assumptions PilotProduction.outputInterface
      (AssemblerInputs.outputHashOffset program) env := by
  unfold OutputHash.Assumptions
  refine ⟨?_, ?_⟩
  · intro expression member
    rw [OutputHash.hashInterface_input] at member
    apply Expr.VarsBelow.mono expression
      (outputPreimageBelowRoot program expression member)
    unfold AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    omega
  · intro lane
    rw [OutputHash.hashInterface_expected]
    apply Expr.VarsBelow.mono _ (outputDigestBelowRoot program lane)
    unfold AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    omega

end NightstreamFPrime.Layout.Stage1.AssemblerPilotBounds
