import NightstreamFPrime.Layout.OptionB.FinishAssembler
import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness

/-! Completeness of the finish circuit at the refresh step's source layout (CW-10.6), the converse
of `FinishAssembler.rows_implies_finish`. An active proof's input and an output that satisfy
`FinishStep.FinishHolds`, with the refresh proof that the input carries, have values on which
every row of `FinishStep.main` at `sharedEnd` holds and whose words carry the step (`complete`).
The values start from the refresh step's honest source values
(`RefreshAssemblerSource.honestEnv`). The prior hashes fill from the first prior slot hash's
offset (`RefreshStepCompleteness.completePriorHashesFrom`), and the fold phase and the slot select
fill from today's Π_DEC offset (`RefreshAssemblerCompleteness.completeFoldFrom`). The finish
circuit places no rows for the application and the next hashes, so their words keep the source
values; the prior hashes' rows read only words below the next slot hashes' offset, which is at or
below today's Π_DEC offset. -/

namespace NightstreamFPrime.Layout.OptionB.FinishAssemblerCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout
open NightstreamFPrime.Layout.OptionB.RefreshAssembler
open NightstreamFPrime.Layout.OptionB.RefreshAssemblerSource
open NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The next slot hashes start at or below today's Π_DEC offset. The layout and the offset are
variables here, so the kernel never evaluates a layout length. -/
private theorem nextSlot_le_dec (program : Stage1.Application.Program) (circuit : Fin 2)
    (layout : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
    (offset : Nat) :
    nextSlotZeroOffset program layout offset ≤ decOffset program circuit layout offset := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have oneLe : nextSlotZeroOffset program layout offset ≤
      nextSlotOneOffset program layout offset := le _ _
  have stateLe : nextSlotOneOffset program layout offset ≤
      nextStateOffset program layout offset := le _ _
  have decLe : nextStateOffset program layout offset ≤
      decOffset program circuit layout offset := le _ _
  omega

/-- An empty prefix ends where it starts. The start is a variable here, so the kernel never
evaluates a layout offset. -/
private theorem empty_end (env : Env) (start : Nat) :
    start + localLength (Sequence.empty env start).operations = start :=
  rfl

/-- Completeness of the finish circuit at the source layout, the converse of
`FinishAssembler.rows_implies_finish`: an active proof's input and an output that satisfy
`FinishStep.FinishHolds` at the production cap, with the refresh proof that the input carries,
have values on which every finish row holds and whose words carry the step. The other premises
are the word bounds and lengths that `StepRepresents` states. -/
theorem complete (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (finish : FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys
      circuit input output)
    (active : input.iteration ≠ 0) (proofEq : input.nifsProof = .refresh proof)
    (pcNext : output.pcNext = circuit)
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (xLength : output.x.length = 4) :
    ∃ env, holdsFlat env (Circuit.ops (FinishStep.main program relation circuit
        (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)) ∧
      RefreshStepSoundness.StepRepresents program (interface base program.witnessWordCount) ajtai
        circuitKeys circuit input output proof env := by
  obtain ⟨priorPcValid, facts⟩ := finish
  dsimp only at facts
  obtain ⟨freshPublic, folds, depthZero, others⟩ := facts
  have selectedPc : input.priorPc = (Paper.selectedIndex priorPcValid).val + 1 := by
    have range := priorPcValid
    unfold Paper.InRange at range
    simp only [Paper.selectedIndex]
    omega
  generalize Paper.selectedIndex priorPcValid = selected at selectedPc folds depthZero others
  rw [proofEq] at folds
  have select : SelectSpec input output selected (output.runningNext selected).1 0 :=
    ⟨fun zero => absurd zero active, fun _ => ⟨selectedPc, rfl, depthZero, others⟩⟩
  obtain ⟨source, sourceEq⟩ : ∃ env : Env, env = honestEnv ajtai circuitKeys input output proof
      selected (output.runningNext selected) base program.witnessWordCount := ⟨_, rfl⟩
  have sourceAgree : ∀ column, column < sharedEnd base program.witnessWordCount →
      source column = honestEnv ajtai circuitKeys input output proof selected
        (output.runningNext selected) base program.witnessWordCount column :=
    fun column _ => congrFun sourceEq column
  have rep0 := stepRepresents ajtai circuitKeys input output proof selected
    (output.runningNext selected) base source program circuit sourceAgree iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength xLength
    pcNext (fun _ => proofEq)
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  obtain ⟨hashes, hashesOps, hashesEnd, -⟩ := RefreshStepCompleteness.completePriorHashesFrom
    program (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount) (sharedEnd base program.witnessWordCount) relation
    (Sequence.empty source (priorSlotZeroOffset program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
    (empty_end _ _) below
    (priorSlot_spec ajtai circuitKeys input output proof selected _ program base source
      sourceAgree circuit rep0 _)
    (priorState_spec ajtai circuitKeys input output proof selected _ program base source
      sourceAgree circuit rep0 freshPublic _)
  have sharedLe : sharedEnd base program.witnessWordCount ≤ priorSlotZeroOffset program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) (sharedEnd base program.witnessWordCount) :=
    (Nat.le_add_right _ _ : _ ≤ Formal.nextOffset _ _)
  have hashesAgree : ∀ column, column < sharedEnd base program.witnessWordCount →
      hashes.current column = honestEnv ajtai circuitKeys input output proof selected
        (output.runningNext selected) base program.witnessWordCount column :=
    fun column bound => (hashes.agrees column (Or.inl (by omega))).trans
      (sourceAgree column bound)
  obtain ⟨env, keep, foldRows, rep⟩ := completeFoldFrom program relation base ajtai circuitKeys
    circuit input output proof selected (output.runningNext selected) depthZero
    (Sequence.empty hashes.current (decOffset program circuit
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) (sharedEnd base program.witnessWordCount)))
    (empty_end _ _) hashesAgree select folds iterationBound priorPcBound priorDepth nextDepth keyLength
    z0Length ziLength zNextLength witnessLength xLength pcNext (fun _ => proofEq)
  have gap := nextSlot_le_dec program circuit (interface (logicalWidth := logicalWidth)
    (publicFits := publicFits) base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount)
  have hashRows : holdsFlat env (FinishStep.hashOps program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) (sharedEnd base program.witnessWordCount)) := by
    have rows := constraintsHold_of_agree_below hashes.current env _ _ hashes.scope
      (fun index bound => keep index (by omega)) hashes.rows
    rw [hashesOps] at rows
    unfold holdsFlat FinishStep.hashOps
    simpa only [Sequence.empty, List.nil_append] using rows
  refine ⟨env, ?_, rep⟩
  change holdsFlat env (FinishStep.opsAt program relation circuit
    (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount) (sharedEnd base program.witnessWordCount))
  unfold FinishStep.opsAt
  simp only [Sequence.empty, List.nil_append] at foldRows
  unfold holdsFlat at foldRows hashRows ⊢
  simp only [List.append_assoc] at foldRows ⊢
  rw [flatConstraints_append]
  exact (constraintsHold_append _ _ _).mpr ⟨hashRows, foldRows⟩

end NightstreamFPrime.Layout.OptionB.FinishAssemblerCompleteness
