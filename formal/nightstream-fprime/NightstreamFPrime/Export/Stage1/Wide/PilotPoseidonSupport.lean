import NightstreamFPrime.Export.Stage1.Wide.ReadSupport
import NightstreamFPrime.Export.Stage1.Wide.PoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem hash_block (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (upper : start + retained.coordinateCount ≤ RetainedLayout.hashEnd program) :
    Form program (retained.form start fits slot) := by
  apply block
  intro column _ high
  exact Or.inl (lt_of_lt_of_le high upper)

theorem prior_sbox (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin (PiRLCRetainedGeometry.priorPoseidonBlock program).slotCount) :
    Form program ((PiRLCRetainedGeometry.priorPoseidonBlock program).form
      (PiRLCRetainedGeometry.priorPoseidonStart program)
      (PiRLCRetainedGeometry.priorPoseidonFits (PiRLCPoseidonGeometry.prefixGeometry geometry)) slot) := by
  apply hash_block
  have endpoint := LaterPoseidonRetainedBlocks.samplerStart_eq program
  unfold PiRLCRetainedGeometry.laterPoseidonStart PiRLCRetainedGeometry.outputPoseidonStart at endpoint
  change _ ≤ LaterPoseidonRetainedBlocks.samplerStart program
  omega

theorem output_sbox (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin (PiRLCRetainedGeometry.outputPoseidonBlock program).slotCount) :
    Form program ((PiRLCRetainedGeometry.outputPoseidonBlock program).form
      (PiRLCRetainedGeometry.outputPoseidonStart program)
      (PiRLCRetainedGeometry.outputPoseidonFits (PiRLCPoseidonGeometry.prefixGeometry geometry)) slot) := by
  apply hash_block
  have endpoint := LaterPoseidonRetainedBlocks.samplerStart_eq program
  unfold PiRLCRetainedGeometry.laterPoseidonStart at endpoint
  change _ ≤ LaterPoseidonRetainedBlocks.samplerStart program
  omega

theorem prior_word (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin (PiRLCPoseidonGeometry.priorInputBlock program).slotCount) :
    Form program ((PiRLCPoseidonGeometry.priorInputBlock program).form
      (PiRLCPoseidonGeometry.priorInputStart program) (PiRLCPoseidonGeometry.priorInputFits geometry) slot) := by
  let within : PiRLCPoseidonGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
    ⟨by rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq, (RetainedLayout.boundaries program).2.2.1]; decide⟩
  exact shared_block program _ _ _ _ (Nat.le_refl _) (PiRLCPoseidonGeometry.priorInputFits within)

theorem output_word (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (slot : Fin (PiRLCPoseidonGeometry.outputInputBlock program).slotCount) :
    Form program ((PiRLCPoseidonGeometry.outputInputBlock program).form
      (PiRLCPoseidonGeometry.outputInputStart program) (PiRLCPoseidonGeometry.outputInputFits geometry) slot) := by
  let within : PiRLCPoseidonGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
    ⟨by rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq, (RetainedLayout.boundaries program).2.2.1]; decide⟩
  apply shared_block program _ _ _ _ _ (PiRLCPoseidonGeometry.outputInputFits within)
  change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + _
  omega

theorem retained_output {sourceWidth columns count : Nat} {predicate : Fin columns → Prop}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth count) (start : Nat) (fits)
    (supported : ∀ slot, Supported predicate (schedule.block.form start fits slot))
    (invocation : Fin count) (lane : Fin 8) :
    Supported predicate (PoseidonRetainedFamily.outputState schedule start fits invocation lane) :=
  external _ (fun _ => supported _) lane

theorem pilot_previous {sourceWidth columns count : Nat} {predicate : Fin columns → Prop}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth count) (start : Nat) (fits)
    (supported : ∀ slot, Supported predicate (schedule.block.form start fits slot))
    (invocation : Fin count) (lane : Fin 8) :
    Supported predicate (PilotPoseidonPlan.previousOutput schedule start fits invocation lane) := by
  unfold PilotPoseidonPlan.previousOutput
  split
  · exact empty _
  · exact retained_output schedule start fits supported _ lane

theorem pilot_prior_input (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (invocation : Fin PilotPoseidonPlan.invocationCount) (lane : Fin 8) :
    Form program (PilotPoseidonPlan.priorInputState geometry invocation lane) := by
  have previous := pilot_previous (PilotPoseidonPlan.priorSchedule program) _ _
    (prior_sbox program geometry) invocation lane
  unfold PilotPoseidonPlan.priorInputState
  dsimp only
  split_ifs
  · exact add previous (prior_word program geometry _)
  · exact previous
  · exact previous
  · exact add previous (singleton _ _ (one program _ rfl))
  · exact previous

theorem pilot_output_input (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (invocation : Fin PilotPoseidonPlan.invocationCount) (lane : Fin 8) :
    Form program (PilotPoseidonPlan.outputInputState geometry invocation lane) := by
  have previous := pilot_previous (PilotPoseidonPlan.outputSchedule program) _ _
    (output_sbox program geometry) invocation lane
  unfold PilotPoseidonPlan.outputInputState
  dsimp only
  split_ifs
  · exact add previous (output_word program geometry _)
  · exact previous
  · exact previous
  · exact add previous (singleton _ _ (one program _ rfl))
  · exact previous

theorem pilot_poseidon (program : Program)
    (geometry : PiRLCPoseidonGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (PilotPoseidonPlan.plan geometry) := by
  apply append
  · apply poseidon_family
    · exact one program _ rfl
    · exact pilot_prior_input program geometry
    · intro invocation slot; exact prior_sbox program geometry _
  · apply poseidon_family
    · exact one program _ rfl
    · exact pilot_output_input program geometry
    · intro invocation slot; exact output_sbox program geometry _

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
