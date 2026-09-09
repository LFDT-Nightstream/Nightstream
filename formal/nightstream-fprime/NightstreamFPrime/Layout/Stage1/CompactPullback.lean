import NightstreamFPrime.Layout.Stage1.Lowering

/-!
Owns the explicit pullback from the compact eight-child logical assembler to
the canonical physical Stage 1 layout. Each compact child-local interval maps
to the already validated phase-local interval. Existing source columns use the
proved Spartan permutation and final suffix shift. Application witness and
local columns map to their final zero-copy intervals.
-/

namespace NightstreamFPrime.Layout.Stage1.CompactPullback

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle

def sourceEnv (program : Lifecycle.Stage1.Application.Program)
    (env : Env) : Env :=
  Spartan.pullback (Lowering.basePullback program env)

/-- The one compact-to-physical environment map. Branch endpoints are the
proved adjacent logical child offsets. -/
def compactEnv (program : Lifecycle.Stage1.Application.Program)
    (env : Env) : Env := fun column =>
  if column < AssemblerInputs.rootOffset program then
    if column < Spartan.SourceColumnCount then
      sourceEnv program env column
    else
      env (ApplicationInputs.witnessStart +
        (column - Spartan.SourceColumnCount))
  else if column < AssemblerInputs.outputHashOffset program then
    sourceEnv program env
      (PilotProduction.witnessOffset +
        (column - AssemblerInputs.priorOffset program))
  else if column < AssemblerInputs.piCcsOffset program then
    sourceEnv program env
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
          PilotProduction.witnessOffset +
        (column - AssemblerInputs.outputHashOffset program))
  else if column < AssemblerInputs.piRlcOffset program then
    sourceEnv program env
      (PilotPiCCS.piCcsOffset +
        (column - AssemblerInputs.piCcsOffset program))
  else if column < AssemblerInputs.piDecOffset program then
    sourceEnv program env
      (PilotPiCCSPiRLC.piRlcOffset +
        (column - AssemblerInputs.piRlcOffset program))
  else if column < AssemblerInputs.runningOffset program then
    sourceEnv program env
      (PilotPiCCSPiRLCPiDEC.piDecOffset +
        (column - AssemblerInputs.piDecOffset program))
  else if column < AssemblerInputs.applicationOffset program then
    sourceEnv program env
      (RunningTransitionInputs.phaseOffset +
        (column - AssemblerInputs.runningOffset program))
  else
    env (ApplicationInputs.localStart program +
      (column - AssemblerInputs.applicationOffset program))

@[simp] theorem compactEnv_source
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (column : Nat) (bound : column < Spartan.SourceColumnCount) :
    compactEnv program env column = sourceEnv program env column := by
  have rootBound : column < AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.rootOffset AssemblerInputs.applicationLocalStart
      AssemblerInputs.applicationWitnessStart
    omega
  simp [compactEnv, rootBound, bound]

@[simp] theorem sourceEnv_applicationInput
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    sourceEnv program env (ApplicationInputs.inputSourceColumn index) =
      env (ApplicationInputs.inputColumn index) := by
  have privateBound : ApplicationInputs.inputColumn index <
      Spartan.constantColumn := by
    rw [ApplicationInputs.inputColumn_value]
    have indexBound := index.isLt
    norm_num [Lifecycle.Stage1.Application.stateWordCount,
      ApplicationInputs.currentWordStart, Spartan.constantColumn] at indexBound ⊢
    omega
  unfold sourceEnv Spartan.pullback Lowering.basePullback
  change env (Lowering.shiftColumn program (ApplicationInputs.inputColumn index)) =
    env (ApplicationInputs.inputColumn index)
  rw [Lowering.shiftColumn_private program _ privateBound]

@[simp] theorem sourceEnv_applicationOutput
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    sourceEnv program env (ApplicationInputs.outputSourceColumn index) =
      env (ApplicationInputs.outputColumn index) := by
  have privateBound : ApplicationInputs.outputColumn index <
      Spartan.constantColumn := by
    rw [ApplicationInputs.outputColumn_value]
    have indexBound := index.isLt
    norm_num [Lifecycle.Stage1.Application.stateWordCount,
      Spartan.constantColumn] at indexBound ⊢
    omega
  unfold sourceEnv Spartan.pullback Lowering.basePullback
  change env (Lowering.shiftColumn program (ApplicationInputs.outputColumn index)) =
    env (ApplicationInputs.outputColumn index)
  rw [Lowering.shiftColumn_private program _ privateBound]

@[simp] theorem compactEnv_applicationInput
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    compactEnv program env (ApplicationInputs.inputSourceColumn index) =
      env (ApplicationInputs.inputColumn index) := by
  rw [compactEnv_source program env _ (by
    have indexBound := index.isLt
    norm_num [ApplicationInputs.inputSourceColumn,
      PilotProduction.priorPreimageStart, ApplicationInputs.currentWordStart,
      Lifecycle.Stage1.Application.stateWordCount, Spartan.SourceColumnCount]
      at indexBound ⊢
    omega)]
  exact sourceEnv_applicationInput program env index

@[simp] theorem compactEnv_applicationOutput
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    compactEnv program env (ApplicationInputs.outputSourceColumn index) =
      env (ApplicationInputs.outputColumn index) := by
  rw [compactEnv_source program env _ (by
    have indexBound := index.isLt
    change 49698 + index.val < 29336724
    norm_num [Lifecycle.Stage1.Application.stateWordCount] at indexBound
    omega)]
  exact sourceEnv_applicationOutput program env index

@[simp] theorem compactEnv_applicationWitness
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Fin program.witnessWordCount) :
    compactEnv program env (AssemblerInputs.applicationWitnessColumn index) =
      env (ApplicationInputs.witnessColumn index) := by
  have indexBound := index.isLt
  have belowRoot : AssemblerInputs.applicationWitnessColumn index <
      AssemblerInputs.rootOffset program := by
    change Spartan.SourceColumnCount + index.val <
      Spartan.SourceColumnCount + program.witnessWordCount
    omega
  have notSource : ¬ AssemblerInputs.applicationWitnessColumn index <
      Spartan.SourceColumnCount := by
    change ¬ Spartan.SourceColumnCount + index.val < Spartan.SourceColumnCount
    omega
  rw [compactEnv, if_pos belowRoot, if_neg notSource]
  simp [AssemblerInputs.applicationWitnessColumn,
    AssemblerInputs.applicationWitnessStart, ApplicationInputs.witnessColumn]

@[simp] theorem compactEnv_priorLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 7311464) :
    compactEnv program env (AssemblerInputs.priorOffset program + index) =
      sourceEnv program env (PilotProduction.witnessOffset + index) := by
  have notRoot : ¬ AssemblerInputs.priorOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.priorOffset
    omega
  have beforeOutput : AssemblerInputs.priorOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.outputHashOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_pos beforeOutput]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_outputHashLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 7311200) :
    compactEnv program env (AssemblerInputs.outputHashOffset program + index) =
      sourceEnv program env
        (Lifecycle.Pilot.outputOffset PilotProduction.interface
          PilotProduction.witnessOffset + index) := by
  have notRoot : ¬ AssemblerInputs.outputHashOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.outputHashOffset program + index <
      AssemblerInputs.outputHashOffset program := by omega
  have beforePiCcs : AssemblerInputs.outputHashOffset program + index <
      AssemblerInputs.piCcsOffset program := by
    unfold AssemblerInputs.piCcsOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_pos beforePiCcs]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_piCcsLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 4581414) :
    compactEnv program env (AssemblerInputs.piCcsOffset program + index) =
      sourceEnv program env (PilotPiCCS.piCcsOffset + index) := by
  have notRoot : ¬ AssemblerInputs.piCcsOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
      AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.piCcsOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.piCcsOffset
    omega
  have notPiCcs : ¬ AssemblerInputs.piCcsOffset program + index <
      AssemblerInputs.piCcsOffset program := by omega
  have beforePiRlc : AssemblerInputs.piCcsOffset program + index <
      AssemblerInputs.piRlcOffset program := by
    unfold AssemblerInputs.piRlcOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_neg notPiCcs,
    if_pos beforePiRlc]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_piRlcLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 315894) :
    compactEnv program env (AssemblerInputs.piRlcOffset program + index) =
      sourceEnv program env (PilotPiCCSPiRLC.piRlcOffset + index) := by
  have notRoot : ¬ AssemblerInputs.piRlcOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.piRlcOffset AssemblerInputs.piCcsOffset
      AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.piRlcOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.piRlcOffset AssemblerInputs.piCcsOffset
    omega
  have notPiCcs : ¬ AssemblerInputs.piRlcOffset program + index <
      AssemblerInputs.piCcsOffset program := by
    unfold AssemblerInputs.piRlcOffset
    omega
  have notPiRlc : ¬ AssemblerInputs.piRlcOffset program + index <
      AssemblerInputs.piRlcOffset program := by omega
  have beforePiDec : AssemblerInputs.piRlcOffset program + index <
      AssemblerInputs.piDecOffset program := by
    unfold AssemblerInputs.piDecOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_neg notPiCcs,
    if_neg notPiRlc, if_pos beforePiDec]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_piDecLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 270) :
    compactEnv program env (AssemblerInputs.piDecOffset program + index) =
      sourceEnv program env (PilotPiCCSPiRLCPiDEC.piDecOffset + index) := by
  have notRoot : ¬ AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
      AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
      AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
      AssemblerInputs.piCcsOffset
    omega
  have notPiCcs : ¬ AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.piCcsOffset program := by
    unfold AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
    omega
  have notPiRlc : ¬ AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.piRlcOffset program := by
    unfold AssemblerInputs.piDecOffset
    omega
  have notPiDec : ¬ AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.piDecOffset program := by omega
  have beforeRunning : AssemblerInputs.piDecOffset program + index <
      AssemblerInputs.runningOffset program := by
    unfold AssemblerInputs.runningOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_neg notPiCcs,
    if_neg notPiRlc, if_neg notPiDec, if_pos beforeRunning]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_runningLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) (bound : index < 1) :
    compactEnv program env (AssemblerInputs.runningOffset program + index) =
      sourceEnv program env (RunningTransitionInputs.phaseOffset + index) := by
  have notRoot : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.runningOffset AssemblerInputs.piDecOffset
      AssemblerInputs.piRlcOffset AssemblerInputs.piCcsOffset
      AssemblerInputs.outputHashOffset AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.runningOffset AssemblerInputs.piDecOffset
      AssemblerInputs.piRlcOffset AssemblerInputs.piCcsOffset
    omega
  have notPiCcs : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.piCcsOffset program := by
    unfold AssemblerInputs.runningOffset AssemblerInputs.piDecOffset
      AssemblerInputs.piRlcOffset
    omega
  have notPiRlc : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.piRlcOffset program := by
    unfold AssemblerInputs.runningOffset AssemblerInputs.piDecOffset
    omega
  have notPiDec : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.piDecOffset program := by
    unfold AssemblerInputs.runningOffset
    omega
  have notRunning : ¬ AssemblerInputs.runningOffset program + index <
      AssemblerInputs.runningOffset program := by omega
  have beforeApplication : AssemblerInputs.runningOffset program + index <
      AssemblerInputs.applicationOffset program := by
    unfold AssemblerInputs.applicationOffset
    omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_neg notPiCcs,
    if_neg notPiRlc, if_neg notPiDec, if_neg notRunning,
    if_pos beforeApplication]
  apply congrArg (sourceEnv program env)
  omega

@[simp] theorem compactEnv_applicationLocal
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (index : Nat) :
    compactEnv program env (AssemblerInputs.applicationOffset program + index) =
      env (ApplicationInputs.localStart program + index) := by
  have notRoot : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.rootOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
      AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
      AssemblerInputs.piCcsOffset AssemblerInputs.outputHashOffset
      AssemblerInputs.priorOffset
    omega
  have notOutput : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.outputHashOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
      AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
      AssemblerInputs.piCcsOffset
    omega
  have notPiCcs : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.piCcsOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
      AssemblerInputs.piDecOffset AssemblerInputs.piRlcOffset
    omega
  have notPiRlc : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.piRlcOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
      AssemblerInputs.piDecOffset
    omega
  have notPiDec : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.piDecOffset program := by
    unfold AssemblerInputs.applicationOffset AssemblerInputs.runningOffset
    omega
  have notRunning : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.runningOffset program := by
    unfold AssemblerInputs.applicationOffset
    omega
  have notApplication : ¬ AssemblerInputs.applicationOffset program + index <
      AssemblerInputs.applicationOffset program := by omega
  unfold compactEnv
  rw [if_neg notRoot, if_neg notOutput, if_neg notPiCcs,
    if_neg notPiRlc, if_neg notPiDec, if_neg notRunning,
    if_neg notApplication]
  apply congrArg env
  omega

end NightstreamFPrime.Layout.Stage1.CompactPullback
