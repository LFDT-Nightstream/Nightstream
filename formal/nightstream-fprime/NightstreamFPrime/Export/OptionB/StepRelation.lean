import NightstreamFPrime.Export.Generic.LogicalPlan
import NightstreamFPrime.Layout.OptionB.StepCircuits

/-! The option-B step relations (CW-5.3). Each step circuit is exported with its shared words at
`base = 270`, after the public input `x` (`LogicalPlan`). From a satisfying assignment of a
circuit's logical relation, the decoders read the step's input, output, key digests and fold proof
from the shared words; `lightRepresents` and `refreshRepresents` hold by construction. With
`LogicalPlan.rowsZero_implies_spec` this gives each circuit's case of the option-B step relation
(`light_stepHolds`, `refresh_stepHolds`), and the output digest that `x` encodes. The decoded key
digests are what the shared words carry; the state hash in `x` binds them to the setup's keys
(`State.stateHash_collision`), which the terminal uses. -/

namespace NightstreamFPrime.Export.OptionB.StepRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The public input occupies source columns 0 to 269; the shared words start after it. -/
def base : Nat := ringDegree * PaperAlgebra.publicRingColumns

theorem natWord_val (value : F) : natWord value.val = value := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

/-- The key digests that the shared words carry, with the setup's relations. -/
def decodedKeys (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (verifierKeys : Fin 2 → Fin 4 → Expr) (env : Env) :
    Fin 2 → CircuitKey logicalWidth publicFits :=
  fun key => ⟨List.ofFn fun lane => (verifierKeys key lane).eval env, (circuitKeys key).relation⟩

section Light

variable (program : Stage1.Application.Program)

abbrev lightInterface :=
  LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount

def lightProof (env : Env) : Spec.Folding.Light.Proof LightShape 8 :=
  ⟨fun roundIndex =>
      ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program).round
        roundIndex).semanticPolynomial env,
    ⟨fun source coefficient =>
        ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
          program).output.padCoordinate source coefficient).eval env,
      fun source matrix coefficient =>
        ((lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
          program).output.matrixCoordinate source matrix coefficient).eval env⟩⟩

def lightInput (env : Env) : StepInput logicalWidth publicFits :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { iteration := (interface.iteration.eval env).val
    z0 := List.ofFn fun lane => (interface.initialState lane).eval env
    zi := List.ofFn fun lane => (interface.currentState lane).eval env
    running := fun slot => (interface.prior slot).value env
    fresh := StatementAbsorption.evalFresh interface.fresh env
    priorPc := (interface.priorPc.eval env).val
    witness := List.ofFn fun index => (interface.witness index).eval env
    nifsProof := .light (lightProof (logicalWidth := logicalWidth) (publicFits := publicFits)
      program env) }

def lightOutput (circuit : Fin 2) (env : Env) : StepOutput logicalWidth publicFits :=
  let interface := lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { zNext := List.ofFn fun lane => (interface.nextState lane).eval env
    runningNext := fun slot => (interface.next slot).value env
    pcNext := circuit
    x := List.ofFn fun lane => (interface.outputDigest lane).eval env }

theorem lightRepresents
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2) (env : Env) :
    LightStepSoundness.StepRepresents program
      (lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program) ajtai
      (decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys env)
      circuit (lightInput program env) (lightOutput program circuit env)
      (lightProof (logicalWidth := logicalWidth) (publicFits := publicFits) program env) env where
  iteration := (natWord_val _).symm
  iterationBound := Fin.isLt _
  priorPc := (natWord_val _).symm
  priorPcBound := Fin.isLt _
  verifierKeys := fun _ => rfl
  initialState := rfl
  currentState := rfl
  nextState := rfl
  witness := rfl
  priorSlots := fun _ => SlotExpr.words_eval _ env
  priorDepth := fun _ => Fin.isLt _
  nextSlots := fun _ => SlotExpr.words_eval _ env
  nextDepth := fun _ => Fin.isLt _
  pcNext := rfl
  outputDigest := rfl
  fresh := rfl
  rounds := fun _ => rfl
  outputPad := fun _ _ => rfl
  outputMatrix := fun _ _ _ => rfl
  proof := fun _ => rfl

/-- The source column of output digest word `word`. -/
def lightDigestColumn (word : Fin 4) : Nat :=
  LightAssembler.digestsStart base program.witnessWordCount + 16 + word.val

/-- A satisfying assignment of the light circuit's logical relation whose public input encodes
`digest` gives the light case of the option-B step relation, with output hash `digest`. -/
theorem light_stepHolds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 0)
    (firstFresh : Nat)
    (fits : LogicalPlan.Fits (LogicalPlan.rows
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
      (lightDigestColumn program)) logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv assignment) = encHash digest)
    (rowsZero : (LogicalPlan.plan _ logicalWidth fits).RowsZero assignment) :
    let env := LogicalPlan.sourceEnv assignment
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (lightInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 0 (lightInput program env) (lightOutput program 0 env) ∧
      (lightOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program 0 env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  intro env
  obtain ⟨rows, words⟩ := LogicalPlan.rowsZero_implies_spec
    (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (lightDigestColumn program) fits assignment digest encoded rowsZero rfl
  refine ⟨LightAssembler.rows_implies_stepHolds program relation base ajtai _ step control 0 rfl
    stepEq validControl env rows _ _ _ (lightRepresents program ajtai circuitKeys 0 env), ?_⟩
  exact congrArg List.ofFn (funext words)

end Light

section Refresh

variable (program : Stage1.Application.Program)

abbrev refreshInterface :=
  RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount

def refreshProof (env : Env) :
    Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8 :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let commitments := fun child row lane => (interface.childCommitment child row lane).eval env
  let evaluations := fun child =>
    StatementAbsorption.evalEvaluation (interface.childEvaluation child) env
  ⟨⟨commitments, evaluations⟩,
    ⟨fun roundIndex => (interface.round roundIndex).semanticPolynomial env,
      ⟨fun source coefficient => (interface.output.padCoordinate source coefficient).eval env,
        fun source matrix coefficient =>
          (interface.output.matrixCoordinate source matrix coefficient).eval env⟩,
      commitments, evaluations⟩⟩

def refreshInput (env : Env) : StepInput logicalWidth publicFits :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { iteration := (interface.iteration.eval env).val
    z0 := List.ofFn fun lane => (interface.initialState lane).eval env
    zi := List.ofFn fun lane => (interface.currentState lane).eval env
    running := fun slot => (interface.prior slot).value env
    fresh := StatementAbsorption.evalFresh interface.fresh env
    priorPc := (interface.priorPc.eval env).val
    witness := List.ofFn fun index => (interface.witness index).eval env
    nifsProof := .refresh (refreshProof (logicalWidth := logicalWidth)
      (publicFits := publicFits) program env) }

def refreshOutput (circuit : Fin 2) (env : Env) : StepOutput logicalWidth publicFits :=
  let interface :=
    refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program
  { zNext := List.ofFn fun lane => (interface.nextState lane).eval env
    runningNext := fun slot => (interface.next slot).value env
    pcNext := circuit
    x := List.ofFn fun lane => (interface.outputDigest lane).eval env }

theorem refreshRepresents
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2) (env : Env) :
    RefreshStepSoundness.StepRepresents program
      (refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program) ajtai
      (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys env)
      circuit (refreshInput program env) (refreshOutput program circuit env)
      (refreshProof (logicalWidth := logicalWidth) (publicFits := publicFits) program env)
      env where
  iteration := (natWord_val _).symm
  iterationBound := Fin.isLt _
  priorPc := (natWord_val _).symm
  priorPcBound := Fin.isLt _
  verifierKeys := fun _ => rfl
  initialState := rfl
  currentState := rfl
  nextState := rfl
  witness := rfl
  priorSlots := fun _ => SlotExpr.words_eval _ env
  priorDepth := fun _ => Fin.isLt _
  nextSlots := fun _ => SlotExpr.words_eval _ env
  nextDepth := fun _ => Fin.isLt _
  pcNext := rfl
  outputDigest := rfl
  fresh := rfl
  childCommitments := fun _ _ _ => rfl
  childEvaluations := fun _ => rfl
  rounds := fun _ => rfl
  outputPad := fun _ _ => rfl
  outputMatrix := fun _ _ _ => rfl
  proof := fun _ => rfl

def refreshDigestColumn (word : Fin 4) : Nat :=
  LightAssembler.digestsStart base program.witnessWordCount + 16 + word.val

/-- A satisfying assignment of the refresh circuit's logical relation whose public input
encodes `digest` gives the refresh case of the option-B step relation, with output hash
`digest`. -/
theorem refresh_stepHolds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness → control state witness = 1)
    (firstFresh : Nat)
    (fits : LogicalPlan.Fits (LogicalPlan.rows
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
      (refreshDigestColumn program)) logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv assignment) = encHash digest)
    (rowsZero : (LogicalPlan.plan _ logicalWidth fits).RowsZero assignment) :
    let env := LogicalPlan.sourceEnv assignment
    StepHolds Folding.Schedule.productionLightFoldCap ajtai
        (decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) program).verifierKeys env)
        step control 1 (refreshInput program env) (refreshOutput program 1 env) ∧
      (refreshOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program 1 env).x =
        List.ofFn fun word : Fin 4 => digest.getD word.val 0 := by
  intro env
  obtain ⟨rows, words⟩ := LogicalPlan.rowsZero_implies_spec
    (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (refreshDigestColumn program) fits assignment digest encoded rowsZero rfl
  refine ⟨RefreshAssembler.rows_implies_stepHolds program relation base ajtai _ step control 1 rfl
    stepEq validControl env rows _ _ _ (refreshRepresents program ajtai circuitKeys 1 env), ?_⟩
  exact congrArg List.ofFn (funext words)

end Refresh

end NightstreamFPrime.Export.OptionB.StepRelation
