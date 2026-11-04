import CompressionLean.Spec.OptionB
import CompressionLean.Spec.OptionBParent
import CompressionLean.Game.Compose
import NightstreamFPrime.Export.OptionB.FinishWalk

/-! The final verifier's claim at the start of option B's IVC walk (CW-10.1).

`LastFoldAt` states `OptionB.ParentHolds` in the walk's terms. It names finish columns, whose
refresh step input has the statement's iteration and states (`terminalOf`) and a nonzero
iteration, with the finish checks under the setup's keys (`FinishStep.FinishHolds`). It also
names witnesses of both next slots under their circuits' relations. `ParentHolds` reads the key
digests from the finish columns. The final program binds those columns to its state keys, so the
digests are the setup's when the state keys are the setup's digests (`KeysAre`).

The finish stage's NIFS adversary draws an outcome of the final verifier's experiment
(`finalGame`). It outputs the finish attempt of the accepted statement's last fold
(`finishAdversary`, `lastAttempt`). The chain's IVC adversary runs that adversary with the finish
extractor's coins. For a statement whose iteration is at most `depth`, it outputs the terminal
that the answer gives (`chainAdversary`, `chainOutput`). Both adversaries choose the last fold with
`Classical.epsilon` (`lastChoice`). The compression claims prove that a last fold exists. They do
not compute one. -/

set_option autoImplicit false

namespace CompressionLean.Spec.OptionB

open CompressionLean.Game
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open NightstreamFPrime.Spec.HyperNova.ReverseStages (NifsAdversary NifsExtractor IvcAdversary)
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.FinishWalk
open NightstreamFPrime.Export.OptionB.IvcWalk (WalkAttempt Answer)
open NightstreamFPrime.Export.OptionB.IvcSecurity (Output)
open NightstreamFPrime.Export.OptionB.StepPackage (Package)

/-- The final verifier's state-restoration experiment of `option_b_bound`: the shrink game with the
choice of both slots' layer-1 games, restored `calls` times. -/
noncomputable abbrev finalGame (finish : Package) (stateKeys : List Replay.Word)
    (keys : Fin 2 → CompressionLean.Circuit.Layer1Key) (S : Fin 2 → Layer1.Setup Gl) (L0 L1 Ls calls : ℕ) :
    Game (Outcome Gl Cubic (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)) :=
  restorePair (Shrink.shrinkGame (F := Gl) CompressionLean.Circuit.Block.poseidonTemplate
      (counts finish stateKeys keys) (shrinkRun finish stateKeys keys) Ls)
    (choice (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)) calls [] []

/-- The statement that the shrink layer accepted, if any. -/
def acceptedOf {F K : Type} (shrink : Option (Option (Shrink.Transcript StateWords F K))) :
    Option StateWords :=
  shrink.join.map Shrink.Transcript.inst

/-- The application statement of the final verifier's words: the iteration and both states. -/
def terminalOf (x : StateWords) : TerminalStatement AppState :=
  ⟨(x 0).val, List.ofFn fun lane : Fin 4 => x ⟨1 + lane, by omega⟩,
    List.ofFn fun lane : Fin 4 => x ⟨5 + lane, by omega⟩⟩

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- The final program's state keys are the circuit keys' digests. -/
def KeysAre (stateKeys : List Replay.Word) (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) :
    Prop :=
  stateKeys.map (Replay.Word.cast (F := NightstreamFPrime.Spec.F)) =
    (circuitKeys 0).digest ++ (circuitKeys 1).digest

/-- A last fold at the finish columns `env` with next-slot witnesses `witnesses`: a nonzero
iteration, the refresh step input's statement is `x`'s, the finish checks hold under the setup's
keys, and both next slots hold under their circuits' relations. -/
def LastFoldAt (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (x : StateWords) (env : Env)
    (witnesses : Fin 2 → Lifecycle.OptionB.Terminal.Witness (logicalWidth := lw) (publicFits := pf)) :
    Prop :=
  x 0 ≠ 0 ∧
    finishStatement (refreshInput (logicalWidth := lw) (publicFits := pf) program env) = terminalOf x ∧
    Lifecycle.OptionB.FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys 1
      (refreshInput program env) (refreshOutput program 1 env) ∧
    ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (Lifecycle.OptionB.toClaim ((circuitKeys slot).system ajtai)
        ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 env).runningNext slot).1)
      (witnesses slot)

/-- Some finish columns and next-slot witnesses give a last fold for `x`. -/
def LastFold (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (x : StateWords) : Prop :=
  ∃ env witnesses, LastFoldAt program ajtai circuitKeys x env witnesses

section Chain

variable (program : Stage1.Application.Program)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
  (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf)

/-- The finish columns and next-slot witnesses of a last fold for `x`, when one exists. -/
noncomputable def lastChoice (x : StateWords) :
    Env × (Fin 2 → Lifecycle.OptionB.Terminal.Witness (logicalWidth := lw) (publicFits := pf)) :=
  Classical.epsilon fun choice => LastFoldAt program ajtai circuitKeys x choice.1 choice.2

/-- The finish attempt of the last fold that `lastChoice` names. -/
noncomputable def lastAttempt (x : StateWords) : WalkAttempt lw pf :=
  finishAttempt circuitKeys (refreshInput program (lastChoice program ajtai circuitKeys x).1)
    (refreshOutput program 1 (lastChoice program ajtai circuitKeys x).1)
    (lastChoice program ajtai circuitKeys x).2

/-- The terminal that the finish answer gives for `x` when its iteration is at most `depth`, and
the base terminal at iteration 0 otherwise. -/
noncomputable def chainOutput (depth : ℕ) (x : StateWords) (answer : Answer lw pf) : Output lw pf :=
  if (x 0).val ≤ depth then
    (terminalOf x, .recursive (finishPayload (refreshInput program (lastChoice program ajtai circuitKeys x).1)
      (lastChoice program ajtai circuitKeys x).2 answer))
  else (⟨0, [], []⟩, .bottom)

/-- The finish stage's NIFS adversary: it draws an outcome from `law` and outputs the finish attempt
of the accepted statement. -/
noncomputable def finishAdversary {Ω : Type} (law : PMF Ω) (accepted : Ω → Option StateWords) :
    NifsAdversary (WalkAttempt lw pf) :=
  ⟨Ω, law, fun ω => (accepted ω).map (lastAttempt program ajtai circuitKeys)⟩

/-- The chain's IVC adversary: the finish stage's adversary and extractor, and the terminal that
the answer gives. -/
noncomputable def chainAdversary {Ω : Type} (law : PMF Ω) (accepted : Ω → Option StateWords)
    (extractor : NifsExtractor (Answer lw pf) (finishAdversary program ajtai circuitKeys law accepted))
    (depth : ℕ) : IvcAdversary (Output lw pf) :=
  ⟨Ω × extractor.Coins, extractor.law, fun draw =>
    chainOutput program ajtai circuitKeys depth ((accepted draw.1).getD 0) (extractor.run draw.1 draw.2)⟩

end Chain

end CompressionLean.Spec.OptionB
