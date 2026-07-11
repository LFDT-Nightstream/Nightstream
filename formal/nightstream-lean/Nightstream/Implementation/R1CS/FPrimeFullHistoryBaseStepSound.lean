import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseFacts
import Nightstream.Protocol.FPrime.Step

/-!
Contract: high-level F' base-step correspondence for the exact supported
plain/stateless full-history base owner.

The theorem decodes the assignment's actual state and x_out wires.  Its
premise is exact R1CS satisfaction; the conclusion is the existing M3
Step.BaseLocalHolds relation.  No accepted or valid flag is imported.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBase
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseFacts

abbrev Digest := List Nat

universe uRunning uFresh

/-- The part of a fresh CCS claim that F' uses for its delayed public link.
The remaining claim coordinates belong to NIFS/terminal relation semantics. -/
structure Fresh where
  publicXOut : Digest
deriving DecidableEq, Repr

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by
  decide

def digestAt (values : List Nat) (start : Nat) : Digest :=
  (List.range 4).map fun lane => values.getD (start + lane) 0

def stateOfValues {Running : Type uRunning} {Fresh : Type uFresh}
    (values : List Nat)
    (initialSemantic : Digest)
    (proofState : ProofState Running Fresh) : State Digest Running Fresh Unit where
  chunkCount := values.getD 8 0
  stepCount := values.getD 9 0
  z0 := digestAt values 10
  zi := digestAt values 14
  initialSemanticState := initialSemantic
  semanticState := digestAt values 19
  pc := values.getD 18 0
  accumulatorDigest := digestAt values 23
  publicTrace := digestAt values 27
  proof := proofState
  nebula := none

def priorValues : List Nat := stateInValues
def initialSemantic : Digest := digestAt priorValues 19
def nextValues : List Nat :=
  stateOutColumns.map (ConstantPins.lookup xOutKnownPins)

def prior : State Digest Unit Fresh Unit :=
  stateOfValues priorValues initialSemantic .initial

def vkDigest : Digest := digestAt priorValues 0
def headerDigest : Digest := digestAt priorValues 4
def initialBoundary : Digest := digestAt priorValues 10
def initialPublicTrace : Digest := digestAt priorValues 27
def emptyAccumulator : Digest := digestAt priorValues 23

def chunkDigestValue : Digest :=
  traceOutputValues chunkTrace chunkInputValues

def next (fresh : Fresh) : State Digest Unit Fresh Unit :=
  { chunkCount := 1
    stepCount := 1
    z0 := initialBoundary
    zi := chunkDigestValue
    initialSemanticState := initialSemantic
    semanticState := emptyAccumulator
    pc := 1
    accumulatorDigest := emptyAccumulator
    publicTrace := chunkDigestValue
    proof := .active () [fresh]
    nebula := none }

def xOutDigestValue : Digest :=
  traceOutputValues xOutTrace xOutInputValues

def expectedXOutInputValues : List Nat :=
  [1313210370] ++ vkDigest ++ headerDigest ++
  [1, 0, 1, 0, 1, 0] ++ chunkDigestValue ++ emptyAccumulator

def low32 (value : Nat) : Nat := value % (2 ^ 32)
def high32 (value : Nat) : Nat := value / (2 ^ 32)

def stateOutputValues
    (preimage : XOut.XOutPreimage Digest Digest Digest) : List Nat :=
  [1313210370] ++
  preimage.vkFsDigest ++
  preimage.piCcsHeader ++
  [low32 preimage.chunkCount, high32 preimage.chunkCount,
   low32 preimage.stepCount, high32 preimage.stepCount,
   low32 preimage.pc, high32 preimage.pc] ++
  preimage.currentBoundary ++
  (match preimage.semanticState with
    | none => []
    | some semantic => semantic) ++
  preimage.construction2Accumulator ++
  (match preimage.nebula with
    | none => []
    | some digest => 1312967745 :: digest)

def hashValues (values : List Nat) : Digest :=
  traceOutputValues xOutTrace values

def profileHash :
    XOut.Message Unit Unit Digest Digest Digest → Digest
  | .verifier _ => vkDigest
  | .initialBoundary _ => initialBoundary
  | .publicTraceSeed _ => initialPublicTrace
  | .stateOutput preimage => hashValues (stateOutputValues preimage)

def hashSemantics : XOut.Semantics Unit Unit Digest Digest Unit Digest where
  hash := profileHash
  nebulaDigest := fun _ => []

def context : XOut.Context Unit Unit Digest Digest where
  params := ()
  structureDigest := ()
  piCcsHeader := headerDigest
  publicInputLength := some 257
  initialSemanticState := initialSemantic

def stepSemantics : Step.Semantics Digest Unit Fresh Unit Unit Unit where
  emptyRunning := ()
  initialNebula := none
  runningDigest := fun _ => emptyAccumulator
  chunkDigest := fun start fresh =>
    if start = 0 ∧ fresh.length = 1 then chunkDigestValue else []
  freshLink := fun digest fresh => decide (digest = fresh.publicXOut)
  nifsVerify := fun _ _ _ _ => none
  applicationStep := fun _ _ _ => false
  nebulaVerify := fun priorLane opening nextLane =>
    decide (priorLane = none ∧ opening = none ∧ nextLane = none)

def input (fresh : Fresh) : Step.Input Fresh Unit Unit where
  nextLatest := [fresh]
  nebulaOpen := none
  nebulaNext := none

def proof : Step.Proof Digest Unit Unit where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := emptyAccumulator
  xOut := xOutDigestValue

theorem artifact_xOut_input_values :
    xOutInputValues = expectedXOutInputValues := by
  unfold xOutInputValues
  rw [xOutTrace_inputColumns]
  simp [expectedXOutInputValues, xOutKnownPins, constantPins, stateInPins,
    chunkStatePins, semanticStatePins, counterHalfPins,
    EqualityPins.transferPins, chunkDigestPairs, semanticAccumulatorPairs,
    chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    traceOutputValues, chunkDigestValue, vkDigest, headerDigest,
    emptyAccumulator, digestAt, priorValues, stateInValues,
    rangeFour, ConstantPins.lookup]

theorem next_preimage_values :
    ∀ fresh, stateOutputValues
      (XOut.preimage hashSemantics .stateless context (next fresh)) =
      xOutInputValues := by
  intro fresh
  rw [artifact_xOut_input_values]
  simp [stateOutputValues, XOut.preimage, hashSemantics, profileHash,
    XOut.verifierDigest, context, next, expectedXOutInputValues,
    low32, high32]

theorem prior_initial :
    Step.InitialState hashSemantics stepSemantics .stateless context prior := by
  simp [Step.InitialState, prior, stateOfValues, priorValues, initialSemantic,
    hashSemantics, profileHash, context, XOut.initialBoundary,
    XOut.initialBoundaryPreimage, XOut.publicTraceSeed, stepSemantics,
    initialBoundary, initialPublicTrace, emptyAccumulator, digestAt,
    stateInValues, rangeFour]

theorem semantic_advance :
    ∀ fresh, Step.SemanticAdvance stepSemantics .stateless prior ()
      (input fresh) proof := by
  intro fresh
  simp [Step.SemanticAdvance, stepSemantics, proof, emptyAccumulator]

theorem nebula_advance :
    ∀ fresh, Step.NebulaAdvance stepSemantics prior (input fresh) proof := by
  intro fresh
  unfold Step.NebulaAdvance
  constructor
  · rfl
  · change decide ((none : Option Unit) = none ∧
        (none : Option Unit) = none ∧
        (none : Option Unit) = none) = true
    decide

theorem installed_nebula :
    ∀ fresh, Step.installedNebula prior (input fresh) = (none : Option Unit) := by
  intro fresh
  simp [Step.installedNebula, input, prior, stateOfValues]

theorem next_advanced :
    ∀ fresh, next fresh =
      Step.advancedState stepSemantics prior () (input fresh) proof := by
  intro fresh
  unfold Step.advancedState
  rw [installed_nebula fresh]
  simp [next, stepSemantics, prior, stateOfValues,
    priorValues, input, proof, initialSemantic, initialBoundary,
    emptyAccumulator, chunkDigestValue, digestAt, stateInValues, rangeFour]

theorem output_binding :
    ∀ fresh, proof.xOut =
      XOut.compute hashSemantics .stateless context (next fresh) := by
  intro fresh
  change xOutDigestValue =
    hashValues
      (stateOutputValues
        (XOut.preimage hashSemantics .stateless context (next fresh)))
  rw [next_preimage_values fresh]
  rfl

/-- Exact supported-profile BaseLocalHolds theorem, independent of any honest
witness vector. -/
theorem profile_baseLocal (fresh : Fresh) :
    Step.BaseLocalHolds hashSemantics stepSemantics .stateless context
      prior (next fresh) (input fresh) proof := by
  refine ⟨prior_initial, rfl, ?_, semantic_advance fresh, nebula_advance fresh,
    next_advanced fresh, output_binding fresh⟩
  simp [input]

def decodedPrior (assignment : Nat → Nat) : State Digest Unit Fresh Unit :=
  stateOfValues (stateInColumns.map assignment) initialSemantic .initial

def decodedNext (assignment : Nat → Nat) (fresh : Fresh) :
    State Digest Unit Fresh Unit :=
  stateOfValues (stateOutColumns.map assignment) initialSemantic
    (.active () [fresh])

def decodedProof (assignment : Nat → Nat) : Step.Proof Digest Unit Unit where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := digestAt (stateOutColumns.map assignment) 19
  xOut := xOutColumns.map assignment

theorem stateInColumnsCovered :
    ConstantPins.Covers stateInColumns stateInPins := by
  native_decide

theorem stateInLookupValues :
    stateInColumns.map (ConstantPins.lookup stateInPins) = stateInValues := by
  native_decide

theorem stateInValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    stateInColumns.map assignment = stateInValues := by
  exact (ConstantPins.map_assignment_eq_lookup facts.stateIn
    stateInColumnsCovered).trans stateInLookupValues

theorem stateOutColumnsCovered :
    ConstantPins.Covers stateOutColumns xOutKnownPins := by
  rw [ConstantPins.covers_iff_keys]
  rw [xOutKnownPins_keys]
  native_decide

theorem stateOutValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    stateOutColumns.map assignment = nextValues := by
  exact ConstantPins.map_assignment_eq_lookup
    (xOutKnownPins_sound facts) stateOutColumnsCovered

theorem xOutColumns_eq_traceOutputs :
    xOutColumns = xOutTrace.outputColumns := by
  native_decide

theorem xOutValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    xOutColumns.map assignment = xOutDigestValue := by
  rw [xOutColumns_eq_traceOutputs, xOutTrace_outputColumns]
  have lane0 := facts.sponge xOutTrace (by native_decide) 0 (by decide)
  have lane1 := facts.sponge xOutTrace (by native_decide) 1 (by decide)
  have lane2 := facts.sponge xOutTrace (by native_decide) 2 (by decide)
  have lane3 := facts.sponge xOutTrace (by native_decide) 3 (by decide)
  rw [xOutInputValues_sound facts] at lane0 lane1 lane2 lane3
  simpa [traceOutputValues, xOutDigestValue, rangeFour,
    xOutTrace_outputColumns] using And.intro lane0
      (And.intro lane1 (And.intro lane2 lane3))

theorem decodedPrior_eq {assignment : Nat → Nat}
    (facts : Facts assignment) : decodedPrior assignment = prior := by
  rw [decodedPrior, stateInValues_sound facts]
  rfl

theorem nextValues_state (fresh : Fresh) :
    stateOfValues nextValues initialSemantic (.active () [fresh]) = next fresh := by
  simp [nextValues, stateOutColumns, stateOfValues, next, xOutKnownPins, constantPins,
    stateInPins, chunkStatePins, semanticStatePins, counterHalfPins,
    EqualityPins.transferPins, chunkDigestPairs, semanticAccumulatorPairs,
    chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    chunkDigestValue, traceOutputValues, initialBoundary, initialSemantic,
    emptyAccumulator, digestAt, priorValues, stateInValues, rangeFour,
    ConstantPins.lookup]

theorem decodedNext_eq {assignment : Nat → Nat}
    (facts : Facts assignment) (fresh : Fresh) :
    decodedNext assignment fresh = next fresh := by
  rw [decodedNext, stateOutValues_sound facts]
  exact nextValues_state fresh

theorem decodedProof_eq {assignment : Nat → Nat}
    (facts : Facts assignment) : decodedProof assignment = proof := by
  rw [decodedProof, stateOutValues_sound facts, xOutValues_sound facts]
  have nextState := nextValues_state (Fresh.mk [])
  have semanticEq := congrArg
    (fun state : State Digest Unit Fresh Unit => state.semanticState) nextState
  simpa [proof, stateOfValues] using congrArg
    (fun digest => ({ fold := Step.FoldProof.noFold
                      nebulaOpen := (none : Option Unit)
                      semanticStateDigest := digest
                      xOut := xOutDigestValue } : Step.Proof Digest Unit Unit))
    semanticEq

/-- `CIR-SOUND` for the exact generated plain/stateless base owner: every
canonical satisfying assignment decodes to the M3 local-step relation. -/
theorem fPrimeFullHistoryBase_local_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment)
    (fresh : Fresh) :
    Step.BaseLocalHolds hashSemantics stepSemantics .stateless context
      (decodedPrior assignment) (decodedNext assignment fresh) (input fresh)
      (decodedProof assignment) := by
  have facts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical one
    satisfies
  rw [decodedPrior_eq facts, decodedNext_eq facts fresh, decodedProof_eq facts]
  exact profile_baseLocal fresh

/-- Branch-dispatched form consumed by full-history circuit composition. -/
theorem fPrimeFullHistoryBase_step_local_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment)
    (fresh : Fresh) :
    Step.LocalHolds hashSemantics stepSemantics .stateless context
      (decodedPrior assignment) (decodedNext assignment fresh) (input fresh)
      (decodedProof assignment) := by
  have base := fPrimeFullHistoryBase_local_sound goldilocksPrime canonical one
    satisfies fresh
  have facts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical one
    satisfies
  rw [decodedPrior_eq facts, decodedProof_eq facts] at base ⊢
  simpa [Step.LocalHolds, prior, proof, stateOfValues] using base

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound
