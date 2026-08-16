import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCS
import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.TypedBridgeFor

/-!
Contract: one authenticated fixed-width PiCCS round and its complete
production composition.

Assurance tier: model-level exact refinement and cryptographic-reduction
boundary.

Owns a message-before-challenge phase in which one fixed-width polynomial
drives both Poseidon2 replay and SumCheck algebra. It also owns exact
composition of all verifier-selected rounds to the current monolithic PiCCS
check and a named collision event for a substituted round message.

Does not own generated rows, physical columns, a Rust assignment, collision
resistance, the start or finish phase artifact, or recursive lifecycle
integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcs
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.SumCheck.Finite

universe uField uState

/-! ## Generic fixed-width fused round pass -/

/-- The only state that must persist between fixed-width SumCheck rounds. -/
structure Continuation (Field : Type uField) (State : Type uState) where
  transcriptState : State
  current : Field
  point : List Field
  cursor : Nat

/-- One message-before-challenge transcript step. -/
def replayRound
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (round : Fin shape.cubeVariables)
    (polynomial : FixedPolynomial Field degree)
    (state : State) : Field × State :=
  let absorbed := oracle.absorbRound state round polynomial.toMessage
  oracle.squeeze absorbed (.sumcheck round)

/-- One phase uses the same polynomial for transcript replay, the incoming
claim equation, and the outgoing claim evaluation. -/
def step
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Ops Field)
    (round : Fin shape.cubeVariables)
    (polynomial : FixedPolynomial Field degree)
    (before : Continuation Field State) : Continuation Field State :=
  let sampled := replayRound oracle round polynomial before.transcriptState
  {
    transcriptState := sampled.2
    current := polynomial.evaluate ops sampled.1
    point := before.point ++ [sampled.1]
    cursor := before.cursor + 1
  }

/-- Exact local relation for one fixed-width round. The round index and
incoming claim are verifier checked; the complete successor is computed. -/
def RoundPhaseRelation
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Ops Field)
    (round : Fin shape.cubeVariables)
    (polynomial : FixedPolynomial Field degree)
    (before after : Continuation Field State) : Prop :=
  before.cursor = round.val /\
    before.current = ops.add
      (polynomial.evaluate ops ops.zero)
      (polynomial.evaluate ops ops.one) /\
    after = step oracle ops round polynomial before

/-- Result of a finite phase suffix. -/
structure RunResult (Field : Type uField) (State : Type uState) where
  accepted : Bool
  continuation : Continuation Field State

/-- Execute a verifier-owned round suffix. Each message is consumed exactly
once, before its challenge is derived. -/
def runRounds
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Ops Field)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree) :
    Continuation Field State -> List (Fin shape.cubeVariables) ->
      RunResult Field State
  | continuation, [] => { accepted := true, continuation }
  | continuation, round :: remaining =>
      let polynomial := rounds round
      let localAccepted := decide (continuation.current = ops.add
        (polynomial.evaluate ops ops.zero)
        (polynomial.evaluate ops ops.one))
      let tail := runRounds oracle ops rounds
        (step oracle ops round polynomial continuation) remaining
      { accepted := localAccepted && tail.accepted
        continuation := tail.continuation }

/-- The phase cursor advances once per verifier-owned round. -/
theorem runRounds_cursor
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Ops Field)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree)
    (continuation : Continuation Field State)
    (indices : List (Fin shape.cubeVariables)) :
    (runRounds oracle ops rounds continuation indices).continuation.cursor =
      continuation.cursor + indices.length := by
  induction indices generalizing continuation with
  | nil => simp [runRounds]
  | cons round remaining inductionHypothesis =>
      simp only [runRounds]
      rw [inductionHypothesis]
      simp [step]
      omega

/-- The fused pass is exactly the fixed-width claimed-chain checker over the
challenges produced by the same transcript replay. -/
theorem runRounds_exact
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (ops : Ops Field)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree)
    (terminal : List Field -> Field)
    (continuation : Continuation Field State)
    (indices : List (Fin shape.cubeVariables)) :
    let run := runRounds oracle ops rounds continuation indices
    let replay := FiatShamir.deriveRoundsFrom oracle
      (fun round => (rounds round).toMessage)
      continuation.transcriptState indices
    run.continuation.transcriptState = replay.2 /\
      run.continuation.point = continuation.point ++ replay.1 /\
      (run.accepted && decide
        (run.continuation.current = terminal run.continuation.point)) =
        FixedPhase.checkChain ops continuation.current
          (indices.map rounds) replay.1
          (terminal (continuation.point ++ replay.1)) := by
  induction indices generalizing continuation with
  | nil =>
      simp [runRounds, FiatShamir.deriveRoundsFrom,
        FixedPhase.checkChain]
  | cons round remaining inductionHypothesis =>
      simp only [runRounds, FiatShamir.deriveRoundsFrom]
      let polynomial := rounds round
      let sampled := replayRound oracle round polynomial
        continuation.transcriptState
      let next := step oracle ops round polynomial continuation
      have tail := inductionHypothesis next
      rcases tail with ⟨stateEqual, pointEqual, acceptedEqual⟩
      refine ⟨?_, ?_, ?_⟩
      · simpa [next, step, sampled, replayRound, polynomial] using stateEqual
      · calc
          (runRounds oracle ops rounds next remaining).continuation.point =
              next.point ++
                (FiatShamir.deriveRoundsFrom oracle
                  (fun coordinate => (rounds coordinate).toMessage)
                  next.transcriptState remaining).1 := pointEqual
          _ = continuation.point ++ sampled.1 ::
                (FiatShamir.deriveRoundsFrom oracle
                  (fun coordinate => (rounds coordinate).toMessage)
                  sampled.2 remaining).1 := by
            simp [next, step, sampled, replayRound, polynomial,
              List.append_assoc]
      · simp only [List.map_cons, FixedPhase.checkChain]
        have acceptedEqual' :
            ((runRounds oracle ops rounds next remaining).accepted &&
              decide
                ((runRounds oracle ops rounds next remaining
                  ).continuation.current =
                  terminal
                    (runRounds oracle ops rounds next remaining
                      ).continuation.point)) =
              FixedPhase.checkChain ops next.current
                (remaining.map rounds)
                (FiatShamir.deriveRoundsFrom oracle
                  (fun coordinate => (rounds coordinate).toMessage)
                  next.transcriptState remaining).1
                (terminal
                  (continuation.point ++ sampled.1 ::
                    (FiatShamir.deriveRoundsFrom oracle
                      (fun coordinate => (rounds coordinate).toMessage)
                      sampled.2 remaining).1)) := by
          simpa [next, step, sampled, replayRound, polynomial,
            List.append_assoc] using acceptedEqual
        have lifted := congrArg
          (fun value : Bool =>
            decide (continuation.current = ops.add
              (polynomial.evaluate ops ops.zero)
              (polynomial.evaluate ops ops.one)) && value)
          acceptedEqual'
        simpa [next, step, sampled, replayRound, polynomial,
          Bool.and_assoc] using lifted

/-! ## Complete typed execution -/

/-- Complete fixed-width execution with the same start, terminal, and output
absorption as the paper PiCCS verifier. -/
structure Execution
    (Field : Type uField) (State : Type uState) (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field
  rounds : RunResult Field State
  outgoingState : State

def derive
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree)
    (output : ProtocolPolynomial.OutputMessage Field shape) :
    Execution Field State shape :=
  let statement : ProtocolVerifier.Statement Field State shape :=
    { priorState, input }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let start : Continuation Field State :=
    { transcriptState := pre.state
      current := input.initial ops pre.gamma
      point := []
      cursor := 0 }
  let executed := runRounds oracle.transcript ops.toOps rounds start
    (canonicalFinIndices shape.cubeVariables)
  {
    alpha := pre.alpha
    gamma := pre.gamma
    rounds := executed
    outgoingState := oracle.absorbOutput
      executed.continuation.transcriptState output
  }

def check
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree)
    (output : ProtocolPolynomial.OutputMessage Field shape) : Bool :=
  let execution := derive oracle priorState ops input rounds output
  execution.rounds.accepted && decide
    (execution.rounds.continuation.current =
      ProtocolPolynomial.terminalFromMessage ops input
        execution.alpha execution.gamma
        (cubePointOrZero ops.toOps.zero
          execution.rounds.continuation.point) output)

/-- Complete phased acceptance is the monolithic fixed-width chain over the
same verifier-derived coins. -/
theorem check_eq_fixedPhase_check
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (input : ProtocolPolynomial.VerifierInput Field shape)
    (rounds : Fin shape.cubeVariables -> FixedPolynomial Field degree)
    (output : ProtocolPolynomial.OutputMessage Field shape) :
    let certificate : ProtocolVerifier.Certificate Field shape :=
      { rounds := fun round => (rounds round).toMessage, output }
    let monolithic := ProtocolVerifier.derive oracle priorState input certificate
    check oracle priorState ops input rounds output =
      FixedPhase.checkChain ops.toOps
        (input.initial ops monolithic.coins.gamma)
        (List.ofFn rounds) monolithic.coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops input
          monolithic.coins.alpha monolithic.coins.gamma
          monolithic.coins.roundPoint output) := by
  dsimp only
  let statement : ProtocolVerifier.Statement Field State shape :=
    { priorState, input }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let indices := canonicalFinIndices shape.cubeVariables
  let replay := FiatShamir.deriveRoundsFrom oracle.transcript
    (fun round => (rounds round).toMessage) pre.state indices
  let terminal := fun coordinates =>
    ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
      (cubePointOrZero ops.toOps.zero coordinates) output
  let start : Continuation Field State :=
    { transcriptState := pre.state
      current := input.initial ops pre.gamma
      point := []
      cursor := 0 }
  have exactReplay := runRounds_exact oracle.transcript ops.toOps rounds
    terminal start indices
  have replayLength : replay.1.length = shape.cubeVariables := by
    dsimp only [replay]
    rw [FiatShamir.deriveRoundsFrom_values_length]
    exact canonicalFinIndices_length shape.cubeVariables
  let replayPoint : CubePoint Field shape.cubeVariables :=
    { coordinates := replay.1, dimension := replayLength }
  have replayPointExact :
      cubePointOrZero ops.toOps.zero replay.1 = replayPoint := by
    simp [cubePointOrZero, replayLength, replayPoint]
  have roundOrder : indices.map rounds = List.ofFn rounds := by
    simp [indices, canonicalFinIndices]
  unfold ProtocolVerifier.derive FiatShamir.derive
  change
    ((runRounds oracle.transcript ops.toOps rounds start indices).accepted &&
      decide
        ((runRounds oracle.transcript ops.toOps rounds start indices
          ).continuation.current =
          terminal
            (runRounds oracle.transcript ops.toOps rounds start indices
              ).continuation.point)) = _
  rw [exactReplay.2.2, roundOrder]
  change
    FixedPhase.checkChain ops.toOps (input.initial ops pre.gamma)
        (List.ofFn rounds) replay.1 (terminal replay.1) =
      FixedPhase.checkChain ops.toOps (input.initial ops pre.gamma)
        (List.ofFn rounds) replay.1
        (ProtocolPolynomial.terminalFromMessage ops input pre.alpha pre.gamma
          replayPoint output)
  unfold terminal
  rw [replayPointExact]

/-! ## Selected production key -/

noncomputable def productionCheck
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) : Bool :=
  let key := ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
    config artifact
  check key.oracle (key.publicInputState running fresh) key.extensionOps
    (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
      statementId config artifact running fresh)
    proof.piCcsRounds (key.piCcsCertificate running fresh proof).output

/-- The complete phased PiCCS check is exactly the current monolithic paper
NIFS PiCCS check. -/
theorem productionCheck_eq_piCcsCheck
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    productionCheck candidate statementId config artifact running fresh proof =
      piCcsCheck
        (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact) running fresh proof := by
  unfold productionCheck
  rw [check_eq_fixedPhase_check]
  rfl

@[simp] theorem production_round_count_exact :
    (canonicalFinIndices
      (ProductNifsCodec.shapeFor 26).cubeVariables).length = 26 := by
  exact canonicalFinIndices_length 26

/-! ## Substituted-message collision boundary -/

abbrev ProductionRound := FixedPolynomial K 9
abbrev BindingState := ProductPoseidon2.State

def roundFrame
    {rowVariables : Nat}
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound) : List Nat :=
  ProductPoseidon2.roundFieldsFor round polynomial.toMessage

@[simp] theorem roundFrame_length
    {rowVariables : Nat}
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound) :
    (roundFrame round polynomial).length = 40 := by
  have labelLength : ProductPoseidon2.proverMessageLabelFields.length = 16 := by
    decide
  have payloadLength :
      (polynomial.coefficients.flatMap ProductPoseidon2.kFields).length = 20 := by
    rw [List.length_flatMap]
    change (polynomial.coefficients.map (fun _ => 2)).sum = 20
    rw [List.map_const', List.sum_replicate_nat,
      polynomial.coefficients_length]
  unfold roundFrame ProductPoseidon2.roundFieldsFor
    ProductPoseidon2.proverMessageFields
  simp only [FixedPolynomial.toMessage, List.length_append,
    List.length_cons, List.length_nil]
  rw [labelLength, payloadLength]

def productionReplay
    {rowVariables : Nat}
    (prior : BindingState)
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound) : K × BindingState :=
  replayRound (ProductPoseidon2.transcriptFor rowVariables)
    round polynomial prior

/-- A distinct encoded round frame reaches the same challenge and successor
state from the same carried Poseidon2 state. -/
def RoundReplayCollision
    {rowVariables : Nat}
    (prior : BindingState)
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (authoritative : ProductionRound) : Prop :=
  exists supplied : ProductionRound,
    roundFrame round supplied ≠ roundFrame round authoritative /\
      productionReplay prior round supplied =
        productionReplay prior round authoritative

private theorem kFields_injective :
    Function.Injective ProductPoseidon2.kFields := by
  intro left right equal
  cases left with
  | mk leftZero leftOne =>
      cases right with
      | mk rightZero rightOne =>
          simp [ProductPoseidon2.kFields] at equal
          congr
          · exact Fin.ext equal.1
          · exact Fin.ext equal.2

private theorem roundFrame_injective
    {rowVariables : Nat}
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables) :
    Function.Injective (roundFrame round) := by
  intro left right equal
  have payloadEqual :
      left.coefficients.flatMap ProductPoseidon2.kFields =
        right.coefficients.flatMap ProductPoseidon2.kFields := by
    have split :
        ProductPoseidon2.word
              (left.coefficients.flatMap ProductPoseidon2.kFields).length =
            ProductPoseidon2.word
              (right.coefficients.flatMap ProductPoseidon2.kFields).length /\
          left.coefficients.flatMap ProductPoseidon2.kFields =
            right.coefficients.flatMap ProductPoseidon2.kFields := by
      simpa [roundFrame, ProductPoseidon2.roundFieldsFor,
        ProductPoseidon2.proverMessageFields, FixedPolynomial.toMessage]
        using equal
    exact split.2
  have leftLengths :
      (left.coefficients.map ProductPoseidon2.kFields).map List.length =
        List.replicate 10 2 := by
    rw [List.map_map]
    have pointwise :
        (List.length ∘ ProductPoseidon2.kFields) = (fun _ : K => 2) := by
      funext value
      rfl
    rw [pointwise, List.map_const', left.coefficients_length]
  have rightLengths :
      (right.coefficients.map ProductPoseidon2.kFields).map List.length =
        List.replicate 10 2 := by
    rw [List.map_map]
    have pointwise :
        (List.length ∘ ProductPoseidon2.kFields) = (fun _ : K => 2) := by
      funext value
      rfl
    rw [pointwise, List.map_const', right.coefficients_length]
  have blocksEqual :
      left.coefficients.map ProductPoseidon2.kFields =
        right.coefficients.map ProductPoseidon2.kFields :=
    WasmResultCodec.flatten_injective_of_lengths leftLengths rightLengths
      (by
        change (left.coefficients.map ProductPoseidon2.kFields).flatten =
          (right.coefficients.map ProductPoseidon2.kFields).flatten at payloadEqual
        exact payloadEqual)
  have coefficientsEqual : left.coefficients = right.coefficients :=
    (List.map_injective_iff.mpr kFields_injective) blocksEqual
  cases left
  cases right
  simp only at coefficientsEqual
  subst coefficientsEqual
  rfl

/-- If an accepted phase substitutes a different fixed-width polynomial while
matching the authoritative successor, it exposes the named Poseidon2 replay
collision. -/
theorem accepted_different_round_implies_collision
    {rowVariables : Nat}
    (ops : Ops K)
    (round : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (authoritative supplied : ProductionRound)
    (different : supplied ≠ authoritative)
    (before after : Continuation K BindingState)
    (accepted : RoundPhaseRelation
      (ProductPoseidon2.transcriptFor rowVariables) ops round supplied
      before after)
    (authoritativeSuccessor :
      after = step (ProductPoseidon2.transcriptFor rowVariables) ops round
        authoritative before) :
    RoundReplayCollision before.transcriptState round authoritative := by
  have stepEqual :
      step (ProductPoseidon2.transcriptFor rowVariables) ops round supplied
          before =
        step (ProductPoseidon2.transcriptFor rowVariables) ops round
          authoritative before :=
    accepted.2.2.symm.trans authoritativeSuccessor
  have stateEqual := congrArg
    (fun continuation => continuation.transcriptState) stepEqual
  have pointEqual := congrArg (fun continuation => continuation.point) stepEqual
  have challengeEqual :
      (productionReplay before.transcriptState round supplied).1 =
        (productionReplay before.transcriptState round authoritative).1 := by
    simpa [step, productionReplay] using pointEqual
  have replayEqual :
      productionReplay before.transcriptState round supplied =
        productionReplay before.transcriptState round authoritative := by
    apply Prod.ext challengeEqual
    simpa [step, productionReplay] using stateEqual
  exact ⟨supplied,
    fun frameEqual => different (roundFrame_injective round frameEqual),
    replayEqual⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority
