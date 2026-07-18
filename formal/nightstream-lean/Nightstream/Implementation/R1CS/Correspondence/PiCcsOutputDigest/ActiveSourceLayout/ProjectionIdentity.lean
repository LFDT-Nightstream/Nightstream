import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionBatchSound
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Pairing
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.TraceNormalForm
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-!
Active-profile semantic contract for the two PiRLC `y_zcol` projection
identities.

Assurance tier: model-level representation and arithmetic correspondence.

Owns: an artifact-independent pair of low/high `ProjectionTrace` values;
their protocol -> phase -> limb-family shape contract; deterministic
source-R1CS-definition/check soundness; a consumer derived directly from the
trace input columns; reconstruction of the independent typed Phi81 `RingK`
fold; and composition with the typed active PiCCS producer boundary.

Does not own: concrete columns or rows, the Rust fixed-point audit, PiCCS
`y_zcol` source truth, transcript derivation of rho or beta, parent-opening
authority, bad-root probability, encoded lowering, cost claims, necessity, or
permission to remove rows.

Emits constraints: no.

Authority boundary: `RowsSatisfied` names the semantic consequences that an
exact row artifact must establish; it is not an acceptance predicate and it
cannot be supplied by a digest. The trace itself determines the PiRLC input
consumer. `ConsumerMatches` and the PiCCS `yZcolOutput` binding remain separate
premises, while transcript and parent-opening authority stay outside this
module.

| Protocol -> phase -> family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `identities.y_zcol.challenge_columns` | both limbs use the same source-indexed rho coefficients | direct dataflow | `ShapeValid.challengeColumnsShared` |
| `identities.y_zcol.{limb}` | each exact trace is one 54-coefficient Phi81 remainder | derived | `lowExact_output`, `highExact_output` |
| `identities.y_zcol.pair` | pair low/high remainders into one typed `RingK` fold | derived | `batchExact_decodedOutput_eq_sourceAggregate` |
| `pi_ccs_to_pi_rlc.y_zcol.inputs` | projection inputs define the consumer; typed producer columns equal every consumer leaf | computed + checked refinement | `TracePair.inputConsumer`, `decodedInputs_eq_inputConsumer`, `ConsumerMatches` |
| `identities.y_zcol.output` | accepted rows imply the typed source aggregate or one named bad root | security boundary | `rows_decodedOutput_eq_messageAggregate_or_badRoot` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.Implementation.R1CS.ProjectionPhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.ProjectionCheck

/-- The two physical base-field identities, indexed by the independent active
semantic source count rather than a generated profile constant. -/
structure TracePair (shape : SemanticShape) where
  low : ProjectionProgram.ProjectionTrace
  high : ProjectionProgram.ProjectionTrace
  lowPairCount : low.pairs.length = shape.sourceCount
  highPairCount : high.pairs.length = shape.sourceCount

def TracePair.traces {shape : SemanticShape}
    (pair : TracePair shape) : List ProjectionProgram.ProjectionTrace :=
  [pair.low, pair.high]

def TracePair.lowPair {shape : SemanticShape}
    (pair : TracePair shape) (index : Fin shape.sourceCount) :
    ProjectionProgram.PairTrace :=
  pair.low.pairs.get (Fin.cast pair.lowPairCount.symm index)

def TracePair.highPair {shape : SemanticShape}
    (pair : TracePair shape) (index : Fin shape.sourceCount) :
    ProjectionProgram.PairTrace :=
  pair.high.pairs.get (Fin.cast pair.highPairCount.symm index)

/-- The PiRLC consumer is determined by the trace input columns themselves;
no assignment-specific value equality is accepted as a substitute for this
physical dataflow. -/
def TracePair.inputConsumer {shape : SemanticShape}
    (pair : TracePair shape) : YZcolConsumer.ConsumerColumns shape where
  column limb source lane :=
    match limb with
    | .c0 => (pair.lowPair source).inputColumns.getD lane.val 0
    | .c1 => (pair.highPair source).inputColumns.getD lane.val 0

/-- Static protocol shape checked independently of assignment values. Shared
rho evaluator equality is stronger than merely sharing source columns: it
also prevents duplicate evaluator outputs from being charged as two leaves. -/
structure ShapeValid {shape : SemanticShape}
    (pair : TracePair shape) : Prop where
  sourceCountPositive : 0 < shape.sourceCount
  lowLayout : pair.low.LayoutValid
  highLayout : pair.high.LayoutValid
  lowPairWidths : ∀ candidate, candidate ∈ pair.low.pairs →
    candidate.rhoColumns.length = ringDegree ∧
      candidate.inputColumns.length = ringDegree
  highPairWidths : ∀ candidate, candidate ∈ pair.high.pairs →
    candidate.rhoColumns.length = ringDegree ∧
      candidate.inputColumns.length = ringDegree
  challengeColumnsShared : forall index : Fin shape.sourceCount,
    (pair.highPair index).rhoColumns =
      (pair.lowPair index).rhoColumns

/-- Exact definition and final-check consequences expected from a physical
row artifact. A later active row bridge must derive these fields from full
R1CS satisfaction; callers must not assert this structure from trace labels. -/
structure RowsSatisfied {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) : Prop where
  definitions : ProjectionProgram.DefinitionsHold assignment
    (pair.traces.flatMap ProjectionProgram.ProjectionTrace.definitions)
  checks : Satisfies
    (pair.traces.flatMap ProjectionProgram.ProjectionTrace.checks) assignment

private theorem pairs_nonempty_of_count
    {Alpha : Type} {items : List Alpha} {count : Nat}
    (lengthEq : items.length = count) (positive : 0 < count) :
    items ≠ [] := by
  intro empty
  rw [empty] at lengthEq
  simp at lengthEq
  omega

/-- The exact generic row consequences make both bounded identities accepted.
No generated artifact or fixed column number occurs in this theorem. -/
theorem rows_batchAccepted
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair)
    (constantOne : assignment 0 = 1)
    (rows : RowsSatisfied pair assignment) :
    BatchAccepted ProjectionProgram.K.ops
      (ProjectionProgram.BatchIdentity pair.traces assignment) := by
  apply ProjectionProgram.ProjectionTrace.census_batchAccepted
    pair.traces assignment constantOne
  · intro trace member
    have member : trace = pair.low ∨ trace = pair.high := by
      simpa [TracePair.traces] using member
    rcases member with rfl | rfl
    · exact valid.lowLayout
    · exact valid.highLayout
  · intro trace member
    have member : trace = pair.low ∨ trace = pair.high := by
      simpa [TracePair.traces] using member
    rcases member with rfl | rfl
    · exact pairs_nonempty_of_count pair.lowPairCount
        valid.sourceCountPositive
    · exact pairs_nonempty_of_count pair.highPairCount
        valid.sourceCountPositive
  · intro trace member candidate candidateMember
    have member : trace = pair.low ∨ trace = pair.high := by
      simpa [TracePair.traces] using member
    rcases member with rfl | rfl
    · exact valid.lowPairWidths candidate candidateMember
    · exact valid.highPairWidths candidate candidateMember
  · exact rows.definitions
  · exact rows.checks

/-- Deterministic one-point boundary for the complete two-limb batch. -/
theorem rows_batchExact_or_badRoot
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair)
    (constantOne : assignment 0 = 1)
    (rows : RowsSatisfied pair assignment) :
    BatchExact (ProjectionProgram.BatchIdentity pair.traces assignment) ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity pair.traces assignment) := by
  exact batchAccepted_implies_exact_or_badRoot ProjectionProgram.K.ops
    (ProjectionProgram.BatchIdentity pair.traces assignment)
    (rows_batchAccepted valid constantOne rows)

def lowChallengeRings {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> Ring :=
  fun index => values assignment (pair.lowPair index).rhoColumns

def highChallengeRings {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> Ring :=
  fun index => values assignment (pair.highPair index).rhoColumns

def lowInputRings {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> Ring :=
  fun index => values assignment (pair.lowPair index).inputColumns

def highInputRings {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> Ring :=
  fun index => values assignment (pair.highPair index).inputColumns

def decodedChallenges {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> RingF :=
  fun index => ringOfList (lowChallengeRings pair assignment index)

def decodedInputs {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) :
    Fin shape.sourceCount -> RingK :=
  fun index => pairRings
    (lowInputRings pair assignment index)
    (highInputRings pair assignment index)

def decodedOutput {shape : SemanticShape}
    (pair : TracePair shape) (assignment : Nat -> Nat) : RingK :=
  pairRings
    (values assignment pair.low.outputColumns)
    (values assignment pair.high.outputColumns)

private theorem lowPair_widths_indexed
    {shape : SemanticShape} {pair : TracePair shape}
    (valid : ShapeValid pair) (index : Fin shape.sourceCount) :
    (pair.lowPair index).rhoColumns.length = ringDegree ∧
      (pair.lowPair index).inputColumns.length = ringDegree := by
  apply valid.lowPairWidths
  exact List.get_mem pair.low.pairs
    (Fin.cast pair.lowPairCount.symm index)

private theorem highPair_widths_indexed
    {shape : SemanticShape} {pair : TracePair shape}
    (valid : ShapeValid pair) (index : Fin shape.sourceCount) :
    (pair.highPair index).rhoColumns.length = ringDegree ∧
      (pair.highPair index).inputColumns.length = ringDegree := by
  apply valid.highPairWidths
  exact List.get_mem pair.high.pairs
    (Fin.cast pair.highPairCount.symm index)

/-- Exactness of the low base-field identity eliminates its quotient and
returns the unique 54-coefficient Phi81 remainder. -/
theorem lowExact_output
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair)
    (exact : (pair.low.identity assignment).Exact) :
    values assignment pair.low.outputColumns =
      phi81Combine (lowChallengeRings pair assignment)
        (lowInputRings pair assignment) := by
  rcases valid.lowLayout with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, outputWidth,
      quotientWidth, maxDegree⟩
  simpa [lowChallengeRings, lowInputRings, TracePair.lowPair] using
    exact_output_eq_phi81Combine
      (count := shape.sourceCount) assignment pair.low pair.lowPairCount
      (fun index => (lowPair_widths_indexed valid index).1)
      (fun index => (lowPair_widths_indexed valid index).2)
      outputWidth quotientWidth maxDegree exact

/-- High-limb counterpart of `lowExact_output`. -/
theorem highExact_output
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair)
    (exact : (pair.high.identity assignment).Exact) :
    values assignment pair.high.outputColumns =
      phi81Combine (highChallengeRings pair assignment)
        (highInputRings pair assignment) := by
  rcases valid.highLayout with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, outputWidth,
      quotientWidth, maxDegree⟩
  simpa [highChallengeRings, highInputRings, TracePair.highPair] using
    exact_output_eq_phi81Combine
      (count := shape.sourceCount) assignment pair.high pair.highPairCount
      (fun index => (highPair_widths_indexed valid index).1)
      (fun index => (highPair_widths_indexed valid index).2)
      outputWidth quotientWidth maxDegree exact

theorem challengeRings_shared
    {shape : SemanticShape} {pair : TracePair shape}
    (valid : ShapeValid pair) (assignment : Nat -> Nat) :
    highChallengeRings pair assignment =
      lowChallengeRings pair assignment := by
  funext index
  unfold highChallengeRings lowChallengeRings
  rw [valid.challengeColumnsShared index]

private theorem lowExact_of_batch
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity pair.traces assignment)) :
    (pair.low.identity assignment).Exact := by
  apply exact
  simp [ProjectionProgram.BatchIdentity, TracePair.traces]

private theorem highExact_of_batch
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity pair.traces assignment)) :
    (pair.high.identity assignment).Exact := by
  apply exact
  simp [ProjectionProgram.BatchIdentity, TracePair.traces]

/-- Coefficient exactness of both base-field identities is precisely the
independent typed extension-ring fold. -/
theorem batchExact_decodedOutput_eq_sourceAggregate
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity pair.traces assignment)) :
    decodedOutput pair assignment =
      sourceAggregate (decodedChallenges pair assignment)
        (decodedInputs pair assignment) := by
  have low := lowExact_output valid (lowExact_of_batch exact)
  have high := highExact_output valid (highExact_of_batch exact)
  rw [challengeRings_shared valid assignment] at high
  unfold decodedOutput decodedChallenges decodedInputs sourceAggregate
  rw [low, high]
  exact pairRings_phi81Combine
    (lowChallengeRings pair assignment)
    (lowInputRings pair assignment)
    (highInputRings pair assignment)

/-- Decoding the trace-derived consumer gives exactly the same complete input
vector as decoding the two trace coefficient lists. -/
theorem decodedInputs_eq_inputConsumer
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    (valid : ShapeValid pair) :
    decodedInputs pair assignment =
      YZcolConsumer.decodedInputs (semanticAssignment assignment)
        pair.inputConsumer := by
  funext source lane
  change
    Concrete.K.mk
      ((values assignment
        (pair.lowPair source).inputColumns).getD lane.val 0)
      ((values assignment
        (pair.highPair source).inputColumns).getD lane.val 0) =
    Concrete.K.mk
      (semanticAssignment assignment
        ((pair.lowPair source).inputColumns.getD lane.val 0))
      (semanticAssignment assignment
        ((pair.highPair source).inputColumns.getD lane.val 0))
  rw [values_getD_of_length assignment
      (pair.lowPair source).inputColumns
      (lowPair_widths_indexed valid source).2 lane,
    values_getD_of_length assignment
      (pair.highPair source).inputColumns
      (highPair_widths_indexed valid source).2 lane]
  rfl

/-- Exact traces plus typed producer/consumer column identity refine the
decoded output to the independently bound PiCCS source aggregate. Transcript
and parent-opening rewrites are intentionally absent. -/
theorem batchExact_decodedOutput_eq_messageAggregate
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    {producer : SourceRole shape -> Nat}
    {message : OutputMessage shape}
    (valid : ShapeValid pair)
    (consumerMatch : YZcolConsumer.ConsumerMatches producer
      pair.inputConsumer)
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment assignment) producer message)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity pair.traces assignment)) :
    decodedOutput pair assignment =
      sourceAggregate (decodedChallenges pair assignment) message.yZcol := by
  have inputsBound := YZcolConsumer.decodedInputs_eq_yZcol_of_bound
    consumerMatch yZcolBound
  calc
    decodedOutput pair assignment =
        sourceAggregate (decodedChallenges pair assignment)
        (decodedInputs pair assignment) :=
      batchExact_decodedOutput_eq_sourceAggregate valid exact
    _ = sourceAggregate (decodedChallenges pair assignment)
        message.yZcol := by
      rw [decodedInputs_eq_inputConsumer valid, inputsBound]

/-- Complete model-level active boundary. The right branch is the exact
sampled bad-root event; no probability or transcript claim is hidden here. -/
theorem rows_decodedOutput_eq_messageAggregate_or_badRoot
    {shape : SemanticShape} {pair : TracePair shape}
    {assignment : Nat -> Nat}
    {producer : SourceRole shape -> Nat}
    {message : OutputMessage shape}
    (valid : ShapeValid pair)
    (constantOne : assignment 0 = 1)
    (rows : RowsSatisfied pair assignment)
    (consumerMatch : YZcolConsumer.ConsumerMatches producer
      pair.inputConsumer)
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment assignment) producer message) :
    decodedOutput pair assignment =
        sourceAggregate (decodedChallenges pair assignment) message.yZcol ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity pair.traces assignment) := by
  rcases rows_batchExact_or_badRoot valid constantOne rows with
    exact | badRoot
  · exact Or.inl (batchExact_decodedOutput_eq_messageAggregate valid
      consumerMatch yZcolBound exact)
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity
