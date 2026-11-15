import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSchedule
import NightstreamFPrime.Lifecycle.RandomOracle.Pieces

/-! Vectors of the hashed encodings that Rust computes outside the step packages
(`SECURITY_MODEL.md`, section Hashed encodings). The value is
`[schema, circuitKeys, foldTranscripts]`.

`circuitKeys` holds `[identity, digest 0, digest 1]` of
`Lifecycle.BoundedDepth.CircuitKeyDigest.digest` for each identity of `identities`. Rust must
give the same words (`crates/nightstream/tests/bounded_depth_lifecycle/circuit_key.rs`).

`foldTranscripts` holds the light and the refresh fold transcript on sample data
(`foldTranscript`): the pieces of `Lifecycle.RandomOracle.Schedule.pieces`, the first and the last
round challenge and the sponge state after the output. Rust must absorb the same lengths and reach the same
state (`crates/nightstream-fprime/tests/fold_transcript_lean.rs`). -/

namespace NightstreamFPrime.Export.EncodingParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle
open NightstreamFPrime.Export.BoundedDepth.StepRelation (keyWidth keyPublicFits)

def schema : Nat := 1

def words (values : List F) : Value :=
  .array (values.map fun value => .atom value.val)

def identities : List (List F) :=
  [List.replicate 4 0, (List.range 4).map Poseidon2.ofNat,
    (List.range 4).map fun word => Poseidon2.ofNat (fieldModulus - 1 - word)]

def circuitKeys : Value :=
  .array (identities.map fun identity =>
    .array [words identity, words (CircuitKeyDigest.digest identity 0),
      words (CircuitKeyDigest.digest identity 1)])

/-- Word `index` of sample list `seed`: a small value that changes with the position. -/
def sample (seed index : Nat) : F := Poseidon2.ofNat ((seed + 31 * index) % 251)

def sampleK (seed index : Nat) : K := ⟨sample seed (2 * index), sample seed (2 * index + 1)⟩

def sampleCommitment (seed : Nat) : PaperAlgebra.Commitment :=
  fun row coefficient => sample seed (ringDegree * row.val + coefficient.val)

def sampleEvaluations (shape : Shape) (seed : Nat) : StrongReduction.EvaluationFamily K shape :=
  ⟨fun coefficient => sampleK seed coefficient.val,
    fun matrix coefficient => sampleK (seed + 1 + matrix.val) coefficient.val⟩

def sampleFresh (shape : Shape) :
    Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := keyWidth) (publicFits := keyPublicFits)) shape :=
  ⟨fun source => sampleCommitment (1 + source.val), fun source cell => sample (2 + source.val) cell.val⟩

def sampleMessages (schedule : Schedule) (width : Nat)
    (output : schedule.Output) : schedule.Messages width :=
  ⟨fun round => ⟨(List.range (width + 1)).map fun index => sampleK (3 + round.val) index,
    by simp⟩, output⟩

def sampleOutput (shape : Shape) : FullOutputCoordinates.FullOutput K shape :=
  ⟨fun source => (sampleEvaluations shape (5 + 7 * source.val)).pad,
    fun source => (sampleEvaluations shape (5 + 7 * source.val)).matrix⟩

/-- `[pieces, round challenges, state]` of a fold transcript: each piece is `[0, words]` for an
absorb or `[1, count]` for reads, the challenges are the reads of `Schedule.challengeCalls` for
the first and the last round, and the state is the sponge state after `Schedule.outputCalls`. -/
def foldTranscript (schedule : Schedule) (width : Nat) (statement : schedule.Statement)
    (messages : schedule.Messages width) : Value :=
  let piece : Sponge.Piece → Value
    | .absorb values => .array [.atom 0, words values]
    | .draw count => .array [.atom 1, .atom count]
  let challenge (round : Fin schedule.shape.cubeVariables) : Value :=
    let value : K := Schedule.Challenge.read (schedule := schedule) (.round round)
      (Sponge.run Transcript.initialState (schedule.challengeCalls statement messages (.round round)))
    words [value.c0, value.c1]
  .array [.array ((schedule.pieces statement messages).map piece),
    .array (((canonicalFinIndices schedule.shape.cubeVariables).filter fun round =>
      round.val = 0 || round.val + 1 = schedule.shape.cubeVariables).map challenge),
    words (Sponge.run Transcript.initialState (schedule.outputCalls statement messages))]

/-- The round polynomials of both folds have degree 8 (`ProductionKey.degreeBound`). -/
def foldDegree : Nat := 8

def foldTranscripts : Value :=
  let light := LightSchedule.schedule keyWidth keyPublicFits
  let refresh := RefreshSchedule.schedule keyWidth keyPublicFits
  .array [
    foldTranscript light foldDegree (sampleFresh LightShape)
      (sampleMessages light foldDegree (sampleOutput LightShape)),
    foldTranscript refresh foldDegree
      (sampleFresh productionShape,
        ⟨fun child => sampleCommitment (20 + child.val),
          fun child => sampleEvaluations productionShape (40 + 5 * child.val)⟩)
      (sampleMessages refresh foldDegree (sampleOutput productionShape))]

def parityValue : Value := .array [.atom schema, circuitKeys, foldTranscripts]

end NightstreamFPrime.Export.EncodingParity
