import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.BoundedDepth.CompressionSponge
import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Lifecycle.BoundedDepth.ApplicationIdentity
import NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSchedule
import NightstreamFPrime.Lifecycle.RandomOracle.Pieces

/-! Vectors of the hashed encodings that Rust computes outside the step packages
(`SECURITY_MODEL.md`, section Hashed encodings). The value is
`[schema, circuitKeys, foldTranscripts, compression, applications, defaultSlot]`.

`circuitKeys` holds `[identity, digest 0, digest 1]` of
`Lifecycle.BoundedDepth.CircuitKeyDigest.digest` for each identity of `identities`. Rust must
give the same words (`crates/nightstream/tests/bounded_depth_lifecycle/encodings.rs`).

`foldTranscripts` holds the light and the refresh fold transcript on sample data
(`foldTranscript`): the pieces of `Lifecycle.RandomOracle.Schedule.pieces`, the first and the last
round challenge and the sponge state after the output. Rust must absorb the same lengths and reach the same
state (`crates/nightstream-fprime/tests/fold_transcript_lean.rs`).

`compression` holds the encodings of `Export.BoundedDepth.CompressionSponge`: the two domains and
the session label, `[bytes, byteWords bytes]` for byte strings of several lengths, a claim with
its words, a sponge state with its session bytes, and the pieces of a sample run with its
session. Rust must give the same words and bytes (`crates/ns-compress/tests/unit/transcript.rs`,
`crates/nightstream/tests/bounded_depth_lifecycle/encodings.rs`).

`applications` holds `[records, words, identity]` of
`Lifecycle.BoundedDepth.ApplicationIdentity` for each application of `sampleApplications`: the
records as nested lists (`applicationValue`), `Application.words` and `Application.identity`.
Rust must give the same words and the same hash for the same records
(`crates/nightstream-fprime/tests/unit/application_identity_lean.rs`). The last application is
one that the public builder makes, and Rust must give its words before the layouts
(`crates/nightstream/tests/application/identity_lean.rs`).

`defaultSlot` holds `Lifecycle.BoundedDepth.slotDigest` of the default slot at depth 0, the slot
digest that the verifier of the light compression puts in the state hash for slot 1 (SPEC.md,
Algorithm 10). Rust must give the same words
(`crates/nightstream/tests/bounded_depth_lifecycle/encodings.rs`). -/

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

section Compression

open NightstreamFPrime.Export.BoundedDepth.CompressionSponge
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

def bytes (values : List Byte) : Value :=
  .array (values.map fun value => .atom value.val)

def sampleBytes (seed length : Nat) : List Byte :=
  (List.range length).map fun index => Fin.ofNat 256 (seed + 37 * index)

def sampleWords (seed length : Nat) : List F := (List.range length).map (sample seed)

/-- A claim of the shape `(m_in, ℓ, t)`. -/
def sampleClaim (seed : Nat) (shape : Nat × Nat × Nat) : Claim where
  blocks := 61 + seed
  norm := 2177 + seed
  commitment := sampleWords (seed + 1) (18 * 64)
  cells := sampleWords (seed + 2) shape.1
  point := sampleWords (seed + 3) (2 * shape.2.1)
  evalK := sampleWords (seed + 4) (2 * 64)
  evalA := sampleWords (seed + 5) (2 * (shape.2.2 * 64))

def sampleSizes : Sizes := ⟨30, 7, 18, 20⟩

def sampleShape : Compression.Rust.Shape where
  claims := [(64, 3, 1), (128, 3, 2)]
  tables := 1
  spanVars := 3
  witnessVars := 4
  keyBlockVars := 1

def sampleRun : Run where
  fresh := sampleWords 50 sampleSizes.fresh
  ccsRounds := (List.range sampleSizes.rounds).map fun round =>
    sampleWords (51 + round) sampleSizes.roundWords
  ccsOutput := sampleWords 60 sampleSizes.output
  argument :=
    { key := sampleBytes 61 30
      claims := [sampleClaim 62 (64, 3, 1), sampleClaim 70 (128, 3, 2)]
      witness := sampleBytes 80 45
      inverse := sampleBytes 81 14
      rounds := (List.range sampleShape.witnessVars).map fun round => sampleWords (82 + round) 14
      witnessValue := sampleWords 90 2
      inverseValue := sampleWords 91 2
      keyWeight := sampleWords 92 2
      keyRounds := (List.range (Compression.Rust.laneVars + sampleShape.keyBlockVars)).map
        fun round => sampleWords (93 + round) 6
      keyValue := sampleWords 110 2 }

def compression : Value :=
  let piece : Piece → Value
    | .absorb values => .array [.atom 0, words values]
    | .draw count => .array [.atom 1, .atom count]
  let state := sampleWords 120 Poseidon2.width
  let claim := sampleClaim 0 (128, 14, 4)
  .array [
    .array [words ccsDomain, words argumentDomain, bytes sessionLabel],
    .array (([0, 1, 6, 7, 8, 13, 14, 15, 50].map (sampleBytes 5)).map fun values =>
      .array [bytes values, words (byteWords values)]),
    .array [.atom claim.blocks, .atom claim.norm, words claim.commitment, words claim.cells,
      words claim.point, words claim.evalK, words claim.evalA, words claim.words],
    .array [words state, bytes (sessionBytes sessionLabel state)],
    .array [.array ((pieces sampleSizes sampleShape sampleRun).map piece),
      bytes (session sampleSizes sampleShape sampleRun)]]

end Compression

section Application

open NightstreamFPrime.Lifecycle.BoundedDepth.ApplicationIdentity

def atoms (values : List Nat) : Value := .array (values.map .atom)

/-- `[constant, [[index, coefficient], ...]]`. -/
def linearValue (linear : Linear) : Value :=
  .array [.atom linear.constant.val,
    .array (linear.terms.map fun term => atoms [term.1, term.2.val])]

def rowValue (row : Row) : Value :=
  .array [linearValue row.a, linearValue row.b, linearValue row.c]

def indexedValue (row : Nat × Row) : Value := .array [.atom row.1, rowValue row.2]

/-- The fields of a layout in the order of `Layout`. A bit, a product, a zero product and a slot
are their words. -/
def layoutValue (layout : Layout) : Value :=
  .array [
    .array (layout.fusion.map fun group => .array [
      .array (group.sourceRecords.map fun pair => atoms [pair.1, pair.2]),
      .array (group.sources.map indexedValue), .array (group.certificates.map indexedValue),
      rowValue group.row]),
    .array (layout.packing.map fun group => .array [
      .array (group.sources.map indexedValue), .array (group.certificates.map indexedValue),
      linearValue group.guard,
      .array (group.pairs.map fun pair => .array [atoms pair.1.words, atoms pair.2.words])]),
    .atom layout.packedStart, atoms layout.packedSources, atoms layout.removed,
    .array (layout.products.map fun product => atoms product.words),
    .array (layout.zeros.map fun zero => atoms zero.words),
    .array (layout.slots.map fun slot => atoms slot.words)]

/-- The fields of an application in the order of `Application`. -/
def applicationValue (application : Application) : Value :=
  .array [words application.package.toList, .atom application.witnessWords,
    .array (application.rows.map fun row =>
      .array [linearValue row.a, linearValue row.b, linearValue row.c]),
    atoms application.recipeRows,
    .array (application.blocks.map fun block =>
      .array [.atom block.firstLocal, .array (block.lanes.toList.map linearValue)]),
    layoutValue application.light, layoutValue application.refresh]

def linear (constant : Nat) (terms : List (Nat × Nat)) : Linear :=
  ⟨Poseidon2.ofNat constant, terms.map fun term => (term.1, Poseidon2.ofNat term.2)⟩

/-- A linear form of `count` terms. -/
def sampleLinear (seed count : Nat) : Linear :=
  ⟨sample seed 0, (List.range count).map fun index => (seed + 3 * index, sample seed (index + 1))⟩

/-- The row at index `index` of a first piece: its forms have 0, 1 and 2 terms in an order
that changes with the index. -/
def sampleRow (index : Nat) : Row :=
  ⟨sampleLinear index (index % 3), sampleLinear (index + 1) ((index + 1) % 3),
    sampleLinear (index + 2) ((index + 2) % 3)⟩

def sampleIndexed (indices : List Nat) : List (Nat × Row) :=
  indices.map fun index => (index, sampleRow index)

def sampleRecordRow (seed : Nat) : RecordRow :=
  ⟨sampleLinear seed (seed % 3), sampleLinear (seed + 1) ((seed + 1) % 3),
    sampleLinear (seed + 2) ((seed + 2) % 3)⟩

def sampleBlock (seed : Nat) : Block :=
  ⟨seed, ⟨(List.range 16).map fun lane => sampleLinear (seed + lane) (lane % 3), by simp⟩⟩

def sampleBit (seed : Nat) : Bit :=
  match seed % 3 with
  | 0 => .zero
  | 1 => .one
  | _ => .column seed

/-- A layout with `count` entries in each list. Group `group` has lists of `group` and of
`group + 1` entries, so the lists of the first group have 0 and 1 entries. -/
def sampleLayout (seed count : Nat) : Layout where
  fusion := (List.range count).map fun group =>
    ⟨(List.range group).map fun index => (seed + index, seed + 2 * index + 1),
      sampleIndexed ((List.range (group + 1)).map (seed + 5 * group + ·)),
      sampleIndexed ((List.range group).map (seed + 5 * group + 3 + ·)),
      sampleRow (seed + group)⟩
  packing := (List.range count).map fun group =>
    ⟨sampleIndexed ((List.range group).map (seed + 40 + 5 * group + ·)),
      sampleIndexed ((List.range (group + 1)).map (seed + 43 + 5 * group + ·)),
      sampleLinear (seed + group) group,
      (List.range (3 * group)).map fun index =>
        (sampleBit (seed + group + index), sampleBit (seed + index / 3))⟩
  packedStart := 1000 + seed
  packedSources := (List.range count).map (seed + 7 * ·)
  removed := (List.range count).map (seed + 5 * ·)
  products := (List.range count).map fun index =>
    ⟨seed + index, seed + 2 * index, seed + 3 * index, seed + 4 * index⟩
  zeros := (List.range count).map fun index =>
    ⟨seed + index, seed + 2 * index, index % 2 == 0, seed + 3 * index, index % 2 == 1⟩
  slots := (List.range count).map fun index => ⟨seed + 64 * index, 1 + index, index % 3⟩

/-- The largest field value, as a constant and as a coefficient. -/
def largestRow : RecordRow :=
  ⟨linear (fieldModulus - 1) [(7, fieldModulus - 1)], linear 0 [], linear 1 []⟩

/-- The records of the application that `built_application` makes with the public builder
(`crates/nightstream/tests/application/identity_lean.rs`): two witness words, a product, a sum,
an equality and the four rows of the output state. -/
def builtApplication : Application where
  package := ⟨(List.range 4).map fun word => sample 9 word, by simp⟩
  witnessWords := 2
  rows := [
    ⟨linear 0 [(0, 1), (4, 3)], linear 5 [(5, 1)], linear 0 [(10, 1)]⟩,
    ⟨linear 0 [(10, 1), (1, 1)], linear 1 [], linear 0 [(11, 1)]⟩,
    ⟨linear 0 [(3, 1)], linear 1 [], linear 0 [(2, 1)]⟩,
    ⟨linear 0 [(6, 1)], linear 1 [], linear 0 [(11, 1)]⟩,
    ⟨linear 0 [(7, 1)], linear 1 [], linear 0 [(1, 1)]⟩,
    ⟨linear 7 [(2, 1), (8, fieldModulus - 1)], linear 1 [], linear 0 []⟩,
    ⟨linear 9 [(9, fieldModulus - 1)], linear 1 [], linear 0 []⟩]
  recipeRows := [0, 1]
  blocks := []
  light := sampleLayout 0 0
  refresh := sampleLayout 0 0

/-- Applications with each list at 0, at 1 and at more entries, then `builtApplication`. -/
def sampleApplications : List Application :=
  [⟨⟨List.replicate 4 0, by simp⟩, 0, [], [], [], sampleLayout 0 0, sampleLayout 0 0⟩,
    ⟨⟨(List.range 4).map Poseidon2.ofNat, by simp⟩, 1, [sampleRecordRow 3], [0], [sampleBlock 20],
      sampleLayout 1 1, sampleLayout 2 1⟩,
    ⟨⟨(List.range 4).map fun word => Poseidon2.ofNat (fieldModulus - 1 - word), by simp⟩, 5,
      [sampleRecordRow 0, sampleRecordRow 1, largestRow, sampleRecordRow 2], [3, 0, 3],
      [sampleBlock 30, sampleBlock 1200], sampleLayout 3 3, sampleLayout 50 2⟩,
    builtApplication]

def applications : Value :=
  .array (sampleApplications.map fun application =>
    .array [applicationValue application, atoms application.words, words application.identity])

end Application

/-- The digest of the default slot at depth 0 (HyperNova Definition 12). -/
def defaultSlotDigest : Value :=
  words (slotDigest (logicalWidth := keyWidth) (publicFits := keyPublicFits) (defaultSlot, 0))

def parityValue : Value :=
  .array [.atom schema, circuitKeys, foldTranscripts, compression, applications, defaultSlotDigest]

end NightstreamFPrime.Export.EncodingParity
