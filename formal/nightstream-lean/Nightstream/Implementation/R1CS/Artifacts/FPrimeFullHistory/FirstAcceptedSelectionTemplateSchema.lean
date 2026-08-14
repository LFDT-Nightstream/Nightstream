import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.AggregateExactness

/-!
Wire schema for the compact production first-accepted selection schedule.

Owns: eight fixed sampler records, their exact expansion into 432 row and
column schedules, proof-free coverage checks, and the connection from each
expanded schedule to the model-level product substitution.

Does not own: Rust source-row validation, final low-norm gate semantics,
one-hotness, complete PiRLC semantics, or row-removal authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.FirstAcceptedSelection

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection

def candidateCount : Nat := 11
def outputCount : Nat := 54
def sourceChunkCount : Nat := 64
def sourceRowsPerBlock : Nat := 36
def emittedRowsPerBlock : Nat := 9
def sourceRowStride : Nat := 48
def emittedRowStride : Nat := 9
def stageStride : Nat := 2
def selectorColumnStride : Nat := 45

structure Sampler where
  arm : Nat
  firstRewrite : Nat
  firstStage : Nat
  firstSourceRow : Nat
  firstEmittedRow : Nat
  firstSelectorColumn : Nat
  accepts : List Nat
  prefixes : List Nat
  symbols : List Nat
  deriving DecidableEq, Repr

structure Occurrence where
  arm : Nat
  rewrite : Nat
  stage : Nat
  position : Nat
  sourceStart : Nat
  sourceStop : Nat
  emittedStart : Nat
  emittedStop : Nat
  selectors : List Nat
  accepts : List Nat
  prefixes : List Nat
  symbols : List Nat
  acceptedProducts : List Nat
  prefixProducts : List Nat
  symbolProducts : List Nat
  output : Nat
  deriving DecidableEq, Repr

structure RawCoverage where
  schemaVersion : Nat
  relationRows : Nat
  relationColumns : Nat
  sourceRows : Nat
  sourceColumns : Nat
  blockCount : Nat
  sourceBlockRows : Nat
  emittedBlockRows : Nat
  samplers : List Sampler
  deriving DecidableEq, Repr

def rangeFrom (start count : Nat) : List Nat :=
  (List.range count).map (fun offset => start + offset)

def stridedFrom (start stride count : Nat) : List Nat :=
  (List.range count).map (fun offset => start + stride * offset)

def window (values : List Nat) (position : Nat) : List Nat :=
  (values.drop position).take candidateCount

def Sampler.occurrence (sampler : Sampler) (position : Nat) : Occurrence :=
  let selectorStart := sampler.firstSelectorColumn + selectorColumnStride * position
  { arm := sampler.arm
    rewrite := sampler.firstRewrite + position
    stage := sampler.firstStage + stageStride * position
    position := position
    sourceStart := sampler.firstSourceRow + sourceRowStride * position
    sourceStop := sampler.firstSourceRow + sourceRowStride * position + sourceRowsPerBlock
    emittedStart := sampler.firstEmittedRow + emittedRowStride * position
    emittedStop := sampler.firstEmittedRow + emittedRowStride * position + emittedRowsPerBlock
    selectors := rangeFrom selectorStart candidateCount
    accepts := window sampler.accepts position
    prefixes := window sampler.prefixes position
    symbols := window sampler.symbols position
    acceptedProducts := stridedFrom (selectorStart + 12) 3 candidateCount
    prefixProducts := stridedFrom (selectorStart + 13) 3 candidateCount
    symbolProducts := stridedFrom (selectorStart + 11) 3 candidateCount
    output := selectorStart + 44 }

def Sampler.occurrences (sampler : Sampler) : List Occurrence :=
  (List.range outputCount).map sampler.occurrence

def RawCoverage.occurrences (raw : RawCoverage) : List Occurrence :=
  raw.samplers.flatMap Sampler.occurrences

theorem Sampler.occurrences_length (sampler : Sampler) :
    sampler.occurrences.length = outputCount := by
  simp [Sampler.occurrences]

private theorem samplerOccurrences_length (samplers : List Sampler) :
    (samplers.flatMap Sampler.occurrences).length =
      samplers.length * outputCount := by
  induction samplers with
  | nil => simp
  | cons sampler samplers inductionHypothesis =>
      simp [Sampler.occurrences_length, inductionHypothesis, Nat.add_mul,
        Nat.add_comm]

theorem RawCoverage.occurrences_length (raw : RawCoverage) :
    raw.occurrences.length = raw.samplers.length * outputCount := by
  exact samplerOccurrences_length raw.samplers

def Sampler.valid (sampler : Sampler) : Bool :=
  sampler.arm == 1 &&
    sampler.accepts.length == sourceChunkCount &&
    sampler.prefixes.length == sourceChunkCount &&
    sampler.symbols.length == sourceChunkCount

def Occurrence.valid (raw : RawCoverage) (occurrence : Occurrence) : Bool :=
  occurrence.position < outputCount &&
    occurrence.sourceStop - occurrence.sourceStart == sourceRowsPerBlock &&
    occurrence.emittedStop - occurrence.emittedStart == emittedRowsPerBlock &&
    occurrence.sourceStop <= raw.sourceRows &&
    occurrence.emittedStop <= raw.relationRows &&
    occurrence.selectors.length == candidateCount &&
    occurrence.accepts.length == candidateCount &&
    occurrence.prefixes.length == candidateCount &&
    occurrence.symbols.length == candidateCount &&
    occurrence.acceptedProducts.length == candidateCount &&
    occurrence.prefixProducts.length == candidateCount &&
    occurrence.symbolProducts.length == candidateCount &&
    ([occurrence.output] ++ occurrence.selectors ++ occurrence.accepts ++
      occurrence.prefixes ++ occurrence.symbols ++ occurrence.acceptedProducts ++
      occurrence.prefixProducts ++ occurrence.symbolProducts).all
        (fun column => column < raw.sourceColumns)

def orderedBy
    (finish start : Occurrence → Nat) : List Occurrence → Bool
  | [] => true
  | [_] => true
  | current :: next :: rest =>
      finish current <= start next && orderedBy finish start (next :: rest)

def CoverageValid (raw : RawCoverage) : Prop :=
  let occurrences := raw.occurrences
  raw.schemaVersion = 1 ∧
  raw.samplers.length = 8 ∧
  raw.samplers.all Sampler.valid = true ∧
  raw.blockCount = raw.samplers.length * outputCount ∧
  occurrences.length = raw.blockCount ∧
  raw.sourceBlockRows = raw.blockCount * sourceRowsPerBlock ∧
  raw.emittedBlockRows = raw.blockCount * emittedRowsPerBlock ∧
  occurrences.all (Occurrence.valid raw) = true ∧
  orderedBy (fun occurrence => occurrence.sourceStop)
    (fun occurrence => occurrence.sourceStart) occurrences = true ∧
  orderedBy (fun occurrence => occurrence.emittedStop)
    (fun occurrence => occurrence.emittedStart) occurrences = true

instance coverageValidDecidable (raw : RawCoverage) : Decidable (CoverageValid raw) := by
  unfold CoverageValid
  infer_instance

def selectorValues {value : Type} (assignment : Nat → value)
    (sampler : Sampler) (position : Nat) : Fin candidateCount → value :=
  fun candidate =>
    assignment (sampler.firstSelectorColumn + selectorColumnStride * position + candidate.val)

def windowValues {value : Type} (assignment : Nat → value)
    (columns : List Nat) (position : Nat) : Fin candidateCount → value :=
  fun candidate => assignment (columns.getD (position + candidate.val) 0)

def CurrentAt {value : Type}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (assignment : Nat → value) (positionValue : Nat → value)
    (sampler : Sampler) (position : Nat) : Prop :=
  CurrentSelectionBlock
    (selectorValues assignment sampler position)
    (windowValues assignment sampler.accepts position)
    (windowValues assignment sampler.prefixes position)
    (windowValues assignment sampler.symbols position)
    (positionValue position)
    (assignment (sampler.firstSelectorColumn + selectorColumnStride * position + 44))

def AggregateAt {value : Type}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (assignment : Nat → value) (positionValue : Nat → value)
    (sampler : Sampler) (position : Nat) : Prop :=
  AggregateSelectionBlock
    (selectorValues assignment sampler position)
    (windowValues assignment sampler.accepts position)
    (windowValues assignment sampler.prefixes position)
    (windowValues assignment sampler.symbols position)
    (positionValue position)
    (assignment (sampler.firstSelectorColumn + selectorColumnStride * position + 44))

theorem currentAt_iff_aggregateAt {value : Type}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (assignment : Nat → value) (positionValue : Nat → value)
    (sampler : Sampler) (position : Nat) :
    CurrentAt assignment positionValue sampler position ↔
      AggregateAt assignment positionValue sampler position := by
  unfold CurrentAt AggregateAt
  exact currentSelectionBlock_iff_aggregate _ _ _ _ _ _

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.FirstAcceptedSelection
