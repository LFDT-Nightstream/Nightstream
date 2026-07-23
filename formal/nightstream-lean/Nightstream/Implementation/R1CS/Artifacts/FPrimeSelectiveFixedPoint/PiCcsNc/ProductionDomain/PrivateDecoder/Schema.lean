/-!
Proof-free schema for the complete steady-recursive private source decoder.

Assurance tier: artifact data only.

Owns: the compact grammar exchanged by the Rust fixed-point exporter and the
Lean checker. Ordinary atoms describe affine source-disposition runs; SIS
batches describe the exact repeated balanced-ternary allocation geometry.

Does not own: source values, eliminated-definition semantics, derived-product
values, sparse matrix equality, CCS/CE membership, commitment binding, or
permission to remove constraints.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

structure RawCensus where
  eliminated : Nat
  unit : Nat
  balanced : Nat
  binary : Nat
  decompositionAliases : Nat
  equalityAliases : Nat
  equalityAliasSavings : Nat
  retainedCoordinatesBeforeAliases : Nat
  centeredColumns : Nat
deriving DecidableEq, Repr

structure RawSummary where
  sourceColumns : Nat
  freshCoordinates : Nat
  census : RawCensus
deriving DecidableEq, Repr

inductive RawAtom where
  | direct
      (length startStride width : Nat)
      (centered : Bool)
  | decompositionAlias
      (length sourceDelta sourceStride digit digitStride startStride : Nat)
      (centered : Bool)
  | equalityAlias
      (length sourceDelta sourceStride startStride width : Nat)
      (centered : Bool)
  | linearDefinition (length : Nat)
  | traceEliminated (length : Nat)
  | sisBatch (batch : Nat)
deriving DecidableEq, Repr

structure RawTemplate where
  atoms : List RawAtom
  summary : RawSummary
deriving DecidableEq, Repr

structure RawCall where
  template : Nat
  sourceStart : Nat
  finalStart : Nat
deriving DecidableEq, Repr

inductive RawOpeningKind where
  | alias (source sourceStride : Nat)
  | direct
deriving DecidableEq, Repr

structure RawOpeningGroup where
  openingStart : Nat
  length : Nat
  directBefore : Nat
  kind : RawOpeningKind
deriving DecidableEq, Repr

structure RawBatch where
  sourceStart : Nat
  sourceEnd : Nat
  inputBinding : Bool
  commitmentFields : Nat
  openings : Nat
  directOpenings : Nat
  groupShardStart : Nat
  groupShardCount : Nat
deriving DecidableEq, Repr

inductive RawOwner where
  | ordinary (call atom offset : Nat)
  | batch (call atom batch offset : Nat)
deriving DecidableEq, Repr

inductive RawConsumer where
  | ordinary (call atom offset : Nat)
  | batch (call atom batch group offset : Nat)
deriving DecidableEq, Repr

structure RawAliasLink where
  consumer : RawConsumer
  length : Nat
  target : RawOwner
  targetOffsetStride : Nat
deriving DecidableEq, Repr

structure RawAliasConsumer where
  consumer : RawConsumer
  length : Nat
  linkStart : Nat
  linkStop : Nat
deriving DecidableEq, Repr

structure RawTemplateChunkContext where
  templateStart : Nat
  templateStop : Nat
  atomCount : Nat
deriving DecidableEq, Repr

structure RawCallChunkContext where
  callStart : Nat
  callStop : Nat
  sourceStart : Nat
  sourceStop : Nat
  finalStart : Nat
  finalStop : Nat
deriving DecidableEq, Repr

structure RawOpeningGroupShardContext where
  batch : Nat
  groupStart : Nat
  groupStop : Nat
  openingStart : Nat
  openingStop : Nat
  directStart : Nat
  directStop : Nat
deriving DecidableEq, Repr

structure RawAliasLinkChunkContext where
  linkStart : Nat
  linkStop : Nat
deriving DecidableEq, Repr

structure RawAliasConsumerChunkContext where
  consumerStart : Nat
  consumerStop : Nat
  linkStart : Nat
  linkStop : Nat
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder
