/-!
Wire schema for a compact source-stage coverage census.

Owns: proof-free owner records, exclusive source-field dispositions, and the
executable checks that their totals agree.

Does not own: compiler path authority, source-to-final decoder semantics,
arithmetic-family identity, relation soundness, or permission to remove rows
or columns.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SourceStageCoverage

inductive Owner where
  | application
  | prelude
  | transcript
  | piCcs
  | runningParentPiDec
  | piRlc
  | piDec
  | pointBinding
  | priorLink
  | nebula
  | accumulator
  | counters
  | output
  | semanticLinks
  deriving DecidableEq, Inhabited, Repr

structure OwnerCensus where
  owner : Owner
  stages : Nat
  sourceFields : Nat
  direct : Nat
  decompositionAlias : Nat
  equalityAlias : Nat
  linearDefinition : Nat
  traceEliminated : Nat
  allocatedCoordinates : Nat
  deriving DecidableEq, Inhabited, Repr

structure RawCoverage where
  schemaVersion : Nat
  physicalStages : Nat
  unownedEmptyStages : Nat
  sourceFields : Nat
  direct : Nat
  decompositionAlias : Nat
  equalityAlias : Nat
  linearDefinition : Nat
  traceEliminated : Nat
  allocatedCoordinates : Nat
  owners : List OwnerCensus
  deriving DecidableEq, Repr

def expectedOwners : List Owner :=
  [ .application
  , .prelude
  , .transcript
  , .piCcs
  , .runningParentPiDec
  , .piRlc
  , .piDec
  , .pointBinding
  , .priorLink
  , .nebula
  , .accumulator
  , .counters
  , .output
  , .semanticLinks
  ]

def OwnerCensus.dispositionFields (census : OwnerCensus) : Nat :=
  census.direct + census.decompositionAlias + census.equalityAlias +
    census.linearDefinition + census.traceEliminated

def OwnerCensus.valid (census : OwnerCensus) : Bool :=
  census.sourceFields == census.dispositionFields

def sumBy (select : OwnerCensus → Nat) : List OwnerCensus → Nat
  | [] => 0
  | census :: rest => select census + sumBy select rest

def RawCoverage.dispositionFields (raw : RawCoverage) : Nat :=
  raw.direct + raw.decompositionAlias + raw.equalityAlias +
    raw.linearDefinition + raw.traceEliminated

def CoverageValid (raw : RawCoverage) : Prop :=
  raw.schemaVersion = 1 ∧
  raw.owners.map (fun census => census.owner) = expectedOwners ∧
  raw.owners.all OwnerCensus.valid = true ∧
  sumBy (fun census => census.stages) raw.owners + raw.unownedEmptyStages =
    raw.physicalStages ∧
  sumBy (fun census => census.sourceFields) raw.owners = raw.sourceFields ∧
  sumBy (fun census => census.direct) raw.owners = raw.direct ∧
  sumBy (fun census => census.decompositionAlias) raw.owners =
    raw.decompositionAlias ∧
  sumBy (fun census => census.equalityAlias) raw.owners = raw.equalityAlias ∧
  sumBy (fun census => census.linearDefinition) raw.owners =
    raw.linearDefinition ∧
  sumBy (fun census => census.traceEliminated) raw.owners =
    raw.traceEliminated ∧
  sumBy (fun census => census.allocatedCoordinates) raw.owners =
    raw.allocatedCoordinates ∧
  raw.sourceFields = raw.dispositionFields

instance coverageValidDecidable (raw : RawCoverage) : Decidable (CoverageValid raw) := by
  unfold CoverageValid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SourceStageCoverage
