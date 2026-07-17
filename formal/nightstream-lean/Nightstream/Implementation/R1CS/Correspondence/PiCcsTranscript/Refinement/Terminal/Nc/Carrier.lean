import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistorySumcheckArtifact
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-!
Exact typed-to-assignment carrier for terminal Split-NC round messages.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the fixed fifteen-round index; exact generated SumCheck map selection;
the five coefficient-pair columns of every round; assignment decoding into
the independent semantic `K` carrier; and the single pointwise boundary
required to identify a typed NC certificate with that decoded carrier.

Does not own: proof of `Bound` from Rust/R1CS input allocation; transcript
replay; Poseidon2 execution; SumCheck algebra; FE messages; costs; necessity;
or row removal.

Emits constraints: no.

Authority boundary: generated maps identify physical columns only.
`decodedRound` interprets those columns in the independent semantic carrier.
`Bound` is the explicit cross-representation obligation that the typed
certificate contains exactly those decoded values; generated-row acceptance
does not imply it.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.certificate.profile` | the repaired 270-column/54-lane domain has exactly fifteen rounds | derived | `domain_roundCount` |
| `nifs.pi_ccs.nc_sumcheck.round.r.map` | select generated affine map `r` without fallback indexing | physical ownership | `columnMap` |
| `nifs.pi_ccs.nc_sumcheck.round.r.coefficients` | round `r` owns five adjacent `(c0,c1)` pairs beginning at `1663447 + 10r` | derived structure | `coefficientColumns_eq` |
| `nifs.pi_ccs.nc_sumcheck.round.r.decode` | interpret the five pairs as one exact degree-four semantic polynomial | computed | `decodedRound` |
| `nifs.pi_ccs.nc_sumcheck.certificate.decode` | every typed round equals its assignment-decoded polynomial | explicit remaining bridge | `RoundBound`, `Bound` |
| `nifs.pi_ccs.nc_sumcheck.certificate.raw_order` | the concrete raw list is exactly the ordered lossless encoding of all typed rounds | derived | `rawRounds_eq_typed` |
| `nifs.pi_ccs.nc_sumcheck.certificate.fixed_order` | the domain-indexed raw list equals the stable fifteen-coordinate view | derived | `typedRawRounds_eq_fixed` |
| `nifs.pi_ccs.nc_sumcheck.certificate.raw_decode` | under `Bound`, the concrete raw list is exactly the ordered assignment-decoded list | derived | `rawRounds_eq_decoded` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Carrier

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

def roundCount : Nat := 15

abbrev domain : FlatNcDomain :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain

theorem domain_roundCount :
    Transcript.Nc.roundCount domain = roundCount := by
  simpa only [roundCount, Transcript.Nc.roundCount] using
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain_variableCount

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  PiCcsTranscript.Exact.Schedule.Input publicInput domain

/-- Embed the stable fixed-round index into the domain-indexed certificate. -/
def domainIndex (round : Fin roundCount) :
    Fin (Transcript.Nc.roundCount domain) :=
  Fin.cast domain_roundCount.symm round

/-- Return a domain-indexed coordinate to the stable fixed profile. -/
def fixedIndex (round : Fin (Transcript.Nc.roundCount domain)) :
    Fin roundCount :=
  Fin.cast domain_roundCount round

@[simp] theorem fixedIndex_domainIndex (round : Fin roundCount) :
    fixedIndex (domainIndex round) = round := by
  apply Fin.ext
  rfl

@[simp] theorem domainIndex_fixedIndex
    (round : Fin (Transcript.Nc.roundCount domain)) :
    domainIndex (fixedIndex round) = round := by
  apply Fin.ext
  rfl

/-- Typed semantic polynomial at fixed NC round `round`. -/
def typedRound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin roundCount) :
    Transcript.Nc.RoundMessage :=
  input.carrier.toNcCertificate.rounds (domainIndex round)

private theorem terminalMaps_length :
    FPrimeFullHistorySumcheckArtifact.terminalNcMaps.length =
      roundCount := by
  decide

def mapIndex (round : Fin roundCount) :
    Fin FPrimeFullHistorySumcheckArtifact.terminalNcMaps.length :=
  Fin.cast terminalMaps_length.symm round

/-- Exact generated affine map at one verifier-owned round index. -/
def columnMap (round : Fin roundCount) : SumcheckChainSound.ColumnMap :=
  FPrimeFullHistorySumcheckArtifact.terminalNcMaps.get (mapIndex round)

/-- Canonical isolated degree-four coefficient pairs after the exact call-site
renaming for `round`. -/
def coefficientColumns (round : Fin roundCount) : List (Nat × Nat) :=
  SumcheckRoundArtifact.coefficientColumns.map fun pair =>
    (Relabel.column (columnMap round) pair.1,
      Relabel.column (columnMap round) pair.2)

def coefficientBase (round : Fin roundCount) : Nat :=
  1663447 + 10 * round.val

def expectedCoefficientColumns
    (round : Fin roundCount) : List (Nat × Nat) :=
  let base := coefficientBase round
  [(base, base + 1),
   (base + 2, base + 3),
   (base + 4, base + 5),
   (base + 6, base + 7),
   (base + 8, base + 9)]

/-- Closed exact audit of all fifteen generated maps. This is kernel
evaluation over a finite index, not a caller-supplied map plan. -/
theorem coefficientColumns_eq :
    ∀ round : Fin roundCount,
      coefficientColumns round = expectedCoefficientColumns round := by
  decide

def semanticFieldAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) : F :=
  ⟨assignment column, canonical column⟩

def semanticCoefficientAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (pair : Nat × Nat) : K where
  c0 := semanticFieldAt assignment canonical pair.1
  c1 := semanticFieldAt assignment canonical pair.2

def artifactCoefficients
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (round : Fin roundCount) : List K :=
  (coefficientColumns round).map
    (semanticCoefficientAt assignment canonical)

/-- Exact assignment-decoded degree-four polynomial for one physical round. -/
def decodedRound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (round : Fin roundCount) :
    Transcript.Nc.RoundMessage where
  coefficients := artifactCoefficients assignment canonical round
  coefficients_length := by
    simp [artifactCoefficients, coefficientColumns,
      SumcheckRoundArtifact.coefficientColumns,
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.ncSumcheckDegreeBound]

/-- One indexed typed-to-assignment coefficient equality. -/
def RoundBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (round : Fin roundCount) : Prop :=
  (typedRound input round).coefficients =
    artifactCoefficients assignment canonical round

/-- Complete NC certificate decoding boundary. -/
def Bound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  ∀ round : Fin roundCount,
    RoundBound input assignment canonical round

/-- The pointwise coefficient boundary identifies the complete typed
fixed-width polynomial, including its proof-irrelevant width witness. -/
theorem typedRound_eq_decodedRound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (round : Fin roundCount)
    (bound : RoundBound input assignment canonical round) :
    typedRound input round = decodedRound assignment canonical round := by
  cases typed : typedRound input round with
  | mk coefficients coefficientsLength =>
      simp only [RoundBound, typed] at bound
      unfold decodedRound
      cases bound
      rfl

/-- Domain-indexed typed coordinate and the stable fixed-index view are the
same certificate element. -/
theorem typedRound_fixedIndex
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin (Transcript.Nc.roundCount domain)) :
    typedRound input (fixedIndex round) =
      input.carrier.toNcCertificate.rounds round := by
  unfold typedRound
  rw [domainIndex_fixedIndex]

/-- Exact raw NC transcript list, preserving verifier-owned order and fixed
five-coefficient width. -/
def typedRawRounds
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    List PiCcsTranscript.SumCheck.RoundMessage :=
  List.ofFn fun round : Fin (Transcript.Nc.roundCount domain) =>
    PiCcsTranscript.ExactMessages.encodeFixed
      (input.carrier.toNcCertificate.rounds round)

/-- Stable fixed-profile view of the same ordered typed raw messages. -/
def fixedTypedRawRounds
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    List PiCcsTranscript.SumCheck.RoundMessage :=
  List.ofFn fun round : Fin (Transcript.Nc.roundCount domain) =>
    PiCcsTranscript.ExactMessages.encodeFixed
      (typedRound input (fixedIndex round))

/-- Domain-indexed and stable fixed-profile views preserve exactly the same
ordered typed certificate. -/
theorem typedRawRounds_eq_fixed
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    typedRawRounds input = fixedTypedRawRounds input := by
  unfold typedRawRounds fixedTypedRawRounds
  congr 1

/-- The raw boundary projection is exactly the canonical ordered encoding of
the typed carrier. -/
theorem rawRounds_eq_typed
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    (PiCcsTranscript.Exact.Schedule.rawMessages input).ncRounds =
      typedRawRounds input := by
  unfold PiCcsTranscript.Exact.Schedule.rawMessages
  rw [PiCcsTranscript.Exact.encode_ncRounds_values]
  unfold typedRawRounds Exact.Carrier.toNcCertificate
  simpa only [List.map_ofFn] using
    congrArg
      (List.map PiCcsTranscript.ExactMessages.encodeFixed)
      (Exact.ExactRoundProjection.ofFn_toFunction
        input.carrier.ncRounds).symm

/-- Canonical ordered raw list decoded directly from one assignment. -/
def decodedRawRounds
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    List PiCcsTranscript.SumCheck.RoundMessage :=
  List.ofFn fun round : Fin (Transcript.Nc.roundCount domain) =>
    PiCcsTranscript.ExactMessages.encodeFixed
      (decodedRound assignment canonical (fixedIndex round))

/-- Complete typed-to-assignment carrier binding identifies the entire raw
NC transcript list, not merely one selected final round. -/
theorem rawRounds_eq_decoded
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (bound : Bound input assignment canonical) :
    (PiCcsTranscript.Exact.Schedule.rawMessages input).ncRounds =
      decodedRawRounds assignment canonical := by
  rw [rawRounds_eq_typed input]
  unfold typedRawRounds decodedRawRounds
  congr 1
  funext round
  rw [← typedRound_fixedIndex input round]
  rw [typedRound_eq_decodedRound
    input canonical (fixedIndex round) (bound (fixedIndex round))]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Carrier
