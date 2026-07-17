import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.FeRefinement
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistorySumcheckArtifact
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-!
Explicit wire-format boundary between the minimal typed FE language and the
legacy terminal degree-four artifact.

Assurance tier: implementation/R1CS gap certificate.

Owns: the six fixed lane coordinates; the five coefficient-pair slots exposed
by every legacy FE affine map; the three coefficients owned by every typed
semantic lane message; and a kernel-checked proof that these formats are not
losslessly identical.

Does not own: a repaired Rust encoding, transcript equivalence between the two
formats, zero authority for the two legacy high slots, SumCheck soundness,
cost savings, or row removal.

Emits constraints: no.

Authority boundary: the independent lane polynomial has degree two. The old
artifact merely allows degree four; its two additional witness coefficients
cannot be treated as computed zeros without an enforcing theorem or rows.
Consequently, this module deliberately blocks a false direct refinement.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe_sumcheck.minimal.lane.width` | every semantic lane message has exactly three extension coefficients | checked by type | `semanticLaneCoefficientCount` |
| `nifs.pi_ccs.fe_sumcheck.legacy.lane.width` | every legacy lane coordinate exposes five extension coefficient pairs | artifact structure | `legacyLaneCoefficientCount` |
| `nifs.pi_ccs.fe_sumcheck.refinement.width_gap` | three-slot semantic messages cannot equal five-slot legacy messages losslessly | counterexample boundary | `laneWireWidthMismatch` |
| `nifs.pi_ccs.fe_sumcheck.optimization.candidate` | six lane rounds carry twelve excess extension slots in the legacy profile | derived diagnostic | `legacyExcessSlots` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.WireFormat

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  PiCcsTranscript.Exact.Schedule.Input publicInput
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain

def legacyRoundCount : Nat := 7
def laneRoundCount : Nat := 6
def semanticLaneCoefficientWidth : Nat := 3
def legacyLaneCoefficientWidth : Nat := 5

private theorem terminalMaps_length :
    FPrimeFullHistorySumcheckArtifact.terminalFeMaps.length =
      legacyRoundCount := by
  decide

theorem domain_laneVariables :
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain.laneVariables =
      laneRoundCount := by
  rfl

def domainLaneIndex (lane : Fin laneRoundCount) :
    Fin Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain.domain.laneVariables :=
  Fin.cast domain_laneVariables.symm lane

/-- Lane coordinate zero is FE round one; the sole row coordinate is round
zero in this fixed seven-round legacy profile. -/
def legacyLaneRoundIndex (lane : Fin laneRoundCount) :
    Fin legacyRoundCount :=
  ⟨lane.val + 1, by
    have laneLt := lane.isLt
    simp only [laneRoundCount, legacyRoundCount] at laneLt ⊢
    omega⟩

def mapIndex (round : Fin legacyRoundCount) :
    Fin FPrimeFullHistorySumcheckArtifact.terminalFeMaps.length :=
  Fin.cast terminalMaps_length.symm round

def columnMap (round : Fin legacyRoundCount) :
    SumcheckChainSound.ColumnMap :=
  FPrimeFullHistorySumcheckArtifact.terminalFeMaps.get (mapIndex round)

def legacyCoefficientColumns
    (lane : Fin laneRoundCount) : List (Nat × Nat) :=
  SumcheckRoundArtifact.coefficientColumns.map fun pair =>
    (Relabel.column (columnMap (legacyLaneRoundIndex lane)) pair.1,
      Relabel.column (columnMap (legacyLaneRoundIndex lane)) pair.2)

def semanticLaneRound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (lane : Fin laneRoundCount) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.LaneMessage :=
  input.carrier.toFeCertificate.laneRounds (domainLaneIndex lane)

theorem semanticLaneCoefficientCount
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (lane : Fin laneRoundCount) :
    (semanticLaneRound input lane).coefficients.length =
      semanticLaneCoefficientWidth := by
  exact (semanticLaneRound input lane).coefficients_length

theorem legacyLaneCoefficientCount
    (lane : Fin laneRoundCount) :
    (legacyCoefficientColumns lane).length =
      legacyLaneCoefficientWidth := by
  simp [legacyCoefficientColumns, legacyLaneCoefficientWidth,
    SumcheckRoundArtifact.coefficientColumns]

/-- A typed lane message cannot be identified coefficient-for-coefficient
with the legacy artifact message: their statically proved lengths differ. -/
theorem laneWireWidthMismatch
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (lane : Fin laneRoundCount) :
    (semanticLaneRound input lane).coefficients.length ≠
      (legacyCoefficientColumns lane).length := by
  rw [semanticLaneCoefficientCount, legacyLaneCoefficientCount]
  decide

/-- The legacy terminal profile exposes two excess extension slots in each
of six lane rounds: twelve extension slots, or twenty-four base-field limbs,
before accounting for the resulting transcript permutation schedule. -/
theorem legacyExcessSlots :
    laneRoundCount *
          (legacyLaneCoefficientWidth - semanticLaneCoefficientWidth) = 12 /\
      2 * laneRoundCount *
          (legacyLaneCoefficientWidth - semanticLaneCoefficientWidth) = 24 := by
  decide

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.WireFormat
