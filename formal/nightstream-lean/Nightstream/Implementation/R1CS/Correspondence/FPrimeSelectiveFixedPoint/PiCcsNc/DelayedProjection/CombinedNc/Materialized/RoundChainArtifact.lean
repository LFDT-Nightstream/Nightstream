import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ClaimedChain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact

/-!
Exact generated-map bridge for the production combined-NC claimed chain.

Owns: the bounded certificate that the 25 generated five-coefficient round
maps form one literal claim-forwarding chain, and the kernel composition from
their mapped 30-row equations to the complete claimed-chain predicate.

Does not own: source-to-selective refinement, initial or terminal formula
rows, transcript derivation, semantic SumCheck soundness, raw-child authority,
state continuity, commitment binding, costs, or row removal.

Emits constraints: none.  The executable certificate consumes exactly 25
proof-free `RawRoundMap` records.  Each record contains 43 mapped columns, 28
allocated columns, and five coefficient-column pairs; no decoded or
proof-carrying structure is evaluated.

Assurance tier: artifact-checked for generated round-map shape and forwarding;
row truth remains a premise until the selective compiler bridge is composed.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundChainArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-- Exact bounded shape checked on the generated map list itself. -/
def GeneratedChainShape : Prop :=
  RoundMaps.values.length = sumcheckRoundCount ∧
    ClaimedChain.Linked RoundMaps.values

private def linkedDecidable :
    (rounds : List RawRoundMap) → Decidable (ClaimedChain.Linked rounds)
  | [] => isTrue trivial
  | [_] => isTrue trivial
  | left :: right :: rounds => by
      letI := linkedDecidable (right :: rounds)
      unfold ClaimedChain.Linked ClaimedChain.Link
      infer_instance

instance (rounds : List RawRoundMap) :
    Decidable (ClaimedChain.Linked rounds) :=
  linkedDecidable rounds

instance : Decidable GeneratedChainShape := by
  unfold GeneratedChainShape
  infer_instance

set_option maxRecDepth 100000 in
theorem generatedChainShape : GeneratedChainShape := by
  native_decide

/-- Satisfaction of the independently defined 30-row quartic program under
every generated affine column map. -/
def GeneratedRoundRowsSatisfy (assignment : Nat → Nat) : Prop :=
  ∀ round ∈ RoundMaps.values,
    Satisfies
      (ProductionRound.rows.map (Relabel.row round.columnMap)) assignment

theorem roundsAccepted_of_generated_rows
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rowsSatisfy : GeneratedRoundRowsSatisfy assignment) :
    ClaimedChain.RoundsAccepted RoundMaps.values assignment := by
  intro round member
  have valid := RoundArtifact.generatedRoundMapsValid.2.2 round member
  have mapsOne : Relabel.column round.columnMap 0 = 0 := by
    simpa [Relabel.column] using valid.2.2.2.1
  exact ProductionRound.mapped_sound round.columnMap mapsOne canonical one
    (rowsSatisfy round member)

/-- The exact generated mapped rows imply one verifier-visible claimed chain.
The first and last values are still physical boundary-column reads; separate
source-row leaves must identify them with the combined-NC initial and terminal
formulas. -/
theorem claimedChain_of_generated_round_rows
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rowsSatisfy : GeneratedRoundRowsSatisfy assignment) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Chain
      ClaimedChain.ops
      (ClaimedChain.initial RoundMaps.values assignment)
      (ClaimedChain.certificate RoundMaps.values assignment).rounds
      (ClaimedChain.challenges RoundMaps.values assignment)
      (ClaimedChain.terminal RoundMaps.values assignment) := by
  exact ClaimedChain.accepted RoundMaps.values generatedChainShape.2
    (roundsAccepted_of_generated_rows canonical one rowsSatisfy)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundChainArtifact
