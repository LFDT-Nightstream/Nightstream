import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane

/-!
Verifier-owned transcript replay for canonical block×lane NC SumCheck.

Assurance tier: model-level.

Owns: an exact block-plus-lane certificate, one NC phase entry, physical
block-then-lane message replay, direct state threading across the block/lane
cut, construction of the typed challenge point, and the adapter to the shared
five-slot claimed-chain checker.

Does not own: concrete transcript encoding, Poseidon2, pre-SumCheck coin
derivation, polynomial truth, packed-output authority, Fiat--Shamir security,
Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate contains only exactly counted five-slot
messages. Challenges and transcript states are computed by the verifier. NC
is entered once; the lane suffix begins from the block prefix's returned
state, with no tag, reset, or separately supplied boundary value. The legacy
flat 15-coordinate adapter is not imported as semantic or arity authority;
only its domain-independent replay machine is reused.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.transcript.certificate` | exactly one five-slot message per block or lane coordinate | checked by type | `Certificate` |
| `nifs.pi_ccs.nc.block_lane.transcript.enter` | enter NC exactly once before the block prefix | verifier transcript | `derive` |
| `nifs.pi_ccs.nc.block_lane.transcript.round` | absorb each message before squeezing its challenge | verifier transcript | `Nc.runRound`, `Nc.runRoundsFrom` |
| `nifs.pi_ccs.nc.block_lane.transcript.phase_cut` | lane replay starts from the block replay's exact final state | direct dataflow | `replay_eq_block_then_lane` |
| `nifs.pi_ccs.nc.block_lane.transcript.point.block` | block challenges are the authoritative prefix | computed | `derive` |
| `nifs.pi_ccs.nc.block_lane.transcript.point.lane` | lane challenges are the authoritative suffix | computed | `derive` |
| `nifs.pi_ccs.nc.block_lane.transcript.chain` | claimed-chain acceptance consumes only derived challenges | checked | `Accepted`, `check` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

/-- Symbolic round count of the canonical product domain. -/
abbrev roundCount (domain : BlockNcDomain) : Nat :=
  domain.blockVariables + domain.laneVariables

/-- Reuse the independently degree-justified five-slot NC message. -/
abbrev RoundMessage :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.RoundMessage

/-- Prover-visible message-only certificate with structural product arity. -/
structure Certificate (domain : BlockNcDomain) where
  rounds : Fin (roundCount domain) → RoundMessage

namespace Certificate

/-- Canonical physical message order: all block rounds, then all lane rounds. -/
def rawRounds
    {domain : BlockNcDomain}
    (certificate : Certificate domain) : List RoundMessage :=
  List.ofFn certificate.rounds

/-- Block-prefix view of the one physical message list. -/
def blockRounds
    {domain : BlockNcDomain}
    (certificate : Certificate domain) : List RoundMessage :=
  certificate.rawRounds.take domain.blockVariables

/-- Lane-suffix view of the one physical message list. -/
def laneRounds
    {domain : BlockNcDomain}
    (certificate : Certificate domain) : List RoundMessage :=
  certificate.rawRounds.drop domain.blockVariables

/-- The two views partition the physical certificate without reordering. -/
theorem blockRounds_append_laneRounds
    {domain : BlockNcDomain}
    (certificate : Certificate domain) :
    certificate.blockRounds ++ certificate.laneRounds =
      certificate.rawRounds := by
  exact List.take_append_drop domain.blockVariables certificate.rawRounds

/-- Projection into the shared physical checker preserves every message. -/
def toSumCheck
    {domain : BlockNcDomain}
    (certificate : Certificate domain) :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Certificate
    where
  rounds := certificate.rawRounds

/-- The physical projection has exactly the typed product arity. -/
@[simp] theorem rawRounds_length
    {domain : BlockNcDomain}
    (certificate : Certificate domain) :
    certificate.rawRounds.length = roundCount domain := by
  simp [rawRounds]

/-- The shared checker sees the same exact product arity. -/
@[simp] theorem toSumCheck_rounds_length
    {domain : BlockNcDomain}
    (certificate : Certificate domain) :
    certificate.toSumCheck.rounds.length = roundCount domain := by
  exact certificate.rawRounds_length

end Certificate

/-- A single replay can be viewed as a block prefix followed directly by a
lane suffix. No second `enterNc` occurs at the cut. -/
theorem replay_eq_block_then_lane
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds =
      let blockResult := Nc.runRoundsFrom machine
        (machine.enterNc initialState) certificate.blockRounds
      let laneResult := Nc.runRoundsFrom machine blockResult.2
        certificate.laneRounds
      (blockResult.1 ++ laneResult.1, laneResult.2) := by
  rw [← certificate.blockRounds_append_laneRounds]
  exact Nc.runRoundsFrom_append machine (machine.enterNc initialState)
    certificate.blockRounds certificate.laneRounds

/-- Typed transcript-derived block/lane challenge point and successor state. -/
structure Derived (domain : BlockNcDomain) (State : Type uState) where
  challengePoint : Point domain
  finalState : State

/-- Enter NC once and replay every exact-count message in canonical order. -/
def derive
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) : Derived domain State :=
  let result := Nc.runRoundsFrom machine (machine.enterNc initialState)
    certificate.rawRounds
  {
    challengePoint := {
      block := {
        coordinates := result.1.take domain.blockVariables
        dimension := by
          rw [List.length_take, Nc.runRoundsFrom_challenges_length,
            Certificate.rawRounds_length]
          simp [roundCount]
      }
      lane := {
        coordinates := result.1.drop domain.blockVariables
        dimension := by
          rw [List.length_drop, Nc.runRoundsFrom_challenges_length,
            Certificate.rawRounds_length]
          simp [roundCount]
      }
    }
    finalState := result.2
  }

/-- Typed point serialization recovers the exact flat transcript challenge
order. -/
theorem derive_point_coordinates
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    (derive machine initialState certificate).challengePoint.coordinates =
      (Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds).1 := by
  unfold derive Point.coordinates
  exact List.take_append_drop domain.blockVariables _

/-- The typed point and successor state are jointly the two projections of
the same one-entry replay. -/
theorem derive_coordinates_finalState
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    ((derive machine initialState certificate).challengePoint.coordinates,
        (derive machine initialState certificate).finalState) =
      Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds := by
  apply Prod.ext
  · exact derive_point_coordinates machine initialState certificate
  · rfl

/-- Logical claimed-chain acceptance under only transcript-derived
challenges. -/
def Accepted
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (initial terminal : K)
    (certificate : Certificate domain) : Prop :=
  let derived := derive machine initialState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Accepted
    initial derived.challengePoint.coordinates terminal certificate.toSumCheck

/-- Executable claimed-chain checker under the same derived point. -/
def check
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (initial terminal : K)
    (certificate : Certificate domain) : Bool :=
  let derived := derive machine initialState certificate
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check
    initial derived.challengePoint.coordinates terminal certificate.toSumCheck

/-- Executable and logical replay acceptance coincide. -/
theorem check_eq_true_iff_accepted
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (initial terminal : K)
    (certificate : Certificate domain) :
    check machine initialState initial terminal certificate = true ↔
      Accepted machine initialState initial terminal certificate := by
  simp only [check, Accepted]
  exact
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.check_eq_true_iff_accepted
      initial (derive machine initialState certificate).challengePoint.coordinates
      terminal certificate.toSumCheck

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane
