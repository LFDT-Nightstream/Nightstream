import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Coins
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Schedule

/-!
Complete exact minimal mixed-width `Pi_CCS` candidate transcript schedule.

Assurance tier: executable implementation refinement.

Owns: one exact input surface from the outer transcript state through binding,
pre-SumCheck coins, exact FE-to-NC messages, and catch-up; the verifier-derived
global degree bound; derivation of the legacy loose shape predicate from the
exact carrier; and equality with the existing complete schedule.

Does not own: authority of binding fields, polynomial truth, honest message
construction, Fiat--Shamir probability, native/gadget/R1CS refinement, costs,
or row removal.

Emits constraints: no.

Authority boundary: the caller supplies an `Exact.Carrier`, not raw message
lists or a `WellShaped` proof. `scheduleInput` is a lossless projection into
the existing complete transcript machine. Its challenges, phase states, and
header digest remain outputs of deterministic replay.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.exact.complete.shape.degree` | one verifier bound covers FE row, FE lane, and NC physical degrees | computed | `degreeBound` |
| `nifs.pi_ccs.exact.complete.binding` | challenge replay starts after the exact five-message binding prefix | direct dataflow | `challengeOutput` |
| `nifs.pi_ccs.exact.complete.coins` | semantic FE/NC coins project from that same challenge execution | computed | `feCoins`, `ncCoins` |
| `nifs.pi_ccs.exact.complete.sumcheck` | exact FE execution threads directly into exact NC execution | direct dataflow | `exactInput` |
| `nifs.pi_ccs.exact.complete.shape` | exact typed counts and widths imply the legacy loose shape predicate | derived | `scheduleInput_wellShaped` |
| `nifs.pi_ccs.exact.complete.refinement` | complete schedule FE/NC fields equal the exact sub-schedule fields | derived | `run_sumcheck_eq_exact` |
| `nifs.pi_ccs.exact.complete.catchup` | header digest and successor come from the exact NC successor | computed | `run_catchup_joint` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev NcDegree : Nat :=
  Polynomial.Nc.Degree.ncSumcheckDegreeBound

/-- The smallest shared physical degree ceiling already implied by the three
exact message languages: the syntax-derived FE row degree or the fixed NC
degree, whichever is larger. The fixed quadratic FE lane degree is below the
NC degree. -/
def degreeBound
    {shape : SemanticShape}
    (publicInput : PublicInput shape) : Nat :=
  max (SumCheck.Fe.Drow publicInput) NcDegree

/-- Exact full-schedule input. The FE initial claim remains verifier-owned and
is therefore separate from the prover message carrier. -/
structure Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape)
    (domain : FlatNcDomain) where
  initialState : State
  binding : Binding.Input
  expectedFeInitial : K
  carrier : Carrier publicInput domain

/-- State after the complete verifier-authority binding prefix. -/
def afterBinding
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : State :=
  Binding.run input.initialState input.binding

/-- One concrete pre-SumCheck challenge execution at the exact semantic
dimensions and verifier-derived degree ceiling. -/
def challengeOutput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : Challenges.Output :=
  Coins.run (afterBinding input) shape domain (degreeBound publicInput)

/-- Typed FE coins from the same pre-SumCheck execution used by the complete
schedule. -/
def feCoins
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) : Polynomial.Fe.Coins shape domain :=
  Coins.feCoins (afterBinding input) shape domain (degreeBound publicInput)

/-- Typed NC coins from the same pre-SumCheck execution used by the complete
schedule. -/
def ncCoins
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Polynomial.Nc.Mixing.Coins domain :=
  Coins.ncCoins (afterBinding input) shape domain (degreeBound publicInput)

/-- Exact post-coin FE-to-NC input. Its incoming state is the successor of the
same challenge execution that defines `feCoins` and `ncCoins`. -/
def exactInput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Exact.Schedule.Input publicInput domain where
  initialState := (challengeOutput input).state
  expectedFeInitial := input.expectedFeInitial
  carrier := input.carrier

/-- Lossless projection into the pre-existing complete schedule. -/
def scheduleInput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.Input where
  initialState := input.initialState
  shape := Coins.shapeWithDegree shape domain (degreeBound publicInput)
  binding := input.binding
  sumcheck := Exact.Schedule.rawMessages (exactInput input)

/-- Replay the complete exact schedule through the existing deterministic
implementation semantics. -/
def run
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.Trace :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run
    (scheduleInput input)

/-- Exact typed widths imply the older loose `WellShaped` condition. This
direction is deliberate: a caller cannot substitute the loose predicate for
the exact carrier language. -/
theorem scheduleInput_wellShaped
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.WellShaped
      (scheduleInput input) := by
  let messages := Exact.Schedule.rawMessages (exactInput input)
  refine {
    feRoundCount := ?_
    ncRoundCount := ?_
    feDegree := ?_
    ncDegree := ?_
  }
  · change
      (encode input.expectedFeInitial input.carrier).feRounds.length =
        shape.rowVariables + domain.laneVariables
    exact encode_feRounds_length input.expectedFeInitial input.carrier
  · change
      (encode input.expectedFeInitial input.carrier).ncRounds.length =
        domain.columnVariables + domain.laneVariables
    exact encode_ncRounds_length input.expectedFeInitial input.carrier
  · intro round member
    change round ∈ (encode input.expectedFeInitial input.carrier).feRounds
      at member
    rw [encode_feRounds_values] at member
    rcases List.mem_append.mp member with rowMember | laneMember
    · rcases List.mem_map.mp rowMember with
        ⟨typed, _, rfl⟩
      simp only [encodeFixed, List.length_map]
      rw [typed.coefficients_length]
      change
        SumCheck.Fe.Drow publicInput + 1 <=
          degreeBound publicInput + 1
      exact Nat.add_le_add_right
        (Nat.le_max_left _ _) 1
    · rcases List.mem_map.mp laneMember with
        ⟨typed, _, rfl⟩
      simp only [encodeFixed, List.length_map]
      rw [typed.coefficients_length]
      change
        Polynomial.Fe.laneSumcheckDegreeBound + 1 <=
          degreeBound publicInput + 1
      unfold degreeBound NcDegree
        Polynomial.Fe.laneSumcheckDegreeBound
        Polynomial.Nc.Degree.ncSumcheckDegreeBound
      omega
  · intro round member
    change round ∈ (encode input.expectedFeInitial input.carrier).ncRounds
      at member
    rw [encode_ncRounds_values] at member
    rcases List.mem_map.mp member with ⟨typed, _, rfl⟩
    simp only [encodeFixed, List.length_map]
    rw [typed.coefficients_length]
    change NcDegree + 1 <= degreeBound publicInput + 1
    exact Nat.add_le_add_right
      (Nat.le_max_right _ _) 1

/-- The complete schedule derives the exact challenge execution used by the
semantic coin projections. -/
@[simp] theorem run_challenges
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).challenges = challengeOutput input := by
  rfl

/-- FE and NC fields of the complete schedule are exactly the fields of the
canonical exact sub-schedule. -/
theorem run_sumcheck_eq_exact
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    ((run input).afterFe, (run input).feChallenges,
        (run input).afterNc, (run input).ncChallenges) =
      ((Exact.Schedule.run (exactInput input)).afterFe,
        (Exact.Schedule.run (exactInput input)).feChallenges,
        (Exact.Schedule.run (exactInput input)).afterNc,
        (Exact.Schedule.run (exactInput input)).ncChallenges) := by
  rfl

/-- State-only projection of the exact FE-to-NC refinement. -/
@[simp] theorem run_afterNc_eq_exact
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).afterNc =
      (Exact.Schedule.run (exactInput input)).afterNc := by
  rfl

/-- Catch-up consumes the exact NC successor and jointly determines the final
state and proof-visible header digest. -/
theorem run_catchup_joint
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    ((run input).afterCatchup, (run input).headerDigest) =
      Primitives.catchup
        (Exact.Schedule.run (exactInput input)).afterNc := by
  calc
    ((run input).afterCatchup, (run input).headerDigest) =
        Primitives.catchup (run input).afterNc := by
      simpa [run] using
        Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_catchup_joint
          (scheduleInput input)
    _ = Primitives.catchup
          (Exact.Schedule.run (exactInput input)).afterNc := by
      rw [run_afterNc_eq_exact]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule
