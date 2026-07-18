import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal

/-!
Canonical verifier-visible block×lane Split-NC phase evaluator.

Assurance tier: model-level.

Owns: composition of exact-count one-entry transcript replay, the typed
block/lane point, packed-message terminal computation, the shared five-slot
claimed-chain checker, and deterministic semantic soundness with an explicit
output-binding failure.

Does not own: pre-SumCheck coin derivation, concrete Poseidon2 replay, a
Fiat--Shamir honest-prover fixed point, construction of packed-output
authority, FE, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate carries only five-slot messages. The
verifier derives every challenge, the final point, the successor state, and
the terminal scalar. `Claims.yZcol` remains untrusted: semantic soundness uses
the full active-lane binding at the final transcript-derived block point and
exposes its absence as a separate outcome. A digest or scalar terminal is
never promoted to authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.verify.transcript` | one-entry exact-count block-then-lane replay | computed | `Transcript.Nc.BlockLane.derive` |
| `nifs.pi_ccs.nc.block_lane.verify.point` | typed block/lane point is the replay result | computed | `derive` |
| `nifs.pi_ccs.nc.block_lane.verify.terminal` | evaluate only the raw packed output at that point | computed payload | `Accepted`, `check` |
| `nifs.pi_ccs.nc.block_lane.verify.chain` | replay the zero initial claim through all derived challenges | checked | `Accepted`, `check` |
| `nifs.pi_ccs.nc.block_lane.verify.output_binding` | every active lane binds at the final block point | explicit proof boundary | `Terminal.PackedYZcolBoundAtBlock` |
| `nifs.pi_ccs.nc.block_lane.verify.soundness` | acceptance yields truth, missing binding, or a named algebraic event | derived | `accepted_implies_truth_or_unbound_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

/-- Verifier-derived typed point and outgoing transcript state. -/
structure Execution (domain : BlockNcDomain) (State : Type uState) where
  point : Point domain
  finalState : State

/-- Derive the complete phase execution from one incoming state and one
message-only exact-count certificate. -/
def derive
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) :
    Execution domain State :=
  let replay := Transcript.Nc.BlockLane.derive
    machine initialState certificate
  {
    point := replay.challengePoint
    finalState := replay.finalState
  }

/-- The phase execution point is exactly the typed transcript-derived point. -/
theorem derive_point
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) :
    (derive machine initialState certificate).point =
      (Transcript.Nc.BlockLane.derive
        machine initialState certificate).challengePoint := by
  rfl

/-- Logical acceptance at the transcript-derived point and message-derived
terminal. -/
def Accepted
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) : Prop :=
  let execution := derive machine initialState certificate
  SumCheck.Nc.Accepted InitialSum.claimedInitial execution.point.coordinates
    (Terminal.terminalFromMessage message coins execution.point)
    certificate.toSumCheck

/-- Executable phase checker. No challenge vector, final point, successor
state, or terminal scalar is certificate data. -/
def check
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) : Bool :=
  let execution := derive machine initialState certificate
  SumCheck.Nc.check InitialSum.claimedInitial execution.point.coordinates
    (Terminal.terminalFromMessage message coins execution.point)
    certificate.toSumCheck

/-- Executable and logical phase acceptance coincide exactly. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) :
    check machine initialState coins message certificate = true ↔
      Accepted machine initialState coins message certificate := by
  unfold check Accepted
  exact SumCheck.Nc.check_eq_true_iff_accepted
    InitialSum.claimedInitial
    (derive machine initialState certificate).point.coordinates
    (Terminal.terminalFromMessage message coins
      (derive machine initialState certificate).point)
    certificate.toSumCheck

/-- Deterministic semantic soundness of the canonical block×lane phase.

This theorem deliberately leaves packed-output authority and all challenge
sampling probabilities outside the conclusion. When binding is present, the
message terminal is rewritten to the independent source polynomial before
using SumCheck soundness. -/
theorem accepted_implies_truth_or_unbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain)
    (challengeSetSize : Nat)
    (accepted : Accepted machine initialState coins message certificate) :
    Semantics.Nc.Truth data ∨
      ¬ Terminal.PackedYZcolBoundAtBlock covers data
        (derive machine initialState certificate).point.block message ∨
      SumCheck.Nc.BlockLane.BadEvent covers data coins
        (derive machine initialState certificate).point
        certificate.toSumCheck challengeSetSize := by
  let execution := derive machine initialState certificate
  change
    Semantics.Nc.Truth data ∨
      ¬ Terminal.PackedYZcolBoundAtBlock covers data
        execution.point.block message ∨
      SumCheck.Nc.BlockLane.BadEvent covers data coins execution.point
        certificate.toSumCheck challengeSetSize
  by_cases bound : Terminal.PackedYZcolBoundAtBlock covers data
      execution.point.block message
  · have terminalBinding :
        Terminal.terminalFromMessage message coins execution.point =
          InitialSum.sumcheckPolynomial covers data coins
            execution.point.coordinates := by
      calc
        Terminal.terminalFromMessage message coins execution.point =
            Mixing.qAtPoint covers data coins execution.point :=
          Terminal.terminal_eq_qAtPoint_of_bound
            covers data coins execution.point message bound
        _ = InitialSum.sumcheckPolynomial covers data coins
              execution.point.coordinates :=
          (InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
            covers data coins execution.point).symm
    have semanticAccepted :
        SumCheck.Nc.Accepted InitialSum.claimedInitial
          execution.point.coordinates
          (InitialSum.sumcheckPolynomial covers data coins
            execution.point.coordinates)
          certificate.toSumCheck := by
      rw [← terminalBinding]
      exact accepted
    rcases SumCheck.Nc.BlockLane.accepted_implies_truth_or_badEvent
        noZeroDivisors covers data coins execution.point
        certificate.toSumCheck challengeSetSize semanticAccepted with
      truth | badEvent
    · exact Or.inl truth
    · exact Or.inr (Or.inr badEvent)
  · exact Or.inr (Or.inl bound)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane
