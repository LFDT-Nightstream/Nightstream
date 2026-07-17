import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc

/-!
Canonical verifier-visible Split-NC phase evaluator.

Owns: composition of exact-count sequential transcript replay, typed
column/lane point decoding, output-derived terminal computation, and the
fixed-width NC claimed-chain checker.

Does not own: pre-SumCheck coin derivation, source-binding construction,
honest sequential prover construction, FE, output commitments, probability
bounds, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate carries only exact-width round messages.
The verifier owns the incoming transcript state and NC coins; it derives every
round challenge, the terminal point, outgoing state, and terminal scalar.
Soundness names a missing `yZcol` source binding explicitly rather than
promoting the output message or its terminal scalar to authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.verify.transcript` | absorb each exact-width message before deriving its challenge | computed | `Transcript.Nc.derive` |
| `nifs.pi_ccs.nc.verify.point` | split the exact challenge vector into typed column/lane points | computed | `derive`, `derive_point_coordinates` |
| `nifs.pi_ccs.nc.verify.terminal` | evaluate the raw output message only at that derived point | computed payload | `Accepted`, `check` |
| `nifs.pi_ccs.nc.verify.chain` | replay the zero initial claim through every derived challenge | checked | `Accepted`, `check_eq_true_iff_accepted` |
| `nifs.pi_ccs.nc.verify.soundness` | acceptance implies truth, missing source binding, or a named bad event | derived | `accepted_implies_truth_or_unbound_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

universe uState

/-- Verifier-derived NC point and outgoing transcript state. -/
structure Execution (domain : FlatNcDomain) (State : Type uState) where
  point : Polynomial.Nc.Point domain
  finalState : State

/-- Convert one exact flat transcript point into the protocol's typed
column/lane product point without changing coordinate order. -/
private def pointOfReplay
    {domain : FlatNcDomain}
    {State : Type uState}
    (replay : Transcript.Nc.Derived domain State) :
    Polynomial.Nc.Point domain :=
  Polynomial.Nc.Point.ofCoordinates replay.challengePoint.coordinates
    replay.challengePoint.dimension

/-- Derive the complete NC verifier state from one incoming transcript state
and one message-only certificate. -/
def derive
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (certificate : Transcript.Nc.Certificate domain) :
    Execution domain State :=
  let replay := Transcript.Nc.derive machine initialState certificate
  { point := pointOfReplay replay
    finalState := replay.finalState }

/-- Typed point conversion preserves the exact transcript-derived challenge
vector; no coordinate is reordered, padded, or supplied separately. -/
theorem derive_point_coordinates
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (certificate : Transcript.Nc.Certificate domain) :
    (derive machine initialState certificate).point.coordinates =
      (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates := by
  unfold derive pointOfReplay Polynomial.Nc.Point.coordinates
    Polynomial.Nc.Point.ofCoordinates
  exact List.take_append_drop domain.columnVariables _

/-- Logical acceptance of the complete verifier-visible NC phase at the
transcript-derived point and output-derived terminal. -/
def Accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain) : Prop :=
  let execution := derive machine initialState certificate
  SumCheck.Nc.Accepted Polynomial.Nc.InitialSum.claimedInitial
    execution.point.coordinates
    (Polynomial.Nc.Terminal.terminalFromMessage
      .paperNc message coins execution.point)
    certificate.toSumCheck

/-- Executable NC phase checker. Its inputs contain no challenge vector,
terminal point, outgoing state, or terminal scalar. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain) : Bool :=
  let execution := derive machine initialState certificate
  SumCheck.Nc.check Polynomial.Nc.InitialSum.claimedInitial
    execution.point.coordinates
    (Polynomial.Nc.Terminal.terminalFromMessage
      .paperNc message coins execution.point)
    certificate.toSumCheck

/-- The executable phase checker is exactly its verifier-visible logical
relation. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain) :
    check machine initialState coins message certificate = true ↔
      Accepted machine initialState coins message certificate := by
  unfold check Accepted
  exact SumCheck.Nc.check_eq_true_iff_accepted
    Polynomial.Nc.InitialSum.claimedInitial
    (derive machine initialState certificate).point.coordinates
    (Polynomial.Nc.Terminal.terminalFromMessage .paperNc message coins
      (derive machine initialState certificate).point)
    certificate.toSumCheck

/-- Deterministic semantic soundness of the canonical verifier-visible NC
phase. Concrete transcript security and source-binding construction remain
the named outer obligations; neither is hidden in this theorem. -/
theorem accepted_implies_truth_or_unbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain)
    (challengeSetSize : Nat)
    (accepted : Accepted machine initialState coins message certificate) :
    Semantics.Nc.Truth data ∨
      ¬ YZcolBoundToSources covers data
        ({ rPrime := data.priorPoint,
           sPrime := (derive machine initialState certificate).point.column } :
          VerifierPoints shape domain)
        message ∨
      SumCheck.Nc.BadEvent covers data coins
        (derive machine initialState certificate).point.coordinates
        certificate.toSumCheck challengeSetSize := by
  exact OutputAuthority.Nc.acceptedFromMessage_implies_truth_or_unbound_or_badEvent
    noZeroDivisors covers data coins
    (derive machine initialState certificate).point message
    certificate.toSumCheck challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc
