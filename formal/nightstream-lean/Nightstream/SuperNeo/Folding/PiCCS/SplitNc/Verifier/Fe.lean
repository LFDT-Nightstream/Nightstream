import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Semantics

/-!
Canonical verifier-visible Split-NC FE phase evaluator.

Owns: composition of physical mixed-width transcript replay, the
verifier-owned FE initial claim, output-derived terminal computation, the
fixed-width claimed-chain checker, conditional honest-message completeness,
and deterministic semantic soundness.

Does not own: pre-SumCheck coin derivation, construction of a transcript
fixed-point honest certificate, NC, output commitments, probability bounds,
Poseidon2 refinement, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate carries only physical row/lane messages.
The verifier derives the row/lane challenge point and terminal scalar. A raw
`yRing` message has no authority merely because the claimed chain accepts;
soundness exposes an explicit output mismatch unless it equals the
source-derived value at the verifier-owned row point.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.verify.transcript` | absorb physical row then lane messages before deriving challenges | computed | `Transcript.Fe.derive` |
| `nifs.pi_ccs.fe.verify.initial` | derive the public carried-evaluation claim | computed | `Polynomial.Fe.initial` |
| `nifs.pi_ccs.fe.verify.terminal` | evaluate raw `yRing` only at the derived point | checked payload | `Accepted`, `check` |
| `nifs.pi_ccs.fe.verify.chain` | replay one claimed chain across the row/lane cut | checked | `Accepted`, `check_eq_true_iff_accepted` |
| `nifs.pi_ccs.fe.verify.completeness` | FE truth and honest source-bound messages are accepted | model-level | `accepted_of_truth_and_honestAt` |
| `nifs.pi_ccs.fe.verify.soundness` | acceptance implies FE truth, output mismatch, or named algebraic bad event | derived | `accepted_implies_truth_or_mismatch_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Logical acceptance of the complete verifier-visible FE phase. The
challenge point is derived from the physical certificate, while the terminal
is derived from the raw output message at that point. -/
def Accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Fe.Certificate input domain) : Prop :=
  let execution := Transcript.Fe.derive machine initialState certificate
  SumCheck.Fe.Accepted
    (Polynomial.Fe.initial profile input coins)
    (Polynomial.Fe.terminalFromMessage profile input coins
      execution.challengePoint message)
    execution.challengePoint certificate

/-- Executable FE phase checker with no caller-supplied challenge point or
terminal scalar. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Fe.Certificate input domain) : Bool :=
  let execution := Transcript.Fe.derive machine initialState certificate
  SumCheck.Fe.check
    (Polynomial.Fe.initial profile input coins)
    (Polynomial.Fe.terminalFromMessage profile input coins
      execution.challengePoint message)
    execution.challengePoint certificate

/-- The executable FE phase checker is exactly its logical relation. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Fe.Certificate input domain) :
    check machine initialState profile input coins message certificate = true ↔
      Accepted machine initialState profile input coins message certificate := by
  unfold check Accepted
  exact SumCheck.Fe.check_eq_true_iff_accepted
    (Polynomial.Fe.initial profile input coins)
    (Polynomial.Fe.terminalFromMessage profile input coins
      (Transcript.Fe.derive machine initialState certificate).challengePoint
      message)
    (Transcript.Fe.derive machine initialState certificate).challengePoint
    certificate

/-- Conditional fixed-transcript completeness.

This theorem deliberately requires honesty at the point derived from the
physical certificate. Constructing such messages sequentially while each
message determines the next Fiat--Shamir challenge is a later honest-prover
transcript theorem, not an assumption hidden here. -/
theorem accepted_of_truth_and_honestAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate :
      SumCheck.Fe.Certificate (PublicInput.ofSources data) domain)
    (truth : Semantics.Fe.Truth data)
    (messageBound :
      message.yRing =
        Polynomial.Fe.sourceYRingAt data
          (Transcript.Fe.derive machine initialState certificate).challengePoint.row)
    (honest :
      SumCheck.Fe.HonestAt
        (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate) :
    Accepted machine initialState profile (PublicInput.ofSources data) coins
      message certificate := by
  unfold Accepted
  change SumCheck.Fe.Accepted
    (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
    (Polynomial.Fe.terminalFromMessage profile (PublicInput.ofSources data)
      coins
      (Transcript.Fe.derive machine initialState certificate).challengePoint
      message)
    (Transcript.Fe.derive machine initialState certificate).challengePoint
    certificate
  have semanticAccepted :=
    SumCheck.Fe.complete_of_truth_and_honestAt
      profile data coins truth
      (Transcript.Fe.derive machine initialState certificate).challengePoint
      certificate honest
  rw [Polynomial.Fe.terminalFromMessage_eq_qAtPoint_of_yRing_eq
    profile data coins
    (Transcript.Fe.derive machine initialState certificate).challengePoint
    message messageBound,
    ← Polynomial.Fe.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint]
  exact semanticAccepted

/-- Deterministic semantic soundness of the canonical FE phase.

No probability claim is made. The result keeps output authority and the
compression/SumCheck bad events separate so later composition cannot promote
a self-consistent raw message to authority. -/
theorem accepted_implies_truth_or_mismatch_or_badEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate :
      SumCheck.Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted machine initialState profile (PublicInput.ofSources data) coins
        message certificate) :
    Semantics.Fe.Truth data ∨
      Polynomial.Fe.OutputMismatch data
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        message ∨
      SumCheck.Fe.BadEvent profile data coins
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate challengeSetSize := by
  let point :=
    (Transcript.Fe.derive machine initialState certificate).challengePoint
  change SumCheck.Fe.Accepted
    (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
    (Polynomial.Fe.terminalFromMessage profile (PublicInput.ofSources data)
      coins point message)
    point certificate at accepted
  by_cases messageBound :
      message.yRing = Polynomial.Fe.sourceYRingAt data point.row
  · have terminalBinding :
        Polynomial.Fe.terminalFromMessage profile
            (PublicInput.ofSources data) coins point message =
          Polynomial.Fe.qAtPoint profile data coins point :=
      Polynomial.Fe.terminalFromMessage_eq_qAtPoint_of_yRing_eq
        profile data coins point message messageBound
    have semanticAccepted :
        SumCheck.Fe.Accepted
          (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
          (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins
            point.coordinates)
          point certificate := by
      rw [Polynomial.Fe.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint]
      rw [← terminalBinding]
      exact accepted
    rcases SumCheck.Fe.accepted_implies_truth_or_badEvent
        profile data coins point certificate challengeSetSize semanticAccepted with
      truth | badEvent
    · exact Or.inl truth
    · exact Or.inr (Or.inr badEvent)
  · exact Or.inr (Or.inl messageBound)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
