import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Semantics

/-!
Output-authority composition for the Split-NC SumCheck terminal.

Owns: the single proof boundary that connects verifier-visible `yZcol`
outputs to the independent NC polynomial and its existing deterministic
SumCheck soundness theorem.

Does not own: the narrow SumCheck checker, transcript challenge derivation,
construction of `YZcolBoundToSources`, output commitments, root probability,
Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the verifier may compute a terminal from an untrusted
output message, but semantic soundness follows only when the same message is
bound to the authoritative CCS/assignment sources at the verifier-derived
column point. Failure of that binding is an explicit bad outcome; scalar
terminal equality is never treated as a substitute.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.output_terminal.compute` | evaluate the terminal from the raw `yZcol` message and verifier point | computed | `Polynomial.Nc.Terminal.terminalFromMessage` |
| `nifs.pi_ccs.nc.output_terminal.bind` | every active `yZcol` value is the source-derived projection at that column point | security boundary | `OutputClaims.YZcolBoundToSources` |
| `nifs.pi_ccs.nc.output_terminal.soundness` | visible acceptance implies NC truth, failed source binding, or a named algebraic bad event | derived | `acceptedFromMessage_implies_truth_or_unbound_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

/-- Verifier-visible NC acceptance is sound up to exactly four disjoint
semantic boundaries: missing output/source authority, selector mixing, gamma
mixing, or a fixed-degree SumCheck collision. The latter three are grouped in
`SumCheck.Nc.BadEvent`; this wrapper adds only the separate `yZcol` authority
failure required by the raw output message.

The typed terminal point supplies the exact SumCheck arity. No caller-provided
length proof or semantic terminal equality is accepted. -/
theorem acceptedFromMessage_implies_truth_or_unbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (point : Polynomial.Nc.Point domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Nc.Certificate)
    (challengeSetSize : Nat)
    (accepted :
      SumCheck.Nc.Accepted Polynomial.Nc.InitialSum.claimedInitial
        point.coordinates
        (Polynomial.Nc.Terminal.terminalFromMessage
          .paperNc message coins point)
        certificate) :
    Semantics.Nc.Truth data ∨
      ¬ YZcolBoundToSources covers data
        ({ rPrime := data.priorPoint, sPrime := point.column } :
          VerifierPoints shape domain)
        message ∨
      SumCheck.Nc.BadEvent covers data coins point.coordinates certificate
        challengeSetSize := by
  by_cases bound :
      YZcolBoundToSources covers data
        ({ rPrime := data.priorPoint, sPrime := point.column } :
          VerifierPoints shape domain)
        message
  · have terminalBinding :
        Polynomial.Nc.Terminal.terminalFromMessage
            .paperNc message coins point =
          Polynomial.Nc.InitialSum.sumcheckPolynomial
            .paperNc covers data coins point.coordinates := by
      calc
        Polynomial.Nc.Terminal.terminalFromMessage
            .paperNc message coins point =
            Polynomial.Nc.Mixing.qAtPoint .paperNc covers data coins point :=
          Polynomial.Nc.Terminal.terminal_eq_qAtPoint_of_yZcolBoundToSources
            .paperNc covers data coins point message bound
        _ = Polynomial.Nc.InitialSum.sumcheckPolynomial
              .paperNc covers data coins point.coordinates :=
          (Polynomial.Nc.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
            .paperNc covers data coins point).symm
    have semanticAccepted :
        SumCheck.Nc.Accepted Polynomial.Nc.InitialSum.claimedInitial
          point.coordinates
          (Polynomial.Nc.InitialSum.sumcheckPolynomial
            .paperNc covers data coins point.coordinates)
          certificate := by
      rw [<- terminalBinding]
      exact accepted
    rcases SumCheck.Nc.accepted_implies_truth_or_badEvent
        noZeroDivisors covers data coins point.coordinates certificate
        challengeSetSize point.coordinates_length semanticAccepted with
      truth | badEvent
    · exact Or.inl truth
    · exact Or.inr (Or.inr badEvent)
  · exact Or.inr (Or.inl bound)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc
