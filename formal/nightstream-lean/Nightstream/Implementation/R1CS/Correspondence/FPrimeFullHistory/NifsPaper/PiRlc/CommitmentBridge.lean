import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Typed commitment refinement for production `Pi_RLC`.

Assurance tier: model-level. These theorems compare independent Lean public
computations; they do not establish Rust-conformant row emission or commitment
binding.

Protocol: SuperNeo `Pi_RLC` inside the fixed F' NIFS.
Phase: public commitment projection decoding and ring action.
Constraint family: the eighteen public commitment identities; this file emits
no rows.

Owns: decoding eighteen coefficient lists into the typed public commitment
carrier and equality between production's `phi81Combine` operation and the
independent typed commitment fold.

Does not own: Ajtai key alignment or binding security; commitment column
serialization; transcript challenges; projection-row soundness; source or
parent authority; costs; or row removal.

Emits constraints: no.

Authority boundary: the output commitment is computed from explicit public
input commitments and challenges. No private opening, key, digest, or
caller-supplied homomorphism law enters this bridge.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.commitment.decode` | eighteen coefficient lists decode to `Fin 18 -> RingF` | computed | `decodeCommitmentRings` |
| `nifs.pi_rlc.verify.identities.commitment.product_sum` | list `phi81Combine` decodes to the shared typed product sum | derived | `RingTransport.ringOfList_phi81Combine` |
| `nifs.pi_rlc.verify.identities.commitment.combine` | every decoded row equals the typed public commitment fold | derived | `decodeCommitmentRings_phi81Combine` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

/-- Decode each production commitment row as one exact Phi81 ring. -/
def decodeCommitmentRings (rings : CommitmentRings) :
    PiRLCAlgebra.Commitment.Value 18 :=
  fun row => ringOfList (rings row)

private theorem combineCommitments_apply
    {count : Nat} (challenges : Fin count -> RingF)
    (values : Fin count -> PiRLCAlgebra.Commitment.Value 18)
    (row : Fin 18) :
    (PiRLCAlgebra.Commitment.combineCommitments challenges values) row =
      productSum challenges (fun index => values index row) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCAlgebra.Commitment.combineCommitments,
        PiRLCAlgebra.Commitment.commitmentAdd,
        PiRLCAlgebra.Commitment.commitmentAct, productSum]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => values index.succ)]

/-- Production's eighteen list-level commitment combinations are exactly the
independent typed public commitment combination. -/
theorem decodeCommitmentRings_phi81Combine
    {count : Nat} (challenges : Fin count -> Ring)
    (inputs : Fin count -> CommitmentRings) :
    decodeCommitmentRings
        (fun row =>
          phi81Combine challenges (fun index => inputs index row)) =
      PiRLCAlgebra.Commitment.combineCommitments
        (fun index => ringOfList (challenges index))
        (fun index => decodeCommitmentRings (inputs index)) := by
  funext row
  rw [decodeCommitmentRings, ringOfList_phi81Combine]
  exact (combineCommitments_apply
    (fun index => ringOfList (challenges index))
    (fun index => decodeCommitmentRings (inputs index)) row).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge
