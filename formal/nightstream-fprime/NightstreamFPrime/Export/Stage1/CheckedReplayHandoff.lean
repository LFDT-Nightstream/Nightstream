import NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor
import NightstreamFPrime.Export.Stage1.HyperNovaInput

/-! Reuse the accepted exact successor as the next checked replay's prior.
The fresh claim and all seventeen source witnesses are the prior result. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplayHandoff

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open PerApplicationCanonicalAssignment
open Poseidon2HashChainV1Package (application)

/-- Re-encoding the next C input preserves the exact accepted predecessor
payload. The next proof supplies only the next execution's local data. -/
theorem prior_eq_payload
    (result : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application) (nextProof : Lifecycle.Proof 9) :
    CheckedReplayStep.prior
      (HyperNovaInput.ofClaims result (CheckedReplaySuccessor.payload result children raw).fresh
        nextProof) children raw.completeAssignment =
      CheckedReplaySuccessor.payload result children raw := by
  unfold CheckedReplayStep.prior
  unfold PiCCSInputCheck.running PiCCSInputCheck.fresh
  rw [HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims]
  rfl

end NightstreamFPrime.Export.Stage1.CheckedReplayHandoff
