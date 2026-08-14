import Nightstream.Implementation.Nebula.FPrime.Claim.NifsReceipt
import Nightstream.Implementation.Nebula.NIFS.Terminal.Relation
import Nightstream.Implementation.Nebula.Commitment.Terminal.ProductCommitmentBridge

/-!
Contract: setup-owned typed output view for the selected V2 NIFS verifier.

Assurance tier: implementation profile boundary.

Owns one selected full-claim verifier together with its exact fourteen-child
terminal output decoder, the canonical bundle codec view of every output
commitment, and the one product-commitment configuration used by all terminal
openings.

Does not assert NIFS soundness. An always-accepting verifier can still inhabit
this structure. A separate reduction must connect `selected.verify` to the
exact SuperNeo transition or a named bad event.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductSelectedVerifier

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Protocol.Nebula.Terminal
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- One verifier-key-owned product profile. The child and bundle functions are
not terminal-proof advice. They are part of the selected verifier profile. -/
structure Profile
    (widths : CompilerWidths)
    (fullShape operationsShape snapshotShape : Shape) where
  selected : SelectedVerifier widths
  config : ProductCommitmentAlgebra.Config fullShape operationsShape
    snapshotShape
  children : selected.Output → ProductTerminalRelation.Children fullShape
  bundles : selected.Output → FoldedChild → CommitmentBundleCodec.Value
  bundleCommitmentExact : ∀ output child,
    TerminalBundleOpeningRows.Layout.codecBundle (bundles output child) =
      (children output child).commitment

end Nightstream.Implementation.Nebula.ProductSelectedVerifier
