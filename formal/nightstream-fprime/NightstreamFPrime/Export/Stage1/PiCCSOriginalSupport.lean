import NightstreamFPrime.Lifecycle.Types

/-! Check every decoded mask block for one unchanged source index.
The caller validates input framing. Preservation proofs remain separate. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport

open NightstreamFPrime.Lifecycle

/-- A source is zero only when both masks are zero in every decoded block.
Missing source entries use the original decoder's zero pair. -/
def isZero (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) : Bool :=
  masks.all (fun block => (block[source.val]?.getD (0, 0)) == (0, 0))

end NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport
