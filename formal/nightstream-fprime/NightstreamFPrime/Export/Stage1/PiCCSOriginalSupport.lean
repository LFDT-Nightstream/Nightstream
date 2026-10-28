import NightstreamFPrime.Lifecycle.Types

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport

open NightstreamFPrime.Lifecycle

def isZero (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) : Bool :=
  masks.all (fun block => (block[source.val]?.getD (0, 0)) == (0, 0))

end NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport
