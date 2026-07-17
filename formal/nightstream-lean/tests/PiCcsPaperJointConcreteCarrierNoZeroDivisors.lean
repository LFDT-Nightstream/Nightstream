import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors

/-! Focused model-level regressions for concrete extension cancellation. -/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The extension no-zero-divisor theorem is derived from the two explicit
base-carrier facts; no extension cancellation callback is assumed. -/
example
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : SevenProjectiveNonresidue) :
    ExtensionNoZeroDivisors :=
  extensionNoZeroDivisors_of_base_and_seven
    baseNoZeroDivisors sevenNonresidue

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Tests
