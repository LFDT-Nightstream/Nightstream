import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.PublicDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.RingPadding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Selectors

/-!
Public artifact surface for the bounded fixed-point 270-coordinate decoder.

Owns: the fail-closed generated public owner schedule.

Does not own: assignment semantics, private coordinates, matrices, protocol
authority, or row removal.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| `PublicDecoder` | exact 270-coordinate owner lookup | assignment values and relation satisfaction |
| `PublicPadding` | exact 13-row public-padding zero schedule | decoded semantics and constant-one authority |
| `PrivatePadding` | exact 38-row private-alignment zero schedule | decoded semantics and private assignment ownership |
| `RingPadding` | exact 52-row final ring-alignment zero schedule | decoded semantics and constant-one authority |
| `Selectors` | exact three domain rows plus selector-total row | selector values and retained-row coverage |
-/
