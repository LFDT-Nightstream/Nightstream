import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.Selectors

/-!
Bounded fixed-point Carrier270 correspondence surface.

Owns: the artifact-backed public-coordinate decoder and its equality to the
independent typed public carrier.

Does not own: private assignment/matrix decoding, CCS/CE membership,
commitment-key alignment, NIFS authority, or row removal.

Emits constraints: no.

| Child | Mathematical obligation | Excluded boundary |
|---|---|---|
| `PublicAssignment` | generated owner values equal the typed public carrier | private assignment and matrix decoding |
| `PublicPadding` | 13 exact rows enforce the typed fixed-public-padding obligation | constant-one and complete assignment decoding |
| `PrivatePadding` | 38 exact emitted rows enforce the prepared zero interval | later private-column ownership and complete relation decoding |
| `Selectors` | four exact rows refine Boolean and sum-to-one equations | gated branch rows and row-removal authority |
-/
