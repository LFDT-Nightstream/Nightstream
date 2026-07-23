import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicWriteTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PhysicalSelectorAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.RingPadding
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
| `PhysicalPublicAssignment` | first 270 production cells refine the typed carrier from source writes and generated padding rows | generated source-write certificate and private suffix |
| `PublicWriteTrace` | a certified exact one-arm write trace computes those first 270 production cells | generic trace model and private suffix |
| `ProductionPublicWriteTrace` | the active post-PiDEC Rust execution artifact supplies that exact 270-write certificate | fixed-profile public prefix only; private suffix |
| `PhysicalSelectorAssignment` | generated steady arm replays the selector unit vector and proves physical column 272 is one | active call-site arm selection and remaining private suffix |
| `PublicPadding` | 13 exact rows enforce the typed fixed-public-padding obligation | constant-one and complete assignment decoding |
| `PrivatePadding` | 38 exact emitted rows enforce the prepared zero interval | later private-column ownership and complete relation decoding |
| `RingPadding` | 52 exact emitted rows enforce the final 64-lane zero interval | constant-one and complete relation satisfaction |
| `Selectors` | four exact rows refine Boolean and sum-to-one equations | gated branch rows and row-removal authority |
-/
