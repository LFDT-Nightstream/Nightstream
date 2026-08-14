import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate.Generated.Coordinates
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate.Generated.DifferentialCases
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate.Generated.Rows

/-!
Proof-free radix-four strict-`PiDEC` canonical-X artifact facade.

Assurance tier: generated data only. The correspondence compiler checks each
record before it gives the artifact Rust-conformant status.

Owns: 270 coordinate maps, seven children and fourteen signed limbs per map,
and all 6,480 exact rows emitted by Rust.

Does not own: row meaning, satisfaction, complete strict-`PiDEC`, or
production-profile selection.

Emits constraints: no; this module describes rows emitted elsewhere.

| Artifact leaf | Exact payload | Excluded boundary |
|---|---|---|
| `coordinates` | 270 seven-child coordinate records | compiler meaning |
| `rows` | 6,480 exact physical sparse rows | satisfaction |
| `differentialCases` | Rust-generated mutation cases | complete `PiDEC` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate

abbrev coordinates := Generated.coordinates
abbrev rows := Generated.rows
abbrev differentialCases := Generated.DifferentialCases.values

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate
