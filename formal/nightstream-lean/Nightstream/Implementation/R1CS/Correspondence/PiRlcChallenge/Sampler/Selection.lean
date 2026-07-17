import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.LinearEquality
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Initialization
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.OneHot
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Position
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.FirstAccepted

/-! Parent for first-accepted candidate-selection correspondence.

Owns: acceptance flags, equality pins, initialization, one-hot selection,
position tracking, and the first-accepted conclusion.

Does not own: candidate decoding, transcript challenges, or ring assembly.

Emits constraints: no.

| Child family | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Acceptance` / `LinearEquality` | candidate acceptance and equality equations | candidate semantics |
| `Initialization` / `OneHot` / `Position` | deterministic selector control flow | transcript order |
| `FirstAccepted` | selected position is the first accepted candidate | scalar/ring output encoding |
-/
