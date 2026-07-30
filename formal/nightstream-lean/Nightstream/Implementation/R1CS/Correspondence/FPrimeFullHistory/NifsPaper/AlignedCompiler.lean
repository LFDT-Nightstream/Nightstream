import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.SelectiveLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ProductionCarrier
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding

/-!
Fail-closed public surface for the typed F' carrier compiler bridge.

Owns: the compiler-bridge facade for the independent 270-coordinate layout,
the exact selective polynomial, public-padding semantics, and the generated
public-carrier artifact slice.

Does not own: complete private-column decoding, all emitted rows, full F′
relation satisfaction, Pi_CCS/CE refinement, Ajtai commitments, NIFS
soundness, or row removal.

Emits constraints: no.

| Child | Mathematical ownership | Excluded boundary |
|---|---|---|
| `SelectiveLayout` | typed public/padding/selector/private ranges and legacy selector-leak witness | concrete compiler instantiation |
| `ProductionCarrier` | exact generated layout plus all thirteen public-padding rows | full fixed-point/private relation |
| `SelectiveCcs.Polynomial` | exact 13-port, 66-term, degree-8 gate syntax and six-family decomposition | protocol minimality |
| `SelectiveCcs.Polynomial.Necessity` | omission counterexample for each chosen term family | complete verifier minimality |
| `SelectiveCcs.Padding` | normalized semantics, polynomial specialization, coefficient classifier, and local necessity | non-padding row families |

The legacy list adapters remain importable by focused diagnostic tests, but
they are deliberately not re-exported here as compiler authority. Sparse
matrix storage, assignment serialization, Ajtai setup artifacts, non-padding
CCS rows, and the complete fixed-point compile remain fail-closed. The
`ProductionCarrier` child closes only the explicitly named layout and
thirteen-row public-padding slice.
-/
