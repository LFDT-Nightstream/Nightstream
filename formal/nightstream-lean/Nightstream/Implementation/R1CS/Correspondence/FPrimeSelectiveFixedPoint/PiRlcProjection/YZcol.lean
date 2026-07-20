import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Soundness

/-!
Stable correspondence surface for the fixed-point PiRLC `y_zcol` projection.

Owns: composition of the source artifact, exact compact selective rows,
producer binding, and independent typed Phi81 semantic boundary.

Does not own: PiCCS source truth, selector enforcement, transcript authority,
bad-root probability, production-wide conformance, final costs, necessity, or
row removal.

Emits constraints: no.

| Child | Role | Exported boundary |
|---|---|---|
| `ProducerBinding` | serializer index and producer/consumer columns | separate physical refinements |
| `ArtifactRows` | exact selected-source-row transport | `RowsSatisfied` |
| `ActiveBridge` | conditional typed semantic composition | aggregate equality or named bad root |
| `Selective.Soundness` | exact compact rows refine source projection obligations | active-row soundness |
| `Selective.HonestAssignment.TerminalSemantics` | honest source equations construct compact-row witnesses | honest completeness |
-/
