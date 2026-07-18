import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge

/-!
Stable correspondence surface for the fixed-point PiRLC `y_zcol` projection.

Owns: composition of the artifact row certificate, producer binding, and
independent typed Phi81 semantic boundary.

Does not own: PiCCS source truth, transcript authority, bad-root probability,
selective-lowering refinement, production-wide conformance, final costs,
necessity, or row removal.

Emits constraints: no.

| Child | Role | Exported boundary |
|---|---|---|
| `ProducerBinding` | serializer index and producer/consumer columns | separate physical refinements |
| `ArtifactRows` | exact selected-source-row transport | `RowsSatisfied` |
| `ActiveBridge` | conditional typed semantic composition | aggregate equality or named bad root |
-/
