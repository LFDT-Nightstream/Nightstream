import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection

/-!
Stable artifact root for selected fixed-point F-prime source-arm evidence.

Owns: the protocol-level artifact hierarchy for this bounded source fixture.

Does not own: selective-lowering refinement, production-wide conformance,
semantic soundness, security reduction, final relation cost, or row removal.

Emits constraints: no.

| Phase | Child | Current checked scope |
|---|---|---|
| PiRLC projection | `PiRlcProjection` | shared + `y_zcol` source rows, fresh columns, and producer coordinates |
-/
