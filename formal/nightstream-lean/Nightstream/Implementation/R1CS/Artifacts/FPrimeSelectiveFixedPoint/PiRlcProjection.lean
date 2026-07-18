import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-!
Stable artifact surface for a cross-branch fixed-point PiRLC source certificate.

Owns: the hierarchy spanning shared projection work and one claim identity.

Does not own: selective lowering, semantic correspondence, transcript
authority, final-relation costs, or row removal.

Emits constraints: no.

| Child | Mathematical obligation | Assurance |
|---|---|---|
| `YZcol` | shared beta/rho work plus two-limb Phi81 identity over 15 sources | bounded source artifact |
-/
