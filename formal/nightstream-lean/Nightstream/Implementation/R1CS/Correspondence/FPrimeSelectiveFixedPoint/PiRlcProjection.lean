import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-!
Correspondence root for the fixed-point PiRLC projection phase.

Owns: the phase → claim-family semantic-refinement hierarchy.

Does not own: other PiRLC phases, transcript generation, global security
bounds, selective lowering, final-relation costs, or row removal.

Emits constraints: no.

| Claim family | Mathematical obligation | Assurance boundary |
|---|---|---|
| `YZcol` | shared beta/rho plus two-limb Phi81 source aggregation | bounded source artifact + conditional model consequence |
-/
