import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.ActivePins

/-!
Stable artifact facade for active constant, selector, and public-coordinate pins.

Owns: the handwritten import boundary for generated pin values, packed public
coordinates, selector-domain rows, and the one-hot row.

Does not own: public-assignment semantics, selector truth, row satisfaction,
acceptance, costs, or row-removal authority.

| Stable stage path | Generated owner | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.active_pins` | `Generated.ActivePins` | computed artifact |
-/
