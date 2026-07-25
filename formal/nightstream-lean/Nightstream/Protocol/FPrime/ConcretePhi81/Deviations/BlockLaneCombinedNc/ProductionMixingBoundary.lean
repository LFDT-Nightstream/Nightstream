import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.Algebra
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.ChallengeSupport
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.CoreSchedule
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.ExactEvents

/-!
Production Split-NC mixing boundary.

Assurance tier: model-level obstruction and concrete algebraic boundary.

This facade deliberately exports no replacement soundness theorem. The
production carriers currently lack a sampled finite support aligned with the
unrestricted denominator and an internal causal order for the opaque core
challenge record. The imported leaves provide exact countermodels at those
interfaces and the concrete no-zero-divisor bridge.

Owns: one facade for the exact event inventory, concrete algebra bridge, and
kernel-checked production-carrier obstructions.

Does not own: a replacement soundness theorem, challenge sampler,
Fiat--Shamir, frozen semantics, Rust/R1CS, encoding, or rows.

Emits constraints: no.

| Boundary | Exported evidence | Excluded claim |
|---|---|---|
| production carrier | exact support and schedule obstructions | obstruction to a future refined carrier |
| existing Split-NC | unchanged collision theorem remains owned separately | loss reassociation |
-/
