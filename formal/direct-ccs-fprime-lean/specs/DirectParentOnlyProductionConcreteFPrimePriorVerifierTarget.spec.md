# DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget

This module states the concrete F′ prior-verifier target used by the
parent-only production path.

The verifier surface is the runtime exact-public-IO opening surface. Its
acceptance predicate replays the compact public image, Construction-2 boundary,
transcript, and terminal/boundary public IO against the canonical statement for
the claimed `(steps, image)` pair.

Acceptance is authoritative only through a fixed authority opener. A verified
prior proof must open to a proof-carrying folded F′ authority, and any opened
authority is accepted only when its step count and public image equal the
verifier's claimed step count and public image.

The target excludes aggregate-only checks over child or public-IO data. Public
pair binding is by canonical statement replay plus authority opening for the
same `(steps, image)` pair.
