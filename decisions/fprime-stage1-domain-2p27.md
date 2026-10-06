# F′ Stage 1 domain `2^27`

The owner requested this change on 2026-10-05. It supersedes
[the 28-round domain decision](fprime-stage1-domain-2p28.md) and the domain
values in the earlier Stage 1 goal.

The production relation uses a 27-variable cube and exactly 27 PiCCS sumcheck
rounds. Both the rows and the ring-padded carrier must fit `2^27 = 134,217,728`.
Complete 54-coordinate ring blocks can use at most 134,217,702 coordinates.

The profile keeps seven CCS matrices, `b = 2`, `k_rho = 16`, `B = 2^16`,
and Poseidon2 protocol binding. The approved maximum key stays at 4,708,530
ring columns; the smaller domain now sets the carrier capacity.

The layout proofs, transcript, state encoding, package identities, native
consumer, and independent conformance checks must use the same 27-round
profile. Saved 28-round proofs are not evidence for it. The selected package
and proof fixtures must be regenerated from this source, with fresh native
folds checked by Lean before fixture promotion.
