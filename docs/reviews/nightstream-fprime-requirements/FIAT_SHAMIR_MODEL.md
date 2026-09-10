# Classical Fiat–Shamir assumption boundary

Status: proposed for owner approval. The conditional code is checked at
`88d394fb4b24cfba21fd1a995bff16c3449312a8`; no model instance is asserted.

For the fixed Nightstream Goldilocks profile, assume an external classical
FS/SuperNeo game transfer for the **actual additive Poseidon2 transcript**.
The real success event is `FiatShamirTransfer.RealSuccess`: the actual
`ProductionKey` NIFS verifier accepts, and the adversary supplies valid
witnesses for its exact 16 returned children. Public acceptance alone is
not a knowledge claim.

The admitted adversaries are adaptive classical algorithms, with the fixed
public seed, setup and protocol code available. The external model must
account for forward and inverse permutation queries, shared Poseidon2 uses,
adaptive statements, and replay. `Q` denotes total permutation queries in
the declared experiment, including extractor replays; it is not the fold
count. Repeated complete prefixes must retain the same raw answers and
state. Any separate base-query budget must state its inflation to `Q`.
This note selects no numerical query limit.

The assumption supplies a translation to the existing checked causal PiCCS
prover and supported PiRLC/PiDEC continuation, preserving the input and key.
For its externally supplied success and error functions `g` and `deltaFS`,
require the exact `FiatShamirModel.successTransfer` inequality:

`g Q (realSuccessProbability) - deltaFS Q <= originalSuccessProbability`.

The source-witness conclusion is proved after this assumption. The selected
checker and primitive correctness proofs are supplied by
`NifsFiatShamir.finishValue_probability_and_expected_work`. Preparation and
call refinement, low-norm invertibility, declared clock bounds and moment
bounds remain explicit. The same context marginal is not itself a proof of
an adversary translation; that translation belongs to the external contract.

Approval would accept this **parametric external assumption boundary**.
It would not select `g`, `deltaFS`, or a numerical security level. A deployment
claim must supply useful bounds for those functions and its total query count;
the code does not infer `g Q p = p`. The conditional theorem remains useful
for the requested implementation proofs and constraint reduction.

This is stronger than Poseidon2 collision resistance. Chiesa–Orrù's
[duplex-sponge result](https://eprint.iacr.org/2025/536) uses overwrite
absorption; its theorem is not currently matched to our additive absorption
and initialization. The note makes no ideal-permutation proof, quantum
security, machine-time, or full-history extraction claim.

The separate finite sampler laws and `VerifierErrorBudget` are checked.
Under the stated per-call laws, the selected PiCCS test and sampler-abort
events have a union bound over arbitrary `n`. For the user's 100,000,000-call
example, their combined budget is at most `2^-87`. This excludes FS/hash/MSIS
attacks and the square-root extraction loss. Neither that example nor the
independent-field sampler law establishes a complete deployed security bound.
