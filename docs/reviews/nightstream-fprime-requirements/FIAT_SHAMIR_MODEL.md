# Classical Fiat–Shamir assumption boundary

Status: the owner approved this explicit assumption boundary on 2026-09-11
UTC after the source review. The approved proposal had SHA-256
`207b67740a6948633870891048091a0c7b2dd90599c04392622ab5cc52cb7498`.
Approval selects no model instance or numerical security level. The
conditional code was checked at `88d394fb4b24cfba21fd1a995bff16c3449312a8`;
the concrete continuation is now consumed by
`NifsClosure.finishValue_probability_and_expected_work`. Its checked source
and gate evidence at `01a8fd8ca68280c7f642451ab33bc714cee5bc98` are recorded in
`formal/nightstream-fprime/NIFS_CLOSURE_STATUS.md`.

For the fixed Nightstream Goldilocks profile, assume an external classical
FS/SuperNeo game transfer for the **actual additive Poseidon2 transcript**.
The real success event is `FiatShamirTransfer.RealSuccess`: the actual
`ProductionKey` NIFS verifier accepts, and the adversary supplies valid
witnesses for its exact 16 returned children. Public acceptance alone is
not a knowledge claim.

The admitted adversaries are adaptive classical algorithms at an externally
declared history depth `d`, with the fixed public seed, setup and protocol
code available. The external model must account for forward and inverse
permutation queries, shared Poseidon2 uses, adaptive statements, and replay.
`Q` denotes total permutation queries in the declared experiment, including
preprocessing and extractor replays; it is not the fold count. Repeated
complete prefixes must retain the same raw answers and state. Any separate
base-query budget must state its inflation to `Q`. This note selects no
numerical query or depth limit.

The assumption supplies a translation to the existing checked causal PiCCS
prover and supported PiRLC/PiDEC continuation, preserving the input and key.
The externally supplied success and error functions `g_d` and `delta_d` must
apply uniformly to all adversaries admitted at depth `d`; they cannot be
chosen separately to fit each adversary's success. Require the exact
`FiatShamirModel.successTransfer` inequality:

`g_d Q (realSuccessProbability) - delta_d Q <= originalSuccessProbability`.

For each fixed `d`, the existing Lean record takes `g := g_d` and
`deltaFS := delta_d`. It does not construct the adversary translation or
prove a history-depth or query bound.

The source-witness conclusion is proved after this assumption. The selected
checker and primitive correctness proofs are supplied by
`NifsClosure.finishValue_probability_and_expected_work`. The selected
provider constructs all suffix and parent checks; the checked prefix call
and identity preparation supply their value/law equalities. Low-norm
invertibility, declared clock bounds and moment bounds remain explicit.
The same context marginal is not itself a proof of an adversary translation; that translation belongs to the external contract.

Approval accepts this **parametric external assumption boundary**.
It does not select `g_d`, `delta_d`, or a numerical security level. A deployment
claim must supply useful bounds for those functions at its declared depth and
total query count; the code does not infer `g_d Q p = p`. The conditional
theorem remains useful for the requested implementation proofs and constraint
reduction.

This is stronger than Poseidon2 collision resistance. Chiesa–Orrù's
[duplex-sponge result](https://eprint.iacr.org/2025/536) uses overwrite
absorption; its theorem is not currently matched to our additive absorption
and initialization. The note makes no ideal-permutation proof, quantum
security, machine-time, or full-history extraction claim.

The separate finite sampler laws and `VerifierErrorBudget` are checked.
Under the stated per-call laws, the selected PiCCS test and sampler-abort
events have a union bound over arbitrary `n`. This excludes FS/hash/MSIS
attacks and the square-root extraction loss. Neither this bound nor the
independent-field sampler law establishes a complete deployed security bound.

## Wide sampler (selected)

Status: the owner approved the whole-vector map and one-block transcript
schedule on 2026-09-24. On 2026-09-25 the owner confirmed in writing that
the approval also extends this Fiat–Shamir boundary to
`WideFiatShamir.RealSuccess` for `PiRLC.Wide.Key.key`. The approved boundary
is the text from the next paragraph to the end of this section; its SHA-256
is `8eada01dde88e02e67101ae1f96a962fa712504893167900ffe426600cbc5b30`. Approval selects no model instance or numerical security level.

The whole-vector map and one-block transcript schedule are selected in
production. `Lifecycle.Nifs.WideFiatShamir` applies the same parametric
boundary to `PiRLC.Wide.Key.key`. Its success event
`WideFiatShamir.RealSuccess` uses that verifier and openings for its exact
sixteen returned children; it is not old-key acceptance. Its
`returned_source_bound_with_adaptive_msis` consumes the existing interactive
extractor without changing the profile or commitment assumptions.

`TranscriptHistory.queryAt_answer` and `WideSamplerSecurity.response_history`
prove a local replay fact: for any seed state and any normalized block
history, replaying that history reproduces the wide key's 17 reads. No
theorem identifies the verifier's PiCCS output state with the replay of the
complete transcript from the initial state. That link, repeated states from
distinct histories (capacity collisions) and inverse-permutation queries are
not modeled by `OracleModel`; they belong to `modelError` or to the external
FS transfer. The wide key has no sampler-abort event, so its trace union bound
contains only the PiCCS test event.

There are two distinct conditional statements:

1. The FS boundary supplies `g Q p_real - deltaFS Q <= p_interactive`, followed
   by the existing extraction, test and same-key MSIS losses. The interactive
   side uses uniform challenges, so the whole difference between the wide
   challenge law and uniform challenges is inside `deltaFS`.
2. `WideSamplerSecurity.adaptive_bias_bound` is the general hybrid bound
   `|p_uniform - p_balanced| <= q*δ`, with `δ < 2^-132`, for a bounded test of
   an oracle run. Its test runs the concrete verifier, which computes its
   challenges with concrete Poseidon2; the oracle enters only through the
   supplied decoder. So `p_balanced` does not give the verifier uniform
   challenges. `concrete_bias_bound` adds the supplied `modelError`.

Statement 2 has no consumer in the extraction chain. It is guidance for
whoever supplies `deltaFS`: a model that identifies the verifier's challenges
with block-oracle replies can use `q*δ` for the law difference. No
monotonicity or Lipschitz property is assumed for `g`, and `q*δ` cannot be
moved across `g` by these proofs.

`q` is the full adaptive block-call count, including repeats and adversarial
calls. `Q` is the existing total permutation-query count, including replay.
The local sampler has 17 reads and 34 permutations per fold, but this does
not give the adversary's query budget or prove a global relation between
`q` and `Q`. No concrete count or security level is selected. For independent
fresh batches only, V6 gives a separate `17*L*δ` term and `17*L/|C|` extraction
loss; these counts must not be substituted for adaptive query accounting.
