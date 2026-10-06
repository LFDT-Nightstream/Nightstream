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

The transfer assumption is stated over the real production key, and its
statement does not change. What is proved is which data the key's transcript
absorbs, and in which order. `Lifecycle/TranscriptCoverage.lean` models each
production challenge (`α`, `γ`, the SumCheck points, the `Π_RLC` scalars) as a
read of the Poseidon2 state after an explicit list of permutation inputs:
zero-padded rate chunks, with a squeeze as the zero chunk. `challenge_seal` ties
every key challenge to its prover-dependent inputs followed by inputs that admit
no prover data, and `coins_eq_reads` states that the PiCCS coin record is
exactly these reads. Equal prover-dependent inputs before a challenge identify
the fresh statement and every earlier prover message (`proverCalls_identify`),
as the dependency specification `AgreeOnAbsorbed` states. The `Π_RLC` reads
reuse only the replay of the sampler's fixed suffix calls, not its ideal-oracle
law.

The transcript does not absorb the verifier key or the running statement
directly. The verifier uses its fixed package key, so the key is bound by the
verifier, not by the transcript. The published HyperNova Construction 3 seeds
its transcript with `hs = ρ(pp, s)`; here nothing about the key seeds it. The
running statement enters only through the prior digest that the fresh public
input carries, as in HyperNova Construction 2. The prior-state link
(`PiCCSSecurity.PriorLink`) states that this digest is the hash of a well-formed
prior preimage that names the verifier context (Construction 2's `vk_fs` slot)
and whose running vector is the NIFS running statement; the terminal recomputes
the digest with its own context. So the link binds the running statement and the
context digest. With the link, equal inputs also identify the prior preimage,
the context and the running statement, or exhibit a state-hash collision
(`PiCCSSecurity.calls_identify_view_or_collision`). Terminal acceptance builds
the link (`terminal_implies_nifsOrBaseOrCollision`), and
`terminal_calls_identify_view_or_collision` applies the result to two accepted
terminals.

The coverage theorems are deterministic. Each ends in identification or a named
event (`StateHashCollision`, `RunCollision`). No probability bound in this
repository charges these events, and none of the coverage theorems is an input
to `history_probability_linear_bound`.

The real success event of the transfer (`FiatShamirTransfer.RealSuccess`) admits
any running statement. In that general event an adaptive adversary could choose
the running statement after the challenges, which the interactive game does not
allow. The HyperNova history theorems (`HyperNovaVisitedSecurity`,
`HyperNovaFalseAcceptance`) apply the transfer to the visited history law, and
at every supported visit real success implies the prior-state link
(`HyperNovaVisitedAcceptance.priorLink_of_realSuccess`). In those theorems every
counted success therefore comes with the link, which identifies the context and
the running statement up to a state-hash collision. The generic NIFS closure
theorems (`NifsClosure`, `NifsFiatShamir`, `NifsProviderLaw`,
`NifsInvalidSource`) apply the transfer at an arbitrary law and carry no link;
there, the binding of the running statement stays inside the transfer
assumption. Restricting the formal success event would change the approved
boundary and needs owner approval.

The native Rust NIFS absorbs the prior digest from `running[0].fold_digest`,
while the Lean key reads it from the fresh public input. The lifecycle makes
them equal: `checked_prior_state` recomputes the digest, and `step_inputs`
rejects a running or parent frame that differs, even when all frames agree on
one wrong value (tested in `step_inputs_rejects_one_consistent_wrong_frame`).
The proof codec does not carry these frames or the PiRLC parent; `extend`
rebuilds both.
`PerApplicationSecurity.replayInput_authority_identifies_or_collision` links the
older committed-statement reductions to this contract: equal replay authority
identifies the fresh statement and every SumCheck round polynomial, or exhibits
two statement call lists that reach one transcript state
(`TranscriptCoverage.RunCollision`). The transfer statement does not consume the
coverage results; they are proved properties of the transcript that the
assumption ranges over. The assumption above still covers the permutation, the
duplex construction, and the transfer bound.

The separate finite sampler laws and `VerifierErrorBudget` are checked.
Under the stated per-call laws, the selected PiCCS test and sampler-abort
events have a union bound over arbitrary `n`. This excludes FS/hash/MSIS
attacks and the square-root extraction loss. Neither this bound nor the
independent-field sampler law establishes a complete deployed security bound.
