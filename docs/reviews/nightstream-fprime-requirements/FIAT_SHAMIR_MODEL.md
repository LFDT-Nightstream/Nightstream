# Classical Fiat–Shamir assumption boundary

**Retired on 2026-10-07 (owner decision).** The black-box transfer to the
causal interactive game has no useful `g`: guessing which oracle query carries
each SumCheck message loses about `(Q+1)^29`. `FiatShamirModel` and every
declaration that only it used are deleted. The history bound now takes
HyperNova errata Assumption 1, plain-model part, at each visit
(`Export.Stage1.HyperNovaVisitedSecurity.NifsKnowledgeSound`). The
random-oracle theorem `Lifecycle.RandomOracleKnowledge.knowledge_error_le`
justifies its per-visit error; see
[ROM_KNOWLEDGE_SOUNDNESS.md](ROM_KNOWLEDGE_SOUNDNESS.md) and
`formal/nightstream-fprime/TRUST_BOUNDARY.md`. The text below records the
retired boundary.

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
The real success event is `FiatShamirTransfer.RealSuccess`: the adversary
outputs a prior preimage whose prior-state link holds for the running statement
and the verifier context digest, the
actual `ProductionKey` NIFS verifier accepts, and the adversary supplies valid
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

The transfer assumption is stated over the real production key. Its statement
changed once since the 2026-09-11 approval: the owner decision of 2026-10-06,
below, adds the prior-state link and a verifier context digest to the success
event. What is proved is which data the key's transcript absorbs, and in which
order. `Lifecycle/TranscriptCoverage.lean` models each
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

Owner decision (2026-10-06): the success event requires the prior-state link.
`FiatShamirTransfer.RealSuccess` takes the verifier context digest and holds
only when the adversary outputs a well-formed prior preimage that hashes to the
absorbed prior digest, names that context, and has the NIFS running statement
as its running vector. No challenge depends on the running statement. Without
the link, an adversary could choose it after `γ` and keep the claimed sum, and
the transfer would have no useful instance. The preimage must be an output, not
merely exist: its initial state, current state and iteration are free, so some
well-formed preimage with any running vector hashes to almost every digest.
With the output link, a running statement chosen after the challenges needs a
preimage of the state hash for a digest that was absorbed before them: a first
preimage if none was known, a second preimage otherwise, or a collision found
in advance. The module moved to
`Export/Stage1/FiatShamirTransfer.lean`, because the link is a Layout
definition. The generic NIFS closure theorems (`NifsClosure`,
`NifsFiatShamir`, `NifsProviderLaw`, `NifsInvalidSource`) take the context
digest as a parameter. The HyperNova history theorems (`HyperNovaVisitedSecurity`,
`HyperNovaFalseAcceptance`) use the package's context digest, and
`HyperNovaRealInput.realSuccess_of_terminal` derives the link from terminal
acceptance for the decoded prior preimage (`HyperNovaRealInput.prior`).

Both native Rust PiCCS engines read the prior digest from the first fresh
public input with Lean's `decodeHash` formula, as `ProductionKey.priorDigest`
does. No check reads a running claim's frame (`fold_digest`), and the proof
codec does not carry it; the running instance holds only the 16 PiDEC children
and their openings. `prior_digest_comes_from_the_fresh_public_input` in
`neo-reductions` tests both engines, and the CI native fixture runs
`step_inputs` and the native NIFS verifier with noncanonical running frames.
`PerApplicationSecurity.replayInput_authority_identifies_or_collision` links the
older committed-statement reductions to this contract: equal replay authority
identifies the fresh statement and every SumCheck round polynomial, or exhibits
two statement call lists that reach one transcript state
(`TranscriptCoverage.RunCollision`). The transfer statement does not consume the
coverage results; they are proved properties of the transcript that the
assumption ranges over. The assumption above still covers the permutation, the
duplex construction, and the transfer bound.

The in-circuit hash class of attacks (Khovratovich–Rothblum–Soukhanov,
[ePrint 2025/118](https://eprint.iacr.org/2025/118); see also Fenzi,
[ePrint 2026/1838](https://eprint.iacr.org/2026/1838)) applies when the proven
relation evaluates the Fiat–Shamir hash. F′ does this by design: it recomputes
the NIFS transcript with the same Poseidon2 permutation. The published attacks
need a relation that the attacker shapes; here the verifier fixes the relation
through its package key. The transfer assumption must still hold with these
in-circuit hash calls. No theorem here proves that.

The transcript and the state hash `Poseidon2.hash` share the permutation and
the zero initial state. Only their leading tag words separate them. Two facts
follow from the definitions. First, the label `[1, 0]` of the first `α`
coordinate equals the hash's final padding, so the first word of that
coordinate is word 0 of `Poseidon2.hash` of the zero-padded statement chunks.
Second, `Poseidon2.hash` does not absorb the input length, so trailing zeros in
the last rate chunk do not change it. The state preimage therefore has a fixed
tag and, by `WellFormed`, a fixed total length; the transcript absorbs
fixed-length tags and labels, and each message as a length-prefixed block.

The separate finite sampler laws and `VerifierErrorBudget` are checked.
Under the stated per-call laws, the selected PiCCS test and sampler-abort
events have a union bound over arbitrary `n`. This excludes FS/hash/MSIS
attacks and the square-root extraction loss. Neither this bound nor the
independent-field sampler law establishes a complete deployed security bound.
