# NIFS Fiat–Shamir model decision

Status: **theorem-to-code match pending; approval question deferred; no model selected**. Transcript review cut:
`773f3d0f29209b33e2325538d5f258f541569c25`, branch `nico/nifs-proof-links`.
This note addresses `N.security.fiat_shamir`. It adds no premise to Lean.
The current finite-law source is
`692641958134b46d021639aed088d5574e1c68ce`. Its source, reviews, passed checks
and retained attempts are in
[NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip](NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip).
The previous unit/output archive remains unchanged.

The contract is to identify the exact current computation, its approved
premises, and the missing security transfer. Success means a precise
conditional statement for owner review, before any later model decision.
No transcript, sampler, profile, query budget, or security level is selected
by this analysis.

## Candidate proof boundaries; no decision requested yet

The current source does not supply an approved Fiat–Shamir security model.
The owner has asked for the exact theorem-to-code match before any request
to approve a model. These are candidate boundaries for that analysis:

| Choice | Exact new premise | Proof work that remains |
|---|---|---|
| Ideal-permutation route | Model the selected width-8, rate-4, capacity-4 Goldilocks Poseidon2 permutation as one public ideal permutation, with classical forward and inverse oracle access. Keep the current additive absorption, statement digest, labels, state transitions, and 54-of-64 sampler. Any claim about the concrete fixed Poseidon2 implementation must retain a separate, explicit premise that transfers the resulting knowledge claim to that implementation. | Prove the security transfer for this exact schedule, its statement initialization and shared hash uses; prove the sampler law; and connect a classical extractor with its query and work bounds. An ideal-permutation theorem alone is not a concrete Poseidon2 theorem. |
| Concrete-transcript route | Assume a classical knowledge reduction for this exact fixed-key NIFS verifier and Poseidon2 transcript, including the bounded sampler. The premise must give an extractor, a success-loss function, and expected-work obligations; it must preserve the selected input, source order, actual final child witnesses, and same commitment projection. | Connect that named premise to actual acceptance and the existing C/R/D witness consumers. Keep the premise visible. No random-oracle conclusion or numerical security bound follows from its name. |

No approval question is active in this note. First identify the exact
absorption, initialization, codec, state-restoration, and error interfaces.
Then present one integrated conditional statement for owner review. Until
that work is complete, the leaf stays open. Finite sampling facts do not
require a new security model.

## Authority already present

- [The owner architecture](/home/nicoarq/develop/Nightstream-nifs-proof-links/FPRIME_LEAN_ARCHITECTURE_SPEC.md:62)
  fixes Goldilocks, `b=2`, `k_rho=16`, `B=65536`, one fresh source, 16 running
  sources, 17 ordered R inputs, 16 D children, 14 matrices, and Poseidon2-only
  binding. The current [shape](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Types.lean:26)
  has 28 C rounds.
- [The goal](/home/nicoarq/develop/Nightstream-nifs-proof-links/FPRIME_STAGE1_GOAL.md:242)
  permits an explicit SuperNeo soundness assumption for missing reductions.
  It does not specify an oracle distribution, reprogramming interface, or
  transfer loss for this transcript. The [architecture security boundary](/home/nicoarq/develop/Nightstream-nifs-proof-links/FPRIME_LEAN_ARCHITECTURE_SPEC.md:226)
  names Fiat–Shamir and sampling separately from collision resistance.
- [The approved fixed-matrix MSIS premise](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md:30)
  applies to the selected public-seed matrix and strict norm 113246208. It
  supplies no numerical success bound and does not imply a hash or oracle
  premise. The note expressly keeps Fiat–Shamir and sampling obligations.
- The normative [HyperNova Appendix B](/home/nicoarq/develop/Nightstream/docs/hypernova-paper/26_B_Achieving_non_interactivity_for_multi_folding_schemes.md:1)
  requires an applicable knowledge theorem for the exact scheduled protocol
  in the random-oracle model. For a concrete hash, it requires explicit
  knowledge and statement-binding assumptions. Public-coin syntax alone
  does not prove the transfer. The normative [SuperNeo preliminaries](/home/nicoarq/develop/Nightstream/docs/superneo-paper-v1_1/04_preliminaries.md:3)
  use classical adversaries and rewinding; they do not prove QROM security.
- [The current progress record](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_PROOF_LINKS.md:638)
  states that model selection is pending. The older
  [security contract](/home/nicoarq/develop/Nightstream-nifs-proof-links/protocol-contract/security-reduction.md:59)
  uses `k_rho=14`, 15 sources, 24 rounds, another sampler, and numerical query
  limits. Those choices do not apply to this source cut.

## Exact computation to which the premise must apply

[Poseidon2](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Poseidon2.lean:4)
uses `q=18446744069414584321`, eight field lanes, rate four, capacity four,
the `x^7` S-box, four initial full rounds, 22 partial rounds, four terminal
full rounds, and the stored constants and matrices. These exact constants
are part of the concrete instance.

[Transcript absorption](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Transcript.lean:23)
adds each rate block to the state, with zeroes in unused lanes, then applies
the permutation. A typed block has a field-word length prefix. A field squeeze
returns lane zero and then permutes. A `K=F[u]/(u^2-7)` squeeze performs two
successive field squeezes; it does not read two adjacent lanes of one state.

The actual C prefix starts at zero and absorbs the ASCII domain
`Nightstream/SuperNeo/PiCCS/digest-only/v1_1`, then the framed prior-state
digest, fresh commitment, and fresh public input. This prefix relies on the
pilot's recomputed prior-state digest to bind the selected key and complete
prior running statement. They are not separately reabsorbed by this C prefix.
See [the selected prefix owner](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:109)
and [the circuit statement owner](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/StatementAbsorption.lean:5).
The unused presence of `verifierInputBlocks` does not add its blocks to this
schedule.

C derives 28 alpha values with labels `[1,i]`, then gamma with `[2]`.
For each round `j`, C absorbs the framed round index and the 10 `K`
polynomial coefficients, then derives the challenge with `[3,j]`. It absorbs
all output Pad and matrix evaluations before R starts. These are 57 `K`
challenges in total. See [the label and round operations](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Transcript.lean:114)
and [the full-output absorber](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:131).

For R coordinate `i=0,...,16`, [the sampler](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Transcript.lean:141)
absorbs `[4,i]`. It then reads all four rate lanes and permutes, eight times.
Each canonical lane integer supplies bits 0–15, then bits 16–31. This yields
64 ordered candidates per coordinate. Reject 65535; map every other
candidate to `(candidate mod 5)-2`; retain the first 54 values. A shortfall
rejects the complete batch. State advancement uses all eight digest blocks,
including blocks after the first 54 accepted candidates. The next coordinate
uses that complete successor state. The abstract digest counter adds no
absorbed counter in this concrete instance.

The sampler therefore reads `17*8*4=544` field lanes. These counts come from
the current [production parameters](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/NonInteractive/PiRlcSampler/ProductionAlphabet.lean:47),
not an adversary budget. The challenge set has exactly `5^54` members. D
derives no further challenge. Native v1.1 [duplex operations](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/neo-transcript/src/poseidon2.rs:41)
and the [coefficient decoder](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/neo-reductions/src/common.rs:650)
implement these operations. Their execution agreement does not assign a
probability distribution to an adversary's accepted transcript.

## Sampler facts and checked scope

The finite comparison spaces give equal mass to every field or chunk
function in the stated domain. They assign no law to Poseidon2 outputs.
The actual decoder event and these probability comparisons have separate
checked owners:

| Owner | Checked result | Focused passing record |
|---|---|---|
| [FieldPairLaw](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/NonInteractive/PiRlcSampler/FieldPairLaw.lean) | Single-lane preimage counts, rational mixture law, sharp event/total-variation bound, and deterministic pair-decoder transport. | [Count check](/tmp/nightstream-nifs-field-pair-law-2.log) and [variation check](/tmp/nightstream-nifs-field-pair-tv-2.log), two seconds each. |
| [ShortfallBound](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/NonInteractive/PiRlcSampler/ShortfallBound.lean) | The 54-of-64 decoder fails iff at least 11 candidates reject. The IID-bit failure frequency is at most `choose(64,11)/65536^11`. | [Resumed round 2](/tmp/nightstream-nifs-shortfall-resume-2.log), two seconds. |
| [SamplerShortfall](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/SamplerShortfall.lean) | The actual `sampleScalar` uses the bounded decoder on its own `sourceAt` prefix; failure is exactly that prefix's 11-rejection event. The same decoder has the stated bound when its input is replaced by the explicit IID-bit comparison. | [Round 2](/tmp/nightstream-nifs-sampler-shortfall-2.log), two seconds. |
| [FieldShortfall](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/NonInteractive/PiRlcSampler/FieldShortfall.lean) | A structural bijection and rejection inclusion prove `a_field <= a_bits` for 32 IID uniform fields, retaining each ordered pair's dependence. The scalar abort bound has no added variation term. | [Round 3](/tmp/nightstream-nifs-field-shortfall-3.log), two seconds. |

The current additions are:

| Owner | Checked result | Focused passing record |
|---|---|---|
| `SamplerFieldShortfall` | Actual eight-state, 32-field window; exact scalar and selected batch failure events. | `nightstream-nifs-field-batch-consumer-1.log`, three seconds. |
| `FieldBatchShortfall` | The explicit 17-window uniform comparison has abort probability at most `17*u`. | `nightstream-nifs-field-batch-2.log` and the consumer check. |
| `FieldOutputLaw` | Every decoder event differs between 32 uniform fields and 64 uniform chunks by at most `32*(M-1)/(M*q)`, including abort. | `nightstream-nifs-field-output-2.log`, one second. |
| `BitOutputLaw` | Equal successful fibers, the full Option mixture, and the field-to-uniform-success error bound below. | `nightstream-nifs-bit-output-law-2.log`, three seconds. |
| `SamplerOutputLaw` | Actual scalar conversion and field-window equality; the same comparison bound on the selected scalar type. | `nightstream-nifs-sampler-output-consumer-1.log`, three seconds. |

The [current combined axiom gate](/tmp/nightstream-nifs-unit-output-axioms-1.log)
passed alone in 26 seconds. It covers 39 new exports and affected consumers,
with 302 audit records, using only `propext`, `Classical.choice` and
`Quot.sound`. The boundary gate passed. Independent source review found no
defect in these finite-law claims. The historical actual-field draft below
has now been completed in the active package; it remains unchanged as a
record of its earlier stopped attempt.

Let `M=2^32`, `h=2^16`, and `q=M*(M-1)+1`. For uniform `X` in `F`, write
`Y=X.val mod M`. Each nonzero `Y` has `M-1` preimages; zero has `M`.
Thus the ordered pair of 16-bit candidates has the exact law

```text
Law(Y) = (1-1/q) * Uniform(Fin M) + (1/q) * PointMass(0).
TV(Law(Y), Uniform(Fin M)) = (M-1)/(M*q).
```

The two candidates are therefore not independent uniform 16-bit values.
The excess pair `(0,0)` decodes to two coefficients `-2`. The existing
`acceptedFactorization` theorem proves equal residue counts among *uniform*
accepted chunks; it does not remove this field-to-bits bias.

For 64 independent uniform 16-bit candidates, the exact scalar-shortfall
probability has the following binomial formula. This identity is not yet
formalized and is not used by the checked upper bound.

```text
r = 1/h
a_bits = sum(j=11..64) choose(64,j) * r^j * (1-r)^(64-j).
```

Eleven rejections cause failure because the scalar needs 54 of 64 candidates.
`ShortfallBound` proves the event equivalence and covers every failure by a
set of eleven rejected positions. It counts the 53 free positions for each
set and applies a finite union bound:

```text
u = choose(64,11)/h^11
a_bits <= u.
```

`BitOutputLaw` now proves the successful-output law. A permutation of
accepted symbols, indexed by their accepted rank, preserves every rejection
flag and gives equal fibers for all `5^54` scalar outputs. Thus the exact
law on the common Option space is

```text
Law(bitDecode) = (1-a_bits)*UniformSome + a_bits*PointMass(none).
```

`UniformSome` is uniform on successful scalars and assigns zero mass to
abort. The checked field/bit coupling and the triangle inequality give,
for every event on the selected scalar type,

```text
|Pr[fieldDecode in event] - Pr[UniformSome in event]|
  <= 32*(M-1)/(M*q) + a_bits
  <= 32*(M-1)/(M*q) + u.
```

`SamplerOutputLaw` uses the actual `scalarOfList` conversion, proves that
its default is unreachable on a successful list, and identifies actual
`sampleScalar` with this decoder on its own field window. The frequency
bound still refers only to the explicit independent-field experiment.

For 32 independent uniform field lanes, let `[z^j]` mean coefficient `j`.
The exact shortfall law retains the dependence inside each pair:

```text
G(z) = (1-1/q) * ((1-r)+r*z)^2 + 1/q
a_field = sum(j=11..64) [z^j] G(z)^32
a_field <= a_bits
```

The polynomial-coefficient identity is not yet formalized. The inequality
is now checked. `FieldShortfall.laneCoupling` swaps the low-32 residue with
an auxiliary pair inside each complete residue block. At the single final
field value, the field pair is zero and the auxiliary pair is unchanged.
This is a finite bijection. Each field rejection is therefore also a
rejection in the coupled bit window. Its 32 copies give an injection between
the two failure spaces and prove

```text
a_field <= a_bits <= u.
```

This proof uses no field-sized enumeration and no arbitrary-event variation
bound. It does not prove that the successful field-decoded scalar is uniform.
The extra `(0,0)` mass still affects the output law.

### Sampler consumers and remaining transfer terms

| Obligation | Checked scope and remaining work |
|---|---|
| Actual 32-field consumer | Closed by `candidateWindow_eq_fieldCandidates` and `sampleScalar_eq_fieldDecode`, for every initial state and coordinate. |
| Batch event and finite union bound | Closed by `sampleBatch_none_iff_field_shortfall` and `iid_field_batch_shortfall_probability_le`. The uniform comparison has 17 windows and 544 field coordinates. No independence law is assigned to actual transcript states. |
| Scalar output bias, including abort | Closed in the stated comparison spaces by `boundedSample_event_frequency_eq_mixture` and `SamplerOutputLaw.field_output_event_error_le`. |
| Joint batch output law | Closed for 17 independent uniform 32-field windows by `SamplerBatchOutputLaw.field_batch_output_event_error_le`. The bound is `544*(M-1)/(M*q)+17*u`, against uniform successful ordered lists in Option. Actual output/state equalities are separate deterministic results. |
| Retry and rewinding work | Bound every failed, repeated and aborting call of the actual extractor under symbolic query/work bounds. No retry policy, numerical retry limit or fresh-state law is selected. |
| Fiat–Shamir use | Prove the exact additive schedule, initialization, total/aborting codec, cache/trace law and classical state-restoration transfer. The scalar comparison is not a distribution theorem for Poseidon2 replay. |

The optional exact identities for scalar binomial failure, the field
polynomial, and `a_batch = 1-(1-a_field)^17` remain unformalized. The checked
bounds do not need them. One comparison batch does not cover an adversary's
search or the extractor's rewound calls.

### Retained attempts and stopping record

The [earlier shortfall draft](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/drafts/NIFS_SHORTFALL_BOUND.lean.txt)
stopped after its three failed checks:
[round 1](/tmp/nightstream-nifs-shortfall-bound-1.log),
[round 2](/tmp/nightstream-nifs-shortfall-bound-2.log), and
[round 3](/tmp/nightstream-nifs-shortfall-bound-3.log). Its subtype-cardinality,
finite-sum, and recursion errors are a historical failed attempt. A later
authorized continuation closed the bound in the active `ShortfallBound`
module listed above. The old draft remains unchanged.

The further [actual-field/batch draft](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/drafts/NIFS_SAMPLER_FIELD_LINK.lean.txt)
stopped after the coordinator applied the project's three-round limit to
these three target invocations. [Link 1](/tmp/nightstream-nifs-sampler-field-link-1.log)
did not reach the new module because a root-owned work dependency failed.
[Link 2](/tmp/nightstream-nifs-sampler-field-link-2.log) reached it and failed
after 128 seconds, with a pair-predicate elaboration error and batch kernel
deep recursion. The coordinator measured 23,695,180 KB RSS at 90 seconds.
[Link 3](/tmp/nightstream-nifs-sampler-field-link-3.log) failed after 122 seconds.
The final errors are the option-match/sampleBatch step conversion, kernel
deep recursion, an `Option.noConfusion` universe mismatch, and the remaining
`Fin.castSucc`/`Fin.last` event equality. The coordinator measured
16,842,284 KB RSS at 46 seconds.

No fourth invocation or narrowed rebuild was run. No heartbeat or recursion
setting was raised. The complete final source was retained byte for byte
outside the active Lean package, and only its new active module was removed.
The draft has 5,978 bytes and SHA-256
`0d188007c24f907ee9055093522e39c0ff18527f86810d126912c007d0ebbf6d`.
That stopped draft supplied no audited theorem at its original cut.
A later continuation completed the actual-field and batch links in
`SamplerFieldShortfall`; the current passing records are listed above.
Its second resume avoided the large fixed-size kernel expansion by first
generalizing the Option values and successor state. The third resume passed.
No heartbeat or recursion setting was raised.

## Exact transfer and extraction obligations

| Obligation | Required statement | Nature |
|---|---|---|
| Model scope | Define the selected setup, classical adversary, auxiliary information, adaptive statement/proof choices, and available transcript/permutation interfaces. Use one permutation consistently for all shared Poseidon2 uses. State how fixed public digests and preprocessing enter that experiment. | Owner model choice, then a mathematical experiment definition. |
| Statement and domain binding | The digest-only prefix identifies the verifier-selected key, profile, prior running statement, and fresh statement, or produces the existing named Poseidon2 binding/collision event. Prove the typed schedule encoding is unambiguous on the actual accepted domains. | Structural proof and a separately named hash premise; collision resistance alone is not challenge security. |
| Public-coin comparison | Derive the C coins and R candidate blocks in the chosen ideal experiment, with a joint law for outputs and successor states. Repeated queries return the same result. No fresh-independent-coin premise may be attached directly to deterministic replay. | Mathematical proof under the approved model. |
| Overwrite/additive match | Chiesa–Orrù uses overwrite absorption; Nightstream adds field words to the current rate lanes. Prove a transfer theorem for this additive schedule or a valid identification with the cited construction. A codec that subtracts the current state is state-dependent and is not automatically covered by a fixed message codec. | Missing mathematical transfer; no silent transcript change or assumed equivalence. |
| Sampler loss | Use the checked scalar and independent-batch output laws, actual decoder links and 17-window abort bound. Connect them to the chosen event map and cover every adversary/extractor invocation under symbolic query/work bounds. Preserve aborts. | Mathematical proof; no new cryptographic premise. |
| C rewinding | Construct the causal oracle/program needed by the existing C extractor. Reprogramming must preserve each earlier message and the bound statement. The second execution must have the exact fresh-coin law used by the pair-agreement proof. | A Fiat–Shamir knowledge proof, not a replay lemma. |
| R coordinate forks | Obtain a base opening and each one-coordinate fork for the same actual 17-source batch. Changing one R coordinate in the current chained state changes later states; a simulator must preserve or reprogram the other coordinates consistently. Include failed, repeated, and aborting calls. | A Fiat–Shamir knowledge proof; the interactive coordinate algebra alone does not supply the oracle. |
| Witness and runtime composition | Reuse the actual D child-opening consumer and same-key commitment projection. Prove expected total work for the constructed oracle/extractor, including permutation simulation, queries, rewinds, sampler failures, and witness checks. | Mathematical work on the actual program; same-key MSIS hardness remains the approved external premise. |
| Concrete transfer | Justify or explicitly assume transfer from the chosen ideal experiment to this fixed Poseidon2 permutation, with any symbolic loss stated. A fixed public permutation is not a secret-key PRP game. | New external premise unless a suitable reduction is supplied and proved. |

The existing interactive targets are
`StrongExtraction.probability_and_expected_work`,
`PaperWeakExtraction.weak_success_bound`,
`CoordinateExtraction.bounds_and_openings`, and
`PaperCompositionProbability.source_success_ge_from_weak`.
Their C test loss is the exact `IndependentExecution.testError` expression:

```text
28*9/q^2 + (productionShape.jointCoefficientCount-1+28)/q^2.
```

Here `jointCoefficientCount=16*54*15+1+17=12978`, so the test loss is
`13257/q^2`. This value follows from the selected source, matrix, coefficient,
degree, and round counts; it is not a selected security level.

C keeps the square root of witness-disagreement probability plus that test
loss. R keeps the loss `17/5^54` and the interactive expected-call bound 18,
with its separate actual-work premises. These are interactive results.
They must not be relabelled as bounds for the Poseidon2 verifier. A chosen
transfer theorem must place its own losses at the correct stage; this note
does not assert that all losses can be added outside the C square root.

## Match to the full Chiesa–Orrù revision

This section uses the full [paper dated 2026-03-27](https://eprint.iacr.org/2025/536),
not the older search-indexed excerpt. The current definitions for rewinding
knowledge are 3.8 and 3.16. The current simplified error is equation (58).

| Paper interface | Current code | Required result |
|---|---|---|
| Alphabet `Σ`, rate `r`, capacity `c` | Goldilocks `F`, `r=4`, `c=4` | Alphabet and dimensions match. The concrete permutation remains fixed; the paper's ideal permutation is a distinct experiment. |
| Construction 3.3 overwrites rate entries | `Poseidon2.absorbBlock` adds each field word to its rate entry | No equality. A proof for additive absorption is missing. Replacing each message word by `stateLane+word` would make the encoder depend on state. Definition 4.1 only gives a fixed injective encoder per round. |
| `Start` puts `h(instance)` in capacity and zeroes the rate | The code starts at zero and absorbs the domain and framed public prefix with the same Poseidon2 permutation | No current initialization theorem. Remark 2.1 discusses shared-permutation instance hashing, but the exact preimage, state placement, shared queries, and resulting loss still need proof. It is not permission to add an independent hash. |
| Absorb and squeeze cursors control permutation calls | Each partial absorb call immediately permutes. Each C field squeeze reads lane zero, then permutes. R reads four lanes, then permutes, even after its final block | Value and successor-state matching are both missing. Reading adjacent lanes of one paper squeeze is not the current `squeezeK`. Adjacent absorbs in the paper can coalesce; current framed calls have fixed block boundaries. |
| Fixed injective `φ_i` and total `ψ_i` | Fixed field encodings, labels, and a decoder that can return sampler shortfall | Prove each message map and malformed-message rule. A partial decoder is not yet the paper's total codec. Keep its abort event in the comparison and runtime laws. |
| Each round has a prover message followed by verifier coins | C has consecutive labelled alpha/gamma challenges; R has 17 labelled coordinates; D has no challenge | Give an explicit event map with fixed singleton messages where needed, exact final-message treatment, and all state transitions. Do not infer this map from the name Fiat–Shamir. |
| IP relation and final acceptance bit | NIFS reduces source knowledge to its 16 child statements | Define the exact game that includes valid final child witnesses, or prove a theorem for reductions. Native NIFS acceptance alone does not supply those witnesses. Preserve the actual child and source identities. |
| Salt size `δ` is a natural number | No separate random salt is present | `δ=0` is the matching candidate and gives `δ*=0`; no salt addition is needed for this knowledge question. This does not establish zero knowledge. |
| Rewinding state-restoration knowledge | Existing C/R results analyze interactive oracle calls with their stated laws | Construct the adaptive prefix-query game of Definition 3.12 and its extractor under Definition 3.16. Honest replay and the interactive expected-call bound do not establish this property. |

The table is a theorem applicability check. It does not adopt the ideal
experiment. The pending additive-absorption proof cannot be replaced by the
general claim that a sponge is indistinguishable from a random oracle.

### Additive schedule and state-restoration review

The retained independent review gives a concrete deterministic candidate.
For a state `S=(S_R,S_C)` and a four-word padded block `b`, use
`E_S(b)=S_R+b`. Overwrite followed by permutation then agrees with the actual
additive block. The inverse subtracts the reconstructed prior rate state.
This is a triangular bijection for a fixed permutation and prior state;
it is not Definition 4.1's fixed message encoder.

The cursor schedule also matters. After translated absorption, a paper
squeeze of 12 field words can retain positions 0 and 4 to match one current
K squeeze and its eager final permutation. A 36-word comparison can retain
the first 32 words to match the eight R digest steps and final state. These
lengths are derived comparison candidates, not protocol changes or proved
Lean schedule equalities. Framed partial blocks must retain their padding
and call boundaries.

Claim 5.23 uses fixed message relabeling. A replacement needs a joint cache
invariant for the original prefix, encoded prefix, raw output and full
sponge state. Decoded C coins alone omit rate lanes and capacity values.
Image tests, prefix inversion, arbitrary forward/inverse query order,
repeated queries and all additional work must be covered. Initialization
with the actual shared Poseidon2 digest prefix remains a separate obligation.
The detailed interfaces and source hashes are retained in
`nightstream-nifs-co25-additive-review.md` inside the evidence archive.

The retained state-restoration review maps Definitions 3.12–3.16 to
`CausalExecution`, selected `StrongExtraction`, `PaperWeakOracle`,
`PaperWeakExtraction` and `PaperCompositionProbability`. The missing adapter
must preserve adaptive statement selection and cached prefix replies while
constructing the exact causal C calls and captured R/D suffix calls.
`FinalOutputSuccess` includes the same invocation's valid final child
witnesses. Public acceptance alone does not provide them. A terminal
witness-bearing game map is proposed but unproved. Its review is retained
as `nightstream-nifs-state-restoration-review.md` in the archive.

### Symbolic error and work interfaces

Use `t_h`, `t_p`, and `t_inv` for symbolic upper bounds on adversary queries
to the instance hash, permutation, and inverse permutation. Put

```text
S = t_h + t_p + t_inv
L_P = sum_i ceil(ell_P(i)/4)
L_V = sum_i ceil(ell_V(i)/4)
L = L_P + L_V.
```

The actual event map must determine the lengths. None has yet been chosen
for the mismatched schedule. Equation (5) of Lemma 5.1 gives the detailed
paper error, with `eps_i` the codec error for round `i`:

```text
eta_detail =
  (7*S^2 + 28*(L+1)*S + 14*(L+1)^2 - 3*S - 13*(L+1)) / (2*q^4)
  + t_p * max_i eps_i + sum_i eps_i.
```

The lemma also states `t_p >= max(L_P,L_V)`. This is a hypothesis on a
symbolic resource bound, not a deployment budget. The formula is a target
for the paper construction only. The additive schedule has not inherited
it.

Equation (58) and Theorem 6.2 state

```text
theta(T) = T
eta(T) = 25*T^2/q^4 + T*max_i eps_i + sum_i eps_i
failure' = failure + eta(T)
work' = work + H(T,n)
kappa_FS(T,n,failure) <= kappa_SR(0,T,n,failure') + eta(T)
ET_FS(T,n,failure,work) <= ET_SR(0,T,n,failure',work') + H(T,n).
```

The time expression has the form

```text
H(T,n) = T^2*log(T)*(n + 8*log(q)) + T*(codec work + salt bit length).
```

Section 5.4 requires the work of inverse message encoding and uniform
conditional preimage sampling for the challenge decoder. The displayed
Theorem 6.2 uses `t_psi` in its abbreviated cost; the proof uses `t_psi^-1`.
The Nightstream cost theorem must bound the actual operations, including
that preimage sampler. It must not account only for forward decoding.

For a checked use of the simplified `25*T^2/q^4` term, require both `S<=T`
and `L+1<=T`, or retain `eta_detail`. These conditions give a direct
derivation: drop the negative terms and bound the numerator by `49*T^2`.
The printed simplification citing `S<=T` alone does not supply the needed
bound on verifier work. This is a mathematical condition on the chosen
symbolic parameter, not a new numerical limit.

For the current scalar decoder, the independent-field comparison now proves
`32*(M-1)/(M*q)+u`, where `u=choose(64,11)/65536^11`. Its target is
`UniformSome`, not uniform on the whole Option type. The aborting decoder still cannot be substituted directly into Definition
4.1. The comparison-only totalization below keeps the target as Scalar;
its verifier adapter and inverse-fiber work still need proof. For `N=5^54`, uniform on
`Option Scalar` assigns abort mass `1/(N+1)` and is that distance from
`UniformSome`; choosing it would require the additional comparison term
and an interactive abort adapter. Neither that message-type choice nor the
adapter has been adopted or proved here.

Transport into `eps_i` needs the missing total/aborting codec and joint
state-transition proof. The final formula must cover adversary queries;
`544*(M-1)/(M*q)` from one comparison batch cannot replace
`T*max_i eps_i + sum_i eps_i`.

An abort is rejection, so it is not automatically an additional accepted
false-statement event. It affects completeness, simulated calls, and
extraction. The proof must state which of these events carries each abort
term. It must not condition on sampler success and discard its cost.

### Inverse-codec coupling that must be repaired

Claim 5.22, pages 49–50 of the inspected revision, uses
`TV(U(B), psi_inverse(U(A)))=0`, where `psi_inverse` samples uniformly from a
fiber. That equality requires balanced fiber sizes. The exact correct
single-query identity for a surjective decoder is

```text
TV(U(B), psi_inverse(U(A))) = TV(psi(U(B)), U(A)).
```

For Nightstream's low-32 projection, the right side is
`(M-1)/(M*q)`, and is positive. This follows from the fiber counts above.
Thus that zero-distance proof line cannot be used for this decoder.
A coupling of the *joint* raw and decoded values can retain a single codec
error per fresh query: in both laws the conditional raw value is uniform
within the same fiber. Proving that coupling, its repeated-query cache, and
its trace use is required before importing the claimed codec term. The
paper's final bound is not declared false by this local proof gap.

### Statement that can be made before model approval

If an exact additive-schedule transfer supplies the joint transcript and
trace law above; if the current aborting codec has the stated event and
conditional-preimage proofs; and if the witness-bearing NIFS game has
classical rewinding state-restoration knowledge with error `kappa_SR` and
work `ET_SR`, then the corresponding ideal experiment has the transferred
error and work bounds. Each "if" identifies mathematical work, not a newly
accepted external assumption. A claim for the fixed concrete Poseidon2
implementation would still need a separately stated concrete transfer
premise. There is no numerical security claim here.

The paper's query bound applies to each adversary run. The interactive R
bound of 18 expected calls concerns a different oracle and a different
experiment. An expected-time adversary with an unbounded query count needs
a proved stopped-execution or query-tail transfer; its mean cannot be used
as a per-run query bound.

## Independent batch output and actual state link

`BatchOutputLaw.independent_batch_event_error_le` proves finite product
transfer with both input cardinalities explicitly positive. The selected
`SamplerBatchOutputLaw.field_batch_output_event_error_le` compares 17
independent uniform 32-field windows with uniform successful ordered
17-scalar lists embedded in Option. All abort mass remains on the field
side. With `delta=(M-1)/(M*q)` and `u=choose(64,11)/65536^11`, every output
event differs by at most `17*(32*delta+u)=544*delta+17*u`. The separate
abort-only bound remains `a_batch<=17*u`.

`sampleBatch_ring_list_eq_fieldDecodeBatch` identifies the actual ring list
with the collected scalar list mapped through `embedScalar`, using that
sampler invocation's own field windows. The output/state theorem also
preserves the exact `stateAt specification initial count`. These equalities
hold for every initial state and count, preserve order and retain `none`.
They do not assign independent field distributions to Poseidon2 states.

The focused product and consumer checks passed in nine and two seconds.
The combined NIFS gate passed all 328 registered results in two seconds,
including these four new exports, with only the permitted axioms. An
independent source review found no defect within these stated claims.

The total/aborting codec, uniform verifier-message target, inverse-codec
work, additive/overwrite schedule and initialization match, joint cache
and successor-state law, and classical state-restoration extraction remain
open. Their symbolic query, retry, abort and rewinding costs remain open.
Transfer to fixed Poseidon2 still requires a precise separate premise.
UniformSome is not uniform on the Option message type. This single-batch
comparison gives no adaptive-query budget and no model approval.

## Scalarwise totalization for comparison only

`SamplerTotalizedOutputLaw` takes any supplied fallback Scalar. Its type
already enforces the selected alphabet. `totalizedFieldDecode` returns that
fallback on a scalar shortfall and otherwise keeps the decoded scalar.
`totalizedFieldBatch` applies this operation separately at each coordinate.
The actual protocol decoder and its rejection behavior are unchanged.

`scalar_event_error_le` proves the same `32*delta+u` bound against uniform
Scalar by precomposing the existing Option event. `batch_event_error_le`
then proves `544*delta+17*u` against uniform ordered Scalar^17 in the
independent field experiment. This is not replacement of a failed complete
batch with a fixed list: successful coordinates keep their values.
`sampleBatch_success_ring_list_eq` agrees with the actual ordered ring
list when `sampleBatch initial count = some batch`. It claims no equality
of failed traces or final states. The complete module passed its first
check in two seconds, and independent source review found no defect in
these three claims.

This comparison can keep the challenge type Scalar, with no
`1/(5^54+1)` Option-uniformity term. A later acceptance-inclusion proof
must use the same statement, proof and permutation calls and show that
actual acceptance implies acceptance by the totalized comparison verifier.
The actual early-abort path and the comparison's eager continuation are
different on failures; successful-list agreement alone is not that full
verifier theorem. The abort fibers are added to the fallback scalar's
fiber, so the conditional inverse sampler and its work need an exact proof.

The fixed codec, pure squeeze/absorption schedule, initialization, cache
and query law, state-restoration extraction and concrete Poseidon2 transfer
remain open. This mathematical comparison selects no new protocol behavior
or security model and supplies no complete Chiesa--Orru applicability claim.

The retained `nightstream-nifs-totalized-inverse-codec-plan.md` gives a
concrete next proof target for the 32-field projection: separate success
and unrestricted-abort dynamic programs, exact rectangle weights from the
field preimage counts, a constructive surjectivity witness, and a proposed
rank/unrank bijection. It does not claim any of those new algorithms are
verified. Counts fit below `2^2048` because there are 32 Goldilocks fields.
Stored access, integer operations, exact uniform-rank sampling and its work
remain explicit obligations. Fair-bit rejection has an expected-time
bound, not a finite worst-case bound. The complete squeeze representation
and its unused words still require the separate schedule/fiber match.

## Exact local field preimages — `4049d613`

`FieldPreimageRectangle` now proves the four candidate-class equivalences
and the exact low/high rectangle preimages. Reject has size 1, a specified
accepted residue has size 13107, all accepted candidates have size 65535,
and unrestricted candidates have size 65536. Each word has `2^32-1` common
field preimages; word zero has the extra field `q-1`. The class-restricted
count is `(2^32-1)*|A|*|B|+[0 in A and 0 in B]`. Rank/unrank are computable
arithmetic equivalences, with no enumeration or random-sampling claim. Raw
alphabet index zero is centered -2, not centered zero. All ten exports passed
the full module check and the 361-record NIFS axiom audit at `4049d6133972475eaa4cd61e27d447da48349392`.

The complete decoder success/abort fibers, totalized scalar inverse, exact
uniform-rank sampler and its expected work remain open. The verifier
comparison draft is also inactive: its third full check still had two
PiDEC decision/output congruence goals. Earlier depth errors were removed,
but this is not a proved acceptance inclusion. The complete draft and logs
are retained in `NIFS_MATRIX_AND_PREIMAGE_EVIDENCE.zip`.

A fresh check of the [17 August 2026 CFRG draft](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-fiat-shamir-03#section-8.4)
found no result that discharges the actual additive Poseidon2 schedule. Its
XOF interface and general security requirements do not replace the missing
schedule, cache, state-restoration or concrete-permutation transfer proofs.
No model, quantum claim or new protocol behavior is approved here.

## Complete 32-field scalar preimages — `ee24bd0c`

The mathematical success and unrestricted-abort recurrences now equal the
cardinalities of the existing bounded decoder's full field-window preimages.
The selected list-view theorem preserves all 64 low/high candidates from
32 fields. The exact-length scalar codec inverse and the consumer theorem
`SamplerFiberCount.totalized_fiber_card` establish, for every full scalar
`s` and fallback `f`, the exact count
`successCount 32 (List.ofFn s) + if s=f then abortCount 32 54 else 0`.
This adds every aborting window only to the fallback's fiber. It does not
restrict accepted symbols on aborting prefixes. Raw alphabet index zero
continues to mean centered -2. All fourteen exports passed the complete
consumer build and the 378-record NIFS audit at `ee24bd0c02930ac068941c9be4f92c2e6200b798`.

These recursive count specifications do not establish efficient table
execution. The next inverse obligations are positive-fiber witnesses, stored
DP value/work, global weighted rank/unrank, exact uniform-rank sampling with
an explicit time contract, joint resampling/cache laws and the 17-scalar
product transport. The actual sampler is unchanged; complete verifier
acceptance inclusion, additive schedule and initialization, state restoration
and fixed-Poseidon2 transfer remain open. Evidence and scoped reviews are in
`NIFS_ROWS_AND_FIBERS_EVIDENCE.zip`.

## Primary proof references and their limits

[Chiesa–Orrù, ePrint 2025/536](https://eprint.iacr.org/2025/536),
Definitions 4.1–4.2, Construction 4.3, and Theorem 6.2 are examined above.
The full 2026-03-27 text supersedes the older search excerpt for this match.

[SAFE, ePrint 2023/522](https://eprint.iacr.org/2023/522), is supporting
material for field encoding and input/output schedules. It does not supply
the missing NIFS knowledge theorem or an approved security model.

[Attema–Fehr–Klooß, Theorems 3–4](https://link.springer.com/article/10.1007/s00145-023-09478-y)
give a `(Q+1)` knowledge-error factor for the stated special-sound protocol
class in the random-oracle model, including distinct challenge spaces and
adaptive statements. Nightstream has not established those exact hypotheses
or the required oracle interface. Its Remark 11 also separates random-bit
outputs from uniform decoded challenges. The factor is not a Nightstream
bound until those conditions are proved.

[VCVio's stateful Sigma bridge](https://raw.githubusercontent.com/Verified-zkEVM/VCVio/main/VCVio/CryptoFoundations/FiatShamir/Sigma/Stateful/Bridge.lean)
is a reference for oracle bookkeeping, not a transfer theorem for this
multi-round NIFS. [ArkLib's oracle-reduction notes](https://verified-zkevm.github.io/ArkLib/blueprint/chap-oracle_reductions.html)
use a transcript-prefix random-oracle model and mark the relevant security
transfer proofs as work in progress. Neither supplies owner approval.

The sampler modules use the coordinator's shared Lean build queue; their
checked status is recorded above. The proof workers ran no Lean/Rust build; the coordinator ran the recorded
checks through the shared queue. Protected owner files were not changed.
Frozen Lean sources were not read, and no site was published.
