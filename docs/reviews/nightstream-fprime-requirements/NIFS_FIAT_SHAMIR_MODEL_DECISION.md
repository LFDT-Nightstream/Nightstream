# NIFS Fiat–Shamir model decision

Status: **theorem-to-code match pending; approval question deferred; no model selected**. Transcript review cut:
`773f3d0f29209b33e2325538d5f258f541569c25`, branch `nico/nifs-proof-links`.
This note addresses `N.security.fiat_shamir`. It adds no premise to Lean.
The finite-law milestone below records the coordinator's continuation from
`0fdd221fe342519440b486eb30c85de27c12d701`; its source and audit scope are
recorded in the main progress report.

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

The [current combined axiom gate](/tmp/nightstream-nifs-primitives-shortfall-axioms-2.log)
passed in 297 seconds, including the dependency rebuild. It covers 51 new
exports and their affected consumers, with 263 audit records. It uses only
`propext`, `Classical.choice`, and `Quot.sound`. The failed actual-field/batch
draft described below is outside this checked scope.

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

The further claim that the 54 coefficients are independent uniform symbols
conditional on bit-sampler success still needs its finite decoder proof.
Keep the abort outcome when transporting that claim to the challenge space.

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

### Remaining sampler consumers and error terms

| Obligation | Exact remaining statement |
|---|---|
| Actual 32-field consumer | Prove `SamplerShortfall.candidateWindow initial i = FieldShortfall.fieldCandidates (fieldWindow initial i)`, where `fieldWindow` reads lanes 0–3 of each of the eight actual pre-permutation states. The stopped draft contains this definition and proposed equality. The active package has no audited theorem for this additional link. |
| Batch event and finite union bound | Prove `sampleBatch initial count = none` iff some `Fin count` scalar returns `none`. For the separate uniform comparison `Fin 17 -> FieldShortfall.FieldWindow`, prove `a_batch <= 17*a_field <= 17*u`. The space has 544 field coordinates. A union bound needs per-coordinate event bounds; it does not require an added independence claim about actual scalar transcripts. |
| Output bias, including abort | Prove the product event/variation transport and the successful IID-bit decoder law. The proposed bounds are `32*(M-1)/(M*q)` for one scalar's candidate-law difference and `544*(M-1)/(M*q)` for the complete comparison batch. Single-pair transport and the scalar abort bound are checked; these product/output statements are not. |
| Retry and rewinding work | Bound failed, repeated, and aborting calls in the actual extractor program, using its symbolic query and work functions. Actual batch shortfall rejects that attempt. Any restart must be part of the constructed extractor and its proof. No retry policy, numerical retry bound, or fresh-state law is selected here. |
| Fiat–Shamir use | Supply the model, additive-schedule comparison, cache/trace law, aborting codec, and classical state-restoration knowledge transfer below. A scalar comparison bound cannot be assigned directly to deterministic Poseidon2 replay. |

In the fully independent batch comparison, the exact product identity
`a_batch = 1-(1-a_field)^17` is also unformalized. It is not needed to use the
proposed union bound. One comparison batch's errors do not account for an
adversary's search or the extractor's rewound calls.

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
It supplies no audited actual-field or batch theorem. All jobs used the
coordinator's single build queue under the 1500-second Lean cap.

## Exact transfer and extraction obligations

| Obligation | Required statement | Nature |
|---|---|---|
| Model scope | Define the selected setup, classical adversary, auxiliary information, adaptive statement/proof choices, and available transcript/permutation interfaces. Use one permutation consistently for all shared Poseidon2 uses. State how fixed public digests and preprocessing enter that experiment. | Owner model choice, then a mathematical experiment definition. |
| Statement and domain binding | The digest-only prefix identifies the verifier-selected key, profile, prior running statement, and fresh statement, or produces the existing named Poseidon2 binding/collision event. Prove the typed schedule encoding is unambiguous on the actual accepted domains. | Structural proof and a separately named hash premise; collision resistance alone is not challenge security. |
| Public-coin comparison | Derive the C coins and R candidate blocks in the chosen ideal experiment, with a joint law for outputs and successor states. Repeated queries return the same result. No fresh-independent-coin premise may be attached directly to deterministic replay. | Mathematical proof under the approved model. |
| Overwrite/additive match | Chiesa–Orrù uses overwrite absorption; Nightstream adds field words to the current rate lanes. Prove a transfer theorem for this additive schedule or a valid identification with the cited construction. A codec that subtracts the current state is state-dependent and is not automatically covered by a fixed message codec. | Missing mathematical transfer; no silent transcript change or assumed equivalence. |
| Sampler loss | Use the checked single-lane law and scalar abort bound. Complete the actual-field/batch consumers, product/output law, and all sampler invocations of the adversary and extractor under symbolic query/work bounds. Preserve aborts. | Mathematical proof; no new cryptographic premise. |
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

For the current scalar decoder, the proposed independent-field comparison
uses `32*(M-1)/(M*q)` for the product field-to-bits error and the checked
`u=choose(64,11)/65536^11` for scalar shortfall. Its required comparison with
a uniform strong-set scalar, in a common space containing abort, has target
bound `32*(M-1)/(M*q)+u`. The product variation and successful bit-decoder
proofs remain open. Transport into `eps_i` also needs the missing
codec-with-abort and state-transition proof. The final formula must include
adversary queries;
`544*(M-1)/(M*q)` from one honest batch cannot replace
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
checked status is recorded above. This worker ran no Lean/Rust command, test,
generator, backend, dependency change, commit, or publication. Protected
owner files and other workers' review files were not changed. Frozen Lean
sources were not read.
