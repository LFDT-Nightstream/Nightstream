# Response to the Ironwood comparison

Reviewed 2026-09-12 against the Nightstream proof source through `30186671`
and the downloaded Zakura fork at `22dfee003b639eff660f68ea69a98a00409a9cb1`.
The proposed improvements contain useful ideas. The overall robustness ranking
is not established, and several claims about Nightstream are false or stale.
This review changes no cryptographic assumption or proof requirement.

## Corrections supported by the current source

| Review claim | Finding |
| --- | --- |
| Nightstream has no recursion/heartbeat overrides or `csimp` declarations. | False. The production source contains 65 `maxRecDepth` sites, two `maxHeartbeats` sites and 29 `csimp` attributes. Examples are `Spec/GoldilocksPrime.lean:42`, `Gadgets/Poseidon2/Permutation/Owned.lean:29`, and `Circuit/Basic.lean:115`. These recorded fixed-size debts are distinct from the axiom trust boundary. The completed R repair added none. |
| Nightstream has no module build-coverage check. | False for this source. `scripts/check-boundaries.sh:124` traverses the declared library/executable roots and their imports; line 165 rejects unreachable source modules. Reachability through an audit target does not imply inclusion in the ordinary library target. Both targets must pass. |
| The binding link is only a vacuous existence result. | The existence helper alone is insufficient to establish an efficient cryptographic reduction. The actual chain already uses `Lifecycle/Nifs/BindingReduction.run` and `runPair`, which emit an integer list. `bindingEvent_implies_success` links the event to that output; `BindingProbability.binding_le_success` bounds event probability by the implemented reduction's success. `SupportedExtraction.returned_source_bound_with_msis` consumes it. The `N.security.binding` record already cites these declarations and work bounds. The helper's docstring explicitly separates search-problem identification from executable reduction and work. |
| A data-returning definition prevents vacuous security statements. | It helps make the output inspectable, but is not sufficient. A function can ignore its input or require an impossible condition. The necessary contract links the actual input event, computed output, success predicate, probability and work. Replacing `Nonempty` with a choice-based definition would not supply an efficient algorithm. |
| The square-root term means a per-fold attack probability near `2^-57`. | That interpretation does not follow. `SupportedExtraction.returned_source_bound_with_msis` puts the square root in an extraction-success lower bound. It is a reduction loss, not an observed forgery probability or the verifier's per-call statistical test error. Compare bounds only after aligning events, models and resource limits. |
| Ironwood proves completeness nowhere outside an excluded contract. | The knowledge-soundness contract does exclude an automatic completeness conclusion, but the downloaded fork also contains `wideActionWitness_completeness_error_bound`. Its README marks that newer result as awaiting final build. Distinguish the soundness contract, later source and validated release evidence. |

The three-axiom audit remains a useful Nightstream property: the checked
exports use only `propext`, `Classical.choice` and `Quot.sound`. It does not
establish zero trust in the whole executable toolchain or universal Rust
correctness. An elaboration-limit override does not add an axiom. A proved
`csimp` equality is also different from an unchecked compiled substitution.
The review's combined table merges these separate questions.

## Findings to retain

Ironwood's explicit trust census, computability checks, direct endpoint pins,
and readable theorem/assumption guides are useful references. Our reachability
check does not by itself prove direct audit coverage of every intended public
endpoint. Inspect that specific gap before adapting a tool, rather than
replacing the current audit system or adding a duplicate coverage checker.

Our Lean-to-package direction reduces one source of transcription drift. Rust
still implements the loader, interpreter, evaluator and protocol execution;
package identity alone does not establish their universal correctness. The
retained Ironwood circuit-fixture provenance does disclose unpublished dump
instrumentation. That is a specific reproducibility limitation, not evidence
that its whole verifier development is weaker.

The fingerprint theorem is also narrower than universal Rust/Lean equivalence.
[`Fingerprint/Match.lean:127`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Fingerprint/Match.lean#L127)
lists external premises, including capture faithfulness, the sampled-point law,
the declared degree class, byte decoding and compiler trust. Its four captures
do not prove equality on arbitrary Rust inputs. The guide states that limit.
The ledger knowledge contract also separates witness-level capstones from the
deployed circuit-to-ledger extraction handoff. Thus the review overstates both
the universal Rust connection and complete system-level closure.

The tighter extraction and Fiat–Shamir results in Ironwood are real results
under their stated models. Its Halo2/IPA algebraic-group result is not a
Module-SIS/SuperNeo theorem. Its numerical remainder cannot be compared with
our reduction loss as a common security score. An advantage function also
remains an external hardness premise, even when its arguments are concrete.
The square-root loss can weaken the concrete guarantee that our reduction
certifies. Correcting the review's attack-probability interpretation does not
remove the need to assess that loss in a complete, useful security bound.
The primitive choice alone does not establish that a fixed-key MSIS break is
less serious than a fixed-group discrete-log break.

The [download review](IRONWOOD_REUSE.md) records the exact fork, source,
licenses and trust conditions. Its pinned aggregate Lean CI was not green;
the formalization job was cancelled while fixture CI passed. This is not
evidence that a theorem is false, and it is not a complete validation result.

## Fiat–Shamir and scope

Our approved transfer contract is stronger than merely assuming a concrete
hash behaves as a random oracle. It remains a material explicit assumption;
the review is right to ask whether a standard transfer result can discharge
part of it.

[Attema–Fehr–Klooß](https://ir.cwi.nl/pub/32771/) gives a linear-in-query loss
for the stated class of multi-round special-sound protocols. Being a
multi-round sum-check protocol alone does not establish all of that theorem's
hypotheses for our composed folding experiment. We still need the exact
special-soundness/extraction interface, challenge laws, oracle queries,
encoding/domain separation, sampler aborts and resource correspondence.
The additive Poseidon2 sponge is not automatically the random-oracle execution
in that theorem. Its knowledge-error statement is not automatically the
inequality required by our `g`/`deltaFS` interface.

Ironwood's Action capstone proves a result for its computed adaptive algebraic
adversary family, DLOG profile and generator-oracle setup. Its verifier guide
keeps the concrete BLAKE2b implementation and byte encodings outside that
theorem. This is a useful in-model transfer result; it is not a proof of a
concrete hash or a generic transfer theorem for our additive transcript.

Do not fill those functions with attractive standard formulas before checking
that correspondence. A bounded applicability review of the named theorem and
any existing Lean formalization is reasonable if requested. The supplied
comparison does not itself authorize reversing the owner's approved scope or
starting a new Fiat–Shamir formalization project.

The same applies to a proposed linear-loss port: establish its exact theorem,
conditions and affected consumers before calling it available or in progress.
No estimate such as “one checkpoint” or “month or season” is established by
this review.

## Attribution

Keep upstream notices and credit actual reused source to Zakura contributors,
including Tal Derei, and the Zcash Protocol Developers and other contributors.
The downloaded source is dual Apache-2.0/MIT. Credit zkSecurity's Clean if code
is adapted from it. Similar circuit-contract ideas alone do not establish
source copying; the claimed field-for-field provenance was not demonstrated
by the supplied review. The proof-map design inspiration is credited
separately to the Ironwood documentation.

The practical recommendation is to retain our package authority direction
and completeness results, make the existing computed reduction more visible,
and assess specific missing audit checks. Full Fiat–Shamir transfer remains
a separate, explicit scope decision. There is no supported single ranking
of the two implementations' overall robustness from these observations.
