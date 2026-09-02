# Open Lean refactor issues

## 1. Remove duplicate PiCCS statement absorption

**Owner decision.** Accepted: use the digest-only schedule below. The issue
remains open until every closing condition is proved and executed.

**Historical problem.** The pilot already recomputed a Poseidon2 digest over
the complete 16-instance running state. PiCCS then absorbed the same running
values again before deriving `α`, `γ`, and the SumCheck challenges.

That schedule used 10,298,432 rows. Approximately 10.1 million processed
running values already bound by the pilot digest. The accepted digest-only
schedule now uses 224,368 statement-absorption rows. The concrete application
has exact joint domain 264,627,486, which is below the approved `2^28` limit.

**Required change.** Initialize the PiCCS transcript from:

- its domain tag;
- the pilot-recomputed prior-state digest;
- the fresh commitment and public input.

Do not absorb the complete running vector or repeat its point, `Eval_K`, and
`Eval_A` claims. Remove the old schedule instead of keeping a second path.

**Safety conditions.** Lean must prove that:

- one domain-separated verifier-context digest covers the exact profile,
  transcript schedule, logical relation and application identity, and all
  static NIFS and commitment-key material;
- the package rows enforce that context digest while the loader separately
  pins the sealed package identity; no unproved equality or self-reference is
  used between the two digests;
- the pilot recomputes the digest from the same authoritative running values;
- parent wiring passes that digest to PiCCS;
- the canonical state encoding is injective on well-formed preimages and no
  valid encoding is a trailing-zero extension of another valid encoding;
- the digest preimage keeps `Eval_K` and `Eval_A` separate;
- the fresh statement is absorbed directly;
- the committed-statement reduction accounts for context binding, canonical
  decoding, state-hash collision, the exact multi-round Fiat--Shamir schedule,
  and the loss of full-absorption defense in depth; and
- the production verifier prevents circuit selection and uses only the
  identity-pinned package relation.

**Impact.** This changes the protocol identity, transcript vectors, emitted
package, and Rust transcript tests. The SuperNeo v1_1 PiCCS formulas in
`paper_exact` and `optimized` do not change.

**Closed when.** Lean proves the new transcript sound and complete, the layout
and joint-domain theorems use it, both Rust engines match its vectors, and the
real prover and verifier use the new package identity.
