# Compression of the final proof over Akita

## Status

Accepted by the owner on 2026-11-13:

- Akita (ePrint 2026/1983) is the polynomial commitment scheme of the
  compression.
- Poseidon2 makes each value that a circuit computes again. Keccak-family
  hashes are used only inside Akita: the Keccak-f duplex sponge, SHA3-256
  digests, the SHAKE128 session identifier and the SHAKE256 setup.
- No recursion over the final proof is planned.
- Folding stays. The prover carries only the accumulator between steps.
- The final proof must be a succinct argument of less than 200 KB.
- The final verifier can read the compiled circuit. It must not read
  gigabytes of key material, and it must not read witness openings.
- The matrices of the circuits are not committed. The verifier keeps its
  pass over the compiled circuit, because a commitment adds prover time to
  each `compress`.
- No Rust code of the compression is translated to Lean with Aeneas.
- Preprocessing commits the Ajtai key with Akita, and the verifier holds the
  commitment in a verification key.
- The verification key belongs to the generation of the circuit on the
  verifier's side. The verifier never takes it from a proof.

Accepted by the owner on 2026-11-16: a circuit whose proofs never reach a
refresh fold, such as each WASM size, has a final proof of two claims, and
its key holds the light relation only
([`wasm-light-compression.md`](./wasm-light-compression.md)).

Proposed: each other entry of this record. The owner did not select these
entries, and they wait for the owner's review. They include the form of the
key polynomial, the sum-check that reduces the key's part of the weights,
the row point, the schedule rows, the bytes of the key and their pin, and
the four lookup challenges.

## Problem

A proof that `Verifier::verify` accepts carries the witnesses of both slots
and of the fresh instance. A caller needs a proof without witnesses that a
verifier checks against the same statement `(i, z0, zi)`. The earlier
Spartan, WHIR and shrink path is removed
([`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md)).

Akita commits and opens a multilinear polynomial. It does not say which
statement to prove, how to prove a norm bound, or how to join its transcript
to a Poseidon2 transcript. This record fixes these choices.

Each CE claim has an Ajtai commitment, so a verifier of the claims needs the
Ajtai key at the point of its check. For the reference application, the key
that the circuits read has `18 · 2^19` ring elements of 64 coefficients. The
owner's requirement is that the verifier reads none of them.

## SuperNeo

SuperNeo v1.2 gives the folding scheme and the CE relation (Definition 21).
This record uses no SuperNeo result about compression. The relations that
the argument targets are the Lean ones: `Spec.CE.Holds` for a slot,
`Spec.CCS.Holds` for the fresh instance, and the verifier
`Export.BoundedDepth.RunCompleteness.Verifies`.

## Decision

`Prover::compression_setup` makes the prover's key once,
`Verifier::compression_key` gives the verification key, `Prover::compress`
makes a final proof from a proof with witnesses, and
`Verifier::verify_final` checks it. [SPEC.md](../SPEC.md) Algorithms 9, 10
and 11 are the rule, with the parameters of section 3 and the transcript of
section 5.4. The final proof has the statement, the slot claims of its mode
with their depths, the CE claim of the fresh instance, one Π_CCS sum-check,
and one argument over Akita. It has no witness and cannot be extended. The
mode has both slots, or slot 0 alone for a circuit whose proofs never reach
a refresh fold (`Export.BoundedDepth.FinalVerifier.Mode`).

| Item | Value |
|---|---|
| Commitment scheme | Akita, git revision `d50f700fb1a292395a717cef0e6f437c7ce2cdb1`, crate version 0.1.0 |
| Akita configuration | `RecursiveCommitmentConfig<proof_optimized::fp64::Dense>`: base field `F_q`, extension `F_q[u]/(u² − 2)`, which is `K`, schedules that offload Akita's own setup |
| Schedule files | `crates/ns-compress/artifacts/key20.aks`, SHA-256 `8ee08715b2120cfb576227f195fde041e5e1dd828511ca5e75d5df73d4c097b5`, and `crates/ns-compress/artifacts/key30.aks`, SHA-256 `b71861d25a4cb23b8eb68b20365bc5b76336224dcacdcada86242c9d9c084f94`, from Akita's planner at the pinned revision (`scripts/akita_schedules.sh`) |
| Committed polynomials | The key polynomial: the Ajtai key, once, in preprocessing. `W`: each witness, then the counts of each table. `V`: the lookup inverses |
| Lookup table of slot `s` | `[−accumulatorBound(depth_s), accumulatorBound(depth_s)]`, at most 20,737 values |
| Lookup challenges `L` | 4 challenges of `F_q`, one for each column of `V` |
| Transcript domains | `Nightstream/OptionB/compress/ccs/v1` for Π_CCS, `Nightstream/compress/akita/v2` for the argument |
| Byte tags | `NS-OPTIONB-FIN-2` for a final proof of both slots, `NS-OPTIONB-FIN-L` for a final proof of slot 0 alone, `NS-COMPRESS-VK-1` for a verification key |
| Rust toolchain | 1.99.1, which Akita requires |
| `jolt-field` | git revision `25d87ba68bde9f4a85ac986f5487b2e7b7db2699`, Akita's own pin |

### The choices

1. **No shrink layer.** The verifier checks the argument directly. The size
   of an Akita opening changes little with the number of variables, so no
   circuit of a verifier is necessary.
2. **Three or two CE claims.** The terminal check of Algorithm 6 has two
   slot accumulators and one fresh instance. Each slot accumulator is a CE
   claim (`Lifecycle.BoundedDepth.SlotClaim`). One Π_CCS run with no running
   claim turns the fresh instance into a CE claim at a random point. This run
   uses the Π_CCS code of the folds. In the light mode, slot 1 is the default
   slot at depth 0, which the verifier builds, so the argument has slot 0
   and the fresh instance of the light circuit, and the key holds the light
   relation only.
3. **Π_CCS binds `W` first.** The Π_CCS transcript absorbs the Akita
   commitment of `W` before its first challenge.
4. **Ring equations at one point.** The Ajtai opening, `Eval_K` and each
   `Eval_A` of a claim are equations in `K[X]/(X^64 + 1)`. Two different
   sides agree at a random `δ ∈ K` with a probability of at most `63/|K|`.
   So each equation is one inner product of `W` with public weights, and the
   proof has no quotient polynomial.
5. **Exact lookup table.** The slot check of Lean is `Spec.normBounded` at
   the bound `accumulatorBound j + 1` for depth `j`
   (`Export.BoundedDepth.StepExtend.Honest`,
   `Spec.GlobalParams.accumulatorBound`). A range of a power of two would
   accept a larger norm than Algorithm 6 accepts.
6. **Lookup challenges of the base field, with repetitions.** Akita commits
   a polynomial over `F_q`, so the inverses and the challenge `τ` are in
   `F_q`. One challenge gives the error `(N + T)/q` for `N` cells and `T`
   table values, and `L = 4` independent challenges give `((N + T)/q)^4`,
   which is below each other term.
7. **Counts are cells of `W`.** `W` is committed before `τ`, so the counts
   need no third commitment. `V` depends on `τ`, so a proof carries two
   commitments. The commitment of the key polynomial is in the verification
   key.
8. **The committed key.** The key's part of the public weights is a sum over
   the key elements that the claims read. The proof carries this sum as a
   claim of the prover. A second sum-check, of degree 2 over the lanes and
   the blocks, reduces the claim to one value of the key polynomial, and
   Akita opens that value against the commitment of the verification key.
   So the verifier reads no key element. One weight batches the three uses
   of the key with the coefficients of the first sum-check.
9. **The rows need no round.** The powers of `ω` that batch the `κ` rows are
   a scaled equality table, so the batch of the rows is a scale times the
   key polynomial at a row point. No proof exists for 7 values of `ω`, with
   probability `7/q²`. `compress` then returns an error.
10. **One key for each circuit size.** The key polynomial is a function of
    the public seeds and of the block count as a power of two only. So each
    circuit of one size has the same verification key. The block count is
    that of the larger relation, or of the light relation in the light
    mode.
11. **One Akita proof.** One proof opens the key polynomial, `W` and `V`.
    The schedule rows offload Akita's setup (ePrint 2026/1983, Section 9),
    and the verification key holds Akita's commitments of the setup
    prefixes.
12. **Transcript.** The Poseidon2 transcript of the Π_CCS run makes each
    challenge of the argument. It absorbs the bytes of the verification key,
    the claims and the bytes of each Akita commitment. Akita makes its own
    challenges from its Keccak sponge, which no circuit computes.
13. **The verification key is pinned.** The code pins the bytes of the key
    of each block count that has schedule rows, and a test computes each
    pinned key again from the public seeds. `Verifier::compression_key`
    gives the pinned key, and `Verifier::decode_compression_key` accepts
    loaded bytes only when they equal that key. Under another key Akita
    gives no binding, so the decoder does not accept a key whose form alone
    is correct.
14. **The Akita schedule comes from the verifier.** The verifier takes the
    number of variables of each polynomial from its key and its own layout
    of the claims, and the schedule row from the embedded files. A value
    that the proof carries is not authority.
15. **The fresh zero tail.** The argument commits only the ring blocks of
    each circuit's package. It does not prove that the fresh witness is zero
    from the width to the end of its last block.
    `Lifecycle.BoundedDepth.BlockSupport.Supported` requires zero only on a
    block that starts at or after the width.
16. **No zero knowledge.** Akita is not zero-knowledge, and the proof
    carries the claims.

Two alternatives were not selected. The estimate for a final proof over WHIR
with the same method was 300 to 450 KB, which is above the owner's limit. A
commitment to the matrices of the circuits removes the verifier's pass over
the rows, but each `compress` must then also prove the matrices' part of the
public weights.

### Error and premises

[SPEC.md](../SPEC.md) section 3 gives the error terms of the argument and
the gate that Rust computes from them (`ns_compress::error_bits`). Lean
bounds the argument in a model with ideal Akita openings: a prover with `Q`
queries has an error of at most `Q · error`, with `error < 2^−119` at the
production shapes (`Spec.Compression.Verifier.soundness`,
`Spec.Compression.Verifier.production_error_lt`). This is not a security
level of the final proof. It does not include:

- the knowledge error of the Akita opening (`[AKITA]`). The documentation of
  the Akita code states a target of a 128-bit Fiat–Shamir rate for each query
  in the classical random-oracle model, and it states no complete bound. The
  paper has a complete formula, that of Akita Theorem 10.46, and no evaluated
  end-to-end number for a schedule. For the claims that end in `_with_akita`,
  Akita Theorem 10.46 stays a premise;
- the error of the Π_CCS run of the fresh instance; and
- Module-SIS for the Akita commitments, which the pinned Akita code prices at
  169 bits of quantum Core-SVP cost in the ADPS16 model (policy
  `Quantum169BitADPS16`), and Module-SIS for the Ajtai commitments of the
  claims (`[MSIS]`). The 169 bits are not in the paper: its floor is 128 bits
  for each instance, more than 125 bits for the family, and at least 147 bits
  for the compression maps (Akita Appendix G). The estimate of the Akita code
  and the estimate of the main key are not in one model (Open points, the
  model of the Module-SIS price). For `key20.aks`, the instances are the 13
  instances of `Claims.Akita.key20_instance_list`.

Lean has claims in which the Akita openings are not ideal (`Claims.Akita`).
The main claim is `Claims.Akita.key20_verifier_sound`, from
`Export.BoundedDepth.FinalKey20.sound`, for the schedule of `key20.aks`. A
prover with at most `Q` queries to one uniform table of stage answers makes
the final verifier accept a statement that has no valid proof with probability
at most
`(Q + 1) · (runCost + 7424/|K| + 135 · p) + (Q + 1) · 144/budget + breakChance`.
`runCost` is the sum of the error bounds of one run of Π_CCS and of the
argument. `p` is the largest probability of one fold challenge, and the claim
has `p = 1/q²` for the Lean decoders. `budget` is the limit of each retry loop
of the extractor. `breakChance` is the `[MSIS]` term: the probability that the
extractor returns a solution of Module-SIS. The result has the composition of
the argument with the opening inside it, by two checkpoints in sequence: `W`
before the lookup challenges, then `W` and `V`. Its hypothesis
`Export.BoundedDepth.FinalKey20.Use` is the use with the key of `key20.aks`:
the claims pass the layout, the key polynomial has 512 blocks, and the opening
point of `V` has the zero count of the layout.

The bound is for one claim shape: one `pc` and one depth for each slot. The
claim is for a finite family of shapes and for the proof of zero steps. Its
bound `shapesBound` is the sum of the bound above over the shapes, so it has
the number of the shapes as a factor. The proof of zero steps adds no error
term. `Claims.Akita.key20_history_sound` and
`Claims.Akita.key20_history_knowledge` bound a false history in this game by
`shapesBound` plus the bound `walkBound` of the reverse extractor for the
depth. The memory results are not stated in this game.

The verifier of these results accepts an honest proof
(`Export.BoundedDepth.FinalKey20.accepts`), for each table of stage answers
that decode to fold challenges and to no challenge of the reject set of the
argument. This acceptance has two hypotheses that no Lean result gives: the
honest run of Π_CCS on the fresh instance (`leads`) and the fit of the
responses of the honest prover (`fit`). The completeness of Akita is not
perfect, and Lean has no bound for the probability that a response does not
fit. The paper has this as the response-model premise of Akita (152) and
(340), which it calls heuristic: its text "motivates this premise but does not
prove it" (Akita Appendix K). For the zero tables, the opening verifier
accepts with no fit condition (`Claims.Akita.key20_opening_accepts`).

The owner decided that these results are claims (recorded on 2026-11-20). They
are the claims of `Claims.Akita`, with the parts of the proof: Akita Theorem
10.17 (`Claims.Akita.checkpoint_extraction`), the fold results
(`Claims.Akita.fold_weak_binding`, `Claims.Akita.fold_extracts`), the units
(`Claims.Akita.short_units`) and the opening
(`Claims.Akita.key20_opening_sound`, `Claims.Akita.key20_opening_bound`). They
do not cover `key30.aks`. Their verifier is a Lean model that differs from the
pinned Akita code. The operator bound of the filtered challenge families is
their hypothesis: the paper proves it for its filter (Akita Theorem H.1), and
Lean has no model of that filter. So the claims that end in `_with_akita` keep
the premises of this section. They are the only claims that cover the shipped
key ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
item 7).

The argument has these premises about the verification key. The key is the
key of Algorithm 11, which the pin of choice 13 gives. The commitment of the
key binds the key polynomial. This is a Lean result under `[MSIS]`
(`Claims.Compression.key_binding`) for the certified relation: Akita
Definition 10.21 with the weak certificate of Akita Definition 10.22. Akita
Definition 10.21 has no norm bound on the source blocks, so it is not
binding alone. The result is for a key commitment that is the output of
`Spec.Compression.Akita.commit` on the key polynomial, and no Lean result
shows that the pinned key holds this output. Two parts stay premises under
`[AKITA]`: the opening is knowledge sound with an extractor that returns the
root weak opening, and the offloading of the setup prefix of `key30.aks` is
sound. With these parts, the value that Akita opens is the value of the key
polynomial. The verifier never reads the polynomial
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), adversary
6 and trust item 7).

### Consequences

- The verifier reads no element of the Ajtai key. It is not succinct: its
  time is linear in the matrices of the circuits.
- A change of the Akita revision, of the schedule files or of the seeds
  changes the verification key, so the pinned bytes must change with it.
- The schedule files have rows only for a key polynomial of `2^9` blocks,
  for tests, and of `2^19` blocks. A circuit of another size needs new rows
  from `scripts/akita_schedules.sh`.
- Akita is version 0.1.0, has no audit, has no proof in the quantum
  random-oracle model, and gives no compatibility between versions.
- The workspace has a second hash family in a proof path, Keccak, inside
  Akita only. The rule of `AGENTS.md` stays in force for the folding scheme
  and the step circuits.

### Open points

- A review of the soundness of the committed key by a party outside the
  project.
- A proof that the Rust layout and the Rust verifier are the Lean models.
  Only tests join them
  ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
  item 7).
- Security against a quantum adversary of the transcript hashes.
- A WASM build of the verifier.

These points are about the claims of `Claims.Akita` for `key20.aks`. Each
needs a decision of the owner ([security
model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust item 7, gives
the facts).

- The claim. Decided: the results are claims, and each hypothesis has a class
  in the premise table. For `key20.aks`, the claims have the premises
  `akita.theorem1046` and `composition` as theorems under their hypotheses.
  What stays open is to replace the claims that end in `_with_akita`, when a
  claim covers `key30.aks`. Two premise rows of the new claims are open: the
  operator bound of the filtered families, and the conditions
  `Export.BoundedDepth.FinalKey20.Opening.Drawn` of the final verifier, which
  also hold the Lean decoders and the certified decommitment `registered` of
  the key commitment. The 512 blocks of the key polynomial are of the class
  link. The term `breakChance` is `[MSIS]`. The uniform table of stage answers
  is `[KECCAK]`. The memory claims have no statement in this game.
- The zero count. The hypothesis `Export.BoundedDepth.FinalKey20.Use` gives
  the opening point of `V` the zero count
  `Export.BoundedDepth.FinalKey20.laidZeros` of the Lean layout. No Lean
  result compares this number with the Rust code.
- The acceptance. `Export.BoundedDepth.FinalKey20.accepts` has the hypotheses
  `leads` and `fit`, which no Lean result gives, so no closed accepting
  instance exists for the whole verifier.
- The model of the verifier. The Lean verifier differs from the pinned Akita
  code in these parts. At the levels with the squared-norm route, its norm
  check is a stage of its own after the checks of the fold, and the code
  merges the norm sum into a sum-check of the fold. A committed level has one
  added row for a zero cell after its last segment, and the quotient digits
  are after that cell. The transcript has no digests and no proof of work.
  Three more differences are with the paper. The Lean verifier has no
  rejection at a zero tensor factor, and the pinned code has none
  (`crates/akita-verifier/src/stages/opening_claims/extension_claim.rs` of the
  Akita code). It has one range sum-check where the paper has a product tree,
  at a certified base above 8. Its fold has no nonce of the response search. A
  change of the pinned Akita code or a change of the Lean model removes a
  difference. The decision is which one changes, for each difference.
- `key30.aks`. The result is for the file of the test key of `2^9` blocks. The
  file of the key of `2^19` blocks has coefficient packing at levels 2 and 3,
  two groups at level 1 and an offloaded setup prefix. The Lean tower has none
  of these. The alternatives are a Lean tower for this file, or a new schedule
  file that the Lean tower covers.

These points are about parameters that the sister project shares, because
it ports the same Akita code. Each needs a decision of the owner or a joint
decision ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md),
trust item 7, gives the facts).

- The stream model of the fold challenges. The sampler of the paper has
  "separately indexed oracle queries from a fixed group root" (Akita Appendix
  I). After the paper, the Akita code replaced it with one stream for a fold
  group (`specs/sparse-challenge-stream.md` of the Akita code), so the paper
  has no proof for the stream of the pinned code. The Lean extraction
  (`Spec.Checkpoint`) reads `[KECCAK]` thus: the squeeze run after a fixed
  prefix is one uniform output of independent slots. The alternative is a
  change of the code to one oracle input for each fold coordinate, as in the
  paper.
- The shapes of the compression maps `F_1` and `F_2`. A schedule file does not
  hold them. Lean takes them from the Akita code, and no check compares them.
  Lean and the pinned code have the ring degrees 64 and 32 for `q = 2^64 − 59`
  (`Spec.Compression.Akita.Key20.maps`,
  `crates/akita-params/src/compression.rs` of the Akita code). Table 10 of the
  paper (Akita Appendix G) has `d_1 = 32`, `m_1 = 2048`, `d_2 = 16`,
  `m_2 = 128`, the rank 1 and the bound 1 for this modulus, with a floor of
  128 bits. The page was read again, and this reading is confirmed. The
  difference has a cause. The Akita commit `dfdbd767` of 2026-10-20 raised the
  floor of each instance to 169 bits and doubled the degrees of the maps,
  because the degrees of the table do not reach 169 bits
  (`specs/sis-nist-level3-policy.md` of the Akita code). So Nightstream runs
  larger degrees than the table of the paper, for a higher floor, and the
  paper has no price for these maps. Ring descent
  (`Claims.Akita.module_sis_descent`) and a larger width
  (`Spec.Compression.Akita.ShortKernel.carry`) give the price of the
  first-stage cell of the table, 147.34 bits for `(32, 1, 2048, ∞, 1)`, to
  `F_1` and `H_1`, because `2 · 1024 = 2048`. A larger width gives it to `F_2`
  and `H_2`. Ring descent moves a price only from a smaller degree to a larger
  degree, and the reductions do not give the 169 bits to an instance. What
  stays open is a check that compares the shapes of Lean with the code.
- The model of the Module-SIS price. `Claims.Akita.key20_instance_list` has
  the 13 instances `(d, n, m, p, η)` of `key20.aks`: `(32, 1, 128, ∞, 1)`,
  `(64, 1, 1024, ∞, 1)`, `(64, 2, 77, ∞, 63)`, `(64, 3, 616, ∞, 63)`,
  `(64, 8, 256, 2, 2^46)`, `(64, 8, 512, 2, 2^45)`, `(128, 1, 1936, ∞, 7)`,
  `(128, 1, 1092, ∞, 31)`, `(128, 4, 512, ∞, 126852)`,
  `(128, 4, 256, 2, 2^44)`, `(512, 2, 56, ∞, 53477172)`,
  `(1024, 1, 224, ∞, 32505732)` and `(1024, 1, 56, ∞, 53477172)`. For `p = 2`
  the bound is on the square of the norm. `Claims.Akita.key20_instances`
  states that the list is complete for `breakChance`, and
  `formal/nightstream-fprime/tests/AkitaSchedules.lean` compares the list with
  the file. Lean does not prove the hardness of an instance. `key30.aks` has
  18 such instances, at the degrees 32 to 2048. They are from a script, and
  Lean does not hold them. The pinned code admits an instance only at 169.07
  bits or more of quantum ADPS16 Core-SVP cost, in a direct ℓ∞ attack model on
  the scalar instance `(n·d, m·d)`, with the estimator of the Akita authors. A
  run of this estimator in a copy of the pinned code reproduces the two cells
  of Table 10 of the paper and gives 171.99 bits or more for each ℓ∞ instance
  of both files. No estimate is independent of the Akita authors. The price
  does not include attacks on the ring or module structure, the loss of the
  reduction, or the sum over the instances that share one setup stream, which
  is about 170.9 bits for each file in that model. The record of the main key
  ([`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md), Security
  calculation) has 127.20 bits in the ADPS16 quantum model for the instance of
  rank 18. The Akita estimator gives this number, with the block size 480, on
  the Euclidean route at the length `β · √(m·d)`. That the record uses this
  route is an inference from the exact match, because the files of the sister
  project were not seen. The direct ℓ∞ model of the Akita table gives 154.23
  bits for the same instance. By the method of the main-key record, the lowest
  Akita instance of `key20.aks` is `B` of level 2, `(128, 1, 1092, ∞, 31)`, at
  110.77 bits, and 6 of the 46 ℓ∞ instances of the two files are below 128
  bits. So the two numbers of `[MSIS]` are not numbers of one model. The
  alternatives are the direct ℓ∞ model for the main key and for Akita, the
  Euclidean route with the factor `√(m·d)` for both, or two models with a
  stated reason for each. This record does not select one. An estimate of the
  instances of the list with `lattice-estimator` is also open.
- The pair `(λ_FS, Q_max)`. The Akita code declares neither number. It has a
  target rate of 128 bits for each query. The paper has a proof of work before
  selected algebraic challenge queries (Akita Section 10.1 and Appendix I),
  and the pinned code also has one for each fold group, which the paper does
  not have. So the budget condition S3 of Akita Definition 9.2 has no value to
  check. S2 and S3 are for the pair that an admitted schedule records in its
  digest, so a schedule with no recorded pair is not admissible, and Akita
  Theorem 10.46 as stated does not apply to the shipped files. The
  alternatives are a declared query budget, or a bound in the address-sum form
  of Akita (211), which can take the credit of the work.
- The price of `A` for the key commitment. The pinned Akita code prices each
  instance at 169 bits, and the floor of the paper is 128 bits (Akita Appendix
  G). The binding of the key compares an extracted opening with the honest
  decommitment of the key, at a mixed norm. The paper prices `A` on the
  coefficient route in the ℓ∞ norm at `2 · κ̄_1 · β̄_∞`, which is the Lean
  `η_A`, and on the squared-norm route at the least power of two at or above
  `64 · Γ² · S_max`. That the schedule file has this norm and this radius for
  `A` is a check of the file:
  `formal/nightstream-fprime/tests/AkitaSchedules.lean` compares both for
  `key20.aks`, and for `key30.aks` it compares only the bound of `A` of the
  key group. Lean proves that the norm of the reduction is at most the bound
  of the file (`Spec.Compression.Akita.Descriptor.key20_covers`,
  `Spec.Compression.Akita.Descriptor.key30_covers`).
- The proof of work and the probability `p`.
  `Export.BoundedDepth.FinalKey20.sound` takes the largest probability `p` of
  one fold challenge as a hypothesis, and the claims have `p = 1/q²`. Its
  bound on the stage errors of the opening is `7559/q²` for a decoder that is
  an injection of one slot of `K` into the challenge family, where `p = 1/q²`.
  No decoder from one slot has `p` at most `2^−128`, because `q²` is below
  `2^128`. The Akita code has a target of 128 bits for each query and a proof
  of work for each fold group, and the Lean model has no proof of work. The
  alternatives are challenge families with `p` at most `2^−128` without the
  work, or a Lean model of the work.
