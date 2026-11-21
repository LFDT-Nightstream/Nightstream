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
`(Q + 1) · (runCost + 317479/(1024·|K|) + 135 · p) + (Q + 1) · 144/budget +
breakChance`. `runCost` is the sum of the error bounds of one run of Π_CCS and
of the argument. `p` is the largest probability of one fold challenge, and the
claim has `p = 1/q²` for the Lean decoders. Each error of a check of the
opening has the factor `2^−g` for the work bits `g` of its site in the code,
and the fold term has no work credit (difference (d)). `budget` is the limit of each retry loop
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
prove it" (Akita Appendix K). The opening verifier accepts the proof of the honest prover for each family
of tables, and for `key30.aks` for each setup prefix, when the fit conditions
hold (`Claims.Akita.key20_opening_accepts`,
`Claims.AkitaKey30.key30_opening_accepts`, the hypothesis `fit`). Lean proves
the fit conditions in two cases: for zero tables and a zero setup prefix with
each decoder of the fold challenges
(`Spec.Compression.Akita.Key20.fit_zero`,
`Spec.Compression.Akita.Key30.fit_zero`), and for each setup prefix and each
family of tables with decoders that give the fold challenge 0
(`Spec.Compression.Akita.Key20.fit_quiet`,
`Spec.Compression.Akita.Key30.fit_quiet`). For the challenge families of the
schedule, the sizes do not give the fit conditions. An example is the group
of the setup prefix at level 1 of `key30.aks`: 128 blocks, challenges with 31
coefficients `±1` and digits up to 128 give only `‖z‖_∞ ≤ 507904`, and the
limit of `Enc` for the base 16 with 4 digits is 34952. The honest prover of
the Akita code has a search over a nonce for a response that fits
(`crates/akita-prover/src/protocol/fold_grind.rs`), and the Lean model has
no nonce.

The owner decided that these results are claims (recorded on 2026-11-20). They
are the claims of `Claims.Akita`, with the parts of the proof: Akita Theorem
10.17 (`Claims.Akita.checkpoint_extraction`), the fold results
(`Claims.Akita.fold_weak_binding`, `Claims.Akita.fold_extracts`), the units
(`Claims.Akita.short_units`) and the opening
(`Claims.Akita.key20_opening_sound`, `Claims.Akita.key20_opening_bound`). The
claims of `Claims.AkitaKey30` are the same results for `key30.aks` (section
"The results for `key30.aks`"). Their verifier is a Lean model that differs
from the pinned Akita code. The filters of the challenge families are the
predicate `accept_strict_parts` of the pinned code
(`Spec.Compression.Akita.TrigTable.codeAccepts`). Lean models the tables of
`build_tables` over the integers and proves for them the error bound of Akita
Appendix H.1 (`Spec.Compression.Akita.TrigTable.code_accurate`), so the operator bound of
the filtered families is a Lean theorem (Akita Theorem H.1, right inclusion,
with the spectral formula (287): `Spec.Compression.Akita.TrigTable.mulBound`,
`Spec.Compression.Akita.TrigTable.filtered_code`). The claims that end in
`_with_akita` do not use these results, so they keep the premises of this
section ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md),
trust item 7).

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

### Differences between the paper, the code and the Lean model

These are the differences that the security model had in trust item 7, (a) to
(j), the differences of the opening with an offloaded root, (k) to (o), and a
difference of the relation rows, (p). The section Open points lists the points
that need a decision.

- (a) The extraction of the paper uses one oracle output for each fold
  coordinate: its sampler has "separately indexed oracle queries from a fixed
  group root" (Akita Appendix I). After the paper, the pinned code replaced
  this sampler with one stream for a fold group, with no absorb between the
  coordinates (`specs/sparse-challenge-stream.md` of the Akita code). So the
  paper has no proof for the stream of the pinned code. This is the reason for
  the slot reading of `[KECCAK]` (security model, section Assumptions): the
  model of `Spec.Checkpoint` is one uniform table, the answer at a stage input
  is one output of independent slots, and the extractor draws one slot again
  and keeps the others (`Spec.Checkpoint.Protocol`). For the filtered challenge
  families of the ℓ2 levels, a slot is the j-th accepted candidate. In
  `Export.BoundedDepth.FinalKey20.sound`, the Poseidon2 transcript of the
  argument and the Keccak transcript of Akita are one table. No Lean statement
  joins the two hash functions of the code to it. Such a statement would have
  the shape of `Spec.Sponge.Transfer`. The session bytes bind the two
  transcripts (security model, trust item 7).
- (b) The verifier `towerAccepts` is not the verifier of the pinned code. The
  difference of the norm check is removed: at the squared-norm levels the norm
  sum (85) is in the last sum-check of the range tree with the coefficient
  `ζ_norm`, Akita (90), and the value `z̃(r_virt)` is in the fused sum-check
  with the coefficient `η_norm`, Akita (91) and (93), in the order of the
  direct shape of the pinned code
  (`crates/akita-verifier/src/stages/physical_l2_norm.rs`, `stage1.rs` and
  `stage2.rs` of the Akita code; `Spec.Compression.Akita.foldCheckSq`). The
  added error is `2/|K|`, which is `N^norm = 2` of Akita (189)
  (`Spec.Compression.Akita.Extract.foldCostsSq_sum`). Each schedule file has
  the direct shape at each squared-norm level, and Lean has no model of the
  digit-expanded shape. The coefficient `ζ_bin` of the binary-support claim
  has a draw only at a level with compression digits, as in the code
  (`stage1.rs`, `CompressionBinary`; `Spec.Compression.Akita.drawBin`): the
  levels with the payload mode `Raw` have no such draw, which are the last two
  levels of each schedule file. The witness of a committed level is the
  witness table of the pinned code (`crates/akita-params/src/witness.rs` and
  `witness/tail.rs` of the Akita code) and of Akita (76) and (348):
  `[ẑ_g | ê_g | t̂_g]` for each group, the quotient digits of the rows before
  the compression rows, and for each compression map the digits of `F_a` of
  each group and of `H_a`, then the quotient digits of these rows
  (`Spec.Compression.Akita.Gaps`, `Spec.Compression.Akita.OpeningShape.gaps`,
  `Spec.Compression.Akita.Batch.gaps`). A row reads no cell for a zero of a
  padded vector: the address `none` is a coefficient 0
  (`Spec.Compression.Akita.RingRow`). The rows of a fold with groups are in the
  order of the groups, as in Akita (76) and in the code
  (`Spec.Compression.Akita.Batch.groupRows`). So the witness of the root ends
  at the cell 3,659,520 in the model and in the code
  (`Spec.Compression.Akita.Key20.root_degrees`), and the file holds 3,659,776,
  with the last alignment of the code to 1,024 cells. The model has no
  alignment zeros, and each alignment of the code before the last one is empty
  at each level of the two files (`tests/AkitaSchedules.lean`). One order
  stays: in the quotient segment of the carrier row the code has the order
  digit, plane, coefficient of Akita (348), and the model has the order plane,
  digit, coefficient in the same cells, because the model has the quotient of
  each plane as one polynomial over `Z_q` (difference (p)). The batching weights of the root are
  independent draws, as in the code, and not the powers of one challenge, as in
  the paper (`Spec.Compression.Akita.Extract.density_batched_le`). The model
  has one claim for each group and the extension degree 2 in its trace and
  tensor layers (`Spec.Compression.Akita.Degree2`). It has no digests: the code
  absorbs the root commitments and the last terminal block as SHA3-256 digests,
  and Lean has only the counting bound of Akita Lemma 10.47
  (`Spec.Compression.Akita.Digest.Run.bad_le`).
- (c) `Spec.Compression.Akita.Schedule.setupGap`, which is `N_setup · q / 2^w`,
  is above 1 at the draw width `w = 64` of the code, so it bounds nothing. The
  code uses exact rejection sampling, and
  `Spec.Compression.Akita.Setup.expand_le` gives the bound for a finite stream.
  This term of `Spec.Compression.Akita.Theorem1046` is to be replaced. Lean
  joins the lists of instances with the views of one setup vector: for keys
  whose matrices are these views, a solution gives a solution for the view of
  an instance of the list itself, and Akita Lemma 10.20 (i) then applies to
  the list (`Claims.Akita.key20_instances_shared`,
  `Claims.AkitaKey30.key30_instances_shared`).
- (d) The code declares no `(λ_FS, Q_max)`. It has a target of 128 bits for
  each query. The paper has a proof of work before selected algebraic challenge
  queries (Akita Section 10.1, (146) and Appendix I). The pinned code has one
  at each challenge site of its grinding plan, with the least `g ≤ 57` such
  that `q² · 2^g ≥ L · 2^128` for the loss `L` of the site
  (`jolt_transcript::grind_bits_for_loss`), and one for each fold group, with
  the fold coordinates `W_l` against the smallest challenge set `2^{K_l}`
  (`FoldChallengeWork`), which the paper does not have. The Lean model has
  this proof of work. The answer at a stage input has one more independent
  slot, the work slot, and the draws of a challenge read only the other slots
  (`Spec.Compression.Akita.Extract.view`). The verifier of each segment and of
  each fold requires the work slot in
  `Spec.Compression.Akita.Extract.workZeros` for the bits of the site, which
  has a uniform slot with probability at most `2^−g`
  (`Spec.Compression.Akita.Extract.mass_workZeros_le`). The bits are schedule
  data by the rule of the code (`Spec.Compression.Akita.Extract.workBits`,
  `Spec.Compression.Akita.Extract.foldWorkBits`), from the losses of the
  sites of the code in the order of its plan. At two sites the Lean loss is
  not the loss of the code: the leaves of the range tree, and the batch of the
  virtual evaluations. At each of them the Lean bits are at
  most the bits of the code, so the Lean verifier accepts each transcript that
  the code accepts at these sites. This is a derivation from the code, not a
  Lean theorem. Lean has no site `SetupBatch`, which the plan of the code has
  only for a setup batch level. Each stage error of a checkpoint or plain
  stage has the factor `2^−g` (Akita Remark 10.18). The code counts a work
  credit on the fold term (`FoldChallengeWork`), the paper refuses it (Akita
  page 71), and the claims do not use it. The Lean verifier keeps the work
  check of the fold.
  The grinding plan of the two files is not stored in the code, so the
  comparison of the Lean bits with the plan is open. The budget condition S3,
  Akita (195), has no value to check (Akita Corollary 10.48:
  `Spec.Compression.Akita.budget_iff`,
  `Spec.Compression.Akita.effectiveStrength_le`), and the budget does not use
  the factor.
- (e) The Lean extractor has a limit `budget` on the attempts of each retry
  loop, which adds the term `(Q + 1) · L / budget` for `L` loops
  (`Claims.Akita.checkpoint_runs` gives the runs).
- (f) The gates do not find a theorem whose hypotheses nothing satisfies.
  Concrete instances found two such defects, and both are corrected: no finite
  type satisfied the stage law of `Spec.Checkpoint.Protocol`, and the block
  counts of `key20.aks` did not satisfy the conditions of the links between the
  levels.
- (g) The verifier of the paper rejects at a zero tensor factor `θ` (Akita
  Section 3.7, Section 5.2 and Section 8.2), and its completeness bound (155)
  has a term for this event. The Lean verifier has no such rejection, so the
  acceptance results of Lean have no such term. The verifier of the pinned code
  also has no such rejection: it keeps `θ` as a scale of the next relation
  (`crates/akita-verifier/src/stages/opening_claims/extension_claim.rs` of the
  Akita code).
- (h) Removed. At a certified base above 8, the Lean verifier has the product
  tree of range sum-checks of Akita Section 6.1, in the order of the pinned
  code (`Spec.Compression.Akita.rangeCheck`, from
  `crates/akita-verifier/src/stages/stage1.rs` and
  `crates/akita-params/src/layout/digit_range.rs` of the Akita code): the
  product degrees `(2)`, `(4)` and `(2, 4)` for the bases 16, 32 and 64, leaf
  polynomials of 4 factors, and the powers of one challenge for the claims of
  the children of a level. Its bad sets have the bound of Akita (187)
  (`Spec.Compression.Akita.rangeCheck_costLe`). One point stays: a round
  message of the Lean sum-check has each coefficient of the round polynomial,
  and the code sends the round polynomial without its equality factor and
  without its constant term. A transcript of the code gives a transcript of
  the Lean verifier, and Lean has no statement of this map.
- (i) The fold of the paper has a nonce of the response search, with a limit
  that the verifier enforces (Akita Section 5.4). The pinned code has a
  response nonce `n` with `n < 2^12` (`FoldResponse`, with no work bits), and
  the proof of work of each site has a nonce of `g + 7` bits. The Lean model
  has no nonce and no range check: a nonce is a part of a message of the
  prover, the factor `Q + 1` covers a prover that tries other messages, and a
  verifier without the range check accepts each transcript that the code
  accepts. Lean states no completeness bound in the nonce limit: the
  acceptance results are for each table whose work slots have 57 zero work
  bits (`Claims.Akita.key20_opening_accepts` and
  `Claims.AkitaKey30.key30_opening_accepts`, the hypothesis `worked`), in
  place of the nonce search of the honest prover.
- (j) `Spec.Checkpoint.Structure.Lawful` has inclusions where Akita Definition
  10.16 has equalities, it permits a plain stage outside each segment, and it
  has no condition on the run time. The Lean theorems use only these
  inclusions.
- (k) The offloaded root fold of `key30.aks` (Akita Section 9). The paper adds
  `σ_S` to the local part in the final check of Stage 2, Akita (132). The
  pinned code multiplies the claim of Stage 3 by a factor before it adds it:
  the evaluation of the powers of `α` at the coefficient coordinates of `r_2`
  (`common_alpha_evaluation` in
  `crates/akita-verifier/src/relation/evaluation.rs` of the Akita code), and
  its Stage 3 is for the weight at the relation-address coordinates `r_x`.
  This difference is removed: the Lean verifier has the form of the code, with
  the factor `scale` of `Spec.Compression.Akita.SetupScan` in the final check
  (`Spec.Compression.Akita.offloadCheck`). The form of the paper is the case
  with the factor 1 (`Spec.Compression.Akita.SetupScan.direct`): for a value
  `σ` of the code, `σ_S = scale · σ`, and the weight `ω_setup` is `scale`
  times the weight of the code
  (`Spec.Compression.Akita.SetupScan.direct_value`). The order of the messages
  is the same in the paper, in the code and in Lean: `v_W`, `σ_S`, the rounds
  of Stage 3, and `s_{ρ_S}`. Stage 3 has the degree 2 and one round for each
  variable of the setup prefix in the three. These points stay. The code
  checks the final equation of Stage 2 after Stage 3, and Lean checks it
  before Stage 3, so the two reject the same transcripts for this equation.
  The code absorbs the identifier of the registry slot of the setup prefix
  before `σ_S`, and the Lean model has the commitment `registry` as a fixed
  value. The form of the code is the form of the paper with the weight
  `ω_setup` written as a product, so it is an identity and not another
  check. Each element of `ẑ`, `ê` and `t̂` that a view of the setup multiplies
  starts at a multiple of 128 cells in the witness, and each ring degree of
  `A`, `B` and `D` is a multiple of 128
  (`Spec.Compression.Akita.Key30.root_blocked`; 128 is
  `relation_coefficient_block_len` of the code for the root). In a block of
  128 cells, the weight of the cell `k` is then `α^k` times the weight of the
  first cell (`Spec.Compression.Akita.Batch.weight_geometric`). Lean proves
  the split in this form (`Spec.Compression.Akita.Batch.codeScan`) when the
  matrices `A`, `B` and `D` of the root are the views of the setup prefix,
  which is the hypothesis `Spec.Compression.Akita.Key30.Tower.Shared`. The
  factor is `Spec.Compression.Akita.powerScale` of `α` at the first 7
  coordinates of `r_2`, which is `evaluate_power_sequence_mle` of the code,
  and the weight of Stage 3 reads only the other coordinates of `r_2`. The
  verifier of the claims of `Claims.AkitaKey30` has this split. One point
  stays: Lean defines the weight of Stage 3 as the extension of the first
  cells of the blocks, and it has no formula for it of the kind of
  `SetupIndexWeightMle` of the code.
- (l) The batching stage of the two claims of level 1 has the weights `(1,
  ϑ_grp)` in the paper, Akita (137). The Lean stage has an independent weight
  for each claim, as at the root
  (`Spec.Compression.Akita.Extract.root_suffix_pinned`).
- (m) Akita Lemma 9.1 has the binding of the setup opening in the
  source-comparison class of the setup group, at the radius of Akita (166). The
  Lean proof (`Spec.Compression.Akita.Batch.pin_fixed`) compares the extracted
  opening with the reference decommitment at the bound `η_A` of the group of
  the setup prefix at level 1, and `Spec.Compression.Akita.Key30.tower_suffix`
  proves that the norm `β̄_∞ + κ̄_1 · ρ` of this comparison, `65535 + 2 · 31 ·
  128`, is at most this `η_A`. So the witness of Lemma 9.1 is a solution of the
  instance of `A` of the setup prefix in the list of
  `Claims.AkitaKey30.key30_instance_list`.
- (n) Each row of `key30.aks` has coefficient packing at levels 2 and 3. The
  family F1 of Akita Section 9.3 has evaluation trace at these levels, so the
  paper does not analyze this schedule (`SPEC.md` section 9). The witness of
  level 3 ends at the cell 523,264 of its cube of `2^19` cells, in the model and
  in the file (`Spec.Compression.Akita.Key30.level₃_sized`).
- (o) At a quotient with 16 digits of the base 16, which levels 1 to 3 of
  `key30.aks` have, the code uses the residues from `−8 · Σ_u 16^u` to `7 · Σ_u
  16^u`: an element above `min(T_δ, ⌊q/2⌋)` has the residue `x − q`
  (`decompose_centering_threshold` in
  `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs` of the Akita
  code; Akita Section 3.3). The Lean digits are those of this lift
  (`Spec.Compression.Akita.lift`, `Spec.Compression.Akita.fieldDigit_sum`).
  The condition of the honest quotient digits was a range of the least
  absolute residue, which is false at these levels. It is now `q ≤ b^δ`
  (`Spec.Compression.Akita.FlatRelation.withQuotients_holds`), which holds at
  the root and at levels 1 to 3 (`Spec.Compression.Akita.Key30.covers`). So
  the honest prover of a fold with quotient digits has its result at the
  numbers of `key30.aks`. Lean has the honest prover of the offloaded check
  of the root (Stage 3, `Spec.Compression.Akita.Extract.offloadCheck_winsAt`)
  and of the levels of this schedule
  (`Spec.Compression.Akita.Key30.tower_complete`). Its acceptance result is
  for each setup prefix with the fit conditions
  (`Claims.AkitaKey30.key30_opening_accepts`).

- (p) Removed. The rows of a fold with coefficient packing. The pinned code
  has one consistency row for each group: it joins the planes of the carrier
  row, Akita (102), with the basis of the extension field before the
  evaluation at `α` (`crates/akita-types/src/proof/coefficient_packing_relation.rs`
  and `crates/akita-verifier/src/coefficient_packing_relation/compact.rs` of
  the Akita code). The paper has this row: the carrier row has the coefficient
  field `E`, its multipliers have the basis elements `β_t`, its quotient is
  `Q_car = Σ_t β_t · Σ_j q_{t,j} · Y^j`, and it counts once at the degree
  `d_car` (Akita Section 7.3, (110) to (113), and the text after Lemma 7.1).
  The Lean relation now has this row. It has one ring row over `Z_q` for each
  plane (`Spec.Compression.Akita.evaluationRow`), and
  `Spec.Compression.Akita.Join` joins the planes into one row over `E` with
  the basis as coefficients (`Spec.Compression.Akita.FlatRelation.freeRows`,
  `Spec.Compression.Akita.LiftRelation.liftRows`). The soundness is the root
  count of one polynomial over `E`: the polynomials of the planes are its
  coordinates in a basis over `Z_q`, so it is zero only when each plane is
  zero (`Spec.Compression.Akita.Join.poly_ne_zero`), and it has at most
  `2 · d − 1` roots on the quotient-lift route and `d − 1` on the
  quotient-free route (`Spec.Compression.Akita.LiftRelation.liftBad_card_le`,
  `Spec.Compression.Akita.FlatRelation.freeBad_card_le`). Lemma 7.1 of the
  paper has its statement for rows over `Z_q`, and the paper applies it to the
  carrier row without a separate proof. The number of rows over `E` is that of
  the code, `Σ_g (1 + n_A + n_B) + n_D`, plus `2 · (G + 1)` at a level with
  compression digits (`crates/akita-params/src/layout/params.rs`;
  `Spec.Compression.Akita.OpeningShape.rowCount`,
  `Spec.Compression.Akita.Batch.rowCount`), and the batching point `τ_1` has
  the number of coordinates of the code at each level of the two files: 5 at
  the root of `key20.aks` and 4 at its level 1, where the model had 6 and 5.
  The files do not hold these two numbers, and `tests/AkitaSchedules.lean`
  computes them from the fields of the files with the formulas of the code.
  The bound of `key20.aks` is smaller by `1/(8 · |K|)` at each of these two
  levels: `317479/(1024 · |K|)` in place of `317735/(1024 · |K|)`. The terms
  of the fused sum-check are the same in the code and in Lean: the claim
  `γ · s + V + η · v`, with the opening row in `V`, and the final check with
  the weights of the relation, of the opening row, of the norm and of the
  binary support (`crates/akita-verifier/src/stages/stage2.rs` of the Akita
  code; `Spec.Compression.Akita.fusedCheck`).

### The results for `key30.aks`

The claims of `Claims.AkitaKey30` are for the two rows of `key30.aks`, the file
of the key of `2^19` blocks. Their model and their form are those of the claims
for `key20.aks`: one uniform table of stage answers with independent slots, a
Lean verifier that is not the verifier of the pinned code, and a bound with the
term `breakChance` under `[MSIS]`.

- The opening. Each output of the extractor is a certified decommitment of each
  root commitment to a table with the claimed value, or a solution of
  Module-SIS (`Claims.AkitaKey30.key30_opening_sound`). The verifier accepts
  while the extractor returns no output with probability at most `(Q + 1) ·
  (7595403 + 4096·B_1 + 4096·B_2)/(4096·q²) + (Q + 1) · (1467 + B_1 + B_2)/budget`
  (`Claims.AkitaKey30.key30_opening_bound`). `B_1` and `B_2` are the live
  blocks of the witness group of level 1 and of level 2: 167 and 44 for `W` of
  27 variables, and 158 and 43 for `W` of 26 variables. The tree size of the
  opening is `54400647168 · (129 + B_1) · (B_2 + 1)`
  (`Spec.Compression.Akita.Key30.tower_counts`).

- The final verifier. For each claim shape and for the proof of zero steps, the
  bound is the sum over the shapes of `(Q + 1) · (runCost + (7595403 + 4096·B_1
  + 4096·B_2)/(4096·q²)) + (Q + 1) · (1469 + B_1 + B_2)/budget + breakChance`
  (`Claims.AkitaKey30.key30_verifier_sound`). The probability of a false
  history is at most this sum plus `walkBound`, and the same bound holds for
  the witnesses of the reverse extractor
  (`Claims.AkitaKey30.key30_history_sound`,
  `Claims.AkitaKey30.key30_history_knowledge`). The three opening points have
  30, 27 or 26, and 28 coordinates (`Claims.AkitaKey30.key30_points`). The part
  that does not read the schedule is in `Export/BoundedDepth/FinalShapes.lean`,
  and the files of `key20.aks` keep their statements.

- The instances. The 54 matrices of each row give 18 instances `(d, n, m, p,
  η)` that are not below another instance: `(32, 1, 128, ∞, 1)`, `(64, 1, 1024,
  ∞, 1)`, `(64, 2, 77, ∞, 63)`, `(64, 3, 616, ∞, 63)`, `(64, 8, 256, 2, 2^46)`,
  `(64, 8, 512, 2, 2^45)`, `(128, 1, 416, ∞, 31)`, `(128, 1, 4224, ∞, 15)`,
  `(128, 1, 11264, ∞, 7)`, `(128, 4, 512, ∞, 126852)`, `(128, 5, 512, ∞,
  835380)`, `(128, 6, 2048, ∞, 835380)`, `(256, 1, 8192, ∞, 15)`, `(256, 1,
  28160, ∞, 7)`, `(512, 2, 1024, ∞, 8126340)`, `(512, 2, 2048, ∞, 507780)`,
  `(512, 3, 5120, ∞, 2080374660)` and `(2048, 1, 4096, ∞, 16642998148)`
  (`Claims.AkitaKey30.key30_instance_list`). The list is complete for
  `breakChance`, with the collision of the key commitment and the witness of
  Akita Lemma 9.1 for the setup prefix (`Claims.AkitaKey30.key30_instances`).
  `formal/nightstream-fprime/tests/AkitaSchedules.lean` compares the list with
  the two rows of the file. The code has one flat setup stream for all roles
  and levels, and it reads each matrix as a prefix view from the offset 0:
  coefficient `l` of entry `(i, j)` is `S[(i·m + j)·d + l]`
  (`akita-params/src/layout/flat_matrix.rs`), which is
  `Spec.Compression.Akita.Setup.view`. Its list of matrices is
  `FoldSchedule::sis_occurrences` (`akita-params/src/schedule/sis_occurrences.rs`):
  `A` and `B` of each group, one `D` of each level, the compression maps of
  each group and of a level with compression, and `A` of the terminal. No `A`,
  `B`, `D` or terminal matrix of the two files is outside the Lean lists. The
  files do not hold the compression maps, and Lean gives them the widths of
  the formula of the code (`akita-params/src/compression.rs`). The verifier of
  the code does not read `A`, `B` and `D` of the offloaded root
  (`setup_envelope.rs`), but the prover reads them, and the Lean list has them.
  When each matrix of the opening is the view of one setup vector `S` with the
  numbers of its instance, each solution gives a solution for the view of `S`
  of one of the 54 views (`Claims.AkitaKey30.key30_instances_shared`). Each
  view reads at most the `2^23` cells of the setup prefix, and `A` of the key
  group reads all of them. So Akita Lemma 10.20 (i) gives the probability for
  a uniform `S` as the sum over the 54 views
  (`Spec.Compression.Akita.Key30.shared_setup`).

- The challenge families. The decoders are
`Spec.Compression.Akita.Extract.draw` of the families of the code: `(31, 0)`
for each group of the root and of level 1 and for level 4, `(31, 10)` for
levels 2 and 3, and `(31, 11)` with a filter for levels 5 and 6 and for the
terminal. The filter is the predicate of the code, so the unit differences,
the norms `ω`, the operator bound `Γ` and the slot probability `1/q²` are
theorems.

The limits are these. The matrices of the root are the views of the setup
prefix by the hypothesis `shared`, and from it Lean proves the split of the
public weight of the root into the setup part. The commitment of the setup prefix in the
verifier setup is the output of the Lean commitment algorithm by the hypothesis
`registered`. The Lean verifier of
the opening accepts the honest proof for each setup prefix and each family of
tables with the fit conditions of the honest prover
(`Claims.AkitaKey30.key30_opening_accepts`). The fit conditions are a
hypothesis. Lean proves them for the zero tables and the zero setup prefix,
and for each setup prefix with decoders that give the fold challenge 0
(`Spec.Compression.Akita.Key30.fit_zero`,
`Spec.Compression.Akita.Key30.fit_quiet`). The final
verifier of this file has no acceptance result. The memory results have no statement in this game. The Lean
verifier and the Lean decoders have no comparison with the code. The
differences (k) to (p) of the section above apply.

The claims that end in `_with_akita` have seven premises about Akita. For the
shipped key, the claims of `Claims.AkitaKey30` have each of them as a theorem
or as a part of the game, under the hypotheses that follow.

- `akita.theorem1046` (`[AKITA]`). The extraction is a theorem:
  `Claims.AkitaKey30.key30_opening_sound` and
  `Claims.AkitaKey30.key30_opening_bound`, with `[MSIS]` for `breakChance` and
  the slot reading of `[KECCAK]`. Hypotheses: `shared`, `registered` and the
  stage layout of the Lean verifier.

- `akita.admissible`. Not used. The claims do not use Akita Definition 9.2. The
  conditions on the numbers of the file are theorems
  (`Spec.Compression.Akita.Key30.root_sized` and the theorems after it), and
  the condition E1 is a theorem for the decoders `draw`.

- `akita.admitted` and `akita.budget`. Not used. The prover has at most `Q`
  queries to the table (`bounded`), and the extractor has a positive limit
  `budget`. The claims have no pair `(λ_FS, Q_max)`.

- `binding`. Inside the result. A different table for the key commitment gives
  a solution of Module-SIS, which is a part of `breakChance`. Hypotheses: the
  key commitment of the verifier has a certified decommitment to the key
  polynomial (`registered` of `Export.BoundedDepth.FinalKey30.Opening.Drawn`)
  and `covers`.

- `accepts`. A part of the game: the verifier is
  `Export.BoundedDepth.FinalKey30.finalAccepts`. It is not the verifier of the
  pinned code.

- `composition`. Inside the result, with two checkpoints in sequence
  (`Spec.Compression.Akita.Extract.Compose.suffix`).

The removal of the claims that end in `_with_akita` is a decision of the owner.
It needs these changes.

- Claims. `Claims.Final.verifier_sound_with_akita`,
  `Claims.Final.history_sound_with_akita`,
  `Claims.Final.history_knowledge_with_akita` and
  `Claims.Final.memory_sound_with_akita` go.
  `Claims.Shipped.verify_final_sound` takes the same seven premises, so it
  needs a new statement over `Claims.AkitaKey30.key30_history_sound`. The
  memory claim has no result in the game with Akita openings: a new theorem is
  necessary, or `Claims.Final.memory_sound_with_akita` stays.

- `SPEC.md`. The two rows of the table of verifier entry points for
  `verify_final` cite `Claims.Final.history_sound_with_akita` and
  `Claims.Final.memory_sound_with_akita`. Algorithm 10 has the pins of the four
  claims, and section 9 has the row of the family F1 with
  `Spec.Compression.Akita.Schedule.not_admissible_key30`.

- Premise rows. `akita.theorem1046`, `akita.admissible`, `akita.admitted` with
  `akita.budget`, `accepts` and `composition` go when no claim takes them.
  `binding`, `msis`, and `honest.*` with `S.*` and `covers` stay while
  `Claims.Compression.key_binding` and `Claims.Shipped.verify_final_sound` take
  them. The entry `[AKITA]` of the list of assumptions goes with the row
  `akita.theorem1046`.

- The cost. Two open rows go (`akita.admissible`, `composition`) and one
  assumption (`[AKITA]`). Two open rows come: `AkitaKey30/drawn.registered`
  and `AkitaKey30/drawn`. The row `AkitaKey30/shared`
  is of the class `game`. The verifier of the claims becomes the Lean verifier, which the
  differences (a) to (p) separate from the pinned code.

### Open points

- A review of the soundness of the committed key by a party outside the
  project.
- A proof that the Rust layout and the Rust verifier are the Lean models.
  Only tests join them
  ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
  item 7).
- Security against a quantum adversary of the transcript hashes.
- A WASM build of the verifier.

These points are about the claims of `Claims.Akita` for `key20.aks` and of
`Claims.AkitaKey30` for `key30.aks`. Each needs a decision of the owner ([security
model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust item 7, gives
the facts).

- The claim. Decided: the results are claims, and each hypothesis has a class
  in the premise table. For `key20.aks`, the claims have the premises
  `akita.theorem1046` and `composition` as theorems under their hypotheses.
  What stays open is to replace the claims that end in `_with_akita`: the
  claims of `Claims.AkitaKey30` cover `key30.aks`, and the section "The
  results for `key30.aks`" gives what the removal needs. For `key30.aks`, the
  rows `AkitaKey30/drawn.registered` and `AkitaKey30/drawn` are open. For `key20.aks`, one premise row is open: the
  conditions `Export.BoundedDepth.FinalKey20.Opening.Drawn` of the final
  verifier, which hold the Lean decoders and the certified decommitment `registered` of
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
  code in these parts. The transcript has no digests and no proof of work.
  Two more differences are with the paper. The Lean verifier has no
  rejection at a zero tensor factor, and the pinned code has none
  (`crates/akita-verifier/src/stages/opening_claims/extension_claim.rs` of the
  Akita code). Its fold has no nonce of the response search. A
  change of the pinned Akita code or a change of the Lean model removes a
  difference. The decision is which one changes, for each difference.
- `key30.aks`. The file of the key of `2^19` blocks has coefficient packing at
  levels 2 and 3, two groups at level 1 and an offloaded setup prefix. The
  Lean tower of `Spec/Compression/Akita/Extract/OffTower.lean` has them, and
  the claims of `Claims.AkitaKey30` are for it. One part is open: a link of
  the prefix commitments of the pinned key to the Lean commitment algorithm.
  The acceptance result is for each setup prefix with the fit conditions
  `Spec.Compression.Akita.Key30.Fit`
  (`Claims.AkitaKey30.key30_opening_accepts`). For the challenge families of
  the schedule, no Lean result gives the fit conditions of a setup prefix
  that is not 0: this needs the nonce of the response search in the model
  and a bound on the probability that a response fits.
  Lean proves the split of the public weight (Akita
  (132)) for matrices that are views of the setup prefix, in the form of the
  pinned code.

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
  the file. For keys whose matrices are the views of one setup vector `S`,
  each solution gives a solution for the view of `S` of one of the 43 views,
  and each view reads at most `2^18` cells of `S`
  (`Claims.Akita.key20_instances_shared`,
  `Spec.Compression.Akita.Key20.shared_setup`). Lean does not prove the
  hardness of an instance. `key30.aks` has
  18 such instances, at the degrees 32 to 2048
  (`Claims.AkitaKey30.key30_instance_list`), and the same test compares them
  with the two rows of the file. The pinned code admits an instance only at
  169.07 bits or more of quantum ADPS16 Core-SVP cost, in a direct ℓ∞ attack model on
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
  `key20.aks` and for the two rows of `key30.aks`. Lean proves that the norm of the reduction is at most the bound
  of the file (`Spec.Compression.Akita.Descriptor.key20_covers`,
  `Spec.Compression.Akita.Descriptor.key30_covers`).
- The proof of work and the probability `p`.
  `Export.BoundedDepth.FinalKey20.sound` takes the largest probability `p` of
  one fold challenge as a hypothesis, and the claims have `p = 1/q²`. Its
  bound on the stage errors of the opening is `455719/(1024·q²)` for a decoder
  that is an injection of one slot of `K` into the challenge family, where
  `p = 1/q²`. No decoder from one slot has `p` at most `2^−128`, because `q²`
  is below `2^128`. The Akita code has a target of 128 bits for each query,
  and the Lean model has its proof of work (difference (d)). One point stays
  open: the comparison of the work bits of Lean with the grinding plan of the
  code, which the code does not store for the two files.
