# Security model

This page states which adversaries the Nightstream F′ Lean results protect
against: what each adversary controls, when it wins, the Lean result that
bounds it, and what you must trust. It does not restate proofs; read the cited
declaration. `SPEC.md` describes the protocol: two step circuits, a light
circuit and a refresh circuit, that fold into two slots. Adversary 6, the
adversary of the compressed proof, has a claim for the argument only, in a
model with ideal Akita openings. No claim gives its aim (trust item 7). Section
Assumptions lists every assumption of the results, section Hashed
encodings lists every place that hashes or absorbs data, and section Premises
gives the class of each hypothesis of the claims. Nightstream claims exactly
the theorems of the namespace `Claims` (`NightstreamFPrime/Claims/`). Each
claim states one result with every hypothesis in its signature, and its proof
is one application of the theorem that this page describes.

Names omit the common `NightstreamFPrime` namespace.
`tests/EndpointCensus.lean` checks that every name on this page that starts
with `Claims.`, `Export.`, `Layout.`, `Lifecycle.` or `Spec.` exists, and that every cited
theorem uses only `propext`, `Classical.choice` and `Quot.sound`. Shorter names
are not checked.

## What Lean checks

- The kernel of the official Lean v4.35.0 toolchain, with Mathlib v4.35.0 and
  the CompPoly revision that `lakefile.toml` pins.
- `scripts/validate.sh` runs on the developer Mac. Its `build`, `axioms`,
  `executables` and `all` phases use `lake build --wfail`, and single files
  run with `warningAsError`, so any warning fails a gate. `all` runs the
  boundary checks, both libraries, every `lean_exe` target and the census.
  The CI job `lean` runs `validate.sh all` on the self-hosted runner (owner
  decision of 2026-11-18). It is not a required check.
- Every audited theorem uses the three axioms above. The audit fails closed:
  `sorryAx` or a new axiom stops the build.
- One linter is off: `Circuit/Basic.lean` turns off `linter.dupNamespace`,
  because the `Circuit` monad has the name of its namespace. It is a naming
  lint, not a check of any proof.
- Strong-set low-norm invertibility (SuperNeo v1.2 Theorem 4) is proved
  (`Claims.Foundations.low_norm_invertible`), so no security result takes it
  as a premise.

## Assumptions

The results use only the assumptions of this list, and each is on a
primitive. A random oracle on a derived object (the Poseidon2 hash, a
transcript, `η`, or a challenge) is not an assumption. Such a model is
`[P2-PERM]` with `[SPONGE]` or `[DSFS]`, with their conditions as Lean
hypotheses, or it is a Lean theorem. Section Hashed encodings gives the status
of each condition. `scripts/check_assumptions.py`, in the `static` phase of
`scripts/validate.sh`, fails when an entry of this list is not on the owner's
list of 2026-11-13 and 2026-11-18, when another part of this page or `SPEC.md` states an
assumption, or when a trust item cites no assumption and is not a gap or a
scope note.

<!-- assumptions:begin -->
- `P2-PERM`: the Poseidon2 permutation of width 16 over `F_q`
  (`Spec.Poseidon2.permute`) is a uniform random permutation. The adversary
  can evaluate it and its inverse, and each evaluation counts toward `N`. Each
  ideal primitive of this list is independent of the others.
- `SPONGE`: the sponge indifferentiability theorem (Bertoni, Daemen, Peeters
  and Van Assche, EUROCRYPT 2008; for prime fields, Khovratovich, Marhuenda
  Beltrán and Mennink, ePrint 2023/520). Under `[P2-PERM]`, the map from a
  nonempty list of 12-lane rate blocks to the rate lanes of the sponge state
  is indifferentiable from a uniform function on such lists. The advantage is
  at most `spongeError N = N²/q⁴` for `N` permutation calls, because the
  capacity is 4 lanes (`Spec.Sponge.spongeError`). `Spec.Sponge.Transfer`
  states the theorem for one use, and `Spec.Sponge.transfer_le` moves a bound
  of the use's random-oracle model to the ideal-permutation model. The theorem applies to a use only under its
  conditions: the hash pads injectively, each use absorbs an injective framing of its data, and
  two uses never absorb the same block list (domain separation). Lean proves the first for the
  `10*` padding of `Spec.Poseidon2.hash` (`Spec.Sponge.padding`, `Spec.Sponge.hash_eq_digest`).
  The other two are Lean hypotheses of each use. Then a finder with `N` calls
  gives two distinct inputs with one 4-lane digest with probability at most
  `C(N + 2, 2)/q⁴ + spongeError N` (`Spec.Sponge.collision_le`), because they
  absorb distinct block lists (`Spec.Sponge.collision_blocks`) and a uniform
  function has this birthday bound (`Spec.RandomOracle.pair_collision_le`).
  Two uses whose words start with tags that differ at a position of both
  never absorb one block list (`Spec.Sponge.blocks_ne_of_differ`).
- `DSFS`: the duplex-sponge Fiat–Shamir lemma (Chiesa and Orrù, ePrint 2025/536, Lemma 5.1, as the proof of
  Theorem 6.2 uses it) for the fold transcripts on Nightstream's sponge. Under `[P2-PERM]` and the
  conditions `Export.BoundedDepth.FoldSponge.Conditions`, each fold prover with at most `N` permutation
  calls has a model prover with at most `N` queries to the uniform function of adversary 3. The probability
  that the fold succeeds and the extractor, which reruns the model prover, does not extract is at most
  `duplexError N = 25·N²/q⁴` larger for the fold prover (`Spec.Sponge.duplexError`,
  `Spec.Sponge.DuplexTransfer`). The number is the error of that theorem for the alphabet `F_q`, a capacity
  of 4 lanes and a codec bias of 0 (each challenge reads lanes of the state), when `N` is larger than the
  number of permutation calls of one verifier run. The paper does not prove two parts, so they are parts of
  this entry (`decisions/duplex-sponge-fiat-shamir.md`). (A) The lemma is for the sponge of ePrint 2025/536,
  Construction 3.3, which writes each block over the rate lanes. Nightstream adds each block to the rate
  lanes (`Spec.Poseidon2.absorbBlock`). The paper says, without a proof, that its analysis adapts to the add
  mode. A start from a fixed state that absorbs the instance is a case that ePrint 2025/536, Section 2.2,
  says the analysis covers. (B) The model prover of the lemma samples values, and the rewinding
  knowledge soundness of ePrint 2025/536, Definition 3.8, is for deterministic provers. So the model prover
  has one fixed random tape for the base run and each retry.
- `KECCAK`: SHAKE128, SHAKE256, SHA3-256 and the Keccak-f duplex of Akita in the classical random-oracle model: the
  Ajtai setup reads SHAKE128 output as uniform 64-bit words (trust item 1), and Akita reads its challenges as uniform
  (trust item 7). For the claims of `Claims.Akita`, the reading is this: the squeeze run of the duplex after one fixed
  absorbed prefix is uniform and independent of each other answer, and it is a function of one uniform output of independent slots `Spec.Compression.Akita.Replay.CodeSlot`, one slot for each coordinate of a challenge. A field challenge reads the field part of its slot, and the work bits are read as `Spec.Compression.Akita.Extract.workZeros`.
  At a fold group without a filter, the word `j` of `K` bits of the message is the word of the slot of the coordinate `j`. At a fold group with a filter, the message is `pool u a` in law (`Spec.Compression.Akita.Replay.expect_pool`): `a j` is the accepted word of the slot of the coordinate `j`, `u` is a vector of independent uniform words that no check of the verifier reads, and the challenge `j` of the code is the decoded `a j` (`Spec.Compression.Akita.Replay.sample_slots`).
  A prover of the code reads `u`. In the model `u` is a part of the random tape of the prover, and each bound of a claim holds for each tape, so it holds for the mean over `u`. This is stronger than one uniform answer for each permutation call. In `Claims.AkitaKey20.key20_verifier_sound` and
  `Claims.AkitaKey30.key30_verifier_sound`, the draws of the Poseidon2 transcript of the argument and the draws of the
  Keccak transcript of Akita are answers of one such table. No Lean statement joins the two hash functions of the code
  to this table (trust item 7). For the SHA3-256 digests that the Akita transcript absorbs (root commitments, last
  terminal blocks), the reading is that of Akita Lemma 10.47: the digest function is a uniform function that is
  independent of the transcript function. The code uses Keccak-f for both, with different first bytes. The owner accepted this hash family for Akita on 2026-11-13
  (`decisions/akita-compression.md`).
- `MSIS`: Module-SIS for a uniform matrix of `κ = 18` rows over `F_q[X]/(X^64 + 1)` at the chain norm, in the
  lattice-estimator model of `decisions/field-2p64-59-ring-x64.md` (trust item 2), and for the Akita commitments at the
  price of the Akita code (trust item 7). The two prices are not numbers of one model. The 169 bits of the Akita code
  are a direct ℓ∞ estimate. The estimator of the Akita authors gives 154.23 bits for the main key in that model, and no Sage
  run gives that number again. On 2026-11-20 the owner selected the method of
  the main-key record for each instance: the Sage `lattice-estimator` at `norm = 2` and the length
  `bound · √(m·d)`. By this method, `sage -python scripts/estimate_akita_sis.sage` gives the recorded 127.20 bits of
  the main key again, and each Akita number below (`scripts/akita_sis_estimates.csv`, trust item 7). **The owner's target is 128 bits against a classical attacker (2026-11-22), and by this method two outer
  matrices of the shipped schedule files are below it.** In the ADPS16 classical model the main key has 140.16 bits.
  Each matrix below 128 bits is a matrix `B` of rank 1 over the ring of degree 128: level 2 of `key20.aks` (width 1092,
  bound 31) at 122.06 bits, and level 2 of `key30.aks` (width 4224, bound 15) at 123.81 bits, with 124.39 bits for the
  second row of the file. Each other instance has 134.90 bits or more. The ADPS16 quantum numbers of the same
  instances are 127.20 bits for the main key and 110.77, 112.36 and 112.89 bits for these matrices.
  Rank 2 for such a matrix gives 285 bits or more, and the next smaller digit base gives 134.6 bits or
  more. The schedule files are not changed, so the claims on Akita openings rest on these instances
  (`decisions/akita-compression.md`). For the matrices of the
  commitment of the compression key, the instances are those of the premise `msis` (trust item 7). For the opening of
  `key20.aks`, the instances are the 13 instances of `Claims.AkitaKey20.key20_instance_list`, and
  `Claims.AkitaKey20.key20_instances` states that this list is complete for the term `breakChance` below. For the opening of
  `key30.aks`, the instances are the 18 instances of `Claims.AkitaKey30.key30_instance_list`, and
  `Claims.AkitaKey30.key30_instances` states that this list is complete for the term `breakChance` of the claims of that
  file. Lean does not prove the hardness of an instance. The Akita matrices are views of one setup vector. For each algorithm `B` on a
  uniform vector and uniform data independent of it, the probability that `B` returns a view and a solution for that view is the sum over the views `I`
  of the success probability, against a uniform matrix of view `I`, of the algorithm `B_I` that Lean derives from `B`
  (Akita Lemma 10.20 (i), `Spec.Compression.Akita.Setup.shared_setup_eq`, `Spec.Compression.Akita.Setup.embedded`).
  This gives no number for `breakChance`. The solver of a Lean extraction result runs the
  prover `K̂ + (Q + 1)(K̂ − 1)` times in expectation, for the tree size `K̂` (`Claims.Akita.checkpoint_runs`). In
  `Claims.AkitaKey20.key20_verifier_sound`, the term of this entry is `Export.BoundedDepth.FinalAkita.breakChance`: the
  probability that the extractor returns a solution for the key commitment, for a matrix of the root or of a level, or
  for the matrix of the terminal. There `K̂ = 85,798,158,336`. `Claims.AkitaKey30.key30_verifier_sound` has the same
  term for `key30.aks` (trust item 7 gives the instances and the radii).
- `HN-A1`: HyperNova errata Assumption 1 for the two fold kinds, in the form of
  trust item 3.
<!-- assumptions:end -->

## Hashed encodings

Each row is a place where Nightstream hashes or absorbs data. The condition
of `[SPONGE]` for a row is that the map from the data to the absorbed words is
injective on the data that the verifier accepts, and that the row's words
start with a domain that no other row starts with. The last column says how
the Rust code is held to the Lean definition (trust item 5).

| Use | Lean definition | Injectivity | Rust parity |
| --- | --- | --- | --- |
| Hash padding | `Spec.Poseidon2.blocks` (`10*`) | `Claims.Encodings.hash_padding_injective`; `Spec.Sponge.padding` meets the conditions of `[SPONGE]` | Direct: Lean hash vectors (`artifacts/nightstream-fprime-poseidon2-vectors-v1.json`, `crates/neo-ccs/tests/poseidon2_lean_vectors.rs`) |
| State hash | `Lifecycle.BoundedDepth.statePreimage` | `Lifecycle.BoundedDepth.statePreimage_injective`, `Claims.Encodings.state_hash_binds` (well-formed states) | Indirect: the rows of the emitted packages (`crates/nightstream/tests/bounded_depth_base_steps.rs`) |
| Slot digest | `Lifecycle.BoundedDepth.serializeSlot` | `Lifecycle.BoundedDepth.serializeSlot_injective` (shapes fixed by the type, depth below `q`) | Indirect, as above |
| Fresh public input | `Lifecycle.encHash` | `Lifecycle.encHash_injective_fixed` | Indirect, as above |
| Package identity | `Export.BoundedDepth.StepPackage.identity` | `Export.BoundedDepth.IdentityBinding.valuePreimage_injective`, `Claims.Encodings.package_identity_binds` (atoms and lengths below `2^64`) | Direct: `identity_code_matches_lean_on_the_probe_packages` |
| Circuit key digest | `Lifecycle.BoundedDepth.CircuitKeyDigest.input` | `Lifecycle.BoundedDepth.CircuitKeyDigest.input_injective`, `Claims.Encodings.circuit_key_binds` (an identity of any length) | Direct: `circuit_key_is_the_lean_digest` on three identities (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/nightstream/tests/bounded_depth_lifecycle/encodings.rs`) |
| Fold transcripts | `Lifecycle.RandomOracle.Schedule.challengeCalls`, `Lifecycle.RandomOracle.Schedule.pieces` | `Lifecycle.RandomOracle.Schedule.challengeAddress_injective`, `Lifecycle.RandomOracle.Schedule.statementCalls_identify`, `Lifecycle.RandomOracle.Schedule.proverCalls_identify`; fixed lengths: `Lifecycle.RandomOracle.Schedule.pieces_alike`, `Lifecycle.RandomOracle.Schedule.calls_pieces`; reads: the state of a `ρ` read is the state of no other challenge (`Lifecycle.RandomOracle.Schedule.rhoCalls_ne`) | Direct: `the_light_transcript_absorbs_the_lean_pieces`, `the_refresh_transcript_absorbs_the_lean_pieces` (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/nightstream-fprime/tests/fold_transcript_lean.rs`). The Π_RLC sampler is direct (`crates/neo-reductions/tests/pi_rlc_lean_parity.rs`). Indirect for the fold prover: package rows |
| Compression transcript | `Export.BoundedDepth.CompressionSponge.pieces`, with `Lifecycle.RandomOracle.Sponge.byteWords` for a byte string and `Export.BoundedDepth.CompressionSponge.Claim.words` for a claim | `Export.BoundedDepth.CompressionSponge.calls_injective` (two runs of one statement shape, each byte string below `q` bytes), from `Export.BoundedDepth.CompressionSponge.pieces_alike`, `Lifecycle.RandomOracle.Sponge.byteWords_injective` and `Export.BoundedDepth.CompressionSponge.Claim.words_injective`. Rows 4 to 17 have the operations and counts of `Spec.Compression.Rust.schedule`: `Export.BoundedDepth.CompressionSponge.argumentPieces_fit` | Direct for the words: `byte_words_are_the_lean_words`, `claim_words_are_the_lean_words`, `the_domain_and_the_session_label_are_the_lean_ones`, `the_compression_domain_is_the_lean_domain`, and for the zero blocks of the reads: `a_replay_of_the_lean_pieces_ends_in_the_lean_session` (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/ns-compress/tests/unit/transcript.rs`). Direct for the order and the counts of rows 4 to 17 on a real proof: `a_replay_in_the_lean_order_draws_the_challenges_of_verify`. Open: no test compares rows 1 to 3 of a real proof with the Lean pieces. Rust makes them with the functions of a fold transcript (section Absorbs without a frame) |
| Akita session bytes | `Export.BoundedDepth.CompressionSponge.session` | `Lifecycle.RandomOracle.Sponge.sessionBytes_injective`; `Export.BoundedDepth.CompressionSponge.session_binds`: two runs with one session are equal, or their transcripts are two distinct lists of sponge calls with the same lanes 0 to 3 | Direct: `the_session_is_the_lean_session`, `a_replay_of_the_lean_pieces_ends_in_the_lean_session` |
| Akita transcript | None | Open: the Akita crates fix these bytes. `akita.verify` (`crates/ns-compress/src/akita/verifier.rs`) gives Akita the session bytes, the three commitments, the three points and the three values, and Akita absorbs them into its own Keccak sponge and makes its challenges there. Lean has no model of that sponge or of Akita's encodings. The Lean extraction results follow the oracle model of the revision of 2026-11-23 of the paper (Akita Section 10.3): the table is on message keys, the stage and the messages before the stage as abstract values (`Spec.Compression.Akita.Extract.Address`), the stage input is rebuilt from the key and the table (`Spec.Checkpoint.Protocol.rebuild`), and the answer is one output of independent slots (the slot reading of `[KECCAK]`). The sponge of the code does not absorb a challenge that it squeezes, as this model needs. The code absorbs public values that it computes from earlier challenges, and no Lean statement shows that its absorbed bytes are an injective key. In `Export.BoundedDepth.FinalKey20.sound`, the draws of the compression transcript and the draws of Akita are stages of one such table, and no Lean statement joins the Poseidon2 sponge and the Keccak sponge to it (trust item 7). The root commitments and the last terminal blocks enter the sponge as SHA3-256 digests. Lean bounds the bad events of such digests (`Spec.Compression.Akita.Digest.Run.bad_le`, the counting part of Akita Lemma 10.47) and has no model of the protocol with digests. The Lean verifier of the Akita claims has no digests, so no result shows that Akita binds these inputs. Two terms of the code are in no Lean bound (`decisions/akita-compression.md`, differences (a) and (b)): the collision term of the session value, `(C(Q_P, 2) + Q_P·Q) / q^4`, because a key of the code has the 32 session bytes (lanes 0 to 3 of the Poseidon2 state) in place of the earlier messages of the argument; and the digest term of Akita Lemma 10.47, `Σ_o (C(Q_o, 2) + Q_o·V_o) / 2^256`, at most `6·Q² / 2^256`, for the commitment digests and the terminal digests. A claimed bound for the code is the Lean bound plus these two terms | None |
| Nebula record chains | `Lifecycle.Nebula.blocks` | `Lifecycle.Nebula.blocks_injective`, `Lifecycle.Nebula.blockChunks_injective` | Direct: fixtures (`crates/nightstream/tests/nebula/memory.rs`) |
| Nebula `η` | `Lifecycle.Nebula.etaInputBlocks` | `Lifecycle.Nebula.etaInputBlocks_injective`, `Lifecycle.Nebula.EtaOracle.pointOf_injective`; the four reads: `Export.BoundedDepth.EtaSponge.reads_injective` | Indirect: package rows |
| Nebula state words | `Lifecycle.Nebula.carryVector` | `Lifecycle.Nebula.carryVector_injective`, `Lifecycle.Nebula.stateWords_eq_or_collision` | Direct: terminal fixtures |
| Application identity | `Lifecycle.BoundedDepth.ApplicationIdentity.Application.words`: a schema of the records that Rust `BoundedDepthPackages::with_application` hashes (`crates/nightstream-fprime/src/bounded_depth/application.rs`). The words are the tag, one word for each byte, the Lean package identity, the witness word count, the rows, the row of each recipe, the blocks, and the layouts of the light and the refresh package (`SPEC.md` section 4.4) | `Lifecycle.BoundedDepth.ApplicationIdentity.Application.words_injective`: the words fix the records. `Lifecycle.BoundedDepth.ApplicationIdentity.Application.body_prefixFree`: the words after the tag fix them before any later words, because each list has its count before it or a length that the format fixes. `Claims.Encodings.application_identity_binds`: two applications with one identity have the same records, or their padded hash inputs are two distinct block lists with one sponge digest. The condition is that no word of the two streams is at or above `q` (`Lifecycle.BoundedDepth.ApplicationIdentity.Application.Canonical`). Then no word is reduced (`Lifecycle.BoundedDepth.ApplicationIdentity.Application.input_val`). The tag meets the condition (`Lifecycle.BoundedDepth.ApplicationIdentity.Application.tagWords_canonical`). Rust refuses a record constant at or above `q` (`ApplicationRecordsWriter::append_row`) and a stream with a word at or above `q` (`with_application`, the test `a_word_at_or_above_q_is_refused`). Limit: these results bind the records. They do not say what the records mean. The Rust application path and the WASM relation have no Lean package (Out of scope), so no theorem states that the rows of the records are the relation that the step circuits check, or that the layouts in the stream are the layouts of the records. The stream does not hold the nodes of a recipe or the rows that give the output state. Rust reads them only to make a witness | Direct: `the_identity_words_are_the_lean_words` on four applications that have each record kind and each list at 0, at 1 and at more entries (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/nightstream-fprime/tests/unit/application_identity_lean.rs`). For an application of the public builder, `a_built_application_has_the_lean_words_and_their_hash`: the words before the layouts are Lean's, the layouts have the form of `Lifecycle.BoundedDepth.ApplicationIdentity.Layout.words`, and the identity is the hash of the words (`crates/nightstream/tests/application/identity_lean.rs`). Indirect for the layouts of a real application: no Lean data has their values |
| Domain separation | `Export.DomainSeparation.uses` | `Claims.Encodings.domains_separated`: two block lists whose words start with the tags of two distinct uses are distinct. A use owns its blocks: `Export.DomainSeparation.hash_owns`, `Export.DomainSeparation.circuitKey_owns`, `Export.DomainSeparation.fold_owns`, `Export.DomainSeparation.nebula_owns`, `Export.DomainSeparation.applicationIdentity_owns`. Open: inputs that an application hashes, which can start with any words. | Indirect for the tags that Lean defines. Direct for the circuit key tag and the two compression domains (rows Circuit key digest and Compression transcript). The compression transcript is one sponge that starts with the domain of its Π_CCS run (`Export.DomainSeparation.compression_owns`) and absorbs the domain of the argument after that run. Direct for the application identity tag (row Application identity) |

### Absorbs without a frame

Rust `absorb_v1_2` (`crates/neo-transcript/src/poseidon2.rs`) adds each 12
words of a message to the rate lanes and permutes. It writes no length word
and no padding. So a message and the same message with one more zero word
give one list of sponge calls
(`Lifecycle.RandomOracle.Sponge.absorbCalls_not_injective`). The sponge
calls of a transcript fix its messages only when, at each position, the two
transcripts read the same number of challenges, or absorb two messages whose
calls fix their length (`Lifecycle.RandomOracle.Sponge.Piece.Alike`,
`Lifecycle.RandomOracle.Sponge.calls_injective`). Two messages of one length
meet this condition (`Lifecycle.RandomOracle.Sponge.Piece.alike_of_length`).
Each row is a call site of `absorb_v1_2` and the reason why its lengths are
fixed. `κd = 1,152`, `d = 64`, `ℓ = 28` and `t = 4`.

| Call site | Words of each absorb | Fixed by |
| --- | --- | --- |
| `read_coin_v1_2` (`crates/neo-transcript/src/poseidon2.rs`) | 12 zero words after the read at position 5 | `Lifecycle.RandomOracle.Sponge.calls`: the count of the reads fixes each zero block |
| `sample_rot_rhos_n` (`crates/neo-reductions/src/common.rs`) | 2: the tag 4 and the coordinate, for each Π_RLC challenge | `Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt`; direct parity (`crates/neo-reductions/tests/pi_rlc_lean_parity.rs`) |
| `bind_and_sample_with_trace`, `prove_phase`, `verify_phase`, `absorb_outputs` (`crates/neo-reductions/src/engines/pi_ccs_joint_protocol.rs`), as the prover of a light or a refresh fold | 12; 1,476 for a light fold or 26,564 for a refresh fold; 18 for each of the `ℓ` rounds; 1,280 for a light fold or 9,600 for a refresh fold | The verifier of a fold is the step circuit, which has the schedule of Lean. `Lifecycle.RandomOracle.Schedule.pieces_alike` gives one length at each position for every statement and every prover message. A prover with another length makes no accepted step |
| The same functions, as the Π_CCS verifier of `verify_final` (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`) | 12; 1,476 plus the byte words of the commitment of `W`; 18 for each round; 640 | `Export.BoundedDepth.CompressionSponge.pieces_alike` for the pieces `Export.BoundedDepth.CompressionSponge.ccsPieces`: the byte words start with the byte count (`Lifecycle.RandomOracle.Sponge.byteWords_alike`), and each other length is a size of the statement. Rust sets `m_in = 320` and checks the shape of the commitment, the count and the length of the rounds and the shape of the output. Open: no test compares these Rust lengths with Lean (row Compression transcript) |
| `append` (`crates/neo-reductions/src/engines/paper_exact_engine/transcript.rs`) | The lengths of the row above for a fold | No function of `nightstream` or `neo-wasm` selects this engine |
| `derive` (`crates/nightstream-fprime/src/package/pi_ccs_v1_2_transcript.rs`) | 12; 1,476 or 26,564; 18 for each round; 1,280 or 9,600 | `validate_shapes` rejects each other length. Direct parity: the two tests of row Fold transcripts compare the state after these absorbs with Lean and check the rejections |
| `absorb_statements` (`crates/ns-compress/src/lib.rs`) | 12; the byte words of the compression key; `5 + κd + m_in + 2ℓ + 2d + 2td` for each claim | `Export.BoundedDepth.CompressionSponge.argumentPieces_fit`: the pieces have the counts of `Spec.Compression.Rust.calls`, and a test replays a proof with them (`crates/ns-compress/tests/unit/transcript_order.rs`). The key and each Akita commitment are byte words, which start with the byte count |
| `absorb_extension`, `absorb_bytes` (`crates/ns-compress/src/transcript.rs`) | 8 for each round; 2, 8 and 2 for the values; 6 for each key round; 2 for the key value; the byte words of each Akita commitment | As the row above |

## Premises

Each row is a hypothesis of a claim. `tests/PremiseCensus.lean` reads every
theorem of the namespace `Claims`. It fails when a declaration of the
namespace is not a theorem, when a claim uses an axiom outside the three of
section What Lean checks, or when the proof of a claim is not one application
of a theorem outside the namespace. Lean writes the hypotheses of the claims to
`artifacts/nightstream-fprime-premises-v1.tsv`, and the Lean build fails when
that file is not current. A hypothesis is a binder whose type is a proposition,
or a proof field of a binder whose type is a structure of the package.
`scripts/check_premises.py`, in the `static` phase of `scripts/validate.sh`,
fails when a hypothesis has no row or two rows, when a row covers no
hypothesis, or when a row does not have the evidence of its class. The classes
are these.

- `assumption`: an assumption of section Assumptions.
- `lean`: Lean proves the hypothesis. The evidence is the theorem or the
  instance that discharges it at the production parameters, or the result
  that joins the theorem and does not take the hypothesis.
- `link`: Rust meets the hypothesis by data that Lean emits. The evidence is
  the emitted file and the Rust test that compares it. A test is a check on
  recorded inputs (trust item 5).
- `game`: the hypothesis is the accept condition or the experiment of the
  theorem. The evidence is the place that states it.
- `open`: no evidence. Each such row is a gap of the results.

A name that ends in `.*` covers each field under it. A name `Scope/name`
covers only the claims of the file or the claim `Scope` in the namespace
`Claims`, and a name without a scope covers each claim. The head is the constant at the head of the hypothesis, and `*` is
each head.

<!-- premises:begin -->
| Premise | Head | Class | Evidence |
| --- | --- | --- | --- |
| `publicFits` | `LE.le` | lean | `Export.BoundedDepth.StepRelation.keyPublicFits`, at the key width `2^26` |
| `cubeFits`, `relation.cubeFits`, `key.relation.cubeFits` | `LE.le` | lean | `Export.BoundedDepth.StepRelation.keyCubeFits`, at the key width `2^26` |
| `program.*`, `exports.program.*` | `*` | lean | `Lifecycle.Stage1.Poseidon2HashChainV1.program` and `Export.Stage1.NebulaMemoryV1Package.application` are programs, so they carry these proofs. A Rust application has no such instance (Out of scope) |
| `stepEq`, `validControl`, `encoded`, `rowsZero`, `others` | `*` | lean | `Claims.Ivc.terminal_bridge` joins the two step results and takes none of these |
| `layout.*` | `*` | lean | `Claims.Ivc.terminal_bridge` and `Claims.Ivc.run_complete` take them as fields of their `exports` (rows below) |
| `programEq` | `Eq` | game | adversary 5: the memory application is the exported program |
| `exports.programStep` | `Eq` | game | `SPEC.md` Algorithm 3: the step of the relation is the step of the application |
| `exports.lightKey`, `exports.refreshKey`, `Shipped/lightKey`, `Shipped/refreshKey` | `Eq` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json`, `relation_rows_match_the_lean_checksum`: the rows that Rust expands from each package have the checksum of Lean's rows |
| `exports.lightLayout.*`, `exports.refreshLayout.*`, `exports.lightInside`, `exports.refreshInside` | `*` | lean | `Export.BoundedDepth.EmittedPackages.Facts.exports` builds the bundle of the two packages that the emitter writes, with the emitter's first fresh columns, source width and batches, so it carries these proofs |
| `fitsPackage.*`, `exports.lightFitsPackage.*`, `exports.refreshFitsPackage.*`, `Shipped/facts.*` | `*` | link | `Export.BoundedDepth.EmittedPackages.Facts.lightFits` and `Export.BoundedDepth.EmittedPackages.Facts.refreshFits` derive them from the capacity of `SPEC.md` section 4.4 (`Export.Generic.RelationFits.fits_of_capacity`): the S-box columns end within the source width and the rows fit the cube. The test `step_packages_fit_their_capacity` checks the capacity on each shipped package, the light and refresh packages of the reference application (`formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` and its refresh package) and of both Nebula programs, with Lean's row counts and logical widths (`factored_rows_match_the_lean_dense_checksum`). Lean does not prove the capacity for an application |
| `accepted` | `Lifecycle.BoundedDepth.Terminal.HoldsFor` | game | `SPEC.md` Algorithm 6 |
| `keysShaped` | `Eq` | lean | `Lifecycle.BoundedDepth.CircuitKeyDigest.keys_shaped`, for the keys `Lifecycle.BoundedDepth.CircuitKeyDigest.keys` of a package identity. Rust `circuit_key` computes these digests (section Hashed encodings, row Circuit key digest) |
| `assumption` | `Spec.HyperNova.ReverseStages.Assumption1` | assumption | `[HN-A1]` |
| `closed.*`, `admitted` | `*` | assumption | `[HN-A1]`, in the form of trust item 3: the closure premise and the admitted adversary |
| `Nebula/premise` | `Spec.Sponge.Transfer` | assumption | `[SPONGE]` under `[P2-PERM]` |
| `Fold/premise` | `Spec.Sponge.DuplexTransfer` | assumption | `[DSFS]` under `[P2-PERM]`, for a fold prover with at most `calls` permutation calls and a model adversary with at most `calls` queries. The event includes the extractor's retries (adversary 3). `calls` is a parameter of the adversary class, as in the row `finder`, and it is larger than the number of permutation calls of one verifier run |
| `Nebula/disjoint`, `Nebula/inRange`, `Nebula/images`, `Nebula/reorders`, `Nebula/bound` | `*` | open | Open: the conditions of a catalog application (`Spec.Nebula.MemoryTrace.Binding`). No Lean result shows that the binding rows of the Rust catalog backend enforce them (Out of scope) |
| `finder` | `Spec.Sponge.Transfer` | assumption | `[SPONGE]` under `[P2-PERM]`, for the collision finder of a fold with at most `calls` permutation calls and inputs of at most `blocks` padded blocks. The event is one output of one run. The finder reruns the adversary on the oracles of its own tape, so no retry is outside the run (adversary 3). `calls` is a parameter of the adversary class, as in the row below |
| `finders` | `Spec.Sponge.Transfer` | assumption | `[SPONGE]` under `[P2-PERM]`, for the finder of stage `j` with at most `calls j` permutation calls and inputs of at most `blocks` padded blocks. Lean has no count of the calls of a stage, which is a Lean function, so `calls` is a parameter of the adversary class, as `depth` is (adversary 4) |
| `msis` | `LE.le` | assumption | `[MSIS]`: for each admitted prover and each efficient extractor, the reduction `Spec.Compression.Akita.solve` returns a solution of Module-SIS with probability at most `bindingError`. The solution is for one of four instances: `A` at the norm `η_A = 2 · κ̄_1 · Δ^cert_f` of Akita (50), `B` at `b⋆ − 1`, or `F_1` or `F_2` at 1. For the shipped key shapes, `η_A` is the bound that the schedule file prices for `A`: 32505732 for `key20.aks` (`Spec.Compression.Akita.Descriptor.key20_innerBound`) and 16642998148 for `key30.aks` (`Spec.Compression.Akita.Descriptor.key30_innerBound`). The four matrices are views of one setup vector, which the code expands in pages of 4096 elements with one SHAKE256 stream for each page (`[KECCAK]`; for independent uniform page streams, `Spec.Compression.Akita.Replay.paged_le` bounds this expansion against a uniform vector), so Akita Lemma 10.20 joins the instances: it bounds the probability by the sum of the four advantages against a uniform matrix. Lean proves item (i) of that lemma for a uniform setup vector (`Spec.Compression.Akita.Setup.shared_setup_eq`). No theorem applies it to `Spec.Compression.Akita.solve` or to the SHAKE256 page streams |
| `honest.*`, `S.*`, `covers` | `*` | lean | `Spec.Compression.Akita.Shape.key20_honest` and `Spec.Compression.Akita.Shape.key30_honest`, for the shapes `Spec.Compression.Akita.Shape.key20` and `Spec.Compression.Akita.Shape.key30` of the key group of each schedule file, which carry the proofs of `S.*`. `tests/AkitaSchedules.lean` compares the numbers of the shapes with both files. A file has no field for `b⋆` and no shape of `F_1` and `F_2`: `b⋆ = 8` is from the bound 7 of the file for `B`, and the shapes of the two maps are from the Akita code (not checked against Lean). For `covers`: `Spec.Compression.Akita.Descriptor.key20_covers` and `Spec.Compression.Akita.Descriptor.key30_covers` give that the norm `β̄_∞ + κ̄_1 · b/2` of the solution of the reduction for `A`, 293887 and 136249343, is at most `η_A`. They are for the descriptors `Spec.Compression.Akita.Descriptor.key20` and `Spec.Compression.Akita.Descriptor.key30` of the root opening of the key group of each file, where `κ̄_1` and `β̄_∞` are the numbers of Akita (50): `κ̄_1 = 2 · 31 = 62`, and `β̄_∞ = Δ^cert_f` is `8^6 − 1` and `8^9 − 1`. The test also compares the challenge counts, the opening base, the fold digits and the bound of `A` with both files. No Lean result shows that the verifier of the root fold certifies these two bounds: that is the root weak opening of Akita Definition 10.22 |
| `Akita/structure_.protocol.*` | `*` | lean | `Spec.Compression.Akita.Extract.transcript` is the protocol of each Akita structure of Lean (`Spec.Compression.Akita.Extract.Tier.Suffix`, field `protocol`), so it carries these proofs. The three checkpoint claims are for each protocol with these laws |
| `Akita/lawful.*` | `*` | lean | `Spec.Compression.Akita.Extract.Tier.Suffix.lawful` gives the conditions of Akita Definition 10.16 for a suffix from the stage 0, and `Spec.Compression.Akita.Key20.Tower.suffix` gives this suffix for the opening of `crates/ns-compress/artifacts/key20.aks`, under the rows `Akita/drawn.*`. For another structure they are conditions of the claim. `Spec.Checkpoint.Structure.Lawful` has inclusions where the definition has equalities (trust item 7) |
| `Akita/bounded`, `AkitaKey20/bounded`, `Akita/positive`, `AkitaKey20/positive`, `AkitaKey30/bounded`, `AkitaKey30/positive`, `Shipped/bounded`, `Shipped/positive` | `*` | game | adversary 6: the prover makes at most `Q` queries to the table of the stage answers, and the limit `budget` on the attempts of a retry loop of the extractor is positive. `Q` and `budget` are parameters of the bound |
| `Akita/nonemptySlot`, `AkitaKey20/nonemptySlot`, `AkitaKey20/nonemptyCoord`, `AkitaKey20/finiteWire`, `AkitaKey20/slots`, `AkitaKey20/coords`, `AkitaKey30/nonemptyCoord`, `AkitaKey30/finiteWire`, `AkitaKey30/slots`, `AkitaKey30/coords`, `Shipped/nonemptyCoord`, `Shipped/finiteWire`, `Shipped/coords` | `*` | game | adversary 6: the experiment of the Akita claims. The wire messages are a finite set, and an answer of a stage has one value for each of its slots: 22 slots or more for the opening of `key20.aks` and 28 or more with the argument, and 28 and 40 for `key30.aks`. The code makes each answer with a sponge, and no Lean statement joins the two (trust item 7) |
| `Akita/result`, `AkitaKey20/result`, `AkitaKey30/result` | `*` | game | adversary 6: the event of `Claims.Akita.checkpoint_sound`, `Claims.AkitaKey20.key20_opening_sound` and `Claims.AkitaKey30.key30_opening_sound` is an output of the extractor |
| `Akita/shape.*`, `Akita/opening.*` | `*` | lean | `Spec.Compression.Akita.Key20.root` and `Spec.Compression.Akita.Key20.traces` have the commitment shapes and the opening shapes of the Lean schedule of `crates/ns-compress/artifacts/key20.aks`, which carry these proofs. The four fold claims are for each shape |
| `Akita/opens.*`, `Akita/opens'.*`, `Akita/different`, `Akita/radius`, `Akita/accepts.*` | `*` | game | adversary 6: the event of a reduction to Module-SIS. It is two weak openings of one statement (Akita Definition 10.22) that differ, with a norm `radius` that covers their bounds, or a fork of two children for each block that satisfy their branch relations with carried digits `opens` (Akita Corollary 10.32) |
| `Akita/units`, `Akita/sparse` | `*` | lean | `Spec.Compression.Akita.Key20.units_of_mem` and `Spec.Compression.Akita.Key20.sparse_of_mem`, for the fold decoders of the code on the slots `Spec.Compression.Akita.Replay.CodeSlot` (`Spec.Compression.Akita.Replay.open_mem`, `Spec.Compression.Akita.Replay.kept64_mem`, `Spec.Compression.Akita.Replay.kept128_mem`), from `Claims.Akita.short_units`. No Lean theorem joins the verifier of the code `Spec.Compression.Akita.Replay.verifyProof` to the Lean verifier of the opening (trust item 7) |
| `Akita/base` | `LE.le` | lean | `Spec.Compression.Akita.Key20.root_sized` and `Spec.Compression.Akita.Key20.traces_sized`: `b_z ≥ 2` at each fold of the Lean schedule of `crates/ns-compress/artifacts/key20.aks` |
| `Akita.short_units/neZeroNatD`, `Akita.short_units/power`, `Akita.short_units/nonzero`, `Akita.short_units/short` | `*` | lean | `Spec.Compression.Akita.SparseFamily.unitDifferences` meets them for each difference of two members of a sparse family of a degree `2^{n+1}`: the difference is not 0, its coefficients are at most 4, and `2 · 4² < q` |
| `AkitaKey20/drawn.injective`, `AkitaKey20/drawn.injective₁`, `AkitaKey20/drawn.injective₂`, `AkitaKey20/drawn.injective₃`, `AkitaKey20/drawn.injective₄`, `AkitaKey20/drawn.injective₅`, `AkitaKey20/drawn.injectiveL`, `AkitaKey20/drawn.next`, `AkitaKey20/drawn.inside`, `AkitaKey20/drawn.decodeL`, `AkitaKey20/inside`, `AkitaKey30/drawn.injective`, `AkitaKey30/drawn.injective₁`, `AkitaKey30/drawn.injective₂`, `AkitaKey30/drawn.injective₃`, `AkitaKey30/drawn.injective₄`, `AkitaKey30/drawn.injective₅`, `AkitaKey30/drawn.injective₆`, `AkitaKey30/drawn.injectiveL`, `AkitaKey30/drawn.next`, `AkitaKey30/drawn.inside`, `AkitaKey30/drawn.decodeL`, `AkitaKey30/inside` | `*` | game | adversary 6: the stage layout of the Lean verifier of the opening. The fold coordinates of a level have different slots, the fold challenge of the terminal is the stage after the last level and a stage of the transcript, and the decoder of the terminal is `Spec.Compression.Akita.Replay.CodeSlot.kept64`, the decoder of the code for the filtered family `(64, 31, 11)`. This stage layout is not the layout of the code, and no Lean theorem joins the two (trust item 7). The filters of the families are the predicate `accept_strict_parts` of the pinned code (`Spec.Compression.Akita.TrigTable.codeAccepts`), and Lean proves the error bound of its integer tables (`Spec.Compression.Akita.TrigTable.code_accurate`), so the operator bound `Γ` is a theorem (`Spec.Compression.Akita.TrigTable.filtered_code`) |
| `AkitaKey20/drawn.kept`, `AkitaKey30/drawn.kept`, `AkitaKey20/tail` | `*` | open | Open: `2^128` members or more of each filtered challenge family pass the filter of the code, the family `(31, 11)` of the degree 64 for `Γ = 18` and the family `(31, 0)` of the degree 128 for `Γ = 13`. Akita Table 12 certifies the lower bounds `2^128.062439` and `2^128.563317` (the size of the base family `C_0` times the pass probability `p_0`) on the accepted families of the fixed-point filter at the scale `2^48`, from exact moments and a rational tail polynomial, and Lean does not check this certificate. The claims use `kept` for two facts only: each slot decodes at a filtered level (`Spec.Compression.Akita.Replay.kept64_isSome`, `Spec.Compression.Akita.Replay.kept128_isSome`; so `Spec.Compression.Akita.Key20.Tower.Drawn.decodes`, `Spec.Compression.Akita.Key30.Tower.Drawn.decodes`, `Export.BoundedDepth.FinalKey20.Opening.Drawn.decodes`), and a filtered fold coordinate has the price `1/card (Kept G) ≤ 1/2^128 ≤ 1/q²` (`Spec.Compression.Akita.Replay.slotChance_kept64_le`, `Spec.Compression.Akita.Replay.slotChance_kept64_field`). The opening of `key30.aks` uses only the family of the degree 64. The pinned Akita checkout has the certificate and a checker (`scripts/operator_norm`), and the Rust code has only the constant 128. Lean proves the union bound and the equal-marginal step of Akita Theorem H.2 for the integer predicate of the code: the count follows from the count at the frequency 0 (`Claims.AkitaKey20.key20_kept`, hypothesis `tail`, `Spec.Compression.Akita.Key20.Tail`), with the error bound `ε + 1` of the tables, because the tables of the code are antisymmetric only up to 1 (`Spec.Compression.Akita.TrigTable.code_antisymmetric`, `Spec.Compression.Akita.TrigTable.code_cos_one`). `Tail` allows a fraction of rejected members at the frequency 0 of at most 0.024219975628 (degree 64) and 0.010546520952 (degree 128); the certificate of the paper gives 0.023909042053 and 0.008120720198. Lean does not prove `Tail`: it needs the moments of Akita (292), the polynomial of (293) and the left inclusion of Theorem H.1 |
| `AkitaKey20/decodes`, `AkitaKey20/decodes₁`, `AkitaKey20/decodes₂`, `AkitaKey20/decodes₃`, `AkitaKey20/decodes₄`, `AkitaKey20/decodes₅`, `AkitaKey20/decodesL`, `AkitaKey30/decodes`, `AkitaKey30/decodes₁`, `AkitaKey30/decodes₂`, `AkitaKey30/decodes₃`, `AkitaKey30/decodes₄`, `AkitaKey30/decodes₅`, `AkitaKey30/decodes₆`, `AkitaKey30/decodesL` | `Eq` | game | adversary 6: `Claims.AkitaKey20.key20_opening_accepts` and `Claims.AkitaKey30.key30_opening_accepts` are for each table on message keys whose slots decode to fold challenges |
| `AkitaKey20/worked`, `AkitaKey30/worked` | `Membership.mem` | game | adversary 6: `Claims.AkitaKey20.key20_opening_accepts` and `Claims.AkitaKey30.key30_opening_accepts` are for each table on message keys whose work slots have 57 zero work bits (`Spec.Compression.Akita.Extract.workZeros`), in place of the nonce search of the honest prover |
| `AkitaKey20/live`, `AkitaKey20/lengths`, `AkitaKey20/fit`, `AkitaKey30/live`, `AkitaKey30/lengths`, `AkitaKey30/fit` | `*` | game | adversary 6: `Claims.AkitaKey20.key20_opening_accepts` and `Claims.AkitaKey30.key30_opening_accepts` are for tables of the key polynomial, `W` and `V` with the sizes of the schedule, for points with their counts of coordinates, and for the setup prefix and the tables whose honest responses satisfy the fit conditions of Akita Definition 10.5 for each fold challenge vector that the decoders give (`Spec.Compression.Akita.Key20.Fit`, `Spec.Compression.Akita.Key30.Fit`). `Spec.Compression.Akita.Key20.fit_zero`, `Spec.Compression.Akita.Key30.fit_zero`, `Spec.Compression.Akita.Key20.fit_quiet` and `Spec.Compression.Akita.Key30.fit_quiet` give data that satisfies them. For the challenge families of the schedule and data that is not 0, no Lean result gives `fit` |
| `AkitaKey20/fresh`, `AkitaKey20/apart`, `AkitaKey20/fair`, `AkitaKey20/energy`, `AkitaKey20/digits`, `AkitaKey20/sampler`, `AkitaKey20/total` | `*` | game | adversary 6: `Claims.AkitaKey20.key20_root_search` and `Claims.AkitaKey20.key20_level1_search` bound the event that each nonce fails, where a nonce fails when a fold challenge does not decode or the response does not fit. `sampler` is the probability `ε` that a fold challenge of one answer does not decode (the sampler term of Akita (157)), and the bound is `(ε + δ)^4096`. `Claims.AkitaKey20.key20_root_search_total` and `Claims.AkitaKey20.key20_level1_search_total` are the case `ε = 0` of decoders that decode each slot (`total`). The claims are for one transcript prefix with different keys of the nonces, different slots of the fold coordinates, decoders on which the sign flips of a challenge act (`Spec.Compression.Akita.SignFair`; `Spec.Compression.Akita.signFair_signed` gives one), challenges with the energy of the families, and source blocks with the digit bound of the level. For the decoders of the code, `Spec.Compression.Akita.Key20.rootDecode_signFair` and `Spec.Compression.Akita.Replay.signFair_open` give `fair`, and `Spec.Compression.Akita.Key20.rootDecode_isSome` and `Spec.Compression.Akita.Replay.open_isSome` give `total` |
| `AkitaKey20/blocks`, `AkitaKey20/use.blocks` | `Eq` | link | `crates/ns-compress/artifacts/key20.aks`, `each_schedule_file_has_its_pinned_digest_and_its_rows`: the file of the key of 512 blocks has its pinned digest, and Rust `SCHEDULES` has the Lean schedules (`schedules_and_variables_are_lean_constants`). The claims are for a verifier that takes this key. The key of the light relation of the reference application has `2^19` blocks (row `Final.light_gate_complete/shipped`), so these claims do not cover it |
| `AkitaKey20/use.laid`, `AkitaKey30/use.laid` | `Export.BoundedDepth.FinalVerifier.Setup.Laid` | game | `SPEC.md` Algorithm 10: step 5 lays out the claims, and the claim is for claims that pass the layout |
| `AkitaKey20/use.zeros`, `AkitaKey30/use.zeros` | `Eq` | lean | `Export.BoundedDepth.FinalKey20.laidSetup_use`, for the setup `Export.BoundedDepth.FinalKey20.laidSetup` of the Akita claims of the final verifier, and `Export.BoundedDepth.FinalKey30.Opening.knowledge` for this setup under `key30.aks`. No Lean result compares `Export.BoundedDepth.FinalKey20.laidZeros` with the Rust code (trust item 7) |
| `AkitaKey20/drawn` | `Export.BoundedDepth.FinalKey20.Opening.Drawn` | open | Open: the conditions `Export.BoundedDepth.FinalKey20.Opening.Drawn` of the opening of each claim shape are one hypothesis, and nothing gives three of its parts. `decode` to `decodeL` fix the fold decoders of the code on the slots `Spec.Compression.Akita.Replay.CodeSlot`, and no Lean theorem joins the verifier of the code `Spec.Compression.Akita.Replay.verifyProof` to the Lean verifier `Spec.Compression.Akita.Extract.towerLinks` (trust item 7). `kept` is the row `AkitaKey20/drawn.kept`. `registered` is a certified decommitment of the key commitment of the verifier to the key polynomial, and no Lean result or Rust test gives it for the pinned key. The other parts are the layout of step 5 and the stage layout of the rows above |
| `AkitaKey20.key20_instances/mass`, `AkitaKey20.key20_instances/weight₁`, `AkitaKey20.key20_instances/weight₂`, `AkitaKey20.key20_instances_shared/mass`, `AkitaKey20.key20_instances_shared/weight₁`, `AkitaKey20.key20_instances_shared/weight₂` | `Eq` | game | adversary 6: the claim is for the opening of `Claims.AkitaKey20.key20_verifier_sound`. The hypothesis `Export.BoundedDepth.FinalKey20.Opening.Drawn` of that claim (row above) has these three equations as fields: the norms `‖c‖_1` of the fold challenges of the root and of the levels 1 and 2, which fix the bounds of the witnesses of `R_bind` |
| `AkitaKey20.key20_instances/solved`, `AkitaKey20.key20_instances_shared/solved`, `AkitaKey30/solved` | `Export.BoundedDepth.FinalAkita.Solved` | game | adversary 6: the event of the claim is an output of the extractor that is a solution of Module-SIS, which is the event of `Export.BoundedDepth.FinalAkita.breakChance` |
| `AkitaKey30/blocks`, `AkitaKey30/use.blocks` | `Eq` | link | `crates/ns-compress/artifacts/key30.aks`, `each_schedule_file_has_its_pinned_digest_and_its_rows`: the file of the key of `2^19` blocks has its pinned digest, and Rust `SCHEDULES` has the Lean schedules (`schedules_and_variables_are_lean_constants`). The claims are for a verifier that takes this key, which is the size of the key of the light relation of the reference application (row `Final.light_gate_complete/shipped`) |
| `AkitaKey30/use.row` | `Eq` | game | `SPEC.md` Algorithm 10: step 5 lays out the claims with a row of the schedule file, and the claim is for the row of `Spec.Compression.Akita.Key30.rows` that has the count of variables of `W`, 27 or 26 |
| `AkitaKey30/member`, `AkitaKey30/o.member` | `Membership.mem` | game | adversary 6: the claim is for an opening under a row of the Lean schedule `Spec.Compression.Akita.Key30.rows` of `crates/ns-compress/artifacts/key30.aks`. `tests/AkitaSchedules.lean` compares the numbers of each row with the file |
| `AkitaKey30/shared`, `AkitaKey30/keys`, `AkitaKey20.key20_instances_shared/keys`, `Shipped/shared` | `*` | game | adversary 6: the claim is for an opening whose matrices `A` and `B` of each group of the root and `D` of the root (`shared`, `Spec.Compression.Akita.Key30.Tower.Shared`), or all matrices (`keys`, `Spec.Compression.Akita.Key30.Shared`, `Spec.Compression.Akita.Key20.Shared`), are the views (Akita (35), `Spec.Compression.Akita.Setup.shared`) of one setup vector with the degree, the rank and the width of their instances, as the shared setup of the paper defines them. The Akita code reads each matrix of each level from one stream from the offset 0. From `shared` Lean proves the split of the public weight of the root into the setup part, Akita (132), in the form of the pinned code with the factor `powerScale` (`Spec.Compression.Akita.Batch.weight_geometric`, `Spec.Compression.Akita.Batch.codeScan`). `Spec.Compression.Akita.Key30.Tower.onSetup_shared`, `Spec.Compression.Akita.Key30.shared_ofSetup` and `Spec.Compression.Akita.Key20.shared_ofSetup` give data that satisfies them, and `keys` at the setup prefix gives `shared` (`Spec.Compression.Akita.Key30.Tower.shared_of_views`), so the claims are not empty |
| `AkitaKey30/drawn.registered` | `Exists` | open | Open: the premise "correct preprocessing" of Akita Lemma 9.1. The commitment of the setup prefix in the verifier setup is the output of `Spec.Compression.Akita.commit` for the table of the setup prefix, with the decommitment of the algorithm as the reference (`Spec.Compression.Akita.Batch.registered_commit`). No Lean result or Rust test gives it for the prefix commitments of the pinned key `crates/ns-compress/artifacts/key30.vk` |
| `AkitaKey30/drawn`, `Shipped/drawn` | `Export.BoundedDepth.FinalKey30.Opening.Drawn` | open | Open: the conditions `Export.BoundedDepth.FinalKey30.Opening.Drawn` of the opening of each claim shape are one hypothesis, and nothing gives four of its parts. `opening.registered` is the row `AkitaKey30/drawn.registered`, and `opening.kept` is the row `AkitaKey20/drawn.kept`. `registered` is a certified decommitment of the key commitment of the verifier to the key polynomial, and no Lean result or Rust test gives it for the pinned key. `opening.decodeL` fixes the decoder of the code `Spec.Compression.Akita.Replay.CodeSlot.kept64` of the terminal, and no Lean theorem joins `Spec.Compression.Akita.Replay.verifyProof` to the Lean verifier `Spec.Compression.Akita.Extract.offLinks` (trust item 7). The other parts are the layout of step 5, the row of the file and the stage layout of the rows above |
| `Akita.module_sis_descent/neZeroNatD`, `Akita.module_sis_descent/neZeroNatDCar`, `Akita.module_sis_descent/tower` | `*` | lean | `Spec.Compression.Akita.Key20.maps_tower`, for the degrees 64 and 32 of the two compression maps, with `stride = 2`. The degree of an instance is not 0 (`Spec.Compression.Akita.Setup.Instance.positive`). The claim is for each pair of degrees with `stride · d_car = d` |
| `valid.toValid.*`, `secure` | `*` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-nebula-gate-parity-v1.json`, `is_valid_and_is_secure_are_the_lean_decisions`: Rust checks rules 1 to 6 of `SPEC.md` section 7.1 on each geometry at run time, and the test compares that check with Lean's decisions. For the fixed WASM sizes Lean proves both (`Spec.Nebula.WasmSize.valid`, `Spec.Nebula.WasmSize.secure`), and `each_wasm_size_has_the_lean_geometry_and_rows` compares the geometries |
| `valid.romWords`, `valid.ramWords` | `LT.lt` | game | `SPEC.md` section 7.1: a program is two images of words below `2^32`, so these bounds define the statements of the game. Rust holds each image as `Vec<u32>` (`nebula::Program`), so each program that the verifier receives meets them |
| `app.*` | `*` | open | Open: `Lifecycle.Nebula.FirstPlan.app` is the test machine. A Rust-built memory application, the WASM relation included, has no instance (Out of scope) |
| `round` | `Export.BoundedDepth.NebulaSecurity.MemoryRound` | open | Open: trust item 6. No result derives the memory round from the assumption list |
| `sampler.*` | `*` | game | adversary 5: the `η` sampler and its query bound |
| `wf.*`, `slots_le`, `key_fits` | `*` | lean | `Spec.Compression.Rust.layoutNew_production`, for the layout of the Lean model of `Layout::new` |
| `counts` | `Spec.Compression.Verifier.CountsNonzero` | lean | `Spec.Compression.Production.params_countsNonzero` |
| `matrix` | `Spec.Compression.Verifier.MatrixSpec` | lean | `Spec.Compression.Production.verifierData_matrixSpec`, for the Lean model of Rust `matrix_part`. Rust `matrix_part` gives the output of the model on three sample cases (`formal/nightstream-fprime/artifacts/nightstream-fprime-matrix-pass-parity-v1.json`, `the_matrix_part_is_the_lean_matrix_pass`). No data compares the two on the rows of the step circuits |
| `columns`, `rows` | `LE.le` | lean | `Claims.Compression.production_sound` |
| `statement.*`, `constantLaw.*` | `*` | game | adversary 6: the statement of the Π_CCS run and the law of its constant |
| `params.rlc_bound`, `freshBound`, `degreeCovers` | `*` | lean | `Claims.Final.verifier_sound` uses the Π_CCS run at the production parameters and takes none of these |
| `Final/rows`, `AkitaKey20/rows`, `AkitaKey30/rows`, `Shipped/rows` | `Eq` | link | `Relation::new` gives `FactoredRows` the package's plan rows and the spans of its layout (`crates/nightstream/src/lifecycle/bounded_depth/relation.rs`). Lean emits the row counts, the logical width and a checksum of the dense matrices of its model `Export.BoundedDepth.FinalRows.packageRows` for both packages of the reference application (`formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json`) and of each Nebula program (`formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-nebula-v1-rows-v1.json`, `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-nebula-n1-rows-v1.json`), and `Export.BoundedDepth.RowsParity.fastSum_eq` proves that the emitted value is that checksum. The test `factored_rows_match_the_lean_dense_checksum` computes the same words from Rust's plan rows and spans for each of these packages. A checksum is evidence, not a proof. The memory claims are about Lean's Nebula programs; a Rust-built memory application is row `app.*` |
| `depthBound` | `LE.le` | game | adversary 4: the result bounds the error over the stages below a bound `depth` on the iteration of the adversary's statements, so `depth` is a parameter of the adversary class, and the error grows with it. The history results of adversary 6 (`Claims.Final.history_sound`, `Claims.AkitaKey20.key20_history_sound`, `Claims.AkitaKey30.key30_history_sound`) take the same parameter. The memory results of adversary 6 do not: the checks of `SPEC.md` Algorithm 8 step 4 accept at most `S_max · N` steps (`Lifecycle.Nebula.EndToEnd.VerifierChecks.iteration_le`), so their walk has `S_max · N` stages |
| `Encodings/same`, `Encodings/equal`, `Encodings/different` | `*` | game | `SPEC.md` section 4: the event of each encoding claim is two inputs with one digest or one block list |
| `Encodings/leftWell.*`, `Encodings/rightWell.*` | `*` | game | `SPEC.md` section 4.2: the two states are well formed |
| `Encodings/leftCanonical`, `Encodings/rightCanonical` | `*` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json`, `the_identity_words_are_the_lean_words`: Rust makes the words of Lean, and it refuses a stream with a word at or above `q` (`a_word_at_or_above_q_is_refused`) |
| `Encodings/fits`, `Encodings/fits'` | `*` | game | `SPEC.md` section 4.4: each atom and each array length of a package is below `2^64`, the range of Rust `u64` |
| `Foundations/accepted`, `Foundations/truth`, `Foundations/falseClaim`, `Foundations/represents`, `Foundations/message.coefficients_length`, `Foundations/semantic.coefficients_length` | `*` | game | `SPEC.md` Algorithm 4: the event of the sum-check claims is an accepted transcript with a false claim, and a round polynomial of the stated degree |
| `Fold/accepted`, `Fold/fork.*`, `Fold/system`, `Fold/combined` | `*` | game | `SPEC.md` Algorithm 4: the fold verifier accepts a claim of the key's relation, and the fork of its challenges is an input of the two extraction claims |
| `Fold/freshOpening`, `Fold/runningOpening`, `Fold/truth`, `Fold/accumulatorHolds`, `Fold/freshHolds` | `*` | game | `SPEC.md` Algorithm 5: the honest prover holds a witness of each input claim |
| `Fold/running.point.dimension`, `Fold/start.point.dimension`, `Fold/result.*`, `Fold/accumulator.*`, `Fold/parent.*` | `*` | game | `SPEC.md` section 2.5: a claim has a point of the cube dimension and a constraint system that fits its cube |
| `Fold.light_complete/key.*`, `Fold.light_extracts/key.*`, `Fold.refresh_complete/key.*`, `Fold.refresh_extracts/key.*` | `*` | lean | `Spec.HyperNova.Construction2.BoundedDepthKeys.lightKey` and `Lifecycle.BoundedDepth.refreshKey` are the keys of the two folds, so they carry these proofs |
| `Fold.light_extracts/laws.*` | `*` | lean | `Lifecycle.BoundedDepth.lightExtraction` is the extraction algebra of the production light key, so it carries these proofs |
| `Fold.refresh_extracts/laws.*` | `*` | lean | `Lifecycle.PaperExtractionAlgebra.extractionAlgebra` is the extraction algebra of the production refresh key, so it carries these proofs |
| `Fold/normLaw.sub_bound` | `*` | lean | `Lifecycle.BoundedDepth.lightExtractionNormLaw` is this law for `Lifecycle.BoundedDepth.lightExtraction` |
| `Fold/oneEq` | `Eq` | lean | `Lifecycle.BoundedDepth.lightExtraction_one` is this equation for the production light key and `Lifecycle.BoundedDepth.lightExtraction` |
| `Fold/capFits`, `Ivc/capFits` | `LE.le` | lean | `Spec.Folding.Schedule.production_cap_fits`, at the production cap |
| `Fold/p.rlc_bound`, `Ivc/p.rlc_bound` | `LT.lt` | lean | `Spec.productionGlobalParams` is the parameter record of the profile, so it carries this proof |
| `Fold/admissible`, `Fold/member`, `Fold/start` | `*` | game | `SPEC.md` section 3: the claim is about each depth that an admissible schedule reaches |
| `Fold/positive` | `LT.lt` | game | adversary 3: the work budget of the extractor is positive |
| `Step/holds`, `Step/represents.*` | `*` | game | `SPEC.md` Algorithm 5: the honest prover has an assignment of the step circuit that carries its step and its fold |
| `Step/publicBelow`, `Ivc/complete.lightLayout`, `Ivc/complete.refreshLayout`, `Ivc/complete.lightPublicBelow`, `Ivc/complete.refreshPublicBelow`, `Ivc/complete.lightFreshAfterLocals`, `Ivc/complete.refreshFreshAfterLocals`, `Nebula/complete.lightLayout`, `Nebula/complete.refreshLayout`, `Nebula/complete.lightPublicBelow`, `Nebula/complete.refreshPublicBelow`, `Nebula/complete.lightFreshAfterLocals`, `Nebula/complete.refreshFreshAfterLocals`, `Final/complete.lightLayout`, `Final/complete.refreshLayout`, `Final/complete.lightPublicBelow`, `Final/complete.refreshPublicBelow`, `Final/complete.lightFreshAfterLocals`, `Final/complete.refreshFreshAfterLocals`, `AkitaKey20/complete.lightLayout`, `AkitaKey30/complete.lightLayout`, `AkitaKey20/complete.refreshLayout`, `AkitaKey30/complete.refreshLayout`, `AkitaKey20/complete.lightPublicBelow`, `AkitaKey30/complete.lightPublicBelow`, `AkitaKey20/complete.refreshPublicBelow`, `AkitaKey30/complete.refreshPublicBelow`, `AkitaKey20/complete.lightFreshAfterLocals`, `AkitaKey30/complete.lightFreshAfterLocals`, `AkitaKey20/complete.refreshFreshAfterLocals`, `AkitaKey30/complete.refreshFreshAfterLocals` | `*` | lean | `Export.BoundedDepth.EmittedPackages.Emitted.complete` builds the completeness bundle of the emitted packages: their layouts are the widened slot layouts, their first fresh columns are after the public columns (`Export.BoundedDepth.EmittedPackages.publicColumns_le_lightFresh`, `Export.BoundedDepth.EmittedPackages.publicColumns_le_refreshFresh`), and they are the ends of the circuits' locals (`Export.BoundedDepth.EmittedPackages.lightFresh`, `Export.BoundedDepth.EmittedPackages.refreshFresh`) |
| `Step/readsOwnColumns`, `Step/publicEq` | `*` | lean | `Export.BoundedDepth.StepExtend.light_extend`, `Export.BoundedDepth.StepExtend.refresh_extend` and `Export.BoundedDepth.StepExtend.base_extend` discharge them from the honest step's assignment (`Layout.BoundedDepth.LightAssemblerCompleteness.complete`, `Layout.BoundedDepth.RefreshAssemblerCompleteness.complete`): its public columns carry `encHash` of the output hash, and each row reads only columns before the end of the circuit's locals, so each piece reads only columns before a first fresh column at or after that end (`Export.BoundedDepth.StepExtend.Complete.lightReadsOwnColumns`, `Export.BoundedDepth.StepExtend.Complete.refreshReadsOwnColumns`) |
| `Step/fitsKeyWidth`, `Ivc/complete.lightFitsKeyWidth`, `Ivc/complete.refreshFitsKeyWidth`, `Nebula/complete.lightFitsKeyWidth`, `Nebula/complete.refreshFitsKeyWidth`, `Final/complete.lightFitsKeyWidth`, `Final/complete.refreshFitsKeyWidth`, `AkitaKey20/complete.lightFitsKeyWidth`, `AkitaKey30/complete.lightFitsKeyWidth`, `AkitaKey20/complete.refreshFitsKeyWidth`, `AkitaKey30/complete.refreshFitsKeyWidth` | `*` | link | `SPEC.md` section 4.4 (Capacity): the slot layout of each package fits the key width `m = 2^26`. These are the fields `lightFitsKeyWidth` and `refreshFitsKeyWidth` of `Export.BoundedDepth.EmittedPackages.Facts`. The test `step_packages_fit_their_capacity` checks the capacity on each shipped package, the light and refresh packages of the reference application (`formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` and its refresh package) and of both Nebula programs, with Lean's row counts and logical widths (`factored_rows_match_the_lean_dense_checksum`). Lean does not prove the capacity for an application. Rust gives each relation the width of its own package's slot layout (`crates/nightstream/src/lifecycle/bounded_depth/relation.rs`) |
| `Ivc.accepted_bounds/accepted`, `Ivc/finalAtMostB`, `Ivc/start` | `*` | game | adversary 4, the result for one slot: an accepted history of the slot |
| `Ivc/validRun`, `Ivc/witnessLengths`, `Ivc/runBelow`, `Ivc/z0Length`, `Ivc/iterationNext`, `Ivc/valid`, `Ivc/verifies`, `Ivc/witnessLength` | `*` | game | `SPEC.md` Algorithm 5: the honest prover extends an accepted proof by a valid step, with a counter below `q` |
| `Nebula/accepted.*`, `Nebula/start`, `Nebula/steps`, `Nebula/terminal.*` | `*` | game | `SPEC.md` Algorithm 8: the memory verifier accepts the chain of steps and its terminal state |
| `Nebula/count`, `Nebula/exec`, `Nebula/range`, `Nebula/width` | `*` | game | `SPEC.md` Algorithm 7: the honest prover runs an execution of the machine |
| `Ivc.light_run_complete/positive`, `Ivc.light_run_complete/short` | `*` | game | `SPEC.md` Algorithm 9: the light mode is for an honest run of 1 to `cap + 1` steps |
| `Nebula.memory_light_complete/light` | `LE.le` | lean | `Spec.Nebula.WasmSize.steps_le_light` proves `S_max · N ≤ cap + 1` for each WASM size (`SPEC.md` Algorithm 9). For another geometry it is a condition of the claim |
| `Fold/bounded` | `Spec.RandomOracle.OracleComp.QueryBound` | game | adversary 3: the adversary makes at most `Q` oracle queries |
| `Ivc.slot_history_failure/assumption` | `Spec.HyperNova.Construction2.BoundedDepthSecurity.Assumption1` | assumption | `[HN-A1]`, in the form for one slot |
| `Compression.production_error/columns`, `Compression.production_error/span_le`, `Compression.production_error/slots` | `*` | lean | `Export.BoundedDepth.FinalVerifier.game_densityLe` meets the three for the verifier of the final verifier. The mode has at most two slot claims (`Export.BoundedDepth.FinalVerifier.Mode.slots_le`). `columnVars = 2` holds by the definition of `Spec.Compression.Production.verifierData`, and `Export.BoundedDepth.FinalVerifier.Setup.layout_facts` gives `spanVars ≤ 26` for the layout of the Lean model of `Layout::new` (`Spec.Compression.Rust.layoutNew_span`) |
| `Compression.production_error/depths_le`, `Compression.light_error_bits/depth_le` | `LE.le` | game | `SPEC.md` Algorithm 10: step 4 requires that the depth of each slot is at most `cap` |
| `Compression.light_error_bits/ok` | `Eq` | game | `SPEC.md` Algorithm 10: step 5 lays out the claims of the light mode with Rust `Layout::new`, whose model is `Spec.Compression.Rust.layoutNew`, and the claim is about each layout that the model accepts |
| `Compression.argument_complete/shaped.*`, `Compression.argument_complete/relation`, `Compression.argument_complete/honest` | `*` | game | `SPEC.md` Algorithm 9: the honest prover names claims of the shape of its layout (step 3), holds their witnesses, and fills the count slices with the honest counts (step 4) |
| `Compression.argument_complete/distinct` | `Spec.Compression.Verifier.TablesDistinct` | lean | `Spec.Compression.Production.params_tablesDistinct`, at the depths up to the cap |
| `Final.light_gate_complete/key` | `Eq` | game | `SPEC.md` Algorithm 10: the verifier takes the pinned key of the block count of Algorithm 11 for its relations and mode |
| `Final.light_gate_complete/shipped` | `Eq` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-compression-parity-v1.json`, `error_bits_and_the_gate_match_the_lean_bound_on_each_light_claim_set`: the key of the light relation of the reference application has the `light_key_blocks` of Lean, `2^19`. For each WASM size, `the_small_light_relation_fits_the_key`, `the_medium_light_relation_fits_the_key` and `the_large_light_relation_fits_the_key` check the key of `2^19` blocks (ignored, by hand) |
| `Final.light_gate_complete/minimum_le` | `LE.le` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-fold-gate-parity-v1.json`, `the_gate_charges_the_lean_error_on_each_step_relation`: `Relation::new` runs the fold gate with the minimum of the verifier, and the gate refuses each minimum above 115 bits, so each verifier that Rust builds has a minimum of at most 115 (`SPEC.md` section 3) |
| `Final.light_gate_complete/verifies`, `Final.light_gate_complete/pcOne`, `Final.light_gate_complete/slotOne` | `*` | lean | `Claims.Ivc.light_run_complete` gives the three for the honest proof of each valid run of 1 to `cap + 1` steps |
<!-- premises:end -->

## Applications and setups

An application is a step program (`Lifecycle.Stage1.Application.Program`).
Both step circuits run the same application step (`Lifecycle.BoundedDepth.machine`).
Adversary 1 holds for every application at the relation that Lean exports for
it, and adversary 2 for every setup whose circuit keys carry these relations.
Adversaries 3 and 4 hold for every key of the light and refresh folds.
`Poseidon2HashChainV1` (`Lifecycle.Stage1.Poseidon2HashChainV1.program`) is
the reference application of the packages and the executables.

No adversary chooses the application or the setup. The verifier's packages
fix the application, the fixed public seed of `SPEC.md` Algorithm 1 fixes the
commitment key, and the verifier pins the identity of the packages (`SPEC.md`
Algorithm 2). A memory application (adversary 5) is such an application: its
packages also fix the memory geometry, and the verifier takes the program,
which the adversary does not choose. The identity is the Poseidon2 hash of the
encoding of the light and refresh packages
(`Export.BoundedDepth.StepPackage.identity`), and every format decodes what it
encodes. Two pairs of packages with one identity are equal, or their two hash
inputs are distinct and have one hash
(`Claims.Encodings.package_identity_binds`), when each atom and each
array length of the encoding is below `2^64`, the range of Rust `u64`. The
node words are injective on such values
(`Export.BoundedDepth.IdentityBinding.valuePreimage_injective`).

The state hash binds the key digest of each circuit
(`Lifecycle.BoundedDepth.statePreimage`). Rust computes each key digest as the
Poseidon2 hash of a domain, the identity and the circuit index, and it
derives each circuit's relation from its package (`SPEC.md` section 4.4, not
checked against Lean). So when the verifier of one pair of packages accepts a
proof, adversary 2's result gives that verifier's keys or a Poseidon2
collision, if each key carries the relation that Lean exports. The Rust tests
compare Rust's expansion of each package with Lean's row checksum (trust
item 5).

## Adversaries

### 1. Malicious step assignment

- **Controls:** every value of the assignment of a step circuit and the
  claimed public input.
- **Wins:** the rows of the circuit's relation are zero and the public input
  encodes a digest, but the circuit's case of the step relation does not hold
  with that digest as its output hash.
- **Result:** no such win, for the light circuit
  (`Claims.Step.light_sound`) and for the refresh circuit
  (`Claims.Step.refresh_sound`). The step holds in an
  environment that agrees with the assignment on the public input and
  differs only on dead Poseidon2 locals. The package rows are the relation's
  rows (`Export.BoundedDepth.StepPackage.relationRows_packageOf`), and the rows of
  either circuit give its case of the step relation
  (`Layout.BoundedDepth.StepCircuits.soundness`). Each result takes as a premise
  that every valid step of the circuit's application has the selector value
  of that circuit. The selector that returns the circuit that runs on a valid step and the
  other circuit otherwise (`Export.BoundedDepth.TerminalBridge.chosen`) meets this
  premise for every application (trust item 4). The decoded key digests are the words that
  the assignment carries; adversary 2 binds them to the verifier's keys. The
  result is deterministic and needs no assumption.

**Packed native Boolean rows.** The adversary also controls both coordinates
of each packed pair. It wins if the accepted fresh relation holds but one of
the replaced application forcing rows fails. There is no such assignment
when each pair names actual one-coordinate slots, each group has at most 31
pairs, and the original rows are covered by retained or matched forcing rows
(`Export.Generic.PackedBoolean.freshHolds_implies_original_rows`). The result
uses the norm of that fresh witness, not the larger accumulator bound.
`Layout.ProductionRelation.Plan.freshHolds_implies_rowsPublicAndNorm` retains
this norm from fresh acceptance. There is no new probabilistic test or
cryptographic assumption. The existing extraction and binding premises still
apply to the history. The constructor packs only the sources, zero products
and products whose added columns fit in the package width, in that order, and
keeps the original row of each other candidate
(`Export.BoundedDepth.PackedBudget.choose_end`). The restorations also hold
for such a partial packing
(`Export.BoundedDepth.PackedBudget.restore_chosen_rows`). Rust must establish
the map, coverage, slot and framing conditions under trust item 5. Honest
completion uses raw application Boolean facts before native-output substitution
(`Export.BoundedDepth.PackedBooleanCompleteness.native_complete`). The terminal
source relation retains its original forcing rows because its own source
assignment has no signed-unit premise.

The adversary also controls parity pairs and zero-product sums from section 7.10.
The product check first recovers each bounded residual
(`Export.Generic.PackedProduct.packed_implies_residuals`). Boolean input facts
then force the original product (`Export.Generic.PackedProduct.half_adder`).
Each removed product uses input derivations from forcing rows or earlier source
rows, not from the row under restoration (`Export.Generic.PackedProduct.restore_rows`).
Zero-product checks require both inputs to be directly forced Boolean values
before decoding their packed residuals (`Export.Generic.PackedZero.packed_implies_residuals`).
The signed-unit sum then forces the original zero product
(`Export.Generic.PackedZero.zero_product`). The actual fresh norm supplies every
new coordinate bound. No accumulator norm or uncommitted affine output supplies
that bound. Honest values follow from `Export.Generic.PackedProduct.honest_values`
and `Export.Generic.PackedZero.honest_values`. These results add no cryptographic
or probabilistic premise. Trust item 5 still covers the Rust constructor's
framing, map, coverage, source-order and slot checks.

**Experimental native guard fusion.** The adversary chooses the fresh assignment
and wins if its fused guard rows hold but an original guarded row fails.
Complementary fusion requires a proved Boolean guard
(`Export.Generic.GuardFusion.complementary_rows`). A family whose directly
forced selectors sum to a Boolean value has mutually exclusive selectors
when its size is below the field modulus
(`Export.Generic.GuardFusion.sum_bit_exclusive`). Subset sums are Boolean,
and disjoint subsets are mutually exclusive
(`Export.Generic.GuardFusion.subset_bit`,
`Export.Generic.GuardFusion.disjoint_sums`). The family rewrite preserves its
rows under that exclusion condition
(`Export.Generic.GuardFusion.family_rows`). The constant coordinate must be 1
when source column 0 is moved into an affine constant. Nonzero field scaling
does not make an unchecked affine form Boolean.

Certificates must remain outside the removed guard rows. Retained
certificates give exact source restoration, including composition with an
acyclic affine substitution and a framed native power chain
(`Export.Generic.GuardFusion.restore_sources`,
`Export.Generic.GuardFusion.affine_restore_sources`,
`Export.Generic.GuardFusion.power_restore_sources`). A packed seed still
requires the actual fresh norm and slot correspondence stated above. A kind
label or a native output alias supplies no Boolean premise. The Rust matcher,
normalization, original-record map, and complete compiler and slot bridge
are not proved. Tests check those boundaries under trust item 5. This
experiment adds no probabilistic test or cryptographic assumption and does
not change terminal source checks.

**Common-guard binary equality packing.** The adversary chooses the fresh
assignment and wins if the packed rows hold but a replaced source row fails.
The source words use at most 32 bits. Output chunks use at most 63 Boolean
pairs. The guard can be any field value, including zero. The exact field
equivalence is `Export.Generic.BoundedWord.common_guard_iff`.
Every used source bit needs a retained forcing row or a retained packed
Boolean certificate. The packed certificate requires the actual fresh norm
and unchanged bit slots, as proved by
`Export.Generic.WordPacking.Certificate.logical_norm`. A signed unit slot
alone supplies no Boolean premise. All certificates remain outside the
selected rows, so one replacement cannot certify another replacement.
`Export.Generic.WordPacking.ordered_iff` preserves source rows and valid
assignments under explicit encoding and coverage checks. Trust item 5 covers
the Rust matcher, normalization, source indices, concrete encodings, group
coverage, identity binding, and slot correspondence. These checks are not a
new cryptographic assumption. The change adds no error probability and does
not change terminal source checks.

### 2. Malicious terminal opening

- **Controls:** the terminal statement and every value of the proof: the
  program counter, both slots with their witnesses, and the fresh claim with
  its witness.
- **Wins:** the terminal relation accepts, but no step holds under the
  verifier's keys that ends in the statement.
- **Result:** acceptance gives the step that the fresh witness decodes to
  (`Export.BoundedDepth.TerminalBridge.Exports.decode`). The step relation holds
  on it at the production cap, under the setup's keys, and it ends in the
  terminal's statement, or its next state and the terminal's state collide,
  or the step is light and its next state's words and the terminal's state
  preimage collide (`Claims.Ivc.terminal_bridge`,
  `Export.BoundedDepth.TerminalBridge.Collision`,
  `Export.BoundedDepth.TerminalBridge.WordCollision`). A light step's other slot
  is the terminal's slot, because the light circuit reads only the active
  slot's words (`Export.BoundedDepth.StepRelation.lightInput`). The terminal is
  `Lifecycle.BoundedDepth.Terminal.HoldsFor`. Two well-formed states with one state
  hash are equal or give a Poseidon2 collision on an input pair that the two
  states fix (`Claims.Encodings.state_hash_binds`,
  `Lifecycle.BoundedDepth.StateCollision`). The bridge reads each counter as
  one word, so it gives the depth words and not the depths
  (`Export.BoundedDepth.TerminalBridge.stateHash_binds_words`). The result is
  deterministic. It is about the Lean terminal relation, not about the Rust
  verifier (trust item 5). It concludes the step relation with the selector
  that returns the circuit that ran, and the terminal does not read the
  selector (`Export.BoundedDepth.TerminalBridge.holdsFor_recursive_control`), so
  its premises hold for the reference packages, whose two circuits run one
  application. The decode is a function of the proof, so the collision names
  two states that a reduction computes from the proof.

### 3. One-fold adversary

- **Controls:** the accumulator, the fresh claim and the proof of one light
  fold or one refresh fold, for a fixed key.
- **Wins:** the fold verifier accepts a fork, and extraction from the fork
  does not give the witnesses. A fork is one base output and one output for
  each Π_RLC challenge coordinate, which differs from the base only in that
  coordinate (`Spec.Folding.PiRLC.PaperForkExtraction`; SuperNeo v1.2
  Definition 12). So a light fork has 2 outputs (1 coordinate) and a
  refresh fork has 16 (15 coordinates). A verifier receives one proof; the
  extractor needs the fork.
- **Result:** for a light fold, the extracted fresh witness opens the fresh
  claim with norm below `b`, the extracted accumulator witness opens the
  input accumulator at the output bound plus `T(b − 1)`, and the claims are
  true at the prior point, or a Π_CCS bad event occurred: a mixing root of
  `α` and `γ`, or a sum-check failure
  (`Claims.Fold.light_extracts`; Cyclo, ePrint
  2026/359, Theorem 3, item 2; `Lifecycle.BoundedDepth.lightExtractionNormLaw`).
  For a refresh fold, the extracted fresh witnesses satisfy the CCS and the
  recomposed children open the input accumulator below `B`, or a Π_CCS bad
  event occurred (`Claims.Fold.refresh_extracts`).
  The fork outputs of a refresh fold need no norm bound. These results are
  deterministic.
- **Binding:** two openings of one commitment at a recovered bound give a
  kernel vector of the Ajtai key below the chain norm
  (`Spec.RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`,
  `Lifecycle.Nifs.BindingBridge.differenceBounded`). Every bound that a
  history recovers is at most `B + cap · T · (b − 1)`
  (`Spec.Folding.Schedule.recoveredBounds_le`). Trust item 2 gives the
  assumption `[MSIS]` at that norm.
- **Error terms:** the light Π_CCS test numerator is 574, against 4,747 for
  the refresh Π_CCS, and a light fork has one `ρ` coordinate, against 15 for
  a refresh fork (`Spec.Folding.Light.ErrorBudget.production_testNumerator`,
  `Spec.Folding.Light.ErrorBudget.production_forkCoordinates`,
  `Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`). Each numerator is a
  per-call count over `q²`.
- **Random-oracle model:** the model is the uniform function of `[SPONGE]`:
  a point is the list of sponge calls before a read, and its answer is the
  rate lanes of the state after them (`Lifecycle.RandomOracle.Test.Answer`).
  A `K` coin of slot `s` reads lanes `2s` and `2s + 1`
  (`Lifecycle.RandomOracle.Test.decodeK`) and a `Π_RLC` draw reads lanes 0
  to 4 (`Lifecycle.RandomOracle.Fork.draw`), so reads of one state read one
  answer, as in the sponge. For an adversary that makes at most `Q` queries,
  the probability that a fold succeeds and the extractor does not return its
  witnesses is at most a statistical term, a state-collision term and a
  kernel term (`Claims.Fold.light_knowledge_in_random_oracle_model`,
  `Claims.Fold.refresh_knowledge_in_random_oracle_model`). The bad set of a
  coin reads the other coins of its state from the answer, and the coin's
  lanes are uniform for each value of the other lanes
  (`Lifecycle.RandomOracle.Test.bad_mass_le`), so the test step costs at most
  `(Q + 1)` times the Π_CCS test error
  (`Lifecycle.RandomOracle.Test.family_error_le`). The statistical term is at
  most `(Q + 1) · 575/2^128` for light and `(Q + 1) · 4748/2^128` for refresh
  (`Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.statisticalError_le`,
  `Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.statisticalError_le`).
  The fold gate of Rust `Params::for_ccs_shape` charges one error for both
  folds and computes its bits in integers. The statistical term of either
  fold is at most `(Q + 1)/2^bits`, and `bits` is 115 at the step relations
  (`Claims.Fold.refresh_error_bits`,
  `Claims.Fold.light_error_bits`,
  `Lifecycle.BoundedDepth.RandomOracle.FoldGate.production_bits`). The gate
  certifies only the term at `Q = 0`.
  The kernel term is the probability that the binding reduction returns a
  short kernel vector of the Ajtai key at the chain norm. A black-box
  rewinding machine outputs that vector with at most `budget` adversary runs,
  so the kernel term is at most its success plus `(3Q + 61)/budget` for light
  and `(31Q + 523)/budget` for refresh
  (`Lifecycle.BoundedDepth.RandomOracle.LightSolver.kernelChance_le_solver`,
  `Lifecycle.BoundedDepth.RandomOracle.RefreshSolver.kernelChance_le_solver`). The
  state-collision terms are the probability of a Poseidon2 collision on two
  named prior states, and a collision finder outputs the colliding input
  pair, whose two inputs have one length
  (`Lifecycle.BoundedDepth.RandomOracle.HashSolver.hashCollisions_le_finder`).
  Under `[P2-PERM]` and `[SPONGE]`, a finder with `N` calls gives two distinct
  inputs with one digest with probability at most
  `C(N + 2, 2)/q⁴ + spongeError N` (`Spec.Sponge.collision_le`), because the
  `10*` padding of the sponge is injective
  (`Claims.Encodings.hash_padding_injective`).
  Averaged over the SHAKE128 key setup at key width `2^26`, the kernel term is
  an MSIS advantage for a uniform matrix at the chain norm plus a programming
  error below `2^−1129`, so each fold's bound uses only the two algorithms
  (`Claims.Fold.light_production_knowledge_error`,
  `Claims.Fold.refresh_production_knowledge_error`).
  The finder is one run with one output, and it reruns the adversary on
  oracles of its own tape. So `[SPONGE]` gives its advantage the number above
  for `N` calls, a parameter of the adversary class
  (`Claims.Fold.light_knowledge_error_numeric`,
  `Claims.Fold.refresh_knowledge_error_numeric`). Lean does not bound the
  MSIS advantage numerically (trust item 2), and it states the machines' work
  in adversary runs, not in time.
- **The sponge:** the model's address of a challenge is the list of sponge
  calls before its read (`Lifecycle.RandomOracle.Schedule.challengeCalls`).
  Distinct challenges have distinct addresses
  (`Lifecycle.RandomOracle.Schedule.challengeAddress_injective`), equal calls
  fix the absorbed data (`Lifecycle.RandomOracle.Schedule.proverCalls_identify`),
  and the light and refresh transcripts share no call prefix
  (`Lifecycle.RandomOracle.Schedule.challengeCalls_separated`). The fold calls
  start with their domain (`Export.DomainSeparation.fold_owns`), and the fold
  domains are separated from the other uses
  (`Claims.Encodings.domains_separated`). The model reads each address as the
  sponge does at the rate lanes
  (`Lifecycle.BoundedDepth.RandomOracle.Fidelity.spongeOracle_deployed`). The
  state of a `ρ` read is the state of no other challenge, because its calls are
  longer than those of a `K` coin and two `ρ` reads have different numbers of
  calls (`Lifecycle.RandomOracle.Schedule.rhoCalls_ne`). So a new answer at a
  `ρ` point keeps the Π_CCS coins
  (`Lifecycle.RandomOracle.Fork.update_rho_ne`). These are the conditions of
  `[DSFS]` (`Export.BoundedDepth.FoldSponge.conditions`). The event is the
  failure event: the fold succeeds and the extractor, with the retries of the
  retry law, does not extract
  (`Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.failureChance`). The
  event has retries, so `[DSFS]` moves it, not `[SPONGE]`, which moves one run
  only. For a prover with `N` permutation calls and a verifier outside a
  circuit, its chance is at most the knowledge error of a model adversary with
  `N` queries plus `duplexError N` (`Claims.Fold.light_knowledge`,
  `Claims.Fold.refresh_knowledge`).
- **Not covered:** the step circuits recompute the challenges with the
  Poseidon2 permutation, and no ideal-permutation model covers a circuit
  that evaluates the permutation. So this result bounds no deployed check of
  the recursion (trust item 3).

### 4. IVC adversary

- **Controls:** a random tape and, from it, a terminal statement and every
  value of a proof, for a fixed application, setup and pair of exported step
  circuits. The adversary must be admitted (trust item 3).
- **Wins:** the terminal relation accepts, and no `iteration` steps of the
  application, each with a witness of the application's word count, lead
  from `z0` to `zi` (a false acceptance).
- **Result:** under Assumption 1 and the closure premise,
  `Pr[false acceptance] ≤ walkBound = Σ_j (finderChance(stage j) + error(stage j))`
  for `j` below a bound on the adversary's iteration
  (`Claims.Ivc.false_acceptance_bound`,
  `Export.BoundedDepth.IvcSecurity.walkBound`). The reverse extractor
  walks back from the terminal one step per stage
  (`Export.BoundedDepth.IvcWalk`). At each visit, the fresh witness decodes to the
  last step (`Export.BoundedDepth.TerminalBridge.Exports.decode`). When that step
  is recursive, the stage's attempt is its fold of the slot that its prior
  program counter selects, linked to the step's prior state, with the slot's
  witness at the slot's bound. An answer that extracts the fold gives the
  prior visit, whose statement and slots are the step's input. Each slot keeps
  its own bound: a light fold adds `T(b − 1)` and a refresh fold returns it to
  `B`, so each attempt stays within `B + cap · T · (b − 1)`
  (`Export.BoundedDepth.IvcWalk.DepthWithin`). At a visit without a collision, an
  answer that extracts gives a valid prior visit one iteration back, or the
  last step is the base step (`Export.BoundedDepth.IvcWalk.walk_step`). So a false
  acceptance has a collision or a failed extraction at one of its first
  `iteration` visits (`Export.BoundedDepth.IvcWalk.covered`).
- **Collision term:** the term of stage `j` is the success chance of an
  explicit Poseidon2 collision finder
  (`Export.BoundedDepth.IvcSecurity.finderChance`). The finder runs stage
  `j` on its tape, decodes the visit, and outputs two different inputs of
  one length with one `Poseidon2.hash`
  (`Export.BoundedDepth.IvcSecurity.finder`,
  `Lifecycle.BoundedDepth.PoseidonCollision`). At a visit it outputs the
  input pair of the visit's collision
  (`Export.BoundedDepth.IvcWalk.visitCollision`): the next state of the
  decoded step and the visit's terminal state, or a light step's next state
  words and the terminal's 38-word state preimage
  (`Export.BoundedDepth.TerminalBridge.Collision`). Each collision at the
  visit gives an output
  (`Export.BoundedDepth.IvcWalk.visitCollision_isSome`). Under `[P2-PERM]`
  and `[SPONGE]`, the term of stage `j` is at most
  `C(N_j + 2, 2)/q⁴ + spongeError N_j` for a finder with `N_j` permutation
  calls, so `walkBound ≤ Σ_j (C(N_j + 2, 2)/q⁴ + spongeError N_j + error(stage j))`
  (`Claims.Ivc.walk_bound_numeric`). A stage is a Lean function, and Lean
  has no count of its calls. So `N_j` is the parameter `calls j` of the
  premise `finders`, and the premise holds only for an adversary class
  whose finder of stage `j` makes at most `N_j` calls (trust item 2).
- **Per slot:** the earlier one-slot result,
  `Claims.Ivc.slot_history_failure`,
  takes the history as an adversary output and has no collision term. Its
  reverse extractor opens each running claim of the slot at its recovered
  bound, at most `B + cap·T·(b − 1)` (`Claims.Ivc.accepted_bounds`).
- **Validity:** a false acceptance has no list of witnesses that are valid
  for the application and that its step maps from `z0` to `zi`
  (`Export.BoundedDepth.IvcWalk.Reaches`). The bridge's selector returns the
  circuit that ran only on a valid step (`Export.BoundedDepth.TerminalBridge.chosen`),
  so each decoded step is valid (`Export.BoundedDepth.TerminalBridge.valid_of_chosen`).
- **The verifier:** the terminal verifier of `SPEC.md` Algorithm 6 also
  holds each slot to the bound of its depth
  (`Export.BoundedDepth.RunCompleteness.Verifies`). The same sum bounds the
  probability that this verifier accepts and no such steps lead from `z0` to
  `zi` (`Claims.Ivc.verifier_sound`). `SPEC.md`
  section 9 gives the theorem of each verifier entry point, or the reason that it is open.
- **Knowledge form:** HyperNova Definition 11 (ii) states knowledge
  soundness: its event is on the witnesses that the extractor returns. The
  same sum bounds the probability of an extraction failure: the terminal
  relation accepts, and the witnesses of the walk's trail are not `iteration`
  valid steps from `z0` to `zi`
  (`Export.BoundedDepth.IvcSecurity.ExtractionFailure`,
  `Claims.Ivc.extraction_bound`). Its law is the adversary's
  tape and the coins of each stage's extractor. A false acceptance is an
  extraction failure, so the result above follows from this one.
- **Gaps:** Lean derives Assumption 1 with an explicit `error` only in a
  random-oracle model of the folds
  (`Claims.Fold.assumption1_in_random_oracle_model`). The bound
  here still takes Assumption 1 for the concrete hash as a premise, and it
  gives no concrete security for a long history (trust item 3).
- **Final proof:** a prover of the final verifier is an adversary of this
  kind, and adversary 6 joins the two bounds.

### 5. Memory adversary

- **Controls:** what adversary 4 controls, for a memory application at a
  fixed geometry and program, and from the same tape a claimed memory
  statement and the final carry. The proof carries the carry, and the
  verifier takes the statement integers from it (`SPEC.md` Algorithm 8). The
  adversary must be admitted (trust item 3).
- **Wins:** the memory verifier accepts (`SPEC.md` Algorithm 8), and the
  history that an extractor returns does not attest the claim: it is not an
  execution from the claim's initial words and the program's memory to its
  final words in which each memory access reads the last value written.
- **Result:** under Assumption 1, the closure premise and the memory round
  (trust item 6), on the joint law of the claim that the adversary's tape
  gives and the reverse extractor's sample, the mass of wins is at most the
  collision and stage terms of adversary 4
  (`Export.BoundedDepth.IvcSecurity.walkBound`), the state-link and chain
  collision masses of the extracted runs, and the memory-round error of the
  last stage, over the walk's `S_max · N` stages (`Claims.Nebula.memory_sound`). The extracted history
  is the steps that adversary 4's walk decodes (`Export.BoundedDepth.IvcWalk.trail`),
  so every event reads data of the sample. The walk meets the extraction
  premise of `Lifecycle.Nebula.EndToEnd.Stage1Extraction`
  (`Export.BoundedDepth.NebulaSecurity.stage1Extraction`), and
  `Lifecycle.Nebula.EndToEnd.bad_subset` reduces a win to a failure of that
  extraction, a state-link collision, a chain collision, or an accepted run
  with an unbalanced segment that passes the product test. The reverse
  extractor runs `S_max · N` stages, and the checks of the memory verifier
  bound the iteration by `S_max · N`
  (`Lifecycle.Nebula.EndToEnd.VerifierChecks.iteration_le`), so every
  sample that the memory verifier accepts counts
  (`Export.BoundedDepth.NebulaSecurity.checked_le`). For a compressed memory proof, adversary 6 joins this
  bound with the bound of the final proof
  (`Claims.AkitaKey20.key20_memory_sound`, `Claims.AkitaKey30.key30_memory_sound`).
- **Deterministic part:** an accepted run gives an execution, a Poseidon2
  collision among the run's own chain inputs and the `D_init` chain, or a
  segment that passes the product test with unbalanced multisets
  (`Claims.Nebula.model_sound`). With the state links, an accepted chain of valid
  steps gives such a run or a state-digest collision
  (`Claims.Nebula.accepts_or_collision`).
- **Challenge game:** in the interactive game where `η` comes after the
  roots and `ts`, segment slot `k` is bad with frequency at most
  `2 · m_mem/q²` and a retry term, and each retry disagreement is a Poseidon2
  transcript collision (`Lifecycle.Nebula.badAt_bound`). Here
  `m_mem = R + M + N · B_ops`, and setup rule 6 keeps `2 · m_mem/q²` at most
  `2^−109.91` (`Spec.Nebula.Geometry.secure_iff`). A proof has at most
  `S_max` segments, so the runtime gate charges `S_max · 2 · m_mem/q²` with
  exact integers (`SPEC.md` section 7.1). Lean defines the bits of that
  gate and proves that the charge is at most `2^−bits`
  (`Spec.Nebula.Geometry.segmentBits`, `Spec.Nebula.Geometry.segmentError_le`).
  Each fixed WASM size certifies exactly 109 bits, and one more segment
  certifies fewer (`Spec.Nebula.WasmSize.segmentBits_eq`,
  `Spec.Nebula.WasmSize.segmentBits_succ_lt`). Each size also has
  `S_max · N ≤ cap + 1`, so no proof that the memory verifier accepts for a
  size reaches a refresh fold, and its final proof has the light mode of
  adversary 6 (`Spec.Nebula.WasmSize.steps_le_light`). Lean emits these
  numbers, and Rust tests compare them (trust item 5).
- **Random-oracle model of `η`:** when `η` comes from a uniform oracle on its
  input (the plan digest, `ts` and the three roots) in place of Poseidon2,
  then for every `η` sampler with `Q_η` queries, the event of the memory
  round has probability at most `(Q_η + 1) · 2 · m_mem/q²` plus the
  sampler's retry term
  (`Lifecycle.Nebula.EtaRound.EtaSampler.unbalancedChance_le`). Each pair
  that the retry term counts is a Poseidon2 transcript collision between the
  chain inputs of two runs with the same `η` input
  (`Lifecycle.Nebula.EtaRound.retry_collision`). For a secure geometry, the
  first term is at most `(Q_η + 1) · 2^−109.91`
  (`Claims.Nebula.memory_round_in_random_oracle_model`). The model takes the
  adversary and the extractor as one sampler, so `Q_η` counts the queries of
  every rerun. At the Poseidon2 oracle the event is the event of the memory
  round (`Lifecycle.Nebula.EtaRound.unbalancedIn_deployed`), but no result
  bounds it there. The model's point is a function of the `η` input that is
  injective on canonical timestamps
  (`Lifecycle.Nebula.EtaOracle.pointOf_injective`), and the absorbed blocks
  are injective (`Lifecycle.Nebula.etaInputBlocks_injective`,
  `Lifecycle.Nebula.blockChunks_injective`). These are conditions of
  `[SPONGE]`, and the `η` domain is separated from the other uses
  (`Claims.Encodings.domains_separated`). With these conditions proved
  (`Export.BoundedDepth.EtaSponge.conditions`), `[P2-PERM]` and `[SPONGE]`
  move the probability of the event to the ideal-permutation model plus
  `spongeError N`. So the bound there is the first term, plus the retry term
  of the sampler that the transfer gives, plus `spongeError N`
  (`Claims.Nebula.eta_balanced`). That
  bound is for a verifier that computes `η` outside a circuit. The step
  circuit computes `η` with the permutation (trust item 6).

### 6. Final-proof adversary

- **Controls:** the statement and each byte of a final proof (`SPEC.md`
  section 4.5), against the verifier's own circuits, security minimum and
  compression key. It can evaluate the Poseidon2 sponge and the Keccak
  functions of Akita. It does not choose the compression key or the mode.
  The verifier takes the pinned key, or loads bytes that must equal it
  (`SPEC.md` Algorithm 11). The verifier's circuit fixes the mode
  (`Export.BoundedDepth.FinalVerifier.Mode`): the argument has both slot
  claims, or in the light mode only the claim of slot 0. In the light mode
  the verifier builds slot 1 as the default slot at depth 0 and requires
  `pc = 1` (`Export.BoundedDepth.FinalVerifier.ModeChecks`).
- **Wins:** the final verifier accepts (`SPEC.md` Algorithm 10), but no
  proof with witnesses exists that the verifier of `SPEC.md` Algorithm 6
  accepts for the statement
  (`Export.BoundedDepth.RunCompleteness.Verifies`).
- **Result:** for the argument only. Lean models the verifier of the argument as a public-coin game
  (`Spec.Compression.Verifier.game`). In the state-restoration experiment, a prover with `Q` oracle queries makes the
  verifier accept claims whose committed `W` does not satisfy `Spec.Compression.Verifier.Relation` with probability at
  most `Q · error` (`Claims.Compression.argument_sound`). `Relation` gives, for each claim, the 18 Ajtai rows of its
  commitment, `Eval_K`, each `Eval_A` relative to the abstract matrix weights, its public input and, for a claim with a
  table, each cell in `[−bound, bound]` (`Spec.Compression.Verifier.game_avoids`). At the production shapes, `error <
  2^−119` (`Claims.Compression.production_error`). The premises of the theorem, and how Rust meets each one, are in
  trust item 7. Lean has no definition of the compression key. Its model of the layout
  (`Spec.Compression.Rust.layoutNew`) meets the layout premises of the theorem
  (`Spec.Compression.Rust.layoutNew_production`), and Rust tests compare it with `Layout::new`. For each slot claim,
  `Relation` gives the claim under `Spec.CE.Holds` with the norm of its depth
  (`Lifecycle.BoundedDepth.CompressionSlot.slot_holds`), under the premise `Spec.Compression.Claim.Represents` of trust
  item 7. Lean models the Π_CCS run with one source as a game of the same kind (`Spec.Compression.Fresh.game`). On a
  path that avoids its bad sets, of density at most `38/q²`, an accepted run whose output claims hold gives the fresh
  instance under `Spec.CCS.Holds` (`Lifecycle.BoundedDepth.CompressionFresh.game_avoids`). Lean joins the checks, Π_CCS
  and the argument in one game of the final verifier (`Export.BoundedDepth.FinalVerifier.Setup.game`), for each mode. In
  the light mode, the zero witness opens the default slot under the relation of slot 1
  (`Export.BoundedDepth.StepExtend.defaultSlot_holds`). With `Q` queries, it accepts a statement for which no proof with
  witnesses satisfies `Verifies` with probability at most `Q · (2^−119 + 38/q²)` (`Claims.Final.verifier_sound`).
  Trust item 7 gives the bounds with Akita openings (`Claims.AkitaKey20.key20_verifier_sound`,
  `Claims.AkitaKey30.key30_verifier_sound`).
  Lean bounds one interactive run of the argument by the exact Rust error sum
  (`Spec.Compression.Rust.params_run_soundness`). Only tests join the model to the Rust layout and error sum, and no
  result states a full honest run. Rust implements the final verifier, and the tests check it on fixed inputs (`SPEC.md`
  lifecycle rules 2 and 3, trust item 5). `decisions/akita-compression.md` gives the reason for each step.
- **Aim:** under the premises of trust item 7, an accepted final proof gives three witnesses: one for each claim of the
  argument, and in the light mode the zero witness of slot 1. The one Akita proof gives the polynomials `W` and `V`, and
  the value of the key polynomial at one point. The key polynomial is the Ajtai key, so the key sum-check makes the
  proof's key part of the public weights equal to the part that the Ajtai key gives. The verifier reads no element of
  the Ajtai key. The first sum-check and the last check of `SPEC.md` Algorithm 10 give, for each claim of the argument,
  a witness in `W` that opens the claim's commitment, has its public input and gives its evaluations (`Spec.CE.Holds`
  without the norm). The lookup gives each slot witness the bound of its depth. The Π_CCS run gives the fresh witness
  the rows of its relation and coefficients in `{−1, 0, 1}` (`Spec.CCS.Holds`). The verifier computes the state hash in
  the fresh public input from the proof's slot claims, so no digest that the proof carries is authority. These are the
  checks of `SPEC.md` Algorithm 6, with one difference. The argument does not prove that the fresh witness is zero from
  the relation's width to the end of its last ring column. `Lifecycle.BoundedDepth.BlockSupport.Supported` does not
  require these zeros. The honest prover's fresh witness has them: it is zero on every cell from the width of its
  circuit's package (`Lifecycle.BoundedDepth.BlockSupport.ZeroTail`,
  `Export.BoundedDepth.RunCompleteness.FreshZeroTail`, `Claims.Ivc.step_extends`, `Claims.Ivc.run_complete`).
- **With the other adversaries:** the verifier of Algorithm 6 accepts only what the terminal relation accepts
  (`Export.BoundedDepth.RunCompleteness.holds_of_verifies`). So the results of adversaries 2, 4 and 5 apply to a
  statement that the final verifier accepts. Lean joins the bound above with the bound of adversary 4. A prover of the
  experiment with `Q` queries is an IVC adversary: its tape is the result of the experiment under the law of its draws,
  and its output is the accepted statement with the proof from `W` (`Export.BoundedDepth.FinalHistory.adversary`,
  `Export.BoundedDepth.FinalVerifier.Setup.proofOf`). The event is a false history: the final verifier accepts `(i, z0,
  zi)`, and no `i` steps of the application, each valid at its state and with a witness of the application's word count,
  lead from `z0` to `zi` (`Export.BoundedDepth.FinalHistory.FalseHistory`). A false history is a result in which
  `Verifies` rejects that proof, or a false acceptance of this IVC adversary. So `Pr[false history] ≤ Q · (2^−119 +
  38/q²) + walkBound`, where `walkBound = Σ_j (finderChance(stage j) + error(stage j))` is the bound of adversary 4 for
  the stages of this IVC adversary (`Claims.Final.history_sound`, `Export.BoundedDepth.IvcSecurity.walkBound`). The same
  bound holds for the witnesses that the reverse extractor returns: the final verifier accepts, and those witnesses are
  not a valid history of the accepted statement (`Claims.Final.history_knowledge`). Trust item 7 gives these results
  with Akita openings. At the packages that the emitter writes for the reference application, Lean proves each premise
  of class `lean` of the direct verifier result (`Claims.Shipped.verify_sound`). At these packages and the 524288 key
  blocks of `key30.aks`, `Claims.Shipped.verify_final_sound` is `Claims.AkitaKey30.key30_history_sound` with the digest
  length of the keys and the block count proved; it keeps the hypotheses `shared`, `drawn` and `coords` on the Akita
  openings. The Lean Nebula packages are test packages that no public verifier loads, so no such result exists
  for the memory verifiers (row `app.*`). The premises are those of trust item 7, `[HN-A1]` and the
  closure premise of trust item 3, an admitted IVC adversary of the prover, the digest length 4 of each circuit key, and
  a bound `depth` on the iteration of each statement that the experiment accepts. The sum has the limits of adversary 4,
  so it gives no concrete security for a long history. Lean also joins the bound above with the bound of adversary 5,
  for the memory verifier of a final proof (Rust `nebula::Verifier::verify_final`, `SPEC.md` Algorithms 8 and 10). The
  prover names a memory statement with each result of the experiment, with any law for each result. With that statement
  on its tape it is an IVC adversary with the output above (`Export.BoundedDepth.NebulaFinal.adversary`). The event is a
  false memory statement: the final verifier accepts `(i, z0, zi)`, the checks of `SPEC.md` Algorithm 8 hold for `(i,
  z0, zi)` and the memory statement, and the history that the walk of adversary 4 decodes does not attest the memory
  statement (`Export.BoundedDepth.NebulaFinal.FalseMemory`). The law is that of the walk's last stage: the result, the
  memory statement and the coins of each stage's extractor. A false memory statement is a tape on which `Verifies`
  rejects the proof from `W`, or a win of adversary 5 for this IVC adversary. So its probability is at most `Q · (2^−119
  + 38/q²)` plus the terms of adversary 5 (`Claims.Final.memory_sound`), and trust item 7 gives the bound with Akita
  openings. The premises are those of the two joined results, but
  without the bound `depth`: the checks of `SPEC.md` Algorithm 8 step 4 accept at most `S_max · N` steps
  (`Lifecycle.Nebula.EndToEnd.VerifierChecks.iteration_le`), so the walk has `S_max · N` stages, and its answers cover
  the iteration of each statement in the event. The memory round (trust item 6) stays a premise. No Lean result joins
  the bound of adversary 2 with the bound of this adversary. No result shows that an extractor of the three witnesses is
  efficient.
- **Error:** Lean bounds the argument by `Q · error`, with `error` below
  `2^−119` at the production shapes (`SPEC.md` section 3). Rust bounds a
  sum of six terms for one interactive run of the argument, about
  `2^−119.02` for the reference application at 4 lookup challenges, by
  `2^−119` in integers, with no factor for the queries. Where the two differ, the Lean bound is correct.
  Neither is a bound on a win. Neither includes the knowledge error of the
  Akita opening or the error of the Π_CCS run.
- **Gaps:** the final proof is not zero-knowledge. No test uses a slot
  depth near the cap, so no test uses the largest lookup table. The tests
  that compress a step of the reference application are ignored. CI runs
  one of them alone, `hash_chain_compresses_and_verifies_without_the_ajtai_key`,
  and the others run by hand. So CI checks the edits of the argument only
  on small claims, with a key polynomial of `2^9` blocks. The verifier takes each Akita schedule row from its key,
  its own layout and the embedded schedule files, whose SHA-256 a test
  pins (`each_schedule_file_has_its_pinned_digest_and_its_rows`).

## What you trust

1. **The Ajtai setup [KECCAK] (premise P1).** Each key element reads one stream of 84 SHAKE128 output words of 64 bits.
   A word below `q` is kept, a word of `q` or more is rejected, and coefficient `h` is kept word `h`, or 0 when fewer
   than `h + 1` words are kept (the fallback) (`Spec.AjtaiSetupV1.Programming.verifierKey_eq`). The experiment is one
   uniform stream. Each ring element is the element of the same number of streams that take no fallback
   (`Spec.AjtaiSetupV1.Rejection.card_element`), so the element is exactly uniform outside the fallback event. The event
   is that 21 or more of the 84 words are rejected, and its probability for one element is at most `fallbackBound =
   C(84, 21) · 59^21/2^1344`, below `2^−1155` (`Spec.AjtaiSetupV1.Rejection.card_fallback_le`,
   `Claims.Foundations.setup_fallback_bound`). With uniform streams, an event has at most its probability under a
   uniform matrix, plus `2 · fallbackBound` for each key element (`Claims.Foundations.setup_element_bias`,
   `Spec.AjtaiSetupV1.Programming.expect_le_programmed`). For the `18 · 2^20` elements of the bounded-depth key the sum
   is below `2^−1129` (`Claims.Foundations.setup_error`). So MSIS for a uniform matrix applies
   (`decisions/fprime-ajtai-shake128-setup.md`, `decisions/fprime-ajtai-rejection-sampling.md`,
   `decisions/fprime-ajtai-shake128-security-argument.md`). The seed is fixed and public, so an adversary that is
   precomputed for that seed is outside this premise. Lean does not join this step to the results above.
2. **The binding and collision terms [MSIS] [P2-PERM] [SPONGE].**
   - *MSIS at the chain norm.* Binding needs MSIS for a uniform matrix of `κ = 18` rows of elements of `F_q[X]/(X^64 +
     1)` (degree 64) at the chain norm `msisChainBound = 8 · T · (B + cap · T · (b − 1)) = 25,165,824`, about `2^24.58`
     (`Claims.Foundations.chain_norm`). It is the SuperNeo v1.2 Section 8 norm `8TB` with `B` replaced by the largest
     recovered bound (`Spec.Folding.Schedule.msisChainBound_zero`). The terminal verifier checks each slot at the bound
     of its depth. By Cyclo Theorem 3, this bound is at most `accumulatorBound cap = 10,112`
     (`Spec.GlobalParams.accumulatorBound`, `SPEC.md` Algorithm 6), and Lean proves that an honest run stays within it
     (`Export.BoundedDepth.StepExtend.Honest`). A refresh fold recovers its input only below `B`, so the terminal
     relation of the security results keeps `B` (`Export.BoundedDepth.RunCompleteness.holds_of_verifies`). The recovered
     bound, and so the chain norm, comes from `B`, not from the bound of the depth.
   - *The κ estimate.* A lattice-estimator run outside Lean, at rank `κ = 18`, a width of `2^28` coefficients and the
     chain norm `8T(2^14 + 64T) = 25,165,824`, gives 127.20 bits in the ADPS16 quantum model (BKZ block size 480),
     140.16 bits in the ADPS16 classical model and 166.84 bits in the default model. The norm is the largest extraction
     norm. There is no allowance for more targets, because the main key is one SHAKE128 matrix and so it has one target.
     The owner approved this method (`decisions/field-2p64-59-ring-x64.md`). The sister project ProtogaLattice computed
     these values with lattice-estimator commit `3e48ef4` and Sage 10.9. A Nightstream run of 2026-11-20 with the same
     commit gives the three numbers again: the row `main key` of `scripts/akita_sis_estimates.csv`, which
     `scripts/estimate_akita_sis.sage` writes (trust item 7). `scripts/estimate_nebula_sis.sage` has the calls of the
     method for the prior profile, another modulus and the ring degree 54, so it does not give these numbers. The
     owner selected `κ = 18` on
     2026-11-13. The models are heuristic generic lattice attacks. They are not a proof for structured Module-SIS or
     for the fixed public seed. The method is that of the SuperNeo parameter estimate: the Euclidean route, `norm = 2` at
     the length `β · √(m·d)`. The direct ℓ∞ model of the Akita table gives 154.23 bits (block size 582) for this
     instance. The owner's target is 128 bits against a classical attacker (2026-11-22), and the owner selected this
     method for the main key and for each Akita instance (2026-11-20). The main key has 140.16 bits against that
     target (the `MSIS` entry of the section Assumptions; trust item 7, the Module-SIS instances).
   - *Poseidon2 collisions.* The state hash, the slot digests, the package identity, the Nebula record chains and the
     state digests of a memory application bind their inputs only up to a Poseidon2 collision. The results name each
     collision as an event on two inputs that a reduction computes. Under `[P2-PERM]` and `[SPONGE]`, an algorithm with
     `N` permutation calls finds two distinct padded inputs with one digest with probability at most `C(N + 2, 2)/q⁴ +
     spongeError N` (`Spec.Sponge.collision_le`). Two distinct inputs of the hash have distinct padded inputs
     (`Spec.Sponge.collision_blocks`), and two distinct values of a row of section Hashed encodings have distinct inputs
     only by that row's injectivity theorem. The deployed permutation is fixed, so `[P2-PERM]` is a heuristic for it;
     algorithms that use its structure are outside the model.
   - *The IVC collision term.* In the bounds of adversaries 4, 5 and 6, the collision term of each reverse stage is the
     success chance of an explicit finder that outputs such a pair of inputs
     (`Export.BoundedDepth.IvcSecurity.finderChance`). Lean bounds it with the bound above
     (`Claims.Ivc.walk_bound_numeric`). `N` counts the permutation calls of the finder: the calls of the stage, and so
     the calls of the earlier stages that it runs again. Lean cannot count them for a Lean function, so `N` is a
     parameter of the adversary class.
   - *The fold collision term.* The collision finder of each fold is such an algorithm too, and Lean bounds its
     advantage with the same bound (`Claims.Fold.light_knowledge_error_numeric`,
     `Claims.Fold.refresh_knowledge_error_numeric`). `N` counts the calls of the adversary in each of the finder's
     reruns.
   - Lean does not bound the MSIS term numerically.
3. **The history: HyperNova errata Assumption 1 for the two fold kinds [HN-A1].**
   `Spec.HyperNova.ReverseStages.Assumption1` states Assumption 1 in the form of HyperNova Definition 7 knowledge
   soundness. Adversary 4 uses it with the success event `Export.BoundedDepth.IvcWalk.WalkSucceeds` and the extraction
   event `Export.BoundedDepth.IvcWalk.WalkExtracted`; the one-slot result uses
   `Spec.HyperNova.Construction2.BoundedDepthSecurity.Assumption1` with the same events. For every admitted fold
   adversary there is an efficient extractor that reads the adversary's tape and its own coins, so it may rerun the
   adversary. The probability that the adversary succeeds and the extractor fails is at most `error` of that adversary.
   The adversary outputs an attempt: a fold, the running claim that it starts from, and a witness of its output at a
   bound. One premise covers both fold kinds. The attempt names its circuit, and a success needs the bounded-depth
   verifier of that circuit to accept, so a light fold at the cap is not a success. A success also needs an output bound
   of at most `B + cap · T · (b − 1)`, so the premise is needed only up to the chain norm of trust item 2
   (`Spec.HyperNova.Construction2.BoundedDepthReverse.Succeeds`). Extraction must open the fold's input at the output
   bound plus `T(b − 1)` after a light fold, and below `B` after a refresh fold, and it must open the fresh claim
   (`Spec.HyperNova.Construction2.BoundedDepthReverse.Extracted`, `Spec.Folding.Schedule.extractedBefore`). These are
   the differences from Definition 7:
   - The success event has a prior-state link. The light transcript does not absorb the running claim
     (`Lifecycle.BoundedDepth.lightAbsorbPublicInput`), and the refresh transcript absorbs the children's commitments
     and evaluations but not the accumulator's point or public input (`Lifecycle.BoundedDepth.refreshWords`). Both bind
     the accumulator only through the prior digest in the fresh public input. So the adversary also outputs a prior
     state, and a success needs `PriorLink` of that state, the running claim and the fold
     (`Spec.HyperNova.Construction2.BoundedDepthReverse.RunsBack`). The prior state is an output, as in PR #155: a link
     to some prior state holds for almost every running claim, because many states share one digest. In an accepted
     execution, each fold of a slot links to the prior hash preimage of its step: the fresh public input encodes the
     hash of that preimage, and its program counter selects the slot that holds the fold's running claim
     (`Spec.HyperNova.Construction2.BoundedDepthSecurity.StepLink`,
     `Spec.HyperNova.Construction2.BoundedDepthSecurity.slot_runsBack`). The bounded-depth scheme's link also requires a
     well-formed prior state (`Lifecycle.BoundedDepth.PriorLink`). Two linked folds with one fresh instance then start
     from one running claim, or their two prior states give a state-hash collision
     (`Lifecycle.BoundedDepth.priorLink_binds`). So the transcript fixes the running claim before the challenges, up to
     a collision on a pair of states that the adversary outputs.
   - The relations have a key. A claim holds under the key of its slot, and every fold of an accepted history uses the
     slot's key.
   - The transcript is not the one of HyperNova Construction 5, the Fiat–Shamir transform of the revision of 2026-02-20.
     The published construction hashes the public parameters and the structure into the first challenge and derives each
     later challenge from the prover's message. The errata's construction absorbs `sid` (the structure and the public
     parameters) and the instances before the first challenge. The fold transcripts absorb a domain block, the fresh
     statement, the children of a refresh fold and the prover messages (`SPEC.md` section 5.2). The structure, the keys
     and the accumulator enter only through the prior digest. So `Assumption1` is a premise about the Nightstream
     transcripts, not the paper's premise about Construction 5.
   - This joint form implies Definition 7's difference form `Pr[success] − Pr[extraction] ≤ error`. The converse needs
     an adversary that stops when its own success check fails.
   - The public parameters are a fixed commitment setup, not sampled by the generator. With a fixed hash and a fixed
     key, an efficient algorithm that contains a collision exists. So the premise can hold only for algorithms that a
     person can write without such knowledge.
   - The structure is a fixed application. Definition 7 lets the adversary choose the structure after the public
     parameters, and Definition 11 lets the IVC prover choose `F` from the public parameters and its coins. A choice
     that depends on the adversary's coins is not covered.
   - `error` replaces `negl(λ)`.

   `Spec.HyperNova.ReverseStages.Closed` is the composition premise. A second class, `StageAdmitted`, holds whole
   stages: the reverse extractor so far, as one algorithm. An admitted stage gives an admitted fold adversary, and one
   reverse step of an admitted stage with an efficient extractor is again an admitted stage. The IVC adversary is the
   start stage, which must be admitted. `Admitted`, `StageAdmitted` and `Efficient` are abstract; their intended meaning
   is expected polynomial time. Lean has no running-time model, so it states no false claim about running time, and the
   closure property is a premise. Lean chooses the stages with `Classical.choose`, so a numerical bound on `error` must
   hold for the whole class. Assumption 1 is the paper's plain-model premise, not a theorem. The step circuits recompute
   the challenges of the previous fold with Poseidon2, so a recursive argument uses the concrete hash, which no
   random-oracle model covers. In a random-oracle model of the folds, Assumption 1 holds with an explicit error:
   adversary 3's knowledge errors for each circuit key plus a truncation loss `2 · ((Q + 1) + 15 · (Q + 15)) / budget`
   for an extractor with its own oracle coins and `budget` reruns (`Claims.Fold.assumption1_in_random_oracle_model`).
   The claim names that extractor (`Lifecycle.BoundedDepth.RandomOracle.Assumption1.RomEfficient`), so no other function
   of the same type meets it. At an oracle that answers like the Poseidon2 sponge, the model's success and extraction
   events are the walk's (`Export.BoundedDepth.RomWalk.succeeds_iff_walkSucceeds`,
   `Export.BoundedDepth.RomWalk.extracted_iff_walkExtracted`), and an honest fold meets the success event
   (`Lifecycle.BoundedDepth.RandomOracle.Honest.light_honest`,
   `Lifecycle.BoundedDepth.RandomOracle.Honest.refresh_honest`). One fixed oracle carries no probability, so this does
   not prove the premise for the concrete hash. The two adversaries also have different types. A random-oracle attempt
   is a pair of an oracle and a walk attempt (`Lifecycle.BoundedDepth.RandomOracle.RomModel.RomAttempt`), and the IVC
   adversary has no oracle (`Spec.HyperNova.ReverseStages.IvcAdversary`), because the step circuits evaluate the fixed
   permutation. So `romError` bounds the same events in another experiment, and no theorem gives the `error` of the
   premise from it. Lean has no random-oracle closure premise, because the query bound of a stage grows with its depth.
   The sum has one Assumption 1 error for each stage of the reverse extractor, which is one visit, and stage `j + 1`
   runs the extractor of stage `j`, which may rerun stage `j`. So the cost of a stage grows with its depth. This is the
   constant-depth limit of HyperNova Lemma 18: the Lean statement holds for every depth, but it gives no concrete
   security for a long history. The errata's Definition 11 states knowledge soundness only for a constant number of
   steps that does not depend on the security parameter, and its proof composes the stage extractors because that number
   is constant. A random-oracle bound for each fold does not remove this limit: each stage's adversary includes the
   earlier stages' reruns.
4. **Scope: the circuit selection.** HyperNova NIVC selects the circuit of a step with `φ(z_i, w_i)`. The Lean step
   relation takes it from the machine's selector of the application state and witness (`Lifecycle.BoundedDepth.machine`,
   `Lifecycle.BoundedDepth.stepHolds_iff`). Both bounded-depth circuits run the same application step, so the selector
   does not change the computed state. `Claims.Ivc.terminal_bridge` takes the selector that returns the circuit that ran
   on a valid step (`Export.BoundedDepth.TerminalBridge.chosen`), so the prover picks the circuit. Rust picks it from
   the depth of the selected slot (`SPEC.md` Algorithm 5): a light fold below the cap and a refresh fold at the cap. The
   light depth guard rejects a light fold at the cap for every pick. The statement `(i, z0, zi)` does not fix the
   circuit of a step; the state hash binds the program counter of the last step only.
5. **Gap: Rust and Lean agree on recorded inputs only.** The Rust tests compare Rust's expansion of the packages, its
   identity code, its witness program and its logical assignment with Lean exports, and they check the lifecycle on
   fixed inputs (`SPEC.md` section 9). No theorem covers arbitrary Rust execution. No Lean theorem states that the
   package's witness program (`Export.BoundedDepth.StepPackage.Package`, field `witness`) computes an assignment that
   the rows accept. The completeness of each gadget states only that such an assignment exists
   (`Circuit.FormalCircuit`), and the rows do not fix each hint, for example the inverse of `Gadgets.Range.CanonicalU64`
   when the value that it inverts is 0. This gap is of completeness only, because the verifier checks the rows and not
   the program.
6. **Gap: the memory round.** `Export.BoundedDepth.NebulaSecurity.MemoryRound` states the memory round for the two step
   circuits as a hypothesis: for each admitted stage and the memory statement of its tape, the extracted run is accepted
   and has a segment that passes the product test with unbalanced multisets with probability at most `error_mem` of that
   stage. It is not an assumption of the list, and no result derives it from the list: the relation computes `η` with
   the Poseidon2 permutation inside the proven step, and no ideal-permutation model covers that check. The random-oracle
   model of `η` in adversary 5 gives the form of `error_mem`: `(Q_η + 1) · 2 · m_mem/q²` plus the retry term, where
   `Q_η` counts the evaluations of the `η` transcript, in the circuit and outside it. Lean does not bound the retry term
   with a number, because each pair that it counts is a Poseidon2 collision. The owner chose this form
   (`decisions/nebula-memory-checking.md`). The model gives the `η` oracle no relation to the fold addresses (trust item
   3). Their domain tags differ, and two block lists with these tags are distinct
   (`Claims.Encodings.domains_separated`). The record chains need only the collision bound of trust item 2.
7. **The compressed proof [KECCAK] [MSIS] [P2-PERM] [SPONGE].**
   Adversary 6 has these premises.
   - *The premises of the Lean result.* `Claims.Compression.argument_sound` takes these hypotheses.
     `Spec.Compression.Verifier.WellFormed` (each slice is aligned and inside the cube of `W`), the bound `2^25` on the
     cells of a witness slice and the bound of the key on the blocks of a claim hold for the layout of the Lean model of
     `Layout::new` (`Spec.Compression.Rust.layoutNew_production`), when each claim has at least its 320 public cells,
     which the shape check of Rust `layout` makes. `Spec.Compression.Rust.layout_soundness` states the bound without
     these premises. The premise is that Rust `Layout::new` is the model: tests compare the two on each claim set of the
     reference application and on generated claim sets, and no proof does. `Spec.Compression.Verifier.CountsNonzero`
     (`2^vars` is below the characteristic of `F_q`) follows from the bound `2^25`
     (`Spec.Compression.Production.params_countsNonzero`). `Spec.Compression.Verifier.MatrixSpec` (the verifier's matrix
     part is the multilinear extension of the matrix weights) is a theorem for the Lean model of Rust `matrix_part`
     (`Spec.Compression.Verifier.matrixPass_spec`), and `Claims.Compression.production_sound` states the bound at the
     production parameters without it. The model is a transcription of the Rust code by hand. Lean emits the output of
     the model on three sample cases: relations of 5 to 16 rows with slots across one and two block borders, two claims
     on one relation and claims on two relations of one size (`Export.MatrixPassParity.pass_matrixSpec` gives
     `MatrixSpec` for the function that the emitter runs). The Rust test `the_matrix_part_is_the_lean_matrix_pass`
     requires that `matrix_part` gives these outputs. No data compares the two on the rows of the step circuits. Lean
     defines the factored rows of a package (`Export.BoundedDepth.FinalRows.packageRows`), but the model of
     `matrix_part` takes about 0.1 s for each row of a sample relation, which is hours for the more than 650,000 rows of
     the two circuits. Rust tests compare `matrix_part` with its Rust definition on those rows. `L ≤ 2^2` and `κ ≤ 2^5`
     are const asserts in Rust, and `params_soundness` proves them at the production values. `slot_holds` also takes the
     hypothesis `Spec.Compression.Claim.Represents`: the dense rows of a slot claim are the matrices of its circuit's
     relation on the claim's cells. Rust builds these rows from the package that `SPEC.md` Algorithm 2 pins. Lean's
     model of them is `Export.BoundedDepth.FinalRows.packageRows`, and these rows represent the circuit's relation
     (`Export.Generic.PackageRows.rows_represents`). The premise `rows` of the final claims is that Rust builds rows
     with the same dense matrices. Lean emits a checksum of the model's dense matrices
     (`Export.BoundedDepth.RowsParity.fastSum_eq`), and the test `factored_rows_match_the_lean_dense_checksum` computes
     it from Rust's plan rows and spans for both packages of the reference application and of each Nebula program. It
     also needs that Rust gives each slot claim to the argument with the values of
     `Lifecycle.BoundedDepth.CompressionSlot.claimValues`, which no check compares. The model has three more premises.
     First, the Akita openings are ideal: an opening is accepted only when the committed table has the opened value. The
     prover commits the tables at two times (`SPEC.md` section 5.4): `W` before the Π_CCS run and the lookup challenges
     `τ` (row 2), and `V`, whose cells are `1/(τ_k − W(x))`, after `τ` (row 9). The game has the same order: `W` before
     `τ`, and `V` after `τ` and before the other challenges (`Spec.Compression.Verifier.afterTables`). The claims with
     Akita openings have the extraction and its composition with the game inside their results (items below).
     `Export.BoundedDepth.FinalVerifier.Premises`
     collects the premises on the setup: `Represents` for the dense rows of each circuit, and at most the blocks of its
     package for each claim. The final claims meet them from the premise `rows`
     (`Export.BoundedDepth.FinalRows.premises_of_rows`). The game runs the gate of step 5 of `SPEC.md` Algorithm 10
     (`Export.BoundedDepth.FinalVerifier.Setup.Gate`). It rejects a key whose block count is not the count of Algorithm
     11, claims that the Lean model of Rust `Layout::new` does not lay out or that have fewer than 320 cells, and a
     layout whose `error_bits` is below the security minimum. It takes its layout from that model. That layout has the
     production bounds (`Export.BoundedDepth.FinalVerifier.Setup.layout_facts`), so the premise on the layout is that
     Rust `Layout::new` is the model, as above. The game takes the slot claims before `W` and the run of Π_CCS, and Rust
     absorbs only `x`, their Poseidon2 hash, before that run, so the slot claims are bound through `x` up to a collision
     of trust item 2. Second, the key polynomial is the honest key. Rust enforces this with the pin of the full bytes of
     the verification key (`crates/ns-compress/artifacts/key20.vk`, `crates/ns-compress/artifacts/key30.vk`), not with a
     digest. Third, each draw is uniform for each query. This is the sponge of `[SPONGE]` under `[P2-PERM]` only if the
     compression transcript absorbs an injective framing of its messages with a domain of its own. Lean proves both for
     the words of the transcript: its sponge calls fix the run among the runs of one statement shape
     (`Export.BoundedDepth.CompressionSponge.calls_injective`), and they start with a domain that no other use starts
     with (`Export.DomainSeparation.compression_owns`, `Claims.Encodings.domains_separated`). Rust tests compare the
     words with Rust (section Hashed encodings). The gap is the join: no theorem states that the draws of
     `Spec.Compression.Verifier.game` are the reads of this sponge, so no result applies `Spec.Sponge.Transfer` to the
     game. Lean models the order: each path of the game receives and draws the rows of `SPEC.md` section 5.4 in order
     (`Spec.Compression.Rust.game_follows`), and a test replays a real proof in that order and checks each challenge of
     `verify`. Akita's own transcript is not modeled (row Akita transcript of section Hashed encodings).
   - *The mode.* The verifier takes the mode from its own circuit, and a proof of the other mode has another tag and
     another slot count, which the verifier rejects (`SPEC.md` Algorithm 10, step 3). Rust selects the light mode for a
     memory circuit whose geometry has `N · S_max ≤ 65` (`SPEC.md` Algorithm 9). Each verifier sees one mode, so the
     compression transcript has no mode word, and `calls_injective` covers the one statement shape of each circuit. The
     soundness of the light mode needs no bound on the iteration. Its completeness needs one: an honest run of at most
     `cap + 1` steps has a proof in the light mode (`Claims.Ivc.light_run_complete`,
     `Export.BoundedDepth.FinalVerifier.checks_light`), and the memory verifier accepts no longer run of a WASM size
     (`Claims.Nebula.memory_light_complete`). When the verifier's key is the key of Algorithm 11 and has `2^19` blocks,
     and its security minimum is at most 119 bits, that proof also passes the whole gate of step 5
     (`Claims.Final.light_gate_complete`). Each light layout that the model accepts at a depth at most the cap has
     `error_bits` 119 (`Claims.Compression.light_error_bits`). No theorem derives the key of `2^19` blocks from a
     package. The test `error_bits_and_the_gate_match_the_lean_bound_on_each_light_claim_set` checks it for the
     reference application, and the fold gate holds the minimum to 115 bits. In the light mode, `compress` checks the
     claim and the depth of slot 1 but not its witness, and it commits only the witness of slot 0
     (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`, `check_light_envelope`). So it turns a proof
     whose slot 1 witness is changed into a final proof that Algorithm 10 accepts. The statement is still true, because
     the zero witness opens the default slot (`Export.BoundedDepth.StepExtend.defaultSlot_holds`). `[HN-A1]` for the
     refresh fold and its knowledge error stay premises of the circuits that keep both slots (trust item 3).
   - *The compression key.* The verifier's key is the key of `SPEC.md` Algorithm 11. It holds the Akita commitment of
     the key polynomial of the production Ajtai key, and Akita's commitments of the prefixes of its setup from Akita's
     public seed. The code pins the bytes of this key for each block count, and a test computes each pinned key again
     from the public seeds. `Verifier::compression_key` gives the pinned key, and `Verifier::decode_compression_key`
     accepts loaded bytes only when they equal it, so the pin enforces the premise for each key that the verifier uses.
     Under another key, Akita gives no binding, and the verifier can accept a false statement. The pin is a value in the
     code, and the test is the check of the pin. No Lean result covers either. The Lean result takes the honest key as a
     premise.
   - *The Akita opening and the gap of `V`.* The one Akita proof opens the key polynomial, `W` and `V`. It uses Akita
     (ePrint 2026/1983) at git revision `d50f700f`, crate version 0.1.0, with the configuration
     `RecursiveCommitmentConfig<fp64::Dense>` and the schedule files of `SPEC.md` section 3. No claim takes Akita
     Theorem 10.46 as a premise. Its conditions (Akita Definition 9.2) fail for `key30.aks`, which is not in the family
     F1 (`Spec.Compression.Akita.Schedule.not_admissible_key30`). The composition theorem of Akita (Theorem 10.45) has
     every root commitment in the first message, before the first challenge of the application, so it does not cover
     the commitment of `V` after `τ`. Akita's checkpoint structures (Definition 10.16) allow more than one checkpoint,
     but no result of the paper states the composition with a second commitment. Lean proves it for `key20.aks`
     (`Claims.AkitaKey20.key20_verifier_sound`) and, under the hypotheses of the items below, for `key30.aks`
     (`Claims.AkitaKey30.key30_verifier_sound`). The commitment of the compression key binds the key polynomial under
     `[MSIS]` for each extractor of Akita Definition 3.7 (`Claims.Compression.key_binding`, item below) for the
     scheme whose decommitment relation is the certified
     relation: Akita Definition 10.21 with the weak certificate of Akita Definition 10.22
     (`Spec.Compression.Akita.Certified.Opens`). Akita's `Rel_com` (Definition 10.21) has no norm bound on the source
     blocks, so it is not binding alone. The documentation of the Akita code states a target of a 128-bit Fiat–Shamir
     rate for each random-oracle query. It says that this target is not an end-to-end bound and that the code declares
     no query budget, no bound on the size of the extraction tree and no reduction loss. The paper has a complete
     formula, that of Theorem 10.46, and no evaluated end-to-end number for a schedule. Akita has no audit and no
     analysis in the quantum random-oracle model.
   - *What Lean proves of Akita.* The results are the claims of `Claims.Akita` and `Claims.AkitaKey20`. They and the
     claims of `Claims.AkitaKey30` are Nightstream bounds, not the bound of Akita Theorem 10.46. One query of the model
     returns all the slots of a key, and the paper counts a query for each indexed coordinate stream (Akita Section
     10.3). No Lean result relates the two counts, and the shipped files have no `(λ_FS, Q_max)`. A bound with no term
     `breakChance` is for acceptance with no output of the extractor, and a solution of Module-SIS is an output, so
     it is not the knowledge error of the paper. The main claim is
     `Claims.AkitaKey20.key20_verifier_sound`, from `Export.BoundedDepth.FinalKey20.sound`. It is for the schedule of
     `key20.aks`: a root of three groups, level 1 with coefficient packing, levels 2 to 5 with evaluation trace and the
     terminal, with the quotient-lift check at the root and at level 1, and with the squared-norm route at levels 3 to 5
     and at the terminal (`Spec.Compression.Akita.Key20.tower_suffix`). The prover has at most `Q` queries to one
     uniform table on message keys (item below). The final verifier with these openings
     (`Spec.Compression.Akita.Extract.towerAccepts`) accepts a statement for which no proof with witnesses satisfies
     `Verifies` with probability at most `(Q + 1) · (runCost + 317479/(1024·|K|) + 135 · p) + (Q + 1) · 144/budget
     + breakChance`. `runCost` is the sum of the bounds of the draws of the Π_CCS run and of the argument
     (`Export.BoundedDepth.FinalVerifier.Setup.runCost`). `317479/(1024·|K|) + 135 · p` bounds the stage errors of
     the opening, which has 135 fold coordinates (`Spec.Compression.Akita.Key20.towerError_le`). A slot is
     `Spec.Compression.Akita.Replay.CodeSlot K`, and the fold decoders are those of the code: `p ≤ 2^−128` without a filter, for words of 128 or 129 bits (`Spec.Compression.Akita.Replay.slotChance_open`), and `p ≤ 1/|Kept G|` with it (`Spec.Compression.Akita.Replay.slotChance_kept64`), so
     with `kept` the exact fold term is at most `135/2^128`. The claims state `p ≤ 1/q²` and the bound `455719/(1024·q²)`
     (`Spec.Compression.Akita.Replay.slotChance_kept64_field`, `Claims.AkitaKey20.key20_opening_bound`). For these
     decoders Lean proves the unit differences (`Claims.Akita.short_units`) and the sparse challenges. The operator bound `Γ` of the filtered families is a theorem for the filter of the code
     (Akita Theorem H.1, `TrigTable.filtered_code`), with the error bound of its tables (`TrigTable.code_accurate`). The extractor has 144 retry loops,
     each with at most `budget` attempts. The result closes the gap of `V` inside itself, with two checkpoints of Akita
     Definition 10.16 in sequence. The first has the candidate `W`, and its stages are the Π_CCS run and the draw of
     `τ`. The second has the candidate `W` and `V` (`Spec.Compression.Akita.Extract.Compose.suffix`, for the two parts
     of the game of `Export.BoundedDepth.FinalVerifier.Setup.game_eq`). The parts of the proof are Akita Theorem 10.17
     (`Claims.Akita.checkpoint_extraction`, `Claims.Akita.checkpoint_sound`), the suffix of Akita Proposition 10.44 with
     the batching root (`Claims.AkitaKey20.key20_opening_sound`, `Claims.AkitaKey20.key20_opening_bound`), and Akita Lemma 10.23
     and Akita Corollary 10.32 for a fold (`Claims.Akita.fold_weak_binding`, `Claims.Akita.fold_extracts`, and
     `Claims.Akita.trace_weak_binding`, `Claims.Akita.trace_extracts` at a trace level). `breakChance` is the `[MSIS]`
     term: the probability that the extractor returns a solution of Module-SIS
     (`Export.BoundedDepth.FinalAkita.breakChance`). The instances are the key commitment, at `η_A` of the key
     group, at or above the norm of the reduction (`covers`); `A` of the root and of each level, at `η_A = 2 · κ̄_1 · Δ^cert_f` on the
     coefficient route and at the squared radius `64 · Γ² · S_max` on the squared-norm route; `B` and `D`, at `b⋆ − 1`;
     a compression map, at 1; and the matrix of the terminal, at its squared radius
     (`Spec.Compression.Akita.Extract.Route.Break`). The solver runs the prover at most `K̂ + (Q + 1)(K̂ − 1)` times in
     expectation, with `K̂ = 4 · 21,449,539,584`: the tree size of the opening and a factor 2 for each checkpoint of the
     argument (`Claims.Akita.checkpoint_runs`, `Export.BoundedDepth.FinalAkita.mixed_runs_le`). These are hypotheses of
     `sound`. Lean proves them for the decoders of the code above, and it proves the operator
     bound for the filter of the code. Each difference of two challenges of a family is a unit, for which Akita cites Lyubashevsky and
     Seiler. Each challenge is sparse, and at the squared-norm levels it has the operator bound `Γ`
     (`Spec.Compression.Akita.UnitDifferences`, `Spec.Compression.Akita.Sparse`, `Spec.Compression.Akita.MulBound`). `p`
     is at least the largest probability of one challenge under the slot decoder of each family
     (`Spec.Compression.Akita.Extract.slotChance`). An answer has 28 slots or more (`coords`), and the fold coordinates
     of a level have different slots. `registered`: the key commitment of the verifier has a certified decommitment to
     the key polynomial. `use` (`Export.BoundedDepth.FinalKey20.Use`): the claims pass the layout of step 5 of `SPEC.md`
     Algorithm 10, the key polynomial has 512 blocks, and the opening point of `V` has the zero count of the layout
     (`Export.BoundedDepth.FinalKey20.laidZeros`). Then the three opening points have 20, 16 and 16 coordinates
     (`Claims.AkitaKey20.key20_points`, `Export.BoundedDepth.FinalKey20.Use.points`), and the block counts of the Lean
     schedule agree with the points of its links (`Claims.AkitaKey20.key20_blocks`). No Lean result compares `laidZeros` with
     the Rust code. The premises of the final verifier are those of `Claims.Final.verifier_sound`. `sound` is for one
     claim shape: one `pc` and one depth for each slot. For a finite family of shapes, the verifier
     `Export.BoundedDepth.FinalKey20.finalAccepts` accepts a statement of zero steps, with no draw and no error term, or
     it is the verifier of the shape that the first message names. Its bound
     `Export.BoundedDepth.FinalKey20.shapesBound` is the sum of the bound above over the shapes, so it has the number of
     the shapes as a factor (`Export.BoundedDepth.FinalKey20.sound_all`, `Export.BoundedDepth.FinalKey20.knowledge_all`,
     and `Claims.AkitaKey20.key20_verifier_sound` in the form of `Claims.Final.verifier_sound`). In this game, a false
     history (`Export.BoundedDepth.FinalKey20.FalseHistory`) has a probability of at most `shapesBound + walkBound`,
     with `Export.BoundedDepth.IvcSecurity.walkBound` for a bound `depth` on the accepted iterations, and the same bound
     holds for the witnesses of the reverse extractor (`Claims.AkitaKey20.key20_history_sound`,
     `Claims.AkitaKey20.key20_history_knowledge`). A false memory statement has the bound `shapesBound` plus the terms of `Claims.Nebula.memory_sound` in this game (`Claims.AkitaKey20.key20_memory_sound`). The opening verifier of
     the tower accepts the honest prover at these numbers if the fit conditions hold for its responses
     (`Spec.Compression.Akita.Key20.tower_complete`). With the fit conditions as the hypothesis `fit` it accepts for each family of
     tables (`Claims.AkitaKey20.key20_opening_accepts`, `Spec.Compression.Akita.Key20.fit_zero`, `Spec.Compression.Akita.Key20.fit_quiet`). Levels 1 to 5 have the bounded source of
     Akita Section 4.5, and the full-field condition is false for them (`Spec.Compression.Akita.Key20.not_honest`). The
     whole verifier accepts an honest proof for each table whose answers decode to fold challenges and to no challenge
     of the reject set of the argument (`Export.BoundedDepth.FinalKey20.complete`,
     `Export.BoundedDepth.FinalKey20.accepts`). One answer decodes to a challenge of this set
     (`Spec.Compression.Verifier.Rejects`) with a probability of at most `tableError + omegaError`
     (`Spec.Compression.Verifier.rejects_mass_le`). This acceptance has two hypotheses that no Lean result gives:
     `leads`, the honest run of Π_CCS on the fresh instance, and `fit`, the fit of the honest responses. The
     completeness of Akita Theorem 10.9 is not perfect: a response does not fit with a probability. The verifier has the nonce of the response search with its limit `2^12` (`Spec.Compression.Akita.Extract.Tier.Decodes`). For the root and level 1 of `key20.aks` and decoders with `Spec.Compression.Akita.SignFair`, each nonce fails after one prefix (a fold challenge does not decode, or the response does not fit) with probability at most `(ε + 2^−12)^4096` and `(ε + 2^−49)^4096`, where `ε` is the sampler term of Akita (157) (`Claims.AkitaKey20.key20_root_search`, `Claims.AkitaKey20.key20_level1_search`), and `ε = 0` for decoders that decode each slot (`Claims.AkitaKey20.key20_root_search_total`, `Claims.AkitaKey20.key20_level1_search_total`). No Lean result joins these bounds to `fit`.
     At the other levels the bound is the response-model premise of Akita (152) and (340), which the paper calls heuristic: its text "motivates this premise but does not prove it" (Akita Appendix K). `key20.aks` is the
     file of the test key of `2^9` blocks, and the claims of this item do not cover `key30.aks` (item below). The Lean
     verifier differs from the pinned code (`decisions/akita-compression.md`). The
     acceptance has the hypotheses `leads` and `fit`, and no closed accepting instance exists for the whole verifier.
   - *What Lean proves for `key30.aks`.* The results are the claims of `Claims.AkitaKey30`, for the two rows of the file
     of the key of `2^19` blocks. The opening has the batching root, the offloaded root fold of Akita Section 9, the
     batching stage of the two claims of level 1, level 1 with the group of the setup prefix and the group of the
     witness, levels 2 and 3 with coefficient packing, levels 4 to 6 with evaluation trace and the terminal
     (`Spec.Compression.Akita.Key30.tower_suffix`). The claims are `Claims.AkitaKey30.key30_opening_sound` and
     `Claims.AkitaKey30.key30_opening_bound` for the opening, `Claims.AkitaKey30.key30_verifier_sound`,
     `Claims.AkitaKey30.key30_history_sound` and `Claims.AkitaKey30.key30_history_knowledge` for the final verifier of
     each claim shape and of the proof of zero steps, `Claims.AkitaKey30.key30_points` for the 30, 27 or 26, and 28
     coordinates of the opening points, and `Claims.AkitaKey30.key30_instance_list` and
     `Claims.AkitaKey30.key30_instances` for the instances of Module-SIS. The bound of one shape is `(Q + 1) · (runCost
     + (7595403 + 4096·B_1 + 4096·B_2)/(4096·q²)) + (Q + 1) · (1469 + B_1 + B_2)/budget + breakChance`, with `B_1 + B_2` equal to 211 for
     `W` of 27 variables and to 201 for `W` of 26 variables (`Export.BoundedDepth.FinalKey30.Opening.shapeBound`). The
     decoders are those of the code on the slots `Spec.Compression.Akita.Replay.CodeSlot K`, so the unit differences, the
     sparse challenges and `p ≤ 1/q²` (with `kept`) are theorems. The premises are those of the claims of `key20.aks` and two more, and one
     of them is open. `shared`: the matrices of the root are views of the setup prefix, and Lean proves the split (132)
     (`Spec.Compression.Akita.Key30.Tower.Shared`). `registered`: the commitment of the setup prefix in the verifier
     setup is the output of the Lean commitment algorithm for the setup prefix, the premise of Akita Lemma 9.1
     (`Spec.Compression.Akita.Batch.registered_commit`). The operator bound `Γ` of the filtered families of levels 5 and
     6 and of the terminal is a theorem for the filter of the code (`Spec.Compression.Akita.TrigTable.filtered_code`). The Lean verifier of the opening accepts the honest proof for each setup prefix and
     each family of tables with the fit conditions (`Claims.AkitaKey30.key30_opening_accepts`). They are the hypothesis `fit`
     (`Spec.Compression.Akita.Key30.fit_zero`, `Spec.Compression.Akita.Key30.fit_quiet`). The memory result is in this game (`Claims.AkitaKey30.key30_memory_sound`). `decisions/akita-compression.md`, section
     "The results for `key30.aks`", gives the limits and the differences from the paper and from the code.
   - *Differences between the paper, the code and the Lean model.* `decisions/akita-compression.md`, section
     "Differences between the paper, the code and the Lean model", has them: (a) to (j) for each schedule, and (k) to
     (o) for the opening with an offloaded root. In short: the model is one uniform table on message keys with
     independent slots, which is the slot reading of `[KECCAK]`, and one query returns all the slots of a key; the
     Lean verifier is not the verifier of the pinned code; the bound `N_setup · q / 2^w` of `Δ_setup` bounds nothing at the draw width of the code; the code
     declares no `(λ_FS, Q_max)`; and the Lean extractor has the limit `budget` on the attempts of each retry loop.
   - *The Keccak family `[KECCAK]`.* Akita makes its challenges with a Keccak-f duplex sponge, its digests with
     SHA3-256, its session identifier with SHAKE128 and its setup with SHAKE256. No step circuit computes these
     functions. The owner accepted this hash family for Akita on 2026-11-13 (`decisions/akita-compression.md`).
   - *Module-SIS for the Akita commitments `[MSIS]`.* Lean reduces the binding of the commitment of the compression key
     to Module-SIS (`Claims.Compression.key_binding`): a certified decommitment to a different table gives, with the
     state of the commitment algorithm, a solution for `A` at `η_A = 2 · κ̄_1 · Δ^cert_f`, for `B` at `b⋆ − 1`, or for
     `F_1` or `F_2` at 1. For the shipped key shapes, `η_A` is the bound that the schedule file prices for `A` (32505732
     for `key20.aks`, 16642998148 for `key30.aks`), and Lean proves that the norm `β̄_∞ + κ̄_1 · b/2` of the reduction
     is at most `η_A` (premise `covers`). Here `κ̄_1` and `β̄_∞ = Δ^cert_f` are the numbers of Akita (50) for the root
     opening descriptor of the file. For the claims, no Lean result shows that the verifier of the root fold certifies
     them: that is the root weak opening (item above). The claim gives no bound on `bindingError` (premise `msis`). It
     is for the output of `Spec.Compression.Akita.commit` on the key polynomial. No Lean result shows that the pinned
     key holds this output for the Akita matrices of the fixed seed: the pin and its test (item above) are the only
     check. The paper prices `A` on the coefficient route in the ℓ∞ norm at `2 · κ̄_1 · β̄_∞`, which is the Lean `η_A`,
     and on the squared-norm route at the least power of two at or above `64 · Γ² · S_max`. The Ajtai commitments of the
     claims keep the estimate of trust item 2.
   - *The Module-SIS instances of Akita `[MSIS]`.* The hardness of an instance depends on the ring degree, and the Lean
     reductions do not. For `key20.aks`, the 43 matrices of the Lean tower give 13 instances `(d, n, m, p, η)` that are
     not below another instance (`Claims.AkitaKey20.key20_instance_list`): the compression maps at the degrees 32 and 64, `B`
     and `D` at 64 and 128, and `A` at 64, 128, 512 and 1024, with the norm ℓ2 for 3 of them. Each bound of a Lean
     witness of `R_bind` is at most the bound that the file prices, and the list is complete for `breakChance`
     (`Claims.AkitaKey20.key20_instances`). `tests/AkitaSchedules.lean` compares the degree, the rank, the width, the norm
     and the bound of each matrix `A`, `B` and `D` of the file with the list. The file does not hold the compression
     maps. For matrices that are the views of one setup vector, `Claims.AkitaKey20.key20_instances_shared` gives a solution
     for a view of the list (Akita Lemma 10.20). For `key30.aks`, the 54 matrices of the Lean tower of each row give 18
     instances that are not below another instance, at the degrees 32 to 2048 (`Claims.AkitaKey30.key30_instance_list`).
     The list is complete for `breakChance` (`Claims.AkitaKey30.key30_instances`), and `tests/AkitaSchedules.lean`
     compares it with the two rows of the file. (i) The price. The pinned code admits an instance only at 169.07 bits or
     more of quantum ADPS16 Core-SVP cost, `2^(0.265·β)` for the BKZ block size `β` (policy `Quantum169BitADPS16`). The
     model is a direct ℓ∞ attack, or the Euclidean attack for a bound on ℓ2, on the scalar instance `(n·d, m·d)`. The
     estimator is the estimator of the Akita authors. A run of it in a copy of the pinned code reproduces the two cells
     of Table 10 of the paper, 147.34 and 153.17 bits. It gives 171.99 bits or more for each of the 46 ℓ∞ instances of
     the two files and 173.57 bits or more for each of the 3 instances in ℓ2. For 27 of the 46 the search was
     exhaustive. For the other 19 it was a search for a local minimum that the estimator does not certify, so these 19
     numbers are estimates and not floors: Akita Appendix G accepts a floor only from a certified search. The files of
     this run are not in the repository. The lowest of `key20.aks` are `A` of level 2, `(128, 4, 512, ∞, 126852)`, at
     171.99 bits and `B` of level 2, `(128, 1, 1092, ∞, 31)`, at 172.51 bits. The Sage `lattice-estimator` runs of (v)
     are in another model. (ii) The degrees. Table 10 of the
     paper (Akita Appendix G) has the degrees 32 and 16 for the compression maps at `q = 2^64 − 59`, with a floor of 128
     bits for each instance and more than 125 bits for the family. The pinned code has 64 and 32 since the Akita commit
     `dfdbd767` of 2026-10-20, which raised the floor to 169 bits. So Nightstream runs larger degrees than the table of
     the paper, and the 169 bits are not in the paper. The paper gives no number for an instance of `A`, `B`, `D` or the
     terminal at this modulus. (iii) The price does not include attacks on the ring or module structure, the loss of the
     reduction, or the sum over the instances that share one setup stream. In the model of (i), this sum is about 170.9
     bits for each file. (iv) The reductions. Lean proves that a solution moves to a matrix with more columns, fewer
     rows or a larger bound (`Spec.Compression.Akita.ShortKernel.carry`), and from the degree `s · d` and the width `m`
     to the degree `d` and the width `s · m`, at one rank and one bound (`Claims.Akita.module_sis_descent`, and
     `Spec.Compression.Akita.ShortKernelSq.lower_chance` for ℓ2). So descent moves a price only from a smaller degree to
     a larger degree. The first-stage cell of Table 10, `(32, 1, 2048, ∞, 1)` at 147.34 bits, covers `F_1` and `H_1` by
     width and descent, and `F_2` and `H_2` by width. The reductions carry the 169 bits to no other instance. (v) The
     target. The owner's target is 128 bits against a classical attacker (2026-11-22), by the method of the main-key
     record in the ADPS16 classical model (the `MSIS` entry of the section Assumptions). The command
     `python3 scripts/akita_sis_instances.py` lists the matrix views of the two files: 43 and 108 views, 49 distinct
     and 24 maximal. With a checkout of `lattice-estimator` at commit `3e48ef4` on `PYTHONPATH`, the command
     `sage -python scripts/estimate_akita_sis.sage` writes the numbers of the main key and of each distinct view to
     `scripts/akita_sis_estimates.csv`. The main key has 140.16 bits.
     Two matrices `B` of rank 1 are below the target: level 2 of `key20.aks` at 122.06 bits, and level 2 of `key30.aks`
     at 123.81 bits and 124.39 bits for its two rows. Each other instance has 134.90 bits or more. The 169 bits of (i)
     are a number of the direct ℓ∞ model and not of this method. The schedule files are not changed, so the change of
     these two matrices is open (`decisions/akita-compression.md`, Open points).
   - *The Poseidon2 sponge for the argument `[P2-PERM]` `[SPONGE]`.* The Poseidon2 sponge makes each challenge of the
     Π_CCS run and of the argument (`SPEC.md` section 5.4). Each term of the Rust sum of `SPEC.md` section 3 is for one
     interactive run, and Lean proves that the exact sum bounds one interactive run
     (`Spec.Compression.Rust.params_run_soundness`). In the state-restoration model with `Q` queries, Lean bounds the
     non-interactive argument by `Q · error` (`Claims.Compression.argument_sound`), and `error` is at most the exact sum
     (`Spec.Compression.Rust.error_le_runError`). Lean bounds the Π_CCS run with one source by `Q · 38/q²`
     (`Claims.Compression.fresh_sound`). The sponge meets this model, up to `spongeError N`, only under the open
     conditions of the third premise above.
   - *Separation of the two transcripts.* The session bytes of the Akita proof are a label and lanes 0 to 3 of the
     Poseidon2 sponge state after the last message of the argument, so they bind the opening to each earlier message,
     the bytes of the compression key included, unless these four lanes collide (trust item 2). The Π_CCS run and the
     argument have their own domain blocks, which differ from the fold domains. `[P2-PERM]` and `[KECCAK]` make the
     permutation and the Keccak functions independent. The compression domains, as Lean restates them, are separated
     from the fold domains (`Claims.Encodings.domains_separated`), but no emitted data compares the restated tags with
     Rust. `Claims.Final.history_sound` puts the draws of the argument and of the Π_CCS run on the tape of an IVC
     adversary whose folds use Poseidon2 (trust item 3), so it also needs the separation of the compression domains from
     the fold domains.
   - *The error of the argument.* Lean bounds it by `Q · error`, with `error < 2^−119` at the production shapes
     (`Claims.Compression.production_error`). Rust bounds the sum of `SPEC.md` section 3 for one interactive run, about
     `2^−119.02` for the reference application at 4 lookup challenges, by `2^−119` in integers. To either, add the error
     of a Π_CCS run with one source, which Rust does not compute. Rust holds the Π_CCS test error for 17 sources to the
     caller's security minimum (`SPEC.md` section 3).
   - *The Rust code.* Trust item 5 applies. The tests check fixed inputs. The verifier does not expand the Ajtai key,
     but the key polynomial that the compression key commits is the SHAKE128 key of `SPEC.md` Algorithm 1. So premise P1
     applies to it as it does to `SPEC.md` Algorithm 6.
## Out of scope

- Completeness does not follow from a knowledge bound: a bound on the
  adversary is not evidence that an honest proof is accepted. The honest
  prover has separate results: an honest light fold and an honest refresh
  fold are accepted (`Claims.Fold.light_complete`,
  `Claims.Fold.refresh_complete`), the bound `accumulatorBound` of each
  depth of an admissible schedule stays below `B`
  (`Claims.Fold.honest_bounds`, arithmetic only), and, given an
  accepted fold, an honest step of either circuit has an assignment in which
  every row of its source layout holds
  (`Layout.BoundedDepth.LightAssemblerCompleteness.complete`,
  `Layout.BoundedDepth.RefreshAssemblerCompleteness.complete`). Lean joins these
  results. Along an honest run, each slot's witness stays within the bound
  of its depth (`Export.BoundedDepth.StepExtend.Honest`). An honest step's witness
  satisfies the exported relation at the key's widened slot layout
  (`Claims.Step.light_complete`,
  `Claims.Step.refresh_complete`). The terminal verifier accepts
  the honest proof of every valid run
  (`Claims.Ivc.run_complete`) and of every memory
  execution (`Claims.Nebula.memory_complete`). HyperNova
  Definition 11 (i) extends each proof that the verifier accepts. The
  terminal verifier holds each slot to the bound of its depth
  (`Export.BoundedDepth.RunCompleteness.Verifies`, `SPEC.md` Algorithm 6), and one
  valid step extends each proof that it accepts
  (`Claims.Ivc.step_extends`). In the honest proof of each of these three
  claims, the fresh witness is zero on every cell from the width of its
  circuit's package (`Export.BoundedDepth.RunCompleteness.FreshZeroTail`).
  `Verifies` requires zeros only on ring blocks, but Rust requires a zero on
  each cell (`SPEC.md` Algorithm 6, Rejects). So the claims rule out a
  rejection of an honest proof by this check of Rust. These
  results take as premises a package check that the emitter evaluates
  (`Export.BoundedDepth.StepExtend.Complete`), key digests of four words, and
  witnesses of the application's witness length. Each application promises that its step keeps the state width
  (`Lifecycle.Stage1.Application.Program.step_length`); the counter must stay
  below `q`.
- The error is not one security level. Lean bounds the fold error of
  adversary 3 in the random-oracle and ideal-permutation models
  (`Claims.Fold`), not for Poseidon2 itself, and the other terms depend on
  MSIS at the solver's work and on Poseidon2 collisions.
- The compressed proof has Lean results for its argument only: soundness
  (adversary 6) and, with ideal openings, completeness, in which the verifier
  rejects the honest argument only when a lookup challenge is a table value or
  `ω` has no row point (`Claims.Compression.argument_complete`). Its error is
  not a security level, and the final proof is not zero-knowledge.
- Attacks on the fixed Poseidon2 permutation that a uniform permutation does
  not have (`[P2-PERM]`) are outside the model. Attacks that use the hash inside the circuit (KRS)
  are outside every random-oracle model; the verifier's packages fix the
  relation.
- The adversary is a classical algorithm. Quantum adversaries are not
  modelled. The κ estimate of trust item 2 gives a quantum Core-SVP cost, but
  no Lean result uses it.
- No adversary chooses the application or the setup (Applications and
  setups). An application chosen from the adversary's coins, or after the
  seed, is outside the results.
- Concrete security for long recursive histories (trust item 3).
- Lean states no machine running time and no Rust execution result beyond the
  recorded inputs.
- A program that the adversary chooses after it sees the setup. Adversary 5
  fixes the program before the adversary runs.
- The step relation that `Circuit::compile` builds with a Rust application
  other than the reference hash chain, whose rows replace Lean's application
  region (`SPEC.md` Algorithm 2, step 3). Lean exports only its own packages.
- The Rust-built memory relation of an application or geometry that no Lean
  package fixes, the binding rows of the Rust catalog backend, the rows of
  `nebula::R1csApplication` that link the rows of a batched step, and the
  WASM relation itself (`SPEC.md` sections 7.6 to 7.8). Lean proves what the slot plan conditions give
  (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder`), that the final catalog
  state then agrees with the final memory (`Claims.Nebula.catalog_agrees`),
  and that batched steps are memory consistent exactly when their rows are
  (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff`), but not
  that the Rust rows enforce those conditions. Tests check the rows on fixed
  inputs (trust item 5). Rust tests also check that each field of the
  typed WASM state changes its words (`every_state_field_changes_the_state_words`
  in `crates/neo-wasm/tests/wasm_nebula.rs`), and that a field of `q` or more
  keeps a word that the terminal check of `SPEC.md` Algorithm 8 refuses
  (`a_state_field_of_q_or_more_has_no_canonical_word`). So a verifier that
  reads a field of a proved state reads a proved word. `None` of
  `memory_pages` and `max_memory_pages` has the word 65,537, which no page
  count has (`memory_page_state_words` in `crates/neo-wasm/src/preprocess.rs`).
- The shared host-event transformation in a multirow WASM step. The adversary
  chooses every row flag, shared input, permutation auxiliary, and shared output.
  The group argument uses soundness and completeness of the exact width-12
  Poseidon2 event commitment with its four-lane feed-forward. It does not use
  a collision bound to replace a row check. The soundness argument allows
  multiple absorbing rows in a group and forces each to have the shared input.
  Completeness uses the checked Boolean decoding, position and kind sum rows,
  cursor transition, absorb equation, Enter exclusion, and cursor continuity.
  The kind count and eight slot positions must be below the field modulus.
  The argument needs no global natural range bound on arbitrary cursor field
  values.
  `Lifecycle.Nebula.SharedAbsorb.CursorRows` states these row premises. The
  Rust tests compare those rows and check column mutations on fixed inputs.
  A universal proof that the Rust transformation and shared-column map realize
  the Lean group relation remains open. The original one-row relation remains
  the source of all constraints outside the selected commitment input rewrite.
- A prover that chooses arbitrary private decomposition bits at internal batch
  boundaries. Continuity-range restoration keeps the payload and all input
  auxiliaries fixed. It requires equal-width input and output checks, retained
  input checks and continuity equalities, and output bits absent from every
  other constraint and observable. The Rust metadata validator checks these conditions
  against row uses, shared columns, continuity endpoints, and the memory catalog.
  The last output check stays in the batch. The proof does not require a valid prior execution.
  Lean proves restoration for the separated representation; universal refinement
  of the Rust metadata validator into that representation remains outside the proof.
- Memory-backed WASM range omissions. The adversary chooses every tagged
  column, including inactive advice. Rocq requires U32 values on every active
  read, write and declared prior-value event. The Rust Nebula bit rows and
  checked catalog bindings supply that premise; a host-only memory check does
  not supply circuit authority. Catalog membership checks only the existence
  of a value or prior-value binding. Each tag still requires a proof that its
  inactive uses admit an accepted bounded replacement preserving events and
  continuity. The row-only range theorem does not receive this premise.
  Exact Rust binding and cross-language event correspondence remain outside
  the formal bridge, as for the other catalog application conditions above.
- A prover that picks the WASM size of its proof
  (`neo_wasm::nebula::WasmSize`). The results hold for each of the four
  fixed sizes, not for a choice among them.
- Values for the adversary time and the query counts `Q_η`. They stay
  symbolic (`decisions/nebula-memory-checking.md`).
