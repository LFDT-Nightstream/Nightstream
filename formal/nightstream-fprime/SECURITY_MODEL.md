# Security model

This page states which adversaries the Nightstream F′ Lean results protect
against: what each adversary controls, when it wins, the Lean result that
bounds it, and what you must trust. It does not restate proofs; read the cited
declaration. `SPEC.md` describes the protocol: two step circuits, a light
circuit and a refresh circuit, that fold into two slots. Adversary 6, the
adversary of the compressed proof, has a Lean result for the argument only,
in a model with ideal Akita openings. No Lean result gives its aim. Section
Assumptions lists every assumption of the results, section Hashed
encodings lists every place that hashes or absorbs data, and section Premises
gives the class of each hypothesis of the top-level theorems.

Names omit the common `NightstreamFPrime` namespace.
`tests/EndpointCensus.lean` checks that every name on this page that starts
with `Export.`, `Layout.`, `Lifecycle.` or `Spec.` exists, and that every cited
theorem uses only `propext`, `Classical.choice` and `Quot.sound`. Shorter names
are not checked.

## What Lean checks

- The kernel of the official Lean v4.35.0 toolchain, with Mathlib v4.35.0 and
  the CompPoly revision that `lakefile.toml` pins.
- `scripts/validate.sh` runs on the developer Mac. Its `build`, `axioms`,
  `executables` and `all` phases use `lake build --wfail`, and single files
  run with `warningAsError`, so any warning fails a gate. `all` runs the
  boundary checks, both libraries, every `lean_exe` target and the census.
  CI runs no Lean (owner decision of 2026-09-26).
- Every audited theorem uses the three axioms above. The audit fails closed:
  `sorryAx` or a new axiom stops the build.
- One linter is off: `Circuit/Basic.lean` turns off `linter.dupNamespace`,
  because the `Circuit` monad has the name of its namespace. It is a naming
  lint, not a check of any proof.
- Strong-set low-norm invertibility (SuperNeo v1.2 Theorem 4) is proved
  (`Spec.RingStrongSet.lowNormInvertibility`), so no security result takes it
  as a premise.

## Assumptions

The results use only the assumptions of this list, and each is on a
primitive. A random oracle on a derived object (the Poseidon2 hash, a
transcript, `η`, or a challenge) is not an assumption. Such a model is
`[P2-PERM]` and `[SPONGE]`, with the conditions of `[SPONGE]` as Lean
hypotheses, or it is a Lean theorem. Section Hashed encodings gives the status
of each condition. `scripts/check_assumptions.py`, in the `static` phase of
`scripts/validate.sh`, fails when an entry of this list is not on the owner's
list of 2026-11-13, when another part of this page or `SPEC.md` states an
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
  The other two are Lean hypotheses of each use. Then two
  distinct inputs give one 4-lane digest with probability at most `N²/q⁴` plus
  `spongeError N`, because they absorb distinct block lists
  (`Spec.Sponge.collision_blocks`). Two uses whose words start with tags that
  differ at a position of both never absorb one block list
  (`Spec.Sponge.blocks_ne_of_differ`).
- `KECCAK`: SHAKE128, SHAKE256, SHA3-256 and the Keccak-f duplex of Akita in
  the classical random-oracle model: the Ajtai setup reads SHAKE128 output
  as uniform 64-bit words (trust item 1), and Akita reads its challenges as
  uniform (trust item 7). The owner accepted this hash family for Akita on
  2026-11-13 (`decisions/akita-compression.md`).
- `MSIS`: Module-SIS for a uniform matrix of `κ = 18` rows over
  `F_q[X]/(X^64 + 1)` at the chain norm, in the lattice-estimator model
  of `decisions/field-2p64-59-ring-x64.md` (trust item 2), and for the
  Akita commitments at Akita's price (trust item 7).
- `AKITA`: the knowledge soundness of Akita (ePrint 2026/1983) with the error
  `εAkita` (`Export.BoundedDepth.FinalVerifier.AkitaKnowledge`).
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
| Hash padding | `Spec.Poseidon2.blocks` (`10*`) | `Spec.Poseidon2.blocks_injective`; `Spec.Sponge.padding` meets the conditions of `[SPONGE]` | Direct: Lean hash vectors (`artifacts/nightstream-fprime-poseidon2-vectors-v1.json`, `crates/neo-ccs/tests/poseidon2_lean_vectors.rs`) |
| State hash | `Lifecycle.BoundedDepth.statePreimage` | `Lifecycle.BoundedDepth.statePreimage_injective`, `Lifecycle.BoundedDepth.stateHash_collision` (well-formed states) | Indirect: the rows of the emitted packages (`crates/nightstream/tests/bounded_depth_base_steps.rs`) |
| Slot digest | `Lifecycle.BoundedDepth.serializeSlot` | `Lifecycle.BoundedDepth.serializeSlot_injective` (shapes fixed by the type, depth below `q`) | Indirect, as above |
| Fresh public input | `Lifecycle.encHash` | `Lifecycle.encHash_injective_fixed` | Indirect, as above |
| Package identity | `Export.BoundedDepth.StepPackage.identity` | `Export.BoundedDepth.IdentityBinding.valuePreimage_injective`, `Export.BoundedDepth.IdentityBinding.identity_binds` (atoms and lengths below `2^64`) | Direct: `identity_code_matches_lean_on_the_probe_packages` |
| Circuit key digest | `Lifecycle.BoundedDepth.CircuitKeyDigest.input` | `Lifecycle.BoundedDepth.CircuitKeyDigest.input_injective`, `Lifecycle.BoundedDepth.CircuitKeyDigest.digest_binds` (an identity of any length) | Direct: `circuit_key_is_the_lean_digest` on three identities (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/nightstream/tests/bounded_depth_lifecycle/encodings.rs`) |
| Fold transcripts | `Lifecycle.RandomOracle.Schedule.challengeCalls`, `Lifecycle.RandomOracle.Schedule.pieces` | `Lifecycle.RandomOracle.Schedule.challengeAddress_injective`, `Lifecycle.RandomOracle.Schedule.statementCalls_identify`, `Lifecycle.RandomOracle.Schedule.proverCalls_identify`; fixed lengths: `Lifecycle.RandomOracle.Schedule.pieces_alike`, `Lifecycle.RandomOracle.Schedule.calls_pieces`; disjoint lanes: `Export.BoundedDepth.FoldSponge.rho_ne` | Direct: `the_light_transcript_absorbs_the_lean_pieces`, `the_refresh_transcript_absorbs_the_lean_pieces` (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/nightstream-fprime/tests/fold_transcript_lean.rs`). The Π_RLC sampler is direct (`crates/neo-reductions/tests/pi_rlc_lean_parity.rs`). Indirect for the fold prover: package rows |
| Compression transcript | `Export.BoundedDepth.CompressionSponge.pieces`, with `Lifecycle.RandomOracle.Sponge.byteWords` for a byte string and `Export.BoundedDepth.CompressionSponge.Claim.words` for a claim | `Export.BoundedDepth.CompressionSponge.calls_injective` (two runs of one statement shape, each byte string below `q` bytes), from `Export.BoundedDepth.CompressionSponge.pieces_alike`, `Lifecycle.RandomOracle.Sponge.byteWords_injective` and `Export.BoundedDepth.CompressionSponge.Claim.words_injective`. Rows 4 to 17 have the operations and counts of `Spec.Compression.Rust.schedule`: `Export.BoundedDepth.CompressionSponge.argumentPieces_fit` | Direct for the words: `byte_words_are_the_lean_words`, `claim_words_are_the_lean_words`, `the_domain_and_the_session_label_are_the_lean_ones`, `the_compression_domain_is_the_lean_domain`, and for the zero blocks of the reads: `a_replay_of_the_lean_pieces_ends_in_the_lean_session` (`artifacts/nightstream-fprime-encoding-parity-v1.json`, `crates/ns-compress/tests/unit/transcript.rs`). Direct for the order and the counts of rows 4 to 17 on a real proof: `a_replay_in_the_lean_order_draws_the_challenges_of_verify`. Open: no test compares rows 1 to 3 of a real proof with the Lean pieces. Rust makes them with the functions of a fold transcript (section Absorbs without a frame) |
| Akita session bytes | `Export.BoundedDepth.CompressionSponge.session` | `Lifecycle.RandomOracle.Sponge.sessionBytes_injective`; `Export.BoundedDepth.CompressionSponge.session_binds`: two runs with one session are equal, or their transcripts are two distinct lists of sponge calls with the same lanes 0 to 3 | Direct: `the_session_is_the_lean_session`, `a_replay_of_the_lean_pieces_ends_in_the_lean_session` |
| Akita transcript | None | Open: the Akita crates fix these bytes. `akita.verify` (`crates/ns-compress/src/akita/verifier.rs`) gives Akita the session bytes, the three commitments, the three points and the three values, and Akita absorbs them into its own Keccak sponge and makes its challenges there. Lean has no model of that sponge or of Akita's encodings. The results trust that Akita binds these inputs, as part of `[AKITA]` under `[KECCAK]` | None |
| Nebula record chains | `Lifecycle.Nebula.blocks` | `Lifecycle.Nebula.blocks_injective`, `Lifecycle.Nebula.blockChunks_injective` | Direct: fixtures (`crates/nightstream/tests/nebula/memory.rs`) |
| Nebula `η` | `Lifecycle.Nebula.etaInputBlocks` | `Lifecycle.Nebula.etaInputBlocks_injective`, `Lifecycle.Nebula.EtaOracle.pointOf_injective`; the four reads: `Export.BoundedDepth.EtaSponge.reads_injective` | Indirect: package rows |
| Nebula state words | `Lifecycle.Nebula.carryVector` | `Lifecycle.Nebula.carryVector_injective`, `Lifecycle.Nebula.stateWords_eq_or_collision` | Direct: terminal fixtures |
| Application identity | None (`BoundedDepthPackages::with_application`, `crates/nightstream-fprime/src/bounded_depth/application.rs`) | Open: Rust only. The hash input is one word for each byte of the tag, the Lean package identity, then the witness word count, the rows, the recipe rows, the shape and both package layouts (`SPEC.md` section 4.4). The results trust two facts that no theorem states. First, the word stream fixes these values: no theorem states that each list of the stream has its count before it or a length that the format fixes. Second, each word is below `q`, so that `Fq::from_u64` reduces no word: no theorem states it. A Lean definition of this stream needs Lean models of the Rust records, the native shape and the realized layouts. A Rust application has no Lean model (Out of scope) | None |
| Domain separation | `Export.DomainSeparation.uses` | `Export.DomainSeparation.separated`: two block lists whose words start with the tags of two distinct uses are distinct. A use owns its blocks: `Export.DomainSeparation.hash_owns`, `Export.DomainSeparation.circuitKey_owns`, `Export.DomainSeparation.fold_owns`, `Export.DomainSeparation.nebula_owns`. Open: inputs that an application hashes, which can start with any words. | Indirect for the tags that Lean defines. Direct for the circuit key tag and the two compression domains (rows Circuit key digest and Compression transcript). The compression transcript is one sponge that starts with the domain of its Π_CCS run (`Export.DomainSeparation.compression_owns`) and absorbs the domain of the argument after that run. None for the tag that Lean restates from Rust (application identity) |

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

Each row is a hypothesis of a top-level security theorem. Lean writes the
hypotheses of the theorems of `tests/PremiseCensus.lean` to
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

A name that ends in `.*` covers each field under it. The head is the constant
at the head of the hypothesis, and `*` is each head.

<!-- premises:begin -->
| Premise | Head | Class | Evidence |
| --- | --- | --- | --- |
| `publicFits` | `LE.le` | lean | `Export.BoundedDepth.StepRelation.keyPublicFits`, at the key width `2^26` |
| `cubeFits`, `relation.cubeFits`, `key.relation.cubeFits` | `LE.le` | lean | `Export.BoundedDepth.StepRelation.keyCubeFits`, at the key width `2^26` |
| `program.*`, `exports.program.*` | `*` | lean | `Lifecycle.Stage1.Poseidon2HashChainV1.program` and `Export.Stage1.NebulaMemoryV1Package.application` are programs, so they carry these proofs. A Rust application has no such instance (Out of scope) |
| `stepEq`, `validControl`, `encoded`, `rowsZero`, `others` | `*` | lean | `Export.BoundedDepth.TerminalBridge.bridge` joins the two step results and takes none of these |
| `fits.*`, `layout.*` | `*` | lean | `Export.BoundedDepth.TerminalBridge.bridge` takes them as fields of its `exports` (rows below) |
| `programEq` | `Eq` | game | adversary 5: the memory application is the exported program |
| `exports.programStep` | `Eq` | game | `SPEC.md` Algorithm 3: the step of the relation is the step of the application |
| `exports.lightKey`, `exports.refreshKey` | `Eq` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json`, `relation_rows_match_the_lean_checksum`: the rows that Rust expands from each package have the checksum of Lean's rows |
| `exports.lightFits.*`, `exports.refreshFits.*`, `exports.lightLayout.*`, `exports.refreshLayout.*`, `exports.lightInside`, `exports.refreshInside` | `*` | open | Open: Lean has no instance of `Export.BoundedDepth.TerminalBridge.Exports` for the emitted packages, so no theorem states that their rows fit their width or that their slot layout has the public form |
| `accepted` | `Lifecycle.BoundedDepth.Terminal.HoldsFor` | game | `SPEC.md` Algorithm 6 |
| `keysShaped` | `Eq` | lean | `Lifecycle.BoundedDepth.CircuitKeyDigest.keys_shaped`, for the keys `Lifecycle.BoundedDepth.CircuitKeyDigest.keys` of a package identity. Rust `circuit_key` computes these digests (section Hashed encodings, row Circuit key digest) |
| `assumption` | `Spec.HyperNova.ReverseStages.Assumption1` | assumption | `[HN-A1]` |
| `closed.*`, `admitted` | `*` | assumption | `[HN-A1]`, in the form of trust item 3: the closure premise and the admitted adversary |
| `premise` | `Spec.Sponge.Transfer` | assumption | `[SPONGE]` under `[P2-PERM]` |
| `akita` | `*` | assumption | `[AKITA]` |
| `valid.toValid.*`, `secure` | `*` | link | `formal/nightstream-fprime/artifacts/nightstream-fprime-nebula-gate-parity-v1.json`, `is_valid_and_is_secure_are_the_lean_decisions`: Rust checks rules 1 to 6 of `SPEC.md` section 7.1 on each geometry at run time, and the test compares that check with Lean's decisions. For the fixed WASM sizes Lean proves both (`Spec.Nebula.WasmSize.valid`, `Spec.Nebula.WasmSize.secure`), and `each_wasm_size_has_the_lean_geometry_and_rows` compares the geometries |
| `valid.romWords`, `valid.ramWords` | `LT.lt` | open | Open: no Lean theorem states that each word of an emitted program image is below `2^32` |
| `app.*` | `*` | open | Open: `Lifecycle.Nebula.FirstPlan.app` is the test machine. A Rust-built memory application, the WASM relation included, has no instance (Out of scope) |
| `round` | `Export.BoundedDepth.NebulaSecurity.MemoryRound` | open | Open: trust item 6. No result derives the memory round from the assumption list |
| `sampler.*` | `*` | game | adversary 5: the `η` sampler and its query bound |
| `wf.*`, `slots_le`, `key_fits` | `*` | lean | `Spec.Compression.Rust.layoutNew_production`, for the layout of the Lean model of `Layout::new` |
| `counts` | `Spec.Compression.Verifier.CountsNonzero` | lean | `Spec.Compression.Production.params_countsNonzero` |
| `matrix` | `Spec.Compression.Verifier.MatrixSpec` | lean | `Spec.Compression.Production.verifierData_matrixSpec`, for the Lean model of Rust `matrix_part` |
| `columns`, `rows` | `LE.le` | lean | `Spec.Compression.Production.params_soundness` |
| `statement.*`, `constantLaw.*` | `*` | game | adversary 6: the statement of the Π_CCS run and the law of its constant |
| `params.rlc_bound`, `freshBound`, `degreeCovers` | `*` | lean | `Export.BoundedDepth.FinalVerifier.soundness` uses the Π_CCS run at the production parameters and takes none of these |
| `premises.layout` | `And` | open | Open: `layout_new_is_lean_layout_new_on_each_claim_set` compares Rust `Layout::new` with the Lean model on recorded claim sets, and `Spec.Compression.Rust.layoutNew_production` gives the bounds of the model. No theorem gives `spanVars ≤ 27`, and no theorem joins the layout of the setup to the model |
| `premises.represents` | `Spec.Compression.Claim.Represents` | open | Open: no Lean result states `Represents` for the dense rows of a package (trust item 7) |
| `premises.support` | `LE.le` | open | Open: no Lean result bounds the blocks of a claim by the blocks of its package (trust item 7) |
| `depthBound` | `LE.le` | open | Open: no result bounds the iteration of an accepted statement. The bound is a parameter of the result |
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
(`Export.BoundedDepth.IdentityBinding.identity_binds`), when each atom and each
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
  (`Export.BoundedDepth.StepRelation.light_stepHolds`) and for the refresh circuit
  (`Export.BoundedDepth.StepRelation.refresh_stepHolds`). The step holds in an
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
  preimage collide (`Export.BoundedDepth.TerminalBridge.bridge`,
  `Export.BoundedDepth.TerminalBridge.Collision`,
  `Export.BoundedDepth.TerminalBridge.WordCollision`). A light step's other slot
  is the terminal's slot, because the light circuit reads only the active
  slot's words (`Export.BoundedDepth.StepRelation.lightInput`). The terminal is
  `Lifecycle.BoundedDepth.Terminal.HoldsFor`. Two well-formed states with one state
  hash are equal or give a Poseidon2 collision on an input pair that the two
  states fix (`Lifecycle.BoundedDepth.stateHash_collision`,
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
- **Wins:** the fold verifier accepts two outputs that differ only in the
  Π_RLC challenges (a fork), and extraction from the fork does not give the
  witnesses.
- **Result:** for a light fold, the extracted fresh witness opens the fresh
  claim with norm below `b`, the extracted accumulator witness opens the
  input accumulator at the output bound plus `T(b − 1)`, and the claims are
  true at the prior point, or a Π_CCS bad event occurred: a mixing root of
  `α` and `γ`, or a sum-check failure
  (`Spec.Folding.Light.accepted_implies_securityOutcome`; Cyclo, ePrint
  2026/359, Theorem 3, item 2; `Lifecycle.BoundedDepth.lightExtractionNormLaw`).
  For a refresh fold, the extracted fresh witnesses satisfy the CCS and the
  recomposed children open the input accumulator below `B`, or a Π_CCS bad
  event occurred (`Spec.Folding.Refresh.accepted_implies_securityOutcome`).
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
- **Random-oracle model:** in a model where each fold challenge is a uniform
  value at the address of its sponge calls, for an adversary that makes at
  most `Q` queries, the probability that a fold succeeds and the extractor does not return its
  witnesses is at most a statistical term, a state-collision term and a
  kernel term (`Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.knowledge_error_le`,
  `Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.knowledge_error_le`). The
  test step charges each challenge's bad set at the challenge's own point, so
  it costs at most `(Q + 1)` times the Π_CCS test error
  (`Lifecycle.RandomOracle.Test.family_error_le`). The statistical term is at
  most `(Q + 1) · 575/2^128` for light and `(Q + 1) · 4748/2^128` for refresh
  (`Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.statisticalError_le`,
  `Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.statisticalError_le`).
  The fold gate of Rust `Params::for_ccs_shape` charges one error for both
  folds and computes its bits in integers. The statistical term of either
  fold is at most `(Q + 1)/2^bits`, and `bits` is 115 at the step relations
  (`Lifecycle.BoundedDepth.RandomOracle.FoldGate.refresh_statisticalError_le_bits`,
  `Lifecycle.BoundedDepth.RandomOracle.FoldGate.light_statisticalError_le_bits`,
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
  Under `[P2-PERM]` and `[SPONGE]`, two distinct inputs give one digest with
  probability at most `N²/q⁴ + spongeError N`, because the `10*` padding of
  the sponge is injective (`Spec.Poseidon2.blocks_injective`,
  `Spec.Sponge.collision_blocks`).
  Averaged over the SHAKE128 key setup at key width `2^26`, the kernel term is
  an MSIS advantage for a uniform matrix at the chain norm plus a programming
  error below `2^−1129`, so each fold's bound uses only the two algorithms
  (`Export.BoundedDepth.RandomOracleSetup.light_production_knowledge_error_lt_solvers`,
  `Export.BoundedDepth.RandomOracleSetup.refresh_production_knowledge_error_lt_solvers`).
  Lean bounds neither the MSIS advantage nor the collision advantage
  numerically (trust item 2), and it states the machines' work in adversary
  runs, not in time.
- **The sponge:** the model's address of a challenge is the list of sponge
  calls before its read (`Lifecycle.RandomOracle.Schedule.challengeCalls`).
  Distinct challenges have distinct addresses
  (`Lifecycle.RandomOracle.Schedule.challengeAddress_injective`), equal calls
  fix the absorbed data (`Lifecycle.RandomOracle.Schedule.proverCalls_identify`),
  and the light and refresh transcripts share no call prefix
  (`Lifecycle.RandomOracle.Schedule.challengeCalls_separated`). These are
  conditions of `[SPONGE]`. Under `[P2-PERM]` and `[SPONGE]`, the bound of
  the model holds for a verifier that reads the sponge outside a circuit, plus
  `spongeError N`, when every condition holds. The fold calls start with
  their domain (`Export.DomainSeparation.fold_owns`), and the fold domains
  are separated from the other uses (`Export.DomainSeparation.separated`).
  The model's answers at distinct points are independent, but a `Π_RLC`
  draw reads lanes 0 to 4 and a coin of slot `s` reads lanes `2s` and
  `2s + 1`. So a draw must never share its calls with a coin of slot 1 or 2:
  it does not, because its calls are longer
  (`Export.BoundedDepth.FoldSponge.rho_ne`). With these conditions proved
  for every fold schedule (`Export.BoundedDepth.FoldSponge.conditions`), the
  knowledge bound moves to the ideal-permutation model as the statistical
  error plus `spongeError N`, with the collision and kernel terms as before
  (`Export.BoundedDepth.FoldSponge.light_knowledge_le`,
  `Export.BoundedDepth.FoldSponge.refresh_knowledge_le`).
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
  `Pr[false acceptance] ≤ Σ_j (Pr[collision at visit j] + error(stage j))`
  for `j` below a bound on the adversary's iteration
  (`Export.BoundedDepth.IvcSecurity.probability_bound`). The reverse extractor
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
- **Collision term:** the collision at a visit is a Poseidon2 collision on
  the next state of the decoded step and the visit's terminal state, or on
  a light step's next state words and the terminal's 38-word state preimage
  (`Export.BoundedDepth.TerminalBridge.Collision`). The stage computes both inputs
  from its tape, so a reduction that runs the stage outputs the colliding
  pair. Lean does not bound this probability (trust item 2).
- **Per slot:** the earlier one-slot result,
  `Spec.HyperNova.Construction2.BoundedDepthSecurity.history_failure_le`,
  takes the history as an adversary output and has no collision term. Without
  probabilities, the reverse run of one slot opens each running claim at its
  recovered bound, or it names a failed fold
  (`Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts`).
- **Validity:** a false acceptance has no list of witnesses that are valid
  for the application and that its step maps from `z0` to `zi`
  (`Export.BoundedDepth.IvcWalk.Reaches`). The bridge's selector returns the
  circuit that ran only on a valid step (`Export.BoundedDepth.TerminalBridge.chosen`),
  so each decoded step is valid (`Export.BoundedDepth.TerminalBridge.valid_of_chosen`).
- **The verifier:** the terminal verifier of `SPEC.md` Algorithm 6 also
  holds each slot to the bound of its depth
  (`Export.BoundedDepth.RunCompleteness.Verifies`). The same sum bounds the
  probability that this verifier accepts and no such steps lead from `z0` to
  `zi` (`Export.BoundedDepth.VerifierSecurity.soundness`). `SPEC.md`
  section 9 gives the theorem of each verifier entry point.
- **Knowledge form:** HyperNova Definition 11 (ii) states knowledge
  soundness: its event is on the witnesses that the extractor returns. The
  same sum bounds the probability of an extraction failure: the terminal
  relation accepts, and the witnesses of the walk's trail are not `iteration`
  valid steps from `z0` to `zi`
  (`Export.BoundedDepth.IvcSecurity.ExtractionFailure`,
  `Export.BoundedDepth.IvcSecurity.extraction_bound`). Its law is the adversary's
  tape and the coins of each stage's extractor. A false acceptance is an
  extraction failure, so the result above follows from this one.
- **Gaps:** Lean derives Assumption 1 with an explicit `error` only in a
  random-oracle model of the folds
  (`Lifecycle.BoundedDepth.RandomOracle.Assumption1.romAssumption1`). The bound
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
  collision and stage terms of adversary 4, the state-link and chain
  collision masses of the extracted runs, and the memory-round error of the
  last stage (`Export.BoundedDepth.NebulaSecurity.bad_le`). The extracted history
  is the steps that adversary 4's walk decodes (`Export.BoundedDepth.IvcWalk.trail`),
  so every event reads data of the sample. The walk meets the extraction
  premise of `Lifecycle.Nebula.EndToEnd.Stage1Extraction`
  (`Export.BoundedDepth.NebulaSecurity.stage1Extraction`), and
  `Lifecycle.Nebula.EndToEnd.bad_subset` reduces a win to a failure of that
  extraction, a state-link collision, a chain collision, or an accepted run
  with an unbalanced segment that passes the product test. A win counts only
  on a sample whose answers cover the statement's iteration; on the law of
  the theorem this holds when the adversary's iteration is at most the
  number of stages. For a compressed memory proof, adversary 6 joins this
  bound with the bound of the final proof
  (`Export.BoundedDepth.NebulaFinal.real_soundness`).
- **Deterministic part:** an accepted run gives an execution, a Poseidon2
  collision among the run's own chain inputs and the `D_init` chain, or a
  segment that passes the product test with unbalanced multisets
  (`Spec.Nebula.soundness`). With the state links, an accepted chain of valid
  steps gives such a run or a state-digest collision
  (`Lifecycle.Nebula.accepts_or_collision`).
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
  `Spec.Nebula.WasmSize.segmentBits_succ_lt`). Lean emits these numbers, and
  Rust tests compare them (trust item 5).
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
  (`Export.BoundedDepth.NebulaSecurity.romMemoryRound`). The model takes the
  adversary and the extractor as one sampler, so `Q_η` counts the queries of
  every rerun. At the Poseidon2 oracle the event is the event of the memory
  round (`Lifecycle.Nebula.EtaRound.unbalancedIn_deployed`), but no result
  bounds it there. The model's point is a function of the `η` input that is
  injective on canonical timestamps
  (`Lifecycle.Nebula.EtaOracle.pointOf_injective`), and the absorbed blocks
  are injective (`Lifecycle.Nebula.etaInputBlocks_injective`,
  `Lifecycle.Nebula.blockChunks_injective`). These are conditions of
  `[SPONGE]`, and the `η` domain is separated from the other uses
  (`Export.DomainSeparation.separated`). With these conditions proved
  (`Export.BoundedDepth.EtaSponge.conditions`), `[P2-PERM]` and `[SPONGE]`
  move the bound, without the retry term, to the ideal-permutation model
  plus `spongeError N` (`Export.BoundedDepth.EtaSponge.unbalanced_le`). That
  bound is for a verifier that computes `η` outside a circuit. The step
  circuit computes `η` with the permutation (trust item 6).

### 6. Final-proof adversary

- **Controls:** the statement and each byte of a final proof (`SPEC.md`
  section 4.5), against the verifier's own circuits, security minimum and
  compression key. It can evaluate the Poseidon2 sponge and the Keccak
  functions of Akita. It does not choose the compression key. The verifier
  takes the pinned key, or loads bytes that must equal it (`SPEC.md`
  Algorithm 11).
- **Wins:** the final verifier accepts (`SPEC.md` Algorithm 10), but no
  proof with witnesses exists that the verifier of `SPEC.md` Algorithm 6
  accepts for the statement
  (`Export.BoundedDepth.RunCompleteness.Verifies`).
- **Result:** for the argument only. Lean models the verifier of the
  argument as a public-coin game (`Spec.Compression.Verifier.game`). In the
  state-restoration experiment, a prover with `Q` oracle queries makes the
  verifier accept claims whose committed `W` does not satisfy
  `Spec.Compression.Verifier.Relation` with probability at most
  `Q · error` (`Spec.Compression.Verifier.soundness`). `Relation` gives,
  for each claim, the 18 Ajtai rows of its commitment, `Eval_K`, each
  `Eval_A` relative to the abstract matrix weights, its public input and,
  for a claim with a table, each cell in `[−bound, bound]`
  (`Spec.Compression.Verifier.game_avoids`). At the production shapes,
  `error < 2^−119` (`Spec.Compression.Verifier.production_error_lt`). The
  premises of the theorem, and how Rust meets each one, are in trust item 7.
  Lean has no definition of the compression key. Its model of the layout
  (`Spec.Compression.Rust.layoutNew`) meets the layout premises of the
  theorem (`Spec.Compression.Rust.layoutNew_production`), and Rust tests
  compare it with `Layout::new`. For each slot claim, `Relation` gives the
  claim under `Spec.CE.Holds` with the norm of its depth
  (`Lifecycle.BoundedDepth.CompressionSlot.slot_holds`), under the premise
  `Spec.Compression.Claim.Represents` of trust item 7. Lean models the
  Π_CCS run with one source as a game of the same kind
  (`Spec.Compression.Fresh.game`). On a path that avoids its bad sets, of
  density at most `38/q²`, an accepted run whose output claims hold gives
  the fresh instance under `Spec.CCS.Holds`
  (`Lifecycle.BoundedDepth.CompressionFresh.game_avoids`). Lean joins the
  checks, Π_CCS and the argument in one game of the final verifier
  (`Export.BoundedDepth.FinalVerifier.Setup.game`). With `Q` queries, it accepts
  a statement for which no proof with witnesses satisfies `Verifies` with
  probability at most `Q · (2^−119 + 38/q²)`
  (`Export.BoundedDepth.FinalVerifier.soundness`), and with Akita openings
  at most `Q · (2^−119 + 38/q²) + εAkita`
  (`Export.BoundedDepth.FinalVerifier.real_soundness`), under the
  premises of trust item 7. Lean bounds one interactive run of the
  argument by the exact Rust error sum
  (`Spec.Compression.Rust.params_run_soundness`). Only tests join the model
  to the Rust layout and error sum, and no result states a full honest
  run. Rust implements the final
  verifier, and the tests check it on fixed inputs (`SPEC.md` lifecycle
  rules 2 and 3, trust item 5). `decisions/akita-compression.md` gives the
  reason for each step.
- **Aim:** under the premises of trust item 7, an accepted final proof
  gives three witnesses. The one Akita proof gives the polynomials `W` and
  `V`, and the value of the key polynomial at one point. The key polynomial
  is the Ajtai key, so the key sum-check makes the proof's key part of the
  public weights equal to the part that the Ajtai key gives. The verifier
  reads no element of the Ajtai key. The first sum-check and the last check
  of `SPEC.md` Algorithm 10 give, for each of the three claims, a witness in
  `W` that opens the claim's commitment, has its public input and gives its
  evaluations (`Spec.CE.Holds` without the norm). The lookup gives each slot witness
  the bound of its depth. The Π_CCS run gives the fresh witness the rows of
  its relation and coefficients in `{−1, 0, 1}` (`Spec.CCS.Holds`). The
  verifier computes the state hash in the fresh public input from the
  proof's slot claims, so no digest that the proof carries is authority.
  These are the checks of `SPEC.md` Algorithm 6, with one difference. The
  argument does not prove that the fresh witness is zero from the
  relation's width to the end of its last ring column.
  `Lifecycle.BoundedDepth.BlockSupport.Supported` does not require these
  zeros.
- **With the other adversaries:** the verifier of Algorithm 6 accepts only
  what the terminal relation accepts
  (`Export.BoundedDepth.RunCompleteness.holds_of_verifies`). So the results
  of adversaries 2, 4 and 5 apply to a statement that the final verifier
  accepts. Lean joins the bound above with the bound of adversary 4. A
  prover of the experiment with `Q` queries is an IVC adversary: its tape
  is the result of the experiment under the law of its draws, and its
  output is the accepted statement with the proof from `W`
  (`Export.BoundedDepth.FinalHistory.adversary`,
  `Export.BoundedDepth.FinalVerifier.Setup.proofOf`). The event is a false
  history: the final verifier accepts `(i, z0, zi)`, and no `i` steps of
  the application, each valid at its state and with a witness of the
  application's word count, lead from `z0` to `zi`
  (`Export.BoundedDepth.FinalHistory.FalseHistory`). A false history is a
  result in which `Verifies` rejects that proof, or a false acceptance of
  this IVC adversary. So
  `Pr[false history] ≤ Q · (2^−119 + 38/q²) + Σ_j (Pr[collision at visit j] + error(stage j))`
  for the stages of this IVC adversary
  (`Export.BoundedDepth.FinalHistory.soundness`). With Akita openings, the
  bound adds `εAkita` (`Export.BoundedDepth.FinalHistory.real_soundness`).
  The premises are those of trust item 7, `[HN-A1]` and the closure
  premise of trust item 3, an admitted IVC adversary of the prover, the
  digest length 4 of each circuit key, and a bound `depth` on the
  iteration of each statement that the experiment accepts. The sum has the
  limits of adversary 4, so it gives no concrete security for a long
  history. Lean also joins the bound above with the bound of adversary 5,
  for the memory verifier of a final proof (Rust
  `nebula::Verifier::verify_final`, `SPEC.md` Algorithms 8 and 10). The
  prover names a memory statement with each result of the experiment, with
  any law for each result. With that statement on its tape it is an IVC
  adversary with the output above
  (`Export.BoundedDepth.NebulaFinal.adversary`). The event is a false
  memory statement: the final verifier accepts `(i, z0, zi)`, the checks of
  `SPEC.md` Algorithm 8 hold for `(i, z0, zi)` and the memory statement, and
  the history that the walk of adversary 4 decodes does not attest the
  memory statement (`Export.BoundedDepth.NebulaFinal.FalseMemory`). The law
  is that of the walk's last stage: the result, the memory statement and
  the coins of each stage's extractor. A false memory statement is a tape
  on which `Verifies` rejects the proof from `W`, or a win of adversary 5
  for this IVC adversary. So its probability is at most
  `Q · (2^−119 + 38/q²)` plus the terms of adversary 5
  (`Export.BoundedDepth.NebulaFinal.soundness`). With Akita openings, the
  bound adds `εAkita` (`Export.BoundedDepth.NebulaFinal.real_soundness`).
  The premises are those of the two joined results, so the memory round
  (trust item 6) stays a premise. No Lean result joins the bound of
  adversary 2 with the bound of this adversary. No result shows that an
  extractor of the three witnesses is efficient.
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

1. **The Ajtai setup [KECCAK] (premise P1).** Each key element reads one
   stream of 84 SHAKE128 output words of 64 bits. A word below `q` is kept, a
   word of `q` or more is rejected, and coefficient `h` is kept word `h`, or
   0 when fewer than `h + 1` words are kept (the fallback)
   (`Spec.AjtaiSetupV1.Programming.verifierKey_eq`). The experiment is one
   uniform stream. Each ring element is the element of the same number of
   streams that take no fallback (`Spec.AjtaiSetupV1.Rejection.card_element`),
   so the element is exactly uniform outside the fallback event. The event is
   that 21 or more of the 84 words are rejected, and its probability for one
   element is at most `fallbackBound = C(84, 21) · 59^21/2^1344`, below
   `2^−1155` (`Spec.AjtaiSetupV1.Rejection.card_fallback_le`,
   `Spec.AjtaiSetupV1.Rejection.fallbackBound_lt`). With uniform streams, an
   event has at most its probability under a uniform matrix, plus
   `2 · fallbackBound` for each key element
   (`Spec.AjtaiSetupV1.Programming.element_bias_le`,
   `Spec.AjtaiSetupV1.Programming.expect_le_programmed`). For the
   `18 · 2^20` elements of the bounded-depth key the sum is below `2^−1129`
   (`Export.BoundedDepth.RandomOracleSetup.programmingError_lt`). So MSIS for
   a uniform matrix applies (`decisions/fprime-ajtai-shake128-setup.md`,
   `decisions/fprime-ajtai-rejection-sampling.md`,
   `decisions/fprime-ajtai-shake128-security-argument.md`). The seed is
   fixed and public, so an adversary that is precomputed for that seed is
   outside this premise. Lean does not join this step to the results above.
2. **The binding and collision terms [MSIS] [P2-PERM] [SPONGE].**
   - *MSIS at the chain norm.* Binding needs MSIS for a uniform matrix of
     `κ = 18` rows of elements of `F_q[X]/(X^64 + 1)` (degree 64) at the chain norm
     `msisChainBound = 8 · T · (B + cap · T · (b − 1)) = 25,165,824`, about
     `2^24.58` (`Spec.Folding.Schedule.production_msisChainBound`). It is the
     SuperNeo v1.2 Section 8 norm `8TB` with `B` replaced by the largest
     recovered bound (`Spec.Folding.Schedule.msisChainBound_zero`). The
     terminal verifier checks each slot at the bound of its depth. By Cyclo
     Theorem 3, this bound is at most `accumulatorBound cap = 10,112`
     (`Spec.GlobalParams.accumulatorBound`, `SPEC.md` Algorithm 6), and Lean
     proves that an honest run stays within it
     (`Export.BoundedDepth.StepExtend.Honest`). A refresh fold recovers its input
     only below `B`, so the terminal relation of the security results keeps
     `B` (`Export.BoundedDepth.RunCompleteness.holds_of_verifies`). The recovered
     bound, and so the chain norm, comes from `B`, not from the bound of the
     depth.
   - *The κ estimate.* A lattice-estimator run outside Lean, at rank
     `κ = 18`, a width of `2^28` coefficients and the chain norm
     `8T(2^14 + 64T) = 25,165,824`, gives 127.20 bits in the ADPS16 quantum
     model (BKZ block size 480), 140.16 bits in the ADPS16 classical model and
     166.84 bits in the default model. The norm is the largest extraction
     norm. There is no allowance for more targets, because the main key is
     one SHAKE128 matrix and so it has one target. The owner approved this
     method (`decisions/field-2p64-59-ring-x64.md`). The sister project
     ProtogaLattice computed these values with lattice-estimator commit
     `3e48ef4` and Sage 10.9. Nightstream did not run them again, because
     Sage stops at start-up on this machine. The owner selected `κ = 18` on
     2026-11-13. The models are heuristic generic lattice attacks. They are
     not a proof for structured Module-SIS or for the fixed public seed, and
     the run's files are not in the repository.
   - *Poseidon2 collisions.* The state hash, the slot digests, the package
     identity, the Nebula record chains and the state digests of a memory
     application bind their inputs only up to a Poseidon2 collision. The
     results name each collision as an event on two inputs that a reduction
     computes. Under `[P2-PERM]` and `[SPONGE]`, an algorithm with `N`
     permutation calls finds two distinct padded inputs with one digest with
     probability at most `N²/q⁴ + spongeError N`. Two distinct inputs of the
     hash have distinct padded inputs (`Spec.Sponge.collision_blocks`), and
     two distinct values of a row of section Hashed encodings have distinct
     inputs only by that row's injectivity theorem. The deployed permutation
     is fixed, so `[P2-PERM]` is a heuristic for it; algorithms that use its
     structure are outside the model.
   - Lean bounds neither term numerically.
3. **The history: HyperNova errata Assumption 1 for the two fold kinds [HN-A1].**
   `Spec.HyperNova.ReverseStages.Assumption1` states Assumption 1 in the form
   of HyperNova Definition 7 knowledge soundness. Adversary 4 uses it with the
   success event `Export.BoundedDepth.IvcWalk.WalkSucceeds` and the extraction
   event `Export.BoundedDepth.IvcWalk.WalkExtracted`; the one-slot result uses
   `Spec.HyperNova.Construction2.BoundedDepthSecurity.Assumption1` with the
   same events.
   For every admitted fold adversary there is an efficient extractor that
   reads the adversary's tape and its own coins, so it may rerun the
   adversary. The probability that the adversary succeeds and the extractor
   fails is at most `error` of that adversary. The adversary outputs an attempt: a fold, the
   running claim that it starts from, and a witness of its output at a
   bound. One premise covers both fold kinds. The attempt names its circuit,
   and a success needs the bounded-depth verifier of that circuit to accept,
   so a light fold at the cap is not a success. A success also needs an
   output bound of at most `B + cap · T · (b − 1)`, so the premise is needed
   only up to the chain norm of trust item 2
   (`Spec.HyperNova.Construction2.BoundedDepthReverse.Succeeds`). Extraction
   must open the fold's input at the output bound plus `T(b − 1)` after a
   light fold, and below `B` after a refresh fold, and it must open the fresh
   claim (`Spec.HyperNova.Construction2.BoundedDepthReverse.Extracted`,
   `Spec.Folding.Schedule.extractedBefore`). These are the differences from
   Definition 7:
   - The success event has a prior-state link. The light transcript does
     not absorb the running claim (`Lifecycle.BoundedDepth.lightAbsorbPublicInput`),
     and the refresh transcript absorbs the children's commitments and
     evaluations but not the accumulator's point or public input
     (`Lifecycle.BoundedDepth.refreshWords`). Both bind the accumulator only
     through the prior digest in the fresh public input. So the adversary
     also outputs a prior state, and a success needs `PriorLink` of that
     state, the running claim and the fold
     (`Spec.HyperNova.Construction2.BoundedDepthReverse.RunsBack`). The
     prior state is an output, as in PR #155: a link to some prior state
     holds for almost every running claim, because many states share one
     digest. In an accepted execution, each fold of a slot links to the
     prior hash preimage of its step: the fresh public input encodes the hash
     of that preimage, and its program counter selects the slot that holds
     the fold's running claim
     (`Spec.HyperNova.Construction2.BoundedDepthSecurity.StepLink`,
     `Spec.HyperNova.Construction2.BoundedDepthSecurity.slot_runsBack`).
     The bounded-depth scheme's link also requires a well-formed prior state
     (`Lifecycle.BoundedDepth.PriorLink`). Two linked folds with one fresh instance
     then start from one running claim, or their two prior states give a
     state-hash collision (`Lifecycle.BoundedDepth.priorLink_binds`). So the
     transcript fixes the running claim before the challenges, up to a
     collision on a pair of states that the adversary outputs.
   - The relations have a key. A claim holds under the key of its slot, and
     every fold of an accepted history uses the slot's key.
   - The transcript is not the one of HyperNova Construction 3. The published
     construction hashes the public parameters and the structure into the
     first challenge and derives each later challenge from the prover's
     message. The errata's construction absorbs `sid` (the structure and the
     public parameters) and the instances before the first challenge. The
     fold transcripts absorb a domain block,
     the fresh statement, the children of a refresh fold and the prover
     messages (`SPEC.md` section 5.2). The structure, the keys and the
     accumulator enter only through the prior digest. So `Assumption1` is a
     premise about the Nightstream transcripts, not the paper's premise about
     Construction 3.
   - This joint form implies Definition 7's difference form
     `Pr[success] − Pr[extraction] ≤ error`. The converse needs an adversary
     that stops when its own success check fails.
   - The public parameters are a fixed commitment setup, not sampled by the
     generator. With a fixed hash and a fixed key, an efficient algorithm that
     contains a collision exists. So the premise can hold only for algorithms
     that a person can write without such knowledge.
   - The structure is a fixed application. Definition 7 lets the adversary
     choose the structure after the public parameters, and Definition 11 lets
     the IVC prover choose `F` from the public parameters and its coins. A
     choice that depends on the adversary's coins is not covered.
   - `error` replaces `negl(λ)`.

   `Spec.HyperNova.ReverseStages.Closed` is the composition premise. A second class, `StageAdmitted`, holds whole stages:
   the reverse extractor so far, as one algorithm. An admitted stage gives an
   admitted fold adversary, and one reverse step of an admitted stage with an
   efficient extractor is again an admitted stage. The IVC adversary is the
   start stage, which must be admitted. `Admitted`, `StageAdmitted` and
   `Efficient` are abstract; their intended meaning is expected polynomial
   time. Lean has no running-time model, so it states no false claim about
   running time, and the closure property is a premise. Lean chooses the
   stages with `Classical.choose`, so a numerical bound on `error` must hold
   for the whole class. Assumption 1 is the paper's plain-model premise, not
   a theorem. The step circuits recompute the challenges of the previous fold
   with Poseidon2, so a recursive argument uses the concrete hash, which no
   random-oracle model covers. In a random-oracle model of the folds,
   Assumption 1 holds with an explicit error: adversary 3's knowledge errors
   for each circuit key plus a truncation loss `2 · ((Q + 1) + 15 · (Q + 15))
   / budget` for an extractor with its own oracle coins and `budget` reruns
   (`Lifecycle.BoundedDepth.RandomOracle.Assumption1.romAssumption1`). At an
   oracle that answers like the Poseidon2 sponge, the model's success and
   extraction events are the walk's
   (`Export.BoundedDepth.RomWalk.succeeds_iff_walkSucceeds`,
   `Export.BoundedDepth.RomWalk.extracted_iff_walkExtracted`), and an honest fold
   meets the success event (`Lifecycle.BoundedDepth.RandomOracle.Honest.light_honest`,
   `Lifecycle.BoundedDepth.RandomOracle.Honest.refresh_honest`). One fixed oracle
   carries no probability, so this does not prove the premise for the
   concrete hash; it gives the value of `error` that the premise asks for.
   Lean has no random-oracle closure premise, because the query bound of a
   stage grows with its depth. The sum has
   one Assumption 1 error for each stage of the reverse extractor, which is
   one visit, and stage `j + 1` runs
   the extractor of stage `j`, which may rerun stage `j`. So the cost of a
   stage grows with its depth. This is the constant-depth limit of HyperNova
   Lemma 17: the Lean statement holds for every depth, but it gives no
   concrete security for a long history. The errata's Definition 11 states
   knowledge soundness only for a constant number of steps that does not
   depend on the security parameter, and its proof composes the stage
   extractors because that number is constant. A random-oracle bound for
   each fold does not remove this limit: each stage's adversary includes
   the earlier stages' reruns.
4. **Scope: the circuit selection.** HyperNova NIVC selects the circuit of a step
   with `φ(z_i, w_i)`. The Lean step relation takes it from the machine's
   selector of the application state and witness (`Lifecycle.BoundedDepth.machine`,
   `Lifecycle.BoundedDepth.stepHolds_iff`). Both bounded-depth circuits run the same
   application step, so the selector does not change the computed state.
   `Export.BoundedDepth.TerminalBridge.bridge` takes the selector that returns the
   circuit that ran on a valid step (`Export.BoundedDepth.TerminalBridge.chosen`),
   so the prover picks the circuit. Rust picks it from the depth of the selected slot
   (`SPEC.md` Algorithm 5): a light fold below the cap and a refresh fold at
   the cap. The light depth guard rejects a light fold at the cap for every
   pick. The statement `(i, z0, zi)` does not fix the circuit of a step; the
   state hash binds the program counter of the last step only.
5. **Gap: Rust and Lean agree on recorded inputs only.** The Rust tests compare
   Rust's expansion of the packages, its identity code, its witness program
   and its logical assignment with Lean exports, and they check the
   lifecycle on fixed inputs (`SPEC.md` section 9). No theorem covers
   arbitrary Rust execution.
6. **Gap: the memory round.** `Export.BoundedDepth.NebulaSecurity.MemoryRound`
   states the memory round for the two step circuits as a hypothesis: for
   each admitted stage and the memory statement of its tape, the extracted run
   is accepted and has a segment that passes the product test with unbalanced
   multisets with probability at most `error_mem` of that stage. It is not an
   assumption of the list, and no result derives it from the list: the
   relation computes `η` with the Poseidon2 permutation inside the proven
   step, and no ideal-permutation model covers that check. The random-oracle
   model of `η` in adversary 5 gives the form of `error_mem`:
   `(Q_η + 1) · 2 · m_mem/q²` plus the retry term, where `Q_η` counts the
   evaluations of the `η` transcript, in the circuit and outside it. Lean does
   not bound the retry term with a number, because each pair that it counts
   is a Poseidon2 collision. The owner chose this form
   (`decisions/nebula-memory-checking.md`). The model gives the `η` oracle no
   relation to the fold addresses (trust item 3). Their domain tags differ,
   and two block lists with these tags are distinct
   (`Export.DomainSeparation.separated`). The record
   chains need only the collision bound of trust item 2.
7. **The compressed proof [AKITA] [KECCAK] [MSIS] [P2-PERM] [SPONGE].**
   Adversary 6 has these premises.
   - *The premises of the Lean result.* `Spec.Compression.Verifier.soundness`
     takes these hypotheses.
     `Spec.Compression.Verifier.WellFormed` (each slice is aligned and
     inside the cube of `W`), the bound `2^25` on the cells of a witness
     slice and the bound of the key on the blocks of a claim hold for the
     layout of the Lean model of `Layout::new`
     (`Spec.Compression.Rust.layoutNew_production`), when each claim has at
     least its 320 public cells, which the shape check of Rust `layout`
     makes. `Spec.Compression.Rust.layout_soundness` states the bound
     without these premises. The premise is that Rust `Layout::new` is the
     model: tests compare the two on each claim set of the reference
     application and on generated claim sets, and no proof does.
     `Spec.Compression.Verifier.CountsNonzero` (`2^vars` is below the
     characteristic of `F_q`) follows from the bound `2^25`
     (`Spec.Compression.Production.params_countsNonzero`). `Spec.Compression.Verifier.MatrixSpec`
     (the verifier's matrix part is the multilinear extension of the matrix
     weights) is a theorem for the Lean model of Rust `matrix_part`
     (`Spec.Compression.Verifier.matrixPass_spec`), and
     `Spec.Compression.Production.params_soundness` states the bound at the
     production parameters without it. The model is a transcription of the
     Rust code by hand, and no check compares the two. `L ≤ 2^2` and
     `κ ≤ 2^5` are const asserts in Rust, and `params_soundness` proves them
     at the production values. `slot_holds` also takes the hypothesis
     `Spec.Compression.Claim.Represents`: the dense rows of a slot claim are
     the matrices of its circuit's relation on the claim's cells. Rust builds
     these rows from the package that `SPEC.md` Algorithm 2 pins, and no
     Lean result states `Represents` for the rows of a package. It also
     needs that Rust gives each slot claim to the argument with the values
     of `Lifecycle.BoundedDepth.CompressionSlot.claimValues`, which no check
     compares. The model has three more
     premises. First, the Akita openings are ideal: an opening is accepted
     only when the committed table, which the prover fixed before the later
     challenges, has the opened value.
     `Export.BoundedDepth.FinalVerifier.AkitaKnowledge` states this premise
     with the error `εAkita`: for each prover with Akita openings, a prover
     of the game wins at most `εAkita` less often. Lean does not bound
     `εAkita`. `Export.BoundedDepth.FinalHistory.real_soundness` takes the
     same premise for the event of a false history, at one prover of the
     game, and that prover's IVC adversary must be admitted (trust item 3).
     `Export.BoundedDepth.FinalVerifier.Premises` collects the premises on
     the setup: `Represents` for the dense rows of each circuit,
     at most the blocks of its package for each claim, and a layout of Rust
     `Layout::new` that is well formed with the production bounds.
     `Spec.Compression.Rust.layoutNew_production` gives these bounds for
     the layout of the model, except `spanVars ≤ 27`, but Lean does not
     join the model to the layout of the setup. The game
     takes the slot claims before `W` and the run of Π_CCS, and Rust absorbs
     only `x`, their Poseidon2 hash, before that run, so the slot claims are
     bound through `x` up to a collision of trust item 2. Second, the key
     polynomial is the honest key. Rust enforces
     this with the pin of the full bytes of the verification key
     (`crates/ns-compress/artifacts/key20.vk`, `crates/ns-compress/artifacts/key30.vk`),
     not with a digest. Third, each draw is uniform for each query. This is
     the sponge of `[SPONGE]` under `[P2-PERM]` only if the compression
     transcript absorbs an injective framing of its messages with a domain of
     its own. Lean proves both for the words of the transcript: its sponge
     calls fix the run among the runs of one statement shape
     (`Export.BoundedDepth.CompressionSponge.calls_injective`), and they
     start with a domain that no other use starts with
     (`Export.DomainSeparation.compression_owns`,
     `Export.DomainSeparation.separated`). Rust tests compare the words with
     Rust (section Hashed encodings). The gap is the join: no theorem states
     that the draws of `Spec.Compression.Verifier.game` are the reads of this
     sponge, so no result applies `Spec.Sponge.Transfer` to the game. Lean
     models the order: each path of the game receives and draws the rows of
     `SPEC.md` section 5.4 in order (`Spec.Compression.Rust.game_follows`),
     and a test replays a real proof in that order and checks each
     challenge of `verify`. Akita's own transcript is not modeled (row Akita
     transcript of section Hashed encodings).
   - *The compression key.* The verifier's key is the key of `SPEC.md`
     Algorithm 11. It holds the Akita commitment of the key polynomial of
     the production Ajtai key, and Akita's commitments of the prefixes of
     its setup from Akita's public seed. The code pins the bytes of this key
     for each block count, and a test computes each pinned key again from
     the public seeds. `Verifier::compression_key` gives the pinned key,
     and `Verifier::decode_compression_key` accepts loaded bytes only when
     they equal it, so the pin enforces the premise for each key that the
     verifier uses. Under another key, Akita gives no binding, and the
     verifier can accept a false statement. The pin is a value in the code,
     and the test is the check of the pin. No Lean result covers either.
     The Lean result takes the honest key as a premise.
   - *Akita binding and knowledge soundness `[AKITA]`.* The one Akita proof opens
     the key polynomial, `W` and `V`. It uses Akita (ePrint 2026/1983) at
     git revision `d50f700f`, crate version 0.1.0, with the configuration
     `RecursiveCommitmentConfig<fp64::Dense>` and the schedule files of
     `SPEC.md` section 3. The premise is that an accepted proof gives
     polynomials with the claimed values, in the classical random-oracle
     model. It includes binding for the key polynomial, so the opened value
     is the value of the Ajtai key's polynomial, and binding for Akita's
     setup prefixes, so the setup that the rows offload is Akita's public
     setup. Akita states a target of a 128-bit Fiat–Shamir rate for each
     random-oracle query. Its documentation says that this target is not an
     end-to-end bound and that it declares no query budget, no bound on the
     size of the extraction tree and no reduction loss. Akita has no audit
     and no analysis in the quantum random-oracle model.
   - *The Keccak family `[KECCAK]`.* Akita makes its challenges with a
     Keccak-f duplex sponge, its digests with SHA3-256, its session
     identifier with SHAKE128 and its setup with SHAKE256. No step circuit
     computes these functions. The owner accepted this hash family for
     Akita on 2026-11-13 (`decisions/akita-compression.md`).
   - *Module-SIS for the Akita commitments `[MSIS]`.* Akita prices each of its
     Module-SIS instances at 169 bits or more in the ADPS16 quantum cost
     model (policy `Quantum169BitADPS16`). Akita states that this price is
     for one attack family on a scalar instance, that it does not price
     attacks on the ring or module structure, and that it does not include
     the loss of the reduction. The Ajtai commitments of the three claims
     keep the lower estimate of trust item 2.
   - *The Poseidon2 sponge for the argument `[P2-PERM]` `[SPONGE]`.* The
     Poseidon2 sponge makes each challenge of the Π_CCS run and of the
     argument (`SPEC.md` section 5.4). Each term of the Rust sum of `SPEC.md`
     section 3 is for one interactive run, and Lean proves that the exact
     sum bounds one interactive run
     (`Spec.Compression.Rust.params_run_soundness`). In the
     state-restoration model with `Q` queries, Lean bounds the
     non-interactive argument by `Q · error`
     (`Spec.Compression.Verifier.soundness`), and `error` is at most the
     exact sum (`Spec.Compression.Rust.error_le_runError`). Lean bounds the
     Π_CCS run with one source by `Q · 38/q²`
     (`Spec.Compression.Fresh.soundness`). The sponge meets this model, up
     to `spongeError N`, only under the open conditions of the third premise
     above.
   - *Separation of the two transcripts.* The session bytes of the Akita
     proof are a label and lanes 0 to 3 of the Poseidon2 sponge state
     after the last message of the argument, so they bind the opening to
     each earlier message, the bytes of the compression key included,
     unless these four lanes collide (trust item 2). The Π_CCS run and the
     argument have their own domain blocks, which differ from the fold
     domains. `[P2-PERM]` and `[KECCAK]` make the permutation and the Keccak
     functions independent. The compression domains, as Lean restates them,
     are separated from the fold domains (`Export.DomainSeparation.separated`),
     but no emitted data compares the restated tags with Rust.
     `Export.BoundedDepth.FinalHistory.soundness` puts the draws of the
     argument and of the Π_CCS run on the tape of an IVC adversary whose
     folds use Poseidon2 (trust item 3), so it also needs the separation of
     the compression domains from the fold domains.
   - *The error of the argument.* Lean bounds it by `Q · error`, with
     `error < 2^−119` at the production shapes
     (`Spec.Compression.Verifier.production_error_lt`). Rust bounds the
     sum of `SPEC.md` section 3 for one interactive run, about `2^−119.02`
     for the reference application at 4 lookup challenges, by `2^−119` in
     integers. To either, add
     the error of a Π_CCS run with one source, which Rust does not compute. Rust holds the Π_CCS test error for 17
     sources to the caller's security minimum (`SPEC.md` section 3).
   - *The Rust code.* Trust item 5 applies. The tests check fixed inputs.
     The verifier does not expand the Ajtai key, but the key polynomial
     that the compression key commits is the SHAKE128 key of `SPEC.md`
     Algorithm 1. So premise P1 applies to it as it does to `SPEC.md`
     Algorithm 6.
## Out of scope

- Completeness does not follow from a knowledge bound: a bound on the
  adversary is not evidence that an honest proof is accepted. The honest
  prover has separate results: an honest light fold and an honest refresh
  fold are accepted (`Spec.Folding.Light.complete`,
  `Spec.Folding.Refresh.complete`), the bound `accumulatorBound` of each
  depth of an admissible schedule stays below `B`
  (`Spec.Folding.Schedule.honest_bounds`, arithmetic only), and, given an
  accepted fold, an honest step of either circuit has an assignment in which
  every row of its source layout holds
  (`Layout.BoundedDepth.LightAssemblerCompleteness.complete`,
  `Layout.BoundedDepth.RefreshAssemblerCompleteness.complete`). Lean joins these
  results. Along an honest run, each slot's witness stays within the bound
  of its depth (`Export.BoundedDepth.StepExtend.Honest`). An honest step's witness
  satisfies the exported relation at the key's widened slot layout
  (`Export.BoundedDepth.StepCompleteness.light_complete`,
  `Export.BoundedDepth.StepCompleteness.refresh_complete`). The terminal accepts
  the honest proof of every valid run
  (`Export.BoundedDepth.RunCompleteness.run_complete`) and of every memory
  execution (`Export.BoundedDepth.NebulaCompleteness.memory_complete`). HyperNova
  Definition 11 (i) extends each proof that the verifier accepts. The
  terminal verifier holds each slot to the bound of its depth
  (`Export.BoundedDepth.RunCompleteness.Verifies`, `SPEC.md` Algorithm 6), and one
  valid step extends each proof that it accepts
  (`Export.BoundedDepth.RunCompleteness.verifies_step`). These
  results take as premises three package checks that the emitter evaluates
  (`Export.BoundedDepth.StepExtend.Complete`), key digests of four words, and
  witnesses of the application's witness length. Each application promises that its step keeps the state width
  (`Lifecycle.Stage1.Application.Program.step_length`); the counter must stay
  below `q`.
- The error is not one security level. Lean bounds no fold error in a
  Fiat–Shamir game (adversary 3), and the other terms depend on MSIS at the
  solver's work and on Poseidon2 collisions.
- The compressed proof has a Lean result for its argument only
  (adversary 6). Its error is not a security level, and the final proof is
  not zero-knowledge.
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
  (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder`) and that batched steps
  are memory consistent exactly when their rows are
  (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff`), but not
  that the Rust rows enforce those conditions. Tests check the rows on fixed
  inputs (trust item 5). Rust tests also check that each field of the
  typed WASM state changes its words (`every_state_field_changes_the_state_words`
  in `crates/neo-wasm/tests/wasm_nebula.rs`), and that a field of `q` or more
  keeps a word that the terminal check of `SPEC.md` Algorithm 8 refuses
  (`a_state_field_of_q_or_more_has_no_canonical_word`). So a verifier that
  reads a field of a proved state reads a proved word, except that `None` and
  `Some(0)` of `memory_pages` and `max_memory_pages` share the word 0.
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
- A prover that picks the WASM size of its proof
  (`neo_wasm::nebula::WasmSize`). The results hold for each of the three
  fixed sizes, not for a choice among them.
- Values for the adversary time and the query counts `Q_η`. They stay
  symbolic (`decisions/nebula-memory-checking.md`).
