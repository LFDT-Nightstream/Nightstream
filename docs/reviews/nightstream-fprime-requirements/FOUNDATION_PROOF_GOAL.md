# Foundations: proofs before links

The user requested this goal on 2026-09-08 UTC. Close the 11 proof records before working on the seven remaining links. Keep the proof denominator at 75. The profile is `b=2`, `k_rho=16`, `B=65536`.

Each proof record needs a precise statement, a named Lean theorem that meets it, focused validation, and an axiom audit. Reuse sufficient existing proofs. A counter or label change alone does not close a record.

| Proof record | Required fact | Current evidence |
|---|---|---|
| `F.field.base.representation` | The active decoder preserves the exact canonical integer and rejects integers at or above `q`. | Closed: `PiCCSInputCheck.decodeField_eq_ok_iff`, `decodeField_rejects_noncanonical`; focused file/build and axiom audit passed. |
| `F.field.extension.representation` | Decoding preserves the ordered pair `(c0,c1)` and rejects an incorrect coefficient count. | Closed: `PiCCSInputCheck.decodeExtension_ordered_pair`, `decodeExtension_rejects_wrong_length`; focused file/build and axiom audit passed. |
| `F.field.tower.specialization` | The selected base and extension carriers have the stated degrees and the base embedding preserves their field operations. | Closed: `FieldTower.base_cardinality`, `extension_cardinality`, `embed_injective`, and `basis_reconstruct` prove the actual carrier counts and basis facts; audit passed. |
| `F.ring.base.coefficients` | Reading a serialized ring coefficient recovers that exact coefficient, including coefficient zero. | Closed: existing `PiCCSRepresentation.serializeRingF_getD` made public; focused validation, dependent rebuild and axiom audit passed. |
| `F.decomposition.split` | The checked split returns exactly the common-sign binary digits under the strict bound, and rejects values outside it. | Closed using existing `Radix.splitScalarChecked_eq_some_iff`, `splitScalarChecked_eq_none`, `boundedDigit_norm`, and `split_recompose`; current axiom audit passed. |
| `F.commitment.binding_collision` | The collision event retains distinct openings, both commitment equations and both strict bounds. | Closed: `PiDEC.parentCollisionEquiv` and `parent_bindingCollision_iff` preserve the two assignments, equations, bounds and distinctness; audit passed. |
| `F.setup.seed` | The selected seed has 32 canonical bytes and key dimensions come from the selected package. | Closed using existing `Poseidon2HashChainV1Setup` seed, dimension and `directProductionAuthorityNats_eq` theorems; current axiom audit passed. |
| `F.setup.chacha_rounds` | The word operations, round schedule, and feed-forward implement the stated 32-bit block function. | Closed: `ChaCha20.add32_eq_bitvec`, `xor32_eq_bitvec`, `rotateLeft32_eq_bitvec`, `quarterRound_eq_bitvec`, `doubleRound_eq_schedule`, `runDoubleRounds_eq_iterate`, and `blockWords_eq_feedForward`, with size and range theorems; audit passed. No pseudorandomness claim. |
| `F.setup.index_encoding` | Bounded seed bytes and row/block/lane indices have the exact little-endian encoding without index aliases. | Closed: structural four-byte recovery proves `littleEndian32_byte`, `initialState_seed_injective`, and `initialState_index_injective`; `SetupIndexEncoding.production_index_injective` supplies the actual bounds. Audit passed. |
| `F.setup.indexed_key` | The semantic key uses the selected seed and exact indexed wide reduction, with canonical field coefficients. | Closed: `Setup.verifierKey_eq_of_authorityWords` and selected `Poseidon2HashChainV1Setup.ajtaiKey_eq_of_authorityWords`, in addition to canonical coefficient range; audit passed. |
| `F.setup.framing` | The complete descriptor determines the seed and dimensions uniquely within their canonical ranges. | Closed: new `Setup.authorityWords_eq_iff` and selected `Poseidon2HashChainV1Setup.authorityWords_injective`; focused validation and axiom audit passed. |

All 11 proof records closed before link work began. The link records are:

- `F.multilinear.root_bound`
- `F.commitment.binding`
- `F.commitment.relaxed_binding`
- `F.setup.rust_vectors` — closed by current executed Lean/Rust setup parity and streaming checks.
- `F.setup.prg_assumption`
- `F.setup.reduction_bias`
- `F.profile.msis_assessment`

These links include probability bounds, implementation evidence, and cryptographic assumptions. Keep their exact premises explicit. Do not claim to prove MSIS hardness or ChaCha20 pseudorandomness, or reuse the paper's security estimate for different parameters.

The runtime algorithms, current root worktree, Stage 1 owner goal and architecture remain intact. The attempted counter-exclusion test was removed when the user set this proof-first goal; no counter change was published.

All 11 proof records have checked evidence. The complete boundary gate passed, followed by the library build (3,650 jobs; 307 seconds) and the axiom/test build (3,686 jobs; 18 seconds). All audited theorems use only the permitted Lean axioms. The proof denominator remains 75.

The earlier byte-recovery attempt did not pass. The successful proof uses an explicit structural decomposition of four bounded bytes. It does not raise tactic limits or change the runtime algorithm.

Phase A validation logs: `/tmp/nightstream-field-tower.log`, `/tmp/nightstream-parent-collision.log`, `/tmp/nightstream-indexed-key.log`, `/tmp/nightstream-setup-index-final.log`, and `/tmp/nightstream-foundation-phase-a-final.log`. Earlier six-entry evidence remains in `/tmp/nightstream-foundation-proof-goal-checkpoint.log`.

After Phase A passed, the current Lean setup emitter and Rust `external_lean_setup_vectors_match_current_rust` test passed. They compare the RFC block, selected coefficients, seeds, and complete 73-word descriptor. The Rust `streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases` test also passed: all 54 lanes for three production-seed cases and one RFC-seed case. These are scoped implementation checks, not a proof of the entire key or its distribution. Log: `/tmp/nightstream-current-ajtai-parity.log`.

The final website checkpoint is **Proof 75/75, Link 84/84, Rust 68/70**. All 11 proof records and seven link records are closed. The owner approved the exact fixed-matrix public-seed MSIS assumption under six explicit conditions, recorded in `PUBLIC_SEED_MSIS_ASSUMPTION.md`. The new selected-setup identity theorem and complete Lean gates pass. See `FOUNDATION_SECURITY_REVIEW.md` for exact evidence and limits. No cryptographic assumption has been relabelled as a proved fact.
