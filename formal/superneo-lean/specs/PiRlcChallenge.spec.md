# PiRlcChallenge

## Purpose

`PiRlcChallenge` specifies the fixed rejection sampler that derives each
length-54 `Pi_RLC` coefficient vector from four Poseidon2 digest rounds. Its
three-level hierarchy separates transcript authority, sampler arithmetic, and
row refinement without creating one file per equation.

## Fixed Parameters

| Name | Value | Meaning |
|---|---:|---|
| `chunkModulus` | 65,536 | Number of little-endian 16-bit values |
| `rejectionBucket` | 65,535 | The unique rejected chunk |
| `alphabetSize` | 5 | Number of centered coefficients |
| `quotientBits` | 14 | Width of the canonical mod-5 quotient |
| `chunksPerSample` | 64 | Four digests times sixteen chunks |
| `outputLength` | 54 | Coefficients in one `rho` vector |
| `slackBits` | 4 | Width of the accepted-count slack |
| `maxRejections` | 10 | Largest rejection count in an accepted fixed sample |
| `selectionWindow` | 11 | Candidate chunk count for each output position |
| `rhoCount` | 15 | Transcript-chained rho vectors in fixed Pi_RLC |

## Mathematical Tree

| Protocol phase | Constraint family | Leaf obligation | Lean file | Rust owner |
|---|---|---|---|---|
| Transcript | Cursor | Raw-field absorption and `digest32` cursor transition | `Transcript/Cursor.lean` | `Poseidon2Transcript` |
| Transcript | Digest rounds | Four rounds yield 64 canonical LE chunks | `Transcript/DigestRounds.lean` | `digest_rounds::collect_chunks` |
| Transcript | Schedule | Fifteen samples thread one cursor over indices 0 through 14 | `Transcript/Schedule.lean` | `enforce_pi_rlc_rhos_from_transcript` |
| Transcript | Projection prefix | Bind Pi_CCS outputs, derive fifteen rhos, bind the typed projection SIS material, then sample `beta`; diagnose that Pi_DEC children are post-beta | `Transcript/ProjectionPrefix.lean` | `nifs/circuit/pi_rlc` and `projection/binding.rs` |
| Transcript refinement | Production schedule artifact | Preserve the exact 15-by-4 stage order, componentwise immediate-child cost tree, and Poseidon permutation/S-box census while distinguishing materialized source dimensions from estimated low-norm dimensions | `Transcript/Refinement/ProductionScheduleArtifact.lean` | fixed recursive production fixture and stage profiler |
| Sampler | Chunk acceptance | Accept iff the chunk is not 65,535 | `Sampler/Chunk.lean` | `chunk::enforce_accept` |
| Sampler | Mod-5 | `chunk = 5 * quotient + residue` with bounded terms | `Sampler/Chunk.lean` | `chunk::enforce_mod5` |
| Sampler | Symbol/prefix | Centered symbol lies in `[-2, 2]`; prefix advances iff accepted | `Sampler/Chunk.lean` | `chunk::enforce_symbol_and_prefix` |
| Sampler | Enough accepts | Four-bit slack is equivalent to at least 54 accepts | `Sampler/Acceptance.lean` | `acceptance::enforce_enough_accepts` |
| Sampler | First accepted | Output is the first 54 accepted symbols within the 11-chunk window | `Sampler/Selection.lean` | `selection::select_first_n_accepts` |
| Composition | One sample | Transcript chunks, enough accepts, and exact output agree | `Semantics.lean` | `enforce_alphabet_sample_5_d` |
| Refinement | Direct chunk mod-5 obligations | Readable sixteen-obligation baseline over fifteen allocated cells: thirteen low-bit roots, one linearly derived high-bit root, one left-centered cubic, and one pair equation; the pair derives the right cubic, the field relation is equivalent to the Nat/source arithmetic relation, and aggregate bitness is a separate theorem conditional on an outer centered-norm premise | `Refinement/ChunkRows.lean` | `alphabet_sampling::chunk` arithmetic model |
| Refinement | Packed chunk mod-5 rows | Eight active equations pack the sixteen direct zero obligations in pairs via `a^2 - 7*b^2`; the inactive equation applies the same nonresidue norm to `L+1,R+1`, uniquely selecting the residue-zero representation; selector-gated degrees three, five, and seven fit the fixed degree-eight CCS | `Refinement/PackedChunkRows.lean` | `alphabet_sampling::chunk` packed trace/refinement bridge and fixed selector |
| Refinement | Generated packed mod-5 artifact | Exact production source rows, projected decoder, active row schedule, and sparse polynomial evaluate directly. Structural-image soundness derives every removed source row for arbitrary encoded witnesses; the reverse direction requires the canonical centered-residue image | `Refinement/PackedMod5Artifact.lean` | `gadget_native::mod5` and the production CCS materializer |
| Refinement | Selection rows | Existential product columns are equivalent to the three emitted aggregate equations; the pointwise guarded form is an optional one-hot corollary | `Refinement/SelectionRows.lean` | `gadget_native::selection` |

The outer and inner absorb schedules are explicit. The width-8 Poseidon2
permutation is supplied through `Poseidon2Core`; any concrete consumer must
prove that its executable permutation instantiates that parameter.

The fixed schedule starts at the pre-Pi_RLC NIFS cursor. `betaPrefixResult`
first absorbs the authoritative Pi_CCS output digest, then derives the fifteen
rho vectors, binds the projection digest, and samples `beta`. A complete NIFS
bridge must prove that this incoming cursor is the production cursor at that
boundary and that the accepted fixed relation has exactly fifteen Pi_CCS
outputs.

The projection-prefix model fixes the Pi_RLC parent commitment, parent
`y_zcol`, and quotient material before `beta`, up to the explicit typed SIS
collision event. Pi_DEC children occur after this prefix and therefore require
an independent opening-binding argument; transcript order alone cannot make
their recomposed coefficients non-adaptive.

The generated packed-Mod-5 closure is intentionally leaf-local. It does not
cover the fixed-selector inactive branch or prove that the outer lowering's
linear substitution of chunk inputs preserves the generated decoder. Those
composition obligations remain separate prerequisites for a full sampler-row
removal claim.

Artifact drift is never promoted automatically. The Rust conformance test
writes `PackedMod5ArtifactData.lean.expected`; review must inspect that file and
deliberately replace the committed generated module.

The production schedule artifact uses the same fail-closed promotion rule. Its
source-row dimensions are materialized by the satisfied recursive builder;
low-norm dimensions are estimator output and are never described as a
materialized relation. Stage-order and nonlinear-event traces do not establish
the absorbed field values, counter values, or Poseidon2 functional semantics.

## Acceptance Criteria

1. `lake build SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge` succeeds.
2. No `sorry`, `admit`, new axiom, or placeholder proposition is introduced.
3. Every file states its ownership and authority boundary and names the Rust
   stage whose mathematics it models.
4. No R1CS row may be removed from Rust until its concrete field-level
   refinement is proved and its trace validator matches every replaced row.
