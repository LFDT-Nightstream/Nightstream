import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Semantics
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Schedule
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.ProjectionPrefix
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Refinement.ProductionScheduleArtifact
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.ChunkRows
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedChunkRows
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedAcceptanceRows
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.CanonicalAcceptanceSource
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.AggregateAcceptanceRows
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.AggregateAcceptanceArtifact
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedMod5Artifact
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.SelectionRows

/-!
Owns: the public import root and three-level ownership map for the fixed
Pi_RLC challenge sampler.

Does not own: the concrete Poseidon2 permutation, the incoming transcript
cursor's authority, or Rust trace conformance.

Emits constraints: no. This module only composes semantic and refinement
theorems.

Authority boundary: transcript modules derive from a caller-supplied cursor;
sampler modules consume only the resulting canonical chunks.

| Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
|---|---|---|---|---|
| `Transcript.Cursor` | Raw-field sponge cursor transition | No | `Poseidon2Transcript` | `Transcript/Cursor.lean` |
| `Transcript.DigestRounds` | Four digest rounds and 64 LE chunks | No | `digest_rounds` | `Transcript/DigestRounds.lean` |
| `Transcript.Schedule` | Cursor chaining for fifteen rho samples | No | `enforce_pi_rlc_rhos_from_transcript` | `Transcript/Schedule.lean` |
| `Transcript.ProjectionPrefix` | Pi_CCS output bind, rho schedule, projection SIS bind, then beta; Pi_DEC children remain post-beta | No | `nifs/circuit/pi_rlc` and `projection/binding.rs` | `Transcript/ProjectionPrefix.lean` |
| `Transcript.Refinement.ProductionScheduleArtifact` | Exact fixed 15-by-4 stage order, source/estimated cost tree, and Poseidon census with untraced surfaces explicit | No | fixed recursive production fixture and stage profiler | `Transcript/Refinement/ProductionScheduleArtifact.lean` |
| `Sampler.Chunk` | Rejection, mod-5, symbol, and prefix arithmetic | No | `alphabet_sampling::chunk` | `Sampler/Chunk.lean` |
| `Sampler.Acceptance` | Exact 64-to-54 acceptance slack | No | `alphabet_sampling::acceptance` | `Sampler/Acceptance.lean` |
| `Sampler.Selection` | First 54 accepted symbols | No | `alphabet_sampling::selection` | `Sampler/Selection.lean` |
| `Semantics` | One transcript-derived sample relation | No | sampler parent | `Semantics.lean` |
| `Refinement.ChunkRows` | Readable 16-obligation/15-cell Goldilocks baseline with a derived right cubic, Nat/source equivalence, and a separate conditional norm-batch lemma | No | `alphabet_sampling::chunk` arithmetic model | `Refinement/ChunkRows.lean` |
| `Refinement.PackedChunkRows` | Eight nonresidue-packed active equations plus the unique inactive residue-pair equation, with explicit degree-eight CCS accounting | No | `alphabet_sampling::chunk` packed trace/refinement bridge and fixed selector | `Refinement/PackedChunkRows.lean` |
| `Refinement.PackedAcceptanceRows` | Two degree-eight rows derive half-products and the exact accept bit from sixteen Boolean chunk coordinates | No | proposed `alphabet_sampling::chunk::enforce_accept` lowering | `Refinement/PackedAcceptanceRows.lean` |
| `Refinement.CanonicalAcceptanceSource` | Exact current inverse rows, one-row witness canonicalization, and existential bridge to the proposed packed acceptance rows | No | `alphabet_sampling::chunk::enforce_accept` | `Refinement/CanonicalAcceptanceSource.lean` |
| `Refinement.AggregateAcceptanceRows` | Fourteen-edge Boolean product tree compressed into seven bit-pair rows, one no-wrap radix-three ProductSum, and one final accept row | No | proposed existing-role acceptance lowering; exact trace bridge still required | `Refinement/AggregateAcceptanceRows.lean` |
| `Refinement.AggregateAcceptanceArtifact` | Generated four-row source image, canonical inverse decoder, global role geometry, and exact nine-row sparse-polynomial refinement | No | `gadget_native::acceptance` active singleton leaf | `Refinement/AggregateAcceptanceArtifact.lean` |
| `Refinement.PackedMod5Artifact` | Exact generated source rows, projected decoder, active row schedule, and sparse polynomial; arbitrary-witness active-leaf soundness plus canonical-image completeness | No | `gadget_native::mod5` and production CCS materializer | `Refinement/PackedMod5Artifact.lean` |
| `Refinement.SelectionRows` | Exact product-row substitution | No | `gadget_native::selection` | `Refinement/SelectionRows.lean` |

Spec: `specs/PiRlcChallenge.spec.md`.
-/
