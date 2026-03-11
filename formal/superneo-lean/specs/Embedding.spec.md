# Embedding Spec

## Purpose
- **What it is**: The coefficient embedding (Definition 7) maps a field vector `z ∈ F^{d·n_R}` into a ring vector `z ∈ R_F^{n_R}` by partitioning into d-sized chunks and placing each chunk as coefficients of a ring element: `z_i = Σ_{j=1}^d z_{i,j}·X^{j-1} ∈ R_F`. The inverse recovers the field vector by concatenating coefficients.
- **Key property**: The embedding is bijective (`unembedVec(embedVec(z)) = z` when `z.size % d = 0`) and linear — it commutes with scalar multiplication and addition: `embedVec(s·z) = s ·_blocks embedVec(z)` and `embedVec(v + w) = embedVec(v) +_blocks embedVec(w)`.
- **Protocol role**: This embedding is what makes field-native sum-check possible (D3) and provides pay-per-bit commitment costs (D2). It is norm-preserving: since the field values ARE the ring coefficients, b-bit field values yield b-bit ring norms. Downstream, `BarLift.lean` and `MatrixTransform.lean` use the closed embedding package to derive linearity for the bar-transform (Theorem 3) and matrix-vector product transform (Theorem 4), which are prerequisites for the evaluation homomorphism (Theorem 5).
- **Scope**: Formalizes the SuperNeo embedding (single field vector → shorter ring vector). The Neo embedding (SIMD, d separate field vectors) is not separately formalized.

## Target Formulas (Paper -> Lean)
- Paper formula(s):
  - Element: `cf(v) = v`, i.e. `unembedElem(embedElem(v)) = v`
  - Vector: `z ∈ F^{d·n_R} → z ∈ R_F^{n_R}` where `z_i = Σ_j z_{i,j}·X^{j-1}`
  - Linearity (scale): `embedVec(s·z) = s ·_blocks embedVec(z)` when `z.size % d = 0`
  - Linearity (add): `embedVec(v+w) = embedVec(v) +_blocks embedVec(w)` when `v.size = w.size` and `v.size % d = 0`
  - Round-trip: `unembedVec(embedVec(z)) = z` when `z.size % d = 0`
  - Matrix extension: row-wise embedding preserves scale/add under per-row `size % d = 0`
- Lean mapping:
  - `embedElem` / `unembedElem` : element-level `F^d ↔ Coeffs`
  - `embedVec` / `unembedVec` : vector-level `F^{d·n_R} ↔ Array Coeffs`
  - `embedMatrix` / `unembedMatrix` : matrix-level (row-wise)
  - `vecAddBlocks` / `vecScaleBlocks` : blockwise ring-vector algebra
  - `embedVec_vecScale_of_mod_eq_zero` : scale linearity theorem
  - `embedVec_vecAdd_of_size_mod_eq_zero` : add linearity theorem
  - `unembedVec_embedVec_of_mod_eq_zero` : vector round-trip theorem
  - `p9EmbeddingAssumption_holds` : combined package closure
- Target statement:
  - All embedding linearity and round-trip properties are proved; the combined `p9EmbeddingAssumption` is closed.

## Paper Anchors
- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Definition 7 (Coefficient Embedding: element, vector, matrix, inverse), lines 358-366
  - Theorem 3 (Inner Product Transform: `ct(ā·b̄) = ⟨a,b⟩`), lines 368-372
  - Theorem 4 (Matrix-Vector Product Transform: `Mz = ct(M̄z)`), lines 384-386
  - Theorem 5 (Evaluation Homomorphism), lines 390-400
  - Section 2.3 (SuperNeo embedding overview), lines 228-238

## Module Mapping
- Implementation: `SuperNeo.Embedding`
- Interface: `SuperNeo.EmbeddingInterface`

## Contract Surface
| Contract group | Lean surface (interface) | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Element embedding | `embedElem`, `unembedElem`, `unembedElem_embedElem`, `embedElem_unembedElem` | None | Bijective: `unembedElem(embedElem(v)) = v` and `embedElem(unembedElem(a)) = a` | Theorem-Target | `BarLift.lean` |
| Element linearity | `embedElem_vecAdd`, `embedElem_vecScale` | None | `embedElem(v+w) = embedElem(v) + embedElem(w)` and `embedElem(s·v) = s·embedElem(v)` | Theorem-Target | `BarLift.lean` |
| Block algebra | `vecAddBlocks`, `vecScaleBlocks`, `vecAddBlocks_size_of_eq`, `vecScaleBlocks_size` | For add: `a.size = b.size` | Blockwise add/scale with size preservation | Theorem-Target | `BarLift.lean`, `Checks.lean` |
| Vector linearity | `embedVec_vecScale_of_mod_eq_zero`, `embedVec_vecAdd_of_size_mod_eq_zero`, `unembedVec_embedVec_of_mod_eq_zero` | Scale: `z.size % d = 0`; Add: `v.size = w.size ∧ v.size % d = 0` | Embedding commutes with scale/add; round-trip is identity | Theorem-Target | `BarLift.lean`, `MatrixTransform.lean` |
| Matrix linearity | `embedMatrix_rowwise_vecScale_of_rows_mod_eq_zero`, `embedMatrix_rowwise_vecAdd_of_rows_size_mod_eq_zero`, `unembedMatrix_embedMatrix_of_rows_mod_eq_zero` | Per-row `size % d = 0`; for add: equal row counts and per-row equal sizes | Row-wise matrix embedding preserves scale/add; round-trip is identity | Theorem-Target | `MatrixTransform.lean` |
| Round-trip bridges | `embeddingVecRoundTrip_iff_prop`, `embeddingMatrixRoundTrip_iff_prop` | None | Bidirectional: `embeddingVecRoundTrip z = true ↔ embeddingVecRoundTripProp z` | Theorem-Target | `Checks.lean` |
| Combined package closure | `p9EmbeddingAssumption`, `p9EmbeddingAssumption_holds` | Shape guards carried in package statement | All element/vector/matrix round-trip and linearity properties bundled; fully discharged | Theorem-Target | `BarLift.lean`, `MatrixTransform.lean` |

## Proof Obligations and Closure Plan
All obligations closed. Vector round-trip proved via `chunkExact`/`flatten` inverse lemmas; matrix lifted row-wise. Combined `p9EmbeddingAssumption_holds` discharges the full package.

## Assumption Ledger
No open boundary assumptions in this module.

## Dependency and Consumer Map
- Upstream dependencies:
  - `SuperNeo/Ring.lean` (provides `Coeffs` type, `vecAdd`, `vecScale`, dimension `d`)
- Downstream consumers:
  - `SuperNeo/BarLift.lean`: uses `p9EmbeddingAssumption_holds` to derive `barLiftLinearityAssumption` (bar-transform linearity from embedding linearity).
  - `SuperNeo/MatrixTransform.lean`: uses `p9EmbeddingAssumption_holds` to derive `matrixTransformAssumption` (Theorem 4 transport from embedding linearity).
  - `SuperNeo/Checks.lean`: uses `embedVec`, `unembedVec`, `embedMatrix`, `unembedMatrix`, `embeddingVecRoundTrip`, `embeddingMatrixRoundTrip`, `embeddingSanity` for executable deterministic checks.

## Implementation Plan (How to Achieve)
1. Element bijection/linearity are trivial (`rfl`) since `embedElem`/`unembedElem` are identity on the `Coeffs` type alias.
2. Vector round-trip (`unembedVec_embedVec_of_mod_eq_zero`) proved by showing `flatten(chunkExact(z, d)) = z` when `z.size % d = 0`.
3. Vector linearity (scale/add) proved by showing `chunkExact` distributes over `vecScale`/`vecAdd` pointwise.
4. Matrix properties lifted row-wise from vector properties via `Array.ofFn`/`map` reasoning.
5. Combined `p9EmbeddingAssumption_holds` assembled from element + vector + matrix closures.

## Quality Expectations
- Round-trip and linearity theorems state explicit equalities with their chunkability preconditions (`z.size % d = 0`).
- The norm-preservation property (field values = ring coefficients → same bit-width) is documented in Purpose, since it is the key motivation for this embedding over NTT.
- Package closures are clearly labeled as proved, not boundary-assumed.

## Acceptance Criteria
1. `unembedVec_embedVec_of_mod_eq_zero` and both linearity theorems exported through the interface.
2. `p9EmbeddingAssumption_holds` closes the combined package with no remaining axioms.
3. `BarLift.lean` and `MatrixTransform.lean` can derive their linearity assumptions from the closed embedding package.
4. `lake build` succeeds.
5. `lake exe check` succeeds with `proof_import_wall=true` and `all_checks=true`.

## Out of Scope
- Neo (SIMD) embedding as a separate formalization path.
- Theorem 3 / Theorem 4 / Theorem 5 proofs (these belong to `BarLift`, `MatrixTransform`, `EvalHom` modules respectively).
- Refactoring the internal `chunkExact`/`flatten` representation.
