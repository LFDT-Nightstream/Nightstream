# Ring

## Purpose

- **What it is**: The polynomial ring `R_q = F_q[X] / Φ_η(X)` represented as coefficient arrays `Coeffs := Array F` of length `d = 54`, with schoolbook cyclic multiplication `mulRq`, vector addition `vecAdd`, scalar scaling `vecScale`, constant-term extraction `ct`, and ring identity/zero elements.
- **Key property**: `mulRq` preserves shape (`mulRq_size : (mulRq a b).size = d`), is commutative (`mulRq_comm`), is associative (`mulRq_assoc`), and distributes over `vecAdd` on both sides (`mulRq_vecAdd_right/left`); `vecAdd` preserves size (`vecAdd_size_of_eq : a.size = b.size → (vecAdd a b).size = a.size`).
- **Protocol role**: `Coeffs` is the universal wire type for ring-element witnesses throughout SuperNeo. `vecAdd` and `vecScale` underlie the folding scheme's linear combination `z' = ρ₁·z₁ + ρ₂·z₂` (Π_CCS, Section 7.3). `ct` extracts the constant term for CCS relation checks. `mulRq` appears in embedding products (Definition 9).

## Target Formulas

- `Coeffs := Array F`, `D := d`
- `vecAdd(a,b)_i = a_i + b_i` when `|a| = |b|`
- `vecScale(s,a)_i = s · a_i`
- `mulRq(a,b)_i = Σ_j a_j · b_{(i+d-j) mod d}` (cyclic convolution)
- `ct(a) = a[0]`
- `linComb2Vec(ρ₁, ρ₂, z₁, z₂) = vecAdd(vecScale(ρ₁, z₁), vecScale(ρ₂, z₂))`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, and Dimensions), Section 4, lines 275-282: `R_F = F[X]/Φ(X)`, degree `d`.
- Section 7.3 (Π_CCS), lines 440-470: folding linear combination `z' = ρ₁·z₁ + ρ₂·z₂`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Ring.lean` | Definition 1 (ring `R_q`) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Types | `Coeffs` | abbrev | Definitional | `Coeffs = Array F` |
| Constants | `D` | def | Definitional | `D = d` |
| Predicates | `hasRingDegreeShape` | def | Definitional | `a.size = d` |
| Predicates | `ringMulShapeProp` | def | Definitional | Both inputs have degree `d` |
| Operations | `vecAdd` | def | Definitional | Pointwise addition |
| Operations | `vecScale` | def | Definitional | Scalar multiplication |
| Operations | `linComb2Vec` | def | Definitional | `ρ₁·z₁ + ρ₂·z₂` |
| Operations | `mulRq` | def | Definitional | Cyclic convolution mod `d` |
| Extraction | `ct`, `coeffAt` | def | Definitional | Constant term / i-th coefficient |
| Identity | `zeroRq`, `oneRq` | def | Definitional | Zero and one in `R_q` |
| Size | `vecScale_size` | theorem | Theorem-Target | `(vecScale s a).size = a.size` |
| Size | `vecAdd_size_of_eq` | theorem | Theorem-Target | Equal inputs → preserved size |
| Size | `vecAdd_size_of_ne` | theorem | Theorem-Target | Unequal inputs → empty result |
| Size | `linComb2Vec_size_of_eq` | theorem | Theorem-Target | Preserved under linear combination |
| Size | `mulRq_size` | theorem | Theorem-Target | `(mulRq a b).size = d` |
| Size | `zeroRq_size`, `oneRq_size` | theorem | Theorem-Target | Both have size `d` |
| Algebra | `mulRq_assoc` | theorem | Theorem-Target | `mulRq (mulRq a b) c = mulRq a (mulRq b c)` |
| Algebra | `mulRq_vecAdd_right` | theorem | Theorem-Target | Right distributivity over `vecAdd` (shape-preserving operands) |
| Algebra | `mulRq_vecAdd_left` | theorem | Theorem-Target | Left distributivity over `vecAdd` (shape-preserving operands) |
| Value | `ct_zeroRq` | theorem | Theorem-Target | `ct zeroRq = 0` |
| Shape | `hasRingDegreeShape_zeroRq`, `hasRingDegreeShape_oneRq`, `hasRingDegreeShape_mulRq` | theorem | Theorem-Target | Canonical degree-shape closure for constants/products |
| Value | `ct_oneRq`, `coeffAt_zeroRq` | theorem | Theorem-Target | `ct oneRq = 1`, zero coefficients for `zeroRq` |
| Shape bundle | `ringMulShapeProp_of_shapes`, `ringMulShapeProp_left/right` | theorem | Theorem-Target | Constructor + projections for multiplication preconditions |

## Proof Obligations and Closure Plan

All current module-scope obligations are closed (no open assumptions in this file), including `mulRq` associativity.
Cross-module closure for `ct`-interaction and embedding-specific identities remains tracked in embedding/theorem-3 milestones.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Field.lean`: `F` type and arithmetic.
- `SuperNeo/Dimensions.lean`: `d` for ring degree.

Downstream consumers:
- `SuperNeo/CoeffMaps.lean`: uses `Coeffs`, `ct`, `mulRq`, `hasRingDegreeShape`.
- `SuperNeo/Norm.lean`: uses `Coeffs`, `vecAdd`, `vecScale`, `mulRq` for norm bounds.
- `SuperNeo/EqPoly.lean`: uses `F` arithmetic (indirectly via `Field`).
- `SuperNeo/Embedding.lean`: uses `Coeffs`, `vecAdd`, `vecScale` for block embedding.
- `SuperNeo/ProtocolRelations.lean`: uses `linComb2Vec` for folding relations.

## Implementation Plan

Stable for module scope; downstream theorem modules consume these shape/value guarantees directly.

## Quality Expectations

All size theorems must be `@[simp]`-tagged or usable in `simp` calls. Shape predicates must compose for protocol-level precondition bundles.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Core size/shape/value theorem surfaces listed in Contract Surface are proved.

## Out of Scope

- NTT-based multiplication.
- Additional `ct`-interaction identities beyond current contract (`ct_mulRq`, evaluation linkage specifics).
- Ring element comparison / ordering.
