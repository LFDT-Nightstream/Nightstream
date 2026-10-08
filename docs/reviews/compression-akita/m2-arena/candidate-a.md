# Compression milestone 2, candidate A: setup pieces regenerated per proof, p3-whir unchanged

Date: 2026-10-07. Base: branch `claude/compression-spartan-whir` at 3198aa960 (M1).
Seat stance: use p3-whir 0.8 unchanged; split large setup commitments into pieces that
the prover regenerates and opens one at a time; use the logUp fraction-sum GKR of
`gkr.rs` for SPARK-style memory checking.

Labels: "(measured)" is from the M1 status table. "(est.)" is an estimate that nobody has
measured. "(verified)" is an identity that I checked with exact arithmetic (see
"Arithmetic checks").

Verdict in one paragraph. This stance gives a succinct layer-1 verifier under the 64 GB
cap. It needs no custom PCS code and no setup files. The verifying key is about 300
bytes. The cost is real. Each proof re-commits 2^31 setup cells, which adds about 80-115 s
of CPU time (est.). Each proof opens six WHIR commitments, so M3 must hash about 71k
Poseidon2 permutations (est.). A design with one stored setup commitment and a custom
streaming WHIR round needs about half of that hashing (est.). For M2, this stance is
not clearly worse, because the alternative buys its savings with new cryptographic code
and about 48 GiB of setup files. For M3, the hashing difference is the main question.
The module boundaries keep the setup PCS replaceable (see "Alternatives").

## Problem

After the M1 linear sum-check, the verifier needs one value:

    Ω̃(s_block) = Σ_b eq(s_block, b) · Ω_b,   Ω_b = Σ_l ζ^l ω_b[l]   (b < blocks; Ω_b = 0 above)

M1 computes it from a SHAKE pass over 967,203,072 key coefficients and a pass over
38,533,993 matrix runs (1,541,414,313 expanded entries). Verification takes 9.8 s
(measured). M2 must make this value cheap to check: polylogarithmic work plus WHIR
verification. M3 will run this verifier inside a circuit, so every new verifier step has
a circuit cost.

With τ = Barᵀ·(1, ζ, …, ζ^53) (so that Emb(bar(u))(ζ) = ⟨u, τ⟩), the value splits into
four terms:

| Term | Formula | Data it needs |
| --- | --- | --- |
| Ω_key | Σ_b eq(s,b) Σ_{i<22} λ^i Σ_{l<54} ζ^l A[i,b,l] | Ajtai key, no structure |
| Ω_pad | Σ_{b<blocks} eq(s,b) Σ_l τ_l π_K(χ_r(54b + l)) | none (tensor + carry structure) |
| Ω_mat | Σ_b eq(s,b) Σ_l τ_l Σ_{j,i} M_j[i, 54b + l] π_j(χ_r(i)) | CCS matrix runs |
| Ω_pub | Σ_{b<5} eq(s,b) λ^{pub(b)} | none |

Constraints that shape the design:

- **Memory.** The machine cap is 64 GB (owner, hard). The M1 whole-test peak is 19.2 GB
  (measured). A p3-whir 0.8 prover keeps the table, the rate-1/2 codeword, the Merkle
  tree and, in round 1, a codeword in Ext. Thus one 2^31 commitment of the key cannot fit.
- **PCS API.** p3-whir 0.8 opens one Merkle matrix per proof ("WHIR opens commitments
  holding exactly one matrix", `pcs/proof.rs`). One commitment can stack several tables,
  but the width of each table must be a power of two (`Table::new`). Two commitments
  cannot share one WHIR proof.
- **Prover data.** `WhirProverData` has no public constructor and no serde. Without a
  change to p3-whir, the only way to get prover data is `commit`. Thus, under this stance,
  the prover must re-commit each setup piece in each proof.
- **Fixed rules.** Poseidon2-only hashing, about 114 bits, paper-faithful published
  techniques, unchanged key derivation and folding, b = 2, k_rho = 16.

## Usage (caller's view)

The public `nightstream` API does not change:

```rust
let circuit = Circuit::compile(reference_bytes, application)?; // also derives the layer-1 setup key
circuit.write(path)?;                                         // the key (about 300 bytes) is saved with the package
let prover = circuit.prover(Engine::Optimized, minimum_security_bits)?;
let finished = prover.finish_with_spartan(&proof)?;
let verifier = Verifier::from_package(&circuit, Engine::Optimized, minimum_security_bits)?;
verifier.verify(&expected_state, &finished)?;                 // reads 4 roots, never the key or the matrices
```

The internal caller (the `nightstream` lifecycle) uses the `neo-spartan` API:

```rust
// Circuit::compile, once per package (about 35-50 s, peak about 16 GiB above baseline, est.)
let setup = neo_spartan::SetupKey::derive(&rows, public_blocks, point_variables, norm_bound)?;

// PreparedLifecycle construction, once per prover or verifier
let relation = neo_spartan::Relation::new(setup.clone(), params.compression_security_bits()?)?;

// finish_with_spartan: the prover gives the matrix source, because it regenerates the setup
let layer1 = neo_spartan::prove(&relation, &rows, transcript.into_inner(), &parent.claim, &parent.witness)?;

// verify_final: no matrix source exists in the signature
neo_spartan::verify(&relation, transcript.into_inner(), &parent, &proof.layer1)?;
```

How the setup data is made and found:

- `SetupKey::derive` regenerates every setup piece, commits it with the proof
  configuration (rate 1/2, folding 4), and keeps only the roots.
- `Circuit::compile` stores the key bytes in the package file (`circuit/storage.rs`).
  `Circuit::load` reads them back. The trust model is that of the matrices today: the
  caller owns package provenance. The package identity does not change. The identity
  determines the key, because `derive` is deterministic in the package matrices, the fixed
  SHAKE seed and the pinned PCS profile.
- Each proof regenerates each piece from the package in memory (SHAKE for the key, one
  `MatrixRows` pass for the run table). The prover re-commits the piece and checks the root
  against the key before it opens the piece. A bad package fails at the prover with
  `Error::Setup`.
- No setup file exists. M3 hard-codes the key words as circuit constants.

## Shape

### Data first

**Setup pieces** (one ordered list in `setup.rs`; each piece is ≤ 2^29 cells):

| Piece | Tables (width × arity) | Cells | Content |
| --- | --- | --- | --- |
| K0, K1, K2 | 8 × 26 | 2^29 each | key rows 8p..8p+7 on the (lane: low 6 bits, block: high 20 bits) cube; rows ≥ 22, lanes ≥ 54, blocks ≥ 814,144 are zero |
| M | 4 × 26, 2 × 26, 2 × 20 | 6·2^26 + 2^21 → 2^29 | run columns `[value, long, j0, j1]`, `[row, block]`; counts `[row_count, block_count]` |

The piece size of 2^29 comes from the cap. A 2^29 opening peaks at about 27-31 GiB (est.).
Add the 19.2 GB measured baseline and the total is about 50 GB. A 2^30 piece peaks at
about 61 GiB with the default rate schedule, or about 37 GiB if round 1 stays at rate 1/2
(est.). Neither leaves a safe margin. Key utilization is 967,203,072 / 3·2^29 = 60%.

**Run table** (the SPARK "sparse representation", setup data, 2^26 entries in production).
Index k = slot + 2^slot_variables · class, with 64 = 2^LANE_VARIABLES classes. Each run
`(j, i, c0 = 54b + l, len, initial, ratio)` goes to a class whose lane is l. A lane with
more runs than 2^slot_variables takes more than one class. The class-to-lane map is in the
key. Columns: value v (the run's initial coefficient; 0 on padding), long (1 for the
41-digit kind), j0 and j1 (the matrix bits), row i, block b. The verifier needs no lane
column, because the lane is a function of the class bits.

**Run kinds.** The production matrices have exactly two kinds of runs: (len 41, ratio 3),
37,572,008 runs; (len 1, ratio 1), 961,985 runs. Setup reads the kinds from the data and
writes them into the key. If more than two kinds occur, setup returns `Error::Shape`. The
verifier builds the head and tail weights from the kinds in the key, so no constant 41
or 3 is copied into the code.

**Head and tail weights** (verified). A run of kind κ that starts at lane l contributes
v·π_j(χ_r(i))·(eq(s,b)·head_κ[l] + eq(s,b+1)·tail_κ[l]), with

    head_κ[l] = Σ_{t < min(len_κ, 54 − l)} ratio_κ^t · τ_{l+t}
    tail_κ[l] = Σ_{54 − l ≤ t < len_κ}     ratio_κ^t · τ_{l+t−54}

For the 41-digit kind, tail[l] = 0 exactly when l < 14. For the single kind,
head[l] = τ_l and tail = 0.

**Per-proof reads** (one committed table, 8 × 26 = 2^29 cells), one row per run entry:
`[Re χ_r(i_k), Im χ_r(i_k), eq(s, b_k) (3 coordinates), eq(s, b_k + 1) (3 coordinates)]`.
The two row coordinates are base-field values, because χ_r is K-valued. The matrix weight
π_j is applied in the sum-check through the setup j bits. This choice keeps the reads at
8 columns (2^29). Reading π_j(χ_r(i)) as an Ext value would need 9 columns and 2^30 cells.

**Verifying key:**

```rust
pub struct SetupKey {
    shape: Shape,              // M1 shape words, plus slot_variables and row_variables
    runs: RunLayout,           // two run kinds, slot_variables, 64 class lanes
    key_roots: [pcs::Commitment; KEY_PIECES],
    matrix_root: pcs::Commitment,
    profile: [usize; 2],       // LOG_INV_RATE, FOLDING: the roots depend on them
}
```

### Module map (`crates/neo-spartan/src`)

| File | Owns | Lines now → est. |
| --- | --- | --- |
| `lib.rs` | Contract header; `SetupKey` re-export, `Relation`, `Proof`, `prove`, `verify`; the whole layer-1 Fiat–Shamir schedule (T0–T21) as one sequence; `outer_terms` | 438 → 570 |
| `setup.rs` (new) | The piece list; key-row regeneration; `RunTable::build` (classify runs, assign classes, counts); `SetupKey::derive`; re-commit with root check | 380 |
| `omega.rs` (new) | The verifier's Ω̃(s) = Ω_key + Ω_pad + Ω_mat + Ω_pub: τ, head and tail weights and class tables, the scaled-eq lane point, Ω_key from opening values, the Ω_pad carry DP, Ω_pub | 300 |
| `spark.rs` (new) | Ω_mat: the reads, the tagged two-table logUp, both GKR trees, the degree-7 sum-check, the reads and matrix-piece openings, all final checks of Ω_mat | 520 |
| `gkr.rs` | Fraction-sum GKR with general leaves (count numerators, closure denominators); `verify` returns the root and the leaf claim (p, q) | 209 → 260 |
| `pcs.rs` | One `Pcs` per commitment shape (z, reads, key piece, matrix piece) with its point schedule; `Suite` with one shared WHIR level from the joint error | 167 → 300 |
| `field.rs` | `Mixed` = Ext ⊗ K = Ext[u]/(u² − 7); `chi_mle`, `next_eval`, Ext coordinates | 83 → 170 |
| `ring.rs` | Unchanged; now only the prover uses it (all ω_b for the quotient and the linear sum-check) | 265 |
| `sumcheck.rs`, `norm.rs`, `hash.rs` | Unchanged | 97, 58, 50 |

Tests (`crates/neo-spartan/tests/internal/`): new `omega.rs` (about 250), `setup.rs`
(200), `spark.rs` (300); `layer1.rs` +150, `gkr.rs` +60, `whir.rs` +60.
`nightstream`: `circuit.rs` +15, `circuit/storage.rs` +20, the lifecycle constructor +10,
and `finish.rs` (the prover passes `&rows`, the verifier drops `matrix_rows()`).
Every file stays below 1,500 lines.

### Type sketch

```rust
// ---------- lib.rs ----------
pub use setup::SetupKey;

pub struct Relation {
    setup: SetupKey,
    pcs: pcs::Suite,
}

impl Relation {
    /// `security_bits` is -log2 of the error this layer may add (caller minimum
    /// minus the fold census, as in M1).
    pub fn new(setup: SetupKey, security_bits: f64) -> Result<Self, Error> { unimplemented!() }
    pub fn security_bits(&self) -> f64 { unimplemented!() }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    root: pcs::Commitment,           // M1 fields
    histogram: Vec<u32>,
    gkr: GkrProof,
    quotient: Vec<Ext>,
    linear: Vec<Vec<Ext>>,
    z_opening: pcs::Opening,
    matrix: spark::Proof,            // M2: Ω_mat, with the reads and matrix-piece openings
    keys: Vec<pcs::Opening>,         // M2: one opening per key piece, in setup order
}

pub fn prove(relation: &Relation, matrices: &dyn MatrixRows, transcript: Poseidon2Transcript,
             claim: &Claim, witness: &Mat<F>) -> Result<Proof, Error> { unimplemented!() }
pub fn verify(relation: &Relation, transcript: Poseidon2Transcript,
              claim: &Claim, proof: &Proof) -> Result<(), Error> { unimplemented!() }

// ---------- setup.rs ----------
/// Key rows per piece: 8 rows × 2^26 cells = 2^29, the largest piece within the
/// memory cap (derivation in DESIGN "Data first").
pub(crate) const KEY_ROWS_PER_PIECE: usize = 8;
pub(crate) const KEY_PIECES: usize = 3; // ceil(PRODUCTION_VERIFIER_ROWS / 8)

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupKey { /* fields above */ }

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct RunKind { len: usize, ratio: u64 }

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RunLayout {
    kinds: [RunKind; 2],                   // index 1 is the "long" bit
    slot_variables: usize,
    class_lanes: [Option<u8>; 1 << LANE_VARIABLES],
}

/// The matrix piece's source, in class order. Prover and setup only.
pub(crate) struct RunTable {
    layout: RunLayout,
    value: Vec<Gl>, long: Vec<bool>, matrix: Vec<u8>, row: Vec<u32>, block: Vec<u32>,
    row_counts: Vec<u32>, block_counts: Vec<u32>,
}

impl SetupKey {
    pub fn derive(matrices: &dyn MatrixRows, public_blocks: usize, point_variables: usize,
                  norm_bound: u32) -> Result<Self, Error> { unimplemented!() }
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    /// Strict; rejects a profile other than this crate's rate and folding.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
    pub(crate) fn shape(&self) -> &Shape { unimplemented!() }
    pub(crate) fn runs(&self) -> &RunLayout { unimplemented!() }
    pub(crate) fn words(&self) -> Vec<Gl> { unimplemented!() }
}

impl RunTable {
    pub(crate) fn build(matrices: &dyn MatrixRows, shape: &Shape) -> Result<Self, Error> { unimplemented!() }
    /// `[value, long, j0, j1]`, `[row, block]`, `[row_count, block_count]`.
    pub(crate) fn tables(&self) -> Vec<Table<Gl>> { unimplemented!() }
}

/// Key rows `8·piece .. 8·piece + 8` as one 8-column table on the (lane, block) cube.
pub(crate) fn key_table(piece: usize, shape: &Shape) -> Table<Gl> { unimplemented!() }

/// Commit a regenerated piece, require the setup root, bind it, return prover data.
pub(crate) fn recommit(pcs: &pcs::Pcs, tables: Vec<Table<Gl>>, root: &pcs::Commitment,
                       challenger: &mut Challenger) -> Result<pcs::ProverData, Error> { unimplemented!() }

// ---------- omega.rs ----------
/// Σ_l ζ^l f(l) = scale · f̃(point) with point_t = ζ^{2^t}/(1 + ζ^{2^t}).
pub(crate) struct LanePoint { point: [Ext; LANE_VARIABLES], scale: Ext }
impl LanePoint {
    /// Rejects when some 1 + ζ^{2^t} is zero (probability at most 63/|Ext|).
    pub(crate) fn new(zeta: Ext) -> Result<Self, Error> { unimplemented!() }
}
/// τ = Barᵀ·(1, ζ, …, ζ^53).
pub(crate) fn bar_weights(zeta: Ext) -> [Ext; D] { unimplemented!() }
/// head_κ and tail_κ per lane, and their 64-entry class tables.
pub(crate) struct RunWeights { head: [[Ext; D]; 2], tail: [[Ext; D]; 2] }
impl RunWeights {
    pub(crate) fn new(tau: &[Ext; D], runs: &RunLayout) -> Self { unimplemented!() }
    /// MLE over the class bits of head_κ[lane(class)] and tail_κ[lane(class)].
    pub(crate) fn at_class(&self, runs: &RunLayout, class_point: &[Ext]) -> [[Ext; 2]; 2] { unimplemented!() }
}
pub(crate) fn key_term(mixing: &Mixing, lanes: &LanePoint, row_values: &[Ext]) -> Ext { unimplemented!() }
pub(crate) fn pad_term(shape: &Shape, r: &[K], mixing: &Mixing, tau: &[Ext; D], s_block: &[Ext]) -> Ext { unimplemented!() }
pub(crate) fn public_term(shape: &Shape, mixing: &Mixing, s_block: &[Ext]) -> Ext { unimplemented!() }

// ---------- spark.rs ----------
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Proof {
    reads_root: pcs::Commitment,
    value: Ext,                     // the claimed Ω_mat
    reads_tree: GkrProof,           // 2 × 2^26 leaves
    tables_tree: GkrProof,          // 2 × 2^20 leaves
    rounds: Vec<Vec<Ext>>,          // 26 rounds of degree 7
    reads_opening: pcs::Opening,
    matrix_opening: pcs::Opening,
}
/// Everything Ω_mat needs that both sides compute: r, s, λ, τ, the run layout.
pub(crate) struct Inputs<'a> { /* borrowed statement point, mixing, tau, s_block, runs */ }
pub(crate) fn prove(relation: &Relation, runs: &RunTable, inputs: &Inputs<'_>,
                    challenger: &mut Challenger) -> Result<(Proof, Ext), Error> { unimplemented!() }
pub(crate) fn verify(relation: &Relation, inputs: &Inputs<'_>, proof: &Proof,
                     challenger: &mut Challenger) -> Result<Ext, Error> { unimplemented!() }

// ---------- gkr.rs ----------
pub(crate) enum Numerators<'a> { Ones, Counts(&'a [u32]) }
pub(crate) fn prove(variables: usize, numerators: Numerators<'_>,
                    denominator: &(dyn Fn(usize) -> Ext + Sync),
                    challenger: &mut Challenger) -> (GkrProof, Vec<Ext>) { unimplemented!() }
pub(crate) struct Root { pub(crate) p: Ext, pub(crate) q: Ext }
pub(crate) struct LeafClaim { pub(crate) point: Vec<Ext>, pub(crate) p: Ext, pub(crate) q: Ext }
pub(crate) fn verify(proof: &GkrProof, variables: usize, ones: bool,
                     challenger: &mut Challenger) -> Result<(Root, LeafClaim), Error> { unimplemented!() }

// ---------- pcs.rs ----------
pub(crate) struct Suite { witness: Pcs, reads: Pcs, key: Pcs, matrix: Pcs, level: usize, security_bits: f64 }
impl Suite {
    /// The smallest shared per-round level such that the six openings (key counted
    /// three times) plus all outer terms stay below 2^-required_bits.
    pub(crate) fn new(shape: &Shape, runs: &RunLayout, required_bits: f64,
                      outer: &[SecurityTerm]) -> Result<Self, Error> { unimplemented!() }
    pub(crate) fn profile_words(&self) -> Vec<Gl> { unimplemented!() }
}
impl Pcs {
    pub(crate) fn commit(&self, tables: Vec<Table<Gl>>, challenger: &mut Challenger) -> (Commitment, ProverData) { unimplemented!() }
    pub(crate) fn observe(&self, root: &Commitment, challenger: &mut Challenger) { unimplemented!() }
    /// One point per opening batch, in protocol order.
    pub(crate) fn open(&self, data: ProverData, points: &[Vec<Ext>], challenger: &mut Challenger) -> Opening { unimplemented!() }
    pub(crate) fn verify(&self, root: &Commitment, opening: &Opening, points: &[Vec<Ext>],
                         challenger: &mut Challenger) -> Result<Vec<Vec<Ext>>, Error> { unimplemented!() }
}

// ---------- field.rs ----------
/// Ext ⊗ K = Ext[u]/(u² − 7): where K-valued χ_r meets Ext-valued weights.
#[derive(Clone, Copy)]
pub(crate) struct Mixed { re: Ext, im: Ext }
impl Mixed {
    pub(crate) fn scale(weight: Ext, value: K) -> Self { unimplemented!() }
    pub(crate) fn project(self, weights: [Ext; 2]) -> Ext { unimplemented!() } // λ^re·re + λ^im·im
}
/// Σ_i eq(point, i)·χ_r(i) over i < 2^point.len(), with r.len() ≥ point.len().
pub(crate) fn chi_mle(r: &[K], point: &[Ext]) -> Mixed { unimplemented!() }
/// Σ_b eq(point, b)·eq(s, b + 1) over b < 2^n − 1.
pub(crate) fn next_eval(point: &[Ext], s: &[Ext]) -> Ext { unimplemented!() }
```

### Depth and ownership

- **Public surface.** `neo-spartan` adds one type (`SetupKey`: `derive`, `to_bytes`,
  `from_bytes`) and one parameter (`prove` takes the matrix source). Hidden behind it:
  the piece list, the class layout, both run kinds, the reads, the lookups, three GKR
  trees, the carry DP and six WHIR configurations. `nightstream`'s public API does not
  change.
- **Invariant in types.** `verify` has no `MatrixRows` parameter, so the verifier cannot
  read the matrices. The key is read only in `setup.rs`, which the verifier does not call.
- **One owner per decision.** `setup.rs` owns the piece order, the kinds and the classes.
  `omega.rs` owns the Ω decomposition. `spark.rs` owns the fingerprint and both trees.
  `pcs.rs` owns the WHIR configurations. `lib.rs` owns the transcript order.
- **Two computations of Ω.** The prover computes every ω_b with `ring.rs`, because the
  quotient needs them. The verifier computes Ω̃(s) with `omega.rs` and the openings. A
  cross-check test ties the two (Test plan, item 6). This duplication is a necessity of the
  protocol, not a second way to do one task.

## Protocol

### Algebra of each term

**Ω_key (verified).** With the 8-column pieces and the scaled-eq lane point,

    Ω_key = scale_ζ · Σ_{i<22} λ^i · Ã_i(x(ζ), s),   x_t(ζ) = ζ^{2^t}/(1 + ζ^{2^t}),   scale_ζ = Π_{t<6} (1 + ζ^{2^t})

Ã_i is column i mod 8 of piece ⌊i/8⌋, a 26-variable MLE. Each piece is opened once at
(x(ζ), s) and returns 8 values. λ is not put into a scaled point, because the openings
return one value per row.

**Ω_pad (verified): carry DP, verifier only.** Write c = 54b + l and process the bits of b
from low to high. The state is v_t = ⌊(54·(b mod 2^t) + l)/2^t⌋. Then bit t of c is
v_t mod 2, and v_{t+1} = (v_t + 54·b_t) >> 1. The value of v stays below 54, so the lane
sum merges into the initial state, D_0[v] = τ_v. A second state bit tracks
"b < blocks" from the low bits: lt' = (b_t < N_t) if b_t ≠ N_t, else lt. For each of the
20 block bits, each of the 108 states has 2 transitions with the weight
eq(s_t, b_t)·eq(r_t, v mod 2) ∈ Ext ⊗ K. The bits of c above 19 come from v_20:
Ω_pad = π_K(Σ_v D_20[v, lt = 1] · Π_{u < P−20} eq(r_{20+u}, bit_u(v_20))). The cost is
4,320 products in Ext ⊗ K plus 432 K products.

**Ω_mat (verified): sparse sum over the run table.**

    Ω_mat = Σ_k v_k · R_k · ( E0_k·(long_k·H1_0(k) + (1 − long_k)·H0_0(k)) + E1_k·(long_k·H1_1(k) + (1 − long_k)·H0_1(k)) )

with R_k = Λre(j0_k, j1_k)·Re_k + Λim(j0_k, j1_k)·Im_k, where Λ(·) is the multilinear
extension over the matrix bits of λ^{eval_a(j)}. Here E0_k = eq(s, b_k) and
E1_k = eq(s, b_k + 1). Hκ_0(k) = head_κ[lane(class(k))] and Hκ_1(k) = tail_κ[lane(class(k))].
The H functions depend only on the 6 class bits, so the verifier evaluates them in O(64).
The summand has degree 7 in each variable.

**Ω_pub.** Five products, as in M1.

### Lookup instances (one tagged logUp, two GKR trees)

Fingerprint of a tuple: fp(a, x, y, tag) = a + γ·x + γ²·y + γ³·tag, with γ ∈ Ext.

| | Reads (left tree, 2^27 leaves, p = 1) | Table (right tree, 2^21 leaves) | Why the verifier can evaluate the table side |
| --- | --- | --- | --- |
| Rows (tag 0) | (i_k, Re_k, Im_k), k < 2^26 | cell i < 2^20: (i, Re χ_r(i), Im χ_r(i)), count = row_count[i] | Re/Im of Σ_i eq(ρ, i)χ_r(i) = Π_u (ρ_u r_u + (1−ρ_u)(1−r_u)) · Π_{u≥20}(1 − r_u), computed in Ext ⊗ K (verified) |
| Blocks (tag 1) | (b_k, E0_k, E1_k), k < 2^26 | cell b < 2^20: (b, eq(s,b), eq(s,b+1)), count = block_count[b] | eq(s, ρ) in O(20); next~(ρ, s) = Σ_k Π_{t<k} ρ_t(1−s_t) · (1−ρ_k)s_k · Π_{t>k} eq(ρ_t, s_t) in O(20) (verified) |

Leaf index of the left tree: k + 2^26·tag (the tag is the top bit, so it is the last GKR
coordinate). Leaf index of the right tree: a + 2^20·tag. Identity:
Σ_leaves(left) 1/(β' − fp) = Σ_leaves(right) count/(β' − fp). Every read count is below
2^27 < p. The address columns, the counts and the index-to-class map are setup data. Only
the values read are per-proof data, and the reads commitment binds them. Padding
entries read row 0 and block 0, and the counts include them.

Leaf claims (verified):
- Left at (ρ_k, ρ_h): q = (1 − ρ_h)(β' − ĩ − γ·R̃e − γ²·Ĩm) + ρ_h(β' − b̃ − γ·Ẽ0 − γ²·Ẽ1 − γ³), all at ρ_k.
  ĩ and b̃ come from the M piece; Re, Im, E0 and E1 come from the reads.
- Right at (ρ_t, ρ_h'): p = (1 − ρ_h')·row_count~ + ρ_h'·block_count~ (M piece), and
  q = β' − Σ_u 2^u ρ_{t,u} − γ[(1 − ρ_h')·Re χ~ + ρ_h'·eq(s, ρ_t)] − γ²[(1 − ρ_h')·Im χ~ + ρ_h'·next~(ρ_t, s)] − γ³ρ_h'.

### Transcript (M1 steps T0–T10 keep their order)

| # | Step |
| --- | --- |
| T0 | Handoff of `tr` by value (M1) |
| T1 | Observe the seed, the six WHIR profile words, the shape words, **the `SetupKey` words (kinds, slot variables, class lanes, 4 roots)**, and the statement |
| T2–T9 | M1: commit z (2^26); histogram; β; norm GKR (26 layers); λ; Q; ζ; linear sum-check (26 rounds) → s = (s_lane, s_block) |
| T10 | WHIR open z at [ρ_norm, s] (M1 step; it now precedes the M2 steps, so the prover frees z early) |
| T11 | WHIR commit the reads (one 8 × 26 table, 2^29) |
| T12 | Observe Ω_mat (1 Ext) |
| T13 | γ ← Ext, then β' ← Ext |
| T14 | Left GKR, 27 layers: root (P_L, Q_L); per layer μ, rounds h(0), h(2), h(3), children, τ |
| T15 | Right GKR, 21 layers, the same form |
| T16 | Matrix sum-check, 26 rounds, h(0), h(2..7); the slot bits first, then the class bits → ρ_S |
| T17 | WHIR open the reads at [ρ_S, ρ_k] (8 columns each) |
| T18 | Observe the M root (the prover re-commits M); WHIR open M: table 0 at ρ_S, table 1 at ρ_k, table 2 at ρ_t |
| T19–T21 | For p = 0, 1, 2: observe the K_p root (the prover re-commits K_p); WHIR open K_p at (x(ζ), s_block) |

**Prover order** (memory): the prover drops each prover data at its opening. The
re-commits of the setup pieces (T18–T21) run one at a time, after the reads data is gone.

**Verifier final checks:**
1. M1: norm root P₀ = Q₀·Σ m_t/(β − t) and Q₀ ≠ 0; norm leaf z(ρ_norm) = β − q.
2. Lookup roots: P_L·Q_R = P_R·Q_L, Q_L ≠ 0, Q_R ≠ 0.
3. Left and right leaf claims, as above.
4. Matrix sum-check: last = v~·(Λre(j0~, j1~)·R̃e + Λim(j0~, j1~)·Ĩm)·(Ẽ0·(long~·H1_0 + (1 − long~)·H0_0) + Ẽ1·(long~·H1_1 + (1 − long~)·H0_1)) at ρ_S.
5. Linear: last_linear = (Ω_key + Ω_pad + Ω_mat + Ω_pub)·Λ̃(s_lane)·z̃(s).

## Soundness map

|Ext| ≈ 2^192. L is the JB list size at rate 1/2: log₂ L = 4.32 (p3-security
`list_size_bits`). The table charges each draw over every commitment that is bound
before the draw and opened after it. The four setup roots are bound at T1, so every new
draw is charged against them. This is conservative, because a setup codeword is exact
(its list has one element). The table rounds the composed terms up.

| Step | Error before × L | Note |
| --- | --- | --- |
| Layer 0 | exact census | unchanged |
| M1 layer-1 terms | as in M1 | unchanged |
| Tagged logUp at (β', γ), K = 2^27 reads, T = 2^21 cells | ≤ 8(K + T)/\|Ext\| ≈ 2^-162 | Schwartz–Zippel on the cleared identity (degree ≤ 4(K+T)) plus the nonzero denominators |
| Left GKR, 27 layers | Σ_{k≤27}(3k + 3)/\|Ext\| = 1,215/\|Ext\| | M1 formula |
| Right GKR, 21 layers | 756/\|Ext\| | |
| Matrix sum-check | 26·7/\|Ext\| = 182/\|Ext\| | |
| Ω_pad DP, Ω_pub, head and tail, closed forms | 0 | exact |
| `LanePoint` zero event | completeness only, ≤ 63/\|Ext\| | the verifier rejects; no soundness term |
| WHIR: z (2^26), reads (2^29), K0–K2, M (2^29) | p3 `prescribed_security` for each | six openings |

All new outer terms together are below 2^-155 after the L factor, and they do not move the
budget. The six WHIR openings set the budget. `Suite::new` searches one per-round level
for all six configurations. It takes the smallest level for which the sum of six
`report.error()` values plus the outer terms is at most 2^-required, with
required = the caller minimum minus the fold census (M1 rule). This costs about log₂ 6 ≈
2.6 bits more per opening than M1, which is about 9 more round-0 queries (est.).

Extraction. WHIR extracts z (M1) and the reads table. The logUp identity holds as a
rational identity (counts below p, no fingerprint collision). Thus every read equals its
table cell: Re + u·Im = χ_r(i_k), E0_k = eq(s, b_k) and E1_k = eq(s, b_k + 1). The
matrix sum-check then gives Ω_mat. The key openings give Ω_key on honest setup
codewords. Ω_pad and Ω_pub are exact. Thus the linear check is the M1 check, with the
streamed Ω̃(s) replaced by a proven value. The M1 extraction applies unchanged.

## Costs

### Per proof (production Poseidon2 package)

| Item | Value | Basis |
| --- | --- | --- |
| WHIR openings | 6: z 2^26 (1 col, 2 points); reads 2^29 (8 cols, 2 points); K0–K2 2^29 (8 cols, 1 point); M 2^29 (3 tables, 3 points) | design |
| Cells committed per proof | 2^26 + 2^29 (per-proof) + 4·2^29 (re-committed setup) ≈ 2.75·10^9 | design |
| Peak RSS of layer 1 | about 27-31 GiB during one 2^29 opening; about 28 GiB during the reads phase (reads data 16 + GKR tree 6 + leaf layer 4.5 + run table 1.1) | est.; table 4, codeword₀ 8, tree₀ 4, codeword₁ (2^29 Ext) 12, tree₁ 2 GiB |
| Peak RSS of the process | about 50 GB (19.2 GB M1 peak + 31 GiB) | est., below the 64 GB cap |
| Prover time added | reads: compute 1 s, commit 6-8 s, open 8-10 s. GKRs 5-8 s. Matrix sum-check 3-5 s. Per setup piece: regeneration 1-4 s, re-commit 6-8 s, open 8-10 s (×4) | est. (Merkle hashing of about 2·10^8 permutations per 2^29 commit; round 1 re-encodes 2^29 Ext) |
| Total `finish_with_spartan` | 25.5 s (measured, M1) + about 80-115 s → about 105-140 s CPU | est. |
| Proof size | z 376 KiB + 5 × 431 KiB of WHIR queries (no dedup) + GKRs 45 KiB + sum-check 4 KiB + M1 non-WHIR 55 KiB → layer 1 about 2.6 MiB; final proof about 3.1 MiB (M1: 0.92 MB) | p3-security formulas at level 118, est. |
| Verifier time | 6 WHIR verifications (about 12k permutations each) + 3 GKR trees (886 rounds) + DP (4.3k products) → about 0.1-0.2 s; it reads no key or matrix data | est. (M1: 9.8 s, measured) |

WHIR query counts (p3-security JB: log₂(1 − δ) = log₂(21/20) − log_inv_rate/2; level 118):

| Commitment | Rounds (log inverse rate: queries) | Query bytes | In-circuit Merkle permutations |
| --- | --- | --- | --- |
| 2^26 (z) | 1: 275, 4: 62, 7: 35, 10: 24, 13: 19 | 376 KiB | 10,375 |
| 2^29 (reads, K0–K2, M) | 1: 275, 4: 62, 7: 35, 10: 24, 13: 19, 16: 15 | 431 KiB | 11,995 |

### Setup (once per package, in `Circuit::compile`)

| Item | Value |
| --- | --- |
| Work | regenerate 3 key pieces (SHAKE, about 1 s each) + build the run table (one `MatrixRows` pass, 3-5 s) + 4 commits of 2^29 (6-10 s each) → about 35-50 s (est.) |
| Peak | about 16 GiB per commit (table 4 + codeword 8 + tree 4) + 1.1 GiB run table, one piece at a time (est.) |
| Storage | `SetupKey` about 300 bytes in the package file; no setup files |

### M3 circuit cost of the M2 verifier

| Part | Work | Est. size |
| --- | --- | --- |
| Six WHIR verifications | 10,375 + 5 × 11,995 = 70,350 Poseidon2 permutations + per-query fold checks (about 36k Ext products) | dominant |
| Transcript | about 1,000 permutations (GKR, sum-check and WHIR absorbs) | |
| Two lookup GKR trees | 561 rounds of degree 3, about 11k Ext products | |
| Matrix sum-check | 26 rounds of degree 7, about 1.3k Ext products | |
| Ω_pad DP | 4,320 Ext ⊗ K products ≈ 156k Goldilocks products | fixed shape, no branches |
| Head and tail, τ | linear with constant coefficients (3^t, Bar) + 53 Ext products | almost free in R1CS |
| Closed forms, `LanePoint` | about 300 Ext products + 6 Ext inverses | |
| Total | about 71k permutations + about 0.6M Goldilocks products. At the F′ ratio of about 76 rows per permutation (828k Poseidon2 rows / 10,921 permutations), about 5.4M + 0.6M rows | est.; M1 alone is about 11k permutations |

## Arithmetic checks

I checked each identity below in Python with exact Goldilocks arithmetic, the cubic
extension w³ = w + 1, K = F[u]/(u² − 7) and Ext ⊗ K. Random inputs; no tolerance.

| Identity | Size | Result |
| --- | --- | --- |
| Scaled-eq point: Σ_l ζ^l f(l) = scale_ζ·f̃(x(ζ)) | 6 lane bits, Ext | holds |
| Head and tail split of 41-digit ratio-3 runs, every lane | 54 lanes, 15 blocks | holds; tail = 0 exactly for l < 14 |
| next~ closed form | 5 bits | holds |
| Re/Im of Σ_i eq(ρ, i)χ_r(i) through Ext ⊗ K | 6 bits | holds |
| Ω_pad carry DP with the "b < blocks" bit | blocks = 101 (7 bits), r with 15 coordinates | holds; the state never exceeds 53 |
| Ω_mat summand vs the `ring.rs` scatter algorithm, both kinds, crossing runs, 4 matrices, K-valued χ_r with Re/Im weights | 300 random runs | holds |
| Ω_key from 8-row pieces with the lane point vs the direct sum | 22 rows, 8 blocks | holds |
| Tagged logUp: honest equality; one wrong read rejected; left and right leaf formulas | 16 reads, 8 cells | holds |

Not checked: every memory, time and size estimate; the per-lane run counts of the
production package.

## Test plan

Toy tests (each well below 300 s; run with `--release` and `timeout: 300000`):

1. `omega.rs`: the lane point, head and tail, `chi_mle` and `next_eval` against direct sums.
   The carry DP against brute force at a toy shape and at the production shape
   (blocks = 814,144, 28 coordinates; the brute force is 44M Ext ⊗ K products, which takes
   seconds).
2. `setup.rs`: `derive` is deterministic; every run is in exactly one slot; both count
   columns sum to the table length; a third run kind gives `Error::Shape`; one changed
   matrix coefficient changes the M root; a changed key-piece cell changes its root;
   `from_bytes` rejects another profile.
3. `gkr.rs`: count numerators; root and leaf against direct sums; tampered children fail.
4. `spark.rs` on synthetic run tables (both kinds, crossing runs at every lane): round trip.
   Ω_mat equals the direct `ring.rs` scatter value. Mutations: Ω_mat ± 1; one E0 read wrong
   (with Ω_mat recomputed to match); one Re read wrong; reads made at s' ≠ s; a proof made
   with a key derived from changed matrices and checked with the honest key; tampered GKR
   children and sum-check rounds; each opening's evaluations.
5. `layer1.rs` (synthetic CE(B) claim, toy carrier, real key prefix): prove and verify;
   mutation of each new proof field; each of the four roots in the verifier's key changed.
6. Cross-check: Ω_key + Ω_pad + Ω_mat + Ω_pub from the M2 verifier path equals the M1
   streamed Ω̃(s) (`ring::block_weights`) on a toy relation. This test is the semantic
   anchor between the two computations of Ω.
7. `nightstream`: the toy finish on both parity fixtures with the M1 layer-0 and layer-1
   mutations; the package round trip keeps the setup key.

Production (ignored tests, run by hand, each with `timeout: 300000`):

- `poseidon_finish_with_spartan_verifies` extended: finish time, verify time, proof bytes,
  peak RSS (`/usr/bin/time -l`). Expected: compile + setup about 60-110 s, finish about
  105-140 s, verify about 0.2 s (est.). If the whole test passes the 300 s cap, split it.
  The bench `compile` command saves the package with the setup key, and a second test loads
  it and runs base, fold, finish and verify.
- Memory gate: the measured peak RSS must stay below 64 GB (owner cap).

## Tradeoffs accepted

- We accept about 80-115 s of CPU time per proof to re-commit 2^31 setup cells, in exchange
  for no custom WHIR code, no setup files and a key of about 300 bytes.
- We accept six WHIR openings (about 71k M3 permutations, est.), in exchange for p3-whir
  unchanged, with the p3 security report for every opening.
- We accept a per-proof reads commitment of 2^29 cells, in exchange for reuse of the
  logUp GKR. Virtual reads (Shout) would need one-hot setup data that WHIR charges per cell.
- We accept 60% fill of the key pieces (tensor layout), in exchange for one point opening
  per piece and no sum-check over the weights. A flat layout fits the key into two pieces
  and saves one opening (about 12k permutations, 431 KiB, about 18 s), but it adds a
  30-round sum-check with streamed weights and a three-radix carry DP.
- We accept a class layout that depends on the per-lane run counts, in exchange for head
  and tail weights in closed form. Generic lookups of these weights would add 2 Ext read
  columns and push the reads to 2^30.
- We accept a degree-7 matrix sum-check with setup j bits, in exchange for base-field row
  reads (8 read columns, not 9).
- We accept one tagged logUp for two tables (γ³ tag), in exchange for one pair of GKR trees.
- We accept that the package identity does not cover the setup key. The key is a
  deterministic function of the identified package. The layer-1 transcript binds it to
  each proof, and M3 hard-codes it.

## Alternatives considered

1. **One stored setup commitment and a custom streaming first WHIR round.** All setup data
   goes into one 2^31 commitment. Its codeword and tree go to disk (about 32 + 16 GiB). Each
   proof streams the polynomial for the first sum-check rounds, reads about 300 rows from
   the stored codeword, and runs the later rounds with p3 code. There are 3 openings per
   proof (setup, z, reads), about 35k M3 permutations (est.), and about 25-35 s added to the
   prover (est.). Interface: the same public API, but every user of the package must
   produce, locate and check a 48 GiB artifact, and the PCS gains new code with its own
   soundness review. It lost for M2 because the M2 contract is a succinct verifier, and this
   stance meets it with less new cryptographic code. It wins if M3 cannot afford about 36k
   more permutations. The swap stays local: only `setup.rs`, `pcs.rs` and transcript steps
   T18–T21 change. `spark.rs` and `omega.rs` consume setup values at points and do not
   depend on how the pieces are committed.
2. **Shout-style virtual reads (one-hot address commitments in setup).** This removes the
   per-proof reads commitment. With 5-bit address chunks, the one-hot block and row
   addresses add about 2^33-2^34 setup cells. WHIR charges per cell, so the per-proof
   re-commit volume grows 4-8 times. Rejected.
3. **SPARK with offline memory checking (timestamps and grand products).** This is the
   construction of the Spartan paper. Read-only memory needs read and audit timestamps in
   setup (at least 2 more 2^26 columns), against one count column per table for logUp. The
   per-proof data is the same. It adds setup cells and needs a product GKR next to the
   fraction GKR. Rejected.

## Open questions and risks

1. Is about 80-115 s of extra CPU time per `finish_with_spartan` acceptable (M1: 25.5 s,
   measured)? If not, alternative 1 is the next design to consider.
2. Is a final proof of about 3 MB acceptable as the M3 input (M1: 0.92 MB)?
3. Can M3 afford about 71k Poseidon2 permutations (about 5.4M rows at the F′ ratio, est.)
   for the M2 openings? This number decides between this stance and alternative 1.
4. Do the per-lane run counts fit 64 classes of 2^20 slots? The mean is 713,592 per lane
   (68% of 2^20), and the spare 10 classes absorb up to 10 lanes with up to 2·2^20 runs
   each. If the counts do not fit, slot_variables becomes 21, the reads become 2^30, and the
   memory derivation fails. In that case, should the reads split into two commitments
   (7 openings)?
5. Is the trust model for the setup key acceptable: stored in the package file, absorbed
   into the layer-1 transcript, and not part of `Circuit::identity()`? Binding it into the
   identity would change the fold-transcript pins.
6. Should `Circuit::compile` always derive the setup key (about 35-50 s and about 16 GiB
   above baseline, est.)? The other choice is to derive it only for callers that compress.
7. Inherited from M1, still open: PoW bits 0, Johnson bound, rate 1/2 and folding 4.
8. Risk: the memory estimate depends on how p3 0.8 keeps the table and the round-1
   codeword. If the measured peak of one 2^29 opening plus the baseline is above 64 GB, set
   round 1 to log inverse rate 2 (codeword 3 GiB, about 110 more round-2 queries) through
   `ProtocolParameters::round_log_inv_rates`. p3-whir stays unchanged.

## Synthesis decision

Not applicable: this file is one runner candidate.

## Next implementation step

Measure before more code: write an ignored spike test that regenerates key piece 0 at the
production shape, commits and opens it with p3-whir (2^29, 8 columns, one point), and
records time and peak RSS. In the same test, print the per-lane run counts of the
production package. These two numbers confirm or reject the piece size and the class
layout.
