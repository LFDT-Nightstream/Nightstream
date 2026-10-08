# M2 candidate B: one virtual WHIR polynomial per proof

Date 2026-10-07. Seat: minimize proof size and M3 circuit cost with ONE batched WHIR
opening that covers the per-proof witness and every setup polynomial. Base: branch
`claude/compression-spartan-whir` at 3198aa960 (M1). Numbers marked "est." are estimates;
nothing in this file is measured.

## Problem

M1's layer-1 verifier needs `Ω̃(s_block) = T1 + T2 + T3 + T4` and gets it by one pass over
the Ajtai key (967,203,072 SHAKE coefficients) and one pass over the matrix runs (38,533,993
runs, 1,541,414,313 expanded entries). M2 must replace both passes by openings of setup
commitments, so that the verifier is polylogarithmic plus WHIR, and every verifier step is
cheap inside the M3 circuit.

Hard constraints that shape the answer:

- **Column encoding.** p3-whir 0.8 encodes the message matrix column by column. Only the
  folded column height must fit the two-adic subgroup (`parameters/whir.rs`, phase 2), so the
  polynomial size is not capped. A virtual row over several trees exists only if all trees
  share F's column height. This decides the packing below.
- **The key is 967 M random cells.** No layout of it uses fewer than 2^30 cells. T1 needs one
  committed polynomial of that size, opened at a per-proof point, so the prover must stream
  a 2^30-cell table and read its codeword from disk at query time. Every candidate needs
  this custom path; stock p3 keeps the whole codeword and tree in RAM.
- **SPARK needs per-proof committed lookup values that depend on `s`.** The column weight of a
  run is `eq(s_block, b)·H(l)`, and `s` comes from the linear sum-check, after `z` is bound.
  So a proof always has two per-proof commitments: `z` (before λ) and the lookup values
  (after `s`). With the setup, that is three Merkle trees.
- 64 GB machine cap; tests at most 300 s; files under 1,500 lines; Poseidon2-only hashing;
  only published techniques; b = 2, k_rho = 16 unchanged; the key derivation and the folding
  protocol unchanged.

Assumptions that matter (stated per CLAUDE.md):

- A1. The setup is deterministic preprocessing. A verifier trusts the setup root only if it
  computed it, or if it reads a pinned constant for the package. The root is never taken
  from a prover.
- A2. p3-whir 0.8 accepts `ConstantFromSecondRound(6, 4)` with explicit round rates. The
  source allows any next rate up to `old rate + folding` (error `RateGrowsDomain` otherwise),
  so the 16× domain shrink after round 0 is legal. The spike runs the 31-variable config.
- A3. Only two run shapes exist (length 41 with ratio 3; length 1 with ratio 1), as the
  census reports. `Setup::build` rejects any other shape.

## Usage (caller's view)

```rust
// One time per package. Minutes of CPU and about 24 GB on the prover's disk.
let setup = circuit.compression_setup(Path::new("/srv/nightstream/setup"), minimum_security_bits)?;
let key: CompressionKey = setup.verifying_key();     // 32-byte root plus shape words; publishable

// Prover: the same verb as M1, plus the setup it reads from.
let finished = prover.finish_with_spartan(&proof, &setup)?;
let bytes = finished.to_bytes();

// Verifier: no setup files. The key comes from its own setup run or from a pinned constant.
let verifier = Verifier::from_package(&circuit, Engine::Optimized, minimum_security_bits)?
    .with_compression_key(&key)?;                    // rejects a key for another package
verifier.verify(&expected_state, &verifier.decode_final_proof(&bytes)?)?;
```

`compression_setup` opens the directory when its stored key matches the package identity
and the security profile; otherwise it builds the setup there. Production pins the key root
as a constant next to `PRODUCTION_SEED`; an ignored test rebuilds it and compares.

Inside nightstream the two layer-1 calls become:

```rust
let layer1 = neo_spartan::prove(setup.inner(), transcript.into_inner(), &parent.claim, &parent.witness)?;
neo_spartan::verify(key.inner(), transcript.into_inner(), &parent, &proof.layer1)?;
```

The matrices no longer pass through layer 1 at prove or verify time. The prover reads the
runs from the setup, so the runs have one source after setup.

## Shape

### The load-bearing data structure: F

Every committed value of a proof lives in ONE virtual polynomial F with 31 variables. In
p3's suffix layout, F's message is a matrix of 2^25 rows and 64 columns; the 6 low index
bits select the column and WHIR folds them in round 0. Each column belongs to exactly one
source polynomial, chosen by fixing low index bits:

| Source | Owner, time | Cells | F columns | Fixed low bits | Tree (leaf width) |
| --- | --- | --- | --- | --- | --- |
| X: key boxes, `col_addr`, `m_row` | setup | 2^30 (1,044,512,768 used) | 32 | b0 = 0 | setup (40) |
| E: `e_row` (3), `e_col` (3), `e_pad` (2) | proof, after `s` | 2^29 | 16 | b0 = 1, b1 = 0 | edges (16) |
| M: `val`, `row_addr`, `m0`, `m1` | setup | 2^28 | 8 | b0 = b1 = 1, b2 = 0 | setup (shared) |
| z | proof, first | 2^26 | 2 | b0..b2 = 1, b3 = b4 = 0 | witness (2) |
| zero | none | 3·2^26 | 6 | b0..b2 = 1, (b3, b4) ≠ 0 | none |

Every source polynomial is an aligned subcube of F, so an evaluation claim on it is an
evaluation claim on F at a point whose fixed bits are constants. Each tree commits the
rate-1/2 codeword rows of its source's columns. All trees have 2^26 codeword rows. Row `j`
of F's codeword is the 64 entries gathered from row `j` of the three trees, plus six zeros.

This identity does not depend on the encoder. p3 encodes each column of the message matrix
on its own (`dft_batch` over columns of height 2^25), and each F column is one source column
of the same height. The numerical check is in "Arithmetic checks"; an oracle test against
p3's own `commit` pins it.

WHIR then runs on F as on any committed polynomial. Only round 0 reads the trees; rounds 1+
commit the folded F (2^25 Ext cells) as usual.

### Other data

- `VerifyingKey` (public, about 200 bytes): package structural identifier (4 words), shape
  words (rows, blocks, run counts K0 and K, public blocks, point length, norm bound),
  WHIR profile words, setup root (4 words). Its words enter the layer-1 transcript at T1.
- `Setup` (prover only): the directory and its parsed key. Files: `rows.bin` (setup tree
  rows, 2^26 × 40 u64, 21.5 GB), `top.bin` (digest levels above the pruned height, about
  8 MB), `metadata.bin` (`val`, `row_addr`, `col_addr`, `m0`, `m1`, `m_row` as u64, 2.7 GB),
  `key.bin`. Key coefficients are not stored: the prover regenerates them from the SHAKE seed,
  which stays their single source.
- Run order in the setup: type-0 runs (length 41) first, then type-1 runs (length 1), then
  zero padding to 2^26. `K0 = 37,572,008`, `K = 38,533,993`. Type and activity are then
  closed-form functions of the run index: `t(k) = [k ≥ K0]`, `act(k) = [k < K]`.
- Key boxes: rows 22 = 16 + 4 + 2, lanes 54 = 32 + 16 + 4 + 2, blocks padded from 814,144 to
  819,200 = 2^19 + 2^18 + 2^15. That gives 36 power-of-two boxes, 973,209,600 cells, packed by
  descending size (so each offset is aligned). Cells for blocks ≥ 814,144 are zero, which keeps
  `Ω_b = 0` for padding blocks as M1 requires.

### Module map (`crates/neo-spartan/src`)

| File | Owns | Lines (est.) |
| --- | --- | --- |
| `lib.rs` | public surface; the layer-1 transcript schedule as one sequence | 438 → 600 |
| `layout.rs` (new) | F's column map, key boxes, the one ordered list of 60 openings and their F points | 250 |
| `setup.rs` (new) | `Setup::build/open`, `VerifyingKey`, run extraction, multiplicities, setup streaming | 450 |
| `disk_tree.rs` (new) | row file plus pruned digest levels; p3-format multiproofs from disk | 250 |
| `virtual_mmcs.rs` (new) | `Mmcs<Gl>` whose round-0 commitment is three roots; verify splits F rows | 250 |
| `pcs.rs` (rewrite) | WHIR config for F; prover port of p3 `open_at`/`prove`/rounds; stock p3 `verify_at` | 167 → 550 |
| `fold0.rs` (new) | streaming round-0: OOD value, per-claim accumulators, 6 rounds, fold to 2^25 Ext | 350 |
| `spark.rs` (new) | lane tables τ, H0, H1; edge columns; SPARK main sum-check; closed-form table MLEs | 450 |
| `lookup.rs` (new) | the one ordered list of 8 GKR families; prover leaves; verifier leaf and root checks | 300 |
| `gkr.rs` (rewrite) | batched fraction GKR over many families (replaces the single-family version) | 209 → 350 |
| `ring.rs` | prover-only block weights for Q, now read from setup runs; verifier use removed | 265 → 250 |
| `sumcheck.rs`, `norm.rs`, `field.rs`, `hash.rs` | small additions (degree-3 product rounds, interval MLE) | +80 |

New and changed code: about 3,100 lines (est.), of which about 1,400 are PCS plumbing
(`disk_tree`, `virtual_mmcs`, `pcs`, `fold0`). nightstream: `finish.rs`, `circuit.rs`, codec,
about +150 lines. No new crates, features or env vars.

### Type sketch

```rust
// lib.rs: the whole public surface.
pub struct Setup { dir: PathBuf, key: VerifyingKey, layout: layout::Layout }
#[derive(Clone, PartialEq, Eq)]
pub struct VerifyingKey { package: [u64; 4], shape: Shape, profile: pcs::Profile, root: pcs::Digest }

impl Setup {
    /// Build or reopen the setup for one CE(B) relation. Rejects run shapes other than
    /// (41, ratio 3) and (1, ratio 1).
    pub fn build(matrices: &dyn MatrixRows, relation: RelationParams, package: [u64; 4], dir: &Path)
        -> Result<Self, Error> { unimplemented!() }
    pub fn open(dir: &Path, expected: &VerifyingKey) -> Result<Self, Error> { unimplemented!() }
    pub fn verifying_key(&self) -> &VerifyingKey { unimplemented!() }
}
pub struct RelationParams { pub public_blocks: usize, pub point_variables: usize,
                            pub norm_bound: u32, pub security_bits: f64 }
impl VerifyingKey {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
    pub fn package(&self) -> [u64; 4] { unimplemented!() }
}
pub fn prove(setup: &Setup, transcript: Poseidon2Transcript, claim: &Claim, witness: &Mat<F>)
    -> Result<Proof, Error> { unimplemented!() }
pub fn verify(key: &VerifyingKey, transcript: Poseidon2Transcript, claim: &Claim, proof: &Proof)
    -> Result<(), Error> { unimplemented!() }

#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    witness_root: pcs::Digest, histogram: Vec<u32>, quotient: Vec<Ext>, linear: Vec<Vec<Ext>>,
    spark_sum: Ext, edges_root: pcs::Digest, spark: Vec<Vec<Ext>>, gkr: gkr::GkrProof,
    opening: pcs::Opening,     // p3 PcsProof over VirtualMmcs; carries the 60 evaluations
}

// layout.rs: the single owner of "where each polynomial lives in F".
pub(crate) enum Source { Key, Metadata, Edges, Witness }
pub(crate) enum Opening {                        // the one ordered list; prover and verifier walk it
    KeyBox(usize), ColumnAddress, RowMultiplicity, Value, RowAddress, TypeZeroCount,
    TypeOneCount, Edge { column: usize, at: Stage }, Witness(Stage),
}
pub(crate) enum Stage { Linear, Main, Leaf }     // points s, ρ_main, ρ
impl Layout {
    pub(crate) fn new(shape: &Shape) -> Result<Self, Error> { unimplemented!() }
    pub(crate) fn openings(&self) -> &[Opening] { unimplemented!() }
    pub(crate) fn point(&self, opening: &Opening, points: &Points) -> Vec<Ext> { unimplemented!() }
    pub(crate) fn key_weight(&self, box_index: usize, lambda: Ext, zeta: Ext, s_block: &[Ext])
        -> Result<Ext, Error> { unimplemented!() }   // κ_box; rejects 1 + c^(2^t) = 0
    pub(crate) fn gather_row(&self, setup: &[Gl], edges: &[Gl], witness: &[Gl]) -> [Gl; 64] {
        unimplemented!() }
}

// virtual_mmcs.rs
pub(crate) struct VirtualMmcs { inner: Mmcs, layout: ColumnMap }
pub(crate) enum Root { Plain(Digest), Virtual { setup: Digest, witness: Digest, edges: Digest } }
pub(crate) enum TreeProof { Plain(InnerMultiProof),
    Virtual { setup: InnerMultiProof, witness: InnerMultiProof, edges: InnerMultiProof } }
impl p3_commit_v08::Mmcs<Gl> for VirtualMmcs { /* plain cases delegate; verify_multi_batch
    on Root::Virtual splits each 64-entry row, checks the 6 zeros, and calls inner.verify on
    the three trees */ }
impl CanObserve<Root> for Challenger { fn observe(&mut self, root: Root) { unimplemented!() } }

// pcs.rs
pub(crate) struct Pcs { whir: Whir, protocol: OpeningProtocol, level: usize, security_bits: f64 }
pub(crate) struct Sources<'a> { setup: &'a Setup, edges: &'a [Gl], witness: &'a [Gl],
                                edge_tree: &'a DiskTree, witness_tree: &'a DiskTree }
impl Pcs {
    pub(crate) fn new(layout: &Layout, required_bits: f64, outer: &[SecurityTerm])
        -> Result<Self, Error> { unimplemented!() }
    pub(crate) fn commit_source(&self, values: &[Gl], columns: usize, scratch: &Path)
        -> Result<(Digest, DiskTree), Error> { unimplemented!() }
    pub(crate) fn open(&self, sources: Sources<'_>, points: &[Vec<Ext>], ch: &mut Challenger)
        -> Result<Opening, Error> { unimplemented!() }
    pub(crate) fn verify(&self, root: &Root, opening: &Opening, points: &[Vec<Ext>],
        ch: &mut Challenger) -> Result<Vec<Ext>, Error> { unimplemented!() }  // stock p3 verify_at
}

// lookup.rs
pub(crate) enum Family { RowRead, ColumnRead, PadRead, RowTable, ColumnTableZero,
                         ColumnTableOne, PadTable, Norm }
pub(crate) const FAMILIES: [Family; 8] = [/* the one list */];
pub(crate) fn leaves(family: Family, inputs: &LeafInputs<'_>) -> (Vec<Ext>, Vec<Ext>) {
    unimplemented!() }
pub(crate) fn leaf_at(family: Family, openings: &LeafOpenings, rho: &[Ext], c: &Challenges)
    -> [Ext; 2] { unimplemented!() }
pub(crate) fn check_roots(roots: &[[Ext; 2]], histogram_sum: Ext) -> Result<(), Error> {
    unimplemented!() }
```

Interface depth. The caller sees two types and two functions. Hidden behind them: the
column map, box packing, disk trees, the virtual MMCS, SPARK, eight lookup families, the
batched GKR and the streaming WHIR prover. `layout.rs` is the single owner of positions and
of the opening list; `lookup.rs` owns the family list; `setup.rs` is the single writer of the
setup files. Every module is private, so nothing below the surface is importable.

## Protocol

### Setup (one time; `Setup::build`)

1. Walk the matrix runs once. For each run emit `(type, matrix j, row i, start c, value v)`.
   Sort type 0 before type 1. Write `val = v`, `row_addr = 2^20·j + i`,
   `col_addr = 64·⌊c/54⌋ + (c mod 54)`. Count `m_row[(j,i)]`, `m0[(b,l)]`, `m1[(b,l)]`.
2. Fill X: the 36 key boxes from `coefficient_block(PRODUCTION_SEED, i, b)`, zero for
   blocks ≥ 814,144, then `col_addr` (2^26) and `m_row` (2^22), all at aligned offsets.
   Fill M with `val`, `row_addr`, `m0`, `m1`.
3. Encode X (32 columns) and M (8 columns) with the same p3 encoder and rate 1/2. Write the
   2^26 rows of 40 values to `rows.bin`. Hash rows with the M1 Merkle hash; keep the digest
   levels above the pruned height.
4. Derive the WHIR profile for F from `security_bits` and the outer terms (M1 mechanism).
   Write `key.bin`: package identity, shape words, profile, root.

### Proof (prover and verifier steps)

`Ω̃(s_block) = T1 + (T2 + T3) + T4`. T1 comes from 36 openings of X. T2 + T3 come from one
SPARK sum-check whose lookups are proven by one batched logUp-GKR. T4 is computed.

Prover:

1. P1: encode z (2 columns, rows of 2), build the witness tree, bind its root.
2. Histogram, λ, Q, ζ, linear sum-check exactly as M1 → `s = (s_lane, s_block)`. Q needs the
   block weights ω_b; the prover computes them from SHAKE, χ_r and the setup runs.
3. Compute and bind `v23 = T2(s_block) + T3(s_block)`.
4. Build the edge columns (2^26 each):
   `e_row(k) = RW[row_addr(k)]`, `e_col(k) = CW_{t(k)}[col_addr(k)]`,
   `e_pad(b,l) = χ_r(54b + l)` (K, two columns). Encode E (16 columns), build the edges
   tree (codeword rows to a scratch file), bind its root. Bind F's virtual root
   `(setup, witness, edges)` with p3's `observe_commitment`.
5. SPARK main sum-check, 26 rounds, degree 3, over x ∈ {0,1}^26:
   `Σ_x val(x)·e_row(x)·e_col(x) + [x_b < nb]·eq(s_block, x_b)·τ̃(x_l)·π_K(e_pad(x)) = v23`,
   where `x_l` = low 6 bits, `x_b` = high 20 bits → point ρ_main.
6. Draw β, γ, δ. Batched GKR over the 8 families below (26 layers) → leaf point ρ.
7. WHIR on F (prover port, round 0 streamed) at the 60 prescribed points of `layout.rs`.

Verifier: replays 1-7, then checks, in this order after `pcs.verify` returns the 60 values:

- (a) GKR roots: every `Q_f ≠ 0`; `Σ_{reads} P_f/Q_f = Σ_{tables} P_f/Q_f`;
  `P_norm/Q_norm = Σ_t m_t/(β − t)` (histogram, as M1).
- (b) GKR leaves: each family's `(p̃_f(ρ), q̃_f(ρ))` from the closed forms and the opened
  values equals the GKR's final claim for that family.
- (c) SPARK: last claim = `val·e_row·e_col (at ρ_main) + below(ρ_b, nb)·eq(s_block, ρ_b)·
  τ̃(ρ_l)·π_K(e_pad)`.
- (d) Linear: last claim = `(T1 + v23 + T4(s_block))·Λ̃(s_lane)·z̃(s)`, with
  `T1 = Σ_box κ_box·X̃(p_box)` and `T4(s_block) = Σ_{b<5} eq(s_block, b)·λ^{pub(b)}`.

### Transcript (changes against M1's T0–T10)

| # | Step |
| --- | --- |
| T0 | handoff as M1 (domain chunk, 4-word seed) |
| T1 | observe shape words, WHIR profile words, verifying-key words (package, root), statement |
| T2 | witness tree root (z only; rows of 2) |
| T3 | histogram |
| T4 | λ ← Ext |
| T5 | Q (53 Ext) |
| T6 | ζ ← Ext |
| T7 | linear sum-check, 26 rounds (h(0), h(2)) → s |
| T8 | v23 (1 Ext) |
| T9 | edges root; then p3 `observe_commitment(Root::Virtual{setup, witness, edges})` |
| T10 | SPARK main sum-check, 26 rounds (h(0), h(2), h(3)) → ρ_main |
| T11 | β, γ, δ ← Ext |
| T12 | GKR: 8 roots; per layer k ≥ 2: μ, k−1 rounds (degree 3), 8 × 4 children, τ_k → ρ |
| T13 | p3 `verify_at` on F at the 60 points: OOD, evaluations, batching, rounds, queries, final |

M1's z-only GKR moves into T12 (family `Norm`); β moves from after T3 to T11. Both moves keep
every challenge after the data it tests: z and the histogram are bound at T2/T3.

### Lookup instances (all in one batched GKR, 26 layers)

One fingerprint for all lookups: `fp(tag, a, v0, v1) = tag·δ³ + a + δ·v0 + δ²·v1`, leaf
denominator `q = γ − fp`. Tags 1, 2, 3 keep the three lookups apart. Index spaces have 26
bits (runs k, cube (b,l), carrier c) except the row table (22 bits, embedded with top bits 0).

| Family | Leaf numerator p | Leaf tuple in q | Verifier's MLE at ρ |
| --- | --- | --- | --- |
| RowRead | `act(k)` | (1, `row_addr(k)`, `e_row(k)`, 0) | `act` closed form; `row_addr`, `e_row` opened |
| ColumnRead | `act(k)` | (2, `col_addr(k)`, `e_col(k)`, `t(k)`) | `act`, `t` closed form; others opened |
| PadRead | `[l<54]·[b<nb]` | (3, `54b + l`, Re `e_pad`, Im `e_pad`) | mask, address linear; `e_pad` opened |
| RowTable | `m_row·[top4 = 0]` | (1, x, `RW(x mod 2^22)`, 0) | `m_row` opened; `RW̃` closed form |
| ColumnTableZero | `m0(x)` | (2, x, `CW0(x)`, 0) | `m0` opened; `CW0̃` closed form |
| ColumnTableOne | `m1(x)` | (2, x, `CW1(x)`, 1) | `m1` opened; `CW1̃` closed form |
| PadTable | `[c < W]` | (3, c, Re `χ_r(c)`, Im `χ_r(c)`) | all closed form |
| Norm | 1 | `β − z(x)` (own challenge, M1) | `z` opened |

Tables and why the verifier evaluates each MLE in O(28):

- `RW[(j,i)] = π_j(χ_r(i))`, `π_j(v) = λ_{re,j} Re v + λ_{im,j} Im v`, i < 2^20.
  `RW̃(y_j, y_i) = Σ_{j<4} eq(y_j, j)·π_j(Π_{t<20} (r_t y_t + (1−r_t)(1−y_t))·Π_{t=20}^{27}(1−r_t))`,
  computed in `Ext[u]/(u² − 7)`. Re and Im commute with the MLE because they are F-linear.
- `CW0[(b,l)] = eq(s_block, b)·H0[l] + eq(s_block, b+1)·H1[l]` (no wrap at b = 2^20 − 1).
  `CW0̃(y_l, y_b) = eq(s_block, y_b)·H̃0(y_l) + EqPlusOne(y_b, s_block)·H̃1(y_l)` with the
  published shift polynomial (Jolt's `EqPlusOne`):
  `EqPlusOne(y, s) = Σ_{k<20} (1−y_k)·s_k·Π_{t<k} y_t(1−s_t)·Π_{t>k} eq(y_t, s_t)`.
  This is how a run that reaches block b + 1 needs one lookup, not two: the shift sits
  inside the table value.
- `CW1[(b,l)] = eq(s_block, b)·τ_l`; `CW1̃ = eq(s_block, y_b)·τ̃(y_l)`.
- Lane tables (64 entries, τ_l = H0[l] = H1[l] = 0 for l ≥ 54): `τ = bar(1, ζ, …, ζ^53)`;
  `H0[l] = Σ_{k<min(41, 54−l)} 3^k τ_{l+k}`, built by `H0[l] = τ_l + 3·H0[l+1] − 3^41·τ_{l+41}`;
  `H1[l] = 0` for l ≤ 13 and `H1[l] = 3^{54−l}·Σ_{j ≤ l−14} 3^j τ_j` otherwise. O(54) each.
- `PW[c] = χ_r(c) = eq(r_{0..26}, c)·(1 − r_26)(1 − r_27)` for c < 2^26.
- Masks and types: `act = below(·, K)`, `t = 1 − below(·, K0)`, `[b < nb] = below(·, 814,144)`,
  `[c < W] = below(·, 43,963,776)`, `[l < 54] = below(·, 54)`, where `below(y, K)` is the
  top-down bit DP `Σ_{t: K_t = 1} (Π_{u>t} eq(y_u, K_u))·(1 − y_t)` (O(n)).
- Addresses: `54b + l` and `int(x)` are linear in the bits: `Σ 2^t y_t` forms.

Why each lookup is complete and sound. Setup multiplicities count each tuple's reads.
Each carrier coordinate c < W is read exactly once by PadRead (the map (b,l) → 54b + l is a
bijection onto [0, W) under the mask). A read tuple not in the table leaves an uncancelled
pole (read counts are below p), so the identity fails as a rational function in γ.

### WHIR on F

- Config: 31 variables, starting rate 1/2 (domain 2^32), `ConstantFromSecondRound(6, 4)`,
  round rates 2^-3, 2^-6, 2^-9, 2^-12, 2^-15, Johnson bound, PoW 0 (owner decision pending),
  level searched as in M1 against the remaining budget.
- Commitment: `Root::Virtual{setup, witness, edges}`. Round-0 queries open row j of each
  tree; `VirtualMmcs::verify_multi_batch` rebuilds F's 64-entry row and checks three p3
  multiproofs with the stock inner MMCS. Later rounds use the plain inner MMCS.
- Prover: a port of p3's `open_at`, `prove`, `round` and `final_round` built only from p3's
  public pieces (`WhirProverTranscript`, `SumcheckProver`, `Constraint`, `EqStatement`,
  `SelectStatement`, proof structs). Two parts are new:
  - `fold0.rs`: one streaming pass over X and M computes the OOD value and, for each
    prescribed claim p, the 64-entry accumulator `A_p[y] = Σ_h eq(p_high, h)·F(y, h)`
    (split-eq tables, each claim only over its own subcube). The six round messages follow
    from the accumulators. A second pass folds F to 2^25 Ext cells.
  - Query rows: setup rows from `rows.bin` and pruned paths recomputed from 256-row
    neighbourhoods; edges and witness rows from their scratch files.
- Verifier: stock `WhirProver::verify_at` with `MT = VirtualMmcs`. No verifier-side WHIR
  code is new except the virtual MMCS.

## Soundness map

`#Ext = p^3 ≈ 2^192`. As in M1, p3 charges every outer term over WHIR's list size L.

| Term | Bound | log2 (before × L) |
| --- | --- | --- |
| M1 terms (λ, ζ, linear sum-check) | unchanged | ≤ −185 |
| SPARK main sum-check (26 rounds, degree 3) | 78/#Ext | −185.7 |
| logUp at (γ, δ): cleared identity of degree ≤ 4 per pole, N = 7·2^26 + 2^22 poles | 4N/#Ext | −161.2 |
| Norm at β (as M1) | (2^26 + 7,345)/#Ext | −166.0 |
| Batched GKR: Σ_{k≤26} (3(k−1) + 15 + 1), 16 claims combined per layer | 1,391/#Ext | −181.6 |
| Key-box points: `1 + λ^{2^t} = 0` (t < 4) or `1 + ζ^{2^t} = 0` (t < 5) | completeness only; verifier rejects | — |
| WHIR on F: 60 prescribed claims + OOD, JB, p3 `prescribed_security` | derived | ≥ remaining budget |
| Merkle binding of the three trees, transcript sponge | Poseidon2, 4-word digests | as today |

Total: the new outer terms add less than 2^-160. The budget stays about 114 bits; layer 0's
Π_CCS term still dominates and the WHIR level absorbs the rest, as in M1.

Extraction. WHIR extracts F*. (1) Setup columns: the honest setup codeword is an exact
codeword; F* agrees with it on more than a sqrt(ρ) > ρ fraction of rows, so each setup
column of F* equals the honest one. (2) z and the edge columns are read off F*. The edge
list is fixed at T9, before every SPARK and GKR challenge. The z list is fixed at T2, before
λ, ζ, s, and that is the same union bound over L that M1 already pays. (3) logUp gives
`e_row`, `e_col`, `e_pad` equal to the table values; SPARK then gives `v23 = T2 + T3`;
the openings give T1; the linear sum-check and the ring check at ζ give the 37 rows as in M1.

## Costs (all est.)

Prover, CPU engine, production shape:

| Phase | Time | Resident memory |
| --- | --- | --- |
| Layer 0 + M1 steps (histogram, Q, linear) | 25 s (M1 measured finish 25.5 s) | as M1 |
| Edge columns (tables CW0, CW1 2×1.6 GB, PW 1 GB, RW 0.1 GB; stream metadata) | 3 s | 4 GB, freed |
| Edges commit: 2^26×16 encode, 2×10^8 Poseidon2 calls | 15–20 s | 4.3 GB table; codeword to scratch disk |
| SPARK main sum-check | 4 s | ≈ 10 GB, halving each round |
| Batched GKR: 8 families × 3.2 GB trees | 15–20 s | 26 GB (peak phase) |
| WHIR on F: 2 streams of 1.3 G setup cells, round-1 codeword 2^28 Ext, queries | 10 s | 6.4 + 1.6 GB |
| Total finish | ≈ 80 s | peak ≈ 26 + 4.3 + 3 + layer-0 residue ≈ 35–42 GB (< 64 GB) |

Setup (one time): 1–2 min CPU (SHAKE 1 s, runs 10 s, encode 40 columns of 2^26 rows,
3.4×10^8 Poseidon2 calls, 21.5 GB write); RAM below 25 GB with column-at-a-time encoding.
Disk: 21.5 GB rows + 2.7 GB metadata + 8 MB digests ≈ 24 GB. Per proof the prover reads the
metadata file twice in sequence (5.4 GB), regenerates the key from SHAKE twice, and reads
about 22 MB of random setup rows (271 queries × 256-row path neighbourhoods).

Proof size (layer 1):

| Part | Size |
| --- | --- |
| WHIR round 0: 271 rows × 64 Gl (139 KB) + 3 multiproofs over 2^26 rows (≈ 3 × 164 KB) | ≈ 630 KB |
| WHIR rounds 1–5 and final (82, 40, 27, 20, 16, 13 queries) | ≈ 175 KB |
| GKR (325 rounds × 3 Ext, 26 × 32 children, 16 roots) | 44 KB |
| Histogram, Q, linear and SPARK rounds, v23, roots, 60 evaluations | ≈ 36 KB |
| Layer 1 total | ≈ 0.89 MB |

The whole final proof is about 1.2 MB with the 275 KB statement and layer 0 (M1: 919 KB).

Verifier (native): stock WHIR verify ≈ 25k Poseidon2 calls plus about 50k Ext operations
(tens of milliseconds), GKR and closed forms below 5 ms. The 9.8 s key and matrix pass is
gone.

M3 circuit (layer-1 verifier only; about 76 rows per Poseidon2 call from the F′ census,
about 10 rows per Ext multiplication):

| Item | Poseidon2 calls | Ext operations |
| --- | --- | --- |
| WHIR round 0: 271 × (4 + 2 + 1 leaf + 3 × 26 path) | 23.0k (18.1k with shared nodes) | 17k (row folds) |
| WHIR rounds 1+ | 5.3k | 8k |
| WHIR final weight (60 points + OOD + ≈ 460 shift points) | — | 12k |
| GKR, sum-checks, Q, histogram absorption | 1.3k | 7k + 7,345 inversions |
| Closed-form tables, masks, 36 key boxes | — | 3k |
| Total | 25–30k | ≈ 50k |

M3 ≈ 2.4–2.8 M rows (est.), against F′ today at 1.0 M rows.

Why ONE opening: the separate-openings shape (setup X, setup M, z, edges: four WHIR proofs)
pays the same round-0 queries and four copies of rounds 1+: about 37–45k Poseidon2 calls
(+40%) and about +0.5 MB of proof. The single opening saves about 0.6–1.0 M M3 rows.

## Arithmetic checks

Checked with a throwaway Python script over Goldilocks at random points (see the appendix).
All passed after one fix to my own check (the interval DP must return 1 when K ≥ 2^n).

| Formula | Verified | How |
| --- | --- | --- |
| Scaled-eq: `Σ_i c^i a_i = Π_t(1 + c^{2^t})·ã(x)`, `x_t = c^{2^t}/(1 + c^{2^t})` | yes | n = 5, random a, c |
| Power-tensor MLE `Σ_i eq(y,i) c^i = Π_t(1 − y_t + y_t c^{2^t})` | yes | n = 5 |
| Key boxes: `T1 = Σ_box κ_box·X̃(p_box)` with buddy packing and `eq(s, b0+b') = eq(s_lo, b')·eq(s_hi, b0_hi)` | yes | 22 rows, 54 lanes, 20 blocks padded to 22, full X MLE |
| H0/H1 split of a 41-digit run, all 54 start lanes | yes | random τ and block weights |
| H0 recurrence and H1 prefix form (O(54)) | yes | equal to the direct sums |
| EqPlusOne closed form (no wrap) | yes | n = 6, brute force |
| `CW0̃ = eq·H̃0 + EqPlusOne·H̃1` | yes | 4 block bits, 6 lane bits, full table MLE |
| `below(y, K)` interval MLE | yes | n = 7, K ∈ {0, 1, 37, 64, 100, 127, 128} |
| Pad address MLE `54b + l` | yes | 10 bits |
| Re/Im of χ_r commute with the MLE in `F[u]/(u²−7)` | yes | n = 5 |
| T2 carry DP (state: last five b bits, carry ≤ 4) | yes | toy 6 block bits, 12 carrier bits, all 54 lanes against brute force |
| T2 DP size at production (20 block bits, 28 carrier bits) | counted | 138,120 transitions over 54 lanes, at most 84 states |
| Virtual row identity (F row j = gathered tree rows, z with two fixed zero bits) | yes | 10-variable F, encoder modelled as column polynomials at `ω^j`; encoder-independent by construction |

Not checked numerically: `Emb(bar(u))(ζ) = <u, τ>`. M1's ring-row test covers it; M2's
oracle test (below) compares the succinct T2 + T3 with M1's streamed weights.

## Test plan

All in `crates/neo-spartan/tests/` and `crates/nightstream/tests/`, `--release`, ≤ 300 s.

Toy relation: 4 matrices, 2^8 rows, 64 blocks, the real 22 key rows, both run types, so F
has about 2^19 cells. Setup, prove and verify take under a few seconds.

Oracle and equivalence tests (completeness, and the riskiest assumptions first):

1. `virtual_rows_match_p3`: commit the materialized toy F with stock p3; the gathered rows
   of the three trees equal p3's rows, row by row.
2. `streaming_round0_matches_p3`: on the materialized toy F with a plain MMCS, `fold0` gives
   byte-identical `initial_sumcheck` data and the same folded table as p3's `SuffixProver`.
3. `ported_prover_matches_p3`: for a plain (non-virtual) toy polynomial, the ported prover's
   `PcsProof` bytes equal p3's `open_at` bytes.
4. `succinct_weights_match_m1`: `T1 + T2 + T3 + T4` from the openings equals M1's
   `block_weights` → `Ω̃(s_block)` on the toy relation (M1's code path stays as the test oracle,
   built inside the test from `MatrixRows`).
5. Closed forms: `RW̃`, `CW0̃`, `CW1̃`, `PW̃`, `below`, `EqPlusOne`, lane tables against full
   table MLEs.
6. Disk tree: multiproofs from disk equal p3's `open_multi_batch` on the same rows.

Mutation tests (the verifier must reject):

- Setup: change one cell of `val`, `row_addr`, `col_addr`, `m0`, `m1`, `m_row` or one key
  coefficient, rebuild the files but keep the honest key → reject. A key from another
  package identity → `with_compression_key` rejects.
- Prover: one wrong `e_row`, `e_col` or `e_pad` cell (and a consistent re-commit) → GKR root
  check fails; a false `v23` → SPARK or linear check fails.
- Proof parts: flip each of the witness root, edges root, histogram, Q, a linear round, a
  SPARK round, a GKR root, round, child; one of the 60 evaluations; an OOD answer; one entry of
  an F row from each tree, and one of the six zero entries; one sibling in each multiproof.
- False witnesses: a z entry above H; a z that breaks one commitment row.

Production (ignored, each under 300 s, CPU):

- `production_compression_setup`: build the setup into `target/nightstream-compression/<id>`
  and compare the root with the pinned constant. Est. 1–2 min.
- `poseidon_finish_with_spartan_m2_verifies`: compile, base, one fold, finish with the setup
  above, verify; record finish time, verify time, proof bytes and peak RSS. Est. 110 s.

## Tradeoffs accepted

- We accept about 1,400 lines of custom PCS plumbing (virtual MMCS, disk trees, ported prover
  loop, streaming round 0) in exchange for one WHIR opening: about 30% fewer M3 Poseidon2
  calls and about 0.5 MB less proof than four openings, and no 26–49 GB in-RAM setup tree.
  The soundness-critical verifier stays stock p3 plus a 250-line MMCS.
- We accept 60 prescribed points (36 for key boxes, with scaled-eq coordinates) in exchange
  for the stock p3 verifier, which takes only point claims.
- We accept a fixed first fold of 6 bits (64 columns of height 2^25). It keeps the folded
  table at 0.8 GB and round-0 rows at 512 bytes. A 5-bit fold saves about 45 KB of proof but
  needs a 26-variable round 1 (6.4–12.9 GB codeword). A larger circuit adds columns, not a
  second polynomial, while each source stays a whole number of columns.
- We accept a trusted deterministic setup (24 GB on the prover's disk, a 32-byte root for
  verifiers) in exchange for a verifier that never reads the key or the matrices.
- We accept a 26 GB GKR phase in exchange for one batched GKR proof.
- We accept `e_pad` (Pad as a lookup) in exchange for removing the verifier's carry DP:
  the two extra columns are free in E (6 or 8 columns both fill the 2^29 source), and the DP
  would cost about 138k K⊗Ext transitions, near 4 M M3 rows.
- We accept zero knowledge is still absent (as M1).

## Alternatives considered

**A1. Four stock openings (X, M, z, edges as separate p3 WHIR proofs).** Same public
surface. It hides less inside layer 1 but exposes nothing new to callers. It drops the
virtual MMCS and the ported prover only if the setup openings run on stock p3, which keeps
the whole setup codeword and tree in RAM: about 28 GB for X and 7 GB for M, read from disk
each proof. Costs: +40% M3 hashing, +0.5 MB proof, near-cap RAM. If the in-RAM route does
not fit, A1 needs the same streaming prover anyway and then saves only the 250-line MMCS.
It lost on the seat's objective and on memory.

**A2. Verifier-heavy shape: T2 by the carry DP, T3 by column aggregation.** The verifier
runs the 138,120-transition DP for Pad, and T3 commits per-proof column sums
`R_t(b,l) = Σ val·e_row` (logUp with weighted numerators) instead of a column lookup. The
interface is the same. It hides less complexity in the prover and pushes it into every
verifier, which is the M3 circuit: about +4 M rows for the DP. Column aggregation needs 11
edge columns, which overflow the 2^29 source and break the 2^31 packing. Rejected.

**A3. Fully custom WHIR, prover and verifier, with weighted claims.** T1 becomes one
power-tensor weighted claim (no boxes, no scaled-eq inverses), and the p3-whir dependency
goes away. Deepest single module (one PCS file owns all of WHIR). It saves 35 points (about
2k Ext operations in M3, 1 KB of proof) but makes the verifier and its security accounting
ours. Rejected for now; reconsider when M3 writes its in-circuit WHIR verifier, because that
circuit is a custom verifier anyway.

## Open questions and risks

1. Does the owner accept that layer 1 now carries a ported copy of p3's prover loop (about
   350 lines) whose only guard against drift is a byte-equality test with p3?
2. Will the owner approve PoW grinding for this opening? 20 bits cut round-0 queries from
   271 to about 225 (−17% of M3 hashing).
3. Who publishes the production verifying key: a pinned constant checked by an ignored test,
   or a key file shipped with the package?
4. Is 24 GB of setup data on the prover's disk acceptable?
5. Is a 26 GB GKR phase acceptable, or should the families split into two batched GKRs
   (13 GB each, +25 KB proof, about +700 M3 Poseidon2 calls)?
6. Risk: the ported prover must match p3's transcript byte for byte. Tests 2 and 3 guard it;
   a p3 upgrade can break it silently only if those tests are removed.
7. Risk: the run census (only two shapes, 38.5 M runs < 2^26) is load-bearing for the
   packing. A circuit with more runs moves E or M past their sources.

## Next implementation step

Build `layout.rs`, `disk_tree.rs` and `virtual_mmcs.rs` at toy size and make
`virtual_rows_match_p3` pass, because the encoder and transcript compatibility with p3 is
the assumption every other part depends on.

## Appendix: condensed check script (Python 3, Goldilocks)

```python
import random; P=2**64-2**32+1; R=random.Random(7); r=lambda:R.randrange(P)
inv=lambda a:pow(a,P-2,P)
def eq(y,b):
    o=1
    for t,v in enumerate(y): o=o*(v if b>>t&1 else 1-v)%P
    return o
mle=lambda T,y:sum(eq(y,i)*v for i,v in enumerate(T))%P
def eqv(a,b):
    o=1
    for u,v in zip(a,b): o=o*((u*v+(1-u)*(1-v))%P)%P
    return o
# scaled-eq
n=5;a=[r() for _ in range(32)];c=r()
x=[pow(c,2**t,P)*inv(1+pow(c,2**t,P))%P for t in range(n)];k=1
for t in range(n): k=k*(1+pow(c,2**t,P))%P
assert sum(pow(c,i,P)*a[i] for i in range(32))%P==k*mle(a,x)%P
# H0/H1 split
D=54;tau=[r() for _ in range(D)];E=[r() for _ in range(4)]
H0=[sum(pow(3,j,P)*tau[l+j] for j in range(min(41,D-l)))%P for l in range(D)]
H1=[sum(pow(3,j,P)*tau[l+j-D] for j in range(D-l,41))%P for l in range(D)]
for b in range(3):
    for l in range(D):
        g=lambda q:E[q//D]*tau[q%D]%P
        assert sum(pow(3,j,P)*g(D*b+l+j) for j in range(41))%P==(E[b]*H0[l]+E[b+1]*H1[l])%P
# EqPlusOne
def ep1(y,s):
    o=0
    for k in range(len(y)):
        t_=(1-y[k])*s[k]%P
        for t in range(k): t_=t_*y[t]%P*(1-s[t])%P
        o=(o+t_*eqv(y[k+1:],s[k+1:]))%P
    return o
y=[r() for _ in range(6)];s=[r() for _ in range(6)]
assert sum(eq(y,b)*eq(s,b+1) for b in range(63))%P==ep1(y,s)
# interval MLE
def below(y,K):
    if K>=1<<len(y): return 1
    o=0;p=1
    for t in range(len(y)-1,-1,-1):
        if K>>t&1: o=(o+p*(1-y[t]))%P; p=p*y[t]%P
        else: p=p*(1-y[t])%P
    return o
y=[r() for _ in range(7)]
assert all(below(y,K)==mle([int(i<K) for i in range(128)],y) for K in (0,1,37,100,128))
print("ok")
```

The key-box, CW0 table, Re/Im, carry-DP and virtual-row checks used the same helpers on the
sizes listed in the table above.
