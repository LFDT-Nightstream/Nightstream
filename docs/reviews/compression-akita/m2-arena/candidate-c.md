# Compression milestone 2, candidate C: honest setup oracles and a two-level scatter

Date: 2026-10-07. Base: branch `claude/compression-spartan-whir` at 3198aa960 (M1).
Seat stance: none fixed; shape derived from first principles.
Marks: "est." is an estimate. "verified" means a numeric check in Goldilocks against a
brute-force sum (see [Verification record](#verification-record)). Nothing is measured.

## Problem

After the M1 linear sum-check, the verifier must know

    Ω~(s_b) = T1 + T2 + T3 + T4      (s = (s_l: 6 lane coords, s_b: 20 block coords))

M1 gets T1 (22 key rows), T2 (Eval_K over c = 54b + l) and T3 (four CCS matrices) by
one SHAKE pass over 967,203,072 key coefficients and one pass over 38,533,993 matrix
runs (verify 9.8 s). M2 must make the verifier polylogarithmic plus WHIR verification,
under these constraints:

- 64 GB machine cap; the prover already peaks near 19 GB.
- Poseidon2 only in proof and transcript paths; about 114 bits; Ext ≈ 2^192.
- Folding, b = 2, k_rho = 16 and the SHAKE128 key derivation stay unchanged.
- Every verifier step will run inside the M3 shrink circuit. The M3 cost is mostly
  Poseidon2 permutations (Merkle paths, transcript) and Ext arithmetic.

The shape is not obvious for three reasons:

1. **The key has no structure.** A plain p3-whir commitment to the key needs about
   48 GiB of prover data at 2^31 cells, so it does not fit with the rest.
2. **The ring and the binary cube disagree.** The ring has 54 lanes per block. The
   CE(B) point r uses binary bits of the logical index c = 54b + l. One of T1 or
   T2/T3 always becomes non-tensor.
3. **The matrix column weights come late.** They depend on s and ζ, which are drawn
   long after z is committed. Standard SPARK then needs a late commitment of size
   nnz.

## Derivation (assumptions challenged)

| # | Assumption | Necessity or convention? | Result |
| --- | --- | --- | --- |
| A1 | z lives on the (block, lane) cube | Necessity for T1: only on this cube is the key weight λ^i·eq(s_b,b)·ζ^l a tensor, so T1 is one point claim. A flat c-cube makes the key claim a general linear form, and p3 accepts only eq claims. | Keep. T2 needs a DP and the matrix columns need a lane split. |
| A2 | λ batches all 37 ring rows | Necessity. It also makes the 22 key rows one virtual oracle A_λ = Σ λ^i A_i. | Keep. Store the key as 22 separate setup oracles, so λ collapses the row dimension for free. |
| A3 | One quotient, ζ, then the linear sum-check | No effect on M2 | Keep unchanged |
| A4 | A setup commitment needs a proximity proof | **Convention.** The setup codeword is computed honestly and deterministically from the package; the verifier, or M3, pins its root. An exact RS codeword needs no proximity test. Two distinct polynomials of degree < d agree on a random domain point with probability < d/\|L\|. | One "honest fold" of all setup data: send 2^k partial evaluations, fold k variables, and check t in-domain queries against the setup tree. The folded remainder moves into a per-proof commitment. Error per query is ρ = 1/2, not √ρ, and there are no OOD samples or WHIR rounds on setup data. |
| A5 | Setup data must be in RAM | Convention | Keep it on disk. Each proof reads t leaves. |
| A6 | SPARK commits per-entry values on both sides | **Convention.** The row weights χ_r are known before layer 1 starts. Only the column weights are late, and they depend only on a slot's position. | Commit per-run row values early. Aggregate runs to slots early (scatter). Aggregate slots to blocks late (second scatter, 2^20 values). |
| A7 | The 41-digit runs need special lookups | Convention. A run is a "slot" (start, len, ratio). A slot spans at most two blocks when len ≤ 54. | The lane split H0/H1 is a 54-entry table per slot class. The b+1 case is the denominator b̂ + 1 in the block scatter. No lookup. |
| A8 | T2 needs a per-lane carry DP | Convention. Long division by 54, read MSB first, gives the lane as the final remainder. | One DP with 54 remainders × 2 bound states, 3,626 transitions (counted). |
| A9 | Key and matrix terms need separate machinery | Partly | They share one setup tree, one set of query positions, one fold challenge α, and one late commitment P1. The norm, the row lookup and the slot scatter share one batched GKR. |
| A10 | Every commitment goes through p3-whir | Kept for per-proof data only | Two p3-whir proofs (P0 early, P1 late) and about 400 lines of honest-fold code |

The number of per-proof commitments is fixed by dependencies:

- z must come before β, λ and ζ.
- The folded setup oracles depend on α, which comes after s.
- Every committed value that a GKR tests must come before that GKR's β.
- A GKR that produces new setup claims must run before α.

A late GKR that touched setup data would therefore force a third commitment. The design
avoids this in two ways. Every late-GKR input is either per-proof (P0/P1) or a P0 copy
of setup data. That copy is checked against the setup at an early random point.

## Usage (caller's view)

README fragment (nightstream):

> Compression needs a one-time setup per circuit. `CompressionSetup::build` encodes the
> Ajtai key prefix and the circuit's matrix structure as Reed-Solomon codewords. It
> writes about 29 GB under a directory of your choice and returns a ~1.1 KB
> `CompressionKey`. The prover reads 125 small leaves per proof from that directory.
> Verifiers need only the key. A key is authoritative only if you derived it from the
> package (`CompressionKey::derive`, about 1 min, no disk) or pinned a copy you trust.
> Never take it from a prover.

```rust
// Once per circuit (prover machine).
let circuit = Circuit::load(&package_bytes)?;
let setup = CompressionSetup::build(&circuit, Path::new("target/app.compress"))?;
std::fs::write("app.key", setup.key().to_bytes())?;

// Prover, any later process.
let key = CompressionKey::from_bytes(&std::fs::read("app.key")?)?;
let setup = CompressionSetup::open(&circuit, Path::new("target/app.compress"), &key)?;
let prover = circuit.prover(Engine::Optimized, minimum_security_bits)?;
let finished = prover.finish_with_spartan(&proof, &setup)?;    // `proof` stays extendable

// Verifier: no SHAKE, no matrix rows, no setup files.
let key = CompressionKey::derive(&circuit)?;                    // or a pinned copy
let verifier = Verifier::from_package(&circuit, Engine::Optimized, minimum_security_bits)?;
verifier.verify_final(&expected_state, &key, &verifier.decode_final_proof(&bytes)?)?;
```

neo-spartan level (what `finish.rs` calls):

```rust
let setup = neo_spartan::Setup::build(&rows, public_blocks, dir)?;          // or Setup::open
let relation = neo_spartan::Relation::new(setup.key(), point_variables, norm_bound, bits)?;
let proof = neo_spartan::prove(&relation, &setup, transcript, &claim, &witness)?;
neo_spartan::verify(&relation, transcript, &claim, &proof)?;  // takes no MatrixRows: succinct by signature
```

## Shape

### Data (setup)

**Runs.** One run is one `GeometricRowRun`. Runs are sorted by (matrix, row, visit order)
and padded to N_R = 2^26 (38,533,993 used, 57%). Per run k, the setup holds:

- i_k: the row;
- σ_k: the slot;
- m_k: the initial coefficient.

The matrix j(k) is a step function with 4 segments (bounds in the key). Padding runs have
m = 0, i = 0, σ = 0.

**Slots.** One slot is a distinct (start, len, ratio) triple. The current package has the
classes (41, 3) and (1, 1). Slots are sorted by (class, lane, block, start) and padded to
2^21 (slot count est. 1.5–2.0 M; confirm by census). Per slot σ, the setup holds the
start block b_σ. The class and lane are step functions with ≤ 109 segments (bounds in the
key). A run longer than 54 coordinates is rejected with `Error::Shape`.

**Setup oracles S.** 26 multilinear oracles of 26 variables each:

- A_0..A_21: the key rows on the (b, l) cube, zero on lanes 54..63 and on blocks ≥ 814,144;
- i, σ, m: on runs;
- b': the slot blocks b, embedded in the low 21 variables.

Each oracle is encoded as the univariate polynomial with its monomial coefficients, on a
coset L with |L| = 2^27 (rate 1/2). The Merkle leaf j is the 8-point coset {y : y^8 = q_j}
of all 26 oracles (208 Goldilocks values), hashed with the workspace Poseidon2 sponge.
There are 2^24 leaves and the path depth is 24.

**Key** (verifier input, ~140 words): the S root, shape words, matrix run bounds, slot
segments (class, lane, start), the slot classes, the package structural identifier, and
the S parameters (k = 3, rate, t).

### Data (per proof)

| Commitment | When | Tables (base-field columns) | Cells |
| --- | --- | --- | --- |
| P0 (p3-whir, rate 1/2) | start of layer 1 | T_big 2^26 × {z, e_re, e_im}; T_slot 2^21 × {U_{j,re}, U_{j,im} (j < 4), b̂}; T_row 2^20 × {mult̂} | 2^27.6 → padded 2^28 |
| P1 (p3-whir, rate 1/8) | after the honest-fold challenge α | T_fold 2^23 × {g_A, g_R as 3+3 Ext coordinates}; T_blk 2^20 × {Ω^A as 3} | 2^25.6 → padded 2^26 |

- **e_k:** χ_r(i_k) ∈ K.
- **U_j(σ):** Σ_{k: j(k)=j, σ_k=σ} m_k e_k. It depends only on r, so it is early.
- **b̂, mult̂:** b̂ is a copy of the setup slot blocks. mult̂ holds the run counts per row;
  the prover chooses it, as logUp allows.
- **g_A, g_R:** the honest folds A_λ(α, ·) and O_R(α, ·).
- **Ω^A(b):** Σ_σ Ū(σ)·(Ĥ_0(σ)[b̂_σ = b] + Ĥ_1(σ)[b̂_σ + 1 = b]).
  - Ū(σ) = Σ_j (λ_{re,j}·Re U_j(σ) + λ_{im,j}·Im U_j(σ)).
  - Ĥ_t is the lane-split table of the slot's class at its lane (below).

### Load-bearing decisions

- **Honest fold, not a third WHIR.** This is per A4. The setup tree is never
  proximity-tested. One custom module (`setup.rs`) owns encoding, storage, the root,
  the per-proof fold and the query check. p3-whir stays unchanged for P0 and P1.
- **Two-level scatter, not SPARK.** This is per A6. Only one per-run per-proof column
  pair (e) is committed, and it is early. The late data is 2^20 Ext (Ω^A).
- **Copy, then check once.** b̂ lives in P0 and is checked against S at ρ[..21], an early
  random point. The late GKR therefore makes no setup claims, and the commitment count
  stays at two.
- **Step functions in the key, not oracles.** The matrix tag, slot class and lane are
  sort-order segments. The verifier evaluates them with interval sums: LT(ρ,a) costs
  O(n) per boundary.
- **The verifier takes only `&Key`.** `verify` has no `MatrixRows` parameter, so
  succinctness is a type-level fact.

What the design deliberately does not do:

- no proximity test on setup data;
- no per-run late values;
- no table-side multiplicity oracle (logUp multiplicities are the prover's choice);
- no change to M1's norm, quotient or linear sum-check.

**Interface depth.** neo-spartan's public surface grows by two types: `Key` (bytes in/out)
and `Setup` (`build`, `open`, `key`). It hides:

- the 26-oracle encoding, the disk layout and the tree;
- the run/slot sort policy and the two scatters;
- five GKR trees, the DP and the fold.

The nightstream surface grows by `CompressionSetup`, `CompressionKey` and `verify_final`.

### Module map (`crates/neo-spartan/src`)

| File | Owns | Lines now → M2 (est.) |
| --- | --- | --- |
| `lib.rs` | Contract header; `Key`/`Relation`/`Proof`/`prove`/`verify`; **the one transcript schedule**; the P0/P1 claim schedules; `outer_terms` | 438 → ~720 |
| `setup.rs` (new) | Honest setup oracles: the ordered oracle list, the monomial encoding, the coset-major disk layout, the streaming root (shared by `build` and `derive`), the prover fold (partials, g), the query read, the verifier leaf combine + fold + path check | ~560 |
| `matrix.rs` (new) | Run/slot tables from `MatrixRows`; e, U, Ū, Ω^A, per-coordinate u_A for the prover's quotient; the leaf formulas of the four matrix trees; the verifier's H tables and step-function MLEs | ~560 |
| `eval_k.rs` (new) | T2 by long division DP over c's bits, in K⊗Ext (verifier only) | ~110 |
| `gkr.rs` | Batched fraction GKR: trees of different depths, shared randomness, explicit-children leaf layers with product leaves (degree 4) | 209 → ~470 |
| `field.rs` | + `Kx` (K⊗Ext), the scaled-eq point, `lt` (interval MLE), identity MLE | 83 → ~180 |
| `pcs.rs` | Multi-table commitments, the rate per instance, budget split between P0 and P1 | 167 → ~260 |
| `ring.rs` | Prover rows and the quotient. It takes u_A from `matrix.rs`; the M1 `Scatter` over `MatrixRows` is deleted. It exposes the key part (A_λ) of its SHAKE pass. | 265 → ~250 |
| `hash.rs`, `sumcheck.rs`, `norm.rs` | + Merkle path check; + degree-4 rounds; unchanged | ~+70 |

nightstream changes:

- `lifecycle/finish.rs`: setup and key parameters, about +40 lines.
- `circuit.rs`: `CompressionSetup`, `CompressionKey`, `verify_final`, about +80 lines.

Every file stays under 1,500 lines.

### Type sketch

```rust
// lib.rs
/// Verifier-owned constants of one circuit's layer 1. Authority: derived from the package.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    root: [Gl; 4],
    shape: Shape,                     // blocks, rows, matrices, public blocks, padded runs/slots
    run_bounds: [u32; MATRICES + 1],  // matrix segments over runs
    slot_segments: Vec<SlotSegment>,  // (class, lane, first slot), slot order
    classes: Vec<SlotClass>,          // (len ≤ 54, ratio)
    package: [u64; 4],
}
impl Key {
    pub fn derive(rows: &dyn MatrixRows, public_blocks: usize) -> Result<Self, Error> { unimplemented!() }
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
    fn words(&self) -> Vec<Gl> { unimplemented!() }
}

/// Prover-side setup: the key, in-RAM run/slot tables, the on-disk oracle store.
pub struct Setup { key: Key, tables: matrix::Tables, store: setup::Store }
impl Setup {
    pub fn build(rows: &dyn MatrixRows, public_blocks: usize, dir: &Path) -> Result<Self, Error> { unimplemented!() }
    /// Reopen files built for `key`; checks the stored root equals `key.root` (not authority).
    pub fn open(rows: &dyn MatrixRows, dir: &Path, key: &Key) -> Result<Self, Error> { unimplemented!() }
    pub fn key(&self) -> &Key { &self.key }
}

pub struct Relation<'k> { key: &'k Key, shape: Shape, p0: Pcs, p1: Pcs, queries: usize }
impl<'k> Relation<'k> {
    pub fn new(key: &'k Key, point_variables: usize, norm_bound: u32, security_bits: f64) -> Result<Self, Error> { unimplemented!() }
    pub fn security_bits(&self) -> f64 { unimplemented!() }
}

pub fn prove(relation: &Relation<'_>, setup: &Setup, transcript: Poseidon2Transcript,
             claim: &Claim, witness: &Mat<F>) -> Result<Proof, Error> { unimplemented!() }
pub fn verify(relation: &Relation<'_>, transcript: Poseidon2Transcript,
              claim: &Claim, proof: &Proof) -> Result<(), Error> { unimplemented!() }

#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    p0: pcs::Commitment, histogram: Vec<u32>,
    early: GkrProof, early_values: matrix::EarlyValues,   // leaf polys at the leaf points
    quotient: Vec<Ext>, linear: Vec<Vec<Ext>>,
    finals: [Ext; 3],                                      // v_A, v_Ω, v_z
    partials: [[Ext; 8]; 2],                               // honest-fold partials (A, R)
    p1: pcs::Commitment,
    queries: Vec<setup::Leaf>,                             // t leaves + paths (not observed)
    late: GkrProof, late_values: matrix::LateValues,
    p1_opening: pcs::Opening, p0_opening: pcs::Opening,
}

// setup.rs
pub(crate) const FOLD: usize = 3;                          // variables folded by the honest fold
pub(crate) enum Oracle { KeyRow(u8), RunRow, RunSlot, RunCoefficient, SlotBlock } // the one list
pub(crate) struct Store { dir: PathBuf, layers: Vec<Vec<[Gl; 4]>> }
pub(crate) struct Leaf { index: u32, values: Vec<Gl>, path: Vec<[Gl; 4]> }
pub(crate) struct Group { point: Vec<Ext>, weights: Vec<(Oracle, Ext)> } // A: λ^i; R: 1, μ, μ², μ³
pub(crate) fn root(oracles: &mut dyn FnMut(Oracle) -> Vec<Gl>, sink: Option<&Path>) -> Result<[Gl; 4], Error> { unimplemented!() }
pub(crate) fn partials(group: &Group, table: &[Ext]) -> [Ext; 1 << FOLD] { unimplemented!() }
pub(crate) fn fold(table: &[Ext], alpha: &[Ext; FOLD]) -> Vec<Ext> { unimplemented!() }
pub(crate) fn read(store: &Store, indices: &[u32]) -> Result<Vec<Leaf>, Error> { unimplemented!() }
/// Path check, group combination, inverse DFT of 8, monomial combination with α.
pub(crate) fn check(key_root: &[Gl; 4], leaf: &Leaf, groups: &[Group; 2], alpha: &[Ext; FOLD]) -> Result<[Ext; 2], Error> { unimplemented!() }

// matrix.rs
pub(crate) struct Tables { row: Vec<u32>, slot: Vec<u32>, coefficient: Vec<Gl>, block: Vec<u32> }
pub(crate) struct Early { e: [Vec<Gl>; 2], u: [Vec<Gl>; 8], mult: Vec<Gl>, block_copy: Vec<Gl> }
impl Tables {
    pub(crate) fn from_rows(rows: &dyn MatrixRows, shape: &Shape) -> Result<(Self, Vec<SlotSegment>, Vec<SlotClass>), Error> { unimplemented!() }
    pub(crate) fn early(&self, point: &[K]) -> Early { unimplemented!() }
    pub(crate) fn column_weights(&self, early: &Early, mixing: &Mixing) -> Vec<Ext> { unimplemented!() } // u_A per coordinate, for ring.rs
    pub(crate) fn block_weights(&self, early: &Early, mixing: &Mixing, zeta: Ext) -> Vec<Ext> { unimplemented!() } // Ω^A
}
pub(crate) fn lane_split(key: &Key, zeta: Ext) -> Vec<[[Ext; D]; 2]> { unimplemented!() } // H0/H1 per class
pub(crate) fn step(key: &Key, values: &[Ext], point: &[Ext]) -> Ext { unimplemented!() } // Σ v_s (LT(a_{s+1}) − LT(a_s))

// eval_k.rs
pub(crate) fn eval_k(blocks: usize, r: &[K], s_b: &[Ext], tau: &[Ext; D], weights: [Ext; 2]) -> Ext { unimplemented!() }

// gkr.rs
pub(crate) trait Leaves: Sync {          // one per tree; the only GKR extension point
    fn depth(&self) -> usize;
    fn degree(&self) -> usize;           // 1 (linear leaves) or 2 (product numerator)
    fn fraction(&self, index: usize) -> (Ext, Ext);
    fn children(&self, point: &[Ext]) -> Vec<Ext>;  // explicit leaf-poly values at (point, 0/1)
}
pub(crate) fn prove(trees: &[&dyn Leaves], challenger: &mut Challenger) -> (GkrProof, Vec<Vec<Ext>>) { unimplemented!() }
pub(crate) fn verify(proof: &GkrProof, depths: &[usize], challenger: &mut Challenger) -> Result<Vec<LeafClaim>, Error> { unimplemented!() }
```

`Leaves` is a real shared capability. Five early trees and two late trees conform to it.

## Protocol

### Formulas the verifier evaluates (no setup reads)

- **Scaled-eq point** (verified). Take κ_ζ = Π_{t<6}(1 + ζ^{2^t}) and
  x_t = ζ^{2^t}/(1 + ζ^{2^t}). Then Σ_{l<64} ζ^l f(l) = κ_ζ·f~(x_ζ) when f is zero on
  lanes ≥ 54. If some 1 + ζ^{2^t} = 0, the verifier rejects; this is a completeness
  loss only, ≤ 63/|Ext|.
- **T1** = κ_ζ·v_A, where v_A = Ã_λ(x_ζ, s_b) is checked by the honest fold.
- **τ_l** = Σ_p bar[p][l]·ζ^p, so that ω_b(ζ) ⊇ <w_b, τ> (M1, ring.rs).
- **Lane split** (verified for len ∈ {41, 1, 54, 17}). For class (len ≤ 54, ratio):
  - H0(l) = Σ_{k<len, l+k<54} ratio^k·τ_{l+k}, by the recursion
    H0(53) = τ_53 and H0(l) = τ_l + ratio·H0(l+1) − [l+len ≤ 53]·ratio^len·τ_{l+len};
  - H1(l) = ratio^{54−l}·P(l+len−55), where P(m) = Σ_{k≤m} ratio^k τ_k (0 if m < 0).
  - O(54) per class. Then Σ_{k<len} ratio^k g(54b+l+k) = E(b)·H0(l) + E(b+1)·H1(l)
    for g(c) = E(⌊c/54⌋)·τ_{c mod 54}.
- **T2** (verified against brute force for blocks ∈ {37, 50, 64}; 3,626 transitions at
  production size, counted). T2 = Σ_{b<blocks} eq(s_b,b)·Σ_l τ_l·π_K(χ_r(54b+l)).
  - The state is (remainder ρ ∈ [0,54), flag ∈ {equal, less} for q against blocks).
    It reads the bits of c from bit 25 down to bit 0.
  - Each step does ρ' = 2ρ + c_t and q_t = [ρ' ≥ 54], with ρ' −= 54·q_t. It multiplies by
    eq(r_t, c_t)·eq(s_{b,t}, q_t) in K⊗Ext, with q_t forced to 0 for t ≥ 20.
  - At the end it keeps flag = less, weighs each remainder by τ_ρ, and multiplies by
    Π_{t≥26}(1 − r_t).
  - T2 = λ_re·A + λ_im·B for the K⊗Ext result A + B·u.
- **T4** = Σ_{j<5} eq(s_b, j)·λ^{pub_j} (M1 `Mixing::public`).
- **χ~_r over rows** at ρ ∈ Ext^20: Π_{t<20}((1−r_t)(1−ρ_t) + r_t ρ_t)·Π_{t≥20}(1−r_t), in
  K⊗Ext. Re and Im are F-linear, so they commute with the MLE.
- **Identity:** id~(ρ) = Σ_t 2^t ρ_t.
- **Step function** with sorted bounds a_s (verified):
  f~(ρ) = Σ_s v_s·(LT(ρ,a_{s+1}) − LT(ρ,a_s)), where
  LT(ρ,a) = Σ_{t: a_t=1} (1−ρ_t)·Π_{t'>t} eq(ρ_{t'}, a_{t'}) and LT(ρ, 2^n) = 1.
  Used for j(k) (5 bounds over 26 variables) and for Ĥ_t (≤ 109 bounds over 21 variables).
- **Embedding** (verified): b~'(ρ) = b~(ρ[..21])·Π_{t=21}^{25}(1 − ρ_t).
- **Honest fold** (verified):
  - Õ(x) = Σ_{u<8} eq(x[..3],u)·w_u, with w_u = Õ(u, x[3..]).
  - For g = O(α,·): g~(x[3..]) = Σ_u eq(α,u)·w_u.
  - From the coset values Ô(y_0 ω_8^v), the fold value is
    ĝ(y_0^8) = Σ_u α^[u]·Ô_u(y_0^8), where Ô_u(y_0^8) = 8^{-1} y_0^{-u} Σ_v ω_8^{-uv} Ô(y_0 ω_8^v)
    and α^[u] = Π_t α_t^{u_t}.
  - The claim on P1 is the eq claim g~(q, q², q⁴, …) = ĝ(q) with q = y_0^8.

### Lookup and scatter instances

All four instances are logUp fraction sums (Haböck 2022/1530) proven by GKR
(Papini–Haböck 2023/1284). In each, "P0", "P1" or "S" names where the claim at the leaf
point is opened.

| Instance | Item side, leaves (p, q) | Other side, leaves (p, q) | MLEs the verifier evaluates |
| --- | --- | --- | --- |
| Norm (M1) | x ∈ cube 2^26: (1, β_N − z) [P0] | public histogram over [−H, H], summed directly | none |
| Row lookup | k ∈ runs 2^26: (1, β_L − (i_k + γ e_re + γ² e_im)) [S: i; P0: e] | i ∈ rows 2^20: (mult̂(i), β_L − (i + γ Re χ_r(i) + γ² Im χ_r(i))) [P0: mult̂] | id~, χ~_r in K⊗Ext |
| Slot scatter | k ∈ runs 2^26: (m_k·(e_re + μ_S e_im), β_S − (σ_k + 2^21 j(k))) [S: m, σ; P0: e] | (σ, j) ∈ 2^23: (U_{j,re}(σ) + μ_S U_{j,im}(σ), β_S − (σ + 2^21 j)) [P0: U] | j~ (step), id~ |
| Block scatter (late) | (σ, t) ∈ 2^22: (Ū(σ)·Ĥ_t(σ), β_B − (b̂(σ) + t)) [P0: U, b̂] | b ∈ 2^20: (Ω^A(b), β_B − b) [P1] | Ĥ_t (step over H tables), id~ |

The root checks are:

- norm: P_N = Q_N·Σ_t m_t/(β_N − t);
- each other pair: P_x·Q_y = P_y·Q_x, with every Q ≠ 0.

The tags σ + 2^21 j and b + t are exact integers below p, so they need no compression
challenge.

### Prover and verifier steps

1. **Bind.** Hand off the transcript (M1 T0, domain `…/compress/v2`). Observe the shape,
   the P0/P1/S profile words, the key words and the statement.
2. **Commit P0.** The prover builds z, e, U, b̂ and mult̂, then commits P0. The verifier
   observes the root and the OOD answers through p3.
3. **Histogram.** Observe it, then draw β_N, β_L, γ_L, β_S and μ_S.
4. **Early batched GKR** over five trees: N (26), L (26), S (26), U (23), T (20).
   - At each layer, draw a tree-batching ν and the layer μ.
   - Rounds have degree 3, or 4 at a product leaf layer.
   - A leaf layer sends its leaf polynomials at both children; τ then reduces them.
   - The leaf claims, at ρ (shared by N, L, S), ρ_U and ρ_T, are:
     z, e_re, e_im, i, σ, m at ρ; U×8 at ρ_U[..21]; mult̂ at ρ_T; b̂ at ρ[..21] (the copy).
   - The verifier checks every leaf formula with the MLEs above.
5. **Ring rows and linear sum-check** (M1 T6–T9): λ; Q; ζ; 26 rounds → s.
   - The prover's ω_b comes from `ring.rs`, using u_A from `matrix.rs`.
   - One SHAKE pass also yields A_λ.
6. **Finals.** The prover sends v_A, v_Ω = Ω~^A(s_b) and v_z = z~(s). The verifier checks
   last = (κ_ζ v_A + T2 + v_Ω + T4)·Λ~(s_l)·v_z.
7. **Honest fold.**
   - Draw μ_R. The verifier forms v_R = ĩ + μ_R σ̃ + μ_R² m̃ + μ_R³·b̂(ρ[..21])·Π_{t≥21}(1 − ρ_t)
     from the step-4 values.
   - The prover sends w_{A,u} and w_{R,u} for u < 8. The verifier checks
     Σ_u eq(x_A[..3],u) w_{A,u} = v_A, with x_A = (x_ζ, s_b), and
     Σ_u eq(ρ[..3],u) w_{R,u} = v_R.
   - Draw α ∈ Ext³.
8. **Commit P1** = {g_A, g_R, Ω^A}, with root and OOD through p3.
9. **Setup queries.** Draw t leaf indices in [0, 2^24). The prover reads the leaves and
   paths from disk. The verifier checks each path against `key.root`, combines the groups
   (λ^i for A; 1, μ_R, μ_R², μ_R³ for R) and folds. This gives y_{A,j} and y_{R,j}.
   Leaves are not observed: the root already binds them.
10. **Late batched GKR.** Draw β_B. Run two trees: B (22) and Ω (20). The leaf claims are
    U×8 and b̂ at y_B (P0), and Ω^A at ρ_Ω (P1).
11. **Open P1** with p3 `open_at`:
    - g_A at x_A[3..] = Σ eq(α,u) w_{A,u};
    - g_R at ρ[3..] = Σ eq(α,u) w_{R,u};
    - (g_A, g_R) at t power points = (y_{A,j}, y_{R,j});
    - Ω^A at s_b = v_Ω and at ρ_Ω.
12. **Open P0** with p3 `open_at`:
    - {z, e_re, e_im} at ρ;
    - z at s = v_z;
    - U×8 at ρ_U[..21] and at y_B;
    - b̂ at ρ[..21] and at y_B;
    - mult̂ at ρ_T.

The verifier accepts if every equality above holds.

### Transcript schedule (one sequence, in `lib.rs`)

| # | Step |
| --- | --- |
| L0 | M1 handoff; domain `Nightstream/SuperNeo/compress/v2`; 4-word seed |
| L1 | observe shape words, P0/P1/S profile words, key words, statement words |
| L2 | P0 commit (p3: root, OOD) |
| L3 | observe histogram; draw β_N, β_L, γ_L, β_S, μ_S |
| L4 | early GKR: 5 roots; per layer ν_k, μ_k, rounds, children, τ_k |
| L5 | observe early leaf values (z, e_re, e_im, i, σ, m, U×8, mult̂, b̂) |
| L6–L9 | λ; observe Q (53 Ext); ζ; linear sum-check (26 rounds) |
| L10 | observe v_A, v_Ω, v_z |
| L11 | draw μ_R; observe 2×8 partials; draw α (3 Ext) |
| L12 | P1 commit (p3: root, OOD) |
| L13 | draw t indices of 24 bits (`sample_bits`) |
| L14 | draw β_B; late GKR: 2 roots, layers, children; observe late leaf values |
| L15 | p3 `open_at` P1 (7 + t batches) |
| L16 | p3 `open_at` P0 (7 batches) |

## Soundness map

The extractor gets z, e, U, b̂ and mult̂ from P0, and g and Ω^A from P1. Each check then
pins the next fact:

- the lookup gives e = χ_r(i);
- the slot scatter gives U = M′ᵀχ_r;
- the copy check gives b̂ = b;
- the block scatter gives the true Ω^A;
- the queries give g = the true folds, so v_A is true;
- T2 and T4 are exact.

So Ω~(s_b) is true, and the M1 argument (λ, quotient, ζ, linear sum-check, logUp norm)
gives all five CE(B) conjuncts.

Each outer term below is a degree over |Ext| ≥ 2^191. Each is charged over both lists,
L_0·L_1 ≤ 2^(4.3+6.3) at JB (p3: log L = log(1/ρ) + log 20 − 1).

| Term | Degree / probability | Bits after charge (est.) |
| --- | --- | --- |
| Norm logUp at β_N (M1) | 2^26 + 7,345 | ~154 |
| Row lookup at (β_L, γ_L) | 3·(2^26 + 2^20) | ~153 |
| Slot scatter at (β_S, μ_S) | 2^26 + 2^23 + 1 | ~154 |
| Early GKR (26 layers, degree ≤ 4, 5 trees) | Σ_k (4k + 8) ≈ 1,600 | ~170 |
| λ, quotient at ζ, linear (M1) | 36 + 106 + 52 | ~172 |
| μ_R group combination; α fold (2 groups) | 3 + 6 | ~177 |
| Copy check b̂ vs b at ρ[..21] | 21 (inside μ_R) | ~176 |
| Block scatter at β_B | 2^22 + 2^20 | ~158 |
| Late GKR (22 layers) | Σ_k (4k + 4) ≈ 1,100 | ~170 |
| **Setup queries** | 2·(1/2)^t (each group: deg 2^23 on 2^24 points), charged over L_1 only | t − 1 − 6.3 |
| P0 WHIR, P1 WHIR | p3 `prescribed_security` (OOD, folds, queries, initial claims) | derived |
| Poseidon2 Merkle (S, P0, P1) and sponge | 4-word digests: 2^-128 classical, as today | n/a |

**Budget.** Let β = `compression_security_bits()` (≈ 114 after layer 0). The parts are:

- P0's report targets β + 1.6.
- P1's report targets β + 1.6. It includes every row above and the query term.
- So t = ⌈β + 1.6 + 1 + log L_1⌉, derived in `Relation::new`; it is not a constant.
  For β = 114 this gives 123. The cost rows round up to 125.
- The sum is ≤ 2·2^-(β+1.6) < 2^-β.

`Error::Budget` is returned when no level reaches the target. PoW stays 0 until the
owner sets it.

**One new argument needs review.** The honest fold's soundness rests on three facts:

1. The setup codeword is exact.
2. Two distinct polynomials of degree < 2^23 agree on < 1/2 of a 2^24-point domain.
3. P1's OOD binds one candidate before the queries.

It is WHIR's fold (2024/1586) applied to a preprocessed honest codeword, the "setup
offloading" idea of Akita (2026/1983). The argument is three lines, but no published
paper states it for WHIR with an external round-1 commitment. A reviewer must confirm it.

## Costs (all est.; this Mac, CPU, production package)

| Item | Estimate | Driver |
| --- | --- | --- |
| Setup time | ~1–1.5 min | 26 × (Möbius + NTT of 2^27) ≈ 35 s; 2^24 leaves × 18 perms ≈ 320 M perms; SHAKE ≈ 3 s |
| Setup storage | 26 GiB codewords + 1 GiB tree (~29 GB) | 23.6 GB of it is the key, circuit-independent |
| Key derivation for a trustless verifier | same compute, no disk, ~3 GiB RAM | streaming sponge state per leaf |
| Prover RAM peak (finish) | ~35 GiB | baseline ~10, P0 data/codeword/tree ~8, early GKR trees ~15 (freed), P0 round-1 codeword 6 |
| Prover time, layer 1 | ~25–40 s; finish ~35–55 s | P0 commit (2^28 cells) 5–10 s, early GKR 5–8 s, M1 ring/linear ~5 s, P1 commit 2–4 s, openings 6–8 s, fold ~1 s, disk reads ~0.1 s |
| Layer-1 proof | ~0.73 MB | P0 opening ~250 KB, setup leaves+paths ~300 KB, P1 ~100 KB, GKR ~45 KB, histogram 29 KB |
| Verifier time | ~50 ms | p3 verifies, 125 × 42 perms, DP, GKR |
| M3: Poseidon2 perms | ~20.8 k (≈1.6 M rows at ~76 rows/perm) | P0 9.7 k, setup 5.25 k (125 × (18 leaf + 24 path)), P1 4.3 k, transcript 1.5 k |
| M3: field arithmetic | ~0.65 M base mults (≈0.65 M rows) | WHIR folds/weights ~130 k, T2 DP 3,626 × 27 ≈ 98 k, leaf combine 125 × 208 Ext×F ≈ 78 k, GKR ~90 k, histogram ~66 k, step MLEs ~45 k |

M3 total, layer 1: ≈ 2.2 M rows. The owner's earlier estimate was 6.2 M.

**Tuning knobs, each set by measurement, not by this design:**

- **The fold width k** (k = 3 chosen over 4 and 2). Setup plus P1 costs 8.9 k perms at
  k = 3, against 10.7 k at k = 4 and 9.4 k at k = 2.
- **The P0 rate** (1/2 from M1). At rate 1/4, P0 needs 125 instead of 271 round-0 queries
  (−3.5 k perms), but its codeword doubles to 8 GiB.
- **PoW.**

## Test plan

All tests live in `tests/` directories, use `--release`, and stay under 300 s each.

**neo-spartan unit tests** (toy sizes, each < 10 s):

1. `formulas.rs` ports the numeric checks:
   - scaled-eq point; H tables against brute force for (41,3), (1,1), (54,r) and random len;
   - T2 DP against brute force (blocks 37/50/64, K-valued r);
   - step MLE; embedding; coset fold; partial-evaluation identities.
   - Each check has a negative control (bound off by one, H1 exponent 53−l, an unscaled
     point) that must fail.
2. `setup_oracle.rs` builds 4 oracles of 2^10 in a temp dir and runs fold and queries.
   Each of these must be rejected: one wrong partial, one wrong coefficient of g in P1,
   one flipped leaf value, one wrong path node, a wrong key root.
3. `gkr_batched.rs` uses depths (8, 8, 6, 5) with linear and product leaves. It checks
   roots and leaves against direct sums, then tampers each round value and each child.
4. `matrix.rs` builds a toy MatrixRows: 32 rows, 4 matrices, both slot classes, slots that
   spill into b+1, and repeated (row, slot) pairs.
   - Ω^A and u_A must equal M1's Eval_A weights. Keep M1's scatter as a test-only oracle.
   - Changing one e, one U, one b̂ or one Ω^A entry must reject at the matching root or
     leaf check.
   - Removing H1 must fail (spill coverage).
5. `prove_verify.rs` uses a synthetic CE(B) claim, the real key prefix for 64 blocks and
   the toy matrices.
   - Every Proof field is mutated: roots, GKR messages and children, v_A/v_Ω/v_z,
     partials, query leaf/path/index, both openings.
   - Statement parts are mutated: c row, X, Eval_K, Eval_A.
   - False witnesses: one z changed; one z outside [−H, H].
   - Every mutation must be rejected.

**nightstream tests:**

6. Toy finish on both parity fixtures with a temp-dir setup.
   - Layer-0 and layer-1 mutations must be rejected.
   - A key from the other fixture must be rejected.
7. `#[ignore]` production tests, each < 300 s:
   - (a) Build the setup into `target/compression-setup/` (est. 1–1.5 min) and compare it
     with `Key::derive`.
   - (b) Run compile + base + 1 fold + finish + `verify_final` with the stored setup
     (est. 1.5–2 min). It prints the finish time, verify time and proof bytes.
   - Peak RSS is measured with `/usr/bin/time -l` and must stay < 64 GB.

## Tradeoffs accepted

- **Setup disk.** We accept ~29 GB of setup files per circuit in exchange for:
  - setup data that is never in RAM;
  - 125 small reads per proof;
  - no proximity proof on setup data.
- **One new argument.** We accept the honest fold, about 400 lines of new protocol code,
  in exchange for removing one WHIR proof and a 2^31 commitment.
- **Bigger P0.** P0 grows to 2^28 (z plus per-run e) in exchange for keeping two
  per-proof commitments and a standard logUp lookup.
- **K-valued U.** We accept 8 U columns per slot (4 matrices × Re/Im, before λ) in
  exchange for computing U before λ exists.
- **Layer-1 proof size.** We accept a larger layer-1 proof (~0.73 MB) in exchange for a
  smaller M3 circuit. M3 hashes, not bytes, set the final size.
- **The 54-coordinate run limit.** We accept rejecting runs longer than 54 coordinates in
  exchange for a two-block lane split. Today's maximum is 41.
- **Layer 0 needs the package.** `verify_final` still needs the package for layer 0.
  Succinct layer-1 verification needs only the key.

## Alternatives considered

1. **p3-whir unchanged, with setup commitments as ordinary WHIR (stance A).**
   - The key needs ~48 GiB of prover data at 2^31 cells, plus a 2^27-Ext round-1
     codeword: over the cap.
   - Splitting the key into 22 commitments of 2^26 means 22 WHIR proofs, about 200 k perms
     in M3.
   - The public surface is the same, but the RAM cost leaks to every caller. Rejected.
2. **One batched opening with a custom streaming WHIR (stance B).**
   - P0, S and P1 would each give a round-0 fold into one shared round-1 commitment.
   - It saves one WHIR tail: about 3–4 k perms (≈15–20% of M3) and ~150 KB.
   - The cost is 1.5–2 k lines of security-critical WHIR code that duplicates p3: OOD,
     STIR, the final round, and our own security accounting.
   - The internals are deeper, but they would hide p3's logic behind our own. Rejected
     for M2; revisit only if the M3 size target needs it.
3. **Standard SPARK.**
   - It needs per-run e_col committed late (3 × 2^26) and a column table of 2^26 (b,l)
     entries with setup multiplicities.
   - Per-run block and lane copies (2 × 2^26) avoid a third commitment.
   - The matrix part costs ~2× the prover work and a larger P1. Rejected.
4. **A flat binary c-cube (Spartan layout).**
   - T2 and the matrix columns become eq weights.
   - The key weight becomes non-tensor, so the setup claim is a general linear form. That
     needs a 30-variable sum-check over the key every proof. Rejected.
5. **Three commitments, no e (eq-mass row scatter).** W(i) = Σ_{k∈row i} eq(ρ,k)·m_k is
   committed after ρ.
   - P0 halves to 2^27.
   - The cost is one more small WHIR (~2.5 k perms), one more GKR and a dense row
     sum-check.
   - Kept as a fallback if the P0 commit dominates when measured.
6. **Fixed-width runs per (matrix, row).** e becomes virtual (χ~_r of the high bits), and
   the row lookup and 2 × 2^26 P0 cells go away. It works only if every (matrix, row) has
   ≤ 32 runs. See open question 4.

## Open questions and risks

1. **Setup storage.** Is ~29 GB of setup files per circuit acceptable?
   - Alternatively, a separate key tree could be shared across circuits. It costs one
     more path per query (+3 k perms in M3).
2. **Proof-of-work.** Do you set PoW bits for the setup queries (t ≈ 125) and for P0
   (271 round-0 queries at rate 1/2)?
3. **Tuning order.** Should k, the P0 rate and the P1 rate be fixed after measuring
   Poseidon2 Merkle throughput? That throughput is the largest risk in every prover-time
   estimate here.
4. **Row profile.** What is the largest run count per (matrix, row)? If it is ≤ 32,
   alternative 6 removes e and the row lookup.
5. **Review of the honest fold.** Will a reviewer confirm the honest-fold soundness
   argument? It is the one new argument.
6. **API.** Is `verify_final(&state, &key, &proof)` acceptable? It replaces M1's sealed
   `verify` for `FinalProof`, because the key must be explicit.
7. **Key authority for native verifiers.** Is it `Key::derive` (≈1 min), a pinned key, or
   both? M3 will pin the key in its circuit.
8. **Long runs.** Is it acceptable to reject packages whose runs exceed 54 coordinates?
9. **Memory risk.** The early GKR holds ~15 GiB at its peak. If the measured peak nears
   50 GB, merge trees L and S into one tree with two fractions per run (−6 GiB).

## Verification record

| Formula | Status | How |
| --- | --- | --- |
| Scaled-eq point for ζ^l over 6 lane bits (54 of 64 used) | verified | Goldilocks, random f, against direct Σ; control: an unscaled point fails |
| H0/H1 split and both O(54) recursions | verified | all run starts in 9 blocks, len ∈ {41,1,54,17}, against direct Σ_k ratio^k g(c+k); control: exponent 53−l fails |
| T2 division DP (remainder × bound flag, K-valued r, zero high bits) | verified | blocks ∈ {37,50,64}, 6-variable s, 12-bit c, against brute force; control: bound+1 fails. Production count: 3,626 transitions, 108 final states |
| runs → slots → blocks equals M1's Eval_A block weight | verified | random slots of both classes, 300 runs, 7 blocks |
| Block-scatter fraction identity (b and b+1 denominators) | verified | random β |
| Step MLE via LT interval sums | verified | 10 segments over 8 variables |
| Zero-high embedding factor | verified | 4 low, 3 high variables |
| Multilinear at (q, q², …) equals the monomial univariate; 8-point coset fold of the low variables; partial-evaluation and folded-claim identities | verified | n = 7, k = 3, rate-1/2 Goldilocks coset |
| JB queries: 271 at rate 1/2, 125 at 1/4, 82 at 1/8, 61 at 1/16, for 116 bits | computed | p3 formulas (√ρ + √ρ/20 per query; log L = log(1/ρ) + log 20 − 1) |
| WHIR round counts, M3 perms, RAM, times | not verified | estimates |

The script is in the scratchpad at `m2c-checks/check_formulas.py` (python3, < 1 s).

## Next implementation step

1. Port the numeric checks into `crates/neo-spartan/tests/internal/formulas.rs`.
2. Build `setup.rs` on toy oracles: encode, disk store, streaming root, partials, fold,
   query read, verifier check. Include its round-trip and tamper tests.

`setup.rs` is the one new primitive. Its measured NTT and Poseidon2 Merkle throughput
fixes k, t and both rates before the GKR and matrix code is written.
