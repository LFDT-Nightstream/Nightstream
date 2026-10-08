# Compression milestone 3, candidate C: one shrink layer over one generic verifier

Date: 2026-10-07. Base: branch `claude/compression-spartan-whir` at dd8a38805 (M2).
Seat stance: none fixed. The shape is derived from first principles.
Marks: "est." is an estimate. "model" is the calibrated query model in
[Cost model](#cost-model-and-calibration). "measured" comes from GROUNDING.md.
Nothing in this file is measured by this candidate.

Owner targets used (2026-10-07 updates): final proof **at most 200 KB (hard)**, goal
**at most 100 KB**; native verifier **about 100-400 ms** on one CPU, which may regenerate
or evaluate part of the circuit; about 114 proven bits (Johnson bound); PoW of at most
20 bits on the last layer only.

## Summary

- **One shrink layer.** It proves "the M2 final verifier accepts" with SuperSpartan
  (ePrint 2023/552) over a CCS whose Poseidon2 S-box is one degree-7 term. The witness
  is committed **once** with p3-whir 0.8 at rate 1/64, PoW 20, folding schedule
  [5, 4, 4, 4].
- **M2 stays as it is.** Only the security budget is split between M2 and the shrink.
  A smaller circuit hardly changes the shrink proof size, so a retune of M2 buys only
  shrink prover time, and it costs M2 prover memory and setup disk.
- **One verifier, written once.** The final verifier (layer 0 replay plus layer 1) is
  generic over a `Backend`. `Native` is the plain M2 verifier (`verify_final`).
  `Recorder` emits the CCS rows and, with a real proof, the witness. The shrink
  verifier re-runs the same code to regenerate the rows. There is no second circuit
  implementation.
- **Structured where it is big, explicit where it is small.** The ~32.5k Poseidon2
  permutations sit in a uniform region whose matrix MLEs have closed forms (this is
  the AIR case of SuperSpartan). All other constraints ("glue", about 1M rows) are
  plain sparse rows. The native verifier streams them at the sum-check point.
- **Estimates.** Proof ≈ **83 KB** (≤ 100 KB goal). Native verifier ≈ **120-170 ms**,
  key ≈ 3 KB. Shrink prover ≈ 23 s, peak ≈ 9 GB, after the M2 finish (39.7 s, 37.3 GB).

## Problem

M3 must turn a `FinalProof` (1,600,157 B) into a proof of at most 200 KB (goal 100 KB)
that a native verifier checks against the expected `State` and a trusted key. The
verifier must take about 100-400 ms on one CPU. The proof must be post-quantum, with
about 114 proven bits, Poseidon2-only hashing and existing published techniques.

Four facts make the shape non-obvious.

1. **The measured 23,589 permutations are a pruned count.** p3-whir 0.8 sends pruned
   Merkle multiproofs (`p3-merkle-tree` `pruning.rs`). Which internal nodes are shared
   depends on the query values, so a fixed-shape circuit cannot follow the pruned walk.
   Checked per query, the M2 layer-1 verifier costs **≈ 29.0k** permutations (model;
   the same model gives the measured pruned counts within 5%). Layer 0 adds ≈ 3.3k
   (state hash 2,319; PiCCS 949; ρ sampling 34; handoff 2). The circuit has
   **≈ 32.5k permutations**.
2. **Proof bytes depend on the WHIR rate and round count, not on the circuit size.**
   One more witness variable adds ≈ 1-3 KB. One more WHIR round adds ≈ 10 KB. At 2^23
   cells the default constant-4 folding schedule needs 5 folds instead of 4. A first
   fold of 5 removes that round (77.5 KB instead of 87.7 KB, model, rate 1/64).
3. **Someone must evaluate the circuit's matrices at a random point.** The Poseidon2
   part has ≈ 3.3k nonzeros per permutation (≈ 107M in total). It must be structured.
   The rest is heterogeneous verifier arithmetic. Holography for it needs a second
   per-proof commitment (Spark e-values, or the folded oracle of an honest fold), which
   costs 60-85 KB. The owner's 100-400 ms verifier budget lets the native verifier
   regenerate and stream that part instead.
4. **PiRLC is the largest arithmetic in layer 0.** Exact ring products of 17 claims
   (commitments, public inputs, evaluation families) cost ≈ 1.8M base products. A
   random evaluation of the same ring identity costs ≈ 0.12M.

Constraints honored: Poseidon2-only (GROUNDING §0), Johnson bound, PoW ≤ 20 on the last
layer only, 64 GB cap, files < 1,500 lines, tests ≤ 300 s in `tests/`, no new Rust
features or environment variables, folding protocol and b = 2, k_rho = 16 unchanged,
M2's soundness argument intact.

## Usage (caller's view)

README fragment:

```text
Compression has three steps. finish_with_spartan (M2) turns a folded proof into a
FinalProof (about 1.6 MB). shrink (M3) turns a FinalProof into a ShrinkProof
(about 83 KB). A chain checks a ShrinkProof with the CompressionKey alone: it needs
neither the package nor the setup files.
```

Prover node:

```rust
use nightstream::{Circuit, CompressionKey, Engine, Prover, Verifier};

let circuit = Circuit::load("app.nspkg")?;
let prover = circuit.prover(Engine::Cpu, 114)?;
let key: CompressionKey = Verifier::from_package(&circuit, Engine::Cpu, 114)?.compression_key()?;
let setup = prover.open_compression_setup("setup/", &key)?;          // M2, unchanged

let final_proof = prover.finish_with_spartan(&envelope, &setup)?;   // M2: about 1.6 MB
let shrink = prover.shrink(&key, &final_proof)?;                    // M3: about 83 KB
publish(shrink.to_bytes());
```

Chain node (no package, no setup files):

```rust
use nightstream::{CompressionKey, ShrinkProof, State};

let key = CompressionKey::from_bytes(PINNED_KEY)?;                  // about 3 KB, strict
let proof = ShrinkProof::from_bytes(&bytes)?;                        // strict canonical bytes
key.verify(&expected_state, &proof)?;                                // about 120-170 ms
```

Off-chain check of an M2 proof (unchanged call, now package-free inside):

```rust
verifier.verify_final(&expected_state, &key, &final_proof)?;
```

neo-spartan level (what nightstream calls inside `shrink` and `verify`):

```rust
use neo_spartan::shrink::{self, Program, Shape};

let program = FinalProgram::new(&key);                     // nightstream type, implements Program
let shape: &Shape = key.shrink_shape();                    // derived once, stored in the key
let proof = shrink::prove(&program, shape, &statement_words, &final_view)?;
shrink::verify(&program, shape, &statement_words, &proof)?;
```

The program author writes verifier code once:

```rust
fn run<B: Backend>(&self, b: &mut B, statement: &[B::F], input: &FinalView) -> Result<(), Error> {
    let digest = state_hash(b, &self.key, statement, &input.running)?;   // layer 0
    let mut transcript = FoldTranscript::new(b);
    let parent = pi_ccs_and_rlc(b, &self.key, &mut transcript, digest, input)?;
    neo_spartan::verify(b, &self.key.layer1_relation()?, transcript, &parent, &input.layer1)
}
```

## Shape

### Data structures first

#### The constraint system

A CCS (SuperSpartan) over Goldilocks with 8 matrices and one polynomial:

    f(z) = (X·z)^7 + Σ_{i<3} (A_i·z) ∘ (B_i·z) − C·z = 0   (row-wise)

- **S-box row:** X holds the S-box input form (MDS combination of earlier cells plus
  the round constant through the constant-one column). C picks the S-box output cell.
- **Product row:** up to three products and a linear part. An Ext×Ext product is three
  rows, one per output coordinate: row k is Σ_i a_i·(Σ_j S_ijk b_j) − d_k with the
  structure constants S of x^3 − x − 1. Base×Ext is one row per coordinate group.
- **Linear row:** all product slots empty; C carries the linear form.

`z = [w | 1, statement, 0…]`. Only `w` is committed. The second half is public and has
9 statement words (iteration, z0, current) after the constant one. The verifier computes
its MLE directly (Spartan, 2019/550 §6).

Layout of `w` (cells) and of the rows. N is the number of permutations, G the number
of glue cells.

| Region | Cells | Rows | Content |
| --- | --- | --- | --- |
| P0 | 64·N | 64·N | full rounds 1-4: 64 S-box outputs and their rows |
| P1 | 64·N | 64·N | full rounds 27-30: 64 S-box outputs and their rows |
| P2 | 64·N | 64·N | 16 inputs, 22 partial S-boxes, 16 outputs, 10 free; 22 S-box rows, 16 output rows |
| Glue | G | glue rows | proof words, bits, Ext arithmetic, materialized forms; wiring rows |

Regions are packed back to back (no power-of-two padding per region). The cube sizes
are derived: `cells = 2^⌈log2(192N + G)⌉`, `rows = 2^⌈log2(192N + glue rows)⌉`.
Expected: N ≈ 32.5k, G ≈ 0.5-0.9M, so w has 2^23 cells (80-85% full) and there are
2^23 rows.

**Matrix evaluation.** For a matrix M and points (rx, ry):

- *Permutation part:* every block (row sub-block s1, column sub-block s2) is the same
  64×64 template T_{s1,s2} on the diagonal b = b'. So
  `M~ = Σ_{s1,s2} D(rx_hi, ry_hi; s1·N, s2·N, N) · T~_{s1,s2}(rx_lo, ry_lo)` with
  `D(x, y; c_x, c_y, N) = Σ_{b<N} eq(x, b + c_x)·eq(y, b + c_y)`. D is a carry DP over
  the high bits (two carries and a less-than flag, 8 states per bit, 17 bits). Round
  constants sit in the constant-one column: `Σ_{b<N} eq(rx_hi, b + c)` times a
  template vector. Cost: about 3.3k template entries plus 9 DPs; under 1 ms.
- *Glue part:* sparse rows. The native verifier regenerates them (below) and
  accumulates `Σ coeff·eq(rx, row)·eq(ry, col)` per matrix while they are emitted. It
  uses split eq tables (two tables of 2^12 entries) and stores no rows.

#### The gadget API (one source of truth)

```rust
// crates/neo-spartan/src/circuit/mod.rs
//! One verifier, two backends. Verifier code is written once over `Backend`.
//! `Native` checks values: it is the plain verifier. `Recorder` (crate-private) emits
//! the CCS rows and, given a real proof, the witness.
//! Owns: the backend contract. Does not own any protocol.
//! Rule: verifier code never branches on values. Every loop bound and length comes
//! from a key. Values reach verifier code only through hint outputs, which are
//! private inputs that later checks must constrain.

pub trait Backend {
    /// A Goldilocks value (Native) or a linear form over wires (Recorder).
    type F: Copy;
    /// An Ext = F[w]/(w^3 - w - 1) value or three forms.
    type E: Copy;

    fn constant(&mut self, value: u64) -> Self::F;
    /// A proof word. It has no authority until a check uses it.
    fn private(&mut self, value: u64) -> Self::F;
    /// A statement word of the shrink proof.
    fn public(&mut self, value: u64) -> Self::F;

    fn add(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn sub(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn scale(&mut self, a: Self::F, by: u64) -> Self::F;
    fn mul(&mut self, a: Self::F, b: Self::F) -> Self::F;

    fn ext(&mut self, coordinates: [Self::F; 3]) -> Self::E;
    fn coordinates(&mut self, a: Self::E) -> [Self::F; 3];
    fn ext_add(&mut self, a: Self::E, b: Self::E) -> Self::E;
    fn ext_sub(&mut self, a: Self::E, b: Self::E) -> Self::E;
    fn ext_mul(&mut self, a: Self::E, b: Self::E) -> Self::E;
    fn ext_scale(&mut self, a: Self::E, by: Self::F) -> Self::E;
    /// Rejects zero: Native returns `Err`; Recorder adds `a·inv = 1`.
    fn ext_inverse(&mut self, a: Self::E, what: &'static str) -> Result<Self::E, Error>;

    fn assert_zero(&mut self, a: Self::F, what: &'static str) -> Result<(), Error>;
    fn assert_ext_zero(&mut self, a: Self::E, what: &'static str) -> Result<(), Error>;
    /// Canonical little-endian bits (the value must be below 2^count and below p).
    fn bits(&mut self, a: Self::F, count: usize, what: &'static str) -> Result<Vec<Self::F>, Error>;
    /// `bit ? one : zero` for a constrained bit.
    fn select(&mut self, bit: Self::F, zero: Self::F, one: Self::F) -> Self::F;
    /// The workspace Poseidon2 permutation (width 16).
    fn permute(&mut self, state: [Self::F; 16]) -> [Self::F; 16];
    /// `len` private words computed from the values of `inputs`. Shape mode passes
    /// zeros; `None` gives zeros. The caller must check every output.
    fn hint(
        &mut self,
        inputs: &[Self::F],
        len: usize,
        compute: &dyn Fn(&[u64]) -> Option<Vec<u64>>,
    ) -> Vec<Self::F>;
}

/// The plain verifier: values only, `Err` on the first failed check.
pub struct Native;
impl Backend for Native { /* F = Value(Gl), E = ExtValue(Ext): opaque newtypes */ }
```

Three uses of the same code:

| Run | Backend | Input | Result |
| --- | --- | --- | --- |
| `verify_final` (native M2 verify) | `Native` | decoded `FinalProof` | `Ok` or the first rejection |
| shrink prover | `Recorder<Witness>` | decoded `FinalProof` | rows, witness `w`, first unsatisfied check |
| shrink verifier | `Recorder<Evaluate>` | `FinalView::zero(key)` | `Σ_j ρ^j M~_j(rx, ry)` of the glue, row and cell counts |

The Recorder never stops at a failed check. It records the row and notes the first
failure. The prover returns an error on any note. Shape determinism (the rows depend
only on the key) is tested by comparing a digest of the rows for a zero input and for
an honest input.

```rust
// crates/neo-spartan/src/circuit/record.rs (crate-private)
/// At most four (wire, coefficient) terms and a constant, with the tracked value.
/// A longer sum becomes a new cell and one linear row.
#[derive(Clone, Copy)]
pub(crate) struct Form { terms: [(Wire, u64); 4], len: u8, constant: u64, value: u64 }

#[derive(Clone, Copy)]
pub(crate) enum Wire { Glue(u32), Perm { slot: u32, cell: u8 }, Public(u16) }

/// One glue row: up to three products and a linear part, by matrix.
pub(crate) struct Row { products: [(Form, Form); 3], linear: Form }

pub(crate) trait Sink {
    fn row(&mut self, row: &Row);
    fn cell(&mut self, value: u64);
    fn perm(&mut self, slot: u32, cells: &[u64; 192]);
}
pub(crate) struct Recorder<S: Sink> { sink: S, cells: u32, perms: u32, failure: Option<&'static str> }
pub(crate) struct Witness { rows: Glue, glue: Vec<u64>, perms: Vec<[u64; 192]> }
pub(crate) struct Evaluate { rx: SplitEq, ry: SplitEq, rho: [Ext; 8], total: Ext, layout: Layout }
```

#### Inputs as domain types

The generic verifier reads decoded views, never p3 types:

```rust
// crates/neo-spartan/src/whir/view.rs
/// One p3-whir 0.8 opening in our profile, as plain words. Built from a p3 proof or
/// as zeros of the right shape. Pruned multiproofs stay opaque; only a hint reads them.
pub(crate) struct WhirView { /* per round: root, OOD answers, sum-check polys,
    PoW witness, query rows, pruned multiproof; final poly; batch evaluations */ }

// crates/neo-spartan/src/lib.rs
pub struct ProofView { /* layer-1 proof words and two WhirViews */ }
impl ProofView {
    pub fn decode(proof: &Proof, relation: &Relation<'_>) -> Result<Self, Error> { unimplemented!() }
    pub fn zero(relation: &Relation<'_>) -> Self { unimplemented!() }
}

// crates/nightstream/src/lifecycle/view.rs
/// A FinalProof as words: 16 running claims, the fresh claim, PiCCS rounds, the
/// PiRLC parent and the layer-1 view.
pub(crate) struct FinalView { /* ... */ }
```

Merkle paths: the generic WHIR verifier draws the round's query bits, then asks
`b.hint(bits, Q·h·4, expand)` for full per-query paths. Natively, and in the witness
run, `expand` walks the pruned multiproof with the concrete indices (input parsing,
no authority). Every path is then checked per query in constraints against the
round root. In shape mode the hint returns zeros.

### Module map

| File | New/changed | Est. lines | Owns |
| --- | --- | --- | --- |
| `neo-spartan/src/circuit/mod.rs` | new | 300 | `Backend`, `Native`, the no-branch rule |
| `neo-spartan/src/circuit/record.rs` | new | 500 | `Recorder`, `Form`, `Row`, sinks |
| `neo-spartan/src/circuit/algebra.rs` | new | 250 | generic Ext, K, Kx helpers: eq, Lagrange, Horner, powers |
| `neo-spartan/src/circuit/hash.rs` | new | 300 | `Duplex<B>` (p3 `DuplexChallenger` semantics incl. `sample_bits`, `sample_uniform_bits`, grinding), `FoldTranscript<B>` (neo-transcript v1_1), leaf sponge, Merkle path, handoff |
| `neo-spartan/src/circuit/poseidon2.rs` | new | 250 | the 3×64 block template from the round constants and MDS; S-box witness fill |
| `neo-spartan/src/whir/mod.rs` | new | 600 | generic p3-whir 0.8 verifier for our profile (OOD, sum-check, STIR, final poly, PoW) |
| `neo-spartan/src/whir/layout.rs` | new | 350 | stacked-layout claims to eq and select statements, and their weights |
| `neo-spartan/src/whir/view.rs` | new | 250 | `WhirView` from p3 proofs, zero views, pruned-path expansion |
| `neo-spartan/src/verify.rs` | moved from `lib.rs` | 450 | generic layer-1 `verify` |
| `neo-spartan/src/{gkr,sumcheck,setup,matrix,mle,norm,ring}.rs` | changed | +450 | verifier halves become generic; `mle::diagonal` DP (+60) |
| `neo-spartan/src/pcs.rs` | changed | +50 | `Profile { log_inv_rate, pow_bits, first_fold }`; M2 keeps (1, 0, 4) |
| `neo-spartan/src/shrink/mod.rs` | new | 350 | `Program`, `Shape`, `shrink::Proof`, `prove`, `verify` |
| `neo-spartan/src/shrink/layout.rs` | new | 350 | regions, closed-form permutation MLE, glue column mapping |
| `neo-spartan/src/shrink/spartan.rs` | new | 450 | SuperSpartan outer and inner sum-checks (prover and verifier) |
| `nightstream/src/lifecycle/final_verify.rs` | new | 650 | generic final program: statement, state hash, encHash, canonical split, PiCCS replay, layer-1 call |
| `nightstream/src/lifecycle/final_rlc.rs` | new | 300 | ρ sampling and base-5 decode (limb gadget), randomized PiRLC |
| `nightstream/src/lifecycle/view.rs` | new | 200 | `FinalView` decode and zero |
| `nightstream/src/lifecycle/finish.rs` | changed | +100 | `verify_final` calls the program on `Native`; `FinalProgram`; `shrink` |
| `nightstream/src/circuit.rs` | changed | +150 | `CompressionKey` struct (layer-1 key, fold key, shrink shape), `ShrinkProof`, `Prover::shrink`, `CompressionKey::verify` |

Total ≈ 5,200 new lines and ≈ 750 changed lines, plus tests. No file passes 1,000 lines.
Call chains stay within three files: `CompressionKey::verify` → `shrink::verify` →
`spartan` / `layout` / `whir`.

### Type sketch

```rust
// crates/neo-spartan/src/shrink/mod.rs
//! The shrink argument: SuperSpartan over the CCS that a `Program` records, with one
//! WHIR commitment. Owns: the CCS layout, the shrink transcript, the proof bytes.
//! Does not own what the program verifies.

/// A verifier written once over `Backend`.
pub trait Program {
    type Input;
    /// Names the program; bound before the first shrink challenge.
    fn words(&self) -> Vec<u64>;
    fn statement_len(&self) -> usize;
    /// An input of the right shape, every word zero (shape mode).
    fn zero_input(&self) -> Self::Input;
    /// `-log2` of the error this layer may add.
    fn security_bits(&self) -> f64;
    fn run<B: Backend>(&self, b: &mut B, statement: &[B::F], input: &Self::Input) -> Result<(), Error>;
}

/// Counts that size the cubes and the WHIR profile. Derived once from the program
/// and kept in the key; every verify re-checks them against the regenerated rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape { perms: u32, glue_cells: u32, glue_rows: u32 }

impl Shape {
    pub fn derive<P: Program>(program: &P) -> Result<Self, Error> { unimplemented!() }
}

/// Opaque shrink proof. It contains no witness coordinate in the clear.
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    commitment: pcs::Commitment,
    outer: Vec<[Ext; 8]>,
    claims: [Ext; 8],
    inner: Vec<[Ext; 2]>,
    opening: pcs::Opening,
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
}

pub fn prove<P: Program>(program: &P, shape: &Shape, statement: &[u64], input: &P::Input) -> Result<Proof, Error> {
    // 1. Recorder<Witness> runs the program; any failure note is an error.
    // 2. Check the counts against `shape`; place regions; build w.
    // 3. Transcript start; commit w; outer sum-check; claims; inner sum-check; open w.
    unimplemented!()
}

pub fn verify<P: Program>(program: &P, shape: &Shape, statement: &[u64], proof: &Proof) -> Result<(), Error> {
    // 1. Transcript start; observe the commitment; replay both sum-checks.
    // 2. Recorder<Evaluate> runs the program on `zero_input()` and streams the glue
    //    MLE at (rx, ry); its counts must equal `shape`.
    // 3. Add the closed-form permutation part and the public half of z.
    // 4. Check the outer and inner final claims; verify the WHIR opening.
    unimplemented!()
}
```

```rust
// crates/neo-spartan/src/shrink/layout.rs
pub(crate) struct Layout { perms: usize, glue_cells: usize, glue_rows: usize, cell_vars: usize, row_vars: usize }
impl Layout {
    pub(crate) fn new(shape: &Shape) -> Self { unimplemented!() }
    /// Global column of a wire.
    pub(crate) fn column(&self, wire: Wire) -> usize { unimplemented!() }
    /// Σ_j ρ^j M~_j(rx, ry) restricted to the permutation regions.
    pub(crate) fn perm_part(&self, rx: &[Ext], ry: &[Ext], rho: &[Ext; 8]) -> Ext { unimplemented!() }
}

// crates/neo-spartan/src/mle.rs
/// Σ_{b<n} eq(x, b + cx)·eq(y, b + cy) over the bits of x and y (carry DP).
pub(crate) fn diagonal(x: &[Ext], y: &[Ext], cx: u64, cy: u64, n: u64) -> Ext { unimplemented!() }
```

```rust
// crates/neo-spartan/src/circuit/hash.rs
/// p3 `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>` over any backend.
pub struct Duplex<B: Backend> { state: [B::F; 16], input: Vec<B::F>, output: Vec<B::F> }
impl<B: Backend> Duplex<B> {
    pub fn observe(&mut self, b: &mut B, word: B::F) { unimplemented!() }
    pub fn sample(&mut self, b: &mut B) -> B::F { unimplemented!() }
    pub fn sample_ext(&mut self, b: &mut B) -> B::E { unimplemented!() }
    /// Low `bits` of a canonical sample.
    pub fn sample_bits(&mut self, b: &mut B, bits: usize) -> Result<Vec<B::F>, Error> { unimplemented!() }
    /// As `sample_bits`, and the sample must not be in p3's resampling band.
    pub fn sample_uniform_bits(&mut self, b: &mut B, bits: usize) -> Result<Vec<B::F>, Error> { unimplemented!() }
    pub fn check_grinding(&mut self, b: &mut B, witness: B::F, bits: usize) -> Result<(), Error> { unimplemented!() }
    /// A copy that leaves `self` unchanged (used for the PiRLC check).
    pub fn fork(&self) -> Self { unimplemented!() }
}

/// neo-transcript `Poseidon2Transcript` v1_1 (additive absorb) over any backend.
pub struct FoldTranscript<B: Backend> { state: [B::F; 16] }
impl<B: Backend> FoldTranscript<B> {
    pub fn new(b: &mut B, label: &[u8]) -> Self { unimplemented!() }
    pub fn reset(&mut self, b: &mut B) { unimplemented!() }
    pub fn absorb(&mut self, b: &mut B, words: &[B::F]) { unimplemented!() }
    pub fn read_pair(&self, pair: usize) -> [B::F; 2] { unimplemented!() }
    pub fn squeeze_digest(&mut self, b: &mut B) -> [B::F; 4] { unimplemented!() }
    /// The layer-1 handoff of `hash::challenger`.
    pub fn hand_off(self, b: &mut B) -> Duplex<B> { unimplemented!() }
}

pub fn leaf_hash<B: Backend>(b: &mut B, values: &[B::F]) -> [B::F; 4] { unimplemented!() }
pub fn merkle_root<B: Backend>(b: &mut B, leaf: [B::F; 4], index_bits: &[B::F], path: &[[B::F; 4]]) -> [B::F; 4] {
    unimplemented!()
}
```

```rust
// crates/neo-spartan/src/verify.rs
/// Verify a layer-1 proof for `claim`, continuing `transcript`. Reads only the key.
/// The same code is the native verifier and the shrink circuit.
pub fn verify<B: Backend>(
    b: &mut B,
    relation: &Relation<'_>,
    transcript: FoldTranscript<B>,
    claim: &ClaimWords<B>,
    proof: &ProofView,
) -> Result<(), Error> {
    unimplemented!()
}

/// The parent CE(B) claim as backend values (commitment, public, point, Eval_K, Eval_A).
pub struct ClaimWords<B: Backend> { /* ... */ }

// crates/neo-spartan/src/whir/mod.rs
/// Replay one p3-whir 0.8 opening of `pcs` and return the opened batch values.
pub(crate) fn verify<B: Backend>(
    b: &mut B,
    pcs: &Pcs,
    root: [B::F; 4],
    view: &WhirView,
    points: &[Vec<B::E>],
    challenger: &mut Duplex<B>,
) -> Result<Vec<Vec<B::E>>, Error> {
    unimplemented!()
}
```

```rust
// crates/nightstream/src/circuit.rs
/// Everything a verifier of a FinalProof or a ShrinkProof needs. Authority: derived
/// from the package (`Verifier::compression_key`) or a pinned copy. Never from a prover.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionKey { layer1: neo_spartan::Key, fold: FoldKey, shrink: neo_spartan::shrink::Shape }

impl CompressionKey {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
    /// The chain entry point. Reads only this key.
    pub fn verify(&self, expected_state: &State, proof: &ShrinkProof) -> Result<(), Error> { unimplemented!() }
}

impl Prover {
    pub fn shrink(&self, key: &CompressionKey, proof: &FinalProof) -> Result<ShrinkProof, Error> { unimplemented!() }
}

pub struct ShrinkProof(neo_spartan::shrink::Proof);

// crates/nightstream/src/lifecycle/final_verify.rs
/// The fold constants the final verifier reads instead of the package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct FoldKey {
    verifier_context: [u64; 4],
    minimum_security_bits: u32,
    /// Exact fold error, stored as f64 bits for canonical bytes.
    fold_error_bits: u64,
    ccs: CcsShape,          // rows, columns, matrices (4), degree, polynomial terms
    params: FoldParams,     // b, k_rho, kappa, T, public width, round count
}

pub(crate) struct FinalProgram<'k> { key: &'k CompressionKey }

impl neo_spartan::shrink::Program for FinalProgram<'_> {
    type Input = FinalView;
    fn words(&self) -> Vec<u64> { unimplemented!() }
    fn statement_len(&self) -> usize { 9 }
    fn zero_input(&self) -> FinalView { unimplemented!() }
    fn security_bits(&self) -> f64 { unimplemented!() }
    fn run<B: Backend>(&self, b: &mut B, statement: &[B::F], input: &FinalView) -> Result<(), neo_spartan::Error> {
        // check_statement → state hash → PiCCS replay → ρ → randomized PiRLC → layer 1
        unimplemented!()
    }
}
```

### Load-bearing decisions

1. **One layer.** It reaches 83 KB (model). A second layer would save ≈ 10 KB and
   needs a fully uniform first layer (its matrices must be evaluated in-circuit).
2. **M2 unchanged.** The shrink cube is 2^23 with or without a retune; a retune only
   halves shrink prover time (2^22) and costs M2 6-16 GB and 2× setup disk.
3. **Generic verifier.** The native verifier, the witness generator and the constraint
   generator are one function, per *single source of truth*. Values cannot drive
   control flow; hints are the only value channel, per *boundary discipline*.
4. **Structure split by size.** The 107M Poseidon2 nonzeros get closed forms. The ≈ 7M
   glue nonzeros are regenerated and streamed by the verifier (100-400 ms budget).
5. **Randomized PiRLC** in the final verifier for both Native and Recorder.
6. **Our own generic WHIR verifier** for the p3-whir 0.8 profile. It replaces
   `verify_at` everywhere in our verifiers (M2 and shrink). p3-whir stays as the
   prover. Versions pinned exactly (`=0.8.0`).
7. **First WHIR fold of 5** for the shrink commitment (`ConstantFromSecondRound(5, 4)`),
   which removes one round at 2^23.

What the design does not do: no zero knowledge (the FinalProof is not secret); no
holography; no Metal path; no change to the fold, M2's PCS profile, the setup or the
SHAKE128 key; the shrink verifier never reads the package or the setup files.

Interface depth: nightstream gains two entry points (`Prover::shrink`,
`CompressionKey::verify`) and one proof type. neo-spartan gains the gadget language
(`Backend`, `Native`, hash and algebra helpers), which nightstream needs to write layer
0 once, and `shrink::{Program, Shape, Proof, prove, verify}`. Behind them it hides the
CCS, regions, the recorder, both sum-checks, WHIR and path expansion. `Recorder` is
crate-private, so no caller can build rows by hand.

## Synthesis decision

Left for the arena.

## The protocol and transcript order

### Inside the circuit (and in `verify_final`)

The program replays M2 exactly, in M2's transcript order, with two changes that keep
the statement: Merkle paths are checked per query, and PiRLC is checked at a random
point.

1. `check_statement`: shapes are static. Children digits are checked as the canonical
   split of the parent public input (sign bit and digit bits per coordinate). The
   27,819-word preimage is assembled from statement words and running claims and
   hashed (2,319 permutations). The four digest words are split into canonical bits;
   the fresh public input must equal `encHash` of them.
2. `prepare_running`: the parent authority is the b-power sum of the children
   (linear). PiDEC's recomputation is then an identity, so it needs no rows.
3. PiCCS (`FoldTranscript`, v1_1): reset, domain chunk, statement (prior digest, fresh
   commitment, fresh x), 28 α coins and γ, 28 rounds (coefficient form, `c0 + Σc =
   claim`, Horner at the coin), outputs absorbed, terminal check (`initial_claim`,
   `terminal_components`: CCS f from the key, range product, γ powers, equality).
4. PiRLC ρ: per source, absorb `(4, i)`, squeeze a digest, decode 54 base-5 digits.
   The decode is a limb gadget: 16-bit limbs for `Σ d_i p^i`, digits c_k ∈ [0, 4]
   (3 bits each, c ≤ 4), quotient Q (131 bits), range-checked carries, exact integer
   equality `Σ d_i p^i = Σ c_k 5^k + 5^54·Q`.
5. Layer-1 handoff: absorb the domain chunk, squeeze, seed `Duplex`. `start` observes
   shape, key, profiles, query count and the parent claim.
6. **Randomized PiRLC** on a fork of the layer-1 challenger after `start` (the main
   challenger is unchanged): sample λ'; the prover supplies Q, the 53 Ext quotient of
   `Σ_f λ'^f (Σ_i ρ_i ⋆ v_{i,f} − v_{parent,f})` by Φ81; observe Q; sample ζ'; check
   `Σ_i ρ_i(ζ')·V_i(ζ') − V_parent(ζ') = Q(ζ')·Φ81(ζ')` with
   `V(ζ') = Σ_{f,k} λ'^f ζ'^k v_{f,k}` over 37 families (22 commitment rows, 5 public
   blocks, Eval_K and 4 Eval_A as Re/Im). Every v is bound before the fork: running
   claims by the state hash, fresh data and the 17 output families by the PiCCS
   transcript, the parent by `start`.
7. Layer 1 exactly as `neo_spartan::verify` today: P0 root, histogram (all 2H+1 = 7,345
   denominators must be nonzero), early GKR, early checks, λ, quotient, ζ, linear
   sum-check, finals, μ, partials, α, P1 root, 125 setup indices (`sample_bits`), setup
   leaves (sponge + 24-level path to the key's root constant), late GKR, late checks,
   P1 opening (129 batches), P0 opening (7 batches).
8. Each WHIR opening (generic replay of p3-whir 0.8, our profile): layout claims →
   batching α → initial sum-check (4 rounds) → per round: root, OOD answers, PoW
   check (zero difficulty in M2), `sample_uniform_bits` STIR queries, hint for full
   paths, leaf hash + path per query, fold of the 16 leaf values at the round's
   randomness, eq and select statements, round sum-check → final polynomial, final
   queries, final sum-check → batch values returned to the caller.

### The shrink proof (outside the circuit)

Transcript: p3 `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>`.

1. Observe the domain chunk `"Nightstream/SuperNeo/shrink/v1"`, the program words (the
   whole `CompressionKey` and the layout words: N, G, rows, cube sizes, profile), and
   the 9 statement words.
2. WHIR commit to w (2^23 base cells, rate 1/64, schedule [5, 4, 4, 4], Johnson bound,
   PoW 20); observe the root.
3. Sample τ ∈ Ext^23.
4. Outer sum-check, 23 rounds, degree 8 (h(0), h(2..8)) → rx.
5. Observe the claims v_j = (M_j z)~(rx), j < 8. Check
   `eq(τ, rx)·(v_X^7 + Σ_i v_{A_i} v_{B_i} − v_C) = last claim`.
6. Sample ρ.
7. Inner sum-check of `Σ_y (Σ_j ρ^j M~_j(rx, y))·z~(y) = Σ_j ρ^j v_j`, 24 rounds,
   degree 2 → ry.
8. WHIR opening of w at ry[..23] (one batch) → w~. Then
   `z~(ry) = (1 − ry_23)·w~ + ry_23·pub~(ry[..23])`.
9. The verifier computes `Σ_j ρ^j M~_j(rx, ry)` = closed-form permutation part +
   streamed glue part, and checks the last inner claim.

Ordering requirements: w before τ; all sum-check messages before their coins; the
glue is regenerated only after (rx, ry) are known, so it is streamed and never stored.

## Soundness map

Budget: caller minimum m (owner: 114). Fold error ε_fold from `Params` (≈ 2^-114.4 at
the production shape). Remainder r = 2^-m − ε_fold ≈ 2^-116.05. **Split r in two
halves**: layer 1 and the shrink each get 2^-117.05.

| Term | Drawn | Bound | Value at m = 114 |
| --- | --- | --- | --- |
| Fold chain (unchanged) | layer 0 | `Params::fold_error_bits` | ≈ 2^-114.4 |
| Layer 1 (M2) | P0, P1, setup queries, all draws | `Relation::new(117.05)`; each commitment targets 118.05 | ≤ 2^-117.05 |
| Randomized PiRLC | fork after `start` | λ': 36/\|Ext\|; ζ': 106/\|Ext\| | 2^-184.8 (charged in layer 1's report) |
| Shrink zero-check τ | after the w root | 23/\|Ext\|, charged over L | ≈ 2^-160 |
| Outer sum-check | 23 rounds, degree 8 | 184/\|Ext\| × L | ≈ 2^-175 |
| Matrix batching ρ | | 7/\|Ext\| × L | ≈ 2^-180 |
| Inner sum-check | 24 rounds, degree 2 | 48/\|Ext\| × L | ≈ 2^-177 |
| WHIR (shrink) | Johnson bound, PoW 20 per site | p3 report incl. the terms above, target 118.05 | ≤ 2^-118.05 |
| **Total** | | sum | **≤ 2^-114** |

|Ext| = p^3 ≈ 2^191.99. L is p3's candidate bound (log2 L ≈ 9-10 at rate 1/64 with
η = √ρ/20). Terms after the commitment are charged over L, as M2 does.

Composition: the shrink argument is knowledge-sound for the CCS. A satisfying w
contains a `FinalView` on which the program accepts. The arithmetization is exact:
every check is a row; bits are canonical (value < p); inverses are checked; hint
outputs are private inputs that later rows constrain. So the extracted FinalProof
passes M2 verification, except that PiRLC holds at a random point (term above). M2's
argument and the fold chain then apply. The in-circuit transcript computes M2's
Fiat-Shamir challenges with the same Poseidon2, so the extracted proof's challenges
are honest.

Completeness note: p3 `sample_uniform_bits::<true>` resamples a sample equal to p − 1
(probability 2^-64). The circuit has a fixed draw count, so it rejects that case.
About 850 STIR draws in M2 give a completeness gap of ≈ 2^-54. Soundness is not
affected.

Security rules: the shrink transcript binds the key words that the verifier holds,
not a carried digest. Every Merkle root inside the circuit is recomputed. The setup
root is a constant of the regenerated rows. The statement words are public cells.

## Costs

### Cost model and calibration

- In-circuit permutations: queries per WHIR round from p3's Johnson formula
  `⌈λ / (0.5·log_inv_rate − log2(21/20))⌉`, leaf sponge + full path per query. The same
  model with pruned paths gives 9,017 for P0 (measured 8,627) and 7,865 + ≈ 600 for P1
  (measured 8,470).
- WHIR bytes: the model is 6-9% above the measured table, so it is scaled by 0.93.
  Anchors: 2^22 at 1/64 → 76.5 KB (measured 75.0); 2^24 at 1/16 → 107.9 KB
  (measured 109.0); 2^22 at 1/16 → 90.7 KB (measured 89.7).

### Permutations and constraints per component (est.)

| Component | Permutations | Main glue arithmetic |
| --- | --- | --- |
| L0 statement, state hash, encHash, canonical split | 2,330 | 4,320 digit checks, 256 + 4×64 bits |
| L0 PiCCS replay | 949 | ≈ 14k K multiplications |
| L0 ρ sampling, decode, randomized PiRLC | ≈ 50 | 17 limb gadgets (≈ 1k bits each), ≈ 36k base×Ext products |
| L1 start, histogram, early GKR, ring, linear | 1,330 | 7,345 inverses, Eval_K DP (3,626 transitions), lane and step MLEs |
| L1 setup leaves (125 → 126 queries after the split) | 5,210 | leaf weights, 2 × 208 base×Ext per leaf |
| L1 P1 opening (2^26, rate 1/2, unpruned) + late GKR | 10,980 | ≈ 415 query folds, 129 prescribed-point weights |
| L1 P0 opening (2^28, rate 1/2, unpruned) | 11,650 | ≈ 432 query folds, constraint weights |
| **Total** | **≈ 32.5k** | **≈ 1.2M product terms** |

| Quantity | Estimate |
| --- | --- |
| Permutation cells | 192 × 32.5k = 6.24M (182 used per block) |
| Glue cells (proof words ≈ 215k, bits ≈ 75k, products ≈ 0.4M, forms) | 0.5-0.9M |
| Witness w | 6.7-7.1M cells → **2^23** (80-85%) |
| Rows | 6.24M + 0.9-1.3M glue → 2^23 |
| Glue nonzeros | 5-8M (wiring ≈ 1.5M: 16 rows per permutation) |
| Permutation-template nonzeros | ≈ 3.3k per block, ≈ 107M implicit |

If the cells pass 2^23, the derived cube becomes 2^24 automatically (proof ≈ 91 KB,
prover ≈ 2×). The reserve lever before that is a Merkle cap of height 4 on M2's
per-proof trees: p3 0.8 supports `cap_height`; it saves ≈ 3.4k permutations for
≈ 70k mux rows and costs the M2 prover nothing.

### Proof bytes

| Part | Bytes (est.) |
| --- | --- |
| WHIR commitment + opening: 2^23 cells, rate 1/64, schedule [5,4,4,4], JB, PoW 20 | ≈ 77.5 KB |
| Outer sum-check: 23 × 8 Ext | 4.4 KB |
| Matrix claims: 8 Ext | 0.2 KB |
| Inner sum-check: 24 × 2 Ext | 1.15 KB |
| Opened value, lengths | 0.1 KB |
| **Total** | **≈ 83.4 KB** (goal ≤ 100 KB: met with ≈ 17% margin; hard ≤ 200 KB: met) |

Other points on the same curve (model, 2^23, schedule [5,4,4,4], plus 5.9 KB):

| Rate | PoW | Total | WHIR commit time (est.) |
| --- | --- | --- | --- |
| 1/256 | 20 | ≈ 75 KB | ≈ 65 s, codeword 16 GB (not recommended) |
| **1/64** | **20** | **≈ 83 KB** | **≈ 17 s, codeword 4 GB** |
| 1/16 | 20 | ≈ 100 KB | ≈ 4.4 s, codeword 1 GB |
| 1/4 | 0 | ≈ 164 KB | ≈ 1.5 s, codeword 0.25 GB |
| 1/2 | 20 | ≈ 212 KB | over the hard limit |

The default constant-4 schedule at 1/64 gives ≈ 94 KB (one more round). The first fold
of 5 is worth ≈ 10 KB.

**What the 200 KB fallback saves.** Rate 1/4 with no PoW (≈ 164 KB) has the same
circuit, the same layer count and the same verifier. It saves ≈ 15 s of shrink prover
time, ≈ 3.5 GB of peak memory and all grinding. Nothing structural changes. The
fallback does not make holography fit: a second per-proof commitment plus setup leaves
would need ≈ 85 + 30 KB on top.

### Prover (shrink only, CPU, this Mac, est.)

| Step | Time | Memory |
| --- | --- | --- |
| Recorder over the FinalProof: 32.5k permutations, ≈ 1.1M glue rows | ≈ 1 s | ≈ 0.5 GB |
| Mz vectors and outer sum-check (2^23 rows, degree 8) | ≈ 3 s | ≈ 1.5 GB |
| Inner sum-check (permutation part from the template: N × 3.3k) | ≈ 2 s | ≈ 0.6 GB |
| WHIR commit + open (2^23 at 1/64; 2^22 at 1/64 measured 8.5 s) | ≈ 17 s | ≈ 6.5 GB |
| **Total** | **≈ 23 s** | **≈ 9 GB peak** |

It runs after `finish_with_spartan` (39.7 s, 37.3 GB peak), whose memory is free by
then. M2 changes only by the budget split: ≈ +0.7% queries (round-0 275 → 277).

### Native verifier (one CPU, est.) against the 100-400 ms budget

| Step | Time |
| --- | --- |
| Shrink transcript, both sum-checks, closed-form permutation part | < 2 ms |
| WHIR verification (≈ 2.2k permutations, folds, PoW checks) | 5-8 ms |
| Regenerate the glue on the zero input (32.5k permutations, ≈ 1.1M rows) | 50-70 ms |
| Stream the glue MLE (5-8M nonzeros, split eq tables) | 60-90 ms |
| **Total** | **≈ 120-170 ms** (inside 100-400 ms) |

Memory under 100 MB. Key ≈ 3 KB: layer-1 key 1,912 B, fold key ≈ 0.6-1 KB (CCS terms,
params, context digest, error bits), shrink shape 12 B. Two optional speed-ups, not in
the plan: skip value computation in the regenerating run (≈ −35 ms), or cache the glue
per key on the node.

`verify_final` (M2, now generic on `Native`, per-query paths): ≈ 40-50 ms instead of
32 ms.

## Tradeoffs accepted

- We accept ≈ 6k new lines (generic verifier, recorder, shrink argument, generic WHIR
  verifier) in exchange for one verifier that is at once the native M2 verifier, the
  witness generator and the constraint generator.
- We accept a 120-170 ms verifier that re-runs the verifier program in exchange for no
  holography, one commitment, an 83 KB proof and a 3 KB key.
- We accept per-query Merkle paths in the circuit (≈ 29.0k instead of 23.6k layer-1
  permutations), because a pruned walk is value-dependent.
- We accept our own copy of the p3-whir 0.8 verifier for our profile, pinned to the
  exact version, in exchange for a WHIR verifier that runs over the recorder. p3-whir's
  `verify_at` is then used only in tests.
- We accept leaving M2 at rate 1/2 with no caps. A smaller circuit moves the proof by
  ≈ 3 KB; the M2 retune costs 6-16 GB and 2× setup disk.
- We accept a randomized PiRLC check in `verify_final` (+2^-184.8) in exchange for
  ≈ 1.7M fewer product terms.
- We accept two PiCCS verifiers: neo-reductions keeps the accumulator path, and the
  final program has its own replay. A trace test (every absorb, coin and terminal
  component) fails the build when they drift.
- We accept explicit `b.mul(x, y)` code instead of operator overloading, because
  overloading needs hidden shared recorder state.
- We accept rate 1/64 and PoW 20 (≈ 17 s, 4 GB) for the 100 KB goal; the 200 KB
  fallback is a one-line profile change.
- We accept a ≈ 2^-54 completeness gap from p3's rejection resampling.

## Alternatives considered

1. **Two layers (a fat inner layer with data-parallel GKR for Poseidon2, ePrint
   2023/1284 style, plus a tiny outer layer).** The inner layer would commit only
   permutation inputs and outputs (≈ 2^21). But the outer layer must evaluate the
   inner layer's matrices in-circuit, which forces a fully uniform inner layer (a
   recursion VM with logUp wiring). It saves ≈ 10 KB against a single layer that
   already meets the goal, and it doubles the recursion machinery. Rejected.
2. **Data-parallel GKR in the final layer.** One GKR layer per Poseidon2 round, 19
   rounds of degree 8 each: ≈ 3.6 KB per layer, ≈ 110 KB for 31 layers. Over the goal
   on its own. Rejected.
3. **R1CS + Spartan.** Four products per S-box: ≈ 600 cells per permutation, w ≈ 2^25.
   About 4× the prover work for the same verifier. Rejected.
4. **Holographic glue (setup-committed matrices, Spark or an honest fold).** It needs a
   second per-proof commitment (Spark's e-values, or the folded oracle) plus setup
   leaves: ≈ +85-115 KB. It would make the verifier ≈ 20 ms, but the owner allows
   100-400 ms. Rejected.
5. **A uniform recursion VM with logUp-GKR wiring.** The native verifier would read
   only a small program table, but the GKR proof adds ≈ 15-20 KB (≈ 103 KB total) and
   a compiler to a VM. Interface: small for the verifier, but it exposes an instruction
   set to every gadget author. Rejected.
6. **Retune M2 first** (P0 1/4, P1 1/8, setup 1/4, caps). Model: ≈ 18k permutations,
   so w fits 2^22. Gain: ≈ 3 KB and ≈ 50% of shrink prover time. Cost: M2 peak
   ≈ +10 GB (≈ 48 GB of 64), M2 finish ≈ +15 s, setup disk 54 GB. Rejected as default;
   caps are the reserve lever.
7. **A separate circuit implementation with differential tests.** Smaller change to
   neo-spartan, but two verifiers to keep in sync, which the owner's "one source of
   truth" rules out. Rejected.
8. **Pruned paths in the circuit** (a routing network for the shared top levels). It
   saves ≈ 5k permutations, but the cube stays 2^23 and the routing adds a value-driven
   wiring gadget. Rejected.

## Open questions and risks

Questions for the owner:

1. Is a verifier that re-runs the final verifier program on each call (≈ 60-70 ms of
   the ≈ 120-170 ms) acceptable on the chain node, or should the node cache the glue per
   key?
2. May `verify_final` use the randomized PiRLC check (+2^-184.8) instead of the exact
   ring products, so that native and circuit share one code path?
3. Is the half-and-half split of the remaining budget between M2 and the shrink the
   right policy, or should the shrink take less (its queries are cheap) so M2 keeps
   more?
4. Is the ≈ 2^-54 completeness gap from p3's resampling acceptable, or must the circuit
   model a second draw?
5. May the workspace pin `p3-whir` and `p3-sumcheck` to exactly 0.8.0, since our verifier
   copies their transcript?
6. Should the accumulator verifier (`Verifier::verify`) later move onto the same generic
   PiCCS code, so that the trace test can go away?

Risks:

- **Biggest: exact replication of the p3-whir 0.8 verifier** (stacked layout to eq and
  select statements, transcript-driver brackets, round batching, OOD, grinding, Suffix
  variable order). One mismatch makes every M2 proof fail natively, so existing tests
  catch it, but finding the cause is slow. Slice 2 builds it alone with parity tests
  against `verify_at`.
- **Circuit size.** The 2^23 fit (80-85%) rests on estimates. Overflow costs ≈ 8 KB and
  2× prover time, or the cap lever.
- **Verifier time** depends on glue nonzeros (5-8M est.). Slice 7 measures it.
- **Layer-0 sharp edges**: the canonical-split rule, the ρ decode limb gadget, the
  preimage layout. Each gets a parity test against the native function.
- **Shape determinism**: one value-dependent branch in generic code breaks the shrink
  verifier. The row-digest test and the hint-only value rule guard it.

## Next implementation step

Write `circuit/{mod,record,hash,poseidon2}.rs` (Backend, Native, Recorder, Duplex,
FoldTranscript, the permutation block) and prove parity with p3's challenger,
permutation and MMCS on recorded transcripts.

## Implementation slices

Each slice has its own tests under `tests/`. Every test is ≤ 300 s with `--release`.

0. **Spike.** Measure p3-whir at 2^23, rate 1/64, PoW 20, schedules [4,4,4,4,4] and
   [5,4,4,4]: bytes, time, peak RSS. Print M2's per-round query counts from the
   production `Relation`. Tests: one ignored measurement test (≈ 30 s).
1. **Circuit core.** `Backend`, `Native`, `Recorder`, `Form`, algebra helpers, `Duplex`,
   `FoldTranscript`, leaf sponge, Merkle path, permutation block. Tests: Native equals
   p3 (permutation, challenger samples and bits, grinding) and neo-transcript v1_1;
   recorded rows are satisfied by honest values; flipping any of 1,000 sampled cells
   breaks a row; non-canonical bits (value + p) are rejected; the row digest of a zero
   input equals that of an honest input.
2. **Generic WHIR verifier and `Profile`.** Tests: parity with p3 `verify_at` on plans of
   2^12-2^18 cells (1-4 tables, rates 1/2, 1/16, 1/64, PoW 0 and 8, schedules constant 4
   and [5,4,…]): both accept honest proofs; every proof field mutation is rejected by
   both; recorded rows are satisfied; counted permutations equal the model.
3. **Generic layer 1.** Move `verify` to `verify.rs` over `Backend`; port the verifier
   halves of gkr, sumcheck, setup, matrix, mle (structural reachability for the Eval_K
   DP), norm (all denominators nonzero), ring. Tests: every existing neo-spartan test
   passes unchanged (behavior preservation); recorded rows satisfied on a toy relation;
   `ProofView::zero` rows equal honest rows.
4. **Generic final verifier.** `final_verify.rs`, `final_rlc.rs`, `view.rs`, `FoldKey`,
   `CompressionKey` struct; `verify_final` calls the program on `Native`. Tests: toy
   finish then `verify_final` passes; PiCCS trace parity with neo-reductions'
   `ProtocolTrace`; ρ decode parity on 10,000 random digests; the randomized PiRLC
   rejects a parent with one changed commitment coefficient, one public word, one
   Eval_A value; key-only verification equals package verification; key bytes are
   strict.
5. **Shrink layout.** Regions, template, `mle::diagonal`, closed-form permutation part,
   glue column mapping. Tests: closed form equals the materialized MLE for N = 1..37 at
   random points; `diagonal` equals brute force; streamed glue equals CSR evaluation.
6. **Shrink argument.** `spartan.rs`, `shrink::{prove, verify, Shape}`. Tests: toy
   programs (a 100-permutation chain with Ext arithmetic and hints) prove and verify;
   every proof field and statement word mutation is rejected; a witness that breaks one
   row cannot be proven; counts that differ from `Shape` are rejected.
7. **nightstream integration.** `Prover::shrink`, `CompressionKey::verify`,
   `ShrinkProof` bytes. Tests: toy package finish → shrink → verify; a wrong state is
   rejected; strict codecs. Ignored production test: setup open, finish, shrink, verify
   with bytes, phase times, verifier time and peak RSS (≈ 15 + 40 + 23 s, within 300 s).
