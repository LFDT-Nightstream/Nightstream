# Candidate B: uniform CCS + SuperSpartan shrink proof (compression milestone 3)

Date 2026-10-07. Seat stance: CCS + SuperSpartan (ePrint 2023/552), WHIR as the PCS, one
degree-7 term so that each Poseidon2 S-box is one constraint, and a uniform layout: N identical
permutation blocks plus an irregular glue part.

"est." marks an estimate. Nothing in this file is measured unless it cites GROUNDING.md or
M2-DESIGN.md. Owner updates received during the design (2026-10-07):

- The final proof must be at most **200 KB (hard limit)**. The **goal is 100 KB or less**.
- The native verifier should take about **100-400 ms** on a CPU. Regenerating or evaluating part
  of the circuit is acceptable inside that budget.

## Summary

- **One shrink layer.** It proves "the M2 final proof verifies" with one WHIR commitment of a
  2^23-word witness. Rate 1/64, PoW 20 bits: **about 89-92 KB** (est.). The goal is met.
- **Uniform part.** Each Poseidon2 permutation is one 256-lane block of one of a few families.
  A block has 150 degree-7 S-box rows and one "predecessor" pointer for sponge and Merkle
  chaining. The verifier evaluates the uniform matrices in O(N) with N ≤ 2^15 blocks.
- **Glue part.** All other verifier arithmetic (about 0.95M rows, 6M nonzeros, est.). The
  native verifier **regenerates** it: it runs the same generic verifier code in an
  "evaluate" mode that adds up the matrix MLEs on the fly. No setup commitment, no extra proof
  bytes, key about 2.2 KB. Verifier **about 130-310 ms** (est.).
- **One source of truth.** The M2 layer-1 verifier is rewritten once, generic over a
  `Machine` trait. `Native` runs it on values. `Recorder` runs it as the circuit (witness mode
  or evaluate mode). The trait gives no way to read a value, so the circuit shape cannot depend
  on a value.
- **Two M2 retunes** keep the witness at 2^23: R1 (fuse PiRLC's combination into the layer-1
  ring row; needs owner approval) and R2 (P0 and P1 at rate 1/4).
- **200 KB fallback** without R1 and R2: a 2^24 witness at rate 1/16, about 119 KB (est.).

## Problem

M3 must turn the M2 `FinalProof` (1,600,157 B) into a proof of at most 200 KB, goal 100 KB.
A native verifier checks it against the expected `State` and a trusted key in 100-400 ms. The
statement is exactly `verify_final`: the state hash and canonical children check, PiCCS,
PiRLC, and the M2 layer-1 verifier. Constraints from GROUNDING.md §0 apply: Poseidon2 only,
post-quantum, about 114 bits proven with the Johnson bound, PoW ≤ 20 bits on the last layer
only, 64 GB, files < 1,500 lines, published techniques only, no new features or env vars.

Three facts make the shape non-obvious:

1. **Hashing dominates.** The M2 layer-1 verifier makes 23,589 permutations (measured, with
   pruned Merkle multiproofs). Layer 0 adds about 3.3k (state hash 2,320, PiCCS 948). In a
   rank-1 system each permutation costs about 600 rows. In a CCS with a degree-7 term it costs
   150 rows.
2. **The verifier must learn the matrix MLEs.** One WHIR opening of a 2^23-2^24 witness fits the
   budget. A second commitment does not fit 100 KB (GROUNDING.md §2). So a holographic opening
   of the circuit matrices (a late commitment) cannot live in the last layer.
3. **The verifier program is irregular.** Its data flow (which value feeds which operation) is
   not a repeated pattern. Some part of the circuit is always irregular, and someone must pay
   for it: the verifier (time), the proof (bytes), or a second layer (prover time and code).

The owner's 100-400 ms budget decides question 3: the verifier pays, but only for the
irregular glue, and the Poseidon2 part costs it O(N).

## Usage (caller's view)

README for the prover service and the chain verifier:

```rust
use nightstream::{Circuit, CompressionKey, Engine, FinalProof};

// Once per circuit (prover machine): the M2 setup files, unchanged from M2.
let circuit = Circuit::load("circuit.bin")?;
let prover = circuit.prover(Engine::Cpu, 114)?;
let setup = prover.compression_setup("/data/setup")?;     // 63-69 s, 27 GB on disk

// Once per circuit (verifier side): derive and pin the key. About 2.2 KB.
let key: CompressionKey = nightstream::Verifier::from_package(&circuit, Engine::Cpu, 114)?
    .compression_key()?;
std::fs::write("compression.key", key.to_bytes())?;

// Per proof (prover machine): fold steps, then finish. Finish now ends with the shrink layer.
let proof = prover.extend(&previous, &inputs)?;
let final_proof: FinalProof = prover.finish(&proof, &setup)?;    // est. 65-70 s, ≤ 100 KB
let bytes = final_proof.to_bytes();
```

Chain-side verifier. It needs no package and no files, only the pinned key:

```rust
let key = CompressionKey::from_bytes(PINNED_KEY)?;           // strict, canonical
let proof = FinalProof::from_bytes(&bytes)?;                  // strict, canonical
nightstream::verify_final(&key, &expected_state, &proof)?;    // est. 130-310 ms, 1 core
```

neo-spartan-level calls (what nightstream does inside `finish` and `verify_final`):

```rust
// nightstream/src/lifecycle/finish.rs (prover)
let layer1 = self.prove_layer1(envelope, setup)?;            // M2 as today (internal type now)
let statement = Layer0::new(key.layer0(), expected_state);  // implements neo_spartan::Statement
let shrink = neo_spartan::shrink::prove(key.shrink(), &statement, &layer1.layer0_witness(), &layer1.proof)?;

// nightstream/src/lifecycle/finish.rs (verifier)
let statement = Layer0::new(key.layer0(), expected_state);
neo_spartan::shrink::verify(key.shrink(), &statement, &proof.0)?;
```

The M2 proof stays reachable for tests through `neo_spartan::verify` (now
`verify_in::<Native>`), but it is no longer a public nightstream type. One proof type, one way
to verify it.

## Shape

### Load-bearing decisions

1. **One shrink layer, one commitment.** The shrink proof is SuperSpartan over one CCS. The
   witness `w` (2^23 Goldilocks words) is the only committed object. The public half `io`
   (`1`, iteration, z0, current) is computed by the verifier.
2. **CCS with four matrices and one degree-7 term**, the same shape as F′ (`A, B, C,
   SboxInput`): `f = (Az)∘(Bz) − Cz + (Sz)^7`. A row is either a product row (S row empty) or
   an S-box row (A and B rows empty). Linear rows use C only.
3. **Uniform blocks with predecessor pointers.** A block is one permutation of one *family*
   (how its 16 input lanes are formed). The only per-block data are the family id and one
   predecessor block index. The family patterns are fixed, so the uniform MLE is a sum over
   blocks with O(1) work each.
4. **Glue regeneration.** The verifier reruns the generic verifier code in evaluate mode and
   adds `ρ_j·coefficient·eq(r_x,row)·eq(r_y,col)` for every glue entry. Nothing is stored or
   committed.
5. **One generic verifier for M2 layer 1** (`verify_in<M: Machine>`). Native verification and
   the circuit are the same code. Layer 0 and the p3-whir verifier get circuit replicas (they
   belong to other owners), tied to their native forms by transcript-log equality tests and
   mutation tests.
6. **M2 retunes.** R1: layer 1 consumes PiRLC's 17 source claims and the ρ_i directly, so the
   circuit never multiplies ring elements exactly (saves 1.83M rows). R2: P0 and P1 at rate 1/4
   (saves about 7k permutations). Both keep M2's soundness argument (see the soundness map).

### Data structures

The column space of `z` has 2^24 positions: `w` (low half, committed, 2^23) and `io` (high
half, public, 10 used positions). The row space has 2^23 positions. Both are cut the same way:

```text
 index = block · 256 + lane                     block < 2^15, lane < 256
 lanes 0..184    the block's family lanes     (one permutation: rows and columns)
 lanes 184..256  glue slots                   (72 per block: 2.36M glue rows, 2.36M glue columns)
```

- **Block.** `(family: u8, pred: u32)`. 24.4k used blocks (est., with R1 and R2); the rest are
  padding blocks of family `Free` with a valid permutation of zero input.
- **Family lanes.** Up to 16 fresh input words `w`, 150 S-box outputs, 16 output lanes, and for
  `Compress` a direction bit, 4 sibling words and 4 mux outputs. At most 178 columns and 171
  rows per block. The S-box rows reference only the block's own lanes, the predecessor's 16
  output lanes, and the `one` column (round constants and length tags).
- **Glue row.** A product row `(A, B, C)` or a linear row `(C)`; entries `(column, coefficient)`.
  Glue row `k` lives at `(k / 72, 184 + k mod 72)`; glue column `g` likewise. Glue rows may
  reference any column: block lanes, glue columns, `io`.
- **Wire.** A `Copy` handle to a linear combination in the recorder's arena. Adds are free.
  A wire gets a column at its first structural use. When that use is a block lane (an absorbed
  proof word, a leaf value, a Merkle sibling) the lane is its home, so no copy row is needed.

The families (all from the workspace permutation, `neo_spartan::hash::permutation()`):

| Family | Input lanes | Used by |
| --- | --- | --- |
| `Absorb{n}` | lanes `<n` fresh, `n..12` zero, lane 12 = pred + n, 13..16 = pred | p3 `DuplexChallenger` absorb |
| `Squeeze` | all 16 = pred | duplex squeeze, `squeeze_digest_v1_1`, zero chunks |
| `Add{n}` | lanes `<n` = pred + fresh, others = pred (no pred: zero) | `absorb_v1_1`, `poseidon2_hash` |
| `Overwrite{n}` | lanes `<n` fresh, others = pred (no pred: zero) | `PaddingFreeSponge` leaf hash |
| `Compress` | `[mux(node, sib), mux(sib, node), 0…]`, node = pred lanes 0..4 | one Merkle level |
| `Free` | all 16 fresh | padding |

### The uniform closed form

Write `r_x = (x_hi, x_lo)` and `r_y = (top, y_hi, y_lo)` with 15 block bits and 8 lane bits.
For each family `f` and matrix `j`, the verifier computes three local numbers in
O(local nonzeros):

- `P_j[f] = Σ eq(x_lo, l')·P_{j,f}[l', l]·eq(y_lo, l)` (own lanes),
- `Q_j[f]` (the predecessor's output lanes),
- `K_j[f] = Σ eq(x_lo, l')·K_{j,f}[l']` (constants into the `one` column).

Then

```text
U_j(r_x, r_y) = (1 − top) · Σ_b eq(x_hi, b) · [ eq(y_hi, b)·P_j[f_b] + eq(y_hi, pred_b)·Q_j[f_b] ]
              +  eq(r_y, one) · Σ_b eq(x_hi, b) · K_j[f_b]
```

This is O(2^15) plus O(families × 3.5k). All families share the 130 inner S-box rows and the
output rows, so the local work is computed once plus a 16-row delta per family. The pointer
`pred_b` is arbitrary, so chains need not be contiguous: block ids follow emission order.

### One source of truth: the `Machine` trait

```text
             verify_in::<M>(…)   (M2 layer 1, one implementation)
             layer0::replay::<M>(…)   whir_replay::verify_at::<M>(…)
                    │
   ┌────────────────┼─────────────────────────┐
 Native        Recorder::witness          Recorder::evaluate(r_x, r_y, ρ)
 values,       glue rows + block list     one Ext accumulator, no storage
 p3 challenger values for every column    (the verifier's matrix MLE)
 p3 verify_at
```

- Shape and values come from one pass of one function. Witness mode and evaluate mode emit the
  same row stream; a test hashes both streams and compares them.
- `Machine` has no method that returns a value. Branches on data are impossible; `if` exists
  only on public constants (shapes, the key). Data-dependent choices of the native verifier
  become operations: a query index becomes bits, `g·ω^index` becomes a product over bits, a
  Merkle direction becomes a mux.
- Hints (`inverse`, bit and digit decompositions) are explicit: `hint` returns unconstrained
  columns and the caller adds the constraints. A missing constraint is visible in one place.
- The p3 parts that we do not own are replicas, generic over `Machine`, with three tests each:
  honest runs give the same challenger log as p3, every single mutation is rejected by both,
  and the recorded constraints hold exactly when the replica accepts.

### Module map

All new code lives in neo-spartan, which keeps every Plonky3 0.8 type. nightstream owns the
layer-0 replica because it owns layer 0.

| File | Owns | Lines (est.) |
| --- | --- | --- |
| `neo-spartan/src/machine/mod.rs` | `Machine` trait, `Ext3`, `K2`, `Kx3`, `Bits` value types and their gadget arithmetic | 450 |
| `neo-spartan/src/machine/native.rs` | `Native`: values, p3 challenger, p3 `verify_at` | 200 |
| `neo-spartan/src/machine/recorder.rs` | `Recorder`: wire arena, column placement, glue rows, blocks, witness and evaluate modes, layout | 650 |
| `neo-spartan/src/machine/families.rs` | Poseidon2 families: local matrices from the workspace constants, block traces, the uniform closed form | 550 |
| `neo-spartan/src/machine/sponge.rs` | Duplex challenger, `PaddingFreeSponge`, compress, full-path Merkle, `absorb_v1_1` replicas | 350 |
| `neo-spartan/src/whir_replay.rs` | Replica of p3-whir 0.8 `verify_at` with the p3-sumcheck stacked-layout verifier | 1,000 |
| `neo-spartan/src/shrink/mod.rs` | `ShrinkShape`, `ShrinkProof`, `Statement`, `prove`, `verify`, the transcript | 350 |
| `neo-spartan/src/shrink/spartan.rs` | Degree-8 zero-check and inner sum-check, prover and verifier | 450 |
| `neo-spartan/src/shrink/products.rs` | `Az, Bz, Cz, Sz` and `M_ρ(r_x, ·)` from blocks and glue | 350 |
| edits: `lib.rs gkr.rs sumcheck.rs matrix.rs mle.rs ring.rs norm.rs setup.rs pcs.rs` | verifier halves become generic (`verify_in`); R1 targets; PCS rate and PoW as parameters | +400 net |
| `nightstream/src/lifecycle/shrink.rs` | Layer-0 replica: state hash, canonical children, encHash, PiCCS replay, PiRLC ρ decode, handoff | 900 |
| edits: `nightstream/src/lifecycle/finish.rs`, `circuit.rs` | `finish`, `verify_final(key, …)`, `CompressionKey` | +200 |

Total about 5.9k new lines plus 0.6k changed lines. No file exceeds 1,000 lines.

### Type sketch

```rust
// ===== neo-spartan/src/machine/mod.rs =====
//! One verifier, three readings: native values, the recorded shrink witness,
//! or the shrink matrices evaluated at a point. Invariant: no method returns a
//! value, so generic code cannot branch on one and the circuit shape depends
//! only on public constants.

pub trait Machine: Sized {
    /// A base-field value (Native) or a wire (Recorder).
    type Gl: Copy;
    /// The layer-1 Fiat-Shamir state: p3 `DuplexChallenger`, or its replica.
    type Challenger;
    /// A WHIR opening: the p3 `PcsProof`, or its wire mirror (full Merkle paths).
    type Opening;
    type Error: From<Rejected>;

    fn constant(&mut self, value: Gl) -> Self::Gl;
    /// A proof or statement word. `None` only in evaluate mode.
    fn input(&mut self, value: Option<Gl>) -> Self::Gl;
    /// `Σ c_i·x_i + c`. Free in a circuit.
    fn linear(&mut self, terms: &[(Gl, Self::Gl)], constant: Gl) -> Self::Gl;
    /// `a·b`. One product row in a circuit.
    fn mul(&mut self, a: Self::Gl, b: Self::Gl) -> Self::Gl;
    /// Unconstrained helper columns computed from `inputs`; the caller constrains them.
    fn hint<const N: usize>(&mut self, inputs: &[Self::Gl], rule: fn(&[Gl]) -> [Gl; N]) -> [Self::Gl; N];
    fn assert_zero(&mut self, value: Self::Gl, what: &'static str) -> Result<(), Self::Error>;
    /// One Poseidon2 permutation. In a circuit: one block of the call's family.
    fn permute(&mut self, call: Call<'_, Self::Gl>) -> Permuted<Self::Gl>;

    fn observe(&mut self, challenger: &mut Self::Challenger, words: &[Self::Gl]);
    fn sample(&mut self, challenger: &mut Self::Challenger) -> Self::Gl;
    /// p3-whir 0.8 `verify_at` (Native) or `whir_replay::verify_at` (Recorder).
    fn verify_opening(
        &mut self,
        pcs: &Pcs,
        challenger: &mut Self::Challenger,
        root: [Self::Gl; 4],
        opening: &Self::Opening,
        points: &[Vec<Ext3<Self::Gl>>],
    ) -> Result<Vec<Vec<Ext3<Self::Gl>>>, Self::Error>;
}

/// How a permutation's input lanes are formed. The variant is the block family.
pub enum Call<'a, G> {
    Absorb { prior: &'a Permuted<G>, words: &'a [G] },
    Squeeze { prior: &'a Permuted<G> },
    Add { prior: Option<&'a Permuted<G>>, words: &'a [G] },
    Overwrite { prior: Option<&'a Permuted<G>>, words: &'a [G] },
    Compress { prior: &'a Permuted<G>, sibling: [G; 4], bit: G },
}

/// A permutation's output lanes and, in a circuit, its block.
#[derive(Clone, Copy)]
pub struct Permuted<G> {
    pub lanes: [G; 16],
    block: Option<u32>,
}

#[derive(Clone, Copy)] pub struct Ext3<G>(pub [G; 3]);   // Gl[w]/(w^3 − w − 1)
#[derive(Clone, Copy)] pub struct K2<G>(pub [G; 2]);     // F[u]/(u^2 − 7)
#[derive(Clone, Copy)] pub struct Kx3<G> { pub re: Ext3<G>, pub im: Ext3<G> }
pub struct Bits<G> { pub low_first: Vec<G> }

pub mod ext {
    /// Karatsuba: 6 product rows.
    pub fn mul<M: Machine>(m: &mut M, a: Ext3<M::Gl>, b: Ext3<M::Gl>) -> Ext3<M::Gl> { unimplemented!() }
    /// A hinted inverse checked by one `mul`; rejects zero.
    pub fn inverse<M: Machine>(m: &mut M, a: Ext3<M::Gl>) -> Result<Ext3<M::Gl>, M::Error> { unimplemented!() }
}
pub mod bits {
    /// Canonical 64-bit decomposition of a sampled word, with p3's uniform-bits rule:
    /// the circuit rejects the single value p3 would resample (p − 1).
    pub fn sampled_index<M: Machine>(m: &mut M, word: M::Gl, width: usize) -> Result<Bits<M::Gl>, M::Error> { unimplemented!() }
    /// `base^index` from the index bits: one product row per bit.
    pub fn pow<M: Machine>(m: &mut M, base: Gl, index: &Bits<M::Gl>) -> M::Gl { unimplemented!() }
}

// ===== neo-spartan/src/machine/native.rs =====
pub struct Native;
impl Machine for Native {
    type Gl = Gl;
    type Challenger = crate::hash::Challenger;
    type Opening = crate::pcs::Opening;
    type Error = crate::Error;
    fn constant(&mut self, value: Gl) -> Gl { unimplemented!() }
    // … the other methods compute directly; assert_zero returns Err(Rejected(what))
}

// ===== neo-spartan/src/machine/recorder.rs =====
pub(crate) struct Recorder<'e> {
    mode: Mode<'e>,
    arena: Vec<Combination>,          // wires: linear combinations of columns
    placement: Vec<Option<Column>>,   // a wire's home, fixed at first structural use
    blocks: Vec<Block>,               // (family, pred), emission order
    glue_rows: usize,
    glue_columns: usize,
}
pub(crate) enum Mode<'e> {
    /// Record rows (CSR) and every column's value.
    Witness(Box<Recorded>),
    /// Add `ρ_j·c·eq(r_x,row)·eq(r_y,col)` for every entry; store nothing.
    Evaluate(&'e mut Evaluation),
}
#[derive(Clone, Copy)] pub(crate) struct Column(u32);
#[derive(Clone, Copy)] pub(crate) struct Block { family: Family, pred: u32 }
pub(crate) struct Recorded { glue: Csr, blocks: Vec<Block>, block_inputs: Vec<[Gl; 16]>, glue_values: Vec<Gl> }
pub(crate) struct Evaluation { x: SplitEq, y: SplitEq, rho: [Ext; 4], glue: Ext, uniform: Ext }

// ===== neo-spartan/src/shrink/mod.rs =====
/// The verifier's constants of the shrink layer. Part of the compression key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShrinkShape {
    witness_variables: usize,   // 23
    log_inv_rate: usize,        // 6
    pow_bits: usize,            // 20
    security_bits: f64,         // the shrink share of the compression budget
    counts: Counts,             // blocks, glue rows, glue columns, nonzeros: detects version drift
}

/// What the shrink proof proves before layer 1. Implemented by the owner of
/// layer 0 (nightstream). `replay` must use only `io`, constants and `witness`.
pub trait Statement {
    type Witness;
    /// Public words, in transcript order; they fill the `io` half.
    fn io(&self) -> Vec<Gl>;
    /// Constants bound before any challenge (the layer-0 part of the key).
    fn key_words(&self) -> Vec<Gl>;
    fn replay<M: Machine>(&self, m: &mut M, io: &[M::Gl], witness: Option<&Self::Witness>)
        -> Result<Handoff<M>, M::Error>;
}

/// Layer 0's output: the transcript state for the layer-1 challenger and,
/// under R1, PiRLC's sources and challenges instead of their combination.
pub struct Handoff<M: Machine> {
    pub seed: [M::Gl; 4],
    pub sources: Vec<ClaimWires<M>>,   // 17 CE claims at one point
    pub rhos: Vec<[M::Gl; 54]>,        // ρ_i coefficients in [−2, 2]
}

/// Opaque shrink proof. About 89-92 KB at the production shape (est.).
#[derive(Clone, Serialize, Deserialize)]
pub struct ShrinkProof {
    commitment: pcs::Commitment,
    outer: Vec<[Ext; 8]>,     // 23 rounds: h(0), h(2..=8)
    claims: [Ext; 4],         // Az, Bz, Cz, Sz at r_x
    inner: Vec<[Ext; 2]>,     // 24 rounds: h(0), h(2)
    opening: pcs::Opening,    // w at r_y[..23]
}

pub fn prove<S: Statement>(key: &ShrinkKey<'_>, statement: &S, witness: &S::Witness, layer1: &crate::Proof)
    -> Result<ShrinkProof, Error> { unimplemented!() }
pub fn verify<S: Statement>(key: &ShrinkKey<'_>, statement: &S, proof: &ShrinkProof)
    -> Result<(), Error> { unimplemented!() }

/// The layer-1 relation plus the shrink shape. Derived, never received.
pub struct ShrinkKey<'k> { relation: Relation<'k>, shape: ShrinkShape }

// ===== neo-spartan/src/lib.rs (M2, now generic) =====
pub fn verify(relation: &Relation<'_>, transcript: Poseidon2Transcript, claim: &Claim, proof: &Proof) -> Result<(), Error> {
    // verify_in::<Native> on wires that wrap the values
    unimplemented!()
}
pub(crate) fn verify_in<M: Machine>(m: &mut M, relation: &Relation<'_>, handoff: Handoff<M>, proof: &ProofWires<M>)
    -> Result<(), M::Error> { unimplemented!() }

// ===== nightstream/src/lifecycle/shrink.rs =====
pub(crate) struct Layer0<'a> { constants: &'a Layer0Constants, state: &'a Stage1State }
pub(crate) struct Layer0Witness { running: Vec<CeClaim>, fresh: CcsClaim, pi_ccs: pi_ccs::Proof, pi_rlc: pi_rlc::Proof }
impl neo_spartan::shrink::Statement for Layer0<'_> {
    type Witness = Layer0Witness;
    fn io(&self) -> Vec<Gl> { unimplemented!() } // iteration, z0, current
    fn key_words(&self) -> Vec<Gl> { unimplemented!() }
    fn replay<M: Machine>(&self, m: &mut M, io: &[M::Gl], w: Option<&Layer0Witness>) -> Result<Handoff<M>, M::Error> { unimplemented!() }
}

// ===== nightstream/src/circuit.rs =====
/// The verifier's trusted constants: M2 key, layer-0 constants, shrink shape. About 2.2 KB.
pub struct CompressionKey { layer1: neo_spartan::Key, layer0: Layer0Constants, shrink: neo_spartan::shrink::ShrinkShape }
/// The finished proof: the shrink proof only.
pub struct FinalProof(neo_spartan::shrink::ShrinkProof);
pub fn verify_final(key: &CompressionKey, expected_state: &State, proof: &FinalProof) -> Result<(), Error> { unimplemented!() }
```

What the public surface hides: the CCS, the layout, the families, the recorder, the replicas,
the sum-checks and the WHIR configuration. nightstream sees `Machine`, the value types, the
gadget functions and `Statement`, because layer 0 is its code. Nothing else of the circuit is
public, and no p3 type crosses the crate boundary.

### Shrink protocol and transcript order

Challenger: a p3 `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>` with the workspace
permutation.

1. Observe the domain chunk `Nightstream/SuperNeo/shrink/v1` (12 words).
2. Observe the key words: M2 key, layer-0 constants, `ShrinkShape` (rate, PoW, counts).
3. Observe `io`: iteration, z0, current (9 words). The verifier checks iteration is canonical
   and positive before step 1.
4. WHIR commit of `w` (2^23 base words, rate 1/64): observe the root.
5. τ ← 23 Ext.
6. Outer sum-check, 23 rounds: `Σ_x eq(τ,x)·((Az)(Bz) − Cz + (Sz)^7)(x) = 0`. Each round
   observes `h(0), h(2..8)` and samples `r_x[i]`.
7. Observe `a, b, c, s` (the four MLEs at `r_x`). Check `last = eq(τ,r_x)·(a·b − c + s^7)`.
8. ρ ← Ext. Claim `a + ρb + ρ²c + ρ³s`.
9. Inner sum-check, 24 rounds over the column space: `Σ_y M_ρ(r_x,y)·z(y)`, degree 2.
10. WHIR `verify_at` of `w` at `r_y[..23]` (p3 binds the value, OOD samples, PoW 20 per round).
11. Final check: `last = M_ρ(r_x,r_y)·((1 − r_y[23])·w(r_y[..23]) + r_y[23]·io(r_y[..23]))`.
    `M_ρ(r_x,r_y) = Σ_j ρ^j (U_j + G_j)`: closed form for `U`, one evaluate-mode pass for `G`.

### What the circuit replays (statement of the shrink proof)

In this order, all on one `Recorder`:

1. **Layer 0** (nightstream replica): state hash (`poseidon2_hash`, 27,819 words, 2,320
   blocks); canonical children split of the parent public input (sign bit and 16 binary digits
   per coordinate, same rule as `split_b_matrix_k`); `encHash` (marker and 256 digest bits,
   canonical); PiCCS: reset, domain chunk, statement (prior digest = state digest, fresh
   commitment, fresh x), α and γ coins by the `read_pair_v1_1` schedule, 28 rounds of 9 K
   coefficients, terminal check with `f`, `range_product`, γ powers, then the output absorb;
   PiRLC: 17 × (absorb `[4, i]`, `squeeze_digest_v1_1`, base-5 decode of the 256-bit integer
   into 54 digits by limb arithmetic); handoff: domain chunk, digest.
2. **Layer 1** (`verify_in::<Recorder>`), unchanged logic except R1: P0 root, histogram, early
   GKR, early values, λ, quotient, ζ, linear sum-check, finals, the honest-fold partials, P1
   root, 125 setup queries (leaf sponge, 24 compress levels, fold weights), late GKR, P1
   opening, P0 opening.
3. **p3-whir replica** for P0 and P1: initial OOD, `add_claim_at` per batch, WHIR seeds as
   constant absorbs, initial fold (batching challenge, sum-check rounds), per round (root,
   OOD, query indices by uniform bits, full Merkle paths, fold evaluations, round batching,
   sum-check), final polynomial, final queries, final sum-check, the final identity with the
   stacked-layout constraint weights.

Two deliberate differences from the native code, both stricter or equivalent:

- **Full Merkle paths instead of pruned multiproofs.** The witness generator restores each
  query's full path from p3's `PrunedMerklePaths`. A full path exists for every query exactly
  when the pruned proof verifies, up to Poseidon2 collisions.
- **No rejection loop.** p3 resamples a uniform-bits draw only when the word is `p − 1`
  (Goldilocks, ≤ 32 bits). The circuit rejects that word instead. Soundness: the circuit
  accepts a subset of what p3 accepts. Completeness: an honest prover fails with probability
  about 760·2^-64 ≈ 2^-54 per proof. In that case the shrink proof cannot be made for this
  M2 proof. This is negligible.

### M2 retunes

**R1: fuse PiRLC's combination into the layer-1 ring row (needs owner approval).** Today the
verifier computes the parent claim `Σ_i ρ_i ⋆ y_i` exactly (17 sources × 37 ring vectors ×
2,916 products = 1.83M products), and layer 1 absorbs the parent claim. Under R1, layer 1
takes the 17 sources and the ρ_i. Its target becomes unreduced:

```text
Σ_b ω_b(X)·z_b(X) − Σ_rows λ^row Σ_i ρ_i(X)·y_{i,row}(X) = Q(X)·Φ81(X)
```

Both sides have degree ≤ 106, so the quotient keeps 53 coefficients and the error stays
`(2D − 2)/|Ext|`, the same "quotient at zeta" term M2 charges today. The verifier needs only
`ρ_i(ζ)` and `y_{i,row}(ζ)` (629 ring evaluations, about 110k circuit rows). The parent claim
words leave the layer-1 statement: the fold transcript already binds every source (the state
digest binds the running claims, PiCCS absorbs the fresh claim and all 17 outputs, the ρ_i are
squeezed after them). PiRLC itself (paper construction) is unchanged; only the place where
its output is evaluated moves. Native M2 verification cost does not change in a measurable way.

**R2: P0 and P1 at rate 1/4.** The Johnson-bound round-0 query count drops from about 240 to
about 120 per opening. In-circuit permutations drop by about 7k (est.). Cost: P0's codeword
grows from 2^29 to 2^30 words (+4 GB), finish +3-5 s (est.), finish peak about 42 GB (est.,
from 37.3 GB). `Pcs::new` still derives each configuration from the same target, so M2's
soundness argument does not change.

### Soundness map

Notation: ε(x) = 2^-x. |Ext| ≈ 2^192. L is a WHIR candidate-list size (Johnson bound).

| # | Term | Bound | Charged where |
| --- | --- | --- | --- |
| 1 | Folding scheme (all folds, incl. the terminal PiCCS + PiRLC) | `fold_error_bits` (existing) | `Params` |
| 2 | M2 layer 1: P0 and P1 openings with every outer draw (as today), incl. "quotient at zeta" `(2D−2)/|Ext|`, unchanged by R1 | ≤ ε(β1), each commitment ε(β1+1) | `Relation::new` |
| 3 | M2 setup queries (honest fold) | `L1·2^-t`, t = ⌈β1 + 3 + log2 L1⌉ | `query_count` |
| 4 | Shrink WHIR opening of `w`, Johnson bound, PoW ≤ 20 per round | ε(β2 + 1) | `Pcs::new` report |
| 5 | Zero-check τ | 23/|Ext| × L | outer term |
| 6 | Outer sum-check | 23 × 8/|Ext| × L | outer term |
| 7 | ρ batching of four matrices | 3/|Ext| × L | outer term |
| 8 | Inner sum-check | 24 × 2/|Ext| × L | outer term |
| 9 | Full paths vs pruned paths | Poseidon2 collision resistance (already assumed by M2) | — |
| 10 | Rejection rule `p − 1` | 0 (circuit is stricter) | — |
| 11 | Circuit correctness (the replicas impose every native check) | tests, not a bound | slices 3, 5, 6, 7 |

Terms 5-8 are about 2^-180·L together, so the shrink layer costs ε(β2) with β2 the WHIR target.
The compression budget is split in two: `ε(β1) + ε(β2) ≤ ε(compression_security_bits)`. With
today's 117-bit share, β1 = β2 = 118. This raises each WHIR target by about 4 bits over the
measured table (114), which adds about 4% to the shrink proof (included below).

Composition: if the native verifier accepts, the WHIR and sum-check extractor gives a `z` that
satisfies the CCS; a satisfying `z` contains an M2 proof and layer-0 data that the replayed
verifiers accept; M2 and the folding scheme then give the accumulator claim. This uses the
standard heuristic of every recursive Fiat-Shamir argument: the Fiat-Shamir hash of M2 is also
computed inside the circuit. No other assumption is added: no capacity conjecture, no new
hash family, PoW only on the last layer.

### Costs

**Poseidon2 blocks in the circuit** (one block per permutation, full Merkle paths):

| Part | Today's M2 (est.) | With R1 + R2 (est.) |
| --- | --- | --- |
| Layer 0: state hash (27,819 words, add-mode, +1 pad) | 2,320 | 2,320 |
| Layer 0: PiCCS (domain 1, statement 122, coins 4, rounds 56, outputs 765) | 948 | 948 |
| Layer 0: PiRLC ρ (17 × 2), handoff (2) | 36 | 36 |
| Layer 1: handoff, shape/key/profile/statement words (measured 201) | 201 | ~30 (R1 drops the parent claim) |
| Layer 1: P0 root, histogram (7,345 words), early GKR (measured 1,068) | 1,068 | 1,068 |
| Layer 1: ring, sum-check, partials, P1 root, indices (measured 57) | 57 | 57 |
| Setup leaves: 125 × (18 sponge + 24 compress); measured 5,166 pruned | ~5,900 | ~5,950 (β1 +1 bit) |
| P1 opening incl. late GKR; measured 8,470 pruned | ~10,500 | ~7,100 |
| P0 opening; measured 8,627 pruned | ~10,600 | ~7,100 |
| **Total** | **~31,600** | **~24,600** |

Capacity is 2^15 = 32,768 blocks. Without R2 the margin is about 3%, which is too small for an
estimate. R2 gives about 25% margin. Slice 0 measures the real count.

**Glue** (est.; "R" = product rows, "L" = linear rows):

| Component | Main operations | Rows |
| --- | --- | --- |
| p3-whir replica arithmetic, P0 + P1 (~520 queries after R2) | per query: 64-bit index decomposition, `ω^index` (≈25 R), fold evaluation (66-90 R), select weight (≈200 R) | 230k |
| Setup honest-fold leaves (125) | `g·ω^j` by bits, `y^-1`, 8 weights, 26 oracles × 8 values × Ext | 135k |
| Norm histogram (7,345 entries) | one Ext inverse and one base × Ext product per entry | 88k |
| Eval_K long division (3,626 transitions) | one `Kx` product (18 R) per transition | 90k |
| GKR, early and late | ~520 rounds of degree ≤ 4 with Lagrange interpolation, `eq` per layer | 60k |
| R1 target | 629 ring evaluations at ζ (162 R each) + 629 Ext products | 110k |
| Matrix structure | lane tables per class, step MLEs, early and late leaf checks | 40k |
| Index bits and direction-bit copies (~760 draws) | 64 boolean rows, canonical check, 1 copy row per Merkle level | 70k |
| Linear sum-check, mixing, finals, honest-fold partials | | 15k |
| PiCCS replay | ~14k K products (γ powers to 4,400, Eval_K/Eval_A weighting twice, 28 Horner rounds), `eq` | 70k |
| PiRLC ρ decode | 17 × (4 canonical words, 256-bit limb identity with carries, 54 digits in [0, 4]) | 22k |
| Statement: children split, encHash, parent packing | 270 coordinates × ~35 rows, 256 digest bits | 12k |
| Constants, roots, zero lanes, seeds | | 10k |
| **Total** | R ≈ 650k, L ≈ 300k | **~950k (range 0.8-1.3M)** |

By operation type: about 60k Ext products (6 R each), 60k base × Ext products (3 R), 14k K
products (3 R), 3.7k Kx products (18 R), 55k boolean rows, 25k base products in power chains.
Merkle muxes are inside the `Compress` family (1 boolean and 4 mux rows per level, about 15k
levels), so they are uniform, not glue.

**Witness and nonzeros** (est.):

| Item | Value |
| --- | --- |
| Committed witness `w` | 2^23 words (8,388,608): blocks 24.6k × ≤178 lanes = 4.4M, glue ≈ 0.75M columns, rest padding |
| Rows | 2^23: blocks ≤171 each, glue ≈ 0.95M |
| Uniform nonzeros | ≈ 3.5k per block, ≈ 86M (prover only) |
| Glue nonzeros | ≈ 6M (range 5-9M) |

**Proof bytes** (WHIR from the measured table; 2^23 interpolated; +4% for the 118-bit target;
plus 4,416 B outer rounds, 96 B claims, 1,152 B inner rounds, about 100 B framing):

| Witness | Rate | PoW | WHIR (est.) | Total (est.) | WHIR prove (est.) | Shrink peak (est.) |
| --- | --- | --- | --- | --- | --- | --- |
| 2^23 | 1/16 | 20 | ~103 KB | ~109 KB | ~4.5 s | ~5 GB |
| 2^23 | 1/32 | 20 | ~94 KB | ~100 KB | ~9 s | ~7 GB |
| **2^23** | **1/64** | **20** | **~86 KB** | **~92 KB** | **~17 s** | **~10 GB** |
| 2^23 | 1/256 | 20 | ~75 KB | ~81 KB | ~65 s | ~25 GB |
| 2^24 | 1/16 | 20 | ~113 KB (109 measured at 114 bits) | ~119 KB | ~9 s (8.7 measured) | ~9 GB |
| 2^24 | 1/64 | 20 | ~95 KB | ~101 KB | ~35 s | ~18 GB |

Recommended: **2^23 at rate 1/64, PoW 20: about 92 KB.** The 200 KB fallback is **2^24 at rate
1/16, about 119 KB**: it needs neither R1 nor R2 (the exact PiRLC products become a second
uniform family of 629 ring-product blocks, 2.6M columns), and it saves about 8 s of shrink
prover time and 5 GB of M2 prover memory. It costs about 27 KB more than the goal design.

**Prover** (est., CPU, production package):

| Step | Time | Peak |
| --- | --- | --- |
| M2 finish with R2 (layer 0 + layer 1) | 43-45 s (measured 39.7 s before R2) | ~42 GB |
| Witness generation (generic verifier in witness mode, 24.6k block traces) | 1-2 s | 0.3 GB |
| `Az, Bz, Cz, Sz` (blocks + glue) | 0.3 s | 0.3 GB |
| Outer sum-check (degree 8, 2^23 rows, parallel) | 1-2 s | 1 GB |
| `M_ρ(r_x, ·)` over 2^24 columns, inner sum-check | 1 s | 0.8 GB |
| WHIR commit and open (2^23, 1/64), PoW grinding | ~18 s | ~10 GB |
| **Finish total** | **~65-70 s** | **~42 GB (M2 part)** |

The production test stays at about 15 s compile + 70 s finish + 0.3 s verify, under 300 s.
The shrink part runs after the M2 prover state is dropped, so it does not add to the 42 GB.

**Native verifier** (est., one core):

| Step | Time |
| --- | --- |
| Decode, shrink transcript, two sum-checks | < 1 ms |
| WHIR `verify_at` (2^23, 1/64, PoW 20: ~35 queries in round 0) | 3-5 ms |
| Uniform closed form (2^15 blocks, ~6 families' local sums) | ~3 ms |
| Glue regeneration in evaluate mode (≈0.95M rows, ≈6M nonzeros, ~12 base products each) | 120-300 ms |
| **Total** | **~130-310 ms** |

Peak verifier memory about 60 MB (the wire arena). Key about 2.2 KB: the M2 key (1,912 B), the
layer-0 constants (verifier-context digest, public width, `f` terms, `b`, `k_rho`, T, κ, dims)
and the shrink shape. If measurement shows more than 400 ms, two options exist: (O1) more
uniform families for repeated arithmetic (STIR query arithmetic, setup-leaf arithmetic,
histogram entries: about −2.5M nonzeros), (O2) regenerate independent sections on several
threads, with section offsets pinned in the key counts.

### Where the stance breaks

1. **The glue is not succinct.** The verifier pays O(glue nonzeros), about 6M. The uniform
   layout makes Poseidon2 cheap for the verifier; it does not make the verifier program
   regular. A succinct glue needs a holographic opening, which needs a second, late
   commitment. That does not fit one layer under 100 KB.
2. **Uniformity pays only for big gadgets with few inputs and outputs.** A permutation has
   about 3.5k internal nonzeros and 16-28 input or output wires, so a block pattern wins by
   about 150×. An Ext product has 6 rows and 6 input or output wires; a uniform Ext family
   would still need one wiring entry per row, so it gains nothing.
3. **The block cliff.** Capacity is 2^15 blocks at 2^23. Crossing it doubles the witness
   (+9 KB proof, +18 s prover, +8 GB). Without R2 the estimate is 97% of capacity.
4. **Lanes are wasted.** A permutation needs at most 178 of 256 lanes. The design puts the glue
   in the spare 72 lanes of each block, so the waste becomes glue capacity, at the cost of a
   less obvious layout function.
5. **Ring products are not Poseidon2-shaped.** Exact PiRLC needs 1.83M products. Either R1
   removes them, or a second uniform family carries them in a 2^24 witness.
6. **p3 has data-dependent structure** (pruned multiproofs, rejection loops, stacked layouts,
   pattern seeds). The circuit uses equivalent fixed forms, so the replica is a second
   implementation that tests must keep equal to p3 0.8.

## Synthesis decision

Filled in by arena.

## Tradeoffs accepted

- We accept a verifier of about 130-310 ms that contains the circuit generator, in exchange for
  no setup commitment, no extra proof bytes and a 2.2 KB key.
- We accept rewriting the M2 layer-1 verifier generic over `Machine` (about 900 lines touched),
  in exchange for one implementation that is the native verifier and the circuit at the same
  time.
- We accept two replicas (layer 0, p3-whir) that tests keep equal to their native forms, because
  making Lean-mirrored neo-reductions code or p3 code generic would touch other owners.
- We accept explicit-call arithmetic (`ext::mul(m, a, b)` instead of `a * b`) in the M2
  verifier, because hidden builder state behind operators would hide the data flow.
- We accept R1 (an M2 protocol-boundary change) and R2 (+5 GB M2 prover memory), in exchange
  for a 2^23 witness: −9 KB proof, −18 s and −8 GB in the shrink prover.
- We accept a 2^-54 per-proof chance that an honest proof hits the `p − 1` sample and cannot
  be shrunk, in exchange for a fixed circuit with no rejection loop.
- We accept that the shrink proof is not zero-knowledge. Nothing in GROUNDING.md asks for it.
- We accept that the M2 `FinalProof` stops being a public type. One proof type, one verify.

## Alternatives considered

1. **R1CS + Spartan (2019/550).** About 600 rows and 600 columns per permutation, so about 15M
   rows and a 2^24 witness. Proof about +9 KB, prover about 2× slower. It hides nothing more
   from callers. Lost on size and prover time.
2. **Holographic glue (setup commitment plus an honest fold or Spark).** It makes the verifier
   O(log) for the glue. But the fold of the setup oracle must be committed after `r_x, r_y`,
   which is a second commitment: about +90-110 KB (est.), plus setup leaves. Over 100 KB in one
   layer; under 200 KB only with two layers. Lost: the 100-400 ms budget makes it unnecessary.
3. **Two layers (holographic inner, small outer).** The outer layer still needs its own glue
   answer, the prover runs twice, and the code doubles. Lost on code size and prover time for
   no verifier-time need.
4. **Data-parallel GKR for the permutations (2023/1284 style).** Only the permutation inputs
   and outputs are committed, so the witness shrinks to about 2^22. But the GKR proof bytes are
   about (30 layers) × (15 rounds) × (8 values) × 24 B ≈ 86 KB. Lost on proof size.
5. **AIR with preprocessed wiring columns.** Wiring needs a permutation argument whose `σ`
   columns the verifier must evaluate: either holography (alternative 2) or regeneration (this
   design, with more machinery). Lost.
6. **Trace p3's own verifier with a symbolic field type.** p3 code branches on values (Merkle
   directions, sample loops), so the traced shape would depend on the witness. p3's `Field`
   trait is also very large. Lost.
7. **A uniform VM with a program ROM.** Every step pays for the widest instruction, about 3-10×
   witness growth, and the ROM is O(steps) for the verifier anyway. Lost.

## Open questions and risks

Questions for the owner:

1. Do you approve **R1**: layer 1 consumes PiRLC's 17 sources and the ρ_i, and the verifier
   evaluates their combination at ζ inside the ring row instead of computing the parent claim?
   Without it, the goal design moves to a 2^24 witness (about 101 KB at rate 1/64).
2. Do you approve **R2** (P0 and P1 at rate 1/4, M2 finish peak about 42 GB, +3-5 s)?
3. Is a verifier that **contains the circuit generator** acceptable on your chain (code size,
   about 60 MB peak memory, 130-310 ms)?
4. Should the compression budget be split **equally** between M2 and the shrink layer
   (β1 = β2 = 118 at today's 117-bit share)?
5. Do you accept the **standard recursion heuristic** (M2's Fiat-Shamir hash is also computed
   in the circuit)? Every recursive Fiat-Shamir design depends on it.
6. Should the M2 `FinalProof` become an internal type, with `finish` returning the shrink proof?

Risks:

- **WHIR sizes at 2^23 are interpolated**, not measured. The 8 KB margin to 100 KB could shrink.
- **Block count.** The unpruned-path overhead (+4.7k) and the R2 saving (−7k) are estimates.
- **Replica fidelity.** The p3-whir 0.8 transcript (seeds, codecs, stacked layout, sum-check
  basis, coefficient order of `sample_algebra_element`, pop order of the duplex output buffer)
  must match exactly. A p3 upgrade breaks the replica; the version must stay pinned.
- **Verifier time** depends on glue nonzeros; 2× the estimate exceeds 400 ms without O1 or O2.
- **Non-native base-5 decoding** of the PiRLC digest (256-bit integers in limbs) is easy to get
  wrong. Edge digests (zero, `p − 1` words) need tests against `decode_pi_rlc_coefficients`.
- **The generic M2 rewrite** could regress M2. All existing M2 tests must pass on `Native`.

## Next implementation step

Slice 0: measure p3-whir 0.8 at 2^23 (rates 1/32 and 1/64, PoW 20, 118 bits) and count the
in-circuit permutations of the production M2 verifier with full paths, with and without R2.

## Implementation slice plan

Every test runs with `--release` and a 300 s timeout. Production tests are `#[ignore]`.

| Slice | Content | Tests |
| --- | --- | --- |
| 0. Spike | WHIR byte/time/peak table at 2^23; an unpruned permutation counter for the M2 verifier; M2 with R2 (finish time, peak RSS, block count) | ignored measurement tests, one per item, each < 300 s |
| 1. Machine core | `Machine`, `Native`, `Recorder` (witness and evaluate), layout, wire placement, `ext`, `k`, `kx`, `bits` gadgets | evaluate mode equals the brute-force MLE of the recorded matrices at random points (toy circuits); witness and evaluate emit the same stream; each gadget: honest values satisfy, each single mutation violates; `Native` equals `Recorder` values |
| 2. Families | Poseidon2 constants from the workspace seed, block traces, local matrices, predecessor links, the closed form | block output equals `hash::permutation()` on random inputs; closed form equals the brute-force MLE for 8-64 blocks with random links; each mutated trace cell violates a row |
| 3. Sponges | duplex, `PaddingFreeSponge`, compress, full-path Merkle, `absorb_v1_1`, pruned-path restoration | challenger logs equal p3 `DuplexChallenger` and neo-transcript on random operation sequences; Merkle roots equal p3 MMCS; restored paths reproduce p3 pruned proofs; mutated siblings reject |
| 4. Shrink argument | CCS products, degree-8 zero-check, inner sum-check, PCS rate and PoW parameters, transcript | toy circuits prove and verify; every proof field and every `io` word mutated rejects; verifier `M_ρ` equals the prover's |
| 5. WHIR replica | `whir_replay::verify_at` generic over `Machine` | on openings from `Pcs` at 2^10-2^16 with several tables and points: replica (`Native`) accepts and logs the same challenger operations as p3; every opening field mutated is rejected by p3 and by the replica, and the recorded circuit is unsatisfied |
| 6. Generic M2 | `verify_in<M>`; R1 targets; R2 rates | all existing M2 tests pass on `Native` (toy, ignored production); `Recorder` at toy size is satisfiable; every mutated M2 proof part makes it unsatisfiable; R1: a parent claim that differs in one coefficient rejects |
| 7. Layer-0 replica | `nightstream/src/lifecycle/shrink.rs` | transcript events equal neo-reductions `ProtocolTrace` on toy packages; state digest equal; ρ digits equal `decode_pi_rlc_coefficients` on random and edge digests; each mutated `FinalProof` part makes the circuit unsatisfiable |
| 8. API | `finish`, `verify_final(key, …)`, `CompressionKey` derivation and codec | toy end-to-end prove and verify (< 300 s); wrong state, wrong key, mutated proof bytes reject; ignored production test reports proof bytes, finish time, peak RSS, verify time |
