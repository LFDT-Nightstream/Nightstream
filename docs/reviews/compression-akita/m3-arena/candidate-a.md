# M3 candidate A: one R1CS + Spartan shrink layer, no holography

Date 2026-10-07. Seat: R1CS + Spartan (ePrint 2019/550) with WHIR as the PCS, one shrink
layer, no holography. The native verifier rebuilds the shrink circuit from the package and
the compression key and evaluates the matrix MLEs itself. "est." marks an estimate. Every
number that is not marked "measured" comes from the estimator described in "Costs".

Owner targets used here (coordinator updates of 2026-10-07): final proof **at most 200 KB
(hard)**, **goal 100 KB or less**; native verifier **about 100-400 ms CPU**.

**Result in one line.** One Spartan proof over a 2^25-row R1CS, one WHIR commitment at rate
1/64 with 20-bit PoW: **about 90 KB**, **about 250 ms** native verify, **about 75 s and 30 GB**
for the shrink prover. M2 does not change, except for a 1-bit security split.

## Problem

M2 gives a `FinalProof` of 1.6 MB that `verify_final(state, key, proof)` checks in 32 ms.
M3 must prove "`verify_final` accepts" in one small proof. Four facts make the shape hard:

1. **The statement is mostly hashing.** The M2 layer-1 verifier does 23,589 Poseidon2
   permutations natively. Layer 0 (terminal state hash, PiCCS and PiRLC transcripts) adds
   3,302 more. A circuit cannot share Merkle siblings across queries the way the native
   multiproof does, so the circuit does about 32,700 permutations. In R1CS one permutation
   is 616 rows (4 rows per x^7 S-box, 150 S-boxes, 16 output rows).
2. **The circuit must match p3-whir 0.8 `verify_at` exactly.** M2's two WHIR openings use
   p3's duplex challenger, its `fs` domain separator, its stacked layout and its pruned
   Merkle multiproofs. Two of these are `pub(crate)` in p3.
3. **No holography in this seat.** Then the shrink verifier must evaluate
   Ã, B̃, C̃ at (r_x, r_y) itself. With 2^25 rows and about 300 M nonzero entries, plain
   evaluation is far too slow. The design only works if almost all rows have a closed
   form.
4. **One source of truth.** The native verifier, the witness generator and the shape the
   shrink verifier rebuilds must be the same code. Otherwise a fix in one place does not
   reach the others, and a gap between the circuit and the native check is a soundness bug.

Constraints honored (GROUNDING §0): Poseidon2-only hashing in proof and transcript paths;
post-quantum; about 114 bits proven (Johnson bound); PoW of at most 20 bits on the last
layer only; 64 GB memory cap; files under 1,500 lines; tests in `tests/`, each at most 300 s
in `--release`; published techniques only; no new Rust features or environment variables;
folding protocol, b = 2, k_rho = 16 and the SHAKE128 key unchanged.

## Usage (caller's view)

### README

> A finished accumulator is compressed into one `CompressedProof` of about 90 KB. A
> verifier needs the expected `Stage1State` and the trusted compression key (1,912 bytes).
> It needs no setup files, no shrink key and no preprocessing.
>
> ```rust
> // Prover machine, once per circuit (M2, unchanged).
> let setup = prover.compression_setup(&setup_dir)?;
>
> // Per proof.
> let proof: CompressedProof = prover.compress(&envelope, &setup)?;
> std::fs::write(&path, proof.to_bytes())?;
>
> // Verifier (for example a chain node). The key is pinned or derived, never sent.
> let key = CompressionKey::from_bytes(PINNED_KEY)?;
> let proof = CompressedProof::from_bytes(&bytes)?; // strict, canonical bytes only
> verifier.verify_compressed(&expected_state, &key, &proof)?;
> ```

`FinalProof` becomes crate-internal. There is one external proof type and one way to
verify it.

### Call site 1: the production test (nightstream)

```rust
let lifecycle = prepared_production_lifecycle()?;
let setup = lifecycle.open_compression_setup(&dir, &key)?;
let started = Instant::now();
let proof = lifecycle.compress(&envelope, &setup)?;          // finish (M2) + shrink (M3)
let bytes = proof.to_bytes();
assert!(bytes.len() <= 200_000);                              // hard limit
eprintln!("compress {:?}, {} bytes", started.elapsed(), bytes.len());
lifecycle.verify_compressed(envelope.state(), &key, &CompressedProof::from_bytes(&bytes)?)?;
```

### Call site 2: inside nightstream (the terminal program)

```rust
/// The check that the shrink proof proves. The same program is the native
/// FinalProof verifier, the witness generator and the verifier's circuit shape.
pub(crate) struct FinalCheck<'a> {
    lifecycle: &'a PreparedLifecycle,
    key: &'a neo_spartan::Key,
    state: &'a Stage1State,
    proof: Option<&'a FinalProof>, // None: shape only (the shrink verifier)
}

// Native verification of a FinalProof (tests and the prover's self-check).
neo_spartan::shrink::check(&FinalCheck { proof: Some(&final_proof), ..check })?;

// Shrink prove and verify.
let shrink = self.shrink_for(key)?;          // cached per key: layout + WHIR config
let proof = shrink.prove(&FinalCheck { proof: Some(&final_proof), ..check })?;
shrink.verify(&FinalCheck { proof: None, ..check }, &proof)?;
```

### Call site 3: neo-spartan-level, the generic M2 verifier inside the program

```rust
impl neo_spartan::shrink::Program for FinalCheck<'_> {
    fn circuit_words(&self) -> Vec<u64> { /* context digest, PiCCS header, key, relation */ }
    fn public_words(&self) -> Vec<u64> { state_words(self.state) } // iteration, z0, current
    fn run<W: World>(&self, w: &mut W) -> Result<(), neo_spartan::Error> {
        let proof = FinalProofWires::alloc(w, self.statics(), self.proof);
        let digest = statement(w, self, &proof)?;                 // state hash, encHash
        let mut fold = FoldSponge::new(w);
        let outputs = pi_ccs::verify(w, &mut fold, self, &proof, digest)?;
        let parent = pi_rlc::verify(w, &mut fold, self, &outputs)?;
        neo_spartan::verify_in(w, &self.relation(), fold, &parent, &proof.layer1)
    }
}
```

## Shape

### Load-bearing decisions

1. **One generic program, three worlds** (single source of truth). Every verifier check is
   written once against a `World` trait. `Native` holds values and returns `Rejected` on a
   failed check: this is the native `FinalProof` verifier. `Circuit` in prover mode records
   rows and witness values: this is the witness generator. `Circuit` in shape mode records
   rows only, straight into an MLE accumulator: this is the shrink verifier's matrix
   evaluation. Generic code cannot read a value (the handle types have no accessor), so the
   circuit shape cannot depend on the witness. Per *encode-lessons-in-structure*.
2. **Uniform templates for repeated blocks.** Poseidon2 (616 rows, 632 columns) and the ring
   product mod Φ81 (161 rows, 269 columns) are templates placed in aligned power-of-two
   slot groups. The MLE of a template group is
   `eq(r_x[slot], r_y[slot]) · T̃(r_x[local], r_y[local]) · eq(r_x[high], row_offset) ·
   eq(r_y[high], column_offset)`. The verifier evaluates it in microseconds (the uniform
   treatment of Jolt, ePrint 2023/1217, and SuperSpartan, ePrint 2023/552). Only about
   1.1 M explicit rows (5.6 M nonzeros) need regeneration.
3. **Fixed shape.** Every data-dependent choice of the native verifier becomes a gadget:
   query indices (canonical 64-bit split), Merkle directions (mux), domain points
   (`ω^index` by bits). Every multiproof path is checked as a full path; the prover
   restores the paths from p3's `PrunedMerklePaths`.
4. **One shrink layer.** It is the only viable count without holography: a second layer
   would have to evaluate the first layer's 5.6 M explicit nonzeros in-circuit, which costs
   more than it saves.
5. **M2 unchanged.** The shrink circuit fits 2^25 rows at 66% fill with M2 as measured.
   Retuning M2 to reach 2^24 halves the shrink prover time but raises the overall peak and
   doubles the setup disk (lever C in "Costs").

### Data structures

**The constraint system.** One R1CS `(A, B, C)` with `2^25` rows and `2^26` columns,
`z = (w ‖ u)`. Only `w` (2^25 base cells) is committed. `u = (1, public words, 0, ...)`.

| Region | Rows | Columns | Placement |
| --- | --- | --- | --- |
| Poseidon2 slots | 32,768 × 640 | 32,768 × 640 | groups from the binary split of the slot count; each slot has a full-round part (512 rows, 512 columns) and a partial-round part (128 rows: 88 S-box rows, 16 output rows; 128 columns: 16 inputs, 88 S-box values, 16 outputs) |
| Ring-product slots | 640 × 256 | 640 × 272 | 107 evaluation rows `ρ(t)·v(t) = P_t` for t = 0..106, then 54 output rows (interpolation and reduction mod Φ81 are constants) |
| Explicit rows | about 1.11 M | about 1.1 M | sponge wiring, Merkle muxes, field arithmetic, bits, copies |

Slot group `g` has `2^p` slots and aligned offsets. Template row `i` of slot `b` is row
`offset + i·2^p + b`; local column `j` of slot `b` is column `offset + j·2^p + b`. Unused
slots hold the trace of a zero input, which every template row accepts.

**Linear combinations.** A base-field handle in the circuit world is an index into an arena
of linear combinations. Linear operations add no row. A combination becomes a column only
when that lowers the total nonzero count (`(terms − 1)·(uses − 1) > terms + 1`), a rule
derived from the cost, not a tuned constant.

**Deferred placement.** A prover word (`World::input`) gets no column at birth. Its first
use places it: as a Poseidon2 input lane it becomes that slot's input column (no copy row);
elsewhere it goes to the explicit column region. Most of the 200 k proof words land in
Poseidon2 input columns this way. Placement is in program order, so prover and verifier
agree.

**Gadget costs** (rows): base product 1; Ext × Ext 5 (Toom-3 at 0, 1, −1, 2, ∞); Ext ×
base 3; Ext inverse 8; K × K 3; K⊗Ext × K⊗Ext 15; bit 1; canonical 64-bit split 67;
select 1 per word; Merkle level 4 mux rows plus one compression slot.

### How one program serves three uses

| Use | World | Program input | Output |
| --- | --- | --- | --- |
| `verify_final` (tests, prover self-check) | `Native` | `FinalProof` values | `Ok` or `Rejected` |
| shrink prover | `Circuit` + `RowStore`, values on | `FinalProof` values | explicit rows, witness `w`; any failed check stops the prover |
| shrink verifier | `Circuit` + `MleSink`, values off | public words only | `Σ_nnz coeff·eq(r_x,row)·eq(r_y,col)` |
| layout derivation (once per key) | `Circuit` + `CountSink`, values off | none | slot counts, explicit row and column counts |

The layer-0 part of the program (state hash, PiDEC child checks, PiCCS, PiRLC) is a second
implementation next to `neo-reductions`, for the selected profile only. The fold path keeps
`neo-reductions`. The terminal program is pinned to it by tests: equal parent claim and
equal transcript state after layer 0 on honest proofs, and equal rejection on every mutation
class. The M2 part has one implementation: `neo_spartan::verify` becomes the generic
`verify_in`, and p3's `verify_at` stays only as a test oracle.

### Where validation lives

- Native, before the circuit: public words (iteration in `[1, p)`, canonical `z0`,
  `current`), strict decoding of `CompressedProof`.
- In the circuit: everything `verify_final` checks. Shape checks of the native verifier
  (lengths) are static in the circuit: a wrong-length proof cannot be a witness.
- Deliberate differences, all stricter or equal in the circuit:
  - Merkle: per-path checks against the root, not the multiproof walk (same acceptance
    under Poseidon2 collision resistance).
  - Uniform-bit sampling: p3 resamples a draw equal to `p − 1`; the circuit rejects it.
    This is a completeness gap of about `draws/p ≈ 2^-54`, never a soundness gap.
  - Native zero-skips (`count == 0`, `value == Kx::ZERO` in Eval_K) become computed zero
    terms; the values are equal.

### What the design does not do

No holography, no preprocessing, no shrink key, no zero knowledge (not required), no second
shrink layer, no change to the folding protocol, to M2's transcript or to M2's rates. No
in-circuit Fiat-Shamir step of its own: the PiRLC ring mix is computed with ring templates,
not checked at a new in-circuit challenge.

### Interface depth

- nightstream adds three public items: `compress`, `verify_compressed`, `CompressedProof`.
  They hide two layers, the M2 setup use, the layout derivation and the WHIR profile.
- neo-spartan adds `World` (17 primitives), `shrink::{Program, Shrink, Proof, check}`,
  `verify_in` and the wire allocators. `World` is the price of keeping the terminal program
  in nightstream, next to the state-preimage format that nightstream owns. The circuit
  world, the layout and every Plonky3 0.8 type stay private.

### Module map

| File | Owns | est. lines |
| --- | --- | --- |
| `neo-spartan/src/world/mod.rs` | `World` trait, handle types, `Error` contract | 220 |
| `neo-spartan/src/world/native.rs` | the value world | 160 |
| `neo-spartan/src/world/circuit.rs` | circuit world: LC arena, deferred placement, slots, sinks | 500 |
| `neo-spartan/src/world/algebra.rs` | Ext/K/K⊗Ext helpers, powers, Horner, eq, interpolation | 220 |
| `neo-spartan/src/world/sponge.rs` | v1.1 fold sponge, p3 duplex replica (incl. `fs` seed words), Merkle leaf/compress/path, bit sampling | 300 |
| `neo-spartan/src/whir/mod.rs` | generic replica of p3-whir 0.8 `verify_at` (adapter, replay, STIR, final checks) | 600 |
| `neo-spartan/src/whir/layout.rs` | replica of p3-sumcheck 0.8 stacked layout verifier (placements, claims, eq/select weights) | 350 |
| `neo-spartan/src/whir/paths.rs` | `PrunedMerklePaths` to full paths (port of p3's `restore_paths`), prover-side only | 160 |
| `neo-spartan/src/verify.rs` | generic M2 layer-1 verifier `verify_in`, proof wire allocation (moved out of `lib.rs`) | 450 |
| `neo-spartan/src/{gkr,sumcheck,setup,matrix,mle,ring,norm}.rs` | verifier halves become generic; prover halves unchanged | +350 net |
| `neo-spartan/src/shrink/mod.rs` | `Program`, `Shrink`, `Proof`, `check`, the shrink transcript | 250 |
| `neo-spartan/src/shrink/layout.rs` | slot groups, aligned regions, Poseidon2 and ring templates, closed-form MLEs, uniform `Az/Bz/Cz` | 420 |
| `neo-spartan/src/shrink/spartan.rs` | outer and inner sum-checks, prove and verify | 400 |
| `neo-spartan/src/shrink/rows.rs` | `RowStore`, `MleSink`, `CountSink`, split eq tables | 220 |
| `neo-spartan/src/pcs.rs` | add `Profile { log_inv_rate, pow_bits }` | +30 |
| `nightstream/src/lifecycle/final_check/mod.rs` | `FinalCheck` program, wire allocation, statement, state hash, encHash, PiDEC child checks, handoff | 320 |
| `nightstream/src/lifecycle/final_check/pi_ccs.rs` | PiCCS replica for the selected profile | 330 |
| `nightstream/src/lifecycle/final_check/pi_rlc.rs` | sampler, base-5 decode gadget, ring-product mix | 220 |
| `nightstream/src/lifecycle/{finish,encoding}.rs`, facade | `compress`, `verify_compressed`, `CompressedProof` bytes, budget split | +200 |

About 5,100 new lines and about 1,800 lines of tests. `lib.rs` shrinks from 1,014 to about
700 lines. Every file stays under 1,500 lines. Call chains: nightstream `verify_compressed`
→ `Shrink::verify` → `Spartan` + `MleSink` (program run). Three files.

### Type sketch

```rust
// ===== neo-spartan/src/world/mod.rs =====
//! The arithmetic that every compression check is written in. One check runs
//! as plain values (`Native`), as a circuit with witness values (shrink
//! prover) and as a circuit without values (shrink verifier). Generic code
//! cannot read a value, so a circuit's shape never depends on the witness.

use neo_math::D;

/// Primitive operations of a verifier world. Linear operations are free in a
/// circuit; each other primitive states its row cost.
pub trait World {
    /// A base-field value or wire.
    type Gl: Copy;
    /// A cubic-extension value or wire (three base coordinates).
    type Ext: Copy;
    /// A boolean value or wire.
    type Bit: Copy;

    /// A prover word. `None` only in the shape world.
    fn input(&mut self, value: Option<u64>) -> Self::Gl;
    /// Public word `index` of the statement.
    fn public(&mut self, index: usize) -> Self::Gl;
    fn constant(&mut self, value: u64) -> Self::Gl;

    fn add(&mut self, a: Self::Gl, b: Self::Gl) -> Self::Gl;
    fn sub(&mut self, a: Self::Gl, b: Self::Gl) -> Self::Gl;
    fn scale(&mut self, a: Self::Gl, factor: u64) -> Self::Gl;
    /// One row.
    fn mul(&mut self, a: Self::Gl, b: Self::Gl) -> Self::Gl;

    fn ext(&mut self, coordinates: [Self::Gl; 3]) -> Self::Ext;
    fn coordinates(&mut self, value: Self::Ext) -> [Self::Gl; 3];
    /// Five rows.
    fn ext_mul(&mut self, a: Self::Ext, b: Self::Ext) -> Self::Ext;
    /// Eight rows; rejects zero.
    fn ext_inverse(&mut self, a: Self::Ext) -> Result<Self::Ext, Error>;

    fn assert_zero(&mut self, a: Self::Gl, check: &'static str) -> Result<(), Error>;
    /// The low `count` bits of the canonical integer of `a`. The other bits
    /// are range-checked and `a < p` is enforced, so the split is unique.
    fn low_bits(&mut self, a: Self::Gl, count: usize) -> Result<Vec<Self::Bit>, Error>;
    fn bit(&mut self, bit: Self::Bit) -> Self::Gl;
    /// One row per word.
    fn select(&mut self, bit: Self::Bit, one: Self::Gl, zero: Self::Gl) -> Self::Gl;

    /// The workspace Poseidon2 permutation: one Poseidon2 slot in a circuit.
    fn permute(&mut self, state: [Self::Gl; 16]) -> [Self::Gl; 16];
    /// `a · b mod Φ81`: one ring-product slot in a circuit.
    fn ring_mul(&mut self, a: &[Self::Gl; D], b: &[Self::Gl; D]) -> [Self::Gl; D];
}

pub use crate::Error; // `Error::Rejected(&'static str)` for a failed check

// ===== neo-spartan/src/world/circuit.rs =====
/// A circuit under construction. `S` decides what happens to explicit rows.
pub(crate) struct Circuit<'l, S: Sink> {
    layout: &'l Layout,
    values: Option<Vec<u64>>, // witness by column; None in shape and count modes
    lcs: LcArena,
    slots: SlotCursor,        // next Poseidon2 and ring slot, by template
    columns: u64,             // next explicit column
    rows: u64,                // next explicit row
    sink: S,
}

pub(crate) trait Sink {
    fn row(&mut self, row: u64, a: &[(u64, u64)], b: &[(u64, u64)], c: &[(u64, u64)]);
}

impl<'l, S: Sink> Circuit<'l, S> {
    pub(crate) fn prover(layout: &'l Layout, sink: S) -> Self { unimplemented!() }
    pub(crate) fn shape(layout: &'l Layout, sink: S) -> Self { unimplemented!() }
    /// The witness `w`, unused slots filled with zero-input traces.
    pub(crate) fn finish(self) -> Result<(Vec<u64>, S), Error> { unimplemented!() }
}

// ===== neo-spartan/src/world/sponge.rs =====
/// `Poseidon2Transcript` v1.1 (add-absorb, pair reads, digest squeeze) over wires.
pub struct FoldSponge<W: World> { state: [W::Gl; 16] }
/// p3 0.8 `DuplexChallenger<Gl, Poseidon2, 16, 12>` over wires, including the
/// length tag at duplexing and the pop order of the output buffer.
pub(crate) struct Duplex<W: World> {
    state: [W::Gl; 16],
    input: Vec<W::Gl>,
    output: Vec<W::Gl>,
}

impl<W: World> FoldSponge<W> {
    pub fn new(w: &mut W) -> Self { unimplemented!() }
    pub fn absorb(&mut self, w: &mut W, words: &[W::Gl]) { unimplemented!() }
    pub fn read_pair(&self, pair: usize) -> [W::Gl; 2] { unimplemented!() }
    pub fn squeeze_digest(&mut self, w: &mut W) -> [W::Gl; 4] { unimplemented!() }
}

// ===== neo-spartan/src/verify.rs =====
/// The CE(B) claim of layer 1 as wires.
pub struct ClaimWires<W: World> {
    pub commitment: Vec<[W::Gl; D]>,
    pub public: Vec<[W::Gl; D]>,
    pub point: Vec<[W::Gl; 2]>,
    pub eval_k: [[W::Gl; 2]; D],
    pub eval_a: Vec<[[W::Gl; 2]; D]>,
}

/// A layer-1 proof as wires: the `Proof` fields, with full Merkle paths.
pub struct ProofWires<W: World> { /* mirrors `Proof`; private */ }

impl<W: World> ProofWires<W> {
    /// Allocate every proof word. `None` gives the shape without values.
    pub fn alloc(w: &mut W, relation: &Relation<'_>, proof: Option<&Proof>) -> Self { unimplemented!() }
}

/// The layer-1 verifier. With `Native` it is the M2 verifier; in a circuit it
/// records the same checks.
pub fn verify_in<W: World>(
    w: &mut W,
    relation: &Relation<'_>,
    transcript: FoldSponge<W>,
    claim: &ClaimWires<W>,
    proof: &ProofWires<W>,
) -> Result<(), Error> {
    unimplemented!()
}

// ===== neo-spartan/src/whir/mod.rs =====
/// Counts and constants of one p3 opening, read from p3's public `WhirShape`
/// and `WhirConfig` accessors, never recomputed.
pub(crate) struct WhirPlan { /* rounds, queries, OOD, folding, domains, seed words */ }

/// p3-whir 0.8 `verify_at` over wires. Returns the opened values per batch.
pub(crate) fn verify_at<W: World>(
    w: &mut W,
    plan: &WhirPlan,
    challenger: &mut Duplex<W>,
    root: [W::Gl; 4],
    opening: &OpeningWires<W>,
    points: &[Vec<W::Ext>],
) -> Result<Vec<Vec<W::Ext>>, Error> {
    unimplemented!()
}

// ===== neo-spartan/src/shrink/layout.rs =====
/// The shrink circuit's shape: template slot groups and explicit regions.
/// Derived by running the program once in count mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    log_rows: u32,                  // 25 in production
    poseidon2: Vec<SlotGroup>,      // binary split of the slot count
    ring: Vec<SlotGroup>,
    explicit_rows: Region,
    explicit_columns: Region,
    public: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotGroup {
    log_slots: u32,
    /// Aligned starts of the template's row parts and column parts.
    rows: [u64; 2],
    columns: [u64; 3],
}

/// A template: sparse `(row, column, coefficient)` per matrix, by part.
pub(crate) struct Template { parts: Vec<TemplatePart> }

impl Template {
    pub(crate) fn poseidon2() -> Self { unimplemented!() } // from `round_constants()`
    pub(crate) fn ring_product() -> Self { unimplemented!() }
    /// `(Ã, B̃, C̃)` of all slot groups of this template, in closed form.
    pub(crate) fn evaluate(&self, groups: &[SlotGroup], r_x: &[Ext], r_y: &[Ext]) -> [Ext; 3] {
        unimplemented!()
    }
}

// ===== neo-spartan/src/shrink/mod.rs =====
/// A fixed-shape check that a shrink proof proves.
pub trait Program {
    /// Words that name the circuit; bound before the first shrink challenge.
    fn circuit_words(&self) -> Vec<u64>;
    /// The statement the native verifier holds.
    fn public_words(&self) -> Vec<u64>;
    fn run<W: World>(&self, w: &mut W) -> Result<(), Error>;
}

/// The shrink argument of one program shape: layout and WHIR configuration.
/// Derived once per key; prover and verifier derive the same value.
pub struct Shrink {
    layout: Layout,
    pcs: Pcs,
}

/// Opaque shrink proof: one WHIR commitment, the Spartan rounds, the three
/// matrix-vector claims and one WHIR opening.
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    commitment: pcs::Commitment,
    outer: Vec<[Ext; 3]>,
    claims: [Ext; 3],
    inner: Vec<[Ext; 2]>,
    opening: pcs::Opening,
}

/// Run `program` on plain values.
pub fn check(program: &impl Program) -> Result<(), Error> { unimplemented!() }

impl Shrink {
    /// `security_bits`: −log2 of the error this layer may add.
    pub fn new(program: &impl Program, security_bits: f64) -> Result<Self, Error> { unimplemented!() }
    pub fn prove(&self, program: &impl Program) -> Result<Proof, Error> { unimplemented!() }
    pub fn verify(&self, program: &impl Program, proof: &Proof) -> Result<(), Error> { unimplemented!() }
    pub fn security_bits(&self) -> f64 { unimplemented!() }
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    /// Strict: the bytes must be the canonical encoding.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
}

// ===== nightstream/src/lifecycle/finish.rs (additions) =====
/// The external compressed proof. The caller holds the state.
#[derive(Clone)]
pub struct CompressedProof { shrink: neo_spartan::shrink::Proof }

impl PreparedLifecycle {
    pub(crate) fn compress(
        &self,
        envelope: &Stage1Envelope,
        setup: &neo_spartan::Setup,
    ) -> Result<CompressedProof, FinishError> {
        unimplemented!() // finish_with_spartan, then shrink.prove(FinalCheck)
    }

    pub(crate) fn verify_compressed(
        &self,
        expected_state: &Stage1State,
        key: &neo_spartan::Key,
        proof: &CompressedProof,
    ) -> Result<(), VerifyError> {
        unimplemented!() // public-word checks, then shrink.verify(FinalCheck, proof: None)
    }

    /// One security budget split: M2 and the shrink layer each get s + 1 bits,
    /// where s = `compression_security_bits()`.
    fn compression_budget(&self) -> Result<[f64; 2], neo_spartan::Error> { unimplemented!() }

    /// The shrink argument of `key`: derived once (count pass and WHIR
    /// configuration) and cached, so prover and verifier use the same value.
    fn shrink_for(&self, key: &neo_spartan::Key) -> Result<&neo_spartan::shrink::Shrink, neo_spartan::Error> {
        unimplemented!()
    }
}
```

## Synthesis decision

To be filled in by the arena.

## Protocol and transcript order

### A. The terminal program `FinalCheck::run` (in the circuit and natively)

Public words: `iteration`, `z0[4]`, `current[4]`. Constants: the verifier-context digest,
the PiCCS header (t = 4, f, n, m, 28 rounds, degree 8), the M2 key, the M2 relation (both
WHIR plans, the setup query count).

1. **Allocate** the `FinalProof` words: 16 running claims, the fresh claim, the PiCCS rounds
   and output evaluations, the layer-1 proof with full Merkle paths. `pi_rlc.combined` is
   not an input: the program computes it.
2. **Statement.** Canonical children: each parent public coordinate as sign and 16 bits,
   child digits `bit · (1 − 2·sign)`. Child public digits in {−1, 0, 1}. The 27,819-word
   state preimage (linear packing) → Poseidon2 sponge hash (2,320 permutations). The fresh
   public input equals `encHash(digest)`: marker 1, then the canonical 64-bit split of each
   digest word.
3. **PiDEC parent authority.** Parent commitment, public input and evaluations as
   `Σ 2^j child_j` (linear, no rows).
4. **PiCCS** on a `FoldSponge`: reset; absorb the fold domain chunk; absorb the prior digest,
   the fresh commitment and the fresh public input (1,462 words); read 28 α and γ (a zero
   chunk after every sixth pair); 28 rounds (absorb 18 words, read r, check
   `p(0) + p(1) = claim`, Horner at r over K); initial and terminal claims (γ powers, f at
   the matrix evaluations, range product for b = 2, two equality polynomials); output public
   parts equal the inputs; absorb the outputs (9,180 words); output digest equals the
   state prefix.
5. **PiRLC.** For i = 0..17: absorb `[4, i]`, squeeze a digest, decode the integer
   `Σ d_k p^k` into 54 base-5 digits (limb arithmetic with carries, digits in {0..4}), then
   `ρ_i = digits − 2`. The parent claim is `Σ_i ρ_i ⋆ source_i` over 37 base ring elements
   per source: 629 ring-product slots.
6. **Handoff.** Absorb `Nightstream/SuperNeo/compress/v1`, squeeze 4 words, seed a `Duplex`.
7. **M2 layer 1** (`verify_in`, the order of today's `neo_spartan::verify`): shape, key,
   profile and statement words; P0 root; histogram; β; early challenges; early GKR (5 trees);
   early values; λ; quotient; ζ; linear sum-check; finals; μ; partials; α; P1 root; 125 setup
   indices; setup leaves (path, honest-fold weights); β; late GKR; late values; P1 opening
   (replica of `verify_at`); P0 opening (replica of `verify_at`); final equalities.

### B. The shrink argument (native only, last layer)

The challenger is a p3 0.8 `DuplexChallenger` over the workspace Poseidon2 (width 16,
rate 12), as in M2.

1. Observe the domain chunk `Nightstream/SuperNeo/shrink/v1`, the circuit words
   (verifier-context digest, PiCCS header, M2 key words, M2 relation words, layout words,
   shrink WHIR profile words) and the 9 public words.
2. The prover commits `w` (2^25 base cells, one table) with WHIR: rate 1/64, folding 4,
   Johnson bound, PoW 20 at every query site, target s + 2. The root is observed.
3. Draw τ ∈ Ext^25.
4. Outer sum-check of `Σ_x eq(τ, x)·(Ãz(x)·B̃z(x) − C̃z(x)) = 0`: 25 rounds, degree 3,
   messages `h(0), h(2), h(3)`, low bit first.
5. The prover sends `v_A, v_B, v_C`; observe. Check `eq(τ, r_x)·(v_A·v_B − v_C)` against
   the last outer claim.
6. Draw ρ. Inner sum-check of `v_A + ρ v_B + ρ² v_C = Σ_y M̃(r_x, y)·z̃(y)` with
   `M = A + ρB + ρ²C`: 26 rounds, degree 2, messages `h(0), h(2)`.
7. WHIR opening of `w` at `r_y[0..25]` (p3 `open_at`; the claimed value is bound inside).
8. The verifier computes `ũ(r_y[0..25])` from 10 words and
   `z̃(r_y) = (1 − r_y[25])·w̃ + r_y[25]·ũ`.
9. The verifier computes `M̃(r_x, r_y)`: the closed form of each template group (both
   templates) plus the explicit rows, which it rebuilds by running `FinalCheck` in shape
   mode into an `MleSink` (split eq tables of 2^13 and 2^12 entries; rows are streamed in
   chunks to worker threads). It checks the last inner claim equals `M̃(r_x, r_y)·z̃(r_y)`.

## Soundness map

Let `s = compression_security_bits()` at the caller's minimum. At 114 bits with the fold
error 2^-114.4 (the existing computed census), `s ≈ 116.05`. Today M2 uses all of `2^-s`. This design
splits it: M2 and the shrink layer each get `2^-(s+1)`.

| Term | Bound | Notes |
| --- | --- | --- |
| Fold (PiCCS γ-mixing, sampler bias) | 2^-114.4 | existing census, unchanged |
| M2 layer 1 | ≤ 2^-(s+1) ≈ 2^-117.05 | `Relation::new(…, s + 1)`; each commitment targets s + 2 (one bit above today; about +1% queries) |
| Shrink WHIR (2^25, rate 1/64, folding 4, JB, PoW 20) | ≤ 2^-(s+2) ≈ 2^-118.05 | p3 report: OOD, folding, query and combination terms |
| Spartan outer: zero check and 25 degree-3 rounds | (25 + 75)/2^192 · L ≈ 2^-176 | charged over L ≈ 2^9.3 candidates (JB list size at rate 1/64), as in M2 |
| Spartan inner: ρ batching and 26 degree-2 rounds | (2 + 52)/2^192 · L ≈ 2^-177 | idem |
| Shrink total | ≤ 2^-(s+1) | the two Spartan terms fit in the 1-bit gap between s + 2 and s + 1 |
| **All** | ≤ 2^-114.4 + 2^-(s+1) + 2^-(s+1) = 2^-114 | by the definition of s |

Notes.

- The circuit is at least as strict as `verify_final` (see "Where validation lives"), so a
  satisfying witness is a `FinalProof` that `verify_final` accepts. The shrink argument is
  an argument of knowledge for that relation; its error adds to M2's.
- PoW of 20 bits sits only on the shrink WHIR query sites. M2 keeps PoW 0.
- Completeness error: about 900 index draws, each rejected in-circuit with probability 1/p
  where p3 would resample: about 2^-54.
- Computational assumptions, as in M2: Poseidon2 collision resistance for every Merkle
  tree, and Poseidon2 as a random oracle for Fiat-Shamir. They are not part of the
  statistical budget.
- No new in-circuit Fiat-Shamir step: every in-circuit challenge is one that M2 or layer 0
  already draws.

## Costs

### Estimator

`whir(n, rate)` rounds: fold 4 until at most 6 variables remain; queries per round
`⌈level / (0.5·log_inv_rate − log2(21/20))⌉` (p3 Johnson bound) at level 120; rates grow by
3 bits per round; round 0 leaves hold 16 base values (2 permutations), later leaves 16 Ext
values (4 permutations); path depth `n + log_inv_rate − round − 4`. Per query, non-hash
rows ≈ 67 (index split) + 2·path (domain point) + 59 or 75 (fold) + 5·variables (weight)
+ 5·path (mux and weight). Layer-0 counts are exact from the code (state preimage 27,819
words; PiCCS 1 + 122 + 4 + 56 + 765 permutations; PiRLC 2 per ρ).

### Constraint census (M2 unchanged)

| Component | Poseidon2 permutations | Poseidon2 rows (× 616) | Other rows |
| --- | --- | --- | --- |
| Terminal state hash | 2,320 | 1,429,120 | about 0 (linear packing) |
| Statement bits (encHash, canonical split, child digits) | 0 | 0 | 19,000 |
| PiCCS (transcript; K arithmetic, γ powers, terminal) | 948 | 583,968 | 50,000 |
| PiRLC (sampler; base-5 decode; ring mix) | 34 | 20,944 | 19,000 + 101,269 template + 67,932 copies |
| M2 fixed part (handoff, statement, histogram, both GKRs' transcript, norm logUp, ring rows, linear sum-check, Eval_K DP) | 1,326 | 816,816 | 191,000 |
| M2 setup leaves (125 × (18 + 24)) | 5,250 | 3,234,000 | 125,000 |
| M2 P1 opening and late GKR (queries 280, 63, 35, 25, 19) | 11,018 | 6,787,088 | 199,000 |
| M2 P0 opening (queries 280, 63, 35, 25, 19, 16) | 11,852 | 7,300,832 | 178,000 |
| Sponge and Merkle wiring (capacity copies, zero and constant lanes) | | | 262,000 |
| **Total** | **32,748** | **20,172,768** | **about 1.21 M** |

Layer-1 permutations in the circuit: 29,446 (native 23,589; the rest is the multiproof
sharing that a fixed-shape circuit cannot use).

By gadget (approximate):

| Gadget | Rows each | Count | Rows |
| --- | --- | --- | --- |
| Poseidon2 permutation (template) | 616 (+24 padding) | 32,748 | 20.17 M (20.97 M with padding) |
| Ring product mod Φ81 (template) | 161 (+95 padding) | 629 | 0.10 M (0.16 M) |
| Ext × Ext | 5 | about 70 k | 0.35 M |
| Ext × base | 3 | about 60 k | 0.18 M |
| Ext inverse (mostly the 7,345-entry histogram) | 8 | about 7.4 k | 0.06 M |
| K × K and K⊗Ext (PiCCS, Eval_K DP) | 3 / 15 | | 0.12 M |
| Index draws (canonical split) and domain points `ω^index` | about 67 + 50 | about 1,000 | 0.12 M |
| Merkle level mux | 4 | about 22.8 k levels | 0.09 M |
| Sponge wiring | about 8 per slot | 32.7 k | 0.26 M |
| Other bits (encHash, split, ρ decode) | | | 0.03 M |

**Sizes.** Rows used: 20.97 M (Poseidon2 slots) + 0.16 M (ring slots) + 1.11 M (explicit) =
22.25 M of 2^25 = 33.55 M (66%). Committed witness `w`: 2^25 base cells, about 22.2 M used.
Explicit nonzeros: about 5.6 M. The circuit fits 2^25 up to about 50 k permutations; the
slot groups follow the real count automatically.

### Proof bytes

Measured WHIR table (one column, 114 bits, folding 4), extrapolated by its own slopes and
scaled by 1.03 for the 118-bit target:

| Configuration | WHIR | Spartan messages | **Total** | vs 100 KB goal / 200 KB limit |
| --- | --- | --- | --- | --- |
| **A (chosen):** M2 unchanged, 2^25, rate 1/64, PoW 20 | about 86.2 KB | 3.2 KB | **about 89 KB** | meets the goal |
| B (200 KB fallback): 2^25, rate 1/16, PoW 20 | about 122 KB | 3.2 KB | about 125 KB | meets the limit only |
| C (lever): M2 retuned, 2^24, rate 1/64, PoW 20 | about 83.2 KB | 3.1 KB | about 86 KB | meets the goal |
| D: C at rate 1/16 | about 112 KB | 3.1 KB | about 115 KB | meets the limit only |

Spartan messages: 25 × 3 + 26 × 2 + 4 Ext values of 24 bytes, plus the root.

### Prover time and memory (shrink stage, est.)

| Step | A | B | C |
| --- | --- | --- | --- |
| Layout derivation (count pass, once per key) | 0.1 s | 0.1 s | 0.1 s |
| Witness and explicit rows (prover world, Merkle path restore) | 2-3 s | 2-3 s | 2 s |
| Uniform `Az, Bz, Cz` (about 330 M template MACs) and sum-checks | 3-5 s | 3-5 s | 2-3 s |
| WHIR commit and open (table slope: ×1.97 per variable) | about 65 s | about 17 s | about 33 s |
| **Shrink total** | **about 75 s** | **about 25 s** | **about 40 s** |
| Shrink peak (codeword 2^31 cells = 16 GB, tree 8 GB, Spartan 4 GB in A) | **about 30 GB** | about 10 GB | about 16 GB |

End-to-end production test for A: compile about 15 s + finish 40 s (measured) + shrink 75 s +
verify 0.3 s ≈ 130 s, under the 300 s cap. The shrink stage starts after the M2 prover data
is dropped (it needs only the 1.6 MB `FinalProof`), so the process peak stays at the M2
finish peak of 37.3 GB (measured).

Lever C retunes M2: P0 and P1 at rate 1/4 (round-0 queries 280 → 130) and the setup
codeword at rate 1/4 (honest-fold queries 125 → 63, 2 bits per query). Its shrink circuit is
15.1 M rows (90% of 2^24). Its cost lands on M2: finish peak 37 → about 42 GB, setup disk
27 → about 54 GB, setup build 63 → about 125 s. It raises the overall peak, so A does not
use it.

### Native verifier (est.)

| Step | A | C |
| --- | --- | --- |
| Public-word checks, transcript, Spartan round checks | < 1 ms | < 1 ms |
| WHIR verify of one 2^25 opening | 3-8 ms | 3-8 ms |
| Template closed forms (about 24 k template nonzeros) | < 1 ms | < 1 ms |
| Shape regeneration and accumulation (5.6 M / 4.1 M explicit nonzeros, 25-60 ns each) | 150-350 ms one core; 80-150 ms on 4 cores | 110-250 ms |
| **Total** | **about 250 ms one core (160-360 ms)** | about 180 ms |

Inside the 100-400 ms budget, with the risk at the top end (see "Open questions").

**Key.** No shrink key. The verifier needs the M2 key (1,912 bytes, measured), the PiCCS
header and the verifier-context digest (under 1 KB): about 3 KB. The layout is derived once
per key (count pass, about 100 ms) and cached.

### What the 200 KB fallback saves

Configuration B instead of A: about 50 s less shrink prover time (75 → 25 s) and about 20 GB
less shrink peak memory (30 → 10 GB), for about 36 KB more proof (89 → 125 KB). It needs no
other change. If the owner accepts B, the M2 retune (C) has no reason left.

### Holography against the new limits

A holographic shrink (SPARK-style commitments to the explicit rows: row, column, value and
memory-checking columns, about 2^26 cells) adds a second WHIR opening of about 85-120 KB.
Total about 175-210 KB: it misses the 100 KB goal and is at the 200 KB limit. It would cut
the verifier to about 15 ms. The 100-400 ms budget does not need it.

## Tradeoffs accepted

- We accept about 4× the committed witness of a degree-7 CCS (616 rows per permutation in
  R1CS) in exchange for the simplest constraint system and a degree-3 outer sum-check. The
  cost lands on the shrink prover (about 65 s of WHIR), not on proof size (2^25 vs 2^23 is
  about 6 KB).
- We accept a native verifier of about 250 ms that rebuilds 1.1 M explicit rows in exchange
  for no holography, no preprocessing and a 3 KB key.
- We accept per-path Merkle checks in the circuit (+22% permutations over the native
  multiproof) in exchange for a fixed circuit shape.
- We accept a second implementation of the selected PiCCS, PiRLC and state-hash checks (the
  terminal program) next to the fold-path verifier in `neo-reductions`, pinned by tests, in
  exchange for one program that is the native verifier, the witness generator and the
  verifier's shape.
- We accept that our replica, not p3's `verify_at`, becomes the native M2 verifier. p3's
  `verify_at` stays as a test oracle. Two p3 internals (`plan_layout`, `restore_paths`) are
  ported because they are `pub(crate)`.
- We accept a completeness error of about 2^-54 (the in-circuit rejection of the sample
  `p − 1`).
- We accept one bit less budget for M2 (each commitment target s + 1 → s + 2) to pay for the
  shrink layer.
- We accept a public `World` trait in neo-spartan so the terminal program can live in
  nightstream, next to the state format that nightstream owns.
- We accept a 2^25 shrink with M2 unchanged over a 2^24 shrink with a retuned M2, because
  the retune raises the overall peak memory and doubles the setup disk.
- We accept 629 ring-product template slots (164 k rows) over an in-circuit Fiat-Shamir
  evaluation check (about 125 k rows), because the template adds no new protocol step.

## Alternatives considered

1. **Degree-7 CCS + SuperSpartan (ePrint 2023/552) for the Poseidon2 template.** The same
   architecture with 150 rows and columns per permutation: a 2^23 circuit, about 80 KB,
   shrink prover about 4× faster (about 20 s), about 8 GB. It loses only because this seat
   fixes R1CS. It is the graft this design takes most easily: only the template, the outer
   sum-check degree (3 → 8) and the row cost table change. The `World` program, the uniform
   closed forms and the WHIR replica stay.
2. **Holographic Spartan (SPARK computation commitments).** Verifier about 15 ms, but a
   second opening: about 175-210 KB. Misses the 100 KB goal; needs a preprocessing setup and
   its key.
3. **A shape cache.** Store the compiled explicit rows per key (about 90 MB) and only
   accumulate at verify time: about 50 ms. Rejected because regeneration fits the budget
   with a 3 KB key. It is the first fallback if the verifier is too slow.
4. **Two shrink layers.** Without holography the outer circuit must evaluate the inner
   circuit's 5.6 M explicit nonzeros, which is larger than the inner verifier it replaces.
   One layer is the only viable count in this seat.
5. **Tracing the native code** (a field type that records a tape) instead of a generic
   `World`. Rejected: bit tests, early returns and zero-skips would make the tape follow one
   value path and silently fork the circuit from the native check.
6. **In-circuit Fiat-Shamir for the PiRLC mix** (one evaluation of the ring identity at a
   challenge from an in-circuit sponge). About 125 k rows, but a new Fiat-Shamir step that
   needs approval and a new soundness term. The ring template costs 164 k rows and adds
   nothing new.
7. **Retuning M2 first** (lever C). Kept as an option with its costs stated; not the default.

## Open questions and risks

1. Is about 250 ms on one core acceptable, given the range of 160-360 ms? The constant per
   rebuilt nonzero is not measured. If it is too slow, do you prefer the shape cache (about
   90 MB per key, about 50 ms) or uniform templates for WHIR query blocks (fewer explicit
   rows, more code)?
2. Do you accept a second implementation of the PiCCS/PiRLC/state-hash checks for the
   terminal statement, pinned to `neo-reductions` by tests? The other option is to make
   `neo-reductions`' PiCCS verifier generic over `World`, which touches Lean-aligned code.
3. Do you accept the 1-bit budget split (M2 and the shrink layer each at s + 1)?
4. p3's `fs` domain separator derives its seed from a Keccak-256 hash of the transcript
   pattern string. The value is a constant of the configuration, not of prover data, and M2
   already uses it. Do you accept it in the shrink transcript too, or must the shrink seed
   its challenger only with Poseidon2-absorbed words?
5. Do you accept that `FinalProof` becomes internal, so `compress`/`verify_compressed` is the
   only external path?
6. Configuration A (89 KB, 75 s, 30 GB) or B (125 KB, 25 s, 10 GB)?
7. Risk: the replica of p3-whir 0.8 must follow p3 bit for bit. A mismatch is a completeness
   bug (honest proofs fail), or a soundness bug if the replica checks less. Mitigation:
   differential tests against `verify_at` on honest and mutated openings, and the version
   pin `=0.8.0` that already exists.
8. Risk: the permutation count is an estimate (±15%). The circuit stays at 2^25 up to about
   50 k permutations; the slot groups follow the real count.
9. Risk: WHIR at 2^25 and rate 1/64 is extrapolated from 2^20 and 2^22 points (bytes and
   time). Slice 0 measures it.

## Next implementation step

Slice 0: measure one WHIR opening at 2^25 (rate 1/64 and 1/16, PoW 20, 118 bits: bytes,
time, peak RSS) and count the per-path in-circuit permutations of the production M2
verifier, before any `World` code.

## Implementation slices

Every test runs in `--release` with a timeout of at most 300 s. Toy sizes keep unit tests
under 60 s.

0. **Spike (measurements only).**
   - Ignored test: one-column WHIR at 2^25, one point, rate 1/64, PoW 20, 118-bit target:
     bytes, prove time, peak RSS (est. 70 s).
   - Same at rate 1/16 (est. 20 s).
   - Ignored test: on the production `FinalProof`, count permutations per M2 phase with
     multiproof sharing removed (from `WhirShape` query counts and path depths). Exit:
     fewer than 50 k permutations; numbers recorded.
1. **World, Native, algebra, sponges** (`world/{mod,native,algebra,sponge}.rs`).
   Tests (`tests/internal/world.rs`): `FoldSponge` equals `Poseidon2Transcript` on random
   absorb/read/squeeze sequences; `Duplex` equals p3 `DuplexChallenger` on random
   observe/sample/`sample_bits`/`sample_uniform_bits` sequences, with a test that pins the
   `p − 1` divergence; Merkle leaf, compression and path equal `MerkleTreeMmcs`; Ext, K and
   K⊗Ext operations equal `field.rs` and `neo_math`.
2. **Circuit world, layout, templates, sinks** (`world/circuit.rs`,
   `shrink/{layout,rows}.rs`). Tests: the Poseidon2 template trace satisfies the template
   and equals p3 `permute`; the ring template equals `neo_math` ring multiplication; random
   generic programs accept natively exactly when the prover-world circuit is satisfiable;
   shape-mode rows equal prover-mode rows (digest); `MleSink` equals a brute-force MLE at
   2^12; template closed forms equal brute force for toy slot groups.
3. **Spartan and the PCS profile** (`shrink/{mod,spartan}.rs`, `pcs.rs`). Tests: toy
   programs (a few hundred permutations) prove and verify; mutating each proof field, a
   public word or a circuit word rejects; `Shrink::new` is deterministic; the security
   report meets the target.
4. **WHIR replica** (`whir/{mod,layout,paths}.rs`). Tests: toy `PrescribedPointPcs`
   openings with M2-like plans (several tables, widths 1, 2 and 9, several points); the
   native replica accepts exactly when p3 `verify_at` accepts, on honest and on each
   mutated field; equal challenger state after verify; restored paths verify exactly when
   p3's multiproof verifies; the prover-world circuit is satisfiable on honest openings.
5. **Generic M2 verifier** (`verify.rs` and the verifier halves of `gkr`, `sumcheck`,
   `setup`, `matrix`, `mle`, `ring`, `norm`). Tests: every existing neo-spartan test passes
   through the native world; toy relation: the circuit is satisfiable on an honest proof and
   unsatisfiable on each mutation class.
6. **Terminal program** (`nightstream/src/lifecycle/final_check/`), with `verify_final`
   switched to it. Tests: toy package `FinalProof`: the native program accepts; its parent
   claim and post-layer-0 transcript state equal `nifs::verify_parent`'s; each mutation
   class rejects in both; the prover-world circuit is satisfiable; the number of allocated
   proof words equals the encoded `FinalProof` word count.
7. **API and production.** `compress`, `verify_compressed`, `CompressedProof` bytes
   (strict). Tests: toy end-to-end; tampered bytes, a wrong state and a wrong key reject.
   Ignored production test: compile + finish + shrink + verify with times, proof bytes
   (assert at most 200 KB, report against 100 KB), peak RSS (est. 130 s). Ignored verifier
   timing test: ten verifications, median printed.
