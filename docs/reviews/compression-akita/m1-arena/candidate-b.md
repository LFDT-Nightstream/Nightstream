# Candidate B: commit the digit planes, get the norm by construction

Milestone 1 of Nightstream compression: layer 0 (Pi_CCS + Pi_RLC, no Pi_DEC) and
layer 1 v1 (sum-check + WHIR argument of knowledge for the CE(B, L) parent).
Numbers marked *est.* are estimates, not measurements. Base: e402a6d99.

**Stance verdict.** The stance holds. Committing the 16 signed planes
D_0..D_15 (z = Σ 2^i D_i, D_i ∈ {−1,0,1}) gives the exact CE(B) bound
(|z| ≤ 2^16 − 1 < B = 2^16) with one cubic zero-check. It reuses the split that
every backend already computes before Pi_DEC, and it keeps the layer-1 verifier
to one sum-check and one WHIR opening, which is what milestones 2 and 3 need.
The price is prover-side and one-time: the committed table has 2^30 cells instead
of 2^26, so the prover pays about 16x on the WHIR commitment (*est.* 20–30 s CPU,
~30 GB peak RSS). Section 4.7 compares the packing options; section 9 asks the
owner whether that memory is acceptable.

## 1. Problem

At the end of folding, the accumulator holds 16 running CE(b) claims plus one
fresh CCS claim, with 17 witnesses (~221.7 MB). We must replace the witnesses with
a post-quantum argument (~114 bits) while keeping every check the terminal
verifier does today. What makes the shape non-obvious:

- **Without Pi_DEC, nothing enforces ||z||_∞ < B.** The RLC parent has |z| ≤ 3,672
  in honest runs, but CE(B) (Definition 21, conjunct 3) must be proven. Lemma 1
  needs a CE(B) witness to be extracted. The (2B, C)-relaxed binding in Definition 22
  is stated for that bound.
- **The claims are ring-valued.** C = A·z holds in R = F[X]/Φ81 (22 rows × 54
  coefficients). Eval_K and Eval_A_j are elements of R_K (54 coefficients in K), and
  each coefficient mixes lanes through `bar`. Φ81 has no root in K or F_{p^5}
  (ord_81(p) = 27), so "evaluate the ring at a point" needs extra structure.
- **Fields.** r ∈ K^28 comes from Pi_CCS. K = F_{p^2} is too small for WHIR at 2^26+
  (proven Johnson-bound terms ≈ 2^-50..2^-80). p3-goldilocks 0.8 provides only
  degree-2 and degree-5 extensions, and F_{p^5} does not contain K.
- **Layout.** The witness has 54 lanes × 814,144 blocks. 54 is not a power of two,
  and the tail lanes of the last block are nonzero after RLC, so we must cover the
  full carrier W = 43,963,776.
- **Two Plonky3 versions.** The workspace pins p3 0.5.3. WHIR requires 0.8.0
  (owner-approved). Our Poseidon2 constants and pinned fixtures depend on 0.5.3.
- **Later milestones.** The v1 verifier may read the key (967M coefficients) and the
  matrices (~85M nonzeros), but the shape must let milestone 2 replace those reads
  with setup-commitment openings, and let milestone 3 run the verifier in a circuit.

## 2. Usage (caller's view)

README quickstart (crates/nightstream/README.md, new section "Final proof"):

```rust
use nightstream::{Circuit, CompressedProof, Engine, Verifier};

let circuit = Circuit::load("poseidon2.pkg")?;
let prover = circuit.prover(Engine::Metal, 114)?;
let mut proof = prover.prove(initial_state, &inputs[0])?;
for step in &inputs[1..] {
    proof = prover.extend(&proof, step)?;          // unchanged folding lifecycle
}
// One more PiCCS + PiRLC fold, then a sum-check/WHIR argument. No witness inside.
let final_proof: CompressedProof = prover.finish_with_spartan(&proof)?;
std::fs::write("final.proof", final_proof.to_bytes())?;
```

Call site 1: an independent verifier service.

```rust
let verifier = Verifier::from_package(&Circuit::load("poseidon2.pkg")?, Engine::Optimized, 114)?;
let final_proof = verifier.decode_proof(&std::fs::read("final.proof")?)?; // length-bounded by the circuit
verifier.verify(&expected_state, &final_proof)?;   // the only public verify
```

Call site 2: a long-running prover that persists its accumulator between sessions.

```rust
std::fs::write("acc.bin", prover.encode_proof(&proof)?)?;           // unchanged
let proof = prover.decode_proof(&std::fs::read("acc.bin")?)?;        // moved from Verifier
let proof = prover.extend(&proof, &next_inputs)?;
```

Call site 3: a crate-internal test of layer 1 on a toy CE(B) claim.

```rust
let shape = neo_spartan::Shape::new(rows.shape(), D, point_len, 16)?;
let profile = neo_spartan::Profile::new(shape, 114.0)?;
let relation = neo_spartan::Relation::new(&rows, workspace_bytes);
let instance = neo_spartan::Instance::from_claim(&parent.claim, profile.shape())?;
let witness = neo_spartan::DigitPlanes::from_planes(parent.digits, profile.shape())?;
let proof = neo_spartan::prove(prover_tr.into_inner(), &profile, &relation, &instance, &witness)?;
neo_spartan::verify(verifier_tr.into_inner(), &profile, &relation, &instance, &proof)?;
```

The client never sees layers, planes, WHIR, or extension fields. `Verifier::verify`
takes only `CompressedProof`. The witness-carrying terminal check becomes the
crate-private test oracle `verify_accumulator`.

## 3. Shape

### 3.1 Data structures (dominant access patterns traced)

| Structure | Owner | Size (production) | Access pattern |
|---|---|---|---|
| `DigitPlanes`: `[plane][block] → (pos: u64, neg: u64)` lane masks | neo-spartan `planes.rs` | 16 × 814,144 × 16 B = 208 MB | Built from the existing compact signed-unit `Mat<F>` (`signed_unit_masks`). Read block-major for the WHIR table and lane-pair-wise for the first sum-check rounds. |
| WHIR table: 2^26 rows × 16 columns over F (row = 64·block + lane, column = plane) | p3 (`WhirProverData`) | 8 GiB + codeword 2^31 × 8 B = 16 GiB | Written once, block-major. p3 reads it for open_at. |
| `BlockWeights` G: `[block] → [Ext; 54]`, the batched ring weight per block | neo-spartan `ring_claims.rs` | 814,144 × 54 × 40 B = 1.76 GB | Key pass in (row, block) order. Pad pass is block-major. The matrix pass scatters into lane slots; it is single-threaded, ~85M nonzeros. |
| Lane weights ω: `[block][lane<64] → Ext` (prover only) | `sumcheck.rs` (consumes) | 2^26 × 40 B = 2.7 GB | Folded in place, lanes first. |
| `Instance`: c (22 × [F;54]), x (5 × [F;54]), r (K^28), y ([K;54]), y_j (t × [K;54]) | `instance.rs` | ~17 KB | Absorbed once, then read by `ring_claims` targets. |

The planes are the parent witness for **both** consumers. Pi_DEC commits them as 16
Ajtai children. Layer 1 commits them as 16 WHIR columns. No backend ever needs an
unsplit-parent API.

### 3.2 Module map

New crate `crates/neo-spartan`. It is the AoK for one CE(B) claim and the only
place that names Plonky3 0.8. The file sizes are *est.*

| File | Owns | Lines |
|---|---|---|
| `src/lib.rs` | Ownership/contract header, public items, `Error` | 110 |
| `src/profile.rs` | `Shape`, `Profile` (WHIR config choice, layer-1 error accounting, profile words) | 220 |
| `src/instance.rs` | `Relation` (matrix rows + fixed production key), `Instance` (validated CE(B) instance) | 170 |
| `src/planes.rs` | `DigitPlanes` (validation, WHIR table export, trit and integer views) | 200 |
| `src/ring_claims.rs` | The single list of ring-linear claims, `Batching`, `block_weights`, `batched_target`, `lane_weights`, `block_mle` | 420 |
| `src/sumcheck.rs` | Batched degree-4 sum-check: SVO prover rounds, verifier, final value | 420 |
| `src/transcript.rs` | `Duplex` wrapper: domain chunk, handoff, observe/sample helpers | 140 |
| `src/whir.rs` | Every p3-whir type: aliases, config, commit/open_at/verify_at, proof bytes | 260 |
| `src/field.rs` | F(0.5)↔F(0.8) conversion, K→(Re, Im), `Ext` ring product mod Φ81, `bar`/`bar⁻¹` on `Ext` | 160 |
| `src/proof.rs` | `Proof` and its strict codec | 140 |
| `src/protocol.rs` | `prove` and `verify`: the step list, side by side | 200 |
| `tests/poseidon2_parity.rs`, `tests/internal/{ring_claims,protocol}.rs` | see §7 | 450 |

Changes in `crates/nightstream`:

| File | Change | Lines (after) |
|---|---|---|
| `src/lib.rs` | export `CompressedProof` | +1 |
| `src/circuit.rs` | `Prover::finish_with_spartan`, `Prover::decode_proof` (moved); `Verifier::{decode_proof, verify}` retyped | ~250 |
| `src/lifecycle/finish.rs` (new) | `CompressedEnvelope`, `finish_with_spartan`, compressed `verify` | ~190 |
| `src/lifecycle/verify.rs` | extract `check_terminal_statement` (steps 1–7, shared); rename to `verify_accumulator` | ~280 |
| `src/lifecycle/encoding.rs` | `NS-FINAL-PROOF01` codec, reusing the claim writers | ~500 |
| `src/lifecycle/mod.rs` | `compression: neo_spartan::Profile` field built in `from_package` | ~175 |
| `src/lifecycle/prove.rs` | `prove_parent` backend dispatch; `prove` = `prove_parent` + per-backend PiDEC | ~150 |
| `src/folding/compose.rs` | `ParentFold`, `prove_parent_with_rows`, `verify_parent`; old functions call them | ~110 |
| `src/folding/pi_dec.rs` | `prove_with_production_key` takes the split digits (the split moves into `prove_parent`) | ~270 |
| `src/folding/params.rs` | `fold_error_log2()` (exact census, unrounded) | ~90 |
| `src/folding/transcript.rs` | `Transcript::into_inner` | +4 |
| `src/engine/{metal,paper_exact,crosscheck}.rs` | each prove = `prove_parent` + its existing D tail | ≤ 160 each |
| `tests/engines/compression.rs` (new, `#[path]` from `engine/mod.rs`) | toy end-to-end + mutations | ~300 |

A flow traces through at most three files: `circuit.rs` → `lifecycle/finish.rs` →
`neo_spartan::protocol.rs`. From there it reaches one leaf (`ring_claims`, `sumcheck`, or `whir`).

### 3.3 Type sketch

```rust
// ---- crates/neo-spartan/src/lib.rs ----
//! Argument of knowledge for one SuperNeo CE(B, L) claim (Definition 21): layer 1
//! of Nightstream's terminal compression.
//! Owns: the digit-plane witness table, the layer-1 duplex after the fold handoff,
//! the batched sum-check, and every Plonky3 0.8 type.
//! Does not own: how the claim was produced, the Ajtai seed or the matrix program
//! (read through `Relation`), or end-to-end security accounting.
//! Contract: `verify` accepts, except with probability 2^-error_bits, only if the
//! prover knows D_i ∈ {-1,0,1}^W (i < planes) with z = Σ 2^i D_i satisfying
//! c = L(z), x = L_in(z), Eval_K(z, r) = y, Eval_A_j(z, r) = y_j. Hence
//! ||z||_∞ ≤ 2^planes − 1 < B.
mod field; mod instance; mod planes; mod profile; mod proof; mod protocol;
mod ring_claims; mod sumcheck; mod transcript; mod whir;
pub use instance::{Instance, Relation};
pub use planes::DigitPlanes;
pub use profile::{Profile, Shape};
pub use proof::Proof;
pub use protocol::{prove, verify};

#[derive(Debug, thiserror::Error)]
pub enum Error { Shape(&'static str), Witness(&'static str), Security { required: f64, available: f64 },
                 Rejected(&'static str), Codec(&'static str), Rows(#[from] neo_reductions::PiCcsError) }

// ---- profile.rs ----
/// Dimensions fixed by the circuit: blocks (W/54), public blocks, matrices t,
/// point length ell, planes (= k_rho). Derived: block_vars = ceil_log2(blocks),
/// lane_vars = 6, variables = block_vars + 6 (26 in production).
pub struct Shape { blocks: usize, public_blocks: usize, matrices: usize, point_len: usize, planes: usize }
impl Shape {
    pub fn new(matrices: MatrixShape, public_width: usize, point_len: usize, planes: usize) -> Result<Self, Error> {
        // TODO: reject public_width % 54 != 0, matrices.columns % 54 != 0, planes == 0,
        // and block_vars + 6 + ceil_log2(planes) + 1 > 32 (WHIR codeword above two-adicity).
        unimplemented!()
    }
}
/// The shape plus WHIR parameters. Holding one proves the layer-1 error is at most 2^-target.
pub struct Profile { shape: Shape, whir: whir::Config, error_bits: f64 }
impl Profile {
    pub fn new(shape: Shape, target_bits: f64) -> Result<Self, Error> {
        // TODO: algebraic = 153 · 2^-319.99 (§5); raise the WHIR per-term level from
        // ceil(target) until log-sum(prescribed_security(), algebraic) ≥ target.
        unimplemented!()
    }
    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn error_bits(&self) -> f64 { self.error_bits }
    pub(crate) fn max_proof_bytes(&self) -> usize { unimplemented!() }
}

// ---- instance.rs ----
/// Public structure: package matrices + the fixed production Ajtai key (PRODUCTION_SEED, 22 rows).
pub struct Relation<'a> { rows: &'a dyn MatrixRows, workspace_bytes: usize } // `new(rows, bytes)`
/// Validated CE(B, L) instance; padding slots (54..64) checked zero and dropped.
pub struct Instance { commitment: Vec<[F; D]> /* Commitment::col(i) */, public: Vec<[F; D]>,
                      point: Vec<K> /* low bit first */, eval_k: [K; D], eval_a: Vec<[K; D]> }
impl Instance { pub fn from_claim(claim: &CeClaim, shape: &Shape) -> Result<Self, Error> { unimplemented!() } }

// ---- planes.rs ----
/// The witness: z = Σ_i 2^i D_i, every D_i ∈ {-1,0,1}^{54×blocks}. Construction is
/// the only validation, so holding one proves ||z||_∞ ≤ 2^planes − 1 on the prover side.
pub struct DigitPlanes { positive: Vec<Vec<u64>>, negative: Vec<Vec<u64>> } // [plane][block]
impl DigitPlanes {
    /// Accepts the PiDEC/Metal split (compact signed-unit matrices, 54 rows).
    pub fn from_planes(planes: Vec<Mat<F>>, shape: &Shape) -> Result<Self, Error> {
        // TODO: count == shape.planes; rows == 54; cols == blocks;
        // `signed_unit_masks()` is Some, or every dense entry ∈ {0, 1, p-1}; pos & neg == 0.
        unimplemented!()
    }
    pub(crate) fn table(&self, variables: usize) -> Vec<F08> { unimplemented!() } // row-major, 16 cols
    #[cfg(test)] pub(crate) fn from_values_unchecked(values: Vec<Vec<i64>>) -> Self { unimplemented!() }
}

// ---- proof.rs ----
pub struct Proof { root: whir::Root, rounds: Vec<sumcheck::RoundPoly>, opening: whir::Opening }
impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    /// Fixed parts are exact. The WHIR part is length-prefixed, bounded by
    /// `profile.max_proof_bytes()`, and bincode-decoded under that limit.
    pub fn from_bytes(profile: &Profile, bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
}

// ---- protocol.rs (the contract; numbers refer to §4.3) ----
pub fn prove(tr: Poseidon2Transcript, profile: &Profile, relation: &Relation,
             instance: &Instance, witness: &DigitPlanes) -> Result<Proof, Error> {
    // L1–L3 let mut duplex = transcript::start(tr, profile, instance)?;
    // L4    let (root, committed) = profile.whir.commit(witness, &mut duplex);
    // L5    let batching = Batching::sample(&mut duplex, profile.shape());
    //       let g = ring_claims::block_weights(relation, instance, &batching, profile.shape())?;
    //       let omega = g.lane_weights(&batching); drop(g);
    // L6    let claim = sumcheck::Claim::sample(&mut duplex, profile.shape(), ring_claims::batched_target(instance, &batching));
    // L7    let (rounds, point) = sumcheck::prove(&mut duplex, witness, omega, &claim);
    // L8    let opening = profile.whir.open(committed, &point, &mut duplex);
    unimplemented!()
}
pub fn verify(tr: Poseidon2Transcript, profile: &Profile, relation: &Relation,
              instance: &Instance, proof: &Proof) -> Result<(), Error> {
    // L1–L4 same start; profile.whir.observe(&proof.root, &mut duplex);
    // L5–L6 same samples; target = batched_target(instance, &batching);
    // L7    let (point, last) = sumcheck::verify(&mut duplex, &proof.rounds, &claim)?;
    // L8    let evals = profile.whir.verify(&proof.root, &proof.opening, &point, &mut duplex)?;
    //       let g_hat = ring_claims::block_weights(...)?.block_mle(&point[6..]);   // the milestone-2 seam
    //       let weight = ring_claims::weight_at(&batching, &g_hat, &point[..6]);    // ⟨γ, Ĝ ⋆ E⟩
    //       require sumcheck::final_value(&claim, weight, &evals, &point) == last
    unimplemented!()
}

// ---- ring_claims.rs ----
/// The one list of ring-linear conjuncts. Every claim is Σ_b W_b ⋆ z_b = t in R_F.
pub(crate) enum RingClaim { CommitmentRow(usize), EvalK(Part), EvalA(usize, Part), Public(usize) }
pub(crate) enum Part { Re, Im }
pub(crate) fn ring_claims(shape: &Shape) -> impl Iterator<Item = RingClaim> { unimplemented!() }
/// μ per claim (commitment rows: μ_C · eq(ρ_c, i)), and γ ∈ Ext^54 for the coefficients.
pub(crate) struct Batching { row_point: [Ext; 5], coefficient: Vec<Ext>, gamma: [Ext; D] }
/// G_b = Σ_k μ_k W^(k)_b for every block (Ext coefficients).
pub(crate) struct BlockWeights(Vec<[Ext; D]>);
pub(crate) fn block_weights(relation: &Relation, instance: &Instance, batching: &Batching,
                            shape: &Shape) -> Result<BlockWeights, Error> {
    // TODO(key):    G_b += μ_C Σ_{i<22} eq(ρ_c,i) · coefficient_block(PRODUCTION_SEED, i, b)   (streamed SHAKE)
    // TODO(pad):    H[54b+l] += μ_K,Re·Re χ_r(54b+l) + μ_K,Im·Im χ_r(54b+l)                       (EqualityWeights, 28 bits)
    // TODO(rows):   H[c] += M_j[i,c] · (μ_A_j,Re·Re χ_r(i) + μ_A_j,Im·Im χ_r(i))                 (one MatrixRows pass)
    // TODO(public): G_b[0] += μ_X_b for b < public_blocks                                            (W_b = 1)
    // G_b += bar(H_b) for every block (bar applied to Ext vectors; F-linear).
    unimplemented!()
}
/// V = ⟨γ, Σ_k μ_k t_k⟩, with t = c_i, Re/Im y, Re/Im y_j, X_b.
pub(crate) fn batched_target(instance: &Instance, batching: &Batching) -> Ext { unimplemented!() }
impl BlockWeights {
    /// Prover: ω_b = bar⁻¹(bar(γ) ⋆ G_b), so ⟨ω_b, z_b⟩ = ⟨γ, G_b ⋆ z_b⟩. Lanes 54..64 are zero.
    pub(crate) fn lane_weights(self, batching: &Batching) -> Vec<Ext> { unimplemented!() }
    /// Verifier: Ĝ(s_hi) = Σ_b eq(s_hi, b) G_b.
    pub(crate) fn block_mle(&self, s_hi: &[Ext]) -> [Ext; D] { unimplemented!() }
}
/// ω~(s) = ⟨γ, Ĝ(s_hi) ⋆ E(s_lo)⟩, E_l = eq(s_lo, l) for l < 54.
pub(crate) fn weight_at(batching: &Batching, g_hat: &[Ext; D], s_lo: &[Ext]) -> Ext { unimplemented!() }
```

nightstream side:

```rust
// ---- lifecycle/finish.rs ----
//! Terminal compression. Owns: one more PiCCS + PiRLC fold without PiDEC (layer 0),
//! the neo-spartan argument for its CE(B) parent (layer 1), and the transcript
//! handoff between them. Does not own the reductions, the statement checks
//! (verify.rs), or the byte format (encoding.rs).
pub struct CompressedEnvelope {
    state: Stage1State,
    running: Vec<CeClaim>,     // 16 terminal running claims, statement form
    fresh: CcsClaim,           // latest fresh claim: c and x
    pi_ccs: pi_ccs::Proof,
    pi_rlc: pi_rlc::Proof,     // redundant `combined`, checked against the recomputation
    layer1: neo_spartan::Proof,
}
impl PreparedLifecycle {
    pub(crate) fn finish_with_spartan(&self, envelope: Stage1Envelope) -> Result<CompressedEnvelope, FinishError> {
        // Active only, else FinishError::Initial. Normalize exactly as extend does:
        //   checked_prior_state → prepare_running → validate_running_parent_authority.
        // let mut tr = Transcript::session();
        // let parent = self.prove_parent(&mut tr, vec![fresh], running)?;          // C, R, split
        // let layer1 = neo_spartan::prove(tr.into_inner(), &self.compression, &self.relation()?,
        //     &Instance::from_claim(&parent.claim, ..)?, &DigitPlanes::from_planes(parent.digits, ..)?)?;
        unimplemented!()
    }
    pub(crate) fn verify(&self, expected: &Stage1State, proof: &CompressedEnvelope) -> Result<(), VerifyError> {
        // let digest = self.check_terminal_statement(expected, &proof.state, &proof.running, &proof.fresh)?;
        // let mut running = RunningInstance::new(proof.running.clone(), vec![], None);
        // prepare_running(&mut running, &self.params, digest);
        // let mut tr = Transcript::session();
        // let parent = nifs::verify_parent(&mut tr, .., &[proof.fresh.clone()], &running, &proof.pi_ccs, &proof.pi_rlc)?;
        // neo_spartan::verify(tr.into_inner(), &self.compression, &self.relation()?, &Instance::from_claim(&parent, ..)?, &proof.layer1)
        unimplemented!()
    }
}

// ---- folding/compose.rs ----
/// The parent of one PiCCS + PiRLC fold. Its witness exists only as the k_rho
/// signed digit planes (Z = Σ 2^i D_i): PiDEC commits them, layer 1 commits them.
pub(crate) struct ParentFold {
    pub pi_ccs: pi_ccs::Proof, pub pi_rlc: pi_rlc::Proof,
    pub claim: CeClaim, pub digits: Vec<Mat<F>>, pub flags: Vec<bool>,
}
pub(crate) fn prove_parent_with_rows(/* as prove_owned_with_rows */) -> Result<ParentFold, Error> { unimplemented!() }
pub(crate) fn verify_parent(tr: &mut Transcript, pp: &Params, s: &Structure, mix: RlcMixer, combine: DecMixer,
    fresh: &[CcsClaim], running: &RunningInstance, c: &pi_ccs::Proof, r: &pi_rlc::Proof) -> Result<CeClaim, Error> {
    // validate_running_parent_authority → pi_ccs::verify → pi_rlc::verify. `verify` = this + pi_dec::verify.
    unimplemented!()
}
```

### 3.4 Plonky3 0.8 isolation

We use **renamed dependencies, isolated in neo-spartan**, and we do not upgrade the
workspace now. neo-spartan's `Cargo.toml` declares `p3-whir = "=0.8.0"` and
`p3-{field,goldilocks,challenger,commit,dft,merkle-tree,symmetric,poseidon2,
multilinear-util,sumcheck,matrix}-08 = { package = "p3-…", version = "=0.8.0" }`.
It enables p3-whir's `parallel` feature. That is an external crate feature, not a
new Nightstream feature. Why:
(1) The Poseidon2 constants, the transcript, and every pinned fixture and Lean
artifact are protocol-binding and are audited on 0.5.3. A workspace bump touches
ten crates for a reason unrelated to this milestone.
(2) The boundary is narrow. Values cross as canonical u64 in `field.rs`, once per
input word or plane trit.
(3) A later standalone upgrade PR, with fixture parity, removes the renames and
leaves the layer-1 design unchanged.
Inside neo-spartan, `whir.rs` is the only file that names p3-whir types.
`transcript.rs` wraps p3's `DuplexChallenger` in a local `Duplex` struct with these
methods: `observe_words`, `observe_ext`, `sample_ext`, and `challenger_mut`. The
last one is crate-private to `whir.rs`. `Ext` (= `BinomialExtensionField<F08, 5>`)
is an internal alias. No public item has a p3 0.8 type in its signature.

### 3.5 Invariants and where validation lives

- **In types.** `DigitPlanes` exists ⇒ every entry is a signed unit, and the count,
  rows, and columns match the shape. `Instance` exists ⇒ the CE(B) shape is valid and
  padding is zero. `Profile` exists ⇒ the layer-1 error is ≤ 2^-target. `Proof`
  fields are private and are decoded only under the profile's length bound.
  `prove`/`verify` take the transcript **by value**, so no code can use the fold
  transcript after the handoff. `ParentFold` has no unsplit witness, so "the parent
  witness is its planes" is the only representation.
- **At boundaries.** Byte decoding is in `encoding.rs` (exact lengths from the
  circuit) and `Proof::from_bytes`. The statement checks are in
  `check_terminal_statement`. One function is shared by the accumulator and
  compressed verifiers, so the two verifiers have one source of truth. The claim
  shape is checked in `Instance::from_claim`, after `pi_rlc::verify`.
- **Inside.** Code trusts the types. `prove` does **not** check that the witness
  satisfies the instance; an invalid witness gives a proof that `verify` rejects.
  The mutation tests depend on this.
- **One list per invariant.** `ring_claims()` enumerates the conjuncts once.
  `Batching`, `block_weights`, and `batched_target` all iterate it. Two quantities
  are computed by two formulas, the prover's ω table and the verifier's
  `weight_at`. A test pins them as equal (§7).

### 3.6 Interface depth

The public surface adds one Prover method, one type with `state()`/`to_bytes()`,
and a retyped `Verifier::{decode_proof, verify}`. Hidden behind it: the layer-0
re-fold, digit planes, five conjuncts → 37 ring claims → 1 scalar claim, the
zero-check, the transcript handoff, two Plonky3 versions, WHIR parameters, and the
security composition. neo-spartan's own surface is 7 items (`Shape`, `Profile`,
`Relation`, `Instance`, `DigitPlanes`, `Proof`, `prove`/`verify`). It is
crate-facing, not client-facing.

## 4. Protocol

### 4.1 Layer 0 (an ordinary fold, stopped before PiDEC)

Prover:
1. Take the Active envelope: state, 16 running claims and their digit witnesses, one
   fresh instance. `checked_prior_state` gives the state digest d. `prepare_running`
   sets fold_digest = d, pads evaluations to 64, and rebuilds parent_authority.
   Then `validate_running_parent_authority` runs. This is the same code `extend` uses.
2. `Transcript::session()`. Pi_CCS: `reset_v1_1` → "Nightstream/SuperNeo/fold/v2" →
   statement stream (d, fresh c, fresh x) → α/γ → 28 rounds → absorb the outputs'
   Eval_K/Eval_A. The result is 17 CE(b) outputs at a shared r ∈ K^28.
3. Pi_RLC: ρ_1..ρ_17 (squeeze_digest + [4, i]). The parent is c, X, r, y, y_j =
   Σ rot(ρ_i)·(…). The witness comes as 16 planes: on CPU and PaperExact,
   `split_b_matrix_k_with_nonzero_flags(Z, 16, 2)`; on Metal, `split_rlc_witnesses`.
   The transcript now has absorbed = 0. **Stop.**

Verifier:
1. `check_terminal_statement`: state equality, canonical iteration, Active, 16
   running claims (no adv, 1,188-word commitment, m_in = 270, shared r of length 28,
   zero surplus evaluations), fresh shape, `check_pi_ccs_v1_1_canonical_children`,
   state preimage → Poseidon2 state hash → encHash == fresh.x. Output: d.
2. `prepare_running` on the claims, then `verify_parent`
   (`validate_running_parent_authority`, `pi_ccs::verify`, `pi_rlc::verify`). Output:
   the parent P, recomputed by the verifier. The carried `combined` is only compared.

### 4.2 Layer 1 algebra: five conjuncts → one scalar linear claim + one zero-check

Layout: the committed table T has 16 columns (planes) over the 26-variable cube
x = (lane bits x_0..x_5, block bits x_6..x_25), low bit first, row = 64·b + l. Lanes
54..63 and blocks ≥ 814,144 are padding. They carry zero weight in every claim.

Every conjunct except the norm is a ring-linear claim Σ_b W_b ⋆ z_b = t in R_F:

| Claims | W_b | t | Count |
|---|---|---|---|
| Commitment row i | a_{i,b} (key element, SHAKE) | c_i = `Commitment::col(i)` | 22 |
| Eval_K, part ∈ {Re, Im} | bar(part(χ_r(54b + ·))) | part(y) | 2 |
| Eval_A_j, part | bar(part((M_j^T χ_r)(54b + ·))) | part(y_j) | 2t = 8 |
| Public block b < 5 | 1 for that block, 0 elsewhere | X_b | 5 |

The Re/Im split is **exact**. z is F-valued and `bar` acts F-linearly on each part,
so Re(Σ bar(w_b) ⋆ z_b) = Σ bar(Re w_b) ⋆ z_b. No error term, and no F_{p^10} in v1.

Batching. Sample μ (commitment rows use μ_C·eq(ρ_c, i), ρ_c ∈ Ext^5, for milestone 2)
and form G_b = Σ_k μ_k W^(k)_b ∈ R_Ext and T = Σ_k μ_k t_k. Sample γ ∈ Ext^54. The
claim becomes ⟨γ, Σ_b G_b ⋆ z_b⟩ = ⟨γ, T⟩ =: V. The SuperNeo identity
ct(bar(a)·b) = ⟨a, b⟩ (ring.rs) gives ⟨γ, G_b ⋆ z_b⟩ = ⟨ω_b, z_b⟩ with
**ω_b = bar⁻¹(bar(γ) ⋆ G_b)**. That is one Ext ring product per block, and it needs
no quotient polynomial. With z = Σ_i 2^i D_i:

  Σ_x ω(x) · Σ_i 2^i D_i(x) = V.                                    (linear)

Norm: Σ_x eq(τ, x) · Σ_i λ^i (D_i(x)^3 − D_i(x)) = 0.               (zero-check)

Sum-check claim: Σ_x [ω(x)·Z(x) + η·eq(τ,x)·N(x)] = V, with Z = Σ 2^i D_i and
N = Σ λ^i (D_i^3 − D_i). The degree per variable is 4.

### 4.3 Transcript schedule (one sponge from the fold reset to the last WHIR query)

The layer-1 `Duplex` is p3 0.8 `DuplexChallenger<F, Perm, 16, 12>`. `Perm` is
**our** width-16 Poseidon2, rebuilt in p3 0.8 from
`neo_ccs::crypto::poseidon2_goldilocks::round_constants()`. A parity test pins it
(§7). If 0.8's internal layer differs, `Perm` wraps our 0.5.3 permutation as a 0.8
`CryptographicPermutation` (correct, slower Merkle hashing). The same `Duplex` is
the `challenger` argument of `WhirProver::commit`, `open_at`, and `verify_at`, so
every WHIR observation, OOD sample, folding challenge, PoW nonce, and query index
comes from it.

| Step | Prover | Verifier | Words |
|---|---|---|---|
| L1 | observe the domain chunk "Nightstream/SuperNeo/compress/v1" (8 LE bytes per word, zero-padded to 12) | same | 12 |
| L2 | observe the full 16-lane Pi_RLC end state (requires absorbed = 0) | same, from its own replay | 16 |
| L3 | observe the profile words (blocks, public blocks, t, κ, planes, ell, variables, WHIR rate/folding/level/pow/assumption), then the instance: c, X, r (Re, Im), y (54 × 2), y_j (t × 54 × 2) | same, from P | ~2,140 |
| L4 | `commit(table)`: p3 observes the Merkle root | `observe_commitment(root)` | 4 |
| L5 | sample ρ_c ∈ Ext^5, μ ∈ Ext^(1+2+2t+5), γ ∈ Ext^54 | same | — |
| L6 | sample τ ∈ Ext^26, λ ∈ Ext, η ∈ Ext | same | — |
| L7 | for t in 0..26: observe h_t(0..=4) (5 Ext), sample s_t | check h_t(0)+h_t(1) = claim; claim := h_t(s_t) | 26 × 25 |
| L8 | `open_at(columns 0..16 at rev(s))`: OOD samples, observe the 16 evaluations, WHIR rounds | `verify_at(...)` returns the evaluations e_i | p3 |
| end | — | require claim == ω~(s)·Σ 2^i e_i + η·eq(τ,s)·Σ λ^i(e_i^3 − e_i) | — |

Domain separation: Pi_CCS resets the transcript to the zero state and absorbs its
own chunk, so layer 0 is byte-identical to the C/R prefix of a normal fold. Nothing
else ever continues a Pi_RLC end state (Pi_DEC has no transcript, and the next fold
resets). L1 marks the continuation as layer 1. L2 chains the full 1,024-bit state
into a fresh duplex, so the handoff uses only p3's public API and stays in one mode.
L3 makes layer 1 a standalone AoK for any instance (the toy tests use this), and it
costs ~180 permutations. The p3 `rev` converts our low-bit-first point to p3's
big-endian `Point`. A test pins this convention.

### 4.4 Sum-check prover (26 rounds, degree 4)

Variables are bound lanes first. The first 6 rounds read the 54-lane masks, and the
data shrinks without leaving the block.
- Rounds 1–3 (SVO). For each (plane, block, remaining lane bits), the folded value
  depends only on a tuple of 2, 4, or 8 trits. The prover accumulates eq(τ_rest)
  into buckets keyed by (plane, tuple), using additions only. Then it multiplies
  each bucket by tabulated cubes at X ∈ {0, 2, 3, 4} (tables of 9 / 81 / 6,561 Ext).
  The prover skips zero tuples. Honest planes 12..15 are all zero, because
  |z| ≤ 3,672 < 2^12.
- Round 4 reads u16 indices into the 6,561-entry table (268 MB). From round 5, the
  arrays are Ext: 16 × 2^22 × 40 B = 2.7 GB, halving each round.
- The linear part is a standard product sum-check on (ω, Z). Z starts as i32 (256 MB).
- ω and the sum-check arrays are freed before L8.

### 4.5 Fields

- Layer 0: unchanged (F, K = F_{p^2}).
- Layer 1: every challenge and the WHIR extension is **Ext = F_{p^5}**
  (`BinomialExtensionField<Goldilocks, 5>`, x^5 − 3, 319.99 bits). K-valued data
  enters only as Re/Im F vectors (χ_r, y, y_j). Milestone 2 evaluates K-point matrix
  rows with Re/Im(eq~(r, σ)). It computes this in Ext[u]/(u² − 7). 7 is a non-residue
  mod p, and odd degree keeps it one, so this is a field. Re is Ext-linear, so it
  commutes with the MLE. That is a ~60-line pair type, not a p3 field.

### 4.6 Final weight in v1, and the 28-cube and 54-lane issues

The verifier computes ω~(s) = Σ_b eq(s_hi, b)·⟨E, ω_b⟩ with E_l = eq(s_lo, l), l < 54.
By the definition of ω_b, ⟨ω_b, E⟩ = ⟨γ, G_b ⋆ E⟩. So
**ω~(s) = ⟨γ, Ĝ(s_hi) ⋆ E(s_lo)⟩, with Ĝ(s_hi) = Σ_b eq(s_hi, b)·G_b ∈ Ext^54.**
That is one block MLE (O(54 · blocks)) and one ring product. In v1, G comes from the
same `block_weights` function that the prover uses. The function reads:
- the key, once, as streamed SHAKE128 via `coefficient_block`;
- the matrices, as one `MatrixRows` pass;
- the Pad weights χ_r over all W coordinates.

The 28-bit point is used only inside χ_r: `EqualityWeights` over 28 bits, with zero
high bits (Π(1 − r_t)), exactly as Pi_CCS uses it. The 54-lane packing never needs
to factor, because every weight is per block (G_b, 54 lanes) and the layer-1 cube
pads lanes to 64.

`block_mle` is the **milestone-2 seam**. It becomes:
- key part: an opening of the setup-committed key polynomial at (ρ_c, s_hi) over 54
  lanes, with one 6-round lane sum-check;
- matrix part: Re/Im(eq(r,·))-weighted matrix openings at (row, s_hi, lane), after a
  row sum-check;
- Pad part: a carry DP over the 28 bits of c = 54b + l (54 · 28 · O(160) states).
  This must be derived and tested in milestone 2.
- public part: O(5·54).

The rest of `verify` is unchanged.

### 4.7 Packing options for the stance

| Option | Committed cells | Verdict |
|---|---|---|
| **16 columns × 26 vars (chosen)** | 2^30 (65.5% dense) | Plane factor applied to 16 opened values. 26 rounds. Equals the k_rho = 16 split. |
| One 30-var poly, plane bits in the sum-check | 2^30 | Same cost, +4 rounds. The plane factor Π(1 − s + s·2^{2^t}) is fine, but it brings no gain. |
| 12 planes, flat (plane, c) into 2^29 | 2^29 | Halves the commit, but it binds an honest-bound plane count (3,672 < 2^12) that is not needed by policy, and it breaks tensor weights (carry DP plus an extra milestone-2 sum-check). Keep as a measured lever only. |
| Trits packed base-3 into one field element | 2^26 | Unpacking is the range proof again. Rejected. |
| Radix-4 digits, 8 planes, degree-7 zero-check | 2^29 | Halves the commit. **AGENTS forbids radix-four without approval.** Owner question. |

Metal reuse: `split_rlc_witnesses` already returns exactly these 16 compact planes.
`DigitPlanes::from_planes` reads their masks without dense expansion. Metal needs
no new API in v1.

## 5. Soundness map

|Ext| = p^5 ≈ 2^319.99. Statistical (RBR) terms follow the existing census convention.

| # | Step | Failure event | Bound | Authority |
|---|---|---|---|---|
| L0 | Pi_CCS + Pi_RLC | output CE(B) claim valid but some input invalid | field_factor/q² + fork_factor/\|C\| ≈ 2^-114.4 (+ 18/5^54 ≈ 2^-121.2), exact from the neo-params census | Lemma 7 (strong), Lemma 8 (weak), Thm 12 → **Lemma 1** |
| A1 | μ batching, eq(ρ_c, ·) rows | a false ring claim survives | ≤ 6/\|Ext\| | Schwartz–Zippel (multilinear in ρ_c, linear in μ) |
| A2 | γ batching of 54 coefficients | nonzero Ext^54 error orthogonal to γ | ≤ 1/\|Ext\| | linear |
| A3 | λ plane combination | some C(D_i(x)) ≠ 0 but Σ λ^i C(D_i(x)) = 0 | ≤ 15/\|Ext\| | degree-15 polynomial in λ |
| A4 | τ zero-check | N ≢ 0 on the cube but Σ eq(τ,x)N(x) = 0 | ≤ 26/\|Ext\| | multilinear in τ |
| A5 | η combination | ≤ 1/\|Ext\| | linear |
| A6 | 26 rounds, degree 4 | ≤ 104/\|Ext\| | sum-check (LFKN) |
| W | WHIR opening of 16 columns at s | composed `prescribed_security()` | ≤ 2^-target (see below) | WHIR, Johnson bound (proven) |
| H | Merkle/duplex hashing (4-word digests) | collisions | O(t²/2^255.9) | computational; same as every existing transcript |

A1–A6 ≤ 153/2^319.99 ≈ 2^-312.7.

**Target derivation (not invented).** The lifecycle needs
ε_L0 + ε_A + ε_W ≤ 2^-minimum. With minimum = 114 and ε_L0 = 2^-114.4:
ε_W ≤ 2^-114 − 2^-114.4 ≈ 2^-116.05. So `Profile::new` gets
target = −log2(2^-min − ε_L0) = 116.05 and selects the WHIR per-term level whose
composed bound reaches it. Lifecycle-level total: 2^-114.4 + 2^-116.1 ≈ 2^-114.0.
`Params::fold_error_log2()` supplies the exact ε_L0 (the census numerator and
denominator, unrounded). If ε_L0 ≥ 2^-min, `from_package` fails with
`InsufficientStatisticalSecurity`.

**Extraction.** WHIR's extractor yields 16 functions D_i on the 26-cube, fixed
before L5. A3–A4 ⇒ D_i(x) ∈ {−1, 0, 1} everywhere. A1, A2, A5, A6 ⇒ the 37 ring
claims hold. So z := Σ 2^i D_i, restricted to blocks < 814,144 and lanes < 54,
satisfies all five conjuncts of Definition 21, and ||z||_∞ ≤ 2^16 − 1 < B. That is
a CE(B, L) witness. Lemma 1, through the (2B, C)-relaxed binding (Module-SIS,
κ = 22, unchanged), then extracts 16 CE(b) running witnesses and one CCS(b) fresh
witness with norm < 2. Lemma 4 composes the AoK after the RoK.

**Checks lost relative to today's accumulator verifier** (see §9): only the fresh
completion-tail-zero check (`validate_fresh_witness_tail_zero`). It reads the
witness. The paper's relation over the W-column padded CCS does not require it.
X == project(witness), the recommit, and the row checks are all implied by the
extracted CE(b)/CCS(b) witnesses.

To settle: (1) pin the per-term WHIR level that reaches 116.05 bits with
`prescribed_security()` at 30 variables; (2) confirm, in the WHIR paper's
prescribed-point theorem, that p3-whir's layout batching is inside
`initial_claims_error`.

## 6. Costs (*est.*, 12-core Apple-class CPU; measure before relying)

| Phase | Prover | Verifier |
|---|---|---|
| Layer 0 (C, R, split) | C/R share of a fold: Metal ~10–15 s, CPU ~40–50 s | Pi_CCS verify (matrix-free) + RLC recompute < 0.1 s |
| Statement checks | — | ms |
| `block_weights` (SHAKE key + matrix pass + Pad) | 3–6 s | 3–6 s |
| ω: 814,144 Ext ring products | 3–5 s | — |
| WHIR commit, 2^30 cells (DFT 2^31 + ~4·10^8 Poseidon2 perms) | 20–30 s | — |
| Sum-check | 5–10 s | µs |
| WHIR open / verify | 5–15 s | < 50 ms |
| **Total layer 1** | **~40–65 s** | **~3–6 s** (today: 12.88 s CPU) |
| Peak memory | ~30 GB (table 8 + codeword 16 + ω or SC arrays ≤ 5.4) | ~2 GB (G 1.76) |

Proof size: 16 running claims 263 KB + 17 Pi_CCS output claims 280 KB + Pi_CCS
rounds 4 KB + combined 16 KB + fresh 12 KB + layer 1 (rounds 5.2 KB, WHIR
~300–500 KB) ≈ **0.9–1.1 MB**, compared with 221.7 MB. With pow_bits = 0, WHIR
queries are not reduced by grinding.

Milestone 2 replaces the verifier's `block_weights` with setup openings at Ĝ's
point. Verifier time goes from seconds to ~10–100 ms, and memory to MBs. The proof
grows by 2–3 openings plus lane and row sum-checks (*est.* +0.8–1.2 MB). Milestone 3
proves the verifier in a circuit: Poseidon2 duplex, 26 degree-4 rounds, one Ext ring
product, WHIR Merkle paths, and the milestone-2 openings. Trims (carry only output
evaluations, drop `combined`) save ~230 KB. The digit-plane choice makes this
verifier as small as it can be: no range-proof sub-protocol and no quotients.

## 7. Test plan (every invocation ≤ 300 s; `--release`; `FoldingMode::Optimized`)

Success criteria: (a) an honest toy end-to-end proof verifies; (b) each mutation
below is rejected; (c) the existing parity and lifecycle tests are byte-identical
after the `prove_parent` refactor.

neo-spartan (ms each):
- `tests/poseidon2_parity.rs`: p3 0.8 `Perm` == `neo_ccs` PERM on the zero state
  and on random states, and the domain chunk equals its pinned word vector.
- `tests/internal/ring_claims.rs` (`#[path]` from `ring_claims.rs`):
  (1) bar identity: ⟨γ, G ⋆ z⟩ == ⟨lane_weight(G), z⟩ for random G, γ, z, with
  `neo_math::Rq::mul` as the reference.
  (2) `weight_at(Ĝ(s_hi), s_lo)` == MLE of the prover's ω table at a random s.
  (3) **Semantic cross-check**: on a 2-block toy relation with the real key prefix
  and a random signed-unit witness, build c with
  `commit_production_signed_unit_prefix_matrix` and y, y_j with
  `eval_real_v1_1_openings`. Then Σ_x ω(x)z(x) == `batched_target`. This fails if
  layer 1's reading of any conjunct drifts from Pi_CCS's.
  (4) The WHIR `rev` convention: open a random 7-variable table and compare with
  our low-bit-first MLE.
- `tests/internal/protocol.rs`: honest prove/verify. The prover and verifier duplex
  states are equal at the end. Each of the following must be **rejected**:
  - one c word; one X word; one r coordinate; one eval_k coefficient; one eval_a
    coefficient;
  - one trit flipped in a plane (linear claims fail);
  - a non-trit with the same z, D_0 = 3 and D_1 = −1 instead of D_0 = 1 (zero-check
    only), via `from_values_unchecked` (`#[cfg(test)]`);
  - a trit in the tail lanes of the last carrier block (Eval_K/commitment);
  - the Merkle root; one h_t value; one opened evaluation; one WHIR proof byte;
  - the handoff state (verifier replays a different transcript).

nightstream (`tests/engines/compression.rs`, `#[path]` from `engine/mod.rs`; uses
the parity `Fixture::bit()` and `Fixture::selected_polynomial()`, m = 55, 2 blocks,
t = 1/4, 16 planes, 7-variable cube, WHIR on 2^11 cells):
- Optimized `prove_parent` → layer-1 prove → `verify_parent` → layer-1 verify accepts.
  The verifier's parent equals the prover's.
- Crosscheck `prove_parent` equals Optimized (proof, parent, digits, transcript).
- Rejected: one Pi_CCS output eval (changes the parent); one fresh-commitment word;
  one running-claim eval.
- Existing `parity.rs` and `dec_children` tests pass unchanged (refactor proof).
- Lifecycle helpers hard-code 16 / 270 / 28 / t = 4, so `finish.rs` glue is kept
  under 100 lines and is covered only by the ignored tests below.

Ignored (production size, run one at a time, Metal and CPU separately):
- `staged_fold.rs` `layer1` phase: load the saved `parent.json` +
  `parent-witness.json` from the existing `rlc` phase, split, prove, verify, and
  report time and peak RSS. Target < 300 s; if it is over, it is a failing slice.
- `circuit_lifecycle.rs`: loaded package, 2 steps, `finish_with_spartan`, `verify`.
  Then: a byte flip in the compressed proof is rejected; `verify_accumulator` and
  compressed `verify` agree on the same envelope.
- Benchmark: a `finish` phase in `nightstream-poseidon2-bench run` (a CLI flag, not
  a feature) times finish and compressed verify.

## 8. Answers to GROUNDING §5

1. **Where it lives.** The new crate `neo-spartan` owns the AoK and every p3 0.8 type.
   nightstream composes layer 0 (folding, refactored to stop at `ParentFold`) with
   layer 1 in `lifecycle/finish.rs`. Public: `finish_with_spartan`, `CompressedProof`,
   `Verifier::verify`.
2. **Field.** Ext = F_{p^5} for WHIR and all layer-1 coins. K data is split exactly
   into Re/Im F vectors before batching. v1 has no F_{p^10}. Milestone 2 adds a
   pair type for K-point rows.
3. **What is committed.** The 16 signed planes, as one WHIR table of 16 columns over
   the 26-variable (block 20 | lane 6) cube, 2^30 cells. z is never committed.
4. **Norm.** By construction: the zero-check Σ eq(τ,x)Σλ^i(D_i^3 − D_i) = 0, batched
   into the sum-check, gives |z| ≤ 2^16 − 1 < B, which is exactly CE(B). It does not
   use the tighter honest bound, because that would invent a plane count.
5. **C = A z.** A random combination of all 22 × 54 coefficient equations, with
   weights μ_C·eq(ρ_c, i)·γ_k, and no quotient. The prover's weight is
   ω_b = bar⁻¹(bar(γ) ⋆ G_b). The verifier's value at s is ⟨γ, Ĝ(s_hi) ⋆ E(s_lo)⟩,
   from one O(key + nnz + W) pass in v1.
6. **Batching.** 22 commitment rows + Eval_K Re/Im + 4 × Eval_A Re/Im + 5 public
   blocks = 37 R_F claims → μ → one R_Ext claim → γ → one scalar claim. Then η
   adds the zero-check, and one degree-4 sum-check over 26 variables runs.
7. **Transcript.** Layer 0 is the fold transcript (with Pi_CCS's reset). Layer 1 is
   a p3 0.8 `DuplexChallenger` over our Poseidon2 constants. It starts with
   "Nightstream/SuperNeo/compress/v1", then the 16-lane Pi_RLC end state, then the
   profile and instance. That same object is WHIR's challenger.
8. **Proof object.** `CompressedProof` (state, 16 running claims, fresh c/x, Pi_CCS
   proof, Pi_RLC combined, layer-1 proof). The codec `NS-FINAL-PROOF01` is in
   `encoding.rs` and reuses the claim writers. The layer-1 bytes are bounded by the
   profile. Entry points: `Verifier::decode_proof` and `Verifier::verify`.
9. **Backends.** Every backend implements `prove_parent` (C, R, split), and Metal
   already returns planes. Layer 1 is one CPU implementation for all engines in v1.
   Crosscheck compares `ParentFold`. Metal acceleration of the WHIR commit and the key
   pass comes later.
10. **Tests.** §7. Toy tests run in milliseconds and include the semantic
    cross-check against existing Pi_CCS kernels. Production tests are ignored and run
    one at a time.

## 9. Decision record

### Synthesis decision
(Filled in by arena.)

### Tradeoffs accepted
- We accept a 16x larger commitment (2^30 cells, *est.* ~30 GB and 20–30 s) in
  exchange for a norm that holds by construction, the exact CE(B) bound, no
  range-proof sub-protocol, and the smallest milestone-3 verifier.
- We accept 1.76 GB of verifier memory in v1 (dense G) in exchange for one code path
  that defines the conjuncts for both prover and verifier. Milestone 2 replaces the
  verifier's use of it.
- We accept two formulas for one weight (prover ω table, verifier ⟨γ, Ĝ ⋆ E⟩),
  pinned equal by a test, in exchange for a verifier formula that survives into
  milestone 2.
- We accept redundant data in the proof in v1 (full Pi_CCS outputs, the `combined`
  claim, ~230 KB) in exchange for reusing the existing claim codec and
  `pi_rlc::verify` unchanged.
- We accept dropping the fresh completion-tail-zero check in compressed mode (owner
  question 1). It is witness-only and outside the paper relation.
- We accept a public API break: `Verifier::verify`/`decode_proof` now take the
  compressed proof, and accumulator decode moves to `Prover`. In exchange there is
  one public verify. Ignored tests and the benchmark verify phase change with it.

### Alternatives considered
- **Commit z (2^26) + logUp-GKR range check on 8-bit limbs.** It commits ~2^27 cells
  (8x less prover commit). But it hides a 27-layer GKR (~380 sum-check rounds) behind
  the same `prove`/`verify`. The verifier and the milestone-3 circuit get several
  times deeper, and the code roughly doubles. The interface is equally narrow, but
  the module is deeper in the wrong place (the verifier). Lost.
- **Keep Pi_DEC and prove the 16 CE(b) children.** It commits the same 16 planes,
  but layer 1 then carries 16 × 37 ring claims and 16 Ajtai commitments as public
  data, and it contradicts the owner's layer-0 decision. Its interface exposes the
  child family to the verifier. Lost.
- **Layer 1 as a module inside nightstream.** It puts two p3 versions into one crate,
  makes p3 0.8 importable next to the lifecycle, and makes layer 1 harder to test
  alone. Lost on isolation.
- Point variants, not whole shapes: quotient-lift ring check (evaluate at α plus
  32 × 53 sent quotients; product-form key lane weight, but more prover code and
  14 KB more proof); 12-plane flat packing (§4.7); workspace p3 upgrade (§3.4).

### Open questions and risks
1. The compressed verifier cannot check that the last fresh witness has a zero
   completion tail. No earlier fresh witness is checked for it, and the padded-CCS
   relation does not need it. Do you accept dropping it, or must the final fresh
   witness get its own small commitment?
2. Is ~30 GB peak and ~1 min CPU for a one-time `finish_with_spartan` acceptable on
   the 64 GB machine (with the Metal session also resident)? If not, do you approve
   radix-4 digits inside layer 1 only (8 planes, degree-7 zero-check, 2^29 cells)?
3. WHIR grinding: v1 uses `pow_bits = 0`, because no authority sets a value. Do you
   want a PoW budget to cut queries and proof size?
4. Do you confirm `SecurityAssumption::JohnsonBound` (proven) rather than
   CapacityBound (conjectured, smaller proofs)?
5. Do you accept the API change (one `Verifier::verify` on `CompressedProof`;
   accumulator decode on `Prover`), or should `verify_compressed` sit beside the
   current `verify`?
6. Do you accept 4-word (256-bit) Merkle and duplex digests for the PQ target? They
   are the same as the existing transcript.
7. Risks: the p3 0.8 Poseidon2 build may not match our constants (fallback: wrap).
   The p3-whir `open_at` memory for a 30-variable stacked table is unmeasured. The
   p3-sumcheck 0.8 `Table`/`OpeningProtocol` API was read only through p3-whir's
   tests. A k_rho = 18 profile would need 32 columns (2^31 cells, at the two-adicity
   limit).

### Next implementation step
Build `crates/neo-spartan` at toy size: `planes`, `ring_claims` with its bar-identity
and semantic cross-check tests, `sumcheck`, and the `whir` wrapper, plus the
protocol mutation tests. Do this before touching the nightstream lifecycle.

### Red-flag screen
Shallow module: no. Leakage: p3 is confined to `whir.rs`/`transcript.rs`.
Temporal decomposition: no; the modules own knowledge. Pass-through:
`circuit.rs` → lifecycle (same as today). Split ownership: the transcript is
consumed at the handoff, and G has one builder. Two ways: no; the old paths are
`*_parent` + D. Importable internals: no. Hand-synced list: `ring_claims()` is the
only list. The ω/`weight_at` pair is pinned by a test.
