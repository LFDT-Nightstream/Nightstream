# Candidate C: compression milestone 1 (layer 0 + layer 1 v1), derived from first principles

Seat stance: none fixed. Result: commit the parent `z` once (54 lane columns) with its
logUp multiplicities in one WHIR commitment. Prove the norm with logUp-GKR against the exact
`B` table. Fold every linear conjunct (C = Az, x, Eval_K, Eval_A) into ONE ring identity with
ONE quotient. Then run ONE 20-round sum-check that ends at ONE opening point. Numbers marked
"est." are estimates, not measurements.

## 1. Problem

At the end of folding, Nightstream holds 16 running CE(b) claims and 1 fresh CCS claim with
17 witnesses (about 221.7 MB). The owner wants a post-quantum proof of about 114 bits that
contains no witness. Layer 0 runs Pi_CCS + Pi_RLC (no Pi_DEC) and gives one CE(B, L) claim.
Lemma 1 makes this a reduction of knowledge to CE(B). Layer 1 must be an argument of
knowledge for all five conjuncts of Definition 21 over the full carrier W = 54·n_R. These
facts make the shape non-obvious:

- K = F_{p²} is the only extension field in the workspace. WHIR's proven (Johnson-bound)
  errors need an extension of about 190 bits or more at 2^26. No odd-degree extension
  contains K.
- Φ81 has no root below F_{p^27}. A ring identity mod Φ81 cannot be checked by evaluation
  at a root. It needs a quotient or an exact reduction.
- The witness packs 54 lanes per ring element. 54 is not a power of two, so `c = 54b + l`
  does not factor as a tensor.
- After Pi_RLC, nothing enforces ‖z‖∞ < B. The parent has |z| ≤ 3,672, so it is not a
  signed-unit vector and it has no direct commit path. Metal keeps the parent on the device
  and returns only 16 digit planes.
- p3-whir 0.8 cannot share types with the workspace's p3 0.5.3. The rules require
  Poseidon2-only hashing in proof paths. Pi_CCS resets the transcript.
- v1 may read the key (967M coefficients, SHAKE128-derived) and the matrices. Milestones 2
  (setup commitments) and 3 (shrink circuit) must reuse the same protocol without a rewrite.

## 2. Usage (caller's view)

### Quickstart (README)

```rust
use nightstream::{Circuit, Engine, Verifier};

// Prover: fold as today, then finish once.
let circuit = Circuit::load("app.nspkg")?;
let prover = circuit.prover(Engine::Metal, 114)?;
let mut proof = prover.prove(z0, &steps[0])?;
for step in &steps[1..] {
    proof = prover.extend(&proof, step)?;
}
let final_proof = prover.finish_with_spartan(&proof)?;   // layer 0 + layer 1, no witnesses
let bytes = final_proof.to_bytes();

// Verifier: an independently configured circuit and an expected state.
let verifier = Verifier::from_package(&expected_circuit, Engine::Optimized, 114)?;
let final_proof = verifier.decode_final_proof(&bytes)?;
verifier.verify(&expected_state, &final_proof)?;
```

### Call site 1: sequencer that publishes a chain head

```rust
fn publish_head(prover: &Prover, head: &Proof, out: &mut impl Write) -> Result<State, Error> {
    let finished = prover.finish_with_spartan(head)?;   // Proof (accumulator) is not consumed
    out.write_all(&finished.to_bytes())?;
    Ok(finished.state().clone())
}
```

### Call site 2: light client

```rust
fn accept(verifier: &Verifier, wire: &[u8], expected: &State) -> Result<(), Error> {
    let proof = verifier.decode_final_proof(wire)?;   // strict, canonical bytes only
    verifier.verify(expected, &proof)                 // layer-0 replay + layer-1 check
}
```

### Call site 3: the only caller of the layer-1 crate (`lifecycle/finish.rs`)

```rust
let relation = neo_spartan::Relation::new(&rows, public_words, params.k_rho())?;
// prover side, after Pi_CCS + Pi_RLC advanced `transcript`:
let statement = neo_spartan::Statement::new(&relation, &parent.claim)?;
let spartan = neo_spartan::prove(&relation, &statement, &parent.witness, transcript.inner_mut())?;
// verifier side, after replaying Pi_CCS and deriving the parent:
neo_spartan::verify(&relation, &neo_spartan::Statement::new(&relation, &parent)?, &spartan,
                    transcript.inner_mut())?;
```

Public surface change: `FinalProof` (opaque; `state()`, `to_bytes()`),
`Prover::finish_with_spartan`, `Verifier::decode_final_proof`, and
`Verifier::verify(&State, &FinalProof)`. The envelope check that needs the witnesses stays
crate-private as a test oracle (see open question Q-A). `Proof` with `encode_proof` and
`decode_proof` stays public because it is still the accumulator that `extend` uses.

## 3. Shape

### 3.1 Derivation: assumptions challenged

| Assumption | Necessity or convention? | Result |
|---|---|---|
| Layer 1 must re-check the CCS gate (Spartan zero-check) | Convention. CE(B) has no gate; Pi_CCS already reduced it into Eval_A. | Drop it. Layer 1 = linear claims + norm. |
| K-valued claims need K arithmetic in layer 1 | Convention. z ∈ F, so each K claim is two F-linear claims (Re, Im). | Project K values with F-linear maps. No compositum field in v1. |
| Each ring row (22 key, Eval, x) needs its own quotient or check | Convention. Mix the rows first. Linearity gives one ring identity. | One quotient Q (53 elements). |
| The norm needs digit planes | Convention (the Pi_DEC habit). The least information to commit is z plus a 2^17 count table. | logUp against S = (−B, B). |
| One sum-check is simpler | It is not necessary. A one-sum-check norm proof must commit a challenge-dependent helper (3× z data) or 12 to 16 digit planes (12× to 16× z data). | logUp-GKR for the norm, plus one linear sum-check. |
| Commit z as one lane-padded 2^26 column | Convention. That layout forces 2^27 once m is added. | 54 columns × 2^20 + m column, stacked into 2^26. |
| WHIR needs F_{p^5} | Not necessary. The cubic extension (x³ − x − 1, about 2^192) clears every term by more than 30 bits (§5). | EF3. Fallback is one type alias. |
| p3 needs its own transcript object | Not necessary. Squeeze a digest from the layer-0 transcript into a p3 DuplexChallenger that uses the same permutation. | One bridge point. |
| Layer 0 needs new protocol code | No. `staged_fold.rs::ccs/rlc` already run C, then R, and replay. | Reuse; add only `pi_rlc::derive_parent`. |
| The verifier stores the weight vector | No. It needs one MLE value at one point. | Streaming passes. Each pass maps to one setup opening in m2. |

These facts remain. (T1) CE(B) has five conjuncts. Four are F-linear in z and one is the
norm. (T2) WHIR commits base-field tables and draws EF challenges. (T3) Linear claims batch
with error deg/|EF|. (T4) A range check that commits no helper needs a fraction-sum protocol.
(T5) Every prover message is bound before the next challenge, and all hashing uses the one
Poseidon2 permutation.

### 3.2 Data structures (layer 1) and access patterns

| Structure | Size (production) | Built from | Read by |
|---|---|---|---|
| `BlockTable` z: 2^20 rows (blocks; rows ≥ n_R are zero) × 54 lane columns, F | 453 MB | one transpose of the parent `Mat<F>` (54 × n_R) with the range check | WHIR commit (moved in; read back through `WhirProverData::table(0)`), GKR leaves, Q, Zζ, Ze. All passes go block by block, in sequence. |
| `m`: logUp counts, 2^17 rows × 1 column | 1 MB | one pass over the 2^26 virtual cells | WHIR commit, table-side GKR |
| GKR layers (p, q) ∈ EF3², sizes 2^25 … 1 (z side), 2^16 … 1 (table) | est. 3.2 GB (all layers; 1.6 GB is layer 25) | bottom-up from leaves | top-down rounds, then freed |
| `u` ∈ EF3^W: projected Eval weights | 1.05 GB | transposed matrix scatter (one column band per thread) + χ_r pass | built into ḡ, then freed |
| `CombinedRow` ḡ_b ∈ EF3^54 per block | 1.05 GB | key pass (SHAKE per (row, block)) + bar(u_b) + public rows | Q (before ζ), G_b = ḡ_b(ζ) (after ζ) |
| Linear tables G, Zζ, E, Ze ∈ EF3^(2^20) | 100 MB | per block | 20-round sum-check |

The verifier allocates nothing of size W. It streams the key (block-parallel), χ_r (as
low/high factor tables), and the matrix runs (row-parallel forward pass, per-thread scalar
accumulator).

### 3.3 Module map

New crate `crates/neo-spartan`. It owns layer 1 and every p3 0.8 dependency.

| File | Owns | Lines (est.) |
|---|---|---|
| `src/lib.rs` | Public API (`Relation`, `Statement`, `Proof`, `prove`, `verify`, `Error`) and the full layer-1 Fiat–Shamir schedule as one readable sequence | 350 |
| `src/ring.rs` | Conjuncts 1, 2, 4, 5: the single list of ring rows, row mixing, quotient, prover weight table, streaming verifier weight `G~(s)` | 460 |
| `src/norm.rs` | Conjunct 3: table S, multiplicities, logUp-GKR prover and verifier | 380 |
| `src/sumcheck.rs` | Degree-generic round prover and verifier on the challenger (`Oracle` trait: GKR layer and linear product) | 150 |
| `src/pcs.rs` | WHIR profile constants, stacked tables, `open_at`/`verify_at`, point-order reversal | 250 |
| `src/hash.rs` | Poseidon2 as a p3 0.8 permutation (built from `neo_ccs` round constants), MMCS and challenger types, transcript bridge | 150 |
| `src/field.rs` | `Gl`, `Ext` aliases, u64 conversions, Re/Im split, eq tables | 100 |
| `src/codec.rs` | Strict `Proof` bytes (our fields by hand, `PcsProof` with bincode and a re-encode check) | 180 |

Changes in `crates/nightstream`:

| File | Change | Lines (est.) |
|---|---|---|
| `src/lifecycle/finish.rs` (new) | `FinalProof`, `finish_with_spartan`, `verify_final`, layer-0 driver | 260 |
| `src/lifecycle/final_encoding.rs` (new) | `NS-FINAL-PROOF01` codec; reuses the claim writers of `encoding.rs` | 200 |
| `src/lifecycle/verify.rs` | Move the statement-only checks (steps 1–7) into `check_terminal_statement`, used by both verifiers | ±0 |
| `src/lifecycle/extend.rs` | Move the sequence checked_prior_state → prepare_running → validate_running_parent_authority into `prepared_running`, used by extend and finish | −15/+10 |
| `src/folding/pi_rlc.rs` | `derive_parent(tr, …, outputs) -> CeClaim` (sample ρ + `rlc_public`); `verify` becomes derive + compare | +15 |
| `src/circuit.rs`, `src/lib.rs` | Three public methods and one re-export | +45 |
| `crates/neo-transcript/src/poseidon2.rs` | `domain_chunk_v1_1(label)`; `fold_domain_chunk_v1_1` calls it | +8 |

Flow traces stay within three files: `circuit.rs` → `lifecycle/finish.rs` →
`neo-spartan/src/lib.rs` (schedule) → one conjunct module.

### 3.4 Type sketch

```rust
// ===== crates/neo-spartan/src/lib.rs =====
//! CE(B) argument of knowledge (layer 1 of compression).
//! Owns: proof that one CE(B, L) claim (SuperNeo Def. 21, B = 2^k_rho) holds for a committed
//! witness, and the Fiat-Shamir schedule that continues the layer-0 transcript.
//! Does not own: layer 0, state binding, the outer FinalProof bytes.
//! Invariants: all five conjuncts over the full carrier; hashing = workspace Poseidon2 only;
//! no p3 0.8 type appears in this public API.
mod codec; mod field; mod hash; mod norm; mod pcs; mod ring; mod sumcheck;

pub type CeClaim = neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>;

/// Fixed CE(B) structure: production Ajtai key prefix (seed and κ from neo-ajtai), CCS
/// matrices, public width, norm exponent. v1 reads key and matrices directly; milestone 2
/// swaps the reads for setup openings behind this type.
pub struct Relation<'a> { matrices: &'a dyn MatrixRows, blocks: usize, public_blocks: usize, k_rho: u32 }
impl<'a> Relation<'a> {
    /// Checks: matrix columns = 54·blocks ≤ key capacity; public_words % 54 == 0; k_rho from
    /// the caller's policy-checked Params (table bits = k_rho + 1).
    pub fn new(matrices: &'a dyn MatrixRows, public_words: usize, k_rho: u32) -> Result<Self, Error> { unimplemented!() }
}

/// A CE(B) instance whose shapes match `Relation`. Only constructor validates: c is κ×54,
/// X is 54×public_blocks, r has the cube length, eval_k and eval_a[t] have 64 slots with
/// zero surplus, adv is None. fold_digest is ignored (non-authoritative).
pub struct Statement<'a> { claim: &'a CeClaim }
impl<'a> Statement<'a> { pub fn new(relation: &Relation<'_>, claim: &'a CeClaim) -> Result<Self, Error> { unimplemented!() } }

/// Opaque layer-1 proof; contains no witness coordinate.
#[derive(Clone)]
pub struct Proof {
    gkr_z: norm::FractionProof, gkr_table: norm::FractionProof,
    quotient: [field::Ext; 53], linear: sumcheck::Messages, pcs: pcs::Opening,
}
impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    /// Strict: canonical words, exact counts from `relation`, re-encode equality.
    pub fn from_bytes(relation: &Relation<'_>, bytes: &[u8]) -> Result<Self, Error> { unimplemented!() }
}

/// Continue `transcript` (layer-0 end state) and prove `statement` for `witness` (54 × n_R).
pub fn prove(relation: &Relation<'_>, statement: &Statement<'_>, witness: &neo_ccs::Mat<neo_math::F>,
             transcript: &mut neo_transcript::Poseidon2Transcript) -> Result<Proof, Error> {
    // TODO P1-P8 of §4.4, in this order, with no other transcript writer.
    unimplemented!()
}
pub fn verify(relation: &Relation<'_>, statement: &Statement<'_>, proof: &Proof,
              transcript: &mut neo_transcript::Poseidon2Transcript) -> Result<(), Error> { unimplemented!() }

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CE(B) shape: {0}")] Shape(&'static str),
    #[error("witness coordinate {0} is outside (-B, B)")] WitnessOutOfRange(usize),
    #[error("statement is false for this witness: {0}")] FalseStatement(&'static str),
    #[error("proof rejected: {0}")] Rejected(&'static str),
    #[error("proof bytes: {0}")] Codec(&'static str),
    #[error("WHIR: {0}")] Pcs(String),
}

// ===== src/ring.rs (pub(crate)) =====
/// The only list of ring rows; prover and verifier both iterate it (γ^ρ follows this order).
pub(crate) enum Row { Commitment(u32), EvalK(Part), EvalA(usize, Part), Public(usize) }
pub(crate) enum Part { Re, Im }
pub(crate) fn rows(relation: &Relation<'_>) -> impl Iterator<Item = Row> + '_ { unimplemented!() }
/// γ^ρ for ρ in `rows` order; built only from a challenge.
pub(crate) struct Mixing { powers: Vec<Ext> }
pub(crate) struct CombinedRow(Vec<[Ext; 54]>);                    // ḡ_b, prover only
pub(crate) fn combined_row(rel: &Relation<'_>, st: &Statement<'_>, mix: &Mixing) -> CombinedRow { unimplemented!() }
pub(crate) fn target_coefficients(st: &Statement<'_>, mix: &Mixing) -> [Ext; 54] { unimplemented!() } // Ȳ
/// Q = (Σ_b ḡ_b·z_b − Ȳ) / Φ81; FalseStatement if the remainder is nonzero.
pub(crate) fn quotient(row: &CombinedRow, z: &pcs::BlockTable, target: &[Ext; 54]) -> Result<[Ext; 53], Error> { unimplemented!() }
pub(crate) fn weight_table(row: &CombinedRow, zeta: Ext) -> Vec<Ext> { unimplemented!() }            // G_b = ḡ_b(ζ)
/// G~(s) without W-sized memory: key pass + χ pass + matrix forward pass + public part.
pub(crate) fn weight_at(rel: &Relation<'_>, st: &Statement<'_>, mix: &Mixing, zeta: Ext, s: &[Ext]) -> Ext { unimplemented!() }

// ===== src/norm.rs (pub(crate)) =====
pub(crate) fn multiplicities(z: &pcs::BlockTable, k_rho: u32) -> Vec<Gl> { unimplemented!() }
pub(crate) fn table_value_mle(s: &[Ext]) -> Ext { unimplemented!() }    // (1 − 2·s_top)·Σ 2^i s_i
pub(crate) struct FractionProof { root: [Ext; 2], layers: Vec<LayerProof> }
pub(crate) struct LayerProof { rounds: Vec<[Ext; 3]>, children: Vec<Ext> } // 4 values; 2 at a unit-numerator leaf layer
pub(crate) struct LeafClaim { pub point: Vec<Ext>, pub numerator: Ext, pub denominator: Ext }
pub(crate) enum Leaves<'a> { Witness { z: &'a pcs::BlockTable, beta: Ext }, Table { m: &'a [Gl], beta: Ext } }
pub(crate) fn prove_fraction_sum(leaves: Leaves<'_>, ch: &mut hash::Challenger) -> (FractionProof, LeafClaim) { unimplemented!() }
pub(crate) fn verify_fraction_sum(p: &FractionProof, vars: usize, unit_numerators: bool, ch: &mut hash::Challenger) -> Result<LeafClaim, Error> { unimplemented!() }

// ===== src/sumcheck.rs (pub(crate)) =====
pub(crate) trait Oracle { const DEGREE: usize; fn evals(&self) -> Vec<Ext>; fn fold(&mut self, r: Ext); }
pub(crate) struct Messages(Vec<Vec<Ext>>);
pub(crate) fn prove<O: Oracle>(o: &mut O, vars: usize, ch: &mut hash::Challenger) -> (Vec<Ext>, Messages) { unimplemented!() }
pub(crate) fn verify(m: &Messages, degree: usize, claim: Ext, ch: &mut hash::Challenger) -> Result<(Vec<Ext>, Ext), Error> { unimplemented!() }

// ===== src/pcs.rs, hash.rs, field.rs (pub(crate)) =====
pub(crate) type Gl = p3_goldilocks_v08::Goldilocks;
pub(crate) type Ext = p3_field_v08::extension::CubicTrinomialExtensionField<Gl>;
pub(crate) type Challenger = p3_challenger_v08::DuplexChallenger<Gl, Perm, 16, 12>;
pub(crate) const PROFILE: Profile = Profile { /* log_inv_rate, folding, security bits, pow, JohnsonBound */ };
pub(crate) fn layer1_challenger(tr: &mut neo_transcript::Poseidon2Transcript) -> Challenger { unimplemented!() }
pub(crate) struct BlockTable { /* view of committed z rows */ }
pub(crate) fn commit(z: BlockTable, m: Vec<Gl>, ch: &mut Challenger) -> Committed { unimplemented!() }
pub(crate) fn open(c: Committed, s_blocks: &[Ext], s_table: &[Ext], ch: &mut Challenger) -> Opening { unimplemented!() }
pub(crate) fn verify(root: &Root, o: &Opening, s_blocks: &[Ext], s_table: &[Ext], ch: &mut Challenger) -> Result<([Ext; 54], Ext), Error> { unimplemented!() }

// ===== crates/nightstream/src/lifecycle/finish.rs =====
//! Compression lifecycle. Owns FinalProof meaning: layer 0 (PiCCS + PiRLC, no PiDEC) on the
//! final running and fresh claims, then the layer-1 CE(B) argument. Not the layer-1 protocol.
#[derive(Clone)] // equality and Debug go through to_bytes(); p3 proof types implement neither
pub struct FinalProof { state: Stage1State, body: FinalBody }
enum FinalBody { Initial, Active(Box<ActiveFinal>) }
struct ActiveFinal {
    running: Vec<CeClaim>,         // 16 semantic claims: the state-hash preimage
    fresh: CcsClaim,               // latest (c, x)
    ccs: pi_ccs::Proof,            // 28 rounds + 17 outputs; codec stores only evals and r'
    spartan: neo_spartan::Proof,   // layer 1
}
impl FinalProof { pub fn state(&self) -> &Stage1State { &self.state } pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() } }
impl PreparedLifecycle {
    pub(crate) fn finish_with_spartan(&self, proof: &Stage1Envelope) -> Result<FinalProof, FinishError> { unimplemented!() }
    pub(crate) fn verify_final(&self, expected: &Stage1State, proof: &FinalProof) -> Result<(), VerifyError> { unimplemented!() }
}
```

### 3.5 p3 0.8 isolation (owner update)

The choice is to isolate p3 0.8 in `neo-spartan` with renamed dependencies, for example
`p3-field-v08 = { package = "p3-field", version = "=0.8.0" }`. The same pattern covers
goldilocks, whir (feature `parallel`), challenger, commit, merkle-tree, symmetric, poseidon2,
dft, matrix, multilinear-util and sumcheck. The workspace stays on 0.5.3. Reasons:

1. 117 workspace files import p3 0.5.3. An upgrade changes field and Poseidon2 APIs across
   all of them, including the Metal glue.
2. An upgrade risks the protocol bytes. Today's Poseidon2 constants come from
   `new_from_rng_128(ChaCha8Rng(SEED))` under p3 0.5.3. The Lean-exported F′ circuit and
   every fixture pin them.
3. With isolation, only one crate can name p3 0.8 types. `nightstream` has no p3 0.8
   dependency, so leakage fails to compile.

The cost is two copies of p3-field and p3-goldilocks in the build, plus one copy of z
(453 MB, done once as part of the transpose). WHIR's Poseidon2 is built in p3 0.8 types from
`neo_ccs::crypto::poseidon2_goldilocks::round_constants()`, the single source. A parity test
pins it. A later workspace upgrade deletes the renames in one manifest and four files.

### 3.6 Invariants, validation, interface depth

- Boundaries validate. `Relation::new`, `Statement::new`, `Proof::from_bytes`,
  `BlockTable::from_witness` (shape and range; this type means "range-checked witness"),
  `check_terminal_statement`, and `decode_final_proof`. Code inside trusts the types.
- One source for each fact. B and k_rho come from the caller's `Params` (neo-params policy).
  κ and the seed come from neo-ajtai. Φ81 and `bar` come from neo-math. The Poseidon2
  constants come from neo-ccs. The ring-row order is `ring::rows`. The WHIR profile is
  `pcs::PROFILE`, and the bridge absorbs it, so a parameter change changes the transcript
  automatically and no list needs a manual sync.
- Depth. neo-spartan exposes six items that hide about 2,000 lines (GKR, two sum-check
  kinds, WHIR, the transcript bridge, ring algebra). nightstream adds three methods and one
  type.
- Out of scope: zero knowledge, Metal acceleration of layer 1, PaperExact/Crosscheck
  finishing (these return `EngineError::Unavailable`; there is no silent substitution).

## 4. Protocol

### 4.1 Fields

- Layer 0 is unchanged: K = F[u]/(u² − 7).
- Layer 1: every challenge and every WHIR extension element is in
  EF3 = F[x]/(x³ − x − 1) ≈ 2^192 (p3 `CubicTrinomialExtensionField`).
- K values (χ_r(·), y, y_j) enter layer 1 only through the F-linear projection
  π_ρ(v) = γ^ρ·Re(v) + γ^(ρ+1)·Im(v). Re and Im each get their own power, so an error in
  one part cannot cancel an error in the other. Because z ∈ F, `y = Σ g_b z_b` is
  equivalent to `Re y = Σ Re(g_b) z_b ∧ Im y = Σ Im(g_b) z_b`.
- v1 never multiplies K by EF3. The compositum F_{p^6} is needed only in m2 (§4.9).

### 4.2 Layer 0 (existing code; prover P0, verifier V0)

- P0.1 `prepared_running`: checked_prior_state (recomputes the state hash, requires
  fresh.x = encHash), then prepare_running, then validate_running_parent_authority.
- P0.2 `tr = Transcript::session()`. Run Pi_CCS: reset → "Nightstream/SuperNeo/fold/v2" →
  statement → α, γ → 28 rounds → 17 outputs' evals. Metal uses
  `optimized_prove_with_matrix_rows(Some(device))`; CPU uses `None`.
- P0.3 `pi_rlc::prove_refs` on the CPU with the 17 CPU witnesses. It returns the parent
  claim and the dense parent `Mat<F>`. Metal's resident parent is not needed.
- V0.1 `check_terminal_statement`: shapes, shared r, canonical children, state preimage,
  Poseidon2 state hash, fresh.x. Then prepare_running with the recomputed digest.
- V0.2 `pi_ccs::verify` returns 17 outputs. V0.3 `pi_rlc::derive_parent` samples ρ and runs
  `rlc_public`. The parent is never carried on the wire.

Layer 0 runs the exact prefix of a normal fold. Its challenges equal those of an `extend`
from the same state. This is harmless: both protocols give the same meaning to the same
transcript prefix, and the continuations differ (Pi_DEC absorbs nothing; the next step
resets).

### 4.3 Bridge and transcript schedule

`tr` is the neo-transcript Poseidon2 duplex. `ch` is a p3 0.8 `DuplexChallenger<Gl, Perm, 16, 12>`
over the same permutation.

| # | Action | Bound items |
|---|---|---|
| B0 | `tr.absorb_v1_1(domain_chunk_v1_1(b"Nightstream/SuperNeo/compress/v1"))`; `tr.absorb_v1_1(PROFILE words)` | layer-1 domain, rate, folding, security bits, PoW, assumption, EF degree, table bits |
| B1 | `seed = tr.squeeze_digest_v1_1()` (4 words); `ch = Challenger::new(perm)`; `ch.observe_slice(seed)` | all of layer 0 (statement, rounds, outputs, ρ) |
| L1 | WHIR commit of the stacked {z: 2^20×54, m: 2^17×1} → `ch.observe(root)` | the witness |
| L2 | β ← EF3 | |
| L3 | observe (P_z, Q_z, P_t, Q_t) | logUp roots |
| L4 | z-side GKR, k = 1..26: λ_k; k−1 rounds {observe h(0), h(2), h(3); r}; observe children (q0, q1 at k = 26); μ_k | |
| L5 | table-side GKR, k = 1..17: same, with four children at each layer | |
| L6 | γ ← EF3 (row mixing γ^ρ, ρ < κ + 2(1+t) + n_R,in = 37) | |
| L7 | observe Q[0..53] | quotient |
| L8 | ζ ← EF3; θ ← EF3 | |
| L9 | 20 linear rounds {observe h(0), h(2); r} | |
| L10 | WHIR `open_at`/`verify_at` at points [s (z, 54 columns), s_t (m, 1 column)]. p3 observes the 55 evaluations, then draws OOD samples, folding, PoW and queries. | |

One owner for each segment: nightstream owns `tr` through B0–B1, called from inside
`neo_spartan`. `neo_spartan::lib.rs` alone owns `ch`.

### 4.4 Layer 1 prover

- P1 `BlockTable::from_witness`. Transpose to block-major order and check |z| < B for every
  coordinate (`WitnessOutOfRange` otherwise). Compute the counts m: index j = (sign, magnitude);
  all zeros, including padding cells, go to +0. Then run B0–B1 and L1.
- P2 L2–L5. Build both GKR trees. The z side has 2^26 leaves (q = β − z64(64b + l); lanes
  54..63 and rows ≥ n_R give q = β; p = 1). The table side has 2^17 leaves (p = m_j,
  q = β − t_j). Send the roots. Run the layer sum-checks. Child index = y + 2^(k−1)·c, so
  each new coordinate is the top bit, and the leaf point s_z = (s_z,lo ∈ EF3^6, s_z,hi ∈ EF3^20)
  uses our low-bit-first order. Free the trees.
- P3 L6. Build u(c) = π_K(χ_r(c)) + Σ_j Σ_i M_j[i,c]·π_j(χ_r(i)) by a column-band
  transposed scatter over the geometric runs. Build
  ḡ_b = Σ_i γ^(ρ_i)·a_{i,b} + Emb(bar(u_b)) + [b < n_R,in]·γ^(ρ_x,b) (key part from
  `coefficient_block`, block-parallel). Free u.
- P4 L7. Ȳ = Σ_ρ γ^ρ·y_ρ, with y_ρ ∈ {c_i, π-projected Eval_K/Eval_A coefficients, X_b}.
  Q = (Σ_b ḡ_b·z_b − Ȳ)/Φ81. This uses only the high half of each block product
  (≈1,431 EF3×F MACs per block). A nonzero remainder gives `FalseStatement`.
- P5 L8–L9. Tables G_b = ḡ_b(ζ), Zζ(b) = z_b(ζ), E_b = eq(s_z,hi, b),
  Ze(b) = Σ_{l<54} eq(s_z,lo, l)·z_{b,l}. Prove
  Σ_b [G_b·Zζ(b) + θ·E_b·Ze(b)] = Ȳ(ζ) + Q(ζ)·Φ(ζ) + θ·(β − q*_z) over 20 variables
  (degree 2). This gives s.
- P6 L10. `open_at` gives the PCS proof with the 54 lane values at s and m~ at s_t.

### 4.5 Layer 1 verifier

- V1 `Statement::new`, `Proof::from_bytes`, then B0–B1, then `ch.observe(root)`.
- V2 Check P_z·Q_t = P_t·Q_z, Q_z ≠ 0 and Q_t ≠ 0. Verify the z-side GKR with the leaf
  numerators fixed to 1. This gives (s_z, q*_z). Verify the table-side GKR. This gives
  (s_t, p*_t, q*_t). Require q*_t = β − t~(s_t), with t~(s) = (1 − 2s_16)·Σ_{i<16} 2^i·s_i.
- V3 γ; observe Q; ζ; θ. Compute
  V = Ȳ(ζ) + Q(ζ)·Φ(ζ) + θ·(β − q*_z), where
  Ȳ(ζ) = Σ_i γ^(ρ_i)·c_i(ζ) + Σ_k ζ^k·[π_K(y_k) + Σ_j π_j(y_{j,k})] + Σ_b γ^(ρ_x,b)·X_b(ζ).
- V4 Verify the linear sum-check from V. This gives (s, F_final).
- V5 `verify_at` gives the lane values e_l and the m value e_m.
- V6 Zζ = Σ_l ζ^l·e_l and Ze = Σ_{l<54} eq(s_z,lo, l)·e_l. Require
  F_final = G~(s)·Zζ + θ·eq(s_z,hi, s)·Ze and e_m = p*_t.

### 4.6 The ring identity (conjuncts 1, 2, 4, 5)

Each conjunct is a set of ring equations Σ_b g_{ρ,b}·z_b ≡ y_ρ (mod Φ81) with g_{ρ,b} ∈ R_F:

- κ key rows: g = a_{i,b}, y = c_i.
- Re and Im of Eval_K: g = Emb(bar(Re/Im χ_r(54b + ·))), by Theorem 10 and Lemma 6.
- Re and Im of each Eval_A_j: g = Emb(bar(Re/Im (M_jᵀχ_r)_b)).
- n_R,in unit rows: g = [b = b′], y = X_{b′}. This is conjunct 2.

Mixing with γ^ρ gives one row ḡ_b ∈ EF3[X]/Φ81. The prover sends Q of degree ≤ 52. The
check Σ_b ḡ_b(X)·z_b(X) = Ȳ(X) + Q(X)·Φ81(X) holds as polynomials of degree ≤ 106. At ζ it
becomes the rank-one F-linear claim Σ_{b,l} z_{b,l}·(G_b·ζ^l) = Ȳ(ζ) + Q(ζ)·Φ(ζ). The lanes
pre-combine into Zζ(b) = z_b(ζ), so the sum-check has 20 variables and needs one point.
Every coefficient of every ring value is bound, not only the constant term. Eval rows
evaluate as ⟨u_b, ω⟩ with ω = bar(ζ⃗), because bar is symmetric.

### 4.7 Norm (conjunct 3)

S = {(1 − 2σ)·μ : σ ∈ {0,1}, μ < 2^k_rho} = (−B, B), with 0 listed twice. It has 2^17
entries and its MLE has a closed form. logUp: Σ_x 1/(β − z64(x)) = Σ_j m_j/(β − t_j). The
leaves cover all 2^26 virtual cells. Padding cells are zero, which lies in S. The m values
may be any elements of F: a z value outside S gives a pole on the left side that the right
side lacks. This needs only count < p, and the count is at most 2^26. The proof is exactly
‖z‖∞ < B, with no new constant. The honest bound 3,672 < 2^12 is not used.

### 4.8 The 54-lane packing and the 28-variable cube

- The committed z is 54 columns × 2^20 rows. GKR uses the virtual z64(64b + l), which is 0
  for l ≥ 54. Its MLE is Σ_{l<54} eq(s_lo, l)·col_l~(s_hi), so the same 54 openings serve
  both uses.
- The block rows n_R..2^20 carry weight 0 in every linear row: the key, χ over the Pad
  rows, and the matrix columns end at W. They are range-checked and otherwise inert. The
  extracted witness is rows < n_R, lanes < 54. This includes the 26 tail lanes of the
  carrier, so all W coordinates are covered.
- χ_r(c) for c < W and χ_r(i) for i < n are computed as low-table × high-table × Π(1 − r_t)
  over the unused top coordinates of r ∈ K^28. The code is generic in every dimension. It
  hard-codes no 814,144 / 28 / 4 / 270.
- p3 `Point` is big-endian. `pcs.rs` reverses our low-bit-first points in one place, and a
  test pins the reversal.

### 4.9 Final weight evaluation G~(s): v1, and the m2 path

G~(s) = Σ_b eq(s, b)·G_b is the sum of four streaming terms:

| Term | v1 computation | m2 replacement |
|---|---|---|
| Key: Σ_{i,b} γ^(ρ_i)·eq(s,b)·a_{i,b}(ζ) | SHAKE pass over 22 × n_R blocks, block-parallel | One point opening of the setup-committed key MLE. γ^i and ζ^k are scaled eq points over 5 and 6 bits; eq(s, ·) covers 20 bits. |
| Eval_K: Σ_c π_K(χ_r(c))·eq(s, ⌊c/54⌋)·ω_{c mod 54} | O(W) | A polylog carry DP for the 54b + l index map, with an F_{p^6} = EF3[u]/(u² − 7) pair helper of about 20 lines |
| Eval_A: Σ_j Σ_i π_j(χ_r(i))·Σ_c M_j[i,c]·eq(s, ⌊c/54⌋)·ω_{c mod 54} | Forward pass over matrix runs, row-parallel | A sparse-matrix (SPARK-style) setup commitment opened at (r, eq(s) ⊗ ω) |
| Public rows: Σ_{b<n_R,in} γ^(ρ_x,b)·eq(s, b) | O(n_R,in) | unchanged |

The sum-checks, the transcript and the public API stay the same in m2. Only `Relation`
gains setup commitments, and `ring::weight_at` switches from streaming to openings. In m3,
every remaining verifier operation is EF3/K arithmetic plus Poseidon2; SHAKE leaves the
verifier in m2. So the shrink circuit runs `verify_final` unchanged.

## 5. Soundness map

L = WHIR list size. p3 draws the initial OOD samples after our challenges, so every outer
error is multiplied by L. In the Johnson regime at rate 1/2, `list_size_bits` ≈ 4.3. |EF3| ≈ 2^192.

| Step | Authority | Error |
|---|---|---|
| Layer 0: Pi_CCS strong (Lemma 7) ∘ Pi_RLC weak (Lemma 8), Theorem 12 → Lemma 1 | existing profile census | ≈ 2^−114.4 (Π_CCS γ-mixing dominates) |
| Bridge digest and Merkle binding (4-word Poseidon2) | collision resistance | 2^−128 (classical) |
| WHIR commitment and opening, Johnson bound (proven; BCSS25 proximity gaps) | p3 `prescribed_security`, test-pinned | ≥ 117 bits required (derivation below) |
| logUp, β | Haböck logUp; Schwartz–Zippel on the cleared rational, degree ≤ 2^26 + 2^17 | 2^26.0·L/2^192 ≈ 2^−161 |
| GKR layers (26 and 17), λ_k, μ_k | Papini–Haböck logUp-GKR; sum-check | ≈ (3·461 + 3·43)·L/2^192 ≈ 2^−176 |
| Row mixing γ (36 powers) | Schwartz–Zippel in γ | 36·L/2^192 |
| Quotient at ζ (degree ≤ 106) | Schwartz–Zippel | 106·L/2^192 |
| θ batching; linear sum-check (20 rounds, degree 2) | sum-check | 41·L/2^192 |

Derivation of the WHIR target: we need 2^−114.4 + ε₁ ≤ 2^−114, so ε₁ ≤ 2^−116.05. The
algebraic terms are below 2^−160 and the hash term is 2^−128. A WHIR composite of
≥ 117 bits gives a total of 2^−114.2. The `security_level` and PoW bits in `PROFILE` must
reach that, and a test asserts it.

Extraction: the WHIR extractor gives the stacked polynomial, so the z columns and m. logUp
and GKR give every cell of z64 in S, so ‖z‖∞ < B. The ring identity gives all 37 rows, so
conjuncts 1, 2, 4 and 5 hold. Hence z is a CE(B) witness for the parent. Lemma 1 composed
with this AoK (Lemma 4) extracts CCS(b)¹ × CE(b)¹⁶. The state-hash recomputation binds them
to `State`. (2B, C)-relaxed binding (Def. 22) is used only inside Lemma 1's extractor, and
its premise ‖z‖ < B is what layer 1 proves.

The model is the same as today's IVC: Fiat–Shamir of a round-by-round sound protocol in the
ROM, with Poseidon2 as the random oracle. EF3 against F_{p^5}: every algebraic term already
has more than 40 bits of margin. WHIR's limiting terms are query terms, set by
`security_level` and independent of |EF|. The cubic field is about 2.5× cheaper than the
quintic field in the GKR and the sum-checks, and 40% smaller per element. Check that settles
the choice: p3 `prescribed_security` at 2^26 with EF3 must report ≥ 117 bits. If it does
not, change the `Ext` alias to `QuinticTrinomialExtensionField` (one line).

## 6. Costs (estimates)

| Item | Estimate |
|---|---|
| Statement (16 running claims, fresh c and x, State) | 275 KB |
| Layer 0 (28 × 9 K rounds, r′, 17 × 270 K evals) | 78 KB |
| Layer 1 sum-check data (GKR ≈ 1,557 EF3, Q 53, linear 40, openings 55) | 41 KB |
| WHIR proximity proof (2^26, rate 1/2, JB, about 117 bits) | about 300 KB (scaled from the WHIR paper's 317 KiB at 128 bits) |
| **Final proof** | **about 0.7 MB (today: 221.7 MB)** |
| Prover, layer 0 | about one fold's C + R. CPU 25–40 s; Metal C 8–12 s; CPU RLC 2–3 s. |
| Prover, layer 1 (CPU, 10 cores) | key SHAKE pass 3–6 s; u scatter 2–10 s; GKR 2–4 s; Q and G < 1 s; WHIR 5–10 s. Total 15–30 s. Peak memory about 8–12 GB on top of today's extend. |
| Verifier v1 (CPU) | SHAKE-bound key pass 3–6 s; matrix pass 1–3 s; χ pass < 1 s; GKR, sum-check, WHIR and Pi_CCS verify ≪ 1 s. Total 5–10 s; about 1 GB. |
| m2 | Verifier: three passes become openings (milliseconds plus WHIR verification). Proof: + key and matrix opening proofs. Prover: + those openings. |
| m3 | The shrink circuit proves `verify_final`: about 600 EF3 sum-check rounds, Pi_CCS in K, rlc_public, the state hash, and WHIR paths. The target of ≤ 150 KB is met in the shrink layer, not here. |

## 7. Test plan (every invocation ≤ 300 s, `--release`, `FoldingMode::Optimized`)

Toy relations are built in neo-spartan tests. They use neo-reductions'
`SuperneoEvalCacheBuilder` + `CachedMatrixRows` (the same pattern as `parity.rs`), the real
production key prefix (2 blocks), and witnesses with |z| ≤ 3,672. The statement comes from
the repository's own code (`eval_real_v1_1_openings`, ring products with `coefficient_block`).
Tests that need internal state attach through `#[cfg(test)] #[path = "../tests/internal/…"]`.

1. **Formula pins.** (a) Key ring rows equal `commit_production_signed_unit_matrix` on a
   ternary witness. This pins the column-major c layout. (b) The ring-row Re/Im Eval_K and
   Eval_A equal `eval_real_v1_1_openings` at a random r. (c) The quotient identity holds on
   random ḡ and z. (d) `weight_at` streaming equals brute force Σ_b eq(s,b)·G_b. (e) t~ and
   the fraction-sum root equal brute force. (f) p3 0.8 Poseidon2 equals the transcript
   permutation on random states. (g) A WHIR opening at our point equals a direct MLE
   evaluation (point order). (h) `prescribed_security` ≥ 117 bits for the production 2^26
   shape. This is config only, with no commit.
2. **Toy end to end (nightstream).** Attach `tests/compression/toy.rs` as a child of
   `parity.rs` so it reuses `Fixture` unchanged. For `bit()` (t = 1) and
   `selected_polynomial()` (t = 4): Pi_CCS + Pi_RLC → `neo_spartan::prove`/`verify`, the
   verifier replay gives an equal transcript, proving twice gives equal bytes, and bytes
   round-trip.
3. **Mutation tests.** Each must reject.
   - z coefficient: tamper a private-block lane, a carrier-tail lane, and a public-block lane.
   - quotient: Q[0] and Q[52].
   - eval coefficient: eval_k[0] Re, eval_k[53] Im, eval_a[t−1][27]. A nonzero surplus
     eval_k[54] must fail `Statement::new`.
   - commitment: c(row 0, coefficient 0) and c(row κ−1, coefficient 53).
   - x: X(53, 0).
   - r: r[last].
   - norm: one entry exactly +B and one exactly −B, with a consistent statement and m forced
     by an internal path. Both reject. The ±(B−1) entries must accept.
   - m: move one count between two table entries.
   - transcript values: a z-GKR round value at layer 26, the root P_t, one linear-round
     value, one opened lane value, one byte of the WHIR proof, and a different layer-0 end
     state (one extra absorbed word).
   - FinalProof (crate-private): a running eval, fresh x, a Pi_CCS round, a Pi_CCS output eval.
4. **Production (`#[ignore]`, each run ≤ 300 s).** New staged steps follow the existing
   `staged_fold.rs::rlc`. `spartan_prove` reads parent.json and parent-witness.json, proves,
   and saves (est. 15–40 s). `spartan_verify` reads and verifies (est. 5–10 s).
   `finish_from_saved_package` loads a saved package and envelope, then runs finish, verify
   and encode/decode (est. 60–90 s). Compilation stays outside these tests.

## 8. Answers to GROUNDING §5

1. **Placement.** The new crate `neo-spartan` owns layer 1 and all p3 0.8 dependencies.
   nightstream owns layer 0 (existing folding code), `FinalProof`, its codec and the lifecycle
   entries. It exposes `finish_with_spartan`, `verify(&State, &FinalProof)` and
   `decode_final_proof`.
2. **Field.** EF3 (cubic trinomial, about 2^192) for all layer-1 challenges and WHIR. K
   values enter only through Re/Im F-linear projections with separate γ powers. F_{p^5} is
   the fallback alias, and the GROUNDING's F_{p^10} is not needed in v1.
3. **What is committed.** The parent z as 54 lane columns × 2^20 rows, plus the 2^17 logUp
   multiplicity column, in one stacked 2^26 WHIR commitment. Digit planes are 16× the data.
   A lane-padded single column forces 2^27.
4. **Norm.** logUp against S = (−B, B) (sign-magnitude, 2^(k_rho+1) entries, closed-form
   MLE), proven with logUp-GKR so that no helper is committed. The proof is exactly
   ‖z‖∞ < B; 2^12 is not used.
5. **C = Az.** The 22 key rows, Re/Im Eval rows and x rows mix into one ring row. One
   quotient Q (degree ≤ 52) is sent before ζ. The identity is checked at ζ ∈ EF3. The
   verifier's weight MLE G~(s) is streamed in v1 (§4.9) and becomes openings in m2.
6. **Batching.** 37 ring rows → one ring identity → one rank-one F-linear claim. The GKR
   leaf claim on z is batched in with θ. One 20-round degree-2 sum-check and one opening
   point for z. m has its own GKR leaf point.
7. **Transcript.** Layer 0 is the unchanged fold prefix (Pi_CCS reset included). A
   domain-separated bridge then runs: the "compress/v1" chunk and the PROFILE words are
   absorbed into the layer-0 transcript, a 4-word squeeze is taken, and a p3 DuplexChallenger
   over the same Poseidon2 permutation is seeded with it. That challenger runs all of
   layer 1 including WHIR. The schedule is in §4.3.
8. **Proof object.** `FinalProof {state, Initial | Active{16 running, fresh (c,x), Pi_CCS
   rounds + r′ + 17 eval families, neo_spartan::Proof}}`. Codec magic is
   `NS-FINAL-PROOF01`. Bytes are strict and canonical; the layer-1 part is checked by
   re-encoding. The verifier entry is `Verifier::verify(&State, &FinalProof)`.
9. **Backends.** Layer 0: Pi_CCS on the backend's row engine (CPU or Metal device), Pi_RLC
   on the CPU for all backends. Layer 1: CPU only. PaperExact and Crosscheck finish returns
   `Unavailable` in v1. The verifier runs on the CPU.
10. **Tests.** See §7: formula pins against the repository code, toy end to end on the
    parity fixtures, a mutation for each component, and staged production steps under the
    cap.

## 9. Tradeoffs, alternatives, questions, next step

### Synthesis decision

(Filled in by arena.)

### Tradeoffs accepted

- We accept the logUp-GKR code (about 380 lines, 461 short sum-checks, about 37 KB of
  proof) in exchange for committing only 2^26 cells: 1× z data instead of 16×.
- We accept one 53-element quotient message and a prover pass over the block products in
  exchange for rank-one weights. This gives a 20-variable sum-check, 100 MB of tables
  instead of 1.6 GB, and an m2 key opening that is one point evaluation.
- We accept the prover's transposed matrix scatter (new code) in exchange for a verifier
  that only runs forward passes and keeps no W-sized state.
- We accept two p3 versions in the build in exchange for unchanged folding bytes and
  pinned constants.
- We accept that the final proof carries the 275 KB statement (16 claims). The shrink layer
  absorbs it in m3.
- We accept that layer 1 is not zero-knowledge (the opened lane values and Q leak linear
  information about z). The owner has not asked for ZK.

### Alternatives considered

- **Commit the 16 signed digit planes** (seat B). The public surface is the same. Inside,
  the code is simpler: a degree-3 zero-check replaces GKR, and the bound Σ 2^i = B − 1 is
  exact. But the commitment holds 864 columns, stacked to 2^30. That is 16× the WHIR commit
  work and a 2^31 codeword (about 17 GB) on top of 17 resident witnesses, which is close to
  the 64 GB cap. The 2^30 zero-check in EF3 also costs more than the whole GKR. It loses on
  cost, not on interface.
- **Commit z, logUp with a committed inverse helper** (one form of seat A). It is one
  sum-check and needs no GKR. But h = 1/(β − z) depends on β. It needs a second WHIR
  commitment of 3 × 2^26 base cells and a second opening proof (+300 KB). It loses on
  prover cost and proof size. The interface is the same.
- **Exact reduction without a quotient.** Weight w(b,l) = (ḡ_b·X^l mod Φ)(ζ). It sends no
  message, but the weight is not rank-one: the sum-check has 26 variables over a 1.6 GB
  weight table, and the m2 key opening needs an extra 6-round sub-sum-check. It loses on m2
  depth.
- **Layer 1 inside nightstream (no new crate).** p3 0.8 types would be importable from any
  nightstream module, and the shortest path that compiles would leak them. It loses on
  boundary.
- **Workspace upgrade to p3 0.8.** See §3.5. It loses on blast radius and on pinned bytes.

### Open questions and risks (for the owner)

- Q-A. Should the public `Verifier::verify` accept only `FinalProof`? The envelope check
  that needs witnesses would become a crate-private test oracle. One public verify avoids
  two ways to check the same statement.
- Q-B. The terminal check "fresh completion tail is zero" has no counterpart in CE(B) or in
  Lemma 1. The final proof cannot show it after RLC. Is it a required semantic? If it is,
  should the circuit enforce it as CCS rows?
- Q-C. Is logUp-GKR (Papini–Haböck, ePrint 2023/1284) acceptable under "published parts
  only"? It is a published sum-check protocol, but it is not Spartan proper.
- Q-D. Which WHIR profile? Rate 1/2 gives the fastest prover and about 300 KB. Rate 1/4 or
  1/8 gives fewer queries and a cheaper m3 shrink circuit, with 2–4× more commit work.
  Also choose the PoW bits.
- Q-E. Do you accept EF3 instead of F_{p^5}? The decision depends on the pinned
  `prescribed_security` result.
- Q-F. Do you accept that PaperExact and Crosscheck return `Unavailable` for finishing in v1?
- Q-G. Post-quantum hashing: 4-word Poseidon2 digests give 128-bit classical collision
  resistance and about 85 bits under BHT. The existing transcript and state hash make the
  same assumption. Accept, or open a system-wide digest-width change?
- Risk: the p3-whir 0.8 API details are not yet confirmed by compiling. These are two-table
  prescribed openings, `CanSampleUniformBits` for the Goldilocks `DuplexChallenger`, and
  the serde form of `PcsProof`. The spike in the next step confirms them.
- Risk: two implementations of the Eval semantics exist (forward in neo-reductions,
  transposed here). A mismatch would break completeness, not soundness. Tests 1(a) and 1(b)
  pin both.

### Red-flag screen

- No shallow module: neo-spartan has 6 public items over about 2,000 lines.
- No information leakage: p3 0.8 types are private, and the wire bytes are owned by codec
  modules.
- No temporal decomposition: modules follow the conjuncts (ring, norm) and resources
  (pcs, hash). The schedule is one function body in lib.rs.
- Pass-through: `Prover::finish_with_spartan` forwards to the lifecycle. This is the
  established facade, the same as `encode_proof`, and it adds error conversion.
- No split ownership: one owner for each transcript segment and each buffer.
- No two ways to do one task, if Q-A is accepted.
- No importable internals: crate privacy.
- No hand-synced lists: `ring::rows` and `pcs::PROFILE` are each a single list, and the
  bridge absorbs PROFILE.

### Next implementation step

Create `crates/neo-spartan` with the p3-whir 0.8 spike. Build our Poseidon2 as a p3 0.8
permutation and pin parity. Commit a toy stacked {z: 2×54 → 2^1×54, m: 2^17} and run
prescribed `open_at`/`verify_at` at two points. Pin `prescribed_security` ≥ 117 bits for
the production 2^26 shape.
