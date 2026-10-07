# Candidate A: one committed witness, logup-GKR norm, quotient-lifted ring claims

Milestone 1 of the compression step: layer 0 (Pi_CCS + Pi_RLC, no Pi_DEC) and
layer 1 v1 (an argument of knowledge for the one CE(B, L) parent claim). Seat
stance: commit the parent witness `z` as ONE base-field multilinear polynomial
and prove the norm with a lookup-style range argument. After analysis the stance
holds, in its strongest form: **the WHIR commitment to `z` is the only
commitment in the whole layer-1 proof.** Everything else is sum-checks, one
logup-GKR, a 53-coefficient quotient and a fixed public histogram.

Numbers marked "est." are derived, not measured.

## 1. Problem

The final proof today is the accumulator plus 17 witnesses (~221.7 MB). The
terminal verifier recommits every witness and streams every matrix row. We must
replace the witnesses with an argument, post-quantum, near 114 bits, from
published parts (Spartan-style sum-checks + Plonky3 WHIR).

Layer 0 is mechanical. `staged_fold.rs::{ccs, rlc}` already runs Pi_CCS then
Pi_RLC and replays the verifier. Lemma 1 says the composition is a reduction of
knowledge from CCS(b)^1 x CE(b)^16 to CE(B). Lemma 4 lets an argument of
knowledge for CE(B) close the chain. The hard part is layer 1. These facts make
its shape non-obvious:

- **Five conjuncts, no CCS gate.** CE(B, L) (Def. 21) is `c = L(z)` (22 ring
  rows over `F[X]/Phi81`), `x = L_in(z)` (5 ring entries), `Eval_K` (54
  K-coefficients), `Eval_A_j` (4 x 54) and `||z||_inf < B`. Without Pi_DEC,
  nothing else enforces the norm.
- **Phi81 = X^54 + X^27 + 1 has no root in K** (ord_81(p) = 27), so ring
  identities need a quotient.
- **54 lanes.** `c = 54*block + lane` does not factor through eq; W > 2^25 forces
  a 2^26 cube in any layout.
- **K is 128 bits.** With EF = K, JB proximity gaps keep 127 - 45 = 82 bits and a
  2^26-term logup keeps 102. The field must be larger, and p3 has no degree 4.
- **Two Plonky3 generations** (0.5.3 workspace, 0.8.0 WHIR) under a
  Poseidon2-only, exact-instance rule.
- **Later milestones.** Milestone 2 commits key and matrices, so the final
  weight needs a structure a setup commitment can serve. Milestone 3 proves this
  verifier in a static circuit, so the proof shape must be fixed.
- **Repo rules.** Lifecycle names, no p3 or wire types on public APIs, files
  under 1,500 lines, tests under 300 s, `b = 2, k_rho = 16, B = 2^16`, digests
  never authority.

## 2. Usage (caller's view)

README quickstart (new section after "Load the saved data"):

```rust
use nightstream::{Circuit, Engine, State, Verifier};

let package = Circuit::load("app.nsc")?;
let prover = package.prover(Engine::Metal, minimum_security_bits)?;

let mut proof = prover.prove(initial_state, &messages[0])?;
for message in &messages[1..] {
    proof = prover.extend(&proof, message)?;
}
// Fold the last fresh instance into the running instance and prove the
// parent claim. The result contains no witness (est. 0.6-0.8 MB).
let finished = prover.finish_with_spartan(&proof)?;
let bytes = finished.to_bytes();

// The verifier chooses its circuit independently of the proof.
let verifier = Verifier::from_package(&package, Engine::Optimized, minimum_security_bits)?;
let finished = verifier.decode_final_proof(&bytes)?;
verifier.verify(&expected_state, &finished)?;
```

`finish_with_spartan` does not change `proof`; `proof` can still be extended.
A finished proof cannot be extended: `FinalProof` has no witness, and
`Prover::extend` does not accept it, so this is a compile error.
`verify` accepts both `Proof` and `FinalProof` through a sealed trait;
existing `verifier.verify(&state, &proof)` call sites compile unchanged.

Call site 1, prover service (finishes once per chain):

```rust
fn publish(prover: &Prover, chain: &Proof, sink: &mut impl Write) -> Result<(), nightstream::Error> {
    let finished = prover.finish_with_spartan(chain)?;
    sink.write_all(&finished.to_bytes())?;
    Ok(())
}
```

Call site 2, verifier node (holds only the package and the expected state):

```rust
fn accept(verifier: &Verifier, state: &State, bytes: &[u8]) -> Result<(), nightstream::Error> {
    let finished = verifier.decode_final_proof(bytes)?; // exact length before allocation
    verifier.verify(state, &finished)                  // layer 0 replay + layer 1
}
```

## 3. Shape

### 3.1 Data structures

| Structure | Owner | Content | Size (production) |
| --- | --- | --- | --- |
| `FinalProof` | `nightstream::lifecycle::finish` | state, 16 running claims, fresh claim, Pi_CCS proof (rounds + output evals), `neo_spartan::Proof` | est. 0.6-0.8 MB |
| `neo_spartan::Relation<'a>` | `neo-spartan::relation` | key prefix, borrowed matrix rows, shape, circuit identity, WHIR config derived from the error budget | small; borrows rows |
| `Statement` (private) | `neo-spartan::relation` | validated view of the parent `CeClaim` | 2,054 words |
| `WitnessTable` (private) | `neo-spartan::layout` | `z` on the (block: 20 bits, lane: 6 bits) cube, `z[64b+l] = Z[l,b]`, zero elsewhere | 2^26 x 8 B = 512 MB |
| `BlockWeights` (private) | `neo-spartan::weights` | lambda-combined carrier weights `E` (len W) + key access; yields omega_b(X) per block | W x 24 B = 1.05 GB |
| `Histogram` (private) | `neo-spartan::norm` | m(t) for t = -H..=H, H = 3,672 | 7,345 x u32 = 29 KB |
| `neo_spartan::Proof` | `neo-spartan::protocol` | root, histogram, GKR, quotient `[Ext; 53]`, linear rounds, z(s), WHIR opening | est. 0.3-0.45 MB |

Dominant access patterns, traced through the table layout. The table puts the
lane in the 6 low index bits, so every pass is a contiguous stream per block:

1. WHIR encode and Merkle: sequential over 2^26 entries.
2. Histogram and GKR leaves: sequential; the first GKR merge pairs adjacent
   entries (lane bit 0).
3. Quotient: per block, 64 contiguous entries times omega_b (54 coefficients).
4. Linear sum-check: `u(b) = sum_l zeta^l z[64b+l]` per block (contiguous), then
   one fold over block variables to get `z(s_blk, l)` for 64 lanes.
5. Weights: the matrix pass scatters into `E` by carrier column (85M adds,
   single thread, est. 0.5 s); the block pass then reads 54 contiguous entries.

The 10 padding lanes per block cost nothing: `W > 2^25` forces a 2^26 cube in
any layout. So the (block, lane) layout is free, and it is the only layout
where the ring structure (`zeta^l`) and the block structure (`Omega(b)`)
factor. That factorization is what milestone 2 needs.

### 3.2 Module map

New crate `crates/neo-spartan` (owns every p3 0.8 type; all modules private):

| File | Owns | Lines (est.) |
| --- | --- | --- |
| `lib.rs` | crate contract header; `prove`, `verify`; re-exports `Relation`, `Proof`, `Error` | 70 |
| `relation.rs` | `Relation`, shape, `Statement` validation, error budget to WHIR config | 210 |
| `protocol.rs` | the layer-1 transcript schedule, prover and verifier side by side | 300 |
| `bridge.rs` | the only file that names both p3 generations: value maps, Poseidon2 0.8 instance from pinned constants, challenger seed, point order, bar on `Ext` | 220 |
| `layout.rs` | cube, `WitnessTable`, `CubePoint`, lane weight `sum_{l<54} zeta^l eq(s,l)` | 140 |
| `weights.rs` | claim slot list, lambda powers, `BlockWeights`, targets t(X) | 330 |
| `ring_lift.rs` | quotient `P div Phi81`, lifted value `t(zeta) + Phi81(zeta) Q(zeta)` | 140 |
| `norm.rs` | histogram, table sum `sum m(t)/(beta - t)` | 120 |
| `gkr.rs` | fractional-sum GKR (Papini-Habock), both sides | 480 |
| `linear.rs` | tensor-weight sum-check `sum Omega(b) zeta^l z(b,l)`, both sides | 230 |
| `pcs.rs` | WHIR types, config, commit, `open_at`, `verify_at` | 240 |
| `codec.rs` | exact-length proof bytes, canonical round trip | 300 |
| `error.rs` | `Error` | 50 |

Total est. 2,830 lines; largest file 480.

Changes elsewhere:

| File | Change | Lines (est.) |
| --- | --- | --- |
| `Cargo.toml` (workspace) | renamed p3 0.8 deps, member `neo-spartan` | +16 |
| `neo-ajtai/src/nightstream_fprime_setup.rs` | `KeyPrefix { seed, rows, blocks }` + `production(blocks)` + `block(row, b)` | +30 |
| `neo-reductions/src/common.rs` | `pi_rlc_parent_bound(params, count) = count*T*(b-1)`; the existing guard calls it | +15 |
| `nightstream/src/lifecycle/finish.rs` (new) | `FinalProof`, `FinishError`, `finish_with_spartan`, `verify_final`, `spartan_relation` | 290 |
| `nightstream/src/lifecycle/verify.rs` | extract `check_active_statement` (shared [S] checks) | +-40 |
| `nightstream/src/lifecycle/encoding.rs` | `FinalProof` codec (magic `NS-FINAL-PROOF01`) | +170 (to ~520) |
| `nightstream/src/folding/compose.rs` | `prove_parent_with_rows`, `verify_parent` (C then R) | +45 |
| `nightstream/src/folding/pi_rlc.rs` | `parent(...)`: sample rho, recompute via `rlc_public` | +20 |
| `nightstream/src/circuit.rs`, `lib.rs` | `finish_with_spartan`, `decode_final_proof`, sealed `Verifiable`, export `FinalProof` | +65 |

Flow traces stay within three files: `circuit.rs` -> `lifecycle/finish.rs` ->
`neo_spartan::protocol` (which calls one sub-protocol module per step).

### 3.3 p3 0.8: renamed dependencies, not a workspace upgrade

Decision: add `p3-whir = "=0.8.0"` and renamed 0.8 crates to
`[workspace.dependencies]` (`p3-field-08 = { package = "p3-field", version = "=0.8.0" }`,
same for goldilocks, challenger, commit, merkle-tree, symmetric, dft,
multilinear-util, sumcheck). Only `neo-spartan` uses them. `bridge.rs` is the
only file that names both generations; values cross as canonical `u64`.

Why not upgrade the workspace: every crate (neo-math's F and K, the Poseidon2
transcript, Metal and CUDA kernels, nightstream-fprime, golden vectors and
Lean-generated fixtures) is built on 0.5.3. An upgrade is a workspace-wide API
migration that can move pinned transcripts and digests, with no protocol gain
for this milestone. Isolation confines the risk to one crate. The bridge is
the migration seam: a later workspace upgrade deletes `bridge.rs` and nothing
else changes shape. Cost: two p3-field builds, and a 2^26-element `u64` map at
the boundary (est. 50 ms).

Hash instance: `bridge::permutation()` builds p3 0.8 `Poseidon2Goldilocks<16>`
from `neo_ccs::crypto::poseidon2_goldilocks::round_constants()` (the pinned
canonical constants that the CUDA backend already uses). This keeps the packed
0.8 Merkle hashing fast. A cross-version test pins equal outputs. If the 0.8
internal diagonal differs, the fallback is a newtype that wraps our 0.5.3
permutation (correct, slower Merkle).

### 3.4 Type sketch

```rust
// ---- nightstream (public) -------------------------------------------------
// lib.rs
pub use circuit::{Circuit, Error, Prover, Verifiable, Verifier};
pub use lifecycle::{FinalProof, Stage1Envelope as Proof, Stage1State as State};

// circuit.rs
impl Prover {
    /// Fold the last fresh instance into the running instance with PiCCS and
    /// PiRLC (no PiDEC), then prove the parent CE(B, L) claim with the
    /// Spartan/WHIR argument. Rejects an initial proof. `proof` is unchanged.
    pub fn finish_with_spartan(&self, proof: &Proof) -> Result<FinalProof, Error> {
        Ok(self.lifecycle.finish_with_spartan(proof)?)
    }
}
impl Verifier {
    /// The configured circuit fixes the exact length; checked before allocation.
    pub fn decode_final_proof(&self, bytes: &[u8]) -> Result<FinalProof, Error> { unimplemented!() }
    /// One verb for both proof kinds.
    pub fn verify<P: Verifiable>(&self, expected_state: &State, proof: &P) -> Result<(), Error> {
        proof.verify_with(&self.lifecycle, expected_state)
    }
}
/// Implemented only by `Proof` and `FinalProof`.
pub trait Verifiable: sealed::Verify {}
mod sealed {
    pub trait Verify {
        fn verify_with(&self, lifecycle: &crate::lifecycle::PreparedLifecycle, state: &crate::State)
            -> Result<(), crate::Error>;
    }
}

// lifecycle/finish.rs
/// A finished proof: the public statement, the layer-0 PiCCS messages and the
/// layer-1 argument. No witness. Carried fold digests are not encoded; the
/// verifier recomputes the state digest and sets them.
pub struct FinalProof {
    state: Stage1State,
    running: Vec<CeClaim>,        // the 16 semantic PiDEC children
    fresh: CcsClaim,
    pi_ccs: pi_ccs::Proof,        // outputs' c and X are copies of the inputs
    argument: neo_spartan::Proof,
}
impl FinalProof {
    pub fn state(&self) -> &Stage1State { &self.state }
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
}

/// Variants: Input, Prior(StepInputError), Reduction(folding::Error),
/// Argument(neo_spartan::Error), Engine(EngineError). VerifyError gains
/// Layer0(folding::Error) and Layer1(neo_spartan::Error).
pub enum FinishError { /* ... */ }

impl PreparedLifecycle {
    pub(crate) fn finish_with_spartan(&self, envelope: &Stage1Envelope) -> Result<FinalProof, FinishError> {
        // 1. Active only. (_, digest) = checked_prior_state(state, running, fresh.claim).
        // 2. prepare_running(&mut running, params, digest); validate_running_parent_authority.
        // 3. Engine: Optimized -> CPU; Metal -> PiCCS on the device through the existing
        //    optimized_prove_with_matrix_rows(.., Some(device)) hook, PiRLC on CPU;
        //    PaperExact | Crosscheck -> EngineError (no fallback, README rule).
        // 4. (pi_ccs, parent) = folding::prove_parent_with_rows(&mut tr, ...).
        // 5. argument = neo_spartan::prove(&self.spartan_relation(&rows)?, tr.into_inner(),
        //                                  &parent.claim, &parent.witness)?.
        unimplemented!()
    }
    pub(crate) fn verify_final(&self, expected: &Stage1State, proof: &FinalProof) -> Result<(), VerifyError> {
        // 1. digest = self.check_active_statement(expected, &proof.running, &proof.fresh)?
        //    (shared with the terminal verifier: shapes, canonical children, state hash).
        // 2. running = claims with fold_digest := digest bytes, evals padded (prepare_running).
        // 3. parent = folding::verify_parent(&mut tr, .., &proof.fresh, &running, &proof.pi_ccs)?
        // 4. neo_spartan::verify(&self.spartan_relation(&rows)?, tr.into_inner(), &parent, &proof.argument)
        unimplemented!()
    }
    /// The one constructor of the layer-1 relation, used by both sides.
    fn spartan_relation<'a>(&self, rows: &'a dyn MatrixRows) -> Result<neo_spartan::Relation<'a>, neo_spartan::Error> {
        // identity = binding.verifier_context().digest(); key = KeyPrefix::production(m.div_ceil(D));
        // public_blocks = m_in / D; point_variables = PI_CCS_V1_1_ROUND_COUNT;
        // max_abs = pi_rlc_parent_bound(params, PI_CCS_V1_1_SOURCE_COUNT);       // 3,672
        // budget  = -log2(2^-minimum_security_bits - 2^-params.fold_security_bits())
        unimplemented!()
    }
}

// folding/compose.rs
/// PiCCS then PiRLC: the layer-0 reduction to one CE(B) claim and its witness.
pub(crate) fn prove_parent_with_rows(tr: &mut Transcript, pp: &Params, s: &Structure,
    rows: &dyn MatrixRows, workspace_bytes: usize, fresh: CcsInstance, running: RunningInstance,
) -> Result<(pi_ccs::Proof, pi_rlc::Output), Error> { unimplemented!() }
/// Verifier replay; the parent is recomputed, never read from the proof.
pub(crate) fn verify_parent(tr: &mut Transcript, pp: &Params, s: &Structure, mix: RlcMixer,
    fresh: &CcsClaim, running: &RunningInstance, proof: &pi_ccs::Proof,
) -> Result<CeClaim, Error> { unimplemented!() }

// ---- neo-spartan (public surface: 2 functions, 3 types) -------------------
/// Argument of knowledge for one CE(B, L) claim (SuperNeo Def. 21) over the
/// complete carrier. One WHIR commitment to the witness; sum-checks and
/// logup-GKR over the cubic extension. Owns every Plonky3 0.8 type.
/// Does not own the claim's provenance: layer 0 produces and checks it.
pub fn prove(relation: &Relation<'_>, transcript: Poseidon2Transcript,
    claim: &CeClaim<Commitment, F, K>, witness: &Mat<F>) -> Result<Proof, Error> { unimplemented!() }
pub fn verify(relation: &Relation<'_>, transcript: Poseidon2Transcript,
    claim: &CeClaim<Commitment, F, K>, proof: &Proof) -> Result<(), Error> { unimplemented!() }

/// The fixed CE(B, L) relation for one circuit. The WHIR profile is derived
/// here from the error budget, so prover and verifier cannot disagree on it.
pub struct Relation<'a> {
    identity: [F; 4],             // circuit verifier-context digest, absorbed
    key: KeyPrefix,               // L: Ajtai key prefix (seed, 22 rows, n_R blocks)
    matrices: &'a dyn MatrixRows, // M_1..M_t over the complete carrier
    shape: Shape,                 // n_R, kappa, t, public blocks, point vars, H, cube vars
    pcs: pcs::Config,             // private p3 0.8 config
}
impl<'a> Relation<'a> {
    pub fn new(identity: [F; 4], key: KeyPrefix, matrices: &'a dyn MatrixRows, public_blocks: usize,
        point_variables: usize, max_abs: u32, error_budget_bits: f64) -> Result<Self, Error> { unimplemented!() }
    /// Composed layer-1 soundness error, -log2.
    pub fn security_bits(&self) -> f64 { unimplemented!() }
}

/// Opaque. Holds p3 0.8 values privately.
pub struct Proof {
    root: pcs::Root,
    histogram: norm::Histogram,       // exactly 2H + 1 counts
    gkr: gkr::GkrProof,               // fixed shape for the cube
    quotient: [Ext; D - 1],           // deg Q <= 52 is a soundness condition; the type enforces it
    linear: linear::LinearProof,      // one [Ext; 2] per cube variable
    linear_value: Ext,                // z(s)
    opening: pcs::Opening,            // WHIR PcsProof at [rho, s]
}
impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> { unimplemented!() }
    pub fn from_bytes(bytes: &[u8], relation: &Relation<'_>) -> Result<Self, Error> { unimplemented!() }
    pub fn encoded_len(relation: &Relation<'_>) -> usize { unimplemented!() }
}

/// Shape, Relation, Budget { reached, needed }, NormBound (prover only),
/// Rejected(&'static str) naming the failed check, Codec, Matrices(PiCcsError).
/// No p3 error type appears here.
pub enum Error { /* ... */ }

// ---- neo-spartan private core ---------------------------------------------
// bridge.rs
pub(crate) type Val = p3_goldilocks_08::Goldilocks;
pub(crate) type Ext = /* p3 0.8 cubic trinomial extension of Val, x^3 - x - 1 */;
pub(crate) type Challenger = p3_challenger_08::DuplexChallenger<Val, Perm, 16, 12>;
/// Squeeze a 4-word seed from the layer-0 transcript (absorb cursor must be 0),
/// start a duplex on the same Poseidon2 instance, observe domain chunk + seed.
pub(crate) fn challenger(transcript: Poseidon2Transcript) -> Challenger { unimplemented!() }
pub(crate) fn p3_point(point: &CubePoint) -> Point<Ext> { unimplemented!() } // ours: low bit first
pub(crate) fn bar(v: &[Ext; D]) -> [Ext; D] { unimplemented!() }             // superneo_bar_block per base coordinate

// weights.rs
/// The single ordered list of batched ring identities. lambda^position.
pub(crate) enum Slot { CommitmentRow(usize), Eval { matrix: Option<usize>, part: Part }, Public(usize) }
pub(crate) fn slots(shape: &Shape) -> impl Iterator<Item = Slot> { unimplemented!() }
pub(crate) struct BlockWeights<'r> { relation: &'r Relation<'r>, carrier: Vec<Ext>, powers: Vec<Ext> }
impl<'r> BlockWeights<'r> {
    /// One pass over matrix rows and chi_r over the carrier, lambda-combined.
    pub(crate) fn new(relation: &'r Relation<'r>, point: &[K], lambda: Ext) -> Result<Self, Error> { unimplemented!() }
    /// omega_b(X) = sum_i lambda^i a_{i,b}(X) + bar(E_b)(X) + [b < n_in] lambda^{X_b}.
    pub(crate) fn block(&self, b: usize) -> [Ext; D] { unimplemented!() }
}
pub(crate) fn targets(statement: &Statement, lambda: Ext) -> [Ext; D] { unimplemented!() }

// ring_lift.rs
pub(crate) fn quotient(weights: &[[Ext; D]], z: &WitnessTable) -> [Ext; D - 1] { unimplemented!() }
pub(crate) fn lifted_value(target: &[Ext; D], quotient: &[Ext; D - 1], zeta: Ext) -> Ext { unimplemented!() }

// norm.rs, gkr.rs, linear.rs, pcs.rs: one prove_* and one verify_* per file, e.g.
pub(crate) fn gkr::prove(z: &WitnessTable, beta: Ext, ch: &mut Challenger) -> (GkrProof, CubePoint, Ext);
pub(crate) fn gkr::verify(proof: &GkrProof, beta: Ext, root_sum: Ext, cube: &Cube, ch: &mut Challenger)
    -> Result<(CubePoint, Ext /* z(rho) = beta - q(rho) */), Error>;
```

### 3.5 Invariants in types; where validation lives

- No witness in a finished proof: `FinalProof` has no witness field.
- A finished proof cannot be extended: `extend` takes `&Proof` only.
- Layer 1 continues layer 0's transcript exactly once: `prove`/`verify` take the
  `Poseidon2Transcript` by value.
- `deg Q <= 52`: `[Ext; D - 1]`.
- Fixed histogram: `Histogram` is built only by `norm::histogram` or the
  decoder, both with exactly 2H + 1 entries.
- One batching list: `weights::slots` feeds lambda powers, the weights and the
  targets. Prover and verifier call the same `BlockWeights::block`.
- One WHIR profile: derived inside `Relation::new` and absorbed.
- No p3 type escapes `neo-spartan`: `Proof`'s fields are private; WHIR errors map
  to `Error::Rejected("WHIR opening")`.

Validation sits at boundaries only: `Verifier::decode_final_proof` (exact
length, canonical words); `check_active_statement` (shapes, canonical children,
state hash); the existing PiCCS/PiRLC validators; `Relation::new` (shape,
budget); `Statement::from_claim` (parent shape, zero eval padding, point
length); `Proof::from_bytes` (exact length, decode-then-re-encode equality).
Inside, functions take `&Statement`, `&WitnessTable` and trust them.

What the design deliberately does not do: zero knowledge; Metal kernels for
layer 1; a succinct verifier (milestone 2); proving the fresh witness's
completion tail is zero (see 9, Q4).

## 4. Protocol

Notation: F Goldilocks; K = F[u]/(u^2 - 7); **Ext = F[w]/(w^3 - w - 1)**,
|Ext| ~ 2^192 (p3 0.8 `CubicTrinomialExtendable`); D = 54; n_R = 814,144
blocks; cube = 20 block bits (high) + 6 lane bits (low) = 26 variables;
bar(v) = Emb(T^-1 v) (`superneo_bar_block`); H = (K+k)*T*(b-1) = 17*216*1 = 3,672.

### 4.1 Layer 0 (existing fold transcript, v1.1 operations)

Prover (`prove_parent_with_rows`):
1. `checked_prior_state` recomputes the state digest; `prepare_running` sets every
   running `fold_digest` to it and pads evals; parent authority is rebuilt and checked.
2. Pi_CCS (`pi_ccs::prove_from_parts_with_rows`): `reset_v1_1`; fold domain chunk
   `Nightstream/SuperNeo/fold/v2`; statement (prior digest 4, fresh C 1,188,
   fresh x 270); alpha, gamma; 28 rounds; absorb all 17 outputs' evals.
3. Pi_RLC (`pi_rlc::prove_refs`): 17 x (absorb [4, i], squeeze) -> rho_i;
   parent claim and dense parent `Z = sum rho_i Z_i` (54 x 814,144).
4. Stop. No Pi_DEC.

Verifier (`verify_parent`): same steps 1-3 as replay; `pi_ccs::verify` returns
17 outputs; `pi_rlc::parent` recomputes the parent with `rlc_public`. The proof
carries only Pi_CCS rounds, the output point, the output digest and the
outputs' evals; output c and X are copied from inputs by the decoder and checked
by the existing validators. The layer-0 transcript equals a normal fold's prefix.
A fold never continues the transcript after Pi_RLC, so the continuation below
is unambiguous.

### 4.2 The ring identities

All five conjuncts except the norm become 37 identities in R_F, each
`sum_b omega_{c,b} * z_b = target_c` with F-coefficient weights:

| Slots | Count | omega_{c,b}(X) | target |
| --- | --- | --- | --- |
| `CommitmentRow(i)` | 22 | key element a_{i,b} | parent c, ring row i |
| `Eval{K, Re/Im}` | 2 | bar(Re/Im e^0_b), e^0_b[l] = chi_r(54b + l) | Re/Im of y |
| `Eval{A_j, Re/Im}` | 2t = 8 | bar(Re/Im e^j_b), e^j = M_j^T chi_r | Re/Im of y_j |
| `Public(j)` | 5 | [b = j] * 1 | X_j |

K-valued claims split into Re and Im because bar and the matrices have F
entries and z is F-valued (Theorem 8, Theorem 10). The prover never computes in
K (x) Ext. chi_r uses all 28 coordinates of r (low bit first) through
`EqualityWeights`; the unused high bits enter as prod(1 - r_t) automatically.
Eval_K covers all W coordinates (Pad), including the nonzero tail lanes.

### 4.3 Layer 1 schedule (one p3 duplex on our Poseidon2 instance)

| # | Who | Message or coin | Content |
| --- | --- | --- | --- |
| T0 | both | v1.1 squeeze | seed = 4 words from the layer-0 transcript (cursor 0, after Pi_RLC) |
| T1 | both | new duplex | `DuplexChallenger<Val, Perm16, 16, 12>`; observe chunk `Nightstream/SuperNeo/compress/v1` (12 words), seed, profile words (identity 4, cube vars, n_R, kappa, t, n_in, point vars, H, WHIR rate, folding factor, assumption, security level, PoW bits) |
| T2 | both | observe | parent statement: c 1,188; X 270; r 56; Eval_K 108; Eval_A 432 |
| T3 | P | WHIR commit | z table (2^26 F, one column); root observed inside `commit` |
| T4 | P | observe | histogram m(-H..H), 7,345 words |
| T5 | V | sample | beta in Ext |
| T6 | P/V | GKR | layer 1: 4 values, check root; per layer k = 1..25: sample lambda_k, k rounds (3 Ext each, degree 3), 4 child values, sample tau; output rho, z(rho) := beta - q(rho) |
| T7 | V | sample | lambda in Ext (claim batching, powers) |
| T8 | P | observe | Q, 53 Ext coefficients |
| T9 | V | sample | zeta in Ext |
| T10 | P/V | linear sum-check | 26 rounds (h(0), h(2) in Ext; h(1) from the claim), sample s_k; observe z(s) |
| T11 | P/V | WHIR `open_at` | one table, two opening batches at [rho, s]; OOD, eval observes, rounds, PoW |

Prover steps (`protocol::prove`), verifier mirror in the same file:

1. `Statement::from_claim`; `ch = bridge::challenger(transcript)`; T1, T2.
2. `WitnessTable::from_packed(Z)`; `pcs::commit` (T3).
3. Histogram over all 2^26 slots (padding zeros count toward m(0)); a value
   with |z| > H returns `Error::NormBound` (honest provers never hit it). T4, T5.
4. GKR over leaves (p, q) = (1, beta - z[x]). Layer k: p_k(y) = p(y,0)q(y,1) +
   p(y,1)q(y,0), q_k(y) = q(y,0)q(y,1). Verifier at the root checks
   P_0 = R * Q_0 with R = sum_t m(t)/(beta - t) (one batch inversion of 7,345
   elements), and rejects Q_0 = 0 or beta = t. At the leaves p == 1 is
   substituted, and z(rho) := beta - q(rho). (T6)
5. T7. `BlockWeights::new` (one matrix pass, one chi_r pass, key rows on demand);
   cache omega_b for all blocks (prover only, 1.05 GB). P(X) = sum_b
   omega_b(X) z_b(X) (degree <= 106, per-block 54 x 54 convolution, parallel);
   Q = P div Phi81 (monic long division). No remainder check: a wrong witness
   still yields a proof, which the verifier rejects (mutation test 6 uses this). T8, T9.
6. Omega(b) = omega_b(zeta) for b < n_R, else 0. Claim v = t(zeta) + Phi81(zeta) Q(zeta).
7. Linear sum-check of g(x) = Omega~(x_blk) * Lambda~(x_lane) * z~(x), with
   Lambda(l) = zeta^l for l < 54 and 0 for 54 <= l < 64. Block rounds first, on
   u(b) = sum_l zeta^l z[64b+l] (valid because Lambda~ is summed over Boolean
   lanes); then lane rounds on z~(s_blk, l). Final check:
   last claim = Omega~(s_blk) * Lambda~(s_lane) * z(s). (T10)
8. `pcs::open` at [rho, s] with values [z(rho), z(s)]; the verifier also requires
   the opening's values to equal the two values it derived. (T11)

**Final weight evaluation in v1.** The verifier computes Omega in full: one
SHAKE expansion of the key prefix (22 x 814,144 elements), one pass over the
matrix runs, one chi_r pass over W, then Omega(b) = <omega_b, (1, zeta, ...,
zeta^53)> and one MLE fold of the 2^20 vector at s_blk. Lambda~(s_lane) costs 64
operations. The verifier needs no K (x) Ext arithmetic in v1.

**Why the lift is sound.** If some identity fails, the batched residual is a
nonzero polynomial in lambda of degree <= 36 in some X-coefficient. Given that,
for every Q with deg Q <= 52, P - t - Phi81*Q is nonzero in Ext[X] (its residue
mod Phi81 is nonzero) and has degree <= 106, so zeta is a root with probability
<= 106/|Ext| (Lemma 2). This is the existing `ProjectionTrace` identity
(|q| = 53, max degree 106), now batched.

### 4.4 What milestone 2 and 3 need from this shape

- Omega~(s_blk) = sum_i lambda^i sum_{l<54} zeta^l A~(i, s_blk, l) + matrix part +
  Pad part + public part. The key part is one small (row, lane) sum-check down
  to one opening of a committed key MLE. The matrix part is
  chi_r^T M_j g with g(54b + l') = eq(s_blk, b) * tau_{l'}, where
  tau = T^-1 (1, zeta, ..., zeta^53) (so bar(e)(zeta) = <e, tau>). g is succinct in
  the (block, lane) column index, so a Spartan/SPARK sparse-matrix evaluation
  serves it. Pad is a fifth structured matrix (W ones). Steps T0-T10 do not
  change; T11 gains openings of setup commitments after s is fixed.
- Every message has a fixed size (fixed histogram, fixed GKR shape, fixed WHIR
  config), so the milestone-3 shrink circuit is static. Its transcript is one
  p3 duplex seeded by 4 words, which is also a clean public-input boundary.

## 5. Soundness map

Errors are statistical, per the repo's 114-bit convention (FS query count not
charged). |Ext| ~ 2^191.9 (`Ext::bits() - 1 = 191` used by p3).

| Step | Error (log2) | Basis |
| --- | --- | --- |
| L0 Pi_CCS gamma-mixing + sum-check over K | -114.4 | Lemma 7; repo estimator |
| L0 Pi_RLC fork (K+k)/|C| | -121.3 | Lemma 8, Theorem 12 |
| L0 Pi_RLC sampler bias, 17 draws | -128.9 | repo estimator |
| L1 claim batching lambda (deg 36) | -186.7 | Lemma 2 |
| L1 ring lift at zeta (deg 106) | -185.2 | Lemma 2, quotient form |
| L1 linear sum-check (26 rounds, deg 2) | -186.2 | sum-check soundness |
| L1 logup at beta ((2^26 + 7,345)/|Ext|) | -165.9 | distinct poles: a value outside T_H has residue count in [1, 2^26] < p |
| L1 GKR (sum_k (3k + 2) / |Ext|, 325 rounds) | -181.9 | Papini-Habock fractional sum-check |
| L1 list multiplier on the five rows above | +0 (UD) / +5.3 (JB) | p3 `log2_max_candidates`: OOD samples come after our challenges |
| L1 WHIR opening, composed | <= -116.1 | p3 `prescribed_security`, config chosen to meet the budget |
| Poseidon2 Merkle and sponge (4-word digest, capacity 4) | 2^-128, computational | hash binding (ROM) |

Budget derivation, no invented margin: with minimum 114 and layer 0 at
2^-114.4 + 2^-121.3 + 2^-128.9 = 2^-114.38, layer 1 may use
2^-114 - 2^-114.38 = 2^-116.1. `Relation::new` raises the WHIR per-term level
until `prescribed_security` plus the outer rows meets that budget, or returns
`Error::Budget`. Total: 2^-114.0 (est.).

Extraction: from an accepting layer-1 proof, the WHIR extractor gives a table
z* (UD: the unique close codeword). The rows above make z* satisfy all 37
identities and |z*[x]| <= H < B on every slot except with the listed errors.
Restricted to real slots, z* is a CE(B, L) witness (Def. 21, all five
conjuncts, full carrier). Lemma 1 with Lemma 4 then extracts the fresh CCS(b, L)
witness and the 16 CE(b, L) children, which is what the terminal verifier
checks with witnesses today (the fresh tail check aside, Q4). Relaxed binding of
the Ajtai key (Theorem 6, MSIS) is the only lattice assumption, unchanged.

To settle if doubted: (a) a property test that the logup check fails for a
witness with one value at H + 1 and an honest-looking histogram; (b) an
independent check that p3's composed bound counts every WHIR phase (its own
unit test `composed_security_charges_every_phase` does this).

## 6. Costs (estimates)

| Item | Prover | Verifier v1 |
| --- | --- | --- |
| Layer 0 | Pi_CCS + Pi_RLC, CPU 20-35 s (Metal: PiCCS on device) | statement + replay: ms |
| Key prefix | SHAKE expansion once (31 GB XOF output), 3-5 s | same, 3-5 s |
| Matrix and chi_r passes | 0.5-1 s | 0.5-1 s |
| WHIR commit (rate 1/4, fold 4) | DFT 2^28 1-2 s, Merkle 50M perms 2-8 s | - |
| GKR | 2-4 s | ms |
| Quotient + linear sum-check | 1-2 s | ms |
| WHIR open / verify | 2-4 s | ms |
| Total | 35-65 s | 4-8 s |
| Peak memory | 10-14 GB | ~1.5 GB |

Proof size: statement 275 KB (16 running claims 263 KB + fresh 12 KB); layer 0
77 KB (output evals 73 KB); layer 1: histogram 29 KB, GKR 26 KB, Q and linear
rounds 2.5 KB, WHIR 200 KB (JB) to 400 KB (UD). Total 0.6-0.8 MB, against
221.7 MB today.

Milestone 2: verifier key and matrix work becomes openings of setup
commitments; verifier time drops to polylog (ms), proof grows by those
openings. Milestone 3: the shrink circuit verifies T0-T11 plus the layer-0
replay; the 0.6-0.8 MB becomes that circuit's witness, and the final proof is
the shrink proof (target <= 150 KB).

## 7. Test plan (every test <= 300 s, `--release`, `FoldingMode::Optimized`)

Unit tests in `crates/neo-spartan/tests/`:
1. `bridge.rs`: 0.8 Poseidon2 from pinned constants equals neo-ccs `PERM` on
   random states; the same layer-0 transcript gives equal challenger streams on
   both sides; `p3_point` agrees with our MLE on a random table.
2. `ring_lift.rs`: random omega and z: remainder equals t, and the lifted value
   equals sum_b omega_b(zeta) z_b(zeta); each Q coefficient + 1 breaks it.
3. `gkr.rs`: round trip; each layer value and round coefficient tampered fails;
   a witness with one value at H + 1 and a histogram that omits it fails.
4. `linear.rs`: round trip against brute force; tampering fails.
5. `end_to_end.rs`: synthetic CE(B) claims with the real production key prefix
   (n_R = 2 and n_R = 2^12), random sparse matrices (t = 4) from
   `SuperneoEvalCacheBuilder`, |z| <= H; c from signed digit planes combined with
   powers of 2; evals from `eval_real_v1_1_openings`. prove, bytes, decode, verify.
6. `mutation.rs` (one cached valid proof, one `#[test]` per component; each must
   reject): z coefficient + 1 before proving; parent c word; X word; Eval_K
   coefficient; Eval_A coefficient; r coordinate; Merkle root; histogram count
   moved by one; GKR layer value; GKR round coefficient; Q coefficient; linear
   round coefficient; z(s); one WHIR leaf byte; a trailing byte; a different
   layer-0 transcript (seed).

Crate-private tests in `crates/nightstream/tests/lifecycle_native/finish.rs`
(attached by `#[path]`, as existing native tests), on `Fixture::bit()` and
`Fixture::selected_polynomial()` from `tests/engines/parity.rs` (real key
prefix, 2 blocks, 16 running claims):
7. Layer-0 parity: `prove_parent_with_rows` + `verify_parent` give the same
   parent and the same transcript snapshot, and the parent equals
   `NifsProof.pi_rlc.combined` of the full fold on the same inputs.
8. Toy end to end: layer 0 + layer 1 prove and verify on both fixtures.
9. Layer-0 mutations: a Pi_CCS round coefficient, an output Eval_K
   coefficient, the fresh x word, a running claim eval: each rejects.
10. `FinalProof` codec: round trip; every truncation and one trailing byte reject.

Ignored production tests (`crates/nightstream/tests/circuit_lifecycle.rs`, like
the existing ones; F' compile exceeds the cap):
11. Two steps, `finish_with_spartan`, verify, encode/decode/verify; one byte
    flipped per codec section rejects; wrong expected state rejects.
12. Metal and Optimized produce byte-equal `FinalProof`.
13. Layer 1 alone at production shape: synthetic parent, full key, synthetic
    sparse matrices of production dimensions; records prover/verifier time and
    peak RSS; run by hand with `timeout 300` (heavy job, est. 15 GB).

## 8. Answers to the open questions (GROUNDING section 5)

1. **Where.** New crate `neo-spartan` owns the argument and every p3 0.8 type
   (public: `prove`, `verify`, `Relation`, `Proof`, `Error`). Layer 0 is
   `folding/compose.rs`; the finished lifecycle is `lifecycle/finish.rs`.
   nightstream exposes `FinalProof`, `finish_with_spartan`,
   `decode_final_proof` and `verify` (sealed `Verifiable`).
2. **Fields.** Cubic extension (~2^192) for WHIR and every layer-1 challenge.
   Non-WHIR terms are <= 2^-165, WHIR algebraic terms <= 2^-140. K claims split
   into Re/Im F-weighted claims; no K (x) Ext arithmetic in v1. Quintic is a
   one-alias change.
3. **Committed.** Only z: one base-field multilinear polynomial on the 2^26
   (block, lane) cube.
4. **Norm.** Logup with a fixed public histogram over {-H..H}, H = 3,672 from
   params (proves |z| <= H < B); the witness side is a fractional GKR ending in
   z(rho). No extra commitment.
5. **C = A z, final weight.** Quotient lift: one Q (53 Ext) for all 37 batched
   identities, checked at random zeta. The v1 verifier computes Omega from the
   key and matrices with the prover's own code and folds it at s_blk.
6. **Batching.** One lambda batch of 37 identities becomes one tensor-weight
   sum-check, Omega(b) * zeta^l, 26 rounds; x = L_in(z) is 5 of the 37.
7. **Transcript.** Unchanged fold transcript for layer 0, a 4-word seed, then one
   p3 `DuplexChallenger` on the same Poseidon2 instance for all of layer 1,
   WHIR included. Own domain chunk; profile and statement absorbed.
8. **Proof object.** `FinalProof`, magic `NS-FINAL-PROOF01`, exact length from
   the circuit, no carried fold digests.
9. **Backends.** Optimized on CPU; Metal runs Pi_CCS on the device through the
   existing hook and Pi_RLC on CPU (unsplit parent, no new Metal API).
   PaperExact and Crosscheck return an engine error for finish in v1. Layer 1
   and the final verifier are CPU only.
10. **Tests.** Section 7.

### Red-flag screen

- Shallow module: no. `neo-spartan` hides WHIR, GKR, logup, the lift and
  weights behind 2 functions and 3 types.
- Information leakage: no p3 or wire type crosses a crate boundary; the claim
  shape is validated at each boundary from one `Shape` that nightstream builds.
- Temporal decomposition: avoided. Modules own sub-protocols (both sides each);
  `protocol.rs` owns the schedule, not prover.rs/verifier.rs.
- Pass-through: `Prover::finish_with_spartan` forwards to the lifecycle, as
  `extend` does today. Accepted for consistency.
- Split ownership: the transcript moves into layer 1; weights have one function.
- Two ways: `encode_proof` (Proof) and `to_bytes` (FinalProof) are one way per
  type. `verify` is one verb.
- Importable internals: all `neo-spartan` modules are private.
- Hand-synced lists: claim slots, profile words and schedule each have one
  source; the codec's encoder and decoder share one length function.

## 9. Decisions, alternatives, questions

### Synthesis decision

(Filled in by arena.)

### Tradeoffs accepted

- A 26 KB GKR transcript and est. 2-4 s of GKR proving, for one WHIR instance in total.
- A 29 KB histogram in the clear, for no committed multiplicities and a fixed proof shape.
- Proving |z| <= 3,672 (stronger than < 2^16), for a fixed-size histogram.
  Completeness holds because T = 216 is a proven expansion bound (Theorem 5).
- 1.05 GB of weights on both sides in v1, for one shared weight function.
  Milestone 2 deletes the verifier's copy.
- Two duplex disciplines (v1.1 add-mode, p3 overwrite-mode) joined by a 4-word
  seed, for using WHIR's challenger unchanged.
- 263 KB of running claims in v1, for an unchanged state-hash binding.
- A sealed trait on `Verifier::verify`, for one lifecycle verb.
- Two p3 generations in the build, for no change to any 0.5.3 code path.

### Alternatives considered

- **Commit the 16 signed digit planes** and prove z = sum 2^i D_i, D_i in
  {-1, 0, 1}, by a degree-3 zero-check. Same interface depth and exactly
  < 2^16, but 16x the committed cells (2^30), 16x WHIR work, a bigger proof and
  a second sum-check family. Lost on cost; it hides nothing this design exposes.
- **Generic Spartan over an R1CS encoding of CE(B)** (bit decomposition
  in-circuit). Deep interface, but about 2^30 constraints, and it discards the
  ring and tensor structure that milestone 2 needs. Lost on cost and fit.
- **Layer 1 as modules inside nightstream.** One crate fewer, but p3 0.8 types
  become importable everywhere and "only the bridge names both generations"
  becomes a comment, not a build boundary. Lost on interface depth.
- **A `ProofState::Finished` variant inside `Proof`.** One type and codec, but
  "cannot extend a finished proof" becomes a runtime check and one type mixes
  witness-carrying and witness-free data. Lost.
- **Committed logup helper h = 1/(beta - z)** instead of GKR: one more
  Ext-valued 2^26 commitment (3x a base commitment) plus a zero-check. Lost on cost.
- **Coefficient-wise random combination** instead of the quotient lift: no
  quotient, but an O(54^2) ring product per block for the verifier and no
  Omega(b) * zeta^l tensor for milestone 2. Lost.

### Open questions and risks (for the owner)

1. Do you accept proving |z| <= H = 3,672 (fixed 29 KB histogram, stronger than
   Def. 21) instead of literal B = 2^16 (sparse histogram, variable length up to
   1 MB, shrink circuit sized for the worst case)?
2. WHIR soundness regime: UniqueDecoding (conjecture-free, WHIR est. 400 KB) or
   JohnsonBound (needs mutual correlated agreement up to Johnson; est. 200 KB)?
   For JB, can a reviewer confirm the published proof that p3-security's JB
   terms rely on?
3. Is the cubic extension acceptable, or do you want quintic for margin when
   FS query count is charged?
4. The terminal verifier checks that the fresh witness's 26 completion-tail lanes
   are zero. Pi_CCS + Pi_RLC do not carry that check, and the paper's CCS(b, L)
   with n_F = W does not contain it. Do finished proofs use the paper relation
   (no tail check), or do you want a non-paper extra claim?
5. Is zero knowledge required? The histogram, Q and WHIR openings leak
   information about z.
6. Freeze the v1 WHIR profile: rate 1/4, folding factor 4, PoW per p3's default
   (`DEFAULT_MAX_POW` = 16)? These values are a performance choice, not a
   derivation; the security level itself is derived from the budget.
7. Should `finish_with_spartan` reject an initial proof (my choice), or return a
   trivial finished proof?
8. Renamed 0.8 dependencies (my choice) or a workspace upgrade?

Risks to check in the first slice: p3 0.8's `Poseidon2Goldilocks<16>` built from
our constants may differ in its internal diagonal (the test decides; fallback
wrapper); `open_at` with two batches on one table and `DuplexChallenger::new` +
seed must behave as read in the 6b6a3b4 source; a fixed WHIR proof length is
assumed for the exact-length codec (otherwise a config-derived upper bound);
verifier key expansion time is unmeasured at 31 GB of XOF output.

### Next implementation step

Create `crates/neo-spartan` with `bridge.rs` and `pcs.rs` only, and make one
test pass: commit a random 2^12 base-field table with WHIR over our Poseidon2
instance, then verify two prescribed openings through a challenger seeded from
a layer-0 transcript, with the cross-version permutation equality test.
