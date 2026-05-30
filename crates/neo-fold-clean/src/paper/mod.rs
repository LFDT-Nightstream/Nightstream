//! Paper layer: SuperNeo §7 + Hypernova §6.3, in paper order, with paper names.
//!
//! This module is the auditor's home. Every identifier here either *is* a
//! paper symbol or has a one-line glossary entry mapping it back to one. If
//! you have to guess what something means, that's a bug — file it.
//!
//! ## Paper symbol → code identifier
//!
//! ### §4 Preliminaries (Definitions 1–6)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | 𝔽 (base field, prime q)              | `neo_math::F`                | `neo-math` |
//! | 𝕂 (extension field, soundness)        | `neo_math::K`                | `neo-math` |
//! | d (cyclotomic degree, ring slot size) | `neo_math::D`                | `neo-math` |
//! | 𝑅_𝔽 = 𝔽[X]/Φ(X)                       | `neo_math::Rq`               | `neo-math` |
//! | cf, ct (coefficient / constant maps)  | `neo_math::cf`, `ct`         | `neo-math` |
//! | ‖·‖_∞ (centered infinity norm)        | `neo_math::balanced::*`      | `neo-math` |
//! | split_b (Def. 3 b-ary decomposition)  | `neo_math::balanced::split_b`| `neo-math` |
//! | sumcheck SumCheck(T; Q) (Def. 6)      | `neo_reductions::sumcheck`   | `neo-reductions` |
//! | Poseidon2 transcript                  | `neo_transcript::Poseidon2Transcript` | `neo-transcript` |
//!
//! ### §5 Embedding (Definitions 7–8, Theorems 3–5)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | bar(·) (lifted transform)             | `neo_math::superneo_bar_*`   | `neo-math` |
//! | Mz vs ct(bar(M)z)                     | `neo_math::superneo_bar_*`   | `neo-math` |
//! | evaluation homomorphism (Thm. 5)      | enforced by `paper/pi_rlc`   | this crate |
//!
//! ### §7.1 Relations (Definitions 11–13)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | s = ({M_j}, f) — structure (Def. 11)  | [`relations::Structure`]     | `paper/relations.rs` |
//! | CCS(b, ℒ) (Def. 12)                   | [`relations::CcsRelation`]   | `paper/relations.rs` |
//! | CE(b, ℒ) (Def. 13)                    | [`relations::CeRelation`]    | `paper/relations.rs` |
//! | one CCS (claim, witness) pair         | [`relations::CcsInstance`]   | `paper/relations.rs` |
//! | (c_i, x_i, r, {y_{i,j}}) CE claim     | [`relations::CeClaim`]       | `paper/relations.rs` |
//! | w_i CCS witness                       | [`relations::CcsWitness`]    | `paper/relations.rs` |
//!
//! ### §7.2 Global parameters (Definition 14)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | (𝔽, 𝕂, d, m, n_𝔽, n_𝑅, n_𝔽,in, k_rho, K, b, B, t, u) | [`params::Params`] | `paper/params.rs` |
//! | strong sampling set 𝒞 (Def. 17)       | [`sampling::StrongSet`]      | `paper/sampling.rs` |
//! | expansion factor T (Thm. 9)           | [`sampling::expansion_T`]    | `paper/sampling.rs` |
//! | Π_RLC norm bound (K+k)·T·(b−1) < B    | [`sampling::check_rlc_bound`]| `paper/sampling.rs` |
//! | ℒ ring commitment (Def. 4, Def. 18)   | `neo_ajtai::AjtaiSModule`    | `neo-ajtai` |
//!
//! ### §7.3 Π_CCS (the sumcheck-based fold)
//!
//! The Q polynomial, sumcheck, terminal identity check, and the
//! (α, γ, r', y'_{i,j}) computations all live in
//! `engine::optimized` (which wraps `neo-reductions`). The paper layer
//! exposes only the *seam*: shape-checked `prove` / `verify` over a
//! `Proof { sumcheck, outputs }` bundle. See `paper/pi_ccs.rs`.
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | (α, γ, r', y'_{i,j}, Q, T, sumcheck) | inside `pi_ccs::Proof::sumcheck` (= `neo_reductions::api::PiCcsProof`) | engine |
//! | Π_CCS prove                           | `pi_ccs::prove`              | `paper/pi_ccs.rs` |
//! | Π_CCS verify                          | `pi_ccs::verify`             | `paper/pi_ccs.rs` |
//! | K+k output CE claims                  | `pi_ccs::Proof::outputs`     | `paper/pi_ccs.rs` |
//!
//! ### §7.4 Π_RLC (random linear combination)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | ρ_i ∈ 𝒞 (rotation matrices)          | [`sampling::RotRho`]         | `paper/sampling.rs` |
//! | combined CE claim of norm B = b^k     | `pi_rlc::Output::claim` (prover) / `pi_rlc::verify` return (verifier; recomputed) | `paper/pi_rlc.rs` |
//!
//! ### §7.5 Π_DEC (decomposition)
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | (z_1,…,z_k) ← split_b(z)              | `pi_dec::Children`           | `paper/pi_dec.rs` |
//! | c ?= Σ b^{i−1}·c_i                    | `pi_dec::verify`             | `paper/pi_dec.rs` |
//! | y_j ?= Σ b^{i−1}·y_{i,j}              | `pi_dec::verify`             | `paper/pi_dec.rs` |
//!
//! ### Hypernova §6.2–§6.3 Construction 2
//!
//! | Paper | Code | Source |
//! |-------|------|--------|
//! | i — chunk counter                     | `construction2::State::chunk_count` | `paper/construction2.rs` |
//! | step counter (K-aware)                | `construction2::State::step_count`  | same |
//! | z_0, z_i — input/state                | `construction2::State::z_0`, `z_i`  | same |
//! | U_i — running accumulator (W_i prover-only) | `construction2::ProofState::Active.running` (`RunningInstance`) | same |
//! | u_i — latest CCS instance(s) to fold (= encoding of F'_{i-1}) | `construction2::ProofState::Active.latest` (`LatestInstance`) | same |
//! | i = 0 base case (U_i = u_⊥)           | `construction2::ProofState::Initial`   | same |
//! | pc_i ∈ [ℓ]                            | `construction2::State::pc`          | same |
//! | vk_fs                                 | `construction2::VerifierKey`        | same |
//! | enc_inst(h)                           | `construction2::EncInst`            | same |
//! | F' (augmented function)               | `f_prime::{prove, verify}`          | `paper/f_prime.rs` |
//! | NIFS.P / NIFS.V                       | `nifs::{prove, verify}`             | `paper/nifs.rs` |
//! | x_{i+1} = H(...) hash-chain absorb    | `digest::state_x_out_digest`        | `paper/digest.rs` |
//!
//! ## ℓ = 1 specialization
//!
//! This crate specializes Construction 2 to ℓ = 1: a single accumulator slot
//! and `pc = TRIVIAL_PC`. Per-opcode case-splits, where they exist in a
//! frontend, live in that frontend — not in the IVC dispatch. This is the same
//! choice the paper allows when there is one step function.
//!
//! ## What is *not* in this layer
//!
//! - Performance counters, traces, shape probes. Move them to `engine/` if
//!   you need them.
//! - Frontend-specific types (RV32IM, CHIP-8). Frontends translate to the
//!   types in this module before calling into it.
//! - Anything that takes a closure with an unbounded lifetime. Clarity over
//!   flexibility.

pub mod construction2;
pub mod decider;
pub(crate) mod decider_ce_relation;
pub mod digest;
pub mod f_prime;
pub mod nifs;
pub mod params;
pub mod proof;
pub mod reductions;
pub mod relations;
pub mod sampling;

// Path stability: keep the short paths working so call sites don't churn
// every time we relocate. The auditor still reads the new structure via
// the directory layout.
pub use reductions::{pi_ccs, pi_dec, pi_rlc};
