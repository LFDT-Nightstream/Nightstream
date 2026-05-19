//! Poseidon2-in-circuit permutation gadget for `Goldilocks` WIDTH=8.
//!
//! Byte-for-byte parity with [`neo_ccs::crypto::poseidon2_goldilocks::PERM`]:
//! the round constants are re-derived from `neo_params::poseidon2_goldilocks::SEED`
//! via the same `ChaCha8Rng` + `StandardUniform` path the native PERM uses.
//!
//! Configuration (from `neo-params` + `p3-goldilocks`):
//! - WIDTH = 8, capacity 4, rate 4
//! - S-box: `x → x^7`  (`GOLDILOCKS_S_BOX_DEGREE = 7`)
//! - Rounds: 4 initial-external + 22 internal + 4 terminal-external (RF=8, RP=22)
//! - External linear layer: `mds_light_permutation` with `MDSMat4`
//! - Internal linear layer: `sum + diag · state` with
//!   `MATRIX_DIAG_8_GOLDILOCKS = [-2, 1, 2, 1/2, 3, -1/2, -3, -4]`
//!
//! ## Cost per permutation
//!
//! Each S-box: 4 mult constraints (`x → x² → x⁴ → x⁶ → x⁷`).
//! - Full rounds (8 × 4 sboxes each): 32 mults/round × 8 rounds = 256 mults
//! - Partial rounds (1 sbox each):    4 mults/round × 22 rounds = 88 mults
//! - **Total: 344 mults per permutation** (plus equality/output wires)
//!
//! Linear layers add no mult constraints — they thread linear combinations
//! through `Lc` arithmetic until each S-box input is materialized.

use std::sync::OnceLock;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use rand_chacha_p3::rand_core::{Rng as RandCoreRng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

const WIDTH: usize = 8;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 22;

/// Round constants for the native `neo_ccs::crypto::poseidon2_goldilocks::PERM`.
///
/// Re-derived from `neo_params::poseidon2_goldilocks::SEED` via exactly the
/// same `ChaCha8Rng` + `StandardUniform` sampling path:
///
/// ```text
/// rng = ChaCha8Rng::from_seed(SEED);
/// initial   = rng.sample_iter(StandardUniform).take(4)  // 4 × [F; 8]
/// terminal  = rng.sample_iter(StandardUniform).take(4)  // 4 × [F; 8]
/// internal  = rng.sample_iter(StandardUniform).take(22) // 22 × F
/// ```
pub struct Poseidon2Constants {
    pub initial: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub terminal: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub internal: [F; PARTIAL_ROUNDS],
}

/// Sample one `Goldilocks` from the RNG using exactly the same rejection
/// sampling as `p3_goldilocks::Distribution<Goldilocks> for StandardUniform`:
/// `next_u64()`, accept if `< Goldilocks::ORDER_U64`, else retry.
fn sample_goldilocks(rng: &mut ChaCha8Rng) -> F {
    loop {
        let x = rng.next_u64();
        if x < Goldilocks::ORDER_U64 {
            return F::from_u64(x);
        }
    }
}

fn constants() -> &'static Poseidon2Constants {
    static CONST: OnceLock<Poseidon2Constants> = OnceLock::new();
    CONST.get_or_init(|| {
        let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);

        // Order mirrors `Poseidon2::new_from_rng_128 → new_from_rng(8, 22)`:
        //   ExternalLayerConstants::new_from_rng(8) consumes:
        //     - 4 samples of [Goldilocks; 8]  (initial)
        //     - 4 samples of [Goldilocks; 8]  (terminal)
        //   Then 22 standalone Goldilocks samples (internal).
        // Each [Goldilocks; 8] is 8 sequential samples.
        let mut initial = [[F::ZERO; WIDTH]; HALF_FULL_ROUNDS];
        for round in initial.iter_mut() {
            for slot in round.iter_mut() {
                *slot = sample_goldilocks(&mut rng);
            }
        }
        let mut terminal = [[F::ZERO; WIDTH]; HALF_FULL_ROUNDS];
        for round in terminal.iter_mut() {
            for slot in round.iter_mut() {
                *slot = sample_goldilocks(&mut rng);
            }
        }
        let mut internal = [F::ZERO; PARTIAL_ROUNDS];
        for slot in internal.iter_mut() {
            *slot = sample_goldilocks(&mut rng);
        }

        Poseidon2Constants {
            initial,
            terminal,
            internal,
        }
    })
}

/// Goldilocks-WIDTH-8 internal diagonal `[-2, 1, 2, 1/2, 3, -1/2, -3, -4]`.
fn internal_diag() -> [F; WIDTH] {
    static DIAG: OnceLock<[F; WIDTH]> = OnceLock::new();
    *DIAG.get_or_init(|| {
        let half = F::from_u64(2).inverse();
        [
            -F::from_u64(2),
            F::ONE,
            F::from_u64(2),
            half,
            F::from_u64(3),
            -half,
            -F::from_u64(3),
            -F::from_u64(4),
        ]
    })
}

/// S-box `x → x^7` via 4 mult constraints.
///
/// Allocates `x²`, `x⁴`, `x⁶`, `x⁷` as fresh witness vars. Returns the
/// `x⁷` wire.
fn enforce_sbox_x7(builder: &mut R1csBuilder, x_lc: &Lc) -> Var {
    let x2 = builder.alloc_mul(x_lc, x_lc);
    let x4 = builder.alloc_mul(&Lc::from_var(x2), &Lc::from_var(x2));
    let x6 = builder.alloc_mul(&Lc::from_var(x2), &Lc::from_var(x4));
    builder.alloc_mul(x_lc, &Lc::from_var(x6))
}

/// Apply the 4×4 MDS matrix `apply_mat4` in-place to a length-4 state of Lcs.
///
/// The matrix is:
/// ```text
/// [2 3 1 1]
/// [1 2 3 1]
/// [1 1 2 3]
/// [3 1 1 2]
/// ```
///
/// Implemented via the same sequence of additions/doublings as
/// `p3_poseidon2::external::apply_mat4`.
fn apply_mat4(state: &mut [Lc; 4]) {
    let x0 = state[0].clone();
    let x1 = state[1].clone();
    let x2 = state[2].clone();
    let x3 = state[3].clone();

    let t01 = x0.clone().add_scaled(&x1, F::ONE);
    let t23 = x2.clone().add_scaled(&x3, F::ONE);
    let t0123 = t01.clone().add_scaled(&t23, F::ONE);
    let t01123 = t0123.clone().add_scaled(&x1, F::ONE);
    let t01233 = t0123.add_scaled(&x3, F::ONE);

    // x[3] = t01233 + 2·x[0]
    state[3] = t01233.clone().add_scaled(&x0, F::from_u64(2));
    // x[1] = t01123 + 2·x[2]
    state[1] = t01123.clone().add_scaled(&x2, F::from_u64(2));
    // x[0] = t01123 + t01  (= 2·x[0] + 3·x[1] + x[2] + x[3])
    state[0] = t01123.add_scaled(&t01, F::ONE);
    // x[2] = t01233 + t23  (= x[0] + x[1] + 2·x[2] + 3·x[3])
    state[2] = t01233.add_scaled(&t23, F::ONE);
}

/// External linear layer `mds_light_permutation` for WIDTH=8.
///
/// 1. Apply `apply_mat4` to elements [0..4] and [4..8] independently.
/// 2. Compute `sums[k] = state[k] + state[k+4]` for k ∈ [0..4].
/// 3. `state[i] += sums[i % 4]`.
fn external_linear_layer(state: &mut [Lc; WIDTH]) {
    let mut lo: [Lc; 4] = [state[0].clone(), state[1].clone(), state[2].clone(), state[3].clone()];
    let mut hi: [Lc; 4] = [state[4].clone(), state[5].clone(), state[6].clone(), state[7].clone()];
    apply_mat4(&mut lo);
    apply_mat4(&mut hi);

    let sums: [Lc; 4] = [
        lo[0].clone().add_scaled(&hi[0], F::ONE),
        lo[1].clone().add_scaled(&hi[1], F::ONE),
        lo[2].clone().add_scaled(&hi[2], F::ONE),
        lo[3].clone().add_scaled(&hi[3], F::ONE),
    ];

    for (i, slot) in state.iter_mut().enumerate() {
        let block = if i < 4 { &lo[i] } else { &hi[i - 4] };
        *slot = block.clone().add_scaled(&sums[i % 4], F::ONE);
    }
}

/// Internal linear layer for Goldilocks WIDTH=8:
///   `sum = Σ state; state[i] = sum + diag[i] · state[i]`.
///
/// All operations are linear; no R1CS mults emitted.
fn internal_linear_layer(state: &mut [Lc; WIDTH]) {
    let diag = internal_diag();
    let mut sum = Lc::zero();
    for s in state.iter() {
        sum = sum.add_scaled(s, F::ONE);
    }
    for (i, slot) in state.iter_mut().enumerate() {
        let scaled = Lc::zero().add_scaled(slot, diag[i]);
        *slot = sum.clone().add_scaled(&scaled, F::ONE);
    }
}

/// Materialize each lane as a fresh witness wire.
///
/// Poseidon2 applies many linear layers between S-boxes. Keeping those layers
/// as symbolic linear combinations across all 22 partial rounds makes the LCs
/// grow explosively. Materializing after each linear layer adds cheap linear
/// constraints and keeps every state lane bounded to one witness term.
fn materialize_state(builder: &mut R1csBuilder, state: &mut [Lc; WIDTH]) {
    for lane in state.iter_mut() {
        let v = builder.alloc(builder.eval(lane));
        builder.enforce_eq(&Lc::from_var(v), lane);
        *lane = Lc::from_var(v);
    }
}

/// Add round constants to the state (linear; no mults).
fn add_round_constants(state: &mut [Lc; WIDTH], rc: &[F; WIDTH]) {
    for (slot, &c) in state.iter_mut().zip(rc.iter()) {
        slot.add_constant(c);
    }
}

/// Apply one full external round: add RC, S-box all 8 lanes, external linear layer.
fn enforce_external_round(builder: &mut R1csBuilder, state: &mut [Lc; WIDTH], rc: &[F; WIDTH]) {
    add_round_constants(state, rc);
    let mut sbox_out: [Lc; WIDTH] = std::array::from_fn(|_| Lc::zero());
    for (i, slot) in state.iter().enumerate() {
        let v = enforce_sbox_x7(builder, slot);
        sbox_out[i] = Lc::from_var(v);
    }
    *state = sbox_out;
    external_linear_layer(state);
    materialize_state(builder, state);
}

/// Apply one partial internal round: add RC to lane 0, S-box lane 0, internal layer.
fn enforce_internal_round(builder: &mut R1csBuilder, state: &mut [Lc; WIDTH], rc: F) {
    state[0].add_constant(rc);
    let v0 = enforce_sbox_x7(builder, &state[0]);
    state[0] = Lc::from_var(v0);
    internal_linear_layer(state);
    materialize_state(builder, state);
}

/// Emit the full Poseidon2 permutation circuit for `Goldilocks` WIDTH=8.
///
/// Allocates the output state as fresh wires (so callers can compose multiple
/// permutations) and returns them. Each output equals the corresponding
/// native `PERM.permute(input)` lane, byte-for-byte.
pub fn enforce_poseidon2_permutation(builder: &mut R1csBuilder, state_in: &[Var; WIDTH]) -> [Var; WIDTH] {
    let c = constants();
    let mut state: [Lc; WIDTH] = std::array::from_fn(|i| Lc::from_var(state_in[i]));

    // Initial mds_light_permutation (the "pre-round" matmul Poseidon2 prescribes).
    external_linear_layer(&mut state);
    materialize_state(builder, &mut state);

    // 4 initial external rounds.
    for round in 0..HALF_FULL_ROUNDS {
        enforce_external_round(builder, &mut state, &c.initial[round]);
    }

    // 22 internal rounds.
    for round in 0..PARTIAL_ROUNDS {
        enforce_internal_round(builder, &mut state, c.internal[round]);
    }

    // 4 terminal external rounds.
    for round in 0..HALF_FULL_ROUNDS {
        enforce_external_round(builder, &mut state, &c.terminal[round]);
    }

    // Materialize the output state as fresh wires.
    let mut out = [Var::ONE; WIDTH];
    for (i, lc) in state.into_iter().enumerate() {
        let v = builder.alloc(builder.eval(&lc));
        builder.enforce_eq(&Lc::from_var(v), &lc);
        out[i] = v;
    }
    out
}

/// Sponge digest length (number of output `F`-limbs). Matches
/// `neo_params::poseidon2_goldilocks::DIGEST_LEN`.
pub const DIGEST_LEN: usize = 4;
const RATE: usize = 4;

/// Variable-length Poseidon2 sponge hash, byte-for-byte parity with
/// [`neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash`]:
///
/// ```text
/// state = [0; WIDTH]
/// for chunk in input.chunks(RATE):
///     state[0..chunk.len()] += chunk
///     state = permute(state)
/// state[0] += 1                  // padding
/// state = permute(state)
/// output = state[0..DIGEST_LEN]
/// ```
///
/// Allocates one zero-wire (reused as the initial state lanes), then per
/// chunk: a fresh wire per absorbed lane + one full permutation. Final
/// padding: one wire + one permutation.
pub fn enforce_poseidon2_hash(builder: &mut R1csBuilder, input: &[Var]) -> [Var; DIGEST_LEN] {
    // Allocate one zero wire and constrain it.
    let zero_var = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(zero_var), &Lc::zero());

    // Initial state: 8 zero lanes.
    let mut state: [Var; WIDTH] = [zero_var; WIDTH];

    // Absorb. Empty input still triggers one permutation (matches the native
    // loop, which iterates zero times — so we skip it). Native code does NOT
    // run a permute on empty input; only the padding-permute.
    for chunk in input.chunks(RATE) {
        // For each lane i in the chunk, new state[i] = old state[i] + chunk[i].
        let mut next = [zero_var; WIDTH];
        for (i, &x) in chunk.iter().enumerate() {
            let lc = Lc::from_var(state[i]).add_scaled(&Lc::from_var(x), F::ONE);
            let v = builder.alloc(builder.eval(&lc));
            builder.enforce_eq(&Lc::from_var(v), &lc);
            next[i] = v;
        }
        // Unaffected lanes carry over.
        for i in chunk.len()..WIDTH {
            next[i] = state[i];
        }
        // Permute.
        state = enforce_poseidon2_permutation(builder, &next);
    }

    // Padding: state[0] += 1.
    let padded_lc = {
        let mut lc = Lc::from_var(state[0]);
        lc.add_constant(F::ONE);
        lc
    };
    let padded = builder.alloc(builder.eval(&padded_lc));
    builder.enforce_eq(&Lc::from_var(padded), &padded_lc);
    state[0] = padded;

    // Final permute.
    state = enforce_poseidon2_permutation(builder, &state);

    // Output first DIGEST_LEN lanes.
    let mut out = [Var::ONE; DIGEST_LEN];
    out.copy_from_slice(&state[..DIGEST_LEN]);
    out
}

#[cfg(test)]
pub fn constants_for_test() -> &'static Poseidon2Constants {
    constants()
}
