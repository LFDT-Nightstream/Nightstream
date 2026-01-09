//! Poseidon2 (Goldilocks, WIDTH=8, RATE=4) gadget.
//!
//! This is the exact permutation used by `neo_transcript::Poseidon2Transcript`.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use bellpepper_core::num::AllocatedNum;
use once_cell::sync::Lazy;
use p3_field::PrimeField64;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::poseidon2_round_numbers_128;
use p3_symmetric::Permutation;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::CircuitF;

pub const WIDTH: usize = 8;
pub const RATE: usize = 4;
pub const SBOX_DEGREE: u64 = 7;

#[derive(Clone, Debug)]
struct Poseidon2ConstantsW8 {
    initial: [[CircuitF; WIDTH]; 4],
    terminal: [[CircuitF; WIDTH]; 4],
    internal: [CircuitF; 22],
}

fn to_circuit(x: Goldilocks) -> CircuitF {
    CircuitF::from(x.as_canonical_u64())
}

static CONSTANTS_W8: Lazy<Poseidon2ConstantsW8> = Lazy::new(|| {
    let (rounds_f, rounds_p) =
        poseidon2_round_numbers_128::<Goldilocks>(WIDTH, SBOX_DEGREE).expect("round numbers");
    assert_eq!(rounds_f, 8, "expected WIDTH=8, D=7 full rounds = 8");
    assert_eq!(rounds_p, 22, "expected WIDTH=8, D=7 partial rounds = 22");

    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);

    // This matches `p3_poseidon2::ExternalLayerConstants::new_from_rng`:
    // draw `half_f` WIDTH-wide vectors for initial, then `half_f` vectors for terminal.
    let mut draw_vec = || -> [Goldilocks; WIDTH] { rng.random() };
    let half_f = rounds_f / 2;
    let mut initial = [[CircuitF::from(0u64); WIDTH]; 4];
    let mut terminal = [[CircuitF::from(0u64); WIDTH]; 4];
    for r in 0..half_f {
        let v = draw_vec();
        initial[r] = v.map(to_circuit);
    }
    for r in 0..half_f {
        let v = draw_vec();
        terminal[r] = v.map(to_circuit);
    }

    // Internal constants: `rounds_p` field elements.
    let mut internal = [CircuitF::from(0u64); 22];
    for r in 0..rounds_p {
        let x: Goldilocks = rng.random();
        internal[r] = to_circuit(x);
    }

    Poseidon2ConstantsW8 {
        initial,
        terminal,
        internal,
    }
});

fn pow7_val(x: CircuitF) -> CircuitF {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * x
}

fn pow7<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    x: &AllocatedNum<CircuitF>,
    label: &str,
) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
    let mul = |cs: &mut CS, a: &AllocatedNum<CircuitF>, b: &AllocatedNum<CircuitF>, lbl: &str| {
        let a_val = a.get_value().unwrap_or(CircuitF::from(0u64));
        let b_val = b.get_value().unwrap_or(CircuitF::from(0u64));
        let out_val = a_val * b_val;
        let out = AllocatedNum::alloc(cs.namespace(|| lbl.to_string()), || Ok(out_val))?;
        cs.enforce(
            || format!("{lbl}_constraint"),
            |lc| lc + a.get_variable(),
            |lc| lc + b.get_variable(),
            |lc| lc + out.get_variable(),
        );
        Ok(out)
    };

    let x2 = mul(cs, x, x, &format!("{label}_x2"))?;
    let x4 = mul(cs, &x2, &x2, &format!("{label}_x4"))?;
    let x6 = mul(cs, &x4, &x2, &format!("{label}_x6"))?;
    mul(cs, &x6, x, &format!("{label}_x7"))
}

fn alloc_linear_comb<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    value: CircuitF,
    terms: &[(CircuitF, &AllocatedNum<CircuitF>)],
) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_out")), || Ok(value))?;
    cs.enforce(
        || format!("{label}_lc"),
        |lc| {
            let mut acc = lc;
            for (coeff, var) in terms {
                acc = acc + (*coeff, var.get_variable());
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
    );
    Ok(out)
}

fn external_linear_layer_w8<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    state: &mut [AllocatedNum<CircuitF>; WIDTH],
    state_val: &mut [CircuitF; WIDTH],
    label: &str,
) -> Result<(), SynthesisError> {
    // MDSMat4 for WIDTH=4 is the 4x4 matrix:
    // [2 3 1 1]
    // [1 2 3 1]
    // [1 1 2 3]
    // [3 1 1 2]
    const A: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

    // WIDTH=8 external linear layer is the block matrix [[2A, A], [A, 2A]].
    let old_vars = state.clone();
    let old_vals = *state_val;

    let mut new_vals = [CircuitF::from(0u64); WIDTH];

    for row in 0..WIDTH {
        let (block_row, top) = if row < 4 { (row, true) } else { (row - 4, false) };
        let mut terms: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(WIDTH);
        let mut acc = CircuitF::from(0u64);
        for col in 0..WIDTH {
            let (block_col, left) = if col < 4 { (col, true) } else { (col - 4, false) };
            let base = A[block_row][block_col];
            let scale = match (top, left) {
                (true, true) => 2,
                (true, false) => 1,
                (false, true) => 1,
                (false, false) => 2,
            };
            let coeff_u64 = base * scale;
            let coeff = CircuitF::from(coeff_u64);
            terms.push((coeff, &old_vars[col]));
            acc += coeff * old_vals[col];
        }
        new_vals[row] = acc;
        let out = alloc_linear_comb(
            cs,
            &format!("{label}_row_{row}"),
            acc,
            &terms,
        )?;
        state[row] = out;
    }

    *state_val = new_vals;
    Ok(())
}

fn internal_linear_layer_w8<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    state: &mut [AllocatedNum<CircuitF>; WIDTH],
    state_val: &mut [CircuitF; WIDTH],
    label: &str,
) -> Result<(), SynthesisError> {
    let diag: [CircuitF; WIDTH] = MATRIX_DIAG_8_GOLDILOCKS.map(to_circuit);

    let old_vars = state.clone();
    let old_vals = *state_val;

    let sum_val: CircuitF = old_vals.iter().copied().fold(CircuitF::from(0u64), |a, b| a + b);
    let mut new_vals = [CircuitF::from(0u64); WIDTH];

    for i in 0..WIDTH {
        let out_val = sum_val + diag[i] * old_vals[i];
        new_vals[i] = out_val;

        // out = sum + diag[i] * state[i]  (sum is a constant-LC over all old vars)
        let mut terms: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(WIDTH);
        for j in 0..WIDTH {
            let coeff = if i == j {
                CircuitF::from(1u64) + diag[i]
            } else {
                CircuitF::from(1u64)
            };
            terms.push((coeff, &old_vars[j]));
        }
        let out = alloc_linear_comb(
            cs,
            &format!("{label}_row_{i}"),
            out_val,
            &terms,
        )?;
        state[i] = out;
    }

    *state_val = new_vals;
    Ok(())
}

fn full_round_w8<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    state: &mut [AllocatedNum<CircuitF>; WIDTH],
    state_val: &mut [CircuitF; WIDTH],
    round_constants: &[CircuitF; WIDTH],
    label: &str,
) -> Result<(), SynthesisError> {
    for i in 0..WIDTH {
        let add_val = state_val[i] + round_constants[i];
        let added = AllocatedNum::alloc(cs.namespace(|| format!("{label}_add_{i}")), || Ok(add_val))?;
        cs.enforce(
            || format!("{label}_add_{i}_enforce"),
            |lc| lc + state[i].get_variable() + (round_constants[i], CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + added.get_variable(),
        );
        let sboxed = pow7(cs, &added, &format!("{label}_sbox_{i}"))?;
        state[i] = sboxed;
        state_val[i] = pow7_val(add_val);
    }
    external_linear_layer_w8(cs, state, state_val, &format!("{label}_mds"))
}

fn partial_round_w8<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    state: &mut [AllocatedNum<CircuitF>; WIDTH],
    state_val: &mut [CircuitF; WIDTH],
    rc: CircuitF,
    label: &str,
) -> Result<(), SynthesisError> {
    let add_val = state_val[0] + rc;
    let added = AllocatedNum::alloc(cs.namespace(|| format!("{label}_add_0")), || Ok(add_val))?;
    cs.enforce(
        || format!("{label}_add_0_enforce"),
        |lc| lc + state[0].get_variable() + (rc, CS::one()),
        |lc| lc + CS::one(),
        |lc| lc + added.get_variable(),
    );
    let sboxed = pow7(cs, &added, &format!("{label}_sbox_0"))?;
    state[0] = sboxed;
    state_val[0] = pow7_val(add_val);

    internal_linear_layer_w8(cs, state, state_val, &format!("{label}_internal"))
}

pub fn permute_w8<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    state: &mut [AllocatedNum<CircuitF>; WIDTH],
) -> Result<(), SynthesisError> {
    let mut state_val = [CircuitF::from(0u64); WIDTH];
    for i in 0..WIDTH {
        state_val[i] = state[i].get_value().unwrap_or(CircuitF::from(0u64));
    }

    // Initial external layer includes an extra MDS.
    external_linear_layer_w8(cs, state, &mut state_val, "poseidon2_init_mds")?;

    for (r, rc) in CONSTANTS_W8.initial.iter().enumerate() {
        full_round_w8(cs, state, &mut state_val, rc, &format!("poseidon2_full_init_{r}"))?;
    }

    for (r, rc) in CONSTANTS_W8.internal.iter().enumerate() {
        partial_round_w8(cs, state, &mut state_val, *rc, &format!("poseidon2_partial_{r}"))?;
    }

    for (r, rc) in CONSTANTS_W8.terminal.iter().enumerate() {
        full_round_w8(cs, state, &mut state_val, rc, &format!("poseidon2_full_term_{r}"))?;
    }

    Ok(())
}

pub fn native_permute_w8(mut state: [Goldilocks; WIDTH]) -> [Goldilocks; WIDTH] {
    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
    let perm = Poseidon2Goldilocks::<WIDTH>::new_from_rng_128(&mut rng);
    perm.permute_mut(&mut state);
    state
}
