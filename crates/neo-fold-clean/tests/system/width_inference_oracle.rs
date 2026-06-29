//! Exhaustive completeness oracle for the app-private width inference.
//!
//! The invariant under test: for every satisfying R1CS assignment whose
//! values lie in the enumerated box, `conservative_app_private_var_widths`
//! must return a width that covers each variable's value. The oracle
//! enumerates *all* assignments of tiny systems, filters to the satisfying
//! ones, computes each variable's true maximum, and checks
//! `inferred_width >= bit_width(true_max)`. Hand-built fixtures cover the
//! known gadget shapes; seeded generators cover random compositions; a SHA
//! corpus check asserts real honest witnesses fit the inferred plan widths.

use ::bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field as FfField;
use neo_ccs::matrix::Mat as NeoMat;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperGoldilocks};
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::r1cs_f_prime::R1csShape;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Largest value enumerated per variable (box `[0, BOX_MAX]`).
const BOX_MAX: u64 = 3;

fn bit_width(max: u64) -> usize {
    if max == 0 {
        1
    } else {
        (64 - max.leading_zeros()) as usize
    }
}

/// Enumerate all assignments with `z[0] = 1` and `z[i] in [0, BOX_MAX]`,
/// keep the satisfying ones, and check the inferred widths cover every
/// variable's true maximum. Returns the satisfying-assignment count so
/// callers can assert non-vacuity.
fn check_widths_against_exhaustive_oracle(r1cs: &R1cs, label: &str) -> usize {
    let m = r1cs.m();
    let n = r1cs.n();
    let widths = R1csShape::from(r1cs).conservative_app_private_var_widths();
    assert_eq!(widths.len(), m);

    let mut true_max = vec![0u64; m];
    let mut satisfying = 0usize;
    let free = m - 1;
    let total = (BOX_MAX + 1).pow(free as u32);
    let mut z = vec![F::ZERO; m];
    z[0] = F::ONE;
    for counter in 0..total {
        let mut acc = counter;
        let mut values = vec![1u64; m];
        for var in 1..m {
            values[var] = acc % (BOX_MAX + 1);
            acc /= BOX_MAX + 1;
            z[var] = F::from_u64(values[var]);
        }
        let mut ok = true;
        for row in 0..n {
            let mut az = F::ZERO;
            let mut bz = F::ZERO;
            let mut cz = F::ZERO;
            for var in 0..m {
                az += r1cs.a[(row, var)] * z[var];
                bz += r1cs.b[(row, var)] * z[var];
                cz += r1cs.c[(row, var)] * z[var];
            }
            if az * bz != cz {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        satisfying += 1;
        for var in 0..m {
            true_max[var] = true_max[var].max(values[var]);
        }
    }

    for var in 0..m {
        assert!(
            widths[var] >= bit_width(true_max[var]),
            "{label}: var {var} has satisfying value {} (needs width {}) but inference returned width {}",
            true_max[var],
            bit_width(true_max[var]),
            widths[var],
        );
    }
    satisfying
}

/// `m` variables, rows appended by each builder. v0 is the constant lane;
/// the first `booleans` variables after it get explicit Boolean rows.
fn system(m: usize, booleans: usize, extra: usize) -> (NeoMat<F>, NeoMat<F>, NeoMat<F>) {
    let n = booleans + extra;
    let mut a = NeoMat::zero(n, m, F::default());
    let mut b = NeoMat::zero(n, m, F::default());
    let c = NeoMat::zero(n, m, F::default());
    for (row, var) in (1..=booleans).enumerate() {
        a[(row, var)] = F::ONE;
        b[(row, 0)] = F::ONE;
        b[(row, var)] = F::ZERO - F::ONE;
    }
    (a, b, c)
}

#[test]
fn oracle_ch_select_gadget() {
    let (mut a, mut b, mut c) = system(6, 3, 1);
    // (v2 - v3) * v1 = v4 - v3
    a[(3, 2)] = F::ONE;
    a[(3, 3)] = F::ZERO - F::ONE;
    b[(3, 1)] = F::ONE;
    c[(3, 4)] = F::ONE;
    c[(3, 3)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let sat = check_widths_against_exhaustive_oracle(&r1cs, "ch");
    assert!(sat >= 8, "all Boolean corners must satisfy, got {sat}");
}

#[test]
fn oracle_maj_pair_gadget() {
    let (mut a, mut b, mut c) = system(7, 3, 2);
    // v2 * v3 = v5 ; (2*v5 - v2 - v3) * v1 = v5 - v6
    a[(3, 2)] = F::ONE;
    b[(3, 3)] = F::ONE;
    c[(3, 5)] = F::ONE;
    a[(4, 5)] = F::from_u64(2);
    a[(4, 2)] = F::ZERO - F::ONE;
    a[(4, 3)] = F::ZERO - F::ONE;
    b[(4, 1)] = F::ONE;
    c[(4, 5)] = F::ONE;
    c[(4, 6)] = F::ZERO - F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let sat = check_widths_against_exhaustive_oracle(&r1cs, "maj");
    assert!(sat >= 8, "all Boolean corners must satisfy, got {sat}");
}

#[test]
fn oracle_interior_extremum_gadget() {
    let (mut a, mut b, mut c) = system(7, 2, 2);
    // (v1 + 2*v2) * v0 = v5 ; (3*v0 - v5) * v5 = v6
    a[(2, 1)] = F::ONE;
    a[(2, 2)] = F::from_u64(2);
    b[(2, 0)] = F::ONE;
    c[(2, 5)] = F::ONE;
    a[(3, 0)] = F::from_u64(3);
    a[(3, 5)] = F::ZERO - F::ONE;
    b[(3, 5)] = F::ONE;
    c[(3, 6)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let sat = check_widths_against_exhaustive_oracle(&r1cs, "interior-extremum");
    assert!(sat >= 4, "all (v1, v2) corners must satisfy, got {sat}");
}

#[test]
fn oracle_affine_sum_gadget() {
    let (mut a, mut b, mut c) = system(6, 3, 1);
    // (v1 + v2 + v3) * v0 = v4   — sum of three Booleans, range [0, 3]
    a[(3, 1)] = F::ONE;
    a[(3, 2)] = F::ONE;
    a[(3, 3)] = F::ONE;
    b[(3, 0)] = F::ONE;
    c[(3, 4)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: 1 };
    let sat = check_widths_against_exhaustive_oracle(&r1cs, "affine-sum");
    assert!(sat >= 8, "all Boolean corners must satisfy, got {sat}");
}

/// Seeded property test: random compositions of the gadget templates over
/// 8 variables, oracle-checked exhaustively. Deterministic LCG seeds keep
/// runs reproducible.
#[test]
fn oracle_seeded_gadget_compositions() {
    let mut nonvacuous = 0usize;
    for seed in 0u64..24 {
        let mut state = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(seed + 1);
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        let m = 8usize;
        let booleans = 3usize;
        let extra = 3usize;
        let (mut a, mut b, mut c) = system(m, booleans, extra);
        // Outputs occupy v4..=v6; supports draw from already-defined vars.
        for (slot, out) in (4..4 + extra).enumerate() {
            let row = booleans + slot;
            let pick = |next: &mut dyn FnMut(u64) -> u64, hi: usize| 1 + next((hi - 1) as u64) as usize;
            match next(4) {
                // AND: x * y = out
                0 => {
                    a[(row, pick(&mut next, out))] = F::ONE;
                    b[(row, pick(&mut next, out))] = F::ONE;
                    c[(row, out)] = F::ONE;
                }
                // XOR: (2x) * y = x + y - out
                1 => {
                    let x = pick(&mut next, out);
                    let mut y = pick(&mut next, out);
                    if y == x {
                        y = if x > 1 { x - 1 } else { x + 1 };
                    }
                    a[(row, x)] = F::from_u64(2);
                    b[(row, y)] = F::ONE;
                    c[(row, x)] = F::ONE;
                    c[(row, y)] += F::ONE;
                    c[(row, out)] = F::ZERO - F::ONE;
                }
                // MUX: (x - y) * s = out - y
                2 => {
                    let x = pick(&mut next, out);
                    let mut y = pick(&mut next, out);
                    if y == x {
                        y = if x > 1 { x - 1 } else { x + 1 };
                    }
                    let s = pick(&mut next, out);
                    a[(row, x)] = F::ONE;
                    a[(row, y)] = F::ZERO - F::ONE;
                    b[(row, s)] = F::ONE;
                    c[(row, out)] = F::ONE;
                    c[(row, y)] = F::ZERO - F::ONE;
                }
                // Affine sum with small coefficients: (x + 2y) * 1 = out
                _ => {
                    let x = pick(&mut next, out);
                    let y = pick(&mut next, out);
                    a[(row, x)] = F::ONE;
                    a[(row, y)] += F::from_u64(2);
                    b[(row, 0)] = F::ONE;
                    c[(row, out)] = F::ONE;
                }
            }
        }
        let r1cs = R1cs { a, b, c, m_in: 1 };
        let sat = check_widths_against_exhaustive_oracle(&r1cs, &format!("seeded-{seed}"));
        if sat > 0 {
            nonvacuous += 1;
        }
    }
    assert!(
        nonvacuous >= 20,
        "seeded systems should mostly be satisfiable, got {nonvacuous}/24"
    );
}

/// Honest-witness corpus check: synthesize real SHA-256 circuits for
/// several distinct preimages and assert every raw assignment value fits
/// the inferred plan width — the end-to-end completeness statement on
/// production-shaped witnesses.
struct ShaPreimageCircuit {
    preimage: Vec<u8>,
}

impl Circuit<BellpepperGoldilocks> for ShaPreimageCircuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let bit_values = ::bellpepper::gadgets::multipack::bytes_to_bits(&self.preimage)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            .map(|(idx, bit)| AllocatedBit::alloc(cs.namespace(|| format!("preimage_bit_{idx}")), bit))
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;
        let hash_bits = ::bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
        let value = hash_bits[0]
            .get_value()
            .map(|bit| {
                if bit {
                    BellpepperGoldilocks::ONE
                } else {
                    BellpepperGoldilocks::ZERO
                }
            })
            .ok_or(SynthesisError::AssignmentMissing)?;
        let input = cs.alloc_input(|| "hash_out_bit_0", || Ok(value))?;
        cs.enforce(
            || "bind hash_out_bit_0",
            |lc| lc + input,
            |lc| lc + CS::one(),
            |_| hash_bits[0].lc(CS::one(), BellpepperGoldilocks::ONE),
        );
        Ok(())
    }
}

#[test]
fn sha256_honest_witness_corpus_fits_inferred_widths() {
    let preimages: [Vec<u8>; 3] = [
        vec![0u8; 64],
        (0u8..64).collect(),
        (0u8..64).map(|i| i.wrapping_mul(37) ^ 0xA5).collect(),
    ];
    let mut widths: Option<Vec<usize>> = None;
    for (which, preimage) in preimages.into_iter().enumerate() {
        let ccs = synthesize_to_ccs(ShaPreimageCircuit { preimage }).expect("synthesize SHA corpus circuit");
        let widths =
            widths.get_or_insert_with(|| R1csShape::from(&ccs.sparse_r1cs).conservative_app_private_var_widths());
        assert_eq!(widths.len(), ccs.assignment.len());
        for (var, (&value, &width)) in ccs.assignment.iter().zip(widths.iter()).enumerate() {
            let value = value.as_canonical_u64();
            assert!(
                width >= 64 || value < (1u64 << width),
                "corpus {which}: var {var} carries value {value} but inferred width is {width}"
            );
        }
    }
}
