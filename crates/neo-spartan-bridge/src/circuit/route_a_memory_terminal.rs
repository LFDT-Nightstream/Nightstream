//! Route-A memory terminal checks (Phase 1 completion).
//!
//! Mirrors `neo_fold::memory_sidecar::memory::verify_route_a_memory_step`:
//! - Shout time-lane terminals (bitness/value/adapter/table linkage)
//! - Twist time-lane terminals (bitness/read/write linkage)
//! - Twist val-eval terminals at `r_val` (lt/total/prev_total + rollover init equation)
//!
//! This module is intentionally "wiring-only": it computes the same K-field expressions as native
//! and enforces equality against the transcript-derived sumcheck terminals.

use bellpepper_core::{ConstraintSystem, Variable};
use neo_math::{F as NeoF, K as NeoK};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::circuit::fold_circuit_helpers as helpers;
use crate::circuit::fold_circuit::{McsInstanceVars, MeInstanceVars};
use crate::error::{Result, SpartanBridgeError};
use crate::gadgets::k_field::{k_add as k_add_raw, k_scalar_mul as k_scalar_mul_raw, KNum, KNumVar};
use crate::CircuitF;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::MeInstance;
use neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use neo_memory::mem_init::MemInit;
use neo_memory::riscv::lookups::RiscvOpcode;
use neo_memory::witness::{LutTableSpec, StepInstanceBundle};

#[derive(Clone, Debug)]
pub(crate) struct ShoutAddrPreCircuitData<'a> {
    pub claimed_sums_vars: &'a [KNumVar], // per LUT
    pub claimed_sums_vals: &'a [NeoK],
    pub finals_vars: &'a [KNumVar], // per LUT
    pub r_addr_vars: &'a [KNumVar], // shared
    pub r_addr_vals: &'a [NeoK],
}

#[derive(Clone, Debug)]
pub(crate) struct TwistAddrPreCircuitData<'a> {
    pub r_addr_vars: &'a [KNumVar],
    pub r_addr_vals: &'a [NeoK],
    pub finals_vars: &'a [KNumVar], // [read_claim_sum, write_claim_sum]
}

#[derive(Clone, Debug)]
pub(crate) struct TwistValEvalCircuitData<'a> {
    pub r_val_vars: &'a [KNumVar],
    pub r_val_vals: &'a [NeoK],
    pub claimed_sums_vars: &'a [KNumVar], // flattened per-mem (lt,total,prev_total?)
    pub claimed_sums_vals: &'a [NeoK],
    pub finals_vars: &'a [KNumVar], // flattened per-mem (lt,total,prev_total?)
}

#[derive(Clone, Debug)]
pub(crate) struct TwistTimeLaneOpeningsVars {
    pub wa_bits: Vec<KNumVar>,
    pub wa_bits_vals: Vec<NeoK>,
    pub has_write: KNumVar,
    pub has_write_val: NeoK,
    pub inc_at_write_addr: KNumVar,
    pub inc_at_write_addr_val: NeoK,
}

fn enforce_f_eq<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, a: Variable, b: Variable, label: &str) {
    cs.enforce(
        || label.to_string(),
        |lc| lc + a,
        |lc| lc + CS::one(),
        |lc| lc + b,
    );
}

fn k_add_with_hint<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &KNumVar,
    a_val: NeoK,
    b: &KNumVar,
    b_val: NeoK,
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    let out_val = a_val + b_val;
    let hint = KNum::<CircuitF>::from_neo_k(out_val);
    let out = k_add_raw(cs, a, b, Some(hint), label).map_err(SpartanBridgeError::BellpepperError)?;
    Ok((out, out_val))
}

fn k_scalar_mul_with_hint<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    k: CircuitF,
    a: &KNumVar,
    a_val: NeoK,
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    // Native scalar is in the base field; lift into K as k + 0*u.
    let k_u64 = k.to_canonical_u64();
    let k_val = NeoK::from(NeoF::from_u64(k_u64));
    let out_val = a_val * k_val;
    let hint = KNum::<CircuitF>::from_neo_k(out_val);
    let out = k_scalar_mul_raw(cs, k, a, Some(hint), label).map_err(SpartanBridgeError::BellpepperError)?;
    Ok((out, out_val))
}

fn k_neg_with_hint<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &KNumVar,
    a_val: NeoK,
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);
    k_scalar_mul_with_hint(cs, minus_one, a, a_val, label)
}

fn k_sub_with_hint<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &KNumVar,
    a_val: NeoK,
    b: &KNumVar,
    b_val: NeoK,
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    let (neg_b, neg_b_val) = k_neg_with_hint(cs, b, b_val, &format!("{label}_neg_b"))?;
    k_add_with_hint(cs, a, a_val, &neg_b, neg_b_val, label)
}

fn k_one_var<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<(KNumVar, NeoK)> {
    Ok((helpers::k_one(cs, label)?, NeoK::ONE))
}

fn k_zero_var<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<(KNumVar, NeoK)> {
    Ok((helpers::k_zero(cs, label)?, NeoK::ZERO))
}

fn y_scalar_from_y_row<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    base_b: u32,
    y_row_vars: &[KNumVar],
    y_row_vals: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    let take = neo_math::D;
    if y_row_vars.len() < take || y_row_vals.len() < take {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: y row too short for y_scalar (vars={}, vals={}, need >= {take})",
            y_row_vars.len(),
            y_row_vals.len()
        )));
    }

    let base_circ = CircuitF::from(base_b as u64);
    let bK = NeoK::from(NeoF::from_u64(base_b as u64));

    let mut acc = NeoK::ZERO;
    let mut pw = NeoK::ONE;
    for rho in 0..take {
        acc += pw * y_row_vals[rho];
        pw *= bK;
    }

    let y_scalar_var = helpers::alloc_k_from_neo(cs, acc, &format!("{label}_ys"))?;

    // Linear recomposition on each limb separately.
    cs.enforce(
        || format!("{label}_ys_c0"),
        |lc| {
            let mut res = lc;
            let mut pow = CircuitF::from(1u64);
            for rho in 0..take {
                res = res + (pow, y_row_vars[rho].c0);
                pow *= base_circ;
            }
            res
        },
        |lc| lc + CS::one(),
        |lc| lc + y_scalar_var.c0,
    );
    cs.enforce(
        || format!("{label}_ys_c1"),
        |lc| {
            let mut res = lc;
            let mut pow = CircuitF::from(1u64);
            for rho in 0..take {
                res = res + (pow, y_row_vars[rho].c1);
                pow *= base_circ;
            }
            res
        },
        |lc| lc + CS::one(),
        |lc| lc + y_scalar_var.c1,
    );

    Ok((y_scalar_var, acc))
}

// eq_bit_affine(bit, u) = bit*(2u-1) + (1-u)
fn eq_bit_affine<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    delta: CircuitF,
    bit: &KNumVar,
    bit_val: NeoK,
    u: &KNumVar,
    u_val: NeoK,
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    let (one, one_val) = k_one_var(cs, &format!("{label}_one"))?;

    let (two_u, two_u_val) = k_add_with_hint(cs, u, u_val, u, u_val, &format!("{label}_two_u"))?;
    let (two_u_minus_one, two_u_minus_one_val) =
        k_sub_with_hint(cs, &two_u, two_u_val, &one, one_val, &format!("{label}_two_u_minus_one"))?;

    let (t0, t0_val) =
        helpers::k_mul_with_hint(cs, bit, bit_val, &two_u_minus_one, two_u_minus_one_val, delta, &format!("{label}_t0"))?;

    let (one_minus_u, one_minus_u_val) = k_sub_with_hint(cs, &one, one_val, u, u_val, &format!("{label}_one_minus_u"))?;
    k_add_with_hint(cs, &t0, t0_val, &one_minus_u, one_minus_u_val, &format!("{label}_eq"))
}

fn eq_bits_prod<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    delta: CircuitF,
    bits: &[KNumVar],
    bits_val: &[NeoK],
    u: &[KNumVar],
    u_val: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    if bits.len() != u.len() || bits_val.len() != u_val.len() || bits.len() != bits_val.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: eq_bits_prod length mismatch"
        )));
    }

    let (mut acc, mut acc_val) = k_one_var(cs, &format!("{label}_init"))?;
    for i in 0..bits.len() {
        let (eq_i, eq_i_val) = eq_bit_affine(
            cs,
            delta,
            &bits[i],
            bits_val[i],
            &u[i],
            u_val[i],
            &format!("{label}_eq_{i}"),
        )?;
        let (new_acc, new_acc_val) =
            helpers::k_mul_with_hint(cs, &acc, acc_val, &eq_i, eq_i_val, delta, &format!("{label}_mul_{i}"))?;
        acc = new_acc;
        acc_val = new_acc_val;
    }
    Ok((acc, acc_val))
}

fn lt_eval<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    delta: CircuitF,
    j_prime: &[KNumVar],
    j_prime_val: &[NeoK],
    j: &[KNumVar],
    j_val: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    if j_prime.len() != j.len() || j_prime_val.len() != j_val.len() || j_prime.len() != j_prime_val.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: lt_eval length mismatch"
        )));
    }

    let ell = j.len();
    let (one, one_val) = k_one_var(cs, &format!("{label}_one"))?;

    // suffix[i] = Π_{k>=i} eq_single(j'_k, j_k)
    let mut suffix_vars: Vec<KNumVar> = Vec::with_capacity(ell + 1);
    let mut suffix_vals: Vec<NeoK> = Vec::with_capacity(ell + 1);
    suffix_vars.resize_with(ell + 1, || one.clone());
    suffix_vals.resize(ell + 1, one_val);

    for i in (0..ell).rev() {
        // eq_single = (1-a)(1-b) + a*b  (use eq_bit_affine with "bit"=a and u=b gives same).
        let (eq_i, eq_i_val) =
            eq_bit_affine(cs, delta, &j_prime[i], j_prime_val[i], &j[i], j_val[i], &format!("{label}_eq_{i}"))?;
        let (s, s_val) = helpers::k_mul_with_hint(
            cs,
            &suffix_vars[i + 1],
            suffix_vals[i + 1],
            &eq_i,
            eq_i_val,
            delta,
            &format!("{label}_suffix_{i}"),
        )?;
        suffix_vars[i] = s;
        suffix_vals[i] = s_val;
    }

    let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_acc_init"))?;
    for i in 0..ell {
        // (1 - j'_i)
        let (one_minus, one_minus_val) =
            k_sub_with_hint(cs, &one, one_val, &j_prime[i], j_prime_val[i], &format!("{label}_one_minus_{i}"))?;

        let (t0, t0_val) =
            helpers::k_mul_with_hint(cs, &one_minus, one_minus_val, &j[i], j_val[i], delta, &format!("{label}_t0_{i}"))?;
        let (t1, t1_val) = helpers::k_mul_with_hint(
            cs,
            &t0,
            t0_val,
            &suffix_vars[i + 1],
            suffix_vals[i + 1],
            delta,
            &format!("{label}_t1_{i}"),
        )?;

        let (new_acc, new_acc_val) =
            k_add_with_hint(cs, &acc, acc_val, &t1, t1_val, &format!("{label}_acc_{i}"))?;
        acc = new_acc;
        acc_val = new_acc_val;
    }

    Ok((acc, acc_val))
}

pub(crate) fn eval_mem_init_at_r_addr<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    base_b: u32,
    delta: CircuitF,
    init: &MemInit<NeoF>,
    k: usize,
    r_addr_vars: &[KNumVar],
    r_addr_vals: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    init.validate(k)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("{label}: invalid MemInit: {e:?}")))?;

    match init {
        MemInit::Zero => k_zero_var(cs, &format!("{label}_zero")),
        MemInit::Sparse(pairs) => {
            if r_addr_vars.len() > 64 && !pairs.is_empty() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: MemInit::Sparse only supports up to 64 bits (got {})",
                    r_addr_vars.len()
                )));
            }
            if r_addr_vars.len() != r_addr_vals.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: r_addr length mismatch"
                )));
            }

            let (one, one_val) = k_one_var(cs, &format!("{label}_one"))?;
            let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_acc"))?;

            for (pair_idx, (addr, val_f)) in pairs.iter().enumerate() {
                let mut chi_var = one.clone();
                let mut chi_val = one_val;
                for (bit_idx, (r_var, &r_val)) in r_addr_vars.iter().zip(r_addr_vals.iter()).enumerate() {
                    let bit = ((*addr >> bit_idx) & 1) as u8;
                    let (factor_var, factor_val) = if bit == 1 {
                        (r_var.clone(), r_val)
                    } else {
                        k_sub_with_hint(
                            cs,
                            &one,
                            one_val,
                            r_var,
                            r_val,
                            &format!("{label}_pair_{pair_idx}_bit_{bit_idx}_1_minus_r"),
                        )?
                    };
                    let (new_chi, new_chi_val) = helpers::k_mul_with_hint(
                        cs,
                        &chi_var,
                        chi_val,
                        &factor_var,
                        factor_val,
                        delta,
                        &format!("{label}_pair_{pair_idx}_bit_{bit_idx}_mul"),
                    )?;
                    chi_var = new_chi;
                    chi_val = new_chi_val;
                }

                // Multiply by val (lifted to K). Since val is in the base field, use scalar mul.
                let val_circ = CircuitF::from(val_f.as_canonical_u64());
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, val_circ, &chi_var, chi_val, &format!("{label}_pair_{pair_idx}_scale"))?;

                let (new_acc, new_acc_val) =
                    k_add_with_hint(cs, &acc, acc_val, &scaled, scaled_val, &format!("{label}_pair_{pair_idx}_add"))?;
                acc = new_acc;
                acc_val = new_acc_val;
            }

            // Keep base_b in the signature for symmetry with other helpers (not used here).
            let _ = base_b;
            Ok((acc, acc_val))
        }
    }
}

fn eval_riscv_opcode_mle<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    delta: CircuitF,
    opcode: RiscvOpcode,
    xlen: usize,
    r_vars: &[KNumVar],
    r_vals: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    if r_vars.len() != r_vals.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: r length mismatch"
        )));
    }
    if r_vars.len() != 2 * xlen {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: expected r.len() == 2*xlen (got {}, xlen={})",
            r_vars.len(),
            xlen
        )));
    }

    // Native value (for witness hints).
    let native_val = neo_memory::riscv::lookups::evaluate_opcode_mle(opcode, r_vals, xlen);

    // Dispatch to the same closed-form formulas used in native `evaluate_opcode_mle`.
    // Note: this is only used for Shout's `table_eval_at_r_addr` terminal linkage, so we
    // implement exactly the subset supported by `RiscvAddressLookupOracleSparse::validate_spec`.
    match opcode {
        RiscvOpcode::And => {
            let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_and_acc"))?;
            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];
                let (prod, prod_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, y_i, y_val, delta, &format!("{label}_and_prod_{i}"))?;
                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) = k_scalar_mul_with_hint(
                    cs,
                    coeff,
                    &prod,
                    prod_val,
                    &format!("{label}_and_scale_{i}"),
                )?;
                let (new_acc, new_acc_val) =
                    k_add_with_hint(cs, &acc, acc_val, &scaled, scaled_val, &format!("{label}_and_add_{i}"))?;
                acc = new_acc;
                acc_val = new_acc_val;
            }
            // Host-side sanity check (should always match).
            if acc_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: AND MLE host mismatch"
                )));
            }
            Ok((acc, acc_val))
        }
        RiscvOpcode::Xor => {
            let two = CircuitF::from(2u64);

            let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_xor_acc"))?;
            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];

                // xor = x + y - 2xy
                let (xy, xy_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, y_i, y_val, delta, &format!("{label}_xor_xy_{i}"))?;
                let (two_xy, two_xy_val) =
                    k_scalar_mul_with_hint(cs, two, &xy, xy_val, &format!("{label}_xor_2xy_{i}"))?;
                let (sum_xy, sum_xy_val) =
                    k_add_with_hint(cs, x_i, x_val, y_i, y_val, &format!("{label}_xor_sum_{i}"))?;
                let (xor_bit, xor_bit_val) = k_sub_with_hint(
                    cs,
                    &sum_xy,
                    sum_xy_val,
                    &two_xy,
                    two_xy_val,
                    &format!("{label}_xor_bit_{i}"),
                )?;

                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, coeff, &xor_bit, xor_bit_val, &format!("{label}_xor_scale_{i}"))?;
                let (new_acc, new_acc_val) =
                    k_add_with_hint(cs, &acc, acc_val, &scaled, scaled_val, &format!("{label}_xor_add_{i}"))?;
                acc = new_acc;
                acc_val = new_acc_val;
            }

            if acc_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: XOR MLE host mismatch"
                )));
            }
            Ok((acc, acc_val))
        }
        RiscvOpcode::Or => {
            let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_or_acc"))?;
            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];

                // or = x + y - xy
                let (xy, xy_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, y_i, y_val, delta, &format!("{label}_or_xy_{i}"))?;
                let (sum_xy, sum_xy_val) =
                    k_add_with_hint(cs, x_i, x_val, y_i, y_val, &format!("{label}_or_sum_{i}"))?;
                let (or_bit, or_bit_val) =
                    k_sub_with_hint(cs, &sum_xy, sum_xy_val, &xy, xy_val, &format!("{label}_or_bit_{i}"))?;

                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, coeff, &or_bit, or_bit_val, &format!("{label}_or_scale_{i}"))?;
                let (new_acc, new_acc_val) =
                    k_add_with_hint(cs, &acc, acc_val, &scaled, scaled_val, &format!("{label}_or_add_{i}"))?;
                acc = new_acc;
                acc_val = new_acc_val;
            }

            if acc_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: OR MLE host mismatch"
                )));
            }
            Ok((acc, acc_val))
        }
        RiscvOpcode::Eq => {
            let (mut acc, mut acc_val) = k_one_var(cs, &format!("{label}_eq_acc"))?;
            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];
                let (eq_i, eq_i_val) =
                    eq_bit_affine(cs, delta, x_i, x_val, y_i, y_val, &format!("{label}_eq_i_{i}"))?;
                let (new_acc, new_acc_val) =
                    helpers::k_mul_with_hint(cs, &acc, acc_val, &eq_i, eq_i_val, delta, &format!("{label}_eq_mul_{i}"))?;
                acc = new_acc;
                acc_val = new_acc_val;
            }

            if acc_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: EQ MLE host mismatch"
                )));
            }
            Ok((acc, acc_val))
        }
        RiscvOpcode::Neq => {
            let (one, one_val) = k_one_var(cs, &format!("{label}_neq_one"))?;
            let (eq, eq_val) = eval_riscv_opcode_mle(
                cs,
                delta,
                RiscvOpcode::Eq,
                xlen,
                r_vars,
                r_vals,
                &format!("{label}_neq_inner"),
            )?;
            let (neq, neq_val) = k_sub_with_hint(cs, &one, one_val, &eq, eq_val, &format!("{label}_neq"))?;
            if neq_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: NEQ MLE host mismatch"
                )));
            }
            Ok((neq, neq_val))
        }
        RiscvOpcode::Sltu => {
            let (one, one_val) = k_one_var(cs, &format!("{label}_sltu_one"))?;
            let (mut lt, mut lt_val) = k_zero_var(cs, &format!("{label}_sltu_lt"))?;
            let (mut eq_prefix, mut eq_prefix_val) = (one.clone(), one_val);

            for bit in (0..xlen).rev() {
                let x_i = &r_vars[2 * bit];
                let y_i = &r_vars[2 * bit + 1];
                let x_val = r_vals[2 * bit];
                let y_val = r_vals[2 * bit + 1];

                let (one_minus_x, one_minus_x_val) =
                    k_sub_with_hint(cs, &one, one_val, x_i, x_val, &format!("{label}_sltu_1mx_{bit}"))?;
                let (t0, t0_val) =
                    helpers::k_mul_with_hint(cs, &one_minus_x, one_minus_x_val, y_i, y_val, delta, &format!("{label}_sltu_t0_{bit}"))?;
                let (t1, t1_val) = helpers::k_mul_with_hint(
                    cs,
                    &t0,
                    t0_val,
                    &eq_prefix,
                    eq_prefix_val,
                    delta,
                    &format!("{label}_sltu_t1_{bit}"),
                )?;
                let (new_lt, new_lt_val) = k_add_with_hint(cs, &lt, lt_val, &t1, t1_val, &format!("{label}_sltu_acc_{bit}"))?;
                lt = new_lt;
                lt_val = new_lt_val;

                let (eq_i, eq_i_val) =
                    eq_bit_affine(cs, delta, x_i, x_val, y_i, y_val, &format!("{label}_sltu_eq_{bit}"))?;
                let (new_eq, new_eq_val) = helpers::k_mul_with_hint(
                    cs,
                    &eq_prefix,
                    eq_prefix_val,
                    &eq_i,
                    eq_i_val,
                    delta,
                    &format!("{label}_sltu_eqp_{bit}"),
                )?;
                eq_prefix = new_eq;
                eq_prefix_val = new_eq_val;
            }

            if lt_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: SLTU MLE host mismatch"
                )));
            }
            Ok((lt, lt_val))
        }
        RiscvOpcode::Slt => {
            if xlen == 0 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: SLT xlen must be > 0"
                )));
            }

            let (one, one_val) = k_one_var(cs, &format!("{label}_slt_one"))?;
            let x_sign = &r_vars[2 * (xlen - 1)];
            let y_sign = &r_vars[2 * (xlen - 1) + 1];
            let x_sign_val = r_vals[2 * (xlen - 1)];
            let y_sign_val = r_vals[2 * (xlen - 1) + 1];

            // Unsigned lt over the full bitstring.
            let (mut lt, mut lt_val) = k_zero_var(cs, &format!("{label}_slt_lt"))?;
            let (mut eq_prefix, mut eq_prefix_val) = (one.clone(), one_val);

            for bit in (0..xlen).rev() {
                let x_i = &r_vars[2 * bit];
                let y_i = &r_vars[2 * bit + 1];
                let x_val = r_vals[2 * bit];
                let y_val = r_vals[2 * bit + 1];

                let (one_minus_x, one_minus_x_val) =
                    k_sub_with_hint(cs, &one, one_val, x_i, x_val, &format!("{label}_slt_1mx_{bit}"))?;
                let (t0, t0_val) =
                    helpers::k_mul_with_hint(cs, &one_minus_x, one_minus_x_val, y_i, y_val, delta, &format!("{label}_slt_t0_{bit}"))?;
                let (t1, t1_val) = helpers::k_mul_with_hint(
                    cs,
                    &t0,
                    t0_val,
                    &eq_prefix,
                    eq_prefix_val,
                    delta,
                    &format!("{label}_slt_t1_{bit}"),
                )?;
                let (new_lt, new_lt_val) = k_add_with_hint(cs, &lt, lt_val, &t1, t1_val, &format!("{label}_slt_acc_{bit}"))?;
                lt = new_lt;
                lt_val = new_lt_val;

                let (eq_i, eq_i_val) =
                    eq_bit_affine(cs, delta, x_i, x_val, y_i, y_val, &format!("{label}_slt_eq_{bit}"))?;
                let (new_eq, new_eq_val) = helpers::k_mul_with_hint(
                    cs,
                    &eq_prefix,
                    eq_prefix_val,
                    &eq_i,
                    eq_i_val,
                    delta,
                    &format!("{label}_slt_eqp_{bit}"),
                )?;
                eq_prefix = new_eq;
                eq_prefix_val = new_eq_val;
            }

            // x_sign - y_sign + lt
            let (x_minus_y, x_minus_y_val) =
                k_sub_with_hint(cs, x_sign, x_sign_val, y_sign, y_sign_val, &format!("{label}_slt_xmy"))?;
            let (out, out_val) = k_add_with_hint(cs, &x_minus_y, x_minus_y_val, &lt, lt_val, &format!("{label}_slt_out"))?;
            if out_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: SLT MLE host mismatch"
                )));
            }
            Ok((out, out_val))
        }
        RiscvOpcode::Add => {
            let two = CircuitF::from(2u64);
            let four = CircuitF::from(4u64);

            let (mut result, mut result_val) = k_zero_var(cs, &format!("{label}_add_res"))?;
            let (mut carry, mut carry_val) = k_zero_var(cs, &format!("{label}_add_carry"))?;

            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];

                let (xy, xy_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, y_i, y_val, delta, &format!("{label}_add_xy_{i}"))?;
                let (xc, xc_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, &carry, carry_val, delta, &format!("{label}_add_xc_{i}"))?;
                let (yc, yc_val) =
                    helpers::k_mul_with_hint(cs, y_i, y_val, &carry, carry_val, delta, &format!("{label}_add_yc_{i}"))?;
                let (xyc, xyc_val) =
                    helpers::k_mul_with_hint(cs, &xy, xy_val, &carry, carry_val, delta, &format!("{label}_add_xyc_{i}"))?;

                // sum_bit = x + y + c - 2xy - 2xc - 2yc + 4xyc
                let (t0, t0_val) = k_add_with_hint(cs, x_i, x_val, y_i, y_val, &format!("{label}_add_t0_{i}"))?;
                let (t1, t1_val) = k_add_with_hint(cs, &t0, t0_val, &carry, carry_val, &format!("{label}_add_t1_{i}"))?;

                let (two_xy, two_xy_val) =
                    k_scalar_mul_with_hint(cs, two, &xy, xy_val, &format!("{label}_add_2xy_{i}"))?;
                let (two_xc, two_xc_val) =
                    k_scalar_mul_with_hint(cs, two, &xc, xc_val, &format!("{label}_add_2xc_{i}"))?;
                let (two_yc, two_yc_val) =
                    k_scalar_mul_with_hint(cs, two, &yc, yc_val, &format!("{label}_add_2yc_{i}"))?;
                let (sub01, sub01_val) =
                    k_add_with_hint(cs, &two_xy, two_xy_val, &two_xc, two_xc_val, &format!("{label}_add_sub01_{i}"))?;
                let (sub012, sub012_val) = k_add_with_hint(
                    cs,
                    &sub01,
                    sub01_val,
                    &two_yc,
                    two_yc_val,
                    &format!("{label}_add_sub012_{i}"),
                )?;
                let (t2, t2_val) =
                    k_sub_with_hint(cs, &t1, t1_val, &sub012, sub012_val, &format!("{label}_add_t2_{i}"))?;
                let (four_xyc, four_xyc_val) =
                    k_scalar_mul_with_hint(cs, four, &xyc, xyc_val, &format!("{label}_add_4xyc_{i}"))?;
                let (sum_bit, sum_bit_val) =
                    k_add_with_hint(cs, &t2, t2_val, &four_xyc, four_xyc_val, &format!("{label}_add_sum_bit_{i}"))?;

                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, coeff, &sum_bit, sum_bit_val, &format!("{label}_add_scale_{i}"))?;
                let (new_result, new_result_val) =
                    k_add_with_hint(cs, &result, result_val, &scaled, scaled_val, &format!("{label}_add_res_add_{i}"))?;
                result = new_result;
                result_val = new_result_val;

                // carry = xy + xc + yc - 2xyc
                let (sum_xy_xc, sum_xy_xc_val) =
                    k_add_with_hint(cs, &xy, xy_val, &xc, xc_val, &format!("{label}_add_csum0_{i}"))?;
                let (sum_xy_xc_yc, sum_xy_xc_yc_val) = k_add_with_hint(
                    cs,
                    &sum_xy_xc,
                    sum_xy_xc_val,
                    &yc,
                    yc_val,
                    &format!("{label}_add_csum1_{i}"),
                )?;
                let (two_xyc, two_xyc_val) =
                    k_scalar_mul_with_hint(cs, two, &xyc, xyc_val, &format!("{label}_add_2xyc_{i}"))?;
                let (new_carry, new_carry_val) = k_sub_with_hint(
                    cs,
                    &sum_xy_xc_yc,
                    sum_xy_xc_yc_val,
                    &two_xyc,
                    two_xyc_val,
                    &format!("{label}_add_carry_{i}"),
                )?;
                carry = new_carry;
                carry_val = new_carry_val;
            }

            if result_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: ADD MLE host mismatch"
                )));
            }
            Ok((result, result_val))
        }
        RiscvOpcode::Sub => {
            let (one, one_val) = k_one_var(cs, &format!("{label}_sub_one"))?;
            let two = CircuitF::from(2u64);
            let four = CircuitF::from(4u64);

            let (mut result, mut result_val) = k_zero_var(cs, &format!("{label}_sub_res"))?;
            let mut carry = one.clone();
            let mut carry_val = one_val;

            for i in 0..xlen {
                let x_i = &r_vars[2 * i];
                let y_i = &r_vars[2 * i + 1];
                let x_val = r_vals[2 * i];
                let y_val = r_vals[2 * i + 1];

                let (y_comp, y_comp_val) =
                    k_sub_with_hint(cs, &one, one_val, y_i, y_val, &format!("{label}_sub_yc_{i}"))?;

                let (xy, xy_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, &y_comp, y_comp_val, delta, &format!("{label}_sub_xy_{i}"))?;
                let (xc, xc_val) =
                    helpers::k_mul_with_hint(cs, x_i, x_val, &carry, carry_val, delta, &format!("{label}_sub_xc_{i}"))?;
                let (yc, yc_val) = helpers::k_mul_with_hint(
                    cs,
                    &y_comp,
                    y_comp_val,
                    &carry,
                    carry_val,
                    delta,
                    &format!("{label}_sub_yc2_{i}"),
                )?;
                let (xyc, xyc_val) =
                    helpers::k_mul_with_hint(cs, &xy, xy_val, &carry, carry_val, delta, &format!("{label}_sub_xyc_{i}"))?;

                // sum_bit = x ⊕ y_comp ⊕ carry (same polynomial as ADD).
                let (t0, t0_val) =
                    k_add_with_hint(cs, x_i, x_val, &y_comp, y_comp_val, &format!("{label}_sub_t0_{i}"))?;
                let (t1, t1_val) = k_add_with_hint(cs, &t0, t0_val, &carry, carry_val, &format!("{label}_sub_t1_{i}"))?;

                let (two_xy, two_xy_val) =
                    k_scalar_mul_with_hint(cs, two, &xy, xy_val, &format!("{label}_sub_2xy_{i}"))?;
                let (two_xc, two_xc_val) =
                    k_scalar_mul_with_hint(cs, two, &xc, xc_val, &format!("{label}_sub_2xc_{i}"))?;
                let (two_yc, two_yc_val) =
                    k_scalar_mul_with_hint(cs, two, &yc, yc_val, &format!("{label}_sub_2yc_{i}"))?;
                let (sub01, sub01_val) =
                    k_add_with_hint(cs, &two_xy, two_xy_val, &two_xc, two_xc_val, &format!("{label}_sub_sub01_{i}"))?;
                let (sub012, sub012_val) = k_add_with_hint(
                    cs,
                    &sub01,
                    sub01_val,
                    &two_yc,
                    two_yc_val,
                    &format!("{label}_sub_sub012_{i}"),
                )?;
                let (t2, t2_val) =
                    k_sub_with_hint(cs, &t1, t1_val, &sub012, sub012_val, &format!("{label}_sub_t2_{i}"))?;
                let (four_xyc, four_xyc_val) =
                    k_scalar_mul_with_hint(cs, four, &xyc, xyc_val, &format!("{label}_sub_4xyc_{i}"))?;
                let (sum_bit, sum_bit_val) =
                    k_add_with_hint(cs, &t2, t2_val, &four_xyc, four_xyc_val, &format!("{label}_sub_sum_bit_{i}"))?;

                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, coeff, &sum_bit, sum_bit_val, &format!("{label}_sub_scale_{i}"))?;
                let (new_result, new_result_val) =
                    k_add_with_hint(cs, &result, result_val, &scaled, scaled_val, &format!("{label}_sub_res_add_{i}"))?;
                result = new_result;
                result_val = new_result_val;

                // carry = xy + xc + yc - 2xyc
                let (sum_xy_xc, sum_xy_xc_val) =
                    k_add_with_hint(cs, &xy, xy_val, &xc, xc_val, &format!("{label}_sub_csum0_{i}"))?;
                let (sum_xy_xc_yc, sum_xy_xc_yc_val) = k_add_with_hint(
                    cs,
                    &sum_xy_xc,
                    sum_xy_xc_val,
                    &yc,
                    yc_val,
                    &format!("{label}_sub_csum1_{i}"),
                )?;
                let (two_xyc, two_xyc_val) =
                    k_scalar_mul_with_hint(cs, two, &xyc, xyc_val, &format!("{label}_sub_2xyc_{i}"))?;
                let (new_carry, new_carry_val) = k_sub_with_hint(
                    cs,
                    &sum_xy_xc_yc,
                    sum_xy_xc_yc_val,
                    &two_xyc,
                    two_xyc_val,
                    &format!("{label}_sub_carry_{i}"),
                )?;
                carry = new_carry;
                carry_val = new_carry_val;
            }

            if result_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: SUB MLE host mismatch"
                )));
            }
            Ok((result, result_val))
        }
        RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra => {
            if !xlen.is_power_of_two() || xlen > 64 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: shift MLE requires power-of-two xlen <= 64 (got {xlen})"
                )));
            }
            let shift_bits = xlen.trailing_zeros() as usize;
            let (one, one_val) = k_one_var(cs, &format!("{label}_shift_one"))?;

            // y low bits: y_k = r[2k+1]
            let mut y_bits_vars = Vec::with_capacity(shift_bits);
            let mut y_bits_vals = Vec::with_capacity(shift_bits);
            for k in 0..shift_bits {
                y_bits_vars.push(r_vars[2 * k + 1].clone());
                y_bits_vals.push(r_vals[2 * k + 1]);
            }
            let mut one_minus_y_vars = Vec::with_capacity(shift_bits);
            let mut one_minus_y_vals = Vec::with_capacity(shift_bits);
            for (k, (y, &y_val)) in y_bits_vars.iter().zip(y_bits_vals.iter()).enumerate() {
                let (om, om_val) =
                    k_sub_with_hint(cs, &one, one_val, y, y_val, &format!("{label}_shift_1my_{k}"))?;
                one_minus_y_vars.push(om);
                one_minus_y_vals.push(om_val);
            }

            // eq_s[s] = 1 iff shamt == s (MLE over low bits).
            let mut eq_s_vars: Vec<KNumVar> = Vec::with_capacity(xlen);
            let mut eq_s_vals: Vec<NeoK> = Vec::with_capacity(xlen);
            for s in 0..xlen {
                let mut eq = one.clone();
                let mut eq_val = one_val;
                for k in 0..shift_bits {
                    let bit = ((s >> k) & 1) as u8;
                    let (f_var, f_val) = if bit == 1 {
                        (y_bits_vars[k].clone(), y_bits_vals[k])
                    } else {
                        (one_minus_y_vars[k].clone(), one_minus_y_vals[k])
                    };
                    let (new_eq, new_eq_val) =
                        helpers::k_mul_with_hint(cs, &eq, eq_val, &f_var, f_val, delta, &format!("{label}_shift_eq_{s}_{k}"))?;
                    eq = new_eq;
                    eq_val = new_eq_val;
                }
                eq_s_vars.push(eq);
                eq_s_vals.push(eq_val);
            }

            let (mut result, mut result_val) = k_zero_var(cs, &format!("{label}_shift_res"))?;

            // sign bit for SRA.
            let sign_var = r_vars[2 * (xlen - 1)].clone();
            let sign_val = r_vals[2 * (xlen - 1)];

            for i in 0..xlen {
                let (mut out_bit, mut out_bit_val) = k_zero_var(cs, &format!("{label}_shift_out_{i}"))?;
                match opcode {
                    RiscvOpcode::Sll => {
                        for s in 0..=i {
                            let x_bit = &r_vars[2 * (i - s)];
                            let x_bit_val = r_vals[2 * (i - s)];
                            let (t, t_val) = helpers::k_mul_with_hint(
                                cs,
                                &eq_s_vars[s],
                                eq_s_vals[s],
                                x_bit,
                                x_bit_val,
                                delta,
                                &format!("{label}_sll_term_{i}_{s}"),
                            )?;
                            let (new_out, new_out_val) =
                                k_add_with_hint(cs, &out_bit, out_bit_val, &t, t_val, &format!("{label}_sll_sum_{i}_{s}"))?;
                            out_bit = new_out;
                            out_bit_val = new_out_val;
                        }
                    }
                    RiscvOpcode::Srl => {
                        for s in 0..(xlen - i) {
                            let x_bit = &r_vars[2 * (i + s)];
                            let x_bit_val = r_vals[2 * (i + s)];
                            let (t, t_val) = helpers::k_mul_with_hint(
                                cs,
                                &eq_s_vars[s],
                                eq_s_vals[s],
                                x_bit,
                                x_bit_val,
                                delta,
                                &format!("{label}_srl_term_{i}_{s}"),
                            )?;
                            let (new_out, new_out_val) =
                                k_add_with_hint(cs, &out_bit, out_bit_val, &t, t_val, &format!("{label}_srl_sum_{i}_{s}"))?;
                            out_bit = new_out;
                            out_bit_val = new_out_val;
                        }
                    }
                    RiscvOpcode::Sra => {
                        for s in 0..xlen {
                            let (bit_var, bit_val) = if i + s < xlen {
                                (r_vars[2 * (i + s)].clone(), r_vals[2 * (i + s)])
                            } else {
                                (sign_var.clone(), sign_val)
                            };
                            let (t, t_val) = helpers::k_mul_with_hint(
                                cs,
                                &eq_s_vars[s],
                                eq_s_vals[s],
                                &bit_var,
                                bit_val,
                                delta,
                                &format!("{label}_sra_term_{i}_{s}"),
                            )?;
                            let (new_out, new_out_val) =
                                k_add_with_hint(cs, &out_bit, out_bit_val, &t, t_val, &format!("{label}_sra_sum_{i}_{s}"))?;
                            out_bit = new_out;
                            out_bit_val = new_out_val;
                        }
                    }
                    _ => unreachable!(),
                }

                let coeff = CircuitF::from(1u64 << i);
                let (scaled, scaled_val) =
                    k_scalar_mul_with_hint(cs, coeff, &out_bit, out_bit_val, &format!("{label}_shift_scale_{i}"))?;
                let (new_result, new_result_val) =
                    k_add_with_hint(cs, &result, result_val, &scaled, scaled_val, &format!("{label}_shift_add_{i}"))?;
                result = new_result;
                result_val = new_result_val;
            }

            if result_val != native_val {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: internal error: shift MLE host mismatch"
                )));
            }
            Ok((result, result_val))
        }
        _ => Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: unsupported RISC-V opcode for implicit Shout table: {opcode:?}"
        ))),
    }
}

fn weighted_bitness_acc<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    delta: CircuitF,
    r_cycle0: &KNumVar,
    r_cycle0_val: NeoK,
    domain_sep: u64,
    opens: &[KNumVar],
    opens_val: &[NeoK],
    label: &str,
) -> Result<(KNumVar, NeoK)> {
    if opens.len() != opens_val.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: opens length mismatch"
        )));
    }

    // base = r_cycle[0] + domain_sep (native additionally tweaks degenerates; we reject them).
    let ds_val = NeoK::from(NeoF::from_u64(domain_sep));
    let base_val = r_cycle0_val + ds_val;
    if base_val == NeoK::ZERO || base_val == NeoK::ONE || base_val == -NeoK::ONE {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{label}: bitness base degenerate (r_cycle[0] + domain_sep in {{0,±1}})"
        )));
    }

    let ds_var = helpers::k_const_from_neo(cs, ds_val, &format!("{label}_ds"))?;
    let (base_var, _base_var_val) = k_add_with_hint(cs, r_cycle0, r_cycle0_val, &ds_var, ds_val, &format!("{label}_base"))?;

    let (one, one_val) = k_one_var(cs, &format!("{label}_one"))?;

    // acc = Σ_i base^i * b_i * (b_i - 1)
    let (mut pow, mut pow_val) = (one.clone(), one_val);
    let (mut acc, mut acc_val) = k_zero_var(cs, &format!("{label}_acc0"))?;

    for (i, (b, &b_val)) in opens.iter().zip(opens_val.iter()).enumerate() {
        let (b_minus_1, b_minus_1_val) = k_sub_with_hint(cs, b, b_val, &one, one_val, &format!("{label}_b_minus_1_{i}"))?;
        let (t0, t0_val) = helpers::k_mul_with_hint(
            cs,
            b,
            b_val,
            &b_minus_1,
            b_minus_1_val,
            delta,
            &format!("{label}_bbm1_{i}"),
        )?;
        let (weighted, weighted_val) =
            helpers::k_mul_with_hint(cs, &pow, pow_val, &t0, t0_val, delta, &format!("{label}_w_{i}"))?;
        let (new_acc, new_acc_val) =
            k_add_with_hint(cs, &acc, acc_val, &weighted, weighted_val, &format!("{label}_acc_{i}"))?;
        acc = new_acc;
        acc_val = new_acc_val;

        let (new_pow, new_pow_val) =
            helpers::k_mul_with_hint(cs, &pow, pow_val, &base_var, base_val, delta, &format!("{label}_pow_{i}"))?;
        pow = new_pow;
        pow_val = new_pow_val;
    }

    Ok((acc, acc_val))
}

pub(crate) fn verify_route_a_memory_terminal_step<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    ccs_m: usize,
    ccs_t: usize,
    base_b: u32,
    delta: CircuitF,
    step_idx: usize,
    step_public: &StepInstanceBundle<Cmt, NeoF, NeoK>,
    prev_step_public: Option<&StepInstanceBundle<Cmt, NeoF, NeoK>>,
    mcs_vars: &McsInstanceVars,
    prev_mcs_c_data_vars: Option<&[Variable]>,
    ccs_out0_vars: &MeInstanceVars,
    ccs_out0_vals: &MeInstance<Cmt, NeoF, NeoK>,
    r_time_vars: &[KNumVar],
    r_time_vals: &[NeoK],
    r_cycle_vars: &[KNumVar],
    r_cycle_vals: &[NeoK],
    bt_claimed_sums_vars: &[KNumVar],
    bt_claimed_sums_vals: &[NeoK],
    bt_final_values_vars: &[KNumVar],
    shout_pre: Option<ShoutAddrPreCircuitData<'_>>,
    twist_pre: &[TwistAddrPreCircuitData<'_>],
    val_eval: Option<TwistValEvalCircuitData<'_>>,
    cpu_me_claims_val_vars: Option<&[MeInstanceVars]>,
    cpu_me_claims_val_vals: &[MeInstance<Cmt, NeoF, NeoK>],
) -> Result<Vec<TwistTimeLaneOpeningsVars>> {
    let ctx = format!("step_{step_idx}_route_a_mem_term");

    // Shared CPU bus layout (ordering only; chunk_size does not affect y_scalar_index).
    let shout_ell_addrs = step_public.lut_insts.iter().map(|inst| inst.d * inst.ell);
    let twist_ell_addrs = step_public.mem_insts.iter().map(|inst| inst.d * inst.ell);
    let cpu_bus = neo_memory::cpu::build_bus_layout_for_instances(
        ccs_m,
        step_public.mcs_inst.m_in,
        /*chunk_size=*/ 1,
        shout_ell_addrs,
        twist_ell_addrs,
    )
    .map_err(|e| SpartanBridgeError::InvalidInput(format!("{ctx}: shared_cpu_bus layout failed: {e}")))?;

    if cpu_bus.shout_cols.len() != step_public.lut_insts.len() || cpu_bus.twist_cols.len() != step_public.mem_insts.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: shared_cpu_bus instance count mismatch"
        )));
    }

    // chi_cycle_at_r_time = eq_points(r_time, r_cycle)
    let (chi_cycle, chi_cycle_val) = {
        if r_time_vars.len() != r_cycle_vars.len() || r_time_vals.len() != r_cycle_vals.len() || r_time_vars.len() != r_time_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!("{ctx}: r_time/r_cycle length mismatch")));
        }
        // Use eq_bits_prod with "bits"=r_time and u=r_cycle (eq_single matches).
        eq_bits_prod(
            cs,
            delta,
            r_time_vars,
            r_time_vals,
            r_cycle_vars,
            r_cycle_vals,
            &format!("{ctx}_chi_cycle"),
        )?
    };

    // Enforce CPU ME at time lane uses the shared r_time.
    if ccs_out0_vars.r.len() != r_time_vars.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: ccs_out0.r length mismatch"
        )));
    }
    for i in 0..r_time_vars.len() {
        helpers::enforce_k_eq(
            cs,
            &ccs_out0_vars.r[i],
            &r_time_vars[i],
            &format!("{ctx}_ccs_out0_r_eq_r_time_{i}"),
        );
    }

    // bus_y_base = y_len - bus_cols (matches native `y_scalars.len() - bus_cols` in shared-bus mode).
    let bus_y_base_time = if cpu_bus.bus_cols > 0 {
        ccs_out0_vals
            .y
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("{ctx}: CPU y too short for bus openings")))? 
    } else {
        0usize
    };
    if cpu_bus.bus_cols > 0 && bus_y_base_time != ccs_t {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: expected bus_y_base_time == ccs_t ({bus_y_base_time} != {ccs_t})"
        )));
    }

    // Claim schedule indices inside batched_time.
    let claim_plan = RouteATimeClaimPlan::build(step_public, 1)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("{ctx}: claim plan build failed: {e:?}")))?;
    if claim_plan.claim_idx_end > bt_final_values_vars.len() || claim_plan.claim_idx_end > bt_claimed_sums_vars.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: batched_time vectors too short for claim plan"
        )));
    }

    // --------------------------------------------------------------------
    // Shout time-lane terminals.
    // --------------------------------------------------------------------
    if step_public.lut_insts.is_empty() {
        if shout_pre.is_some() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: shout_pre present but no LUT instances"
            )));
        }
    } else {
        let Some(shout_pre) = shout_pre else {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: missing shout_pre circuit data"
            )));
        };
        if shout_pre.claimed_sums_vars.len() != step_public.lut_insts.len()
            || shout_pre.finals_vars.len() != step_public.lut_insts.len()
            || shout_pre.claimed_sums_vals.len() != shout_pre.claimed_sums_vars.len()
        {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: shout_pre length mismatch"
            )));
        }

        for (lut_idx, inst) in step_public.lut_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: shared bus requires metadata-only Shout instances (lut_idx={lut_idx})"
                )));
            }
            let layout = inst.shout_layout();
            let ell_addr = layout.ell_addr;

            let shout_cols = cpu_bus.shout_cols.get(lut_idx).ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!("{ctx}: missing shout_cols[{lut_idx}]"))
            })?;
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: shout bus layout mismatch at lut_idx={lut_idx}"
                )));
            }
            if shout_pre.r_addr_vars.len() != ell_addr || shout_pre.r_addr_vals.len() != ell_addr {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: shout r_addr length mismatch (lut_idx={lut_idx})"
                )));
            }

            // Open addr_bits, has_lookup, val from CPU ME time lane.
            let mut addr_bits_open = Vec::with_capacity(ell_addr);
            let mut addr_bits_open_val = Vec::with_capacity(ell_addr);
            for (j, col_id) in shout_cols.addr_bits.clone().enumerate() {
                let y_idx = bus_y_base_time + col_id;
                let (open_var, open_val) = y_scalar_from_y_row(
                    cs,
                    base_b,
                    &ccs_out0_vars.y[y_idx],
                    &ccs_out0_vals.y[y_idx],
                    &format!("{ctx}_shout_{lut_idx}_addr_bit_{j}"),
                )?;
                addr_bits_open.push(open_var);
                addr_bits_open_val.push(open_val);
            }
            let (has_lookup_open, has_lookup_open_val) = {
                let y_idx = bus_y_base_time + shout_cols.has_lookup;
                y_scalar_from_y_row(
                    cs,
                    base_b,
                    &ccs_out0_vars.y[y_idx],
                    &ccs_out0_vals.y[y_idx],
                    &format!("{ctx}_shout_{lut_idx}_has_lookup"),
                )?
            };
            let (val_open, val_open_val) = {
                let y_idx = bus_y_base_time + shout_cols.val;
                y_scalar_from_y_row(
                    cs,
                    base_b,
                    &ccs_out0_vars.y[y_idx],
                    &ccs_out0_vals.y[y_idx],
                    &format!("{ctx}_shout_{lut_idx}_val"),
                )?
            };

            let shout_claims = claim_plan.shout.get(lut_idx).ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!("{ctx}: missing shout claim schedule at lut_idx={lut_idx}"))
            })?;

            // bitness over (addr_bits, has_lookup)
            {
                let mut opens = Vec::with_capacity(ell_addr + 1);
                let mut opens_val = Vec::with_capacity(ell_addr + 1);
                opens.extend(addr_bits_open.iter().cloned());
                opens_val.extend(addr_bits_open_val.iter().copied());
                opens.push(has_lookup_open.clone());
                opens_val.push(has_lookup_open_val);

                let (acc, acc_val) = weighted_bitness_acc(
                    cs,
                    delta,
                    &r_cycle_vars[0],
                    r_cycle_vals[0],
                    0x5348_4F55_54u64 + lut_idx as u64,
                    &opens,
                    &opens_val,
                    &format!("{ctx}_shout_{lut_idx}_bitness"),
                )?;
                let (expected, expected_val) = helpers::k_mul_with_hint(
                    cs,
                    &chi_cycle,
                    chi_cycle_val,
                    &acc,
                    acc_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_bitness_expected"),
                )?;
                // expected == batched_final_values[bitness]
                helpers::enforce_k_eq(
                    cs,
                    &expected,
                    &bt_final_values_vars[shout_claims.bitness],
                    &format!("{ctx}_shout_{lut_idx}_bitness_final"),
                );
                let _ = expected_val;
            }

            // value terminal: chi_cycle * has_lookup * val
            {
                let (t0, t0_val) = helpers::k_mul_with_hint(
                    cs,
                    &has_lookup_open,
                    has_lookup_open_val,
                    &val_open,
                    val_open_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_value_inner"),
                )?;
                let (expected, _expected_val) = helpers::k_mul_with_hint(
                    cs,
                    &chi_cycle,
                    chi_cycle_val,
                    &t0,
                    t0_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_value_expected"),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &expected,
                    &bt_final_values_vars[shout_claims.value],
                    &format!("{ctx}_shout_{lut_idx}_value_final"),
                );
            }

            // adapter terminal: chi_cycle * has_lookup * eq_addr
            {
                let (eq_addr, eq_addr_val) = eq_bits_prod(
                    cs,
                    delta,
                    addr_bits_open.as_slice(),
                    addr_bits_open_val.as_slice(),
                    shout_pre.r_addr_vars,
                    shout_pre.r_addr_vals,
                    &format!("{ctx}_shout_{lut_idx}_eq_addr"),
                )?;
                let (t0, t0_val) = helpers::k_mul_with_hint(
                    cs,
                    &has_lookup_open,
                    has_lookup_open_val,
                    &eq_addr,
                    eq_addr_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_adapter_inner"),
                )?;
                let (expected, _expected_val) = helpers::k_mul_with_hint(
                    cs,
                    &chi_cycle,
                    chi_cycle_val,
                    &t0,
                    t0_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_adapter_expected"),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &expected,
                    &bt_final_values_vars[shout_claims.adapter],
                    &format!("{ctx}_shout_{lut_idx}_adapter_final"),
                );
            }

            // value_claim == addr_claim_sum
            helpers::enforce_k_eq(
                cs,
                &bt_claimed_sums_vars[shout_claims.value],
                &shout_pre.claimed_sums_vars[lut_idx],
                &format!("{ctx}_shout_{lut_idx}_value_claim_eq_addr_claim"),
            );

            // addr_final == table_eval_at_r_addr * adapter_claim
            {
                let table_eval = match &inst.table_spec {
                    None => {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "{ctx}: Shout LUT table_spec=None not supported in compression profile (lut_idx={lut_idx})"
                        )))
                    }
                    Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                        let xlen = *xlen;
                        if shout_pre.r_addr_vars.len() != 2 * xlen {
                            return Err(SpartanBridgeError::InvalidInput(format!(
                                "{ctx}: RiscvOpcode LUT expects ell_addr=2*xlen (lut_idx={lut_idx})"
                            )));
                        }
                        eval_riscv_opcode_mle(
                            cs,
                            delta,
                            *opcode,
                            xlen,
                            shout_pre.r_addr_vars,
                            shout_pre.r_addr_vals,
                            &format!("{ctx}_shout_{lut_idx}_table_eval"),
                        )?
                    }
                };

                let adapter_claim_val = bt_claimed_sums_vals[shout_claims.adapter];
                let (expected_addr_final, _expected_addr_final_val) = helpers::k_mul_with_hint(
                    cs,
                    &table_eval.0,
                    table_eval.1,
                    &bt_claimed_sums_vars[shout_claims.adapter],
                    adapter_claim_val,
                    delta,
                    &format!("{ctx}_shout_{lut_idx}_addr_final_expected"),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &expected_addr_final,
                    &shout_pre.finals_vars[lut_idx],
                    &format!("{ctx}_shout_{lut_idx}_addr_final_eq"),
                );
            }
        }
    }

    // --------------------------------------------------------------------
    // Twist time-lane terminals + record openings for output binding.
    // --------------------------------------------------------------------
    if twist_pre.len() != step_public.mem_insts.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: twist_pre length mismatch"
        )));
    }

    let mut twist_time_openings: Vec<TwistTimeLaneOpeningsVars> = Vec::with_capacity(step_public.mem_insts.len());
    for (mem_idx, inst) in step_public.mem_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: shared bus requires metadata-only Twist instances (mem_idx={mem_idx})"
            )));
        }
        let layout = inst.twist_layout();
        let ell_addr = layout.ell_addr;

        let twist_cols = cpu_bus.twist_cols.get(mem_idx).ok_or_else(|| {
            SpartanBridgeError::InvalidInput(format!("{ctx}: missing twist_cols[{mem_idx}]"))
        })?;
        if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: twist bus layout mismatch at mem_idx={mem_idx}"
            )));
        }

        let pre = &twist_pre[mem_idx];
        if pre.r_addr_vars.len() != ell_addr || pre.r_addr_vals.len() != ell_addr {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: twist r_addr length mismatch at mem_idx={mem_idx}"
            )));
        }
        if pre.finals_vars.len() != 2 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: twist addr-pre finals malformed at mem_idx={mem_idx}"
            )));
        }

        let bus_y_base_time = bus_y_base_time;
        let mut ra_bits_open = Vec::with_capacity(ell_addr);
        let mut ra_bits_open_val = Vec::with_capacity(ell_addr);
        for (j, col_id) in twist_cols.ra_bits.clone().enumerate() {
            let y_idx = bus_y_base_time + col_id;
            let (open_var, open_val) = y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_ra_bit_{j}"),
            )?;
            ra_bits_open.push(open_var);
            ra_bits_open_val.push(open_val);
        }
        let mut wa_bits_open = Vec::with_capacity(ell_addr);
        let mut wa_bits_open_val = Vec::with_capacity(ell_addr);
        for (j, col_id) in twist_cols.wa_bits.clone().enumerate() {
            let y_idx = bus_y_base_time + col_id;
            let (open_var, open_val) = y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_wa_bit_{j}"),
            )?;
            wa_bits_open.push(open_var);
            wa_bits_open_val.push(open_val);
        }
        let (has_read_open, has_read_open_val) = {
            let y_idx = bus_y_base_time + twist_cols.has_read;
            y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_has_read"),
            )?
        };
        let (has_write_open, has_write_open_val) = {
            let y_idx = bus_y_base_time + twist_cols.has_write;
            y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_has_write"),
            )?
        };
        let (wv_open, wv_open_val) = {
            let y_idx = bus_y_base_time + twist_cols.wv;
            y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_wv"),
            )?
        };
        let (rv_open, rv_open_val) = {
            let y_idx = bus_y_base_time + twist_cols.rv;
            y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_rv"),
            )?
        };
        let (inc_write_open, inc_write_open_val) = {
            let y_idx = bus_y_base_time + twist_cols.inc;
            y_scalar_from_y_row(
                cs,
                base_b,
                &ccs_out0_vars.y[y_idx],
                &ccs_out0_vals.y[y_idx],
                &format!("{ctx}_twist_{mem_idx}_inc"),
            )?
        };

        let twist_claims = claim_plan.twist.get(mem_idx).ok_or_else(|| {
            SpartanBridgeError::InvalidInput(format!("{ctx}: missing twist claim schedule at mem_idx={mem_idx}"))
        })?;

        // claimed sums must match addr-pre finals (read_check/write_check).
        helpers::enforce_k_eq(
            cs,
            &bt_claimed_sums_vars[twist_claims.read_check],
            &pre.finals_vars[0],
            &format!("{ctx}_twist_{mem_idx}_read_claim_eq_pre"),
        );
        helpers::enforce_k_eq(
            cs,
            &bt_claimed_sums_vars[twist_claims.write_check],
            &pre.finals_vars[1],
            &format!("{ctx}_twist_{mem_idx}_write_claim_eq_pre"),
        );

        // bitness over (ra_bits, wa_bits, has_read, has_write)
        {
            let mut opens = Vec::with_capacity(2 * ell_addr + 2);
            let mut opens_val = Vec::with_capacity(2 * ell_addr + 2);
            opens.extend(ra_bits_open.iter().cloned());
            opens_val.extend(ra_bits_open_val.iter().copied());
            opens.extend(wa_bits_open.iter().cloned());
            opens_val.extend(wa_bits_open_val.iter().copied());
            opens.push(has_read_open.clone());
            opens_val.push(has_read_open_val);
            opens.push(has_write_open.clone());
            opens_val.push(has_write_open_val);

            let (acc, acc_val) = weighted_bitness_acc(
                cs,
                delta,
                &r_cycle_vars[0],
                r_cycle_vals[0],
                0x5457_4953_54u64 + mem_idx as u64,
                &opens,
                &opens_val,
                &format!("{ctx}_twist_{mem_idx}_bitness"),
            )?;
            let (expected, _expected_val) = helpers::k_mul_with_hint(
                cs,
                &chi_cycle,
                chi_cycle_val,
                &acc,
                acc_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_bitness_expected"),
            )?;
            helpers::enforce_k_eq(
                cs,
                &expected,
                &bt_final_values_vars[twist_claims.bitness],
                &format!("{ctx}_twist_{mem_idx}_bitness_final"),
            );
        }

        // init_at_r_addr + claimed_inc_sum_lt
        let Some(val_eval) = val_eval.as_ref() else {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: missing val_eval circuit data (mem present)"
            )));
        };
        let claims_per_mem = if step_idx > 0 { 3 } else { 2 };
        let base = claims_per_mem * mem_idx;
        if base + 1 >= val_eval.claimed_sums_vars.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: val_eval claimed_sums too short (mem_idx={mem_idx})"
            )));
        }

        let (init_at_r_addr, init_at_r_addr_val) = eval_mem_init_at_r_addr(
            cs,
            base_b,
            delta,
            &inst.init,
            inst.k,
            pre.r_addr_vars,
            pre.r_addr_vals,
            &format!("{ctx}_twist_{mem_idx}_init"),
        )?;
        let claimed_inc_sum_lt_var = &val_eval.claimed_sums_vars[base];
        let claimed_inc_sum_lt_val = val_eval.claimed_sums_vals[base];
        let (claimed_val, claimed_val_val) = k_add_with_hint(
            cs,
            &init_at_r_addr,
            init_at_r_addr_val,
            claimed_inc_sum_lt_var,
            claimed_inc_sum_lt_val,
            &format!("{ctx}_twist_{mem_idx}_claimed_val"),
        )?;

        let (read_eq_addr, read_eq_addr_val) = eq_bits_prod(
            cs,
            delta,
            ra_bits_open.as_slice(),
            ra_bits_open_val.as_slice(),
            pre.r_addr_vars,
            pre.r_addr_vals,
            &format!("{ctx}_twist_{mem_idx}_eq_ra"),
        )?;
        let (write_eq_addr, write_eq_addr_val) = eq_bits_prod(
            cs,
            delta,
            wa_bits_open.as_slice(),
            wa_bits_open_val.as_slice(),
            pre.r_addr_vars,
            pre.r_addr_vals,
            &format!("{ctx}_twist_{mem_idx}_eq_wa"),
        )?;

        // read terminal
        {
            let (claimed_minus_rv, claimed_minus_rv_val) = k_sub_with_hint(
                cs,
                &claimed_val,
                claimed_val_val,
                &rv_open,
                rv_open_val,
                &format!("{ctx}_twist_{mem_idx}_claimed_minus_rv"),
            )?;
            let (t0, t0_val) = helpers::k_mul_with_hint(
                cs,
                &has_read_open,
                has_read_open_val,
                &claimed_minus_rv,
                claimed_minus_rv_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_read_t0"),
            )?;
            let (t1, t1_val) = helpers::k_mul_with_hint(
                cs,
                &t0,
                t0_val,
                &read_eq_addr,
                read_eq_addr_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_read_t1"),
            )?;
            let (expected, _expected_val) = helpers::k_mul_with_hint(
                cs,
                &chi_cycle,
                chi_cycle_val,
                &t1,
                t1_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_read_expected"),
            )?;
            helpers::enforce_k_eq(
                cs,
                &expected,
                &bt_final_values_vars[twist_claims.read_check],
                &format!("{ctx}_twist_{mem_idx}_read_final"),
            );
        }

        // write terminal
        {
            let (wv_minus_claimed, wv_minus_claimed_val) = k_sub_with_hint(
                cs,
                &wv_open,
                wv_open_val,
                &claimed_val,
                claimed_val_val,
                &format!("{ctx}_twist_{mem_idx}_wv_minus_claimed"),
            )?;
            let (wv_minus_claimed_minus_inc, wv_minus_claimed_minus_inc_val) = k_sub_with_hint(
                cs,
                &wv_minus_claimed,
                wv_minus_claimed_val,
                &inc_write_open,
                inc_write_open_val,
                &format!("{ctx}_twist_{mem_idx}_wv_minus_claimed_minus_inc"),
            )?;
            let (t0, t0_val) = helpers::k_mul_with_hint(
                cs,
                &has_write_open,
                has_write_open_val,
                &wv_minus_claimed_minus_inc,
                wv_minus_claimed_minus_inc_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_write_t0"),
            )?;
            let (t1, t1_val) = helpers::k_mul_with_hint(
                cs,
                &t0,
                t0_val,
                &write_eq_addr,
                write_eq_addr_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_write_t1"),
            )?;
            let (expected, _expected_val) = helpers::k_mul_with_hint(
                cs,
                &chi_cycle,
                chi_cycle_val,
                &t1,
                t1_val,
                delta,
                &format!("{ctx}_twist_{mem_idx}_write_expected"),
            )?;
	            helpers::enforce_k_eq(
	                cs,
	                &expected,
	                &bt_final_values_vars[twist_claims.write_check],
	                &format!("{ctx}_twist_{mem_idx}_write_final"),
	            );
	        }

	        twist_time_openings.push(TwistTimeLaneOpeningsVars {
	            wa_bits: wa_bits_open,
	            wa_bits_vals: wa_bits_open_val,
	            has_write: has_write_open,
	            has_write_val: has_write_open_val,
	            inc_at_write_addr: inc_write_open,
	            inc_at_write_addr_val: inc_write_open_val,
	        });
	    }

	    // --------------------------------------------------------------------
	    // Twist val-eval terminals at r_val + rollover.
	    // --------------------------------------------------------------------
	    if step_public.mem_insts.is_empty() {
	        if val_eval.is_some() || cpu_me_claims_val_vars.is_some() || !cpu_me_claims_val_vals.is_empty() {
	            return Err(SpartanBridgeError::InvalidInput(format!(
	                "{ctx}: unexpected val-eval artifacts with no mem instances"
	            )));
	        }
	        return Ok(twist_time_openings);
	    }

    let Some(val_eval) = val_eval else {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: missing val_eval circuit data"
        )));
    };
    let Some(cpu_me_vars) = cpu_me_claims_val_vars else {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: missing cpu_me_claims_val vars"
        )));
    };

    let has_prev = prev_step_public.is_some();
    let expected_cpu_me = 1usize + usize::from(has_prev);
    if cpu_me_vars.len() != expected_cpu_me || cpu_me_claims_val_vals.len() != expected_cpu_me {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: cpu_me_claims_val count mismatch"
        )));
    }

    // Commitment binding for CPU ME(val) claims (current + optional prev).
    if cpu_me_vars[0].c_data.len() != mcs_vars.c_data.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: cpu_me_cur.c_data length mismatch vs mcs_inst.c"
        )));
    }
    for (i, (&a, &b)) in cpu_me_vars[0].c_data.iter().zip(mcs_vars.c_data.iter()).enumerate() {
        enforce_f_eq(cs, a, b, &format!("{ctx}_cpu_me_val_cur_c_eq_mcs_c_{i}"));
    }
    if has_prev {
        let Some(prev_c_data) = prev_mcs_c_data_vars else {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: missing prev_mcs_c_data_vars with has_prev"
            )));
        };
        if cpu_me_vars[1].c_data.len() != prev_c_data.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: cpu_me_prev.c_data length mismatch vs prev mcs_inst.c"
            )));
        }
        for (i, (&a, &b)) in cpu_me_vars[1].c_data.iter().zip(prev_c_data.iter()).enumerate() {
            enforce_f_eq(cs, a, b, &format!("{ctx}_cpu_me_val_prev_c_eq_prev_mcs_c_{i}"));
        }
    }

    // lt = lt_eval(r_val, r_time)
    let (lt, lt_val) = lt_eval(
        cs,
        delta,
        val_eval.r_val_vars,
        val_eval.r_val_vals,
        r_time_vars,
        r_time_vals,
        &format!("{ctx}_lt_eval"),
    )?;

    let bus_y_base_val = cpu_me_claims_val_vals[0]
        .y
        .len()
        .checked_sub(cpu_bus.bus_cols)
        .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("{ctx}: CPU y(val) too short for bus openings")))?;

    for (mem_idx, inst) in step_public.mem_insts.iter().enumerate() {
        let layout = inst.twist_layout();
        let ell_addr = layout.ell_addr;
        let twist_cols = cpu_bus.twist_cols.get(mem_idx).ok_or_else(|| {
            SpartanBridgeError::InvalidInput(format!("{ctx}: missing twist_cols[{mem_idx}]"))
        })?;

        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * mem_idx;
        if base + (claims_per_mem - 1) >= val_eval.finals_vars.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: val_eval finals too short (mem_idx={mem_idx})"
            )));
        }

        // Open wa_bits/has_write/inc at r_val from cpu_me_cur.
        let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
        let mut wa_bits_val_open_val = Vec::with_capacity(ell_addr);
        for (j, col_id) in twist_cols.wa_bits.clone().enumerate() {
            let y_idx = bus_y_base_val + col_id;
            let (open_var, open_val) = y_scalar_from_y_row(
                cs,
                base_b,
                &cpu_me_vars[0].y[y_idx],
                &cpu_me_claims_val_vals[0].y[y_idx],
                &format!("{ctx}_val_cur_mem{mem_idx}_wa_{j}"),
            )?;
            wa_bits_val_open.push(open_var);
            wa_bits_val_open_val.push(open_val);
        }
        let (has_write_val_open, has_write_val_open_val) = {
            let y_idx = bus_y_base_val + twist_cols.has_write;
            y_scalar_from_y_row(
                cs,
                base_b,
                &cpu_me_vars[0].y[y_idx],
                &cpu_me_claims_val_vals[0].y[y_idx],
                &format!("{ctx}_val_cur_mem{mem_idx}_has_write"),
            )?
        };
        let (inc_val_open, inc_val_open_val) = {
            let y_idx = bus_y_base_val + twist_cols.inc;
            y_scalar_from_y_row(
                cs,
                base_b,
                &cpu_me_vars[0].y[y_idx],
                &cpu_me_claims_val_vals[0].y[y_idx],
                &format!("{ctx}_val_cur_mem{mem_idx}_inc"),
            )?
        };

        let r_addr_vars = twist_pre[mem_idx].r_addr_vars;
        let r_addr_vals = twist_pre[mem_idx].r_addr_vals;
        let (eq_wa, eq_wa_val) = eq_bits_prod(
            cs,
            delta,
            wa_bits_val_open.as_slice(),
            wa_bits_val_open_val.as_slice(),
            r_addr_vars,
            r_addr_vals,
            &format!("{ctx}_val_cur_mem{mem_idx}_eq_wa"),
        )?;

        let (t0, t0_val) = helpers::k_mul_with_hint(
            cs,
            &has_write_val_open,
            has_write_val_open_val,
            &inc_val_open,
            inc_val_open_val,
            delta,
            &format!("{ctx}_val_cur_mem{mem_idx}_t0"),
        )?;
        let (inc_at_r_addr_val, inc_at_r_addr_val_val) = helpers::k_mul_with_hint(
            cs,
            &t0,
            t0_val,
            &eq_wa,
            eq_wa_val,
            delta,
            &format!("{ctx}_val_cur_mem{mem_idx}_inc_at_r_addr"),
        )?;

        // expected_lt_final = inc_at_r_addr_val * lt
        {
            let (expected_lt_final, _expected_lt_final_val) = helpers::k_mul_with_hint(
                cs,
                &inc_at_r_addr_val,
                inc_at_r_addr_val_val,
                &lt,
                lt_val,
                delta,
                &format!("{ctx}_val_cur_mem{mem_idx}_expected_lt"),
            )?;
            helpers::enforce_k_eq(
                cs,
                &expected_lt_final,
                &val_eval.finals_vars[base],
                &format!("{ctx}_val_cur_mem{mem_idx}_lt_final_eq"),
            );
        }
        // expected_total_final = inc_at_r_addr_val
        helpers::enforce_k_eq(
            cs,
            &inc_at_r_addr_val,
            &val_eval.finals_vars[base + 1],
            &format!("{ctx}_val_cur_mem{mem_idx}_total_final_eq"),
        );

        if has_prev {
            // Prev openings at current r_val.
            let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
            let mut wa_bits_prev_open_val = Vec::with_capacity(ell_addr);
            for (j, col_id) in twist_cols.wa_bits.clone().enumerate() {
                let y_idx = bus_y_base_val + col_id;
                let (open_var, open_val) = y_scalar_from_y_row(
                    cs,
                    base_b,
                    &cpu_me_vars[1].y[y_idx],
                    &cpu_me_claims_val_vals[1].y[y_idx],
                    &format!("{ctx}_val_prev_mem{mem_idx}_wa_{j}"),
                )?;
                wa_bits_prev_open.push(open_var);
                wa_bits_prev_open_val.push(open_val);
            }
            let (has_write_prev_open, has_write_prev_open_val) = {
                let y_idx = bus_y_base_val + twist_cols.has_write;
                y_scalar_from_y_row(
                    cs,
                    base_b,
                    &cpu_me_vars[1].y[y_idx],
                    &cpu_me_claims_val_vals[1].y[y_idx],
                    &format!("{ctx}_val_prev_mem{mem_idx}_has_write"),
                )?
            };
            let (inc_prev_open, inc_prev_open_val) = {
                let y_idx = bus_y_base_val + twist_cols.inc;
                y_scalar_from_y_row(
                    cs,
                    base_b,
                    &cpu_me_vars[1].y[y_idx],
                    &cpu_me_claims_val_vals[1].y[y_idx],
                    &format!("{ctx}_val_prev_mem{mem_idx}_inc"),
                )?
            };
            let (eq_wa_prev, eq_wa_prev_val) = eq_bits_prod(
                cs,
                delta,
                wa_bits_prev_open.as_slice(),
                wa_bits_prev_open_val.as_slice(),
                r_addr_vars,
                r_addr_vals,
                &format!("{ctx}_val_prev_mem{mem_idx}_eq_wa"),
            )?;
            let (t0, t0_val) = helpers::k_mul_with_hint(
                cs,
                &has_write_prev_open,
                has_write_prev_open_val,
                &inc_prev_open,
                inc_prev_open_val,
                delta,
                &format!("{ctx}_val_prev_mem{mem_idx}_t0"),
            )?;
            let (inc_at_r_addr_prev, _inc_at_r_addr_prev_val) = helpers::k_mul_with_hint(
                cs,
                &t0,
                t0_val,
                &eq_wa_prev,
                eq_wa_prev_val,
                delta,
                &format!("{ctx}_val_prev_mem{mem_idx}_inc_at_r_addr"),
            )?;
            helpers::enforce_k_eq(
                cs,
                &inc_at_r_addr_prev,
                &val_eval.finals_vars[base + 2],
                &format!("{ctx}_val_prev_mem{mem_idx}_prev_total_final_eq"),
            );

            // Rollover init equation: Init_cur(r_addr) == Init_prev(r_addr) + claimed_prev_total
            let prev_step = prev_step_public.ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!("{ctx}: missing prev_step_public with has_prev"))
            })?;
            if prev_step.mem_insts.len() != step_public.mem_insts.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: rollover requires stable mem instance count"
                )));
            }
            let prev_inst = prev_step.mem_insts.get(mem_idx).ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!("{ctx}: missing prev mem instance at {mem_idx}"))
            })?;
            if prev_inst.d != inst.d || prev_inst.ell != inst.ell || prev_inst.k != inst.k {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: rollover requires stable geometry at mem_idx={mem_idx}"
                )));
            }
            let (init_prev, init_prev_val) = eval_mem_init_at_r_addr(
                cs,
                base_b,
                delta,
                &prev_inst.init,
                prev_inst.k,
                r_addr_vars,
                r_addr_vals,
                &format!("{ctx}_roll_mem{mem_idx}_init_prev"),
            )?;
            let (init_cur, init_cur_val) = eval_mem_init_at_r_addr(
                cs,
                base_b,
                delta,
                &inst.init,
                inst.k,
                r_addr_vars,
                r_addr_vals,
                &format!("{ctx}_roll_mem{mem_idx}_init_cur"),
            )?;
            let claimed_prev_total_var = &val_eval.claimed_sums_vars[base + 2];
            let claimed_prev_total_val = val_eval.claimed_sums_vals[base + 2];
            let (rhs, _rhs_val) = k_add_with_hint(
                cs,
                &init_prev,
                init_prev_val,
                claimed_prev_total_var,
                claimed_prev_total_val,
                &format!("{ctx}_roll_mem{mem_idx}_rhs"),
            )?;
            helpers::enforce_k_eq(
                cs,
                &init_cur,
                &rhs,
                &format!("{ctx}_roll_mem{mem_idx}_eq"),
            );
            let _ = init_cur_val;
        }
    }

    Ok(twist_time_openings)
}
