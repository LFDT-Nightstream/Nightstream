#![allow(deprecated)]

use neo_spartan_bridge::{
    compress_ivc_verifier_to_lean_proof_with_linkage, verify_lean_proof,
};
use neo_spartan_bridge::me_to_r1cs::{IvcEvEmbed, IvcLinkageInputs, CommitEvoEmbed};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Witness digits (len = 8 = 2^3)
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    // Two Ajtai rows (unit vectors) so constraints become z0=c0 and z1=c1
    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];
    let dot = |row: &[F]| -> F {
        row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
            let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            acc + (*a) * zi_f
        })
    };
    let c_coords = vec![dot(&ajtai_rows[0]), dot(&ajtai_rows[1])];

    // Two weight vectors for two y outputs
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE];
    let dotf = |row: &[F]| -> F { row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
        let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
        acc + (*a) * zi_f
    })};
    let y_outputs = vec![dotf(&w0), dotf(&w1)];

    let me = MEInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c_coords,
        y_outputs,
        r_point: vec![F::from_u64(3); 2],
        base_b: 4, // digits in [-3,3]
        header_digest: [0u8; 32],
    };
    let wit = MEWitness { z_digits: z, weight_vectors: vec![w0, w1], ajtai_rows: Some(ajtai_rows) };
    (me, wit)
}

fn linkage_for_column_0(step_io_override: Option<F>, z_digits: &[i64], base_b: u64) -> IvcLinkageInputs {
    // Bind column 0 value reconstructed from its digits (r=0..upto)
    let d = neo_math::ring::D; // 54
    let upto = core::cmp::min(d, z_digits.len());
    let mut val: i128 = 0;
    let mut pow = 1i128;
    for r in 0..upto {
        let zi = z_digits[r] as i128;
        val += zi * pow;
        pow *= base_b as i128;
    }
    let expected = if let Some(x) = step_io_override { x } else {
        if val >= 0 { F::from_u64(val as u64) } else { -F::from_u64((-val) as u64) }
    };
    IvcLinkageInputs {
        x_indices_abs: vec![0],
        y_prev_indices_abs: vec![],
        const1_index_abs: None,
        step_io: vec![expected],
    }
}

#[test]
fn ev_linkage_ok_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(5);
    let y_prev = vec![F::from_u64(10), F::from_u64(20)];
    // Bind y_step to the tail of ME outputs (security linkage enabled)
    let y_step = me.y_outputs.clone();
    let y_next = vec![y_prev[0] + rho * y_step[0], y_prev[1] + rho * y_step[1]];

    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: Some(y_step.clone()),
        fold_chain_digest: None,
        acc_c_prev: None,
        acc_c_step: None,
        acc_c_next: None,
        rho_eff: None,
    };
    let link = linkage_for_column_0(None, &wit.z_digits, me.base_b as u64);

    // No commit evolution in this test
    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, Some(link))
        .expect("prove");
    assert!(verify_lean_proof(&proof).expect("verify runs"));
}

#[test]
fn linkage_attack_fails_v1() {
    let (me, wit) = tiny_me_instance();
    // Malicious: provide an obviously wrong expected value for linkage
    let rho = F::from_u64(3);
    let y_prev = vec![F::from_u64(7), F::from_u64(9)];
    let y_step = vec![F::from_u64(1), F::from_u64(0)];
    let y_next = vec![y_prev[0] + rho * y_step[0], y_prev[1] + rho * y_step[1]];
    let ev = IvcEvEmbed { rho, y_prev, y_next, y_step_public: Some(y_step), fold_chain_digest: None,
        acc_c_prev: None, acc_c_step: None, acc_c_next: None, rho_eff: None };
    // Linkage expects a wrong constant to force failure
    let link = linkage_for_column_0(Some(F::from_u64(123456)), &wit.z_digits, me.base_b as u64);

    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, Some(link))
        .expect("prove");
    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(!ok, "verification must fail when linkage constraints are violated");
}

#[test]
fn y_step_public_mismatch_ev_fails_v1() {
    let (me, wit) = tiny_me_instance();
    // Honest step
    let rho = F::from_u64(7);
    let y_prev = vec![F::from_u64(1), F::from_u64(2)];
    let y_step_honest = me.y_outputs.clone();
    let y_next = vec![
        y_prev[0] + rho * y_step_honest[0],
        y_prev[1] + rho * y_step_honest[1],
    ];

    // Tamper y_step_public only (keep ρ, y_prev, y_next fixed)
    let mut y_step_bad = y_step_honest.clone();
    y_step_bad[0] += F::ONE;
    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: Some(y_step_bad),
        fold_chain_digest: None,
        acc_c_prev: None,
        acc_c_step: None,
        acc_c_next: None,
        rho_eff: None,
    };

    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");
    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(
        !ok,
        "verification must fail when y_step_public mismatches EV relation"
    );
}

#[test]
fn rho_flip_with_digest_fails_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(9);
    let y_prev = vec![F::from_u64(3), F::from_u64(4)];
    let y_step = me.y_outputs.clone();
    let y_next = vec![y_prev[0] + rho * y_step[0], y_prev[1] + rho * y_step[1]];
    let fold_digest = [0x55u8; 32];

    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: Some(y_step.clone()),
        fold_chain_digest: Some(fold_digest),
        acc_c_prev: None,
        acc_c_step: None,
        acc_c_next: None,
        rho_eff: None,
    };
    let mut proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");

    // Compute offset to rho scalar in public IO with y_step_public present
    let scalars_before_rho = me.c_coords.len()
        + (me.y_outputs.len() + y_step.len())
        + me.r_point.len()
        + 1 /*base_b*/
        + y_prev.len()
        + y_next.len();
    let byte_off = scalars_before_rho * 8;
    assert!(proof.public_io_bytes.len() >= byte_off + 8, "public io too short for rho");
    // Flip a byte in rho
    proof.public_io_bytes[byte_off] ^= 0xA5;

    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(
        !ok,
        "verification must fail when rho is tampered while digest stays bound"
    );
}

#[test]
fn ajtai_step_linkage_digit_tamper_fails_v1() {
    let (me, mut wit) = tiny_me_instance();
    // Bind Ajtai to acc_c_step; start from honest step = current c_coords
    let acc_c_step = me.c_coords.clone();
    let rho = F::from_u64(5);
    let y_prev = vec![F::from_u64(1), F::from_u64(2)];
    let y_step = me.y_outputs.clone();
    let y_next = vec![y_prev[0] + rho * y_step[0], y_prev[1] + rho * y_step[1]];
    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: Some(y_step),
        fold_chain_digest: None,
        acc_c_prev: None,
        acc_c_step: Some(acc_c_step),
        acc_c_next: None,
        rho_eff: None,
    };

    // Tamper one Ajtai digit in the witness (break ⟨L_i, z⟩ = acc_c_step[i])
    if !wit.z_digits.is_empty() { wit.z_digits[0] += 1; }
    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");
    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(
        !ok,
        "verification must fail when Ajtai/step linkage is violated by digit tamper"
    );
}

#[test]
fn commit_evo_bad_c_next_fails_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(3);
    let y_prev = vec![F::from_u64(1), F::from_u64(2)];
    let y_step = me.y_outputs.clone();
    let y_next = vec![y_prev[0] + rho * y_step[0], y_prev[1] + rho * y_step[1]];

    // Honest commit-evo inputs
    let c_prev = vec![F::from_u64(5), F::from_u64(6)];
    let c_step = vec![F::from_u64(7), F::from_u64(8)];
    let mut c_next = vec![c_prev[0] + rho * c_step[0], c_prev[1] + rho * c_step[1]];
    // Tamper c_next only
    c_next[0] += F::ONE;

    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: Some(y_step),
        fold_chain_digest: None,
        acc_c_prev: Some(c_prev),
        acc_c_step: Some(c_step),
        acc_c_next: Some(c_next),
        rho_eff: None,
    };

    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");
    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(
        !ok,
        "verification must fail when commit evolution public c_next is incorrect"
    );
}

#[test]
fn commit_evolution_ok_v1() {
    let (me, wit) = tiny_me_instance();
    // Commit evolution enforces: c_next (me.c_coords) = c_prev + rho * c_step
    // Use c_next = original Ajtai dot(row,Z), set c_step=0, c_prev=c_next so both constraints hold
    let rho = F::from_u64(11);
    let c_next = me.c_coords.clone();
    let c_prev = c_next.clone();
    let c_step = vec![F::ZERO; c_next.len()];
    let commit = CommitEvoEmbed { rho, c_prev, c_step };

    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, None, Some(commit), None)
        .expect("prove");
    assert!(verify_lean_proof(&proof).expect("verify runs"));
}

#[test]
fn commit_evo_attack_fails_v1() {
    let (me, wit) = tiny_me_instance();
    // Honest relation: choose c_prev=0, c_step=me.c_coords, rho=1 so equality holds
    let _rho = F::from_u64(1);
    let c_step = me.c_coords.clone();
    let c_prev = vec![F::ZERO; c_step.len()];

    // Attack: change rho to break equality
    let bad_rho = F::from_u64(2);
    let commit = CommitEvoEmbed { rho: bad_rho, c_prev, c_step };

    let proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, None, Some(commit), None)
        .expect("prove");
    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(!ok, "verification must fail when commit evolution is false");
}

#[test]
fn acc_vectors_public_io_flip_fails_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(3);
    let y_prev = vec![F::from_u64(1), F::from_u64(2)];
    let y_next = vec![F::from_u64(4), F::from_u64(8)];

    // Consistent acc vectors
    let c_prev = vec![F::from_u64(5), F::from_u64(6)];
    let c_step = vec![F::from_u64(7), F::from_u64(8)];
    let c_next = vec![c_prev[0] + rho * c_step[0], c_prev[1] + rho * c_step[1]];

    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: None,
        fold_chain_digest: None,
        acc_c_prev: Some(c_prev.clone()),
        acc_c_step: Some(c_step.clone()),
        acc_c_next: Some(c_next.clone()),
        rho_eff: None,
    };

    let mut proof = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");

    // Locate acc_c_prev in public IO and flip a byte
    let scalars_before_acc = me.c_coords.len() + me.y_outputs.len() + me.r_point.len()
        + 1 /*base_b*/ + y_prev.len() + y_next.len() + 1 /*rho*/;
    let byte_off = scalars_before_acc * 8;
    assert!(proof.public_io_bytes.len() >= byte_off + 8, "public io too short");
    proof.public_io_bytes[byte_off] ^= 0xA5;

    let ok = neo_spartan_bridge::verify_lean_proof(&proof).unwrap_or(false);
    assert!(!ok, "verification must fail when acc_c_prev limb is tampered in public IO");
}

#[test]
fn fold_chain_digest_flip_fails_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(5);
    let y_prev = vec![F::from_u64(2), F::from_u64(4)];
    let y_next = vec![F::from_u64(12), F::from_u64(24)];
    let fold_digest = [0x77u8; 32];

    let ev = IvcEvEmbed {
        rho,
        y_prev: y_prev.clone(),
        y_next: y_next.clone(),
        y_step_public: None,
        fold_chain_digest: Some(fold_digest),
        acc_c_prev: None,
        acc_c_step: None,
        acc_c_next: None,
        rho_eff: None,
    };
    let mut proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");

    // Locate digest start and flip a byte
    let scalars_before_digest = me.c_coords.len() + me.y_outputs.len() + me.r_point.len()
        + 1 /*base_b*/ + y_prev.len() + y_next.len() + 1 /*rho*/;
    let byte_off = scalars_before_digest * 8;
    assert!(proof.public_io_bytes.len() >= byte_off + 1);
    proof.public_io_bytes[byte_off] ^= 0xFF;

    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(!ok, "verification must fail when fold_chain_digest is tampered in public IO");
}

#[test]
fn rho_flip_fails_v1() {
    let (me, wit) = tiny_me_instance();
    let rho = F::from_u64(9);
    let y_prev = vec![F::from_u64(3)];
    let y_next = vec![F::from_u64(12)];
    let ev = IvcEvEmbed { rho, y_prev: y_prev.clone(), y_next: y_next.clone(), y_step_public: None,
        fold_chain_digest: None, acc_c_prev: None, acc_c_step: None, acc_c_next: None, rho_eff: None };
    let mut proof = compress_ivc_verifier_to_lean_proof_with_linkage(&me, &wit, None, Some(ev), None, None)
        .expect("prove");

    // Offset to rho scalar in EV public IO
    let scalars_before_rho = me.c_coords.len() + me.y_outputs.len() + me.r_point.len() + 1 /*base_b*/
        + y_prev.len() + y_next.len();
    let byte_off = scalars_before_rho * 8;
    proof.public_io_bytes[byte_off] ^= 0xA5;

    let ok = verify_lean_proof(&proof).unwrap_or(false);
    assert!(!ok, "verification must fail when rho is tampered in public IO");
}
