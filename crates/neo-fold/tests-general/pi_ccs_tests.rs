use neo_fold::pi_ccs::*;
use neo_ccs::{Mat, SparsePoly, Term, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_params::NeoParams;
use neo_math::F;
use neo_ajtai::Commitment as Cmt;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

/// Minimal S-module homomorphism for tests:
/// - commit() returns a zero commitment with the right shape
/// - project_x() returns a zero matrix with m_in columns
struct DummyS;
impl neo_ccs::traits::SModuleHomomorphism<F, Cmt> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Cmt { Cmt::zeros(z.rows(), 2) }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        Mat::zero(z.rows(), m_in, F::ZERO)
    }
}

/// Build a CCS with non-power-of-two row count `n` and `m` variables.
/// One matrix (t=1). Polynomial f(y) = y - y (degree 1) ⇒ identically 0.
fn make_ccs(n: usize, m: usize) -> neo_ccs::CcsStructure<F> {
    assert!(n >= 1 && m >= 1);
    // Simple matrix with an identity block so Y is non-trivial if needed
    let mut m0 = Mat::zero(n, m, F::ZERO);
    let diag = n.min(m);
    for i in 0..diag { m0[(i, i)] = F::ONE; }

    // f(y) = y - y  (arity=1)
    let terms = vec![
        Term { coeff: F::ONE,  exps: vec![1] },
        Term { coeff: -F::ONE, exps: vec![1] },
    ];
    let f = SparsePoly::new(1, terms);
    neo_ccs::CcsStructure::new(vec![m0], f).expect("valid CCS")
}

/// Build a consistent MCS instance and witness:
/// - w = z (all private, m_in = 0)
/// - Z is the base-b decomposition of z (row-major D×m)
/// - c is a dummy zero commitment matching Z's shape
#[allow(non_snake_case)] // Allow mathematical notation Z
fn make_instance_and_witness(z: Vec<F>, params: &NeoParams) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let d = neo_math::D;
    let m = z.len();

    // Decompose z into base-b digits (Balanced), then convert to row-major D×m
    let z_digits = neo_ajtai::decomp_b(&z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { for row in 0..d { row_major[row * m + col] = z_digits[col * d + row]; } }
    let Z = Mat::from_row_major(d, m, row_major);

    // Dummy commitment that matches Z's shape; m_in=0 (all private)
    let l = DummyS;
    let c = l.commit(&Z);
    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit  = McsWitness::<F> { w: z, Z };
    (inst, wit)
}

#[test]
fn pi_ccs_non_power_of_two_n_works() {
    // Use the same params as your example; any sane params work here.
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);

    // Exercise several non-power-of-two n; m is small/fixed.
    let ns = [3usize, 5, 6, 7, 9];
    let m  = 4usize;

    for &n in &ns {
        let s = make_ccs(n, m);

        // Simple witness vector z = [1,2,3,4] over F
        let z: Vec<F> = (0..m).map(|i| F::from_u64((i as u64) + 1)).collect();
        let (inst, wit) = make_instance_and_witness(z, &params);

        // Prove
        let l = DummyS;
        let mut tr_p = Poseidon2Transcript::new(b"neo/fold");
        let (out_me, prf) = pi_ccs_prove_simple(&mut tr_p, &params, &s, &[inst.clone()], &[wit], &l)
            .expect("pi_ccs_prove should succeed for non-power-of-two n");

        // ℓ must equal log(dn) per paper (not just log n)
        let d_pad = neo_math::D.next_power_of_two();
        let n_pad = s.n.next_power_of_two().max(2);
        let ell_expected = (d_pad * n_pad).trailing_zeros() as usize;
        assert_eq!(prf.sumcheck_rounds.len(), ell_expected, "sum-check rounds must equal log(dn)");

        // Verify
        let mut tr_v = Poseidon2Transcript::new(b"neo/fold");
        let ok = pi_ccs_verify_simple(&mut tr_v, &params, &s, &[inst], &out_me, &prf)
            .expect("pi_ccs_verify should run");
        assert!(ok, "pi_ccs_verify should accept for n={}", n);
    }
}
