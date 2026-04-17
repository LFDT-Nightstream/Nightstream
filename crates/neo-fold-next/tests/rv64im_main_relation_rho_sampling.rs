use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_fold_next::rv64im::main_relation_circuit::rho_sampling::{
    materialize_goldilocks_rot_matrices, sample_goldilocks_rot_rhos,
};
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::D;
use neo_params::NeoParams;
use neo_reductions::{sample_rot_rhos_n, RotRing};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeField64;
use spartan2::provider::goldi::F as SpartanF;

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_rho_sampler_matches_native_goldilocks_sampler() {
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();
    let mut native_tr = Poseidon2Transcript::new(b"test/rv64im/rho_sampling");
    let native_rhos = sample_rot_rhos_n(&mut native_tr, &params, &ring, 2).expect("native rho sampling");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut circuit_tr =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "tr"), b"test/rv64im/rho_sampling").expect("transcript");
    let sampled = sample_goldilocks_rot_rhos(&mut cs, &mut circuit_tr, 2, "rho").expect("circuit rho sampling");
    let sampled_mats = materialize_goldilocks_rot_matrices(&mut cs, &sampled, "rho_mat").expect("materialize rho mats");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
    assert_eq!(sampled.len(), native_rhos.len());
    assert_eq!(sampled_mats.len(), native_rhos.len());
    for (rho_idx, ((sampled_rho, sampled_mat), native_rho)) in sampled
        .iter()
        .zip(sampled_mats.iter())
        .zip(native_rhos.iter())
        .enumerate()
    {
        assert_eq!(sampled_rho.coeffs.len(), D);
        assert_eq!(sampled_rho.coeff_values.len(), D);
        for coeff_idx in 0..D {
            let expected = native_rho[(coeff_idx, 0)];
            assert_eq!(
                sampled_rho.coeff_values[coeff_idx], expected,
                "rho {rho_idx} coeff {coeff_idx} native mismatch"
            );
            let actual = sampled_rho.coeffs[coeff_idx]
                .get_value()
                .expect("coeff witness value")
                .to_canonical_u64();
            assert_eq!(
                actual,
                expected.as_canonical_u64(),
                "rho {rho_idx} coeff {coeff_idx} circuit value mismatch"
            );
        }
        for row in 0..D {
            for col in 0..D {
                let expected = native_rho[(row, col)];
                assert_eq!(
                    sampled_mat.entry_value(row, col).expect("matrix value"),
                    expected,
                    "rho {rho_idx} entry ({row}, {col}) native mismatch"
                );
                let actual = sampled_mat
                    .entry(row, col)
                    .expect("matrix entry")
                    .get_value()
                    .expect("matrix witness value")
                    .to_canonical_u64();
                assert_eq!(
                    actual,
                    expected.as_canonical_u64(),
                    "rho {rho_idx} entry ({row}, {col}) circuit mismatch"
                );
            }
        }
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_rho_sampler_handles_k_rho_plus_one_claims() {
    let params = NeoParams::goldilocks_127();
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut circuit_tr =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "tr"), b"test/rv64im/rho_sampling/full").expect("transcript");
    let sampled = sample_goldilocks_rot_rhos(&mut cs, &mut circuit_tr, (params.k_rho as usize) + 1, "rho")
        .expect("circuit rho sampling");
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
    assert_eq!(sampled.len(), (params.k_rho as usize) + 1);
    assert!(sampled.iter().all(|rho| rho.coeffs.len() == D));
}
