use neo::{F, zero_mcs_instance_for_shape};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn base_case_sets_const1_and_commits() {
    let d = neo_math::D;
    let m_in = 1usize;
    let m_step = 4usize;
    let const1_idx = Some(0usize);

    // Ensure Ajtai PP for (D, m_step)
    let mut rng = StdRng::from_seed([7u8; 32]);
    let pp = ajtai_setup(&mut rng, d, 8, m_step).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);

    let (inst, wit) = zero_mcs_instance_for_shape(m_in, m_step, const1_idx)
        .expect("zero MCS instance should be constructed");
    // Check Z[0, m_in + const1_idx] == 1
    assert_eq!(wit.Z[(0, m_in + const1_idx.unwrap())], F::ONE);

    // Commitment matches AjtaiSModule • Z
    let l = AjtaiSModule::from_global_for_dims(d, m_step).unwrap();
    let c = l.commit(&wit.Z);
    assert_eq!(c.data, inst.c.data);
}

