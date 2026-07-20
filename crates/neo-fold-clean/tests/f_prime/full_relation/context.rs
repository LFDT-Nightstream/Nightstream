use super::*;

#[test]
fn full_relation_rejects_a_noncanonical_nifs_verifier_configuration() {
    let carrier = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&carrier, 41).expect("carrier preprocessing");
    let mut cfg = step_config(&prep);
    cfg.nifs.pi_ccs.header_bundle[0] += F::ONE;
    let application = fibonacci_step_r1cs();
    let initial = semantic_state_digest_fields(&[F::from_u64(3), F::from_u64(5)]);
    let context = full_context(&prep, initial);

    let result = FullFPrimeRelation::new(context, cfg, &application, vec![1, 2], vec![3, 4]);
    assert!(matches!(
        result,
        Err(FullFPrimeError::NifsConfigMismatch { field: "header bundle" })
    ));
}
