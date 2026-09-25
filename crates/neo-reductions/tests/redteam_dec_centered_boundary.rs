use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeField64;

#[test]
fn neo_params_rejects_dec_bound_outside_centered_field_interval() {
    let q = F::ORDER_U64;
    let params = NeoParams::new(q, 81, D as u32, 18, 1, 2, 63, 216, 2, 125);

    assert!(
        params.is_err(),
        "accepted B=2^63 even though SuperNeo requires 2*B < q and Goldilocks q={q}"
    );
}
