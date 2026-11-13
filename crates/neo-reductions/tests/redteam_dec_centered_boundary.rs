use neo_math::F;
use neo_params::{nightstream_p64_k16 as profile, NeoParams};
use p3_field::PrimeField64;

#[test]
fn neo_params_rejects_dec_bound_outside_centered_field_interval() {
    let q = F::ORDER_U64;
    let params = NeoParams::new(
        q,
        profile::ETA as u32,
        profile::D as u32,
        profile::KAPPA,
        1,
        2,
        63,
        profile::T,
        2,
        125,
    );

    assert!(
        params.is_err(),
        "accepted B=2^63 even though SuperNeo requires 2*B < q and q={q}"
    );
}
