use neo_ccs::matrix::Mat;
use neo_memory::shout::split_lut_mats;
use neo_memory::witness::{LutInstance, LutWitness};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn split_lut_mats_orders_fields() {
    // With index-bit addressing: d=3, ell=1 (since n_side=2)
    // Total matrices = d*ell + has_lookup + val = 3*1 + 2 = 5
    let inst: LutInstance<(), Goldilocks> = LutInstance {
        comms: vec![(); 5],
        k: 8,
        d: 3,
        n_side: 2,
        steps: 16,
        ell: 1, // log2(2) = 1
        table: vec![Goldilocks::ZERO; 8],
        _phantom: std::marker::PhantomData,
    };

    let dummy_mat = Mat::from_row_major(1, 1, vec![Goldilocks::ZERO]);
    let wit = LutWitness {
        mats: vec![dummy_mat.clone(); 5], // 3 addr bits + has_lookup + val
    };

    let parts = split_lut_mats(&inst, &wit);
    // With ell=1, d=3: we have d*ell = 3 bit columns
    assert_eq!(parts.addr_bit_mats.len(), 3);
    // Plus has_lookup and val columns
    assert_eq!(parts.has_lookup_mat.cols(), 1);
    assert_eq!(parts.val_mat.cols(), 1);
}
