use neo_ccs::matrix::Mat;
use neo_memory::twist::split_mem_mats;
use neo_memory::witness::{MemInstance, MemWitness};
use neo_memory::MemInit;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn split_mem_mats_orders_fields() {
    // With index-bit addressing: d=2, ell=1 (since n_side=2)
    // Route A layout:
    // Total matrices = 2*d*ell + 5 = 2*2*1 + 5 = 9
    // (ra_bits, wa_bits, has_read, has_write, wv, rv, inc_at_write_addr)
    let inst: MemInstance<(), Goldilocks> = MemInstance {
        comms: vec![(); 9],
        k: 4,
        d: 2,
        n_side: 2,
        steps: 8,
        ell: 1, // log2(2) = 1
        init: MemInit::Zero,
        _phantom: std::marker::PhantomData,
    };

    let dummy_mat = Mat::from_row_major(1, 1, vec![Goldilocks::ZERO]);
    let wit = MemWitness {
        mats: vec![dummy_mat.clone(); 9],
    };

    let parts = split_mem_mats(&inst, &wit);

    // With ell=1, d=2: we have d*ell = 2 bit columns for read and write
    assert_eq!(parts.ra_bit_mats.len(), 2);
    assert_eq!(parts.wa_bit_mats.len(), 2);
    // Check the scalar columns exist
    assert_eq!(parts.has_read_mat.cols(), 1);
    assert_eq!(parts.has_write_mat.cols(), 1);
    assert_eq!(parts.wv_mat.cols(), 1);
    assert_eq!(parts.rv_mat.cols(), 1);
    assert_eq!(parts.inc_at_write_addr_mat.cols(), 1);

    let layout = inst.twist_layout();
    assert_eq!(layout.ell_addr, 2);
    assert_eq!(layout.expected_len(), 9);
    assert_eq!(layout.val_lane_len(), 4);
    assert_eq!(layout.ra_bits, 0..2);
    assert_eq!(layout.wa_bits, 2..4);
    assert_eq!(layout.has_read, 4);
    assert_eq!(layout.has_write, 5);
    assert_eq!(layout.wv, 6);
    assert_eq!(layout.rv, 7);
    assert_eq!(layout.inc_at_write_addr, 8);
}
