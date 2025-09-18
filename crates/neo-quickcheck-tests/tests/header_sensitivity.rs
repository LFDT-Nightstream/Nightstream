#![cfg(feature = "quickcheck")]
//! QuickCheck: flipping any one bit of the digest changes the header bytes.

#![allow(deprecated)]
use quickcheck_macros::quickcheck;
#[allow(deprecated)]
use neo_ccs::MEInstance;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use neo_spartan_bridge::encode_bridge_io_header;

fn fvec(xs: &[u64]) -> Vec<F> { xs.iter().copied().map(F::from_u64).collect() }

#[quickcheck]
fn header_digest_flip_changes_bytes(
    c_vals: Vec<u64>, y_vals: Vec<u64>, r_vals: Vec<u64>,
    base_raw: u64, digest_vec: Vec<u8>, flip_idx_raw: u8
) -> bool {
    // Convert Vec<u8> to [u8; 32], padding or truncating as needed
    let mut digest = [0u8; 32];
    for (i, &byte) in digest_vec.iter().take(32).enumerate() {
        digest[i] = byte;
    }
    let c = fvec(&c_vals[..(c_vals.len().min(4))]);
    let y = fvec(&y_vals[..(y_vals.len().min(4))]);
    let r = fvec(&r_vals[..(r_vals.len().min(4))]);
    let base = 2 + (base_raw % 9); // base in [2..10]

    let me0 = MEInstance {
        c_step_coords: vec![], // TODO: Pattern A - populate with actual pre-commit coords
        u_offset: 0,
        u_len: 0,
        c_coords: c.clone(),
        y_outputs: y.clone(),
        r_point: r.clone(),
        base_b: base,
        header_digest: digest,
    };
    let h0 = encode_bridge_io_header(&me0);

    // flip one bit
    let idx = (flip_idx_raw as usize) % 32;
    digest[idx] ^= 0x01;

    let me1 = MEInstance {
        c_step_coords: vec![], // TODO: Pattern A - populate with actual pre-commit coords
        u_offset: 0,
        u_len: 0,
        c_coords: c,
        y_outputs: y,
        r_point: r,
        base_b: base,
        header_digest: digest,
    };
    let h1 = encode_bridge_io_header(&me1);

    h0 != h1
}
