//! Convert a Midnight Powers-of-Tau transcript into `midnight-proofs` `ParamsKZG` files.
//!
//! This mirrors `external/midnight-ledger/transient-crypto/examples/translate-params.rs`,
//! but lives in this repo so we can generate params for outer compression proofs without
//! relying on Midnight ledger tooling.
//!
//! Usage:
//!   cargo run -p neo-midnight-bridge --example params_from_powers_of_tau -- \
//!     <powers_of_tau_path> <out_dir> [k_max]
//!
//! Output:
//!   `<out_dir>/bls_midnight_2p{k}` for k in 0..=k_max (written in RawBytes format).

use midnight_curves::{serde::SerdeObject, Bls12, G1Affine, G2Affine};
use midnight_proofs::{poly::kzg::params::ParamsKZG, utils::SerdeFormat};
use std::path::PathBuf;

fn floor_log2(mut n: usize) -> u32 {
    let mut log = 0u32;
    while n > 1 {
        n >>= 1;
        log += 1;
    }
    log
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pot_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("powers_of_tau"));
    let out_dir = args.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let k_max_arg = args.next().map(|s| s.parse::<u32>().expect("k_max must be an integer"));

    let bytes = std::fs::read(&pot_path).expect("read powers_of_tau");
    let g1_size = G1Affine::uncompressed_size();
    let g2_size = G2Affine::uncompressed_size();

    assert!(
        bytes.len() >= 2 * g2_size,
        "powers_of_tau too short: need >= 2*G2"
    );
    let offset = bytes.len() - 2 * g2_size;
    assert_eq!(
        offset % g1_size,
        0,
        "powers_of_tau length not aligned to G1 size"
    );

    let g1_count = offset / g1_size;
    let k_max_default = floor_log2(g1_count);
    let k_max = k_max_arg.unwrap_or(k_max_default);
    let needed = 1usize << k_max;
    assert!(
        g1_count >= needed,
        "powers_of_tau has {g1_count} G1 points, but k_max={k_max} needs >= {needed}"
    );

    println!("Reading powers_of_tau from {pot_path:?}");
    println!("G1 points: {g1_count} (max k by length: {k_max_default})");
    println!("Generating ParamsKZG for k_max={k_max} into {out_dir:?}");

    // Read G1 powers.
    let mut g1s = Vec::with_capacity(g1_count);
    for chunk in bytes[..offset].chunks(g1_size) {
        let p = G1Affine::from_raw_bytes(chunk).expect("decode G1");
        g1s.push(p.into());
    }

    // Read trailing G2 points (beta_g2, g2).
    let g2_0 = G2Affine::from_raw_bytes(&bytes[offset..offset + g2_size]).expect("decode G2[0]");
    let g2_1 =
        G2Affine::from_raw_bytes(&bytes[offset + g2_size..offset + 2 * g2_size]).expect("decode G2[1]");

    std::fs::create_dir_all(&out_dir).expect("create out_dir");

    // Build params at k_max, then downsize in-place and write all smaller k.
    let mut params = ParamsKZG::<Bls12>::from_parts(k_max, g1s, None, g2_0.into(), g2_1.into());
    for k in (0..=k_max).rev() {
        println!("Writing k={k}");
        params.downsize(k);
        let out_path = out_dir.join(format!("bls_midnight_2p{k}"));
        let mut f = std::fs::File::create(&out_path).expect("create output file");
        params
            .write_custom(&mut f, SerdeFormat::RawBytes)
            .expect("write ParamsKZG");
    }
}

