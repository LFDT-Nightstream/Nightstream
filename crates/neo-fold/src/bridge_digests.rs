//! Digest helpers used by the Spartan bridge and closure proofs.
//!
//! These functions are consensus-critical: they define how the Phase-1 public statement binds
//! initial/final accumulators and how Phase-2 closure proofs bind to the final obligations.

#![forbid(unsafe_code)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::MeInstance;
use neo_math::{F as NeoF, K as NeoK, KExtensions};
use p3_field::PrimeCharacteristicRing;

/// Digest an accumulator (list of ME instances).
///
/// This is the canonical definition used by `neo-spartan-bridge` (`acc_digest/v2`).
pub fn compute_accumulator_digest_v2(base_b: u32, acc: &[MeInstance<Cmt, NeoF, NeoK>]) -> [u8; 32] {
    use neo_ccs::crypto::poseidon2_goldilocks as p2;
    use p3_field::PrimeField64;
    use p3_symmetric::Permutation;

    let perm = p2::permutation();
    let mut st = [NeoF::ZERO; p2::WIDTH];
    let mut absorbed = 0usize;

    let mut absorb = |x: NeoF| {
        if absorbed >= p2::RATE {
            st = perm.permute(st);
            absorbed = 0;
        }
        st[absorbed] = x;
        absorbed += 1;
    };

    for &b in b"neo/spartan-bridge/acc_digest/v2" {
        absorb(NeoF::from_u64(b as u64));
    }

    absorb(NeoF::from_u64(acc.len() as u64));

    for me in acc {
        absorb(NeoF::from_u64(me.m_in as u64));

        // Commitment data (binds the Ajtai commitment relation used by native verifier).
        absorb(NeoF::from_u64(me.c.data.len() as u64));
        for &c in &me.c.data {
            absorb(NeoF::from_u64(c.as_canonical_u64()));
        }

        // X matrix (binds the public linear projection).
        absorb(NeoF::from_u64(me.X.rows() as u64));
        absorb(NeoF::from_u64(me.X.cols() as u64));
        for &x in me.X.as_slice() {
            absorb(NeoF::from_u64(x.as_canonical_u64()));
        }

        // r point.
        absorb(NeoF::from_u64(me.r.len() as u64));
        for limb in &me.r {
            let coeffs = limb.as_coeffs();
            absorb(coeffs[0]);
            absorb(coeffs[1]);
        }

        // y vectors (padded).
        absorb(NeoF::from_u64(me.y.len() as u64));
        for yj in &me.y {
            absorb(NeoF::from_u64(yj.len() as u64));
            for y_elem in yj {
                let coeffs = y_elem.as_coeffs();
                absorb(coeffs[0]);
                absorb(coeffs[1]);
            }
        }

        // Canonical y_scalars: base-b recomposition of the first D digits of y[j].
        //
        // IMPORTANT: in shared-bus mode, `me.y_scalars` may include extra scalars appended after
        // the core `t=s.t()` entries (bus openings). Those must not affect the public accumulator
        // digest, so we derive scalars from `y` directly.
        let bK = NeoK::from(NeoF::from_u64(base_b as u64));
        for yj in &me.y {
            let mut acc_k = NeoK::ZERO;
            let mut pw = NeoK::ONE;
            for rho in 0..neo_math::D {
                acc_k += pw * yj[rho];
                pw *= bK;
            }
            let coeffs = acc_k.as_coeffs();
            absorb(coeffs[0]);
            absorb(coeffs[1]);
        }
    }

    // Same squeeze gate as `neo_transcript::Poseidon2Transcript::digest32`.
    absorb(NeoF::ONE);
    st = perm.permute(st);

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    out
}

/// Digest binding for the final obligations list.
///
/// This is the canonical definition used by `neo-spartan-bridge` (`obligations_digest/v1`).
pub fn compute_obligations_digest_v1(
    acc_final_main_digest: [u8; 32],
    acc_final_val_digest: [u8; 32],
    pp_id_digest: [u8; 32],
) -> [u8; 32] {
    use blake3::Hasher;

    let mut h = Hasher::new();
    h.update(b"neo/spartan-bridge/obligations_digest/v1");
    h.update(&acc_final_main_digest);
    h.update(&acc_final_val_digest);
    h.update(&pp_id_digest);
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}
