//! NIFS claim views stored in the F-prime image.

use neo_math::F;
use p3_field::PrimeField64;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;

use super::{
    decode_u64_lane, read_digest_bits, write_digest_bits, write_lane_bits, write_u64_bits, FPrimeImage,
    NIFS_FOLD_DIGEST_BITS, NIFS_K_LIMB_BITS, NIFS_LEN_HEADER_BITS,
};

/// nifs_payloads view of one fresh `CcsClaim` payload. Mirrors
/// `paper::digest::ccs_claim_digest`'s preimage shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NifsCcsClaimView {
    /// `Commitment::d` (`usize` in production; encoded as `u64` here).
    pub d: u64,
    /// `Commitment::kappa` (same encoding contract).
    pub kappa: u64,
    pub c_data: Vec<F>,
    pub x: Vec<F>,
    pub m_in: u64,
}

/// Shape of one fresh CcsClaim payload — the sizes the encoder/decoder
/// agree on. Computed from a real `CcsClaim` or set explicitly by the
/// caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NifsCcsClaimShape {
    pub c_data_entries: usize,
    pub x_entries: usize,
}

impl NifsCcsClaimShape {
    pub fn bits(&self) -> usize {
        // d + kappa + c_data_len + c_data + x_len + x + m_in
        2 * NIFS_LEN_HEADER_BITS
            + NIFS_LEN_HEADER_BITS
            + self.c_data_entries * POSEIDON2_GOLDILOCKS_BITS
            + NIFS_LEN_HEADER_BITS
            + self.x_entries * POSEIDON2_GOLDILOCKS_BITS
            + NIFS_LEN_HEADER_BITS
    }
}

/// nifs_payloads view of one CE claim payload. Covers commitment, active `X`,
/// evaluation point `r`, identity-first `y_ring`, `m_in`, and `fold_digest`.
/// Encoding order mirrors `ce_claim_digest`'s preimage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NifsCeClaimView {
    pub d: u64,
    pub kappa: u64,
    pub c_data: Vec<F>,
    pub x_rows: u64,
    pub x_cols: u64,
    pub x_active_cols: u64,
    /// `x_rows × x_active_cols` F values, row-major.
    pub x_active_flat: Vec<F>,
    /// K-extension `r` as `[c0, c1]` F pairs.
    pub r: Vec<[F; 2]>,
    /// `y_ring[row][col]` as K-element pairs. Each inner Vec may have a
    /// different length (per `y_ring_inner_lens` in the shape).
    pub y_ring: Vec<Vec<[F; 2]>>,
    pub m_in: u64,
    /// `digest32_as_fields(fold_digest)` — four F values, each 64 bits.
    pub fold_digest_fields: [F; 4],
}

/// Shape of one CE claim payload.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NifsCeClaimShape {
    pub c_data_entries: usize,
    pub x_rows: usize,
    pub x_active_cols: usize,
    pub r_len: usize,
    pub y_ring_inner_lens: Vec<usize>,
}

impl NifsCeClaimShape {
    pub fn bits(&self) -> usize {
        let mut total = 0;
        total += 2 * NIFS_LEN_HEADER_BITS; // d, kappa
        total += NIFS_LEN_HEADER_BITS + self.c_data_entries * POSEIDON2_GOLDILOCKS_BITS; // c_data
        total += 3 * NIFS_LEN_HEADER_BITS; // x_rows, x_cols, x_active_cols
        total += self.x_rows * self.x_active_cols * POSEIDON2_GOLDILOCKS_BITS; // X entries
        total += NIFS_LEN_HEADER_BITS + self.r_len * NIFS_K_LIMB_BITS; // r
        total += NIFS_LEN_HEADER_BITS; // y_ring outer
        for &inner in &self.y_ring_inner_lens {
            total += NIFS_LEN_HEADER_BITS + inner * NIFS_K_LIMB_BITS;
        }
        total += NIFS_LEN_HEADER_BITS; // m_in
        total += NIFS_FOLD_DIGEST_BITS; // fold_digest
        total
    }
}

impl FPrimeImage {
    /// Encode a fresh `CcsClaim` payload starting at `nifs_offset` (a
    /// nifs_payloads-relative offset). Returns the next free nifs_payloads-relative offset.
    pub fn fill_nifs_ccs_claim_at(&mut self, nifs_offset: usize, view: &NifsCcsClaimView) -> usize {
        let shape = NifsCcsClaimShape {
            c_data_entries: view.c_data.len(),
            x_entries: view.x.len(),
        };
        let total = shape.bits();
        assert!(
            nifs_offset + total <= self.layout.nifs_payloads.bits,
            "nifs_payloads CcsClaim payload at offset {nifs_offset} ({total} bits) overflows region ({} bits)",
            self.layout.nifs_payloads.bits,
        );

        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        write_u64_bits(&mut self.values, cursor, view.d as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.kappa as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.c_data.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.c_data {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.x.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.x {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.m_in);
        cursor += NIFS_LEN_HEADER_BITS;

        debug_assert_eq!(cursor, self.layout.nifs_payloads.offset + nifs_offset + total);
        nifs_offset + total
    }

    /// Decode a fresh `CcsClaim` payload from `nifs_offset` using `shape`
    /// to size the variable-length fields.
    pub fn decode_nifs_ccs_claim_at(&self, nifs_offset: usize, shape: &NifsCcsClaimShape) -> NifsCcsClaimView {
        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        let d = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let kappa = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let c_data_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            c_data_len, shape.c_data_entries,
            "nifs_payloads CcsClaim c_data len mismatch"
        );
        let c_data: Vec<F> = (0..c_data_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += c_data_len * POSEIDON2_GOLDILOCKS_BITS;
        let x_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(x_len, shape.x_entries, "nifs_payloads CcsClaim x len mismatch");
        let x: Vec<F> = (0..x_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += x_len * POSEIDON2_GOLDILOCKS_BITS;
        let m_in = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        NifsCcsClaimView {
            d,
            kappa,
            c_data,
            x,
            m_in,
        }
    }

    /// Encode one CE claim payload starting at `nifs_offset`. Returns the
    /// next free nifs_payloads-relative offset.
    pub fn fill_nifs_ce_claim_at(&mut self, nifs_offset: usize, view: &NifsCeClaimView) -> usize {
        let shape = NifsCeClaimShape {
            c_data_entries: view.c_data.len(),
            x_rows: view.x_rows as usize,
            x_active_cols: view.x_active_cols as usize,
            r_len: view.r.len(),
            y_ring_inner_lens: view.y_ring.iter().map(|row| row.len()).collect(),
        };
        let total = shape.bits();
        assert!(
            nifs_offset + total <= self.layout.nifs_payloads.bits,
            "nifs_payloads CeClaim payload at offset {nifs_offset} ({total} bits) overflows region ({} bits)",
            self.layout.nifs_payloads.bits,
        );

        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        write_u64_bits(&mut self.values, cursor, view.d as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.kappa as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.c_data.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.c_data {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.x_rows);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.x_cols);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.x_active_cols);
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            view.x_active_flat.len(),
            (view.x_rows * view.x_active_cols) as usize,
            "nifs_payloads CeClaim x_active_flat length must match x_rows × x_active_cols"
        );
        for &v in &view.x_active_flat {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.r.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for k in &view.r {
            write_lane_bits(&mut self.values, cursor, k[0]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
            write_lane_bits(&mut self.values, cursor, k[1]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.y_ring.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for row in &view.y_ring {
            write_u64_bits(&mut self.values, cursor, row.len() as u64);
            cursor += NIFS_LEN_HEADER_BITS;
            for k in row {
                write_lane_bits(&mut self.values, cursor, k[0]);
                cursor += POSEIDON2_GOLDILOCKS_BITS;
                write_lane_bits(&mut self.values, cursor, k[1]);
                cursor += POSEIDON2_GOLDILOCKS_BITS;
            }
        }
        // FS-bound prefix ends here: production `ce_claim_digest` puts
        // `m_in` and `fold_digest` immediately after `y_ring`. Keep that
        // exact order so the first N bits of nifs_payloads round-trip back to the
        // production preimage prefix.
        write_u64_bits(&mut self.values, cursor, view.m_in);
        cursor += NIFS_LEN_HEADER_BITS;
        write_digest_bits(&mut self.values, cursor, view.fold_digest_fields);
        cursor += NIFS_FOLD_DIGEST_BITS;
        debug_assert_eq!(cursor, self.layout.nifs_payloads.offset + nifs_offset + total);
        nifs_offset + total
    }

    /// Decode one CE claim payload from `nifs_offset` using `shape` to
    /// size the variable-length fields.
    pub fn decode_nifs_ce_claim_at(&self, nifs_offset: usize, shape: &NifsCeClaimShape) -> NifsCeClaimView {
        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        let d = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let kappa = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let c_data_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(c_data_len, shape.c_data_entries, "nifs_payloads CeClaim c_data len");
        let c_data: Vec<F> = (0..c_data_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += c_data_len * POSEIDON2_GOLDILOCKS_BITS;
        let x_rows = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let x_cols = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let x_active_cols = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(x_rows as usize, shape.x_rows, "nifs_payloads CeClaim x_rows");
        assert_eq!(
            x_active_cols as usize, shape.x_active_cols,
            "nifs_payloads CeClaim x_active_cols"
        );
        let x_count = (x_rows * x_active_cols) as usize;
        let x_active_flat: Vec<F> = (0..x_count)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += x_count * POSEIDON2_GOLDILOCKS_BITS;
        let r_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(r_len, shape.r_len, "nifs_payloads CeClaim r_len");
        let r: Vec<[F; 2]> = (0..r_len)
            .map(|i| {
                let base = cursor + i * NIFS_K_LIMB_BITS;
                let c0 = decode_u64_lane(&self.values, base);
                let c1 = decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS);
                [c0, c1]
            })
            .collect();
        cursor += r_len * NIFS_K_LIMB_BITS;
        let y_ring_outer = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            y_ring_outer,
            shape.y_ring_inner_lens.len(),
            "nifs_payloads CeClaim y_ring outer"
        );
        let mut y_ring: Vec<Vec<[F; 2]>> = Vec::with_capacity(y_ring_outer);
        for &expected_inner in &shape.y_ring_inner_lens {
            let inner = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
            cursor += NIFS_LEN_HEADER_BITS;
            assert_eq!(inner, expected_inner, "nifs_payloads CeClaim y_ring inner");
            let row: Vec<[F; 2]> = (0..inner)
                .map(|i| {
                    let base = cursor + i * NIFS_K_LIMB_BITS;
                    [
                        decode_u64_lane(&self.values, base),
                        decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS),
                    ]
                })
                .collect();
            cursor += inner * NIFS_K_LIMB_BITS;
            y_ring.push(row);
        }
        // FS-bound prefix: read `m_in` + `fold_digest` immediately after
        // `y_ring`, mirroring `ce_claim_digest`'s preimage order.
        let m_in = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let fold_digest_fields = read_digest_bits(&self.values, cursor);
        cursor += NIFS_FOLD_DIGEST_BITS;
        debug_assert_eq!(cursor, self.layout.nifs_payloads.offset + nifs_offset + shape.bits());
        NifsCeClaimView {
            d,
            kappa,
            c_data,
            x_rows,
            x_cols,
            x_active_cols,
            x_active_flat,
            r,
            y_ring,
            m_in,
            fold_digest_fields,
        }
    }
}
