//! Π_CCS NC-channel (column-phase) kernels: round evaluation and the digit
//! table folds.
//!
//! Contract: bit-identical to `NcOracle`'s `col_phase_coeffs_b2` and
//! `NcDigitTable::fold_inplace` in neo-reductions. The digit layouts mirror
//! the CPU's representation evolution: diagonal/strided flat windows while
//! merge windows stay lane-disjoint (`2·width ≤ 54`), dense `[K; 54]` rows
//! afterwards. Mask-based zero skips on the CPU are value-equal
//! optimizations; these kernels compute plainly.
//!
//! Thread shape for evaluation: one thread per (column-pair chunk, output
//! coefficient), scalar accumulators only — see `pi_ccs_fe` for why.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::goldilocks::Kx;

/// Digit-table representation marker.
pub const NC_MODE_STRIDED: u32 = 0;
pub const NC_MODE_DENSE: u32 = 1;
/// Column pairs each eval thread walks.
pub const NC_CHUNK_PAIRS: usize = 2;
/// The NC column round polynomial has 5 coefficients for b = 2. Ajtai-tail
/// proof rounds may be padded wider to the protocol degree bound by the host.
pub const NC_COEFFS: usize = 5;
/// Ring lane count (= neo_math::D), asserted host-side.
pub const RING_LANES: usize = 54;

pub use pi_ccs_nc_kernels::LoadedModule as NcKernelModule;

pub fn load_nc_kernels(ctx: &Arc<CudaContext>) -> Result<NcKernelModule, EmbeddedModuleError> {
    pi_ccs_nc_kernels::load(ctx)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_col_partials(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    eq_tbl: &DeviceBuffer<u64>,
    digits: &DeviceBuffer<u64>,
    weights: &DeviceBuffer<u64>,
    mode: u32,
    width: usize,
    live_len: usize,
    wit_stride: usize,
    num_wits: usize,
    tail_len: usize,
    pair_groups: usize,
    inner_partials: &mut DeviceBuffer<u64>,
    partials: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_inner_partials(
        stream,
        LaunchConfig::for_num_elems((num_wits * tail_len) as u32),
        digits,
        weights,
        mode,
        width as u32,
        live_len as u32,
        wit_stride as u32,
        num_wits as u32,
        tail_len as u32,
        inner_partials,
    )?;
    module.nc_col_partials_from_inner(
        stream,
        LaunchConfig::for_num_elems((num_wits * pair_groups) as u32),
        eq_tbl,
        inner_partials,
        num_wits as u32,
        tail_len as u32,
        pair_groups as u32,
        partials,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_strided(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    src_len: usize,
    width: usize,
    out_len: usize,
    wit_stride: usize,
    num_wits: usize,
    r_c0: u64,
    r_c1: u64,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_strided(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_len) as u32),
        src,
        src_len as u32,
        width as u32,
        out_len as u32,
        wit_stride as u32,
        num_wits as u32,
        r_c0,
        r_c1,
        dst,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_strided_from_challenge(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    src_len: usize,
    width: usize,
    out_len: usize,
    wit_stride: usize,
    num_wits: usize,
    challenge: &DeviceBuffer<u64>,
    challenge_offset: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_strided_from_challenge(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_len) as u32),
        src,
        src_len as u32,
        width as u32,
        out_len as u32,
        wit_stride as u32,
        num_wits as u32,
        challenge,
        challenge_offset as u32,
        dst,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_strided_to_dense(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    src_len: usize,
    width: usize,
    rows: usize,
    out_rows: usize,
    wit_stride: usize,
    dense_stride: usize,
    num_wits: usize,
    r_c0: u64,
    r_c1: u64,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_strided_to_dense(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_rows * RING_LANES) as u32),
        src,
        src_len as u32,
        width as u32,
        rows as u32,
        out_rows as u32,
        wit_stride as u32,
        dense_stride as u32,
        num_wits as u32,
        r_c0,
        r_c1,
        dst,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_strided_to_dense_from_challenge(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    src_len: usize,
    width: usize,
    rows: usize,
    out_rows: usize,
    wit_stride: usize,
    dense_stride: usize,
    num_wits: usize,
    challenge: &DeviceBuffer<u64>,
    challenge_offset: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_strided_to_dense_from_challenge(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_rows * RING_LANES) as u32),
        src,
        src_len as u32,
        width as u32,
        rows as u32,
        out_rows as u32,
        wit_stride as u32,
        dense_stride as u32,
        num_wits as u32,
        challenge,
        challenge_offset as u32,
        dst,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_dense(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    rows: usize,
    out_rows: usize,
    dense_stride: usize,
    num_wits: usize,
    r_c0: u64,
    r_c1: u64,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_dense(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_rows * RING_LANES) as u32),
        src,
        rows as u32,
        out_rows as u32,
        dense_stride as u32,
        num_wits as u32,
        r_c0,
        r_c1,
        dst,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_fold_dense_from_challenge(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    rows: usize,
    out_rows: usize,
    dense_stride: usize,
    num_wits: usize,
    challenge: &DeviceBuffer<u64>,
    challenge_offset: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_fold_dense_from_challenge(
        stream,
        LaunchConfig::for_num_elems((num_wits * out_rows * RING_LANES) as u32),
        src,
        rows as u32,
        out_rows as u32,
        dense_stride as u32,
        num_wits as u32,
        challenge,
        challenge_offset as u32,
        dst,
    )
}

/// Pack the finalized column state — each witness's live digit row plus
/// `eq[0]` — into `num_wits * D * 2 + 2` words so the host downloads bytes,
/// not the full ping-pong buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_nc_pack_final_state(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    digits: &DeviceBuffer<u64>,
    eq: &DeviceBuffer<u64>,
    mode: u32,
    width: usize,
    wit_stride: usize,
    num_wits: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_pack_final_state(
        stream,
        LaunchConfig::for_num_elems((num_wits * RING_LANES + 1) as u32),
        digits,
        eq,
        mode,
        width as u32,
        wit_stride as u32,
        num_wits as u32,
        out,
    )
}

/// Initialize the digit ping-pong buffer from resident base-field witness
/// planes: `digits[wit][col] = (plane[wit][col], 0)` for `col < len` — the
/// exact K lift `build_nc_digit_table_compact` produces for the unfolded
/// table. `digits` must be zeroed (stale headroom past `len` stays zero).
pub fn launch_nc_widen_planes(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    plane_len: usize,
    len: usize,
    wit_stride: usize,
    num_wits: usize,
    digits: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_widen_planes(
        stream,
        LaunchConfig::for_num_elems((num_wits * len) as u32),
        planes,
        plane_len as u32,
        len as u32,
        wit_stride as u32,
        num_wits as u32,
        digits,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_nc_ajtai_tail_partials(
    module: &NcKernelModule,
    stream: &Arc<CudaStream>,
    digits: &DeviceBuffer<u64>,
    eq: &DeviceBuffer<u64>,
    beta_a: &DeviceBuffer<u64>,
    gamma: &DeviceBuffer<u64>,
    challenges: &DeviceBuffer<u64>,
    mode: u32,
    width: usize,
    wit_stride: usize,
    num_wits: usize,
    col_rounds: usize,
    tail_round: usize,
    tail_rounds: usize,
    partials: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.nc_ajtai_tail_partials(
        stream,
        LaunchConfig::for_num_elems(num_wits as u32),
        digits,
        eq,
        beta_a,
        gamma,
        challenges,
        mode,
        width as u32,
        wit_stride as u32,
        num_wits as u32,
        col_rounds as u32,
        tail_round as u32,
        tail_rounds as u32,
        partials,
    )
}

#[cuda_module]
pub mod pi_ccs_nc_kernels {
    use super::*;

    /// The four coefficients of the weighted b=2 range polynomial
    /// `w · N(a + b·X)` (N(y) = y³ − y) over one affine digit lane —
    /// `accumulate_inner_b2_at`'s array update as one named tuple, so the
    /// accumulators stay in registers.
    fn range_components(w: Kx, a: Kx, b: Kx) -> (Kx, Kx, Kx, Kx) {
        let three = Kx::from_words(3, 0);
        let a2 = a * a;
        let b2 = b * b;
        (
            w * (a2 * a - a),
            w * (a2 * b * three - b),
            w * (a * b2 * three),
            w * (b2 * b),
        )
    }

    fn read_k_pair(words: &[u64], k_index: usize) -> Kx {
        let base = k_index * 2;
        if base + 1 >= words.len() {
            return Kx::ZERO;
        }
        Kx::from_words(words[base], words[base + 1])
    }

    fn eq_lin(a: Kx, b: Kx) -> Kx {
        (Kx::ONE - a) * (Kx::ONE - b) + a * b
    }

    fn point_weight(values: &[u64], word_offset: usize, mask: usize, len: usize) -> Kx {
        let mut out = Kx::ONE;
        for bit in 0..len {
            let point = read_k_pair(values, word_offset / 2 + bit);
            out = if ((mask >> bit) & 1) == 0 {
                out * (Kx::ONE - point)
            } else {
                out * point
            };
        }
        out
    }

    fn read_final_digit_lane(
        digits: &[u64],
        mode: u32,
        width: usize,
        wit_stride: usize,
        witness: usize,
        lane: usize,
    ) -> Kx {
        if lane >= RING_LANES {
            return Kx::ZERO;
        }
        if mode == NC_MODE_STRIDED && lane >= width {
            return Kx::ZERO;
        }
        read_k_pair(digits, witness * wit_stride + lane)
    }

    fn prefolded_digit(
        digits: &[u64],
        mode: u32,
        width: usize,
        wit_stride: usize,
        witness: usize,
        lane_base: usize,
        prefix_bits: usize,
        challenges: &[u64],
        challenge_word_offset: usize,
    ) -> Kx {
        let prefix_count = 1usize << prefix_bits;
        let mut out = Kx::ZERO;
        for mask in 0..prefix_count {
            let lane = lane_base + mask;
            if lane < RING_LANES {
                let weight = point_weight(challenges, challenge_word_offset, mask, prefix_bits);
                out = out + weight * read_final_digit_lane(digits, mode, width, wit_stride, witness, lane);
            }
        }
        out
    }

    fn tail_weight(beta_a: &[u64], bit_start: usize, mask: usize, len: usize) -> Kx {
        point_weight(beta_a, 2 * bit_start, mask, len)
    }

    fn gamma_power(gamma: Kx, witness: usize) -> Kx {
        let mut out = gamma;
        for _ in 0..witness {
            out = out * gamma;
        }
        out
    }

    /// Stage A: one thread per (witness, column pair), computing the inner
    /// cubic without keeping the five outer coefficients live.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_inner_partials(
        digits: &[u64],
        weights: &[u64],
        mode: u32,
        width: u32,
        live_len: u32,
        wit_stride: u32,
        num_wits: u32,
        tail_len: u32,
        mut inner_partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let width = width as usize;
        let live_len = live_len as usize;
        let wit_stride = wit_stride as usize;
        let num_wits = num_wits as usize;
        let tail_len = tail_len as usize;
        if idx >= num_wits * tail_len {
            return;
        }
        let wit = idx / tail_len;
        let t = idx % tail_len;
        {
            let idx_pair = 2 * t;

            // The inner cubic's four coefficients for this witness, in
            // named scalars.
            let mut i0 = Kx::ZERO;
            let mut i1 = Kx::ZERO;
            let mut i2 = Kx::ZERO;
            let mut i3 = Kx::ZERO;
            {
                let base = wit * wit_stride * 2;
                let weight_at = |rho: usize| {
                    Kx::from_words(
                        weights[(wit * RING_LANES + rho) * 2],
                        weights[(wit * RING_LANES + rho) * 2 + 1],
                    )
                };
                if mode == NC_MODE_STRIDED {
                    // Strided: `live_len` counts flat slots; folds leave
                    // stale words past it in the ping-pong buffers and the
                    // padded column domain extends past every table.
                    let read_slot = |flat: usize| -> Kx {
                        let at = base + 2 * flat;
                        if flat < live_len && at + 2 <= digits.len() {
                            Kx::from_words(digits[at], digits[at + 1])
                        } else {
                            Kx::ZERO
                        }
                    };
                    if 2 * width <= RING_LANES {
                        // Lane-disjoint windows: each slot sees the other
                        // operand as zero, giving per-slot closed forms.
                        // rho advances with the slot; wrap by subtraction so
                        // the loop stays modulo-free (width ≤ 27 here).
                        let mut lo_rho = (idx_pair * width) % RING_LANES;
                        let mut hi_rho = ((idx_pair + 1) * width) % RING_LANES;
                        for j in 0..width {
                            let lo_flat = idx_pair * width + j;
                            let hi_flat = (idx_pair + 1) * width + j;
                            let lo = read_slot(lo_flat);
                            if lo != Kx::ZERO {
                                // lo window lane: (a, y1) = (lo, 0) ⇒ b = -lo.
                                let (r0, r1, r2, r3) = range_components(weight_at(lo_rho), lo, Kx::ZERO - lo);
                                i0 = i0 + r0;
                                i1 = i1 + r1;
                                i2 = i2 + r2;
                                i3 = i3 + r3;
                            }
                            let hi = read_slot(hi_flat);
                            if hi != Kx::ZERO {
                                // hi window lane: (a, y1) = (0, hi) ⇒ b = hi.
                                let (r0, r1, r2, r3) = range_components(weight_at(hi_rho), Kx::ZERO, hi);
                                i0 = i0 + r0;
                                i1 = i1 + r1;
                                i2 = i2 + r2;
                                i3 = i3 + r3;
                            }
                            lo_rho += 1;
                            if lo_rho >= RING_LANES {
                                lo_rho -= RING_LANES;
                            }
                            hi_rho += 1;
                            if hi_rho >= RING_LANES {
                                hi_rho -= RING_LANES;
                            }
                        }
                    } else {
                        // Overlapping windows (2·width > 54): gather both
                        // operands per lane, as the CPU `lane()` does.
                        let lo_start = (idx_pair * width) % RING_LANES;
                        let hi_start = ((idx_pair + 1) * width) % RING_LANES;
                        let windowed = |row: usize, start: usize, rho: usize| -> Kx {
                            let mut j = rho + RING_LANES - start;
                            if j >= RING_LANES {
                                j -= RING_LANES;
                            }
                            if j < width {
                                read_slot(row * width + j)
                            } else {
                                Kx::ZERO
                            }
                        };
                        for rho in 0..RING_LANES {
                            let a = windowed(idx_pair, lo_start, rho);
                            let y1 = windowed(idx_pair + 1, hi_start, rho);
                            let b = y1 - a;
                            if a != Kx::ZERO || b != Kx::ZERO {
                                let (r0, r1, r2, r3) = range_components(weight_at(rho), a, b);
                                i0 = i0 + r0;
                                i1 = i1 + r1;
                                i2 = i2 + r2;
                                i3 = i3 + r3;
                            }
                        }
                    }
                } else {
                    // Dense: `live_len` counts rows; rows past it read zero.
                    let read_lane = |row: usize, rho: usize| -> Kx {
                        let at = base + (row * RING_LANES + rho) * 2;
                        if row < live_len && at + 2 <= digits.len() {
                            Kx::from_words(digits[at], digits[at + 1])
                        } else {
                            Kx::ZERO
                        }
                    };
                    for rho in 0..RING_LANES {
                        let a = read_lane(idx_pair, rho);
                        let y1 = read_lane(idx_pair + 1, rho);
                        let b = y1 - a;
                        if a != Kx::ZERO || b != Kx::ZERO {
                            let (r0, r1, r2, r3) = range_components(weight_at(rho), a, b);
                            i0 = i0 + r0;
                            i1 = i1 + r1;
                            i2 = i2 + r2;
                            i3 = i3 + r3;
                        }
                    }
                }
            }

            let out_base = idx * 4 * 2;
            if out_base + 4 * 2 > inner_partials.len() {
                return;
            }
            let words = [i0.as_words(), i1.as_words(), i2.as_words(), i3.as_words()];
            for (coefficient, words) in words.iter().enumerate() {
                unsafe {
                    *inner_partials.get_unchecked_mut(out_base + 2 * coefficient) = words[0];
                    *inner_partials.get_unchecked_mut(out_base + 2 * coefficient + 1) = words[1];
                }
            }
        }
    }

    /// Stage B: one thread combines up to two resident inner cubics. Inner
    /// values are loaded in coefficient-sized scopes so they do not overlap
    /// the range-polynomial temporaries from stage A.
    #[kernel]
    pub fn nc_col_partials_from_inner(
        eq_tbl: &[u64],
        inner_partials: &[u64],
        num_wits: u32,
        tail_len: u32,
        pair_groups: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let partial_group = idx;
        let num_wits = num_wits as usize;
        let tail_len = tail_len as usize;
        let pair_groups = pair_groups as usize;
        if partial_group >= num_wits * pair_groups {
            return;
        }
        let wit = partial_group / pair_groups;
        let group = partial_group % pair_groups;
        let pair_start = group * NC_CHUNK_PAIRS;
        let pair_end = if pair_start + NC_CHUNK_PAIRS < tail_len {
            pair_start + NC_CHUNK_PAIRS
        } else {
            tail_len
        };

        let mut c0 = Kx::ZERO;
        let mut c1 = Kx::ZERO;
        let mut c2 = Kx::ZERO;
        let mut c3 = Kx::ZERO;
        let mut c4 = Kx::ZERO;
        for t in pair_start..pair_end {
            if 4 * t + 4 > eq_tbl.len() {
                return;
            }
            let e0 = Kx::from_words(eq_tbl[4 * t], eq_tbl[4 * t + 1]);
            let e1 = Kx::from_words(eq_tbl[4 * t + 2], eq_tbl[4 * t + 3]) - e0;
            let inner_base = (wit * tail_len + t) * 4 * 2;
            if inner_base + 4 * 2 > inner_partials.len() {
                return;
            }
            {
                let i0 = Kx::from_words(inner_partials[inner_base], inner_partials[inner_base + 1]);
                c0 = c0 + e0 * i0;
            }
            {
                let i0 = Kx::from_words(inner_partials[inner_base], inner_partials[inner_base + 1]);
                let i1 = Kx::from_words(inner_partials[inner_base + 2], inner_partials[inner_base + 3]);
                c1 = c1 + e0 * i1 + e1 * i0;
            }
            {
                let i1 = Kx::from_words(inner_partials[inner_base + 2], inner_partials[inner_base + 3]);
                let i2 = Kx::from_words(inner_partials[inner_base + 4], inner_partials[inner_base + 5]);
                c2 = c2 + e0 * i2 + e1 * i1;
            }
            {
                let i2 = Kx::from_words(inner_partials[inner_base + 4], inner_partials[inner_base + 5]);
                let i3 = Kx::from_words(inner_partials[inner_base + 6], inner_partials[inner_base + 7]);
                c3 = c3 + e0 * i3 + e1 * i2;
                c4 = c4 + e1 * i3;
            }
        }

        let out_base = partial_group * NC_COEFFS * 2;
        if out_base + NC_COEFFS * 2 > partials.len() {
            return;
        }
        let coeff_words = [
            c0.as_words(),
            c1.as_words(),
            c2.as_words(),
            c3.as_words(),
            c4.as_words(),
        ];
        for (coefficient, words) in coeff_words.iter().enumerate() {
            unsafe {
                *partials.get_unchecked_mut(out_base + 2 * coefficient) = words[0];
                *partials.get_unchecked_mut(out_base + 2 * coefficient + 1) = words[1];
            }
        }
    }

    /// One thread per (witness, column): lift one plane word to a K pair.
    #[kernel]
    pub fn nc_widen_planes(
        planes: &[u64],
        plane_len: u32,
        len: u32,
        wit_stride: u32,
        num_wits: u32,
        mut digits: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let plane_len = plane_len as usize;
        let len = len as usize;
        if idx >= (num_wits as usize) * len {
            return;
        }
        let wit = idx / len;
        let col = idx % len;
        let src_at = wit * plane_len + col;
        if src_at >= planes.len() {
            return;
        }
        let at = (wit * (wit_stride as usize) + col) * 2;
        if at + 2 > digits.len() {
            return;
        }
        unsafe {
            *digits.get_unchecked_mut(at) = planes[src_at];
            *digits.get_unchecked_mut(at + 1) = 0;
        }
    }

    /// One thread per witness: coefficients of the NC Ajtai-tail round after
    /// the column point is fixed. This mirrors `NcOracle::evals_ajtai_phase`
    /// for b=2, but emits coefficients directly instead of host interpolation.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_ajtai_tail_partials(
        digits: &[u64],
        eq: &[u64],
        beta_a: &[u64],
        gamma: &[u64],
        challenges: &[u64],
        mode: u32,
        width: u32,
        wit_stride: u32,
        num_wits: u32,
        col_rounds: u32,
        tail_round: u32,
        tail_rounds: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let witness = thread::index_1d().get();
        let num_wits = num_wits as usize;
        if witness >= num_wits || tail_round >= tail_rounds {
            return;
        }
        let width = width as usize;
        let wit_stride = wit_stride as usize;
        let col_rounds = col_rounds as usize;
        let bit = tail_round as usize;
        let ell_d = tail_rounds as usize;
        if beta_a.len() < 2 * ell_d || gamma.len() < 2 || challenges.len() < 2 * (col_rounds + bit) || eq.len() < 2 {
            return;
        }

        let beta_j = read_k_pair(beta_a, bit);
        let beta0 = Kx::ONE - beta_j;
        let beta1 = beta_j + beta_j - Kx::ONE;
        let eq_beta_m = read_k_pair(eq, 0);
        let mut eq_beta_pref = Kx::ONE;
        let tail_challenge_word_offset = 2 * col_rounds;
        for prefix_bit in 0..bit {
            let challenge = read_k_pair(challenges, col_rounds + prefix_bit);
            let beta = read_k_pair(beta_a, prefix_bit);
            eq_beta_pref = eq_beta_pref * eq_lin(challenge, beta);
        }

        let stride = 1usize << bit;
        let head_stride = stride << 1;
        let free_tail_bits = ell_d - bit - 1;
        let tail_count = 1usize << free_tail_bits;
        let mut inner0 = Kx::ZERO;
        let mut inner1 = Kx::ZERO;
        let mut inner2 = Kx::ZERO;
        let mut inner3 = Kx::ZERO;
        for tail_mask in 0..tail_count {
            let lane0 = tail_mask * head_stride;
            let lane1 = lane0 + stride;
            let lo = prefolded_digit(
                digits,
                mode,
                width,
                wit_stride,
                witness,
                lane0,
                bit,
                challenges,
                tail_challenge_word_offset,
            );
            let hi = prefolded_digit(
                digits,
                mode,
                width,
                wit_stride,
                witness,
                lane1,
                bit,
                challenges,
                tail_challenge_word_offset,
            );
            let weight = tail_weight(beta_a, bit + 1, tail_mask, free_tail_bits);
            let (r0, r1, r2, r3) = range_components(weight, lo, hi - lo);
            inner0 = inner0 + r0;
            inner1 = inner1 + r1;
            inner2 = inner2 + r2;
            inner3 = inner3 + r3;
        }

        let scale = eq_beta_m * eq_beta_pref * gamma_power(Kx::from_words(gamma[0], gamma[1]), witness);
        let c0 = scale * beta0 * inner0;
        let c1 = scale * (beta0 * inner1 + beta1 * inner0);
        let c2 = scale * (beta0 * inner2 + beta1 * inner1);
        let c3 = scale * (beta0 * inner3 + beta1 * inner2);
        let c4 = scale * beta1 * inner3;
        let coeffs = [
            c0.as_words(),
            c1.as_words(),
            c2.as_words(),
            c3.as_words(),
            c4.as_words(),
        ];
        let out_base = witness * NC_COEFFS * 2;
        if out_base + NC_COEFFS * 2 > partials.len() {
            return;
        }
        for (degree, words) in coeffs.iter().enumerate() {
            unsafe {
                *partials.get_unchecked_mut(out_base + 2 * degree) = words[0];
                *partials.get_unchecked_mut(out_base + 2 * degree + 1) = words[1];
            }
        }
    }

    /// One thread per (witness, lane) plus one for `eq[0]`: gather the
    /// fully folded digit row exactly as the host decode did — strided
    /// row 0's window starts at lane 0 (lane j = slot j, zero past the
    /// width), dense reads the row directly.
    #[kernel]
    pub fn nc_pack_final_state(
        digits: &[u64],
        eq: &[u64],
        mode: u32,
        width: u32,
        wit_stride: u32,
        num_wits: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let num_wits = num_wits as usize;
        let width = width as usize;
        if idx > num_wits * RING_LANES {
            return;
        }
        if idx == num_wits * RING_LANES {
            let at = 2 * idx;
            if eq.len() < 2 || at + 2 > out.len() {
                return;
            }
            unsafe {
                *out.get_unchecked_mut(at) = eq[0];
                *out.get_unchecked_mut(at + 1) = eq[1];
            }
            return;
        }
        let wit = idx / RING_LANES;
        let rho = idx % RING_LANES;
        let live = mode != NC_MODE_STRIDED || rho < width;
        let base = wit * (wit_stride as usize) * 2 + 2 * rho;
        let (w0, w1) = if live && base + 2 <= digits.len() {
            (digits[base], digits[base + 1])
        } else {
            (0, 0)
        };
        let at = 2 * idx;
        if at + 2 > out.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(at) = w0;
            *out.get_unchecked_mut(at + 1) = w1;
        }
    }

    /// Strided digit fold while merge windows stay lane-disjoint: a pure
    /// slot-wise scale, `dst[f] = src[f] · (1-r | r)` by half-window parity.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_strided(
        src: &[u64],
        src_len: u32,
        width: u32,
        out_len: u32,
        wit_stride: u32,
        num_wits: u32,
        r_c0: u64,
        r_c1: u64,
        mut dst: DisjointSlice<u64>,
    ) {
        fold_strided_at(
            thread::index_1d().get(),
            src,
            src_len as usize,
            width as usize,
            out_len as usize,
            wit_stride as usize,
            num_wits as usize,
            Kx::from_words(r_c0, r_c1),
            &mut dst,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_strided_from_challenge(
        src: &[u64],
        src_len: u32,
        width: u32,
        out_len: u32,
        wit_stride: u32,
        num_wits: u32,
        challenge: &[u64],
        challenge_offset: u32,
        mut dst: DisjointSlice<u64>,
    ) {
        let offset = challenge_offset as usize;
        if offset + 1 >= challenge.len() {
            return;
        }
        fold_strided_at(
            thread::index_1d().get(),
            src,
            src_len as usize,
            width as usize,
            out_len as usize,
            wit_stride as usize,
            num_wits as usize,
            Kx::from_words(challenge[offset], challenge[offset + 1]),
            &mut dst,
        );
    }

    fn fold_strided_at(
        idx: usize,
        src: &[u64],
        src_len: usize,
        width: usize,
        out_len: usize,
        wit_stride: usize,
        num_wits: usize,
        r: Kx,
        dst: &mut DisjointSlice<u64>,
    ) {
        if idx >= num_wits * out_len {
            return;
        }
        let wit = idx / out_len;
        let f = idx % out_len;
        let base = wit * wit_stride * 2;
        let value = if f < src_len && base + 2 * f + 2 <= src.len() {
            Kx::from_words(src[base + 2 * f], src[base + 2 * f + 1])
        } else {
            Kx::ZERO
        };
        // Strided widths double from 1, so `2 * width` is a power of two.
        let scale = if (f & (2 * width - 1)) < width { Kx::ONE - r } else { r };
        let words = (value * scale).as_words();
        let out_at = base + 2 * f;
        if out_at + 2 > dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(out_at) = words[0];
            *dst.get_unchecked_mut(out_at + 1) = words[1];
        }
    }

    /// Terminal strided fold (`2·width > 54`): gather each output lane from
    /// the two windowed source rows and fold, materializing dense rows.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_strided_to_dense(
        src: &[u64],
        src_len: u32,
        width: u32,
        rows: u32,
        out_rows: u32,
        wit_stride: u32,
        dense_stride: u32,
        num_wits: u32,
        r_c0: u64,
        r_c1: u64,
        mut dst: DisjointSlice<u64>,
    ) {
        fold_strided_to_dense_at(
            thread::index_1d().get(),
            src,
            src_len as usize,
            width as usize,
            rows as usize,
            out_rows as usize,
            wit_stride as usize,
            dense_stride as usize,
            num_wits as usize,
            Kx::from_words(r_c0, r_c1),
            &mut dst,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_strided_to_dense_from_challenge(
        src: &[u64],
        src_len: u32,
        width: u32,
        rows: u32,
        out_rows: u32,
        wit_stride: u32,
        dense_stride: u32,
        num_wits: u32,
        challenge: &[u64],
        challenge_offset: u32,
        mut dst: DisjointSlice<u64>,
    ) {
        let offset = challenge_offset as usize;
        if offset + 1 >= challenge.len() {
            return;
        }
        fold_strided_to_dense_at(
            thread::index_1d().get(),
            src,
            src_len as usize,
            width as usize,
            rows as usize,
            out_rows as usize,
            wit_stride as usize,
            dense_stride as usize,
            num_wits as usize,
            Kx::from_words(challenge[offset], challenge[offset + 1]),
            &mut dst,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_strided_to_dense_at(
        idx: usize,
        src: &[u64],
        src_len: usize,
        width: usize,
        rows: usize,
        out_rows: usize,
        wit_stride: usize,
        dense_stride: usize,
        num_wits: usize,
        r: Kx,
        dst: &mut DisjointSlice<u64>,
    ) {
        if idx >= num_wits * out_rows * RING_LANES {
            return;
        }
        let wit = idx / (out_rows * RING_LANES);
        let rest = idx % (out_rows * RING_LANES);
        let out_row = rest / RING_LANES;
        let rho = rest % RING_LANES;

        let base = wit * wit_stride * 2;
        let lane_value = |row: usize| -> Kx {
            if row >= rows {
                return Kx::ZERO;
            }
            let start = (row * width) % RING_LANES;
            let j = (rho + RING_LANES - start) % RING_LANES;
            let flat = row * width + j;
            if j < width && flat < src_len && base + 2 * flat + 2 <= src.len() {
                Kx::from_words(src[base + 2 * flat], src[base + 2 * flat + 1])
            } else {
                Kx::ZERO
            }
        };
        let lo = lane_value(2 * out_row);
        let hi = lane_value(2 * out_row + 1);
        let words = (lo + (hi - lo) * r).as_words();
        let out_at = (wit * dense_stride + out_row * RING_LANES + rho) * 2;
        if out_at + 2 > dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(out_at) = words[0];
            *dst.get_unchecked_mut(out_at + 1) = words[1];
        }
    }

    /// Dense digit fold: pairwise rows, one thread per output lane.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_dense(
        src: &[u64],
        rows: u32,
        out_rows: u32,
        dense_stride: u32,
        num_wits: u32,
        r_c0: u64,
        r_c1: u64,
        mut dst: DisjointSlice<u64>,
    ) {
        fold_dense_at(
            thread::index_1d().get(),
            src,
            rows as usize,
            out_rows as usize,
            dense_stride as usize,
            num_wits as usize,
            Kx::from_words(r_c0, r_c1),
            &mut dst,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nc_fold_dense_from_challenge(
        src: &[u64],
        rows: u32,
        out_rows: u32,
        dense_stride: u32,
        num_wits: u32,
        challenge: &[u64],
        challenge_offset: u32,
        mut dst: DisjointSlice<u64>,
    ) {
        let offset = challenge_offset as usize;
        if offset + 1 >= challenge.len() {
            return;
        }
        fold_dense_at(
            thread::index_1d().get(),
            src,
            rows as usize,
            out_rows as usize,
            dense_stride as usize,
            num_wits as usize,
            Kx::from_words(challenge[offset], challenge[offset + 1]),
            &mut dst,
        );
    }

    fn fold_dense_at(
        idx: usize,
        src: &[u64],
        rows: usize,
        out_rows: usize,
        dense_stride: usize,
        num_wits: usize,
        r: Kx,
        dst: &mut DisjointSlice<u64>,
    ) {
        if idx >= num_wits * out_rows * RING_LANES {
            return;
        }
        let wit = idx / (out_rows * RING_LANES);
        let rest = idx % (out_rows * RING_LANES);
        let out_row = rest / RING_LANES;
        let rho = rest % RING_LANES;

        let read_row = |row: usize| -> Kx {
            if row >= rows {
                return Kx::ZERO;
            }
            let at = (wit * dense_stride + row * RING_LANES + rho) * 2;
            if at + 2 > src.len() {
                return Kx::ZERO;
            }
            Kx::from_words(src[at], src[at + 1])
        };
        let lo = read_row(2 * out_row);
        let hi = read_row(2 * out_row + 1);
        let words = (lo + (hi - lo) * r).as_words();
        let out_at = (wit * dense_stride + out_row * RING_LANES + rho) * 2;
        if out_at + 2 > dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(out_at) = words[0];
            *dst.get_unchecked_mut(out_at + 1) = words[1];
        }
    }
}
