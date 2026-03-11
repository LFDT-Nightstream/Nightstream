//! Minimal protocol utilities for paper-exact implementation
//!
//! Contains only the essential functions needed by prove and verify.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CeClaim, SparsePoly};
use neo_gpu::ProverComputeBackend;
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::{labels as tr_labels, Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

/// Computed dimensions for the CCS reduction
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub ell_d: usize,
    pub ell_n: usize,
    pub ell_m: usize,
    pub ell: usize,
    pub ell_nc: usize,
    pub ell_max: usize,
    pub d_sc: usize,
}

pub use crate::optimized_engine::Challenges;

#[inline]
fn degree_bound_nc(params: &NeoParams) -> usize {
    // `range_product` has degree (2b-1) in `y` for the symmetric range [-(b-1), ..., (b-1)].
    // Multiplying by an `eq_lin` factor in the current sumcheck variable adds +1.
    //
    // So the per-round univariate degree bound for the NC polynomial is ≤ 2b.
    core::cmp::max(2, 2 * (params.b as usize))
}

/// Build dimensions and validate extension field security policy
pub fn build_dims_and_policy(params: &NeoParams, s: &CcsStructure<F>) -> Result<Dims, PiCcsError> {
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }

    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;

    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;

    let m_pad = s.m.next_power_of_two().max(2);
    let ell_m = m_pad.trailing_zeros() as usize;

    let ell = ell_d + ell_n;
    let ell_nc = ell_d + ell_m;
    let ell_max = core::cmp::max(ell, ell_nc);

    let d_sc = core::cmp::max(s.max_degree() as usize + 1, degree_bound_nc(params));

    let ext = params
        .extension_check(ell_max as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits",
            ext.slack_bits
        )));
    }

    Ok(Dims {
        ell_d,
        ell_n,
        ell_m,
        ell,
        ell_nc,
        ell_max,
        d_sc,
    })
}

/// Bind header and MCS instances to transcript
pub fn bind_header_and_instances(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    dims: Dims,
) -> Result<(), PiCcsError> {
    let digest = digest_ccs_matrices(s);
    bind_header_and_instances_with_digest(tr, params, s, mcs_list, dims, &digest)
}

/// Bind CCS header and MCS instances to transcript, using a precomputed CCS matrix digest.
///
/// This is performance-critical in shard folding, where the same `s` is reused across many steps.
pub fn bind_header_and_instances_with_digest(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    dims: Dims,
    mat_digest: &[Goldilocks],
) -> Result<(), PiCcsError> {
    tr.append_message(tr_labels::PI_CCS, b"");

    let ext = params
        .extension_check(dims.ell_max as u32, dims.d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_message(b"neo/pi_ccs/variant/v1", b"SplitNcV1");

    tr.append_u64s(
        b"ccs/header",
        &[
            64,
            ext.s_supported as u64,
            params.lambda as u64,
            dims.ell as u64,
            dims.ell_nc as u64,
            dims.ell_max as u64,
            dims.d_sc as u64,
            ext.slack_bits.unsigned_abs() as u64,
        ],
    );
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 { 1 } else { 0 }]);

    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(
        b"dims/v2",
        &[
            s.n as u64,
            s.m as u64,
            s.t() as u64,
            dims.ell_d as u64,
            dims.ell_n as u64,
            dims.ell_m as u64,
        ],
    );

    if mat_digest.len() != 4 {
        return Err(PiCcsError::InvalidInput(format!(
            "CCS matrix digest must have len 4, got {}",
            mat_digest.len()
        )));
    }
    for &digest_elem in mat_digest {
        tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]);
    }

    absorb_sparse_polynomial(tr, &s.f);

    for inst in mcs_list.iter() {
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        tr.append_fields(b"c_data", &inst.c.data);
    }

    Ok(())
}

/// Bind ME inputs to transcript
pub fn bind_me_inputs(tr: &mut Poseidon2Transcript, me_inputs: &[CeClaim<Cmt, F, K>]) -> Result<(), PiCcsError> {
    bind_me_inputs_with_backend(tr, me_inputs, &ProverComputeBackend::Cpu)
}

/// Bind ME inputs to transcript using the selected compute backend for batched Poseidon2 hashing.
pub fn bind_me_inputs_with_backend(
    tr: &mut Poseidon2Transcript,
    me_inputs: &[CeClaim<Cmt, F, K>],
    compute_backend: &ProverComputeBackend,
) -> Result<(), PiCcsError> {
    let backend_ctx = crate::accelerator::BackendContext::new(compute_backend)?;
    bind_me_inputs_with_context(tr, me_inputs, &backend_ctx)
}

pub fn bind_me_inputs_with_context(
    tr: &mut Poseidon2Transcript,
    me_inputs: &[CeClaim<Cmt, F, K>],
    backend_ctx: &crate::accelerator::BackendContext,
) -> Result<(), PiCcsError> {
    // v5: bind each ME input via a compact domain-separated Poseidon2 digest.
    // Encoding update: K-coefficient width is encoded once per K-slice (not per K element).
    tr.append_message(b"neo/ccs/me_inputs/v5", b"");
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    #[inline]
    fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
        const BYTES_PER_LIMB: usize = 7;
        dst.push(F::from_u64(bytes.len() as u64));
        for chunk in bytes.chunks(BYTES_PER_LIMB) {
            let mut limb = [0u8; 8];
            limb[..chunk.len()].copy_from_slice(chunk);
            dst.push(F::from_u64(u64::from_le_bytes(limb)));
        }
    }

    #[inline]
    fn extend_f_slice(dst: &mut Vec<F>, vals: &[F]) {
        dst.push(F::from_u64(vals.len() as u64));
        dst.extend_from_slice(vals);
    }

    #[inline]
    fn extend_k_slice(dst: &mut Vec<F>, vals: &[K]) {
        dst.push(F::from_u64(vals.len() as u64));
        let coeffs_len = vals.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        dst.push(F::from_u64(coeffs_len as u64));
        for v in vals {
            let coeffs = v.as_coeffs();
            debug_assert_eq!(
                coeffs.len(),
                coeffs_len,
                "non-uniform K coeff length in ME digest encoding"
            );
            dst.extend(coeffs.iter().copied());
        }
    }

    #[inline]
    fn poseidon_digest32_fields(input: &[F]) -> [u8; 32] {
        let digest = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(input);
        let mut out = [0u8; 32];
        for (i, limb) in digest.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.as_canonical_u64().to_le_bytes());
        }
        out
    }

    #[inline]
    fn me_digest_poseidon_input(me: &CeClaim<Cmt, F, K>) -> Vec<F> {
        let mut digest_input = Vec::<F>::with_capacity(2048);
        extend_packed_bytes_as_fields(&mut digest_input, b"neo/ccs/me_input_digest_poseidon/v2");

        extend_f_slice(&mut digest_input, &me.c.data);
        extend_f_slice(&mut digest_input, me.X.as_slice());
        extend_k_slice(&mut digest_input, &me.r);
        extend_k_slice(&mut digest_input, &me.s_col);
        extend_k_slice(&mut digest_input, &me.y_zcol);

        digest_input.push(F::from_u64(me.y_ring.len() as u64));
        for row in &me.y_ring {
            extend_k_slice(&mut digest_input, row);
        }

        extend_k_slice(&mut digest_input, &me.ct);
        extend_k_slice(&mut digest_input, &me.aux_openings);
        extend_f_slice(&mut digest_input, &me.c_step_coords);
        digest_input.push(F::from_u64(me.m_in as u64));
        digest_input.push(F::from_u64(me.u_offset as u64));
        digest_input.push(F::from_u64(me.u_len as u64));
        extend_packed_bytes_as_fields(&mut digest_input, &me.fold_digest);
        digest_input
    }

    if backend_ctx.supports_poseidon2() {
        let digest_inputs = me_inputs
            .iter()
            .map(me_digest_poseidon_input)
            .collect::<Vec<_>>();
        if let Some(digests) = crate::accelerator::poseidon2_digest32_many_with_context(backend_ctx, &digest_inputs)? {
            for digest in digests {
                tr.append_bytes_packed(b"me_digest", &digest);
            }
            return Ok(());
        }

        for digest_input in digest_inputs {
            tr.append_bytes_packed(b"me_digest", &poseidon_digest32_fields(&digest_input));
        }
        return Ok(());
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
    #[cfg(not(any(not(target_arch = "wasm32"), feature = "wasm-threads")))]
    let _allow_parallel = false;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let digests: Vec<[u8; 32]> = if allow_parallel && me_inputs.len() >= 8 {
        use rayon::prelude::*;
        me_inputs
            .par_iter()
            .map(|me| poseidon_digest32_fields(&me_digest_poseidon_input(me)))
            .collect()
    } else {
        me_inputs
            .iter()
            .map(|me| poseidon_digest32_fields(&me_digest_poseidon_input(me)))
            .collect()
    };
    #[cfg(not(any(not(target_arch = "wasm32"), feature = "wasm-threads")))]
    let digests: Vec<[u8; 32]> = me_inputs
        .iter()
        .map(|me| poseidon_digest32_fields(&me_digest_poseidon_input(me)))
        .collect();

    for digest in digests {
        tr.append_bytes_packed(b"me_digest", &digest);
    }

    Ok(())
}

/// Validate CE `ct` semantics on outputs (SuperNeo-only).
///
/// `ct[j] = constant_term(y_ring[j])`.
pub fn validate_ct_constant_term<Ff>(
    s: &CcsStructure<Ff>,
    _params: &NeoParams,
    me_outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    for (idx, out) in me_outputs.iter().enumerate() {
        if out.ct.len() < s.t() {
            return Err(PiCcsError::ProtocolError(format!(
                "me_outputs[{idx}].ct.len()={} is smaller than s.t()={}",
                out.ct.len(),
                s.t()
            )));
        }

        for j in 0..s.t() {
            let row = &out.y_ring[j];
            let want = crate::common::ct_from_y_digits(row);
            if out.ct[j] != want {
                return Err(PiCcsError::ProtocolError(format!(
                    "me_outputs[{idx}].ct[{j}] does not match SuperNeo constant-term semantics"
                )));
            }
        }
    }

    Ok(())
}

/// Validate that all ME inputs share the same evaluation point `r`.
///
/// Returns `None` when `me_inputs` is empty, otherwise returns a shared `r` slice.
pub fn shared_me_input_r<'a, C, Ff>(
    me_inputs: &'a [CeClaim<C, Ff, K>],
    ell_n: usize,
) -> Result<Option<&'a [K]>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if me_inputs.is_empty() {
        return Ok(None);
    }

    let r0 = me_inputs[0].r.as_slice();
    if r0.len() != ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "ME input r length mismatch at accumulator #0: expected ell_n = {}, got {}",
            ell_n,
            r0.len()
        )));
    }

    for (idx, me) in me_inputs.iter().enumerate().skip(1) {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx,
                ell_n,
                me.r.len()
            )));
        }
        if me.r.as_slice() != r0 {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r mismatch at accumulator #{}: all ME inputs must share the same r",
                idx
            )));
        }
    }

    Ok(Some(r0))
}

/// Validate MCS-output `X` content against public `x` under SuperNeo packed semantics.
///
/// `x[c] = X[c % D, c]` and all off-lane rows in column `c` must be zero.
pub fn validate_mcs_output_x_recomposition<Ff>(
    _params: &NeoParams,
    ccs_m: usize,
    mcs_list: &[CcsClaim<Cmt, Ff>],
    me_outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if ccs_m == 0 {
        return Err(PiCcsError::InvalidInput("CCS width m must be > 0".into()));
    }
    for (idx, inst) in mcs_list.iter().enumerate() {
        let out = me_outputs.get(idx).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "missing me_outputs entry for mcs_list index {} (|me_outputs|={})",
                idx,
                me_outputs.len()
            ))
        })?;

        if inst.x.len() != inst.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "mcs_list[{idx}].x.len()={}, expected m_in={}",
                inst.x.len(),
                inst.m_in
            )));
        }
        if out.X.cols() != inst.m_in {
            return Err(PiCcsError::ProtocolError(format!(
                "me_outputs[{idx}].X cols mismatch (got {}, expected {})",
                out.X.cols(),
                inst.m_in
            )));
        }

        for c in 0..inst.m_in {
            let lane = c % D;
            let got = out.X[(lane, c)];
            if got != inst.x[c] {
                return Err(PiCcsError::ProtocolError(format!(
                    "me_outputs[{idx}].X lane {} at column {} does not match mcs_list[{idx}].x[{}]",
                    lane, c, c
                )));
            }
            if let Some((rho, _)) = (0..D)
                .filter(|&rho| rho != lane)
                .map(|rho| (rho, out.X[(rho, c)]))
                .find(|(_, v)| *v != Ff::ZERO)
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "me_outputs[{idx}].X column {} has non-zero off-lane row {} in SuperNeo packed layout",
                    c, rho
                )));
            }
        }
    }

    Ok(())
}

/// Sample challenges α, β, γ from transcript
pub fn sample_challenges(tr: &mut Poseidon2Transcript, ell_d: usize, ell: usize) -> Result<Challenges, PiCcsError> {
    tr.append_message(b"neo/ccs/chals/v1", b"");

    let alpha: Vec<K> = (0..ell_d)
        .map(|_| {
            let c = tr.challenge_fields(b"chal/k", 2);
            neo_math::from_complex(c[0], c[1])
        })
        .collect();

    let beta: Vec<K> = (0..ell)
        .map(|_| {
            let c = tr.challenge_fields(b"chal/k", 2);
            neo_math::from_complex(c[0], c[1])
        })
        .collect();

    let (beta_a, beta_r) = beta.split_at(ell_d);

    let g = tr.challenge_fields(b"chal/k", 2);
    let gamma = neo_math::from_complex(g[0], g[1]);

    Ok(Challenges {
        alpha,
        beta_a: beta_a.to_vec(),
        beta_r: beta_r.to_vec(),
        beta_m: Vec::new(),
        gamma,
    })
}

/// Sample the column-domain β_m ∈ K^{log m} for the split-NC variant.
pub fn sample_beta_m(tr: &mut Poseidon2Transcript, ell_m: usize) -> Result<Vec<K>, PiCcsError> {
    tr.append_message(b"neo/ccs/chals/nc_beta_m/v1", b"");

    let beta_m: Vec<K> = (0..ell_m)
        .map(|_| {
            let c = tr.challenge_fields(b"chal/k", 2);
            neo_math::from_complex(c[0], c[1])
        })
        .collect();

    Ok(beta_m)
}

pub fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

    const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);

    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;

    const DOMAIN_STRING: &[u8] = b"neo/ccs/matrices/v1";
    for &byte in DOMAIN_STRING {
        if absorbed >= 15 {
            poseidon2.permute_mut(&mut state);
            absorbed = 0;
        }
        state[absorbed] = Goldilocks::from_u32(byte as u32);
        absorbed += 1;
    }

    if absorbed + 3 >= 16 {
        poseidon2.permute_mut(&mut state);
        absorbed = 0;
    }
    state[absorbed] = Goldilocks::from_u64(s.n as u64);
    state[absorbed + 1] = Goldilocks::from_u64(s.m as u64);
    state[absorbed + 2] = Goldilocks::from_u64(s.t() as u64);

    poseidon2.permute_mut(&mut state);

    for (j, matrix) in s.matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;

        let mut emit = |row: usize, col: usize, val_u64: u64| {
            if absorbed + 3 > 15 {
                poseidon2.permute_mut(&mut state);
                absorbed = 0;
            }
            state[absorbed] = Goldilocks::from_u64(row as u64);
            state[absorbed + 1] = Goldilocks::from_u64(col as u64);
            state[absorbed + 2] = Goldilocks::from_u64(val_u64);
            absorbed += 3;
        };

        match matrix {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, s.n);
                debug_assert_eq!(*n, s.m);
                let one_u = F::ONE.as_canonical_u64();
                for row in 0..s.n {
                    emit(row, row, one_u);
                }
            }
            CcsMatrix::Csc(csc) => {
                // Enumerate non-zeros in row-major order (matches dense scan) without allocating
                // a `Vec<Vec<_>>` of length `nrows` (which is massive for large circuits).
                //
                // Strategy: build CSR-style row segments in one contiguous allocation.
                let nrows = csc.nrows;
                let nnz = csc.vals.len();
                debug_assert_eq!(csc.row_idx.len(), nnz);

                // 1) Count entries per row.
                let mut row_counts = vec![0u32; nrows];
                for &r in csc.row_idx.iter() {
                    row_counts[r] += 1;
                }

                // 2) Prefix sums to get row offsets.
                let mut row_offsets = vec![0usize; nrows + 1];
                for r in 0..nrows {
                    row_offsets[r + 1] = row_offsets[r] + (row_counts[r] as usize);
                }
                debug_assert_eq!(row_offsets[nrows], nnz);

                // 3) Fill (col,val) pairs into per-row segments while scanning columns in order.
                let mut write_pos = row_offsets[..nrows].to_vec();
                let mut entries = vec![(0usize, 0u64); nnz];

                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        let idx = write_pos[row];
                        write_pos[row] = idx + 1;
                        entries[idx] = (col, csc.vals[k].as_canonical_u64());
                    }
                }

                // 4) Emit in row-major order.
                for row in 0..nrows {
                    let start = row_offsets[row];
                    let end = row_offsets[row + 1];
                    for &(col, val_u64) in &entries[start..end] {
                        emit(row, col, val_u64);
                    }
                }
            }
        }

        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
}

/// Compute the CCS matrix digest, optionally using a prebuilt sparse cache.
///
/// This cache-aware variant uses a native CSC encoding (`v2-csc`) to avoid the expensive
/// to avoid the expensive row-major reconstruction/allocation required by `digest_ccs_matrices`.
/// Prover/verifier soundness is preserved because both sides bind this digest into transcript
/// under the same domain and full structural content.
pub fn digest_ccs_matrices_with_sparse_cache<Ff: Field + PrimeField64>(
    s: &CcsStructure<Ff>,
    sparse: Option<&crate::engines::optimized_engine::oracle::SparseCache<Ff>>,
) -> Vec<Goldilocks> {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

    #[inline]
    fn absorb_u64(poseidon2: &Poseidon2Goldilocks<16>, state: &mut [Goldilocks; 16], absorbed: &mut usize, v: u64) {
        if *absorbed >= 15 {
            poseidon2.permute_mut(state);
            *absorbed = 0;
        }
        state[*absorbed] = Goldilocks::from_u64(v);
        *absorbed += 1;
    }

    const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);

    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0usize;
    const DOMAIN_STRING: &[u8] = b"neo/ccs/matrices/v2-csc";
    for &byte in DOMAIN_STRING {
        absorb_u64(&poseidon2, &mut state, &mut absorbed, byte as u64);
    }
    absorb_u64(&poseidon2, &mut state, &mut absorbed, s.n as u64);
    absorb_u64(&poseidon2, &mut state, &mut absorbed, s.m as u64);
    absorb_u64(&poseidon2, &mut state, &mut absorbed, s.t() as u64);
    poseidon2.permute_mut(&mut state);

    for (j, matrix) in s.matrices.iter().enumerate() {
        absorbed = 0;
        absorb_u64(&poseidon2, &mut state, &mut absorbed, j as u64);

        match matrix {
            CcsMatrix::Identity { n } => {
                absorb_u64(&poseidon2, &mut state, &mut absorbed, 1); // tag=identity
                absorb_u64(&poseidon2, &mut state, &mut absorbed, *n as u64);
            }
            CcsMatrix::Csc(csc_from_s) => {
                let cached_csc = sparse.and_then(|sp| sp.csc(j));
                #[cfg(debug_assertions)]
                if let Some(c) = cached_csc {
                    debug_assert_eq!(c.nrows, csc_from_s.nrows, "CSC cache nrows mismatch for matrix {j}");
                    debug_assert_eq!(c.ncols, csc_from_s.ncols, "CSC cache ncols mismatch for matrix {j}");
                    debug_assert_eq!(
                        c.col_ptr, csc_from_s.col_ptr,
                        "CSC cache col_ptr mismatch for matrix {j}"
                    );
                    debug_assert_eq!(
                        c.row_idx, csc_from_s.row_idx,
                        "CSC cache row_idx mismatch for matrix {j}"
                    );
                    debug_assert_eq!(c.vals, csc_from_s.vals, "CSC cache vals mismatch for matrix {j}");
                }
                let (nrows, ncols, col_ptr, row_idx, vals) = if let Some(c) = cached_csc {
                    (
                        c.nrows,
                        c.ncols,
                        c.col_ptr.as_slice(),
                        c.row_idx.as_slice(),
                        c.vals.as_slice(),
                    )
                } else {
                    (
                        csc_from_s.nrows,
                        csc_from_s.ncols,
                        csc_from_s.col_ptr.as_slice(),
                        csc_from_s.row_idx.as_slice(),
                        csc_from_s.vals.as_slice(),
                    )
                };

                absorb_u64(&poseidon2, &mut state, &mut absorbed, 2); // tag=CSC
                absorb_u64(&poseidon2, &mut state, &mut absorbed, nrows as u64);
                absorb_u64(&poseidon2, &mut state, &mut absorbed, ncols as u64);
                absorb_u64(&poseidon2, &mut state, &mut absorbed, vals.len() as u64);
                absorb_u64(&poseidon2, &mut state, &mut absorbed, col_ptr.len() as u64);

                for &cp in col_ptr {
                    absorb_u64(&poseidon2, &mut state, &mut absorbed, cp as u64);
                }
                for &r in row_idx {
                    absorb_u64(&poseidon2, &mut state, &mut absorbed, r as u64);
                }
                for &v in vals {
                    absorb_u64(&poseidon2, &mut state, &mut absorbed, v.as_canonical_u64());
                }
            }
        }

        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
}

fn absorb_sparse_polynomial(tr: &mut Poseidon2Transcript, f: &SparsePoly<F>) {
    tr.append_message(b"neo/ccs/poly", b"");
    tr.append_u64s(b"arity", &[f.arity() as u64]);
    tr.append_u64s(b"terms_len", &[f.terms().len() as u64]);

    let mut terms: Vec<_> = f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps);

    for term in terms {
        tr.append_fields(b"coeff", &[term.coeff]);
        let exps: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
        tr.append_u64s(b"exps", &exps);
    }
}
