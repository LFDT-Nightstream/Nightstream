use std::sync::Arc;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_gpu::{connect, DeviceApi, FlatK, MojoSession, MojoSplitNcEvaluator, ProverComputeBackend};
use neo_math::{from_complex, F as BaseF, K, KExtensions};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::engines::optimized_engine::oracle::{NcOracle, OptimizedOracle, SparseCache};
use crate::sumcheck::RoundOracle;
use crate::PiCcsError;

const POSEIDON2_MIN_PERMUTATIONS_CPU_DIRECT: usize = 32;
const POSEIDON2_MIN_PERMUTATIONS_METAL: usize = 128;
const POSEIDON2_MIN_PERMUTATIONS_CUDA: usize = 32;
const POSEIDON2_MIN_PERMUTATIONS_HIP: usize = 32;

pub struct BackendContext {
    session: Option<MojoSession>,
    requested_mojo: bool,
    allow_cpu_fallback: bool,
    supports_split_nc: bool,
    supports_poseidon2: bool,
}

impl BackendContext {
    pub fn new(backend: &ProverComputeBackend) -> Result<Self, PiCcsError> {
        match backend {
            ProverComputeBackend::Cpu => Ok(Self {
                session: None,
                requested_mojo: false,
                allow_cpu_fallback: false,
                supports_split_nc: false,
                supports_poseidon2: false,
            }),
            ProverComputeBackend::Mojo(cfg) => match connect(cfg) {
                Ok(session) => {
                    let supports_split_nc = session.supports_split_nc_api();
                    let supports_poseidon2 = session.supports_poseidon2_api();
                    if !supports_split_nc && !supports_poseidon2 {
                        if cfg.fallback_to_cpu {
                            return Ok(Self {
                                session: None,
                                requested_mojo: true,
                                allow_cpu_fallback: true,
                                supports_split_nc: false,
                                supports_poseidon2: false,
                            });
                        }
                        return Err(PiCcsError::ProtocolError(
                            "Mojo backend loaded but does not expose Poseidon2 or Split-NC symbols".into(),
                        ));
                    }
                    Ok(Self {
                        session: Some(session),
                        requested_mojo: true,
                        allow_cpu_fallback: cfg.fallback_to_cpu,
                        supports_split_nc,
                        supports_poseidon2,
                    })
                }
                Err(_err) if cfg.fallback_to_cpu => Ok(Self {
                    session: None,
                    requested_mojo: true,
                    allow_cpu_fallback: true,
                    supports_split_nc: false,
                    supports_poseidon2: false,
                }),
                Err(err) => Err(PiCcsError::ProtocolError(format!(
                    "failed to initialize Mojo backend: {err}"
                ))),
            },
        }
    }

    #[inline]
    pub fn supports_poseidon2(&self) -> bool {
        self.supports_poseidon2
    }

    #[inline]
    pub fn poseidon_session(&self) -> Option<&MojoSession> {
        if self.supports_poseidon2 {
            self.session.as_ref()
        } else {
            None
        }
    }

    pub fn split_nc_session(&self) -> Result<Option<&MojoSession>, PiCcsError> {
        if self.supports_split_nc {
            return Ok(self.session.as_ref());
        }
        if self.requested_mojo && !self.allow_cpu_fallback {
            return Err(PiCcsError::ProtocolError(
                "Mojo backend loaded but does not expose Split-NC evaluator symbols".into(),
            ));
        }
        Ok(None)
    }

    #[inline]
    pub fn split_nc_required(&self) -> bool {
        self.requested_mojo && !self.allow_cpu_fallback
    }

    #[inline]
    pub fn poseidon2_min_permutations(&self) -> usize {
        match self.poseidon_session().map(MojoSession::device_api) {
            Some(DeviceApi::Metal) => POSEIDON2_MIN_PERMUTATIONS_METAL,
            Some(DeviceApi::Cuda) => POSEIDON2_MIN_PERMUTATIONS_CUDA,
            Some(DeviceApi::Hip) => POSEIDON2_MIN_PERMUTATIONS_HIP,
            Some(DeviceApi::Cpu) => POSEIDON2_MIN_PERMUTATIONS_CPU_DIRECT,
            Some(DeviceApi::Auto) | None => usize::MAX,
        }
    }
}

#[inline]
fn add_goldilocks_u64(lhs: u64, rhs: BaseF) -> u64 {
    (BaseF::from_u64(lhs) + rhs).as_canonical_u64()
}

#[inline]
fn flat_k_from_ext(x: K) -> FlatK {
    let (re, im) = x.to_limbs_u64();
    FlatK { re, im }
}

#[inline]
fn ext_k_from_flat(x: FlatK) -> K {
    from_complex(BaseF::from_u64(x.re), BaseF::from_u64(x.im))
}

pub fn poseidon2_digest32_many_with_context(
    backend_ctx: &BackendContext,
    inputs: &[Vec<BaseF>],
) -> Result<Option<Vec<[u8; 32]>>, PiCcsError> {
    if inputs.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let total_permutations = inputs
        .iter()
        .map(|input| input.len().div_ceil(p2::RATE) + 1)
        .sum::<usize>();
    if total_permutations < backend_ctx.poseidon2_min_permutations() {
        return Ok(None);
    }

    let Some(session) = backend_ctx.poseidon_session() else {
        return Ok(None);
    };

    let max_chunks = inputs
        .iter()
        .map(|input| input.len().div_ceil(p2::RATE))
        .max()
        .unwrap_or(0);
    let mut states = vec![[0u64; p2::WIDTH]; inputs.len()];
    let mut batch = Vec::<[u64; p2::WIDTH]>::with_capacity(inputs.len());
    let mut batch_indices = Vec::<usize>::with_capacity(inputs.len());

    for chunk_idx in 0..max_chunks {
        batch.clear();
        batch_indices.clear();
        let start = chunk_idx * p2::RATE;
        for (input_idx, input) in inputs.iter().enumerate() {
            if start >= input.len() {
                continue;
            }
            let end = (start + p2::RATE).min(input.len());
            for (lane, x) in input[start..end].iter().enumerate() {
                states[input_idx][lane] = add_goldilocks_u64(states[input_idx][lane], *x);
            }
            batch_indices.push(input_idx);
            batch.push(states[input_idx]);
        }

        session
            .permute_poseidon2_batch_u64x8(&mut batch)
            .map_err(|err| PiCcsError::ProtocolError(format!("batched Poseidon2 permutation failed: {err}")))?;

        for (input_idx, state) in batch_indices.iter().copied().zip(batch.iter().copied()) {
            states[input_idx] = state;
        }
    }

    batch.clear();
    for state in &mut states {
        state[0] = add_goldilocks_u64(state[0], BaseF::ONE);
        batch.push(*state);
    }

    session
        .permute_poseidon2_batch_u64x8(&mut batch)
        .map_err(|err| PiCcsError::ProtocolError(format!("final batched Poseidon2 permutation failed: {err}")))?;

    let mut digests = Vec::with_capacity(batch.len());
    for state in batch {
        let mut out = [0u8; 32];
        for (i, limb) in state[..p2::DIGEST_LEN].iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        digests.push(out);
    }
    Ok(Some(digests))
}

pub struct SplitNcOptimizedOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    inner: OptimizedOracle<'a, Ff>,
    fe_evaluator: Option<MojoSplitNcEvaluator<'ctx>>,
    split_nc_required: bool,
}

impl<'a, 'ctx, Ff> SplitNcOptimizedOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_sparse(
        s: &'a CcsStructure<Ff>,
        params: &'a NeoParams,
        mcs_witnesses: &'a [CcsWitness<Ff>],
        me_witnesses: &'a [Mat<Ff>],
        ch: crate::engines::optimized_engine::Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
        sparse: Arc<SparseCache<Ff>>,
        backend_ctx: &'ctx BackendContext,
    ) -> Result<Self, PiCcsError> {
        let inner = OptimizedOracle::new_with_sparse(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_n,
            d_sc,
            r_inputs,
            sparse,
        );
        let split_nc_required = backend_ctx.split_nc_required();
        let fe_evaluator = match backend_ctx.split_nc_session()? {
            Some(session) => match session.create_fe_evaluator(&inner.fe_row_snapshot_bytes()) {
                Ok(evaluator) => Some(evaluator),
                Err(err) if split_nc_required => {
                    return Err(PiCcsError::ProtocolError(format!(
                        "failed to initialize Mojo FE evaluator: {err}"
                    )))
                }
                Err(_) => None,
            },
            None => None,
        };
        Ok(Self {
            inner,
            fe_evaluator,
            split_nc_required,
        })
    }

    pub fn build_me_outputs_from_ajtai_precomp<L>(
        &mut self,
        mcs_list: &[CcsClaim<Cmt, Ff>],
        me_inputs: &[CeClaim<Cmt, Ff, K>],
        s_col: &[K],
        fold_digest: [u8; 32],
        l: &L,
    ) -> Vec<CeClaim<Cmt, Ff, K>>
    where
        L: neo_ccs::traits::SModuleHomomorphism<Ff, Cmt>,
    {
        self.inner
            .build_me_outputs_from_ajtai_precomp(mcs_list, me_inputs, s_col, fold_digest, l)
    }
}

impl<'a, 'ctx, Ff> SplitNcOptimizedOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[inline]
    fn should_use_gpu(&self) -> bool {
        self.fe_evaluator.is_some() && self.inner.round_idx < self.inner.ell_n
    }

    fn drop_fe_evaluator(&mut self, context: &str, err: impl std::fmt::Display) {
        if self.split_nc_required {
            panic!("{context}: {err}");
        }
        self.fe_evaluator = None;
    }
}

impl<'a, 'ctx, Ff> RoundOracle for SplitNcOptimizedOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.should_use_gpu() {
            let flat_points = points.iter().copied().map(flat_k_from_ext).collect::<Vec<_>>();
            let gpu_out = {
                let evaluator = self.fe_evaluator.as_ref().expect("checked by should_use_gpu");
                evaluator.evals_at(&flat_points)
            };
            match gpu_out {
                Ok(out) => return out.into_iter().map(ext_k_from_flat).collect(),
                Err(err) => self.drop_fe_evaluator("Mojo FE evals_at failed", err),
            }
        }
        self.inner.evals_at(points)
    }

    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        self.inner.degree_bound()
    }

    fn fold(&mut self, r: K) {
        if self.should_use_gpu() {
            let challenge = flat_k_from_ext(r);
            let gpu_result = {
                let evaluator = self.fe_evaluator.as_mut().expect("checked by should_use_gpu");
                evaluator.fold(challenge)
            };
            if let Err(err) = gpu_result {
                self.drop_fe_evaluator("Mojo FE fold failed", err);
            }
        }
        self.inner.fold(r);
        if self.inner.round_idx >= self.inner.ell_n {
            self.fe_evaluator = None;
        }
    }
}

pub struct SplitNcNcOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    inner: NcOracle<'a, Ff>,
    nc_evaluator: Option<MojoSplitNcEvaluator<'ctx>>,
    split_nc_required: bool,
}

impl<'a, 'ctx, Ff> SplitNcNcOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: &'a CcsStructure<Ff>,
        params: &'a NeoParams,
        mcs_witnesses: &'a [CcsWitness<Ff>],
        me_witnesses: &'a [Mat<Ff>],
        ch: crate::engines::optimized_engine::Challenges,
        ell_d: usize,
        ell_m: usize,
        d_sc: usize,
        backend_ctx: &'ctx BackendContext,
    ) -> Result<Self, PiCcsError> {
        let inner = NcOracle::new(s, params, mcs_witnesses, me_witnesses, ch, ell_d, ell_m, d_sc);
        let split_nc_required = backend_ctx.split_nc_required();
        let nc_evaluator = match backend_ctx.split_nc_session()? {
            Some(session) => match session.create_nc_evaluator(&inner.nc_col_snapshot_bytes()) {
                Ok(evaluator) => Some(evaluator),
                Err(err) if split_nc_required => {
                    return Err(PiCcsError::ProtocolError(format!(
                        "failed to initialize Mojo NC evaluator: {err}"
                    )))
                }
                Err(_) => None,
            },
            None => None,
        };
        Ok(Self {
            inner,
            nc_evaluator,
            split_nc_required,
        })
    }
}

impl<'a, 'ctx, Ff> SplitNcNcOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    #[inline]
    fn should_use_gpu(&self) -> bool {
        self.nc_evaluator.is_some() && self.inner.round_idx < self.inner.ell_m
    }

    fn drop_nc_evaluator(&mut self, context: &str, err: impl std::fmt::Display) {
        if self.split_nc_required {
            panic!("{context}: {err}");
        }
        self.nc_evaluator = None;
    }
}

impl<'a, 'ctx, Ff> RoundOracle for SplitNcNcOracle<'a, 'ctx, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.should_use_gpu() {
            let flat_points = points.iter().copied().map(flat_k_from_ext).collect::<Vec<_>>();
            let gpu_out = {
                let evaluator = self.nc_evaluator.as_ref().expect("checked by should_use_gpu");
                evaluator.evals_at(&flat_points)
            };
            match gpu_out {
                Ok(out) => return out.into_iter().map(ext_k_from_flat).collect(),
                Err(err) => self.drop_nc_evaluator("Mojo NC evals_at failed", err),
            }
        }
        self.inner.evals_at(points)
    }

    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        self.inner.degree_bound()
    }

    fn fold(&mut self, r: K) {
        if self.should_use_gpu() {
            let challenge = flat_k_from_ext(r);
            let gpu_result = {
                let evaluator = self.nc_evaluator.as_mut().expect("checked by should_use_gpu");
                evaluator.fold(challenge)
            };
            if let Err(err) = gpu_result {
                self.drop_nc_evaluator("Mojo NC fold failed", err);
            }
        }
        self.inner.fold(r);
        if self.inner.round_idx >= self.inner.ell_m {
            self.nc_evaluator = None;
        }
    }
}
