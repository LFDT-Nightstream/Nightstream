//! Digest-binding proof for Phase 2 obligations (Milestone 3).
//!
//! This module provides a succinct proof that a private obligations witness hashes (via the
//! canonical Poseidon2/Goldilocks digest functions in `neo_fold::bridge_digests`) to the public
//! `stmt.obligations_digest` value.
//!
//! Important: this is only the *digest binding* piece. The obligations-private closure backend
//! still needs an in-proof binding that the committed weights/claims used by the closure sumcheck
//! are derived from the *same* private obligations witness.

#![forbid(unsafe_code)]

use bellpepper_core::boolean::Boolean;
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use bincode::Options;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{Mat, MeInstance};
use neo_fold::shard::ShardObligations;
use neo_math::{KExtensions as _, F as NeoF, K as NeoK};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing as _;
use p3_field::PrimeField64 as _;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use spartan2::traits::circuit::SpartanCircuit as SpartanCircuitTrait;
use spartan2::traits::snark::R1CSSNARKTrait;

use crate::whir_p3_backend::derive_seed_v1;
use crate::{codec, ClosureProofError, ClosureStatementV1};

type CircuitF = spartan2::provider::goldi::F;
type SpartanEngine = spartan2::provider::GoldilocksMerkleMleEngine;
type SpartanSnark = spartan2::spartan::R1CSSNARK<SpartanEngine>;
type SpartanProverKey = spartan2::spartan::SpartanProverKey<SpartanEngine>;
type SpartanVerifierKey = spartan2::spartan::SpartanVerifierKey<SpartanEngine>;

const MAX_DIGEST_BINDING_PROOF_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB

fn snark_bincode_opts() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_DIGEST_BINDING_PROOF_BYTES)
}

fn digest_u64_limbs(d: [u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, chunk) in d.chunks_exact(8).enumerate() {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(chunk);
        out[i] = u64::from_le_bytes(limb);
    }
    out
}

fn digest_u32_chunks(d: [u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, chunk) in d.chunks_exact(4).enumerate() {
        let mut limb = [0u8; 4];
        limb.copy_from_slice(chunk);
        out[i] = u32::from_le_bytes(limb);
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DigestBindingShapeV1 {
    pub base_b: u32,
    pub d: u32,
    pub kappa: u32,
    pub m_in: u32,
    pub r_len: u32,
    pub y_len: u32,
    pub y_row_len: u32,
    pub main_len: u32,
    pub val_len: u32,
}

impl DigestBindingShapeV1 {
    pub fn from_obligations(params: &NeoParams, obligations: &ShardObligations<Cmt, NeoF, NeoK>) -> Result<Self, String> {
        let d = params.d as usize;
        let kappa = params.kappa as usize;

        let mut it = obligations.iter_all();
        let Some(me0) = it.next() else {
            return Ok(Self {
                base_b: params.b,
                d: params.d,
                kappa: params.kappa,
                m_in: 0,
                r_len: 0,
                y_len: 0,
                y_row_len: 0,
                main_len: obligations.main.len() as u32,
                val_len: obligations.val.len() as u32,
            });
        };

        let m_in0 = me0.m_in;
        let r_len0 = me0.r.len();
        let y_len0 = me0.y.len();
        let y_row_len0 = me0.y.first().map(|r| r.len()).unwrap_or(0);
        if me0.y_scalars.len() != y_len0 {
            return Err("y_scalars length mismatch vs y (obligations[0])".into());
        }

        if me0.c.d != d || me0.c.kappa != kappa || me0.c.data.len() != d * kappa {
            return Err("commitment shape mismatch vs params".into());
        }
        if me0.X.rows() != d || me0.X.cols() != m_in0 {
            return Err("X shape mismatch vs m_in/params".into());
        }
        if y_len0 > 0 && y_row_len0 < d {
            return Err("y row too short for y_scalars recomposition".into());
        }

        for me in core::iter::once(me0).chain(it) {
            if me.m_in != m_in0 {
                return Err("m_in mismatch across obligations".into());
            }
            if me.c.d != d || me.c.kappa != kappa || me.c.data.len() != d * kappa {
                return Err("commitment shape mismatch across obligations".into());
            }
            if me.X.rows() != d || me.X.cols() != m_in0 {
                return Err("X shape mismatch across obligations".into());
            }
            if me.r.len() != r_len0 {
                return Err("r length mismatch across obligations".into());
            }
            if me.y.len() != y_len0 {
                return Err("y length mismatch across obligations".into());
            }
            if me.y_scalars.len() != y_len0 {
                return Err("y_scalars length mismatch vs y across obligations".into());
            }
            for row in &me.y {
                if row.len() != y_row_len0 {
                    return Err("y row length mismatch across obligations".into());
                }
            }
        }

        Ok(Self {
            base_b: params.b,
            d: params.d,
            kappa: params.kappa,
            m_in: m_in0 as u32,
            r_len: r_len0 as u32,
            y_len: y_len0 as u32,
            y_row_len: y_row_len0 as u32,
            main_len: obligations.main.len() as u32,
            val_len: obligations.val.len() as u32,
        })
    }

    pub fn dummy_obligations(&self) -> ShardObligations<Cmt, NeoF, NeoK> {
        let d = self.d as usize;
        let kappa = self.kappa as usize;
        let m_in = self.m_in as usize;
        let r_len = self.r_len as usize;
        let y_len = self.y_len as usize;
        let y_row_len = self.y_row_len as usize;

        let make_me = || MeInstance {
            c: Cmt::zeros(d, kappa),
            X: Mat::from_row_major(d, m_in, vec![NeoF::ZERO; d * m_in]),
            r: vec![NeoK::ZERO; r_len],
            y: vec![vec![NeoK::ZERO; y_row_len]; y_len],
            y_scalars: vec![NeoK::ZERO; y_len],
            m_in,
            fold_digest: [0u8; 32],
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
        };

        ShardObligations {
            main: vec![make_me(); self.main_len as usize],
            val: vec![make_me(); self.val_len as usize],
        }
    }

    fn require_matches_params(&self, params: &NeoParams) -> Result<(), String> {
        if self.base_b != params.b {
            return Err("base_b mismatch vs params".into());
        }
        if self.d != params.d {
            return Err("d mismatch vs params".into());
        }
        if self.kappa != params.kappa {
            return Err("kappa mismatch vs params".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DigestBindingProofV1 {
    shape: DigestBindingShapeV1,
    spartan_snark: Vec<u8>,
}

#[derive(Clone, Debug)]
struct DigestBindingCircuit {
    base_b: u32,
    d: usize,
    kappa: usize,
    m_in: usize,
    r_len: usize,
    y_len: usize,
    y_row_len: usize,
    core_t: usize,
    // Deterministic coefficients derived from `(stmt, commitment_root_z_u64)`.
    u_vecs: Vec<Vec<NeoF>>,   // κ vectors in F^d
    lambdas: Vec<NeoF>,       // per-obligation
    gamma_x: NeoF,            // mixer
    betas_x: Vec<NeoF>,       // per (obligation,row,col<m_in)
    gamma_me: NeoF,           // mixer
    delta_k: NeoF,            // K->F mixer
    nu: Vec<NeoF>,            // row weights ν_ρ
    mu_core: Vec<NeoF>,       // matrix weights μ_j (core)
    mu_bus: Vec<NeoF>,        // matrix weights μ_bus[col_id]
    obligations: ShardObligations<Cmt, NeoF, NeoK>,
    // Public IO
    pp_id_u32: [u32; 8],
    obligations_digest_u64: [u64; 4],
    claimed_sum_u64: u64,
}

impl DigestBindingCircuit {
    fn new(
        params: &NeoParams,
        ccs_t: usize,
        commitment_root_z_u64: &[u64],
        claimed_sum_u64: u64,
        shape: DigestBindingShapeV1,
        obligations: ShardObligations<Cmt, NeoF, NeoK>,
        stmt: &ClosureStatementV1,
    ) -> Result<Self, String> {
        shape.require_matches_params(params)?;
        let d = shape.d as usize;
        let kappa = shape.kappa as usize;
        let m_in = shape.m_in as usize;
        let r_len = shape.r_len as usize;
        let y_len = shape.y_len as usize;
        let y_row_len = shape.y_row_len as usize;

        if y_len > 0 && y_row_len < d {
            return Err("y_row_len < d".into());
        }
        if y_len < ccs_t {
            return Err("y_len < ccs.t()".into());
        }
        let bus_cols = y_len - ccs_t;

        let pp_id_u32 = digest_u32_chunks(stmt.pp_id_digest);
        let obligations_digest_u64 = digest_u64_limbs(stmt.obligations_digest);

        let obligation_count = obligations.main.len() + obligations.val.len();
        if obligation_count != shape.main_len as usize + shape.val_len as usize {
            return Err("obligation_count mismatch vs declared shape".into());
        }

        // --- Derive deterministic coefficients for claimed_sum (must match `weights_claims.rs`) ---
        let (u_vecs, lambdas) = {
            let seed = derive_seed_v1(b"ajtai_opening_only/u_and_lambdas", stmt, Some(commitment_root_z_u64));
            let mut rng = ChaCha8Rng::from_seed(seed);

            let mut u_vecs = Vec::with_capacity(kappa);
            for _ in 0..kappa {
                let mut v = Vec::with_capacity(d);
                for _ in 0..d {
                    v.push(NeoF::from_u64(rng.next_u64()));
                }
                u_vecs.push(v);
            }

            let mut lambdas = Vec::with_capacity(obligation_count);
            for _ in 0..obligation_count {
                lambdas.push(NeoF::from_u64(rng.next_u64()));
            }
            (u_vecs, lambdas)
        };

        // X-projection RNG: mixer scalar γ_x and β coefficients for each X entry.
        let (gamma_x, betas_x) = {
            let seed = derive_seed_v1(b"ajtai_opening_plus_x/rng", stmt, Some(commitment_root_z_u64));
            let mut rng = ChaCha8Rng::from_seed(seed);

            let mut gamma_x = NeoF::from_u64(rng.next_u64());
            if gamma_x == NeoF::ZERO {
                gamma_x = NeoF::ONE;
            }

            let mut betas_x = Vec::with_capacity(obligation_count * d * m_in);
            for _ in 0..(obligation_count * d * m_in) {
                betas_x.push(NeoF::from_u64(rng.next_u64()));
            }
            (gamma_x, betas_x)
        };

        // ME-consistency RNG: mixer scalar γ_me, K->F mixer δ_k, ν vector, and μ weights.
        let (gamma_me, delta_k, nu, mu_core, mu_bus) = {
            let seed = derive_seed_v1(b"full_closure/rng", stmt, Some(commitment_root_z_u64));
            let mut rng = ChaCha8Rng::from_seed(seed);

            let mut gamma_me = NeoF::from_u64(rng.next_u64());
            if gamma_me == NeoF::ZERO {
                gamma_me = NeoF::ONE;
            }

            let mut delta_k = NeoF::from_u64(rng.next_u64());
            if delta_k == NeoF::ZERO {
                delta_k = NeoF::ONE;
            }

            let mut nu = Vec::with_capacity(d);
            for _ in 0..d {
                nu.push(NeoF::from_u64(rng.next_u64()));
            }

            let mut mu_core = Vec::with_capacity(ccs_t);
            for _ in 0..ccs_t {
                mu_core.push(NeoF::from_u64(rng.next_u64()));
            }
            let mut mu_bus = Vec::with_capacity(bus_cols);
            for _ in 0..bus_cols {
                mu_bus.push(NeoF::from_u64(rng.next_u64()));
            }

            (gamma_me, delta_k, nu, mu_core, mu_bus)
        };

        Ok(Self {
            base_b: params.b,
            d,
            kappa,
            m_in,
            r_len,
            y_len,
            y_row_len,
            core_t: ccs_t,
            u_vecs,
            lambdas,
            gamma_x,
            betas_x,
            gamma_me,
            delta_k,
            nu,
            mu_core,
            mu_bus,
            obligations,
            pp_id_u32,
            obligations_digest_u64,
            claimed_sum_u64,
        })
    }

    fn synthesize_inner<CS: ConstraintSystem<CircuitF>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
        // Public inputs
        let mut pp_id_u32_vars = Vec::with_capacity(8);
        for (i, &x) in self.pp_id_u32.iter().enumerate() {
            let v = AllocatedNum::alloc_input(cs.namespace(|| format!("pp_id_u32_{i}")), || Ok(CircuitF::from(x as u64)))?;
            pp_id_u32_vars.push(v);
        }

        let mut obligations_digest_vars = Vec::with_capacity(4);
        for (i, &x) in self.obligations_digest_u64.iter().enumerate() {
            let v =
                AllocatedNum::alloc_input(cs.namespace(|| format!("obligations_digest_u64_{i}")), || Ok(CircuitF::from(x)))?;
            obligations_digest_vars.push(v);
        }

        let claimed_sum_var =
            AllocatedNum::alloc_input(cs.namespace(|| "claimed_sum_u64".to_string()), || Ok(CircuitF::from(self.claimed_sum_u64)))?;

        // Compute acc digests.
        let acc_main = acc_digest_v2(
            &mut cs.namespace(|| "acc_main_digest_v2"),
            self.base_b,
            self.d,
            self.kappa,
            self.m_in,
            self.r_len,
            self.y_len,
            self.y_row_len,
            self.obligations.main.as_slice(),
        )?;
        let acc_val = acc_digest_v2(
            &mut cs.namespace(|| "acc_val_digest_v2"),
            self.base_b,
            self.d,
            self.kappa,
            self.m_in,
            self.r_len,
            self.y_len,
            self.y_row_len,
            self.obligations.val.as_slice(),
        )?;

        // Compute obligations_digest_v2, matching `neo_fold::bridge_digests::compute_obligations_digest_v2`.
        let out = obligations_digest_v2(
            &mut cs.namespace(|| "obligations_digest_v2"),
            &acc_main,
            &acc_val,
            pp_id_u32_vars.as_slice(),
        )?;

        // Enforce output equals public statement digest (4 u64 limbs).
        for i in 0..4 {
            cs.enforce(
                || format!("obligations_digest_match_{i}"),
                |lc| lc + out[i].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + obligations_digest_vars[i].get_variable(),
            );
        }

        // Compute and enforce the deterministic full-closure claimed_sum:
        //   claimed_sum = Σ_i λ_i·<u, c_i> + γ_x·Σ β·X + γ_me·Σ λ_i·μ_j·⟨ν, mix(y_j,δ_k)⟩.
        //
        // This is linear in the obligations witness variables; all coefficients are derived
        // deterministically from `(stmt, commitment_root_z)` in host code.
        let obligation_count = self.obligations.main.len() + self.obligations.val.len();
        if obligation_count != self.lambdas.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        if self.kappa != self.u_vecs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for v in self.u_vecs.iter() {
            if v.len() != self.d {
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        // Pre-allocate all obligation witness variables needed for claimed_sum into a flat term list.
        let mut terms: Vec<(CircuitF, bellpepper_core::Variable)> = Vec::new();

        let gamma_x = self.gamma_x;
        let gamma_me = self.gamma_me;
        let delta_k = self.delta_k;

        let mut beta_idx = 0usize;
        for (ob_idx, (me, lambda_i)) in self
            .obligations
            .main
            .iter()
            .chain(self.obligations.val.iter())
            .zip(self.lambdas.iter().copied())
            .enumerate()
        {
            // Commitment opening claim terms: λ_i * <u, c_i>
            if me.c.d != self.d || me.c.kappa != self.kappa || me.c.data.len() != self.d * self.kappa {
                return Err(SynthesisError::Unsatisfiable);
            }
            for col in 0..self.kappa {
                for rho in 0..self.d {
                    let c_val = me.c.data[col * self.d + rho];
                    let c_var = alloc_witness_f(cs, &format!("claim_me_{ob_idx}_c_{col}_{rho}"), c_val)?;
                    let coeff = lambda_i * self.u_vecs[col][rho];
                    if coeff != NeoF::ZERO {
                        terms.push((CircuitF::from(coeff.as_canonical_u64()), c_var.get_variable()));
                    }
                }
            }

            // X-projection claim terms: γ_x * Σ β * X
            if me.X.rows() != self.d || me.X.cols() != self.m_in {
                return Err(SynthesisError::Unsatisfiable);
            }
            for row in 0..self.d {
                for col in 0..self.m_in {
                    let beta = self
                        .betas_x
                        .get(beta_idx)
                        .copied()
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    beta_idx += 1;

                    let x_val = me.X[(row, col)];
                    let x_var = alloc_witness_f(cs, &format!("claim_me_{ob_idx}_X_{row}_{col}"), x_val)?;
                    let coeff = gamma_x * beta;
                    if coeff != NeoF::ZERO {
                        terms.push((CircuitF::from(coeff.as_canonical_u64()), x_var.get_variable()));
                    }
                }
            }

            // ME-consistency claim terms (core + bus y rows).
            if me.y.len() != self.y_len || me.y_scalars.len() != self.y_len {
                return Err(SynthesisError::Unsatisfiable);
            }
            for (j, mu_j) in self.mu_core.iter().copied().enumerate() {
                if mu_j == NeoF::ZERO {
                    continue;
                }
                let yj = me.y.get(j).ok_or(SynthesisError::Unsatisfiable)?;
                if yj.len() < self.d {
                    return Err(SynthesisError::Unsatisfiable);
                }
                for rho in 0..self.d {
                    let y = yj[rho];
                    let [c0, c1] = alloc_witness_k_coeffs(cs, &format!("claim_me_{ob_idx}_y_{j}_{rho}"), y)?;
                    let base = gamma_me * lambda_i * mu_j * self.nu[rho];
                    if base != NeoF::ZERO {
                        terms.push((CircuitF::from(base.as_canonical_u64()), c0.get_variable()));
                        let base1 = base * delta_k;
                        if base1 != NeoF::ZERO {
                            terms.push((CircuitF::from(base1.as_canonical_u64()), c1.get_variable()));
                        }
                    }
                }
            }

            for (bus_j, mu) in self.mu_bus.iter().copied().enumerate() {
                if mu == NeoF::ZERO {
                    continue;
                }
                let j = self
                    .core_t
                    .checked_add(bus_j)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let yj = me.y.get(j).ok_or(SynthesisError::Unsatisfiable)?;
                if yj.len() < self.d {
                    return Err(SynthesisError::Unsatisfiable);
                }
                for rho in 0..self.d {
                    let y = yj[rho];
                    let [c0, c1] = alloc_witness_k_coeffs(cs, &format!("claim_me_{ob_idx}_y_bus_{bus_j}_{rho}"), y)?;
                    let base = gamma_me * lambda_i * mu * self.nu[rho];
                    if base != NeoF::ZERO {
                        terms.push((CircuitF::from(base.as_canonical_u64()), c0.get_variable()));
                        let base1 = base * delta_k;
                        if base1 != NeoF::ZERO {
                            terms.push((CircuitF::from(base1.as_canonical_u64()), c1.get_variable()));
                        }
                    }
                }
            }
        }
        if beta_idx != self.betas_x.len() {
            return Err(SynthesisError::Unsatisfiable);
        }

        cs.enforce(
            || "claimed_sum_match",
            |lc| {
                let mut acc = lc;
                for (coeff, var) in terms.iter().copied() {
                    acc = acc + (coeff, var);
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + claimed_sum_var.get_variable(),
        );

        Ok(())
    }

    fn public_io(&self) -> Vec<CircuitF> {
        let mut out = Vec::with_capacity(13);
        out.extend(self.pp_id_u32.iter().copied().map(|x| CircuitF::from(x as u64)));
        out.extend(self.obligations_digest_u64.iter().copied().map(CircuitF::from));
        out.push(CircuitF::from(self.claimed_sum_u64));
        out
    }
}

impl SpartanCircuitTrait<SpartanEngine> for DigestBindingCircuit {
    fn public_values(&self) -> std::result::Result<Vec<CircuitF>, SynthesisError> {
        Ok(self.public_io())
    }

    fn shared<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
        _precommitted: &[AllocatedNum<CircuitF>],
        _challenges: Option<&[CircuitF]>,
    ) -> std::result::Result<(), SynthesisError> {
        self.synthesize_inner(cs)
    }
}

pub fn prove_obligations_digest_binding_proof_v1(
    stmt: &ClosureStatementV1,
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &ShardObligations<Cmt, NeoF, NeoK>,
    commitment_root_z_u64: &[u64],
    claimed_sum_u64: u64,
) -> Result<Vec<u8>, ClosureProofError> {
    let shape = DigestBindingShapeV1::from_obligations(params, obligations)
        .map_err(|e| ClosureProofError::Spartan2(format!("digest-binding shape extraction failed: {e}")))?;

    let circuit = DigestBindingCircuit::new(
        params,
        ccs.t(),
        commitment_root_z_u64,
        claimed_sum_u64,
        shape.clone(),
        obligations.clone(),
        stmt,
    )
        .map_err(|e| ClosureProofError::Spartan2(format!("digest-binding circuit init failed: {e}")))?;

    let (pk, _vk): (SpartanProverKey, SpartanVerifierKey) = SpartanSnark::setup(circuit.clone())
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 setup failed: {e}")))?;

    let prep = SpartanSnark::prep_prove(&pk, circuit.clone(), true)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 prep_prove failed: {e}")))?;
    let snark = SpartanSnark::prove(&pk, circuit, &prep, true)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 prove failed: {e}")))?;

    let snark_bytes = snark_bincode_opts()
        .serialize(&snark)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 proof serialization failed: {e}")))?;

    let proof = DigestBindingProofV1 { shape, spartan_snark: snark_bytes };
    codec::serialize_payload(&proof)
}

pub fn verify_obligations_digest_binding_proof_v1(
    stmt: &ClosureStatementV1,
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    commitment_root_z_u64: &[u64],
    proof_bytes: &[u8],
) -> Result<(), ClosureProofError> {
    let (_shape, _claimed_sum) =
        verify_obligations_digest_binding_proof_v1_with_shape_and_claimed_sum(stmt, params, ccs, commitment_root_z_u64, proof_bytes)?;
    Ok(())
}

pub(crate) fn verify_obligations_digest_binding_proof_v1_with_shape_and_claimed_sum(
    stmt: &ClosureStatementV1,
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    commitment_root_z_u64: &[u64],
    proof_bytes: &[u8],
) -> Result<(DigestBindingShapeV1, u64), ClosureProofError> {
    let proof: DigestBindingProofV1 = codec::deserialize_payload(proof_bytes)?;
    proof
        .shape
        .require_matches_params(params)
        .map_err(|e| ClosureProofError::Spartan2(format!("digest-binding shape mismatch: {e}")))?;
    if (proof.shape.y_len as usize) < ccs.t() {
        return Err(ClosureProofError::Spartan2("digest-binding shape y_len < ccs.t()".into()));
    }

    if proof.spartan_snark.len() > MAX_DIGEST_BINDING_PROOF_BYTES as usize {
        return Err(ClosureProofError::Spartan2(format!(
            "digest-binding proof too large: {} > {}",
            proof.spartan_snark.len(),
            MAX_DIGEST_BINDING_PROOF_BYTES
        )));
    }

    // Deterministically rebuild the verifier key from the declared shape (dummy witness, all zeros).
    //
    // This is the same pattern used by `neo-spartan-bridge` for pinned VK setup, but here the
    // shape is carried alongside the digest-binding proof bytes.
    let dummy_obligations = proof.shape.dummy_obligations();
    let dummy_circuit = DigestBindingCircuit::new(
        params,
        ccs.t(),
        commitment_root_z_u64,
        0,
        proof.shape.clone(),
        dummy_obligations,
        stmt,
    )
        .map_err(|e| ClosureProofError::Spartan2(format!("digest-binding dummy circuit init failed: {e}")))?;
    let (_pk, vk): (SpartanProverKey, SpartanVerifierKey) = SpartanSnark::setup(dummy_circuit)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 setup (vk) failed: {e}")))?;

    let snark: SpartanSnark = snark_bincode_opts()
        .deserialize(&proof.spartan_snark)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 proof deserialization failed: {e}")))?;

    let expected = {
        let mut out = Vec::with_capacity(12);
        out.extend(digest_u32_chunks(stmt.pp_id_digest).iter().copied().map(|x| CircuitF::from(x as u64)));
        out.extend(digest_u64_limbs(stmt.obligations_digest).iter().copied().map(CircuitF::from));
        out
    };

    let io = snark
        .verify(&vk)
        .map_err(|e| ClosureProofError::Spartan2(format!("Spartan2 verification failed: {e}")))?;
    if io.len() != expected.len() + 1 {
        return Err(ClosureProofError::Spartan2("Spartan2 public IO length mismatch".into()));
    }
    if io[..expected.len()] != expected {
        return Err(ClosureProofError::Spartan2("Spartan2 public IO mismatch".into()));
    }

    let claimed_sum_u64 = io[expected.len()].to_canonical_u64();
    Ok((proof.shape, claimed_sum_u64))
}

/// Decode the public shape carried by a v1 digest-binding proof container.
///
/// This does *not* verify the Spartan2 proof; callers should additionally invoke
/// [`verify_obligations_digest_binding_proof_v1`] before relying on the result.
pub fn decode_obligations_digest_binding_shape_v1(proof_bytes: &[u8]) -> Result<DigestBindingShapeV1, ClosureProofError> {
    let proof: DigestBindingProofV1 = codec::deserialize_payload(proof_bytes)?;
    if proof.spartan_snark.len() > MAX_DIGEST_BINDING_PROOF_BYTES as usize {
        return Err(ClosureProofError::Spartan2(format!(
            "digest-binding proof too large: {} > {}",
            proof.spartan_snark.len(),
            MAX_DIGEST_BINDING_PROOF_BYTES
        )));
    }
    Ok(proof.shape)
}

// -----------------------------------------------------------------------------
// R1CS Poseidon2 sponge (WIDTH=8, RATE=4) gadget.
// -----------------------------------------------------------------------------

mod poseidon2 {
    use super::*;
    use once_cell::sync::Lazy;
    use p3_field::PrimeField64;
    use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
    use p3_poseidon2::poseidon2_round_numbers_128;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    pub(super) const WIDTH: usize = 8;
    pub(super) const RATE: usize = 4;
    pub(super) const SBOX_DEGREE: u64 = 7;

    #[derive(Clone, Debug)]
    struct Poseidon2ConstantsW8 {
        initial: [[CircuitF; WIDTH]; 4],
        terminal: [[CircuitF; WIDTH]; 4],
        internal: [CircuitF; 22],
    }

    fn to_circuit(x: Goldilocks) -> CircuitF {
        CircuitF::from(x.as_canonical_u64())
    }

    static CONSTANTS_W8: Lazy<Poseidon2ConstantsW8> = Lazy::new(|| {
        let (rounds_f, rounds_p) =
            poseidon2_round_numbers_128::<Goldilocks>(WIDTH, SBOX_DEGREE).expect("round numbers");
        assert_eq!(rounds_f, 8, "expected WIDTH=8, D=7 full rounds = 8");
        assert_eq!(rounds_p, 22, "expected WIDTH=8, D=7 partial rounds = 22");

        let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);

        // Draw `half_f` WIDTH-wide vectors for initial, then `half_f` vectors for terminal.
        let mut draw_vec = || -> [Goldilocks; WIDTH] { rng.random() };
        let half_f = rounds_f / 2;
        let mut initial = [[CircuitF::from(0u64); WIDTH]; 4];
        let mut terminal = [[CircuitF::from(0u64); WIDTH]; 4];
        for r in 0..half_f {
            let v = draw_vec();
            initial[r] = v.map(to_circuit);
        }
        for r in 0..half_f {
            let v = draw_vec();
            terminal[r] = v.map(to_circuit);
        }

        // Internal constants: `rounds_p` field elements.
        let mut internal = [CircuitF::from(0u64); 22];
        for r in 0..rounds_p {
            let x: Goldilocks = rng.random();
            internal[r] = to_circuit(x);
        }

        Poseidon2ConstantsW8 {
            initial,
            terminal,
            internal,
        }
    });

    fn pow7_val(x: CircuitF) -> CircuitF {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        x6 * x
    }

    fn pow7<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        x: &AllocatedNum<CircuitF>,
        label: &str,
    ) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
        let mul = |cs: &mut CS, a: &AllocatedNum<CircuitF>, b: &AllocatedNum<CircuitF>, lbl: &str| {
            let a_val = a.get_value().unwrap_or(CircuitF::from(0u64));
            let b_val = b.get_value().unwrap_or(CircuitF::from(0u64));
            let out_val = a_val * b_val;
            let out = AllocatedNum::alloc(cs.namespace(|| lbl.to_string()), || Ok(out_val))?;
            cs.enforce(
                || format!("{lbl}_constraint"),
                |lc| lc + a.get_variable(),
                |lc| lc + b.get_variable(),
                |lc| lc + out.get_variable(),
            );
            Ok(out)
        };

        let x2 = mul(cs, x, x, &format!("{label}_x2"))?;
        let x4 = mul(cs, &x2, &x2, &format!("{label}_x4"))?;
        let x6 = mul(cs, &x4, &x2, &format!("{label}_x6"))?;
        mul(cs, &x6, x, &format!("{label}_x7"))
    }

    pub(super) fn alloc_linear_comb<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        label: &str,
        value: CircuitF,
        terms: &[(CircuitF, &AllocatedNum<CircuitF>)],
    ) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
        let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_out")), || Ok(value))?;
        cs.enforce(
            || format!("{label}_lc"),
            |lc| {
                let mut acc = lc;
                for (coeff, var) in terms {
                    acc = acc + (*coeff, var.get_variable());
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + out.get_variable(),
        );
        Ok(out)
    }

    fn external_linear_layer_w8<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        state: &mut [AllocatedNum<CircuitF>; WIDTH],
        state_val: &mut [CircuitF; WIDTH],
        label: &str,
    ) -> Result<(), SynthesisError> {
        // MDSMat4 for WIDTH=4 is the 4x4 matrix:
        // [2 3 1 1]
        // [1 2 3 1]
        // [1 1 2 3]
        // [3 1 1 2]
        const A: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];

        // WIDTH=8 external linear layer is the block matrix [[2A, A], [A, 2A]].
        let old_vars = state.clone();
        let old_vals = *state_val;

        let mut new_vals = [CircuitF::from(0u64); WIDTH];

        for row in 0..WIDTH {
            let (block_row, top) = if row < 4 { (row, true) } else { (row - 4, false) };
            let mut terms: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(WIDTH);
            let mut acc = CircuitF::from(0u64);
            for col in 0..WIDTH {
                let (block_col, left) = if col < 4 { (col, true) } else { (col - 4, false) };
                let base = A[block_row][block_col];
                let scale = match (top, left) {
                    (true, true) => 2,
                    (true, false) => 1,
                    (false, true) => 1,
                    (false, false) => 2,
                };
                let coeff_u64 = base * scale;
                let coeff = CircuitF::from(coeff_u64);
                terms.push((coeff, &old_vars[col]));
                acc += coeff * old_vals[col];
            }
            new_vals[row] = acc;
            let out = alloc_linear_comb(cs, &format!("{label}_row_{row}"), acc, &terms)?;
            state[row] = out;
        }

        *state_val = new_vals;
        Ok(())
    }

    fn internal_linear_layer_w8<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        state: &mut [AllocatedNum<CircuitF>; WIDTH],
        state_val: &mut [CircuitF; WIDTH],
        label: &str,
    ) -> Result<(), SynthesisError> {
        let diag: [CircuitF; WIDTH] = MATRIX_DIAG_8_GOLDILOCKS.map(to_circuit);

        let old_vars = state.clone();
        let old_vals = *state_val;

        let sum_val: CircuitF = old_vals
            .iter()
            .copied()
            .fold(CircuitF::from(0u64), |a, b| a + b);
        let mut new_vals = [CircuitF::from(0u64); WIDTH];

        for i in 0..WIDTH {
            let out_val = sum_val + diag[i] * old_vals[i];
            new_vals[i] = out_val;

            // out = sum + diag[i] * state[i]  (sum is a constant-LC over all old vars)
            let mut terms: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(WIDTH);
            for j in 0..WIDTH {
                let coeff = if i == j {
                    CircuitF::from(1u64) + diag[i]
                } else {
                    CircuitF::from(1u64)
                };
                terms.push((coeff, &old_vars[j]));
            }
            let out = alloc_linear_comb(cs, &format!("{label}_row_{i}"), out_val, &terms)?;
            state[i] = out;
        }

        *state_val = new_vals;
        Ok(())
    }

    fn full_round_w8<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        state: &mut [AllocatedNum<CircuitF>; WIDTH],
        state_val: &mut [CircuitF; WIDTH],
        round_constants: &[CircuitF; WIDTH],
        label: &str,
    ) -> Result<(), SynthesisError> {
        for i in 0..WIDTH {
            let add_val = state_val[i] + round_constants[i];
            let added = AllocatedNum::alloc(cs.namespace(|| format!("{label}_add_{i}")), || Ok(add_val))?;
            cs.enforce(
                || format!("{label}_add_{i}_enforce"),
                |lc| lc + state[i].get_variable() + (round_constants[i], CS::one()),
                |lc| lc + CS::one(),
                |lc| lc + added.get_variable(),
            );
            let sboxed = pow7(cs, &added, &format!("{label}_sbox_{i}"))?;
            state[i] = sboxed;
            state_val[i] = pow7_val(add_val);
        }
        external_linear_layer_w8(cs, state, state_val, &format!("{label}_mds"))
    }

    fn partial_round_w8<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        state: &mut [AllocatedNum<CircuitF>; WIDTH],
        state_val: &mut [CircuitF; WIDTH],
        rc: CircuitF,
        label: &str,
    ) -> Result<(), SynthesisError> {
        let add_val = state_val[0] + rc;
        let added = AllocatedNum::alloc(cs.namespace(|| format!("{label}_add_0")), || Ok(add_val))?;
        cs.enforce(
            || format!("{label}_add_0_enforce"),
            |lc| lc + state[0].get_variable() + (rc, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + added.get_variable(),
        );
        let sboxed = pow7(cs, &added, &format!("{label}_sbox_0"))?;
        state[0] = sboxed;
        state_val[0] = pow7_val(add_val);
        internal_linear_layer_w8(cs, state, state_val, &format!("{label}_int"))
    }

    pub(super) fn permute_w8<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        state: &mut [AllocatedNum<CircuitF>; WIDTH],
    ) -> Result<(), SynthesisError> {
        // Maintain a parallel native state for witness generation.
        let mut state_val = [CircuitF::from(0u64); WIDTH];
        for i in 0..WIDTH {
            state_val[i] = state[i].get_value().unwrap_or(CircuitF::from(0u64));
        }

        // Initial external layer includes an extra MDS.
        external_linear_layer_w8(cs, state, &mut state_val, "poseidon2_init_mds")?;

        // Initial full rounds
        for r in 0..4 {
            full_round_w8(cs, state, &mut state_val, &CONSTANTS_W8.initial[r], &format!("fr_init_{r}"))?;
        }
        // Partial rounds
        for r in 0..22 {
            partial_round_w8(
                cs,
                state,
                &mut state_val,
                CONSTANTS_W8.internal[r],
                &format!("pr_{r}"),
            )?;
        }
        // Terminal full rounds
        for r in 0..4 {
            full_round_w8(cs, state, &mut state_val, &CONSTANTS_W8.terminal[r], &format!("fr_term_{r}"))?;
        }

        Ok(())
    }
}

#[derive(Clone)]
struct Poseidon2Sponge {
    state: [AllocatedNum<CircuitF>; poseidon2::WIDTH],
    absorbed: usize,
    permute_count: usize,
    one: AllocatedNum<CircuitF>,
    scope: String,
}

impl Poseidon2Sponge {
    fn new<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<Self, SynthesisError> {
        let mut state: Vec<AllocatedNum<CircuitF>> = Vec::with_capacity(poseidon2::WIDTH);
        for i in 0..poseidon2::WIDTH {
            let z = AllocatedNum::alloc(cs.namespace(|| format!("{label}_st_{i}")), || Ok(CircuitF::from(0u64)))?;
            cs.enforce(
                || format!("{label}_st_{i}_is_zero"),
                |lc| lc + z.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
            state.push(z);
        }
        let state: [AllocatedNum<CircuitF>; poseidon2::WIDTH] = state
            .try_into()
            .map_err(|_| SynthesisError::Unsatisfiable)?;

        let one = AllocatedNum::alloc(cs.namespace(|| format!("{label}_one")), || Ok(CircuitF::from(1u64)))?;
        cs.enforce(
            || format!("{label}_one_is_one"),
            |lc| lc + one.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (CircuitF::from(1u64), CS::one()),
        );
        Ok(Self {
            state,
            absorbed: 0,
            permute_count: 0,
            one,
            scope: label.to_owned(),
        })
    }

    fn absorb<CS: ConstraintSystem<CircuitF>>(&mut self, cs: &mut CS, x: AllocatedNum<CircuitF>) -> Result<(), SynthesisError> {
        // `absorb_elem` semantics: permute *before* writing if buffer is full.
        if self.absorbed >= poseidon2::RATE {
            self.permute(cs, &format!("{}_permute_{}", self.scope, self.permute_count))?;
        }
        self.state[self.absorbed] = x;
        self.absorbed += 1;
        Ok(())
    }

    fn permute<CS: ConstraintSystem<CircuitF>>(&mut self, cs: &mut CS, label: &str) -> Result<(), SynthesisError> {
        let mut cs_ns = cs.namespace(|| label.to_string());
        self.permute_count += 1;
        poseidon2::permute_w8(&mut cs_ns, &mut self.state)?;
        self.absorbed = 0;
        Ok(())
    }

    fn digest32<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &str,
    ) -> Result<[AllocatedNum<CircuitF>; 4], SynthesisError> {
        self.absorb(cs, self.one.clone())?;
        self.permute(cs, &format!("{}_{}_permute_{}", self.scope, label, self.permute_count))?;
        Ok([
            self.state[0].clone(),
            self.state[1].clone(),
            self.state[2].clone(),
            self.state[3].clone(),
        ])
    }
}

fn alloc_const<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    x: CircuitF,
) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
    let v = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(x))?;
    cs.enforce(
        || format!("{label}_is_const"),
        |lc| lc + v.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (x, CS::one()),
    );
    Ok(v)
}

fn alloc_witness_f<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    x: NeoF,
) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
    let x_f = CircuitF::from(x.as_canonical_u64());
    AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(x_f))
}

fn alloc_witness_k_coeffs<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    x: NeoK,
) -> Result<[AllocatedNum<CircuitF>; 2], SynthesisError> {
    let [c0, c1] = x.as_coeffs();
    let c0v = alloc_witness_f(cs, &format!("{label}_c0"), c0)?;
    let c1v = alloc_witness_f(cs, &format!("{label}_c1"), c1)?;
    Ok([c0v, c1v])
}

fn u32_from_bits<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    bits: &[Boolean],
) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
    if bits.len() != 32 {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut val_u64 = 0u64;
    for (i, b) in bits.iter().enumerate() {
        let bit = match b.get_value() {
            Some(v) => v,
            None => false,
        };
        if bit {
            val_u64 |= 1u64 << i;
        }
    }

    let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_u32")), || Ok(CircuitF::from(val_u64)))?;
    cs.enforce(
        || format!("{label}_u32_pack"),
        |lc| {
            let mut acc = lc;
            for (i, b) in bits.iter().enumerate() {
                let coeff = CircuitF::from(1u64 << i);
                match b {
                    Boolean::Constant(true) => acc = acc + (coeff, CS::one()),
                    Boolean::Constant(false) => {}
                    Boolean::Is(var) => acc = acc + (coeff, var.get_variable()),
                    Boolean::Not(var) => {
                        // (1 - var)
                        acc = acc + (coeff, CS::one()) + (-coeff, var.get_variable());
                    }
                }
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
    );
    Ok(out)
}

fn split_field_u64_to_u32_chunks_strict<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    label: &str,
    x: &AllocatedNum<CircuitF>,
) -> Result<[AllocatedNum<CircuitF>; 2], SynthesisError> {
    let bits = x.to_bits_le_strict(cs.namespace(|| format!("{label}_bits")))?;
    if bits.len() < 64 {
        return Err(SynthesisError::Unsatisfiable);
    }
    let lo = u32_from_bits(cs, &format!("{label}_lo"), &bits[0..32])?;
    let hi = u32_from_bits(cs, &format!("{label}_hi"), &bits[32..64])?;
    Ok([lo, hi])
}

fn acc_digest_v2<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    base_b: u32,
    d: usize,
    kappa: usize,
    m_in: usize,
    r_len: usize,
    y_len: usize,
    y_row_len: usize,
    acc: &[MeInstance<Cmt, NeoF, NeoK>],
) -> Result<[AllocatedNum<CircuitF>; 4], SynthesisError> {
    let mut sponge = Poseidon2Sponge::new(cs, "acc_digest")?;

    for (i, &b) in b"neo/spartan-bridge/acc_digest/v2".iter().enumerate() {
        let v = alloc_const(cs, &format!("acc_dom_{i}"), CircuitF::from(b as u64))?;
        sponge.absorb(cs, v)?;
    }

    let acc_len = alloc_const(cs, "acc_len", CircuitF::from(acc.len() as u64))?;
    sponge.absorb(cs, acc_len)?;

    let b_f = NeoF::from_u64(base_b as u64);
    let mut b_pow_coeffs = Vec::with_capacity(d);
    {
        let mut pw = NeoF::ONE;
        for _ in 0..d {
            b_pow_coeffs.push(CircuitF::from(pw.as_canonical_u64()));
            pw *= b_f;
        }
    }

    for (me_idx, me) in acc.iter().enumerate() {
        // Shape sanity (host-side guard for circuit determinism).
        if me.c.d != d || me.c.kappa != kappa || me.c.data.len() != d * kappa {
            return Err(SynthesisError::Unsatisfiable);
        }
        if me.X.rows() != d || me.X.cols() != m_in {
            return Err(SynthesisError::Unsatisfiable);
        }
        if me.r.len() != r_len {
            return Err(SynthesisError::Unsatisfiable);
        }
        if me.y.len() != y_len {
            return Err(SynthesisError::Unsatisfiable);
        }
        if me.y_scalars.len() != y_len {
            return Err(SynthesisError::Unsatisfiable);
        }
        for row in &me.y {
            if row.len() != y_row_len {
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        let me_m_in = alloc_const(cs, &format!("me_{me_idx}_m_in"), CircuitF::from(me.m_in as u64))?;
        sponge.absorb(cs, me_m_in)?;

        let c_len = alloc_const(cs, &format!("me_{me_idx}_c_len"), CircuitF::from(me.c.data.len() as u64))?;
        sponge.absorb(cs, c_len)?;
        for (i, &c) in me.c.data.iter().enumerate() {
            let c_var = alloc_witness_f(cs, &format!("me_{me_idx}_c_{i}"), c)?;
            sponge.absorb(cs, c_var)?;
        }

        let x_rows = alloc_const(cs, &format!("me_{me_idx}_X_rows"), CircuitF::from(d as u64))?;
        sponge.absorb(cs, x_rows)?;
        let x_cols = alloc_const(cs, &format!("me_{me_idx}_X_cols"), CircuitF::from(m_in as u64))?;
        sponge.absorb(cs, x_cols)?;
        for (i, &x) in me.X.as_slice().iter().enumerate() {
            let x_var = alloc_witness_f(cs, &format!("me_{me_idx}_X_{i}"), x)?;
            sponge.absorb(cs, x_var)?;
        }

        let r_len_var = alloc_const(cs, &format!("me_{me_idx}_r_len"), CircuitF::from(r_len as u64))?;
        sponge.absorb(cs, r_len_var)?;
        for (i, &r_k) in me.r.iter().enumerate() {
            let [c0, c1] = alloc_witness_k_coeffs(cs, &format!("me_{me_idx}_r_{i}"), r_k)?;
            sponge.absorb(cs, c0)?;
            sponge.absorb(cs, c1)?;
        }

        let y_len_var = alloc_const(cs, &format!("me_{me_idx}_y_len"), CircuitF::from(y_len as u64))?;
        sponge.absorb(cs, y_len_var)?;
        let mut y_coeffs: Vec<Vec<[AllocatedNum<CircuitF>; 2]>> = Vec::with_capacity(y_len);
        for (j, yj) in me.y.iter().enumerate() {
            let row_len_var = alloc_const(
                cs,
                &format!("me_{me_idx}_y_{j}_len"),
                CircuitF::from(y_row_len as u64),
            )?;
            sponge.absorb(cs, row_len_var)?;
            let mut row = Vec::with_capacity(y_row_len);
            for (rho, &y_elem) in yj.iter().enumerate() {
                let [c0, c1] = alloc_witness_k_coeffs(cs, &format!("me_{me_idx}_y_{j}_{rho}"), y_elem)?;
                // Enforce canonical padding: entries beyond `d` MUST be zero.
                if rho >= d {
                    cs.enforce(
                        || format!("me_{me_idx}_y_{j}_{rho}_pad_c0_is_zero"),
                        |lc| lc + c0.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc,
                    );
                    cs.enforce(
                        || format!("me_{me_idx}_y_{j}_{rho}_pad_c1_is_zero"),
                        |lc| lc + c1.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc,
                    );
                }
                sponge.absorb(cs, c0.clone())?;
                sponge.absorb(cs, c1.clone())?;
                row.push([c0, c1]);
            }
            y_coeffs.push(row);
        }

        // Derived y_scalars: base-b recomposition of the first D digits of y[j].
        for (j, yj) in me.y.iter().enumerate() {
            let mut acc0 = NeoF::ZERO;
            let mut acc1 = NeoF::ZERO;
            let mut pw = NeoF::ONE;
            for rho in 0..d {
                let [c0, c1] = yj[rho].as_coeffs();
                acc0 += pw * c0;
                acc1 += pw * c1;
                pw *= b_f;
            }
            let acc0v = CircuitF::from(acc0.as_canonical_u64());
            let acc1v = CircuitF::from(acc1.as_canonical_u64());

            let mut terms0: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(d);
            let mut terms1: Vec<(CircuitF, &AllocatedNum<CircuitF>)> = Vec::with_capacity(d);
            for rho in 0..d {
                let coeff = b_pow_coeffs[rho];
                terms0.push((coeff, &y_coeffs[j][rho][0]));
                terms1.push((coeff, &y_coeffs[j][rho][1]));
            }

            let c0 = poseidon2::alloc_linear_comb(cs, &format!("me_{me_idx}_y_sc_{j}_c0"), acc0v, &terms0)?;
            let c1 = poseidon2::alloc_linear_comb(cs, &format!("me_{me_idx}_y_sc_{j}_c1"), acc1v, &terms1)?;

            // Enforce that the provided y_scalars matches the canonical recomposition.
            let y_scalar = me
                .y_scalars
                .get(j)
                .copied()
                .ok_or(SynthesisError::Unsatisfiable)?;
            let [ys0, ys1] = alloc_witness_k_coeffs(cs, &format!("me_{me_idx}_y_scalar_{j}"), y_scalar)?;
            cs.enforce(
                || format!("me_{me_idx}_y_scalar_{j}_c0_match"),
                |lc| lc + c0.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + ys0.get_variable(),
            );
            cs.enforce(
                || format!("me_{me_idx}_y_scalar_{j}_c1_match"),
                |lc| lc + c1.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + ys1.get_variable(),
            );

            sponge.absorb(cs, c0)?;
            sponge.absorb(cs, c1)?;
        }
    }

    sponge.digest32(cs, "acc_digest32")
}

fn obligations_digest_v2<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    acc_main_digest_u64: &[AllocatedNum<CircuitF>; 4],
    acc_val_digest_u64: &[AllocatedNum<CircuitF>; 4],
    pp_id_u32: &[AllocatedNum<CircuitF>],
) -> Result<[AllocatedNum<CircuitF>; 4], SynthesisError> {
    if pp_id_u32.len() != 8 {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut sponge = Poseidon2Sponge::new(cs, "obligations_digest")?;

    for (i, &b) in b"neo/spartan-bridge/obligations_digest/v2".iter().enumerate() {
        let v = alloc_const(cs, &format!("obl_dom_{i}"), CircuitF::from(b as u64))?;
        sponge.absorb(cs, v)?;
    }

    for (i, limb) in acc_main_digest_u64.iter().enumerate() {
        let [lo, hi] = split_field_u64_to_u32_chunks_strict(cs, &format!("acc_main_u32_{i}"), limb)?;
        sponge.absorb(cs, lo)?;
        sponge.absorb(cs, hi)?;
    }
    for (i, limb) in acc_val_digest_u64.iter().enumerate() {
        let [lo, hi] = split_field_u64_to_u32_chunks_strict(cs, &format!("acc_val_u32_{i}"), limb)?;
        sponge.absorb(cs, lo)?;
        sponge.absorb(cs, hi)?;
    }
    for v in pp_id_u32.iter() {
        sponge.absorb(cs, v.clone())?;
    }

    sponge.digest32(cs, "obligations_digest32")
}
