//! Public API for proving and verifying FoldRuns with Spartan2
//!
//! This module provides the high-level interface for:
//! - Converting a FoldRun into a Spartan proof
//! - Verifying a Spartan proof for a FoldRun
//!
//! **STATUS**: WIP / non-production. Phase 1 (sound folding lane compression) is implemented.
//! Route‑A memory terminal checks and optional output binding are implemented, but program/VM binding
//! (`program_digest`) and step-linking profiles are still evolving.

use crate::circuit::fold_circuit::CircuitPolyTerm;
use crate::circuit::{FoldRunCircuit, FoldRunInstance, FoldRunWitness};
use crate::error::{Result, SpartanBridgeError};
use crate::statement::{SpartanShardStatement, STATEMENT_VERSION};
use crate::CircuitF;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, MeInstance};
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::shard::{BatchedTimeProof, FoldStep, MemSidecarProof, ShardProof as FoldRun, StepProof};
use neo_math::{F as NeoF, K as NeoK};
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
#[cfg(feature = "debug-logs")]
use neo_reductions::common::format_ext;
#[cfg(feature = "debug-logs")]
use neo_reductions::paper_exact_engine::claimed_initial_sum_from_inputs;
use p3_field::PrimeCharacteristicRing;

use spartan2::{provider::GoldilocksP3MerkleMleEngine, spartan::R1CSSNARK, traits::snark::R1CSSNARKTrait};
use spartan2::bellpepper::shape_cs::ShapeCS;

pub type SpartanEngine = GoldilocksP3MerkleMleEngine;
pub type SpartanSnark = R1CSSNARK<SpartanEngine>;
pub type SpartanProverKey = spartan2::spartan::SpartanProverKey<SpartanEngine>;
pub type SpartanVerifierKey = spartan2::spartan::SpartanVerifierKey<SpartanEngine>;

pub fn compute_accumulator_digest_v2(base_b: u32, acc: &[MeInstance<Cmt, NeoF, NeoK>]) -> [u8; 32] {
    use crate::gadgets::poseidon2::{native_permute_w8, RATE, WIDTH};
    use neo_math::KExtensions;
    use p3_field::PrimeField64;
    use p3_goldilocks::Goldilocks;

    let mut st = [Goldilocks::ZERO; WIDTH];
    let mut absorbed = 0usize;

    let mut absorb = |x: Goldilocks| {
        if absorbed >= RATE {
            st = native_permute_w8(st);
            absorbed = 0;
        }
        st[absorbed] = x;
        absorbed += 1;
    };

    for &b in b"neo/spartan-bridge/acc_digest/v2" {
        absorb(Goldilocks::from_u64(b as u64));
    }

    absorb(Goldilocks::from_u64(acc.len() as u64));

    for me in acc {
        absorb(Goldilocks::from_u64(me.m_in as u64));

        // Commitment data (binds the Ajtai commitment relation used by native verifier).
        absorb(Goldilocks::from_u64(me.c.data.len() as u64));
        for &c in &me.c.data {
            absorb(Goldilocks::from_u64(c.as_canonical_u64()));
        }

        // X matrix (binds the public linear projection).
        absorb(Goldilocks::from_u64(me.X.rows() as u64));
        absorb(Goldilocks::from_u64(me.X.cols() as u64));
        for &x in me.X.as_slice() {
            absorb(Goldilocks::from_u64(x.as_canonical_u64()));
        }

        absorb(Goldilocks::from_u64(me.r.len() as u64));
        for limb in &me.r {
            let coeffs = limb.as_coeffs();
            absorb(coeffs[0]);
            absorb(coeffs[1]);
        }

        absorb(Goldilocks::from_u64(me.y.len() as u64));
        for yj in &me.y {
            absorb(Goldilocks::from_u64(yj.len() as u64));
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

    absorb(Goldilocks::ONE);
    st = native_permute_w8(st);

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    out
}

/// Circuit shape parameters needed to pin a Spartan2 VK/PK.
///
/// Phase 1 currently fixes the circuit shape to:
/// - the number of steps, and
/// - the MCS public input length `m_in` (which drives X dimensions).
#[derive(Clone, Debug)]
pub struct FoldRunShape {
    pub step_count: u32,
    /// Initial accumulator length (for step 0).
    ///
    /// In shard folding, this is often `0` (seeded run) or `params.k_rho` (continuation).
    pub initial_accumulator_len: usize,
    /// Per-step public instances (MCS + optional Twist/Shout instances).
    pub steps_public: Vec<StepInstanceBundle<Cmt, NeoF, NeoK>>,
    /// Optional output binding config (determines output sumcheck shape on the last step).
    pub output_binding: Option<OutputBindingConfig>,
    /// Optional step-linking constraints pinned into the circuit.
    pub step_linking: Vec<(usize, usize)>,
}

impl FoldRunShape {
    pub fn from_witness(witness: &FoldRunWitness) -> Result<Self> {
        let step_count = u32::try_from(witness.fold_run.steps.len()).map_err(|_| {
            SpartanBridgeError::InvalidInput(format!(
                "FoldRun has too many steps for u32: {}",
                witness.fold_run.steps.len()
            ))
        })?;
        if witness.steps_public.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(
                "steps_public must be non-empty".into(),
            ));
        }
        Ok(Self {
            step_count,
            initial_accumulator_len: witness.initial_accumulator.len(),
            steps_public: witness.steps_public.clone(),
            output_binding: witness.output_binding.clone(),
            step_linking: witness.step_linking.clone(),
        })
    }
}

fn pad_sumcheck_rounds_in_place(d_sc: usize, proof: &mut neo_reductions::PiCcsProof) {
    let target = d_sc + 1;
    for round in proof.sumcheck_rounds.iter_mut() {
        if round.len() < target {
            round.resize(target, NeoK::ZERO);
        }
    }
}

fn dummy_me_instance(m_in: usize, y_len: usize, r_len: usize, kappa: usize) -> MeInstance<Cmt, NeoF, NeoK> {
    use neo_ccs::Mat;
    use neo_math::{D, K};

    let c = Cmt::zeros(D, kappa);
    let X = Mat::zero(D, m_in, NeoF::ZERO);
    let r = vec![K::ZERO; r_len];
    // NOTE: Neo pads the ME `y_j` vectors to the next power-of-two of `D` for MLE-friendly layouts.
    // The circuit shape must match real proofs, so the dummy witness uses the same padding.
    let y_pad = D.next_power_of_two();
    let y = vec![vec![K::ZERO; y_pad]; y_len];
    let y_scalars = vec![K::ZERO; y_len];

    MeInstance::<Cmt, NeoF, NeoK> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
    }
}

fn dummy_witness_for_shape(params: &NeoParams, ccs: &CcsStructure<NeoF>, shape: &FoldRunShape) -> Result<FoldRunWitness> {
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, ccs).map_err(SpartanBridgeError::NeoError)?;
    let d_sc = dims.d_sc;
    let r_len = dims.ell_n;
    let core_t = ccs.t();

    let k_dec = params.k_rho as usize;
    let kappa = params.kappa as usize;

    if shape.steps_public.len() != shape.step_count as usize {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "dummy_witness_for_shape: steps_public length mismatch (steps_public={}, step_count={})",
            shape.steps_public.len(),
            shape.step_count
        )));
    }

    let m_in = shape
        .steps_public
        .first()
        .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: steps_public empty".into()))?
        .mcs_inst
        .m_in;

    // Shared CPU bus appends implicit openings as extra ME rows; match the real `y.len()` shape.
    let bus_cols = {
        let has_instances = shape
            .steps_public
            .iter()
            .any(|s| !s.lut_insts.is_empty() || !s.mem_insts.is_empty());
        if !has_instances {
            0usize
        } else {
            let step0 = shape
                .steps_public
                .first()
                .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: steps_public empty".into()))?;
            let shout_ell_addrs: Vec<usize> = step0.lut_insts.iter().map(|inst| inst.d * inst.ell).collect();
            let twist_ell_addrs: Vec<usize> = step0.mem_insts.iter().map(|inst| inst.d * inst.ell).collect();

            let bus = neo_memory::cpu::build_bus_layout_for_instances(
                ccs.m,
                m_in,
                /*chunk_size=*/ 1,
                shout_ell_addrs.into_iter(),
                twist_ell_addrs.into_iter(),
            )
            .map_err(|e| SpartanBridgeError::InvalidInput(format!("dummy_witness_for_shape: bus layout failed: {e}")))?;
            bus.bus_cols
        }
    };
    let y_len_total = core_t
        .checked_add(bus_cols)
        .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: core_t + bus_cols overflow".into()))?;

    let mut initial_accumulator = Vec::with_capacity(shape.initial_accumulator_len);
    for _ in 0..shape.initial_accumulator_len {
        initial_accumulator.push(dummy_me_instance(m_in, y_len_total, r_len, kappa));
    }

    let mut steps = Vec::with_capacity(shape.step_count as usize);
    for step_idx in 0..(shape.step_count as usize) {
        let step_public = shape
            .steps_public
            .get(step_idx)
            .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: missing step_public".into()))?;

        let inputs_len = if step_idx == 0 {
            shape.initial_accumulator_len
        } else {
            k_dec
        };
        let ccs_out_len = inputs_len + 1;

        let mut ccs_out = Vec::with_capacity(ccs_out_len);
        for _ in 0..ccs_out_len {
            ccs_out.push(dummy_me_instance(m_in, y_len_total, r_len, kappa));
        }

        let mut proof = neo_reductions::PiCcsProof {
            sumcheck_rounds: vec![vec![NeoK::ZERO; d_sc + 1]; dims.ell],
            sc_initial_sum: Some(NeoK::ZERO),
            sumcheck_challenges: vec![NeoK::ZERO; dims.ell],
            challenges_public: neo_reductions::Challenges {
                alpha: vec![NeoK::ZERO; dims.ell_d],
                beta_a: vec![NeoK::ZERO; dims.ell_d],
                beta_r: vec![NeoK::ZERO; dims.ell_n],
                gamma: NeoK::ZERO,
            },
            sumcheck_final: NeoK::ZERO,
            header_digest: vec![0u8; 32],
            _extra: None,
        };
        pad_sumcheck_rounds_in_place(d_sc, &mut proof);

        let rlc_rhos = vec![neo_ccs::Mat::zero(neo_math::D, neo_math::D, NeoF::ZERO); ccs_out_len];
        let rlc_parent = dummy_me_instance(m_in, y_len_total, r_len, kappa);

        let mut dec_children = Vec::with_capacity(k_dec);
        for _ in 0..k_dec {
            dec_children.push(dummy_me_instance(m_in, y_len_total, r_len, kappa));
        }

        let fold = FoldStep {
            ccs_out,
            ccs_proof: proof,
            rlc_rhos,
            rlc_parent,
            dec_children,
        };

        let n_lut = step_public.lut_insts.len();
        let n_mem = step_public.mem_insts.len();
        let has_prev = step_idx > 0;

        let ob_inc_total_degree_bound = if step_idx + 1 == shape.step_count as usize {
            if let Some(cfg) = shape.output_binding.as_ref() {
                let mem_inst = step_public.mem_insts.get(cfg.mem_idx).ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(
                        "dummy_witness_for_shape: output binding mem_idx out of range".into(),
                    )
                })?;
                let ell_addr = mem_inst.twist_layout().ell_addr;
                Some(2 + ell_addr)
            } else {
                None
            }
        } else {
            None
        };

        // Route-A batched time proof shape (multi-claim when mem is enabled).
        let metas = neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan::time_claim_metas_for_step(
            step_public,
            d_sc,
            ob_inc_total_degree_bound,
        );
        let labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
        let degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
        let mut round_polys: Vec<Vec<Vec<NeoK>>> = Vec::with_capacity(metas.len());
        for &deg in degree_bounds.iter() {
            round_polys.push(vec![vec![NeoK::ZERO; deg + 1]; dims.ell_n]);
        }

        // Memory sidecar proof shape.
        let shout_addr_pre = if n_lut == 0 {
            Default::default()
        } else {
            // Fixed-count profile: one addr-pre sumcheck per LUT instance.
            let ell_addr = step_public
                .lut_insts
                .first()
                .map(|inst| inst.d * inst.ell)
                .unwrap_or(0);
            neo_fold::shard::ShoutAddrPreProof {
                claimed_sums: vec![NeoK::ZERO; n_lut],
                round_polys: vec![vec![vec![NeoK::ZERO; 3]; ell_addr]; n_lut],
                r_addr: vec![NeoK::ZERO; ell_addr],
            }
        };

        let mut proofs: Vec<neo_fold::shard::MemOrLutProof> = Vec::with_capacity(n_lut + n_mem);
        for _ in 0..n_lut {
            proofs.push(neo_fold::shard::MemOrLutProof::Shout(neo_memory::shout::ShoutProof::<NeoK>::default()));
        }
        for mem_idx in 0..n_mem {
            let mem_inst = &step_public.mem_insts[mem_idx];
            let ell_addr = mem_inst.d * mem_inst.ell;

            let addr_pre = neo_memory::sumcheck_proof::BatchedAddrProof {
                claimed_sums: vec![NeoK::ZERO; 2],
                round_polys: vec![vec![vec![NeoK::ZERO; 3]; ell_addr]; 2],
                r_addr: vec![NeoK::ZERO; ell_addr],
            };

            let val_plan =
                neo_fold::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step_public.mem_insts.iter(), has_prev);
            let base = val_plan.base(mem_idx);
            let lt_deg = val_plan
                .degree_bounds
                .get(base)
                .copied()
                .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: val-eval lt degree missing".into()))?;
            let total_deg = val_plan
                .degree_bounds
                .get(base + 1)
                .copied()
                .ok_or_else(|| SpartanBridgeError::InvalidInput("dummy_witness_for_shape: val-eval total degree missing".into()))?;
            let prev_total_deg = has_prev.then(|| {
                val_plan
                    .degree_bounds
                    .get(base + 2)
                    .copied()
                    .ok_or_else(|| {
                        SpartanBridgeError::InvalidInput("dummy_witness_for_shape: val-eval prev_total degree missing".into())
                    })
            });

            let val_eval = Some(neo_memory::twist::TwistValEvalProof {
                claimed_inc_sum_lt: NeoK::ZERO,
                rounds_lt: vec![vec![NeoK::ZERO; lt_deg + 1]; dims.ell_n],
                claimed_inc_sum_total: NeoK::ZERO,
                rounds_total: vec![vec![NeoK::ZERO; total_deg + 1]; dims.ell_n],
                claimed_prev_inc_sum_total: has_prev.then_some(NeoK::ZERO),
                rounds_prev_total: match prev_total_deg {
                    None => None,
                    Some(Ok(deg)) => Some(vec![vec![NeoK::ZERO; deg + 1]; dims.ell_n]),
                    Some(Err(e)) => return Err(e),
                },
            });

            proofs.push(neo_fold::shard::MemOrLutProof::Twist(neo_memory::twist::TwistProof { addr_pre, val_eval }));
        }

        let cpu_me_claims_val = if n_mem == 0 {
            Vec::new()
        } else {
            let expected = 1 + if has_prev { 1 } else { 0 };
            let mut v = Vec::with_capacity(expected);
            for _ in 0..expected {
                v.push(dummy_me_instance(m_in, y_len_total, r_len, kappa));
            }
            v
        };

        let val_fold = if n_mem == 0 {
            None
        } else {
            let inputs = cpu_me_claims_val.len();
            let rlc_rhos = vec![neo_ccs::Mat::zero(neo_math::D, neo_math::D, NeoF::ZERO); inputs];
            let rlc_parent = dummy_me_instance(m_in, y_len_total, r_len, kappa);
            let mut dec_children = Vec::with_capacity(k_dec);
            for _ in 0..k_dec {
                dec_children.push(dummy_me_instance(m_in, y_len_total, r_len, kappa));
            }
            Some(neo_fold::shard::RlcDecProof {
                rlc_rhos,
                rlc_parent,
                dec_children,
            })
        };

        steps.push(StepProof {
            fold,
            mem: MemSidecarProof {
                cpu_me_claims_val,
                shout_addr_pre,
                proofs,
            },
            batched_time: BatchedTimeProof {
                claimed_sums: vec![NeoK::ZERO; metas.len()],
                degree_bounds,
                labels,
                round_polys,
            },
            val_fold,
        });
    }

    let run = FoldRun {
        steps,
        output_proof: shape.output_binding.as_ref().map(|cfg| neo_memory::output_check::OutputBindingProof {
            output_sc: neo_memory::output_check::OutputSumcheckProof {
                round_polys: vec![vec![NeoK::ZERO; 4]; cfg.num_bits],
            },
        }),
    };

    Ok(FoldRunWitness::new(
        run,
        shape.steps_public.clone(),
        initial_accumulator,
        shape.output_binding.clone(),
    )
    .with_step_linking(shape.step_linking.clone()))
}

/// Proof output from Spartan2
#[derive(Clone, Debug)]
pub struct SpartanProof {
    /// The actual Spartan proof bytes
    pub proof_data: Vec<u8>,

    /// Public statement (small, fixed-size).
    pub statement: SpartanShardStatement,
}

/// Setup a pinned Spartan2 (prover key, verifier key) for a particular circuit shape.
///
/// This does **not** require the full FoldRun witness: it constructs a dummy witness of the same
/// shape (all zeros) and runs `SpartanSnark::setup` on that circuit.
pub fn setup_fold_run_shape(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    shape: &FoldRunShape,
) -> Result<(SpartanProverKey, SpartanVerifierKey)> {
    let dummy = dummy_witness_for_shape(params, ccs, shape)?;

    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);
    let steps_digest = compute_steps_digest_v3(dummy.steps_public.as_slice())?;
    let acc_init_digest = compute_accumulator_digest_v2(params.b, dummy.initial_accumulator.as_slice());
    let obligations = dummy.fold_run.compute_final_obligations(dummy.initial_accumulator.as_slice());
    let acc_final_main_digest = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_final_val_digest = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());

    let mem_enabled = dummy
        .steps_public
        .iter()
        .any(|s| !s.lut_insts.is_empty() || !s.mem_insts.is_empty());

    let output_binding_enabled = dummy.output_binding.is_some();
    let program_io_digest = if let Some(cfg) = dummy.output_binding.as_ref() {
        compute_program_io_digest_v1(cfg)?
    } else {
        [0u8; 32]
    };
    let statement = SpartanShardStatement::new(
        params_digest,
        ccs_digest,
        [0u8; 32], // program_digest (Phase 1 placeholder)
        steps_digest, // steps_digest (Phase 1)
        program_io_digest,
        acc_init_digest,
        acc_final_main_digest,
        acc_final_val_digest,
        shape.step_count,
        mem_enabled,
        output_binding_enabled,
    );
    let instance = FoldRunInstance { statement };

    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, ccs).map_err(SpartanBridgeError::NeoError)?;
    let ext = params
        .extension_check(dims.ell as u32, dims.d_sc as u32)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("extension policy failed: {e:?}")))?;
    let slack_sign: u8 = if ext.slack_bits >= 0 { 1 } else { 0 };

    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices(ccs);
    if mat_digest.len() != 4 {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "CCS matrix digest must have len 4, got {}",
            mat_digest.len()
        )));
    }
    let mut ccs_mat_digest = [0u64; 4];
    for (i, d) in mat_digest.iter().enumerate() {
        use p3_field::PrimeField64;
        ccs_mat_digest[i] = d.as_canonical_u64();
    }

    let mut terms: Vec<_> = ccs.f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps);
    let poly_f: Vec<CircuitPolyTerm> = terms
        .into_iter()
        .map(|term| {
            use p3_field::PrimeField64;
            let coeff_circ = CircuitF::from(term.coeff.as_canonical_u64());
            CircuitPolyTerm {
                coeff: coeff_circ,
                coeff_native: term.coeff,
                exps: term.exps.iter().map(|e| *e as u32).collect(),
            }
        })
        .collect();

    let delta = CircuitF::from(7u64);
    let circuit = FoldRunCircuit::new(
        instance,
        Some(dummy),
        ccs.n,
        ccs.m,
        ccs.t(),
        ccs.f.arity(),
        dims.ell_d,
        dims.ell_n,
        dims.ell,
        dims.d_sc,
        params.lambda,
        ext.s_supported,
        ext.slack_bits.unsigned_abs() as u32,
        slack_sign,
        ccs_mat_digest,
        delta,
        params.b,
        poly_f,
        shape.step_linking.clone(),
    );

    let (pk, vk) = SpartanSnark::setup(circuit)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 setup failed: {e}")))
        ?;

    if std::env::var("NEO_SPARTAN_BRIDGE_DEBUG_SHAPE").ok().as_deref() == Some("1") {
        eprintln!(
            "[neo-spartan-bridge] setup shape sizes = {:?}",
            pk.sizes()
        );
    }

    Ok((pk, vk))
}

/// Generate a pinned Spartan2 (prover key, verifier key) for a particular circuit shape.
///
/// The returned `SpartanVerifierKey` must be treated as **public and pinned**: verifiers must not
/// accept a verifier key bundled by the prover inside the proof.
pub fn setup_fold_run(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    witness: &FoldRunWitness,
) -> Result<(SpartanProverKey, SpartanVerifierKey)> {
    // Shape guardrails (cheap host checks).
    enforce_sumcheck_degree_bounds(params, ccs, witness)?;

    let shape = FoldRunShape::from_witness(witness)?;
    setup_fold_run_shape(params, ccs, &shape)
}

/// Generate a Spartan proof for a FoldRun.
///
/// This:
/// - Builds a `FoldRunInstance` from the FoldRun + Neo params/CCS digests.
/// - Uses the caller-provided initial accumulator (ME(b, L)^k inputs to step 0).
/// - Extracts Π-CCS challenges per step from the embedded proofs.
/// - Synthesizes the FoldRun circuit as a Spartan2 `SpartanCircuit`.
/// - Runs Spartan2 setup/prep/prove using the Goldilocks + Hash-MLE PCS engine.
pub fn prove_fold_run(
    pk: &SpartanProverKey,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    mut witness: FoldRunWitness,
) -> Result<SpartanProof> {
    let steps_public = witness.steps_public.as_slice();
    let initial_accumulator = witness.initial_accumulator.as_slice();

    // Enforce sumcheck degree bounds on the Π-CCS proofs before we even
    // build the circuit. This mirrors the native verifier's policy that
    // each round polynomial must have degree ≤ d_sc.
    enforce_sumcheck_degree_bounds(params, ccs, &witness)?;

    // Normalize: pad each sumcheck round polynomial to fixed length (d_sc+1) so the
    // circuit shape is determined by (params, ccs, step_count) rather than trimming.
    let d_sc = {
        let max_deg = ccs.max_degree() as usize + 1;
        let range_bound = core::cmp::max(2, 2 * (params.b as usize) + 2);
        core::cmp::max(max_deg, range_bound)
    };
    for (step_idx, step) in witness.fold_run.steps.iter_mut().enumerate() {
        pad_sumcheck_rounds_in_place(d_sc, &mut step.fold.ccs_proof);

        // Route-A batched time: pad each claim to (degree_bound+1).
        if step.batched_time.round_polys.len() != step.batched_time.degree_bounds.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: batched_time round_polys/degree_bounds length mismatch"
            )));
        }
        for (claim_idx, (claim_rounds, &deg)) in step
            .batched_time
            .round_polys
            .iter_mut()
            .zip(step.batched_time.degree_bounds.iter())
            .enumerate()
        {
            let target = deg + 1;
            for (round_idx, round) in claim_rounds.iter_mut().enumerate() {
                if round.len() > target {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: batched_time claim {claim_idx} round {round_idx} exceeds degree bound (len={} > {})",
                        round.len(),
                        target
                    )));
                }
                if round.len() < target {
                    round.resize(target, NeoK::ZERO);
                }
            }
        }

        // Memory sidecar proof normalization (fixed circuit shape).
        let step_public = steps_public.get(step_idx).ok_or_else(|| {
            SpartanBridgeError::InvalidInput(format!("step {step_idx}: missing StepInstanceBundle"))
        })?;
        let n_lut = step_public.lut_insts.len();
        let n_mem = step_public.mem_insts.len();

        // Shout addr-pre (fixed-count profile): degree bound 2 => 3 coeffs.
        for (lut_idx, rounds) in step.mem.shout_addr_pre.round_polys.iter_mut().enumerate() {
            for (round_idx, round) in rounds.iter_mut().enumerate() {
                if round.len() > 3 {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: shout_addr_pre round_polys[{lut_idx}][{round_idx}] too long (len={})",
                        round.len()
                    )));
                }
                if round.len() < 3 {
                    round.resize(3, NeoK::ZERO);
                }
            }
        }

        // Twist proofs live after the n_lut Shout placeholders.
        let has_prev = step_idx > 0;
        let val_plan = neo_fold::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step_public.mem_insts.iter(), has_prev);

        for mem_idx in 0..n_mem {
            let proof_idx = n_lut + mem_idx;
            let twist = match step.mem.proofs.get_mut(proof_idx) {
                Some(neo_fold::shard::MemOrLutProof::Twist(p)) => p,
                _ => {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: expected Twist proof at mem.proofs[{proof_idx}]"
                    )))
                }
            };

            // Addr-pre: degree bound 2 => 3 coeffs (2-claim batch).
            for (claim_idx, claim_rounds) in twist.addr_pre.round_polys.iter_mut().enumerate() {
                for (round_idx, round) in claim_rounds.iter_mut().enumerate() {
                    if round.len() > 3 {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: twist addr_pre claim {claim_idx} round {round_idx} too long (len={})",
                            round.len()
                        )));
                    }
                    if round.len() < 3 {
                        round.resize(3, NeoK::ZERO);
                    }
                }
            }

            // Val-eval (time-domain): pad to plan degree bounds.
            if let Some(val_eval) = twist.val_eval.as_mut() {
                let base = val_plan.base(mem_idx);
                let lt_deg = *val_plan
                    .degree_bounds
                    .get(base)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput("val_plan missing lt degree".into()))?;
                let total_deg = *val_plan
                    .degree_bounds
                    .get(base + 1)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput("val_plan missing total degree".into()))?;

                let lt_target = lt_deg + 1;
                for (round_idx, round) in val_eval.rounds_lt.iter_mut().enumerate() {
                    if round.len() > lt_target {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: twist val_eval lt round {round_idx} too long (len={} > {lt_target})",
                            round.len()
                        )));
                    }
                    if round.len() < lt_target {
                        round.resize(lt_target, NeoK::ZERO);
                    }
                }

                let total_target = total_deg + 1;
                for (round_idx, round) in val_eval.rounds_total.iter_mut().enumerate() {
                    if round.len() > total_target {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: twist val_eval total round {round_idx} too long (len={} > {total_target})",
                            round.len()
                        )));
                    }
                    if round.len() < total_target {
                        round.resize(total_target, NeoK::ZERO);
                    }
                }

                if has_prev {
                    let prev_deg = *val_plan
                        .degree_bounds
                        .get(base + 2)
                        .ok_or_else(|| SpartanBridgeError::InvalidInput("val_plan missing prev_total degree".into()))?;
                    let prev_target = prev_deg + 1;

                    if val_eval.rounds_prev_total.is_none() || val_eval.claimed_prev_inc_sum_total.is_none() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: expected rollover prev_total fields in val_eval"
                        )));
                    }
                    let rounds_prev = val_eval.rounds_prev_total.as_mut().expect("checked");
                    for (round_idx, round) in rounds_prev.iter_mut().enumerate() {
                        if round.len() > prev_target {
                            return Err(SpartanBridgeError::InvalidInput(format!(
                                "step {step_idx}: twist val_eval prev_total round {round_idx} too long (len={} > {prev_target})",
                                round.len()
                            )));
                        }
                        if round.len() < prev_target {
                            round.resize(prev_target, NeoK::ZERO);
                        }
                    }
                } else {
                    if val_eval.rounds_prev_total.is_some() || val_eval.claimed_prev_inc_sum_total.is_some() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: unexpected rollover prev_total fields in val_eval (no prev step)"
                        )));
                    }
                }
            }
        }
    }

    // Output binding proof normalization (fixed circuit shape).
    if let Some(cfg) = witness.output_binding.as_ref() {
        cfg.program_io
            .validate(cfg.num_bits)
            .map_err(|e| SpartanBridgeError::InvalidInput(format!("invalid ProgramIO for output binding: {e:?}")))?;

        let output_proof = witness
            .fold_run
            .output_proof
            .as_mut()
            .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding enabled but fold_run.output_proof is None".into()))?;

        if output_proof.output_sc.round_polys.len() != cfg.num_bits {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "output sumcheck rounds mismatch: proof has {}, cfg.num_bits={}",
                output_proof.output_sc.round_polys.len(),
                cfg.num_bits
            )));
        }

        for (round_idx, coeffs) in output_proof.output_sc.round_polys.iter_mut().enumerate() {
            if coeffs.len() > 4 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "output sumcheck round {round_idx} too long (len={} > 4)",
                    coeffs.len()
                )));
            }
            if coeffs.len() < 4 {
                coeffs.resize(4, NeoK::ZERO);
            }
        }

        for (step_idx, step_public) in steps_public.iter().enumerate() {
            let mem_inst = step_public.mem_insts.get(cfg.mem_idx).ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding mem_idx={} out of range (mem_insts.len()={})",
                    cfg.mem_idx,
                    step_public.mem_insts.len()
                ))
            })?;
            if mem_inst.d != cfg.num_bits {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding num_bits mismatch: cfg.num_bits={}, mem_inst.d={}",
                    cfg.num_bits, mem_inst.d
                )));
            }
        }
    }

    let fold_run = &witness.fold_run;
    // 1. Compute digests of params, CCS, and MCS
    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);
    let steps_digest = compute_steps_digest_v3(steps_public)?;
    let obligations = fold_run.compute_final_obligations(initial_accumulator);
    let acc_init_digest = compute_accumulator_digest_v2(params.b, initial_accumulator);
    let acc_final_main_digest = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_final_val_digest = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    let step_count = u32::try_from(fold_run.steps.len()).map_err(|_| {
        SpartanBridgeError::InvalidInput(format!(
            "FoldRun has too many steps for u32: {}",
            fold_run.steps.len()
        ))
    })?;
    if steps_public.len() != fold_run.steps.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "steps_public length mismatch: steps_public={}, fold_run.steps={}",
            steps_public.len(),
            fold_run.steps.len()
        )));
    }

    let mem_enabled = steps_public
        .iter()
        .any(|s| !s.lut_insts.is_empty() || !s.mem_insts.is_empty());

    let output_binding_enabled = witness.output_binding.is_some();
    match (output_binding_enabled, fold_run.output_proof.is_some()) {
        (true, false) => {
            return Err(SpartanBridgeError::InvalidInput(
                "output binding enabled but fold_run.output_proof is None".into(),
            ))
        }
        (false, true) => {
            return Err(SpartanBridgeError::InvalidInput(
                "fold_run.output_proof is Some but witness.output_binding is None".into(),
            ))
        }
        _ => {}
    }
    if output_binding_enabled && !mem_enabled {
        return Err(SpartanBridgeError::InvalidInput(
            "output binding requires mem_enabled=true (non-empty Twist instances)".into(),
        ));
    }

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[spartan-bridge] Proving FoldRun with {} steps", fold_run.steps.len());
        eprintln!(
            "[spartan-bridge] initial_accumulator.len() = {}",
            initial_accumulator.len()
        );
        for (step_idx, step) in fold_run.steps.iter().enumerate() {
            let proof = &step.fold.ccs_proof;
            eprintln!("\n[spartan-bridge] === Step {} ===", step_idx);
            eprintln!(
                "[spartan-bridge]   ccs_out.len() = {}, dec_children.len() = {}",
                step.fold.ccs_out.len(),
                step.fold.dec_children.len()
            );
            eprintln!(
                "[spartan-bridge]   sumcheck_rounds = {}, sumcheck_challenges = {}",
                proof.sumcheck_rounds.len(),
                proof.sumcheck_challenges.len()
            );
            eprintln!(
                "[spartan-bridge]   alpha.len() = {}, beta_a.len() = {}, beta_r.len() = {}",
                proof.challenges_public.alpha.len(),
                proof.challenges_public.beta_a.len(),
                proof.challenges_public.beta_r.len()
            );

            // Compute scalar T and RHS using the native paper-exact utilities
            // for comparison.
            if !step.fold.ccs_out.is_empty() {
                let dims = neo_reductions::engines::utils::build_dims_and_policy(params, ccs)
                    .map_err(SpartanBridgeError::NeoError)?;
                let ell_n = dims.ell_n;
                let ell = dims.ell;
                eprintln!("[spartan-bridge]   dims.ell_n = {}, dims.ell = {}", ell_n, ell);

                // The ME inputs for this step as seen by the native verifier:
                let me_inputs: Vec<MeInstance<Cmt, NeoF, NeoK>> = if step_idx == 0 {
                    initial_accumulator.to_vec()
                } else {
                    fold_run.steps[step_idx - 1].fold.dec_children.clone()
                };

                let T_native = claimed_initial_sum_from_inputs::<NeoF>(ccs, &proof.challenges_public, &me_inputs);
                eprintln!(
                    "[spartan-bridge]   native claimed_initial_sum T = {}",
                    format_ext(T_native)
                );
                // Host-side recomputation of T using the same formula as
                // `claimed_initial_sum_from_inputs` (including the outer γ^k).
                let T_bridge_host: NeoK = {
                    use core::cmp::min;

                    let k_total = 1 + me_inputs.len();
                    if k_total < 2 {
                        NeoK::ZERO
                    } else {
                        // Build χ_{α} over the Ajtai domain
                        let d_sz = 1usize
                            .checked_shl(proof.challenges_public.alpha.len() as u32)
                            .unwrap_or(0);
                        let mut chi_a = vec![NeoK::ZERO; d_sz];
                        for rho in 0..d_sz {
                            let mut w = NeoK::ONE;
                            for (bit, &a) in proof.challenges_public.alpha.iter().enumerate() {
                                let is_one = ((rho >> bit) & 1) == 1;
                                w *= if is_one { a } else { NeoK::ONE - a };
                            }
                            chi_a[rho] = w;
                        }

                        // Number of matrices t: use y-table length from ME inputs.
                        let t = if me_inputs.is_empty() { 0 } else { me_inputs[0].y.len() };

                        // γ^k
                        let mut gamma_to_k = NeoK::ONE;
                        for _ in 0..k_total {
                            gamma_to_k *= proof.challenges_public.gamma;
                        }

                        let mut inner = NeoK::ZERO;
                        for j in 0..t {
                            for (idx, out) in me_inputs.iter().enumerate() {
                                let i_abs = idx + 2;

                                let yj = &out.y[j];
                                let mut y_eval = NeoK::ZERO;
                                let limit = min(d_sz, yj.len());
                                for rho in 0..limit {
                                    y_eval += yj[rho] * chi_a[rho];
                                }

                                let mut weight = NeoK::ONE;
                                for _ in 0..(i_abs - 1) {
                                    weight *= proof.challenges_public.gamma;
                                }
                                for _ in 0..j {
                                    weight *= gamma_to_k;
                                }

                                inner += weight * y_eval;
                            }
                        }

                        // Match paper-exact engine: T = γ^k · inner.
                        gamma_to_k * inner
                    }
                };
                eprintln!(
                    "[spartan-bridge]   bridge claimed_initial_sum T (host) = {}",
                    format_ext(T_bridge_host)
                );
                if let Some(sc0) = proof.sc_initial_sum {
                    eprintln!("[spartan-bridge]   proof.sc_initial_sum = {}", format_ext(sc0));
                } else {
                    eprintln!("[spartan-bridge]   proof.sc_initial_sum = <None>");
                }

                // Compute native RHS terminal identity for debugging
                let rhs_native = neo_reductions::paper_exact_engine::rhs_terminal_identity_paper_exact(
                    &ccs.ensure_identity_first()
                        .map_err(|e| SpartanBridgeError::InvalidInput(format!("Identity check failed: {:?}", e)))?,
                    params,
                    &proof.challenges_public,
                    &proof.sumcheck_challenges[..dims.ell_n],
                    &proof.sumcheck_challenges[dims.ell_n..],
                    &step.fold.ccs_out,
                    if step_idx == 0 {
                        Some(&initial_accumulator[0].r)
                    } else {
                        Some(&fold_run.steps[step_idx - 1].fold.dec_children[0].r)
                    },
                );
                eprintln!("[spartan-bridge]   rhs_native(α′,r′) = {}", format_ext(rhs_native));

                eprintln!(
                    "[spartan-bridge]   proof.sumcheck_final = {}",
                    format_ext(proof.sumcheck_final)
                );
            }
        }
    }

    // 3. Build the public statement.
    let program_io_digest = if let Some(cfg) = witness.output_binding.as_ref() {
        compute_program_io_digest_v1(cfg)?
    } else {
        [0u8; 32]
    };
    let statement = SpartanShardStatement::new(
        params_digest,
        ccs_digest,
        [0u8; 32], // program_digest (Phase 1 placeholder)
        steps_digest, // steps_digest (Phase 1)
        program_io_digest,
        acc_init_digest,
        acc_final_main_digest,
        acc_final_val_digest,
        step_count,
        mem_enabled,
        output_binding_enabled,
    );
    let instance = FoldRunInstance { statement };

    // 4. Extract CCS polynomial f into circuit-friendly representation
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, ccs).map_err(SpartanBridgeError::NeoError)?;
    let ext = params
        .extension_check(dims.ell as u32, dims.d_sc as u32)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("extension policy failed: {e:?}")))?;
    let slack_sign: u8 = if ext.slack_bits >= 0 { 1 } else { 0 };

    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices(ccs);
    if mat_digest.len() != 4 {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "CCS matrix digest must have len 4, got {}",
            mat_digest.len()
        )));
    }
    let mut ccs_mat_digest = [0u64; 4];
    for (i, d) in mat_digest.iter().enumerate() {
        use p3_field::PrimeField64;
        ccs_mat_digest[i] = d.as_canonical_u64();
    }

    let mut terms: Vec<_> = ccs.f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps);
    let poly_f: Vec<CircuitPolyTerm> = terms
        .into_iter()
        .map(|term| {
            use p3_field::PrimeField64;
            let coeff_circ = CircuitF::from(term.coeff.as_canonical_u64());
            CircuitPolyTerm {
                coeff: coeff_circ,
                coeff_native: term.coeff,
                exps: term.exps.iter().map(|e| *e as u32).collect(),
            }
        })
        .collect();

    // 5. Create circuit
    let delta = CircuitF::from(7u64); // Goldilocks K delta
    let step_linking = witness.step_linking.clone();
    let circuit = FoldRunCircuit::new(
        instance.clone(),
        Some(witness),
        ccs.n,
        ccs.m,
        ccs.t(),
        ccs.f.arity(),
        dims.ell_d,
        dims.ell_n,
        dims.ell,
        dims.d_sc,
        params.lambda,
        ext.s_supported,
        ext.slack_bits.unsigned_abs() as u32,
        slack_sign,
        ccs_mat_digest,
        delta,
        params.b,
        poly_f,
        step_linking,
    );

    if std::env::var("NEO_SPARTAN_BRIDGE_CHECK_SHAPE").ok().as_deref() == Some("1") {
        let pk_sizes = pk.sizes();
        let shape = <ShapeCS<SpartanEngine> as spartan2::bellpepper::r1cs::SpartanShape<SpartanEngine>>::r1cs_shape(&circuit)
            .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 shape extraction failed: {e}")))?;
        let got = shape.sizes();
        if got != pk_sizes {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Circuit shape mismatch vs pinned PK: pk_sizes={pk_sizes:?}, circuit_sizes={got:?} (this usually means witness-dependent synthesis)",
            )));
        }
    }

    // Preprocess: build preprocessed state (witness commitments, etc.).
    let prep = SpartanSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 prep_prove failed: {e}")))?;

    // Prove: produce the SNARK proof object.
    let snark = SpartanSnark::prove(pk, circuit, &prep, true)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 prove failed: {e}")))?;

    // Pack SNARK into proof bytes. The verifier key must be pinned out-of-band.
    let proof_data = bincode::serialize(&snark)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 proof serialization failed: {e}")))?;

    Ok(SpartanProof {
        proof_data,
        statement: instance.statement,
    })
}

/// Verify a Spartan proof for a FoldRun.
///
/// This:
/// - Checks Neo params/CCS digests against the proof's instance.
/// - Recomputes expected public IO from the instance digests.
/// - Deserializes the Spartan2 verifier key and SNARK.
/// - Runs Spartan2 verification and checks the returned public IO matches.
pub fn verify_fold_run(
    vk: &SpartanVerifierKey,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    steps_public: &[StepInstanceBundle<Cmt, NeoF, NeoK>],
    output_binding: Option<&OutputBindingConfig>,
    proof: &SpartanProof,
) -> Result<bool> {
    // 1. Verify digests match
    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);
    let steps_digest = compute_steps_digest_v3(steps_public)?;

    if proof.statement.version != STATEMENT_VERSION {
        return Err(SpartanBridgeError::VerificationError(format!(
            "Unsupported statement version: {}",
            proof.statement.version
        )));
    }
    if proof.statement.output_binding_enabled {
        let cfg = output_binding.ok_or_else(|| {
            SpartanBridgeError::VerificationError("output binding enabled but config missing".into())
        })?;
        let digest = compute_program_io_digest_v1(cfg)?;
        if proof.statement.program_io_digest != digest {
            return Err(SpartanBridgeError::VerificationError(
                "program_io_digest mismatch".into(),
            ));
        }
    } else {
        if output_binding.is_some() {
            return Err(SpartanBridgeError::VerificationError(
                "unexpected output binding config (statement.output_binding_enabled=false)".into(),
            ));
        }
        if proof.statement.program_io_digest != [0u8; 32] {
            return Err(SpartanBridgeError::VerificationError(
                "program_io_digest must be 0 when output binding is disabled".into(),
            ));
        }
    }

    if proof.statement.params_digest != params_digest {
        return Err(SpartanBridgeError::VerificationError("Params digest mismatch".into()));
    }

    if proof.statement.ccs_digest != ccs_digest {
        return Err(SpartanBridgeError::VerificationError("CCS digest mismatch".into()));
    }
    if proof.statement.steps_digest != steps_digest {
        return Err(SpartanBridgeError::VerificationError("Steps digest mismatch".into()));
    }
    let expected_mem_enabled = steps_public
        .iter()
        .any(|s| !s.lut_insts.is_empty() || !s.mem_insts.is_empty());
    if proof.statement.mem_enabled != expected_mem_enabled {
        return Err(SpartanBridgeError::VerificationError(
            "mem_enabled flag mismatch vs steps_public".into(),
        ));
    }

    // 2. Expected public IO must mirror `FoldRunCircuit::{allocate_public_inputs, public_values}`.
    //
    let expected_statement_io = proof.statement.public_io();

    // 3. Deserialize snark from proof bytes.
    let snark: SpartanSnark = bincode::deserialize(&proof.proof_data)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("Spartan2 proof deserialization failed: {e}")))?;

    // 4. Run Spartan2 verification.
    let io = snark
        .verify(vk)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("Spartan2 verification failed: {e}")))?;

    // 5. Check that the public IO returned by Spartan matches the expected statement encoding.
    if io != expected_statement_io {
        return Err(SpartanBridgeError::VerificationError("Spartan2 public IO mismatch".into()));
    }

    Ok(true)
}

/// Compute a digest of the Neo parameters
///
/// TODO: This is a minimal digest. In production, include more parameters.
fn compute_params_digest(params: &NeoParams) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(b"neo/spartan-bridge/params_digest/v1");
    let bytes = bincode::serialize(params).expect("NeoParams bincode serialize");
    hasher.update(&bytes);
    let hash = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(hash.as_bytes());
    digest
}

/// Compute a digest of the CCS structure
///
/// TODO: This is a minimal digest. In production, include matrix contents.
fn compute_ccs_digest(ccs: &CcsStructure<NeoF>) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(b"neo/spartan-bridge/ccs_digest/v1");
    hasher.update(&(ccs.n as u64).to_le_bytes());
    hasher.update(&(ccs.m as u64).to_le_bytes());
    hasher.update(&(ccs.t() as u64).to_le_bytes());

    // Matrix digest matches the native transcript header binder.
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices(ccs);
    hasher.update(&(mat_digest.len() as u64).to_le_bytes());
    for d in mat_digest {
        use p3_field::PrimeField64;
        hasher.update(&d.as_canonical_u64().to_le_bytes());
    }

    // Polynomial digest (coeff + exponent vector).
    let terms = ccs.f.terms();
    hasher.update(&(terms.len() as u64).to_le_bytes());
    for term in terms {
        use p3_field::PrimeField64;
        hasher.update(&term.coeff.as_canonical_u64().to_le_bytes());
        hasher.update(&(term.exps.len() as u64).to_le_bytes());
        for &e in &term.exps {
            hasher.update(&(e as u64).to_le_bytes());
        }
    }
    let hash = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(hash.as_bytes());
    digest
}

fn compute_steps_digest_v3(steps_public: &[StepInstanceBundle<Cmt, NeoF, NeoK>]) -> Result<[u8; 32]> {
    use neo_transcript::{Poseidon2Transcript, Transcript};

    let mut tr = Poseidon2Transcript::new(b"neo/spartan-bridge/steps_digest/v3");
    tr.append_message(b"steps/len", &(steps_public.len() as u64).to_le_bytes());

    for (step_idx, step) in steps_public.iter().enumerate() {
        tr.append_message(b"step/idx", &(step_idx as u64).to_le_bytes());

        // Bind memory/table metadata in the same canonical form as the native shard verifier.
        neo_fold::memory_sidecar::memory::absorb_step_memory(&mut tr, step);

        // Bind the MCS instance (x, m_in, c_data). This mirrors the instance absorption in
        // `neo_reductions::engines::utils::bind_header_and_instances_with_digest`.
        let inst = &step.mcs_inst;
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        tr.append_fields(b"c_data", &inst.c.data);
    }

    Ok(tr.digest32())
}

fn compute_program_io_digest_v1(cfg: &OutputBindingConfig) -> Result<[u8; 32]> {
    use neo_transcript::{Poseidon2Transcript, Transcript};

    cfg.program_io
        .validate(cfg.num_bits)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("invalid ProgramIO for output binding: {e:?}")))?;

    let mut tr = Poseidon2Transcript::new(b"neo/spartan-bridge/program_io_digest/v1");
    tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
    tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);
    cfg.program_io.absorb_into_transcript(&mut tr);
    Ok(tr.digest32())
}

/// Enforce that every sumcheck round polynomial in the Π-CCS proofs respects
/// the degree bound d_sc used by the native verifier.
///
/// This is a host-side check only; inside the circuit we assume that
/// `sumcheck_rounds` have already been truncated to the allowed length.
fn enforce_sumcheck_degree_bounds(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    witness: &FoldRunWitness,
) -> Result<()> {
    // Match the definition of d_sc in `neo_reductions::engines::utils`.
    let d_sc = {
        let max_deg = ccs.max_degree() as usize + 1;
        let range_bound = core::cmp::max(2, 2 * (params.b as usize) + 2);
        core::cmp::max(max_deg, range_bound)
    };

    for (step_idx, step_proof) in witness.fold_run.steps.iter().enumerate() {
        let proof = &step_proof.fold.ccs_proof;
        for (round_idx, round_poly) in proof.sumcheck_rounds.iter().enumerate() {
            if round_poly.len() > d_sc + 1 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Sumcheck round {} in step {} exceeds degree bound: len={} > d_sc+1={}",
                    round_idx,
                    step_idx,
                    round_poly.len(),
                    d_sc + 1,
                )));
            }
        }
    }

    Ok(())
}
