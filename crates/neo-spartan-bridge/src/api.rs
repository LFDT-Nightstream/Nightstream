//! Public API for proving and verifying FoldRuns with Spartan2
//!
//! This module provides the high-level interface for:
//! - Converting a FoldRun into a Spartan proof
//! - Verifying a Spartan proof for a FoldRun
//!
//! **STATUS**: Experimental. Spartan2 integration is implemented and used for
//! benchmarking/proof-size measurement; the public statement/soundness story is
//! still evolving.

#![allow(unused_imports)]

use crate::circuit::fold_circuit::CircuitPolyTerm;
use crate::circuit::{FoldRunCircuit, FoldRunInstance, FoldRunWitness};
use crate::error::{Result, SpartanBridgeError};
use crate::CircuitF;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_fold::shard::ShardProof as FoldRun;
use neo_math::{F as NeoF, K as NeoK};
use neo_params::NeoParams;
use neo_reductions::common::format_ext;
use neo_reductions::paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs;
use p3_field::PrimeCharacteristicRing;

use spartan2::{provider::GoldilocksP3MerkleMleEngine, spartan::R1CSSNARK, traits::snark::R1CSSNARKTrait};

pub type SpartanEngine = GoldilocksP3MerkleMleEngine;
pub type SpartanSnark = R1CSSNARK<SpartanEngine>;
pub type SpartanProverKey = spartan2::spartan::SpartanProverKey<SpartanEngine>;
pub type SpartanVerifierKey = spartan2::spartan::SpartanVerifierKey<SpartanEngine>;

/// (ProverKey, VerifierKey) for a fixed FoldRun circuit shape.
///
/// In production, the verifier key is deployed once (not carried per proof).
pub struct SpartanKeypair {
    pub pk: SpartanProverKey,
    pub vk: SpartanVerifierKey,
}

/// Proof output from Spartan2
#[derive(Clone, Debug)]
pub struct SpartanProof {
    /// The Spartan SNARK proof bytes (does **not** include `vk`).
    pub snark_data: Vec<u8>,

    /// Public instance (for verification)
    pub instance: FoldRunInstance,
}

impl SpartanProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

fn materialize_fold_run_for_circuit(fold_run: &FoldRun) -> Result<FoldRun> {
    // CCS-only batched segments can legally have `proof_steps < public_steps`
    // without embedding nested substeps in `compressed_substeps`.
    // In that case, keep the run as-is.
    if !fold_run
        .steps
        .iter()
        .any(|step| step.compressed_substeps.is_some())
    {
        return Ok(fold_run.clone());
    }

    let mut materialized_steps = Vec::new();
    for (idx, step) in fold_run.steps.iter().enumerate() {
        if let Some(prefix_steps) = step.compressed_substeps.as_ref() {
            for (sub_idx, sub) in prefix_steps.iter().enumerate() {
                if sub.compressed_substeps.is_some() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "FoldRun step {} substep {} is nested-compressed; bridge materialization supports one compression level",
                        idx, sub_idx
                    )));
                }
                materialized_steps.push(sub.clone());
            }
            let mut terminal = step.clone();
            terminal.compressed_substeps = None;
            materialized_steps.push(terminal);
        } else {
            materialized_steps.push(step.clone());
        }
    }

    Ok(FoldRun {
        steps: materialized_steps,
        output_proof: fold_run.output_proof.clone(),
        riscv_profile: fold_run.riscv_profile.clone(),
        riscv_memory_layout: fold_run.riscv_memory_layout.clone(),
        segment_meta: None,
    })
}

fn build_fold_run_circuit(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    initial_accumulator: &[CeClaim<Cmt, NeoF, NeoK>],
    fold_run: &FoldRun,
    mut witness: FoldRunWitness,
) -> Result<FoldRunCircuit> {
    let materialized_run = materialize_fold_run_for_circuit(fold_run)?;
    witness.fold_run = materialized_run.clone();

    enforce_sumcheck_degree_bounds(params, ccs, &witness)?;

    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);

    // Extract challenges from FoldRun's Π-CCS proofs (native FS).
    let pi_ccs_challenges = extract_challenges_from_fold_run(&materialized_run, params, ccs)?;

    let instance = FoldRunInstance::from_fold_run(
        &materialized_run,
        params_digest,
        ccs_digest,
        initial_accumulator.to_vec(),
        pi_ccs_challenges,
    );

    // Extract CCS polynomial f into circuit-friendly representation.
    let poly_f: Vec<CircuitPolyTerm> = ccs
        .f
        .terms()
        .iter()
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

    let delta = CircuitF::from(7u64); // Goldilocks K delta
    Ok(FoldRunCircuit::new(instance, Some(witness), delta, params.b, poly_f))
}

/// Build `(pk, vk)` for a fixed FoldRun circuit shape.
///
/// In production, `vk` is deployed once and reused across many proofs.
pub fn setup_fold_run(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    initial_accumulator: &[CeClaim<Cmt, NeoF, NeoK>],
    fold_run: &FoldRun,
    witness_for_setup: FoldRunWitness,
) -> Result<SpartanKeypair> {
    let circuit = build_fold_run_circuit(params, ccs, initial_accumulator, fold_run, witness_for_setup)?;
    let (pk, vk) = SpartanSnark::setup(circuit)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 setup failed: {e}")))?;
    Ok(SpartanKeypair { pk, vk })
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
    initial_accumulator: &[CeClaim<Cmt, NeoF, NeoK>],
    fold_run: &FoldRun,
    witness: FoldRunWitness,
) -> Result<SpartanProof> {
    #[cfg(feature = "debug-logs")]
    {
        // Extract challenges from FoldRun's Π-CCS proofs (native FS) for logging.
        let pi_ccs_challenges = extract_challenges_from_fold_run(fold_run, params, ccs)?;

        eprintln!("[spartan-bridge] Proving FoldRun with {} steps", fold_run.steps.len());
        eprintln!(
            "[spartan-bridge] initial_accumulator.len() = {}",
            initial_accumulator.len()
        );
        for (step_idx, (step, ch)) in fold_run
            .steps
            .iter()
            .zip(pi_ccs_challenges.iter())
            .enumerate()
        {
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
                ch.alpha.len(),
                ch.beta_a.len(),
                ch.beta_r.len()
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
                let me_inputs: Vec<CeClaim<Cmt, NeoF, NeoK>> = if step_idx == 0 {
                    initial_accumulator.to_vec()
                } else {
                    fold_run.steps[step_idx - 1].fold.dec_children.clone()
                };

                if step.fold.ccs_out.len() < me_inputs.len() + 1 {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {} invalid output/input partition: k_total={}, me_inputs={}",
                        step_idx,
                        step.fold.ccs_out.len(),
                        me_inputs.len()
                    )));
                }
                let k_mcs = step.fold.ccs_out.len() - me_inputs.len();
                let T_native = claimed_initial_sum_from_inputs_with_k_mcs::<NeoF>(
                    ccs,
                    &proof.challenges_public,
                    k_mcs,
                    &me_inputs,
                );
                eprintln!(
                    "[spartan-bridge]   native claimed_initial_sum T = {}",
                    format_ext(T_native)
                );
                eprintln!(
                    "[spartan-bridge]   k_total = {}, k_mcs = {}, me_inputs = {}",
                    step.fold.ccs_out.len(),
                    k_mcs,
                    me_inputs.len()
                );
                if let Some(sc0) = proof.sc_initial_sum {
                    eprintln!("[spartan-bridge]   proof.sc_initial_sum = {}", format_ext(sc0));
                } else {
                    eprintln!("[spartan-bridge]   proof.sc_initial_sum = <None>");
                }

                // Compute native RHS terminal identity for debugging
                let rhs_native = neo_reductions::paper_exact_engine::rhs_terminal_identity_paper_exact_with_k_mcs(
                    &ccs.ensure_identity_first()
                        .map_err(|e| SpartanBridgeError::InvalidInput(format!("Identity check failed: {:?}", e)))?,
                    params,
                    &proof.challenges_public,
                    &ch.r_prime,
                    &ch.alpha_prime,
                    &step.fold.ccs_out,
                    k_mcs,
                    me_inputs.first().map(|me| me.r.as_slice()),
                );
                eprintln!("[spartan-bridge]   rhs_native(α′,r′) = {}", format_ext(rhs_native));

                eprintln!(
                    "[spartan-bridge]   proof.sumcheck_final = {}",
                    format_ext(proof.sumcheck_final)
                );
            }
        }
    }

    // 2. Build the FoldRun circuit from the instance + witness.
    let circuit = build_fold_run_circuit(params, ccs, initial_accumulator, fold_run, witness)?;
    let instance = circuit.instance.clone();

    // Preprocess: build preprocessed state (witness commitments, etc.).
    let prep = SpartanSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 prep_prove failed: {e}")))?;

    // Prove: produce the SNARK proof object.
    let snark = SpartanSnark::prove(pk, circuit, &prep, true)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 prove failed: {e}")))?;

    let snark_data = bincode::serialize(&snark)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("Spartan2 proof serialization failed: {e}")))?;

    Ok(SpartanProof { snark_data, instance })
}

/// Verify a Spartan proof for a FoldRun.
///
/// This:
/// - Checks Neo params/CCS digests against the proof's instance.
/// - Recomputes expected public IO from the instance digests.
/// - Deserializes the Spartan2 SNARK proof.
/// - Runs Spartan2 verification under the deployed `vk` and checks the returned public IO matches.
pub fn verify_fold_run(
    vk: &SpartanVerifierKey,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    proof: &SpartanProof,
) -> Result<bool> {
    // 1. Verify digests match
    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);

    if proof.instance.params_digest != params_digest {
        return Err(SpartanBridgeError::VerificationError("Params digest mismatch".into()));
    }

    if proof.instance.ccs_digest != ccs_digest {
        return Err(SpartanBridgeError::VerificationError("CCS digest mismatch".into()));
    }

    // 2. Recompute expected public IO from instance digests (must mirror
    // `FoldRunCircuit::public_values` / `allocate_public_inputs`).
    fn append_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
        for chunk in digest.chunks(4) {
            let mut limb_bytes = [0u8; 4];
            limb_bytes.copy_from_slice(chunk);
            let limb_u32 = u32::from_le_bytes(limb_bytes);
            out.push(CircuitF::from(limb_u32 as u64));
        }
    }

    let mut expected_io = Vec::with_capacity(16);
    append_digest(&mut expected_io, &proof.instance.params_digest);
    append_digest(&mut expected_io, &proof.instance.ccs_digest);

    // 3. Deserialize SNARK from proof bytes.
    let snark: SpartanSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("Spartan2 proof deserialization failed: {e}")))?;

    // 4. Run Spartan2 verification.
    let io = snark
        .verify(vk)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("Spartan2 verification failed: {e}")))?;

    // 5. Check that the public IO returned by Spartan matches the expected
    // digest limbs encoded in the FoldRun instance.
    if io != expected_io {
        return Err(SpartanBridgeError::VerificationError(
            "Spartan2 public IO mismatch FoldRun instance".into(),
        ));
    }

    Ok(true)
}

/// Compute a digest of the Neo parameters
///
/// TODO: This is a minimal digest. In production, include more parameters.
fn compute_params_digest(params: &NeoParams) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    let bytes = bincode::serialize(params).expect("NeoParams should be serializable");
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
    let bytes = bincode::serialize(ccs).expect("CcsStructure should be serializable");
    hasher.update(&bytes);
    let hash = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(hash.as_bytes());
    digest
}

fn extract_challenges_from_fold_run(
    fold_run: &FoldRun,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
) -> Result<Vec<crate::circuit::witness::PiCcsChallenges>> {
    use crate::circuit::witness::PiCcsChallenges;
    use neo_reductions::optimized_engine::PiCcsProofVariant;

    // Use the same dimension builder as the Π-CCS engines to recover
    // (ell_n, ell, ell_m, ell_nc) so we can split sumcheck points.
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(params, ccs).map_err(SpartanBridgeError::NeoError)?;
    let ell_n = dims.ell_n;
    let ell = dims.ell;
    let ell_m = dims.ell_m;
    let ell_nc = dims.ell_nc;

    let mut out = Vec::with_capacity(fold_run.steps.len());
    for (step_idx, step) in fold_run.steps.iter().enumerate() {
        let proof = &step.fold.ccs_proof;

        if proof.variant != PiCcsProofVariant::SplitNcV1 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRun step {}: Spartan bridge currently requires SplitNcV1 Π-CCS proofs (got {:?})",
                step_idx, proof.variant
            )));
        }

        if proof.sumcheck_rounds.len() != ell {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRun step {}: expected {} sumcheck rounds, got {}",
                step_idx,
                ell,
                proof.sumcheck_rounds.len()
            )));
        }
        if proof.sumcheck_challenges.len() != ell {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRun step {}: expected {} sumcheck challenges, got {}",
                step_idx,
                ell,
                proof.sumcheck_challenges.len()
            )));
        }
        if proof.sumcheck_rounds_nc.len() != ell_nc {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRun step {}: expected {} NC sumcheck rounds, got {}",
                step_idx,
                ell_nc,
                proof.sumcheck_rounds_nc.len()
            )));
        }
        if proof.sumcheck_challenges_nc.len() != ell_nc {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRun step {}: expected {} NC sumcheck challenges, got {}",
                step_idx,
                ell_nc,
                proof.sumcheck_challenges_nc.len()
            )));
        }

        let alpha = proof.challenges_public.alpha.clone();
        let beta_a = proof.challenges_public.beta_a.clone();
        let beta_r = proof.challenges_public.beta_r.clone();
        let beta_m = proof.challenges_public.beta_m.clone();
        let gamma = proof.challenges_public.gamma;

        let sumcheck_challenges = proof.sumcheck_challenges.clone();
        let (r_prime_slice, alpha_prime_slice) = sumcheck_challenges.split_at(ell_n);

        let sumcheck_challenges_nc = proof.sumcheck_challenges_nc.clone();
        let (s_col_prime_slice, alpha_prime_nc_slice) = sumcheck_challenges_nc.split_at(ell_m);

        out.push(PiCcsChallenges {
            alpha,
            beta_a,
            beta_r,
            beta_m,
            gamma,
            r_prime: r_prime_slice.to_vec(),
            alpha_prime: alpha_prime_slice.to_vec(),
            sumcheck_challenges,
            s_col_prime: s_col_prime_slice.to_vec(),
            alpha_prime_nc: alpha_prime_nc_slice.to_vec(),
            sumcheck_challenges_nc,
        });
    }

    Ok(out)
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

    for (step_idx, step) in witness.fold_run.steps.iter().enumerate() {
        let proof = &step.fold.ccs_proof;
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
        for (round_idx, round_poly) in proof.sumcheck_rounds_nc.iter().enumerate() {
            if round_poly.len() > d_sc + 1 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "NC sumcheck round {} in step {} exceeds degree bound: len={} > d_sc+1={}",
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
