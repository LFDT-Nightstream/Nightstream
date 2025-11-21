//! Public API for proving and verifying FoldRuns with Spartan2
//!
//! This module provides the high-level interface for:
//! - Converting a FoldRun into a Spartan proof
//! - Verifying a Spartan proof for a FoldRun
//!
//! **STATUS**: Stubs only. No Spartan2 integration yet.

#![allow(unused_imports)]

use crate::circuit::{FoldRunCircuit, FoldRunWitness, FoldRunInstance};
use crate::circuit::fold_circuit::CircuitPolyTerm;
use crate::error::{Result, SpartanBridgeError};
use crate::CircuitF;
use neo_fold::folding::FoldRun;
use neo_params::NeoParams;
use neo_ccs::{CcsStructure, MeInstance};
use neo_math::{F as NeoF, K as NeoK};
use neo_ajtai::Commitment as Cmt;
use neo_reductions::paper_exact_engine::claimed_initial_sum_from_inputs;
use neo_reductions::common::format_ext;
use p3_field::PrimeCharacteristicRing;

use spartan2::{
    spartan::R1CSSNARK,
    traits::snark::R1CSSNARKTrait,
    provider::GoldilocksP3MerkleMleEngine,
};

/// Proof output from Spartan2
#[derive(Clone, Debug)]
pub struct SpartanProof {
    /// The actual Spartan proof bytes
    pub proof_data: Vec<u8>,
    
    /// Public instance (for verification)
    pub instance: FoldRunInstance,
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
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    initial_accumulator: &[MeInstance<Cmt, NeoF, NeoK>],
    fold_run: &FoldRun,
    witness: FoldRunWitness,
) -> Result<SpartanProof> {
    // Enforce sumcheck degree bounds on the Π-CCS proofs before we even
    // build the circuit. This mirrors the native verifier's policy that
    // each round polynomial must have degree ≤ d_sc.
    enforce_sumcheck_degree_bounds(params, ccs, &witness)?;
    // 1. Compute digests of params, CCS, and MCS
    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);
    let mcs_digest = [0u8; 32]; // Would hash the MCS instances
    
    // 2. Extract challenges from FoldRun's Π-CCS proofs
    let pi_ccs_challenges = extract_challenges_from_fold_run(fold_run, params, ccs)?;
    #[cfg(feature = "debug-logs")]
    {
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
            let proof = &witness.pi_ccs_proofs[step_idx];
            eprintln!("\n[spartan-bridge] === Step {} ===", step_idx);
            eprintln!(
                "[spartan-bridge]   ccs_out.len() = {}, dec_children.len() = {}",
                step.ccs_out.len(),
                step.dec_children.len()
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
            if !step.ccs_out.is_empty() {
                let dims = neo_reductions::engines::utils::build_dims_and_policy(params, ccs)
                    .map_err(SpartanBridgeError::NeoError)?;
                let ell_n = dims.ell_n;
                let ell = dims.ell;
                eprintln!(
                    "[spartan-bridge]   dims.ell_n = {}, dims.ell = {}",
                    ell_n, ell
                );

                // The ME inputs for this step as seen by the native verifier:
                let me_inputs: Vec<MeInstance<Cmt, NeoF, NeoK>> = if step_idx == 0 {
                    initial_accumulator.to_vec()
                } else {
                    fold_run.steps[step_idx - 1].dec_children.clone()
                };

                let T_native =
                    claimed_initial_sum_from_inputs::<NeoF>(ccs, &proof.challenges_public, &me_inputs);
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
                        let t = if me_inputs.is_empty() {
                            0
                        } else {
                            me_inputs[0].y.len()
                        };

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
                    eprintln!(
                        "[spartan-bridge]   proof.sc_initial_sum = {}",
                        format_ext(sc0)
                    );
                } else {
                    eprintln!("[spartan-bridge]   proof.sc_initial_sum = <None>");
                }

                // Compute native RHS terminal identity for debugging
                let rhs_native = neo_reductions::paper_exact_engine::rhs_terminal_identity_paper_exact(
                    &ccs.ensure_identity_first().map_err(|e| SpartanBridgeError::InvalidInput(format!("Identity check failed: {:?}", e)))?,
                    params,
                    &proof.challenges_public,
                    &ch.r_prime,
                    &ch.alpha_prime,
                    &step.ccs_out,
                    if step_idx == 0 {
                        Some(&initial_accumulator[0].r)
                    } else {
                        Some(&fold_run.steps[step_idx - 1].dec_children[0].r)
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
    
    // 3. Build instance with the actual initial accumulator used by the
    // folding engine (ME(b, L)^k inputs to the first Π-CCS step).
    let initial_accumulator = initial_accumulator.to_vec();
    let instance = FoldRunInstance::from_fold_run(
        fold_run,
        params_digest,
        ccs_digest,
        mcs_digest,
        initial_accumulator,
        pi_ccs_challenges,
    );
    
    // 4. Extract CCS polynomial f into circuit-friendly representation
    let poly_f: Vec<CircuitPolyTerm> = ccs
        .f
        .terms()
        .iter()
        .map(|term| {
            use p3_field::PrimeField64;
            let coeff_circ = CircuitF::from(term.coeff.as_canonical_u64());
            CircuitPolyTerm {
                coeff: coeff_circ,
                exps: term.exps.iter().map(|e| *e as u32).collect(),
            }
        })
        .collect();
    
    // 5. Create circuit
    let delta = CircuitF::from(7u64);  // Goldilocks K delta
    let circuit = FoldRunCircuit::new(
        instance.clone(),
        Some(witness),
        delta,
        params.b,
        poly_f,
    );
    
    // 6. Run Spartan2 setup → prep → prove using the Goldilocks Hash-MLE engine.
    type E = GoldilocksP3MerkleMleEngine;
    type SNARK = R1CSSNARK<E>;

    // Setup: derive prover/verifier keys from the circuit shape.
    let (pk, vk) = SNARK::setup(circuit.clone()).map_err(|e| {
        SpartanBridgeError::ProvingError(format!("Spartan2 setup failed: {e}"))
    })?;

    // Preprocess: build preprocessed state (witness commitments, etc.).
    let prep = SNARK::prep_prove(&pk, circuit.clone(), true).map_err(|e| {
        SpartanBridgeError::ProvingError(format!("Spartan2 prep_prove failed: {e}"))
    })?;

    // Prove: produce the SNARK proof object.
    let snark = SNARK::prove(&pk, circuit, &prep, true).map_err(|e| {
        SpartanBridgeError::ProvingError(format!("Spartan2 prove failed: {e}"))
    })?;

    // Pack verifier key + SNARK into proof bytes.
    let packed = (vk, snark);
    let proof_data = bincode::serialize(&packed).map_err(|e| {
        SpartanBridgeError::ProvingError(format!("Spartan2 proof serialization failed: {e}"))
    })?;

    Ok(SpartanProof { proof_data, instance })
}

/// Verify a Spartan proof for a FoldRun.
///
/// This:
/// - Checks Neo params/CCS digests against the proof's instance.
/// - Recomputes expected public IO from the instance digests.
/// - Deserializes the Spartan2 verifier key and SNARK.
/// - Runs Spartan2 verification and checks the returned public IO matches.
pub fn verify_fold_run(
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    proof: &SpartanProof,
) -> Result<bool> {
    // 1. Verify digests match
    let params_digest = compute_params_digest(params);
    let ccs_digest = compute_ccs_digest(ccs);
    
    if proof.instance.params_digest != params_digest {
        return Err(SpartanBridgeError::VerificationError(
            "Params digest mismatch".into()
        ));
    }
    
    if proof.instance.ccs_digest != ccs_digest {
        return Err(SpartanBridgeError::VerificationError(
            "CCS digest mismatch".into()
        ));
    }
    
    // 2. Recompute expected public IO from instance digests (must mirror
    // `FoldRunCircuit::public_values` / `allocate_public_inputs`).
    fn append_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
        for chunk in digest.chunks(8) {
            let mut limb_bytes = [0u8; 8];
            limb_bytes.copy_from_slice(chunk);
            let limb_u64 = u64::from_le_bytes(limb_bytes);
            out.push(CircuitF::from(limb_u64));
        }
    }

    let mut expected_io = Vec::with_capacity(12);
    append_digest(&mut expected_io, &proof.instance.params_digest);
    append_digest(&mut expected_io, &proof.instance.ccs_digest);
    append_digest(&mut expected_io, &proof.instance.mcs_digest);

    // 3. Deserialize (vk, snark) from proof bytes.
    type E = GoldilocksP3MerkleMleEngine;
    type SNARK = R1CSSNARK<E>;
    type VK<E> = spartan2::spartan::SpartanVerifierKey<E>;

    let (vk, snark): (VK<E>, SNARK) = bincode::deserialize(&proof.proof_data).map_err(|e| {
        SpartanBridgeError::VerificationError(format!(
            "Spartan2 proof deserialization failed: {e}"
        ))
    })?;

    // 4. Run Spartan2 verification.
    let io = snark.verify(&vk).map_err(|e| {
        SpartanBridgeError::VerificationError(format!("Spartan2 verification failed: {e}"))
    })?;

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
    hasher.update(&params.q.to_le_bytes());
    hasher.update(&params.b.to_le_bytes());
    hasher.update(&params.k_rho.to_le_bytes());
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
    hasher.update(&ccs.m.to_le_bytes());
    hasher.update(&ccs.n.to_le_bytes());
    hasher.update(&ccs.t().to_le_bytes());
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

    // Use the same dimension builder as the Π-CCS engines to recover
    // (ell_n, ell) so we can split the sumcheck challenges.
    let s_norm = ccs
        .ensure_identity_first()
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("identity-first required: {e:?}")))?;

    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, &s_norm)
        .map_err(SpartanBridgeError::NeoError)?;
    let ell_n = dims.ell_n;
    let ell = dims.ell;

    let mut out = Vec::with_capacity(fold_run.steps.len());
    for (step_idx, step) in fold_run.steps.iter().enumerate() {
        let proof = &step.ccs_proof;

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

        let alpha = proof.challenges_public.alpha.clone();
        let beta_a = proof.challenges_public.beta_a.clone();
        let beta_r = proof.challenges_public.beta_r.clone();
        let gamma = proof.challenges_public.gamma;

        let sumcheck_challenges = proof.sumcheck_challenges.clone();
        let (r_prime_slice, alpha_prime_slice) = sumcheck_challenges.split_at(ell_n);

        out.push(PiCcsChallenges {
            alpha,
            beta_a,
            beta_r,
            gamma,
            r_prime: r_prime_slice.to_vec(),
            alpha_prime: alpha_prime_slice.to_vec(),
            sumcheck_challenges,
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

    for (step_idx, proof) in witness.pi_ccs_proofs.iter().enumerate() {
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
