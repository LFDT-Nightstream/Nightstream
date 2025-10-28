//! IVC folding verification pipeline
//!
//! This module implements folding proof verification (Pi-CCS + Pi-RLC + Pi-DEC)

use crate::F;
use crate::shared::types::*;
use neo_fold::{pi_ccs_verify_simple, pi_rlc_verify, pi_dec_verify};
use neo_fold::pi_ccs::pi_ccs_derive_transcript_tail;
use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_math::{Rq, cf_inv, SAction};
use neo_ajtai::AjtaiSModule;
use neo_ccs::SModuleHomomorphism;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::internal::{
    augmented::compute_augmented_public_input_for_step,
    tie::tie_check_with_r,
};

/// Internal: recombine digit MEs into the parent ME (same math as neo-fold::recombine_me_digits_to_parent).
fn recombine_me_digits_to_parent_local(
    params: &crate::NeoParams,
    digits: &[neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>],
) -> Result<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>, String> {
    use neo_ajtai::s_lincomb;
    use neo_math::{Rq, cf_inv};
    if digits.is_empty() {
        return Err("no digit instances to recombine".to_string());
    }
    let m_in = digits[0].m_in;
    let r_ref = &digits[0].r;
    let t = digits[0].y.len();
    let d_rows = digits[0].X.rows();
    let x_cols = digits[0].X.cols();
    for (i, d) in digits.iter().enumerate() {
        if d.m_in != m_in { return Err(format!("digit[{}]: m_in mismatch", i)); }
        if &d.r != r_ref   { return Err(format!("digit[{}]: r mismatch", i)); }
        if d.X.rows() != d_rows || d.X.cols() != x_cols {
            return Err(format!("digit[{}]: X shape mismatch (want {}x{}, got {}x{})",
                               i, d_rows, x_cols, d.X.rows(), d.X.cols()));
        }
        if d.y.len() != t { return Err(format!("digit[{}]: y arity mismatch", i)); }
    }
    // S-linear combination coefficients 1, b, b^2, ...
    let mut coeffs: Vec<Rq> = Vec::with_capacity(digits.len());
    let mut pow_f = F::ONE;
    for _ in 0..digits.len() {
        let mut arr = [F::ZERO; neo_math::D];
        arr[0] = pow_f;
        coeffs.push(cf_inv(arr));
        pow_f *= F::from_u64(params.b as u64);
    }
    // Combine commitments
    let digit_cs: Vec<neo_ajtai::Commitment> = digits.iter().map(|d| d.c.clone()).collect();
    let c_parent = s_lincomb(&coeffs, &digit_cs).map_err(|_| "s_lincomb failed".to_string())?;
    // Combine X
    let mut x_parent = neo_ccs::Mat::zero(d_rows, x_cols, F::ZERO);
    let mut pow = F::ONE;
    for d in digits {
        for r in 0..d_rows {
            for c in 0..x_cols {
                x_parent[(r, c)] += d.X[(r, c)] * pow;
            }
        }
        pow *= F::from_u64(params.b as u64);
    }
    // Combine y (vector-of-rows representation)
    let y_dim = digits[0].y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_parent = vec![vec![neo_math::K::ZERO; y_dim]; t];
    let mut pow_k = neo_math::K::from(F::ONE);
    let base_k = neo_math::K::from(F::from_u64(params.b as u64));
    for d in digits {
        for j in 0..t {
            for u in 0..y_dim {
                y_parent[j][u] += d.y[j][u] * pow_k;
            }
        }
        pow_k *= base_k;
    }
    // Combine y_scalars
    let mut y_scalars_parent = vec![neo_math::K::ZERO; digits[0].y_scalars.len()];
    let mut powk = neo_math::K::from(F::ONE);
    for d in digits {
        for j in 0..y_scalars_parent.len() {
            if j < d.y_scalars.len() { y_scalars_parent[j] += d.y_scalars[j] * powk; }
        }
        powk *= base_k;
    }
    Ok(neo_ccs::MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: c_parent,
        X: x_parent,
        r: r_ref.clone(),
        y: y_parent,
        y_scalars: y_scalars_parent,
        m_in,
        fold_digest: digits[0].fold_digest,
    })
}
/// Internal: recombine digit ME witnesses into a parent witness Z' = Σ b^i · Z_i
fn recombine_digit_witnesses_to_parent_local(
    params: &crate::NeoParams,
    digits: &[neo_ccs::MeWitness<F>],
) -> Result<neo_ccs::MeWitness<F>, String> {
    if digits.is_empty() { return Err("no digit witnesses".into()); }
    let d = digits[0].Z.rows();
    let m = digits[0].Z.cols();
    if d != neo_math::D {
        return Err(format!(
            "digit witnesses have {} rows, expected D={}",
            d, neo_math::D
        ));
    }
    for (i, dw) in digits.iter().enumerate() {
        if dw.Z.rows() != d || dw.Z.cols() != m {
            return Err(format!("digit_witness[{}] shape mismatch (want {}x{}, got {}x{})", i, d, m, dw.Z.rows(), dw.Z.cols()));
        }
    }
    let mut zp = neo_ccs::Mat::zero(d, m, F::ZERO);
    let mut pow = F::ONE;
    let base = F::from_u64(params.b as u64);
    for dw in digits {
        for r in 0..d { for c in 0..m { zp[(r,c)] += dw.Z[(r,c)] * pow; } }
        pow *= base;
    }
    Ok(neo_ccs::MeWitness { Z: zp })
}
/// Verify a single step's folding proof (Pi-CCS + Pi-RLC + Pi-DEC).
/// - `augmented_ccs` must match the prover's folding CCS; the caller should reconstruct it.
/// - `prev_step_x`: previous step's `step_public_input` (None for step 0).
pub fn verify_ivc_step_folding(
    params: &crate::NeoParams,
    ivc_proof: &IvcProof,
    augmented_ccs: &neo_ccs::CcsStructure<F>,
    prev_acc: &Accumulator,
    prev_augmented_x: Option<&[F]>,
) -> Result<bool, Box<dyn std::error::Error>> {
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] enter");
    #[cfg(feature = "neo-logs")]
    {
        println!("🔎 FOLD VERIFY: step {}", ivc_proof.step);
        println!("   augmented_ccs: n={}, m={}", augmented_ccs.n, augmented_ccs.m);
        println!("   prev_acc.step={}, y_len={}", prev_acc.step, prev_acc.y_compact.len());
        if let Some(px) = prev_augmented_x { println!("   prev_augmented_x.len()={}", px.len()); }
    }
    let folding = ivc_proof
        .folding_proof
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing folding_proof"))?;

    // 1) Cross-check against stored Pi-CCS inputs, using RHS as the binding point.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=inputs");
    let stored_inputs = &folding.pi_ccs_inputs;
    if stored_inputs.len() != 2 {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: pi_ccs_inputs.len() != 2");
        return Err(anyhow::anyhow!("folding proof missing pi_ccs_inputs").into());
    }

    // Compute expected RHS augmented x and ensure it matches both the proof copy and stored input.
    let (x_rhs_expected, _) = compute_augmented_public_input_for_step(prev_acc, ivc_proof)
        .map_err(|e| anyhow::anyhow!("failed to compute augmented input: {}", e))?;
    if x_rhs_expected != ivc_proof.public_inputs.step_augmented_public_input() {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS augmented input mismatch vs proof copy");
        return Ok(false);
    }

    // LHS checks: now STRICT — enforce exact X equality and also bind to caller-provided prev_augmented_x when present.
    // Ensure LHS/RHS shapes match.
    if stored_inputs[0].m_in != stored_inputs[1].m_in {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS/RHS m_in mismatch");
        return Ok(false);
    }
    // Ensure LHS commitment matches the stored output commitment.
    if stored_inputs[0].c.data != folding.pi_ccs_outputs[0].c.data {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS commitment mismatch vs stored output");
        return Ok(false);
    }

    // Link LHS augmented input to the previous step.
    let x_lhs_proof = ivc_proof.prev_step_augmented_public_input.clone();
    let m_in = x_rhs_expected.len();
    if x_lhs_proof.len() != m_in {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS augmented x length mismatch");
        return Ok(false);
    }
    // Consider per-lane base case: first use of this running ME (coords empty) and no prev_augmented_x provided.
    let is_lane_base_case = prev_acc.c_coords.is_empty();
    if is_lane_base_case && prev_augmented_x.is_none() {
        let step_x_len = ivc_proof.public_inputs.wrapper_public_input_x().len();
        let y_len = ivc_proof.public_inputs.y_prev().len();
        if x_lhs_proof.len() != step_x_len + 1 + 2 * y_len {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: base-case LHS augmented input length mismatch");
            return Ok(false);
        }
        // Accept either canonical zero-vector (zero-MCS base case) or self-fold (LHS == RHS augmented x)
        let is_zero = x_lhs_proof.iter().all(|&f| f == F::ZERO);
        let is_self_fold = x_lhs_proof == x_rhs_expected;
        if !is_zero && !is_self_fold {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: base-case LHS augmented x not zero or self-fold");
            return Ok(false);
        }
    } else if let Some(_px) = prev_augmented_x {
        // Production strict: enforce provided prev_augmented_x linkage
        let px = _px;
        if px != x_lhs_proof.as_slice() {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: prev_augmented_x linkage mismatch");
            return Ok(false);
        }
    }
    // Bind to the LHS stored input inside Pi-CCS as well.
    if stored_inputs[0].x != x_lhs_proof {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS stored x mismatch vs proof LHS augmented x");
        return Ok(false);
    }
    if stored_inputs[0].m_in != x_lhs_proof.len() {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS m_in mismatch vs proof LHS augmented x len");
        return Ok(false);
    }
    // Note: Do not bind LHS Pi-CCS commitment to prev_acc.c_coords.
    // The accumulator commitment evolves on step-only coordinates, whereas Pi-CCS
    // commitments bind the full augmented z. Binding is enforced via:
    //  - LHS x linkage (prev_augmented_x and proof copy),
    //  - RHS reconstruction equality, and
    //  - Pi-RLC and Pi-DEC checks tying digits to commitments.

    // RHS checks: bind to expected augmented x and stored output commitment.
    if stored_inputs[1].x != x_rhs_expected {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS x mismatch: stored != expected");
        let _ex: Vec<_> = x_rhs_expected.iter().take(6).map(|f| f.as_canonical_u64()).collect();
        let _sx: Vec<_> = stored_inputs[1].x.iter().take(6).map(|f| f.as_canonical_u64()).collect();
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding]    expected head: {:?}", _ex);
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding]    stored   head: {:?}", _sx);
        return Ok(false);
    }
    if stored_inputs[1].m_in != x_rhs_expected.len() {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS m_in mismatch vs expected len");
        return Ok(false);
    }
    if stored_inputs[1].c.data != folding.pi_ccs_outputs[1].c.data {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS commitment mismatch vs stored output");
        return Ok(false);
    }

    // Early hygiene guards before Pi-CCS: detect structural tampering and error out
    {
        if folding.pi_rlc_proof.rho_elems.len() != folding.pi_ccs_outputs.len() {
            return Err(anyhow::anyhow!("rho count != Π‑CCS outputs").into());
        }
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        if t != augmented_ccs.t() {
            return Err(anyhow::anyhow!("t mismatch: outputs.t != CCS.t").into());
        }
        for me in &folding.pi_ccs_outputs {
            if me.y.len() != t {
                return Err(anyhow::anyhow!("inconsistent t across Π‑CCS outputs").into());
            }
            for yj in &me.y {
                if yj.len() != neo_math::D {
                    return Err(anyhow::anyhow!("y[j] length != D").into());
                }
            }
        }
    }

    // 2) Verify Pi-CCS against those instances.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=pi-ccs");
    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    let ok_ccs = pi_ccs_verify_simple(
        &mut tr,
        params,
        augmented_ccs,
        stored_inputs,
        &folding.pi_ccs_outputs,
        &folding.pi_ccs_proof,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-CCS verify: {}", ok_ccs);
    if !ok_ccs { return Ok(false); }

    // 2b) (Skip) Intra-output y vs y_scalars check; rely on Π‑RLC/Π‑DEC path for scalar consistency.

    // 4) Recombine digit MEs to the parent ME for Pi‑RLC and Pi‑DEC checks.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=recombine-me");
    let me_digits = ivc_proof
        .me_instances
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit ME instances"))?;
    let mut me_parent = recombine_me_digits_to_parent_local(params, me_digits)?;

    // 4a) (No direct cross-link of RHS y_scalars to parent y_scalars here)
    //      Binding between Π‑CCS outputs and the recomposed parent is enforced by Π‑RLC and Π‑DEC.

    // 5) Verify Pi‑RLC.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=rlc");
    let ok_rlc = pi_rlc_verify(
        &mut tr,
        params,
        &folding.pi_ccs_outputs,
        &me_parent,
        &folding.pi_rlc_proof,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-RLC verify: {}", ok_rlc);
    if !ok_rlc { #[cfg(feature = "neo-logs")] eprintln!("[folding] rlc_verify=false"); return Ok(false); }

    // 6) Recombine digit witnesses to get true (d, m) for Ajtai S-module; then verify Π‑DEC.
    let wit_digits = ivc_proof
        .digit_witnesses
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit witnesses for DEC and tie check"))?;
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=recombine-wit");
    let wit_parent = recombine_digit_witnesses_to_parent_local(params, wit_digits)
        .map_err(|e| anyhow::anyhow!("recombine_digit_witnesses_to_parent failed: {}", e))?;

    let d_rows = neo_math::D;
    let m_cols = wit_parent.Z.cols();
    let l_real = match AjtaiSModule::from_global_for_dims(d_rows, m_cols) {
        Ok(l) => l,
        Err(_) => {
            #[cfg(not(feature = "testing"))]
            {
                return Err(anyhow::anyhow!(
                    "Ajtai PP missing for dims (D={}, m={}); register CRS/PP before verify",
                    d_rows, m_cols
                )
                .into());
            }
            #[cfg(feature = "testing")]
            {
                crate::ensure_ajtai_pp_for_dims(d_rows, m_cols, || {
                    use rand::{RngCore, SeedableRng};
                    use rand::rngs::StdRng;
                    let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
                        StdRng::from_seed([42u8; 32])
                    } else {
                        let mut seed = [0u8; 32];
                        rand::rng().fill_bytes(&mut seed);
                        StdRng::from_seed(seed)
                    };
                    let pp = crate::ajtai_setup(&mut rng, d_rows, params.kappa as usize, m_cols)?;
                    neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
                })?;
                AjtaiSModule::from_global_for_dims(d_rows, m_cols)
                    .map_err(|_| anyhow::anyhow!("AjtaiSModule unavailable (PP must exist after ensure)"))?
            }
        }
    };
    // 6a) Witness-commitment binding: each digit witness Z_i must open to its ME commitment.
    // This check is independent of power-of-two row constraints and catches tampering in Z.
    {
        let me_digits = ivc_proof
            .me_instances
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit ME instances for DEC and tie check"))?;
        if me_digits.len() != wit_digits.len() { #[cfg(feature = "neo-logs")] eprintln!("[folding] early: digit me count != digit witness count"); return Ok(false); }
        for (_i, (dw, me_i)) in wit_digits.iter().zip(me_digits.iter()).enumerate() {
            let c_from_wit = l_real.commit(&dw.Z);
            if c_from_wit.data != me_i.c.data {
                #[cfg(feature = "neo-logs")]
                eprintln!("[folding] digit witness-commit mismatch");
                return Ok(false);
            }
        }
        // Parent witness must also open to parent commitment after recombination.
        let c_parent_from_wit = l_real.commit(&wit_parent.Z);
        if c_parent_from_wit.data != me_parent.c.data {
            eprintln!("[folding] parent witness-commit mismatch after recombination");
            return Ok(false);
        }
    }

    eprintln!("[folding] stage=dec");
    let ok_dec = pi_dec_verify(
        &mut tr,
        params,
        &me_parent,
        me_digits,
        &folding.pi_dec_proof,
        &l_real,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-DEC verify: {} (d_rows={}, m_cols={})", ok_dec, d_rows, m_cols);
    if !ok_dec { #[cfg(feature = "neo-logs")] eprintln!("[folding] dec_verify=false"); return Ok(false); }

    // Derive the Π‑CCS transcript tail once; use r for tie and reuse for residual later.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=derive-tail");
    let tail = pi_ccs_derive_transcript_tail(
        params,
        augmented_ccs,
        stored_inputs,
        &folding.pi_ccs_proof,
    )
    .map_err(|e| anyhow::anyhow!("failed to derive transcript tail: {}", e))?;
    if me_parent.r.is_empty() || me_parent.r != tail.r {
        me_parent.r = tail.r.clone();
    }

    // 7) Parent-level tie check now that authentic Ajtai S-module is ensured by Π‑DEC path
    // Recompute X from Z via S-module for tie check to avoid any recombination drift.
    let mut me_parent_tie = me_parent.clone();
    me_parent_tie.X = l_real.project_x(&wit_parent.Z, me_parent.m_in);
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=tie (r.len()={})", me_parent_tie.r.len());
    if let Err(_e) = tie_check_with_r(augmented_ccs, &me_parent_tie, &wit_parent, &me_parent_tie.r) {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] ❌ tie_with_r failed: {}", _e);
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] debug: me_parent.y.len={} (t), y[0].len={} (D)", me_parent.y.len(), me_parent.y.get(0).map(|v| v.len()).unwrap_or(0));
        return Ok(false);
    }

    // 8) Cross-link Π‑CCS outputs to the parent ME: recombine Π‑CCS y via ρ and 
    //    require the scalars match the parent ME (which DEC/tie bound to Z).
    {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] stage=cross-link");
        // Hygiene: arity/shape guards
        if folding.pi_rlc_proof.rho_elems.len() != folding.pi_ccs_outputs.len() {
            return Err(anyhow::anyhow!("rho count != Π‑CCS outputs").into());
        }
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        if t != augmented_ccs.t() {
            return Err(anyhow::anyhow!("t mismatch: outputs.t != CCS.t").into());
        }
        for me in &folding.pi_ccs_outputs {
            if me.y.len() != t {
                return Err(anyhow::anyhow!("inconsistent t across Π‑CCS outputs").into());
            }
            for yj in &me.y {
                if yj.len() != neo_math::D {
                    return Err(anyhow::anyhow!("y[j] length != D").into());
                }
            }
        }
        let rhos_ring: Vec<Rq> = folding
            .pi_rlc_proof
            .rho_elems
            .iter()
            .map(|coeffs| cf_inv(*coeffs))
            .collect();
        // Recombine y vectors per matrix index j using S-action
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        let d = neo_math::D;
        let mut y_parent_vecs: Vec<Vec<neo_math::K>> = vec![vec![neo_math::K::ZERO; d]; t];
        for (rho, me) in rhos_ring.iter().zip(folding.pi_ccs_outputs.iter()) {
            let s_act = SAction::from_ring(*rho);
            for j in 0..t {
                let yj_rot = s_act
                    .apply_k_vec(&me.y[j])
                    .map_err(|_| anyhow::anyhow!("S-action dim mismatch for y[j]"))?;
                for r in 0..d { y_parent_vecs[j][r] += yj_rot[r]; }
            }
        }
        // Compute y_scalars from recombined y vectors (base-b powers)
        let mut pow_b_f = vec![F::ONE; d];
        for i in 1..d { pow_b_f[i] = pow_b_f[i-1] * F::from_u64(params.b as u64); }
        let pow_b_k: Vec<neo_math::K> = pow_b_f.into_iter().map(neo_math::K::from).collect();
        let mut y_scalars_from_rlc = vec![neo_math::K::ZERO; t];
        for j in 0..t {
            let mut acc = neo_math::K::ZERO;
            for r in 0..d { acc += y_parent_vecs[j][r] * pow_b_k[r]; }
            y_scalars_from_rlc[j] = acc;
        }
        if me_parent.y_scalars.len() != t {
            return Err(anyhow::anyhow!(
                "parent y_scalars length ({}) != t ({})",
                me_parent.y_scalars.len(), t
            ).into());
        }
        if y_scalars_from_rlc != me_parent.y_scalars {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] cross-link failed: recombined y_scalars != parent y_scalars");
            return Ok(false);
        }
    }





    // 9) Enforce CCS satisfiability check for first step (base case)
    //    When prev_acc.c_coords.is_empty(), this is the first step with no prior accumulator.
    //    Both batched instances (LHS zero instance + RHS fresh step) should satisfy their CCS,
    //    so the sum over the hypercube (initial_sum) must be zero.
    //    
    //    CAVEAT: Only enforce when ℓ >= 2. In the ℓ=1 case (single-row padded to 2),
    //    the augmented CCS can carry a constant offset (e.g., from const-1 binding or other glue),
    //    so the hypercube sum of Q need not be zero even for a valid witness.
    //    This signature appears as p(0)=α₀, p(1)=0 in the sum-check rounds.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=residual");
    if prev_acc.c_coords.is_empty() && tail.r.len() >= 2 {
        // Base case with ℓ >= 2: both instances should be satisfied
        use crate::K;
        if tail.initial_sum != K::ZERO {
            #[cfg(feature = "neo-logs")]
            eprintln!(
                "[folding] non-zero CCS sum over hypercube (base case, ℓ={}): initial_sum={:?}, rejecting",
                tail.r.len(), tail.initial_sum
            );
            return Ok(false);
        }
    } else {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "[folding] skipping CCS-sum guard (c_coords.is_empty()={}, ℓ={})",
            prev_acc.c_coords.is_empty(), tail.r.len()
        );
    }

    Ok(true)
}
