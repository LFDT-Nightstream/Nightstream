//! Base case construction for IVC
//!
//! Provides utilities for creating zero MCS instances to start IVC chains.

use super::prelude::*;

/// Build a canonical zero MCS instance for a given shape (m_in, m_step) to start a fold chain.
///
/// This avoids the unsound base case where the prover folded the step with itself.
pub fn zero_mcs_instance_for_shape(
    m_in: usize,
    m_step: usize,
    const1_witness_index: Option<usize>,
) -> anyhow::Result<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> {
    let d = neo_math::ring::D;
    anyhow::ensure!(m_step >= m_in, "zero_mcs_instance_for_shape: m_step < m_in ({} < {})", m_step, m_in);

    // Construct a zero Ajtai commitment directly; avoid PP setup/commit computation.
    // Derive kappa from any registered Ajtai PP if available to avoid drift; fallback to 16.
    let _kappa = neo_ajtai::get_global_pp().map(|pp| pp.kappa).unwrap_or(16usize);
    let w_len = m_step - m_in;
    let x_zero = vec![F::ZERO; m_in];
    let mut w_zero = vec![F::ZERO; w_len];
    let mut z_zero = neo_ccs::Mat::zero(d, m_step, F::ZERO);
    // Ensure the constant-1 witness column is actually 1 in the base-case Z
    if let Some(idx) = const1_witness_index {
        if idx < w_len {
            // Column absolute index of const1 within z = [public || witness]
            let col_abs = m_in + idx;
            z_zero[(0, col_abs)] = F::ONE; // least-significant digit 1
            // Ensure undigitized z has 1 at the const1 witness index
            w_zero[idx] = F::ONE;
        }
    }

    // Use actual commitment for the base-case Z to satisfy c = L(Z)
    // Ensure Ajtai PP exists for (d, m_step). In testing, auto-ensure; in prod, error out.
    let l = match neo_ajtai::AjtaiSModule::from_global_for_dims(d, m_step) {
        Ok(l) => l,
        Err(_) => {
            #[cfg(not(feature = "testing"))]
            {
                return Err(anyhow::anyhow!(
                    "Ajtai PP missing for dims (D={}, m={}); register CRS/PP before proving base case",
                    d, m_step
                ));
            }
            #[cfg(feature = "testing")]
            {
                let kappa_guess = neo_ajtai::get_global_pp().map(|pp| pp.kappa).unwrap_or(16usize);
                super::super::super::ensure_ajtai_pp_for_dims(d, m_step, || {
                    use rand::{RngCore, SeedableRng};
                    use rand::rngs::StdRng;
                    let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
                        StdRng::from_seed([42u8; 32])
                    } else {
                        let mut seed = [0u8; 32];
                        rand::rng().fill_bytes(&mut seed);
                        StdRng::from_seed(seed)
                    };
                    let pp = crate::ajtai_setup(&mut rng, d, kappa_guess, m_step)?;
                    neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
                })?;
                neo_ajtai::AjtaiSModule::from_global_for_dims(d, m_step)
                    .map_err(|e| anyhow::anyhow!("AjtaiSModule unavailable for (d={}, m={}): {}", d, m_step, e))?
            }
        }
    };
    use neo_ccs::traits::SModuleHomomorphism;
    let c_zero = l.commit(&z_zero);

    Ok((
        neo_ccs::McsInstance { c: c_zero, x: x_zero, m_in },
        neo_ccs::McsWitness::<F> { w: w_zero, Z: z_zero },
    ))
}

