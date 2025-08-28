// neo-ccs/src/integration.rs
use crate::{legacy::CcsInstance, legacy::CcsStructure, legacy::CcsWitness};
use neo_math::{from_base, project_ext_to_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

/// Return type: ((A, B, C), x, w)
/// A, B, C are dense matrices laid out as Vec<rows>, each row is Vec<F> with width = 1 + nvars.
/// x = public inputs (base field)
/// w = witness variables (base field)
///
/// Assumptions (as used throughout the repo):
/// - structure.witness_size = total z length (|public_input| + |witness|), padded if needed.
/// - z = [public_input..., witness...], where each entry is embedded in ExtF with imag = 0.
pub fn convert_ccs_for_spartan2(
    structure: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
) -> Result<((Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>), Vec<F>, Vec<F>), String> {
    let m = structure.num_constraints;
    let n = structure.witness_size;
    let s = structure.mats.len();

    // 1) Recover affine coefficients of f(y) = sum_j alpha_j * y_j + c
    let mut alpha: Vec<F> = Vec::with_capacity(s);
    for j in 0..s {
        let mut basis = vec![ExtF::ZERO; s];
        basis[j] = ExtF::ONE;
        let val = structure.f.evaluate(&basis);
        let base = project_ext_to_base(val)
            .ok_or_else(|| "f(e_j) not in base field; only affine/base-valued f supported".to_string())?;
        alpha.push(base);
    }
    let c = {
        let val0 = structure.f.evaluate(&vec![ExtF::ZERO; s]);
        project_ext_to_base(val0)
            .ok_or_else(|| "f(0) not in base field; only affine/base-valued f supported".to_string())?
    };

    // Sanity: ensure "affine, not multilinear" — if the CCS used a non-affine f,
    // the converter must error out (we'll add full multilinear arithmetization later).
    if structure.f.max_individual_degree() > 1 {
        return Err("CCS→R1CS converter (v1) supports only affine f (max degree per var ≤ 1 AND no cross terms)".to_string());
    }

    // Reject mixed terms: f must be affine in its inputs (no X_i * X_j).
    for i in 0..s {
        for j in (i+1)..s {
            let mut eij = vec![ExtF::ZERO; s];
            eij[i] = ExtF::ONE;
            eij[j] = ExtF::ONE;
            let lhs = structure.f.evaluate(&eij);
            let lhs0 = project_ext_to_base(lhs)
                .ok_or_else(|| "f(e_i + e_j) not in base field".to_string())?;
            let rhs = alpha[i] + alpha[j] + c;
            if lhs0 != rhs {
                return Err("CCS→R1CS (v1) only supports affine f(y)=c+Σα_j y_j".to_string());
            }
        }
    }

    // 2) Build linear form coefficients L_b over z for each row b:
    //    L_b[k] = sum_j alpha[j] * M_j[b,k], and constant term c goes into column 0.
    // Dimensions for R1CS matrices: rows = m, cols = 1 + n (col 0 is the "1" wire).
    let cols = 1 + n;
    #[allow(non_snake_case)]
    let mut A = vec![vec![F::ZERO; cols]; m];
    #[allow(non_snake_case)]
    let mut B = vec![vec![F::ZERO; cols]; m];
    #[allow(non_snake_case)]
    let C = vec![vec![F::ZERO; cols]; m];

    for b in 0..m {
        // Constant term
        A[b][0] = c;

        // Variable columns (z_0..z_{n-1})
        for k in 0..n {
            let mut coeff = F::ZERO;
            for j in 0..s {
                // structure.mats[j] stores ExtF entries, project to base field
                let m_j_bk_ext = structure.mats[j].get(b, k).unwrap_or(ExtF::ZERO);
                let m_j_bk = project_ext_to_base(m_j_bk_ext)
                    .ok_or_else(|| format!("Matrix entry M_{}[{},{}] not in base field", j, b, k))?;
                coeff += alpha[j] * m_j_bk;
            }
            A[b][1 + k] = coeff;
        }

        // B selects the constant "1"
        B[b][0] = F::ONE;

        // C stays zero → enforces L_b(v) * 1 = 0
    }

    // 3) Build (x, w) split.
    let mut z_ext: Vec<ExtF> = instance
        .public_input
        .iter()
        .map(|&x| from_base(x))
        .collect();
    z_ext.extend_from_slice(&witness.z);
    if z_ext.len() > n {
        return Err(format!(
            "Instance+Witness length {} exceeds structure.witness_size {}",
            z_ext.len(),
            n
        ));
    }
    z_ext.resize(n, ExtF::ZERO);

    // Convert to base field and check imag = 0
    let mut z_base: Vec<F> = Vec::with_capacity(n);
    for (i, &zi) in z_ext.iter().enumerate() {
        let base = project_ext_to_base(zi).ok_or_else(|| {
            format!(
                "z[{}] not in base field (imag ≠ 0) — CCS→R1CS v1 requires base-valued z",
                i
            )
        })?;
        z_base.push(base);
    }

    let x = instance.public_input.clone();
    let w = z_base[instance.public_input.len()..].to_vec();

    Ok(((A, B, C), x, w))
}
