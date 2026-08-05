//! Claim validation, allocation, and canonical bootstrap constraints.
//!
//! Owns: claim shape checks, wire allocation, canonical padding, and claim
//! digest inputs.
//!
//! Does not own: SumCheck, transcript order, or the terminal equation.
//!
//! Emits constraints: canonical claim fields, padding, and digest binding.
//!
//! | Input | Constraint family |
//! | --- | --- |
//! | fresh claim | whole-ring public input and canonical commitment |
//! | running/output claim | point, ring value, and padding canonicality |

use super::*;
use p3_field::PrimeField64;

pub(super) fn validate_fresh_shape(cfg: &PiCcsVerifierConfig<'_>, idx: usize, fresh: &CcsClaim) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    if fresh.m_in > cfg.structure.m() {
        return Err(Error::Shape(format!(
            "fresh[{idx}].m_in ({}) > structure.m ({})",
            fresh.m_in,
            cfg.structure.m()
        )));
    }
    if fresh.x.len() != fresh.m_in {
        return Err(Error::Shape(format!(
            "fresh[{idx}].x.len ({}) != m_in ({})",
            fresh.x.len(),
            fresh.m_in
        )));
    }
    if fresh.c.d != D {
        return Err(Error::Shape(format!("fresh[{idx}].c.d ({}) != D ({D})", fresh.c.d)));
    }
    if fresh.c.kappa != kappa {
        return Err(Error::Shape(format!(
            "fresh[{idx}].c.kappa ({}) != params.kappa ({kappa})",
            fresh.c.kappa
        )));
    }
    if fresh.c.data.len() != D * kappa {
        return Err(Error::Shape(format!(
            "fresh[{idx}].c.data.len ({}) != D*kappa ({})",
            fresh.c.data.len(),
            D * kappa
        )));
    }
    validate_adv_shape(fresh.adv.as_ref(), D, kappa, &format!("fresh[{idx}]")).map_err(Error::Shape)
}

pub(super) fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
    let [c0, c1] = value.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

pub(super) fn alloc_k_vec(builder: &mut R1csBuilder, values: &[K]) -> Vec<KVar> {
    values
        .iter()
        .copied()
        .map(|value| alloc_k(builder, value))
        .collect()
}

fn alloc_k_rows(builder: &mut R1csBuilder, rows: &[Vec<K>]) -> Vec<Vec<KVar>> {
    rows.iter().map(|row| alloc_k_vec(builder, row)).collect()
}

fn digest32_witness_fields(builder: &mut R1csBuilder, bytes: &[u8]) -> Result<[Var; 4], Error> {
    let fields = header_digest_bytes_to_fields(bytes)?;
    Ok(std::array::from_fn(|index| builder.alloc(fields[index])))
}

fn header_digest_bytes_to_fields(bytes: &[u8]) -> Result<[F; 4], Error> {
    if bytes.len() != 32 {
        return Err(Error::Shape(format!(
            "PiCCS fold digest must be 32 bytes, got {}",
            bytes.len()
        )));
    }
    let mut fields = [F::ZERO; 4];
    for (index, field) in fields.iter_mut().enumerate() {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&bytes[index * 8..(index + 1) * 8]);
        let value = u64::from_le_bytes(limb);
        if value >= F::ORDER_U64 {
            return Err(Error::Shape(format!("PiCCS fold digest limb {index} is not canonical")));
        }
        *field = F::from_u64(value);
    }
    Ok(fields)
}

pub(super) fn alloc_fresh_wires(builder: &mut R1csBuilder, fresh: &CcsClaim) -> CcsClaimWires {
    CcsClaimWires {
        c_d: fresh.c.d,
        c_d_var: alloc_usize(builder, fresh.c.d),
        c_kappa: fresh.c.kappa,
        c_kappa_var: alloc_usize(builder, fresh.c.kappa),
        c_data: builder.alloc_vec(&fresh.c.data),
        adv: alloc_adv(builder, fresh.adv.as_ref()),
        x: builder.alloc_vec(&fresh.x),
        m_in: fresh.m_in,
        m_in_var: alloc_usize(builder, fresh.m_in),
    }
}

pub(super) fn alloc_ce_wires(builder: &mut R1csBuilder, claim: &CeClaim) -> Result<CeClaimWires, Error> {
    let mut x = Vec::with_capacity(claim.X.rows() * claim.X.cols());
    let active_cols = crate::paper::relations::superneo_public_x_cols(claim.m_in);
    let inactive_nonzero =
        (0..claim.X.rows()).any(|row| (active_cols..claim.X.cols()).any(|column| claim.X[(row, column)] != F::ZERO));
    let inactive_zero = builder.alloc(if inactive_nonzero { F::ONE } else { F::ZERO });
    builder.enforce_eq(&Lc::from_var(inactive_zero), &Lc::zero());
    for row in 0..claim.X.rows() {
        for column in 0..claim.X.cols() {
            x.push(if column < active_cols {
                builder.alloc(claim.X[(row, column)])
            } else {
                inactive_zero
            });
        }
    }
    Ok(CeClaimWires {
        c_d: claim.c.d,
        c_d_var: alloc_usize(builder, claim.c.d),
        c_kappa: claim.c.kappa,
        c_kappa_var: alloc_usize(builder, claim.c.kappa),
        c_data: builder.alloc_vec(&claim.c.data),
        adv: alloc_adv(builder, claim.adv.as_ref()),
        x,
        x_rows: claim.X.rows(),
        x_rows_var: alloc_usize(builder, claim.X.rows()),
        x_cols: claim.X.cols(),
        x_cols_var: alloc_usize(builder, claim.X.cols()),
        r: alloc_k_vec(builder, &claim.r),
        y_ring: alloc_k_rows(builder, &claim.y_ring),
        ct: alloc_k_vec(builder, &claim.ct),
        m_in: claim.m_in,
        m_in_var: alloc_usize(builder, claim.m_in),
        fold_digest_fields: digest32_witness_fields(builder, &claim.fold_digest)?,
    })
}
