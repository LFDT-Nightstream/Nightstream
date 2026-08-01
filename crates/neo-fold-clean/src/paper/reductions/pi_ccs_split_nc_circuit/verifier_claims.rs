//! Claim validation, allocation, and canonical bootstrap constraints.

use super::*;

pub(super) fn validate_fresh_shape(cfg: &SplitNcPiCcsVConfig<'_>, idx: usize, fresh: &CcsClaim) -> Result<(), Error> {
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

pub(super) fn validate_running_parent_authority_shape(
    cfg: &SplitNcPiCcsVConfig<'_>,
    claim: &CeClaim,
) -> Result<(), Error> {
    validate_ce_shape_without_y_zcol(cfg, "running_parent_authority", claim)
}

pub(super) fn validate_canonical_bootstrap_shape(msg: &SplitNcPiCcsVMessages<'_>) -> Result<(), Error> {
    let fresh = msg
        .fresh
        .first()
        .ok_or_else(|| Error::Shape("canonical bootstrap requires one fresh claim".into()))?;
    let m_in = fresh.m_in;
    let has_lane_commitments = fresh.adv.is_some();
    let parent = msg
        .running_parent_authority
        .ok_or_else(|| Error::Shape("canonical bootstrap requires its zero parent".into()))?;
    if parent.m_in != m_in || parent.adv.is_some() != has_lane_commitments {
        return Err(Error::Shape(
            "canonical bootstrap parent has the wrong public width or lane commitments".into(),
        ));
    }
    for claim in msg.running {
        if claim.m_in != m_in
            || claim.adv.is_some() != has_lane_commitments
            || claim.y_zcol.iter().any(|value| *value != K::ZERO)
        {
            return Err(Error::Shape(
                "canonical bootstrap child has noncanonical public structure".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_ce_shape_without_y_zcol(
    cfg: &SplitNcPiCcsVConfig<'_>,
    label: &str,
    claim: &CeClaim,
) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    let d_pad = 1usize << cfg.ell_d;
    validate_ce_commitment_shape(cfg, label, claim, kappa)?;
    if claim.X.rows() != D || claim.X.cols() != claim.m_in {
        return Err(Error::Shape(format!(
            "{label}.X shape ({}×{}) != ({D}×{})",
            claim.X.rows(),
            claim.X.cols(),
            claim.m_in
        )));
    }
    validate_ce_common_shape(cfg, label, claim, d_pad)
}

pub(super) fn validate_output_ce_shape(
    cfg: &SplitNcPiCcsVConfig<'_>,
    label: &str,
    claim: &CeClaim,
) -> Result<(), Error> {
    let kappa = cfg.params.kappa() as usize;
    let d_pad = 1usize << cfg.ell_d;
    validate_ce_commitment_shape(cfg, label, claim, kappa)?;
    if claim.X.rows() != D {
        return Err(Error::Shape(format!("{label}.X.rows ({}) != D ({D})", claim.X.rows())));
    }
    validate_ce_common_shape(cfg, label, claim, d_pad)?;
    validate_y_zcol_shape(label, claim, d_pad)
}

fn validate_ce_commitment_shape(
    cfg: &SplitNcPiCcsVConfig<'_>,
    label: &str,
    claim: &CeClaim,
    kappa: usize,
) -> Result<(), Error> {
    if claim.c.d != D {
        return Err(Error::Shape(format!("{label}.c.d ({}) != D ({D})", claim.c.d)));
    }
    if claim.c.kappa != kappa {
        return Err(Error::Shape(format!(
            "{label}.c.kappa ({}) != params.kappa ({kappa})",
            claim.c.kappa
        )));
    }
    if claim.c.data.len() != D * kappa {
        return Err(Error::Shape(format!(
            "{label}.c.data.len ({}) != D*kappa ({})",
            claim.c.data.len(),
            D * kappa
        )));
    }
    validate_adv_shape(claim.adv.as_ref(), D, kappa, label).map_err(Error::Shape)?;
    if claim.m_in > cfg.structure.m() {
        return Err(Error::Shape(format!(
            "{label}.m_in ({}) > structure.m ({})",
            claim.m_in,
            cfg.structure.m()
        )));
    }
    Ok(())
}

fn validate_ce_common_shape(
    cfg: &SplitNcPiCcsVConfig<'_>,
    label: &str,
    claim: &CeClaim,
    d_pad: usize,
) -> Result<(), Error> {
    if claim.r.len() != cfg.ell_n {
        return Err(Error::Shape(format!(
            "{label}.r.len ({}) != ell_n ({})",
            claim.r.len(),
            cfg.ell_n
        )));
    }
    let expected_s_col = nc_column_variables(cfg);
    if claim.s_col.len() != expected_s_col {
        return Err(Error::Shape(format!(
            "{label}.s_col.len ({}) != selected NC column variables ({expected_s_col})",
            claim.s_col.len()
        )));
    }
    if claim.y_ring.len() != cfg.structure.t() {
        return Err(Error::Shape(format!(
            "{label}.y_ring.len ({}) != structure.t ({})",
            claim.y_ring.len(),
            cfg.structure.t()
        )));
    }
    if claim.ct.len() != cfg.structure.t() {
        return Err(Error::Shape(format!(
            "{label}.ct.len ({}) != structure.t ({})",
            claim.ct.len(),
            cfg.structure.t()
        )));
    }
    for (index, row) in claim.y_ring.iter().enumerate() {
        if row.len() != d_pad {
            return Err(Error::Shape(format!(
                "{label}.y_ring[{index}].len ({}) != d_pad ({d_pad})",
                row.len()
            )));
        }
    }
    validate_ce_sidecars(label, claim)
}

fn validate_y_zcol_shape(label: &str, claim: &CeClaim, d_pad: usize) -> Result<(), Error> {
    if claim.y_zcol.len() != d_pad {
        return Err(Error::Shape(format!(
            "{label}.y_zcol.len ({}) != d_pad ({d_pad})",
            claim.y_zcol.len()
        )));
    }
    Ok(())
}

fn validate_ce_sidecars(label: &str, claim: &CeClaim) -> Result<(), Error> {
    if !claim.aux_openings.is_empty() {
        return Err(Error::Shape(format!(
            "{label}.aux_openings.len ({}) != 0 for clean SplitNc circuit",
            claim.aux_openings.len()
        )));
    }
    if !claim.c_step_coords.is_empty() || claim.u_offset != 0 || claim.u_len != 0 {
        return Err(Error::Shape(format!(
            "{label} carries unsupported Pattern-A fields (c_step_coords.len={}, u_offset={}, u_len={})",
            claim.c_step_coords.len(),
            claim.u_offset,
            claim.u_len
        )));
    }
    Ok(())
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
    alloc_ce_wires_from_y_zcol(builder, claim, &claim.y_zcol)
}

pub(super) fn alloc_ce_wires_without_y_zcol(builder: &mut R1csBuilder, claim: &CeClaim) -> Result<CeClaimWires, Error> {
    let d_pad = D.next_power_of_two();
    if claim.y_zcol.len() != d_pad || claim.y_zcol.iter().skip(D).any(|value| *value != K::ZERO) {
        return Err(Error::Shape(
            "running CE y_zcol must have the padded ring shape with zero padding".into(),
        ));
    }
    alloc_ce_wires_from_y_zcol(builder, claim, &[])
}

pub(super) fn alloc_ce_wires_with_canonical_y_zcol(
    builder: &mut R1csBuilder,
    claim: &CeClaim,
    lanes: usize,
) -> Result<CeClaimWires, Error> {
    let canonical: Vec<K> = (0..lanes)
        .map(|lane| claim.y_zcol.get(lane).copied().unwrap_or(K::ZERO))
        .collect();
    alloc_ce_wires_from_y_zcol(builder, claim, &canonical)
}

fn alloc_ce_wires_from_y_zcol(builder: &mut R1csBuilder, claim: &CeClaim, y_zcol: &[K]) -> Result<CeClaimWires, Error> {
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
        s_col: alloc_k_vec(builder, &claim.s_col),
        y_ring: alloc_k_rows(builder, &claim.y_ring),
        ct: alloc_k_vec(builder, &claim.ct),
        y_zcol: alloc_k_vec(builder, y_zcol),
        m_in: claim.m_in,
        m_in_var: alloc_usize(builder, claim.m_in),
        fold_digest_fields: digest32_witness_fields(builder, &claim.fold_digest)?,
    })
}

pub(super) fn enforce_unique_zero_wires(builder: &mut R1csBuilder, wires: impl Iterator<Item = Var>) {
    let mut constrained = std::collections::HashSet::new();
    for wire in wires {
        if constrained.insert(wire.col()) {
            builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
        }
    }
}

pub(super) fn enforce_canonical_bootstrap_zero(
    builder: &mut R1csBuilder,
    running: &[CeClaimWires],
    parent: &CeClaimWires,
) -> Result<(), Error> {
    for claim in running.iter().chain(std::iter::once(parent)) {
        let adv = claim.adv.iter().flat_map(|adv| {
            adv.ops
                .data
                .iter()
                .chain(&adv.is.data)
                .chain(&adv.fs.data)
                .copied()
        });
        enforce_unique_zero_wires(
            builder,
            claim
                .c_data
                .iter()
                .copied()
                .chain(adv)
                .chain(claim.x.iter().copied())
                .chain(claim.r.iter().flat_map(|value| [value.c0, value.c1]))
                .chain(claim.s_col.iter().flat_map(|value| [value.c0, value.c1]))
                .chain(
                    claim
                        .y_ring
                        .iter()
                        .flatten()
                        .flat_map(|value| [value.c0, value.c1]),
                )
                .chain(claim.ct.iter().flat_map(|value| [value.c0, value.c1]))
                .chain(claim.y_zcol.iter().flat_map(|value| [value.c0, value.c1]))
                .chain(claim.fold_digest_fields),
        );
    }
    Ok(())
}
