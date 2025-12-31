use crate::PiCcsError;
use neo_ccs::poly::SparsePoly;
use neo_ccs::{CcsStructure, Mat};
use neo_math::{F, K};
use neo_memory::ajtai::decode_vector as ajtai_decode_vector;
use neo_memory::cpu::{build_bus_layout_for_instances, BusLayout};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use std::collections::HashSet;

pub(crate) trait BusStepView<Cmt> {
    fn m_in(&self) -> usize;
    fn public_x(&self) -> &[F];
    fn lut_insts_len(&self) -> usize;
    fn mem_insts_len(&self) -> usize;
    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F>;
    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F>;
}

impl<Cmt, KK> BusStepView<Cmt> for StepWitnessBundle<Cmt, F, KK> {
    fn m_in(&self) -> usize {
        self.mcs.0.m_in
    }

    fn public_x(&self) -> &[F] {
        &self.mcs.0.x
    }

    fn lut_insts_len(&self) -> usize {
        self.lut_instances.len()
    }

    fn mem_insts_len(&self) -> usize {
        self.mem_instances.len()
    }

    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F> {
        &self.lut_instances[idx].0
    }

    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F> {
        &self.mem_instances[idx].0
    }
}

impl<Cmt, KK> BusStepView<Cmt> for StepInstanceBundle<Cmt, F, KK> {
    fn m_in(&self) -> usize {
        self.mcs_inst.m_in
    }

    fn public_x(&self) -> &[F] {
        &self.mcs_inst.x
    }

    fn lut_insts_len(&self) -> usize {
        self.lut_insts.len()
    }

    fn mem_insts_len(&self) -> usize {
        self.mem_insts.len()
    }

    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F> {
        &self.lut_insts[idx]
    }

    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F> {
        &self.mem_insts[idx]
    }
}

fn infer_chunk_size_from_steps<Cmt, S: BusStepView<Cmt>>(steps: &[S]) -> Result<usize, PiCcsError> {
    let mut max_steps = 0usize;
    for step in steps {
        for i in 0..step.lut_insts_len() {
            max_steps = max_steps.max(step.lut_inst(i).steps);
        }
        for i in 0..step.mem_insts_len() {
            max_steps = max_steps.max(step.mem_inst(i).steps);
        }
    }
    if max_steps == 0 {
        // No instances => no bus; chunk_size is irrelevant but must be >=1 for layout helpers.
        return Ok(1);
    }
    Ok(max_steps)
}

fn infer_bus_layout_for_steps<Cmt, S: BusStepView<Cmt>>(
    s: &CcsStructure<F>,
    steps: &[S],
) -> Result<BusLayout, PiCcsError> {
    if steps.is_empty() {
        return Err(PiCcsError::InvalidInput("no steps".into()));
    }
    let m_in = steps[0].m_in();
    for (i, step) in steps.iter().enumerate() {
        let cur_m_in = step.m_in();
        if cur_m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "m_in mismatch across steps (step 0 has {m_in}, step {i} has {cur_m_in})"
            )));
        }
    }

    let chunk_size = infer_chunk_size_from_steps(steps)?;

    let base_shout_ell_addrs: Vec<usize> = (0..steps[0].lut_insts_len())
        .map(|i| {
            let inst = steps[0].lut_inst(i);
            inst.d * inst.ell
        })
        .collect();
    let base_twist_ell_addrs: Vec<usize> = (0..steps[0].mem_insts_len())
        .map(|i| {
            let inst = steps[0].mem_inst(i);
            inst.d * inst.ell
        })
        .collect();

    for (i, step) in steps.iter().enumerate().skip(1) {
        let cur_shout: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| {
                let inst = step.lut_inst(j);
                inst.d * inst.ell
            })
            .collect();
        let cur_twist: Vec<usize> = (0..step.mem_insts_len())
            .map(|j| {
                let inst = step.mem_inst(j);
                inst.d * inst.ell
            })
            .collect();
        if cur_shout != base_shout_ell_addrs || cur_twist != base_twist_ell_addrs {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus layout mismatch across steps (step 0 vs step {i})"
            )));
        }
    }

    let layout = build_bus_layout_for_instances(
        s.m,
        m_in,
        chunk_size,
        base_shout_ell_addrs.iter().copied(),
        base_twist_ell_addrs.iter().copied(),
    )
    .map_err(PiCcsError::InvalidInput)?;

    if m_in
        .checked_add(layout.chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + chunk_size overflow".into()))?
        > s.n
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time rows out of range: m_in({m_in}) + chunk_size({}) > n({})",
            layout.chunk_size, s.n
        )));
    }

    Ok(layout)
}

pub(crate) fn prepare_ccs_for_shared_cpu_bus_steps<Cmt, S: BusStepView<Cmt>>(
    s0: &CcsStructure<F>,
    steps: &[S],
) -> Result<(CcsStructure<F>, BusLayout), PiCcsError> {
    let bus = infer_bus_layout_for_steps(s0, steps)?;
    let padding_rows = ensure_ccs_has_shared_bus_padding_for_steps(s0, &bus, steps)?;
    ensure_ccs_binds_shared_bus_for_steps(s0, &bus, &padding_rows)?;
    let s = extend_ccs_with_cpu_bus_copyouts(s0, &bus)?;
    Ok((s, bus))
}

pub(crate) fn extend_ccs_with_cpu_bus_copyouts(
    s: &CcsStructure<F>,
    bus: &BusLayout,
) -> Result<CcsStructure<F>, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(s.clone());
    }
    if s.n != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "shared-bus requires square CCS (n==m) for identity-first ME semantics, got {}×{}",
            s.n, s.m
        )));
    }

    let mut matrices = s.matrices.clone();
    let old_len = matrices.len();
    let f: SparsePoly<F> = s.f.append_zero_vars(bus.bus_cols);

    for col_id in 0..bus.bus_cols {
        let mut mat = Mat::zero(s.n, s.m, F::ZERO);
        for j in 0..bus.chunk_size {
            let row = bus.time_index(j);
            let col = bus.bus_cell(col_id, j);
            if row >= s.n || col >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "bus copy-out index out of range (row={row}, col={col}, n={}, m={})",
                    s.n, s.m
                )));
            }
            mat.set(row, col, F::ONE);
        }
        matrices.push(mat);
    }

    let out = CcsStructure::new(matrices, f)
        .map_err(|e| PiCcsError::InvalidInput(format!("invalid CCS after bus extension: {e:?}")))?;
    assert_eq!(
        out.matrices.len(),
        old_len + bus.bus_cols,
        "shared-bus copyouts: matrix count mismatch"
    );
    Ok(out)
}

fn active_matrix_indices(s: &CcsStructure<F>) -> Vec<usize> {
    let t = s.matrices.len();
    let mut active = vec![false; t];
    for term in s.f.terms() {
        for (j, &exp) in term.exps.iter().enumerate() {
            if exp != 0 {
                active[j] = true;
            }
        }
    }
    active
        .iter()
        .enumerate()
        .filter_map(|(j, &is_active)| is_active.then_some(j))
        .collect()
}

struct BusColLabel {
    col_id: usize,
    label: String,
}

fn required_bus_cols_for_layout(layout: &BusLayout) -> Vec<BusColLabel> {
    let mut out = Vec::<BusColLabel>::new();
    for (lut_idx, shout) in layout.shout_cols.iter().enumerate() {
        for (b, col_id) in shout.addr_bits.clone().enumerate() {
            out.push(BusColLabel {
                col_id,
                label: format!("shout[{lut_idx}].addr_bits[{b}]"),
            });
        }
        out.push(BusColLabel {
            col_id: shout.has_lookup,
            label: format!("shout[{lut_idx}].has_lookup"),
        });
        out.push(BusColLabel {
            col_id: shout.val,
            label: format!("shout[{lut_idx}].val"),
        });
    }

    for (mem_idx, twist) in layout.twist_cols.iter().enumerate() {
        for (b, col_id) in twist.ra_bits.clone().enumerate() {
            out.push(BusColLabel {
                col_id,
                label: format!("twist[{mem_idx}].ra_bits[{b}]"),
            });
        }
        for (b, col_id) in twist.wa_bits.clone().enumerate() {
            out.push(BusColLabel {
                col_id,
                label: format!("twist[{mem_idx}].wa_bits[{b}]"),
            });
        }
        out.push(BusColLabel {
            col_id: twist.has_read,
            label: format!("twist[{mem_idx}].has_read"),
        });
        out.push(BusColLabel {
            col_id: twist.has_write,
            label: format!("twist[{mem_idx}].has_write"),
        });
        out.push(BusColLabel {
            col_id: twist.wv,
            label: format!("twist[{mem_idx}].wv"),
        });
        out.push(BusColLabel {
            col_id: twist.rv,
            label: format!("twist[{mem_idx}].rv"),
        });
        out.push(BusColLabel {
            col_id: twist.inc,
            label: format!("twist[{mem_idx}].inc_at_write_addr"),
        });
    }

    out
}

fn required_bus_binding_cols_for_layout(layout: &BusLayout) -> Vec<BusColLabel> {
    let inc_cols: HashSet<usize> = layout.twist_cols.iter().map(|t| t.inc).collect();
    required_bus_cols_for_layout(layout)
        .into_iter()
        .filter(|c| !inc_cols.contains(&c.col_id))
        .collect()
}

fn ensure_ccs_references_bus_cols(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    required_cols: &[BusColLabel],
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // If the CCS polynomial does not depend on any matrix (e.g. f == 0), the CCS imposes
        // no constraints at all. In that case this check is not meaningful, so we skip it.
        return Ok(());
    }

    let mut missing: Vec<&BusColLabel> = Vec::new();
    for col in required_cols {
        if col.col_id >= bus.bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: required col_id {} out of range (bus_cols={})",
                col.col_id, bus.bus_cols
            )));
        }
        let z_idx = bus.bus_cell(col.col_id, 0);
        if z_idx >= s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                z_idx, s.m
            )));
        }

        let mut found = false;
        'active_mats: for &mj in &active {
            let mat = &s.matrices[mj];
            for r in 0..mat.rows() {
                if mat[(r, z_idx)] != F::ZERO {
                    found = true;
                    break 'active_mats;
                }
            }
        }
        if !found {
            missing.push(col);
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("col_id {} ({})", c.col_id, c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS does not reference required bus columns in any active constraint matrix.\n\
         This makes the bus a dead witness: CPU semantics can fork from Twist/Shout semantics.\n\
         Fix: make CPU semantics use the bus coordinates directly, or add equality constraints tying any shadow columns to the bus.\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

fn ensure_ccs_references_bus_cols_outside_padding_rows(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    padding_rows: &HashSet<usize>,
    required_cols: &[BusColLabel],
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // If the CCS polynomial does not depend on any matrix (e.g. f == 0), the CCS imposes
        // no constraints at all. In that case this check is not meaningful, so we skip it.
        return Ok(());
    }

    let mut is_padding_row = vec![false; s.n];
    for &r in padding_rows {
        if r < s.n {
            is_padding_row[r] = true;
        }
    }

    let mut missing: Vec<&BusColLabel> = Vec::new();
    for col in required_cols {
        if col.col_id >= bus.bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: required col_id {} out of range (bus_cols={})",
                col.col_id, bus.bus_cols
            )));
        }
        let z_idx = bus.bus_cell(col.col_id, 0);
        if z_idx >= s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                z_idx, s.m
            )));
        }

        let mut found = false;
        'active_mats: for &mj in &active {
            let mat = &s.matrices[mj];
            for r in 0..mat.rows() {
                if is_padding_row[r] {
                    continue;
                }
                if mat[(r, z_idx)] != F::ZERO {
                    found = true;
                    break 'active_mats;
                }
            }
        }
        if !found {
            missing.push(col);
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("col_id {} ({})", c.col_id, c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS never references core bus columns outside the canonical padding rows.\n\
         This is a common linkage footgun: padding constraints (1-has_*)*field=0 only restrict inactive fields, \
         and do not bind active bus semantics to CPU semantics.\n\
         Fix: inject binding constraints tying CPU semantics columns to the shared bus (recommended: \
         `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints` / `CpuConstraintBuilder`).\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

#[derive(Clone, Debug)]
struct BusPaddingLabel {
    flag_z_idx: usize,
    field_z_idx: usize,
    label: String,
}

fn required_bus_padding_for_layout(bus: &BusLayout) -> Vec<BusPaddingLabel> {
    let mut out = Vec::<BusPaddingLabel>::new();

    for (lut_idx, shout) in bus.shout_cols.iter().enumerate() {
        let has_lookup_z = bus.bus_cell(shout.has_lookup, 0);
        let val_z = bus.bus_cell(shout.val, 0);

        // (1 - has_lookup) * val = 0
        out.push(BusPaddingLabel {
            flag_z_idx: has_lookup_z,
            field_z_idx: val_z,
            label: format!("shout[{lut_idx}]: (1-has_lookup)*val"),
        });

        // (1 - has_lookup) * addr_bits[b] = 0
        for (b, col_id) in shout.addr_bits.clone().enumerate() {
            let bit_z = bus.bus_cell(col_id, 0);
            out.push(BusPaddingLabel {
                flag_z_idx: has_lookup_z,
                field_z_idx: bit_z,
                label: format!("shout[{lut_idx}]: (1-has_lookup)*addr_bits[{b}]"),
            });
        }
    }

    for (mem_idx, twist) in bus.twist_cols.iter().enumerate() {
        let has_read_z = bus.bus_cell(twist.has_read, 0);
        let has_write_z = bus.bus_cell(twist.has_write, 0);

        let wv_z = bus.bus_cell(twist.wv, 0);
        let rv_z = bus.bus_cell(twist.rv, 0);
        let inc_z = bus.bus_cell(twist.inc, 0);

        // (1 - has_read) * rv = 0
        out.push(BusPaddingLabel {
            flag_z_idx: has_read_z,
            field_z_idx: rv_z,
            label: format!("twist[{mem_idx}]: (1-has_read)*rv"),
        });

        // (1 - has_read) * ra_bits[b] = 0
        for (b, col_id) in twist.ra_bits.clone().enumerate() {
            let bit_z = bus.bus_cell(col_id, 0);
            out.push(BusPaddingLabel {
                flag_z_idx: has_read_z,
                field_z_idx: bit_z,
                label: format!("twist[{mem_idx}]: (1-has_read)*ra_bits[{b}]"),
            });
        }

        // (1 - has_write) * wv = 0
        out.push(BusPaddingLabel {
            flag_z_idx: has_write_z,
            field_z_idx: wv_z,
            label: format!("twist[{mem_idx}]: (1-has_write)*wv"),
        });

        // (1 - has_write) * inc_at_write_addr = 0
        out.push(BusPaddingLabel {
            flag_z_idx: has_write_z,
            field_z_idx: inc_z,
            label: format!("twist[{mem_idx}]: (1-has_write)*inc_at_write_addr"),
        });

        // (1 - has_write) * wa_bits[b] = 0
        for (b, col_id) in twist.wa_bits.clone().enumerate() {
            let bit_z = bus.bus_cell(col_id, 0);
            out.push(BusPaddingLabel {
                flag_z_idx: has_write_z,
                field_z_idx: bit_z,
                label: format!("twist[{mem_idx}]: (1-has_write)*wa_bits[{b}]"),
            });
        }
    }

    out
}

fn infer_public_constant_one_cols_from_steps<Cmt, S: BusStepView<Cmt>>(steps: &[S]) -> Vec<usize> {
    if steps.is_empty() {
        return Vec::new();
    }
    let m_in = steps[0].m_in();
    if m_in == 0 {
        return Vec::new();
    }

    let mut is_const_one = vec![true; m_in];
    for step in steps {
        let x = step.public_x();
        if x.len() != m_in {
            // Treat as no const-one. Other layers validate x length.
            return Vec::new();
        }
        for c in 0..m_in {
            if x[c] != F::ONE {
                is_const_one[c] = false;
            }
        }
    }
    is_const_one
        .iter()
        .enumerate()
        .filter_map(|(c, &ok)| ok.then_some(c))
        .collect()
}

fn row_is_all_zero(mat: &Mat<F>, row: usize) -> bool {
    for &v in mat.row(row) {
        if v != F::ZERO {
            return false;
        }
    }
    true
}

fn row_single_nonzero_col(mat: &Mat<F>, row: usize) -> Option<usize> {
    let mut found: Option<usize> = None;
    for (c, &v) in mat.row(row).iter().enumerate() {
        if v == F::ZERO {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(c);
    }
    found
}

fn row_padding_flag_col(mat: &Mat<F>, row: usize, const_one_cols: &[usize]) -> Option<usize> {
    let mut first: Option<(usize, F)> = None;
    let mut second: Option<(usize, F)> = None;

    for (c, &v) in mat.row(row).iter().enumerate() {
        if v == F::ZERO {
            continue;
        }
        if first.is_none() {
            first = Some((c, v));
            continue;
        }
        if second.is_none() {
            second = Some((c, v));
            continue;
        }
        // More than 2 non-zero entries -> not a simple (1-flag) form.
        return None;
    }

    let (c1, v1) = first?;
    let (c2, v2) = second?;

    // Look for the canonical form: a*(one - flag), up to scaling by nonzero a.
    // That means the two coefficients must be equal magnitude and opposite sign.
    if v1 != -v2 {
        return None;
    }

    if const_one_cols.contains(&c1) {
        return Some(c2);
    }
    if const_one_cols.contains(&c2) {
        return Some(c1);
    }
    None
}

fn ensure_ccs_has_bus_padding_constraints(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    const_one_cols: &[usize],
    required: &[BusPaddingLabel],
) -> Result<HashSet<usize>, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(HashSet::new());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // CCS has no active constraints (e.g., f == 0); skip guardrail to preserve plumbing tests.
        return Ok(HashSet::new());
    }

    if const_one_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus=true but no public constant-one column found to validate required padding constraints.\n\
             Fix: include a public input column that is always 1 (e.g., x[0]=1) and build padding constraints using it \
             (recommended: use `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints` or \
              `neo_memory::cpu::constraints::CpuConstraintBuilder`)."
                .into(),
        ));
    }

    // This validation is intentionally strict and recognizes the canonical R1CS embedding:
    // A(z) * B(z) - C(z) = 0, with padding rows using:
    //   A(z) = (1 - flag), B(z) = field, C(z) = 0.
    let (a_idx, b_idx, c_idx) = if s.matrices.len() >= 4 && s.matrices[0].is_identity() {
        (1usize, 2usize, 3usize)
    } else if s.matrices.len() >= 3 {
        (0usize, 1usize, 2usize)
    } else {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus=true but CPU CCS has fewer than 3 matrices; cannot validate padding constraints".into(),
        ));
    };

    let mut present: HashSet<(usize, usize)> = HashSet::new();
    let mut padding_rows: HashSet<usize> = HashSet::new();
    for row in 0..s.n {
        if !row_is_all_zero(&s.matrices[c_idx], row) {
            continue;
        }
        let Some(field_col) = row_single_nonzero_col(&s.matrices[b_idx], row) else {
            continue;
        };
        let Some(flag_col) = row_padding_flag_col(&s.matrices[a_idx], row, const_one_cols) else {
            continue;
        };
        present.insert((flag_col, field_col));
        padding_rows.insert(row);
    }

    let mut missing: Vec<&BusPaddingLabel> = Vec::new();
    for req in required {
        if !present.contains(&(req.flag_z_idx, req.field_z_idx)) {
            missing.push(req);
        }
    }

    if missing.is_empty() {
        return Ok(padding_rows);
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("{}", c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS is missing required padding constraints that force inactive bus fields to zero.\n\
         This is a common footgun: Twist/Shout gate checks by has_* flags, so unconstrained bus fields become arbitrary degrees of freedom.\n\
         Fix: inject the canonical shared-bus constraints (binding + padding) using `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints`, \
         or add constraints of the form (1 - has_*) * field = 0 for each gated bus field (recommended: use `neo_memory::cpu::constraints::CpuConstraintBuilder`).\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

fn ensure_ccs_binds_shared_bus_for_steps(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    padding_rows: &HashSet<usize>,
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }
    let required = required_bus_cols_for_layout(bus);
    ensure_ccs_references_bus_cols(s, bus, &required)?;

    let binding_required = required_bus_binding_cols_for_layout(bus);
    ensure_ccs_references_bus_cols_outside_padding_rows(s, bus, padding_rows, &binding_required)
}

fn ensure_ccs_has_shared_bus_padding_for_steps<Cmt, S: BusStepView<Cmt>>(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    steps: &[S],
) -> Result<HashSet<usize>, PiCcsError> {
    if steps.is_empty() || bus.bus_cols == 0 {
        return Ok(HashSet::new());
    }
    let const_one_cols = infer_public_constant_one_cols_from_steps(steps);
    let required = required_bus_padding_for_layout(bus);
    ensure_ccs_has_bus_padding_constraints(s, bus, &const_one_cols, &required)
}

pub(crate) fn decode_cpu_z_to_k(params: &NeoParams, Z: &Mat<F>) -> Vec<K> {
    ajtai_decode_vector(params, Z)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub(crate) fn build_time_sparse_from_bus_col(
    z: &[K],
    bus: &BusLayout,
    col_id: usize,
    steps_len: usize,
    pow2_cycle: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
    if col_id >= bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus col_id out of range: {col_id} >= {}",
            bus.bus_cols
        )));
    }
    if steps_len > bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "steps_len({steps_len}) > bus.chunk_size({})",
            bus.chunk_size
        )));
    }
    let mut entries: Vec<(usize, K)> = Vec::new();
    for j in 0..bus.chunk_size {
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        if j >= steps_len {
            continue;
        }
        let idx = bus.bus_cell(col_id, j);
        let v = z
            .get(idx)
            .copied()
            .ok_or_else(|| PiCcsError::InvalidInput(format!("CPU witness too short for bus idx={idx}")))?;
        if v != K::ZERO {
            entries.push((t, v));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}
