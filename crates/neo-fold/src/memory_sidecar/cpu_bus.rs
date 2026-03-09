use crate::PiCcsError;
use neo_ccs::{CcsMatrix, CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_memory::cpu::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, BusLayout, ShoutInstanceShape,
};
use neo_memory::riscv::lookups::{PROG_ID, REG_ID};
use neo_memory::riscv::trace::{
    riscv_is_decode_lookup_table_id, riscv_trace_is_width_lookup_table_id, riscv_trace_lookup_n_vals_for_table_id,
    rv32_trace_cpu_cols, rv64_trace_cpu_cols,
};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use std::collections::{BTreeMap, HashMap, HashSet};

pub(crate) trait BusStepView<Cmt> {
    fn m_in(&self) -> usize;
    fn public_x(&self) -> &[F];
    fn lut_insts_len(&self) -> usize;
    fn mem_insts_len(&self) -> usize;
    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F>;
    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F>;
    fn time_t(&self) -> usize {
        0
    }
    fn time_cpu_cols_len(&self) -> usize {
        0
    }
    fn time_mem_cols_len(&self) -> usize {
        0
    }
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

    fn time_t(&self) -> usize {
        self.time_columns.t
    }

    fn time_cpu_cols_len(&self) -> usize {
        self.time_columns.cpu_cols.len()
    }

    fn time_mem_cols_len(&self) -> usize {
        self.time_columns.mem_cols.len()
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

    fn time_t(&self) -> usize {
        self.time_columns.t
    }

    fn time_cpu_cols_len(&self) -> usize {
        self.time_columns.cpu_cols.len()
    }

    fn time_mem_cols_len(&self) -> usize {
        self.time_columns.mem_cols.len()
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
    let base_shout_lanes: Vec<usize> = (0..steps[0].lut_insts_len())
        .map(|i| {
            let inst = steps[0].lut_inst(i);
            inst.lanes
        })
        .collect();
    let base_shout_n_vals: Vec<usize> = (0..steps[0].lut_insts_len())
        .map(|i| riscv_trace_lookup_n_vals_for_table_id(steps[0].lut_inst(i).table_id))
        .collect();
    let base_shout_addr_groups: Vec<Option<u64>> = (0..steps[0].lut_insts_len())
        .map(|i| steps[0].lut_inst(i).addr_group)
        .collect();
    let base_shout_selector_groups: Vec<Option<u64>> = (0..steps[0].lut_insts_len())
        .map(|i| steps[0].lut_inst(i).selector_group)
        .collect();
    let base_twist_ell_addrs: Vec<usize> = (0..steps[0].mem_insts_len())
        .map(|i| {
            let inst = steps[0].mem_inst(i);
            inst.d * inst.ell
        })
        .collect();
    let base_twist_lanes: Vec<usize> = (0..steps[0].mem_insts_len())
        .map(|i| {
            let inst = steps[0].mem_inst(i);
            inst.lanes
        })
        .collect();

    for (i, step) in steps.iter().enumerate().skip(1) {
        let cur_shout: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| {
                let inst = step.lut_inst(j);
                inst.d * inst.ell
            })
            .collect();
        let cur_shout_lanes: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| {
                let inst = step.lut_inst(j);
                inst.lanes
            })
            .collect();
        let cur_shout_n_vals: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| riscv_trace_lookup_n_vals_for_table_id(step.lut_inst(j).table_id))
            .collect();
        let cur_shout_addr_groups: Vec<Option<u64>> = (0..step.lut_insts_len())
            .map(|j| step.lut_inst(j).addr_group)
            .collect();
        let cur_shout_selector_groups: Vec<Option<u64>> = (0..step.lut_insts_len())
            .map(|j| step.lut_inst(j).selector_group)
            .collect();
        let cur_twist: Vec<usize> = (0..step.mem_insts_len())
            .map(|j| {
                let inst = step.mem_inst(j);
                inst.d * inst.ell
            })
            .collect();
        let cur_twist_lanes: Vec<usize> = (0..step.mem_insts_len())
            .map(|j| {
                let inst = step.mem_inst(j);
                inst.lanes
            })
            .collect();
        if cur_shout != base_shout_ell_addrs
            || cur_shout_lanes != base_shout_lanes
            || cur_shout_n_vals != base_shout_n_vals
            || cur_shout_addr_groups != base_shout_addr_groups
            || cur_shout_selector_groups != base_shout_selector_groups
            || cur_twist != base_twist_ell_addrs
            || cur_twist_lanes != base_twist_lanes
        {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus layout mismatch across steps (step 0 vs step {i})"
            )));
        }
    }

    let shout_shapes = base_shout_ell_addrs
        .iter()
        .copied()
        .zip(base_shout_lanes.iter().copied())
        .zip(base_shout_n_vals.iter().copied())
        .zip(base_shout_addr_groups.iter().copied())
        .zip(base_shout_selector_groups.iter().copied())
        .map(
            |((((ell_addr, lanes), n_vals), addr_group), selector_group)| ShoutInstanceShape {
                ell_addr,
                lanes,
                n_vals,
                addr_group,
                selector_group,
            },
        );
    let twist_ell_addrs_and_lanes = base_twist_ell_addrs
        .iter()
        .copied()
        .zip(base_twist_lanes.iter().copied())
        .map(|(ell_addr, lanes)| (ell_addr, lanes));
    let uniform_width_matches_ccs = m_in
        .checked_add(steps[0].time_cpu_cols_len())
        .and_then(|v| v.checked_add(steps[0].time_mem_cols_len()))
        .map_or(false, |uniform_m| uniform_m == s.m);
    let use_time_columns = uniform_width_matches_ccs
        && steps[0].time_t() == chunk_size
        && steps[0].time_mem_cols_len() > 0
        && steps.iter().all(|step| {
            step.time_t() == chunk_size
                && step.time_mem_cols_len() == steps[0].time_mem_cols_len()
                && step.time_cpu_cols_len() == steps[0].time_cpu_cols_len()
        });
    let m_virtual = if use_time_columns {
        let cpu_region = steps[0]
            .time_cpu_cols_len()
            .checked_mul(chunk_size)
            .ok_or_else(|| PiCcsError::InvalidInput("time cpu_cols*chunk_size overflow".into()))?;
        let mem_region = steps[0]
            .time_mem_cols_len()
            .checked_mul(chunk_size)
            .ok_or_else(|| PiCcsError::InvalidInput("time mem_cols*chunk_size overflow".into()))?;
        m_in.checked_add(cpu_region)
            .and_then(|v| v.checked_add(mem_region))
            .ok_or_else(|| PiCcsError::InvalidInput("time virtual m overflow".into()))?
    } else {
        s.m
    };

    let layout = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_virtual,
        m_in,
        chunk_size,
        shout_shapes,
        twist_ell_addrs_and_lanes,
    )
    .map_err(PiCcsError::InvalidInput)?;

    if use_time_columns && layout.bus_cols != steps[0].time_mem_cols_len() {
        return Err(PiCcsError::InvalidInput(format!(
            "time mem column count mismatch: layout.bus_cols={} vs time_columns.mem_cols={}",
            layout.bus_cols,
            steps[0].time_mem_cols_len()
        )));
    }

    // If there are no bus columns (no Twist/Shout instances), Route A doesn't use the bus time rows.
    // Allow small CCS instances (including m_in == n) in this case.
    if layout.bus_cols == 0 {
        return Ok(layout);
    }

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

pub(crate) fn prepare_ccs_for_shared_cpu_bus_steps<'a, Cmt, S: BusStepView<Cmt>>(
    s0: &'a CcsStructure<F>,
    steps: &[S],
) -> Result<(&'a CcsStructure<F>, BusLayout), PiCcsError> {
    let bus = infer_bus_layout_for_steps(s0, steps)?;
    let m_in = steps.first().map(|s| s.m_in()).unwrap_or(0usize);
    let uniform_width_matches_ccs = steps.first().and_then(|step0| {
        m_in.checked_add(step0.time_cpu_cols_len())
            .and_then(|v| v.checked_add(step0.time_mem_cols_len()))
    }) == Some(s0.m);
    let has_physical_bus_refs = ccs_references_any_bus_cols(s0, &bus);
    let using_time_columns = !steps.is_empty()
        && steps[0].time_t() == bus.chunk_size
        && steps[0].time_mem_cols_len() > 0
        && steps.iter().all(|step| {
            step.time_t() == bus.chunk_size
                && step.time_mem_cols_len() == steps[0].time_mem_cols_len()
                && step.time_cpu_cols_len() == steps[0].time_cpu_cols_len()
        });
    let rv32_trace_cpu_cols = rv32_trace_cpu_cols();
    let rv64_trace_cpu_cols = rv64_trace_cpu_cols();
    let canonical_trace_mode = !steps.is_empty()
        && m_in == 5
        && matches!(steps[0].time_cpu_cols_len(), n if n == rv32_trace_cpu_cols || n == rv64_trace_cpu_cols);
    let route_a_uniform_mode = using_time_columns && uniform_width_matches_ccs && canonical_trace_mode;
    if route_a_uniform_mode && !has_physical_bus_refs {
        // Only the canonical RV32/RV64 trace-wiring paths are allowed to satisfy shared-bus
        // linkage purely through committed time columns. Ad hoc test bundles may also synthesize
        // `time_columns`, but those columns are not authoritative and must still satisfy the
        // physical-tail guardrails below.
        return Ok((s0, bus));
    }
    let padding_rows = ensure_ccs_has_shared_bus_padding_for_steps(s0, &bus, steps)?;
    ensure_ccs_binds_shared_bus_for_steps(s0, &bus, &padding_rows, steps)?;
    // Performance: do NOT materialize bus copyout matrices into the CCS. Instead, we append the
    // corresponding ME openings directly from the witness (see `append_bus_openings_to_me_*`).
    Ok((s0, bus))
}

#[inline]
fn chi_for_row_index(r: &[K], idx: usize) -> K {
    // Multilinear basis polynomial χ_r(idx) where idx is interpreted in little-endian bits:
    // χ_r(idx) = ∏_i (idx_i ? r_i : (1 - r_i)).
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

#[inline]
fn precompute_contiguous_time_weights(
    r: &[K],
    start_row: usize,
    len: usize,
    n_pad: usize,
) -> Result<Vec<K>, PiCcsError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let end_row = start_row
        .checked_add(len)
        .ok_or_else(|| PiCcsError::InvalidInput("time-weight range overflow".into()))?;
    if end_row > n_pad {
        return Err(PiCcsError::InvalidInput(format!(
            "time-weight range out of bounds (start_row={}, len={}, n_pad={})",
            start_row, len, n_pad
        )));
    }

    // For large contiguous windows, build χ_r over the full boolean domain once and slice.
    // This avoids repeated per-index basis recomputation in hot Route-A paths.
    const FULL_CHI_MAX_PAD: usize = 1 << 20;
    let naive_ops = len.saturating_mul(r.len().max(1));
    let use_full_table = len >= 1024 && n_pad <= FULL_CHI_MAX_PAD && naive_ops >= n_pad;

    if use_full_table {
        let mut chi = Vec::with_capacity(n_pad);
        chi.push(K::ONE);
        for &ri in r {
            let one_minus_ri = K::ONE - ri;
            let cur_len = chi.len();
            chi.reserve(cur_len);
            for i in 0..cur_len {
                let v = chi[i];
                chi[i] = v * one_minus_ri;
                chi.push(v * ri);
            }
        }
        return Ok(chi[start_row..start_row + len].to_vec());
    }

    let mut out = Vec::with_capacity(len);
    for off in 0..len {
        out.push(chi_for_row_index(r, start_row + off));
    }
    Ok(out)
}

pub(crate) fn point_covers_bus_time_rows(bus: &BusLayout, point: &[K]) -> Result<bool, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(true);
    }
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    let start_row = bus.time_index(0);
    let end_row = start_row
        .checked_add(bus.chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("bus time row range overflow".into()))?;
    Ok(end_row <= n_pad)
}

pub(crate) fn append_bus_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    bus: &BusLayout,
    core_t: usize,
    Z: &Mat<F>,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let want_len = core_t
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
    let y_pad = (params.d as usize).next_power_of_two();

    let expected_cols = bus.m.div_ceil(neo_math::D);
    if Z.cols() != expected_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require packed witness width (Z.cols()={}, expected ceil(bus.m/D)={} for bus.m={})",
            Z.cols(),
            expected_cols,
            bus.m
        )));
    }

    // Idempotent append: allow callers to call this once; reject unexpected shapes.
    if me.y_ring.len() >= want_len && me.ct.len() >= want_len && me.y_ring.len() == me.ct.len() {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }

    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if me.m_in != bus.m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require ME.m_in==bus.m_in (got {}, want {})",
            me.m_in, bus.m_in
        )));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    let start_row = bus.time_index(0);
    for j in 0..bus.chunk_size {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time_index({j})={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
        if row != start_row + j {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time_index must be contiguous: time_index({j})={row}, expected {}",
                start_row + j
            )));
        }
    }
    if start_row
        .checked_add(bus.chunk_size)
        .map_or(true, |end| end > n_pad)
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time row range out of bounds: start_row={} chunk_size={} n_pad={}",
            start_row, bus.chunk_size, n_pad
        )));
    }
    if core_t < me.ct.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require core_t >= ME core ct len (core_t={}, ct.len()={})",
            core_t,
            me.ct.len(),
        )));
    }
    let aux_base = core_t
        .checked_sub(me.ct.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t underflow for bus aux base".into()))?;

    // Idempotent append: allow callers to call this once; reject unexpected aux lengths.
    let want_aux_len = aux_base
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("aux_base + bus_cols overflow".into()))?;
    if me.aux_openings.len() >= want_aux_len {
        return Ok(());
    }
    if me.aux_openings.len() != aux_base {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME aux_openings to start at aux_base (aux_openings.len()={}, aux_base={})",
            me.aux_openings.len(),
            aux_base
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(time_index(j)) weights for the bus time rows.
    let time_weights = precompute_contiguous_time_weights(&me.r, start_row, bus.chunk_size, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Decode once into logical witness coordinates so openings are layout-agnostic.
    let z_logical = decode_cpu_z_to_k(params, Z, bus.m)?;

    // Append bus openings in canonical col_id order so `bus_y_base = ct.len() - bus_cols`
    // remains valid.
    for col_id in 0..bus.bus_cols {
        let col_base = bus
            .bus_base
            .checked_add(
                col_id
                    .checked_mul(bus.chunk_size)
                    .ok_or_else(|| PiCcsError::InvalidInput("bus col_id * chunk_size overflow".into()))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput("bus col_base overflow".into()))?;
        let mut y_scalar = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            let idx = col_base
                .checked_add(j)
                .ok_or_else(|| PiCcsError::InvalidInput("bus cell index overflow".into()))?;
            let v = z_logical.get(idx).copied().ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "bus openings: logical index out of range (idx={idx}, m={})",
                    z_logical.len()
                ))
            })?;
            y_scalar += w * v;
        }

        me.aux_openings.push(y_scalar);
    }

    Ok(())
}

pub(crate) fn shared_bus_openings_from_time_columns_at_point(
    bus: &BusLayout,
    mem_cols: &[Vec<F>],
    point: &[K],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(BTreeMap::new());
    }
    if mem_cols.len() != bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: mem_cols.len()={} != bus.bus_cols={}",
            mem_cols.len(),
            bus.bus_cols
        )));
    }
    if point.is_empty() {
        return Err(PiCcsError::InvalidInput(format!("{label}: point must be non-empty")));
    }

    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: 2^ell_n overflow")))?;
    let start_row = bus.time_index(0);
    for j in 0..bus.chunk_size {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: bus time row index out of range: t={row} >= 2^ell_n={n_pad}"
            )));
        }
        if row != start_row + j {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: bus time_index must be contiguous: time_index({j})={row}, expected {}",
                start_row + j
            )));
        }
    }
    if start_row
        .checked_add(bus.chunk_size)
        .map_or(true, |end| end > n_pad)
    {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: bus time row range out of bounds: start_row={} chunk_size={} n_pad={}",
            start_row, bus.chunk_size, n_pad
        )));
    }

    let time_weights = precompute_contiguous_time_weights(point, start_row, bus.chunk_size, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    let mut out = BTreeMap::new();
    for col_id in 0..bus.bus_cols {
        let vals = mem_cols
            .get(col_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: missing mem column {col_id}")))?;
        if vals.len() != bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: time mem column length mismatch for bus col_id={col_id}: len={}, chunk_size={}",
                vals.len(),
                bus.chunk_size
            )));
        }
        let mut eval = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            eval += w * K::from(vals[j]);
        }
        out.insert(col_id, eval);
    }
    Ok(out)
}

pub(crate) fn time_columns_openings_from_time_columns_at_point(
    m_in: usize,
    t_len: usize,
    cpu_cols: &[Vec<F>],
    col_ids: &[usize],
    point: &[K],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(format!("{label}: t_len must be non-zero")));
    }
    if point.is_empty() {
        return Err(PiCcsError::InvalidInput(format!("{label}: point must be non-empty")));
    }

    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: 2^ell_n overflow")))?;
    if m_in.checked_add(t_len).map_or(true, |end| end > n_pad) {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: time row range out of bounds: m_in={} t_len={} n_pad={}",
            m_in, t_len, n_pad
        )));
    }

    let time_weights = precompute_contiguous_time_weights(point, m_in, t_len, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    let mut out = BTreeMap::new();
    for &col_id in col_ids {
        let vals = cpu_cols.get(col_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "{label}: trace col_id {} out of range for cpu_cols.len()={}",
                col_id,
                cpu_cols.len()
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: time cpu column length mismatch for col_id={col_id}: len={}, t_len={t_len}",
                vals.len()
            )));
        }
        let mut eval = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            eval += w * K::from(vals[j]);
        }
        out.insert(col_id, eval);
    }
    Ok(out)
}

pub(crate) fn append_bus_openings_to_me_instance_from_time_columns<Cmt>(
    params: &NeoParams,
    bus: &BusLayout,
    core_t: usize,
    mem_cols: &[Vec<F>],
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if bus.bus_cols == 0 {
        return Ok(());
    }
    if mem_cols.len() != bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require mem_cols.len()==bus.bus_cols (got {}, want {})",
            mem_cols.len(),
            bus.bus_cols
        )));
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    let start_row = bus.time_index(0);
    for j in 0..bus.chunk_size {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time row index out of range: t={row} >= 2^ell_n={n_pad}"
            )));
        }
        if row != start_row + j {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time_index must be contiguous: time_index({j})={row}, expected {}",
                start_row + j
            )));
        }
    }
    if start_row
        .checked_add(bus.chunk_size)
        .map_or(true, |end| end > n_pad)
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time row range out of bounds: start_row={} chunk_size={} n_pad={}",
            start_row, bus.chunk_size, n_pad
        )));
    }
    let want_len = core_t
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    let time_weights = precompute_contiguous_time_weights(&me.r, start_row, bus.chunk_size, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow_b = Vec::with_capacity(d);
    let mut cur = K::ONE;
    for _ in 0..d {
        pow_b.push(cur);
        cur *= bK;
    }

    for col_id in 0..bus.bus_cols {
        let vals = mem_cols
            .get(col_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing mem column {col_id} for bus openings")))?;
        if vals.len() != bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "time mem column length mismatch for bus col_id={col_id}: len={}, chunk_size={}",
                vals.len(),
                bus.chunk_size
            )));
        }

        let mut y_row = vec![K::ZERO; y_pad];
        let mut acc = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            acc += w * K::from(vals[j]);
        }
        y_row[0] = acc;
        let y_scalar = acc * pow_b[0];

        me.y_ring.push(y_row);
        me.ct.push(y_scalar);
    }

    Ok(())
}

pub(crate) fn append_zero_bus_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    bus: &BusLayout,
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }

    let want_len = core_t
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    for _ in 0..bus.bus_cols {
        me.y_ring.push(vec![K::ZERO; y_pad]);
        me.ct.push(K::ZERO);
    }
    Ok(())
}

pub(crate) fn append_zero_time_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    num_openings: usize,
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if num_openings == 0 {
        return Ok(());
    }
    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    let want_len = core_t
        .checked_add(num_openings)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + num_openings overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }
    for _ in 0..num_openings {
        me.y_ring.push(vec![K::ZERO; y_pad]);
        me.ct.push(K::ZERO);
    }
    Ok(())
}

pub(crate) fn append_time_columns_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    cpu_cols: &[Vec<F>],
    cols: &[usize],
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    append_time_columns_openings_to_me_instance_with_row_base(params, m_in, m_in, t_len, cpu_cols, cols, core_t, me)
}

pub(crate) fn append_time_columns_openings_to_me_instance_with_row_base<Cmt>(
    params: &NeoParams,
    m_in: usize,
    row_base: usize,
    t_len: usize,
    cpu_cols: &[Vec<F>],
    cols: &[usize],
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("time openings require t_len >= 1".into()));
    }
    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("time openings require non-empty ME.r".into()));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for j in 0..t_len {
        let row = row_base
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + j overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time row index out of range: (row_base + j)={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }
    let want_len = core_t
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + cols.len overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    let time_weights = precompute_contiguous_time_weights(&me.r, row_base, t_len, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    for &col_id in cols {
        let vals = cpu_cols.get(col_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "time openings: col_id={} out of range for cpu_cols.len()={}",
                col_id,
                cpu_cols.len()
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings: cpu_cols[{col_id}].len()={} != t_len={t_len}",
                vals.len()
            )));
        }
        let mut acc = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            acc += w * K::from(vals[j]);
        }
        let mut y_row = vec![K::ZERO; y_pad];
        y_row[0] = acc;
        me.y_ring.push(y_row);
        me.ct.push(acc);
    }
    Ok(())
}

pub(crate) fn append_mixed_time_columns_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    cpu_cols: &[Vec<F>],
    mem_cols: &[Vec<F>],
    logical_col_ids: &[usize],
    cols: &[usize],
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("time openings require t_len >= 1".into()));
    }
    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("time openings require non-empty ME.r".into()));
    }
    let total_cols = cpu_cols
        .len()
        .checked_add(mem_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("time openings: cpu_cols + mem_cols overflow".into()))?;
    if logical_col_ids.len() != total_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require logical_col_ids.len()==cpu_cols+mem_cols (got {}, expected {})",
            logical_col_ids.len(),
            total_cols
        )));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for j in 0..t_len {
        let row = m_in
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + j overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time row index out of range: (m_in + j)={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }
    let want_len = core_t
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + cols.len overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    let time_weights = precompute_contiguous_time_weights(&me.r, m_in, t_len, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    for &col_id in cols {
        let logical_pos = logical_col_ids
            .iter()
            .position(|&id| id == col_id)
            .ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "time openings: logical col_id={} is not present in logical_col_ids",
                    col_id
                ))
            })?;
        let vals: &Vec<F> = if logical_pos < cpu_cols.len() {
            cpu_cols.get(logical_pos).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "time openings: cpu column position {} out of range for cpu_cols.len()={}",
                    logical_pos,
                    cpu_cols.len()
                ))
            })?
        } else {
            let mem_col = logical_pos - cpu_cols.len();
            mem_cols.get(mem_col).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "time openings: mem column position {} out of range for mem_cols.len()={}",
                    mem_col,
                    mem_cols.len()
                ))
            })?
        };
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings: selected column len={} != t_len={t_len} for col_id={col_id}",
                vals.len()
            )));
        }

        let mut acc = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            acc += w * K::from(vals[j]);
        }
        let mut y_row = vec![K::ZERO; y_pad];
        y_row[0] = acc;
        me.y_ring.push(y_row);
        me.ct.push(acc);
    }
    Ok(())
}

pub(crate) fn append_time_columns_openings_to_me_instance_at_js<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    cpu_cols: &[Vec<F>],
    cols: &[usize],
    core_t: usize,
    me: &mut CeClaim<Cmt, F, K>,
    js: &[usize],
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("time openings require t_len >= 1".into()));
    }
    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("time openings require non-empty ME.r".into()));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for &j in js {
        if j >= t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings js out of range: j={j} >= t_len={t_len}"
            )));
        }
        let row = m_in
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + j overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time row index out of range: (m_in + j)={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }
    let want_len = core_t
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + cols.len overflow".into()))?;
    if me.y_ring.len() == want_len && me.ct.len() == want_len {
        return Ok(());
    }
    if me.y_ring.len() != core_t || me.ct.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "time openings expect ME y/ct to start at core_t (y.len()={}, ct.len()={}, core_t={})",
            me.y_ring.len(),
            me.ct.len(),
            core_t
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    let dense_selection = js.len().saturating_mul(3) >= t_len;
    let time_weights: Vec<(usize, K)> = if dense_selection {
        let all = precompute_contiguous_time_weights(&me.r, m_in, t_len, n_pad)?;
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push((j, all[j]));
        }
        out
    } else {
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push((j, chi_for_row_index(&me.r, m_in + j)));
        }
        out
    };
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    for &col_id in cols {
        let vals = cpu_cols.get(col_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "time openings: col_id={} out of range for cpu_cols.len()={}",
                col_id,
                cpu_cols.len()
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "time openings: cpu_cols[{col_id}].len()={} != t_len={t_len}",
                vals.len()
            )));
        }
        let mut acc = K::ZERO;
        for &(j, w) in weighted_rows.iter() {
            acc += w * K::from(vals[j]);
        }
        let mut y_row = vec![K::ZERO; y_pad];
        y_row[0] = acc;
        me.y_ring.push(y_row);
        me.ct.push(acc);
    }
    Ok(())
}

/// Append time-indexed openings for a column-major region of the CPU witness.
///
/// This is a "no shared CPU bus tail" bridge: instead of materializing copyout matrices for
/// per-row columns (e.g. an execution trace), we compute their Route-A time-combined openings
/// directly from the committed witness matrix `Z` and append them to the ME instance.
///
/// Semantics: for each `col_id` in `cols`, append an opening for the vector
/// `{ z[col_base + col_id * t_len + j] }_{j=0..t_len-1}` combined with weights
/// `χ_r(m_in + j)` where `r = me.r`.
pub(crate) fn append_col_major_time_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    col_base: usize,
    cols: &[usize],
    core_t: usize,
    expected_m: usize,
    Z: &Mat<F>,
    me: &mut CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("trace openings require t_len >= 1".into()));
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("trace openings require non-empty ME.r".into()));
    }
    if col_base >= expected_m {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require col_base < m (col_base={}, m={})",
            col_base, expected_m
        )));
    }

    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for j in 0..t_len {
        let row = m_in
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + t overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace time row index out of range: (m_in + j)={} out of range for ell_n={} (n_pad={})",
                row,
                me.r.len(),
                n_pad
            )));
        }
    }

    if me.y_ring.len() != me.ct.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require ME core y/ct alignment (y.len()={}, ct.len()={})",
            me.y_ring.len(),
            me.ct.len(),
        )));
    }
    if core_t < me.ct.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require core_t >= ME core ct len (core_t={}, ct.len()={})",
            core_t,
            me.ct.len(),
        )));
    }
    let aux_base = core_t
        .checked_sub(me.ct.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t underflow for trace aux base".into()))?;

    // Idempotent append: allow callers to call this once; reject unexpected aux lengths.
    let want_aux_len = aux_base
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("aux_base + cols.len overflow".into()))?;
    if me.aux_openings.len() == want_aux_len {
        return Ok(());
    }
    if me.aux_openings.len() != aux_base {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings expect ME aux_openings to start at aux_base (aux_openings.len()={}, aux_base={})",
            me.aux_openings.len(),
            aux_base
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings require ME.y_ring[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(m_in + j) weights for the time rows.
    let time_weights = precompute_contiguous_time_weights(&me.r, m_in, t_len, n_pad)?;
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Decode once into logical witness coordinates so openings are layout-agnostic.
    let z_logical = decode_cpu_z_to_k(params, Z, expected_m)?;

    for &col_id in cols {
        let col_offset = col_id
            .checked_mul(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
        let col_start = col_base
            .checked_add(col_offset)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_base + col_offset overflow".into()))?;
        let col_end = col_start
            .checked_add(t_len - 1)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_end overflow".into()))?;
        if col_end >= expected_m {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings: column span out of range (col_start={col_start}, col_end={col_end}, m={})",
                expected_m
            )));
        }

        let mut y_scalar = K::ZERO;
        for (j, w) in weighted_rows.iter() {
            let idx = col_start
                .checked_add(*j)
                .ok_or_else(|| PiCcsError::InvalidInput("trace logical index overflow".into()))?;
            let v = z_logical.get(idx).copied().ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "trace openings: logical index out of range (idx={idx}, m={})",
                    z_logical.len()
                ))
            })?;
            y_scalar += *w * v;
        }

        me.aux_openings.push(y_scalar);
    }

    Ok(())
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

fn ccs_col_has_any_nonzero(mat: &CcsMatrix<F>, col: usize) -> bool {
    match mat {
        CcsMatrix::Identity { n } => col < *n,
        CcsMatrix::Csc(csc) => csc.col_ptr[col] < csc.col_ptr[col + 1],
    }
}

fn ccs_col_has_nonzero_outside_padding_rows(mat: &CcsMatrix<F>, col: usize, is_padding_row: &[bool]) -> bool {
    match mat {
        CcsMatrix::Identity { n } => {
            if col >= *n {
                return false;
            }
            let row = col;
            row < is_padding_row.len() && !is_padding_row[row]
        }
        CcsMatrix::Csc(csc) => {
            let s0 = csc.col_ptr[col];
            let e0 = csc.col_ptr[col + 1];
            for k in s0..e0 {
                let row = csc.row_idx[k];
                if row < is_padding_row.len() && !is_padding_row[row] {
                    return true;
                }
            }
            false
        }
    }
}

struct BusColLabel {
    col_id: usize,
    label: String,
}

fn required_bus_cols_for_layout(layout: &BusLayout) -> Vec<BusColLabel> {
    let mut out = Vec::<BusColLabel>::new();
    for (lut_idx, inst) in layout.shout_cols.iter().enumerate() {
        for (lane_idx, shout) in inst.lanes.iter().enumerate() {
            for (b, col_id) in shout.addr_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("shout[{lut_idx}].lane[{lane_idx}].addr_bits[{b}]"),
                });
            }
            out.push(BusColLabel {
                col_id: shout.has_lookup,
                label: format!("shout[{lut_idx}].lane[{lane_idx}].has_lookup"),
            });
            for (v_idx, &val_col) in shout.vals.iter().enumerate() {
                out.push(BusColLabel {
                    col_id: val_col,
                    label: format!("shout[{lut_idx}].lane[{lane_idx}].val[{v_idx}]"),
                });
            }
        }
    }

    for (mem_idx, inst) in layout.twist_cols.iter().enumerate() {
        for (lane_idx, twist) in inst.lanes.iter().enumerate() {
            for (b, col_id) in twist.ra_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}].ra_bits[{b}]"),
                });
            }
            for (b, col_id) in twist.wa_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}].wa_bits[{b}]"),
                });
            }
            out.push(BusColLabel {
                col_id: twist.has_read,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].has_read"),
            });
            out.push(BusColLabel {
                col_id: twist.has_write,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].has_write"),
            });
            out.push(BusColLabel {
                col_id: twist.wv,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].wv"),
            });
            out.push(BusColLabel {
                col_id: twist.rv,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].rv"),
            });
            out.push(BusColLabel {
                col_id: twist.inc,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].inc_at_write_addr"),
            });
        }
    }

    out
}

fn required_bus_binding_cols_for_layout<Cmt, S: BusStepView<Cmt>>(layout: &BusLayout, steps: &[S]) -> Vec<BusColLabel> {
    // Note: `inc_at_write_addr` is a Twist-internal witness field derived from the sparse
    // memory state. Many CPU CCSes do not (and should not) constrain it outside padding rows;
    // it is constrained by the Twist Route-A checks instead. We still require the canonical
    // padding constraint `(1 - has_write) * inc_at_write_addr = 0`.
    let inc_cols: HashSet<usize> = layout
        .twist_cols
        .iter()
        .flat_map(|inst| inst.lanes.iter().map(|t| t.inc))
        .collect();

    // Shout key `addr_bits` are often constrained outside the *main* CPU CCS:
    // - by a decode/semantics sidecar CCS, and/or
    // - by VM-specific constraints that live outside the shared-bus binding gadget.
    //
    // The Route-A Shout argument already constrains `(addr_bits, val)` internally via:
    // - per-lane Shout value/adaptor terminal checks, and
    // - trace linkage checks (`verify_route_a_memory_step`) that bind the
    //   CPU trace's `(shout_has_lookup, shout_val, shout_lhs, shout_rhs)` to the sidecar openings.
    //
    // In RV32 trace shared-bus mode, Shout table-linkage ownership is moved to reduction-time
    // aggregate checks, so the shared-bus adapter may intentionally omit CPU-linkage equalities
    // for Shout lanes. Keep only canonical Shout padding/bitness constraints in the CPU CCS and
    // exempt all Shout columns from the "outside-padding binding" guard.
    let shout_addr_cols: HashSet<usize> = layout
        .shout_cols
        .iter()
        .flat_map(|inst| inst.lanes.iter().flat_map(|s| s.addr_bits.clone()))
        .collect();
    let shout_selector_and_val_cols: HashSet<usize> = layout
        .shout_cols
        .iter()
        .flat_map(|inst| {
            inst.lanes.iter().flat_map(|s| {
                let mut cols = Vec::with_capacity(1 + s.vals.len());
                cols.push(s.has_lookup);
                cols.extend_from_slice(&s.vals);
                cols
            })
        })
        .collect();

    let mut twist_unbound_cols: HashSet<usize> = HashSet::new();
    if let Some(step0) = steps.first() {
        let has_trace_lookup_families = (0..step0.lut_insts_len()).any(|idx| {
            let table_id = step0.lut_inst(idx).table_id;
            riscv_is_decode_lookup_table_id(table_id) || riscv_trace_is_width_lookup_table_id(table_id)
        });
        if has_trace_lookup_families {
            for (mem_idx, inst) in layout.twist_cols.iter().enumerate() {
                let mem_id = step0.mem_inst(mem_idx).mem_id;
                for (lane_idx, twist) in inst.lanes.iter().enumerate() {
                    let read_bound = if mem_id == PROG_ID.0 {
                        lane_idx == 0
                    } else if mem_id == REG_ID.0 {
                        lane_idx <= 1
                    } else {
                        lane_idx == 0
                    };
                    let write_bound = if mem_id == REG_ID.0 {
                        lane_idx == 0
                    } else if mem_id == PROG_ID.0 {
                        false
                    } else {
                        lane_idx == 0
                    };

                    if !read_bound {
                        twist_unbound_cols.insert(twist.has_read);
                        twist_unbound_cols.insert(twist.rv);
                        twist_unbound_cols.extend(twist.ra_bits.clone());
                    }
                    if !write_bound {
                        twist_unbound_cols.insert(twist.has_write);
                        twist_unbound_cols.insert(twist.wv);
                        twist_unbound_cols.insert(twist.inc);
                        twist_unbound_cols.extend(twist.wa_bits.clone());
                    }
                }
            }
        }
    }

    required_bus_cols_for_layout(layout)
        .into_iter()
        .filter(|c| !inc_cols.contains(&c.col_id))
        .filter(|c| !shout_addr_cols.contains(&c.col_id))
        .filter(|c| !shout_selector_and_val_cols.contains(&c.col_id))
        .filter(|c| !twist_unbound_cols.contains(&c.col_id))
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
        let mut all_js_present = true;
        for j in 0..bus.chunk_size {
            let z_idx = bus.bus_cell(col.col_id, j);
            if z_idx >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                    z_idx, s.m
                )));
            }

            let mut found = false;
            'active_mats: for &mj in &active {
                let mat = &s.matrices[mj];
                if ccs_col_has_any_nonzero(mat, z_idx) {
                    found = true;
                    break 'active_mats;
                }
            }
            if !found {
                all_js_present = false;
                break;
            }
        }
        if !all_js_present {
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

fn ccs_references_any_bus_cols(s: &CcsStructure<F>, bus: &BusLayout) -> bool {
    if bus.bus_cols == 0 {
        return false;
    }
    let active = active_matrix_indices(s);
    if active.is_empty() {
        return false;
    }

    for col_id in 0..bus.bus_cols {
        for j in 0..bus.chunk_size {
            let z_idx = bus.bus_cell(col_id, j);
            if z_idx >= s.m {
                continue;
            }
            for &mj in &active {
                if ccs_col_has_any_nonzero(&s.matrices[mj], z_idx) {
                    return true;
                }
            }
        }
    }
    false
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
        let mut all_js_present = true;
        for j in 0..bus.chunk_size {
            let z_idx = bus.bus_cell(col.col_id, j);
            if z_idx >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                    z_idx, s.m
                )));
            }

            let mut found = false;
            'active_mats: for &mj in &active {
                let mat = &s.matrices[mj];
                if ccs_col_has_nonzero_outside_padding_rows(mat, z_idx, &is_padding_row) {
                    found = true;
                    break 'active_mats;
                }
            }
            if !found {
                all_js_present = false;
                break;
            }
        }
        if !all_js_present {
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
    let mut shout_addr_range_counts = std::collections::HashMap::<(usize, usize), usize>::new();
    for inst in bus.shout_cols.iter() {
        for shout in inst.lanes.iter() {
            let key = (shout.addr_bits.start, shout.addr_bits.end);
            *shout_addr_range_counts.entry(key).or_insert(0) += 1;
        }
    }

    for (lut_idx, inst) in bus.shout_cols.iter().enumerate() {
        for (lane_idx, shout) in inst.lanes.iter().enumerate() {
            let key = (shout.addr_bits.start, shout.addr_bits.end);
            let shared_addr_group = shout_addr_range_counts.get(&key).copied().unwrap_or(0) > 1;
            for j in 0..bus.chunk_size {
                let has_lookup_z = bus.bus_cell(shout.has_lookup, j);
                let val_z = bus.bus_cell(shout.primary_val(), j);

                // (1 - has_lookup) * val = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_lookup_z,
                    field_z_idx: val_z,
                    label: format!("shout[{lut_idx}].lane[{lane_idx}][j={j}]: (1-has_lookup)*val"),
                });

                if !shared_addr_group {
                    // (1 - has_lookup) * addr_bits[b] = 0
                    for (b, col_id) in shout.addr_bits.clone().enumerate() {
                        let bit_z = bus.bus_cell(col_id, j);
                        out.push(BusPaddingLabel {
                            flag_z_idx: has_lookup_z,
                            field_z_idx: bit_z,
                            label: format!("shout[{lut_idx}].lane[{lane_idx}][j={j}]: (1-has_lookup)*addr_bits[{b}]"),
                        });
                    }
                }
            }
        }
    }

    for (mem_idx, inst) in bus.twist_cols.iter().enumerate() {
        for (lane_idx, twist) in inst.lanes.iter().enumerate() {
            for j in 0..bus.chunk_size {
                let has_read_z = bus.bus_cell(twist.has_read, j);
                let has_write_z = bus.bus_cell(twist.has_write, j);

                let wv_z = bus.bus_cell(twist.wv, j);
                let rv_z = bus.bus_cell(twist.rv, j);
                let inc_z = bus.bus_cell(twist.inc, j);

                // (1 - has_read) * rv = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_read_z,
                    field_z_idx: rv_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_read)*rv"),
                });

                // (1 - has_read) * ra_bits[b] = 0
                for (b, col_id) in twist.ra_bits.clone().enumerate() {
                    let bit_z = bus.bus_cell(col_id, j);
                    out.push(BusPaddingLabel {
                        flag_z_idx: has_read_z,
                        field_z_idx: bit_z,
                        label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_read)*ra_bits[{b}]"),
                    });
                }

                // (1 - has_write) * wv = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_write_z,
                    field_z_idx: wv_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*wv"),
                });

                // (1 - has_write) * inc_at_write_addr = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_write_z,
                    field_z_idx: inc_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*inc_at_write_addr"),
                });

                // (1 - has_write) * wa_bits[b] = 0
                for (b, col_id) in twist.wa_bits.clone().enumerate() {
                    let bit_z = bus.bus_cell(col_id, j);
                    out.push(BusPaddingLabel {
                        flag_z_idx: has_write_z,
                        field_z_idx: bit_z,
                        label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*wa_bits[{b}]"),
                    });
                }
            }
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

    // This validation is intentionally strict and recognizes two safe patterns in the canonical
    // R1CS embedding:
    //
    // 1) Explicit padding constraints:
    //      (1 - flag) * field = 0
    //
    // 2) For *bit* fields only (addr bits), padding implied by a gated-bit constraint:
    //      bit * (bit - flag) = 0
    //    combined with a boolean constraint on `flag`. When `flag ∈ {0,1}`, this implies:
    //      - flag=0 => bit=0
    //      - flag=1 => bit ∈ {0,1}
    //
    // We accept (2) to allow circuits to drop redundant per-bit padding rows while staying safe.
    let (a_idx, b_idx, c_idx) = if s.matrices.len() >= 4 && s.matrices[0].is_identity() {
        (1usize, 2usize, 3usize)
    } else if s.matrices.len() >= 3 {
        (0usize, 1usize, 2usize)
    } else {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus=true but CPU CCS has fewer than 3 matrices; cannot validate padding constraints".into(),
        ));
    };

    let n = s.n;
    let const_one_cols: HashSet<usize> = const_one_cols.iter().copied().collect();

    let mut c_has_nonzero = vec![false; n];
    let mut b_count = vec![0u8; n];
    let mut b_col1 = vec![0usize; n];
    let mut b_val1 = vec![F::ZERO; n];
    let mut b_col2 = vec![0usize; n];
    let mut b_val2 = vec![F::ZERO; n];

    let mut a_count = vec![0u8; n];
    let mut a_col1 = vec![0usize; n];
    let mut a_val1 = vec![F::ZERO; n];
    let mut a_col2 = vec![0usize; n];
    let mut a_val2 = vec![F::ZERO; n];

    let scan_c = |mat: &CcsMatrix<F>, c_has_nonzero: &mut [bool]| match mat {
        CcsMatrix::Identity { n } => {
            let cap = core::cmp::min(*n, c_has_nonzero.len());
            for row in 0..cap {
                c_has_nonzero[row] = true;
            }
        }
        CcsMatrix::Csc(csc) => {
            for &row in &csc.row_idx {
                if row < c_has_nonzero.len() {
                    c_has_nonzero[row] = true;
                }
            }
        }
    };

    let scan_b = |mat: &CcsMatrix<F>,
                  b_count: &mut [u8],
                  b_col1: &mut [usize],
                  b_val1: &mut [F],
                  b_col2: &mut [usize],
                  b_val2: &mut [F]| {
        match mat {
            CcsMatrix::Identity { n } => {
                let cap = core::cmp::min(*n, b_count.len());
                for row in 0..cap {
                    b_count[row] = 1;
                    b_col1[row] = row;
                    b_val1[row] = F::ONE;
                }
            }
            CcsMatrix::Csc(csc) => {
                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        if row >= b_count.len() {
                            continue;
                        }
                        match b_count[row] {
                            0 => {
                                b_count[row] = 1;
                                b_col1[row] = col;
                                b_val1[row] = csc.vals[k];
                            }
                            1 => {
                                b_count[row] = 2;
                                b_col2[row] = col;
                                b_val2[row] = csc.vals[k];
                            }
                            _ => {
                                b_count[row] = 3;
                            }
                        }
                    }
                }
            }
        }
    };

    let scan_a = |mat: &CcsMatrix<F>,
                  a_count: &mut [u8],
                  a_col1: &mut [usize],
                  a_val1: &mut [F],
                  a_col2: &mut [usize],
                  a_val2: &mut [F]| {
        match mat {
            CcsMatrix::Identity { n } => {
                let cap = core::cmp::min(*n, a_count.len());
                for row in 0..cap {
                    a_count[row] = 1;
                    a_col1[row] = row;
                    a_val1[row] = F::ONE;
                }
            }
            CcsMatrix::Csc(csc) => {
                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        if row >= a_count.len() {
                            continue;
                        }
                        match a_count[row] {
                            0 => {
                                a_count[row] = 1;
                                a_col1[row] = col;
                                a_val1[row] = csc.vals[k];
                            }
                            1 => {
                                a_count[row] = 2;
                                a_col2[row] = col;
                                a_val2[row] = csc.vals[k];
                            }
                            _ => {
                                a_count[row] = 3;
                            }
                        }
                    }
                }
            }
        }
    };

    scan_c(&s.matrices[c_idx], &mut c_has_nonzero);
    scan_b(
        &s.matrices[b_idx],
        &mut b_count,
        &mut b_col1,
        &mut b_val1,
        &mut b_col2,
        &mut b_val2,
    );
    scan_a(
        &s.matrices[a_idx],
        &mut a_count,
        &mut a_col1,
        &mut a_val1,
        &mut a_col2,
        &mut a_val2,
    );

    // Find boolean constraints for flag columns: flag * (flag - 1) = 0.
    let mut flag_is_boolean: HashSet<usize> = HashSet::new();
    let mut flag_boolean_row: HashMap<usize, usize> = HashMap::new();
    let mut selector_rows: HashSet<usize> = HashSet::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if a_count[row] != 1 || a_val1[row] != F::ONE {
            continue;
        }
        let flag_col = a_col1[row];
        if b_count[row] != 2 {
            continue;
        }

        let (c1, v1) = (b_col1[row], b_val1[row]);
        let (c2, v2) = (b_col2[row], b_val2[row]);

        let ok = (c1 == flag_col && v1 == F::ONE && const_one_cols.contains(&c2) && v2 == -F::ONE)
            || (c2 == flag_col && v2 == F::ONE && const_one_cols.contains(&c1) && v1 == -F::ONE);
        if !ok {
            continue;
        }

        flag_is_boolean.insert(flag_col);
        flag_boolean_row.entry(flag_col).or_insert(row);
        selector_rows.insert(row);
    }

    // Explicit padding constraints present in CCS: (1 - flag) * field = 0.
    let mut present: HashSet<(usize, usize)> = HashSet::new();
    let mut padding_rows: HashSet<usize> = HashSet::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if b_count[row] != 1 || b_val1[row] != F::ONE {
            continue;
        }
        let field_col = b_col1[row];
        if a_count[row] != 2 {
            continue;
        }
        let (c1, v1) = (a_col1[row], a_val1[row]);
        let (c2, v2) = (a_col2[row], a_val2[row]);
        if v1 != -v2 {
            continue;
        }

        let flag_col = if const_one_cols.contains(&c1) {
            Some(c2)
        } else if const_one_cols.contains(&c2) {
            Some(c1)
        } else {
            None
        };
        let Some(flag_col) = flag_col else {
            continue;
        };

        present.insert((flag_col, field_col));
        padding_rows.insert(row);
    }

    // Bitness constraints that imply address-bit padding: bit * (bit - flag) = 0.
    //
    // We only treat these as satisfying padding when `flag` is also boolean.
    let mut implied: HashMap<(usize, usize), usize> = HashMap::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if a_count[row] != 1 || a_val1[row] != F::ONE {
            continue;
        }
        let bit_col = a_col1[row];
        if b_count[row] != 2 {
            continue;
        }

        let (c1, v1) = (b_col1[row], b_val1[row]);
        let (c2, v2) = (b_col2[row], b_val2[row]);

        let flag_col = if c1 == bit_col && v1 == F::ONE && v2 == -F::ONE {
            Some(c2)
        } else if c2 == bit_col && v2 == F::ONE && v1 == -F::ONE {
            Some(c1)
        } else {
            None
        };
        let Some(flag_col) = flag_col else {
            continue;
        };
        if !flag_is_boolean.contains(&flag_col) {
            continue;
        }

        implied.insert((flag_col, bit_col), row);
    }

    let mut missing: Vec<&BusPaddingLabel> = Vec::new();
    for req in required {
        if present.contains(&(req.flag_z_idx, req.field_z_idx)) {
            continue;
        }
        if implied.contains_key(&(req.flag_z_idx, req.field_z_idx)) {
            continue;
        }
        if !present.contains(&(req.flag_z_idx, req.field_z_idx)) {
            missing.push(req);
        }
    }

    if missing.is_empty() {
        // Treat selector boolean and bitness constraints as padding-only rows so they don't
        // accidentally satisfy "binding" presence checks.
        for row in selector_rows {
            padding_rows.insert(row);
        }
        for &row in implied.values() {
            padding_rows.insert(row);
        }
        for &row in flag_boolean_row.values() {
            padding_rows.insert(row);
        }
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

fn ensure_ccs_binds_shared_bus_for_steps<Cmt, S: BusStepView<Cmt>>(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    padding_rows: &HashSet<usize>,
    steps: &[S],
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }
    let required = required_bus_cols_for_layout(bus);
    ensure_ccs_references_bus_cols(s, bus, &required)?;

    let binding_required = required_bus_binding_cols_for_layout(bus, steps);
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

pub(crate) fn decode_cpu_z_to_k(params: &NeoParams, Z: &Mat<F>, expected_m: usize) -> Result<Vec<K>, PiCcsError> {
    neo_reductions::common::decode_z_from_witness_mat(params, Z, expected_m)
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

pub(crate) fn build_time_sparse_from_bus_col_at_js(
    z: &[K],
    bus: &BusLayout,
    col_id: usize,
    js: &[usize],
    pow2_cycle: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
    if col_id >= bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus col_id out of range: {col_id} >= {}",
            bus.bus_cols
        )));
    }
    let mut entries: Vec<(usize, K)> = Vec::new();
    for &j in js {
        if j >= bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "bus j out of range: j={j} >= bus.chunk_size={}",
                bus.chunk_size
            )));
        }
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
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

pub(crate) fn build_time_sparse_from_mem_cols(
    mem_cols: &[Vec<F>],
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
    if col_id >= mem_cols.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "time mem col_id out of range: {col_id} >= {}",
            mem_cols.len()
        )));
    }
    if steps_len > bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "steps_len({steps_len}) > bus.chunk_size({})",
            bus.chunk_size
        )));
    }
    let vals = &mem_cols[col_id];
    if vals.len() != bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "time mem column length mismatch: col_id={col_id}, len={}, chunk_size={}",
            vals.len(),
            bus.chunk_size
        )));
    }
    let mut entries: Vec<(usize, K)> = Vec::new();
    for (j, v) in vals.iter().copied().enumerate().take(steps_len) {
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        let vk = K::from(v);
        if vk != K::ZERO {
            entries.push((t, vk));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}

pub(crate) fn build_time_sparse_from_mem_cols_at_js(
    mem_cols: &[Vec<F>],
    bus: &BusLayout,
    col_id: usize,
    js: &[usize],
    pow2_cycle: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
    if col_id >= bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus col_id out of range: {col_id} >= {}",
            bus.bus_cols
        )));
    }
    if col_id >= mem_cols.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "time mem col_id out of range: {col_id} >= {}",
            mem_cols.len()
        )));
    }
    let vals = &mem_cols[col_id];
    if vals.len() != bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "time mem column length mismatch: col_id={col_id}, len={}, chunk_size={}",
            vals.len(),
            bus.chunk_size
        )));
    }
    let mut entries: Vec<(usize, K)> = Vec::new();
    for &j in js {
        if j >= bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "bus j out of range: j={j} >= bus.chunk_size={}",
                bus.chunk_size
            )));
        }
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        let vk = K::from(vals[j]);
        if vk != K::ZERO {
            entries.push((t, vk));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}
