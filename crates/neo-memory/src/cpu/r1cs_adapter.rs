//! R1CS-based CPU adapter for the shared memory/lookup bus.
//!
//! This module provides an adapter that implements `CpuArithmetization` for a generic
//! R1CS-based CPU, allowing integration with Neo's shared CPU bus architecture.

use crate::addr::{write_addr_bits_dim_major_le_into_bus, write_addr_bits_dim_major_le_u128_into_bus};
use crate::ajtai::encode_vector_for_ccs_m;
use crate::builder::CpuArithmetization;
use crate::cpu::bus_layout::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, BusLayout, ShoutInstanceShape,
};
use crate::cpu::constraints::{
    extend_ccs_with_shared_cpu_bus_constraints_optional_shout, ShoutCpuBinding, TwistCpuBinding, CPU_BUS_COL_DISABLED,
};
use crate::mem_init::MemInit;
use crate::plain::LutTable;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::uninterleave_bits;
use crate::riscv::trace::riscv_trace_lookup_n_vals_for_table_id;
use crate::witness::{LutInstance, LutTableSpec, MemInstance};
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64};
use p3_goldilocks::Goldilocks;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for the shared CPU bus.
#[derive(Clone, Debug)]
pub struct SharedCpuBusConfig<F> {
    /// Geometry for each Twist instance (twist_id -> layout).
    pub mem_layouts: HashMap<u32, PlainMemLayout>,
    /// Sparse initial memory values: (twist_id, addr) -> value.
    pub initial_mem: HashMap<(u32, u64), F>,
    /// Public witness column index that must be fixed to 1 for all steps.
    ///
    /// Required because selector-binding constraints are represented as unconditional equalities:
    /// `1 * (cpu_flag - bus_flag) = 0`.
    ///
    /// Must satisfy: `const_one_col < m_in`.
    pub const_one_col: usize,
    /// Per-table CPU→bus bindings (shout_id -> binding).
    ///
    /// The bus tail contains one Shout instance per `table_id` known to this CPU (from `tables` in `R1csCpu::new`).
    ///
    /// Each Shout instance may have multiple lookup lanes.
    /// - Non-empty vector: CPU linkage bindings in lane-index order.
    /// - Empty vector: no CPU linkage for that table's bus lanes (padding/bitness only).
    pub shout_cpu: HashMap<u32, Vec<ShoutCpuBinding>>,
    /// Per-memory CPU→bus bindings (twist_id -> binding).
    ///
    /// The bus tail contains one Twist instance per `mem_id` in `mem_layouts`.
    ///
    /// Each Twist instance may have multiple access lanes (`PlainMemLayout.lanes`); this map must
    /// provide one `TwistCpuBinding` per lane in lane-index order.
    pub twist_cpu: HashMap<u32, Vec<TwistCpuBinding>>,
    /// Optional per-table address-sharing group ids (table_id -> group_id).
    ///
    /// Tables with the same group_id share `addr_bits` columns in the bus layout.
    /// Leave empty when no families share address bits. Populated by trace mode for column efficiency.
    pub shout_addr_groups: HashMap<u32, u64>,
    /// Optional per-table selector-sharing group ids (table_id -> group_id).
    ///
    /// Tables with the same group_id share `has_lookup` columns in the bus layout.
    /// Leave empty when no families share selectors. Populated by trace mode for column efficiency.
    pub shout_selector_groups: HashMap<u32, u64>,
}

#[derive(Clone, Debug)]
struct SharedCpuBusState<F> {
    cfg: SharedCpuBusConfig<F>,
    table_ids: Vec<u32>,
    mem_ids: Vec<u32>,
    layout: BusLayout,
}

/// Adapter that implements CpuArithmetization for a generic R1CS-based CPU.
///
/// It assumes the user provides:
/// 1. The CCS structure (wrapped R1CS matrices).
/// 2. A witness builder function that maps a `StepTrace` to the R1CS witness vector `w`.
/// 3. Shout tables to verify trace correctness.
pub struct R1csCpu<F, Cmt, L, Key = u64>
where
    F: PrimeField + PrimeField64,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub ccs: CcsStructure<F>,
    pub params: NeoParams,
    pub committer: L,

    /// Number of public inputs m_in (prefix of z treated as x).
    /// The witness z is split as z = (x[0..m_in], w[m_in..]).
    pub m_in: usize,

    /// Shout table metadata needed to write the shared CPU bus (d, n_side).
    pub shout_meta: HashMap<u32, (usize, usize)>,
    /// Optional virtual table specs by table_id, used for packed bus witness filling.
    pub shout_specs: HashMap<u32, LutTableSpec>,

    /// Optional shared CPU-bus configuration.
    /// When present, we overwrite a reserved tail segment of `z_vec` with Twist/Shout access rows
    /// (in deterministic id order) before Ajtai decomposition + commitment.
    shared_cpu_bus: Option<SharedCpuBusState<F>>,
    /// Optional lookup-family sharing metadata used by Route-A claim planning in uniform mode.
    /// This is intentionally independent from physical shared-bus constraint injection.
    lookup_shout_addr_groups: HashMap<u32, u64>,
    lookup_shout_selector_groups: HashMap<u32, u64>,

    /// Function to map a trace chunk (up to `chunk_size` steps) to the full witness z = (x, w).
    /// The witness MUST satisfy the CCS relation.
    pub chunk_to_witness: Box<dyn Fn(&[StepTrace<u64, u64, Key>]) -> Vec<F> + Send + Sync>,

    _phantom: PhantomData<Cmt>,
}

impl<F, Cmt, L, Key> R1csCpu<F, Cmt, L, Key>
where
    F: PrimeField + PrimeField64 + Copy,
    L: SModuleHomomorphism<F, Cmt>,
    Key: Copy + Send + Sync + 'static,
{
    pub fn new(
        ccs: CcsStructure<F>,
        params: NeoParams,
        committer: L,
        m_in: usize,
        tables: &HashMap<u32, LutTable<F>>,
        table_specs: &HashMap<u32, LutTableSpec>,
        chunk_to_witness: Box<dyn Fn(&[StepTrace<u64, u64, Key>]) -> Vec<F> + Send + Sync>,
    ) -> Result<Self, String> {
        let mut shout_meta = HashMap::new();
        let mut shout_specs = HashMap::new();
        for (id, table) in tables {
            shout_meta.insert(*id, (table.d, table.n_side));
        }
        for (id, spec) in table_specs {
            let (d, n_side) = match spec {
                LutTableSpec::RiscvOpcode { xlen, .. } => (xlen.saturating_mul(2), 2usize),
                LutTableSpec::RiscvOpcodePacked { opcode, xlen } => (
                    crate::riscv::packed::rv_packed_d(*opcode, *xlen)
                        .map_err(|e| format!("invalid packed opcode spec: {e}"))?,
                    2usize,
                ),
                LutTableSpec::RiscvOpcodeEventTablePacked { .. } => {
                    return Err("RiscvOpcodeEventTablePacked is not supported in the shared-bus R1csCpu path".into());
                }
                LutTableSpec::IdentityU32 => (32usize, 2usize),
            };
            shout_specs.insert(*id, spec.clone());
            match shout_meta.entry(*id) {
                Entry::Vacant(v) => {
                    v.insert((d, n_side));
                }
                Entry::Occupied(existing) => {
                    // Prefer explicit table metadata when both are present.
                    debug_assert_eq!(
                        *existing.get(),
                        (d, n_side),
                        "shout_meta mismatch for table_id={id}: explicit={:?} spec={:?}",
                        existing.get(),
                        (d, n_side)
                    );
                }
            }
        }

        Ok(Self {
            ccs,
            params,
            committer,
            m_in,
            shout_meta,
            shout_specs,
            shared_cpu_bus: None,
            lookup_shout_addr_groups: HashMap::new(),
            lookup_shout_selector_groups: HashMap::new(),
            chunk_to_witness,
            _phantom: PhantomData,
        })
    }

    /// Attach lookup-family sharing metadata without enabling physical shared-bus tail injection.
    pub fn with_lookup_sharing_groups(
        mut self,
        shout_addr_groups: HashMap<u32, u64>,
        shout_selector_groups: HashMap<u32, u64>,
    ) -> Self {
        self.lookup_shout_addr_groups = shout_addr_groups;
        self.lookup_shout_selector_groups = shout_selector_groups;
        self
    }

    fn shared_bus_schema(
        &self,
        bus: &SharedCpuBusConfig<F>,
        chunk_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, BusLayout), String> {
        let mut table_ids: Vec<u32> = self.shout_meta.keys().copied().collect();
        table_ids.sort_unstable();
        let mut mem_ids: Vec<u32> = bus.mem_layouts.keys().copied().collect();
        mem_ids.sort_unstable();

        let mut shout_shapes = Vec::with_capacity(table_ids.len());
        for table_id in &table_ids {
            let (d, n_side) = self
                .shout_meta
                .get(table_id)
                .copied()
                .ok_or_else(|| format!("shared_cpu_bus: missing shout_meta for table_id={table_id}"))?;
            if n_side == 0 || !n_side.is_power_of_two() {
                return Err(format!(
                    "shared_cpu_bus: shout n_side must be power-of-two, got {n_side}"
                ));
            }
            let ell = n_side.trailing_zeros() as usize;
            let ell_addr = d * ell;
            let lanes = bus
                .shout_cpu
                .get(table_id)
                .map(|v| v.len())
                .unwrap_or(0)
                .max(1);
            shout_shapes.push(ShoutInstanceShape {
                ell_addr,
                lanes,
                n_vals: riscv_trace_lookup_n_vals_for_table_id(*table_id),
                addr_group: bus.shout_addr_groups.get(table_id).copied(),
                selector_group: bus.shout_selector_groups.get(table_id).copied(),
            });
        }

        let mut twist_ell_addrs_and_lanes = Vec::with_capacity(mem_ids.len());
        for mem_id in &mem_ids {
            let layout = bus
                .mem_layouts
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing mem_layout for mem_id={mem_id}"))?;
            if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
                return Err(format!(
                    "shared_cpu_bus: twist n_side must be power-of-two, got {}",
                    layout.n_side
                ));
            }
            let ell = layout.n_side.trailing_zeros() as usize;
            let ell_addr = layout.d * ell;
            twist_ell_addrs_and_lanes.push((ell_addr, layout.lanes.max(1)));
        }

        let layout = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
            self.ccs.m,
            self.m_in,
            chunk_size,
            shout_shapes,
            twist_ell_addrs_and_lanes,
        )?;
        Ok((table_ids, mem_ids, layout))
    }

    pub fn with_shared_cpu_bus(mut self, cfg: SharedCpuBusConfig<F>, chunk_size: usize) -> Result<Self, String> {
        if chunk_size == 0 {
            return Err("shared_cpu_bus: chunk_size must be >= 1".into());
        }
        if cfg.const_one_col >= self.m_in {
            return Err(format!(
                "shared_cpu_bus: const_one_col={} must be < m_in={}",
                cfg.const_one_col, self.m_in
            ));
        }

        let (table_ids, mem_ids, layout) = self.shared_bus_schema(&cfg, chunk_size)?;
        let bus_base = layout.bus_base;

        // Validate initial memory keys.
        for ((mem_id, addr), _val) in cfg.initial_mem.iter() {
            let layout = cfg
                .mem_layouts
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: initial_mem refers to unknown mem_id={mem_id}"))?;
            if (*addr as usize) >= layout.k {
                return Err(format!(
                    "shared_cpu_bus: initial_mem out of range for mem_id={mem_id}: addr={addr} >= k={}",
                    layout.k
                ));
            }
        }

        fn validate_cpu_binding_cols(
            kind: &str,
            id: u32,
            bus_base: usize,
            chunk_size: usize,
            cols: &[(&str, usize, bool)],
        ) -> Result<(), String> {
            let max_step_offset = chunk_size
                .checked_sub(1)
                .ok_or_else(|| "shared_cpu_bus: chunk_size must be >= 1".to_string())?;
            for (label, col, allow_bus_overlap) in cols {
                if *col == CPU_BUS_COL_DISABLED {
                    continue;
                }
                let max_col = col
                    .checked_add(max_step_offset)
                    .ok_or_else(|| format!("shared_cpu_bus: {kind} binding for id={id} overflows usize"))?;
                if !allow_bus_overlap && max_col >= bus_base {
                    return Err(format!(
                        "shared_cpu_bus: {kind} binding for id={id} uses {label}={col} (max={max_col}), but bus_base={bus_base} (CPU bindings must be < bus_base to avoid overlapping the bus tail)"
                    ));
                }
            }
            Ok(())
        }

        // Build per-lane binding vectors in canonical order (id-sorted, then lane index).
        let total_shout_lanes: usize = table_ids
            .iter()
            .map(|id| cfg.shout_cpu.get(id).map(|v| v.len().max(1)).unwrap_or(1))
            .sum();
        let mut shout_cpu: Vec<Option<ShoutCpuBinding>> = Vec::with_capacity(total_shout_lanes);
        for table_id in &table_ids {
            let bindings = cfg
                .shout_cpu
                .get(table_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if bindings.is_empty() {
                shout_cpu.push(None);
                continue;
            }
            for (lane_idx, b) in bindings.iter().enumerate() {
                let mut cols = vec![("has_lookup", b.has_lookup, false), ("val", b.val, false)];
                if let Some(addr) = b.addr {
                    cols.push(("addr", addr, false));
                }
                validate_cpu_binding_cols(
                    &format!("shout_cpu[lane={lane_idx}]"),
                    *table_id,
                    bus_base,
                    chunk_size,
                    &cols,
                )?;
                shout_cpu.push(Some(b.clone()));
            }
        }
        let total_twist_lanes: usize = mem_ids
            .iter()
            .map(|mem_id| {
                cfg.mem_layouts
                    .get(mem_id)
                    .map(|l| l.lanes.max(1))
                    .unwrap_or(0)
            })
            .sum();
        let mut twist_cpu: Vec<TwistCpuBinding> = Vec::with_capacity(total_twist_lanes);
        for mem_id in &mem_ids {
            let layout = cfg
                .mem_layouts
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing mem_layout for mem_id={mem_id}"))?;
            let lanes = layout.lanes.max(1);
            let bindings = cfg
                .twist_cpu
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing twist_cpu binding for mem_id={mem_id}"))?;
            if bindings.len() != lanes {
                return Err(format!(
                    "shared_cpu_bus: twist_cpu bindings for mem_id={mem_id} has len={}, expected lanes={lanes}",
                    bindings.len()
                ));
            }
            for (lane_idx, b) in bindings.iter().enumerate() {
                let mut cols = vec![
                    // Selector columns may intentionally source from bus-tail decode lookups.
                    ("has_read", b.has_read, true),
                    ("has_write", b.has_write, true),
                    ("read_addr", b.read_addr, false),
                    ("write_addr", b.write_addr, false),
                    ("rv", b.rv, false),
                    ("wv", b.wv, false),
                ];
                if let Some(inc) = b.inc {
                    cols.push(("inc", inc, false));
                }
                let kind = format!("twist_cpu[lane={lane_idx}]");
                validate_cpu_binding_cols(&kind, *mem_id, bus_base, chunk_size, &cols)?;
                twist_cpu.push(b.clone());
            }
        }

        // Catch typos: refuse extra bindings for unknown ids.
        for table_id in cfg.shout_cpu.keys() {
            if !table_ids.contains(table_id) {
                return Err(format!(
                    "shared_cpu_bus: shout_cpu has binding for unknown table_id={table_id}"
                ));
            }
        }
        for mem_id in cfg.twist_cpu.keys() {
            if !mem_ids.contains(mem_id) {
                return Err(format!(
                    "shared_cpu_bus: twist_cpu has binding for unknown mem_id={mem_id}"
                ));
            }
        }

        // Build metadata-only instances for constraint injection (steps=chunk_size, comms empty).
        let mut lut_insts: Vec<LutInstance<Cmt, F>> = Vec::with_capacity(table_ids.len());
        for table_id in &table_ids {
            let (d, n_side) = self
                .shout_meta
                .get(table_id)
                .copied()
                .ok_or_else(|| format!("shared_cpu_bus: missing shout_meta for table_id={table_id}"))?;
            let ell = n_side.trailing_zeros() as usize;
            let lanes = cfg
                .shout_cpu
                .get(table_id)
                .map(|v| v.len())
                .unwrap_or(0)
                .max(1);
            lut_insts.push(LutInstance {
                table_id: *table_id,
                comms: Vec::new(),
                k: 0,
                d,
                n_side,
                steps: chunk_size,
                lanes,
                ell,
                table_spec: self.shout_specs.get(table_id).cloned(),
                table: Vec::new(),
                table_digest: None,
                addr_group: cfg.shout_addr_groups.get(table_id).copied(),
                selector_group: cfg.shout_selector_groups.get(table_id).copied(),
            });
        }

        let mut mem_insts: Vec<MemInstance<Cmt, F>> = Vec::with_capacity(mem_ids.len());
        for mem_id in &mem_ids {
            let layout = cfg
                .mem_layouts
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing mem_layout for mem_id={mem_id}"))?;
            let ell = layout.n_side.trailing_zeros() as usize;
            mem_insts.push(MemInstance {
                mem_id: *mem_id,
                comms: Vec::new(),
                k: layout.k,
                d: layout.d,
                n_side: layout.n_side,
                steps: chunk_size,
                lanes: layout.lanes.max(1),
                ell,
                init: MemInit::Zero,
                init_digest: None,
                guest_addr_remap: None,
            });
        }

        self.ccs = extend_ccs_with_shared_cpu_bus_constraints_optional_shout(
            &self.ccs,
            self.m_in,
            cfg.const_one_col,
            &shout_cpu,
            &twist_cpu,
            &lut_insts,
            &mem_insts,
            &cfg.shout_addr_groups,
            &cfg.shout_selector_groups,
        )
        .map_err(|e| format!("shared_cpu_bus: failed to inject constraints: {e}"))?;

        self.shared_cpu_bus = Some(SharedCpuBusState {
            cfg,
            table_ids,
            mem_ids,
            layout,
        });
        Ok(self)
    }
}

// R1csCpu implementation specifically for Goldilocks field because neo_ajtai::decomp_b uses Goldilocks
// The trait impl needs to be generic F, but decomp_b expects &[Fq] (Goldilocks).
// We need to cast F to Goldilocks if possible, or restrict F to Goldilocks.
// Given neo-ajtai is hardcoded to Goldilocks (Fq), we should restrict here.

impl<Cmt, L, Key> CpuArithmetization<Goldilocks, Cmt, Key> for R1csCpu<Goldilocks, Cmt, L, Key>
where
    L: SModuleHomomorphism<Goldilocks, Cmt>,
    Key: Copy + TryInto<u128> + Eq + Send + Sync + 'static,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64, Key>,
        chunk_size: usize,
    ) -> Result<Vec<(CcsClaim<Cmt, Goldilocks>, CcsWitness<Goldilocks>)>, Self::Error> {
        if chunk_size == 0 {
            return Err("chunk_size must be >= 1".into());
        }

        // Shared CPU-bus bookkeeping (optional).
        let shared = self.shared_cpu_bus.as_ref();
        if let Some(shared) = shared {
            if shared.layout.chunk_size != chunk_size {
                return Err(format!(
                    "shared_cpu_bus: chunk_size mismatch (cpu configured with {}, got {})",
                    shared.layout.chunk_size, chunk_size
                ));
            }
        }

        let mut mcss = Vec::with_capacity(trace.steps.len().div_ceil(chunk_size));

        // Track sparse memory state across the full trace to compute inc_at_write_addr.
        let mut mem_state: HashMap<u32, HashMap<u64, Goldilocks>> = HashMap::new();
        if let Some(shared) = shared {
            for ((mem_id, addr), &val) in shared.cfg.initial_mem.iter() {
                if val == Goldilocks::ZERO {
                    continue;
                }
                let layout = shared
                    .cfg
                    .mem_layouts
                    .get(mem_id)
                    .ok_or_else(|| format!("shared_cpu_bus: initial_mem refers to unknown mem_id={mem_id}"))?;
                if (*addr as usize) >= layout.k {
                    return Err(format!(
                        "shared_cpu_bus: initial_mem out of range for mem_id={mem_id}: addr={addr} >= k={}",
                        layout.k
                    ));
                }
                mem_state.entry(*mem_id).or_default().insert(*addr, val);
            }
        }

        let mut chunk_start = 0usize;
        while chunk_start < trace.steps.len() {
            let chunk_end = (chunk_start + chunk_size).min(trace.steps.len());
            let chunk = &trace.steps[chunk_start..chunk_end];

            // 1) Build witness z for this chunk.
            let mut z_vec = (self.chunk_to_witness)(chunk);

            // Allow witness builders to omit trailing dummy variables (including the shared-bus tail).
            if z_vec.len() < self.m_in {
                return Err(format!(
                    "Witness length {} is shorter than m_in={}",
                    z_vec.len(),
                    self.m_in
                ));
            }
            if z_vec.len() > self.ccs.m {
                return Err(format!(
                    "Witness length {} exceeds CCS width {}",
                    z_vec.len(),
                    self.ccs.m
                ));
            }
            let z_len_before_resize = z_vec.len();
            if z_vec.len() != self.ccs.m {
                z_vec.resize(self.ccs.m, Goldilocks::ZERO);
            }

            // Force the constant-one public input (required by shared-bus constraints and guardrails).
            if let Some(shared) = shared {
                if shared.cfg.const_one_col >= self.m_in {
                    return Err(format!(
                        "shared_cpu_bus: const_one_col={} must be < m_in={}",
                        shared.cfg.const_one_col, self.m_in
                    ));
                }
                z_vec[shared.cfg.const_one_col] = Goldilocks::ONE;
            }

            // 2) Overwrite the shared bus tail from the trace events.
            if let Some(shared) = shared {
                let bus_base = shared.layout.bus_base;
                let bus_region_len = shared.layout.bus_region_len();
                debug_assert_eq!(bus_base + bus_region_len, self.ccs.m);

                // Zero the bus tail so that inactive instances/ports are guaranteed zero.
                //
                // Common case: CPU witness builders omit the bus tail and return a vector of length `bus_base`.
                // In that case, the `resize` above already filled the entire bus region with zeros.
                if z_len_before_resize > bus_base {
                    for i in 0..bus_region_len {
                        z_vec[bus_base + i] = Goldilocks::ZERO;
                    }
                }

                // Scratch buffers for per-step event lookup (avoid per-step HashMap allocation).
                #[derive(Clone)]
                struct ShoutLaneEvent {
                    key: u128,
                    vals: Vec<Option<Goldilocks>>,
                }
                let mut shout_events: Vec<Vec<Option<ShoutLaneEvent>>> = shared
                    .layout
                    .shout_cols
                    .iter()
                    .map(|inst| vec![None; inst.lanes.len()])
                    .collect();
                let mut used_shout: Vec<usize> = vec![0; shout_events.len()];
                let mut twist_reads: Vec<Vec<Option<(u64, Goldilocks)>>> = shared
                    .layout
                    .twist_cols
                    .iter()
                    .map(|inst| vec![None; inst.lanes.len()])
                    .collect();
                let mut twist_writes: Vec<Vec<Option<(u64, Goldilocks)>>> = shared
                    .layout
                    .twist_cols
                    .iter()
                    .map(|inst| vec![None; inst.lanes.len()])
                    .collect();

                for (j, step) in chunk.iter().enumerate() {
                    used_shout.fill(0);
                    for lanes in shout_events.iter_mut() {
                        lanes.fill(None);
                    }
                    for lanes in twist_reads.iter_mut() {
                        lanes.fill(None);
                    }
                    for lanes in twist_writes.iter_mut() {
                        lanes.fill(None);
                    }

                    // Collect Shout events keyed by (sorted) table_id list.
                    for ev in &step.shout_events {
                        let id = ev.shout_id.0;
                        let key = ev
                            .key
                            .try_into()
                            .expect("shared-bus shout lane keys must fit in u128");
                        let idx = shared.table_ids.binary_search(&id).map_err(|_| {
                            format!("unexpected shout_id={id} in one step (chunk_start={chunk_start}, j={j})")
                        })?;
                        let lanes = shout_events[idx].len();
                        let n_vals = shared
                            .layout
                            .shout_cols
                            .get(idx)
                            .and_then(|inst_cols| inst_cols.lanes.first())
                            .map(|lane| lane.vals.len())
                            .unwrap_or(1)
                            .max(1);
                        let slot_idx = used_shout[idx];
                        let total_slots = lanes.checked_mul(n_vals).ok_or_else(|| {
                            format!("shout slot overflow for shout_id={id} (lanes={lanes}, n_vals={n_vals})")
                        })?;
                        if slot_idx >= total_slots {
                            return Err(format!(
                                "too many shout events for shout_id={id} in one step (chunk_start={chunk_start}, j={j}): lanes={lanes}, n_vals={n_vals}, total_slots={total_slots}"
                            ));
                        }
                        let lane_idx = slot_idx / n_vals;
                        let val_slot = slot_idx % n_vals;
                        let lane_entry = shout_events[idx][lane_idx].get_or_insert_with(|| ShoutLaneEvent {
                            key,
                            vals: vec![None; n_vals],
                        });
                        if lane_entry.key != key {
                            return Err(format!(
                                "mixed shout keys for shout_id={id} in one step (chunk_start={chunk_start}, j={j}, lane={lane_idx}): first_key={}, new_key={}",
                                lane_entry.key, key
                            ));
                        }
                        if lane_entry.vals[val_slot].is_some() {
                            return Err(format!(
                                "duplicate shout value slot for shout_id={id} in one step (chunk_start={chunk_start}, j={j}, lane={lane_idx}, val_slot={val_slot})"
                            ));
                        }
                        lane_entry.vals[val_slot] = Some(Goldilocks::from_u64(ev.value));
                        used_shout[idx] += 1;
                    }

                    // Collect Twist events keyed by (sorted) mem_id list.
                    for ev in &step.twist_events {
                        let id = ev.twist_id.0;
                        let idx = shared.mem_ids.binary_search(&id).map_err(|_| {
                            format!("unexpected twist_id={id} in one step (chunk_start={chunk_start}, j={j})")
                        })?;
                        match ev.kind {
                            neo_vm_trace::TwistOpKind::Read => {
                                let lanes = twist_reads
                                    .get_mut(idx)
                                    .ok_or_else(|| format!("missing twist read lanes for twist_id={id}"))?;

                                let lane_idx = if let Some(lane) = ev.lane {
                                    let lane_idx = usize::try_from(lane).map_err(|_| {
                                        format!(
                                            "invalid twist read lane for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lane={lane}"
                                        )
                                    })?;
                                    if lane_idx >= lanes.len() {
                                        return Err(format!(
                                            "twist read lane out of range for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lane={lane_idx}, lanes={}",
                                            lanes.len()
                                        ));
                                    }
                                    if lanes[lane_idx].is_some() {
                                        return Err(format!(
                                            "multiple twist reads for twist_id={id} in one step (chunk_start={chunk_start}, j={j}) in lane={lane_idx}"
                                        ));
                                    }
                                    lane_idx
                                } else {
                                    lanes.iter().position(|x| x.is_none()).ok_or_else(|| {
                                        format!(
                                            "too many twist reads for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lanes={}",
                                            lanes.len()
                                        )
                                    })?
                                };
                                lanes[lane_idx] = Some((ev.addr, Goldilocks::from_u64(ev.value)));
                            }
                            neo_vm_trace::TwistOpKind::Write => {
                                let lanes = twist_writes
                                    .get_mut(idx)
                                    .ok_or_else(|| format!("missing twist write lanes for twist_id={id}"))?;

                                if lanes.iter().flatten().any(|(addr, _)| *addr == ev.addr) {
                                    return Err(format!(
                                        "duplicate twist write addr for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): addr={}",
                                        ev.addr
                                    ));
                                }

                                let lane_idx = if let Some(lane) = ev.lane {
                                    let lane_idx = usize::try_from(lane).map_err(|_| {
                                        format!(
                                            "invalid twist write lane for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lane={lane}"
                                        )
                                    })?;
                                    if lane_idx >= lanes.len() {
                                        return Err(format!(
                                            "twist write lane out of range for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lane={lane_idx}, lanes={}",
                                            lanes.len()
                                        ));
                                    }
                                    if lanes[lane_idx].is_some() {
                                        return Err(format!(
                                            "multiple twist writes for twist_id={id} in one step (chunk_start={chunk_start}, j={j}) in lane={lane_idx}"
                                        ));
                                    }
                                    lane_idx
                                } else {
                                    lanes.iter().position(|x| x.is_none()).ok_or_else(|| {
                                        format!(
                                            "too many twist writes for twist_id={id} in one step (chunk_start={chunk_start}, j={j}): lanes={}",
                                            lanes.len()
                                        )
                                    })?
                                };
                                lanes[lane_idx] = Some((ev.addr, Goldilocks::from_u64(ev.value)));
                            }
                        }
                    }

                    let reg_read_pair_u32 = shared
                        .mem_ids
                        .binary_search(&crate::riscv::lookups::REG_ID.0)
                        .ok()
                        .and_then(|reg_idx| {
                            let lanes = twist_reads.get(reg_idx)?;
                            let lhs = lanes
                                .get(0)
                                .and_then(|slot| *slot)
                                .map(|(_, v)| v.as_canonical_u64() as u32)?;
                            let rhs = lanes
                                .get(1)
                                .and_then(|slot| *slot)
                                .map(|(_, v)| v.as_canonical_u64() as u32)?;
                            Some((lhs, rhs))
                        });

                    // Shout lanes: addr_bits, has_lookup, vals[0..n_vals).
                    for (i, table_id) in shared.table_ids.iter().enumerate() {
                        let inst_cols = &shared.layout.shout_cols[i];
                        let (d, n_side) = self
                            .shout_meta
                            .get(table_id)
                            .copied()
                            .ok_or_else(|| format!("missing shout_meta for table_id={table_id}"))?;
                        let ell = n_side.trailing_zeros() as usize;

                        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
                            if let Some(ref lane_event) = shout_events[i][lane_idx] {
                                let key = lane_event.key;
                                let primary_val = lane_event
                                    .vals
                                    .first()
                                    .and_then(|slot| *slot)
                                    .ok_or_else(|| {
                                        format!(
                                            "missing primary shout value for table_id={table_id} at step j={j}, lane={lane_idx}"
                                        )
                                    })?;
                                match self.shout_specs.get(table_id) {
                                    Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => {
                                        // Packed lanes should be derived from architectural operands, not key encoding.
                                        // This keeps packed synthesis correct when some opcodes migrate to combined keys.
                                        let val_u64 = primary_val.as_canonical_u64();
                                        let (lhs, rhs) = if *xlen == 32 {
                                            let (lhs, rhs) = reg_read_pair_u32.unwrap_or_else(|| {
                                                let (lhs_raw, rhs_raw) = uninterleave_bits(key as u128);
                                                (lhs_raw as u32, rhs_raw as u32)
                                            });
                                            (lhs as u64, rhs as u64)
                                        } else {
                                            uninterleave_bits(key as u128)
                                        };
                                        let packed_cols = crate::riscv::packed::build_rv_packed_cols::<Goldilocks>(
                                            *opcode, lhs, rhs, val_u64, *xlen,
                                        )
                                            .map_err(|e| {
                                                format!(
                                                    "packed shout column synthesis failed for table_id={table_id}, opcode={opcode:?}, step_j={j}: {e}"
                                                )
                                            })?;
                                        if packed_cols.len() != (shout_cols.addr_bits.end - shout_cols.addr_bits.start)
                                        {
                                            return Err(format!(
                                                "packed shout column width mismatch for table_id={table_id}: cols_len={} bus_ell_addr={}",
                                                packed_cols.len(),
                                                shout_cols.addr_bits.end - shout_cols.addr_bits.start
                                            ));
                                        }
                                        for (packed_idx, col_id) in shout_cols.addr_bits.clone().enumerate() {
                                            z_vec[shared.layout.bus_cell(col_id, j)] = packed_cols[packed_idx];
                                        }
                                    }
                                    Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => {
                                        return Err(format!(
                                            "RiscvOpcodeEventTablePacked is not supported in shared-bus R1csCpu path (table_id={table_id})"
                                        ));
                                    }
                                    _ => {
                                        write_addr_bits_dim_major_le_u128_into_bus(
                                            &mut z_vec,
                                            &shared.layout,
                                            shout_cols.addr_bits.clone(),
                                            j,
                                            key,
                                            d,
                                            n_side,
                                            ell,
                                        );
                                    }
                                }
                                z_vec[shared.layout.bus_cell(shout_cols.has_lookup, j)] = Goldilocks::ONE;
                                if lane_event.vals.len() != shout_cols.vals.len() {
                                    return Err(format!(
                                        "shared-bus shout value slot mismatch for table_id={table_id} at step j={j}, lane={lane_idx}: event_slots={} bus_slots={}",
                                        lane_event.vals.len(),
                                        shout_cols.vals.len()
                                    ));
                                }
                                for (val_slot, &col_id) in shout_cols.vals.iter().enumerate() {
                                    let slot_val = lane_event.vals[val_slot].ok_or_else(|| {
                                        format!(
                                            "missing shout value for table_id={table_id} at step j={j}, lane={lane_idx}, val_slot={val_slot}"
                                        )
                                    })?;
                                    z_vec[shared.layout.bus_cell(col_id, j)] = slot_val;
                                }
                            }
                        }
                    }

                    // Twist: ra_bits, wa_bits, has_read, has_write, wv, rv, inc_at_write_addr.
                    for (i, mem_id) in shared.mem_ids.iter().enumerate() {
                        let inst_cols = &shared.layout.twist_cols[i];
                        let layout = shared
                            .cfg
                            .mem_layouts
                            .get(mem_id)
                            .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
                        let ell = layout.n_side.trailing_zeros() as usize;

                        let st = mem_state.entry(*mem_id).or_default();
                        let mut writes_to_apply: Vec<(u64, Goldilocks)> = Vec::new();

                        for (lane_idx, twist_cols) in inst_cols.lanes.iter().enumerate() {
                            // Read port.
                            let (has_read, ra, rv) = if let Some((addr, val)) = twist_reads[i][lane_idx] {
                                (Goldilocks::ONE, addr, val)
                            } else {
                                (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                            };
                            if has_read == Goldilocks::ONE {
                                if ra >= layout.k as u64 {
                                    return Err(format!(
                                        "shared_cpu_bus: twist read addr out of range: mem_id={mem_id}, addr={ra}, k={}",
                                        layout.k
                                    ));
                                }
                                write_addr_bits_dim_major_le_into_bus(
                                    &mut z_vec,
                                    &shared.layout,
                                    twist_cols.ra_bits.clone(),
                                    j,
                                    ra,
                                    layout.d,
                                    layout.n_side,
                                    ell,
                                );
                                z_vec[shared.layout.bus_cell(twist_cols.rv, j)] = rv;
                            }

                            // Write port.
                            let (has_write, wa, wv) = if let Some((addr, val)) = twist_writes[i][lane_idx] {
                                (Goldilocks::ONE, addr, val)
                            } else {
                                (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                            };

                            let mut inc = Goldilocks::ZERO;
                            if has_write == Goldilocks::ONE {
                                if wa >= layout.k as u64 {
                                    return Err(format!(
                                        "shared_cpu_bus: twist write addr out of range: mem_id={mem_id}, addr={wa}, k={}",
                                        layout.k
                                    ));
                                }
                                write_addr_bits_dim_major_le_into_bus(
                                    &mut z_vec,
                                    &shared.layout,
                                    twist_cols.wa_bits.clone(),
                                    j,
                                    wa,
                                    layout.d,
                                    layout.n_side,
                                    ell,
                                );
                                z_vec[shared.layout.bus_cell(twist_cols.wv, j)] = wv;

                                let old = st.get(&wa).copied().unwrap_or(Goldilocks::ZERO);
                                inc = wv - old;
                                writes_to_apply.push((wa, wv));
                            }

                            z_vec[shared.layout.bus_cell(twist_cols.has_read, j)] = has_read;
                            z_vec[shared.layout.bus_cell(twist_cols.has_write, j)] = has_write;
                            z_vec[shared.layout.bus_cell(twist_cols.inc, j)] = inc;
                        }

                        for (wa, wv) in writes_to_apply {
                            if wv == Goldilocks::ZERO {
                                st.remove(&wa);
                            } else {
                                st.insert(wa, wv);
                            }
                        }
                    }
                }
            }

            // 3) Encode z -> Z matrix
            let m = z_vec.len(); // == ccs.m after padding

            // Validate m_in
            let m_in = self.m_in;
            if m_in > m {
                return Err(format!("m_in={} exceeds witness length m={}", m_in, m));
            }

            let z_mat = encode_vector_for_ccs_m(&self.params, self.ccs.m, &z_vec)
                .map_err(|e| format!("shared_cpu_bus: witness encode failed: {e}"))?;

            // 4) Commit to Z
            let c = self.committer.commit(&z_mat);

            // 5) Build Instance/Witness
            let x = z_vec[..m_in].to_vec();
            let w = z_vec[m_in..].to_vec();

            mcss.push((CcsClaim { c, x, m_in }, CcsWitness { w, Z: z_mat }));

            chunk_start = chunk_end;
        }

        Ok(mcss)
    }

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64, Key>,
    ) -> Result<Vec<(CcsClaim<Cmt, Goldilocks>, CcsWitness<Goldilocks>)>, Self::Error> {
        self.build_ccs_chunks(trace, 1)
    }

    fn shout_addr_groups(&self) -> &HashMap<u32, u64> {
        if let Some(shared) = self.shared_cpu_bus.as_ref() {
            &shared.cfg.shout_addr_groups
        } else {
            &self.lookup_shout_addr_groups
        }
    }

    fn shout_selector_groups(&self) -> &HashMap<u32, u64> {
        if let Some(shared) = self.shared_cpu_bus.as_ref() {
            &shared.cfg.shout_selector_groups
        } else {
            &self.lookup_shout_selector_groups
        }
    }

    type Error = String;
}
