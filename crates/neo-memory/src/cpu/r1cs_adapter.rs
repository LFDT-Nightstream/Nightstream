//! R1CS-based CPU adapter for the shared memory/lookup bus.
//!
//! This module provides an adapter that implements `CpuArithmetization` for a generic
//! R1CS-based CPU, allowing integration with Neo's shared CPU bus architecture.

use crate::addr::write_addr_bits_dim_major_le_into_bus;
use crate::builder::CpuArithmetization;
use crate::cpu::bus_layout::{build_bus_layout_for_instances, BusLayout};
use crate::cpu::constraints::{extend_ccs_with_shared_cpu_bus_constraints, ShoutCpuBinding, TwistCpuBinding};
use crate::mem_init::MemInit;
use crate::plain::LutTable;
use crate::plain::PlainMemLayout;
use crate::witness::{LutInstance, MemInstance};
use neo_ajtai::{decomp_b, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64};
use p3_goldilocks::Goldilocks;
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
    pub shout_cpu: HashMap<u32, ShoutCpuBinding>,
    /// Per-memory CPU→bus bindings (twist_id -> binding).
    ///
    /// The bus tail contains one Twist instance per `mem_id` in `mem_layouts`.
    pub twist_cpu: HashMap<u32, TwistCpuBinding>,
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
pub struct R1csCpu<F, Cmt, L>
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

    /// Cached Shout tables for verification: shout_id -> (key -> val)
    /// Used to ensure trace Shout lookups are valid against the table.
    pub shout_cache: HashMap<u32, HashMap<u64, F>>,

    /// Shout table metadata needed to write the shared CPU bus (d, n_side).
    pub shout_meta: HashMap<u32, (usize, usize)>,

    /// Optional shared CPU-bus configuration.
    /// When present, we overwrite a reserved tail segment of `z_vec` with Twist/Shout access rows
    /// (in deterministic id order) before Ajtai decomposition + commitment.
    shared_cpu_bus: Option<SharedCpuBusState<F>>,

    /// Function to map a step trace to the full witness z = (x, w).
    /// The witness MUST satisfy the CCS relation.
    pub step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,

    _phantom: PhantomData<Cmt>,
}

impl<F, Cmt, L> R1csCpu<F, Cmt, L>
where
    F: PrimeField + PrimeField64 + Copy,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub fn new(
        ccs: CcsStructure<F>,
        params: NeoParams,
        committer: L,
        m_in: usize,
        tables: &HashMap<u32, LutTable<F>>,
        step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,
    ) -> Self {
        // Build fast Shout cache from tables
        let mut shout_cache = HashMap::new();
        let mut shout_meta = HashMap::new();
        for (id, table) in tables {
            // Assuming table content is dense (val = content[addr])
            // LutTable currently has `content: Vec<F>`.
            // We'll map key -> val.
            let mut map = HashMap::new();
            for (key, val) in table.content.iter().enumerate() {
                map.insert(key as u64, *val);
            }
            shout_cache.insert(*id, map);
            shout_meta.insert(*id, (table.d, table.n_side));
        }

        Self {
            ccs,
            params,
            committer,
            m_in,
            shout_cache,
            shout_meta,
            shared_cpu_bus: None,
            step_to_witness,
            _phantom: PhantomData,
        }
    }

    fn shared_bus_schema(&self, bus: &SharedCpuBusConfig<F>) -> Result<(Vec<u32>, Vec<u32>, BusLayout), String> {
        let mut table_ids: Vec<u32> = self.shout_cache.keys().copied().collect();
        table_ids.sort_unstable();
        let mut mem_ids: Vec<u32> = bus.mem_layouts.keys().copied().collect();
        mem_ids.sort_unstable();

        let mut shout_ell_addrs = Vec::with_capacity(table_ids.len());
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
            shout_ell_addrs.push(ell_addr);
        }

        let mut twist_ell_addrs = Vec::with_capacity(mem_ids.len());
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
            twist_ell_addrs.push(ell_addr);
        }

        let layout = build_bus_layout_for_instances(self.ccs.m, self.m_in, 1, shout_ell_addrs, twist_ell_addrs)?;
        Ok((table_ids, mem_ids, layout))
    }

    pub fn with_shared_cpu_bus(mut self, cfg: SharedCpuBusConfig<F>) -> Result<Self, String> {
        if cfg.const_one_col >= self.m_in {
            return Err(format!(
                "shared_cpu_bus: const_one_col={} must be < m_in={}",
                cfg.const_one_col, self.m_in
            ));
        }

        let (table_ids, mem_ids, layout) = self.shared_bus_schema(&cfg)?;
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
            cols: &[(&str, usize)],
        ) -> Result<(), String> {
            for (label, col) in cols {
                if *col >= bus_base {
                    return Err(format!(
                        "shared_cpu_bus: {kind} binding for id={id} uses {label}={col}, but bus_base={bus_base} (CPU bindings must be < bus_base to avoid overlapping the bus tail)"
                    ));
                }
            }
            Ok(())
        }

        // Build per-instance binding vectors in canonical order (id-sorted).
        let mut shout_cpu: Vec<ShoutCpuBinding> = Vec::with_capacity(table_ids.len());
        for table_id in &table_ids {
            let b = cfg
                .shout_cpu
                .get(table_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing shout_cpu binding for table_id={table_id}"))?;
            validate_cpu_binding_cols(
                "shout_cpu",
                *table_id,
                bus_base,
                &[("has_lookup", b.has_lookup), ("addr", b.addr), ("val", b.val)],
            )?;
            shout_cpu.push(b.clone());
        }
        let mut twist_cpu: Vec<TwistCpuBinding> = Vec::with_capacity(mem_ids.len());
        for mem_id in &mem_ids {
            let b = cfg
                .twist_cpu
                .get(mem_id)
                .ok_or_else(|| format!("shared_cpu_bus: missing twist_cpu binding for mem_id={mem_id}"))?;
            let mut cols = vec![
                ("has_read", b.has_read),
                ("has_write", b.has_write),
                ("read_addr", b.read_addr),
                ("write_addr", b.write_addr),
                ("rv", b.rv),
                ("wv", b.wv),
            ];
            if let Some(inc) = b.inc {
                cols.push(("inc", inc));
            }
            validate_cpu_binding_cols("twist_cpu", *mem_id, bus_base, &cols)?;
            twist_cpu.push(b.clone());
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

        // Build metadata-only instances for constraint injection (steps=1, comms empty).
        let mut lut_insts: Vec<LutInstance<Cmt, F>> = Vec::with_capacity(table_ids.len());
        for table_id in &table_ids {
            let (d, n_side) = self
                .shout_meta
                .get(table_id)
                .copied()
                .ok_or_else(|| format!("shared_cpu_bus: missing shout_meta for table_id={table_id}"))?;
            let ell = n_side.trailing_zeros() as usize;
            let k = self.shout_cache.get(table_id).map(|m| m.len()).unwrap_or(0);
            lut_insts.push(LutInstance {
                comms: Vec::new(),
                k,
                d,
                n_side,
                steps: 1,
                ell,
                table_spec: None,
                table: Vec::new(),
                _phantom: PhantomData,
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
                comms: Vec::new(),
                k: layout.k,
                d: layout.d,
                n_side: layout.n_side,
                steps: 1,
                ell,
                init: MemInit::Zero,
                _phantom: PhantomData,
            });
        }

        self.ccs = extend_ccs_with_shared_cpu_bus_constraints(
            &self.ccs,
            self.m_in,
            cfg.const_one_col,
            &shout_cpu,
            &twist_cpu,
            &lut_insts,
            &mem_insts,
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

impl<Cmt, L> CpuArithmetization<Goldilocks, Cmt> for R1csCpu<Goldilocks, Cmt, L>
where
    L: SModuleHomomorphism<Goldilocks, Cmt>,
{
    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64>,
        chunk_size: usize,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        if chunk_size != 1 {
            return Err(format!(
                "R1csCpu does not support chunk_size={} (expected 1)",
                chunk_size
            ));
        }
        self.build_ccs_steps(trace)
    }

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64>,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        let mut mcss = Vec::with_capacity(trace.steps.len());

        // Shared CPU-bus bookkeeping (optional).
        let shared = self.shared_cpu_bus.as_ref();
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

        for step in &trace.steps {
            // 1. Verify Shout lookups in this step against the table cache
            for shout in &step.shout_events {
                if let Some(table) = self.shout_cache.get(&shout.shout_id.0) {
                    // Check val == table[key]
                    if let Some(&expected) = table.get(&shout.key) {
                        let trace_val = Goldilocks::from_u64(shout.value);
                        if trace_val != expected {
                            return Err(format!(
                                "Shout mismatch at step pc={:?}: shout {} key {} has {}, trace has {}",
                                step.pc_before,
                                shout.shout_id.0,
                                shout.key,
                                expected.as_canonical_u64(),
                                trace_val.as_canonical_u64()
                            ));
                        }
                    } else {
                        return Err(format!(
                            "Shout key {} not found in shout {}",
                            shout.key, shout.shout_id.0
                        ));
                    }
                } else {
                    return Err(format!("Missing Shout table for shout_id {}", shout.shout_id.0));
                }
            }

            // 2. Build witness z
            let mut z_vec = (self.step_to_witness)(step);

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
            if z_vec.len() != self.ccs.m {
                z_vec.resize(self.ccs.m, Goldilocks::ZERO);
            }

            // Enforce the constant-one public input (required by shared-bus constraints and guardrails).
            if let Some(shared) = shared {
                if shared.cfg.const_one_col >= self.m_in {
                    return Err(format!(
                        "shared_cpu_bus: const_one_col={} must be < m_in={}",
                        shared.cfg.const_one_col, self.m_in
                    ));
                }
                z_vec[shared.cfg.const_one_col] = Goldilocks::ONE;
            }

            // 2.1 Overwrite the shared CPU-bus tail (if enabled).
            if let Some(shared) = shared {
                let bus_base = shared.layout.bus_base;
                let bus_region_len = shared.layout.bus_region_len();
                debug_assert_eq!(bus_base + bus_region_len, self.ccs.m);

                // Zero the bus tail so that inactive instances/ports are guaranteed zero.
                for i in 0..bus_region_len {
                    z_vec[bus_base + i] = Goldilocks::ZERO;
                }

                // Build per-step maps for fast lookup.
                let mut shout_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                for ev in &step.shout_events {
                    let id = ev.shout_id.0;
                    if shout_by_id
                        .insert(id, (ev.key, Goldilocks::from_u64(ev.value)))
                        .is_some()
                    {
                        return Err(format!("multiple shout events for shout_id={id} in one step"));
                    }
                }

                let mut read_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                let mut write_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                for ev in &step.twist_events {
                    let id = ev.twist_id.0;
                    match ev.kind {
                        neo_vm_trace::TwistOpKind::Read => {
                            if read_by_id
                                .insert(id, (ev.addr, Goldilocks::from_u64(ev.value)))
                                .is_some()
                            {
                                return Err(format!("multiple twist reads for twist_id={id} in one step"));
                            }
                        }
                        neo_vm_trace::TwistOpKind::Write => {
                            if write_by_id
                                .insert(id, (ev.addr, Goldilocks::from_u64(ev.value)))
                                .is_some()
                            {
                                return Err(format!("multiple twist writes for twist_id={id} in one step"));
                            }
                        }
                    }
                }

                // Shout: addr_bits, has_lookup, val.
                for (i, table_id) in shared.table_ids.iter().enumerate() {
                    let shout_cols = &shared.layout.shout_cols[i];
                    let (d, n_side) = self
                        .shout_meta
                        .get(table_id)
                        .copied()
                        .ok_or_else(|| format!("missing shout_meta for table_id={table_id}"))?;
                    let ell = n_side.trailing_zeros() as usize;

                    if let Some((key, val)) = shout_by_id.get(table_id).copied() {
                        write_addr_bits_dim_major_le_into_bus(
                            &mut z_vec,
                            &shared.layout,
                            shout_cols.addr_bits.clone(),
                            0,
                            key,
                            d,
                            n_side,
                            ell,
                        );
                        z_vec[shared.layout.bus_cell(shout_cols.has_lookup, 0)] = Goldilocks::ONE;
                        z_vec[shared.layout.bus_cell(shout_cols.val, 0)] = val;
                    }
                }

                // Twist: ra_bits, wa_bits, has_read, has_write, wv, rv, inc_at_write_addr.
                for (i, mem_id) in shared.mem_ids.iter().enumerate() {
                    let twist_cols = &shared.layout.twist_cols[i];
                    let layout = shared
                        .cfg
                        .mem_layouts
                        .get(mem_id)
                        .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
                    let ell = layout.n_side.trailing_zeros() as usize;

                    // Read port.
                    let (has_read, ra, rv) = if let Some((addr, val)) = read_by_id.get(mem_id).copied() {
                        (Goldilocks::ONE, addr, val)
                    } else {
                        (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                    };
                    if has_read == Goldilocks::ONE {
                        write_addr_bits_dim_major_le_into_bus(
                            &mut z_vec,
                            &shared.layout,
                            twist_cols.ra_bits.clone(),
                            0,
                            ra,
                            layout.d,
                            layout.n_side,
                            ell,
                        );
                        z_vec[shared.layout.bus_cell(twist_cols.rv, 0)] = rv;
                    }

                    // Write port.
                    let (has_write, wa, wv) = if let Some((addr, val)) = write_by_id.get(mem_id).copied() {
                        (Goldilocks::ONE, addr, val)
                    } else {
                        (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                    };
                    if has_write == Goldilocks::ONE {
                        write_addr_bits_dim_major_le_into_bus(
                            &mut z_vec,
                            &shared.layout,
                            twist_cols.wa_bits.clone(),
                            0,
                            wa,
                            layout.d,
                            layout.n_side,
                            ell,
                        );
                        z_vec[shared.layout.bus_cell(twist_cols.wv, 0)] = wv;
                    }

                    z_vec[shared.layout.bus_cell(twist_cols.has_read, 0)] = has_read;
                    z_vec[shared.layout.bus_cell(twist_cols.has_write, 0)] = has_write;

                    let mut inc = Goldilocks::ZERO;
                    if has_write == Goldilocks::ONE && (wa as usize) < layout.k {
                        let st = mem_state.entry(*mem_id).or_default();
                        let old = st.get(&wa).copied().unwrap_or(Goldilocks::ZERO);
                        inc = wv - old;
                        if wv == Goldilocks::ZERO {
                            st.remove(&wa);
                        } else {
                            st.insert(wa, wv);
                        }
                    }
                    z_vec[shared.layout.bus_cell(twist_cols.inc, 0)] = inc;
                }
            }

            // 3. Decompose z -> Z matrix
            let d = self.params.d as usize;
            let m = z_vec.len(); // == ccs.m after padding

            // Validate m_in
            let m_in = self.m_in;
            if m_in > m {
                return Err(format!("m_in={} exceeds witness length m={}", m_in, m));
            }

            // Decompose: Z is d x m
            let z_digits = decomp_b(&z_vec, self.params.b, d, DecompStyle::Balanced);

            // Convert to Mat (row-major d x m)
            // decomp_b returns digits "per element", i.e. column-major for the (d × m) matrix:
            // z_digits[c*d + r] = digit r (row) of value c (column).
            let mut mat_data = vec![Goldilocks::ZERO; d * m];
            for c in 0..m {
                for r in 0..d {
                    mat_data[r * m + c] = z_digits[c * d + r];
                }
            }
            let z_mat = Mat::from_row_major(d, m, mat_data);

            // 4. Commit to Z
            let c = self.committer.commit(&z_mat);

            // 5. Build Instance/Witness
            // Split z into public inputs x and private witness w
            let x = z_vec[..m_in].to_vec();
            let w = z_vec[m_in..].to_vec();

            mcss.push((McsInstance { c, x, m_in }, McsWitness { w, Z: z_mat }));
        }

        Ok(mcss)
    }
    type Error = String;
}
