use super::{CcsBuilder, FoldingSession, SharedBusResources, WitnessLayout};
use crate::PiCcsError;
use neo_ccs::CcsStructure;
use neo_ccs::traits::SModuleHomomorphism;
use neo_memory::cpu::{CpuConstraintBuilder, R1csCpu, SharedCpuBusConfig, ShoutCpuBinding, TwistCpuBinding};
use neo_memory::plain::LutTable;
use neo_memory::witness::LutTableSpec;
use neo_params::NeoParams;
use neo_vm_trace::StepTrace;
use std::collections::HashMap;
use std::sync::Arc;

/// Circuit-level interface for building an ergonomic shared-CPU-bus R1CS prover.
///
/// This is designed to keep `FoldingSession` as the front-facing API:
/// callers compile a circuit once, then execute/prove using session methods.
pub trait NeoCircuit: Send + Sync + 'static {
    type Layout: WitnessLayout + Clone + Send + Sync + 'static;

    /// Folding chunk size (lanes per folding step).
    fn chunk_size(&self) -> usize;

    /// Construct the witness layout for this circuit.
    fn layout(&self) -> Self::Layout {
        <Self::Layout as WitnessLayout>::new_layout()
    }

    /// Column index inside the public prefix that is fixed to 1.
    fn const_one_col(&self, layout: &Self::Layout) -> usize;

    /// Declare shared-bus resources (Twist layouts/init + Shout tables/specs).
    fn resources(&self, resources: &mut SharedBusResources);

    /// Per-instance CPU→bus bindings (id -> base column indices).
    fn cpu_bindings(
        &self,
        layout: &Self::Layout,
    ) -> Result<(HashMap<u32, Vec<ShoutCpuBinding>>, HashMap<u32, Vec<TwistCpuBinding>>), String>;

    /// Define the CPU semantic constraints (not including shared-bus injected constraints).
    fn define_cpu_constraints(&self, cs: &mut CcsBuilder<super::F>, layout: &Self::Layout) -> Result<(), String>;

    /// Build the CPU witness prefix z[0..USED_COLS) for a single trace chunk.
    ///
    /// The shared-bus tail is filled separately by `neo_memory::cpu::R1csCpu`.
    fn build_witness_prefix(&self, layout: &Self::Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<super::F>, String>;
}

/// Shared preprocessing for a shared-bus R1CS circuit (no commitment key material).
#[derive(Clone, Debug)]
pub struct SharedBusR1csPreprocessing<C: NeoCircuit> {
    pub circuit: Arc<C>,
    pub layout: C::Layout,
    pub resources: SharedBusResources,
    pub shout_cpu: HashMap<u32, Vec<ShoutCpuBinding>>,
    pub twist_cpu: HashMap<u32, Vec<TwistCpuBinding>>,
    pub m_in: usize,
    pub const_one_col: usize,
    pub chunk_size: usize,
    pub base_ccs: CcsStructure<super::F>,
}

impl<C: NeoCircuit> SharedBusR1csPreprocessing<C> {
    #[inline]
    pub fn m(&self) -> usize {
        self.base_ccs.m
    }

    /// Build a prover-side `R1csCpu` (injecting shared-bus constraints).
    ///
    /// Note: witness building errors currently panic because `neo_memory::cpu::R1csCpu` requires a
    /// `Fn(&[StepTrace]) -> Vec<F>` callback. If we evolve the arithmetization interface to return
    /// `Result`, this can become a structured error instead.
    pub fn into_prover<L>(self, params: NeoParams, committer: L) -> Result<SharedBusR1csProver<L, C>, PiCcsError>
    where
        L: SModuleHomomorphism<super::F, super::Cmt> + Clone + Sync,
    {
        let chunk_to_witness = {
            let circuit = Arc::clone(&self.circuit);
            let layout = self.layout.clone();
            Box::new(move |chunk: &[StepTrace<u64, u64>]| {
                circuit
                    .build_witness_prefix(&layout, chunk)
                    .unwrap_or_else(|e| panic!("build_witness_prefix failed: {e}"))
            })
        };

        let cpu = R1csCpu::<super::F, super::Cmt, _>::new(
            self.base_ccs,
            params.clone(),
            committer.clone(),
            self.m_in,
            &self.resources.lut_tables,
            &self.resources.lut_table_specs,
            chunk_to_witness,
        )
        .with_shared_cpu_bus(
            SharedCpuBusConfig {
                mem_layouts: self.resources.mem_layouts.clone(),
                initial_mem: self.resources.initial_mem.clone(),
                const_one_col: self.const_one_col,
                shout_cpu: self.shout_cpu.clone(),
                twist_cpu: self.twist_cpu.clone(),
            },
            self.chunk_size,
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("R1csCpu shared-bus config failed: {e}")))?;

        Ok(SharedBusR1csProver {
            circuit: self.circuit,
            layout: self.layout,
            resources: self.resources,
            cpu,
            params,
            committer,
        })
    }
}

/// Prover-side artifact (includes commitment key material).
pub struct SharedBusR1csProver<L, C>
where
    L: SModuleHomomorphism<super::F, super::Cmt> + Clone + Sync,
    C: NeoCircuit,
{
    pub circuit: Arc<C>,
    pub layout: C::Layout,
    pub resources: SharedBusResources,
    pub cpu: R1csCpu<super::F, super::Cmt, L>,
    pub params: NeoParams,
    pub committer: L,
}

impl<L, C> SharedBusR1csProver<L, C>
where
    L: SModuleHomomorphism<super::F, super::Cmt> + Clone + Sync,
    C: NeoCircuit,
{
    #[inline]
    pub fn ccs(&self) -> &CcsStructure<super::F> {
        &self.cpu.ccs
    }

    /// Execute the VM shard using this circuit and add step bundles to the given session.
    pub fn execute_into_session<V, Tw, Sh>(
        &self,
        session: &mut FoldingSession<L>,
        vm: V,
        twist: Tw,
        shout: Sh,
        max_steps: usize,
    ) -> Result<(), PiCcsError>
    where
        V: neo_vm_trace::VmCpu<u64, u64>,
        Tw: neo_vm_trace::Twist<u64, u64>,
        Sh: neo_vm_trace::Shout<u64>,
    {
        session.set_shared_bus_resources(self.resources.clone());
        session.execute_shard_shared_cpu_bus_configured(vm, twist, shout, max_steps, self.circuit.chunk_size(), &self.cpu)
    }
}

/// Shared preprocessing for a `NeoCircuit` in shared-bus R1CS mode.
pub fn preprocess_shared_bus_r1cs<C>(circuit: Arc<C>) -> Result<SharedBusR1csPreprocessing<C>, PiCcsError>
where
    C: NeoCircuit,
{
    let layout = circuit.layout();
    let m_in = <C::Layout as WitnessLayout>::M_IN;
    let used_cols = <C::Layout as WitnessLayout>::USED_COLS;
    let chunk_size = circuit.chunk_size();
    if chunk_size == 0 {
        return Err(PiCcsError::InvalidInput("chunk_size must be >= 1".into()));
    }

    let const_one_col = circuit.const_one_col(&layout);
    if const_one_col >= m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "const_one_col({const_one_col}) must be < m_in({m_in})"
        )));
    }

    let mut resources = SharedBusResources::new();
    circuit.resources(&mut resources);

    let (shout_cpu, twist_cpu) = circuit
        .cpu_bindings(&layout)
        .map_err(|e| PiCcsError::InvalidInput(format!("cpu bindings invalid: {e}")))?;

    // Ensure Shout lane counts are consistent across resources + cpu bindings.
    //
    // If lanes aren't set explicitly in resources, infer them from the binding vector length
    // so witness building + bus layout inference remain consistent.
    {
        let mut table_ids: Vec<u32> = resources
            .lut_tables
            .keys()
            .copied()
            .chain(resources.lut_table_specs.keys().copied())
            .collect();
        table_ids.sort_unstable();
        table_ids.dedup();

        for table_id in table_ids {
            let bindings = shout_cpu
                .get(&table_id)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("missing shout_cpu binding for table_id={table_id}")))?;
            if bindings.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout_cpu bindings for table_id={table_id} must be non-empty"
                )));
            }
            match resources.lut_lanes.get(&table_id) {
                Some(&lanes) => {
                    if lanes.max(1) != bindings.len() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "shout lanes mismatch for table_id={table_id}: resources.lut_lanes={} but cpu_bindings provides {}",
                            lanes,
                            bindings.len()
                        )));
                    }
                }
                None => {
                    resources.lut_lanes.insert(table_id, bindings.len());
                }
            }
        }
    }

    let (bus_region_len, bus_constraints) = shared_bus_buslen_and_constraints(
        used_cols,
        m_in,
        chunk_size,
        const_one_col,
        &resources,
        &shout_cpu,
        &twist_cpu,
    )
    .map_err(|e| PiCcsError::InvalidInput(format!("shared-bus sizing failed: {e}")))?;

    // Build base CPU R1CS with enough slack for shared-bus injection in the last rows.
    let mut cs = CcsBuilder::<super::F>::new(m_in, const_one_col)
        .map_err(|e| PiCcsError::InvalidInput(format!("ccs builder init failed: {e}")))?;
    circuit
        .define_cpu_constraints(&mut cs, &layout)
        .map_err(|e| PiCcsError::InvalidInput(format!("define_cpu_constraints failed: {e}")))?;

    let m_min = used_cols
        .checked_add(bus_region_len)
        .ok_or_else(|| PiCcsError::InvalidInput("shared-bus witness width overflow".into()))?;
    let base_ccs = cs
        .build_square(m_min, bus_constraints)
        .map_err(|e| PiCcsError::InvalidInput(format!("build_square failed: {e}")))?;

    Ok(SharedBusR1csPreprocessing {
        circuit,
        layout,
        resources,
        shout_cpu,
        twist_cpu,
        m_in,
        const_one_col,
        chunk_size,
        base_ccs,
    })
}

fn shout_meta_for_bus(
    table_id: u32,
    tables: &HashMap<u32, LutTable<super::F>>,
    specs: &HashMap<u32, LutTableSpec>,
) -> Result<(usize, usize), String> {
    if let Some(t) = tables.get(&table_id) {
        return Ok((t.d, t.n_side));
    }
    if let Some(spec) = specs.get(&table_id) {
        match spec {
            LutTableSpec::RiscvOpcode { xlen, .. } => {
                let d = xlen
                    .checked_mul(2)
                    .ok_or_else(|| "2*xlen overflow for RISC-V shout table".to_string())?;
                Ok((d, 2usize))
            }
        }
    } else {
        Err(format!("missing shout table metadata for table_id={table_id}"))
    }
}

fn shared_bus_buslen_and_constraints(
    cpu_used_cols: usize,
    m_in: usize,
    chunk_size: usize,
    const_one_col: usize,
    resources: &SharedBusResources,
    shout_cpu: &HashMap<u32, Vec<ShoutCpuBinding>>,
    twist_cpu: &HashMap<u32, Vec<TwistCpuBinding>>,
) -> Result<(usize, usize), String> {
    // Deterministic bus order: Shout tables (sorted), then Twist mems (sorted).
    let mut table_ids: Vec<u32> = resources
        .lut_tables
        .keys()
        .copied()
        .chain(resources.lut_table_specs.keys().copied())
        .collect();
    table_ids.sort_unstable();
    table_ids.dedup();

    let mut mem_ids: Vec<u32> = resources.mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    // Compute ell_addr (d * ell) and lanes for each instance.
    let mut shout_ell_addrs_and_lanes = Vec::with_capacity(table_ids.len());
    for table_id in &table_ids {
        let (d, n_side) = shout_meta_for_bus(*table_id, &resources.lut_tables, &resources.lut_table_specs)?;
        if n_side == 0 || !n_side.is_power_of_two() {
            return Err(format!("shout n_side must be power-of-two, got {n_side}"));
        }
        let ell = n_side.trailing_zeros() as usize;
        let ell_addr = d
            .checked_mul(ell)
            .ok_or_else(|| format!("ell_addr overflow for shout table_id={table_id}"))?;
        let lanes = resources.lut_lanes.get(table_id).copied().unwrap_or(1).max(1);
        let bindings = shout_cpu
            .get(table_id)
            .ok_or_else(|| format!("missing shout_cpu binding for table_id={table_id}"))?;
        if bindings.len() != lanes {
            return Err(format!(
                "shout_cpu bindings for table_id={table_id} has len={}, expected lanes={lanes}",
                bindings.len()
            ));
        }
        shout_ell_addrs_and_lanes.push((ell_addr, lanes));
    }

    let mut twist_ell_addrs_and_lanes = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let layout = resources
            .mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
        if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
            return Err(format!("twist n_side must be power-of-two, got {}", layout.n_side));
        }
        let ell = layout.n_side.trailing_zeros() as usize;
        let ell_addr = layout
            .d
            .checked_mul(ell)
            .ok_or_else(|| format!("ell_addr overflow for twist mem_id={mem_id}"))?;
        let lanes = layout.lanes.max(1);
        twist_ell_addrs_and_lanes.push((ell_addr, lanes));
    }

    // Bus columns per lane (same layout helper used by Route-A + constraints).
    let bus_cols = shout_ell_addrs_and_lanes
        .iter()
        .map(|&(ell, lanes)| lanes * (ell + 2))
        .chain(
            twist_ell_addrs_and_lanes
                .iter()
                .map(|&(ell, lanes)| lanes * (2 * ell + 5)),
        )
        .sum::<usize>();
    let bus_region_len = bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| "bus_region_len overflow".to_string())?;

    // Use a minimal feasible `m` to build the bus layout and count injected constraints.
    let m_min = cpu_used_cols
        .checked_add(bus_region_len)
        .ok_or_else(|| "shared-bus witness width overflow".to_string())?;

    let bus_layout = neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m_min,
        m_in,
        chunk_size,
        shout_ell_addrs_and_lanes.iter().copied(),
        twist_ell_addrs_and_lanes.iter().copied(),
    )?;

    let mut builder = CpuConstraintBuilder::<super::F>::new(/*n=*/ 1, /*m=*/ m_min, const_one_col);
    for (i, table_id) in table_ids.iter().enumerate() {
        let cpus = shout_cpu
            .get(table_id)
            .ok_or_else(|| format!("missing shout_cpu binding for table_id={table_id}"))?;
        let inst_cols = &bus_layout.shout_cols[i];
        if cpus.len() != inst_cols.lanes.len() {
            return Err(format!(
                "shared-bus shout lanes mismatch for table_id={table_id}: shout_cpu has len={} but bus layout expects {}",
                cpus.len(),
                inst_cols.lanes.len()
            ));
        }
        for lane_idx in 0..cpus.len() {
            builder.add_shout_instance_bound(&bus_layout, &inst_cols.lanes[lane_idx], &cpus[lane_idx]);
        }
    }
    for (i, mem_id) in mem_ids.iter().enumerate() {
        let inst_layout = resources
            .mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
        let lanes = inst_layout.lanes.max(1);
        let cpus = twist_cpu
            .get(mem_id)
            .ok_or_else(|| format!("missing twist_cpu binding for mem_id={mem_id}"))?;
        if cpus.len() != lanes {
            return Err(format!(
                "twist_cpu bindings for mem_id={mem_id} has len={}, expected lanes={lanes}",
                cpus.len()
            ));
        }
        for lane in 0..lanes {
            builder.add_twist_instance_bound(&bus_layout, &bus_layout.twist_cols[i].lanes[lane], &cpus[lane]);
        }
    }

    Ok((bus_region_len, builder.constraints().len()))
}
