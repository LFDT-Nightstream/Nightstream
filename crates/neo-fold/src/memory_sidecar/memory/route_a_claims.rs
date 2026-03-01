use super::*;

struct ValueCursor<'a> {
    vals: &'a [K],
    idx: usize,
}

impl<'a> ValueCursor<'a> {
    fn new(vals: &'a [K]) -> Self {
        Self { vals, idx: 0 }
    }

    fn take(&mut self) -> K {
        let v = self.vals[self.idx];
        self.idx += 1;
        v
    }

    fn take_arr<const N: usize>(&mut self) -> [K; N] {
        core::array::from_fn(|_| self.take())
    }

    fn consumed(&self) -> usize {
        self.idx
    }
}

struct DenseCols<T> {
    cols: Vec<Option<Vec<T>>>,
}

impl<T> DenseCols<T> {
    fn from_cols(cols: Vec<Vec<T>>) -> Self {
        Self {
            cols: cols.into_iter().map(Some).collect(),
        }
    }

    fn get(&self, col_id: &usize) -> Option<&Vec<T>> {
        self.cols.get(*col_id).and_then(|v| v.as_ref())
    }

    fn insert(&mut self, col_id: usize, vals: Vec<T>) {
        if col_id >= self.cols.len() {
            self.cols.resize_with(col_id + 1, || None);
        }
        self.cols[col_id] = Some(vals);
    }
}

#[inline]
fn ensure_column_len<T>(ctx: &'static str, col_id: usize, vals: &[T], expected: usize) -> Result<(), PiCcsError> {
    if vals.len() != expected {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: column {col_id} length mismatch: expected {expected}, got {}",
            vals.len()
        )));
    }
    Ok(())
}

pub struct RouteAShoutTimeClaimsGuard<'a> {
    pub lane_ranges: Vec<core::ops::Range<usize>>,
    pub lanes: Vec<RouteAShoutTimeLaneClaims<'a>>,
    pub gamma_groups: Vec<RouteAShoutTimeGammaGroupClaims<'a>>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub struct RouteAShoutTimeLaneClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub event_table_hash_prefix: Option<RoundOraclePrefix<'a>>,
    pub value_claim: K,
    pub adapter_claim: K,
    pub event_table_hash_claim: Option<K>,
    pub gamma_group: Option<usize>,
    pub transport_only: bool,
}

pub struct RouteAShoutTimeGammaGroupClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub bitness_prefix: RoundOraclePrefix<'a>,
    pub value_claim: K,
    pub adapter_claim: K,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [RouteAShoutTimeOracles],
    shout_gamma_groups: &'a mut [RouteAShoutGammaGroupOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut lane_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(shout_oracles.len());
    let mut lanes: Vec<RouteAShoutTimeLaneClaims<'a>> = Vec::new();
    let mut gamma_groups: Vec<RouteAShoutTimeGammaGroupClaims<'a>> = Vec::with_capacity(shout_gamma_groups.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        let start = lanes.len();
        for lane in o.lanes.iter_mut() {
            lanes.push(RouteAShoutTimeLaneClaims {
                value_prefix: RoundOraclePrefix::new(lane.value.as_mut(), ell_n),
                adapter_prefix: RoundOraclePrefix::new(lane.adapter.as_mut(), ell_n),
                event_table_hash_prefix: lane
                    .event_table_hash
                    .as_deref_mut()
                    .map(|o| RoundOraclePrefix::new(o, ell_n)),
                value_claim: lane.value_claim,
                adapter_claim: lane.adapter_claim,
                event_table_hash_claim: lane.event_table_hash_claim,
                gamma_group: lane.gamma_group,
                transport_only: lane.transport_only,
            });
        }
        let end = lanes.len();
        lane_ranges.push(start..end);
    }

    for g in shout_gamma_groups.iter_mut() {
        gamma_groups.push(RouteAShoutTimeGammaGroupClaims {
            value_prefix: RoundOraclePrefix::new(g.value.as_mut(), ell_n),
            adapter_prefix: RoundOraclePrefix::new(g.adapter.as_mut(), ell_n),
            bitness_prefix: RoundOraclePrefix::new(g.bitness.as_mut(), ell_n),
            value_claim: g.value_claim,
            adapter_claim: g.adapter_claim,
        });
    }

    RouteAShoutTimeClaimsGuard {
        lane_ranges,
        lanes,
        gamma_groups,
        bitness,
    }
}

pub struct ShoutRouteAProtocol<'a> {
    guard: RouteAShoutTimeClaimsGuard<'a>,
}

impl<'a> ShoutRouteAProtocol<'a> {
    pub fn new(
        shout_oracles: &'a mut [RouteAShoutTimeOracles],
        shout_gamma_groups: &'a mut [RouteAShoutGammaGroupOracles],
        ell_n: usize,
    ) -> Self {
        Self {
            guard: build_route_a_shout_time_claims_guard(shout_oracles, shout_gamma_groups, ell_n),
        }
    }
}

impl<'o> TimeBatchedClaims for ShoutRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) -> Result<(), PiCcsError> {
        append_route_a_shout_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        )
    }
}

pub fn append_route_a_shout_time_claims<'a>(
    guard: &'a mut RouteAShoutTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) -> Result<(), PiCcsError> {
    if guard.lane_ranges.is_empty() {
        return Ok(());
    }
    if guard.bitness.len() != guard.lane_ranges.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "shout bitness count mismatch (bitness={}, lane_ranges={})",
            guard.bitness.len(),
            guard.lane_ranges.len()
        )));
    }

    let mut lane_ranges_iter = guard.lane_ranges.iter();
    let mut next_end = lane_ranges_iter
        .next()
        .ok_or_else(|| PiCcsError::ProtocolError("shout lane_ranges unexpectedly empty".into()))?
        .end;
    let mut bitness_iter = guard.bitness.iter_mut();

    for (lane_idx, lane) in guard.lanes.iter_mut().enumerate() {
        if !lane.transport_only && lane.gamma_group.is_none() {
            claimed_sums.push(lane.value_claim);
            degree_bounds.push(lane.value_prefix.degree_bound());
            labels.push(b"shout/value");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: &mut lane.value_prefix,
                claimed_sum: lane.value_claim,
                label: b"shout/value",
            });

            claimed_sums.push(lane.adapter_claim);
            degree_bounds.push(lane.adapter_prefix.degree_bound());
            labels.push(b"shout/adapter");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: &mut lane.adapter_prefix,
                claimed_sum: lane.adapter_claim,
                label: b"shout/adapter",
            });
        }

        if let Some(prefix) = lane.event_table_hash_prefix.as_mut() {
            let claim = lane.event_table_hash_claim.ok_or_else(|| {
                PiCcsError::ProtocolError(format!("event_table_hash_claim missing for shout lane_idx={lane_idx}"))
            })?;
            claimed_sums.push(claim);
            degree_bounds.push(prefix.degree_bound());
            labels.push(b"shout/event_table_hash");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: prefix,
                claimed_sum: claim,
                label: b"shout/event_table_hash",
            });
        }

        if lane_idx + 1 == next_end {
            let bitness_vec = bitness_iter.next().ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "shout bitness idx drift at lane_idx={lane_idx} (missing bitness vector)"
                ))
            })?;
            for bit_oracle in bitness_vec.iter_mut() {
                claimed_sums.push(K::ZERO);
                degree_bounds.push(bit_oracle.degree_bound());
                labels.push(b"shout/bitness");
                claim_is_dynamic.push(false);
                claims.push(BatchedClaim {
                    oracle: bit_oracle.as_mut(),
                    claimed_sum: K::ZERO,
                    label: b"shout/bitness",
                });
            }

            next_end = lane_ranges_iter.next().map(|r| r.end).unwrap_or(usize::MAX);
        }
    }

    for group in guard.gamma_groups.iter_mut() {
        claimed_sums.push(group.value_claim);
        degree_bounds.push(group.value_prefix.degree_bound());
        labels.push(b"shout/value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut group.value_prefix,
            claimed_sum: group.value_claim,
            label: b"shout/value",
        });

        claimed_sums.push(group.adapter_claim);
        degree_bounds.push(group.adapter_prefix.degree_bound());
        labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut group.adapter_prefix,
            claimed_sum: group.adapter_claim,
            label: b"shout/adapter",
        });

        claimed_sums.push(K::ZERO);
        degree_bounds.push(group.bitness_prefix.degree_bound());
        labels.push(b"shout/bitness");
        claim_is_dynamic.push(false);
        claims.push(BatchedClaim {
            oracle: &mut group.bitness_prefix,
            claimed_sum: K::ZERO,
            label: b"shout/bitness",
        });
    }

    if bitness_iter.next().is_some() {
        return Err(PiCcsError::ProtocolError(
            "shout bitness not fully consumed after lane claim assembly".into(),
        ));
    }
    Ok(())
}

pub struct RouteATwistTimeClaimsGuard<'a> {
    pub read_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_check_claims: Vec<K>,
    pub write_check_claims: Vec<K>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
    pub virtual_write_domain_prefixes: Vec<Option<RoundOraclePrefix<'a>>>,
    pub nonvirtual_arch_domain_prefixes: Vec<Option<RoundOraclePrefix<'a>>>,
}

pub fn build_route_a_twist_time_claims_guard<'a>(
    twist_oracles: &'a mut [RouteATwistTimeOracles],
    ell_n: usize,
    read_check_claims: Vec<K>,
    write_check_claims: Vec<K>,
) -> Result<RouteATwistTimeClaimsGuard<'a>, PiCcsError> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(twist_oracles.len());
    let mut virtual_write_domain_prefixes: Vec<Option<RoundOraclePrefix<'a>>> = Vec::with_capacity(twist_oracles.len());
    let mut nonvirtual_arch_domain_prefixes: Vec<Option<RoundOraclePrefix<'a>>> =
        Vec::with_capacity(twist_oracles.len());

    if read_check_claims.len() != twist_oracles.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "twist read-check claim count mismatch (claims={}, oracles={})",
            read_check_claims.len(),
            twist_oracles.len()
        )));
    }
    if write_check_claims.len() != twist_oracles.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "twist write-check claim count mismatch (claims={}, oracles={})",
            write_check_claims.len(),
            twist_oracles.len()
        )));
    }

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_check_prefixes.push(RoundOraclePrefix::new(o.read_check.as_mut(), ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(o.write_check.as_mut(), ell_n));
        let vd_prefix = o
            .virtual_write_domain
            .as_mut()
            .map(|oracle| RoundOraclePrefix::new(oracle.as_mut(), ell_n));
        virtual_write_domain_prefixes.push(vd_prefix);
        let nvd_prefix = o
            .nonvirtual_arch_domain
            .as_mut()
            .map(|oracle| RoundOraclePrefix::new(oracle.as_mut(), ell_n));
        nonvirtual_arch_domain_prefixes.push(nvd_prefix);
    }

    Ok(RouteATwistTimeClaimsGuard {
        read_check_prefixes,
        write_check_prefixes,
        read_check_claims,
        write_check_claims,
        bitness,
        virtual_write_domain_prefixes,
        nonvirtual_arch_domain_prefixes,
    })
}

pub fn append_route_a_twist_time_claims<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) -> Result<(), PiCcsError> {
    for (
        ((((read_check_time, write_check_time), bitness_vec), virtual_write_domain), nonvirtual_arch_domain),
        (read_claim, write_claim),
    ) in guard
        .read_check_prefixes
        .iter_mut()
        .zip(guard.write_check_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
        .zip(guard.virtual_write_domain_prefixes.iter_mut())
        .zip(guard.nonvirtual_arch_domain_prefixes.iter_mut())
        .zip(
            guard
                .read_check_claims
                .iter()
                .zip(guard.write_check_claims.iter()),
        )
    {
        claimed_sums.push(*read_claim);
        degree_bounds.push(read_check_time.degree_bound());
        labels.push(b"twist/read_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_check_time,
            claimed_sum: *read_claim,
            label: b"twist/read_check",
        });

        claimed_sums.push(*write_claim);
        degree_bounds.push(write_check_time.degree_bound());
        labels.push(b"twist/write_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_check_time,
            claimed_sum: *write_claim,
            label: b"twist/write_check",
        });

        for bit_oracle in bitness_vec.iter_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle.as_mut(),
                claimed_sum: K::ZERO,
                label: b"twist/bitness",
            });
        }
        if let Some(virtual_write_domain_oracle) = virtual_write_domain.as_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(virtual_write_domain_oracle.degree_bound());
            labels.push(b"twist/virtual_write_domain");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: virtual_write_domain_oracle,
                claimed_sum: K::ZERO,
                label: b"twist/virtual_write_domain",
            });
        }
        if let Some(nonvirtual_arch_domain_oracle) = nonvirtual_arch_domain.as_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(nonvirtual_arch_domain_oracle.degree_bound());
            labels.push(b"twist/nonvirtual_arch_domain");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: nonvirtual_arch_domain_oracle,
                claimed_sum: K::ZERO,
                label: b"twist/nonvirtual_arch_domain",
            });
        }
    }
    Ok(())
}

pub struct TwistRouteAProtocol<'a> {
    guard: RouteATwistTimeClaimsGuard<'a>,
}

impl<'a> TwistRouteAProtocol<'a> {
    pub fn new(
        twist_oracles: &'a mut [RouteATwistTimeOracles],
        ell_n: usize,
        read_check_claims: Vec<K>,
        write_check_claims: Vec<K>,
    ) -> Result<Self, PiCcsError> {
        Ok(Self {
            guard: build_route_a_twist_time_claims_guard(twist_oracles, ell_n, read_check_claims, write_check_claims)?,
        })
    }
}

impl<'o> TimeBatchedClaims for TwistRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) -> Result<(), PiCcsError> {
        append_route_a_twist_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        )
    }
}

#[inline]
pub(crate) fn has_trace_lookup_families_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    step.lut_insts
        .iter()
        .any(|inst| rv32_is_decode_lookup_table_id(inst.table_id) || rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn has_trace_lookup_families_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    step.lut_instances
        .iter()
        .any(|(inst, _)| rv32_is_decode_lookup_table_id(inst.table_id) || rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn wb_wp_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    // Stage gating is keyed by lookup-family presence instead of fixed `m_in`/`mem_id`
    // assumptions so adapter-side routing can evolve without hardcoding RV32 shapes here.
    has_trace_lookup_families_instance(step)
}

#[inline]
pub(crate) fn wb_wp_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    has_trace_lookup_families_witness(step)
}

pub(crate) fn build_bus_layout_for_step_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
) -> Result<BusLayout, PiCcsError> {
    let m_in = step.mcs.0.m_in;
    if step.time_columns.t != t_len || step.time_columns.cpu_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(format!(
            "step bus layout requires canonical time columns (time_t={}, cpu_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len()
        )));
    }
    let cpu_region_len = step
        .time_columns
        .cpu_cols
        .len()
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("step bus layout: cpu_cols*t_len overflow".into()))?;
    let bus_region_len = step
        .time_columns
        .mem_cols
        .len()
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("step bus layout: mem_cols*t_len overflow".into()))?;
    let m = m_in
        .checked_add(cpu_region_len)
        .and_then(|v| v.checked_add(bus_region_len))
        .ok_or_else(|| PiCcsError::InvalidInput("step bus layout: virtual m overflow".into()))?;
    let shout_shapes: Vec<ShoutInstanceShape> = step
        .lut_instances
        .iter()
        .map(|(inst, _)| ShoutInstanceShape {
            ell_addr: inst.d * inst.ell,
            lanes: inst.lanes.max(1),
            n_vals: neo_memory::riscv::trace::rv32_trace_lookup_n_vals_for_table_id(inst.table_id),
            addr_group: inst.addr_group,
            selector_group: inst.selector_group,
        })
        .collect();
    let grouped_shout_instances = shout_shapes
        .iter()
        .filter(|shape| shape.addr_group.is_some())
        .count();
    let twist = step
        .mem_instances
        .iter()
        .map(|(inst, _)| (inst.d * inst.ell, inst.lanes.max(1)));
    let layout =
        build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(m, m_in, t_len, shout_shapes, twist).map_err(
            |e| {
                PiCcsError::InvalidInput(format!(
                    "step bus layout failed: m={m}, m_in={m_in}, t_len={t_len}, lut_insts={}, grouped_lut_insts={grouped_shout_instances}: {e}",
                    step.lut_instances.len()
                ))
            },
        )?;
    if layout.bus_cols != step.time_columns.mem_cols.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step bus layout mismatch: layout.bus_cols={} != time_columns.mem_cols.len()={}",
            layout.bus_cols,
            step.time_columns.mem_cols.len()
        )));
    }
    Ok(layout)
}

#[inline]
pub(crate) fn decode_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_instance(step)
        && step
            .lut_insts
            .iter()
            .any(|inst| rv32_is_decode_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn decode_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_witness(step)
        && step
            .lut_instances
            .iter()
            .any(|(inst, _)| rv32_is_decode_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn width_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_instance(step)
        && step
            .lut_insts
            .iter()
            .any(|inst| rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn width_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_witness(step)
        && step
            .lut_instances
            .iter()
            .any(|(inst, _)| rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn control_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    decode_stage_required_for_step_instance(step)
}

#[inline]
pub(crate) fn control_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    decode_stage_required_for_step_witness(step)
}

pub(crate) fn build_route_a_wb_wp_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<(Option<(Box<dyn RoundOracle>, K)>, Option<(Box<dyn RoundOracle>, K)>), PiCcsError> {
    if !wb_wp_required_for_step_witness(step) {
        return Ok((None, None));
    }

    let trace = Rv32TraceLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let wb_bool_cols = rv32_trace_wb_columns(&trace);
    let wp_cols = rv32_trace_wp_columns(&trace);

    let mut decode_cols = Vec::with_capacity(1 + wb_bool_cols.len() + wp_cols.len());
    decode_cols.push(trace.active);
    decode_cols.extend(wb_bool_cols.iter().copied());
    decode_cols.extend(wp_cols.iter().copied());
    let decoded = decode_trace_col_values_batch(params, step, t_len, &decode_cols)?;

    let wb_weights = wb_weight_vector(r_cycle, wb_bool_cols.len());
    let mut wb_bool_sparse_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(wb_bool_cols.len());
    for &col_id in wb_bool_cols.iter() {
        let vals = decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("WB: missing decoded bool column {col_id}")))?;
        wb_bool_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let wb_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, wb_bool_sparse_cols, wb_weights);

    let wp_cols = rv32_trace_wp_columns(&trace);
    let weights = wp_weight_vector(r_cycle, wp_cols.len());
    let active_vals = decoded
        .get(&trace.active)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("WP: missing decoded active column {}", trace.active)))?;
    let active = sparse_trace_col_from_values(m_in, ell_n, &active_vals)?;

    let mut sparse_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(wp_cols.len());
    for &col_id in wp_cols.iter() {
        let vals = decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("WP: missing decoded column {col_id}")))?;
        sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, &vals)?);
    }

    let oracle = WeightedMaskOracleSparseTime::new(active, sparse_cols, weights, r_cycle);
    Ok((Some((Box::new(wb_oracle), K::ZERO)), Some((Box::new(oracle), K::ZERO))))
}

pub(crate) fn build_route_a_decode_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<(Option<(Box<dyn RoundOracle>, K)>, Option<(Box<dyn RoundOracle>, K)>), PiCcsError> {
    if !decode_stage_required_for_step_witness(step) {
        return Ok((None, None));
    }

    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();

    let cpu_cols = vec![
        trace.active,
        trace.halted,
        trace.is_virtual,
        trace.virtual_sequence_remaining,
        trace.instr_word,
        trace.rs1_addr,
        trace.rs2_addr,
        trace.rd_addr,
        trace.rs1_val,
        trace.rs2_val,
        trace.rd_val,
        trace.rd_has_write,
        trace.ram_addr,
        trace.shout_has_lookup,
        trace.shout_table_id,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
        trace.shout_add_sub_key,
    ];
    let cpu_decoded = decode_trace_col_values_batch(params, step, t_len, &cpu_cols)?;

    let decode_decoded = {
        let instr_vals = cpu_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing instr_word decode column".into()))?;
        let active_vals = cpu_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "W2(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded_cols: Vec<Vec<K>> = (0..decode.cols)
            .map(|_| Vec::with_capacity(t_len))
            .collect();
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "W2(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for (col_id, value) in row.into_iter().enumerate() {
                decoded_cols[col_id].push(K::from(value));
            }
        }
        let mut decoded = DenseCols::from_cols(decoded_cols);

        // In shared lookup-backed mode, overwrite lookup-backed decode columns with the values
        // actually committed on the shared Shout bus so prover oracles and verifier terminals
        // are sourced from identical openings.
        let (decode_open_cols, decode_lut_slots) = resolve_shared_decode_lookup_lut_indices(step, &decode)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        if bus.shout_cols.len() != step.lut_instances.len() {
            return Err(PiCcsError::ProtocolError(
                "W2(shared): bus layout shout lane count drift".into(),
            ));
        }
        let mut bus_val_cols = Vec::with_capacity(decode_open_cols.len());
        for &(lut_idx, val_slot) in decode_lut_slots.iter() {
            let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): missing shout cols for decode lookup table".into())
            })?;
            let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): expected one shout lane for decode lookup table".into())
            })?;
            let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): decode val_slot={} out of range for lut_idx={} (n_vals={})",
                    val_slot,
                    lut_idx,
                    lane0.vals.len()
                ))
            })?;
            bus_val_cols.push(val_col);
        }
        let lookup_vals = decode_lookup_backed_col_values_batch(
            t_len,
            bus.bus_cols,
            Some(&step.time_columns.mem_cols),
            &bus_val_cols,
        )?;
        for (open_idx, &decode_col_id) in decode_open_cols.iter().enumerate() {
            let bus_col_id = bus_val_cols[open_idx];
            let values = lookup_vals.get(&bus_col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): missing decoded lookup values for bus_col={bus_col_id}"
                ))
            })?;
            decoded.insert(decode_col_id, values.clone());
        }

        // Recompute derived decode helper columns from opened lookup-backed decode columns.
        let rd_is_zero_vals = decoded
            .get(&decode.rd_is_zero)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing rd_is_zero decode column".into()))?;
        let funct7_b0_vals = decoded
            .get(&decode.funct7_bit[0])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[0] decode column".into()))?;
        let funct7_b1_vals = decoded
            .get(&decode.funct7_bit[1])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[1] decode column".into()))?;
        let funct7_b2_vals = decoded
            .get(&decode.funct7_bit[2])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[2] decode column".into()))?;
        let funct7_b3_vals = decoded
            .get(&decode.funct7_bit[3])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[3] decode column".into()))?;
        let funct7_b4_vals = decoded
            .get(&decode.funct7_bit[4])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[4] decode column".into()))?;
        let funct7_b5_vals = decoded
            .get(&decode.funct7_bit[5])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[5] decode column".into()))?;
        let funct7_b6_vals = decoded
            .get(&decode.funct7_bit[6])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[6] decode column".into()))?;
        let op_lui_vals = decoded
            .get(&decode.op_lui)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_lui decode column".into()))?;
        let op_auipc_vals = decoded
            .get(&decode.op_auipc)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_auipc decode column".into()))?;
        let op_jal_vals = decoded
            .get(&decode.op_jal)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_jal decode column".into()))?;
        let op_jalr_vals = decoded
            .get(&decode.op_jalr)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_jalr decode column".into()))?;
        let op_alu_imm_vals = decoded
            .get(&decode.op_alu_imm)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_alu_imm decode column".into()))?;
        let op_alu_reg_vals = decoded
            .get(&decode.op_alu_reg)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_alu_reg decode column".into()))?;
        let funct3_is0_vals = decoded
            .get(&decode.funct3_is[0])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[0] decode column".into()))?;
        let funct3_is1_vals = decoded
            .get(&decode.funct3_is[1])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[1] decode column".into()))?;
        let funct3_is2_vals = decoded
            .get(&decode.funct3_is[2])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[2] decode column".into()))?;
        let funct3_is3_vals = decoded
            .get(&decode.funct3_is[3])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[3] decode column".into()))?;
        let funct3_is4_vals = decoded
            .get(&decode.funct3_is[4])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[4] decode column".into()))?;
        let funct3_is5_vals = decoded
            .get(&decode.funct3_is[5])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[5] decode column".into()))?;
        let funct3_is6_vals = decoded
            .get(&decode.funct3_is[6])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[6] decode column".into()))?;
        let funct3_is7_vals = decoded
            .get(&decode.funct3_is[7])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[7] decode column".into()))?;
        let rs2_vals = decoded
            .get(&decode.rs2)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing rs2 decode column".into()))?;
        let imm_i_vals = decoded
            .get(&decode.imm_i)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing imm_i decode column".into()))?;
        for (col_id, vals) in [
            (decode.rd_is_zero, rd_is_zero_vals.as_slice()),
            (decode.funct7_bit[0], funct7_b0_vals.as_slice()),
            (decode.funct7_bit[1], funct7_b1_vals.as_slice()),
            (decode.funct7_bit[2], funct7_b2_vals.as_slice()),
            (decode.funct7_bit[3], funct7_b3_vals.as_slice()),
            (decode.funct7_bit[4], funct7_b4_vals.as_slice()),
            (decode.funct7_bit[5], funct7_b5_vals.as_slice()),
            (decode.funct7_bit[6], funct7_b6_vals.as_slice()),
            (decode.op_lui, op_lui_vals.as_slice()),
            (decode.op_auipc, op_auipc_vals.as_slice()),
            (decode.op_jal, op_jal_vals.as_slice()),
            (decode.op_jalr, op_jalr_vals.as_slice()),
            (decode.op_alu_imm, op_alu_imm_vals.as_slice()),
            (decode.op_alu_reg, op_alu_reg_vals.as_slice()),
            (decode.funct3_is[0], funct3_is0_vals.as_slice()),
            (decode.funct3_is[1], funct3_is1_vals.as_slice()),
            (decode.funct3_is[2], funct3_is2_vals.as_slice()),
            (decode.funct3_is[3], funct3_is3_vals.as_slice()),
            (decode.funct3_is[4], funct3_is4_vals.as_slice()),
            (decode.funct3_is[5], funct3_is5_vals.as_slice()),
            (decode.funct3_is[6], funct3_is6_vals.as_slice()),
            (decode.funct3_is[7], funct3_is7_vals.as_slice()),
            (decode.rs2, rs2_vals.as_slice()),
            (decode.imm_i, imm_i_vals.as_slice()),
        ] {
            ensure_column_len("W2(shared)", col_id, vals, t_len)?;
        }

        let mut op_lui_write = Vec::with_capacity(t_len);
        let mut op_auipc_write = Vec::with_capacity(t_len);
        let mut op_jal_write = Vec::with_capacity(t_len);
        let mut op_jalr_write = Vec::with_capacity(t_len);
        let mut op_alu_imm_write = Vec::with_capacity(t_len);
        let mut op_alu_reg_write = Vec::with_capacity(t_len);
        let mut alu_reg_delta = Vec::with_capacity(t_len);
        let mut alu_imm_delta = Vec::with_capacity(t_len);
        let mut alu_imm_shift_rhs_delta = Vec::with_capacity(t_len);
        for j in 0..t_len {
            let rd_keep = K::ONE - rd_is_zero_vals[j];
            op_lui_write.push(op_lui_vals[j] * rd_keep);
            op_auipc_write.push(op_auipc_vals[j] * rd_keep);
            op_jal_write.push(op_jal_vals[j] * rd_keep);
            op_jalr_write.push(op_jalr_vals[j] * rd_keep);
            op_alu_imm_write.push(op_alu_imm_vals[j] * rd_keep);
            op_alu_reg_write.push(op_alu_reg_vals[j] * rd_keep);
            let funct7_bits = [
                funct7_b0_vals[j],
                funct7_b1_vals[j],
                funct7_b2_vals[j],
                funct7_b3_vals[j],
                funct7_b4_vals[j],
                funct7_b5_vals[j],
                funct7_b6_vals[j],
            ];
            let funct3_is = [
                funct3_is0_vals[j],
                funct3_is1_vals[j],
                funct3_is2_vals[j],
                funct3_is3_vals[j],
                funct3_is4_vals[j],
                funct3_is5_vals[j],
                funct3_is6_vals[j],
                funct3_is7_vals[j],
            ];
            alu_reg_delta.push(w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is));
            alu_imm_delta.push(funct7_bits[5] * funct3_is[5]);
            alu_imm_shift_rhs_delta.push((funct3_is1_vals[j] + funct3_is5_vals[j]) * (rs2_vals[j] - imm_i_vals[j]));
        }
        decoded.insert(decode.op_lui_write, op_lui_write);
        decoded.insert(decode.op_auipc_write, op_auipc_write);
        decoded.insert(decode.op_jal_write, op_jal_write);
        decoded.insert(decode.op_jalr_write, op_jalr_write);
        decoded.insert(decode.op_alu_imm_write, op_alu_imm_write);
        decoded.insert(decode.op_alu_reg_write, op_alu_reg_write);
        decoded.insert(decode.alu_reg_table_delta, alu_reg_delta);
        decoded.insert(decode.alu_imm_table_delta, alu_imm_delta);
        decoded.insert(decode.alu_imm_shift_rhs_delta, alu_imm_shift_rhs_delta);

        decoded
    };

    let imm_i_vals = decode_decoded
        .get(&decode.imm_i)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.imm_i)))?;
    let imm_s_vals = decode_decoded
        .get(&decode.imm_s)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.imm_s)))?;
    let imm_b_vals = decode_decoded
        .get(&decode.imm_b)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.imm_b)))?;
    let imm_j_vals = decode_decoded
        .get(&decode.imm_j)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.imm_j)))?;
    let funct3_bit0_vals = decode_decoded.get(&decode.funct3_bit[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_bit[0]
        ))
    })?;
    let funct3_bit1_vals = decode_decoded.get(&decode.funct3_bit[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_bit[1]
        ))
    })?;
    let funct3_bit2_vals = decode_decoded.get(&decode.funct3_bit[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_bit[2]
        ))
    })?;
    let funct7_bit0_vals = decode_decoded.get(&decode.funct7_bit[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[0]
        ))
    })?;
    let funct7_bit1_vals = decode_decoded.get(&decode.funct7_bit[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[1]
        ))
    })?;
    let funct7_bit2_vals = decode_decoded.get(&decode.funct7_bit[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[2]
        ))
    })?;
    let funct7_bit3_vals = decode_decoded.get(&decode.funct7_bit[3]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[3]
        ))
    })?;
    let funct7_bit4_vals = decode_decoded.get(&decode.funct7_bit[4]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[4]
        ))
    })?;
    let funct7_bit5_vals = decode_decoded.get(&decode.funct7_bit[5]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[5]
        ))
    })?;
    let funct7_bit6_vals = decode_decoded.get(&decode.funct7_bit[6]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct7_bit[6]
        ))
    })?;
    let rd_bit0_vals = decode_decoded.get(&decode.rd_bit[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_bit[0]))
    })?;
    let rd_bit1_vals = decode_decoded.get(&decode.rd_bit[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_bit[1]))
    })?;
    let rd_bit2_vals = decode_decoded.get(&decode.rd_bit[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_bit[2]))
    })?;
    let rd_bit3_vals = decode_decoded.get(&decode.rd_bit[3]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_bit[3]))
    })?;
    let rd_bit4_vals = decode_decoded.get(&decode.rd_bit[4]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_bit[4]))
    })?;
    let rs1_bit0_vals = decode_decoded.get(&decode.rs1_bit[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1_bit[0]))
    })?;
    let rs1_bit1_vals = decode_decoded.get(&decode.rs1_bit[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1_bit[1]))
    })?;
    let rs1_bit2_vals = decode_decoded.get(&decode.rs1_bit[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1_bit[2]))
    })?;
    let rs1_bit3_vals = decode_decoded.get(&decode.rs1_bit[3]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1_bit[3]))
    })?;
    let rs1_bit4_vals = decode_decoded.get(&decode.rs1_bit[4]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1_bit[4]))
    })?;
    let rs2_bit0_vals = decode_decoded.get(&decode.rs2_bit[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2_bit[0]))
    })?;
    let rs2_bit1_vals = decode_decoded.get(&decode.rs2_bit[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2_bit[1]))
    })?;
    let rs2_bit2_vals = decode_decoded.get(&decode.rs2_bit[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2_bit[2]))
    })?;
    let rs2_bit3_vals = decode_decoded.get(&decode.rs2_bit[3]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2_bit[3]))
    })?;
    let rs2_bit4_vals = decode_decoded.get(&decode.rs2_bit[4]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2_bit[4]))
    })?;
    for (col_id, vals) in [
        (decode.imm_i, imm_i_vals.as_slice()),
        (decode.imm_s, imm_s_vals.as_slice()),
        (decode.imm_b, imm_b_vals.as_slice()),
        (decode.imm_j, imm_j_vals.as_slice()),
        (decode.funct3_bit[0], funct3_bit0_vals.as_slice()),
        (decode.funct3_bit[1], funct3_bit1_vals.as_slice()),
        (decode.funct3_bit[2], funct3_bit2_vals.as_slice()),
        (decode.funct7_bit[0], funct7_bit0_vals.as_slice()),
        (decode.funct7_bit[1], funct7_bit1_vals.as_slice()),
        (decode.funct7_bit[2], funct7_bit2_vals.as_slice()),
        (decode.funct7_bit[3], funct7_bit3_vals.as_slice()),
        (decode.funct7_bit[4], funct7_bit4_vals.as_slice()),
        (decode.funct7_bit[5], funct7_bit5_vals.as_slice()),
        (decode.funct7_bit[6], funct7_bit6_vals.as_slice()),
        (decode.rd_bit[0], rd_bit0_vals.as_slice()),
        (decode.rd_bit[1], rd_bit1_vals.as_slice()),
        (decode.rd_bit[2], rd_bit2_vals.as_slice()),
        (decode.rd_bit[3], rd_bit3_vals.as_slice()),
        (decode.rd_bit[4], rd_bit4_vals.as_slice()),
        (decode.rs1_bit[0], rs1_bit0_vals.as_slice()),
        (decode.rs1_bit[1], rs1_bit1_vals.as_slice()),
        (decode.rs1_bit[2], rs1_bit2_vals.as_slice()),
        (decode.rs1_bit[3], rs1_bit3_vals.as_slice()),
        (decode.rs1_bit[4], rs1_bit4_vals.as_slice()),
        (decode.rs2_bit[0], rs2_bit0_vals.as_slice()),
        (decode.rs2_bit[1], rs2_bit1_vals.as_slice()),
        (decode.rs2_bit[2], rs2_bit2_vals.as_slice()),
        (decode.rs2_bit[3], rs2_bit3_vals.as_slice()),
        (decode.rs2_bit[4], rs2_bit4_vals.as_slice()),
    ] {
        ensure_column_len("W2(shared)", col_id, vals, t_len)?;
    }

    #[cfg(debug_assertions)]
    let cpu_active_vals = cpu_decoded
        .get(&trace.active)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.active)))?;
    #[cfg(debug_assertions)]
    let cpu_halted_vals = cpu_decoded
        .get(&trace.halted)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.halted)))?;
    #[cfg(debug_assertions)]
    let cpu_is_virtual_vals = cpu_decoded
        .get(&trace.is_virtual)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.is_virtual)))?;
    #[cfg(debug_assertions)]
    let cpu_virtual_sequence_remaining_vals = cpu_decoded
        .get(&trace.virtual_sequence_remaining)
        .ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W2 missing CPU decoded column {}",
                trace.virtual_sequence_remaining
            ))
        })?;
    #[cfg(debug_assertions)]
    let cpu_rs1_addr_vals = cpu_decoded
        .get(&trace.rs1_addr)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rs1_addr)))?;
    #[cfg(debug_assertions)]
    let cpu_rs2_addr_vals = cpu_decoded
        .get(&trace.rs2_addr)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rs2_addr)))?;
    #[cfg(debug_assertions)]
    let cpu_rd_addr_vals = cpu_decoded
        .get(&trace.rd_addr)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rd_addr)))?;
    #[cfg(debug_assertions)]
    let cpu_rs1_val_vals = cpu_decoded
        .get(&trace.rs1_val)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rs1_val)))?;
    #[cfg(debug_assertions)]
    let cpu_rs2_val_vals = cpu_decoded
        .get(&trace.rs2_val)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rs2_val)))?;
    #[cfg(debug_assertions)]
    let cpu_rd_val_vals = cpu_decoded
        .get(&trace.rd_val)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rd_val)))?;
    #[cfg(debug_assertions)]
    let cpu_rd_has_write_vals = cpu_decoded
        .get(&trace.rd_has_write)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.rd_has_write)))?;
    #[cfg(debug_assertions)]
    let cpu_ram_addr_vals = cpu_decoded
        .get(&trace.ram_addr)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.ram_addr)))?;
    #[cfg(debug_assertions)]
    let cpu_shout_has_lookup_vals = cpu_decoded.get(&trace.shout_has_lookup).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_has_lookup))
    })?;
    #[cfg(debug_assertions)]
    let cpu_shout_table_id_vals = cpu_decoded
        .get(&trace.shout_table_id)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_table_id)))?;
    #[cfg(debug_assertions)]
    let cpu_shout_val_vals = cpu_decoded
        .get(&trace.shout_val)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_val)))?;
    #[cfg(debug_assertions)]
    let cpu_shout_lhs_vals = cpu_decoded
        .get(&trace.shout_lhs)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_lhs)))?;
    #[cfg(debug_assertions)]
    let cpu_shout_rhs_vals = cpu_decoded
        .get(&trace.shout_rhs)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_rhs)))?;
    #[cfg(debug_assertions)]
    let cpu_shout_add_sub_key_vals = cpu_decoded.get(&trace.shout_add_sub_key).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {}", trace.shout_add_sub_key))
    })?;
    #[cfg(debug_assertions)]
    let decode_opcode_vals = decode_decoded.get(&decode.opcode).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.opcode))
    })?;
    #[cfg(debug_assertions)]
    let decode_rs1_addr_vals = decode_decoded
        .get(&decode.rs1)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs1)))?;
    #[cfg(debug_assertions)]
    let decode_rs2_addr_vals = decode_decoded
        .get(&decode.rs2)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rs2)))?;
    #[cfg(debug_assertions)]
    let decode_rd_addr_vals = decode_decoded
        .get(&decode.rd)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd)))?;
    #[cfg(debug_assertions)]
    let decode_rd_is_zero_vals = decode_decoded.get(&decode.rd_is_zero).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.rd_is_zero))
    })?;
    #[cfg(debug_assertions)]
    let decode_rd_has_write_vals = decode_decoded.get(&decode.rd_has_write).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.rd_has_write
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_ram_has_read_vals = decode_decoded.get(&decode.ram_has_read).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.ram_has_read
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_ram_has_write_vals = decode_decoded.get(&decode.ram_has_write).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.ram_has_write
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_lui_vals = decode_decoded.get(&decode.op_lui).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_lui))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_auipc_vals = decode_decoded.get(&decode.op_auipc).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_auipc))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_jal_vals = decode_decoded.get(&decode.op_jal).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_jal))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_jalr_vals = decode_decoded.get(&decode.op_jalr).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_jalr))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_branch_vals = decode_decoded.get(&decode.op_branch).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_branch))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_load_vals = decode_decoded.get(&decode.op_load).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_load))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_store_vals = decode_decoded.get(&decode.op_store).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_store))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_alu_imm_vals = decode_decoded.get(&decode.op_alu_imm).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_alu_imm))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_alu_reg_vals = decode_decoded.get(&decode.op_alu_reg).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_alu_reg))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_misc_mem_vals = decode_decoded.get(&decode.op_misc_mem).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_misc_mem))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_system_vals = decode_decoded.get(&decode.op_system).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_system))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_amo_vals = decode_decoded.get(&decode.op_amo).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_amo))
    })?;
    #[cfg(debug_assertions)]
    let decode_op_custom_vals = decode_decoded.get(&decode.op_custom).ok_or_else(|| {
        PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {}", decode.op_custom))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is0_vals = decode_decoded.get(&decode.funct3_is[0]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[0]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is1_vals = decode_decoded.get(&decode.funct3_is[1]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[1]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is2_vals = decode_decoded.get(&decode.funct3_is[2]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[2]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is3_vals = decode_decoded.get(&decode.funct3_is[3]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[3]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is4_vals = decode_decoded.get(&decode.funct3_is[4]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[4]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is5_vals = decode_decoded.get(&decode.funct3_is[5]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[5]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is6_vals = decode_decoded.get(&decode.funct3_is[6]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[6]
        ))
    })?;
    #[cfg(debug_assertions)]
    let decode_funct3_is7_vals = decode_decoded.get(&decode.funct3_is[7]).ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "W2 missing decode lookup-backed column {}",
            decode.funct3_is[7]
        ))
    })?;

    let mut imm_residual_vals: Vec<Vec<K>> = (0..W2_IMM_RESIDUAL_COUNT)
        .map(|_| Vec::with_capacity(t_len))
        .collect();
    for j in 0..t_len {
        let funct3_bits = [funct3_bit0_vals[j], funct3_bit1_vals[j], funct3_bit2_vals[j]];
        let funct7_bits = [
            funct7_bit0_vals[j],
            funct7_bit1_vals[j],
            funct7_bit2_vals[j],
            funct7_bit3_vals[j],
            funct7_bit4_vals[j],
            funct7_bit5_vals[j],
            funct7_bit6_vals[j],
        ];
        let imm = w2_decode_immediate_residuals(
            imm_i_vals[j],
            imm_s_vals[j],
            imm_b_vals[j],
            imm_j_vals[j],
            [
                rd_bit0_vals[j],
                rd_bit1_vals[j],
                rd_bit2_vals[j],
                rd_bit3_vals[j],
                rd_bit4_vals[j],
            ],
            funct3_bits,
            [
                rs1_bit0_vals[j],
                rs1_bit1_vals[j],
                rs1_bit2_vals[j],
                rs1_bit3_vals[j],
                rs1_bit4_vals[j],
            ],
            [
                rs2_bit0_vals[j],
                rs2_bit1_vals[j],
                rs2_bit2_vals[j],
                rs2_bit3_vals[j],
                rs2_bit4_vals[j],
            ],
            funct7_bits,
        );

        #[cfg(debug_assertions)]
        {
            let active = cpu_active_vals[j];
            let halted = cpu_halted_vals[j];
            let is_virtual = cpu_is_virtual_vals[j];
            let virtual_sequence_remaining = cpu_virtual_sequence_remaining_vals[j];
            let trace_rs1_addr = cpu_rs1_addr_vals[j];
            let trace_rs2_addr = cpu_rs2_addr_vals[j];
            let trace_rd_addr = cpu_rd_addr_vals[j];
            let trace_rd_has_write = cpu_rd_has_write_vals[j];
            let decode_opcode = decode_opcode_vals[j];
            let decode_rd_has_write = decode_rd_has_write_vals[j];
            let decode_rs1_addr = decode_rs1_addr_vals[j];
            let decode_rs2_addr = decode_rs2_addr_vals[j];
            let decode_rd_addr = decode_rd_addr_vals[j];
            let rd_is_zero = decode_rd_is_zero_vals[j];
            let rs1_val = cpu_rs1_val_vals[j];
            let rs2_val = cpu_rs2_val_vals[j];
            let rd_val = cpu_rd_val_vals[j];
            let ram_has_read = decode_ram_has_read_vals[j];
            let ram_has_write = decode_ram_has_write_vals[j];
            let ram_addr = cpu_ram_addr_vals[j];
            let shout_has_lookup = cpu_shout_has_lookup_vals[j];
            let shout_table_id = cpu_shout_table_id_vals[j];
            let shout_val = cpu_shout_val_vals[j];
            let shout_lhs = cpu_shout_lhs_vals[j];
            let shout_rhs = cpu_shout_rhs_vals[j];
            let shout_add_sub_key = cpu_shout_add_sub_key_vals[j];
            let opcode_flags = [
                decode_op_lui_vals[j],
                decode_op_auipc_vals[j],
                decode_op_jal_vals[j],
                decode_op_jalr_vals[j],
                decode_op_branch_vals[j],
                decode_op_load_vals[j],
                decode_op_store_vals[j],
                decode_op_alu_imm_vals[j],
                decode_op_alu_reg_vals[j],
                decode_op_misc_mem_vals[j],
                decode_op_system_vals[j],
                decode_op_amo_vals[j],
            ];
            let op_custom = decode_op_custom_vals[j];
            let funct3_is = [
                decode_funct3_is0_vals[j],
                decode_funct3_is1_vals[j],
                decode_funct3_is2_vals[j],
                decode_funct3_is3_vals[j],
                decode_funct3_is4_vals[j],
                decode_funct3_is5_vals[j],
                decode_funct3_is6_vals[j],
                decode_funct3_is7_vals[j],
            ];

            let op_write_flags = [
                opcode_flags[0] * (K::ONE - rd_is_zero),
                opcode_flags[1] * (K::ONE - rd_is_zero),
                opcode_flags[2] * (K::ONE - rd_is_zero),
                opcode_flags[3] * (K::ONE - rd_is_zero),
                opcode_flags[7] * (K::ONE - rd_is_zero),
                opcode_flags[8] * (K::ONE - rd_is_zero),
            ];
            let alu_reg_table_delta = w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is);
            let alu_imm_table_delta = funct7_bits[5] * funct3_is[5];
            let alu_imm_shift_rhs_delta = (funct3_is[1] + funct3_is[5]) * (decode_rs2_addr - imm_i_vals[j]);
            let selector_residuals = w2_decode_selector_residuals(
                active,
                decode_opcode,
                opcode_flags,
                op_custom,
                funct3_is,
                funct3_bits,
                opcode_flags[11],
            );
            let bitness_residuals = w2_decode_bitness_residuals(opcode_flags, funct3_is);
            let alu_branch_residuals = w2_alu_branch_lookup_residuals(
                active,
                is_virtual,
                virtual_sequence_remaining,
                halted,
                shout_has_lookup,
                shout_lhs,
                shout_rhs,
                shout_add_sub_key,
                shout_table_id,
                trace_rs1_addr,
                trace_rs2_addr,
                trace_rd_addr,
                decode_rs1_addr,
                decode_rs2_addr,
                decode_rd_addr,
                rs1_val,
                rs2_val,
                trace_rd_has_write,
                decode_rd_has_write,
                rd_is_zero,
                rd_val,
                ram_has_read,
                ram_has_write,
                ram_addr,
                shout_val,
                funct3_bits,
                funct7_bits,
                opcode_flags,
                op_write_flags,
                funct3_is,
                alu_reg_table_delta,
                alu_imm_table_delta,
                alu_imm_shift_rhs_delta,
                decode_rs2_addr,
                imm_i_vals[j],
                imm_s_vals[j],
            );
            if let Some((idx, _)) = selector_residuals
                .iter()
                .enumerate()
                .find(|(_, r)| **r != K::ZERO)
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "decode/fields selector residual non-zero at row={j}, idx={idx}"
                )));
            }
            if let Some((idx, _)) = bitness_residuals
                .iter()
                .enumerate()
                .find(|(_, r)| **r != K::ZERO)
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "decode/fields bitness residual non-zero at row={j}, idx={idx}"
                )));
            }
            if let Some((idx, _)) = alu_branch_residuals
                .iter()
                .enumerate()
                .find(|(_, r)| **r != K::ZERO)
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "decode/fields alu_branch residual non-zero at row={j}, idx={idx}"
                )));
            }
        }

        for (k, r) in imm.iter().enumerate() {
            imm_residual_vals[k].push(*r);
        }
    }

    let main_field_cols = vec![
        trace.active,
        trace.halted,
        trace.is_virtual,
        trace.virtual_sequence_remaining,
        trace.rs1_addr,
        trace.rs2_addr,
        trace.rd_addr,
        trace.rs1_val,
        trace.rs2_val,
        trace.rd_val,
        trace.rd_has_write,
        trace.ram_addr,
        trace.shout_has_lookup,
        trace.shout_table_id,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
        trace.shout_add_sub_key,
    ];
    let decode_field_cols = vec![
        decode.opcode,
        decode.rs1,
        decode.rs2,
        decode.rd,
        decode.rd_is_zero,
        decode.rd_has_write,
        decode.ram_has_read,
        decode.ram_has_write,
        decode.op_lui,
        decode.op_auipc,
        decode.op_jal,
        decode.op_jalr,
        decode.op_branch,
        decode.op_load,
        decode.op_store,
        decode.op_alu_imm,
        decode.op_alu_reg,
        decode.op_misc_mem,
        decode.op_system,
        decode.op_amo,
        decode.op_custom,
        decode.funct3_is[0],
        decode.funct3_is[1],
        decode.funct3_is[2],
        decode.funct3_is[3],
        decode.funct3_is[4],
        decode.funct3_is[5],
        decode.funct3_is[6],
        decode.funct3_is[7],
        decode.funct3_bit[0],
        decode.funct3_bit[1],
        decode.funct3_bit[2],
        decode.funct7_bit[0],
        decode.funct7_bit[1],
        decode.funct7_bit[2],
        decode.funct7_bit[3],
        decode.funct7_bit[4],
        decode.funct7_bit[5],
        decode.funct7_bit[6],
        decode.imm_i,
        decode.imm_s,
    ];
    let mut fields_sparse_cols = Vec::with_capacity(main_field_cols.len() + decode_field_cols.len());
    for &col_id in main_field_cols.iter() {
        let vals = cpu_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {col_id}")))?;
        fields_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    for &col_id in decode_field_cols.iter() {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {col_id}")))?;
        fields_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let mut imm_sparse_cols = Vec::with_capacity(imm_residual_vals.len());
    for vals in imm_residual_vals.iter() {
        imm_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let pow2_cycle = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("W2: 2^ell_n overflow".into()))?;
    let active_zero = SparseIdxVec::from_entries(pow2_cycle, Vec::new());
    let fields_weights = w2_decode_pack_weight_vector(r_cycle, W2_FIELDS_RESIDUAL_COUNT);
    let fields_oracle = FormulaOracleSparseTime::new(
        fields_sparse_cols,
        // Virtual-stage shape+semantic selectors introduce higher multiplicative degree;
        // use the shared decode/fields bound with one slack degree.
        W2_FIELDS_DEGREE_BOUND,
        r_cycle,
        move |vals: &[K]| {
            let mut cursor = ValueCursor::new(vals);
            let decode_inputs = W2DecodeFieldsOpenings {
                active: cursor.take(),
                halted: cursor.take(),
                is_virtual: cursor.take(),
                virtual_sequence_remaining: cursor.take(),
                trace_rs1_addr: cursor.take(),
                trace_rs2_addr: cursor.take(),
                trace_rd_addr: cursor.take(),
                rs1_val: cursor.take(),
                rs2_val: cursor.take(),
                rd_val: cursor.take(),
                trace_rd_has_write: cursor.take(),
                ram_addr: cursor.take(),
                shout_has_lookup: cursor.take(),
                shout_table_id: cursor.take(),
                shout_val: cursor.take(),
                shout_lhs: cursor.take(),
                shout_rhs: cursor.take(),
                shout_add_sub_key: cursor.take(),
                decode_opcode: cursor.take(),
                decode_rs1_addr: cursor.take(),
                decode_rs2_addr: cursor.take(),
                decode_rd_addr: cursor.take(),
                rd_is_zero: cursor.take(),
                decode_rd_has_write: cursor.take(),
                ram_has_read: cursor.take(),
                ram_has_write: cursor.take(),
                opcode_flags: cursor.take_arr::<12>(),
                op_custom: cursor.take(),
                funct3_is: cursor.take_arr::<8>(),
                funct3_bits: cursor.take_arr::<3>(),
                funct7_bits: cursor.take_arr::<7>(),
                imm_i: cursor.take(),
                imm_s: cursor.take(),
            };
            debug_assert_eq!(cursor.consumed(), vals.len());
            w2_decode_fields_weighted_residual(&decode_inputs, &fields_weights)
        },
    );
    let imm_oracle = WeightedMaskOracleSparseTime::new(
        active_zero,
        imm_sparse_cols,
        w2_decode_imm_weight_vector(r_cycle, 4),
        r_cycle,
    );

    Ok((
        Some((Box::new(fields_oracle), K::ZERO)),
        Some((Box::new(imm_oracle), K::ZERO)),
    ))
}

pub(crate) type W3TimeClaims = (
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
);
