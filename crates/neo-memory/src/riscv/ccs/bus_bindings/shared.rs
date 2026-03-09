use std::collections::HashMap;

use p3_goldilocks::Goldilocks as F;

use crate::cpu::bus_layout::BusLayout;
use crate::cpu::constraints::CpuConstraintBuilder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceShoutBusSpec {
    pub table_id: u32,
    pub ell_addr: usize,
    pub n_vals: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceShoutShape {
    pub table_id: u32,
    pub ell_addr: usize,
    pub n_vals: usize,
    pub addr_group: Option<u32>,
    pub selector_group: Option<u32>,
}

pub(crate) fn derive_trace_shout_shapes<FBase, FAddrGroup, FSelectorGroup>(
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mut base_shape_for_table_id: FBase,
    mut addr_group_for_table_id: FAddrGroup,
    mut selector_group_for_table_id: FSelectorGroup,
    context: &str,
) -> Result<Vec<TraceShoutShape>, String>
where
    FBase: FnMut(u32, &[TraceShoutBusSpec]) -> Result<Option<(usize, usize)>, String>,
    FAddrGroup: FnMut(u32, usize) -> Option<u32>,
    FSelectorGroup: FnMut(u32) -> Option<u32>,
{
    let mut shape_by_table_id = HashMap::<u32, TraceShoutShape>::new();

    for &table_id in shout_table_ids {
        if let Some((ell_addr, n_vals)) = base_shape_for_table_id(table_id, extra_shout_specs)? {
            shape_by_table_id.insert(
                table_id,
                TraceShoutShape {
                    table_id,
                    ell_addr,
                    n_vals,
                    addr_group: addr_group_for_table_id(table_id, ell_addr),
                    selector_group: selector_group_for_table_id(table_id),
                },
            );
        }
    }

    for spec in extra_shout_specs {
        if spec.ell_addr == 0 {
            return Err(format!(
                "{context}: extra shout spec for table_id={} has ell_addr=0",
                spec.table_id
            ));
        }
        if spec.n_vals == 0 {
            return Err(format!(
                "{context}: extra shout spec for table_id={} has n_vals=0",
                spec.table_id
            ));
        }

        let inferred_addr_group = addr_group_for_table_id(spec.table_id, spec.ell_addr);
        let inferred_selector_group = selector_group_for_table_id(spec.table_id);
        if let Some(prev) = shape_by_table_id.get(&spec.table_id) {
            if prev.ell_addr != spec.ell_addr {
                return Err(format!(
                    "{context}: conflicting ell_addr for table_id={} (base/spec mismatch: {} vs {})",
                    spec.table_id, prev.ell_addr, spec.ell_addr
                ));
            }
            if prev.n_vals != spec.n_vals {
                return Err(format!(
                    "{context}: conflicting n_vals for table_id={} (base/spec mismatch: {} vs {})",
                    spec.table_id, prev.n_vals, spec.n_vals
                ));
            }
            if prev.addr_group != inferred_addr_group {
                return Err(format!(
                    "{context}: conflicting addr_group for table_id={} (base/spec mismatch: {:?} vs {:?})",
                    spec.table_id, prev.addr_group, inferred_addr_group
                ));
            }
            if prev.selector_group != inferred_selector_group {
                return Err(format!(
                    "{context}: conflicting selector_group for table_id={} (base/spec mismatch: {:?} vs {:?})",
                    spec.table_id, prev.selector_group, inferred_selector_group
                ));
            }
        } else {
            shape_by_table_id.insert(
                spec.table_id,
                TraceShoutShape {
                    table_id: spec.table_id,
                    ell_addr: spec.ell_addr,
                    n_vals: spec.n_vals,
                    addr_group: inferred_addr_group,
                    selector_group: inferred_selector_group,
                },
            );
        }
    }

    let mut shapes: Vec<TraceShoutShape> = shape_by_table_id.into_values().collect();
    shapes.sort_unstable_by_key(|shape| shape.table_id);
    Ok(shapes)
}

pub(crate) fn audit_bus_tail_constraint_coverage(
    builder: &CpuConstraintBuilder<F>,
    bus: &BusLayout,
    context: &str,
) -> Result<(), String> {
    let mut referenced = vec![false; bus.bus_cols];
    let bus_end = bus
        .bus_base
        .checked_add(bus.bus_region_len())
        .ok_or_else(|| format!("{context}: bus tail end overflow during coverage audit"))?;

    let mut mark_col = |col: usize| {
        if col >= bus.bus_base && col < bus_end {
            let rel = col - bus.bus_base;
            let col_id = rel / bus.chunk_size;
            if col_id < referenced.len() {
                referenced[col_id] = true;
            }
        }
    };

    for c in builder.constraints() {
        mark_col(c.condition_col);
        for &col in &c.additional_condition_cols {
            mark_col(col);
        }
        for &(col, _) in &c.b_terms {
            mark_col(col);
        }
    }

    let dead: Vec<usize> = referenced
        .iter()
        .enumerate()
        .filter_map(|(i, used)| if *used { None } else { Some(i) })
        .collect();

    if dead.is_empty() {
        return Ok(());
    }

    let preview: Vec<usize> = dead.iter().copied().take(24).collect();
    Err(format!(
        "{context}: dead bus-tail columns are not referenced by constraints (count={}, first={preview:?})",
        dead.len()
    ))
}
