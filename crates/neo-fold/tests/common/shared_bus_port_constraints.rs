#![allow(dead_code)]

use neo_fold::session::{CcsBuilder, ShoutPort, TwistPortWithInc};
use neo_math::F;
use neo_memory::cpu::constraints::{CpuConstraint, CpuConstraintBuilder};
use neo_memory::cpu::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutCpuBinding, ShoutInstanceShape,
    TwistCpuBinding,
};
use p3_field::PrimeCharacteristicRing;

pub fn constrain_boolean(cs: &mut CcsBuilder<F>, one: usize, col: usize) {
    cs.r1cs_terms([(col, F::ONE)], [(col, F::ONE), (one, -F::ONE)], []);
}

pub fn constrain_zero_when_inactive(cs: &mut CcsBuilder<F>, one: usize, flag: usize, field: usize) {
    cs.r1cs_terms([(one, F::ONE), (flag, -F::ONE)], [(field, F::ONE)], []);
}

pub fn constrain_shout_port<const N: usize>(cs: &mut CcsBuilder<F>, one: usize, port: &ShoutPort<N>) {
    for j in 0..N {
        let has_lookup = port.has_lookup.at(j);
        constrain_boolean(cs, one, has_lookup);
        constrain_zero_when_inactive(cs, one, has_lookup, port.addr.at(j));
        constrain_zero_when_inactive(cs, one, has_lookup, port.val.at(j));
    }
}

pub fn constrain_twist_port<const N: usize>(cs: &mut CcsBuilder<F>, one: usize, port: &TwistPortWithInc<N>) {
    for j in 0..N {
        let has_read = port.has_read.at(j);
        let has_write = port.has_write.at(j);
        constrain_boolean(cs, one, has_read);
        constrain_boolean(cs, one, has_write);
        constrain_zero_when_inactive(cs, one, has_read, port.read_addr.at(j));
        constrain_zero_when_inactive(cs, one, has_read, port.rv.at(j));
        constrain_zero_when_inactive(cs, one, has_write, port.write_addr.at(j));
        constrain_zero_when_inactive(cs, one, has_write, port.wv.at(j));
        constrain_zero_when_inactive(cs, one, has_write, port.inc.at(j));
    }
}

pub fn append_cpu_constraint(cs: &mut CcsBuilder<F>, one: usize, constraint: &CpuConstraint<F>) {
    let mut a_terms = Vec::with_capacity(1 + constraint.additional_condition_cols.len());
    if constraint.negate_condition {
        a_terms.push((one, F::ONE));
        a_terms.push((constraint.condition_col, -F::ONE));
        for &col in &constraint.additional_condition_cols {
            a_terms.push((col, -F::ONE));
        }
    } else {
        a_terms.push((constraint.condition_col, F::ONE));
        for &col in &constraint.additional_condition_cols {
            a_terms.push((col, F::ONE));
        }
    }
    cs.r1cs_terms(a_terms, constraint.b_terms.iter().copied(), []);
}

pub fn append_cpu_constraints(cs: &mut CcsBuilder<F>, one: usize, constraints: &[CpuConstraint<F>]) {
    for constraint in constraints {
        append_cpu_constraint(cs, one, constraint);
    }
}

pub fn append_unshared_bus_binding_constraints(
    cs: &mut CcsBuilder<F>,
    one: usize,
    used_cols: usize,
    chunk_size: usize,
    shout_shapes: &[ShoutInstanceShape],
    shout_bindings: &[Vec<ShoutCpuBinding>],
    twist_ell_addrs_and_lanes: &[(usize, usize)],
    twist_bindings: &[Vec<TwistCpuBinding>],
) -> Result<(), String> {
    for shape in shout_shapes {
        if shape.addr_group.is_some() || shape.selector_group.is_some() {
            return Err("append_unshared_bus_binding_constraints only supports unshared shout groups".into());
        }
    }

    let shout_bus_cols = shout_shapes
        .iter()
        .map(|shape| shape.lanes.max(1) * (shape.ell_addr + 1 + shape.n_vals.max(1)))
        .sum::<usize>();
    let twist_bus_cols = twist_ell_addrs_and_lanes
        .iter()
        .map(|(ell_addr, lanes)| (*lanes).max(1) * (2 * ell_addr + 5))
        .sum::<usize>();
    let bus_cols = shout_bus_cols
        .checked_add(twist_bus_cols)
        .ok_or_else(|| "bus_cols overflow".to_string())?;
    let bus_region_len = bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| "bus_region_len overflow".to_string())?;
    let m = used_cols
        .checked_add(bus_region_len)
        .ok_or_else(|| "used_cols + bus_region_len overflow".to_string())?;

    let bus_layout = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m,
        cs.m_in(),
        chunk_size,
        shout_shapes.iter().cloned(),
        twist_ell_addrs_and_lanes.iter().copied(),
    )?;

    let mut builder = CpuConstraintBuilder::<F>::new(/*n=*/ 1, /*m=*/ m, one);
    if shout_shapes.len() != shout_bindings.len() {
        return Err(format!(
            "shout_shapes len {} != shout_bindings len {}",
            shout_shapes.len(),
            shout_bindings.len()
        ));
    }
    if twist_ell_addrs_and_lanes.len() != twist_bindings.len() {
        return Err(format!(
            "twist specs len {} != twist_bindings len {}",
            twist_ell_addrs_and_lanes.len(),
            twist_bindings.len()
        ));
    }

    for (i, cpus) in shout_bindings.iter().enumerate() {
        let inst_cols = &bus_layout.shout_cols[i];
        if cpus.len() != inst_cols.lanes.len() {
            return Err(format!(
                "shout binding lanes mismatch for instance {i}: {} vs {}",
                cpus.len(),
                inst_cols.lanes.len()
            ));
        }
        for (lane_idx, cpu) in cpus.iter().enumerate() {
            builder.add_shout_instance_bound(&bus_layout, &inst_cols.lanes[lane_idx], cpu);
        }
    }

    for (i, cpus) in twist_bindings.iter().enumerate() {
        let inst_cols = &bus_layout.twist_cols[i];
        if cpus.len() != inst_cols.lanes.len() {
            return Err(format!(
                "twist binding lanes mismatch for instance {i}: {} vs {}",
                cpus.len(),
                inst_cols.lanes.len()
            ));
        }
        for (lane_idx, cpu) in cpus.iter().enumerate() {
            builder.add_twist_instance_bound(&bus_layout, &inst_cols.lanes[lane_idx], cpu);
        }
    }

    append_cpu_constraints(cs, one, builder.constraints());
    Ok(())
}
