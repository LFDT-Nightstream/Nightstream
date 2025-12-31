#![allow(non_snake_case)]

#[path = "common/fixtures.rs"]
mod fixtures;

use fixtures::{build_twist_shout_2step_fixture, prove};
use neo_fold::pi_ccs::FoldingMode;
use neo_math::K;
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::mle::chi_at_index;

#[test]
fn shared_cpu_bus_copyout_indices_match_bus_layout() {
    let fx = build_twist_shout_2step_fixture(123);
    let proof = prove(FoldingMode::Optimized, &fx);

    let step0_wit = &fx.steps_witness[0];
    let step0_inst = &fx.steps_instance[0];
    let ccs_out0 = &proof.steps[0].fold.ccs_out[0];

    let s0 = fx.ccs.ensure_identity_first().expect("identity-first");
    let base_t = s0.t();

    let bus = build_bus_layout_for_instances(
        s0.m,
        step0_inst.mcs_inst.m_in,
        1,
        step0_inst.lut_insts.iter().map(|inst| inst.d * inst.ell),
        step0_inst.mem_insts.iter().map(|inst| inst.d * inst.ell),
    )
    .expect("bus layout");

    assert_eq!(
        ccs_out0.y_scalars.len(),
        base_t + bus.bus_cols,
        "copyout matrices count must match bus_cols"
    );

    let bus_y_base = ccs_out0.y_scalars.len() - bus.bus_cols;

    let mut z = Vec::new();
    z.extend_from_slice(&step0_wit.mcs.0.x);
    z.extend_from_slice(&step0_wit.mcs.1.w);

    let time_row = bus.time_index(0);
    let chi = chi_at_index(&ccs_out0.r, time_row);

    let shout0 = &bus.shout_cols[0];
    let twist0 = &bus.twist_cols[0];
    let col_ids = [shout0.has_lookup, shout0.val, twist0.has_write, twist0.wv, twist0.inc];

    for col_id in col_ids {
        let z_idx = bus.bus_cell(col_id, 0);
        let expected: K = z[z_idx].into();
        let expected = expected * chi;
        let actual = ccs_out0.y_scalars[bus.y_scalar_index(bus_y_base, col_id)];
        assert_eq!(actual, expected, "copyout mismatch at col_id={col_id}");
    }
}
