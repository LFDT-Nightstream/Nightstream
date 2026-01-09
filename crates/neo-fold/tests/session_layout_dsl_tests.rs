use neo_fold::session::{
    Lane, Public, Scalar, SharedBusResources, ShoutPort, TwistPort, TwistPortWithInc, WitnessLayout,
    WitnessLayoutAllocator,
};
use neo_fold::session::witness_layout;
use neo_math::F;
use neo_memory::riscv::lookups::RiscvOpcode;
use neo_memory::witness::LutTableSpec;
use neo_vm_trace::{ShoutEvent, ShoutId, StepTrace, TwistEvent, TwistId, TwistOpKind};
use p3_field::PrimeCharacteristicRing;

witness_layout! {
    pub TestLayout<const N: usize> {
        pub one: Public<Scalar>,
        pub a: Lane<N>,
        pub twist0: TwistPort<N>,
        pub shout0: ShoutPort<N>,
    }
}

witness_layout! {
    pub TestLayoutWithInc<const N: usize> {
        pub one: Public<Scalar>,
        pub twist0: TwistPortWithInc<N>,
    }
}

#[test]
fn witness_layout_allocates_contiguously() {
    type L = TestLayout<4>;
    assert_eq!(L::M_IN, 1);
    assert_eq!(L::USED_COLS, 1 + 4 + 6 * 4 + 3 * 4);

    let l = L::new();
    assert_eq!(l.one, 0);
    assert_eq!(l.a.base(), 1);
    assert_eq!(l.a.at(0), 1);
    assert_eq!(l.a.at(3), 4);

    assert_eq!(l.twist0.has_read.base(), 5);
    assert_eq!(l.twist0.has_write.base(), 9);
    assert_eq!(l.twist0.read_addr.base(), 13);
    assert_eq!(l.twist0.write_addr.base(), 17);
    assert_eq!(l.twist0.rv.base(), 21);
    assert_eq!(l.twist0.wv.base(), 25);

    assert_eq!(l.shout0.has_lookup.base(), 29);
    assert_eq!(l.shout0.addr.base(), 33);
    assert_eq!(l.shout0.val.base(), 37);

    let tb = l.twist0.cpu_binding();
    assert_eq!(tb.has_read, l.twist0.has_read.base());
    assert_eq!(tb.has_write, l.twist0.has_write.base());
    assert_eq!(tb.read_addr, l.twist0.read_addr.base());
    assert_eq!(tb.write_addr, l.twist0.write_addr.base());
    assert_eq!(tb.rv, l.twist0.rv.base());
    assert_eq!(tb.wv, l.twist0.wv.base());
    assert!(tb.inc.is_none());

    let sb = l.shout0.cpu_binding();
    assert_eq!(sb.has_lookup, l.shout0.has_lookup.base());
    assert_eq!(sb.addr, l.shout0.addr.base());
    assert_eq!(sb.val, l.shout0.val.base());
}

#[test]
fn witness_layout_with_inc_binds_inc() {
    type L = TestLayoutWithInc<2>;
    assert_eq!(L::M_IN, 1);
    assert_eq!(L::USED_COLS, 1 + 7 * 2);

    let l = L::new();
    let tb = l.twist0.cpu_binding();
    assert_eq!(tb.inc, Some(l.twist0.inc.base()));
}

#[test]
#[should_panic(expected = "public columns must be allocated before private columns")]
fn allocator_rejects_public_after_private() {
    let mut alloc = WitnessLayoutAllocator::new();
    let _ = alloc.scalar();
    let _ = alloc.public_scalar();
}

#[test]
fn shared_bus_resources_resolve_shout_conflicts() {
    let mut r = SharedBusResources::new();
    r.set_binary_table(0, vec![F::ZERO, F::ONE]);
    assert!(r.lut_tables.contains_key(&0));
    assert!(!r.lut_table_specs.contains_key(&0));

    r.shout(0).spec(LutTableSpec::RiscvOpcode {
        opcode: RiscvOpcode::And,
        xlen: 32,
    });
    assert!(!r.lut_tables.contains_key(&0));
    assert!(r.lut_table_specs.contains_key(&0));
}

#[test]
fn shared_bus_resources_binary_table_uses_pow2_geometry() {
    let mut r = SharedBusResources::new();
    r.set_binary_table(7, (0u64..8).map(F::from_u64).collect());

    let t = r.lut_tables.get(&7).expect("table inserted");
    assert_eq!(t.k, 8);
    assert_eq!(t.n_side, 2);
    assert_eq!(t.d, 3, "expected d = log2(k) for binary tables");
}

#[test]
fn lane_set_helpers_write_expected_cells() {
    type L = TestLayout<4>;
    let l = L::new();
    let mut z = <L as WitnessLayout>::zero_witness_prefix();

    l.a.set(&mut z, 2, F::from_u64(123));
    assert_eq!(z[l.a.at(2)], F::from_u64(123));

    l.a.set_from_iter(
        &mut z,
        [F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)],
    )
    .expect("set_from_iter should succeed");
    assert_eq!(z[l.a.at(0)], F::from_u64(1));
    assert_eq!(z[l.a.at(3)], F::from_u64(4));

    assert!(
        l.a.set_from_iter(&mut z, [F::from_u64(9)]).is_err(),
        "too-few values should error"
    );
    assert!(
        l.a.set_from_iter(&mut z, vec![F::from_u64(7); 5]).is_err(),
        "too-many values should error"
    );
}

#[test]
fn ports_fill_from_trace_writes_expected_cells() {
    type L = TestLayout<2>;
    let l = L::new();
    let mut z = <L as WitnessLayout>::zero_witness_prefix();

    let step0 = StepTrace::<u64, u64> {
        cycle: 0,
        pc_before: 0,
        pc_after: 4,
        opcode: 0,
        regs_before: vec![0, 1],
        regs_after: vec![1, 1],
        twist_events: vec![
            TwistEvent {
                twist_id: TwistId(0),
                kind: TwistOpKind::Read,
                addr: 7,
                value: 9,
                lane: None,
            },
            TwistEvent {
                twist_id: TwistId(0),
                kind: TwistOpKind::Write,
                addr: 7,
                value: 10,
                lane: None,
            },
        ],
        shout_events: vec![ShoutEvent {
            shout_id: ShoutId(0),
            key: 1,
            value: 123,
        }],
        halted: false,
    };
    let step1 = StepTrace::<u64, u64> {
        cycle: 1,
        pc_before: 4,
        pc_after: 8,
        opcode: 0,
        regs_before: vec![1, 1],
        regs_after: vec![1, 2],
        twist_events: Vec::new(),
        shout_events: Vec::new(),
        halted: false,
    };
    let chunk = vec![step0, step1];

    l.shout0
        .fill_from_trace(&chunk, 0, &mut z)
        .expect("shout fill should succeed");
    assert_eq!(z[l.shout0.has_lookup.at(0)], F::ONE);
    assert_eq!(z[l.shout0.addr.at(0)], F::from_u64(1));
    assert_eq!(z[l.shout0.val.at(0)], F::from_u64(123));
    assert_eq!(z[l.shout0.has_lookup.at(1)], F::ZERO);

    l.twist0
        .fill_from_trace(&chunk, 0, &mut z)
        .expect("twist fill should succeed");
    assert_eq!(z[l.twist0.has_read.at(0)], F::ONE);
    assert_eq!(z[l.twist0.read_addr.at(0)], F::from_u64(7));
    assert_eq!(z[l.twist0.rv.at(0)], F::from_u64(9));
    assert_eq!(z[l.twist0.has_write.at(0)], F::ONE);
    assert_eq!(z[l.twist0.write_addr.at(0)], F::from_u64(7));
    assert_eq!(z[l.twist0.wv.at(0)], F::from_u64(10));
    assert_eq!(z[l.twist0.has_read.at(1)], F::ZERO);
    assert_eq!(z[l.twist0.has_write.at(1)], F::ZERO);
}

#[test]
fn ports_fill_from_trace_rejects_multiple_events() {
    type L = TestLayout<1>;
    let l = L::new();
    let mut z = <L as WitnessLayout>::zero_witness_prefix();

    let step = StepTrace::<u64, u64> {
        cycle: 0,
        pc_before: 0,
        pc_after: 0,
        opcode: 0,
        regs_before: Vec::new(),
        regs_after: Vec::new(),
        twist_events: vec![
            TwistEvent {
                twist_id: TwistId(0),
                kind: TwistOpKind::Read,
                addr: 0,
                value: 0,
                lane: None,
            },
            TwistEvent {
                twist_id: TwistId(0),
                kind: TwistOpKind::Read,
                addr: 1,
                value: 0,
                lane: None,
            },
        ],
        shout_events: vec![
            ShoutEvent {
                shout_id: ShoutId(0),
                key: 0,
                value: 0,
            },
            ShoutEvent {
                shout_id: ShoutId(0),
                key: 1,
                value: 0,
            },
        ],
        halted: false,
    };
    let chunk = vec![step];

    assert!(l.twist0.fill_from_trace(&chunk, 0, &mut z).is_err());
    assert!(l.shout0.fill_from_trace(&chunk, 0, &mut z).is_err());
}
