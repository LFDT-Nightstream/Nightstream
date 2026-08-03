use neo_fold_clean::frontends::nebula::application::{
    ApplicationError, MemoryOpSlot, MemoryPort, MemoryPortActivation, MemoryPortKind, MemoryPortLayout, MemoryRegion,
    MemoryRegionKind,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn region(name: &str, kind: MemoryRegionKind, base: u64) -> MemoryRegion {
    MemoryRegion::new(name, kind, base, vec![2]).expect("test region")
}

fn port(region: usize, kind: MemoryPortKind, activation: MemoryPortActivation) -> MemoryPort {
    MemoryPort::new(region, vec![0], 1, kind, activation)
}

fn singleton(port: MemoryPort) -> MemoryOpSlot {
    MemoryOpSlot::new(vec![port])
}

fn params() -> NebulaParams {
    NebulaParams::new(2, 2, 1, 1, 4).expect("test geometry")
}

#[test]
fn logical_regions_reject_same_namespace_aliases() {
    let error = MemoryPortLayout::new(
        vec![
            region("left", MemoryRegionKind::Ram, 0),
            region("right", MemoryRegionKind::Ram, 2),
        ],
        Vec::new(),
    )
    .expect_err("overlapping RAM ranges must be rejected");
    assert!(matches!(error, ApplicationError::OverlappingRegions { .. }));

    MemoryPortLayout::new(
        vec![
            region("rom", MemoryRegionKind::Rom, 0),
            region("ram", MemoryRegionKind::Ram, 0),
        ],
        Vec::new(),
    )
    .expect("ROM and RAM are disjoint protocol namespaces");
}

#[test]
fn layout_rejects_rom_writes_and_excess_slots() {
    let empty = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![MemoryOpSlot::new(Vec::new())],
    )
    .expect_err("physical slots must have at least one candidate");
    assert!(matches!(empty, ApplicationError::EmptyMemorySlot { slot: 0 }));

    let error = MemoryPortLayout::new(
        vec![region("rom", MemoryRegionKind::Rom, 0)],
        vec![singleton(port(
            0,
            MemoryPortKind::Write {
                value_before_column: None,
            },
            MemoryPortActivation::Always,
        ))],
    )
    .expect_err("ROM write declaration must be rejected");
    assert!(matches!(error, ApplicationError::RomWritePort { slot: 0 }));

    let compacted = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![MemoryOpSlot::new(vec![
            port(0, MemoryPortKind::Read, MemoryPortActivation::Column(2)),
            port(0, MemoryPortKind::Read, MemoryPortActivation::Column(3)),
        ])],
    )
    .expect("compacted slot declaration");
    compacted
        .validate_for(4, &params())
        .expect("two logical ports fit in one physical slot");

    let layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![
            singleton(port(0, MemoryPortKind::Read, MemoryPortActivation::Always)),
            singleton(port(0, MemoryPortKind::Read, MemoryPortActivation::Always)),
        ],
    )
    .expect("slot declarations");
    assert!(matches!(
        layout.validate_for(2, &params()),
        Err(ApplicationError::TooManySlots {
            declared: 2,
            available: 1
        })
    ));
}

#[test]
fn mixed_radix_components_and_activation_are_range_checked() {
    let ram = region("ram", MemoryRegionKind::Ram, 0);
    assert!(matches!(
        ram.address(&[4]),
        Err(ApplicationError::AddressComponentRange { value: 4, bits: 2, .. })
    ));

    let layout = MemoryPortLayout::new(
        vec![ram],
        vec![singleton(port(
            0,
            MemoryPortKind::Read,
            MemoryPortActivation::Column(2),
        ))],
    )
    .expect("layout");
    let geometry = params();
    let rom = vec![0; geometry.rom_cells() as usize];
    let mut memory = Memory::new(geometry, &rom).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");
    let assignment = [F::ZERO, F::ZERO, F::from_u64(2)];
    assert!(matches!(
        layout.execute_assignment(&mut segment, &assignment),
        Err(ApplicationError::NonBooleanActivation { slot: 0, value: 2 })
    ));
}

#[test]
fn read_and_rmw_values_are_checked_against_real_memory() {
    let read_layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![singleton(port(0, MemoryPortKind::Read, MemoryPortActivation::Always))],
    )
    .expect("read layout");
    let geometry = params();
    let rom = vec![0; geometry.rom_cells() as usize];
    let ram = vec![9, 0, 0, 0];
    let mut memory = Memory::new_with_initial_ram(geometry, &rom, &ram).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");
    assert!(matches!(
        read_layout.execute_assignment(&mut segment, &[F::ZERO, F::from_u64(8)]),
        Err(ApplicationError::ReadValueMismatch {
            expected: 9,
            actual: 8,
            ..
        })
    ));

    let write_layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![singleton(port(
            0,
            MemoryPortKind::Write {
                value_before_column: Some(2),
            },
            MemoryPortActivation::Always,
        ))],
    )
    .expect("write layout");
    let mut memory = Memory::new_with_initial_ram(geometry, &rom, &ram).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");
    assert!(matches!(
        write_layout.execute_assignment(&mut segment, &[F::ZERO, F::from_u64(5), F::from_u64(8)],),
        Err(ApplicationError::WriteBeforeMismatch {
            expected: 9,
            actual: 8,
            ..
        })
    ));
}

#[test]
fn one_physical_slot_selects_one_logical_port() {
    let layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![MemoryOpSlot::new(vec![
            MemoryPort::new(0, vec![0], 1, MemoryPortKind::Read, MemoryPortActivation::Column(4)),
            MemoryPort::new(0, vec![2], 3, MemoryPortKind::Read, MemoryPortActivation::Column(5)),
        ])],
    )
    .expect("candidate port layout");
    assert_eq!(layout.slot_count(), 1);
    assert_eq!(layout.logical_port_count(), 2);

    let geometry = params();
    let rom = vec![0; geometry.rom_cells() as usize];
    let ram = vec![9, 8, 0, 0];
    let mut memory = Memory::new_with_initial_ram(geometry, &rom, &ram).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");

    let first = layout
        .execute_assignment(
            &mut segment,
            &[F::ZERO, F::from_u64(9), F::ONE, F::from_u64(8), F::ONE, F::ZERO],
        )
        .expect("first candidate");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].expect("active first candidate").addr, 0);

    let second = layout
        .execute_assignment(
            &mut segment,
            &[F::ZERO, F::from_u64(9), F::ONE, F::from_u64(8), F::ZERO, F::ONE],
        )
        .expect("second candidate");
    assert_eq!(second[0].expect("active second candidate").addr, 1);

    let padded = layout
        .execute_assignment(
            &mut segment,
            &[F::ZERO, F::from_u64(9), F::ONE, F::from_u64(8), F::ZERO, F::ZERO],
        )
        .expect("inactive slot");
    assert_eq!(padded, vec![None]);
}

#[test]
fn physical_slot_rejects_multiple_active_candidates() {
    let layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![MemoryOpSlot::new(vec![
            port(0, MemoryPortKind::Read, MemoryPortActivation::Column(2)),
            port(0, MemoryPortKind::Read, MemoryPortActivation::Column(3)),
        ])],
    )
    .expect("candidate port layout");
    let geometry = params();
    let rom = vec![0; geometry.rom_cells() as usize];
    let mut memory = Memory::new(geometry, &rom).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");

    assert!(matches!(
        layout.execute_assignment(&mut segment, &[F::ZERO, F::ZERO, F::ONE, F::ONE]),
        Err(ApplicationError::MemorySlotCollision {
            slot: 0,
            first: 0,
            second: 1
        })
    ));
}

#[test]
fn physical_slot_can_select_between_rom_and_ram() {
    let layout = MemoryPortLayout::new(
        vec![
            region("rom", MemoryRegionKind::Rom, 0),
            region("ram", MemoryRegionKind::Ram, 0),
        ],
        vec![MemoryOpSlot::new(vec![
            MemoryPort::new(0, vec![0], 1, MemoryPortKind::Read, MemoryPortActivation::Column(4)),
            MemoryPort::new(1, vec![2], 3, MemoryPortKind::Read, MemoryPortActivation::Column(5)),
        ])],
    )
    .expect("cross-namespace candidate port");
    let geometry = params();
    let rom = vec![7, 0, 0, 0];
    let ram = vec![0, 8, 0, 0];
    let mut memory = Memory::new_with_initial_ram(geometry, &rom, &ram).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");

    let rom_read = layout
        .execute_assignment(
            &mut segment,
            &[F::ZERO, F::from_u64(7), F::ONE, F::from_u64(8), F::ONE, F::ZERO],
        )
        .expect("ROM candidate")[0]
        .expect("active ROM candidate");
    assert_eq!(rom_read.space, neo_fold_clean::frontends::nebula::layout::MemSpace::Rom);

    let ram_read = layout
        .execute_assignment(
            &mut segment,
            &[F::ZERO, F::from_u64(7), F::ONE, F::from_u64(8), F::ZERO, F::ONE],
        )
        .expect("RAM candidate")[0]
        .expect("active RAM candidate");
    assert_eq!(ram_read.space, neo_fold_clean::frontends::nebula::layout::MemSpace::Ram);
}
