use neo_fold_clean::frontends::nebula::application::{
    ApplicationError, MemoryPort, MemoryPortActivation, MemoryPortKind, MemoryPortLayout, MemoryRegion,
    MemoryRegionKind,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn region(name: &str, kind: MemoryRegionKind, base: u64) -> MemoryRegion {
    MemoryRegion::new(name, kind, base, vec![2]).expect("test region")
}

fn port(kind: MemoryPortKind, activation: MemoryPortActivation) -> MemoryPort {
    MemoryPort::new(0, vec![0], 1, kind, activation)
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
fn layout_rejects_rom_writes_and_excess_ports() {
    let error = MemoryPortLayout::new(
        vec![region("rom", MemoryRegionKind::Rom, 0)],
        vec![port(
            MemoryPortKind::Write {
                value_before_column: None,
            },
            MemoryPortActivation::Always,
        )],
    )
    .expect_err("ROM write declaration must be rejected");
    assert!(matches!(error, ApplicationError::RomWritePort { slot: 0 }));

    let layout = MemoryPortLayout::new(
        vec![region("ram", MemoryRegionKind::Ram, 0)],
        vec![
            port(MemoryPortKind::Read, MemoryPortActivation::Always),
            port(MemoryPortKind::Read, MemoryPortActivation::Always),
        ],
    )
    .expect("port declarations");
    assert!(matches!(
        layout.validate_for(2, &params()),
        Err(ApplicationError::TooManyPorts { ports: 2, slots: 1 })
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
        vec![port(MemoryPortKind::Read, MemoryPortActivation::Column(2))],
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
        vec![port(MemoryPortKind::Read, MemoryPortActivation::Always)],
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
        vec![port(
            MemoryPortKind::Write {
                value_before_column: Some(2),
            },
            MemoryPortActivation::Always,
        )],
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
