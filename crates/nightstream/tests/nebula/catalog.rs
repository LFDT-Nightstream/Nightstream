//! The catalog backend on a test catalog: a ROM with a two-part address, a
//! heap RAM whose write names its prior column, a stack RAM, and a
//! host-checked RAM; `When` and `Unless` ports; and three slots, two of them
//! shared. The native ports must bind the events of a Rust copy of Lean
//! `eventsForRow` in order. Each broken row must hold the memory rows and fail
//! the binding rows: a circuit without the binding rows accepts its witness,
//! and the circuit with them rejects it. Each broken row escapes every
//! binding row but one kind, so each kind of row has a witness that needs it.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use crate::application::ApplicationCircuit;
use crate::nebula::circuit::memory_circuit;
use crate::nebula::segment::initial_memory;
use crate::nebula::{
    Application, CatalogBinding, CatalogBindingError, CatalogRowError, ColumnFamilySpec, ColumnRegistry, ColumnWidth,
    Context, Expr, Geometry, Invocation, Memory, MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind,
    MemoryPortSpec, MemoryPreload, MemorySpec, PortAccess, Program, Region, Run, Transition, View,
};

const FETCH: usize = 0;
const STORE: usize = 1;
const PUSH: usize = 2;
const POP: usize = 3;
const CODE_LOW: usize = 4;
const CODE_HIGH: usize = 5;
const CODE_VALUE: usize = 6;
const HEAP_READ: usize = 7;
const HEAP_READ_VALUE: usize = 8;
const HEAP_WRITE: usize = 9;
const HEAP_WRITE_VALUE: usize = 10;
const HEAP_PRIOR: usize = 11;
const PUSH_ADDRESS: usize = 12;
const PUSH_VALUE: usize = 13;
const POP_ADDRESS: usize = 14;
const POP_VALUE: usize = 15;
const IO_ADDRESS: usize = 16;
const IO_VALUE: usize = 17;
const COLUMNS: usize = 18;

/// Checked-port ranks: code fetch 0, heap read 1, heap write 2, push 3, pop 4.
/// Fetch and the heap read are exclusive by construction. The pop sits
/// before the push, so a row that pushes and pops is out of catalog order.
const PLAN: [&[usize]; 3] = [&[0, 1], &[2, 4], &[3]];

/// ROM of 4 cells and RAM of 8, three slots, one segment of four steps.
fn geometry() -> Geometry {
    Geometry {
        r: 2,
        mu: 3,
        w_ts: 4,
        b_ops: 3,
        b_scan: 3,
        n: 4,
        s_max: 1,
    }
}

fn port(address: &[usize], value: usize, kind: MemoryPortKind, activation: MemoryPortActivation) -> MemoryPortSpec {
    MemoryPortSpec {
        address_columns: address.to_vec(),
        value_column: value,
        kind,
        activation,
    }
}

fn registry(flags: usize, columns: usize) -> ColumnRegistry {
    let family = |start, len, name, width| ColumnFamilySpec {
        region: "test",
        start,
        len,
        name,
        role: "test",
        width,
    };
    ColumnRegistry::new([
        family(0, flags, "flags", ColumnWidth::Boolean),
        family(flags, columns - flags, "words", ColumnWidth::U32),
    ])
    .unwrap()
}

const READ: MemoryPortKind = MemoryPortKind::Read;

fn write(prior: Option<usize>) -> MemoryPortKind {
    MemoryPortKind::Write {
        value_before_column: prior,
    }
}

/// Memories 0 (code ROM), 1 (heap), 2 (stack), and 3 (host-checked I/O).
fn catalog() -> MemoryCatalog<u32> {
    let memory = |id, kind, ports| MemorySpec { id, kind, ports };
    MemoryCatalog::new(
        [
            memory(
                0,
                MemoryKind::Rom,
                vec![port(
                    &[CODE_LOW, CODE_HIGH],
                    CODE_VALUE,
                    READ,
                    MemoryPortActivation::When(FETCH),
                )],
            ),
            memory(
                1,
                MemoryKind::Ram,
                vec![
                    port(&[HEAP_READ], HEAP_READ_VALUE, READ, MemoryPortActivation::Unless(FETCH)),
                    port(
                        &[HEAP_WRITE],
                        HEAP_WRITE_VALUE,
                        write(Some(HEAP_PRIOR)),
                        MemoryPortActivation::When(STORE),
                    ),
                ],
            ),
            memory(
                2,
                MemoryKind::Ram,
                vec![
                    port(
                        &[PUSH_ADDRESS],
                        PUSH_VALUE,
                        write(None),
                        MemoryPortActivation::When(PUSH),
                    ),
                    port(&[POP_ADDRESS], POP_VALUE, READ, MemoryPortActivation::When(POP)),
                ],
            ),
            memory(
                3,
                MemoryKind::Ram,
                vec![port(&[IO_ADDRESS], IO_VALUE, write(None), MemoryPortActivation::Always)],
            ),
        ],
        &registry(4, COLUMNS),
    )
    .unwrap()
}

fn region(base: u64, widths: &[usize]) -> Region {
    Region {
        base,
        widths: widths.to_vec(),
    }
}

fn regions() -> Vec<(u32, Region)> {
    vec![(0, region(0, &[1, 1])), (1, region(0, &[2])), (2, region(4, &[1]))]
}

fn plan() -> Vec<Vec<usize>> {
    PLAN.iter().map(|slot| slot.to_vec()).collect()
}

/// A catalog with its geometry, program, binding, and column count.
struct Setup {
    geometry: Geometry,
    program: Program,
    binding: CatalogBinding<u32>,
    columns: usize,
}

fn setup() -> Setup {
    Setup {
        geometry: geometry(),
        program: Program::new(geometry(), vec![11, 12, 13, 14], vec![0, 0, 3, 0, 0, 0, 0, 0]).unwrap(),
        binding: CatalogBinding::new(&catalog(), &geometry(), regions(), &plan()).unwrap(),
        columns: COLUMNS,
    }
}

/// Ports that a forged step gives in place of the binding's.
type Forge = fn(&[F], &Memory) -> Vec<Option<PortAccess>>;

#[derive(Clone, Debug)]
struct CatalogApp {
    binding: CatalogBinding<u32>,
    columns: usize,
    /// `false`: no row at all, so only the memory rows check a witness.
    bound: bool,
    forge: Option<Forge>,
}

impl Application for CatalogApp {
    type Input = Vec<F>;
    type State = ();
    type Error = CatalogRowError;

    fn width(&self) -> usize {
        0
    }

    fn aux(&self) -> usize {
        self.columns
    }

    fn words(&self, _: &()) -> Vec<u64> {
        Vec::new()
    }

    fn rows(&self, view: &View) -> Vec<Expr> {
        let columns: Vec<Expr> = (0..self.columns).map(|column| view.aux(column)).collect();
        if self.bound {
            self.binding.rows(view, &columns)
        } else {
            Vec::new()
        }
    }

    fn step(&self, _: &(), row: &Vec<F>, memory: &Memory) -> Result<Transition<()>, CatalogRowError> {
        let ports = match self.forge {
            None => self.binding.ports(row, memory)?,
            Some(forge) => forge(row, memory),
        };
        Ok(Transition {
            next: (),
            ports,
            aux: row.clone(),
        })
    }
}

impl Setup {
    fn app(&self, bound: bool, forge: Option<Forge>) -> CatalogApp {
        CatalogApp {
            binding: self.binding.clone(),
            columns: self.columns,
            bound,
            forge,
        }
    }

    fn circuit(&self, bound: bool) -> ApplicationCircuit {
        memory_circuit(&self.geometry, &self.app(bound, None)).unwrap()
    }

    fn run(&self, forge: Option<Forge>, rows: &[Vec<F>]) -> (Run<CatalogApp>, Vec<Invocation>) {
        let mut run = Run::new(Context::new(self.app(true, forge), &self.program), ());
        let invocations = run.segment(rows).unwrap();
        (run, invocations)
    }

    /// The memory cells before any step.
    fn memory(&self) -> Vec<crate::nebula::Cell> {
        initial_memory(&self.program.rom, &self.program.ram)
    }

    /// Invocation `step` of `rows`, with `change` made to its columns, holds
    /// the memory rows and fails the binding rows.
    fn reject(&self, name: &str, forge: Option<Forge>, rows: &[Vec<F>], step: usize, change: &[(usize, u64)]) {
        let (run, invocations) = self.run(forge, rows);
        let mut words = invocations[step].words.clone();
        for (column, value) in change {
            *aux_word(&mut words, self.columns, *column) = F::from_u64(*value);
        }
        let input = match step {
            0 => run.initial_state(),
            _ => invocations[step - 1].output,
        };
        self.circuit(false)
            .execute(input, &words)
            .unwrap_or_else(|error| panic!("{name}: the memory rows must hold: {error}"));
        assert!(self.circuit(true).execute(input, &words).is_err(), "{name}");
    }
}

/// Word of column `column` in an invocation's words: the private words are
/// the last words, and the application has no state word.
fn aux_word(words: &mut [F], columns: usize, column: usize) -> &mut F {
    let start = words.len() - columns;
    &mut words[start + column]
}

fn row(values: &[(usize, u64)]) -> Vec<F> {
    let mut row = vec![F::ZERO; COLUMNS];
    for (column, value) in values {
        row[*column] = F::from_u64(*value);
    }
    row
}

/// Fetch ROM (0, 0) = 11 and nothing else.
fn fetch_row() -> Vec<F> {
    row(&[(FETCH, 1), (CODE_VALUE, 11)])
}

/// Four honest rows. Each writes the I/O memory, which no row checks.
fn honest_rows() -> Vec<Vec<F>> {
    vec![
        // Fetch ROM (1, 1) = 3; write heap 2 (3 → 7); push 9 to stack 1.
        row(&[
            (FETCH, 1),
            (CODE_LOW, 1),
            (CODE_HIGH, 1),
            (CODE_VALUE, 14),
            (STORE, 1),
            (HEAP_WRITE, 2),
            (HEAP_WRITE_VALUE, 7),
            (HEAP_PRIOR, 3),
            (PUSH, 1),
            (PUSH_ADDRESS, 1),
            (PUSH_VALUE, 9),
            (IO_ADDRESS, 5),
            (IO_VALUE, 99),
        ]),
        // Read heap 2 = 7; pop stack 1 = 9.
        row(&[
            (HEAP_READ, 2),
            (HEAP_READ_VALUE, 7),
            (POP, 1),
            (POP_ADDRESS, 1),
            (POP_VALUE, 9),
            (IO_VALUE, 1 << 40),
        ]),
        // Fetch ROM (0, 1) = 2; write heap 2 (7 → 8).
        row(&[
            (FETCH, 1),
            (CODE_HIGH, 1),
            (CODE_VALUE, 13),
            (STORE, 1),
            (HEAP_WRITE, 2),
            (HEAP_WRITE_VALUE, 8),
            (HEAP_PRIOR, 7),
        ]),
        // Read heap 2 = 8; push 5 to stack 0.
        row(&[(HEAP_READ, 2), (HEAP_READ_VALUE, 8), (PUSH, 1), (PUSH_VALUE, 5)]),
    ]
}

/// A Lean `MemoryTrace.Event` with the integer value of each column.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Read {
        memory: u32,
        address: Vec<u64>,
        value: u64,
    },
    Write {
        memory: u32,
        address: Vec<u64>,
        value: u64,
        prior: Option<u64>,
    },
}

/// Lean `active`.
fn active(w: &[F], activation: MemoryPortActivation) -> bool {
    match activation {
        MemoryPortActivation::Always => true,
        MemoryPortActivation::When(column) => w[column] == F::ONE,
        MemoryPortActivation::Unless(column) => w[column] == F::ZERO,
    }
}

/// Lean `portEvent`.
fn port_event(memory: u32, w: &[F], port: &MemoryPortSpec) -> Option<Event> {
    if !active(w, port.activation) {
        return None;
    }
    let value_of = |column: usize| w[column].as_canonical_u64();
    let address = port
        .address_columns
        .iter()
        .map(|column| value_of(*column))
        .collect();
    let value = value_of(port.value_column);
    Some(match port.kind {
        MemoryPortKind::Read => Event::Read { memory, address, value },
        MemoryPortKind::Write { value_before_column } => Event::Write {
            memory,
            address,
            value,
            prior: value_before_column.map(value_of),
        },
    })
}

/// Lean `eventsForRow`: catalog memories in order, then each memory's ports.
fn events_for_row(catalog: &[MemorySpec<u32>], w: &[F]) -> Vec<Event> {
    catalog
        .iter()
        .flat_map(|memory| {
            memory
                .ports
                .iter()
                .filter_map(|port| port_event(memory.id, w, port))
        })
        .collect()
}

/// The memories with a region: the catalog of the Lean theorem.
fn checked_catalog() -> Vec<MemorySpec<u32>> {
    let regions = regions();
    catalog()
        .entries()
        .iter()
        .filter(|memory| regions.iter().any(|(id, _)| *id == memory.id))
        .cloned()
        .collect()
}

/// Lean `packParts`.
fn pack_parts(widths: &[usize], parts: &[u64]) -> u64 {
    match (widths, parts) {
        ([w, ws @ ..], [x, xs @ ..]) => x + (1 << w) * pack_parts(ws, xs),
        _ => 0,
    }
}

fn region_of(memory: u32) -> Region {
    regions()
        .into_iter()
        .find(|(id, _)| *id == memory)
        .expect("a checked memory")
        .1
}

/// Lean `Binding.Binds` with the regions of `regions()`.
fn binds(event: &Event, access: &PortAccess) -> bool {
    let (memory, address) = match event {
        Event::Read { memory, address, .. } | Event::Write { memory, address, .. } => (*memory, address),
    };
    let region = region_of(memory);
    let is_ram = catalog().entries()[memory as usize].kind == MemoryKind::Ram;
    let at = access.is_ram == is_ram && access.addr == region.base + pack_parts(&region.widths, address);
    at && match event {
        Event::Read { value, .. } => !access.is_write && access.vr == *value,
        Event::Write { value, prior, .. } => {
            access.is_write && access.vw == *value && prior.is_none_or(|v| access.vr == v)
        }
    }
}

/// The access of the active port of rank `rank`, without the binding's
/// checks. A write without a prior column reads its cell after `earlier`.
fn access_of(rank: usize, w: &[F], memory: &Memory, earlier: &[Option<PortAccess>]) -> PortAccess {
    let (spec, port) = checked_catalog()
        .into_iter()
        .flat_map(|memory| {
            memory
                .ports
                .clone()
                .into_iter()
                .map(move |port| (memory.clone(), port))
        })
        .nth(rank)
        .unwrap();
    let region = region_of(spec.id);
    let value_of = |column: usize| w[column].as_canonical_u64();
    let parts: Vec<u64> = port
        .address_columns
        .iter()
        .map(|column| value_of(*column))
        .collect();
    let (is_ram, addr) = (
        spec.kind == MemoryKind::Ram,
        region.base + pack_parts(&region.widths, &parts),
    );
    let value = value_of(port.value_column);
    let vr = match port.kind {
        MemoryPortKind::Read => value,
        MemoryPortKind::Write {
            value_before_column: Some(prior),
        } => value_of(prior),
        MemoryPortKind::Write {
            value_before_column: None,
        } => earlier
            .iter()
            .flatten()
            .rfind(|access| access.is_ram == is_ram && access.addr == addr)
            .map(|access| access.vw)
            .or_else(|| if is_ram { memory.ram(addr) } else { memory.rom(addr) })
            .unwrap(),
    };
    PortAccess {
        is_write: matches!(port.kind, MemoryPortKind::Write { .. }),
        is_ram,
        addr,
        vr,
        vw: value,
    }
}

/// Each slot shows its first active port, in slot order.
fn forged_ports(w: &[F], memory: &Memory) -> Vec<Option<PortAccess>> {
    let activations: Vec<MemoryPortActivation> = checked_catalog()
        .iter()
        .flat_map(|memory| memory.ports.iter().map(|port| port.activation))
        .collect();
    let mut ports = Vec::new();
    for slot in PLAN {
        let shown = slot
            .iter()
            .copied()
            .find(|rank| active(w, activations[*rank]))
            .map(|rank| access_of(rank, w, memory, &ports));
        ports.push(shown);
    }
    ports
}

fn rom_read(addr: u64, value: u64) -> PortAccess {
    PortAccess {
        is_write: false,
        is_ram: false,
        addr,
        vr: value,
        vw: value,
    }
}

#[test]
fn honest_rows_bind_the_lean_events_in_order_and_hold() {
    let setup = setup();
    let (run, invocations) = setup.run(None, &honest_rows());
    let checked = checked_catalog();
    let mut memory = setup.memory();
    for (k, w) in honest_rows().iter().enumerate() {
        let ports = setup
            .binding
            .ports(w, &Memory::new(&setup.geometry, &memory))
            .unwrap();
        let accesses: Vec<PortAccess> = ports.iter().flatten().copied().collect();
        let events = events_for_row(&checked, w);
        assert_eq!(accesses.len(), events.len(), "row {k}: one access per event");
        for (event, access) in events.iter().zip(&accesses) {
            assert!(binds(event, access), "row {k}: {access:?} binds {event:?}");
        }
        for access in accesses {
            memory[access.global_index(&setup.geometry)].value = access.vw;
        }
    }
    assert_eq!(setup.binding.host_checked(), &[3]);

    let (bound, unbound) = (setup.circuit(true), setup.circuit(false));
    let mut input = run.initial_state();
    for invocation in &invocations {
        bound.execute(input, &invocation.words).unwrap();
        input = invocation.output;
    }
    assert_eq!(bound.private_input_count(), unbound.private_input_count());
    assert_eq!(
        bound.generated_range().len(),
        unbound.generated_range().len(),
        "the binding rows add no witness word"
    );
    eprintln!(
        "binding rows={} (5 checked ports in 3 slots, 4 activation columns)",
        bound.row_count() - unbound.row_count()
    );
}

/// Rows that need a dishonest slot: the forged ports give it and keep every
/// memory row true. The binding's ports refuse the row or give other slots.
#[test]
fn binding_rows_reject_each_forged_slot() {
    let setup = setup();
    let fetch = || vec![fetch_row(); 4];
    let lead = |first: Vec<F>| [vec![first], vec![fetch_row(); 3]].concat();
    let cases: [(&str, Vec<Vec<F>>, Forge); 4] = [
        // Push 9 to stack 1 and pop stack 1. In catalog order the pop reads
        // 9; in slot order it runs first and reads 0.
        (
            "co-active ports out of catalog order",
            lead(row(&[
                (FETCH, 1),
                (CODE_VALUE, 11),
                (PUSH, 1),
                (PUSH_ADDRESS, 1),
                (PUSH_VALUE, 9),
                (POP, 1),
                (POP_ADDRESS, 1),
            ])),
            forged_ports,
        ),
        // Slot 2 reads ROM 0 with no active port: an access without an event.
        ("an active slot without an active port", fetch(), |w, memory| {
            let mut ports = forged_ports(w, memory);
            ports[2] = Some(rom_read(0, 11));
            ports
        }),
        // The heap read of 3 shows as a write of 0 at the same cell.
        (
            "a read port whose slot writes",
            lead(row(&[(HEAP_READ, 2), (HEAP_READ_VALUE, 3)])),
            |w, memory| {
                let mut ports = forged_ports(w, memory);
                if w[FETCH] == F::ZERO {
                    let slot = ports[0].as_mut().unwrap();
                    slot.is_write = true;
                    slot.vw = 0;
                }
                ports
            },
        ),
        // The pop of stack 1 (RAM 5) shows as a read of heap 1 (RAM 1): the
        // address bits of the part agree, the bits above the region do not.
        (
            "an address outside the port's region",
            lead(row(&[(FETCH, 1), (CODE_VALUE, 11), (POP, 1), (POP_ADDRESS, 1)])),
            |w, memory| {
                let mut ports = forged_ports(w, memory);
                if w[POP] == F::ONE {
                    ports[1] = Some(PortAccess {
                        is_ram: true,
                        addr: 1,
                        ..rom_read(1, 0)
                    });
                }
                ports
            },
        ),
    ];
    let cells = setup.memory();
    let memory = Memory::new(&setup.geometry, &cells);
    for (name, rows, forge) in cases {
        assert_ne!(
            setup.binding.ports(&rows[0], &memory).ok(),
            Some(forge(&rows[0], &memory)),
            "{name}: the binding gives other slots or no slots"
        );
        setup.reject(name, Some(forge), &rows, 0, &[]);
    }
}

/// Rows that change only columns of an honest step: the slots and every
/// memory word stay the same.
#[test]
fn binding_rows_reject_each_changed_column() {
    let setup = setup();
    let cases: [(&str, usize, &[(usize, u64)]); 7] = [
        // Step 2 has no push, so slot 2 is inactive.
        ("an active port whose slot is inactive", 2, &[(PUSH, 1)]),
        // Slot 0 shows the fetch of ROM 3 = 14; the heap read of heap 3 = 14
        // claims it: same address and value, another memory space.
        (
            "an active port whose slot shows another port",
            0,
            &[(FETCH, 0), (HEAP_READ, 3), (HEAP_READ_VALUE, 14)],
        ),
        // (3, 0) packs to ROM 3, as (1, 1) does, but 3 does not fit 1 bit.
        (
            "an address part at or above 2^width",
            0,
            &[(CODE_LOW, 3), (CODE_HIGH, 0)],
        ),
        // Slot 1 shows the pop; the heap write of the same slot is active too.
        (
            "two co-active ports in one slot",
            1,
            &[(STORE, 1), (HEAP_WRITE, 2), (HEAP_WRITE_VALUE, 8)],
        ),
        (
            "a read whose value differs from its slot's v_r",
            1,
            &[(HEAP_READ_VALUE, 6)],
        ),
        ("a write whose v_r differs from its prior column", 0, &[(HEAP_PRIOR, 4)]),
        ("a write whose value differs from its slot's v_w", 3, &[(PUSH_VALUE, 6)]),
    ];
    for (name, step, change) in cases {
        setup.reject(name, None, &honest_rows(), step, change);
    }
}

/// One ROM with two reads in one slot, `When(0)` and `Unless(0)`. With column
/// 0 at 2, Lean `active` holds for neither port, so the row has no event,
/// but the two activations still sum to 1. Only the bit row of column 0
/// rejects the slot's access.
#[test]
fn binding_rows_reject_a_non_bit_activation() {
    let geometry = Geometry {
        r: 2,
        mu: 2,
        w_ts: 2,
        b_ops: 1,
        b_scan: 4,
        n: 2,
        s_max: 1,
    };
    let catalog = MemoryCatalog::new(
        [MemorySpec {
            id: 0,
            kind: MemoryKind::Rom,
            ports: vec![
                port(&[1], 2, READ, MemoryPortActivation::When(0)),
                port(&[3], 4, READ, MemoryPortActivation::Unless(0)),
            ],
        }],
        &registry(1, 5),
    )
    .unwrap();
    let setup = Setup {
        geometry,
        program: Program::new(geometry, vec![11, 12, 13, 14], vec![0; 4]).unwrap(),
        binding: CatalogBinding::new(&catalog, &geometry, [(0, region(0, &[2]))], &[vec![0, 1]]).unwrap(),
        columns: 5,
    };
    let w: Vec<F> = [2, 1, 12, 1, 12].map(F::from_u64).to_vec();
    assert!(events_for_row(catalog.entries(), &w).is_empty());
    assert_eq!(
        setup
            .binding
            .ports(&w, &Memory::new(&geometry, &setup.memory())),
        Err(CatalogRowError::Activation)
    );
    setup.reject(
        "a non-bit activation",
        Some(|_, _| vec![Some(rom_read(1, 12))]),
        &[w.clone(), w],
        0,
        &[],
    );
}

#[test]
fn bad_configurations_make_no_binding() {
    let catalog = catalog();
    let with = |memory: u32, changed: Region| {
        let mut regions = regions();
        regions.iter_mut().find(|(id, _)| *id == memory).unwrap().1 = changed;
        regions
    };
    let mut plan_dropping = plan();
    plan_dropping[2].clear();
    let mut plan_repeating = plan();
    plan_repeating[2].push(0);
    let cases: [(&str, Vec<(u32, Region)>, Vec<Vec<usize>>, CatalogBindingError); 9] = [
        (
            "overlapping regions",
            with(2, region(2, &[1])),
            plan(),
            CatalogBindingError::Overlap,
        ),
        (
            "a region past its space",
            with(2, region(8, &[1])),
            plan(),
            CatalogBindingError::Region,
        ),
        // The region fits RAM (8 cells) but the code memory is ROM (4 cells).
        (
            "a ROM memory in RAM space",
            with(0, region(4, &[1, 1])),
            plan(),
            CatalogBindingError::Region,
        ),
        (
            "an unaligned base",
            with(1, region(1, &[2])),
            plan(),
            CatalogBindingError::Region,
        ),
        (
            "too few address widths",
            with(0, region(0, &[2])),
            plan(),
            CatalogBindingError::AddressParts,
        ),
        (
            "a plan that drops a port",
            regions(),
            plan_dropping,
            CatalogBindingError::Plan,
        ),
        (
            "a plan that repeats a port",
            regions(),
            plan_repeating,
            CatalogBindingError::Plan,
        ),
        (
            "a plan with too few slots",
            regions(),
            plan()[..2].to_vec(),
            CatalogBindingError::SlotCount,
        ),
        (
            "a region of an unknown memory",
            [regions(), vec![(9, region(6, &[1]))]].concat(),
            plan(),
            CatalogBindingError::UnknownMemory,
        ),
    ];
    for (name, regions, slots, expected) in cases {
        assert_eq!(
            CatalogBinding::new(&catalog, &geometry(), regions, &slots).err(),
            Some(expected),
            "{name}"
        );
    }
}

/// Lean `Binding.Initial`: the images hold each checked memory's preloaded
/// cells at `base + packParts(address)` and zero elsewhere, and the
/// host-checked I/O memory has no cell in them. A cell with a part at or
/// above `2^width`, or with another number of parts, makes no program.
#[test]
fn program_images_hold_the_preloaded_cells_of_checked_memories() {
    let binding = setup().binding;
    let mut preload = MemoryPreload::default();
    for (address, value) in [([0, 0], 11), ([1, 0], 12), ([0, 1], 13), ([1, 1], 14)] {
        preload.insert(0, address.to_vec(), value);
    }
    preload.insert(1, vec![2], 3);
    preload.insert(2, vec![1], 5);
    preload.insert(3, vec![5], 99);
    // Code (x₀, x₁) is ROM x₀ + 2·x₁; heap a is RAM a; stack a is RAM 4 + a.
    let expected = Program::new(geometry(), vec![11, 12, 13, 14], vec![0, 0, 3, 0, 0, 5, 0, 0]).unwrap();
    assert_eq!(binding.program(&preload), Ok(expected));
    for (memory, address) in [(0, vec![2, 0]), (0, vec![1]), (2, vec![2])] {
        let mut outside = preload.clone();
        outside.insert(memory, address.clone(), 1);
        assert_eq!(
            binding.program(&outside),
            Err(CatalogBindingError::Preload),
            "memory {memory} address {address:?}"
        );
    }
}

/// One segment of the test catalog through `Circuit::compile` and the memory
/// verifier. The honest proof verifies. A proof whose last step pushes
/// another value than its slot writes, with every memory word unchanged, is
/// rejected.
#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn catalog_proof_verifies_and_rejects_a_broken_row_on_cpu() {
    use crate::lifecycle::{VerifyError as Stage1Error, FAILED_ASSERTION_ROW, UNCHECKED_WITNESS};
    use crate::nebula::{Circuit, Verifier, VerifyError};
    let started = std::time::Instant::now();
    let setup = setup();
    let engine = crate::Engine::Optimized;
    let reference = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap();
    let circuit = Circuit::compile(&reference, setup.geometry, setup.app(true, None)).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::new(&circuit, engine, 114).unwrap();
    eprintln!("catalog setup elapsed={:?}", started.elapsed());

    let (run, invocations) = setup.run(None, &honest_rows());
    let mut proofs = vec![prover
        .prove(run.initial_state(), &invocations[0].words)
        .unwrap()];
    for invocation in &invocations[1..] {
        let next = prover
            .extend(proofs.last().unwrap(), &invocation.words)
            .unwrap();
        proofs.push(next);
    }
    verifier
        .verify(&setup.program, &run.statement(), run.carry(), &proofs[3])
        .unwrap();
    eprintln!("catalog honest proof elapsed={:?}", started.elapsed());

    let mut words = invocations[3].words.clone();
    *aux_word(&mut words, COLUMNS, PUSH_VALUE) = F::from_u64(6);
    UNCHECKED_WITNESS.set(true);
    FAILED_ASSERTION_ROW.set(None);
    let bad = prover.extend_with_output(&proofs[2], &words, invocations[3].output);
    UNCHECKED_WITNESS.set(false);
    assert!(
        FAILED_ASSERTION_ROW.take().is_some(),
        "the broken step fails an assertion row"
    );
    assert!(matches!(
        verifier.verify(&setup.program, &run.statement(), run.carry(), &bad.unwrap()),
        Err(VerifyError::Proof(crate::Error::Verify(Stage1Error::FreshRelation(_))))
    ));
    eprintln!("catalog elapsed={:?}", started.elapsed());
}
