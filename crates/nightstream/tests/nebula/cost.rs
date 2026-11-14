//! The per-step cost of the Rust-built memory relation for the free
//! application (no state word, no private word, no row) with `B_ops = 21`, a
//! ROM of `2^12` cells, and a RAM of `2^16` cells, at each feasible `N` with
//! `S_max = 1` and the smallest `W_ts` that SPEC.md section 7.1, rule 2 allows; and for
//! the geometry, state words, and private words of the WASM test of
//! `neo-wasm` (`tests/wasm_nebula.rs`), without its rows. A step's
//! Stage 1 words are its witness words (the memory words of
//! `WitnessWords.lean`) and its local words (the variables of the Poseidon2
//! sponges and of the product rows). The refresh relation with the application
//! in place, whose logical width is the widest step's, must fit the approved
//! key capacity (`MAX_CARRIER_WIDTH`).

use std::sync::OnceLock;

use neo_ajtai::nightstream_fprime_setup::MAX_CARRIER_WIDTH;
use nightstream_fprime::{load_bounded_depth_packages, BoundedDepthPackages, BOUNDED_DEPTH_PACKAGES_IDENTITY};

use super::memory_circuit;
use crate::application::ApplicationCircuit;
use crate::nebula::{Application, Expr, Geometry, Memory, Transition, View};

/// The logical width of the refresh relation with `circuit` in place of Lean's application.
fn logical_width(circuit: &ApplicationCircuit) -> Result<usize, crate::Error> {
    static PACKAGES: OnceLock<BoundedDepthPackages> = OnceLock::new();
    let packages = PACKAGES.get_or_init(|| {
        let bytes = crate::nebula::test_packages::bounded_depth_packages("");
        load_bounded_depth_packages(&bytes[0], &bytes[1], BOUNDED_DEPTH_PACKAGES_IDENTITY).unwrap()
    });
    let packages = packages.with_application(
        circuit.records().clone(),
        circuit.private_input_count(),
        circuit.shape(),
    )?;
    Ok(packages.refresh().clone().expand()?.layout().width())
}

/// An application with `width` state words, `aux` private words, `ports`
/// ports, and no row.
#[derive(Clone, Copy, Debug)]
struct Free {
    width: usize,
    aux: usize,
    ports: usize,
}

impl Application for Free {
    type Input = ();
    type State = ();
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        self.width
    }

    fn aux(&self) -> usize {
        self.aux
    }

    fn port_count(&self) -> usize {
        self.ports
    }

    fn words(&self, _: &()) -> Vec<u64> {
        vec![0; self.width]
    }

    fn rows(&self, _: &View) -> Result<Vec<Expr>, crate::Error> {
        Ok(Vec::new())
    }

    fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
        unreachable!("the cost measurement runs no step")
    }
}

/// Build the free relation of `N = n` and `S_max = s_max` with the smallest
/// `W_ts`, print its counts, and return whether it fits; `None` for a
/// geometry that breaks SPEC.md section 7.1.
fn measure(n: usize, s_max: usize) -> Option<bool> {
    let (r, mu, b_ops) = (12, 16, 21);
    let cells = (1usize << r) + (1 << mu);
    let geometry = Geometry {
        r,
        mu,
        w_ts: (usize::BITS - (s_max * n * b_ops).leading_zeros()) as usize,
        b_ops,
        b_scan: cells / n,
        n,
        s_max,
    };
    count(
        &geometry,
        Free {
            width: 0,
            aux: 0,
            ports: geometry.b_ops,
        },
    )
}

/// Print the counts of the relation of `geometry` and `app`, and return
/// whether it fits; `None` for a geometry that breaks SPEC.md section 7.1.
fn count(geometry: &Geometry, app: Free) -> Option<bool> {
    if !geometry.is_valid() || !geometry.is_secure() {
        return None;
    }
    let n = geometry.n;
    let s_max = geometry.s_max;
    let started = std::time::Instant::now();
    let circuit = memory_circuit(geometry, &app).unwrap();
    let (witness, local) = (circuit.private_input_count(), circuit.generated_range().len());
    let carrier = logical_width(&circuit);
    let fits = carrier
        .as_ref()
        .is_ok_and(|width| *width <= MAX_CARRIER_WIDTH);
    eprintln!(
        "{n} {s_max} {} {} {witness} {local} {} {} {} {fits} {:.1}",
        geometry.b_scan,
        geometry.w_ts,
        witness + local,
        circuit.row_count(),
        carrier.map_or_else(|error| error.to_string(), |width| width.to_string()),
        started.elapsed().as_secs_f64()
    );
    Some(fits)
}

#[test]
#[ignore = "Builds memory relations of up to millions of words; run separately under the 300-second cap."]
fn free_application_words_per_step() {
    let cells = (1usize << 12) + (1 << 16);
    eprintln!("N S_max B_scan W_ts witness local words rows carrier fits seconds");
    let mut smallest = None;
    for n in (1..=cells).rev() {
        if measure(n, 1) == Some(true) {
            smallest = Some(n);
        }
    }
    let smallest = smallest.expect("some feasible N fits");
    eprintln!("budget MAX_CARRIER_WIDTH={MAX_CARRIER_WIDTH} smallest fitting N={smallest}");
    // One more timestamp bit at the smallest N: S_max = 2.
    measure(smallest, 2);
}

/// The WASM test geometry with its 61 state words and 8,549 private words.
#[test]
#[ignore = "Prints the per-step words of the WASM test geometry."]
fn wasm_geometry_words_per_step() {
    let geometry = Geometry {
        r: 8,
        mu: 8,
        w_ts: 8,
        b_ops: 21,
        b_scan: 64,
        n: 8,
        s_max: 1,
    };
    eprintln!("N S_max B_scan W_ts witness local words rows carrier fits seconds");
    assert_eq!(
        count(
            &geometry,
            Free {
                width: 61,
                aux: 8_549,
                ports: geometry.b_ops,
            }
        ),
        Some(true)
    );
    eprintln!("budget MAX_CARRIER_WIDTH={MAX_CARRIER_WIDTH}");
}

/// The parts of `size_parts` sum to the counts of the memory circuit, and
/// each phase and region has its own name.
#[test]
fn size_parts_sum_to_the_circuit_counts() {
    let geometry = Geometry {
        r: 2,
        mu: 3,
        w_ts: 4,
        b_ops: 3,
        b_scan: 3,
        n: 4,
        s_max: 1,
    };
    let app = Free {
        width: 2,
        aux: 3,
        ports: geometry.b_ops,
    };
    let parts = crate::nebula::size_parts(&geometry, &app).unwrap();
    let circuit = memory_circuit(&geometry, &app).unwrap();
    let total = parts
        .iter()
        .fold((0, 0, 0), |(witness, local, rows), part| {
            (
                witness + part.size.witness_words,
                local + part.size.local_words,
                rows + part.size.rows,
            )
        });
    assert_eq!(
        total,
        (
            circuit.private_input_count(),
            circuit.generated_range().len(),
            circuit.row_count()
        )
    );
    let mut names: Vec<&str> = parts.iter().map(|part| part.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), parts.len());
}

/// The committed size of the step packages with `circuit` in place of the
/// fixture's application: per package, the logical width, the slot
/// coordinates of each kind, the plan rows and the blocks.
fn print_package_size(name: &str, packages: &BoundedDepthPackages) {
    for (kind, package) in [("light", packages.light()), ("refresh", packages.refresh())] {
        let expanded = package.clone().expand().unwrap();
        let (layout, rows) = (expanded.layout(), expanded.rows());
        eprintln!(
            "{name} {kind} logical={} plan_rows={} blocks={}",
            layout.width(),
            rows.row_count(),
            package.block_count()
        );
    }
}

/// The step cost of the first memory package's geometry and machine through
/// `Circuit::compile`: the committed size of both step packages, next to
/// Lean's own `nebula-v1` packages, and the CPU time of three honest steps.
#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn native_memory_step_cost() {
    use crate::nebula::{Circuit, Machine, MachineState, Package, Run, Step};
    let geometry = crate::nebula::Package::V1.geometry();
    let circuit = memory_circuit(&geometry, &Machine).unwrap();
    eprintln!(
        "native application witness={} local={} rows={}",
        circuit.private_input_count(),
        circuit.generated_range().len(),
        circuit.row_count()
    );
    let bytes = crate::nebula::test_packages::bounded_depth_packages("");
    let fixture = load_bounded_depth_packages(&bytes[0], &bytes[1], BOUNDED_DEPTH_PACKAGES_IDENTITY).unwrap();
    print_package_size(
        "native",
        &fixture
            .with_application(
                circuit.records().clone(),
                circuit.private_input_count(),
                circuit.shape(),
            )
            .unwrap(),
    );
    let lean = crate::nebula::test_packages::bounded_depth_packages("nebula-v1-");
    let computed = match load_bounded_depth_packages(&lean[0], &lean[1], [0; 4]) {
        Err(nightstream_fprime::PackageError::IdentityMismatch { computed, .. }) => computed,
        other => panic!("{:?}", other.map(|packages| packages.identity())),
    };
    print_package_size(
        "lean",
        &load_bounded_depth_packages(&lean[0], &lean[1], computed).unwrap(),
    );

    let started = std::time::Instant::now();
    let circuit = Circuit::compile(geometry, Machine).unwrap();
    let prover = circuit
        .circuit
        .prover(crate::Engine::Optimized, 109)
        .unwrap();
    eprintln!("setup elapsed={:?}", started.elapsed());
    let mut run = Run::new(Machine, &Package::V1.program(), MachineState::default());
    let z0 = run.initial_state();
    let mut invocations = run.segment(&[Step::Execute, Step::Execute]).unwrap();
    invocations.extend(run.segment(&[Step::Idle, Step::Idle]).unwrap());
    let mut proof: Option<crate::Proof> = None;
    for (step, invocation) in invocations.iter().take(3).enumerate() {
        let proving = std::time::Instant::now();
        proof = Some(
            match &proof {
                None => prover.prove_with_output(z0, &invocation.words, invocation.output),
                Some(proof) => prover.extend_with_output(proof, &invocation.words, invocation.output),
            }
            .unwrap(),
        );
        eprintln!("native step={step} prove={:?}", proving.elapsed());
    }
}

/// The refresh package's logical width for an `R1csApplication` of two rows
/// a step whose one-row relation is a counter that reads a ROM, with `bits`
/// more columns, each made 0 or 1 by a row `x · (x − one)`.
fn r1cs_width(bits: usize) -> usize {
    use crate::nebula::{batched_slots, pack_regions, R1csApplication, RowRelation, Size};
    use crate::relation::{
        ColumnFamilySpec, ColumnRegistry, ColumnWidth, ConstraintTag, ContinuityCatalog, ContinuityGroup,
        ContinuityLink, MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
        R1csBuilder,
    };
    use crate::{Goldilocks as F, PrimeCharacteristicRing};
    const ONE: usize = 0;
    const BEFORE: usize = 1;
    const AFTER: usize = 2;
    const VALUE: usize = 3;
    let columns = 4 + bits;
    let registry = ColumnRegistry::new([ColumnFamilySpec {
        region: "counter",
        start: 0,
        len: columns,
        name: "words".into(),
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap();
    let mut builder = R1csBuilder::new(1, ONE).unwrap();
    builder.with_tag(ConstraintTag::new("count", "counter"), |rows| {
        rows.push_linear_zero([(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)]);
        for bit in 4..columns {
            rows.push_boolean(bit);
        }
    });
    let relation = builder.build(&registry).unwrap();
    let catalog = MemoryCatalog::new(
        [MemorySpec {
            id: 0u32,
            kind: MemoryKind::Rom,
            ports: vec![MemoryPortSpec {
                address_columns: vec![BEFORE],
                value_column: VALUE,
                kind: MemoryPortKind::Read,
                activation: MemoryPortActivation::Always,
            }],
        }],
        &registry,
    )
    .unwrap();
    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "count",
            role: "test",
            links: vec![ContinuityLink {
                previous_step_column: AFTER,
                next_step_column: BEFORE,
            }],
        }],
        &registry,
    )
    .unwrap();
    let rows = std::num::NonZeroUsize::new(2).unwrap();
    let geometry = Geometry::new(2, 2, 2, 2, 1).unwrap();
    let (regions, _) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    let size = Size {
        geometry,
        rows,
        regions,
        slots: batched_slots(&catalog, &[vec![0]], rows).unwrap(),
    };
    let app = R1csApplication::new(RowRelation::of(&relation).unwrap(), &continuity, &catalog, size).unwrap();
    logical_width(&memory_circuit(&geometry, &app).unwrap()).unwrap()
}

#[test]
fn each_forced_bit_of_an_r1cs_application_has_one_companion() {
    let (base, bits) = (r1cs_width(0), r1cs_width(8));
    assert_eq!(
        bits - base,
        2 * 8 * 2,
        "8 bit pairs in each of 2 rows: {base} -> {bits}"
    );
}
