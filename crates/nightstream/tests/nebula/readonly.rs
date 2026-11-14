use std::{cell::Cell, convert::Infallible};

use super::*;
use crate::nebula::{
    circuit::{memory_circuit, memory_circuit_unshared},
    Application, Circuit, Memory, PortAccess, Program, Run, Transition, View,
};

#[derive(Clone)]
struct Ports {
    reads: usize,
    count: usize,
}

impl Application for Ports {
    type Input = ();
    type State = u64;
    type Error = Infallible;
    fn width(&self) -> usize {
        1
    }
    fn aux(&self) -> usize {
        0
    }
    fn port_count(&self) -> usize {
        self.count
    }
    fn words(&self, state: &u64) -> Vec<u64> {
        vec![*state]
    }
    fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
        let mut rows = Vec::new();
        let mut sum = Expr::constant(0);
        for slot in 0..self.count {
            let write = u64::from(slot >= self.reads);
            rows.extend([
                view.pad(slot),
                view.is_write(slot) - Expr::constant(write),
                view.is_ram(slot) - Expr::constant(1),
                view.addr(slot) - Expr::constant(slot as u64),
                view.vw(slot) - (view.vr(slot) + Expr::constant(write)),
            ]);
            sum = sum + view.vr(slot);
        }
        rows.push(view.output(0) - sum);
        Ok(rows)
    }
    fn step(&self, _: &u64, _: &(), memory: &Memory) -> Result<Transition<u64>, Infallible> {
        let mut ports = Vec::new();
        let mut sum = 0;
        for slot in 0..self.count {
            let value = memory.ram(slot as u64).unwrap();
            let is_write = slot >= self.reads;
            ports.push(Some(PortAccess {
                is_write,
                is_ram: true,
                addr: slot as u64,
                vr: value,
                vw: value + u64::from(is_write),
            }));
            sum += value;
        }
        Ok(Transition {
            next: sum,
            ports,
            aux: Vec::new(),
        })
    }
}

fn geometry(ports: usize) -> Geometry {
    Geometry::new(2, 2, ports, 2, 2).unwrap()
}

fn packages(circuit: &crate::application::ApplicationCircuit) -> nightstream_fprime::BoundedDepthPackages {
    let bytes = crate::nebula::test_packages::bounded_depth_packages("");
    nightstream_fprime::load_bounded_depth_packages(
        &bytes[0],
        &bytes[1],
        nightstream_fprime::BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .unwrap()
    .with_application(
        circuit.records().clone(),
        circuit.private_input_count(),
        circuit.shape(),
    )
    .unwrap()
}

#[test]
fn only_an_authoritative_nonzero_linear_zero_row_marks_a_read() {
    let geometry = geometry(2);
    let layout = Layout::new(&geometry, 1, 0);
    let builder = crate::application::ApplicationBuilder::new(layout.count()).unwrap();
    let wires = builder.private_inputs();
    let flag = Expr::Var(wires[layout.op(0, 1)]);
    let other = Expr::Var(wires[layout.app_in(0)]);
    for row in [
        flag.clone(),
        Expr::constant(7) * flag.clone(),
        flag.clone() + other.clone() - other.clone(),
    ] {
        let map = ReadOnlyValues::checked(&[row], &geometry, &layout, wires);
        assert_eq!(map.aliases.len(), 32);
        let mapped = map.wires(wires);
        for bit in 0..32 {
            assert_eq!(
                mapped[layout.op(0, layout.op_vw(bit))],
                wires[layout.op(0, layout.op_vr(bit))]
            );
            assert_eq!(
                mapped[layout.op(1, layout.op_vw(bit))],
                wires[layout.op(1, layout.op_vw(bit))]
            );
        }
    }
    for row in [
        flag.clone() - Expr::constant(1),
        flag.clone() * other.clone(),
        flag.clone() * Expr::constant(0),
        flag.clone() + other,
    ] {
        assert!(ReadOnlyValues::checked(&[row], &geometry, &layout, wires).is_empty());
    }
}

#[test]
fn shared_values_reconstruct_every_original_memory_and_hash_row() {
    let mut rejected = 0;
    let mut discarded = 0;
    for reads in 0..=3 {
        let app = Ports { reads, count: 3 };
        let geometry = geometry(3);
        let layout = Layout::new(&geometry, 1, 0);
        let old = memory_circuit_unshared(&geometry, &app).unwrap();
        let new = memory_circuit(&geometry, &app).unwrap();
        assert_eq!(old.row_count() - new.row_count(), reads * 33);
        assert_eq!(old.private_input_count(), new.private_input_count());
        assert_eq!(old.generated_range().len(), new.generated_range().len());
        assert_eq!(old.shape().blocks.len(), new.shape().blocks.len());
        let image = Program::new(geometry, vec![0; 4], vec![0, 0x1234_5678, 0xffff_ff00, 0]).unwrap();
        let mut run = Run::new(app, &image, 0);
        let mut prior = run.initial_state();
        for invocation in run.segment(&[(), ()]).unwrap() {
            old.execute(prior, &invocation.words).unwrap();
            assert_eq!(
                new.execute(prior, &invocation.words)
                    .unwrap()
                    .output_state(),
                invocation.output
            );
            let mut erased = invocation.words.clone();
            for slot in 0..reads {
                for bit in 0..32 {
                    erased[layout.op(slot, layout.op_vw(bit))] = F::from_u64(99);
                    discarded += 1;
                }
            }
            let accepted = new.execute(prior, &erased).unwrap();
            assert_eq!(accepted.output_state(), invocation.output);
            if reads > 0 {
                assert!(old.execute(prior, &erased).is_err());
            }
            for slot in 0..reads {
                for bit in 0..32 {
                    erased[layout.op(slot, layout.op_vw(bit))] = erased[layout.op(slot, layout.op_vr(bit))];
                }
            }
            assert_eq!(erased, invocation.words);
            old.execute(prior, &erased).unwrap();
            for slot in 0..3 {
                for bit in 0..32 {
                    let mut wrong = invocation.words.clone();
                    wrong[layout.op(slot, layout.op_vr(bit))] = F::from_u64(2);
                    assert!(new.execute(prior, &wrong).is_err());
                    rejected += 1;
                }
                if slot >= reads {
                    let mut wrong = invocation.words.clone();
                    wrong[layout.op(slot, layout.op_vw(0))] = F::from_u64(2);
                    assert!(new.execute(prior, &wrong).is_err());
                    rejected += 1;
                }
                let mut wrong = invocation.words.clone();
                wrong[layout.op(slot, 1)] = F::ONE - wrong[layout.op(slot, 1)];
                assert!(new.execute(prior, &wrong).is_err());
                rejected += 1;
            }
            prior = invocation.output;
        }
    }
    eprintln!("readonly_adverse rejected={rejected} discarded_words_reconstructed={discarded}");
}

/// Each proved read port removes 33 source rows, 32 bit slots and their 32
/// companions (`SPEC.md` sections 7.3 and 7.9). The plan rows fall only by the
/// packed Boolean rows, 31 sources or products each (section 7.9). The port's
/// other two rows read `vw - vr`: without sharing, section 7.13 packs them in
/// one row (`Export.Generic.WordPacking.group_iff`); with sharing, the
/// operation-slot row is gone and the application row is identically zero.
#[test]
fn compiled_slot_cost_is_an_integer_per_proved_read_port() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("nightstream-readonly-packages-{stamp}"));
    std::fs::create_dir(&directory).unwrap();
    for reads in 0..=3 {
        let app = Ports { reads, count: 3 };
        let geometry = geometry(3);
        let old = memory_circuit_unshared(&geometry, &app).unwrap();
        let new = memory_circuit(&geometry, &app).unwrap();
        crate::application::check_blocks(&new).unwrap();
        let old_packages = packages(&old);
        let new_packages = packages(&new);
        if reads == 0 {
            assert_eq!(old_packages.identity(), new_packages.identity());
        } else {
            assert_ne!(old_packages.identity(), new_packages.identity());
        }
        for (name, old, new) in [
            ("light", old_packages.light(), new_packages.light()),
            ("refresh", old_packages.refresh(), new_packages.refresh()),
        ] {
            let (old_expanded, new_expanded) = (old.clone().expand().unwrap(), new.clone().expand().unwrap());
            let (before, after) = (old_expanded.rows(), new_expanded.rows());
            let before_source = old.source_rows().unwrap();
            let after_source = new.source_rows().unwrap();
            assert_eq!(before_source.row_count() - after_source.row_count(), 33 * reads);
            let pairs = |expanded: &nightstream_fprime::ExpandedPackage| {
                let package = expanded.package();
                expanded.layout().spans().widths.len() - (package.sbox_start() + 150 * package.block_count())
            };
            let (before_pairs, after_pairs) = (pairs(&old_expanded), pairs(&new_expanded));
            assert_eq!(before_pairs - after_pairs, 32 * reads);
            assert_eq!(
                before.row_count() - after.row_count(),
                old.packed_boolean_count().div_ceil(31) - new.packed_boolean_count().div_ceil(31)
            );
            assert_eq!(
                old_expanded.layout().width() - new_expanded.layout().width(),
                64 * reads
            );
            eprintln!("readonly_cost reads={reads} package={name} before_rows={} after_rows={} before_entries={} after_entries={} before_coordinates={} after_coordinates={}",before.row_count(),after.row_count(),before.entry_count(),after.entry_count(),old_expanded.layout().width(),new_expanded.layout().width());
        }
        for (name, application) in [("before", old), ("after", new)] {
            crate::Circuit::compile(application)
                .unwrap()
                .write(directory.join(format!("{reads}-{name}.nsopt")))
                .unwrap();
        }
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn the_original_application_rows_are_observed_once() {
    struct Once(Cell<usize>);
    impl Application for Once {
        type Input = ();
        type State = ();
        type Error = Infallible;
        fn width(&self) -> usize {
            0
        }
        fn aux(&self) -> usize {
            0
        }
        fn port_count(&self) -> usize {
            1
        }
        fn words(&self, _: &()) -> Vec<u64> {
            Vec::new()
        }
        fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
            self.0.set(self.0.get() + 1);
            assert_eq!(self.0.get(), 1);
            Ok(vec![view.is_write(0)])
        }
        fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Infallible> {
            unreachable!()
        }
    }
    let app = Once(Cell::new(0));
    memory_circuit(&geometry(1), &app).unwrap();
    assert_eq!(app.0.get(), 1);
}

#[test]
#[ignore = "Runs actual public proofs. Invoke alone under the300-second cap."]
fn readonly_public_proof_binds_the_authorized_memory() {
    let geometry = geometry(2);
    let app = Ports { reads: 2, count: 2 };
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(crate::Engine::Optimized, 109).unwrap();
    let verifier = circuit.verifier(crate::Engine::Optimized, 109).unwrap();
    let images = [vec![17, 29, 0, 0], vec![29, 17, 0, 0]].map(|ram| Program::new(geometry, vec![0; 4], ram).unwrap());
    for index in 0..2 {
        let mut proving = prover.start(&images[index], 0).unwrap();
        proving.segment(&[(), ()]).unwrap();
        assert_eq!(*proving.state(), 46);
        let proof = proving.finish().unwrap();
        let encoded = prover.encode_proof(&proof).unwrap();
        let proof = verifier.decode_proof(&encoded).unwrap();
        verifier.verify(&images[index], &0, &46, &proof).unwrap();
        assert!(verifier
            .verify(&images[1 - index], &0, &46, &proof)
            .is_err());
        assert!(verifier.verify(&images[index], &0, &47, &proof).is_err());
        eprintln!("readonly_public case={index} encoded_bytes={}", encoded.len());
    }
}
