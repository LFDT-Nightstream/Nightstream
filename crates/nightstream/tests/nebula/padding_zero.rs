use super::{Geometry, Layout, Sym, F};
use crate::application::{Affine, ApplicationBuilder, ApplicationCircuit};
use crate::nebula::expr::lower_constraint;
use p3_field::PrimeCharacteristicRing;

fn slot_circuit(check_bits: bool) -> (ApplicationCircuit, Layout) {
    let geometry = Geometry::first();
    let layout = Layout::new(&geometry, 0, 0);
    let mut builder = ApplicationBuilder::new(layout.count()).unwrap();
    let wires = builder.private_inputs().to_vec();
    let symbols = Sym::new(&geometry, layout, &wires, [F::ZERO; 12]);
    if check_bits {
        for row in symbols.bit_rows() {
            lower_constraint(&mut builder, &row).unwrap();
        }
    }
    for row in symbols.slot_rows(0) {
        lower_constraint(&mut builder, &row).unwrap();
    }
    let circuit = builder
        .finish(std::array::from_fn(|_| Affine::constant(F::ZERO)))
        .unwrap();
    (circuit, layout)
}

fn padded_words(layout: Layout) -> Vec<F> {
    let mut words = vec![F::ZERO; layout.count()];
    words[layout.op(0, 0)] = F::ONE;
    for pair in 0..2 {
        words[layout.ops_factor(0, pair, 2)] = F::ONE;
    }
    words
}

#[test]
fn padding_sum_rejects_nonzero_boolean_payloads() {
    let (circuit, layout) = slot_circuit(true);
    let honest = padded_words(layout);
    circuit.execute([F::ZERO; 4], &honest).unwrap();

    let geometry = Geometry::first();
    for write in [F::ZERO, F::ONE] {
        let mut changed = honest.clone();
        changed[layout.op(0, 1)] = write;
        changed[layout.op(0, 2)] = F::ONE;
        for pair in 0..2 {
            changed[layout.ops_factor(0, pair, 0)] = F::from_u64(geometry.rom_size() as u64);
        }
        assert!(circuit.execute([F::ZERO; 4], &changed).is_err());
    }
    for bit in 0..32 {
        let mut changed = honest.clone();
        changed[layout.op(0, layout.op_vr(bit))] = F::ONE;
        changed[layout.op(0, layout.op_vw(bit))] = F::ONE;
        assert!(circuit.execute([F::ZERO; 4], &changed).is_err());
    }
    for bit in 0..geometry.w_ts {
        let mut changed = honest.clone();
        changed[layout.op(0, layout.op_rt(bit))] = F::ONE;
        assert!(circuit.execute([F::ZERO; 4], &changed).is_err());
    }
    for bit in 0..geometry.addr_width() {
        let mut changed = honest.clone();
        changed[layout.op(0, layout.op_addr(bit))] = F::ONE;
        for pair in 0..2 {
            changed[layout.ops_factor(0, pair, 0)] = F::from_u64(1 << bit);
        }
        assert!(circuit.execute([F::ZERO; 4], &changed).is_err());
    }
}

#[test]
fn padding_sum_keeps_active_payloads_and_inactive_difference_bits() {
    let (circuit, layout) = slot_circuit(true);
    let mut active = vec![F::ZERO; layout.count()];
    active[layout.cnt(0)] = F::ONE;
    for bit in 0..32 {
        active[layout.op(0, layout.op_vr(bit))] = F::ONE;
        active[layout.op(0, layout.op_vw(bit))] = F::ONE;
    }
    circuit.execute([F::ZERO; 4], &active).unwrap();

    let mut inactive = padded_words(layout);
    for bit in 0..Geometry::first().w_ts {
        inactive[layout.op(0, layout.op_diff(bit))] = F::ONE;
    }
    circuit.execute([F::ZERO; 4], &inactive).unwrap();
}

#[test]
fn retained_bit_rows_reject_non_boolean_cancellation() {
    let (checked, layout) = slot_circuit(true);
    let (without_bits, _) = slot_circuit(false);
    let mut changed = padded_words(layout);
    for offset in [layout.op_vr(0), layout.op_vw(0)] {
        changed[layout.op(0, offset)] = F::ONE;
        changed[layout.op(0, offset + 1)] = F::NEG_ONE;
    }
    assert!(checked.execute([F::ZERO; 4], &changed).is_err());
    without_bits.execute([F::ZERO; 4], &changed).unwrap();
}
