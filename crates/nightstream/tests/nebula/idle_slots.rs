//! Row O7 (SPEC.md section 7.3) for an application whose rows ignore idle
//! slots: an idle slot may hold any fields, and its gate still keeps its
//! record out of both products.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{Layout, Sym};
use crate::application::Variable;
use crate::nebula::{Expr, Geometry};

fn eval(expr: &Expr, words: &[F]) -> F {
    match expr {
        Expr::Var(variable) => words[variable.index()],
        Expr::Const(value) => *value,
        Expr::Add(left, right) => eval(left, words) + eval(right, words),
        Expr::Mul(left, right) => eval(left, words) * eval(right, words),
    }
}

fn set_bits(words: &mut [F], position: impl Fn(usize) -> usize, count: usize, value: u64) {
    for k in 0..count {
        words[position(k)] = F::from_u64((value >> k) & 1);
    }
}

/// Slot 0 of a step with `η = 0`, start products `1`, and an idle slot whose
/// fields are a write record `(rt 3, addr 5, v_r 7, v_w 9)` in RAM. With
/// `gate`, its gate factor is the record's fingerprint in place of `1`.
fn idle_slot_words(geometry: &Geometry, layout: &Layout, gate: bool) -> Vec<F> {
    let mut words = vec![F::ZERO; layout.count()];
    words[layout.op(0, 0)] = F::ONE;
    words[layout.op(0, 1)] = F::ONE;
    words[layout.op(0, 2)] = F::ONE;
    set_bits(
        &mut words,
        |k| layout.op(0, layout.op_addr(k)),
        geometry.addr_width(),
        5,
    );
    set_bits(&mut words, |k| layout.op(0, layout.op_vr(k)), 32, 7);
    set_bits(&mut words, |k| layout.op(0, layout.op_vw(k)), 32, 9);
    set_bits(&mut words, |k| layout.op(0, layout.op_rt(k)), geometry.w_ts, 3);
    let g = F::from_u64(5 + geometry.rom_size() as u64);
    for s in 0..2 {
        words[layout.start(s, 0)] = F::ONE;
    }
    for pair in 0..2 {
        let factor = |k| layout.ops_factor(0, pair, k);
        words[factor(0)] = g;
        let x0 = if gate { g } else { F::ONE };
        words[factor(2)] = x0;
        words[layout.ops_products(0, 2 * pair % 4)] = x0;
    }
    words
}

fn failing_slot_rows(zero_idle: bool, gate: bool) -> usize {
    let geometry = Geometry::new(4, 4, 2, 2, 1).expect("a valid geometry");
    let layout = Layout::new(&geometry, 0, 0);
    let wires: Vec<Variable> = (0..layout.count()).map(Variable).collect();
    let sym = Sym::new(&geometry, layout, &wires, [F::ZERO; 12], zero_idle);
    let words = idle_slot_words(&geometry, &layout, gate);
    sym.slot_rows(0)
        .iter()
        .filter(|row| eval(row, &words) != F::ZERO)
        .count()
}

#[test]
fn an_idle_slot_with_any_fields_holds_without_row_o7() {
    assert_eq!(failing_slot_rows(false, false), 0, "no row reads the idle fields");
    assert!(failing_slot_rows(true, false) > 0, "row O7 rejects the same fields");
}

#[test]
fn an_idle_slot_cannot_add_its_record_to_a_product() {
    assert!(
        failing_slot_rows(false, true) > 0,
        "the gate row rejects a factor other than 1 for an idle slot"
    );
}
