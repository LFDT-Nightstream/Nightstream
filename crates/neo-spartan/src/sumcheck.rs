//! Sum-check rounds shared by the GKR layers and the linear claim.
//!
//! A round message is `h(0), h(2), ..., h(degree)`; the verifier derives
//! `h(1) = claim - h(0)`. Variables bind low index bit first.

use p3_challenger_v08::FieldChallenger;
use p3_field_v08::{Field, PrimeCharacteristicRing};

use crate::field::{eq_table, fold_low, Ext, Gl};
use crate::hash::Challenger;
use crate::Error;

/// Bind one round message and draw its challenge.
pub(crate) fn send(message: &[Ext], challenger: &mut Challenger) -> Ext {
    challenger.observe_algebra_slice(message);
    challenger.sample_algebra_element()
}

/// Replay `rounds` against `claim`; return the point and the final claim.
pub(crate) fn verify(
    rounds: &[Vec<Ext>],
    degree: usize,
    mut claim: Ext,
    challenger: &mut Challenger,
) -> Result<(Vec<Ext>, Ext), Error> {
    let mut point = Vec::with_capacity(rounds.len());
    for message in rounds {
        if message.len() != degree {
            return Err(Error::Rejected("sum-check round length"));
        }
        let mut values = Vec::with_capacity(degree + 1);
        values.push(message[0]);
        values.push(claim - message[0]);
        values.extend_from_slice(&message[1..]);
        let r = send(message, challenger);
        claim = interpolate(&values, r);
        point.push(r);
    }
    Ok((point, claim))
}

/// The polynomial through `(i, values[i])` for `i = 0..values.len()`, at `x`.
pub(crate) fn interpolate(values: &[Ext], x: Ext) -> Ext {
    let nodes: Vec<Ext> = (0..values.len())
        .map(|i| Ext::from(Gl::from_usize(i)))
        .collect();
    let mut total = Ext::ZERO;
    for (i, &value) in values.iter().enumerate() {
        let mut numerator = Ext::ONE;
        let mut denominator = Gl::ONE;
        for (j, &node) in nodes.iter().enumerate() {
            if i != j {
                numerator *= x - node;
                denominator *= Gl::from_usize(i) - Gl::from_usize(j);
            }
        }
        total += value * numerator * denominator.inverse();
    }
    total
}

/// Prove `Σ_x a(x)·b(x) = claim` over `log2(a.len())` variables.
/// Returns the round messages, the point, and `(a(point), b(point))`.
pub(crate) fn prove_product(
    mut a: Vec<Ext>,
    mut b: Vec<Ext>,
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>, [Ext; 2]) {
    assert_eq!(a.len(), b.len());
    let mut rounds = Vec::new();
    let mut point = Vec::new();
    while a.len() > 1 {
        let (mut at_zero, mut at_two) = (Ext::ZERO, Ext::ZERO);
        for k in 0..a.len() / 2 {
            let (a0, a1, b0, b1) = (a[2 * k], a[2 * k + 1], b[2 * k], b[2 * k + 1]);
            at_zero += a0 * b0;
            at_two += (a1.double() - a0) * (b1.double() - b0);
        }
        let message = vec![at_zero, at_two];
        let r = send(&message, challenger);
        fold_low(&mut a, r);
        fold_low(&mut b, r);
        rounds.push(message);
        point.push(r);
    }
    (rounds, point, [a[0], b[0]])
}

/// `Σ_x eq(point, x)·table(x)`: the multilinear extension at `point`.
pub(crate) fn evaluate(table: &[Ext], point: &[Ext]) -> Ext {
    assert_eq!(table.len(), 1 << point.len());
    eq_table(point)
        .iter()
        .zip(table)
        .map(|(&w, &v)| w * v)
        .sum()
}
