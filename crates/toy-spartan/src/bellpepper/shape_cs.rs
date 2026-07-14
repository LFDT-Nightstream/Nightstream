//! Fast production `ShapeCS` for R1CS shape synthesis.
//!
//! This builder owns only the data needed to derive the final split R1CS shape:
//! constraint linear combinations and variable counts. Human-readable names and
//! namespace-path uniqueness live in `test_shape_cs.rs`, not here.

use std::{cmp::Ordering, collections::BTreeMap};

use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use core::fmt::Write;
use ff::{Field, PrimeField};

#[derive(Clone, Copy)]
struct OrderedVariable(Variable);

impl Eq for OrderedVariable {}
impl PartialEq for OrderedVariable {
  fn eq(&self, other: &OrderedVariable) -> bool {
    match (self.0.get_unchecked(), other.0.get_unchecked()) {
      (Index::Input(ref a), Index::Input(ref b)) => a == b,
      (Index::Aux(ref a), Index::Aux(ref b)) => a == b,
      _ => false,
    }
  }
}
impl PartialOrd for OrderedVariable {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}
impl Ord for OrderedVariable {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self.0.get_unchecked(), other.0.get_unchecked()) {
      (Index::Input(ref a), Index::Input(ref b)) => a.cmp(b),
      (Index::Aux(ref a), Index::Aux(ref b)) => a.cmp(b),
      (Index::Input(_), Index::Aux(_)) => Ordering::Less,
      (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
    }
  }
}

#[allow(clippy::upper_case_acronyms)]
/// `ShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct ShapeCS<E: Engine>
where
  E::Scalar: PrimeField + Field,
{
  #[allow(clippy::type_complexity)]
  /// All constraints added to the `ShapeCS`.
  pub constraints: Vec<(
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
  )>,
  inputs: usize,
  aux: usize,
  namespace_depth: usize,
}

fn proc_lc<Scalar: PrimeField>(
  terms: &LinearCombination<Scalar>,
) -> BTreeMap<OrderedVariable, Scalar> {
  let mut map = BTreeMap::new();
  for (var, &coeff) in terms.iter() {
    map
      .entry(OrderedVariable(var))
      .or_insert_with(|| Scalar::ZERO)
      .add_assign(&coeff);
  }

  // Remove terms that have a zero coefficient to normalize
  let mut to_remove = vec![];
  for (var, coeff) in map.iter() {
    if coeff.is_zero().into() {
      to_remove.push(*var)
    }
  }

  for var in to_remove {
    map.remove(&var);
  }

  map
}

impl<E: Engine> ShapeCS<E>
where
  E::Scalar: PrimeField,
{
  /// Create a new, default `ShapeCS`,
  pub fn new() -> Self {
    ShapeCS::default()
  }

  /// Returns the number of constraints defined for this `ShapeCS`.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Returns the number of inputs defined for this `ShapeCS`.
  pub fn num_inputs(&self) -> usize {
    self.inputs
  }

  /// Returns the number of aux inputs defined for this `ShapeCS`.
  pub fn num_aux(&self) -> usize {
    self.aux
  }

  /// Print all public inputs, aux inputs, and constraint names.
  #[allow(dead_code)]
  pub fn pretty_print_list(&self) -> Vec<String> {
    let mut result = Vec::new();

    for input in 0..self.inputs {
      result.push(format!("INPUT I{input}"));
    }
    for aux in 0..self.aux {
      result.push(format!("AUX A{aux}"));
    }

    for (idx, _) in self.constraints.iter().enumerate() {
      result.push(format!("C{idx}"));
    }

    result
  }

  /// Print all iputs and a detailed representation of each constraint.
  #[allow(dead_code)]
  pub fn pretty_print(&self) -> String {
    let mut s = String::new();

    for input in 0..self.inputs {
      writeln!(s, "INPUT I{input}").unwrap()
    }

    let negone = -<E::Scalar>::ONE;

    let powers_of_two = (0..E::Scalar::NUM_BITS)
      .map(|i| E::Scalar::from(2u64).pow_vartime([u64::from(i)]))
      .collect::<Vec<_>>();

    let pp = |s: &mut String, lc: &LinearCombination<E::Scalar>| {
      s.push('(');
      let mut is_first = true;
      for (var, coeff) in proc_lc::<E::Scalar>(lc) {
        if coeff == negone {
          s.push_str(" - ")
        } else if !is_first {
          s.push_str(" + ")
        }
        is_first = false;

        if coeff != <E::Scalar>::ONE && coeff != negone {
          for (i, x) in powers_of_two.iter().enumerate() {
            if x == &coeff {
              write!(s, "2^{i} . ").unwrap();
              break;
            }
          }

          write!(s, "{coeff:?} . ").unwrap()
        }

        match var.0.get_unchecked() {
          Index::Input(i) => {
            write!(s, "`I{i}`").unwrap();
          }
          Index::Aux(i) => {
            write!(s, "`A{i}`").unwrap();
          }
        }
      }
      if is_first {
        // Nothing was visited, print 0.
        s.push('0');
      }
      s.push(')');
    };

    for (idx, (a, b, c)) in self.constraints.iter().enumerate() {
      s.push('\n');

      write!(s, "C{idx}: ").unwrap();
      pp(&mut s, a);
      write!(s, " * ").unwrap();
      pp(&mut s, b);
      s.push_str(" = ");
      pp(&mut s, c);
    }

    s.push('\n');

    s
  }
}

impl<E: Engine> Default for ShapeCS<E>
where
  E::Scalar: PrimeField,
{
  fn default() -> Self {
    ShapeCS {
      constraints: vec![],
      inputs: 1,
      aux: 0,
      namespace_depth: 0,
    }
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for ShapeCS<E>
where
  E::Scalar: PrimeField,
{
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let _ = annotation;
    self.aux += 1;
    Ok(Variable::new_unchecked(Index::Aux(self.aux - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let _ = annotation;
    self.inputs += 1;
    Ok(Variable::new_unchecked(Index::Input(self.inputs - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    let _ = annotation;

    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.constraints.push((a, b, c));
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    let _ = name_fn;
    self.namespace_depth += 1;
  }

  fn pop_namespace(&mut self) {
    assert!(self.namespace_depth > 0);
    self.namespace_depth -= 1;
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}
