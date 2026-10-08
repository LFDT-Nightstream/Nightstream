//! The fixed words each Plonky3 0.8 transcript phase of a prescribed-point
//! WHIR opening absorbs first: its domain separator.
//!
//! Plonky3 keeps three of the separator shapes crate-private (layout opening,
//! out-of-domain claim, batching); they are rebuilt here from the same public
//! parts. `DomainSeparator::seed` does the encoding, including the approved
//! Keccak-256 fingerprint of the fixed pattern string (M2-DESIGN.md, Result
//! item 7). No prover data passes through it. Parity tests against Plonky3's
//! own verifier pin every word.

use p3_challenger_v08::fs::{DomainSeparator, FieldUnit, Hierarchy, Interaction, InteractionPattern, Kind, Length};
use p3_challenger_v08::CanObserve;
use p3_sumcheck_v08::layout::commitment_domain_separator;
use p3_sumcheck_v08::strategy::Basis;
use p3_sumcheck_v08::transcript::SumcheckShape;
use p3_sumcheck_v08::TableShape;
use p3_whir::transcript::WhirShape;

use crate::field::{Ext, Gl};
use crate::pcs::Config;

const LAYOUT_VERSION: u8 = 1;
const OPENING_NAME: &[u8] = b"p3-sumcheck-layout-opening";
const VIRTUAL_NAME: &[u8] = b"p3-sumcheck-layout-ood";
const BATCHING_NAME: &[u8] = b"p3-sumcheck-layout-batching";

/// Collects the words a separator absorbs.
struct Words(Vec<Gl>);

impl CanObserve<Gl> for Words {
    fn observe(&mut self, value: Gl) {
        self.0.push(value);
    }
}

fn words(separator: &DomainSeparator<FieldUnit<Gl>>) -> Vec<Gl> {
    let mut words = Words(Vec::new());
    separator.seed(&mut words);
    words.0
}

fn scalar(kind: Kind, label: &'static str) -> Interaction {
    Interaction::algebra::<Gl, Ext>(Hierarchy::Atomic, kind, label, Length::Scalar)
}

fn instance(separator: &mut DomainSeparator<FieldUnit<Gl>>, value: usize) {
    separator.instance(&(value as u64).to_be_bytes());
}

/// The stacked layout as the separators bind it: `SuffixProver`'s strategy
/// (selectors not reversed, suffix variable order) and the padded tables.
fn bind(separator: &mut DomainSeparator<FieldUnit<Gl>>, num_variables: usize, tables: &[TableShape]) {
    instance(separator, num_variables);
    separator.instance(&[0]);
    separator.instance(&[1]);
    instance(separator, tables.len());
    for table in tables {
        instance(separator, table.num_variables());
        instance(separator, table.width());
    }
}

/// A claim on `columns` of table `table` at a point the caller fixes.
pub(crate) fn opening(num_variables: usize, tables: &[TableShape], table: usize, columns: &[usize]) -> Vec<Gl> {
    let pattern = InteractionPattern::new(vec![Interaction::algebra::<Gl, Ext>(
        Hierarchy::Atomic,
        Kind::Message,
        "current_evals",
        Length::Fixed(columns.len()),
    )])
    .expect("one leaf step");
    let mut separator = DomainSeparator::new(LAYOUT_VERSION, OPENING_NAME, pattern);
    bind(&mut separator, num_variables, tables);
    instance(&mut separator, table);
    instance(&mut separator, tables[table].num_variables());
    instance(&mut separator, columns.len());
    for &column in columns {
        instance(&mut separator, column);
    }
    instance(&mut separator, 0);
    words(&separator)
}

/// One out-of-domain claim on the stacked polynomial.
pub(crate) fn virtual_claim(num_variables: usize, tables: &[TableShape]) -> Vec<Gl> {
    let pattern = InteractionPattern::new(vec![
        scalar(Kind::Challenge, "virtual_point"),
        scalar(Kind::Message, "virtual_eval"),
    ])
    .expect("two leaf steps");
    let mut separator = DomainSeparator::new(LAYOUT_VERSION, VIRTUAL_NAME, pattern);
    bind(&mut separator, num_variables, tables);
    words(&separator)
}

/// The claim-batching challenge.
pub(crate) fn batching(num_variables: usize, tables: &[TableShape], claims: usize, virtual_claims: usize) -> Vec<Gl> {
    let pattern = InteractionPattern::new(vec![scalar(Kind::Challenge, "batching")]).expect("one leaf step");
    let mut separator = DomainSeparator::new(LAYOUT_VERSION, BATCHING_NAME, pattern);
    bind(&mut separator, num_variables, tables);
    instance(&mut separator, claims);
    instance(&mut separator, virtual_claims);
    words(&separator)
}

/// A quadratic sum-check of `rounds` rounds without grinding.
pub(crate) fn sumcheck(rounds: usize) -> Vec<Gl> {
    words(&SumcheckShape::new(rounds, 0, Basis::Evaluation).domain_separator::<Gl, Ext>())
}

/// The WHIR run of `claims` opening batches.
pub(crate) fn whir(config: &Config, claims: usize) -> Vec<Gl> {
    words(&WhirShape::new(config, claims).domain_separator::<Gl, Ext>())
}

/// The commitment phase that binds a root.
pub(crate) fn commitment() -> Vec<Gl> {
    words(&commitment_domain_separator::<Gl>())
}
