//! Owns terminal private-witness label collection and padding.

use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};
use spartan2::traits::circuit::SpartanCircuit;

use crate::spartan_backend::{NeoFoldDeciderEngine, SpartanF, SplitR1CSShape};

pub(crate) fn collect_private_witness_labels<C>(circuit: &C, context: &str) -> Result<Vec<String>, String>
where
    C: SpartanCircuit<NeoFoldDeciderEngine>,
{
    let mut cs = LabelOnlyConstraintSystem::new();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| format!("{context} label shared allocation failed: {err}"))?;
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| format!("{context} label precommitted allocation failed: {err}"))?;
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| format!("{context} label synthesis failed: {err}"))?;
    Ok(cs.aux_labels())
}

pub(crate) fn padded_private_witness_labels(
    split_shape: &SplitR1CSShape<NeoFoldDeciderEngine>,
    private_witness_labels: &[String],
    context: &str,
) -> Result<Vec<Option<String>>, String> {
    if private_witness_labels.len() != split_shape.num_variables_unpadded() {
        return Err(format!(
            "{context} unpadded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables_unpadded(),
            private_witness_labels.len()
        ));
    }

    let mut padded = Vec::with_capacity(split_shape.num_variables());
    let mut cursor = 0usize;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_shared_unpadded(),
        split_shape.num_shared(),
        context,
        "shared",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_precommitted_unpadded(),
        split_shape.num_precommitted(),
        context,
        "precommitted",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_rest_unpadded(),
        split_shape.num_rest(),
        context,
        "rest",
    )?;

    if cursor != private_witness_labels.len() {
        return Err(format!(
            "{context} witness label padding consumed {cursor} labels but {} were supplied",
            private_witness_labels.len()
        ));
    }
    if padded.len() != split_shape.num_variables() {
        return Err(format!(
            "{context} padded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables(),
            padded.len()
        ));
    }
    Ok(padded)
}

fn push_padded_witness_label_segment(
    padded: &mut Vec<Option<String>>,
    labels: &[String],
    cursor: &mut usize,
    unpadded_len: usize,
    padded_len: usize,
    context: &str,
    segment_name: &str,
) -> Result<(), String> {
    if padded_len < unpadded_len {
        return Err(format!(
            "{context} {segment_name} witness segment has padded length {padded_len} below unpadded length {unpadded_len}"
        ));
    }
    let end = cursor
        .checked_add(unpadded_len)
        .ok_or_else(|| format!("{context} witness label cursor overflow"))?;
    if end > labels.len() {
        return Err(format!(
            "{context} {segment_name} witness labels exceed collected label count"
        ));
    }
    padded.extend(labels[*cursor..end].iter().cloned().map(Some));
    padded.resize(padded.len() + (padded_len - unpadded_len), None);
    *cursor = end;
    Ok(())
}

#[derive(Clone, Debug)]
struct LabelOnlyConstraintSystem {
    current_namespace: Vec<String>,
    inputs: usize,
    aux_labels: Vec<String>,
}

impl LabelOnlyConstraintSystem {
    fn new() -> Self {
        Self {
            current_namespace: Vec::new(),
            inputs: 1,
            aux_labels: Vec::new(),
        }
    }

    fn alloc_path(&self, annotation: &str) -> String {
        if self.current_namespace.is_empty() {
            return annotation.to_owned();
        }
        let mut path = self.current_namespace.join("/");
        path.push('/');
        path.push_str(annotation);
        path
    }

    fn aux_labels(self) -> Vec<String> {
        self.aux_labels
    }
}

impl ConstraintSystem<SpartanF> for LabelOnlyConstraintSystem {
    type Root = Self;

    fn alloc<FN, A, AR>(&mut self, annotation: A, _: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let var = Variable::new_unchecked(bellpepper_core::Index::Aux(self.aux_labels.len()));
        self.aux_labels.push(self.alloc_path(&annotation().into()));
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, _: A, _: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let var = Variable::new_unchecked(bellpepper_core::Index::Input(self.inputs));
        self.inputs = self
            .inputs
            .checked_add(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        Ok(var)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _: LA, _: LB, _: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.current_namespace.push(name_fn().into());
    }

    fn pop_namespace(&mut self) {
        assert!(self.current_namespace.pop().is_some());
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
