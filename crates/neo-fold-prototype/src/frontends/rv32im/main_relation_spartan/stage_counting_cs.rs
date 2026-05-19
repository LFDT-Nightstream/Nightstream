//! Counts rows and variables emitted by named proof-gadget stages.

use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConstraintStageCounts {
    pub(crate) stage: &'static str,
    pub(crate) rows: usize,
    pub(crate) aux_columns: usize,
    pub(crate) public_columns: usize,
}

pub(crate) struct StageCountingCs<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> {
    inner: &'a mut CS,
    namespace_stack: Vec<String>,
    stages: Vec<ConstraintStageCounts>,
    _marker: std::marker::PhantomData<Scalar>,
}

impl<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>> StageCountingCs<'a, Scalar, CS> {
    pub(crate) fn new(inner: &'a mut CS) -> Self {
        Self {
            inner,
            namespace_stack: Vec::new(),
            stages: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn into_stage_counts(mut self) -> Vec<ConstraintStageCounts> {
        self.stages.sort_by_key(|counts| stage_order(counts.stage));
        self.stages
    }

    fn classify(&self, annotation: &str) -> &'static str {
        let mut label = self.namespace_stack.join("/");
        if !label.is_empty() {
            label.push('/');
        }
        label.push_str(annotation);
        classify_pi_ccs_label(&label)
    }

    fn add_row(&mut self, annotation: &str) {
        self.add_to_stage(annotation, |counts| counts.rows += 1);
    }

    fn add_aux(&mut self, annotation: &str) {
        self.add_to_stage(annotation, |counts| counts.aux_columns += 1);
    }

    fn add_public(&mut self, annotation: &str) {
        self.add_to_stage(annotation, |counts| counts.public_columns += 1);
    }

    fn add_to_stage(&mut self, annotation: &str, update: impl FnOnce(&mut ConstraintStageCounts)) {
        let stage = self.classify(annotation);
        if let Some(counts) = self.stages.iter_mut().find(|counts| counts.stage == stage) {
            update(counts);
            return;
        }
        let mut counts = ConstraintStageCounts {
            stage,
            ..ConstraintStageCounts::default()
        };
        update(&mut counts);
        self.stages.push(counts);
    }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> ConstraintSystem<Scalar> for StageCountingCs<'_, Scalar, CS> {
    type Root = Self;

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let annotation = annotation().into();
        self.add_aux(&annotation);
        self.inner.alloc(|| annotation, f)
    }

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let annotation = annotation().into();
        self.add_public(&annotation);
        self.inner.alloc_input(|| annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        let annotation = annotation().into();
        self.add_row(&annotation);
        self.inner.enforce(|| annotation, a, b, c);
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        let name = name_fn().into();
        self.namespace_stack.push(name.clone());
        self.inner.push_namespace(|| name);
    }

    fn pop_namespace(&mut self) {
        self.inner.pop_namespace();
        let _ = self.namespace_stack.pop();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_witness_generator(&self) -> bool {
        self.inner.is_witness_generator()
    }
}

fn classify_pi_ccs_label(label: &str) -> &'static str {
    if label.contains("terminal_fe") {
        "terminal_fe_identity"
    } else if label.contains("terminal_nc") {
        "terminal_nc_identity"
    } else if label.contains("output_binding") {
        "output_binding"
    } else if label.contains("_ccs_output_") {
        "alloc_output_ce_surfaces"
    } else if label.contains("fold_digest") {
        "fold_digest"
    } else if label.contains("nc_sumcheck") || label.contains("nc_round") || label.contains("initial_sum_nc") {
        "nc_sumcheck"
    } else if label.contains("fe_sumcheck") || label.contains("fe_round") || label.contains("initial_sum_fe") {
        "fe_sumcheck"
    } else if label.contains("sample_challenges") {
        "sample_challenges"
    } else if label.contains("bind_me_input") || label.contains("bind_me_inputs") {
        "bind_me_inputs"
    } else if label.contains("bind_header") {
        "bind_header"
    } else if label.contains("_fresh_claim_") || label.contains("public_step") || label.contains("public_chunk") {
        "fresh_claim_and_public_chunk"
    } else {
        "other"
    }
}

fn stage_order(stage: &str) -> usize {
    match stage {
        "fresh_claim_and_public_chunk" => 0,
        "bind_header" => 1,
        "bind_me_inputs" => 2,
        "sample_challenges" => 3,
        "fe_sumcheck" => 4,
        "nc_sumcheck" => 5,
        "fold_digest" => 6,
        "alloc_output_ce_surfaces" => 7,
        "output_binding" => 8,
        "terminal_fe_identity" => 9,
        "terminal_nc_identity" => 10,
        _ => 11,
    }
}
