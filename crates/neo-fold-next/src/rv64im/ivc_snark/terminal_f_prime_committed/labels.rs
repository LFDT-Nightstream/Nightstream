//! Owns label collection for terminal `F'` auxiliary variables.
//!
//! This module is classification-only: it records the R1CS allocation paths
//! used to choose the SuperNeo low-norm limb encoding. Shape and assignment
//! synthesis remain owned by the terminal committed-step relation.

use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

use crate::rv64im::SimpleKernelError;

use super::super::{Rv64imDeciderEngine, SpartanCircuit, SpartanF};

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
        let var = Variable::new_unchecked(Index::Aux(self.aux_labels.len()));
        self.aux_labels.push(self.alloc_path(&annotation().into()));
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, annotation: A, _: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let var = Variable::new_unchecked(Index::Input(self.inputs));
        let _ = annotation;
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

pub(super) fn collect_private_witness_labels<C>(circuit: &C) -> Result<Vec<String>, SimpleKernelError>
where
    C: SpartanCircuit<Rv64imDeciderEngine>,
{
    let mut cs = LabelOnlyConstraintSystem::new();
    let shared = circuit.shared(&mut cs).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM terminal F' label shared allocation failed: {err}"))
    })?;
    let precommitted = circuit.precommitted(&mut cs, &shared).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM terminal F' label precommitted allocation failed: {err}"
        ))
    })?;
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal F' label synthesis failed: {err}")))?;
    Ok(cs.aux_labels())
}
