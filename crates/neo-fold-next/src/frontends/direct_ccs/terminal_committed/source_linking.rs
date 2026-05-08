use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, Variable};

use super::types::DirectCcsTerminalR2Layout;
use crate::construction2_terminal::TerminalPrivateColumnEncoding;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::witness::PackedWitnessVar;

pub(super) struct DirectSourceWitnessLinkingCs<'a, 'b, CS: ConstraintSystem<SpartanF>> {
    inner: &'a mut CS,
    layout: &'b DirectCcsTerminalR2Layout,
    packed_z: &'b PackedWitnessVar,
    committed_width: usize,
    public_len: usize,
    current_namespace: Vec<String>,
    pub(super) source_link_constraints: usize,
}

impl<'a, 'b, CS: ConstraintSystem<SpartanF>> DirectSourceWitnessLinkingCs<'a, 'b, CS> {
    pub(super) fn new(
        inner: &'a mut CS,
        layout: &'b DirectCcsTerminalR2Layout,
        packed_z: &'b PackedWitnessVar,
        committed_width: usize,
        public_len: usize,
    ) -> Self {
        Self {
            inner,
            layout,
            packed_z,
            committed_width,
            public_len,
            current_namespace: Vec::new(),
            source_link_constraints: 0,
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

    fn source_lc_with_encoding(
        &self,
        offset: usize,
        encoding: TerminalPrivateColumnEncoding,
    ) -> Result<LinearCombination<SpartanF>, SynthesisError> {
        let mut lc = LinearCombination::<SpartanF>::zero();
        for limb_idx in 0..encoding.limb_count() {
            let logical_col = self
                .public_len
                .checked_add(offset)
                .and_then(|value| value.checked_add(limb_idx))
                .ok_or(SynthesisError::Unsatisfiable)?;
            let limb = self
                .packed_z
                .logical_entry(self.committed_width, logical_col)?;
            lc = lc + (SpartanF::from_canonical_u64(1u64 << limb_idx), limb.get_variable());
        }
        Ok(lc)
    }
}

impl<CS: ConstraintSystem<SpartanF>> ConstraintSystem<SpartanF> for DirectSourceWitnessLinkingCs<'_, '_, CS> {
    type Root = Self;

    fn alloc<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let annotation = annotation().into();
        let label = self.alloc_path(&annotation);
        let var = self.inner.alloc(|| annotation.clone(), f)?;
        if let Some((offset, encoding)) = self.layout.source_binding(&label) {
            let source_lc = self.source_lc_with_encoding(offset, encoding)?;
            self.inner.enforce(
                || format!("direct_terminal_r2_source_link_{label}"),
                |lc| lc + var,
                |lc| lc + CS::one(),
                |_| source_lc,
            );
            self.source_link_constraints += 1;
        }
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.inner.alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
        self.inner.enforce(annotation, a, b, c);
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        let name = name_fn().into();
        self.current_namespace.push(name.clone());
        self.inner.push_namespace(|| name);
    }

    fn pop_namespace(&mut self) {
        assert!(self.current_namespace.pop().is_some());
        self.inner.pop_namespace();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
