//! Owns Bellpepper R1CS export for direct CCS/R1CS frontends.

use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsMatrix, CscMat};
use neo_math::{balanced::to_balanced_i128, D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsProgram};
use super::super::step::{direct_ccs_step_from_low_norm_full_witness, DirectCcsStep};
use super::r1cs::direct_ccs_program_from_sparse_r1cs_with_public_input_len;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanCircuit, SpartanF};
use spartan2::bellpepper::solver::SatisfyingAssignment;

#[derive(Clone, Debug)]
pub struct DirectSparseR1csExport {
    pub a: CcsMatrix<F>,
    pub b: CcsMatrix<F>,
    pub c: CcsMatrix<F>,
    /// Full R1CS assignment in CCS column order.
    ///
    /// Values must already fit the SuperNeo low-norm packing budget. Frontends
    /// with arbitrary field-valued internals must bit/limb encode them before
    /// exporting this direct step.
    pub witness: Vec<F>,
    pub public_input_len: usize,
    pub constraint_count: usize,
    pub variable_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectSparseR1csLowNormReport {
    pub variable_count: usize,
    pub public_input_len: usize,
    pub low_norm_packable: bool,
    pub violation_count: usize,
    pub first_violations: Vec<DirectSparseR1csLowNormViolation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectSparseR1csLowNormViolation {
    pub index: usize,
    pub is_public: bool,
    pub centered_value: i128,
}

impl DirectSparseR1csExport {
    pub fn to_direct_ccs_program(&self) -> Result<DirectCcsProgram, DirectCcsFPrimeSnarkError> {
        let params = NeoParams::goldilocks_auto_r1cs_ccs(self.constraint_count)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        direct_ccs_program_from_sparse_r1cs_with_public_input_len(
            &params,
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.public_input_len,
        )
    }

    pub fn into_direct_ccs_step<L>(
        self,
        program: &DirectCcsProgram,
        log: &L,
        label: impl Into<String>,
    ) -> Result<DirectCcsStep, DirectCcsFPrimeSnarkError>
    where
        L: SModuleHomomorphism<F, Commitment>,
    {
        let report = self.low_norm_report(program.params(), 4);
        if !report.low_norm_packable {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "direct R1CS witness is not SuperNeo low-norm packable: {} violation(s); {}",
                report.violation_count,
                report.first_violation_summary()
            )));
        }
        direct_ccs_step_from_low_norm_full_witness(program, log, label, &self.witness, self.public_input_len)
    }

    pub fn into_direct_ccs_program_and_step<L>(
        self,
        log: &L,
        label: impl Into<String>,
    ) -> Result<(DirectCcsProgram, DirectCcsStep), DirectCcsFPrimeSnarkError>
    where
        L: SModuleHomomorphism<F, Commitment>,
    {
        let program = self.to_direct_ccs_program()?;
        let step = self.into_direct_ccs_step(&program, log, label)?;
        Ok((program, step))
    }

    pub fn low_norm_report(&self, params: &NeoParams, max_violations: usize) -> DirectSparseR1csLowNormReport {
        let mut first_violations = Vec::new();
        let mut violation_count = 0usize;
        for (index, &value) in self.witness.iter().enumerate() {
            if !is_superneo_digit_representable(value, params.b) {
                violation_count += 1;
                if first_violations.len() < max_violations {
                    first_violations.push(DirectSparseR1csLowNormViolation {
                        index,
                        is_public: index < self.public_input_len,
                        centered_value: to_balanced_i128(value),
                    });
                }
            }
        }
        DirectSparseR1csLowNormReport {
            variable_count: self.variable_count,
            public_input_len: self.public_input_len,
            low_norm_packable: violation_count == 0,
            violation_count,
            first_violations,
        }
    }
}

impl DirectSparseR1csLowNormReport {
    fn first_violation_summary(&self) -> String {
        let Some(first) = self.first_violations.first() else {
            return "no representative violation captured".into();
        };
        let visibility = if first.is_public { "public" } else { "private" };
        format!(
            "first violation index {} ({visibility}) centered={}",
            first.index, first.centered_value
        )
    }
}

/// Exports a Bellpepper/Spartan circuit into the direct sparse R1CS adapter.
///
/// Column order is deliberately direct-CCS friendly: all Bellpepper public
/// inputs, including the `one` input, come first, followed by all auxiliary
/// variables. Callers still need the usual SuperNeo low-norm check when turning
/// the export into a `DirectCcsStep`; circuits with arbitrary field-valued
/// internals will be rejected there until they are lowered into low-norm limbs.
pub fn direct_sparse_r1cs_export_from_spartan_circuit<C>(
    circuit: &C,
) -> Result<DirectSparseR1csExport, DirectCcsFPrimeSnarkError>
where
    C: SpartanCircuit<NeoFoldDeciderEngine>,
{
    if circuit.num_challenges() != 0 {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct R1CS export does not support randomized Spartan challenge inputs".into(),
        ));
    }

    let mut shape_cs = ShapeCS::<NeoFoldDeciderEngine>::new();
    synthesize_spartan_circuit(circuit, &mut shape_cs)
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("direct R1CS shape synthesis failed: {err}")))?;

    let mut witness_cs = SatisfyingAssignment::<NeoFoldDeciderEngine>::new();
    synthesize_spartan_circuit(circuit, &mut witness_cs)
        .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("direct R1CS witness synthesis failed: {err}")))?;

    if shape_cs.num_inputs() != witness_cs.inputs_slice().len() || shape_cs.num_aux() != witness_cs.aux_slice().len() {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct R1CS export shape/witness allocation mismatch".into(),
        ));
    }
    validate_public_values(circuit, &witness_cs)?;
    validate_satisfied_assignment(&shape_cs, &witness_cs)?;

    let public_input_len = witness_cs.inputs_slice().len();
    let variable_count = public_input_len + witness_cs.aux_slice().len();
    let mut witness = Vec::with_capacity(variable_count);
    witness.extend(
        witness_cs
            .inputs_slice()
            .iter()
            .copied()
            .map(spartan_to_native),
    );
    witness.extend(
        witness_cs
            .aux_slice()
            .iter()
            .copied()
            .map(spartan_to_native),
    );

    let rows = shape_cs.num_constraints();
    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let mut c_trips = Vec::new();
    for (row, (a, b, c)) in shape_cs.constraints.iter().enumerate() {
        push_lc_trips(row, public_input_len, a, &mut a_trips)?;
        push_lc_trips(row, public_input_len, b, &mut b_trips)?;
        push_lc_trips(row, public_input_len, c, &mut c_trips)?;
    }

    Ok(DirectSparseR1csExport {
        a: CcsMatrix::Csc(CscMat::from_triplets(a_trips, rows, variable_count)),
        b: CcsMatrix::Csc(CscMat::from_triplets(b_trips, rows, variable_count)),
        c: CcsMatrix::Csc(CscMat::from_triplets(c_trips, rows, variable_count)),
        witness,
        public_input_len,
        constraint_count: rows,
        variable_count,
    })
}

fn validate_public_values<C>(
    circuit: &C,
    witness_cs: &SatisfyingAssignment<NeoFoldDeciderEngine>,
) -> Result<(), DirectCcsFPrimeSnarkError>
where
    C: SpartanCircuit<NeoFoldDeciderEngine>,
{
    let declared = circuit.public_values().map_err(|err| {
        DirectCcsFPrimeSnarkError::Input(format!("direct R1CS public value extraction failed: {err}"))
    })?;
    let allocated = witness_cs
        .inputs_slice()
        .get(1..)
        .ok_or_else(|| DirectCcsFPrimeSnarkError::Input("direct R1CS export missing Bellpepper one input".into()))?;
    if witness_cs.inputs_slice()[0] != SpartanF::from_canonical_u64(1) {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct R1CS export Bellpepper one input is not one".into(),
        ));
    }
    if declared.as_slice() != allocated {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct R1CS public_values mismatch: declared {} field(s), allocated {} field(s)",
            declared.len(),
            allocated.len()
        )));
    }
    Ok(())
}

fn validate_satisfied_assignment(
    shape_cs: &ShapeCS<NeoFoldDeciderEngine>,
    witness_cs: &SatisfyingAssignment<NeoFoldDeciderEngine>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let inputs = witness_cs.inputs_slice();
    let aux = witness_cs.aux_slice();
    for (row, (a, b, c)) in shape_cs.constraints.iter().enumerate() {
        let a_value = eval_lc(a, inputs, aux)?;
        let b_value = eval_lc(b, inputs, aux)?;
        let c_value = eval_lc(c, inputs, aux)?;
        if a_value * b_value != c_value {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "direct R1CS export assignment does not satisfy row {row}"
            )));
        }
    }
    Ok(())
}

fn eval_lc(
    lc: &LinearCombination<SpartanF>,
    inputs: &[SpartanF],
    aux: &[SpartanF],
) -> Result<SpartanF, DirectCcsFPrimeSnarkError> {
    let mut acc = SpartanF::from_canonical_u64(0);
    for (var, coeff) in lc.iter() {
        let value = match var.get_unchecked() {
            Index::Input(idx) => inputs.get(idx).copied(),
            Index::Aux(idx) => aux.get(idx).copied(),
        }
        .ok_or_else(|| DirectCcsFPrimeSnarkError::Input("direct R1CS export variable index out of bounds".into()))?;
        acc += *coeff * value;
    }
    Ok(acc)
}

fn synthesize_spartan_circuit<C, CS>(circuit: &C, cs: &mut CS) -> Result<(), SynthesisError>
where
    C: SpartanCircuit<NeoFoldDeciderEngine>,
    CS: ConstraintSystem<SpartanF>,
{
    let shared = circuit.shared(cs)?;
    let precommitted = circuit.precommitted(cs, &shared)?;
    circuit.synthesize(cs, &shared, &precommitted, None)
}

fn push_lc_trips(
    row: usize,
    public_input_len: usize,
    lc: &LinearCombination<SpartanF>,
    trips: &mut Vec<(usize, usize, F)>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (var, coeff) in lc.iter() {
        let col = match var.get_unchecked() {
            Index::Input(idx) => idx,
            Index::Aux(idx) => public_input_len
                .checked_add(idx)
                .ok_or_else(|| DirectCcsFPrimeSnarkError::Input("direct R1CS export column index overflow".into()))?,
        };
        trips.push((row, col, spartan_to_native(*coeff)));
    }
    Ok(())
}

fn spartan_to_native(value: SpartanF) -> F {
    F::from_u64(value.to_canonical_u64())
}

fn is_superneo_digit_representable(value: F, base: u32) -> bool {
    if base < 2 {
        return false;
    }
    let mut remainder = to_balanced_i128(value);
    let base = base as i128;
    for _ in 0..D {
        let (_, quotient) = balanced_divrem(remainder, base);
        remainder = quotient;
    }
    remainder == 0
}

fn balanced_divrem(value: i128, base: i128) -> (i128, i128) {
    debug_assert!(base >= 2);
    let mut remainder = value % base;
    let mut quotient = (value - remainder) / base;
    let half = base / 2;
    if remainder > half {
        remainder -= base;
        quotient += 1;
    } else if remainder < -half {
        remainder += base;
        quotient -= 1;
    }
    (remainder, quotient)
}
