//! Bellpepper frontend — synthesize a Bellpepper circuit into sparse CCS.
//!
//! This module owns only the adapter boundary from Bellpepper's R1CS API
//! to the CCS relation the lifecycle already folds. It preserves sparse
//! matrices, pads `x` to complete public rings, returns `z = [x | w]`, and leaves
//! preprocessing / folding to the normal lifecycle entrypoints.

use bellpepper_core::{Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;
use neo_ccs::{sparse_r1cs_to_ccs, CcsMatrix, CcsStructure, CscMat};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::lifecycle::Preprocessing;
use crate::paper::relations::{CcsInstance, RelationError};

const AUX_FLAG: u32 = 1 << 31;

/// Goldilocks field wrapper that implements Bellpepper's `ff::PrimeField`.
///
/// Bellpepper is built around the `ff` traits, while the rest of this
/// crate uses `neo_math::F` / Plonky3 traits. The modulus is exactly the
/// Goldilocks modulus, so conversion into `neo_math::F` is just canonical
/// `u64` conversion.
#[derive(PrimeField)]
#[PrimeFieldModulus = "18446744069414584321"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct BellpepperGoldilocks([u64; 2]);

/// Bellpepper synthesis dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BellpepperShape {
    pub constraints: usize,
    pub inputs: usize,
    pub aux: usize,
}

/// Sparse CCS relation plus the satisfying assignment Bellpepper produced.
#[derive(Clone, Debug)]
pub struct BellpepperCcs {
    pub structure: CcsStructure<F>,
    pub sparse_r1cs: SparseR1cs,
    pub assignment: Vec<F>,
    pub shape: BellpepperShape,
}

impl BellpepperCcs {
    pub fn public_input_len(&self) -> usize {
        self.assignment.len() - self.shape.aux
    }

    pub fn public_inputs(&self) -> &[F] {
        &self.assignment[..self.public_input_len()]
    }

    pub fn private_witness(&self) -> &[F] {
        &self.assignment[self.public_input_len()..]
    }

    /// Build one lifecycle input instance from this assignment.
    pub fn build_instance(&self, prep: &Preprocessing) -> Result<CcsInstance, BellpepperFrontendError> {
        if self.assignment.first() != Some(&F::ONE) {
            return Err(BellpepperFrontendError::NonCanonicalConstant);
        }
        if let Some(offset) = self.assignment[self.shape.inputs..self.public_input_len()]
            .iter()
            .position(|value| *value != F::ZERO)
        {
            return Err(BellpepperFrontendError::NonCanonicalPublicPadding {
                index: self.shape.inputs + offset,
            });
        }
        Ok(CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            &self.assignment,
            self.public_input_len(),
        )?)
    }
}

#[derive(Debug, Error)]
pub enum BellpepperFrontendError {
    #[error("Bellpepper synthesis failed: {0:?}")]
    Synthesis(SynthesisError),
    #[error(transparent)]
    Ccs(#[from] neo_ccs::RelationError),
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error("Bellpepper implicit constant-one input is not one")]
    NonCanonicalConstant,
    #[error("Bellpepper public-ring completion is nonzero at index {index}")]
    NonCanonicalPublicPadding { index: usize },
}

/// Synthesize any Bellpepper circuit over Goldilocks into sparse CCS.
pub fn synthesize_to_ccs<C>(circuit: C) -> Result<BellpepperCcs, BellpepperFrontendError>
where
    C: Circuit<BellpepperGoldilocks>,
{
    let mut cs = TripletConstraintSystem::new();
    circuit
        .synthesize(&mut cs)
        .map_err(BellpepperFrontendError::Synthesis)?;

    let TripletConstraintSystem {
        inputs,
        aux,
        num_constraints,
        a_trips,
        b_trips,
        c_trips,
    } = cs;
    let source_constraints = num_constraints as usize;
    let num_inputs = inputs.len();
    let num_aux = aux.len();
    let public_input_len = num_inputs.div_ceil(D) * D;
    let public_padding_len = public_input_len - num_inputs;
    let num_constraints = source_constraints + public_padding_len;
    let num_variables = public_input_len + num_aux;

    let mut assignment = inputs;
    assignment.resize(public_input_len, F::ZERO);
    assignment.extend(aux);

    let mut a_trips = TripletConstraintSystem::resolve_triplets(a_trips, public_input_len);
    let mut b_trips = TripletConstraintSystem::resolve_triplets(b_trips, public_input_len);
    let c_trips = TripletConstraintSystem::resolve_triplets(c_trips, public_input_len);
    for (offset, column) in (num_inputs..public_input_len).enumerate() {
        let row = source_constraints + offset;
        a_trips.push((row, column, F::ONE));
        b_trips.push((row, 0, F::ONE));
    }

    let a = CcsMatrix::Csc(CscMat::from_triplets(a_trips, num_constraints, num_variables));
    let b = CcsMatrix::Csc(CscMat::from_triplets(b_trips, num_constraints, num_variables));
    let c = CcsMatrix::Csc(CscMat::from_triplets(c_trips, num_constraints, num_variables));

    let sparse_r1cs = SparseR1cs::new(
        a.clone(),
        b.clone(),
        c.clone(),
        num_constraints,
        num_variables,
        public_input_len,
    )?;

    Ok(BellpepperCcs {
        structure: sparse_r1cs_to_ccs(a, b, c)?,
        sparse_r1cs,
        assignment,
        shape: BellpepperShape {
            constraints: num_constraints,
            inputs: num_inputs,
            aux: num_aux,
        },
    })
}

fn fp_to_f(x: &BellpepperGoldilocks) -> F {
    let bytes = x.to_repr();
    F::from_u64(u64::from_le_bytes(
        bytes.0[0..8]
            .try_into()
            .expect("Goldilocks repr is at least 8 bytes"),
    ))
}

struct TripletConstraintSystem {
    inputs: Vec<F>,
    aux: Vec<F>,
    num_constraints: u32,
    a_trips: Vec<(u32, u32, F)>,
    b_trips: Vec<(u32, u32, F)>,
    c_trips: Vec<(u32, u32, F)>,
}

impl TripletConstraintSystem {
    fn new() -> Self {
        Self {
            inputs: vec![F::ONE],
            aux: Vec::new(),
            num_constraints: 0,
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
        }
    }

    fn push_lc_trips(row: u32, lc: &LinearCombination<BellpepperGoldilocks>, trips: &mut Vec<(u32, u32, F)>) {
        for (var, coeff) in lc.iter() {
            let value = fp_to_f(coeff);
            if value == F::ZERO {
                continue;
            }
            let col = match var.0 {
                Index::Input(idx) => u32::try_from(idx).expect("input index fits u32"),
                Index::Aux(idx) => AUX_FLAG | u32::try_from(idx).expect("aux index fits u32"),
            };
            trips.push((row, col, value));
        }
    }

    fn resolve_triplets(trips: Vec<(u32, u32, F)>, num_inputs: usize) -> Vec<(usize, usize, F)> {
        trips
            .into_iter()
            .map(|(row, col, value)| {
                let row = row as usize;
                if (col & AUX_FLAG) == 0 {
                    (row, col as usize, value)
                } else {
                    let aux_idx = (col & !AUX_FLAG) as usize;
                    (row, num_inputs + aux_idx, value)
                }
            })
            .collect()
    }
}

impl ConstraintSystem<BellpepperGoldilocks> for TripletConstraintSystem {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<BellpepperGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.aux.len();
        self.aux.push(fp_to_f(&f()?));
        Ok(Variable::new_unchecked(Index::Aux(idx)))
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<BellpepperGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.inputs.len();
        self.inputs.push(fp_to_f(&f()?));
        Ok(Variable::new_unchecked(Index::Input(idx)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<BellpepperGoldilocks>) -> LinearCombination<BellpepperGoldilocks>,
        LB: FnOnce(LinearCombination<BellpepperGoldilocks>) -> LinearCombination<BellpepperGoldilocks>,
        LC: FnOnce(LinearCombination<BellpepperGoldilocks>) -> LinearCombination<BellpepperGoldilocks>,
    {
        let row = self.num_constraints;
        self.num_constraints += 1;
        Self::push_lc_trips(row, &a(LinearCombination::zero()), &mut self.a_trips);
        Self::push_lc_trips(row, &b(LinearCombination::zero()), &mut self.b_trips);
        Self::push_lc_trips(row, &c(LinearCombination::zero()), &mut self.c_trips);
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
