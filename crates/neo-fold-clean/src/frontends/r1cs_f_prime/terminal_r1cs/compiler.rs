use std::collections::HashMap;

use neo_ajtai::{precompute_rot_columns, AjtaiSModule};
use neo_ccs::build_superneo_ring_forms;
use neo_math::{KExtensions, D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use toy_spartan::{provider::goldi::F as SpartanF, SparseMatrix, SplitR1CSShape};

use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::relations::{superneo_has_canonical_x_shape, CcsClaim, CcsWitness, CeClaim, Structure, WitnessMat};

use super::{
    CompiledTerminalR1cs, CompiledTerminalR1csStatement, LeanNativeCcsManifest, TerminalR1csError, TerminalR1csInput,
    TerminalR1csStatement, TerminalSpartanEngine,
};
use crate::frontends::r1cs_f_prime::lean_manifest::{ColumnId, ManifestCost, ManifestTerm};
use crate::frontends::r1cs_f_prime::lean_nebula_combined_manifest::{
    map_native_index, map_nebula_index, CombinedLayout, LeanNebulaCombinedManifest, NebulaFamily, NebulaTerm,
};

pub(super) const MAX_DIRECT_TERMINAL_ROWS: usize = 1_000_000;

type Rotations = [[F; D]; D];

pub(super) fn compile(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    input: TerminalR1csInput<'_>,
) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
    let compiled = compile_relation(
        manifest,
        log,
        input.running_claims,
        Some(input.running_witnesses),
        &input.fresh.claim,
        Some(&input.fresh.witness),
    )?;
    if let Some(row) = compiled.builder.first_unsatisfied_row() {
        return Err(TerminalR1csError::Unsatisfied(row));
    }
    let witness = compiled.builder.witness();
    let private_values = compiled
        .private_vars
        .iter()
        .map(|variable| to_spartan(witness[variable.col()]))
        .collect();
    Ok(CompiledTerminalR1cs {
        shape: compiled.shape,
        private_values,
        public_values: compiled.public_values,
        lean_public_columns: compiled.lean_public_columns,
    })
}

pub(super) fn compile_statement(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    statement: TerminalR1csStatement<'_>,
) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
    let compiled = compile_relation(
        manifest,
        log,
        statement.running_claims,
        None,
        statement.fresh_claim,
        None,
    )?;
    Ok(CompiledTerminalR1csStatement {
        shape: compiled.shape,
        public_values: compiled.public_values,
        lean_public_columns: compiled.lean_public_columns,
    })
}

pub(super) fn compile_combined(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
    input: TerminalR1csInput<'_>,
) -> Result<CompiledTerminalR1cs, TerminalR1csError> {
    let compiled = compile_combined_relation(
        manifest,
        log,
        input.running_claims,
        Some(input.running_witnesses),
        &input.fresh.claim,
        Some(&input.fresh.witness),
    )?;
    if let Some(row) = compiled.builder.first_unsatisfied_row() {
        return Err(TerminalR1csError::Unsatisfied(row));
    }
    let witness = compiled.builder.witness();
    let private_values = compiled
        .private_vars
        .iter()
        .map(|variable| to_spartan(witness[variable.col()]))
        .collect();
    Ok(CompiledTerminalR1cs {
        shape: compiled.shape,
        private_values,
        public_values: compiled.public_values,
        lean_public_columns: compiled.lean_public_columns,
    })
}

pub(super) fn compile_combined_statement(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
    statement: TerminalR1csStatement<'_>,
) -> Result<CompiledTerminalR1csStatement, TerminalR1csError> {
    let compiled = compile_combined_relation(
        manifest,
        log,
        statement.running_claims,
        None,
        statement.fresh_claim,
        None,
    )?;
    Ok(CompiledTerminalR1csStatement {
        shape: compiled.shape,
        public_values: compiled.public_values,
        lean_public_columns: compiled.lean_public_columns,
    })
}

struct CompiledRelation {
    builder: R1csBuilder,
    private_vars: Vec<Var>,
    shape: SplitR1CSShape<TerminalSpartanEngine>,
    public_values: Vec<SpartanF>,
    lean_public_columns: usize,
}

fn compile_relation(
    manifest: &LeanNativeCcsManifest,
    log: &AjtaiSModule,
    running_claims: &[CeClaim],
    running_witnesses: Option<&[WitnessMat]>,
    fresh_claim: &CcsClaim,
    fresh_witness: Option<&CcsWitness>,
) -> Result<CompiledRelation, TerminalR1csError> {
    let descriptor = manifest.terminal_r1cs();
    validate_ajtai_setup(manifest, log)?;
    let expected_cost = descriptor.cost();
    if expected_cost.recurring_rows() > MAX_DIRECT_TERMINAL_ROWS {
        return Err(TerminalR1csError::ResourceLimit {
            rows: expected_cost.recurring_rows(),
            cap: MAX_DIRECT_TERMINAL_ROWS,
        });
    }
    require_len("running claims", manifest.running_claim_count(), running_claims.len())?;
    if let Some(witnesses) = running_witnesses {
        require_len("running witnesses", manifest.running_claim_count(), witnesses.len())?;
    }
    require_len("fresh claims", manifest.fresh_claim_count(), 1)?;

    let step = manifest
        .emit_phi81_step(|_| Some(F::ZERO))
        .map_err(|error| TerminalR1csError::Manifest(error.to_string()))?;
    let structure = step.structure();
    validate_structure(manifest, structure)?;
    let rotations = verifier_rotations(log, structure.m, descriptor.verifier_rows())?;

    let zero_running;
    let running_witnesses = match running_witnesses {
        Some(witnesses) => witnesses,
        None => {
            zero_running =
                vec![neo_ccs::Mat::virtual_constant(D, structure.m.div_ceil(D), F::ZERO); running_claims.len()];
            &zero_running
        }
    };
    let zero_fresh;
    let fresh_witness = match fresh_witness {
        Some(witness) => witness,
        None => {
            zero_fresh = CcsWitness {
                w: Vec::new(),
                Z: neo_ccs::Mat::virtual_constant(D, structure.m.div_ceil(D), F::ZERO),
            };
            &zero_fresh
        }
    };

    let mut builder = R1csBuilder::new();
    let mut private_vars = Vec::with_capacity(expected_cost.committed_columns() + expected_cost.auxiliary_columns());
    let mut public_vars = Vec::with_capacity(expected_cost.public_columns().saturating_sub(1));

    for (claim, witness) in running_claims.iter().zip(running_witnesses) {
        compile_running(
            &mut builder,
            &mut private_vars,
            &mut public_vars,
            structure,
            &rotations,
            manifest.public_carrier_width(),
            claim,
            witness,
        )?;
    }
    compile_fresh(
        &mut builder,
        &mut private_vars,
        &mut public_vars,
        manifest,
        structure,
        &step,
        &rotations,
        fresh_claim,
        fresh_witness,
    )?;

    finish_relation(builder, private_vars, public_vars, expected_cost)
}

fn compile_combined_relation(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
    running_claims: &[CeClaim],
    running_witnesses: Option<&[WitnessMat]>,
    fresh_claim: &CcsClaim,
    fresh_witness: Option<&CcsWitness>,
) -> Result<CompiledRelation, TerminalR1csError> {
    let descriptor = manifest.terminal_r1cs();
    validate_combined_ajtai_setup(manifest, log)?;
    let expected_cost = descriptor.cost();
    if expected_cost.recurring_rows() > MAX_DIRECT_TERMINAL_ROWS {
        return Err(TerminalR1csError::ResourceLimit {
            rows: expected_cost.recurring_rows(),
            cap: MAX_DIRECT_TERMINAL_ROWS,
        });
    }
    require_len("running claims", manifest.running_claim_count(), running_claims.len())?;
    if let Some(witnesses) = running_witnesses {
        require_len("running witnesses", manifest.running_claim_count(), witnesses.len())?;
    }
    require_len("fresh claims", manifest.fresh_claim_count(), 1)?;

    let structure = manifest
        .terminal_structure()
        .map_err(|error| TerminalR1csError::Manifest(error.to_string()))?;
    validate_combined_structure(manifest, &structure)?;
    let rotations = verifier_rotations(log, structure.m, descriptor.verifier_rows())?;

    let zero_running;
    let running_witnesses = match running_witnesses {
        Some(witnesses) => witnesses,
        None => {
            zero_running =
                vec![neo_ccs::Mat::virtual_constant(D, structure.m.div_ceil(D), F::ZERO); running_claims.len()];
            &zero_running
        }
    };
    let zero_fresh;
    let fresh_witness = match fresh_witness {
        Some(witness) => witness,
        None => {
            zero_fresh = CcsWitness {
                w: Vec::new(),
                Z: neo_ccs::Mat::virtual_constant(D, structure.m.div_ceil(D), F::ZERO),
            };
            &zero_fresh
        }
    };

    let mut builder = R1csBuilder::new();
    let mut private_vars = Vec::with_capacity(expected_cost.committed_columns() + expected_cost.auxiliary_columns());
    let mut public_vars = Vec::with_capacity(expected_cost.public_columns().saturating_sub(1));

    for (claim, witness) in running_claims.iter().zip(running_witnesses) {
        compile_running(
            &mut builder,
            &mut private_vars,
            &mut public_vars,
            &structure,
            &rotations,
            manifest.public_carrier_width(),
            claim,
            witness,
        )?;
    }
    compile_combined_fresh(
        &mut builder,
        &mut private_vars,
        &mut public_vars,
        manifest,
        &structure,
        &rotations,
        fresh_claim,
        fresh_witness,
    )?;

    finish_relation(builder, private_vars, public_vars, expected_cost)
}

fn finish_relation(
    builder: R1csBuilder,
    private_vars: Vec<Var>,
    public_vars: Vec<Var>,
    expected_cost: ManifestCost,
) -> Result<CompiledRelation, TerminalR1csError> {
    check_count("terminal rows", expected_cost.recurring_rows(), builder.rows())?;
    check_count(
        "terminal private columns",
        expected_cost.committed_columns() + expected_cost.auxiliary_columns(),
        private_vars.len(),
    )?;
    check_count(
        "terminal public columns",
        expected_cost.public_columns(),
        public_vars.len() + 1,
    )?;
    check_count(
        "terminal allocated columns",
        builder.cols(),
        private_vars.len() + public_vars.len() + 1,
    )?;
    let old_to_new = column_permutation(builder.cols(), &private_vars, &public_vars)?;
    let total_columns = builder.cols();
    let (a_trips, b_trips, c_trips) = builder.sparse_triplets();
    let a = canonical_matrix(builder.rows(), total_columns, a_trips, &old_to_new)?;
    let b = canonical_matrix(builder.rows(), total_columns, b_trips, &old_to_new)?;
    let c = canonical_matrix(builder.rows(), total_columns, c_trips, &old_to_new)?;
    let shape = SplitR1CSShape::<TerminalSpartanEngine>::new(
        2,
        builder.rows(),
        0,
        0,
        private_vars.len(),
        public_vars.len(),
        0,
        a,
        b,
        c,
    )
    .map_err(|error| TerminalR1csError::Spartan(error.to_string()))?;
    let witness = builder.witness();
    let public_values = public_vars
        .iter()
        .map(|variable| to_spartan(witness[variable.col()]))
        .collect();
    Ok(CompiledRelation {
        builder,
        private_vars,
        shape,
        public_values,
        lean_public_columns: public_vars.len() + 1,
    })
}

fn validate_ajtai_setup(manifest: &LeanNativeCcsManifest, log: &AjtaiSModule) -> Result<(), TerminalR1csError> {
    let expected_rows = manifest.terminal_r1cs().verifier_rows();
    match log.seeded_params() {
        Some((rows, seed)) if rows == expected_rows && seed == manifest.ajtai_setup_seed() => Ok(()),
        _ => Err(TerminalR1csError::SetupMismatch),
    }
}

fn validate_combined_ajtai_setup(
    manifest: &LeanNebulaCombinedManifest,
    log: &AjtaiSModule,
) -> Result<(), TerminalR1csError> {
    let expected_rows = manifest.terminal_r1cs().verifier_rows();
    match log.seeded_params() {
        Some((rows, seed)) if rows == expected_rows && seed == manifest.ajtai_setup_seed() => Ok(()),
        _ => Err(TerminalR1csError::SetupMismatch),
    }
}

fn validate_structure(manifest: &LeanNativeCcsManifest, structure: &Structure) -> Result<(), TerminalR1csError> {
    let descriptor = manifest.terminal_r1cs();
    require_len(
        "terminal carrier width",
        descriptor.logical_width().div_ceil(D) * D,
        structure.m,
    )?;
    require_len("terminal row domain", 1usize << descriptor.row_variables(), structure.n)?;
    require_len("terminal matrix count", descriptor.matrix_count(), structure.t())?;
    require_len(
        "terminal public carrier",
        descriptor.public_ring_columns() * D,
        manifest.public_carrier_width(),
    )
}

fn validate_combined_structure(
    manifest: &LeanNebulaCombinedManifest,
    structure: &Structure,
) -> Result<(), TerminalR1csError> {
    let descriptor = manifest.terminal_r1cs();
    require_len(
        "terminal carrier width",
        descriptor.logical_width().div_ceil(D) * D,
        structure.m,
    )?;
    require_len("terminal row domain", 1usize << descriptor.row_variables(), structure.n)?;
    require_len("terminal matrix count", descriptor.matrix_count(), structure.t())?;
    require_len(
        "terminal public carrier",
        descriptor.public_ring_columns() * D,
        manifest.public_carrier_width(),
    )
}

fn verifier_rotations(
    log: &AjtaiSModule,
    carrier_width: usize,
    verifier_rows: usize,
) -> Result<Vec<Vec<Rotations>>, TerminalR1csError> {
    require_len("Ajtai ring degree", D, log.dims().0)?;
    require_len("Ajtai witness blocks", carrier_width.div_ceil(D), log.dims().1)?;
    require_len("Ajtai verifier rows", verifier_rows, log.kappa())?;
    let pp = log
        .materialize_pp()
        .map_err(|error| TerminalR1csError::Coefficients(error.to_string()))?;
    require_len("Ajtai materialized rows", verifier_rows, pp.m_rows.len())?;
    let mut all = Vec::with_capacity(pp.m_rows.len());
    for row in &pp.m_rows {
        require_len("Ajtai materialized columns", carrier_width / D, row.len())?;
        let mut row_rotations = Vec::with_capacity(row.len());
        for &ring_element in row {
            let mut rotations = [[F::ZERO; D]; D];
            precompute_rot_columns(ring_element, &mut rotations);
            row_rotations.push(rotations);
        }
        all.push(row_rotations);
    }
    Ok(all)
}

#[allow(clippy::too_many_arguments)]
fn compile_running(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    public_vars: &mut Vec<Var>,
    structure: &Structure,
    rotations: &[Vec<Rotations>],
    public_width: usize,
    claim: &CeClaim,
    witness: &WitnessMat,
) -> Result<(), TerminalR1csError> {
    validate_running_claim(structure, rotations.len(), public_width, claim, witness)?;
    let witness_values = packed_witness(witness, structure.m);
    let witness_wires = alloc_private_vec(builder, private_vars, &witness_values);
    let commitment_wires = alloc_public_vec(builder, public_vars, &claim.c.data);
    let projected_values = projected_ce_values(claim, public_width);
    let projection_wires = alloc_public_vec(builder, public_vars, &projected_values);

    let mut low_values = Vec::with_capacity(structure.t() * D);
    let mut high_values = Vec::with_capacity(structure.t() * D);
    for evaluation in &claim.y_ring {
        for value in &evaluation[..D] {
            let [low, high] = value.as_coeffs();
            low_values.push(low);
            high_values.push(high);
        }
    }
    let low_wires = alloc_public_vec(builder, public_vars, &low_values);
    let high_wires = alloc_public_vec(builder, public_vars, &high_values);
    let square_wires = alloc_squares(builder, private_vars, &witness_wires);

    enforce_ajtai(builder, rotations, &witness_wires, &commitment_wires);
    enforce_projection(builder, &witness_wires, &projection_wires);
    enforce_norm(builder, &witness_wires, &square_wires);
    enforce_fixed_evaluations(builder, structure, claim, &witness_wires, &low_wires, &high_wires)
}

#[allow(clippy::too_many_arguments)]
fn compile_fresh(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    public_vars: &mut Vec<Var>,
    manifest: &LeanNativeCcsManifest,
    structure: &Structure,
    step: &super::super::lean_native_ccs_manifest::NativePhi81StepEmission,
    rotations: &[Vec<Rotations>],
    claim: &CcsClaim,
    witness: &CcsWitness,
) -> Result<(), TerminalR1csError> {
    validate_fresh(
        structure,
        rotations.len(),
        manifest.public_carrier_width(),
        claim,
        witness,
    )?;
    let witness_values = packed_witness(&witness.Z, structure.m);
    let witness_wires = alloc_private_vec(builder, private_vars, &witness_values);
    let commitment_wires = alloc_public_vec(builder, public_vars, &claim.c.data);
    let projection_wires = alloc_public_vec(builder, public_vars, &claim.x);
    let square_wires = alloc_squares(builder, private_vars, &witness_wires);

    enforce_ajtai(builder, rotations, &witness_wires, &commitment_wires);
    enforce_projection(builder, &witness_wires, &projection_wires);
    enforce_norm(builder, &witness_wires, &square_wires);
    enforce_fresh_ccs(builder, private_vars, manifest, step, &witness_wires)
}

#[allow(clippy::too_many_arguments)]
fn compile_combined_fresh(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    public_vars: &mut Vec<Var>,
    manifest: &LeanNebulaCombinedManifest,
    structure: &Structure,
    rotations: &[Vec<Rotations>],
    claim: &CcsClaim,
    witness: &CcsWitness,
) -> Result<(), TerminalR1csError> {
    validate_fresh(
        structure,
        rotations.len(),
        manifest.public_carrier_width(),
        claim,
        witness,
    )?;
    let witness_values = packed_witness(&witness.Z, structure.m);
    let witness_wires = alloc_private_vec(builder, private_vars, &witness_values);
    let commitment_wires = alloc_public_vec(builder, public_vars, &claim.c.data);
    let projection_wires = alloc_public_vec(builder, public_vars, &claim.x);
    let square_wires = alloc_squares(builder, private_vars, &witness_wires);

    enforce_ajtai(builder, rotations, &witness_wires, &commitment_wires);
    enforce_projection(builder, &witness_wires, &projection_wires);
    enforce_norm(builder, &witness_wires, &square_wires);
    enforce_combined_fresh_relation(builder, private_vars, manifest, &witness_wires)
}

fn validate_running_claim(
    structure: &Structure,
    verifier_rows: usize,
    public_width: usize,
    claim: &CeClaim,
    witness: &WitnessMat,
) -> Result<(), TerminalR1csError> {
    if claim.adv.is_some() {
        return Err(TerminalR1csError::Unsupported("Nebula running commitment sidecars"));
    }
    validate_witness(witness, structure.m)?;
    validate_commitment(&claim.c, verifier_rows)?;
    require_len("running public width", public_width, claim.m_in)?;
    if !superneo_has_canonical_x_shape(&claim.X, claim.m_in) {
        return Err(TerminalR1csError::Unsupported(
            "running X is not a canonical whole-ring coefficient embedding",
        ));
    }
    require_len("running evaluation count", structure.t(), claim.y_ring.len())?;
    for values in &claim.y_ring {
        if values.len() != D && values.len() != D.next_power_of_two() {
            return Err(TerminalR1csError::Shape {
                what: "running evaluation lanes",
                expected: D,
                got: values.len(),
            });
        }
        if values[D..].iter().any(|value| *value != neo_math::K::ZERO) {
            return Err(TerminalR1csError::Unsupported("nonzero running evaluation padding"));
        }
    }
    let assignment_width = neo_reductions::common::superneo_carrier_width(structure.m);
    let expected_point = structure
        .n
        .max(assignment_width)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    require_len("running evaluation point", expected_point, claim.r.len())
}

fn validate_fresh(
    structure: &Structure,
    verifier_rows: usize,
    public_width: usize,
    claim: &CcsClaim,
    witness: &CcsWitness,
) -> Result<(), TerminalR1csError> {
    if claim.adv.is_some() {
        return Err(TerminalR1csError::Unsupported("Nebula fresh commitment sidecars"));
    }
    validate_witness(&witness.Z, structure.m)?;
    validate_commitment(&claim.c, verifier_rows)?;
    require_len("fresh public width", public_width, claim.m_in)?;
    require_len("fresh public values", public_width, claim.x.len())?;
    if claim.x.first() != Some(&F::ONE) {
        return Err(TerminalR1csError::Unsupported("fresh constant-one coordinate"));
    }
    Ok(())
}

fn validate_witness(witness: &WitnessMat, carrier_width: usize) -> Result<(), TerminalR1csError> {
    require_len("witness rows", D, witness.rows())?;
    require_len("witness blocks", carrier_width.div_ceil(D), witness.cols())
}

fn validate_commitment(commitment: &neo_ajtai::Commitment, verifier_rows: usize) -> Result<(), TerminalR1csError> {
    require_len("commitment ring degree", D, commitment.d)?;
    require_len("commitment verifier rows", verifier_rows, commitment.kappa)?;
    require_len("commitment coordinates", verifier_rows * D, commitment.data.len())
}

fn packed_witness(witness: &WitnessMat, carrier_width: usize) -> Vec<F> {
    (0..carrier_width)
        .map(|coordinate| witness[(coordinate % D, coordinate / D)])
        .collect()
}

fn projected_ce_values(claim: &CeClaim, public_width: usize) -> Vec<F> {
    (0..public_width)
        .map(|coordinate| claim.X[(coordinate % D, coordinate / D)])
        .collect()
}

fn alloc_private_vec(builder: &mut R1csBuilder, private_vars: &mut Vec<Var>, values: &[F]) -> Vec<Var> {
    values
        .iter()
        .map(|&value| {
            let variable = builder.alloc(value);
            private_vars.push(variable);
            variable
        })
        .collect()
}

fn alloc_public_vec(builder: &mut R1csBuilder, public_vars: &mut Vec<Var>, values: &[F]) -> Vec<Var> {
    values
        .iter()
        .map(|&value| {
            let variable = builder.alloc(value);
            public_vars.push(variable);
            variable
        })
        .collect()
}

fn alloc_squares(builder: &mut R1csBuilder, private_vars: &mut Vec<Var>, witness: &[Var]) -> Vec<Var> {
    witness
        .iter()
        .map(|&variable| {
            let value = builder.witness()[variable.col()];
            let square = builder.alloc(value * value);
            private_vars.push(square);
            square
        })
        .collect()
}

fn enforce_ajtai(builder: &mut R1csBuilder, rotations: &[Vec<Rotations>], witness: &[Var], commitment: &[Var]) {
    for (commitment_column, row) in rotations.iter().enumerate() {
        for output in 0..D {
            let mut left = Lc::zero();
            for (block, block_rotations) in row.iter().enumerate() {
                for witness_lane in 0..D {
                    left.add_term(witness[block * D + witness_lane], block_rotations[witness_lane][output]);
                }
            }
            builder.enforce_eq(&left, &Lc::from_var(commitment[commitment_column * D + output]));
        }
    }
}

fn enforce_projection(builder: &mut R1csBuilder, witness: &[Var], projection: &[Var]) {
    for (&witness, &public) in witness.iter().zip(projection) {
        builder.enforce_eq(&Lc::from_var(witness), &Lc::from_var(public));
    }
}

fn enforce_norm(builder: &mut R1csBuilder, witness: &[Var], squares: &[Var]) {
    for (&value, &square) in witness.iter().zip(squares) {
        builder.enforce(&Lc::from_var(value), &Lc::from_var(value), &Lc::from_var(square));
        builder.enforce(&Lc::from_var(square), &Lc::from_var(value), &Lc::from_var(value));
    }
}

fn enforce_fixed_evaluations(
    builder: &mut R1csBuilder,
    structure: &Structure,
    claim: &CeClaim,
    witness: &[Var],
    low_claims: &[Var],
    high_claims: &[Var],
) -> Result<(), TerminalR1csError> {
    let forms = build_superneo_ring_forms(structure, &claim.r)
        .map_err(|error| TerminalR1csError::Coefficients(error.to_string()))?;
    for (matrix, matrix_forms) in forms.iter().enumerate() {
        for lane in 0..D {
            let mut low = Lc::zero();
            let mut high = Lc::zero();
            for (coordinate, form) in matrix_forms.iter().enumerate() {
                let [low_coefficient, high_coefficient] = form[lane].as_coeffs();
                low.add_term(witness[coordinate], low_coefficient);
                high.add_term(witness[coordinate], high_coefficient);
            }
            let index = matrix * D + lane;
            builder.enforce_eq(&low, &Lc::from_var(low_claims[index]));
            builder.enforce_eq(&high, &Lc::from_var(high_claims[index]));
        }
    }
    Ok(())
}

fn enforce_fresh_ccs(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    manifest: &LeanNativeCcsManifest,
    step: &super::super::lean_native_ccs_manifest::NativePhi81StepEmission,
    witness: &[Var],
) -> Result<(), TerminalR1csError> {
    let mut indices = HashMap::with_capacity(manifest.terminal_r1cs().logical_width());
    for receipt in &manifest.step_program().receipts {
        for allocation in &receipt.allocations {
            let index = step
                .column_index(&allocation.id)
                .ok_or_else(|| TerminalR1csError::Manifest("missing validated logical column".into()))?;
            indices.insert(allocation.id.clone(), index);
        }
    }
    for receipt in &manifest.step_program().receipts {
        let selector_index = *indices
            .get(&receipt.selector)
            .ok_or_else(|| TerminalR1csError::Manifest("missing validated selector column".into()))?;
        for row in &receipt.rows {
            let a = mapped_combination(&row.a, &indices, witness)?;
            let b = mapped_combination(&row.b, &indices, witness)?;
            let c = mapped_combination(&row.c, &indices, witness)?;
            let residual_value = builder.eval(&a) * builder.eval(&b) - builder.eval(&c);
            let residual = builder.alloc(residual_value);
            private_vars.push(residual);
            let mut lifted_c = c;
            lifted_c.add_term(residual, F::ONE);
            builder.enforce(&a, &b, &lifted_c);
            builder.enforce(
                &Lc::from_var(witness[selector_index]),
                &Lc::from_var(residual),
                &Lc::zero(),
            );
        }
    }
    Ok(())
}

fn enforce_combined_fresh_relation(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    manifest: &LeanNebulaCombinedManifest,
    witness: &[Var],
) -> Result<(), TerminalR1csError> {
    let layout = manifest.combined_layout();
    let mut native_indices = HashMap::with_capacity(layout.native_logical_width);
    for allocation in manifest
        .core()
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.allocations)
    {
        let native_index = native_indices.len();
        if native_indices
            .insert(allocation.id.clone(), native_index)
            .is_some()
        {
            return Err(TerminalR1csError::Manifest(
                "native Step allocates one logical column twice".into(),
            ));
        }
    }
    check_count(
        "native terminal logical columns",
        layout.native_logical_width,
        native_indices.len(),
    )?;

    for receipt in &manifest.core().step_program().receipts {
        let selector_native = *native_indices
            .get(&receipt.selector)
            .ok_or_else(|| TerminalR1csError::Manifest("missing validated selector column".into()))?;
        let selector = witness_var(witness, map_native_index(layout, selector_native))?;
        for row in &receipt.rows {
            let a = mapped_native_combination(&row.a, &native_indices, layout, witness)?;
            let b = mapped_native_combination(&row.b, &native_indices, layout, witness)?;
            let c = mapped_native_combination(&row.c, &native_indices, layout, witness)?;
            let residual_value = builder.eval(&a) * builder.eval(&b) - builder.eval(&c);
            let residual = alloc_private_value(builder, private_vars, residual_value);
            let lifted_c = c.add_scaled(&Lc::from_var(residual), F::ONE);
            builder.enforce(&a, &b, &lifted_c);
            builder.enforce(&Lc::from_var(selector), &Lc::from_var(residual), &Lc::zero());
        }
    }

    for row in manifest.application_rows() {
        enforce_nebula_row(builder, private_vars, row.family(), &row.images, layout, witness)?;
    }
    Ok(())
}

fn mapped_native_combination(
    terms: &[ManifestTerm],
    native_indices: &HashMap<ColumnId, usize>,
    layout: CombinedLayout,
    witness: &[Var],
) -> Result<Lc, TerminalR1csError> {
    let mut combination = Lc::zero();
    for term in terms {
        let native_index = *native_indices
            .get(&term.column)
            .ok_or_else(|| TerminalR1csError::Manifest("row refers to an undeclared native column".into()))?;
        let variable = witness_var(witness, map_native_index(layout, native_index))?;
        combination.add_term(variable, F::from_u64(term.coefficient));
    }
    Ok(combination)
}

fn mapped_nebula_combination(
    terms: &[NebulaTerm],
    layout: CombinedLayout,
    witness: &[Var],
) -> Result<Lc, TerminalR1csError> {
    let mut combination = Lc::zero();
    for term in terms {
        let variable = witness_var(witness, map_nebula_index(layout, term.column))?;
        combination.add_term(variable, F::from_u64(term.coefficient));
    }
    Ok(combination)
}

fn witness_var(witness: &[Var], index: usize) -> Result<Var, TerminalR1csError> {
    witness
        .get(index)
        .copied()
        .ok_or_else(|| TerminalR1csError::Manifest("terminal row column exceeds the combined carrier".into()))
}

fn alloc_private_value(builder: &mut R1csBuilder, private_vars: &mut Vec<Var>, value: F) -> Var {
    let variable = builder.alloc(value);
    private_vars.push(variable);
    variable
}

fn enforce_nebula_row(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    family: NebulaFamily,
    images: &super::super::lean_nebula_combined_manifest::NebulaImages,
    layout: CombinedLayout,
    witness: &[Var],
) -> Result<(), TerminalR1csError> {
    match family {
        NebulaFamily::OperationBit | NebulaFamily::InitialScanBit | NebulaFamily::FinalScanBit => {
            let bit = mapped_nebula_combination(&images.bit, layout, witness)?;
            let one = Lc::from_var(witness_var(witness, 0)?);
            let bit_minus_one = bit.clone().add_scaled(&one, -F::ONE);
            builder.enforce(&bit, &bit_minus_one, &Lc::zero());
        }
        NebulaFamily::ReadWrite
        | NebulaFamily::TimestampOrder
        | NebulaFamily::RomWrite
        | NebulaFamily::RomRange
        | NebulaFamily::Padding => {
            let left = mapped_nebula_combination(&images.product_left, layout, witness)?;
            let right = mapped_nebula_combination(&images.product_right, layout, witness)?;
            builder.enforce(&left, &right, &Lc::zero());
        }
        NebulaFamily::Filler
        | NebulaFamily::OperationCount
        | NebulaFamily::BoundaryTimestamp
        | NebulaFamily::BoundaryProduct => {
            let one = Lc::from_var(witness_var(witness, 0)?);
            let left = mapped_nebula_combination(&images.linear_left, layout, witness)?;
            let right = mapped_nebula_combination(&images.linear_right, layout, witness)?;
            builder.enforce(&one, &left, &right);
        }
        NebulaFamily::ReadProduct
        | NebulaFamily::WriteProduct
        | NebulaFamily::InitialScanProduct
        | NebulaFamily::FinalScanProduct => {
            enforce_nebula_extension(builder, private_vars, images, layout, witness)?;
        }
    }
    Ok(())
}

fn enforce_nebula_extension(
    builder: &mut R1csBuilder,
    private_vars: &mut Vec<Var>,
    images: &super::super::lean_nebula_combined_manifest::NebulaImages,
    layout: CombinedLayout,
    witness: &[Var],
) -> Result<(), TerminalR1csError> {
    let value_a = mapped_nebula_combination(&images.value_a, layout, witness)?;
    let value_b = mapped_nebula_combination(&images.value_b, layout, witness)?;
    let value = mapped_nebula_combination(&images.value, layout, witness)?;
    let extension_a = mapped_nebula_combination(&images.extension_a, layout, witness)?;
    let extension_b = mapped_nebula_combination(&images.extension_b, layout, witness)?;
    let fingerprint_a = mapped_nebula_combination(&images.fingerprint_a, layout, witness)?;
    let fingerprint_b = mapped_nebula_combination(&images.fingerprint_b, layout, witness)?;
    let active = mapped_nebula_combination(&images.active, layout, witness)?;
    let pad = mapped_nebula_combination(&images.pad, layout, witness)?;
    let output = mapped_nebula_combination(&images.output, layout, witness)?;

    let value_a_product = alloc_private_value(builder, private_vars, builder.eval(&value_a) * builder.eval(&value));
    builder.enforce(&value_a, &value, &Lc::from_var(value_a_product));
    let value_b_product = alloc_private_value(builder, private_vars, builder.eval(&value_b) * builder.eval(&value));
    builder.enforce(&value_b, &value, &Lc::from_var(value_b_product));

    let fingerprint_a_minus_product = fingerprint_a.add_scaled(&Lc::from_var(value_a_product), -F::ONE);
    let extension_a_contribution = alloc_private_value(
        builder,
        private_vars,
        builder.eval(&extension_a) * builder.eval(&fingerprint_a_minus_product),
    );
    builder.enforce(
        &extension_a,
        &fingerprint_a_minus_product,
        &Lc::from_var(extension_a_contribution),
    );

    let fingerprint_b_minus_product = fingerprint_b.add_scaled(&Lc::from_var(value_b_product), -F::ONE);
    let extension_b_contribution = alloc_private_value(
        builder,
        private_vars,
        builder.eval(&extension_b) * builder.eval(&fingerprint_b_minus_product),
    );
    builder.enforce(
        &extension_b,
        &fingerprint_b_minus_product,
        &Lc::from_var(extension_b_contribution),
    );

    let contributions =
        Lc::from_var(extension_a_contribution).add_scaled(&Lc::from_var(extension_b_contribution), F::ONE);
    let active_contribution = alloc_private_value(
        builder,
        private_vars,
        builder.eval(&active) * builder.eval(&contributions),
    );
    builder.enforce(&active, &contributions, &Lc::from_var(active_contribution));

    let output_minus_active = output.add_scaled(&Lc::from_var(active_contribution), -F::ONE);
    builder.enforce(&extension_a, &pad, &output_minus_active);
    Ok(())
}

fn mapped_combination(
    terms: &[ManifestTerm],
    indices: &HashMap<ColumnId, usize>,
    witness: &[Var],
) -> Result<Lc, TerminalR1csError> {
    let mut combination = Lc::zero();
    for term in terms {
        let index = *indices
            .get(&term.column)
            .ok_or_else(|| TerminalR1csError::Manifest("row refers to an undeclared logical column".into()))?;
        combination.add_term(witness[index], F::from_u64(term.coefficient));
    }
    Ok(combination)
}

fn column_permutation(
    columns: usize,
    private_vars: &[Var],
    public_vars: &[Var],
) -> Result<Vec<usize>, TerminalR1csError> {
    let mut map = vec![usize::MAX; columns];
    for (index, variable) in private_vars.iter().enumerate() {
        map_column(&mut map, *variable, index)?;
    }
    map_column(&mut map, Var::ONE, private_vars.len())?;
    for (index, variable) in public_vars.iter().enumerate() {
        map_column(&mut map, *variable, private_vars.len() + 1 + index)?;
    }
    if map.iter().any(|&column| column == usize::MAX) {
        return Err(TerminalR1csError::Manifest(
            "terminal compiler allocated an unclassified column".into(),
        ));
    }
    Ok(map)
}

fn map_column(map: &mut [usize], variable: Var, new_column: usize) -> Result<(), TerminalR1csError> {
    let slot = map
        .get_mut(variable.col())
        .ok_or_else(|| TerminalR1csError::Manifest("terminal column exceeds allocation".into()))?;
    if *slot != usize::MAX {
        return Err(TerminalR1csError::Manifest(
            "terminal column has two ownership classes".into(),
        ));
    }
    *slot = new_column;
    Ok(())
}

fn canonical_matrix(
    rows: usize,
    columns: usize,
    triplets: &[(usize, usize, F)],
    permutation: &[usize],
) -> Result<SparseMatrix<SpartanF>, TerminalR1csError> {
    let mut sorted: Vec<_> = triplets
        .iter()
        .map(|&(row, column, value)| (row, permutation[column], value))
        .collect();
    sorted.sort_unstable_by_key(|&(row, column, _)| (row, column));

    let mut canonical: Vec<(usize, usize, F)> = Vec::with_capacity(sorted.len());
    for (row, column, value) in sorted {
        if let Some((last_row, last_column, last_value)) = canonical.last_mut() {
            if *last_row == row && *last_column == column {
                *last_value += value;
                continue;
            }
        }
        canonical.push((row, column, value));
    }
    canonical.retain(|entry| entry.2 != F::ZERO);

    let mut data = Vec::with_capacity(canonical.len());
    let mut indices = Vec::with_capacity(canonical.len());
    let mut indptr = vec![0usize; rows + 1];
    let mut cursor = 0usize;
    for row in 0..rows {
        while cursor < canonical.len() && canonical[cursor].0 == row {
            indices.push(canonical[cursor].1);
            data.push(to_spartan(canonical[cursor].2));
            cursor += 1;
        }
        indptr[row + 1] = cursor;
    }
    if cursor != canonical.len() {
        return Err(TerminalR1csError::Manifest(
            "terminal matrix row exceeds relation".into(),
        ));
    }
    SparseMatrix::from_csr(rows, columns, data, indices, indptr)
        .map_err(|error| TerminalR1csError::Spartan(error.to_string()))
}

fn to_spartan(value: F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

fn require_len(what: &'static str, expected: usize, got: usize) -> Result<(), TerminalR1csError> {
    if expected != got {
        return Err(TerminalR1csError::Shape { what, expected, got });
    }
    Ok(())
}

fn check_count(what: &'static str, expected: usize, got: usize) -> Result<(), TerminalR1csError> {
    require_len(what, expected, got)
}
