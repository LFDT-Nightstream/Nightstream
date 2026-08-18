//! Drift gate for the Nightstream k_rho=16 strict-PiDEC canonical-X receipt.
//!
//! This exporter drives the live `enforce_dec_v_strict` emitter directly on
//! the fixed `54 x 5`, sixteen-child public carrier. It never constructs or
//! scans the fixed-point private assignment domain.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::engine::r1cs_circuit::{
    CanonicalSparseRow, PiDecCanonicalXReceipt, PiDecCanonicalXRowOwner, R1csBuilder,
};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::reductions::pi_dec;
use neo_fold_clean::paper::reductions::pi_dec_circuit::{alloc_dec_inputs, enforce_dec_v_strict};
use neo_fold_clean::{config, preprocess, CcsInstance, Params, Preprocessing};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const BINARY_GENERATED_DIRECTORY: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Nifs/PiDecCanonicalX/Generated";
const RADIX_FOUR_GENERATED_DIRECTORY: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Nifs/PiDecCanonicalX/RadixFourCandidate/Generated";
const SCHEMA_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX";
const COORDINATE_SHARD_SIZE: usize = 100;
const ROW_SHARD_SIZE: usize = 240;
const X_ROWS: usize = 54;
const ACTIVE_COLUMNS: usize = 5;
const LOGICAL_COORDINATES: usize = X_ROWS * ACTIVE_COLUMNS;
const RECOMPOSITION_ROWS: usize = LOGICAL_COORDINATES;
const APPLICATION_MATRIX_COUNT: usize = 13;
const PAPER_MATRIX_COUNT: usize = APPLICATION_MATRIX_COUNT + 1;
const RECURSIVE_SELECTOR: u64 = 1;

#[derive(Clone, Copy)]
struct ArtifactProfile {
    generated_directory: &'static str,
    generated_namespace: &'static str,
    radix: u32,
    child_count: usize,
    profile_tag: usize,
}

impl ArtifactProfile {
    fn canonicality_rows(self) -> usize {
        let rows_per_child = if self.radix == 2 { 1 } else { 3 };
        LOGICAL_COORDINATES * (2 + rows_per_child * self.child_count)
    }

    fn total_rows(self) -> usize {
        RECOMPOSITION_ROWS + self.canonicality_rows()
    }

    fn canonical_columns(self) -> usize {
        let limbs = if self.radix == 2 { 0 } else { 2 * self.child_count };
        1 + LOGICAL_COORDINATES * (1 + self.child_count + 2 + limbs)
    }
}

const BINARY_PROFILE: ArtifactProfile = ArtifactProfile {
    generated_directory: BINARY_GENERATED_DIRECTORY,
    generated_namespace:
        "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated",
    radix: 2,
    child_count: 16,
    profile_tag: 0,
};

const RADIX_FOUR_PROFILE: ArtifactProfile = ArtifactProfile {
    generated_directory: RADIX_FOUR_GENERATED_DIRECTORY,
    generated_namespace:
        "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate.Generated",
    radix: 4,
    child_count: 7,
    profile_tag: 1,
};

struct GeneratedLeanFile {
    relative_path: String,
    contents: String,
}

#[derive(Clone)]
struct CoordinateColumns {
    parent: usize,
    children: Vec<usize>,
    sign: usize,
    product: usize,
    limbs: Vec<[usize; 2]>,
}

#[derive(Clone)]
struct PhysicalRow {
    relative_index: usize,
    physical_index: usize,
    owner: PiDecCanonicalXRowOwner,
    row: CanonicalSparseRow,
}

struct Artifact {
    strict_rows: std::ops::Range<usize>,
    recomposition_rows: std::ops::Range<usize>,
    canonicality_rows: std::ops::Range<usize>,
    coordinates: Vec<CoordinateColumns>,
    rows: Vec<PhysicalRow>,
    differential_cases: Vec<DifferentialCase>,
}

#[derive(Clone)]
struct DifferentialCase {
    case_id: usize,
    profile_tag: usize,
    recursive_selector: u64,
    public_column: usize,
    parent: F,
    children: Vec<F>,
    child_evaluation_arities: Vec<usize>,
    rust_accepted: bool,
}

struct ProductionFixture {
    builder: R1csBuilder,
    receipt: PiDecCanonicalXReceipt,
    parent_values: Vec<F>,
    child_values: Vec<Vec<F>>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn compare_or_write_expected(root: &Path, file: GeneratedLeanFile, drifted: &mut Vec<String>) {
    let path = root.join(&file.relative_path);
    if fs::read_to_string(&path).unwrap_or_default() == file.contents {
        return;
    }
    let expected = path.with_extension("lean.expected");
    fs::create_dir_all(expected.parent().expect("generated canonical-X parent"))
        .expect("create generated canonical-X directory");
    fs::write(&expected, file.contents).expect("write generated canonical-X candidate");
    drifted.push(
        expected
            .strip_prefix(root)
            .unwrap_or(&expected)
            .display()
            .to_string(),
    );
}

fn committed_lean_files(directory: &Path, root: &Path, files: &mut BTreeSet<String>) {
    if !directory.exists() {
        return;
    }
    for entry in fs::read_dir(directory).expect("read generated canonical-X directory") {
        let path = entry.expect("read generated canonical-X entry").path();
        if path.is_dir() {
            committed_lean_files(&path, root, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "lean")
        {
            files.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
}

fn compact_preprocessing(profile: ArtifactProfile) -> Preprocessing {
    let matrices = (0..APPLICATION_MATRIX_COUNT)
        .map(|_| Mat::identity(LOGICAL_COORDINATES))
        .collect::<Vec<_>>();
    let structure = CcsStructure::new(matrices, SparsePoly::new(APPLICATION_MATRIX_COUNT, vec![]))
        .expect("compact 270-coordinate paper-shape structure");
    let base = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("compact paper-shape params");
    let params = if profile.radix == 2 {
        base
    } else {
        Params::test_only_from_neo_params(
            neo_params::NeoParams::new(
                base.q(),
                base.eta(),
                base.d(),
                base.kappa(),
                base.m(),
                profile.radix,
                profile.child_count as u32,
                base.T(),
                base.extension_degree(),
                114,
            )
            .expect("radix-four compact artifact params"),
        )
    };
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(LOGICAL_COORDINATES)).expect("compact paper-shape preprocessing")
}

fn compact_assignment() -> Vec<F> {
    (0..LOGICAL_COORDINATES)
        .map(|column| match column % 5 {
            0 | 3 => F::ONE,
            1 => F::ZERO - F::ONE,
            _ => F::ZERO,
        })
        .collect()
}

fn drive_nifs(profile: ArtifactProfile) -> (Preprocessing, RunningInstance, nifs::NifsProof) {
    let prep = compact_preprocessing(profile);
    let fresh = vec![CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &compact_assignment(),
        LOGICAL_COORDINATES,
    )
    .expect("compact paper-shape low-norm instance")];
    let running = RunningInstance::default();
    let mut transcript = Transcript::session();
    let (next, proof) = nifs::prove(
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P for canonical-X artifact");
    (prep, next, proof)
}

fn production_fixture(profile: ArtifactProfile) -> ProductionFixture {
    let (prep, next, proof) = drive_nifs(profile);
    let parent = proof.pi_rlc.combined;
    let children = proof.pi_dec.children;
    assert_eq!(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, LOGICAL_COORDINATES);
    assert_eq!(next.witnesses.len(), profile.child_count);
    assert_eq!(children.len(), profile.child_count);
    assert_eq!(parent.m_in, LOGICAL_COORDINATES);
    assert_eq!(parent.X.rows(), D);
    assert_eq!(parent.X.cols(), ACTIVE_COLUMNS);

    let mut child_values = vec![vec![F::ZERO; LOGICAL_COORDINATES]; profile.child_count];
    for (child_index, ((claim, witness), values)) in children
        .iter()
        .zip(&next.witnesses)
        .zip(&mut child_values)
        .enumerate()
    {
        assert_eq!(claim.m_in, LOGICAL_COORDINATES);
        assert_eq!(
            claim.m_in % D,
            0,
            "production public input must contain whole ring elements"
        );
        assert_eq!(claim.y_ring.len(), PAPER_MATRIX_COUNT);
        assert_eq!(witness.rows(), D);
        assert_eq!(witness.cols(), LOGICAL_COORDINATES / D);
        for (public_column, value) in values.iter_mut().enumerate() {
            let row = public_column % D;
            let column = public_column / D;
            *value = witness[(row, column)];
            assert_eq!(
                claim.X[(row, column)],
                *value,
                "child {child_index} public coordinate {public_column} must come from its raw WitnessMat"
            );
        }
    }

    let mut parent_values = vec![F::ZERO; LOGICAL_COORDINATES];
    for public_column in 0..LOGICAL_COORDINATES {
        let mut power = F::ONE;
        for values in &child_values {
            parent_values[public_column] += power * values[public_column];
            power *= F::from_u64(profile.radix as u64);
        }
        let row = public_column % D;
        let column = public_column / D;
        assert_eq!(
            parent.X[(row, column)],
            parent_values[public_column],
            "parent coordinate {public_column} must be the raw-child radix recomposition"
        );
    }
    assert_eq!(parent.y_ring.len(), PAPER_MATRIX_COUNT);
    pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof {
            children: children.clone(),
        },
    )
    .expect("native PiDEC verifier accepts the compact raw-WitnessMat fixture");

    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    let receipt =
        enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("live strict PiDEC canonical-X receipt");
    assert!(builder.is_satisfied(), "production-shaped canonical-X fixture");
    ProductionFixture {
        builder,
        receipt,
        parent_values,
        child_values,
    }
}

fn collect_artifact(profile: ArtifactProfile) -> Artifact {
    let ProductionFixture {
        builder,
        receipt,
        parent_values,
        child_values,
    } = production_fixture(profile);
    let snapshot = builder.snapshot();
    let program = receipt.program();
    let plan = program.plan();
    assert_eq!(
        (plan.x_rows(), plan.active_columns(), plan.child_count()),
        (X_ROWS, ACTIVE_COLUMNS, profile.child_count)
    );
    assert_eq!(plan.radix(), profile.radix);
    assert_eq!(plan.logical_coordinates(), LOGICAL_COORDINATES);
    assert_eq!(plan.recomposition_rows(), RECOMPOSITION_ROWS);
    assert_eq!(plan.canonicality_rows(), profile.canonicality_rows());
    assert_eq!(program.row_count(), profile.total_rows());
    assert_eq!(plan.canonical_column_count(), profile.canonical_columns());

    let coordinates = (0..LOGICAL_COORDINATES)
        .map(|active_index| CoordinateColumns {
            parent: receipt
                .columns()
                .actual_column(
                    program
                        .parent_canonical_column(active_index)
                        .expect("parent canonical column"),
                )
                .expect("parent actual column"),
            children: (0..profile.child_count)
                .map(|child| {
                    receipt
                        .columns()
                        .actual_column(
                            program
                                .child_canonical_column(child, active_index)
                                .expect("child canonical column"),
                        )
                        .expect("child actual column")
                })
                .collect(),
            sign: receipt
                .columns()
                .actual_column(
                    program
                        .sign_canonical_column(active_index)
                        .expect("sign canonical column"),
                )
                .expect("sign actual column"),
            product: receipt
                .columns()
                .actual_column(
                    program
                        .product_canonical_column(active_index)
                        .expect("product canonical column"),
                )
                .expect("product actual column"),
            limbs: if plan.limbs_per_child() == 0 {
                Vec::new()
            } else {
                (0..profile.child_count)
                    .map(|child| {
                        core::array::from_fn(|limb| {
                            receipt
                                .columns()
                                .actual_column(
                                    program
                                        .limb_canonical_column(child, limb, active_index)
                                        .expect("limb canonical column"),
                                )
                                .expect("limb actual column")
                        })
                    })
                    .collect()
            },
        })
        .collect::<Vec<_>>();

    let mut reconstructed_columns = vec![0];
    reconstructed_columns.extend(coordinates.iter().map(|coordinate| coordinate.parent));
    for child in 0..profile.child_count {
        reconstructed_columns.extend(
            coordinates
                .iter()
                .map(|coordinate| coordinate.children[child]),
        );
    }
    reconstructed_columns.extend(
        coordinates
            .iter()
            .flat_map(|coordinate| [coordinate.sign, coordinate.product]),
    );
    for child in 0..profile.child_count {
        for limb in 0..program.plan().limbs_per_child() {
            reconstructed_columns.extend(
                coordinates
                    .iter()
                    .map(|coordinate| coordinate.limbs[child][limb]),
            );
        }
    }
    assert_eq!(
        reconstructed_columns,
        receipt.columns().canonical_to_actual(),
        "coordinate records must reconstruct the exact canonical-to-actual map"
    );

    let rows = (0..profile.total_rows())
        .map(|relative_index| {
            let physical_index = receipt
                .physical_row(relative_index)
                .expect("every canonical-X row has one physical owner");
            let row = CanonicalSparseRow {
                a: snapshot.a_row(physical_index).to_vec(),
                b: snapshot.b_row(physical_index).to_vec(),
                c: snapshot.c_row(physical_index).to_vec(),
            };
            assert_eq!(
                receipt.actual_row_at(relative_index),
                Some(row.clone()),
                "indexed compiler must equal the live physical row"
            );
            PhysicalRow {
                relative_index,
                physical_index,
                owner: program.owner(relative_index).expect("indexed owner"),
                row,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows.iter()
            .map(|row| row.physical_index)
            .collect::<BTreeSet<_>>()
            .len(),
        profile.total_rows(),
        "physical canonical-X rows must be uniquely owned"
    );

    Artifact {
        strict_rows: receipt.strict_rows(),
        recomposition_rows: receipt.recomposition_rows(),
        canonicality_rows: receipt.canonicality_rows(),
        coordinates,
        rows,
        differential_cases: differential_cases(profile, &parent_values, &child_values),
    }
}

fn native_case_accepts(profile: ArtifactProfile, case: &DifferentialCase) -> bool {
    if case.profile_tag != profile.profile_tag
        || case.recursive_selector != RECURSIVE_SELECTOR
        || case.public_column >= LOGICAL_COORDINATES
        || case.children.len() != profile.child_count
        || case.child_evaluation_arities != vec![PAPER_MATRIX_COUNT; profile.child_count]
    {
        return false;
    }
    let mut parent = Mat::zero(D, 1, F::ZERO);
    parent.set(0, 0, case.parent);
    let Ok(expected) = neo_reductions::common::split_b_matrix_k(&parent, profile.child_count, profile.radix) else {
        return false;
    };
    expected
        .iter()
        .zip(&case.children)
        .all(|(matrix, child)| matrix[(0, 0)] == *child)
}

fn checked_case(
    profile: ArtifactProfile,
    case_id: usize,
    profile_tag: usize,
    recursive_selector: u64,
    public_column: usize,
    parent: F,
    children: Vec<F>,
    child_evaluation_arities: Vec<usize>,
) -> DifferentialCase {
    let mut case = DifferentialCase {
        case_id,
        profile_tag,
        recursive_selector,
        public_column,
        parent,
        children,
        child_evaluation_arities,
        rust_accepted: false,
    };
    case.rust_accepted = native_case_accepts(profile, &case);
    case
}

fn differential_cases(profile: ArtifactProfile, parent_values: &[F], child_values: &[Vec<F>]) -> Vec<DifferentialCase> {
    assert_eq!(parent_values.len(), LOGICAL_COORDINATES);
    assert_eq!(child_values.len(), profile.child_count);
    assert!(child_values
        .iter()
        .all(|values| values.len() == LOGICAL_COORDINATES));
    let public_column = (0..LOGICAL_COORDINATES)
        .find(|&column| {
            (0..profile.child_count).any(|left| {
                (left + 1..profile.child_count).any(|right| child_values[left][column] != child_values[right][column])
            })
        })
        .expect("raw fixture must expose a nonconstant canonical digit vector");
    let raw_children = child_values
        .iter()
        .map(|values| values[public_column])
        .collect::<Vec<_>>();
    let arities = vec![PAPER_MATRIX_COUNT; profile.child_count];
    let honest = checked_case(
        profile,
        0,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        raw_children.clone(),
        arities.clone(),
    );
    assert!(
        honest.rust_accepted,
        "raw WitnessMat coordinate must pass native split_b"
    );

    let mut cases = vec![honest];
    let last = checked_case(
        profile,
        1,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        LOGICAL_COORDINATES - 1,
        parent_values[LOGICAL_COORDINATES - 1],
        child_values
            .iter()
            .map(|values| values[LOGICAL_COORDINATES - 1])
            .collect(),
        arities.clone(),
    );
    assert!(
        last.rust_accepted,
        "last raw WitnessMat coordinate must pass native split_b"
    );
    cases.push(last);

    let mut changed_child = raw_children.clone();
    changed_child[0] += F::ONE;
    cases.push(checked_case(
        profile,
        2,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        changed_child,
        arities.clone(),
    ));

    let (left, right) = (0..profile.child_count)
        .find_map(|left| {
            (left + 1..profile.child_count)
                .find(|&right| raw_children[left] != raw_children[right])
                .map(|right| (left, right))
        })
        .expect("selected raw coordinate has two distinct child digits");
    let mut reordered = raw_children.clone();
    reordered.swap(left, right);
    cases.push(checked_case(
        profile,
        3,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        reordered,
        arities.clone(),
    ));

    cases.push(checked_case(
        profile,
        4,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column] + F::ONE,
        raw_children.clone(),
        arities.clone(),
    ));

    let mut short = raw_children.clone();
    short.pop();
    cases.push(checked_case(
        profile,
        5,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        short,
        arities.clone(),
    ));

    let mut mixed_sign = vec![F::ZERO; profile.child_count];
    mixed_sign[0] = F::ZERO - F::from_u64((profile.radix - 1) as u64);
    mixed_sign[1] = F::ONE;
    cases.push(checked_case(
        profile,
        6,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        0,
        F::ONE,
        mixed_sign,
        arities.clone(),
    ));

    cases.push(checked_case(
        profile,
        7,
        profile.profile_tag + 1,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        raw_children.clone(),
        arities.clone(),
    ));
    cases.push(checked_case(
        profile,
        8,
        profile.profile_tag,
        0,
        public_column,
        parent_values[public_column],
        raw_children.clone(),
        arities.clone(),
    ));
    let mut wrong_arity = arities.clone();
    wrong_arity[0] += 1;
    cases.push(checked_case(
        profile,
        9,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        public_column,
        parent_values[public_column],
        raw_children.clone(),
        wrong_arity,
    ));
    cases.push(checked_case(
        profile,
        10,
        profile.profile_tag,
        RECURSIVE_SELECTOR,
        LOGICAL_COORDINATES,
        parent_values[public_column],
        raw_children,
        arities,
    ));
    assert!(cases.iter().take(2).all(|case| case.rust_accepted));
    assert!(cases.iter().skip(2).all(|case| !case.rust_accepted));
    cases
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_field_list(values: &[F]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| value.as_canonical_u64().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_owner(owner: PiDecCanonicalXRowOwner) -> String {
    match owner {
        PiDecCanonicalXRowOwner::Recomposition { active_index } => {
            format!(".recomposition {active_index}")
        }
        PiDecCanonicalXRowOwner::SignProduct { active_index } => {
            format!(".signProduct {active_index}")
        }
        PiDecCanonicalXRowOwner::SignZero { active_index } => {
            format!(".signZero {active_index}")
        }
        PiDecCanonicalXRowOwner::ChildDigit { active_index, child } => {
            format!(".childDigit {active_index} {child}")
        }
        PiDecCanonicalXRowOwner::RadixFourLimb {
            active_index,
            child,
            limb,
        } => format!(".radixFourLimb {active_index} {child} {limb}"),
        PiDecCanonicalXRowOwner::RadixFourReconstruction { active_index, child } => {
            format!(".radixFourReconstruction {active_index} {child}")
        }
    }
}

fn render_coordinate_chunk(profile: ArtifactProfile, index: usize, coordinates: &[CoordinateColumns]) -> String {
    let values = coordinates
        .iter()
        .map(|coordinate| {
            let limbs = format!(
                "[{}]",
                coordinate
                    .limbs
                    .iter()
                    .map(|pair| lean_nat_list(pair.iter().copied()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            format!(
                "{{ parent := {}, children := {}, sign := {}, product := {}, limbs := {} }}",
                coordinate.parent,
                lean_nat_list(coordinate.children.iter().copied()),
                coordinate.sign,
                coordinate.product,
                limbs,
            )
        })
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Schema\n\n\
         /-! Generated exact PiDEC canonical-X columns. Do not hand-edit. -/\n\n\
         namespace {namespace}.Coordinates.Chunk{index}\n\n\
         open {SCHEMA_NAMESPACE}\n\n\
         set_option maxRecDepth 100000 in\n\
         def values : List CoordinateColumns := [\n  {values}\n]\n\n\
         end {namespace}.Coordinates.Chunk{index}\n",
        namespace = profile.generated_namespace,
    )
}

fn render_row_chunk(profile: ArtifactProfile, index: usize, rows: &[PhysicalRow]) -> String {
    let values = rows
        .iter()
        .map(|row| {
            format!(
                "{{ relativeIndex := {}, physicalIndex := {}, owner := {}, row := ⟨{}, {}, {}⟩ }}",
                row.relative_index,
                row.physical_index,
                lean_owner(row.owner),
                lean_terms(&row.row.a),
                lean_terms(&row.row.b),
                lean_terms(&row.row.c)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Schema\n\n\
         /-! Generated exact PiDEC canonical-X rows. Do not hand-edit. -/\n\n\
         namespace {namespace}.Rows.Chunk{index}\n\n\
         open {SCHEMA_NAMESPACE}\n\n\
         set_option maxRecDepth 100000 in\n\
         def values : List PhysicalRow := [\n  {values}\n]\n\n\
         end {namespace}.Rows.Chunk{index}\n",
        namespace = profile.generated_namespace,
    )
}

fn row_ranges(profile: ArtifactProfile) -> Vec<std::ops::Range<usize>> {
    let mut ranges = vec![0..100, 100..200, 200..RECOMPOSITION_ROWS];
    let mut start = RECOMPOSITION_ROWS;
    while start < profile.total_rows() {
        let end = (start + ROW_SHARD_SIZE).min(profile.total_rows());
        ranges.push(start..end);
        start = end;
    }
    assert_eq!(ranges.last().expect("row range").end, profile.total_rows());
    assert!(ranges.iter().all(|range| range.len() <= ROW_SHARD_SIZE));
    ranges
}

fn render_metadata(profile: ArtifactProfile, artifact: &Artifact, row_counts: &[usize]) -> String {
    let coordinate_counts = (0..LOGICAL_COORDINATES)
        .step_by(COORDINATE_SHARD_SIZE)
        .map(|start| COORDINATE_SHARD_SIZE.min(LOGICAL_COORDINATES - start));
    format!(
        "/-! Generated exact strict-PiDEC canonical-X metadata. Do not hand-edit. -/\n\n\
         namespace {namespace}.Metadata\n\n\
         def schemaVersion : Nat := 1\n\
         def radix : Nat := {}\n\
         def xRows : Nat := {X_ROWS}\n\
         def activeColumns : Nat := {ACTIVE_COLUMNS}\n\
         def childCount : Nat := {}\n\
         def logicalCoordinates : Nat := {LOGICAL_COORDINATES}\n\
         def canonicalColumnCount : Nat := {}\n\
         def strictRowStart : Nat := {}\n\
         def strictRowEnd : Nat := {}\n\
         def recompositionRowStart : Nat := {}\n\
         def recompositionRowEnd : Nat := {}\n\
         def canonicalityRowStart : Nat := {}\n\
         def canonicalityRowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def coordinateShardSize : Nat := {COORDINATE_SHARD_SIZE}\n\
         def rowShardSize : Nat := {ROW_SHARD_SIZE}\n\
         def coordinateChunkCounts : List Nat := {}\n\
         def rowChunkCounts : List Nat := {}\n\n\
         end {namespace}.Metadata\n",
        profile.radix,
        profile.child_count,
        profile.canonical_columns(),
        artifact.strict_rows.start,
        artifact.strict_rows.end,
        artifact.recomposition_rows.start,
        artifact.recomposition_rows.end,
        artifact.canonicality_rows.start,
        artifact.canonicality_rows.end,
        profile.total_rows(),
        lean_nat_list(coordinate_counts),
        lean_nat_list(row_counts.iter().copied()),
        namespace = profile.generated_namespace,
    )
}

fn render_differential_cases(profile: ArtifactProfile, cases: &[DifferentialCase]) -> String {
    let values = cases
        .iter()
        .map(|case| {
            format!(
                "{{ caseId := {}, profileTag := {}, recursiveSelector := {}, publicColumn := {}, parent := {}, children := {}, childEvaluationArities := {}, rustAccepted := {} }}",
                case.case_id,
                case.profile_tag,
                case.recursive_selector,
                case.public_column,
                case.parent.as_canonical_u64(),
                lean_field_list(&case.children),
                lean_nat_list(case.child_evaluation_arities.iter().copied()),
                case.rust_accepted,
            )
        })
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Schema\n\n\
         /-! Generated compact Rust/Lean PiDEC-X differential cases. Do not hand-edit. -/\n\n\
         namespace {namespace}.DifferentialCases\n\n\
         open {SCHEMA_NAMESPACE}\n\n\
         def values : List DifferentialCase := [\n  {values}\n]\n\n\
         end {namespace}.DifferentialCases\n",
        namespace = profile.generated_namespace,
    )
}

fn render_aggregate(
    profile: ArtifactProfile,
    kind: &str,
    shard_count: usize,
    value_type: &str,
    value_name: &str,
) -> String {
    let imports = (0..shard_count)
        .map(|index| {
            format!(
                "import {namespace}.{kind}.Chunk{index}",
                namespace = profile.generated_namespace,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let chunks = (0..shard_count)
        .map(|index| format!("{kind}.Chunk{index}.values"))
        .collect::<Vec<_>>()
        .join(",\n      ");
    format!(
        "{imports}\n\n\
         /-! Generated ordered PiDEC canonical-X {kind}. Do not hand-edit. -/\n\n\
         namespace {namespace}\n\n\
         open {SCHEMA_NAMESPACE}\n\n\
         def {value_name} : List {value_type} :=\n\
           [{chunks}].flatten\n\n\
         end {namespace}\n",
        namespace = profile.generated_namespace,
    )
}

fn generated_files(profile: ArtifactProfile) -> Vec<GeneratedLeanFile> {
    let artifact = collect_artifact(profile);
    let row_ranges = row_ranges(profile);
    let row_counts = row_ranges
        .iter()
        .map(std::ops::Range::len)
        .collect::<Vec<_>>();
    let mut files = vec![
        GeneratedLeanFile {
            relative_path: format!("{}/Metadata.lean", profile.generated_directory),
            contents: render_metadata(profile, &artifact, &row_counts),
        },
        GeneratedLeanFile {
            relative_path: format!("{}/DifferentialCases.lean", profile.generated_directory),
            contents: render_differential_cases(profile, &artifact.differential_cases),
        },
    ];
    files.extend(
        artifact
            .coordinates
            .chunks(COORDINATE_SHARD_SIZE)
            .enumerate()
            .map(|(index, coordinates)| GeneratedLeanFile {
                relative_path: format!("{}/Coordinates/Chunk{index}.lean", profile.generated_directory),
                contents: render_coordinate_chunk(profile, index, coordinates),
            }),
    );
    files.push(GeneratedLeanFile {
        relative_path: format!("{}/Coordinates.lean", profile.generated_directory),
        contents: render_aggregate(
            profile,
            "Coordinates",
            LOGICAL_COORDINATES.div_ceil(COORDINATE_SHARD_SIZE),
            "CoordinateColumns",
            "coordinates",
        ),
    });
    files.extend(
        row_ranges
            .iter()
            .enumerate()
            .map(|(index, range)| GeneratedLeanFile {
                relative_path: format!("{}/Rows/Chunk{index}.lean", profile.generated_directory),
                contents: render_row_chunk(profile, index, &artifact.rows[range.clone()]),
            }),
    );
    files.push(GeneratedLeanFile {
        relative_path: format!("{}/Rows.lean", profile.generated_directory),
        contents: render_aggregate(profile, "Rows", row_ranges.len(), "PhysicalRow", "rows"),
    });
    files
}

#[test]
fn production_pi_dec_canonical_x_artifact_matches_live_rows() {
    check_generated_profile(BINARY_PROFILE);
}

#[test]
fn radix_four_candidate_pi_dec_canonical_x_artifact_matches_live_rows() {
    check_generated_profile(RADIX_FOUR_PROFILE);
}

fn check_generated_profile(profile: ArtifactProfile) {
    let files = generated_files(profile);
    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    let mut drifted = Vec::new();
    for file in files {
        compare_or_write_expected(&root, file, &mut drifted);
    }
    let mut committed_paths = BTreeSet::new();
    committed_lean_files(&root.join(profile.generated_directory), &root, &mut committed_paths);
    for stale in committed_paths.difference(&expected_paths) {
        drifted.push(format!("stale generated module: {stale}"));
    }
    assert!(
        drifted.is_empty(),
        "PiDEC canonical-X artifact drifted; inspect every generated `.lean.expected` candidate before promotion: {drifted:?}"
    );
}

#[test]
#[ignore = "deliberately promotes reviewed generated Lean candidates"]
fn regenerate_production_pi_dec_canonical_x_artifact() {
    let root = repo_root();
    for profile in [BINARY_PROFILE, RADIX_FOUR_PROFILE] {
        for file in generated_files(profile) {
            let path = root.join(&file.relative_path);
            fs::create_dir_all(path.parent().expect("generated canonical-X parent"))
                .expect("create generated canonical-X directory");
            fs::write(&path, file.contents).expect("write generated canonical-X artifact");
            let expected = path.with_extension("lean.expected");
            if expected.exists() {
                fs::remove_file(expected).expect("remove promoted canonical-X candidate");
            }
        }
    }
}
