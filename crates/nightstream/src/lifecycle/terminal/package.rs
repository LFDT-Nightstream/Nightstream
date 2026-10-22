//! The Lean terminal package in the final program
//! (`formal/compression-lean`): the F′ rows that check the last fold before
//! Π_DEC.
//!
//! Owns: the pinned artifact, its rows as one `R1cs` block, the wire of every
//! column the rows name, and the layer-1 claim and transcript read from the
//! output columns. Does not own the rows or their meaning (Lean proves them)
//! nor the input values (`encode_terminal_inputs`).
//!
//! Wires: the constant column is one; the domain, context and statement words
//! of the prior preimage and the public context words are constants or
//! statement words; every other input is a proof word; the columns the
//! package computes come from one hint, which the rows then constrain.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_ccs::SparsePoly;
use neo_math::{D, F};
use neo_spartan::{Backend, ClaimWords, Error, FoldTranscript, R1cs};
use nightstream_fprime::{
    load_terminal_package, LoadedPackage, PackageError, PackageSparseMatrix, PiCcsV1_2EncodedInputs, TerminalOutputs,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_STATE_PREIMAGE_WORDS,
};
use p3_field::PrimeField64;

use super::super::inputs::{STATE_CONTEXT_START, STATE_DOMAIN_CHUNK, STATE_STATEMENT_WORDS};

const ARTIFACT: &[u8] = include_bytes!("../../../artifacts/compression-terminal-v1.json");
// Without its Git LFS object, the file is the pointer text: fail the build,
// not every verification.
const _: () = assert!(
    !ARTIFACT.is_empty() && ARTIFACT[0] == b'[',
    "compression-terminal-v1.json is not the package: run `git lfs pull`"
);

/// The identifier of the embedded artifact.
pub(crate) const IDENTIFIER: [u64; 4] = [
    15_606_775_949_683_203_332,
    10_806_517_727_170_708_686,
    18_361_906_505_571_982_029,
    3_191_860_797_915_537_918,
];

/// The public inputs: the prior public input, the output digest, the context.
const PUBLIC_CONTEXT_START: usize = PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS + 4;

/// Where the value of one system column comes from.
#[derive(Clone, Copy)]
enum Source {
    One,
    /// Private input `index` of `execute_witness`.
    Private(usize),
    /// Public input `index`.
    Public(usize),
    /// Hint output `index`.
    Computed(usize),
}

pub(crate) struct Package {
    loaded: LoadedPackage,
    system: R1cs,
    sources: Vec<Source>,
    /// The package column of each computed value, in hint order.
    computed: Vec<usize>,
    /// The output map in system columns.
    outputs: TerminalOutputs,
}

/// Start loading the package on another thread (about 5 s, once per
/// process). A later `package()` waits for it.
pub(crate) fn preload() {
    std::thread::spawn(|| {
        let _ = package();
    });
}

/// The embedded package does not load; the text is the loader's reason.
#[derive(Clone, Debug, thiserror::Error)]
#[error("the terminal package does not load: {0}")]
pub struct TerminalPackageError(String);

impl From<PackageError> for TerminalPackageError {
    fn from(error: PackageError) -> Self {
        Self(error.to_string())
    }
}

/// The embedded package, loaded once.
pub(crate) fn package() -> Result<&'static Package, TerminalPackageError> {
    static PACKAGE: OnceLock<Result<Package, TerminalPackageError>> = OnceLock::new();
    PACKAGE.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn load() -> Result<Package, TerminalPackageError> {
    let (loaded, outputs) = load_terminal_package(ARTIFACT, IDENTIFIER)?;
    // `apply` reads the first caller inputs as the prior state preimage.
    if loaded.prior_preimage_inputs() != Some(0..PI_CCS_V1_2_STATE_PREIMAGE_WORDS) {
        return Err(TerminalPackageError(
            "the prior state preimage is not the first caller input".into(),
        ));
    }
    let matrices = loaded.ordered_r1cs()?;
    let sides = [matrices.a(), matrices.b(), matrices.c()];
    let total = loaded.total_column_count();
    let mut used = vec![false; total];
    for matrix in sides {
        for &column in matrix.column_indices() {
            used[column] = true;
        }
    }
    let constant = loaded.private_column_count();
    let mut index = vec![u32::MAX; total];
    let mut sources = Vec::new();
    let mut computed = Vec::new();
    for column in (0..total).filter(|&column| used[column]) {
        index[column] = sources.len() as u32;
        sources.push(if column == constant {
            Source::One
        } else if column > constant {
            Source::Public(column - constant - 1)
        } else if let Some(input) = loaded.private_input_index(column) {
            Source::Private(input)
        } else {
            computed.push(column);
            Source::Computed(computed.len() - 1)
        });
    }
    let side = |matrix: &PackageSparseMatrix, row: usize| -> Vec<(usize, u64)> {
        let offsets = matrix.row_offsets();
        (offsets[row]..offsets[row + 1])
            .map(|entry| (index[matrix.column_indices()[entry]] as usize, matrix.values()[entry]))
            .collect()
    };
    let mut system = R1cs::new(sources.len());
    for row in 0..matrices.a().rows() {
        system.push(&side(sides[0], row), &side(sides[1], row), &side(sides[2], row));
    }
    let map = |columns: &[usize]| -> Result<Vec<usize>, TerminalPackageError> {
        columns
            .iter()
            .map(|&column| match index[column] {
                u32::MAX => Err(TerminalPackageError("terminal output column without a row".into())),
                system => Ok(system as usize),
            })
            .collect()
    };
    let outputs = TerminalOutputs {
        commitment: map(&outputs.commitment)?,
        public_input: map(&outputs.public_input)?,
        point: map(&outputs.point)?,
        eval_k: map(&outputs.eval_k)?,
        eval_a: map(&outputs.eval_a)?,
        state: map(&outputs.state)?,
    };
    Ok(Package {
        loaded,
        system,
        sources,
        computed,
        outputs,
    })
}

impl Package {
    pub(crate) fn loaded(&self) -> &LoadedPackage {
        &self.loaded
    }

    /// Whether the package checks PiCCS for the CCS polynomial `f`.
    pub(crate) fn checks(&self, f: &SparsePoly<F>) -> bool {
        let terms = self.loaded.ccs_relation().terms();
        terms.len() == f.terms().len()
            && terms.iter().zip(f.terms()).all(|(ours, theirs)| {
                ours.coefficient() == theirs.coeff.as_canonical_u64()
                    && ours.exponents().len() == theirs.exps.len()
                    && ours
                        .exponents()
                        .iter()
                        .zip(&theirs.exps)
                        .all(|(&a, &b)| a == b as usize)
            })
    }

    /// Apply the rows to `statement` (the iteration, `z0`, the current state)
    /// and `inputs` (zeros in a shape run). Returns the layer-1 claim and the
    /// transcript after PiRLC.
    pub(crate) fn apply<B: Backend>(
        &self,
        b: &mut B,
        statement: &[B::F],
        context: [u64; 4],
        inputs: Option<&PiCcsV1_2EncodedInputs>,
    ) -> Result<(ClaimWords<B::F>, FoldTranscript<B>), Error> {
        assert_eq!(statement.len(), STATE_STATEMENT_WORDS);
        let computed = b.hint(&[], self.computed.len(), &|_| self.execute(inputs));
        let domain = STATE_DOMAIN_CHUNK;
        let private = |index: usize| inputs.map_or(0, |inputs| inputs.private_values()[index]);
        let public = |index: usize| inputs.map_or(0, |inputs| inputs.public_values()[index]);
        let columns: Vec<B::F> = self
            .sources
            .iter()
            .map(|&source| match source {
                Source::One => b.constant(1),
                Source::Private(index) if index < domain.len() => b.constant(domain[index]),
                Source::Private(index) if (STATE_CONTEXT_START..STATE_CONTEXT_START + 4).contains(&index) => {
                    b.constant(context[index - STATE_CONTEXT_START])
                }
                Source::Private(index)
                    if (STATE_CONTEXT_START + 4..PI_CCS_V1_2_STATE_PREIMAGE_WORDS).contains(&index) =>
                {
                    statement[index - STATE_CONTEXT_START - 4]
                }
                Source::Private(index) => b.private(private(index)),
                Source::Public(index) if index >= PUBLIC_CONTEXT_START => {
                    b.constant(context[index - PUBLIC_CONTEXT_START])
                }
                Source::Public(index) => b.private(public(index)),
                Source::Computed(index) => computed[index],
            })
            .collect();
        b.r1cs(&self.system, &columns, "terminal package")?;

        let outputs = &self.outputs;
        let ring = |list: &[usize]| -> Vec<[B::F; D]> {
            list.chunks_exact(D)
                .map(|chunk| std::array::from_fn(|lane| columns[chunk[lane]]))
                .collect()
        };
        let extension = |list: &[usize]| -> [[B::F; 2]; D] {
            std::array::from_fn(|coefficient| [columns[list[2 * coefficient]], columns[list[2 * coefficient + 1]]])
        };
        let claim = ClaimWords {
            commitment: ring(&outputs.commitment),
            public: ring(&outputs.public_input),
            point: outputs
                .point
                .chunks_exact(2)
                .map(|pair| [columns[pair[0]], columns[pair[1]]])
                .collect(),
            eval_k: extension(&outputs.eval_k),
            eval_a: outputs.eval_a.chunks_exact(2 * D).map(extension).collect(),
        };
        let state: [B::F; WIDTH] = std::array::from_fn(|lane| columns[outputs.state[lane]]);
        Ok((claim, FoldTranscript::from_state(state)))
    }

    /// The computed columns of an honest run on `inputs`; zeros without
    /// inputs or when a row fails, so that the rows then reject.
    fn execute(&self, inputs: Option<&PiCcsV1_2EncodedInputs>) -> Vec<u64> {
        let assignment = inputs.and_then(|inputs| {
            self.loaded
                .execute_witness(inputs.private_values(), inputs.public_values())
                .ok()
        });
        match assignment {
            Some(assignment) => self
                .computed
                .iter()
                .map(|&column| assignment.private_values()[column])
                .collect(),
            None => vec![0; self.computed.len()],
        }
    }
}
