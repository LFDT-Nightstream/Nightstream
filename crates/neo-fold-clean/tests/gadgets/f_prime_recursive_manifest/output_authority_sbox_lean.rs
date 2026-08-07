//! Rust-to-Lean bridge for the audited output-authority Poseidon2 S-box manifest.
//!
//! Owns: normalization of the production audit into compact Lean data,
//! deterministic rendering, generator-local invariants, and fail-closed
//! mutation tests.
//!
//! Does not own: the production audit, Lean proofs, centered substitution,
//! row removal, or promotion of generated review output.
//!
//! Emits constraints: no.
//!
//! Authority boundary: only a successful
//! `audit_output_authority_poseidon2_sboxes` call may construct this bridge.
//! The two emitted Rust evidence booleans report that exact row replay and the
//! complete A/B/C matrix-use scan succeeded; they are not Lean proofs of the
//! corresponding global source predicates.
//!
//! | Bridge branch | Generated data | Generator validation | Lean consumer |
//! |---|---|---|---|
//! | boundaries | stage, prehash, hash, digest, trace intervals | exact nesting and widths | `Boundaries.Valid` |
//! | calls | 5 call geometries | order, range, 600-row/column ABI | `CallGeometry.Valid` |
//! | offsets | 86 isolated `x^7` output offsets | exact Poseidon2 family formulas | `offsetsExact` |
//! | census | candidate and A/C-use totals | independently recomputed arithmetic | `Census.Valid` |
//! | Rust evidence | row replay and whole-matrix scan accepted | successful production audit | metadata only |

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_fold_clean::frontends::f_prime::output_authority::{
    audit_output_authority_poseidon2_sboxes, OutputAuthorityPoseidon2SboxCensus,
    OutputAuthorityPoseidon2SboxFamilyLayout, OutputAuthorityPoseidon2SboxManifest, Poseidon2PermutationCall,
};

use super::{build_recursive_program, repo_root};

const LEAN_DATA_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/FPrimeRecursiveOutputAuthoritySboxManifestData.lean";
const EXPECTED_STAGE_ROWS: usize = 3_034;
const EXPECTED_STAGE_COLUMNS: usize = 3_034;
const EXPECTED_PREHASH_ROWS: usize = 12;
const EXPECTED_PREHASH_COLUMNS: usize = 12;
const EXPECTED_HASH_INPUTS: usize = 16;
const EXPECTED_FULL_ABSORBS: usize = 4;
const EXPECTED_PARTIAL_ABSORB_FIELDS: usize = 0;
const EXPECTED_PERMUTATIONS: usize = 5;
const EXPECTED_PERMUTATION_ROWS: usize = 600;
const EXPECTED_PERMUTATION_COLUMNS: usize = 600;
const EXPECTED_INITIAL_SBOXES: usize = 32;
const EXPECTED_PARTIAL_SBOXES: usize = 22;
const EXPECTED_TERMINAL_SBOXES: usize = 32;
const EXPECTED_SBOXES_PER_PERMUTATION: usize = 86;
const EXPECTED_CANDIDATES: usize = 430;
const EXPECTED_LINEAR_USES: usize = 3_440;
const EXPECTED_TOTAL_USES: usize = 3_870;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LeanBridgeManifest {
    stage_rows: Range<usize>,
    stage_columns: Range<usize>,
    prehash_rows: Range<usize>,
    prehash_columns: Range<usize>,
    hash_rows: Range<usize>,
    hash_zero_column: usize,
    hash_output_columns: [usize; 4],
    claimed_digest_columns: [usize; 4],
    semantic_state_output_columns: [usize; 4],
    permutation_trace_range: Range<usize>,
    calls: Vec<Poseidon2PermutationCall>,
    isolated_output_offsets: Vec<usize>,
    families: OutputAuthorityPoseidon2SboxFamilyLayout,
    census: OutputAuthorityPoseidon2SboxCensus,
    exact_call_rows_accepted: bool,
    whole_matrix_no_escape_accepted: bool,
}

impl LeanBridgeManifest {
    fn from_audited(manifest: &OutputAuthorityPoseidon2SboxManifest) -> Self {
        Self {
            stage_rows: manifest.stage_rows.clone(),
            stage_columns: manifest.stage_columns.clone(),
            prehash_rows: manifest.prehash_rows.clone(),
            prehash_columns: manifest.prehash_columns.clone(),
            hash_rows: manifest.hash_rows.clone(),
            hash_zero_column: manifest.prehash_columns.end,
            hash_output_columns: manifest.hash_output_columns,
            claimed_digest_columns: manifest.claimed_digest_columns,
            semantic_state_output_columns: manifest.semantic_state_output_columns,
            permutation_trace_range: manifest.permutation_trace_range.clone(),
            calls: manifest.calls.clone(),
            isolated_output_offsets: manifest.isolated_sbox_output_offsets().to_vec(),
            families: manifest.family_layout.clone(),
            census: manifest.census,
            exact_call_rows_accepted: true,
            whole_matrix_no_escape_accepted: true,
        }
    }

    fn validate(&self, source_rows: usize, source_columns: usize) -> Result<(), String> {
        self.validate_boundaries(source_rows, source_columns)?;
        self.validate_calls(source_rows, source_columns)?;
        self.validate_offsets_and_families()?;
        let candidates = self.derived_candidates()?;
        self.validate_census(source_rows, source_columns, candidates.len())?;
        if !self.exact_call_rows_accepted || !self.whole_matrix_no_escape_accepted {
            return Err("Rust conformance evidence must come from a successful audit".to_owned());
        }
        Ok(())
    }

    fn validate_boundaries(&self, source_rows: usize, source_columns: usize) -> Result<(), String> {
        let expected_claimed = std::array::from_fn(|lane| self.claimed_digest_columns[0] + lane);
        let semantic_start = self
            .stage_columns
            .end
            .checked_sub(4)
            .ok_or_else(|| "stage columns cannot contain the digest boundary".to_owned())?;
        let expected_semantic = std::array::from_fn(|lane| semantic_start + lane);
        if self.stage_rows.len() != EXPECTED_STAGE_ROWS
            || self.stage_columns.len() != EXPECTED_STAGE_COLUMNS
            || self.stage_rows.end > source_rows
            || self.stage_columns.end > source_columns
            || self.prehash_rows.start != self.stage_rows.start
            || self.prehash_rows.len() != EXPECTED_PREHASH_ROWS
            || self.prehash_rows.end != self.hash_rows.start
            || self.hash_rows.end + 4 != self.stage_rows.end
            || self.prehash_columns.start != self.stage_columns.start
            || self.prehash_columns.len() != EXPECTED_PREHASH_COLUMNS
            || self.prehash_columns.end != self.hash_zero_column
            || self.claimed_digest_columns != expected_claimed
            || self.claimed_digest_columns[3] >= self.stage_columns.start
            || self.semantic_state_output_columns != expected_semantic
            || self.permutation_trace_range.len() != EXPECTED_PERMUTATIONS
            || self
                .hash_output_columns
                .iter()
                .any(|&column| column >= source_columns)
        {
            return Err("stage/hash/prehash/digest boundary geometry drifted".to_owned());
        }
        Ok(())
    }

    fn validate_calls(&self, source_rows: usize, source_columns: usize) -> Result<(), String> {
        if self.calls.len() != EXPECTED_PERMUTATIONS {
            return Err("Poseidon2 call count drifted".to_owned());
        }
        for (index, call) in self.calls.iter().enumerate() {
            let expected_trace_index = self.permutation_trace_range.start + index;
            let expected_outputs: [usize; 8] = std::array::from_fn(|lane| call.first_allocated_column + 592 + lane);
            if call.trace_index != expected_trace_index
                || call.source_rows.len() != EXPECTED_PERMUTATION_ROWS
                || call.allocated_column_count != EXPECTED_PERMUTATION_COLUMNS
                || call.output_columns != expected_outputs
                || call.source_rows.start < self.hash_rows.start
                || call.source_rows.end > self.hash_rows.end
                || call.source_rows.end > source_rows
                || call.first_allocated_column + call.allocated_column_count > source_columns
                || call
                    .input_columns
                    .iter()
                    .any(|&column| column >= source_columns)
            {
                return Err(format!("Poseidon2 call {index} geometry drifted"));
            }
        }
        for (index, pair) in self.calls.windows(2).enumerate() {
            if pair[0].trace_index >= pair[1].trace_index
                || pair[0].source_rows.end > pair[1].source_rows.start
                || pair[0].first_allocated_column + pair[0].allocated_column_count > pair[1].first_allocated_column
            {
                return Err(format!("Poseidon2 call order drifted after call {index}"));
            }
        }
        let final_call = self
            .calls
            .last()
            .ok_or_else(|| "Poseidon2 calls are empty".to_owned())?;
        if self.hash_output_columns != final_call.output_columns[..4] {
            return Err("hash outputs do not match the final Poseidon2 call".to_owned());
        }
        Ok(())
    }

    fn validate_offsets_and_families(&self) -> Result<(), String> {
        if self.isolated_output_offsets != expected_isolated_output_offsets() {
            return Err("isolated S-box output offsets drifted".to_owned());
        }
        if self.families.initial_external != (0..EXPECTED_INITIAL_SBOXES)
            || self.families.partial != (EXPECTED_INITIAL_SBOXES..EXPECTED_INITIAL_SBOXES + EXPECTED_PARTIAL_SBOXES)
            || self.families.terminal_external
                != (EXPECTED_INITIAL_SBOXES + EXPECTED_PARTIAL_SBOXES
                    ..EXPECTED_INITIAL_SBOXES + EXPECTED_PARTIAL_SBOXES + EXPECTED_TERMINAL_SBOXES)
            || EXPECTED_INITIAL_SBOXES + EXPECTED_PARTIAL_SBOXES + EXPECTED_TERMINAL_SBOXES
                != EXPECTED_SBOXES_PER_PERMUTATION
        {
            return Err("Poseidon2 S-box family ranges drifted".to_owned());
        }
        Ok(())
    }

    fn derived_candidates(&self) -> Result<Vec<usize>, String> {
        let mut candidates = Vec::with_capacity(self.calls.len() * self.isolated_output_offsets.len());
        let mut unique = BTreeSet::new();
        let protected = self
            .hash_output_columns
            .iter()
            .chain(&self.claimed_digest_columns)
            .chain(&self.semantic_state_output_columns)
            .copied()
            .chain(std::iter::once(self.hash_zero_column))
            .collect::<BTreeSet<_>>();
        for call in &self.calls {
            for &offset in &self.isolated_output_offsets {
                let column = call.first_allocated_column + offset;
                if offset >= call.allocated_column_count || protected.contains(&column) || !unique.insert(column) {
                    return Err("candidate S-box outputs overlap, escape, or alias a protected boundary".to_owned());
                }
                candidates.push(column);
            }
        }
        if candidates.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("candidate S-box output order drifted".to_owned());
        }
        Ok(candidates)
    }

    fn validate_census(&self, source_rows: usize, source_columns: usize, candidate_count: usize) -> Result<(), String> {
        let census = self.census;
        let expected_candidates = self.calls.len() * self.isolated_output_offsets.len();
        let expected_definitions = expected_candidates;
        let expected_linear = expected_candidates * 8;
        if census.scanned_source_rows != source_rows
            || census.scanned_source_columns != source_columns
            || census.stage_rows != self.stage_rows.len()
            || census.stage_columns != self.stage_columns.len()
            || census.prehash_binding_rows != self.prehash_rows.len()
            || census.prehash_fresh_columns != self.prehash_columns.len()
            || census.hash_input_fields != EXPECTED_HASH_INPUTS
            || census.full_absorb_rounds != EXPECTED_FULL_ABSORBS
            || census.partial_absorb_fields != EXPECTED_PARTIAL_ABSORB_FIELDS
            || census.pad_rounds != 1
            || census.permutations != self.calls.len()
            || census.initial_external_sboxes != self.calls.len() * self.families.initial_external.len()
            || census.partial_sboxes != self.calls.len() * self.families.partial.len()
            || census.terminal_external_sboxes != self.calls.len() * self.families.terminal_external.len()
            || candidate_count != expected_candidates
            || census.candidate_sbox_outputs != expected_candidates
            || census.definition_uses != expected_definitions
            || census.linear_consumer_uses != expected_linear
            || census.total_matrix_uses != expected_definitions + expected_linear
            || expected_candidates != EXPECTED_CANDIDATES
            || expected_linear != EXPECTED_LINEAR_USES
            || expected_definitions + expected_linear != EXPECTED_TOTAL_USES
        {
            return Err("output-authority S-box census drifted".to_owned());
        }
        Ok(())
    }
}

fn expected_isolated_output_offsets() -> Vec<usize> {
    let mut offsets = Vec::with_capacity(EXPECTED_SBOXES_PER_PERMUTATION);
    for round in 0..4 {
        for lane in 0..8 {
            offsets.push(11 + 40 * round + 4 * lane);
        }
    }
    for round in 0..EXPECTED_PARTIAL_SBOXES {
        offsets.push(171 + 12 * round);
    }
    for round in 0..4 {
        for lane in 0..8 {
            offsets.push(435 + 40 * round + 4 * lane);
        }
    }
    offsets
}

fn audited_bridge() -> (LeanBridgeManifest, usize, usize) {
    let builder = build_recursive_program();
    let source = builder.snapshot();
    let manifest = audit_output_authority_poseidon2_sboxes(&source, builder.encoding_trace(), &[])
        .expect("exact output-authority Poseidon2 S-box audit");
    let bridge = LeanBridgeManifest::from_audited(&manifest);
    bridge
        .validate(source.rows(), source.cols())
        .expect("valid Rust-to-Lean output-authority S-box bridge");
    (bridge, source.rows(), source.cols())
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

fn lean_range(range: &Range<usize>) -> String {
    format!("{{ start := {}, finish := {} }}", range.start, range.end)
}

fn render_lean_bridge(bridge: &LeanBridgeManifest) -> String {
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.OutputAuthoritySboxManifestSchema\n\n",
    );
    rendered.push_str(
        "/-! Generated by `gadgets_f_prime_recursive_manifest`; do not hand-edit.\n\nOwns: exact compact data from the audited output-authority Poseidon2 S-box\ncall manifest and a kernel-checked geometry/census certificate.\n\nDoes not own: global source-row identity, a Lean complete-matrix extractor,\ncentered substitution, or permission to remove rows or slots.\n\nEmits constraints: no.\n\nAuthority boundary: Rust accepted exact per-call row replay and scanned the\nwhole A/B/C source matrix. The booleans below record that external conformance\nevidence; they do not prove `SourceCallRowsMatch` or `WholeMatrixNoEscape`.\n\n| Generated branch | Exact content | Lean check | Permits row removal? |\n|---|---|---|---|\n| boundaries | stage/prehash/hash/digest intervals | `Boundaries.Valid` | no |\n| calls | 5 call geometries | `CallGeometry.Valid` plus order | no |\n| offsets/families | 86 isolated outputs in 32/22/32 phases | exact equality | no |\n| census | 430 candidates and 3,870 roles | `Census.Valid` | no |\n| Rust evidence | exact rows and whole-matrix scan accepted | metadata only | no |\n-/\n\n",
    );
    rendered.push_str("namespace Nightstream.Implementation.R1CS.OutputAuthoritySboxManifestData\n\n");
    rendered.push_str("open Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest\n\n");
    rendered.push_str("set_option maxRecDepth 1048576\n\n");
    rendered.push_str("def calls : List CallGeometry :=\n  [ ");
    for (index, call) in bridge.calls.iter().enumerate() {
        if index != 0 {
            rendered.push_str("  , ");
        }
        writeln!(
            rendered,
            "{{ traceIndex := {}, rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {}, allocatedColumnCount := {}, outputColumns := {} }}",
            call.trace_index,
            call.source_rows.start,
            call.source_rows.end,
            lean_nat_list(call.input_columns),
            call.first_allocated_column,
            call.allocated_column_count,
            lean_nat_list(call.output_columns),
        )
        .expect("render call");
    }
    rendered.push_str("  ]\n\n");
    writeln!(
        rendered,
        "def isolatedOutputOffsets : List Nat := {}\n",
        lean_nat_list(bridge.isolated_output_offsets.iter().copied())
    )
    .expect("render offsets");
    writeln!(rendered, "def manifest : Manifest :=").expect("render manifest");
    writeln!(rendered, "  {{ schemaVersion := 1").expect("render manifest");
    writeln!(rendered, "    boundaries :=").expect("render manifest");
    writeln!(rendered, "      {{ stageRows := {}", lean_range(&bridge.stage_rows)).expect("render manifest");
    writeln!(
        rendered,
        "        stageColumns := {}",
        lean_range(&bridge.stage_columns)
    )
    .expect("render manifest");
    writeln!(rendered, "        prehashRows := {}", lean_range(&bridge.prehash_rows)).expect("render manifest");
    writeln!(
        rendered,
        "        prehashColumns := {}",
        lean_range(&bridge.prehash_columns)
    )
    .expect("render manifest");
    writeln!(rendered, "        hashRows := {}", lean_range(&bridge.hash_rows)).expect("render manifest");
    writeln!(rendered, "        hashZeroColumn := {}", bridge.hash_zero_column).expect("render manifest");
    writeln!(
        rendered,
        "        hashOutputColumns := {}",
        lean_nat_list(bridge.hash_output_columns)
    )
    .expect("render manifest");
    writeln!(
        rendered,
        "        claimedDigestColumns := {}",
        lean_nat_list(bridge.claimed_digest_columns)
    )
    .expect("render manifest");
    writeln!(
        rendered,
        "        semanticStateOutputColumns := {}",
        lean_nat_list(bridge.semantic_state_output_columns)
    )
    .expect("render manifest");
    writeln!(
        rendered,
        "        permutationTraceRange := {} }}",
        lean_range(&bridge.permutation_trace_range)
    )
    .expect("render manifest");
    writeln!(rendered, "    calls := calls").expect("render manifest");
    writeln!(rendered, "    isolatedOutputOffsets := isolatedOutputOffsets").expect("render manifest");
    writeln!(
        rendered,
        "    families := {{ initialExternal := {}, partialRounds := {}, terminalExternal := {} }}",
        lean_range(&bridge.families.initial_external),
        lean_range(&bridge.families.partial),
        lean_range(&bridge.families.terminal_external),
    )
    .expect("render manifest");
    let census = bridge.census;
    writeln!(rendered, "    census :=").expect("render manifest");
    writeln!(rendered, "      {{ scannedSourceRows := {}", census.scanned_source_rows).expect("render census");
    writeln!(
        rendered,
        "        scannedSourceColumns := {}",
        census.scanned_source_columns
    )
    .expect("render census");
    writeln!(
        rendered,
        "        prehashBindingRows := {}",
        census.prehash_binding_rows
    )
    .expect("render census");
    writeln!(
        rendered,
        "        prehashFreshColumns := {}",
        census.prehash_fresh_columns
    )
    .expect("render census");
    writeln!(rendered, "        hashInputFields := {}", census.hash_input_fields).expect("render census");
    writeln!(rendered, "        fullAbsorbRounds := {}", census.full_absorb_rounds).expect("render census");
    writeln!(
        rendered,
        "        partialAbsorbFields := {}",
        census.partial_absorb_fields
    )
    .expect("render census");
    writeln!(rendered, "        padRounds := {}", census.pad_rounds).expect("render census");
    writeln!(rendered, "        permutations := {}", census.permutations).expect("render census");
    writeln!(
        rendered,
        "        sboxesPerPermutation := {EXPECTED_SBOXES_PER_PERMUTATION}"
    )
    .expect("render census");
    writeln!(
        rendered,
        "        initialExternalSboxes := {}",
        census.initial_external_sboxes
    )
    .expect("render census");
    writeln!(rendered, "        partialSboxes := {}", census.partial_sboxes).expect("render census");
    writeln!(
        rendered,
        "        terminalExternalSboxes := {}",
        census.terminal_external_sboxes
    )
    .expect("render census");
    writeln!(
        rendered,
        "        candidateSboxOutputs := {}",
        census.candidate_sbox_outputs
    )
    .expect("render census");
    writeln!(rendered, "        definitionCUses := {}", census.definition_uses).expect("render census");
    writeln!(rendered, "        linearAUses := {}", census.linear_consumer_uses).expect("render census");
    writeln!(rendered, "        totalMatrixUses := {} }}", census.total_matrix_uses).expect("render census");
    writeln!(
        rendered,
        "    rustEvidence := {{ exactCallRowsAccepted := {}, wholeMatrixNoEscapeAccepted := {} }} }}\n",
        bridge.exact_call_rows_accepted, bridge.whole_matrix_no_escape_accepted,
    )
    .expect("render manifest");
    rendered.push_str("def certificate : manifest.Certificate where\n");
    for field in [
        "schemaVersion",
        "boundariesValid",
        "everyCallValid",
        "callsAdjacent",
        "traceOrder",
        "offsetsExact",
        "offsetsInAllocatedRange",
        "familiesValid",
        "censusValid",
        "callCount",
        "offsetCount",
        "candidateColumnsIncreasing",
        "boundaryDisjoint",
        "exactRowsEvidence",
        "noEscapeEvidence",
    ] {
        writeln!(rendered, "  {field} := by native_decide").expect("render certificate");
    }
    rendered.push_str("\nend Nightstream.Implementation.R1CS.OutputAuthoritySboxManifestData\n");
    rendered
}

#[test]
fn output_authority_sbox_lean_bridge_rejects_generator_mutations_and_is_deterministic() {
    let (bridge, source_rows, source_columns) = audited_bridge();
    assert_eq!(render_lean_bridge(&bridge), render_lean_bridge(&bridge));

    let mut call_order = bridge.clone();
    call_order.calls.swap(0, 1);
    assert!(call_order.validate(source_rows, source_columns).is_err());

    let mut call_count = bridge.clone();
    call_count.calls.pop();
    assert!(call_count.validate(source_rows, source_columns).is_err());

    let mut call_geometry = bridge.clone();
    call_geometry.calls[0].source_rows.end -= 1;
    assert!(call_geometry.validate(source_rows, source_columns).is_err());

    let mut offset = bridge.clone();
    offset.isolated_output_offsets[0] += 1;
    assert!(offset.validate(source_rows, source_columns).is_err());

    let mut family = bridge.clone();
    family.families.partial.end -= 1;
    assert!(family.validate(source_rows, source_columns).is_err());

    let mut census = bridge.clone();
    census.census.total_matrix_uses -= 1;
    assert!(census.validate(source_rows, source_columns).is_err());
}

#[test]
fn output_authority_sbox_lean_manifest_matches_audited_production() {
    let (bridge, _, _) = audited_bridge();
    let rendered = render_lean_bridge(&bridge);
    let path = repo_root().join(LEAN_DATA_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("output-authority manifest parent"))
            .expect("create output-authority manifest directory");
        fs::write(&expected, &rendered).expect("write expected output-authority S-box Lean manifest");
    }
    assert_eq!(
        committed, rendered,
        "output-authority S-box Lean manifest drifted; inspect and deliberately promote the generated .expected file"
    );
}
