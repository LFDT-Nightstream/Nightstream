use std::fmt::Write as _;

use neo_ccs::SeededPhi81LinearBlock;
use p3_field::PrimeField64;

use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySeededPhi81Artifact.lean";
const SCHEMA_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySeededPhi81Schema.lean";
const SHARD_PREFIX: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySeededPhi81Block";
const REJECTION_FUEL: usize = 16;

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    let values = values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn lean_seed(seed: &[u8; 32]) -> String {
    lean_nat_list(seed.iter().copied().map(usize::from))
}

fn lean_nat_sequence(values: &[usize]) -> String {
    if values.len() >= 4 {
        let step = values[1] - values[0];
        if values
            .iter()
            .enumerate()
            .all(|(index, value)| *value == values[0] + index * step)
        {
            return format!(
                "(List.range {}).map fun index => {} + index * {}",
                values.len(),
                values[0],
                step
            );
        }
    }
    lean_nat_list(values.iter().copied())
}

fn lean_seed_rows(rows: &[Vec<[u8; 32]>]) -> String {
    format!(
        "[{}]",
        rows.iter()
            .map(|seeds| format!("[{}]", seeds.iter().map(lean_seed).collect::<Vec<_>>().join(", ")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn output_columns(builder: &R1csBuilder, row_start: usize, row_end: usize) -> Vec<usize> {
    let (a, b, c) = builder.sparse_triplets();
    assert!(
        a.iter()
            .all(|(row, _, _)| *row < row_start || *row >= row_end),
        "compact seeded Phi81 rows must not also carry explicit A terms"
    );
    (row_start..row_end)
        .map(|row| {
            let b_row = b
                .iter()
                .filter(|(candidate, _, _)| *candidate == row)
                .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
                .collect::<Vec<_>>();
            assert_eq!(b_row, vec![(0, 1)], "seeded Phi81 row has non-unit B");
            let c_row = c
                .iter()
                .filter(|(candidate, _, _)| *candidate == row)
                .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
                .collect::<Vec<_>>();
            let [(column, 1)] = c_row.as_slice() else {
                panic!("seeded Phi81 row {row} must define exactly one output: {c_row:?}");
            };
            *column
        })
        .collect()
}

fn render_schema(builder: &R1csBuilder) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.SeededPhi81\n\n\
         /-! Generated placement schema for production full-history SeededPhi81 blocks. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.SeededPhi81\n\n\
         def totalRows : Nat := {}\n\
         def totalColumns : Nat := {}\n\
         def rejectionFuel : Nat := {REJECTION_FUEL}\n\n\
         def rowEnd (block : SeededPhi81.Block) : Nat :=\n\
         \x20 block.rowStart + SeededPhi81.dimension * block.kappa\n\n\
         def MetadataValid (block : SeededPhi81.Block) : Prop :=\n\
         \x20 0 < block.wordWidth ∧ 0 < block.kappa ∧\n\
         \x20 0 < block.schedule.chunkSize ∧\n\
         \x20 block.superneoTransformedColumns = false ∧\n\
         \x20 block.messageCols =\n\
         \x20   (block.wordStarts.length * block.wordWidth + SeededPhi81.dimension - 1) /\n\
         \x20     SeededPhi81.dimension ∧\n\
         \x20 block.outputColumns.length = SeededPhi81.dimension * block.kappa ∧\n\
         \x20 block.schedule.seedsByOutput.length = block.kappa ∧\n\
         \x20 ∀ seeds ∈ block.schedule.seedsByOutput,\n\
         \x20   seeds.length =\n\
         \x20     (block.messageCols + block.schedule.chunkSize - 1) /\n\
         \x20       block.schedule.chunkSize\n\n\
         def RowsMapped (block : SeededPhi81.Block) : Prop :=\n\
         \x20 rowEnd block ≤ totalRows ∧\n\
         \x20 block.outputColumns.length = SeededPhi81.dimension * block.kappa\n\n\
         instance (block : SeededPhi81.Block) : Decidable (MetadataValid block) := by\n\
         \x20 unfold MetadataValid\n\
         \x20 infer_instance\n\n\
         instance (block : SeededPhi81.Block) : Decidable (RowsMapped block) := by\n\
         \x20 unfold RowsMapped rowEnd\n\
         \x20 infer_instance\n\n\
         def CertifiedBlock :=\n\
         \x20 {{ block : SeededPhi81.Block //\n\
         \x20   block.Valid ∧ MetadataValid block ∧ RowsMapped block ∧\n\
         \x20   block.superneoTransformedColumns = false }}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n",
        builder.rows(),
        builder.cols(),
    )
}

fn render_block(builder: &R1csBuilder, index: usize, block: &SeededPhi81LinearBlock) -> String {
    assert!(
        !block.has_superneo_transformed_columns(),
        "Lean compact compiler does not yet implement transformed SeededPhi81 columns"
    );
    let outputs = output_columns(builder, block.row_start(), block.row_end());
    let mut rendered = String::new();
    rendered.push_str("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema\n\n");
    writeln!(rendered, "/-! Generated production SeededPhi81 block {index}. -/\n").expect("render");
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n\n");
    rendered.push_str("open Nightstream.Implementation.R1CS.SeededPhi81\n\n");
    rendered.push_str("set_option maxRecDepth 1048576\n");
    rendered.push_str("set_option maxHeartbeats 0\n\n");
    writeln!(rendered, "def block{index} : SeededPhi81.Block :=").expect("render");
    writeln!(rendered, "  {{ rowStart := {}", block.row_start()).expect("render");
    writeln!(rendered, "    wordStarts := {}", lean_nat_sequence(block.word_starts())).expect("render");
    writeln!(rendered, "    wordWidth := {}", block.word_width()).expect("render");
    writeln!(rendered, "    kappa := {}", block.kappa()).expect("render");
    writeln!(rendered, "    messageCols := {}", block.message_cols()).expect("render");
    writeln!(rendered, "    outputColumns := {}", lean_nat_sequence(&outputs)).expect("render");
    writeln!(
        rendered,
        "    superneoTransformedColumns := {}",
        block.has_superneo_transformed_columns()
    )
    .expect("render");
    rendered.push_str("    schedule :=\n");
    writeln!(rendered, "      {{ chunkSize := {}", block.chunk_size()).expect("render");
    writeln!(
        rendered,
        "        seedsByOutput := {}",
        lean_seed_rows(block.chunk_seeds_by_row())
    )
    .expect("render");
    rendered.push_str("        rejectionFuel := rejectionFuel } }\n\n");
    writeln!(
        rendered,
        "theorem block{index}_certified :\n    block{index}.Valid ∧ MetadataValid block{index} ∧ RowsMapped block{index} ∧\n      block{index}.superneoTransformedColumns = false := by native_decide\n"
    )
    .expect("render");
    writeln!(
        rendered,
        "def certifiedBlock{index} : CertifiedBlock := ⟨block{index}, block{index}_certified⟩\n"
    )
    .expect("render");
    rendered.push_str("end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n");
    rendered
}

fn render_artifact(block_count: usize) -> String {
    let imports = (0..block_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Block{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let certified = (0..block_count)
        .map(|index| format!("certifiedBlock{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{imports}\n\n\
         /-! Generated certified index of every production full-history SeededPhi81 block. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n\n\
         def certifiedBlocks : List CertifiedBlock :=\n  [{certified}]\n\n\
         def blocks : List SeededPhi81.Block := certifiedBlocks.map Subtype.val\n\n\
         theorem blocks_valid {{block : SeededPhi81.Block}} (member : block ∈ blocks) :\n\
         \x20   block.Valid := by\n\
         \x20 rw [blocks] at member\n\
         \x20 rcases List.mem_map.mp member with ⟨certified, _, rfl⟩\n\
         \x20 exact certified.property.1\n\n\
         theorem metadata_valid {{block : SeededPhi81.Block}} (member : block ∈ blocks) :\n\
         \x20   MetadataValid block := by\n\
         \x20 rw [blocks] at member\n\
         \x20 rcases List.mem_map.mp member with ⟨certified, _, rfl⟩\n\
         \x20 exact certified.property.2.1\n\n\
         theorem rows_mapped {{block : SeededPhi81.Block}} (member : block ∈ blocks) :\n\
         \x20   RowsMapped block := by\n\
         \x20 rw [blocks] at member\n\
         \x20 rcases List.mem_map.mp member with ⟨certified, _, rfl⟩\n\
         \x20 exact certified.property.2.2.1\n\n\
         theorem transformed_columns_status {{block : SeededPhi81.Block}}\n\
         \x20   (member : block ∈ blocks) : block.superneoTransformedColumns = false := by\n\
         \x20 rw [blocks] at member\n\
         \x20 rcases List.mem_map.mp member with ⟨certified, _, rfl⟩\n\
         \x20 exact certified.property.2.2.2\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81\n"
    )
}

pub fn compare_seeded_phi81_artifact(builder: &R1csBuilder) {
    let mut blocks = builder.seeded_phi81_a_blocks().iter().collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.row_start());
    for pair in blocks.windows(2) {
        assert!(
            pair[0].row_end() <= pair[1].row_start(),
            "seeded Phi81 row ranges overlap"
        );
    }

    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write full-history SeededPhi81 artifact");
            drifted.push(expected);
        }
    };
    compare(root.join(SCHEMA_PATH), render_schema(builder));
    for (index, block) in blocks.iter().enumerate() {
        compare(
            root.join(format!("{SHARD_PREFIX}{index}.lean")),
            render_block(builder, index, block),
        );
    }
    compare(root.join(ARTIFACT_PATH), render_artifact(blocks.len()));
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history SeededPhi81 artifacts drifted: {drifted:?}"
    );
}
