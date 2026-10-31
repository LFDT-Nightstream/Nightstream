use neo_verified::circuit::{Kind, Wire};
use neo_verified::layout::{
    block_cell, block_row, cell_variables, column, entry_column, glue_row, one, place, row_variables, slot, split,
    variables, Part, Shape,
};
use std::panic::{catch_unwind, UnwindSafe};
use std::path::Path;
use std::process::Command;

const EDGES: [usize; 16] = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    (1 << 63) - 1,
    1 << 63,
    (1 << 63) + 1,
    usize::MAX,
];

const SHIFTS: [u32; 11] = [0, 1, 31, 32, 63, 64, 65, 127, 128, 1 << 31, u32::MAX];

/// Block counts at the edges of `place`: `192·count` is the end of both part kinds.
const COUNTS: [usize; 8] = [
    0,
    1,
    3,
    1000,
    usize::MAX / 192,
    usize::MAX / 192 + 1,
    usize::MAX / 128 + 1,
    usize::MAX,
];

/// Glue rows, glue cells and statement words; `2^63` makes `cell_variables` 64.
const GLUES: [[usize; 3]; 9] = [
    [0, 0, 0],
    [70, 90, 5],
    [1 << 32, 1 << 32, 1 << 32],
    [usize::MAX, 0, 0],
    [0, usize::MAX, 0],
    [0, 0, usize::MAX],
    [0, 0, (1 << 63) - 1],
    [0, 0, 1 << 63],
    [1 << 63, 1 << 63, 0],
];

fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> (*state % 64)) as usize
}

fn inputs() -> Vec<usize> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut values = EDGES.to_vec();
    for _ in 0..200 {
        values.push(random(&mut state));
    }
    values
}

fn shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();
    for count in COUNTS {
        for [glue_rows, glue_cells, publics] in GLUES {
            shapes.push(Shape {
                blocks: [count],
                glue_rows,
                glue_cells,
                publics,
            });
        }
    }
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    for _ in 0..60 {
        shapes.push(Shape {
            blocks: [random(&mut state) >> 8],
            glue_rows: random(&mut state),
            glue_cells: random(&mut state),
            publics: random(&mut state),
        });
    }
    shapes
}

fn shape_words(shape: &Shape) -> [u64; 4] {
    [shape.blocks[0], shape.glue_rows, shape.glue_cells, shape.publics].map(|word| word as u64)
}

fn wire_words(wire: Wire) -> [u64; 3] {
    match wire {
        Wire::Glue(cell) => [0, cell.into(), 0],
        Wire::Block(_, block, cell) => [1, block.into(), cell.into()],
        Wire::Public(index) => [2, index.into(), 0],
    }
}

const WIRES: [Wire; 6] = [
    Wire::Glue(0),
    Wire::Glue(u32::MAX),
    Wire::Block(Kind::Permutation, 1, 129),
    Wire::Block(Kind::Permutation, u32::MAX, u16::MAX),
    Wire::Public(0),
    Wire::Public(u32::MAX),
];

fn show_parts(([wide, narrow], end): ([Part; 2], usize)) -> String {
    format!(
        "{} {} {} {} {end}",
        wide.start, wide.stride, narrow.start, narrow.stride
    )
}

fn words(values: &[usize]) -> Vec<u64> {
    values.iter().map(|&value| value as u64).collect()
}

fn outcome<T>(run: impl FnOnce() -> T + UnwindSafe, show: impl Fn(T) -> String) -> String {
    match catch_unwind(run) {
        Ok(value) => format!("ok {}", show(value)),
        Err(_) => "fail".to_string(),
    }
}

fn unwrap(value: Option<usize>) -> usize {
    value.unwrap()
}

fn option(value: Option<impl std::fmt::Display>) -> String {
    value.map_or("none".to_string(), |v| format!("some {v}"))
}

fn cases() -> Vec<String> {
    let mut cases = Vec::new();
    let mut case = |name: &str, args: &[u64], output: String| {
        let args: Vec<String> = args.iter().map(u64::to_string).collect();
        cases.push(format!("{name} {} -> {output}", args.join(" ")).replace("  ->", " ->"));
    };
    for x in inputs() {
        let w = x as u64;
        case("split", &[w], outcome(|| split(x), |[a, b]| format!("{a} {b}")));
        case("variables", &[w], outcome(|| variables(x), |v| v.to_string()));
        case(
            "core.num.Usize.leading_zeros",
            &[w],
            format!("ok {}", x.leading_zeros()),
        );
        for y in EDGES {
            case(
                "Usize.checked_sub",
                &[w, y as u64],
                format!("ok {}", option(x.checked_sub(y))),
            );
        }
    }
    for k in SHIFTS {
        for x in [1usize, 3, usize::MAX] {
            case(
                "core.num.Usize.wrapping_shl",
                &[x as u64, k.into()],
                format!("ok {}", x.wrapping_shl(k)),
            );
        }
        for j in SHIFTS {
            case(
                "U32.checked_sub",
                &[k.into(), j.into()],
                format!("ok {}", option(k.checked_sub(j))),
            );
        }
        case("UScalar.cast", &[k.into()], format!("ok {}", k as usize));
    }
    case(
        "core.option.Option.unwrap.some",
        &[5],
        outcome(|| unwrap(Some(5)), |v| v.to_string()),
    );
    case(
        "core.option.Option.unwrap.none",
        &[],
        outcome(|| unwrap(None), |v| v.to_string()),
    );
    case("core.num.Usize.BITS", &[], format!("ok {}", usize::BITS));
    case("Array.make", &[0, u64::MAX], format!("ok 0 {}", u64::MAX));
    layout_cases(&mut case);
    model_cases(&mut case);
    cases
}

fn layout_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    let kind = Kind::Permutation;
    case("Kind.index", &[], format!("ok {}", kind.index()));
    case("Kind.rows", &[], format!("ok {}", kind.rows()));
    case("Kind.cells", &[], format!("ok {}", kind.cells()));
    let sizes = [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        166,
        182,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        usize::MAX,
    ];
    let counts = [0, 1, 2, 1000, 1 << 32, 1 << 62, usize::MAX];
    for size in sizes {
        for count in counts {
            case(
                "place",
                &words(&[size, count]),
                outcome(|| place(size, count), show_parts),
            );
        }
    }
    let mut state: u64 = 0x6a09_e667_f3bc_c908;
    for _ in 0..100 {
        let (size, count) = (random(&mut state), random(&mut state));
        case(
            "place",
            &words(&[size, count]),
            outcome(|| place(size, count), show_parts),
        );
    }
    for size in [1, 2, 3, 7, 166, 182, 1 << 63, usize::MAX] {
        for count in [0, 1, 5, 1 << 57, usize::MAX] {
            for b in [0, 1, 4, 1 << 60] {
                for k in [0, 1, 6, 127, 128, 165, usize::MAX] {
                    case(
                        "slot",
                        &words(&[size, count, b, k]),
                        outcome(|| slot(size, count, b, k), |v| v.to_string()),
                    );
                }
            }
        }
    }
    for shape in shapes() {
        let s = shape_words(&shape);
        let with = |extra: &[u64]| [s.as_slice(), extra].concat();
        case(
            "row_variables",
            &s,
            outcome(|| row_variables(&shape), |v| v.to_string()),
        );
        case(
            "cell_variables",
            &s,
            outcome(|| cell_variables(&shape), |v| v.to_string()),
        );
        case("one", &s, outcome(|| one(&shape), |v| v.to_string()));
        for g in [0, 1, usize::MAX] {
            case(
                "glue_row",
                &with(&words(&[g])),
                outcome(|| glue_row(&shape, g), |v| v.to_string()),
            );
        }
        for (b, k) in [(0, 0), (2, 130), (usize::MAX, 0)] {
            case(
                "block_row",
                &with(&words(&[b, k])),
                outcome(|| block_row(&shape, kind, b, k), |v| v.to_string()),
            );
        }
        for (b, k) in [(0, 0), (2, 150), (1 << 20, 181)] {
            case(
                "block_cell",
                &with(&words(&[b, k])),
                outcome(|| block_cell(&shape, kind, b, k), |v| v.to_string()),
            );
        }
        for wire in WIRES {
            case(
                "column",
                &with(&wire_words(wire)),
                outcome(|| column(&shape, wire), |v| v.to_string()),
            );
        }
        for (b, cell) in [(0, None), (1, Some(3)), (usize::MAX, Some(usize::MAX))] {
            let cell_words = cell.map_or([0, 0], |cell| [1, cell as u64]);
            case(
                "entry_column",
                &with(&[b as u64, cell_words[0], cell_words[1]]),
                outcome(|| entry_column(&shape, kind, b, cell), |v| v.to_string()),
            );
        }
    }
}

/// The models that `place`, `cell_variables` and the derived `Clone` and `PartialEq` impls call.
/// A clone of a `Copy` value is the value, so the Rust side of a clone case is its input.
fn model_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    for x in EDGES {
        for y in EDGES {
            let args = words(&[x, y]);
            case("Usize.checked_add", &args, format!("ok {}", option(x.checked_add(y))));
            case("Usize.checked_mul", &args, format!("ok {}", option(x.checked_mul(y))));
            case("core.cmp.impls.OrdUsize.max", &args, format!("ok {}", x.max(y)));
            case("core.cmp.PartialEqUsize.eq", &args, format!("ok {}", x == y));
            case(
                "core.array.equality.PartialEqArray.eq",
                &args,
                format!("ok {}", [x] == [y]),
            );
        }
    }
    for i in [0, 1, 2, usize::MAX] {
        let array = [7usize, 9];
        case(
            "Array.index_usize",
            &words(&[array[0], array[1], i]),
            outcome(|| array[i], |v| v.to_string()),
        );
    }
    for x in [0u32, 1, 1 << 31, u32::MAX] {
        for y in [0u32, 1, 1 << 31, u32::MAX] {
            case(
                "core.cmp.impls.PartialEqU32.eq",
                &[x.into(), y.into()],
                format!("ok {}", x == y),
            );
        }
    }
    for x in [0u16, 1, 1 << 15, u16::MAX] {
        for y in [0u16, 1, 1 << 15, u16::MAX] {
            case(
                "core.cmp.impls.PartialEqU16.eq",
                &[x.into(), y.into()],
                format!("ok {}", x == y),
            );
        }
    }
    for a in WIRES {
        let words_a = wire_words(a);
        case(
            "core.clone.Clone.clone.wire",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        case(
            "core.marker.Copy.cloneInst.wire",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        case("core.marker.StructuralPartialEq.wire", &words_a, "ok".to_string());
        for b in WIRES {
            let args = [words_a, wire_words(b)].concat();
            case("core.cmp.PartialEq.eq.wire", &args, format!("ok {}", a == b));
        }
    }
    let kind = Kind::Permutation;
    case("core.clone.Clone.clone.kind", &[], format!("ok {}", kind.index()));
    case(
        "core.cmp.PartialEq.eq.kind",
        &[],
        format!("ok {}", kind == Kind::Permutation),
    );
    let shapes = shapes();
    for a in &shapes[..12] {
        let words_a = shape_words(a);
        case(
            "core.clone.Clone.clone.shape",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        for b in &shapes[..12] {
            let args = [words_a, shape_words(b)].concat();
            case("core.cmp.PartialEq.eq.shape", &args, format!("ok {}", a == b));
        }
    }
    case("core.clone.Clone.clone.part", &[3, 4], "ok 3 4".to_string());
}

fn show(words: &[u64]) -> String {
    words
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs Lean: the pinned toolchain and `lake build CompressionLeanRust` in formal/compression-lean"]
fn translation_runs_as_rust() {
    assert_eq!(usize::BITS, 64, "the cases assume a 64-bit usize");
    std::panic::set_hook(Box::new(|_| {}));
    let cases = cases();
    let models = include_str!("../../../formal/aeneas/std-models.txt");
    for model in models
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        assert!(
            cases
                .iter()
                .any(|case| case.starts_with(&format!("{model} ")) || case.starts_with(&format!("{model}."))),
            "formal/aeneas/std-models.txt lists {model}, which has no case"
        );
    }
    let _ = std::panic::take_hook();
    let file = Path::new(env!("CARGO_TARGET_TMPDIR")).join("neo_verified_run_check.txt");
    std::fs::write(&file, cases.join("\n") + "\n").unwrap();
    let lean = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../formal/compression-lean");
    let output = Command::new("lake")
        .args(["env", "lean", "--run", "CompressionLeanRust/RunCheck.lean"])
        .arg(&file)
        .current_dir(&lean)
        .output()
        .expect("run lake from formal/compression-lean");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    assert_eq!(stdout.trim(), format!("rustCheck: {} cases agree", cases.len()));
}
