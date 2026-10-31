use neo_verified::layout::{split, variables};
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

fn inputs() -> Vec<usize> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut values = EDGES.to_vec();
    for _ in 0..200 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push((state >> (state % 64)) as usize);
    }
    values
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
    cases
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
