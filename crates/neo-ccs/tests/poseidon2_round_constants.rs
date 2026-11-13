//! Pins the Poseidon2 constants against `scripts/poseidon2/params.json`, against a permutation
//! built from the matrices of that file, and against the selected F-prime Lean specification.

use neo_ccs::crypto::poseidon2 as p2;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn u64s(words: &serde_json::Value) -> Vec<u64> {
    words
        .as_array()
        .expect("array")
        .iter()
        .map(|word| word.as_str().expect("decimal").parse().expect("u64"))
        .collect()
}

fn rows(table: &serde_json::Value) -> Vec<Vec<u64>> {
    table.as_array().expect("rows").iter().map(u64s).collect()
}

fn canonical<const T: usize>(table: &[[Fq; T]]) -> Vec<Vec<u64>> {
    table
        .iter()
        .map(|row| row.iter().map(Fq::as_canonical_u64).collect())
        .collect()
}

fn matvec(matrix: &[Vec<Fq>], x: &[Fq]) -> Vec<Fq> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// The permutation of `params.json` with dense matrices: `M_E = circ(2·M4, M4, …)` and
/// `M_I = J + diag(D)`.
fn dense_permute(instance: &serde_json::Value, m4: &[Vec<u64>], state: &[Fq]) -> Vec<Fq> {
    let t = state.len();
    let fq = |values: Vec<u64>| values.into_iter().map(Fq::new).collect::<Vec<_>>();
    let external: Vec<Vec<Fq>> = (0..t)
        .map(|i| {
            (0..t)
                .map(|j| Fq::from_u64(m4[i % 4][j % 4] * if i / 4 == j / 4 { 2 } else { 1 }))
                .collect()
        })
        .collect();
    let diagonal = fq(u64s(&instance["internal_diagonal"]));
    let internal: Vec<Vec<Fq>> = (0..t)
        .map(|i| {
            (0..t)
                .map(|j| if i == j { diagonal[i] + Fq::ONE } else { Fq::ONE })
                .collect()
        })
        .collect();
    let sbox = |x: Fq| x.exp_u64(instance["d"].as_u64().expect("degree"));
    let full = |x: Vec<Fq>, table: &serde_json::Value| {
        rows(table).into_iter().fold(x, |x, constants| {
            let after: Vec<Fq> = x
                .iter()
                .zip(fq(constants))
                .map(|(&v, c)| sbox(v + c))
                .collect();
            matvec(&external, &after)
        })
    };
    let mut x = full(matvec(&external, state), &instance["external_initial"]);
    for constant in fq(u64s(&instance["internal"])) {
        x[0] = sbox(x[0] + constant);
        x = matvec(&internal, &x);
    }
    full(x, &instance["external_terminal"])
}

fn check_instance<const T: usize>(
    instance: &serde_json::Value,
    m4: &[Vec<u64>],
    rust: &p2::Instance<T>,
    rng: &mut StdRng,
) {
    assert_eq!(instance["t"], T);
    assert_eq!(instance["q"], Fq::ORDER_U64.to_string());
    assert_eq!((instance["rf"].as_u64(), instance["rp"].as_u64()), (Some(8), Some(22)));
    assert_eq!(rows(&instance["external_initial"]), canonical(&rust.initial));
    assert_eq!(rows(&instance["external_terminal"]), canonical(&rust.terminal));
    assert_eq!(u64s(&instance["internal"]), canonical(&[rust.internal])[0]);
    assert_eq!(u64s(&instance["internal_diagonal"]), canonical(&[rust.diagonal])[0]);
    for _ in 0..64 {
        let state: [Fq; T] = core::array::from_fn(|_| Fq::new(rng.random::<u64>()));
        assert_eq!(dense_permute(instance, m4, &state), rust.permute(state), "width {T}");
    }
}

#[test]
fn checked_in_constants_equal_params_json() {
    let params: serde_json::Value =
        serde_json::from_str(include_str!("../../../scripts/poseidon2/params.json")).expect("params.json");
    let m4: Vec<Vec<u64>> = params["m4"]
        .as_array()
        .expect("m4")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_u64().expect("entry"))
                .collect()
        })
        .collect();
    let instances = params["instances"].as_array().expect("instances");
    assert_eq!(instances.len(), 2);
    let mut rng = StdRng::seed_from_u64(0x7032_5f72_635f_7631);
    check_instance(&instances[0], &m4, &p2::INSTANCE_16, &mut rng);
    check_instance(&instances[1], &m4, &p2::INSTANCE_12, &mut rng);
}

fn lean_numbers_after(source: &str, declaration: &str) -> Vec<u64> {
    let body = source
        .split_once(declaration)
        .expect("selected Lean Poseidon2 constant declaration")
        .1
        .split_once("\n\n")
        .expect("next Lean declaration")
        .0;
    body.split(|c: char| !(c.is_ascii_hexdigit() || c == 'x'))
        .filter(|word| !word.is_empty())
        .map(|word| {
            if let Some(hex) = word.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).expect("Lean hexadecimal constant")
            } else {
                word.parse().expect("Lean decimal constant")
            }
        })
        .collect()
}

#[test]
fn exported_round_constants_match_fprime_lean_spec() {
    let source = include_str!("../../../formal/nightstream-fprime/NightstreamFPrime/Spec/Poseidon2.lean");
    let rust = &p2::INSTANCE_16;
    let initial = lean_numbers_after(source, "def initialConstants : List (List Nat) :=");
    let internal = lean_numbers_after(source, "def internalConstants : List Nat :=");
    let terminal = lean_numbers_after(source, "def terminalConstants : List (List Nat) :=");
    let diagonal = lean_numbers_after(source, "def internalDiagonal : List Nat :=");
    assert_eq!(initial, canonical(&rust.initial).concat());
    assert_eq!(internal, canonical(&[rust.internal])[0]);
    assert_eq!(terminal, canonical(&rust.terminal).concat());
    assert_eq!(diagonal, canonical(&[rust.diagonal])[0]);
}
