//! Lean renderer for the exact verifier-native terminal guard ledger.

use std::fmt::Write;

use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    TERMINAL_CONTEXT_GUARD_NAMES, TERMINAL_PROOF_GUARD_NAMES, TERMINAL_STATEMENT_GUARD_NAMES,
};

const SCHEMA: u32 = 1;

pub fn checked_terminal_native_guard_names() -> String {
    let names = TERMINAL_CONTEXT_GUARD_NAMES
        .into_iter()
        .chain(TERMINAL_STATEMENT_GUARD_NAMES)
        .chain(TERMINAL_PROOF_GUARD_NAMES)
        .collect::<Vec<_>>();

    assert_eq!(names.len(), 18, "terminal native guard count");
    assert!(names.iter().all(|name| !name.is_empty()));

    render(&names)
}

fn render(names: &[&str]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "/-!\nGENERATED FILE — do not edit by hand.\n\n\
         Exact ordered verifier-native terminal guard names emitted from the\n\
         Rust check-site enums. This is structural drift evidence. It does not\n\
         prove that a Rust check refines the matching Lean model.\n-/\n"
    )
    .unwrap();
    writeln!(
        out,
        "namespace Nightstream.Implementation.R1CS.Artifacts.\
         TerminalVerifierNativeGuards.Generated.Names\n"
    )
    .unwrap();
    writeln!(out, "def schema : Nat := {SCHEMA}\n").unwrap();
    writeln!(out, "def values : List String :=\n  [").unwrap();
    for (index, name) in names.iter().enumerate() {
        let separator = if index + 1 == names.len() { "" } else { "," };
        writeln!(out, "    \"{name}\"{separator}").unwrap();
    }
    writeln!(out, "  ]\n").unwrap();
    writeln!(
        out,
        "end Nightstream.Implementation.R1CS.Artifacts.\
         TerminalVerifierNativeGuards.Generated.Names"
    )
    .unwrap();
    out
}
