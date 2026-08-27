//! Exact conformance check for one Lean package plan and expanded reference.

#[path = "check_package_conformance/support.rs"]
mod support;

use std::{env, path::PathBuf};

fn main() {
    let mut arguments = env::args_os().skip(1);
    let plan_path = PathBuf::from(arguments.next().expect("package-plan path"));
    let reference_path = PathBuf::from(arguments.next().expect("expanded-package path"));
    let parity_path = PathBuf::from(arguments.next().expect("PiCCS parity path"));
    let mut identity = [0u64; 4];
    for word in &mut identity {
        *word = arguments
            .next()
            .expect("four expected identity words")
            .into_string()
            .expect("identity word text")
            .parse()
            .expect("identity word u64");
    }
    assert!(arguments.next().is_none(), "unexpected argument");
    support::run(&plan_path, &reference_path, &parity_path, identity);
}
