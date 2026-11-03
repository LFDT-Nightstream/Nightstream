//! The option-B packages that the Nebula tests compile and load: an
//! application's light and refresh step packages (`""` for today's fixture,
//! `"nebula-v1-"`, `"nebula-n1-"`) and the finish package.

pub(crate) fn option_b_packages(stem: &str) -> [Vec<u8>; 3] {
    let read = |name: String| {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../formal/nightstream-fprime/artifacts")
                .join(name),
        )
        .expect("Lean option-B package")
    };
    [
        read(format!("nightstream-fprime-option-b-{stem}light-v1.json")),
        read(format!("nightstream-fprime-option-b-{stem}refresh-v1.json")),
        read("nightstream-fprime-option-b-finish-v1.json".into()),
    ]
}

pub(crate) fn references(packages: &[Vec<u8>; 3]) -> crate::option_b::References<'_> {
    crate::option_b::References {
        light: &packages[0],
        refresh: &packages[1],
        finish: &packages[2],
    }
}
