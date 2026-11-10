//! The bounded-depth packages that the Nebula tests compile and load: an
//! application's light and refresh step packages (`""` for today's fixture,
//! `"nebula-v1-"`, `"nebula-n1-"`) and the finish package.

pub(crate) fn bounded_depth_packages(stem: &str) -> [Vec<u8>; 3] {
    let read = |name: String| {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../formal/nightstream-fprime/artifacts")
                .join(name),
        )
        .expect("Lean bounded-depth package")
    };
    [
        read(format!("nightstream-fprime-bounded-depth-{stem}light-v1.json")),
        read(format!("nightstream-fprime-bounded-depth-{stem}refresh-v1.json")),
        read("nightstream-fprime-bounded-depth-finish-v1.json".into()),
    ]
}

pub(crate) fn references(packages: &[Vec<u8>; 3]) -> crate::References<'_> {
    crate::References {
        light: &packages[0],
        refresh: &packages[1],
        finish: &packages[2],
    }
}
