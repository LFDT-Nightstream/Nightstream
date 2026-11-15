use std::num::NonZeroUsize;

use nightstream::nebula::{Geometry, R1csApplication, R1csApplicationError, RowRelation, Size};
use nightstream::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityGroup, ContinuityLink, MemoryCatalog,
};
use nightstream::{Fq as F, PrimeCharacteristicRing};

fn columns(width: usize) -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "test",
        start: 0,
        len: width,
        name: "row".into(),
        role: "test",
        width: ColumnWidth::Field,
    }])
    .unwrap()
}

fn size() -> Size<u8> {
    Size {
        geometry: Geometry::new(0, 0, 1, 1, 1).unwrap(),
        rows: NonZeroUsize::new(1).unwrap(),
        regions: vec![],
        slots: vec![vec![]],
    }
}

fn app(
    width: usize,
    one: usize,
    link: Option<ContinuityLink>,
    registry_width: usize,
) -> Result<R1csApplication<u8>, R1csApplicationError> {
    let columns = columns(registry_width);
    let groups = link.into_iter().map(|link| ContinuityGroup {
        name: "state",
        role: "test",
        links: vec![link],
    });
    let continuity = ContinuityCatalog::new(groups, &columns).unwrap();
    let catalog = MemoryCatalog::<u8>::new([], &columns).unwrap();
    R1csApplication::new(
        RowRelation {
            rows: vec![],
            width,
            one,
        },
        &continuity,
        &catalog,
        size(),
    )
}

#[test]
fn rejects_continuity_column_outside_relation() {
    assert!(matches!(
        app(
            2,
            0,
            Some(ContinuityLink {
                previous_step_column: 2,
                next_step_column: 1
            }),
            3
        ),
        Err(R1csApplicationError::Column)
    ));
}

#[test]
fn rejects_constant_column_outside_relation() {
    assert!(matches!(app(2, 2, None, 2), Err(R1csApplicationError::Column)));
}

#[test]
fn accepts_and_reads_valid_continuity() {
    let app = app(
        2,
        0,
        Some(ContinuityLink {
            previous_step_column: 1,
            next_step_column: 1,
        }),
        2,
    )
    .unwrap();
    assert_eq!(app.output_words(&[F::ONE, F::from_u64(7)]), vec![7]);
}
