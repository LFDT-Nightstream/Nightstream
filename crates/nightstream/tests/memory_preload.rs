use nightstream::relation::MemoryPreload;

#[test]
fn cells_for_gives_the_cells_of_one_memory_in_address_order() {
    let mut preload = MemoryPreload::default();
    preload.insert(1, vec![2, 0], 20);
    preload.insert(2, vec![0], 99);
    preload.insert(1, vec![1, 3], 10);

    assert_eq!(
        preload.cells_for(&1).collect::<Vec<_>>(),
        vec![(&[1, 3][..], 10), (&[2, 0][..], 20)]
    );
    assert_eq!(preload.cells_for(&3).next(), None);
}
