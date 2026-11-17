use super::*;

#[test]
fn oversized_buffer_is_rejected_without_allocation() {
    let session = CudaSession::new().unwrap();
    let before = session.activity();
    for bytes in [usize::MAX, BUFFER_LIMIT_BYTES + 1] {
        assert!(matches!(
            session.buffer(bytes),
            Err(CudaError::MemoryLimit {
                requested,
                limit: BUFFER_LIMIT_BYTES,
                ..
            }) if requested == bytes
        ));
    }
    let after = session.activity();
    assert_eq!(before.allocated_bytes, after.allocated_bytes);
    assert_eq!(before.launches, after.launches);
}

#[test]
fn buffers_start_zeroed() {
    let session = CudaSession::new().unwrap();
    let buffer = session.buffer(4096).unwrap();
    assert!(session
        .read_buffer::<u64>(&buffer, 512)
        .unwrap()
        .iter()
        .all(|&word| word == 0));
    session.write_shared(&buffer, &[7u64, 8, 9]).unwrap();
    assert_eq!(session.read_at::<u64>(&buffer, 8, 2).unwrap(), [8, 9]);
    assert!(CudaSession::peak_allocated_bytes() >= 4096);
}
