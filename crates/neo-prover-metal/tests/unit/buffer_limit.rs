use super::*;

#[test]
fn oversized_buffer_is_rejected_without_allocation_or_submission() {
    let session = MetalSession::new().unwrap();
    let before = session.activity();
    assert!(matches!(
        session.buffer(usize::MAX),
        Err(MetalError::MemoryLimit {
            requested: usize::MAX,
            limit: BUFFER_LIMIT_BYTES,
            ..
        })
    ));
    let after = session.activity();
    assert_eq!(before.allocated_bytes, after.allocated_bytes);
    assert_eq!(before.current_allocated_bytes, after.current_allocated_bytes);
    assert_eq!(before.command_buffers, after.command_buffers);
}
