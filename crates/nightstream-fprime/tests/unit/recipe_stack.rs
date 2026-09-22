use super::RecipeStack;

#[test]
fn recipe_stack_buffers_shallow_frames_and_replays_spilled_blocks() {
    let mut stack = RecipeStack::new().unwrap();
    for i in 0..stack.block_frames {
        stack.push([i as u64, 1, 2]).unwrap();
    }
    assert_eq!(stack.file.metadata().unwrap().len(), 0);
    let count = stack.block_frames * 3 + 1;
    for i in stack.block_frames..count {
        stack.push([i as u64, 1, 2]).unwrap();
    }
    assert!(stack.file.metadata().unwrap().len() > 0);
    for i in (0..count).rev() {
        assert_eq!(stack.pop().unwrap(), Some([i as u64, 1, 2]));
    }
    assert_eq!(stack.pop().unwrap(), None);
    stack.push([3, 4, 5]).unwrap();
    stack.reset();
    assert_eq!(stack.pop().unwrap(), None);
    stack.push([6, 7, 8]).unwrap();
    assert_eq!(stack.pop().unwrap(), Some([6, 7, 8]));
}
