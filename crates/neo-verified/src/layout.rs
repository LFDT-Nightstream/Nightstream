pub fn split(size: usize) -> [usize; 2] {
    let wide = two_pow(floor_log2(size));
    let rest = size.checked_sub(wide).unwrap();
    if rest == 0 {
        [wide, 0]
    } else {
        [wide, two_pow(ceil_log2(rest))]
    }
}

pub fn variables(count: usize) -> usize {
    if count <= 2 {
        1
    } else {
        ceil_log2(count) as usize
    }
}

fn floor_log2(n: usize) -> u32 {
    usize::BITS
        .checked_sub(1)
        .unwrap()
        .checked_sub(n.leading_zeros())
        .unwrap()
}

fn ceil_log2(n: usize) -> u32 {
    usize::BITS
        .checked_sub(n.checked_sub(1).unwrap().leading_zeros())
        .unwrap()
}

fn two_pow(k: u32) -> usize {
    1usize.wrapping_shl(k)
}
