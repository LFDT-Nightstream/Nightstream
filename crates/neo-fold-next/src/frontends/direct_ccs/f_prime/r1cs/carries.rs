//! Native carry-bit witnesses for low-norm F' source counter constraints.

use super::*;

pub(super) fn push_increment_carries(carries: &mut Vec<u8>, input: u64) {
    let mut carry = 1u128;
    for bit_index in 0..U64_BITS {
        let sum = ((input >> bit_index) & 1) as u128 + carry;
        carry = sum >> 1;
        if bit_index + 1 < U64_BITS {
            carries.push(carry as u8);
        }
    }
}

pub(super) fn push_addition_carries(carries: &mut Vec<u8>, lhs: u64, rhs: u64) {
    let mut carry = 0u128;
    for bit_index in 0..U64_BITS {
        let sum = ((lhs >> bit_index) & 1) as u128 + ((rhs >> bit_index) & 1) as u128 + carry;
        carry = sum >> 1;
        if bit_index + 1 < U64_BITS {
            carries.push(carry as u8);
        }
    }
}
