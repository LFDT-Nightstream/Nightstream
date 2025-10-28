//! Pretty-printing utilities for debugging and logging

#![allow(dead_code)] // These utilities are called conditionally via #[cfg(feature = "neo-logs")]

use crate::F;
use p3_field::PrimeField64;

/// Pretty-print a field element (shows -1, -2 instead of p-1, p-2)
/// 
/// # Examples
/// 
/// ```ignore
/// let f = F::from_u64(18446744069414584320); // p-1
/// assert_eq!(pretty_field(f), "-1");
/// 
/// let f = F::from_u64(42);
/// assert_eq!(pretty_field(f), "42");
/// ```
pub fn pretty_field(f: F) -> String {
    let val = f.as_canonical_u64();
    let p = F::ORDER_U64;
    
    if val == 0 {
        "0".to_string()
    } else if val < 1000 {
        // Small positive numbers: show as-is
        val.to_string()
    } else if val > p - 1000 {
        // Large numbers near p: show as negative
        format!("-{}", p - val)
    } else {
        // Everything else: show raw value
        format!("{}", val)
    }
}

/// Pretty-print a slice of field elements
/// 
/// Shows first few elements with pretty formatting, then "... N more" if too long.
/// 
/// # Examples
/// 
/// ```ignore
/// let v = vec![F::ZERO, F::ONE, F::from_u64(18446744069414584320)];
/// assert_eq!(pretty_field_vec(&v), "[0, 1, -1]");
/// 
/// let long_v = vec![F::ZERO; 20];
/// assert_eq!(pretty_field_vec(&long_v), "[0, 0, 0, ... 17 more]");
/// ```
pub fn pretty_field_vec(v: &[F]) -> String {
    if v.is_empty() {
        "[]".to_string()
    } else if v.len() <= 5 {
        // Show all elements for short vectors
        let elements: Vec<String> = v.iter().map(|&f| pretty_field(f)).collect();
        format!("[{}]", elements.join(", "))
    } else {
        // Show first 3 + count for long vectors
        let first_three: Vec<String> = v[..3].iter().map(|&f| pretty_field(f)).collect();
        format!("[{}, ... {} more]", first_three.join(", "), v.len() - 3)
    }
}

/// Pretty-print row of a matrix (CCS constraint)
/// 
/// Useful for debugging constraint rows in augmented CCS.
/// 
/// # Examples
/// 
/// ```ignore
/// let row = vec![F::ONE, F::ZERO, F::from_u64(18446744069414584320)];
/// assert_eq!(pretty_matrix_row(&row, 0), "Row[0]: [1, 0, -1]");
/// ```
pub fn pretty_matrix_row(row: &[F], row_idx: usize) -> String {
    format!("Row[{}]: {}", row_idx, pretty_field_vec(row))
}
