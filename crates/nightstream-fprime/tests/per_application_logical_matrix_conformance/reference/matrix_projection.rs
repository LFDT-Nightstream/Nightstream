//! Checked relabeling of serialized matrix operands for the independent decoder.
//! Source-row schedules and physical-source projections keep their own indices.

use serde_json::Value;

use super::super::{Result, RetainedBlock};
use super::{array, checked_add, checked_mul, exact_array, word, SourceProjection};

struct Projection<'a> {
    source_width: usize,
    mapping: &'a SourceProjection,
}

impl Projection<'_> {
    fn column(&self, value: &Value) -> Result<Value> {
        let column = word(value, "mapped matrix column")?;
        if column >= self.source_width {
            return Err("mapped matrix source column is out of range".into());
        }
        Ok(Value::from(self.mapping.column(column)?))
    }

    fn retained(&self, value: &Value) -> Result<Value> {
        let retained = RetainedBlock::decode(value)?;
        retained.validate(self.source_width)?;
        let mut fields = exact_array(value, 3, "mapped retained block")?.to_vec();
        let count = checked_mul(retained.slot_count, retained.kind.width(), "mapped retained count")?;
        if count == 0 {
            return Ok(Value::Array(fields));
        }
        let start = word(&fields[2], "mapped retained start")?;
        let end = checked_add(start, count, "mapped retained end")?;
        let mapped_start = self.mapping.column(start)?;
        if let SourceProjection::Mapped(ranges) = self.mapping {
            let mut intersections = Vec::new();
            for range in ranges {
                let first = start.max(range.package_start);
                let last = end.min(checked_add(range.package_start, range.count, "mapped range end")?);
                if first < last {
                    intersections.push((first, last));
                }
            }
            intersections.sort_unstable();
            let mut cursor = start;
            for (first, last) in intersections {
                if first != cursor {
                    return Err("missing or overlapping mapped retained coordinates".into());
                }
                let expected = checked_add(mapped_start, first - start, "mapped retained coordinate")?;
                if self.mapping.column(first)? != expected {
                    return Err("mapped retained coordinates are not contiguous".into());
                }
                cursor = last;
            }
            if cursor != end {
                return Err("missing mapped retained coordinates".into());
            }
        }
        fields[2] = Value::from(mapped_start);
        Ok(Value::Array(fields))
    }

    fn form(&self, value: &Value) -> Result<Value> {
        let entries = array(value, "mapped sparse form")?
            .iter()
            .map(|entry| {
                let mut fields = exact_array(entry, 2, "mapped sparse entry")?.to_vec();
                // Check every stored read before form normalization removes zero terms.
                fields[0] = self.column(&fields[0])?;
                Ok(Value::Array(fields))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::Array(entries))
    }

    fn substitution(&self, value: &Value) -> Result<Value> {
        let fields = exact_array(value, 2, "mapped source substitution")?;
        let ranges = array(&fields[0], "mapped source ranges")?
            .iter()
            .map(|value| {
                let mut range = exact_array(value, 4, "mapped source range")?.to_vec();
                range[2] = self.retained(&range[2])?;
                Ok(Value::Array(range))
            })
            .collect::<Result<Vec<_>>>()?;
        let grids = array(&fields[1], "mapped source grids")?
            .iter()
            .map(|value| {
                let mut grid = exact_array(value, 11, "mapped source grid")?.to_vec();
                grid[6] = self.retained(&grid[6])?;
                Ok(Value::Array(grid))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::Array(vec![Value::Array(ranges), Value::Array(grids)]))
    }

    fn ordinary(&self, value: &Value) -> Result<Value> {
        let mut fields = exact_array(value, 4, "mapped ordinary block")?.to_vec();
        fields[1] = self.column(&fields[1])?;
        fields[2] = self.substitution(&fields[2])?;
        Ok(Value::Array(fields))
    }

    fn affine(&self, value: &Value) -> Result<Value> {
        let rules = array(value, "mapped affine program")?
            .iter()
            .map(|value| {
                let mut rule = exact_array(value, 2, "mapped affine rule")?.to_vec();
                let mut term = array(&rule[1], "mapped affine term")?.to_vec();
                match term.first().and_then(Value::as_u64) {
                    Some(0) if term.len() == 7 => term[1] = self.retained(&term[1])?,
                    Some(1) if term.len() == 2 => {}
                    _ => return Err("unknown mapped affine term".into()),
                }
                rule[1] = Value::Array(term);
                Ok(Value::Array(rule))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::Array(rules))
    }

    fn poseidon_input(&self, value: &Value) -> Result<Value> {
        let rules = array(value, "mapped Poseidon2 inputs")?
            .iter()
            .map(|value| {
                let mut rule = exact_array(value, 2, "mapped Poseidon2 rule")?.to_vec();
                let mut term = array(&rule[1], "mapped Poseidon2 term")?.to_vec();
                match term.first().and_then(Value::as_u64) {
                    Some(0) if term.len() == 5 => term[1] = self.retained(&term[1])?,
                    Some(2) if term.len() == 4 => term[1] = self.retained(&term[1])?,
                    Some(3) if term.len() == 7 => term[1] = self.retained(&term[1])?,
                    Some(5) if term.len() == 6 => term[2] = self.substitution(&term[2])?,
                    Some(6) if term.len() == 3 => {
                        term[1] = Value::Array(
                            array(&term[1], "mapped Poseidon2 sparse inputs")?
                                .iter()
                                .map(|form| self.form(form))
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    Some(1) if term.len() == 2 => {}
                    Some(4) if term.len() == 3 => {}
                    _ => return Err("unknown mapped Poseidon2 term".into()),
                }
                rule[1] = Value::Array(term);
                Ok(Value::Array(rule))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::Array(rules))
    }
}

pub(super) fn block(value: &Value, source_width: usize, mapping: &SourceProjection) -> Result<Value> {
    let fields = array(value, "mapped matrix block")?;
    let projection = Projection { source_width, mapping };
    if let [tag, width, inner_mapping, inner] = fields {
        if tag.as_u64() == Some(5) {
            let width = word(width, "inner mapped source width")?;
            let inner_mapping = SourceProjection::decode(inner_mapping, width)?;
            let resolved = block(inner, width, &inner_mapping)?;
            return block(&resolved, source_width, mapping);
        }
    }
    if let [tag, ordinary, rows] = fields {
        if tag.as_u64() == Some(6) {
            return Ok(Value::Array(vec![
                tag.clone(),
                projection.ordinary(ordinary)?,
                rows.clone(),
            ]));
        }
    }
    let fields = exact_array(value, 2, "mapped matrix block")?;
    let opcode = word(&fields[0], "mapped matrix opcode")?;
    let payload = match opcode {
        0 => projection.ordinary(&fields[1])?,
        1 => {
            let mut pin = exact_array(&fields[1], 2, "mapped pin block")?.to_vec();
            pin[0] = projection.column(&pin[0])?;
            pin[1] = Value::Array(
                array(&pin[1], "mapped pin values")?
                    .iter()
                    .map(|form| projection.form(form))
                    .collect::<Result<Vec<_>>>()?,
            );
            Value::Array(pin)
        }
        2 => {
            let mut poseidon = exact_array(&fields[1], 4, "mapped Poseidon2 block")?.to_vec();
            poseidon[1] = projection.column(&poseidon[1])?;
            poseidon[2] = projection.retained(&poseidon[2])?;
            poseidon[3] = projection.poseidon_input(&poseidon[3])?;
            Value::Array(poseidon)
        }
        3 => {
            let mut product = array(&fields[1], "mapped Phi81 block")?.to_vec();
            let input_index = match product.len() {
                8 => {
                    product[2] = projection.retained(&product[2])?;
                    5
                }
                7 => {
                    product[2] = Value::Array(
                        array(&product[2], "mapped Phi81 challenges")?
                            .iter()
                            .map(|form| projection.form(form))
                            .collect::<Result<Vec<_>>>()?,
                    );
                    4
                }
                _ => return Err("invalid mapped Phi81 block".into()),
            };
            product[1] = projection.column(&product[1])?;
            product[input_index] = projection.substitution(&product[input_index])?;
            product[input_index + 1] = projection.retained(&product[input_index + 1])?;
            product[input_index + 2] = projection.retained(&product[input_index + 2])?;
            Value::Array(product)
        }
        4 => {
            let mut product = exact_array(&fields[1], 5, "mapped multiplication block")?.to_vec();
            product[1] = projection.column(&product[1])?;
            for operand in &mut product[2..] {
                *operand = projection.affine(operand)?;
            }
            Value::Array(product)
        }
        _ => return Err("unknown mapped matrix opcode".into()),
    };
    Ok(Value::Array(vec![fields[0].clone(), payload]))
}
