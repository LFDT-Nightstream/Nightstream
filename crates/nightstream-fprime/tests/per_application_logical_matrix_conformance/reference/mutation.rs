//! Structurally valid mutations of the derived assignment transport.

use serde_json::Value;

use super::{array, exact_array, word, Result};

#[derive(Clone, Copy, Debug)]
pub enum RecipeFamily {
    Phi81,
    First54,
    Payload,
    OutputDigest,
}

impl RecipeFamily {
    pub const ALL: [Self; 4] = [Self::Phi81, Self::First54, Self::Payload, Self::OutputDigest];
}

pub fn self_consistent_bytes(sealed_bytes: &[u8], family: RecipeFamily) -> Result<Vec<u8>> {
    let mut sealed: Value =
        serde_json::from_slice(sealed_bytes).map_err(|error| format!("recipe-mutation package decode: {error}"))?;
    let transport = sealed
        .as_array_mut()
        .and_then(|fields| fields.get_mut(4))
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "missing assignment transport".to_string())?;
    if transport.len() != 8 || transport[0].as_u64() != Some(1) {
        return Err("unexpected assignment transport for mutation".into());
    }
    match family {
        RecipeFamily::Phi81 => {
            shift_block_sources(transport, 7)?;
        }
        RecipeFamily::First54 => {
            shift_block_sources(transport, 4)?;
        }
        RecipeFamily::Payload => {
            let expressions = transport[5]
                .as_array_mut()
                .ok_or_else(|| "missing payload expressions".to_string())?;
            let original = expressions
                .first()
                .cloned()
                .ok_or_else(|| "empty payload expressions".to_string())?;
            expressions[0] = Value::Array(vec![
                Value::from(2u64),
                original,
                Value::Array(vec![Value::from(1u64), Value::from(1u64)]),
            ]);
        }
        RecipeFamily::OutputDigest => {
            shift_block_sources(transport, 27)?;
            let sources = block_sources(transport, 27)?;
            if sources.len() != 4 {
                return Err("output-digest block does not have four sources".into());
            }
            transport[7] = Value::Array(
                sources
                    .into_iter()
                    .map(|source| Value::Array(vec![Value::from(0u64), Value::from(source as u64)]))
                    .collect(),
            );
        }
    }
    let mut bytes = serde_json::to_vec(&sealed).map_err(|error| format!("recipe-mutation encode: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn shift_block_sources(transport: &mut [Value], opcode: usize) -> Result<()> {
    let blocks = transport[1]
        .as_array_mut()
        .ok_or_else(|| "missing assignment blocks".to_string())?;
    let block = blocks
        .get_mut(opcode)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| format!("missing assignment block {opcode}"))?;
    if block.len() != 5 || block[0].as_u64() != Some(opcode as u64) {
        return Err(format!("unexpected assignment block {opcode}"));
    }
    let runs = block[4]
        .as_array_mut()
        .ok_or_else(|| format!("missing assignment block {opcode} runs"))?;
    let first_run = runs
        .first_mut()
        .and_then(Value::as_array_mut)
        .ok_or_else(|| format!("empty assignment block {opcode} runs"))?;
    if first_run.len() != 3 {
        return Err("invalid assignment source run".into());
    }
    let first = first_run[0]
        .as_u64()
        .ok_or_else(|| "invalid assignment source-run first".to_string())?;
    first_run[0] = Value::from(if first == 0 { 1 } else { first - 1 });
    Ok(())
}

fn block_sources(transport: &[Value], opcode: usize) -> Result<Vec<usize>> {
    let blocks = array(&transport[1], "assignment blocks")?;
    let block = exact_array(
        blocks
            .get(opcode)
            .ok_or_else(|| format!("missing assignment block {opcode}"))?,
        5,
        "assignment block",
    )?;
    let expected = word(&block[2], "assignment block slot count")?;
    let mut sources = Vec::with_capacity(expected);
    for run in array(&block[4], "assignment source runs")? {
        let fields = exact_array(run, 3, "assignment source run")?;
        let first = word(&fields[0], "assignment source first")?;
        let step = word(&fields[1], "assignment source step")?;
        let count = word(&fields[2], "assignment source count")?;
        for offset in 0..count {
            sources.push(
                first
                    .checked_add(
                        step.checked_mul(offset)
                            .ok_or_else(|| "assignment source offset overflow".to_string())?,
                    )
                    .ok_or_else(|| "assignment source value overflow".to_string())?,
            );
        }
    }
    if sources.len() != expected {
        return Err("assignment source-run coverage mismatch".into());
    }
    Ok(sources)
}
