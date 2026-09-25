//! Structurally valid mutations of the derived assignment transport.

use serde_json::Value;

use super::{array, exact_array, word, Result};

#[derive(Clone, Copy, Debug)]
pub enum RecipeFamily {
    Phi81,
    ChallengeBits,
    OutputDigest,
}

pub fn self_consistent_bytes(sealed_bytes: &[u8], family: RecipeFamily) -> Result<Vec<u8>> {
    let mut sealed: Value =
        serde_json::from_slice(sealed_bytes).map_err(|error| format!("recipe-mutation package decode: {error}"))?;
    let transport = sealed
        .as_array_mut()
        .and_then(|fields| fields.get_mut(4))
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "missing assignment transport".to_string())?;
    if transport.len() != 4 || transport[0].as_u64() != Some(4) {
        return Err("unexpected assignment transport for mutation".into());
    }
    match family {
        RecipeFamily::Phi81 => {
            // Keep the retained digits fixed, but derive each quotient with
            // the other scalar's valid three-bit challenge coefficients.
            swap_challenge_sources(transport)?;
        }
        RecipeFamily::ChallengeBits => {
            swap_challenge_sources(transport)?;
            let blocks = transport[1]
                .as_array_mut()
                .ok_or_else(|| "missing assignment blocks".to_string())?;
            // The emitted per-scalar triples start at block 23. Their third
            // block contains 353 checked bits, including all 54 digit triples.
            // Swap the first two scalar bit blocks with the quotient recipe.
            for index in [25, 28] {
                let block = exact_array(
                    blocks
                        .get(index)
                        .ok_or_else(|| format!("missing bit block {index}"))?,
                    3,
                    "sampler bit block",
                )?;
                if word(&block[0], "sampler bit kind")? != 0 || word(&block[1], "sampler bit count")? != 353 {
                    return Err("unexpected sampler bit block".into());
                }
            }
            blocks.swap(25, 28);
        }
        RecipeFamily::OutputDigest => {
            shift_block_sources(transport, 17, 0)?;
            let sources = block_sources(transport, 17)?;
            if sources.len() != 4 {
                return Err("output-digest block does not have four sources".into());
            }
            transport[3] = Value::Array(
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

fn swap_challenge_sources(transport: &mut [Value]) -> Result<()> {
    let recipe = transport[2]
        .as_array_mut()
        .filter(|fields| fields.len() == 3)
        .ok_or_else(|| "missing wide Phi81 recipe".to_string())?;
    let sources = recipe[2]
        .as_array_mut()
        .filter(|runs| runs.len() == 17)
        .ok_or_else(|| "unexpected challenge source runs".to_string())?;
    for source in &sources[..2] {
        let run = exact_array(source, 3, "challenge source run")?;
        if word(&run[1], "challenge source stride")? != 1 || word(&run[2], "challenge bit count")? != 162 {
            return Err("unexpected challenge bit run".into());
        }
    }
    sources.swap(0, 1);
    Ok(())
}

fn shift_block_sources(transport: &mut [Value], index: usize, slot: usize) -> Result<()> {
    let blocks = transport[1]
        .as_array_mut()
        .ok_or_else(|| "missing assignment blocks".to_string())?;
    let block = blocks
        .get_mut(index)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| format!("missing assignment block {index}"))?;
    if block.len() != 3 {
        return Err(format!("unexpected assignment block {index}"));
    }
    let runs = block[2]
        .as_array_mut()
        .ok_or_else(|| format!("missing assignment block {index} runs"))?;
    let mut end = 0usize;
    for run in runs {
        let fields = run
            .as_array_mut()
            .filter(|fields| fields.len() == 3)
            .ok_or_else(|| "invalid assignment source run".to_string())?;
        end = end
            .checked_add(word(&fields[2], "assignment source-run count")?)
            .ok_or_else(|| "assignment source-run coverage overflow".to_string())?;
        if slot < end {
            let first = word(&fields[0], "assignment source-run first")?;
            fields[0] = Value::from(if first == 0 { 1 } else { first - 1 });
            return Ok(());
        }
    }
    Err(format!("assignment block {index} has no source for slot {slot}"))
}

fn block_sources(transport: &[Value], index: usize) -> Result<Vec<usize>> {
    let blocks = array(&transport[1], "assignment blocks")?;
    let block = exact_array(
        blocks
            .get(index)
            .ok_or_else(|| format!("missing assignment block {index}"))?,
        3,
        "assignment block",
    )?;
    let expected = word(&block[1], "assignment block slot count")?;
    let mut sources = Vec::with_capacity(expected);
    for run in array(&block[2], "assignment source runs")? {
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
