//! Relocates only the typed physical-column fields declared by SharedVerifier.

use serde_json::{json, Value};

use super::{
    manifest::{Counts, Manifest},
    wire::*,
    AssemblyError,
};

struct Mapping<'a> {
    manifest: &'a Manifest,
    from: Counts,
    to: Counts,
}

impl Mapping<'_> {
    fn column(&self, column: &mut usize) -> Result<(), AssemblyError> {
        let m = self.manifest;
        let old_constant = m.geometry.source_constant.eval(self.from)?;
        let old_total = m.geometry.source_total.eval(self.from)?;
        if *column < m.source_relocation.source_private_start {
            return Ok(());
        }
        if *column < old_constant || *column >= old_total {
            return Err(AssemblyError::Invalid(
                "shared row reads application-private or invalid column",
            ));
        }
        *column = m
            .geometry
            .source_constant
            .eval(self.to)?
            .checked_add(*column - old_constant)
            .ok_or(AssemblyError::Overflow)?;
        Ok(())
    }

    fn row(&self, row: &mut usize) -> Result<(), AssemblyError> {
        let source = &self.manifest.source_relocation;
        if *row < source.application_row_start {
            return Ok(());
        }
        let old_next = source.next_preimage_row_start.eval(self.from)?;
        let offset = row
            .checked_sub(old_next)
            .filter(|offset| *offset < source.next_preimage_row_count)
            .ok_or(AssemblyError::Invalid("shared row is inside application insertion"))?;
        *row = source
            .next_preimage_row_start
            .eval(self.to)?
            .checked_add(offset)
            .ok_or(AssemblyError::Overflow)?;
        Ok(())
    }

    fn combination(&self, combination: &mut Combination) -> Result<(), AssemblyError> {
        for (column, _) in &mut combination.terms {
            self.column(column)?;
        }
        Ok(())
    }

    fn expression(&self, value: &mut Value) -> Result<(), AssemblyError> {
        let fields = value
            .as_array_mut()
            .ok_or(AssemblyError::Invalid("source witness expression"))?;
        let tag = fields
            .first()
            .and_then(Value::as_u64)
            .ok_or(AssemblyError::Invalid("source witness expression tag"))?;
        match (tag, fields.len()) {
            (0, 2) => {
                let mut column = super::word(&fields[1])?;
                self.column(&mut column)?;
                fields[1] = json!(column);
            }
            (1, 2) => {}
            (2 | 3, 3) => {
                self.expression(&mut fields[1])?;
                self.expression(&mut fields[2])?;
            }
            _ => return Err(AssemblyError::Invalid("source witness expression shape")),
        }
        Ok(())
    }

    fn batch(&self, batch: &mut Batch) -> Result<(), AssemblyError> {
        self.column(&mut batch.start)?;
        for recipe in &mut batch.recipes {
            self.expression(recipe)?;
        }
        for hint in &mut batch.hints {
            let fields = hint
                .as_array_mut()
                .ok_or(AssemblyError::Invalid("source witness hint"))?;
            let tag = fields
                .first()
                .and_then(Value::as_u64)
                .ok_or(AssemblyError::Invalid("source witness hint tag"))?;
            if !matches!((tag, fields.len()), (0, 3) | (1..=3, 2)) {
                return Err(AssemblyError::Invalid("source witness hint shape"));
            }
            self.expression(&mut fields[1])?;
        }
        Ok(())
    }

    fn layout(&self, layout: &mut Layout) -> Result<(), AssemblyError> {
        let g = &self.manifest.geometry;
        layout.rows = g.source_rows.eval(self.to)?;
        layout.private = g.source_private.eval(self.to)?;
        layout.constant = g.source_constant.eval(self.to)?;
        layout.total = g.source_total.eval(self.to)?;
        for segment in &mut layout.private_segments {
            match segment.role {
                17 => {
                    segment.start = self
                        .manifest
                        .port("application_witness")?
                        .source_start
                        .eval(self.to)?;
                    segment.length = self.to.witness;
                }
                18 => {
                    segment.start = self
                        .manifest
                        .port("application_local")?
                        .source_start
                        .eval(self.to)?;
                    segment.length = self.to.local;
                }
                _ => {}
            }
        }
        for segment in &mut layout.public_segments {
            self.column(&mut segment.start)?;
        }
        Ok(())
    }
}

fn checked_relocation<T: Clone + PartialEq>(
    value: &mut T,
    forward: &Mapping<'_>,
    inverse: &Mapping<'_>,
    relocate: impl Fn(&Mapping<'_>, &mut T) -> Result<(), AssemblyError>,
) -> Result<(), AssemblyError> {
    let original = value.clone();
    relocate(forward, value)?;
    let mut restored = value.clone();
    relocate(inverse, &mut restored)?;
    if restored != original {
        return Err(AssemblyError::Invalid(
            "shared source relocation changed undeclared data",
        ));
    }
    Ok(())
}

pub(super) fn replace_application(
    reference: &mut Envelope,
    application: ApplicationPlan,
    manifest: &Manifest,
    counts: Counts,
) -> Result<(), AssemblyError> {
    let old = manifest.reference();
    let source = &mut reference.source;
    let start = manifest.source_relocation.application_row_start;
    let end = manifest
        .source_relocation
        .next_preimage_row_start
        .eval(old)?;
    let private_start = manifest.source_relocation.source_private_start;
    let private_end = manifest.source_relocation.reference_constant;
    source.rows.retain(|row| !(start..end).contains(&row.index));
    source
        .instructions
        .retain(|instruction| !(start..end).contains(&instruction.row));
    source
        .batches
        .retain(|batch| !(private_start..private_end).contains(&batch.start));

    let forward = Mapping {
        manifest,
        from: old,
        to: counts,
    };
    let inverse = Mapping {
        manifest,
        from: counts,
        to: old,
    };
    for chain in &mut source.hash_chains {
        checked_relocation(chain, &forward, &inverse, |m, value| {
            m.column(&mut value.input_start)?;
            m.column(&mut value.witness_start)?;
            m.column(&mut value.digest_start)
        })?;
    }
    for invocation in &mut source.permutation_invocations {
        checked_relocation(invocation, &forward, &inverse, |m, value| {
            m.column(&mut value.witness_start)?;
            for input in &mut value.inputs {
                m.combination(input)?;
            }
            Ok(())
        })?;
    }
    for invocation in &mut source.compact_invocations {
        checked_relocation(invocation, &forward, &inverse, |m, value| {
            m.column(&mut value.local_start)?;
            for input in &mut value.inputs {
                if input.input_count > 0 {
                    let last = input
                        .column_start
                        .checked_add(
                            input
                                .column_stride
                                .checked_mul(input.input_count - 1)
                                .ok_or(AssemblyError::Overflow)?,
                        )
                        .ok_or(AssemblyError::Overflow)?;
                    let boundary = m.manifest.source_relocation.source_private_start;
                    if input.column_start < boundary && last >= boundary {
                        return Err(AssemblyError::Invalid("compact input crosses application insertion"));
                    }
                }
                m.column(&mut input.column_start)?;
            }
            Ok(())
        })?;
    }
    for batch in &mut source.batches {
        checked_relocation(batch, &forward, &inverse, |m, value| m.batch(value))?;
    }
    for instruction in &mut source.instructions {
        checked_relocation(instruction, &forward, &inverse, |m, value| {
            m.row(&mut value.row)?;
            m.column(&mut value.target)?;
            m.combination(&mut value.a)?;
            m.combination(&mut value.b)
        })?;
    }
    for row in &mut source.rows {
        checked_relocation(row, &forward, &inverse, |m, value| {
            m.row(&mut value.index)?;
            m.combination(&mut value.a)?;
            m.combination(&mut value.b)?;
            m.combination(&mut value.c)
        })?;
    }
    checked_relocation(&mut source.layout, &forward, &inverse, |m, value| m.layout(value))?;
    for expression in &mut reference.assignment.digest_expressions {
        checked_relocation(expression, &forward, &inverse, |m, value| m.expression(value))?;
    }
    let split = source.rows.partition_point(|row| row.index < start);
    source
        .rows
        .splice(split..split, application.rows.iter().cloned());
    source.batches.extend(application.batches.iter().cloned());
    source
        .instructions
        .extend(application.instructions.iter().cloned());
    source.relation.rows = manifest.geometry.logical_rows.eval(counts)?;
    source.relation.columns = manifest.geometry.logical_width.eval(counts)?;
    source.terminal = json!([1, [0, source.relation.rows, 16, 1]]);
    reference.next_preimage = [
        manifest
            .source_relocation
            .next_preimage_row_start
            .eval(counts)?,
        manifest.source_relocation.next_preimage_row_count,
    ];
    reference.application = application;
    Ok(())
}
