const test = require('node:test');
const assert = require('node:assert/strict');
const diagram = require('../proof-map.json');
const {proofMapEdges} = require('../proof-map.js');

function proofMapTrace(diagram, selected) {
  const walk = reverse => {
    const found = new Set([selected]), pending = [selected];
    while (pending.length) {
      const id = pending.pop();
      for (const edge of diagram.edges) {
        if ((reverse ? edge.target : edge.source) !== id) continue;
        const next = reverse ? edge.source : edge.target;
        if (!found.has(next)) { found.add(next); pending.push(next); }
      }
    }
    return found;
  };
  return {inputs: walk(true), outputs: walk(false)};
}

test('history security uses its actual premises without depending on native completeness', () => {
  const trace = proofMapTrace(diagram, 'probability');
  for (const id of ['accept', 'children', 'extraction', 'predecessor', 'history', 'visited', 'fs', 'msis']) assert.ok(trace.inputs.has(id), id);
  for (const id of ['native', 'extension', 'encoding']) assert.ok(!trace.inputs.has(id), id);
  assert.ok(trace.outputs.has('deployed'));
  assert.equal(diagram.edges.find(e => e.source === 'actual-success' && e.target === 'children').kind, 'supplies');
  assert.equal(diagram.edges.find(e => e.source === 'probability' && e.target === 'deployed').kind, 'open');
});

test('selection isolates direct connections; the full graph remains available', () => {
  const selected = proofMapEdges(diagram, 'rows', false);
  assert.deepEqual(selected.map(e => [e.source, e.target]).sort(),
    [['accept', 'rows'], ['rows', 'ccs'], ['rows', 'dec'], ['rows', 'rlc']].sort());
  assert.equal(proofMapEdges(diagram, null, true).length, diagram.edges.length);
  assert.ok(proofMapEdges(diagram, null, false).length < diagram.edges.length);
  assert.ok(proofMapEdges(diagram, 'extraction', false).some(e => e.source === 'msis'));
});
