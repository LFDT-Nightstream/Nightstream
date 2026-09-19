const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {techAssembly, techTreeRoute} = require('../tech-tree.js');
const {nodes} = JSON.parse(readFileSync(new URL('../requirements.json', 'file://' + __filename), 'utf8'));

test('explicit tree and saved group links use the tech tree, including scope exclusions', () => {
  const byId = new Map(nodes.map(node => [node.id, node]));
  for (const fragment of ['group-all', 'tech-tree']) {
    assert.deepEqual(techTreeRoute(fragment, byId), {id: 'root'});
  }
  for (const group of nodes.filter(node => node.parent === 'root')) {
    assert.deepEqual(techTreeRoute('group-' + group.id, byId), {id: group.id});
  }
  for (const node of nodes) {
    const route = techTreeRoute('req-' + node.id, byId);
    assert.equal(route.recordId, node.id);
    let owner = node;
    while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
    assert.equal(route.proofScopeId, owner.id);
  }
  assert.deepEqual(techTreeRoute('proof-C:C.security.probability', byId),
    {id: 'C.security.probability', proofScopeId: 'C'});
});

test('the full picture contains every Stage 1 group exactly once', () => {
  const expected = nodes.filter(node => node.id === 'root' ||
    (node.parent === 'root' && node.origin !== 'out_of_scope')).map(node => node.id).sort();
  assert.deepEqual(techAssembly.map(node => node.id).sort(), expected);
  assert.deepEqual(techAssembly.filter(node => node.row === 0).map(node => node.id), ['root']);
});

test('all support comes from a lower row and reaches the end goal, with no circular unlocks', () => {
  const byId = new Map(techAssembly.map(node => [node.id, node]));
  const reached = new Set();
  function visit(id) {
    reached.add(id);
    const parent = byId.get(id);
    for (const child of parent.needs) {
      assert(byId.has(child), 'Unknown group: ' + child);
      assert(byId.get(child).row > parent.row, child + ' must be below ' + id);
      if (!reached.has(child)) visit(child);
    }
  }
  visit('root');
  assert.equal(reached.size, techAssembly.length);
  for (const phase of ['C', 'R', 'D']) assert(byId.get('N').needs.includes(phase));
  assert(!byId.get('C').needs.includes('R'));
  assert(!byId.get('R').needs.includes('C'));
});

test('cards do not overlap within a row and every group opens an existing requirement hierarchy', () => {
  const occupied = new Set();
  for (const item of techAssembly) {
    const group = nodes.find(node => node.id === item.id);
    assert.equal(group.kind, 'group');
    assert(nodes.some(node => node.parent === item.id), 'No parts for ' + item.id);
    for (const column of [item.column, item.column + 1]) {
      const cell = item.row + ':' + column;
      assert(!occupied.has(cell), 'Overlapping cards at ' + cell);
      occupied.add(cell);
    }
  }
});
