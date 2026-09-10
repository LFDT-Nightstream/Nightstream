const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {counts, scenario} = require('../assurance.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../requirements.json')));
const leaves = data.nodes.filter(n => n.kind === 'leaf');
for (const axis of ['proof', 'connection', 'rust']) {
  assert.equal(Object.values(counts(leaves, axis)).reduce((a, b) => a + b), leaves.filter(n => n.origin !== 'out_of_scope').length);
}
assert.equal(counts([{origin: 'paper', rust: 'implemented'}], 'rust').tested, 0);
const result = scenario(data.error_budget, '100000000');
assert.equal(result.numerator, 13257n);
assert.equal(result.denominator, 18446744069414584321n ** 2n);
assert.ok(Math.abs(result.bits - 87.7301085) < 0.000001);
assert.equal(scenario(data.error_budget, '1' + '0'.repeat(100)).bound, 1);
for (const input of ['0', '-1', '1.5', '1e8', '', 'NaN']) assert.throws(() => scenario(data.error_budget, input));
console.log('JavaScript status and error-scenario checks passed.');
