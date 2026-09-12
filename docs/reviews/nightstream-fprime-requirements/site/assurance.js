/* Views of recorded assumptions, deployment scenarios, readiness and evidence. */
const RequirementAssurance = (() => {
  const categories = {
    proof: [['proved', 'Proved'], ['assumed', 'Assumed'], ['open', 'Open'], ['not_applicable', 'N/A']],
    connection: [['connected', 'Connected'], ['open', 'Open'], ['not_applicable', 'N/A']],
    rust: [['tested', 'Scoped tests'], ['implemented', 'Code only'], ['recorded', 'Recorded'], ['open', 'Open'], ['not_applicable', 'N/A']]
  };
  function category(axis, status) {
    if (status === 'not_required') return 'not_applicable';
    if (axis === 'proof') return ({proved: 'proved', assumption: 'assumed'})[status] || 'open';
    if (axis === 'connection') return status === 'connected' ? 'connected' : 'open';
    return ({tested_scoped: 'tested', implemented: 'implemented', recorded_only: 'recorded'})[status] || 'open';
  }
  function counts(leaves, axis) {
    const result = Object.fromEntries(categories[axis].map(([key]) => [key, 0]));
    for (const node of leaves) if (node.origin !== 'out_of_scope') result[category(axis, node[axis])]++;
    return result;
  }
  function progress(axis, values) {
    const finished = axis === 'proof' ? values.proved : axis === 'connection' ? values.connected : values.tested + values.implemented;
    const total = Object.entries(values).reduce((sum, [key, count]) => sum + (['assumed', 'not_applicable'].includes(key) ? 0 : count), 0);
    return {finished, total};
  }
  function scenario(budget, text) {
    if (!/^\d+$/.test(text) || BigInt(text) < 1n) throw Error('Enter a positive whole number.');
    const p = budget.parameters;
    const joint = p.running_sources * p.coefficient_lanes * (p.matrices + 1) + 2 * p.fresh_sources + p.running_sources;
    const numerator = BigInt(p.rounds * p.message_width + joint - 1 + p.rounds);
    const denominator = BigInt(p.q) ** 2n;
    const sum = BigInt(text) * numerator;
    const bound = sum >= denominator ? 1 : Number(sum) / Number(denominator);
    return {numerator, denominator, bound, bits: -Math.log2(bound)};
  }
  function build(data, publication, references) {
    const byId = new Map(data.nodes.map(n => [n.id, n]));
    const el = (tag, text, cls) => {
      const node = document.createElement(tag);
      if (text !== undefined) node.textContent = text;
      if (cls) node.className = cls;
      return node;
    };
    const link = (text, href) => { const a = el('a', text); a.href = href; return a; };
    const value = v => v === null || v === undefined ? 'Not specified' : String(v);
    const sourceUrl = (path, line) => {
      if (references.locations?.[path]?.repository_committed === false) return null;
      return data.provenance.repository + '/blob/' + data.provenance.code_commit + '/' + path.split('/').map(encodeURIComponent).join('/') + (line ? '#L' + line : '');
    };
    function fields(parent, entries) {
      const dl = el('dl', undefined, 'facts');
      for (const [name, v] of entries) dl.append(el('dt', name), el('dd', value(v)));
      parent.append(dl);
    }
    function recordLinks(parent, ids, title = 'Requirement records') {
      if (!ids?.length) return;
      const box = el('div', undefined, 'uses'); box.append(el('strong', title));
      for (const id of ids || []) box.append(link(byId.get(id).label, '#req-' + id));
      parent.append(box);
    }
    function premiseLink(id) {
      const premise = data.assumptions.find(a => a.id === id);
      return link(premise.label, '#assumption-' + id);
    }

    const readiness = el('div');
    readiness.append(el('p', 'These checks describe the two intended uses of the proof work. They do not add requirement leaves or award phase completion. A passed check is limited to its stated evidence.', 'view-summary'));
    const panels = el('div', undefined, 'readiness-grid');
    for (const panel of data.readiness) {
      const box = el('section', undefined, 'panel');
      box.append(el('h2', panel.title), el('p', panel.description));
      const list = el('ul', undefined, 'check-list');
      for (const item of panel.items) {
        const row = el('li');
        const title = el('div', undefined, 'check-title');
        title.append(el('strong', item.label), el('span', ({passed: 'Passed in scope', assumed: 'Assumed', open: 'Open'})[item.status], 'status status-' + (item.status === 'passed' ? 'good' : item.status === 'assumed' ? 'assumption' : 'open')));
        row.append(title, el('p', item.requirement), el('p', item.evidence, 'muted'));
        recordLinks(row, item.records); list.append(row);
      }
      box.append(list); panels.append(box);
    }
    readiness.append(panels);

    const assumptions = el('div');
    const assumed = data.nodes.filter(n => n.kind === 'leaf' && n.proof === 'assumption' && n.origin !== 'out_of_scope').length;
    assumptions.append(el('p', assumed + ' assumption-status records refer to ' + data.assumptions.length + ' ledger entries. Repeated uses of one premise are linked here; they are not independent security assumptions.', 'view-summary'));
    for (const entry of data.assumptions) {
      const card = el('details', undefined, 'assumption-card'); card.id = 'assumption-' + entry.id;
      const summary = el('summary');
      summary.append(el('strong', entry.label), el('span', entry.kind, 'status'));
      const body = el('div', undefined, 'panel-body');
      body.append(el('p', entry.statement));
      fields(body, Object.entries(entry.parameters).map(([k, v]) => [k.replaceAll('_', ' '), v]));
      fields(body, [['Approval state', entry.approval.state], ['Recorded by', entry.approval.by], ['Approval date', entry.approval.date]]);
      body.append(el('p', entry.approval.qualification));
      if (entry.estimates) {
        for (const estimate of entry.estimates) body.append(el('p', estimate.name + ': log₂ cost ' + estimate.log2_cost.toFixed(2)));
      }
      if (entry.note) body.append(el('p', entry.note, 'muted'));
      const url = sourceUrl(entry.approval.source);
      body.append(url ? link('Assumption or scope record', url) : el('code', entry.approval.source));
      recordLinks(body, entry.records, 'Assumption-status records');
      const direct = data.nodes.filter(n => n.assumption_ids?.includes(entry.id) && !entry.records.includes(n.id)).map(n => n.id);
      recordLinks(body, [...new Set([...direct, ...(entry.inherited_by || [])])], 'Recorded dependent uses');
      if (entry.depends_on) {
        const deps = el('div', undefined, 'uses'); deps.append(el('strong', 'Uses other premises'));
        for (const id of entry.depends_on) deps.append(premiseLink(id)); body.append(deps);
      }
      card.append(summary, body); assumptions.append(card);
    }

    const risk = el('div'); const budget = data.error_budget;
    risk.append(el('p', budget.not_a_total_bound, 'scope-notice'), el('h2', 'Interactive algebraic test term'), el('p', budget.event), el('p', budget.formula, 'formula'));
    const perTest = scenario(budget, '1');
    risk.append(el('p', 'Per specified test: ' + perTest.numerator + ' / ' + perTest.denominator + '. Calculated decimal and bit values are rounded.', 'muted'));
    const form = el('form', undefined, 'risk-form');
    const label = el('label', 'Number of specified tests'); label.htmlFor = 'risk-uses';
    const input = el('input'); input.id = 'risk-uses'; input.type = 'text'; input.inputMode = 'numeric'; input.value = '';
    input.setAttribute('aria-describedby', 'risk-scenario-note');
    const output = el('output'); output.setAttribute('for', 'risk-uses'); output.setAttribute('aria-live', 'polite');
    const calculate = () => {
      if (!input.value.trim()) {
        input.setCustomValidity('');
        output.textContent = 'Enter a use count to calculate the conditional bound.';
        return;
      }
      try {
        const result = scenario(budget, input.value.trim());
        input.setCustomValidity('');
        output.textContent = result.bound === 1 ? 'Upper bound: 1. This bound is uninformative at this use count.' : 'Sum of test-error bounds: ' + result.bound.toExponential(4) + ' (approximately 2^−' + result.bits.toFixed(2) + ').';
      } catch (error) { input.setCustomValidity(error.message); output.textContent = error.message; }
    };
    form.addEventListener('submit', event => { event.preventDefault(); calculate(); }); input.addEventListener('input', calculate);
    form.append(label, input, output); risk.append(form);
    const note = el('p', budget.scenario_note, 'muted'); note.id = 'risk-scenario-note'; risk.append(note, el('p', budget.accumulation)); calculate();
    recordLinks(risk, [budget.owner_record], 'Full error-budget obligation');
    risk.append(el('h2', 'Parameters still needed for a deployment claim'));
    fields(risk, budget.deployment_parameters.map(p => [p.name, p.value]));
    const terms = el('ul', undefined, 'check-list');
    for (const term of budget.missing_terms) {
      const row = el('li'); row.append(el('strong', term.label), el('p', term.symbol), premiseLink(term.assumption)); terms.append(row);
    }
    risk.append(terms, el('h2', 'Knowledge extraction is a separate claim'), el('p', budget.extraction_formula, 'formula'), el('p', budget.extraction_note));
    risk.append(el('h2', 'Setup and attack-cost accounting'), el('p', 'The ideal modular-reduction calculation belongs to its setup model. It is not a new random event at every fold. The 153.20 and 114.22 log₂ attack-cost estimates use different units; neither is a per-fold failure probability.'));
    risk.append(premiseLink('ideal_setup_inputs'), el('span', ' · '), premiseLink('fixed_matrix_msis'));

    const evidence = el('div');
    evidence.append(el('p', 'Code, map content and validation evidence have separate revisions. A successful website build does not rerun or approve protocol conformance.', 'view-summary'));
    fields(evidence, [['Protocol code commit', data.provenance.code_commit], ['Map commit', publication.map_commit], ['Map source state', publication.source_state], ['Map SHA-256', publication.map_sha256], ['Reference checks at', references.checked_at], ['Reference-check scope', references.scope]]);
    evidence.append(link('Protocol code on GitHub', data.provenance.repository + '/tree/' + data.provenance.code_commit), el('span', ' · '), link('Publication metadata', 'publication.json'), el('span', ' · '), link('Reference-check record', 'reference-report.json'));
    for (const run of data.provenance.evidence) {
      const card = el('section', undefined, 'panel'); card.append(el('h2', run.label));
      fields(card, [['Checked code', run.code_commit], ['Performed by', run.performed_by], ['Date', run.date], ['Authority', run.authority]]);
      card.append(el('p', run.scope), link('Report', sourceUrl(run.report)), el('span', ' · '), link('Retained archive', sourceUrl(run.archive))); evidence.append(card);
    }
    evidence.append(el('p', 'Required independent approvals remain open. Publishing requires a clean committed source snapshot and reruns the reference check. No secret checker key or self-issued approval is created by this site.', 'scope-notice'));
    return {views: {readiness, assumptions, risk, evidence}, sourceUrl, premiseLink};
  }
  return {categories, category, counts, progress, scenario, build};
})();
if (typeof module !== 'undefined') module.exports = RequirementAssurance;
