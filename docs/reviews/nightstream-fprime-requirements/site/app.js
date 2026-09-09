(() => {
  const data = JSON.parse(document.getElementById('requirements-data').textContent);
  const nodes = data.nodes;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map();
  for (const node of nodes) {
    if (node.id === 'root') continue;
    if (!children.has(node.parent)) children.set(node.parent, []);
    children.get(node.parent).push(node);
  }
  const names = {
    proved: 'local proof exists', partial: 'partial', definition: 'defined',
    assumption: 'explicit assumption', not_required: 'not required here',
    not_reviewed: 'not verified by this review', connected: 'connected locally',
    open: 'open', tested_scoped: 'scoped tests passed', recorded_only: 'recorded evidence only',
    implemented: 'code exists'
  };
  const origins = {
    paper: 'Paper requirement', profile: 'Nightstream profile',
    implementation: 'Implementation requirement', assumption: 'External assumption',
    out_of_scope: 'Outside selected Stage 1'
  };
  const navNames = {
    F: 'Foundations', T: 'Hash & transcript', C: 'PiCCS', R: 'PiRLC', D: 'PiDEC',
    N: 'NIFS composition', H: 'HyperNova recursion', L: 'Circuit & export',
    P: 'Production acceptance', O: 'Outside Stage 1'
  };
  const sourceRoot = '/Users/nicarq/starstream/develop/nightstream-clean-up/';
  const el = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const descendants = id => (children.get(id) || []).flatMap(node => node.kind === 'leaf' ? [node] : descendants(node.id));
  const unresolved = node => node.connection === 'open' || node.connection === 'partial' || node.proof === 'partial';
  const tone = status => ['open', 'partial'].includes(status) ? 'open' : status === 'assumption' ? 'assumption' : ['proved', 'connected', 'tested_scoped'].includes(status) ? 'good' : 'neutral';
  const groups = children.get('root');
  const included = nodes.filter(node => node.kind === 'leaf' && node.origin !== 'out_of_scope').length;
  const excluded = nodes.filter(node => node.kind === 'leaf' && node.origin === 'out_of_scope').length;
  const axes = [
    {key: 'proof', label: 'Proof', complete: ['proved']},
    {key: 'connection', label: 'Link', complete: ['connected']},
    {key: 'rust', label: 'Rust', complete: ['implemented', 'tested_scoped']}
  ];
  const progress = new Map(nodes.filter(node => node.kind === 'group').map(node => {
    const leaves = descendants(node.id).filter(leaf => leaf.origin !== 'out_of_scope');
    return [node.id, axes.map(axis => {
      const applicable = leaves.filter(leaf => !['not_required', 'assumption'].includes(leaf[axis.key]));
      const finished = applicable.filter(leaf => axis.complete.includes(leaf[axis.key])).length;
      return {...axis, finished, total: applicable.length};
    })];
  }));
  const fraction = axis => axis.total ? axis.finished + '/' + axis.total : 'n/a';
  function countBadges(id) {
    const counts = el('span', undefined, 'axis-counts');
    for (const axis of progress.get(id)) {
      const badge = el('span', undefined, 'axis-count axis-' + axis.key);
      badge.append(el('span', axis.label), el('strong', fraction(axis)));
      badge.setAttribute('aria-label', axis.total ? axis.label + ': ' + axis.finished + ' of ' + axis.total + ' applicable requirements finished' : axis.label + ': no applicable requirements');
      counts.append(badge);
    }
    return counts;
  }
  const stats = node => {
    const leaves = descendants(node.id);
    if (node.origin === 'out_of_scope') return leaves.length + ' scope exclusions';
    const open = leaves.filter(unresolved).length;
    const assumptions = leaves.filter(item => item.proof === 'assumption').length;
    return leaves.length + ' requirements' + (open ? ' · ' + open + ' unresolved proof/link ' + (open === 1 ? 'item' : 'items') : '') + (assumptions ? ' · ' + assumptions + ' explicit ' + (assumptions === 1 ? 'assumption' : 'assumptions') : '');
  };
  function pair(box, label, value, className = '') {
    if (!value) return;
    const dl = el('dl', undefined, 'req-pair ' + className);
    dl.append(el('dt', label), el('dd', value));
    box.append(dl);
  }
  function anchorActions(node) {
    const row = el('div', undefined, 'req-anchor-row');
    const anchor = el('a', '#req-' + node.id, 'req-anchor');
    anchor.href = '#req-' + node.id;
    anchor.setAttribute('aria-label', 'Permanent link to ' + node.label);
    const copy = el('button', 'Copy link', 'copy-link');
    copy.type = 'button';
    copy.setAttribute('aria-label', 'Copy link to ' + node.label);
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(anchor.href);
        copy.textContent = 'Copied';
        document.getElementById('view-announcement').textContent = 'Link copied for ' + node.label;
      } catch {
        let field = row.querySelector('input');
        if (!field) {
          field = el('input', undefined, 'copy-link-fallback');
          field.type = 'text';
          field.readOnly = true;
          field.setAttribute('aria-label', 'Select and copy this item link');
          row.append(field);
        }
        field.value = anchor.href;
        field.focus();
        field.select();
        document.getElementById('view-announcement').textContent = 'Select and copy the displayed link.';
      }
    });
    row.append(el('span', 'Linked item', 'target-indicator'), anchor, copy);
    return row;
  }
  function render(node) {
    const li = el('li');
    if (node.parent === 'root') li.dataset.root = node.id;
    const details = el('details');
    details.id = 'req-' + node.id;
    const summary = el('summary');
    summary.append(el('span', node.label, 'req-title'));
    if (node.kind === 'group' && node.origin !== 'out_of_scope') summary.append(countBadges(node.id));
    const state = el('span', undefined, 'req-state');
    if (node.kind === 'group') state.textContent = stats(node);
    else if (node.origin === 'out_of_scope') state.textContent = 'Outside selected Stage 1';
    else for (const [axis, label] of [['proof', 'Proof'], ['connection', 'Link'], ['rust', 'Rust']]) {
      state.append(el('span', label + ': ' + names[node[axis]], 'status status-' + tone(node[axis])));
    }
    summary.append(state);
    details.append(summary);
    const content = el('div', undefined, 'req-detail');
    content.append(anchorActions(node), el('span', origins[node.origin], 'req-id'));
    pair(content, 'Requirement', node.requirement);
    pair(content, 'Remaining', node.remaining, 'remaining' + (unresolved(node) ? ' remaining-open' : ''));
    const references = [...(node.paper || []), ...(node.code || [])];
    if (references.length) {
      const evidence = el('details', undefined, 'req-sources');
      evidence.append(el('summary', 'Paper & code references (' + references.length + ')'));
      const refs = el('div', undefined, 'req-evidence');
      for (const source of references) {
        refs.append(el('div', source.section || source.symbol || source.path, 'source-name'));
        refs.append(el('div', sourceRoot + source.path + ':' + source.line, 'source-path'));
      }
      evidence.append(refs);
      content.append(evidence);
    }
    if (node.depends_on?.length) {
      const uses = el('div', undefined, 'uses');
      uses.append(el('strong', 'Uses'));
      for (const id of node.depends_on) {
        const link = el('a', byId.get(id).label);
        link.href = '#req-' + id;
        uses.append(link);
      }
      content.append(uses);
    }
    details.append(content);
    if (children.has(node.id)) {
      const list = el('ul', undefined, 'req-children');
      children.get(node.id).forEach(child => list.append(render(child)));
      details.append(list);
    }
    li.append(details);
    return li;
  }
  const tree = document.getElementById('requirements-tree');
  const list = el('ul', undefined, 'req-roots');
  groups.forEach(node => list.append(render(node)));
  tree.append(list);
  const navigation = document.getElementById('group-nav');
  const select = document.getElementById('group-select');
  const navigationItems = [{id: 'all', label: 'All groups'}, ...groups];
  for (const node of navigationItems) {
    const label = navNames[node.id] || node.label;
    const link = el('a', undefined, 'nav-link');
    link.href = '#group-' + node.id;
    link.dataset.group = node.id;
    link.append(el('span', node.id === 'all' ? '—' : node.id, 'nav-id'), el('span', label));
    link.append(node.origin === 'out_of_scope' ? el('span', descendants(node.id).length + ' excluded', 'nav-count') : countBadges(node.id === 'all' ? 'root' : node.id));
    navigation.append(link);
    const countText = node.origin === 'out_of_scope' ? 'excluded' : progress.get(node.id === 'all' ? 'root' : node.id).map(axis => axis.label + ' ' + fraction(axis)).join(' · ');
    const option = el('option', label + ' — ' + countText);
    option.value = node.id;
    select.append(option);
  }
  document.getElementById('source-meta').textContent = 'SuperNeo v1.1 + HyperNova · Base ' + data.commit.slice(0, 8) + (data.source_note ? ' · ' + data.source_note : '');
  function showGroup(id) {
    const node = byId.get(id);
    const all = id === 'all' || !groups.includes(node);
    const selected = all ? 'all' : id;
    tree.classList.toggle('single-group', !all);
    for (const item of list.children) {
      item.hidden = !all && item.dataset.root !== id;
      if (!all && item.dataset.root === id) item.querySelector('details').open = true;
    }
    for (const link of navigation.children) {
      if (link.dataset.group === selected) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    select.value = selected;
    const title = all ? 'Requirements map' : node.label;
    document.getElementById('view-title').textContent = title;
    document.getElementById('view-progress').replaceChildren(...(!all && node.origin === 'out_of_scope' ? [] : [countBadges(all ? 'root' : id)]));
    document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / ' + (all ? 'All groups' : navNames[id]);
    document.getElementById('view-summary').textContent = all ? included + ' indexed requirements · ' + excluded + ' scope exclusions. Expand a group to inspect its primitives and proof connections.' : stats(node);
    document.getElementById('view-announcement').textContent = title;
    document.title = title + ' — Nightstream';
  }
  function navigate() {
    const fragment = decodeURIComponent(location.hash.slice(1));
    if (fragment.startsWith('req-') && byId.has(fragment.slice(4))) {
      const id = fragment.slice(4);
      let owner = byId.get(id);
      while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
      showGroup(owner.id);
      const target = document.getElementById('req-' + id);
      if (!target) return;
      let current = target;
      while (current && current !== tree) {
        if (current.tagName === 'DETAILS') current.open = true;
        current = current.parentElement;
      }
      target.querySelector('.req-anchor').focus({preventScroll: true});
      target.scrollIntoView({block: 'start'});
    } else {
      showGroup(fragment.startsWith('group-') ? fragment.slice(6) : 'all');
      if (fragment.startsWith('group-')) window.scrollTo({top: 0});
    }
  }
  select.addEventListener('change', () => { location.hash = 'group-' + select.value; });
  window.addEventListener('hashchange', navigate);
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#req-"]');
    if (anchor && anchor.hash === location.hash) { event.preventDefault(); navigate(); }
  });
  navigate();
})();
