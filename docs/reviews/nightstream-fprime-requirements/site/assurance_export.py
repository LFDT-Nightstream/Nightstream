"""Static counterparts of the assurance views, from the same recorded data."""

import re

from site_model import error_scenario


def text(value):
    if value is None:
        return 'Not specified'
    return re.sub(r'([\\`*_\[\]<>|])', r'\\\1', str(value)).replace('\n', ' ')


def export_assurance(data, publication, references):
    if not data.get('assumptions'):
        return {}
    by_id = {n['id']: n for n in data['nodes']}

    def records(ids):
        links = []
        for node_id in ids:
            group = by_id[node_id]
            while group['parent'] not in {'root', None}:
                group = by_id[group['parent']]
            path = 'requirements.md' if node_id == 'root' else 'markdown/' + group['id'] + '.md'
            links.append(f'[{node_id}]({path}#req-{node_id})')
        return '; '.join(links) or 'None recorded.'

    def source(path):
        return data['provenance']['repository'] + '/blob/' + data['provenance']['code_commit'] + '/' + path

    def intro(title):
        return ['# ' + title, '', '[Requirement index](requirements.md) · [Source JSON](requirements.json)', '']

    assumptions = intro('Assumption ledger')
    count = sum(n.get('proof') == 'assumption' and n['kind'] == 'leaf' and n['origin'] != 'out_of_scope' for n in data['nodes'])
    assumptions += [f'{count} assumption-status records use {len(data["assumptions"])} ledger entries. Shared uses of one premise are not independent assumptions.', '']
    for entry in data['assumptions']:
        assumptions += [f'## <a id="assumption-{entry["id"]}"></a>{text(entry["label"])}', '',
                        '- Kind: ' + text(entry['kind']), '- Statement: ' + text(entry['statement'])]
        for key, value in entry['parameters'].items():
            assumptions.append('- ' + text(key.replace('_', ' ')) + ': ' + text(value))
        approval = entry['approval']
        assumptions += ['- Approval: ' + text(approval['state']) + '; by ' + text(approval['by']) + '; date ' + text(approval['date']) + '.',
                        '- Approval scope: ' + text(approval['qualification']),
                        '- Source: [Recorded premise](' + source(approval['source']) + ')']
        for estimate in entry.get('estimates', []):
            assumptions.append(f'- {text(estimate["name"])}: log₂ cost {estimate["log2_cost"]:.2f}.')
        if entry.get('note'):
            assumptions.append('- Qualification: ' + text(entry['note']))
        direct = [n['id'] for n in data['nodes'] if entry['id'] in n.get('assumption_ids', []) and n['id'] not in entry['records']]
        uses = list(dict.fromkeys(direct + entry.get('inherited_by', [])))
        assumptions += ['- Assumption-status records: ' + records(entry['records']), '- Recorded dependent uses: ' + records(uses)]
        if entry.get('depends_on'):
            assumptions.append('- Other premises: ' + '; '.join(f'[{p}](#assumption-{p})' for p in entry['depends_on']))
        assumptions.append('')

    readiness = intro('Readiness for the intended uses')
    readiness += ['These checks reuse requirement records. They do not add proof credit or grant phase closure. Passed checks apply only to their recorded scope.', '']
    for panel in data['readiness']:
        readiness += ['## ' + text(panel['title']), '', text(panel['description']), '']
        for item in panel['items']:
            readiness += ['### ' + text(item['label']), '', '- Status: `' + item['status'] + '`.',
                          '- Required: ' + text(item['requirement']), '- Evidence: ' + text(item['evidence']),
                          '- Records: ' + records(item['records']), '']

    budget = data['error_budget']
    result = error_scenario(budget, '1')
    risk = intro('Conditional error bounds')
    risk += [text(budget['not_a_total_bound']), '', '## Specified interactive test', '', text(budget['event']), '',
             text(budget['formula']), '', f'Per test: `{result["numerator"]} / {result["denominator"]}`.', '',
             text(budget['scenario_note']), '', text(budget['accumulation']), '',
             'Full obligation: ' + records([budget['owner_record']]), '', '## Deployment parameters', '']
    risk += ['- ' + text(p['name']) + ': ' + text(p['value']) for p in budget['deployment_parameters']]
    risk += ['', '## Terms still needed for a total claim', '']
    for term in budget['missing_terms']:
        risk.append(f'- {text(term["label"])}: {text(term["symbol"])}. [Premise](assumptions.md#assumption-{term["assumption"]}).')
    risk += ['', '## Separate extraction claim', '', text(budget['extraction_formula']), '', text(budget['extraction_note']), '',
             '## Setup and attack-cost accounting', '',
             'The ideal modular-reduction calculation applies to its setup model, once per setup. It is not a per-fold failure term. The 153.20 and 114.22 log₂ attack-cost estimates use different units; neither is a failure probability.', '',
             '[Setup premise](assumptions.md#assumption-ideal_setup_inputs) · [MSIS premise](assumptions.md#assumption-fixed_matrix_msis)', '']

    evidence = intro('Source and evidence revisions')
    evidence += ['Code, map content and test evidence have separate revisions. Website checks do not grant protocol conformance.', '',
                 '- Protocol code: `' + data['provenance']['code_commit'] + '`.',
                 '- Map commit: ' + text(publication.get('map_commit')),
                 '- Map source state: ' + text(publication.get('source_state')),
                 '- Map SHA-256: ' + text(publication.get('map_sha256')),
                 '- Reference check date: ' + text(references.get('checked_at')),
                 '- Reference check scope: ' + text(references.get('scope')), '',
                 '[Publication metadata](publication.json) · [Reference-check record](reference-report.json)', '']
    for run in data['provenance']['evidence']:
        evidence += ['## ' + text(run['label']), '', '- Checked code: `' + run['code_commit'] + '`.',
                     '- Performed by: ' + text(run['performed_by']), '- Date: ' + text(run['date']),
                     '- Authority: ' + text(run['authority']), '- Scope: ' + text(run['scope']), '',
                     '[Report](' + source(run['report']) + ') · [Retained archive](' + source(run['archive']) + ')', '']
    evidence += ['Required independent approvals remain open. Publication requires committed source inputs and a new reference check. File hashes identify evidence; they do not establish proof meaning or conformance.', '']
    return {name + '.md': '\n'.join(lines) for name, lines in [
        ('assumptions', assumptions), ('readiness', readiness), ('error-budget', risk), ('evidence', evidence)]}
