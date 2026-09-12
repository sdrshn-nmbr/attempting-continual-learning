import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_tool_streams import need, rescore, sha


def tensor_record(value):
    if isinstance(value, torch.Tensor):
        return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k):tensor_record(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [tensor_record(v) for v in value]
    return value


def tensor_identity(value):
    return hashlib.sha256(json.dumps(tensor_record(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit_delete_rehearsal(root, wave, tokenizer):
    config=json.loads((root/'config.json').read_text())
    result=json.loads((root/'result.json').read_text())
    execution=json.loads((root/'execution.json').read_text())
    need(execution['status']=='completed' and execution['exit_code']==0,'delete rehearsal completion')
    source=wave/'runs'/config['source_task_id']
    reference=wave/'runs'/config['reference_task_id']
    compression=wave/'runs'/config['compression_task_id']
    for label,path in [('source',source),('reference',reference),('compression',compression)]:
        need(result[f'{label}_result_sha256']==sha(path/'result.json'),f'delete rehearsal {label} binding')
    for index,ancestor in enumerate(result['process_proof']['ancestors']):
        path=wave/'runs'/ancestor['task_id']/'execution.json'
        previous=json.loads(path.read_text())
        need(ancestor['execution_sha256']==sha(path)==sha(root/f'ancestor-{index}-execution.json'),'delete ancestor hash')
        need(previous['status']=='completed' and previous['finished_at']<=execution['started_at'],'delete ancestor completion')
        need(previous['attempt_id']!=execution['attempt_id'],'delete distinct execution')
    dataset=json.loads((source/'dataset.json').read_text())
    conventions=json.loads((source/'stream.json').read_text())['conventions']
    rows={r['id']:r for families in dataset.values() for splits in families.values() for examples in splits.values() for r in examples}
    panels={}
    for name,checksum in result['artifact_manifest'].items():
        path=root/name
        need(sha(path)==checksum,f'delete artifact {name}')
        if not name.endswith('.json'):
            continue
        panel=json.loads(path.read_text())
        if not isinstance(panel,dict) or set(panel)!={'metrics','records'}:
            continue
        for record in panel['records']:
            rescore(record,rows[record['id']],conventions,tokenizer,248046,248044)
        n=len(panel['records']); correct=sum(r['correct'] for r in panel['records'])
        need(panel['metrics']['n']==n and panel['metrics']['correct']==correct and panel['metrics']['accuracy']==correct/n,'delete aggregate')
        panels[name]=panel
    training=root if config['stage']=='train' else wave/'runs'/config['training_task_id']
    trained=json.loads((training/'result.json').read_text())
    arm=trained['methods']['agent_dice']['arms']['delete_rehearsal']
    initial=torch.load(compression/'agent_dice/checkpoint/state.pt',map_location='cpu',weights_only=True)['adapter']
    need(arm['initial_adapter_sha256']==tensor_identity(initial),'delete same compressed initial factors')
    replay=json.loads((reference/'result.json').read_text())['methods']['agent_dice']['arms']['compressed_replay']
    need(arm['initial_optimizer_sha256']==replay['initial_optimizer_sha256'],'delete same fresh optimizer')
    updates=[json.loads(line) for line in (training/'agent_dice/delete_rehearsal/updates.jsonl').read_text().splitlines()]
    original=[json.loads(line) for line in (reference/'agent_dice/compressed_replay/updates.jsonl').read_text().splitlines()]
    need([r['step'] for r in updates]==list(range(1,129)),'delete complete trace')
    for row,old in zip(updates,original,strict=True):
        need(row['current_ids']==old['current_ids'] and len(row['current_ids'])==2 and not row['replay_ids'],'delete exact current exposure order')
        need(row['loss_denominator']==4 and row['loss']==row['current_mean_loss']/2,'delete current loss coefficient')
        need(all(rows[i]['split']=='train' and rows[i]['kind']=='workflow' for i in row['current_ids']),'delete TRAIN only')
    saved=torch.load(training/arm['checkpoint']/'state.pt',map_location='cpu',weights_only=True)
    resident=0
    for layer in saved['adapter'].values():
        need(layer['offset'] is None and layer['A'].shape[0]==layer['B'].shape[1]==8,'delete fixed rank without offset')
        for key in ('A','B'):
            need(layer[key].dtype==torch.float32 and bool(torch.isfinite(layer[key]).all()),'delete finite FP32')
            resident+=layer[key].numel()*layer[key].element_size()
    need(resident==3670016,'delete resident bytes')
    need({int(v['step']) for v in saved['optimizer']['state'].values()}=={128},'delete optimizer clocks')
    details={'updates':128,'current_exposures':256,'old_exposures':0,'resident_adapter_bytes':resident}
    if config['stage']=='audit':
        measured=result['methods']['agent_dice']['arms']['delete_rehearsal']
        need(measured['status']=='fresh_process_audited' and measured['test_evaluated'],'delete final process')
        for name,metrics in measured['dev'].items():
            panel=panels[f'agent_dice/delete_rehearsal/{name}.json']
            need(panel==json.loads((training/f'agent_dice/delete_rehearsal/dev-128/{name}.json').read_text()),'delete DEV reload exact')
            need(metrics==panel['metrics'],'delete DEV report binding')
        for name,metrics in measured['test'].items():
            need(metrics==panels[f'agent_dice/delete_rehearsal/{name}.json']['metrics'],'delete TEST report binding')
        details['test']={k:[v['correct'],v['n']] for k,v in measured['test'].items()}
    return {'status':'passed','stage':config['stage'],'records':sum(len(p['records']) for p in panels.values()),'panels':len(panels),'source_sha256':execution['source_sha256'],'result_sha256':sha(root/'result.json'),'details':details,'scope':'Posthoc old-loss ablation on one calendar stream. Current rows, coefficient and update count matched; batch shape, clipping and Adam interactions remain part of the intervention.'}


def audit_hybrid(root, wave, tokenizer):
    config=json.loads((root/'config.json').read_text())
    result=json.loads((root/'result.json').read_text())
    execution=json.loads((root/'execution.json').read_text())
    need(execution['status']=='completed' and execution['exit_code']==0,'hybrid completion')
    source=wave/'runs'/config['source_task_id']
    compression=wave/'runs'/config['compression_task_id']
    need(result['source_result_sha256']==sha(source/'result.json'),'hybrid source binding')
    need(result['compression_result_sha256']==sha(compression/'result.json'),'hybrid compression binding')
    need(result['source_archives_preserved'] and result['no_hidden_dense_offset'],'hybrid preservation claims')
    for index,ancestor in enumerate(result['process_proof']['ancestors']):
        path=wave/'runs'/ancestor['task_id']/'execution.json'
        previous=json.loads(path.read_text())
        need(ancestor['execution_sha256']==sha(path)==sha(root/f'ancestor-{index}-execution.json'),'hybrid ancestor hash')
        need(previous['status']=='completed' and previous['exit_code']==0 and previous['finished_at']<=execution['started_at'],'hybrid ancestor completion')
        need(previous['attempt_id']!=execution['attempt_id'] and previous['task_id']!=execution['task_id'],'hybrid new execution')
    dataset=json.loads((source/'dataset.json').read_text())
    conventions=json.loads((source/'stream.json').read_text())['conventions']
    rows={r['id']:r for families in dataset.values() for splits in families.values() for examples in splits.values() for r in examples}
    panels={}
    for name,checksum in result['artifact_manifest'].items():
        path=root/name
        need(sha(path)==checksum,f'hybrid artifact {name}')
        if not name.endswith('.json'):
            continue
        panel=json.loads(path.read_text())
        if not isinstance(panel,dict) or set(panel)!={'metrics','records'}:
            continue
        for record in panel['records']:
            rescore(record,rows[record['id']],conventions,tokenizer,248046,248044)
        n=len(panel['records']);correct=sum(r['correct'] for r in panel['records'])
        need(panel['metrics']['n']==n and panel['metrics']['correct']==correct and panel['metrics']['accuracy']==correct/n,'hybrid panel counts')
        panels[name]=panel
    details={}
    training=root if config['stage']=='train' else wave/'runs'/config['training_task_id']
    trained=json.loads((training/'result.json').read_text())
    memory=json.loads((training/'replay-memory.json').read_text())
    ids=set(memory['ids'])
    need(memory['capacity']==memory['actual_examples']==len(ids)==64 and not memory['growth'],'fixed replay capacity')
    need(Counter(rows[i]['family'] for i in ids)=={f:16 for f in conventions},'replay family balance')
    need(all(rows[i]['split']=='train' and rows[i]['kind']=='primitive' for i in ids),'replay TRAIN only')
    exposed=set()
    for path in (source/'qualification').glob('*/*/updates.jsonl'):
        for line in path.read_text().splitlines():
            exposed.update(json.loads(line)['current_ids'])
    need(ids<=exposed,'replay examples previously exposed')
    need(trained['methods']['arithmetic']['status']=='branch_gate_failed' and trained['methods']['arithmetic']['training_updates']==0,'failed arithmetic gate did no training')
    initial=torch.load(compression/'agent_dice/checkpoint/state.pt',map_location='cpu',weights_only=True)['adapter']
    initial_hash=tensor_identity(initial)
    optimizer_ids=set()
    for arm,value in trained['methods']['agent_dice']['arms'].items():
        need(value['updates']==128 and not value['checkpoint_selected_using_test'],'fixed workflow budget')
        need(value['initial_adapter_sha256']==initial_hash,'same compressed initial factors')
        optimizer_ids.add(value['initial_optimizer_sha256'])
        saved=torch.load(training/value['checkpoint']/'state.pt',map_location='cpu',weights_only=True)
        resident=0
        for layer in saved['adapter'].values():
            need(layer['offset'] is None and layer['A'].shape[0]==layer['B'].shape[1]==8,'hybrid fixed rank without dense offset')
            for key in ('A','B'):
                need(layer[key].dtype==torch.float32 and bool(torch.isfinite(layer[key]).all()),'hybrid finite FP32 factors')
                resident+=layer[key].numel()*layer[key].element_size()
        need(resident==3670016==value['resident']['resident_adapter_tensor_bytes'],'hybrid resident bytes')
        need({int(v['step']) for v in saved['optimizer']['state'].values()}=={128},'actual saved optimizer clocks')
        updates=[json.loads(line) for line in (training/'agent_dice'/arm/'updates.jsonl').read_text().splitlines()]
        need([r['step'] for r in updates]==list(range(1,129)),'hybrid complete update trace')
        for row in updates:
            old=row['replay_ids'];current=row['current_ids']
            need(len(old)==(2 if arm=='compressed_replay' else 0) and set(old)<=ids and len(current)+len(old)==4,'hybrid batch composition')
            need(all(rows[i]['split']=='train' and rows[i]['kind']=='workflow' for i in current),'workflow TRAIN only')
        for key,field in (('old_exposures','replay_ids'),('current_exposures','current_ids')):
            need(value['exposures'][key]==sum(len(row[field]) for row in updates),'hybrid exposure counters')
        details[arm]={'resident_adapter_bytes':resident,'updates':128,'exposures':value['exposures'],'replay_capacity':value['memory_examples']}
        if config['stage']=='audit':
            measured=result['methods']['agent_dice']['arms'][arm]
            need(measured['status']=='fresh_process_audited' and measured['test_evaluated'],'hybrid final process audited')
            for panel,metrics in measured['dev'].items():
                actual=panels[f'agent_dice/{arm}/{panel}.json']
                need(actual==json.loads((training/f'agent_dice/{arm}/dev-128/{panel}.json').read_text()),'hybrid fresh DEV records exact')
                need(metrics==actual['metrics'],'hybrid DEV report binding')
            for panel,metrics in measured['test'].items():
                need(metrics==panels[f'agent_dice/{arm}/{panel}.json']['metrics'],'hybrid TEST report binding')
            details[arm]['test']={k:[v['correct'],v['n']] for k,v in measured['test'].items()}
    need(len(optimizer_ids)==1,'identical initial optimizers')
    return {'status':'passed','stage':config['stage'],'records':sum(len(p['records']) for p in panels.values()),'panels':len(panels),'source_sha256':execution['source_sha256'],'result_sha256':sha(root/'result.json'),'details':details,'scope':'One qualified calendar stream; same rank8 factors and128 updates, replay substitutes256 old exposures for256 current exposures. Full base and archival expert weights are excluded from resident adapter bytes.'}


def audit(root, wave, tokenizer):
    config = json.loads((root / 'config.json').read_text())
    source = wave / 'runs' / config['source_task_id']
    result = json.loads((root / 'result.json').read_text())
    execution = json.loads((root / 'execution.json').read_text())
    previous = json.loads((source / 'execution.json').read_text())
    need(execution['status'] == 'completed' and execution['exit_code'] == 0, 'followup completion')
    need(previous['status'] == 'completed' and previous['exit_code'] == 0, 'source completion')
    need(previous['task_id'] != execution['task_id'] and previous['attempt_id'] != execution['attempt_id'], 'distinct supervised executions')
    need(previous['finished_at'] <= execution['started_at'], 'execution order')
    need(result['source_execution_sha256'] == sha(source / 'execution.json') == sha(root / 'source-execution.json'), 'source execution binding')
    need(result['source_result_sha256'] == sha(source / 'result.json'), 'source result binding')
    need(result['source_unchanged'], 'source mutated')
    dataset = json.loads((source / 'dataset.json').read_text())
    conventions = json.loads((source / 'stream.json').read_text())['conventions']
    rows = {r['id']:r for families in dataset.values() for splits in families.values() for examples in splits.values() for r in examples}
    panels, total = {}, 0
    for name, checksum in result['artifact_manifest'].items():
        path = root / name
        if not name.endswith('.json'):
            continue
        need(sha(path) == checksum, f'artifact hash {name}')
        panel = json.loads(path.read_text())
        if not isinstance(panel, dict) or set(panel) != {'metrics','records'}:
            continue
        for row in panel['records']:
            rescore(row, rows[row['id']], conventions, tokenizer, 248046, 248044)
        n, correct = len(panel['records']), sum(r['correct'] for r in panel['records'])
        need(panel['metrics']['n'] == n and panel['metrics']['correct'] == correct and panel['metrics']['accuracy'] == correct/n, 'followup aggregate')
        panels[name] = panel
        total += n
    details = {}
    if 'conditions' in result:
        original_result = json.loads((source / 'result.json').read_text())
        need(set(result['conditions']) == set(original_result['transfer']), 'missing final conditions')
        for condition, value in result['conditions'].items():
            for name, panel in panels.items():
                if not name.startswith(condition + '/'):
                    continue
                stem = Path(name).stem
                if stem.startswith('primitive-'):
                    previous_path = source/'transfer'/condition/'primitive-after'/f'{stem.removeprefix("primitive-")}.json'
                else:
                    previous_path = source/'transfer'/condition/('novel-test.json' if stem == 'workflow.novel-test' else 'test.json')
                need(panel == json.loads(previous_path.read_text()), 'fresh process saved tokens changed')
            need(value['all_records_exact'], 'reported persistence mismatch')
        details['conditions'] = len(result['conditions'])
    if 'methods' in result:
        need(result['training_updates'] == 0 and result['teacher_or_bank_distillation_updates'] == 0, 'compression update accounting')
        for method, value in result['methods'].items():
            if not value['gate']['passed']:
                need(not value['svd_performed'], 'SVD before eligibility')
                details[method] = {'status':'ineligible_before_SVD'}
                continue
            need(value['rank'] == 8 and value['offsets_removed'] and not value['selected_using_test'], 'fixed compression recipe')
            saved = torch.load(root/value['checkpoint']/'state.pt', map_location='cpu', weights_only=True)
            state = saved['adapter']
            resident = 0
            for module in state.values():
                need(module['offset'] is None and module['A'].shape[0] == module['B'].shape[1] == 8, 'compressed capacity')
                for field in ('A','B'):
                    tensor = module[field]
                    need(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), 'compressed tensor type')
                    resident += tensor.numel()*tensor.element_size()
            need(resident == value['memory']['compressed']['resident_adapter_tensor_bytes'], 'resident compressed bytes')
            counts = {}
            for surface, comparison in value['behavior'].items():
                for version in ('original','compressed'):
                    need(comparison[version] == panels[f'{method}/{version}/{surface}.json']['metrics'], 'compression report binding')
                need(comparison['accuracy_change'] == comparison['compressed']['accuracy']-comparison['original']['accuracy'], 'compression delta')
                counts[surface] = [comparison['original']['correct'],comparison['compressed']['correct'],comparison['original']['n']]
            details[method] = {'status':'audited','resident_bytes':resident,'surfaces_original_compressed_n':counts}
    return {'status':'passed','records':total,'panels':len(panels),'source':source.name,'source_sha256':execution['source_sha256'],
            'execution_sha256':sha(root/'execution.json'),'result_sha256':sha(root/'result.json'),'details':details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave',type=Path,required=True)
    args = parser.parse_args()
    path = args.wave/'tokenizer35/tokenizer.json'
    need(sha(path) == '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42','tokenizer identity')
    tokenizer = Tokenizer.from_file(str(path))
    report = {}
    for root in sorted((args.wave/'runs').iterdir()):
        if not root.name.startswith(('followthrough-20260912-skills-audit','followthrough-20260912-skills-compress','followthrough-20260912-skills-delete-rehearsal')):
            continue
        config = json.loads((root/'config.json').read_text())
        if not (args.wave/'runs'/config['source_task_id']).exists():
            continue
        auditors={'skill-transfer-compressed-replay':audit_hybrid,'skill-transfer-delete-rehearsal':audit_delete_rehearsal}
        report[root.name] = auditors.get(config['experiment'],audit)(root,args.wave,tokenizer)
        print(json.dumps({'audited':root.name,'records':report[root.name]['records']}),flush=True)
    need(report,'no collected followups with collected source')
    (args.wave/'tool-followup-audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
