import hashlib
import json
import math
import random
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

root = Path(__file__).resolve().parents[2]
portfolio = root.parent.parent / 'outputs/portfolio/followthrough-20260912'
run_id = 'followthrough-20260912-portal-example-learned-target64'
run = portfolio / 'runs' / run_id
output = Path(__file__).with_suffix('.json')
assert not output.exists(), 'AUDIT_OUTPUT_ALREADY_EXISTS'
torch.set_num_threads(4)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())

def pretty(value):
    return sha((json.dumps(value, indent=2) + '\n').encode())

def pin(path):
    raw = path.read_bytes()
    return {'bytes':len(raw), 'sha256':sha(raw)}

def read(path):
    return json.loads(path.read_text())

def differences(left, right, path=''):
    if type(left) is not type(right):
        return [{'path':path, 'before':left, 'after':right}]
    if isinstance(left,dict):
        return [change for key in sorted(left.keys() | right.keys()) for change in (differences(left[key],right[key],f'{path}/{key}') if key in left and key in right else [{'path':f'{path}/{key}','before':left.get(key),'after':right.get(key)}])]
    if isinstance(left,list) and len(left)==len(right):
        return [change for i,(a,b) in enumerate(zip(left,right,strict=True)) for change in differences(a,b,f'{path}/{i}')]
    return [] if left==right else [{'path':path,'before':left,'after':right}]

def archive_files(code_hash):
    path=portfolio/'code'/f'{code_hash}.tar'
    with tarfile.open(path) as archive:
        files={member.name:archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    checksum=hashlib.sha256()
    for name in sorted(files,key=Path):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        checksum.update(name.encode()+b'\0'+files[name])
    assert checksum.hexdigest()==code_hash, 'ARCHIVED_CODE_HASH_MISMATCH'
    return files



def tensors_sha(tensors):
    checksum = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        checksum.update(name.encode())
        checksum.update(str((tuple(value.shape), value.dtype)).encode())
        checksum.update(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()))
    return checksum.hexdigest()

def verify_collection(location):
    collected = read(location / 'collection.json')
    for name, expected in collected['files'].items():
        assert pin(location / name) == expected, f'COLLECTED_FILE_HASH_MISMATCH:{location.name}/{name}'
    return len(collected['files'])

def transitions(before, after):
    left = {row['id']: row for row in before['predictions']}
    right = {row['id']: row for row in after['predictions']}
    assert left.keys() == right.keys()
    result = {}
    for task in sorted(before['metrics']):
        identities = [key for key, row in left.items() if row['task'] == task]
        result[task] = {
            'examples': len(identities),
            'before_correct': sum(left[key]['correct'] for key in identities),
            'after_correct': sum(right[key]['correct'] for key in identities),
            'correct_to_wrong': sum(left[key]['correct'] and not right[key]['correct'] for key in identities),
            'wrong_to_correct': sum(not left[key]['correct'] and right[key]['correct'] for key in identities),
            'choice_changes': sum(left[key]['prediction'] != right[key]['prediction'] for key in identities),
            'score_record_changes': sum(left[key] != right[key] for key in identities),
        }
    return result

config, result, training, task, execution, manifest, schedule, rows = [
    read(run / name) for name in [
        'config.json', 'result.json', 'training_receipt.json', 'task.json',
        'execution.json', 'input_manifest.json', 'schedule.json', 'training_rows.json'
    ]
]
verified_files = verify_collection(run)
assert pin(run / 'training_receipt.json') == read(run / 'training_receipt.pin.json')
assert execution['status'] == 'completed' and execution['exit_code'] == 0 and not execution['timed_out']
assert execution['task_id'] == task['id'] == run_id
assert task == execution['task'] and config == task['config']
assert execution['task_sha256'] == pretty(task) and execution['config_sha256'] == pretty(config)
assert canonical(config) == result['config_sha256'] == training['config_sha256']
assert result['status'] == 'completed' and result['new_pid_reload']
assert result['training_pid'] == training['pid'] != result['evaluation_pid']
assert training['input_manifest'] == manifest
assert read(run / 'partial_training_receipt.json') == training
assert config['kind'] == 'portal_example_learned_target_calibration'
assert canonical(config['protocol']) == config['protocol_sha256']
assert config['runtime']['dtype'] == 'float32' and not config['runtime']['autocast']
assert config['protocol']['mode'] == 'calibration'
assert config['protocol']['checkpoints'] == [0, 20, 80, 160, 320]
assert config['protocol']['acquisition_train_floor'] == 0.9
assert config['protocol']['acquisition_validation_floor'] == 0.85
files = archive_files(execution['source_sha256'])
assert len(files) == 95
for name, payload in files.items():
    assert (root / name).read_bytes() == payload, f'FROZEN_LOCAL_SOURCE_CHANGED:{name}'
for name, expected in config['code_dependencies'].items():
    assert expected == {'bytes': len(files[name]), 'sha256': sha(files[name])}
assert training['entrypoint'] == config['code_dependencies']['learned_target.py']
assert training['code_dependencies'] == config['code_dependencies']
assert 'steps = sorted(training["checkpoints"], key=int)' in files['follow_through.py'].decode()
prior_audit = read(output.parent / 'target64_conservative.json')
prior_run = portfolio / 'runs' / prior_audit['run']
assert pin(prior_run / 'input_manifest.json') == prior_audit['input_pins']['input_manifest.json']
assert pin(prior_run / 'config.json') == prior_audit['input_pins']['config.json']
assert config['inputs']['base'] == read(prior_run / 'config.json')['inputs']['base']
assert manifest['base'] == read(prior_run / 'input_manifest.json')['base']
assert training['base_sha256'] == read(prior_run / 'training_receipt.json')['base_sha256']
verified_input_paths = []
for name, spec in config['inputs'].items():
    location = Path(spec['path'])
    if not location.is_absolute():
        payload = files[spec['path']]
        assert spec['sha256'] == sha(payload) and spec['bytes'] == len(payload)
        location = Path(task['code_dir']) / location
    records = manifest[name] if 'files' in spec else {'': manifest[name]}
    for filename, record in records.items():
        expected = spec['files'][filename] if filename else {key: spec[key] for key in ['bytes', 'sha256']}
        assert record['bytes'] == expected['bytes'], f'INPUT_SIZE_CHANGED:{name}/{filename}'
        if 'sha256' in expected:
            assert record['sha256'] == expected['sha256'], f'INPUT_SHA256_CHANGED:{name}/{filename}'
        else:
            assert name == 'base' and set(expected) == {'bytes', 'git_blob_sha1'}
        expected_path = location / filename if filename else location
        assert record['path'] == str(expected_path), f'INPUT_PATH_CHANGED:{name}/{filename}'
        verified_input_paths.append(str(expected_path))

export = portfolio / 'learned-source-export'
qualification_path = export / 'source_qualification.json'
qualification = read(qualification_path)
assert pin(qualification_path) == {key: config['inputs']['source_qualification'][key] for key in ['bytes', 'sha256']}
assert read(export / 'target64.json') == config
assert qualification['qualification_scope'] == 'acquisition_and_retention'
assert qualification['acquisition_qualified'] and qualification['retention_qualified']
assert qualification['source_arm'] == 'heads_new_latents' and qualification['source_step'] == 512
assert qualification['exported_source'] == config['inputs']['source_learned']
source_run = portfolio / 'runs' / qualification['source_run']['task_id']
for name, expected in qualification['original_source_verification']['metadata'].items():
    assert pin(source_run / name) == expected, f'SOURCE_RECEIPT_CHANGED:{name}'
corrected_run = portfolio / 'runs' / qualification['corrected_retention_run']['task_id']
for name, expected in qualification['corrected_receipt_files'].items():
    assert pin(corrected_run / name) == expected
corrected = read(corrected_run / 'result.json')['arms']['heads_new_latents']
assert corrected['retention_steps'] == {'before': 0, 'after': 512}
assert corrected['retention'] == qualification['retention']
assert all(value['change'] >= -2 / value['examples'] and value['initial_ability_qualified'] for value in qualification['retention'].values())
source_training = read(source_run / 'training_receipt.json')
source_checkpoint = source_training['arms']['heads_new_latents']['checkpoints']['512']
assert source_checkpoint['saved'] == qualification['source_checkpoint']
assert all(value == 1.0 for panel in ['train', 'validation'] for value in source_checkpoint['qualification'][panel].values())
source_location = source_run / 'heads_new_latents/checkpoints/step-0512'
for name, expected in qualification['source_checkpoint']['artifact']['files'].items():
    assert pin(source_location / name) == expected
source_state = load_file(str(source_location / 'model.safetensors'))
source_config = read(source_location / 'config.json')
assert tensors_sha(source_state) == qualification['source_checkpoint']['tensor_sha256']
source_initial_location = source_run / 'heads_new_latents/checkpoints/step-0000'
source_initial = load_file(str(source_initial_location / 'model.safetensors'))
assert pin(source_initial_location / 'model.safetensors') == config['inputs']['source_initial']['files']['model.safetensors']
assert tensors_sha(source_initial) == config['inputs']['source_initial']['tensor_sha256']
assert tensors_sha({name: value for name, value in source_state.items() if not name.startswith('alignment.')}) == qualification['source_checkpoint']['shared_sha256']

fixture=json.loads(files[config['inputs']['fixture']['path']])
old=json.loads(files[config['inputs']['old_holdout']['path']])
length=json.loads(files[config['inputs']['length_holdout']['path']])
released=json.loads(files[config['inputs']['retention_probes']['path']])
tasks=('sequence_a','sequence_b','sequence_c')
old_tasks=('boolq','commonsense_qa','hellaswag','winogrande')
release_rows=[]
for row,identity in zip(released['validation'],released['indices'],strict=True):
    assert row['task']==identity['task'] and sha(' '.join(row['prompt'].split()).casefold().encode())==identity['prompt_sha256']
    release_rows.append({**row,'id':'released-'+identity['prompt_sha256'],'group':identity['prompt_sha256']})
validation=[row for name in tasks for row in fixture['splits'][f'{name}_validation']]
panels={'old_exhaustive_triples':old['rows'],'sealed_length4':length['rows'],'old_validation_test':[row for name in tasks for split in ['validation','test'] for row in fixture['splits'][f'{name}_{split}']], 'released_tasks':release_rows}
all_rows=[row for split in fixture['splits'].values() for row in split]+old['rows']+length['rows']
for row in all_rows:
    oracle=' '+' '.join(str(fixture['provenance']['rules'][row['task']][int(value)]) for value in row['group'].split())
    assert row['choices'][row['gold_idx']]==oracle
training_groups={row['group'] for name in tasks for row in fixture['splits'][f'{name}_train']}
for panel in ['old_exhaustive_triples','sealed_length4','old_validation_test']:
    assert not training_groups & {row['group'] for row in panels[panel]}
for name in tasks:
    assert len(rows[name])==64 and [row['id'] for row in rows[name]]==config['protocol']['training_ids'][name]
    source_index={row['id']:row for row in fixture['splits'][f'{name}_train']}
    assert all(row==source_index[row['id']] for row in rows[name])


for name in tasks:
    index = source_config['tasks'].index(name)
    assert tensors_sha({'vector': source_state['task_latents'][index]}) == qualification['ABC_vector_sha256'][name]
    assert not torch.equal(source_state['task_latents'][index], source_initial['task_latents'][source_config['tasks'].index('rte')])
    assert torch.equal(source_initial['task_latents'][index], source_initial['task_latents'][source_config['tasks'].index('rte')])
    assert {row['id'] for row in rows[name]} <= {
        identity for entry in source_training['arms']['heads_new_latents']['trace'] for identity in entry['ids'][name]
    }
for name in source_config['tasks']:
    if name not in tasks:
        index = source_config['tasks'].index(name)
        assert torch.equal(source_state['task_latents'][index], source_initial['task_latents'][index])

expected_schedule = []
for offset in range(320):
    epoch, batch = divmod(offset, 16)
    selected = {}
    for name in tasks:
        identities = [row['id'] for row in rows[name]]
        random.Random(int(canonical([config['protocol']['data_seed'], epoch, name]), 16)).shuffle(identities)
        selected[name] = identities[batch * 4:(batch + 1) * 4]
    expected_schedule.append({'step': offset + 1, 'epoch': epoch + 1, 'ids': selected})
assert schedule == expected_schedule
for name in tasks:
    assert Counter(identity for entry in schedule for identity in entry['ids'][name]) == Counter({row['id']: 20 for row in rows[name]})
assert config['protocol']['batch_per_task'] == 4

prediction_count=0
score_ties=0

def rescore(panel, expected):
    global prediction_count,score_ties
    by_id={row['id']:row for row in expected}
    assert len(by_id)==len(expected)==len(panel['predictions'])
    assert {row['id'] for row in panel['predictions']}==set(by_id)
    correct=Counter()
    total=Counter()
    gold_scores=defaultdict(list)
    for prediction in panel['predictions']:
        row=by_id[prediction['id']]
        scores=prediction['scores']
        assert len(scores)==len(row['choices']) and all(math.isfinite(score) for score in scores)
        choice=max(range(len(scores)),key=scores.__getitem__)
        assert prediction['prediction']==choice
        assert prediction['gold']==row['gold_idx'] and prediction['correct']==int(choice==row['gold_idx'])
        assert prediction['task']==row['task'] and prediction['group']==row['group']
        assert prediction['prompt_sha256']==canonical(row['prompt'])
        total[row['task']]+=1
        correct[row['task']]+=int(choice==row['gold_idx'])
        gold_scores[row['task']].append(scores[row['gold_idx']])
        prediction_count+=1
        score_ties+=int(scores.count(max(scores))>1)
    assert set(panel['metrics'])==set(total)
    for name in total:
        metric=panel['metrics'][name]
        assert metric['examples']==total[name] and metric['accuracy']==correct[name]/total[name]
        assert math.isclose(metric['gold_char_mean_logp'],statistics.mean(gold_scores[name]),rel_tol=1e-6,abs_tol=1e-7)
        assert math.isfinite(metric['gold_nll']) and metric['gold_nll']>=0
    return {'correct':sum(correct.values()),'examples':sum(total.values()),'tasks':{name:{'correct':correct[name],'examples':total[name]} for name in sorted(total)}}



raw = {name: rescore(panel, panels[name]) for name, panel in result['raw'].items()}
transplant = read(run / 'transplant_verification.json')
arm_results = {}
model_files = optimizer_files = 0
initial_alignment = None
for arm_spec in config['protocol']['arms']:
    name = arm_spec['name']
    assert name in ['learned', 'untouched', 'lora']
    assert arm_spec['learning_rate'] == 0.001 and not arm_spec['train_latents']
    assert arm_spec['routing'] == ('persistent' if name == 'lora' else 'task_latents')
    trained = training['arms'][name]
    assert trained == read(run / name / 'training.json'), f'ARM_RECEIPT_DIFFERS:{name}'
    assert trained['status'] == 'completed_budget'
    assert trained['optimizer_updates'] == trained['finite_completed_updates'] == len(trained['trace']) == 320
    assert trained['example_exposures_per_task'] == dict.fromkeys(tasks, 1280)
    assert trained['schedule_sha256'] == canonical(schedule)
    ema = {}
    for entry, scheduled in zip(trained['trace'], schedule, strict=True):
        assert {key: entry[key] for key in ['step', 'epoch', 'ids']} == scheduled
        assert math.isfinite(entry['gradient_norm']) and entry['gradient_norm'] > 0
        assert math.isclose(entry['gradient_norm'], math.sqrt(sum(group['norm'] ** 2 for group in entry['gradients'].values())), rel_tol=1e-12)
        for group in entry['gradients'].values():
            assert not group['missing'] and math.isfinite(group['norm']) and group['norm'] >= 0
        for name_task, loss in entry['losses'].items():
            assert math.isfinite(loss['nll']) and loss['nll'] >= 0 and loss['tokens'] == 24
            ema[name_task] = 0.9 * ema.get(name_task, loss['nll']) + 0.1 * loss['nll']
            assert ema[name_task] == loss['ema']
    measured = result['arms'][name]
    assert measured['retention_steps'] == {'before': 0, 'after': 320}
    assert measured['reload_predictions_exact'] and measured['status'] == trained['status']
    steps = sorted(trained['checkpoints'], key=int)
    assert list(map(int, steps)) == config['protocol']['checkpoints']
    assert set(measured['checkpoints']) == set(steps)
    curve = {}
    checkpoint_audits = {}
    for step in steps:
        checkpoint = trained['checkpoints'][step]
        assert checkpoint == read(run / name / f'checkpoint-{int(step):04d}.json')
        assert checkpoint['step'] == int(step)
        train_panel = rescore(checkpoint['train'], [row for values in rows.values() for row in values])
        val_panel = rescore(checkpoint['validation'], validation)
        expected_qualification = {
            task_name: train_panel['tasks'][task_name]['correct'] / 64 >= 0.9 and val_panel['tasks'][task_name]['correct'] / 32 >= 0.85
            for task_name in tasks
        }
        q = checkpoint['qualification']
        assert q['qualified_tasks'] == expected_qualification and q['all_tasks_qualified'] == all(expected_qualification.values())
        for panel_name in ['train', 'validation']:
            assert q[panel_name] == {task_name: checkpoint[panel_name]['metrics'][task_name]['accuracy'] for task_name in tasks}
        assert q['gain_from_start'] == val_panel['correct'] / val_panel['examples'] - sum(p['correct'] for p in trained['checkpoints']['0']['validation']['predictions']) / 96
        destination = run / name / 'checkpoints' / f'step-{int(step):04d}'
        saved = checkpoint['saved']
        assert saved['artifact']['path'] == str(Path(manifest['source_qualification']['path']).parents[2] / 'runs' / run_id / name / 'checkpoints' / destination.name)
        for filename, expected in saved['artifact']['files'].items():
            assert pin(destination / filename) == expected
        state = load_file(str(destination / 'model.safetensors'))
        assert tensors_sha(state) == saved['tensor_sha256']
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
        model_files += 1
        checkpoint_config = read(destination / 'config.json')
        assert checkpoint_config['base_model_name_or_path'] == config['base']['repo_id']
        assert checkpoint_config['base_model_revision'] == config['base']['revision']
        assert checkpoint_config['tasks'] == source_config['tasks']
        trainable = {key: value for key, value in state.items() if name == 'lora' or key.startswith('alignment.')}
        assert sum(value.numel() for value in trainable.values()) == trained['trainable_parameters']
        assert sum(value.numel() for value in state.values()) == trained['resident_parameters']
        if name != 'lora':
            reference = source_state if name == 'learned' else source_initial
            frozen = {key: value for key, value in state.items() if not key.startswith('alignment.')}
            assert all(torch.equal(value, reference[key]) for key, value in frozen.items())
            assert tensors_sha(frozen) == saved['shared_sha256']
            freeze_receipt = {key: value for key, value in frozen.items() if key != 'task_latents'}
            old_indices = [i for i, task_name in enumerate(checkpoint_config['tasks']) if task_name not in tasks]
            freeze_receipt['published_task_latents'] = state['task_latents'][old_indices]
            vector_hashes = {task_name: tensors_sha({'vector': state['task_latents'][checkpoint_config['tasks'].index(task_name)]}) for task_name in tasks}
            assert vector_hashes == transplant[name]['ABC_vector_sha256']
            if step == '0':
                assert all(torch.count_nonzero(value) == 0 for key, value in state.items() if key.startswith('alignment.output.'))
                if initial_alignment is None:
                    initial_alignment = {key: value.clone() for key, value in state.items() if key.startswith('alignment.')}
                else:
                    assert all(torch.equal(state[key], value) for key, value in initial_alignment.items())
        else:
            freeze_receipt = {}
            if step == '0':
                assert all(torch.count_nonzero(value) == 0 for key, value in state.items() if key.endswith('_b'))
        assert tensors_sha(freeze_receipt) == trained['frozen_adapter_sha256'] == checkpoint['frozen_adapter_sha256']
        if step == '0':
            assert tensors_sha(state) == transplant[name]['initial_tensor_sha256']
            assert trained['trainable_parameters'] == transplant[name]['trainable_parameters']
        optimizer_path = destination.parent / f'optimizer-{int(step):04d}.pt'
        assert pin(optimizer_path) == checkpoint['optimizer']
        optimizer = torch.load(optimizer_path, map_location='cpu', weights_only=True)
        optimizer_files += 1
        assert len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert group['lr'] == 0.001 and group['weight_decay'] == 0 and group['betas'] == (0.9, 0.999)
        assert group['eps'] == 1e-8 and group['foreach'] is False
        assert len(group['params']) == len(trainable)
        expected_states = 0 if step == '0' else len(trainable)
        assert len(optimizer['state']) == expected_states
        if expected_states:
            assert Counter(tuple(value['exp_avg'].shape) for value in optimizer['state'].values()) == Counter(tuple(value.shape) for value in trainable.values())
        for value in optimizer['state'].values():
            assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'} and value['step'].item() == int(step)
            assert value['exp_avg'].shape == value['exp_avg_sq'].shape
            assert all(tensor.dtype == torch.float32 and torch.isfinite(tensor).all() for tensor in value.values())
            assert (value['exp_avg_sq'] >= 0).all()
        final_panels = {panel_name: rescore(panel, panels[panel_name]) for panel_name, panel in measured['checkpoints'][step].items()}
        validation_ids = {row['id'] for row in validation}
        scored_validation = {row['id']: row for row in measured['checkpoints'][step]['old_validation_test']['predictions'] if row['id'] in validation_ids}
        assert scored_validation == {row['id']: row for row in checkpoint['validation']['predictions']}
        curve[step] = {
            'train': train_panel, 'validation': val_panel,
            'mean_train_nll': statistics.mean(checkpoint['train']['metrics'][task_name]['gold_nll'] for task_name in tasks),
            'mean_validation_nll': statistics.mean(checkpoint['validation']['metrics'][task_name]['gold_nll'] for task_name in tasks),
            'acquisition_qualified': all(expected_qualification.values()),
            'example_exposures_total': int(step) * 12,
            'example_exposures_per_task': int(step) * 4,
            'heldout': final_panels,
        }
        with safe_open(str(destination / 'model.safetensors'), framework='pt', device='cpu') as reader:
            metadata = reader.metadata()
        checkpoint_audits[step] = {
            'model': pin(destination / 'model.safetensors'), 'tensor_sha256': saved['tensor_sha256'],
            'optimizer': pin(optimizer_path), 'optimizer_states': expected_states,
            'optimizer_tensor_bytes': sum(t.numel() * t.element_size() for v in optimizer['state'].values() for t in v.values()),
            'model_tensor_bytes': sum(t.numel() * t.element_size() for t in state.values()),
            'frozen_tensor_sha256': checkpoint['frozen_adapter_sha256'],
            'shared_sha256': saved['shared_sha256'], 'metadata': metadata,
            'finite_fp32_model_and_optimizer': True,
            'validation_scores_equal_new_pid_holdout_subset': True,
        }
        del state, optimizer
    initial = measured['checkpoints']['0']['released_tasks']
    final = measured['checkpoints']['320']['released_tasks']
    assert initial == result['raw']['released_tasks']
    for panel in ['old_exhaustive_triples', 'sealed_length4', 'old_validation_test']:
        assert measured['checkpoints']['0'][panel] == result['raw'][panel]
    transition = transitions(initial, final)
    for old_task in old_tasks:
        summary = measured['retention'][old_task]
        before, after = transition[old_task]['before_correct'] / 32, transition[old_task]['after_correct'] / 32
        assert summary == {'before': before, 'after': after, 'change': after - before, 'raw': before, 'examples': 32, 'initial_ability_qualified': before >= 0.6, 'initial_adapter_gain_qualified': False}
    first_positive = {
        family: next(entry['step'] for entry in trained['trace'] if entry['gradients'][family]['norm'] > 0)
        for family in trained['active_gradient_families']
    }
    assert all(step <= config['protocol']['gradient_gate_step'] for step in first_positive.values())
    events = [json.loads(line) for line in (run / name / 'events.jsonl').read_text().splitlines()]
    assert [{key: value for key, value in event.items() if key not in ['event', 'pid']} for event in events if event['event'] == 'training_update'] == trained['trace']
    assert all(event['pid'] == training['pid'] for event in events)
    arm_results[name] = {
        'curve': curve, 'checkpoint_audits': checkpoint_audits, 'trainable_parameters': trained['trainable_parameters'],
        'resident_parameters': trained['resident_parameters'],
        'first_qualifying_checkpoint': next(int(step) for step in steps if curve[step]['acquisition_qualified']),
        'qualification_crossing_interval_updates': [20, 80],
        'first_all_train_and_validation_correct': next((int(step) for step in steps if curve[step]['train']['correct'] == 192 and curve[step]['validation']['correct'] == 96), None),
        'first_all_triples_and_length4_correct': next((int(step) for step in steps if curve[step]['heldout']['old_exhaustive_triples']['correct'] == 864 and curve[step]['heldout']['sealed_length4']['correct'] == 384), None),
        'gradient_clipped_updates': sum(entry['gradient_norm'] > config['protocol']['gradient_clip'] for entry in trained['trace']),
        'max_gradient_norm': max(entry['gradient_norm'] for entry in trained['trace']),
        'first_positive_gradient_by_family': first_positive,
        'retention': measured['retention'], 'old_row_transitions': transition,
        'within_two_row_net_drop_on_initially_qualified_panels': all(value['change'] >= -2 / value['examples'] for value in measured['retention'].values() if value['initial_ability_qualified']),
    }
    print(json.dumps({'verified_arm': name, 'first_qualified': arm_results[name]['first_qualifying_checkpoint'], 'first_perfect_heldout': arm_results[name]['first_all_triples_and_length4_correct']}), flush=True)

top_events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
assert top_events[0]['adapters'] == transplant and top_events[0]['pid'] == training['pid']
assert [(entry['arm'], int(entry['step'])) for entry in top_events if entry['event'] == 'heldout_checkpoint_evaluated'] == [
    (arm['name'], step) for arm in config['protocol']['arms'] for step in config['protocol']['checkpoints']
]
assert all(entry['pid'] == result['evaluation_pid'] for entry in top_events if entry['event'] == 'heldout_checkpoint_evaluated')

retention_followup = {}
for audit_name in ['head_nullspace', 'preservation']:
    audit_path = output.parent / f'{audit_name}.json'
    audit = read(audit_path)
    old_run = portfolio / 'runs' / audit['run']
    for filename, key in [('result.json', 'result_pin'), ('training_receipt.json', 'training_pin'), ('collection.json', 'collection_pin')]:
        assert pin(old_run / filename) == audit[key]
    assert verify_collection(old_run) == audit['collection_files_verified']
    old_result = read(old_run / 'result.json')
    old_arms = {}
    for arm_name, arm in old_result['arms'].items():
        numeric_steps = sorted(arm['checkpoints'], key=int)
        first, last = numeric_steps[0], numeric_steps[-1]
        before, after = [arm['checkpoints'][step]['released_tasks'] for step in [first, last]]
        rescore(before, release_rows)
        rescore(after, release_rows)
        old_arms[arm_name] = transitions(before, after)
    retention_followup[audit_name] = {
        'prior_audit': {'path': str(audit_path), **pin(audit_path)},
        'all_prior_collected_file_hashes_reverified': audit['collection_files_verified'],
        'old_row_transitions': old_arms,
        'prior_matrix_and_precision_audits_still_bound_to_identical_artifacts': True,
    }

report = {
    'status': 'independently_verified',
    'run': run_id, 'source_sha256': execution['source_sha256'], 'source_files': len(files),
    'auditor': {'path': str(Path(__file__).resolve()), **pin(Path(__file__))},
    'collection_files_verified': verified_files, 'model_checkpoints_loaded': model_files, 'optimizer_checkpoints_loaded': optimizer_files,
    'pins': {name: pin(run / name) for name in ['collection.json', 'execution.json', 'config.json', 'task.json', 'result.json', 'training_receipt.json', 'training_receipt.pin.json', 'input_manifest.json', 'training_rows.json', 'schedule.json', 'transplant_verification.json']},
    'verified_input_paths': verified_input_paths, 'source_qualification': qualification,
    'scoring_rechecks': prediction_count, 'argmax_tie_rows_in_rechecks': score_ties,
    'metric_scope': 'Choice accuracy and character-normalized gold log probability rederived from stored raw scores. Token-normalized gold_nll is checked for finiteness. Rechecks contain repeated panels, not independent samples.',
    'ABC_oracle_choices_verified': len(all_rows),
    'runtime': {'local_torch': torch.__version__, 'training_pid': training['pid'], 'evaluation_pid': result['evaluation_pid'], 'base_sha256': training['base_sha256']},
    'base_verification_boundary': 'Frozen base identity and new-process reload are bound through the hashed runtime receipt and archived checks. Base tensors were not independently reloaded locally.',
    'data': {'unique_calibration_examples_per_task': 64, 'updates': 320, 'example_exposures_per_task': 1280, 'total_exposures_per_arm': 3840, 'training_rows_and_order_identical_across_arms': True, 'target_training_rows_subset_of_source_training_rows': True, 'heldout_groups_disjoint_from_source_and_target_training': True},
    'raw': raw, 'arms': arm_results, 'retention_followup': retention_followup,
    'interpretation': [
        'All three arms first qualify at the same recorded checkpoint80. The actual qualification crossing is only localized to(20,80]; these checkpoints do not establish faster threshold crossing.',
        'The learned source improves triples and validation over untouched at20 and80, but not length4 at80. This is a modest early trajectory advantage within one fixed-task single-seed stream.',
        'Learned and untouched first reach perfect triples and length4 at160. LoRA reaches both at80, remains perfect at160, then loses two answers on each panel at320. The final endpoint alone would hide the earlier LoRA lead.',
        'Learned and untouched each finish86/128 old answers, with different per-task transitions. LoRA finishes74/128. Both native arms have at most two net lost rows on the three initially qualified target tasks; WinoGrande starts19/32 and is ineligible under the60percent ability gate.',
        'The learned core and all learned ABC vectors are exactly frozen in every target checkpoint; fresh target alignment is identical atstep0 across native arms and trains in both. The learned-source effect bundles trained heads and task vectors, so their separate causal contributions are not identified.',
        'Native calibration trains10486912 parameters versus2949120 for LoRA, plus512 source-training updates for the learned source. No end-to-end compute advantage or wall-clock speed claim follows.',
        'Raw scores use the declared four-choice contract. These results do not establish free generation, unfamiliar related-skill transfer, repeated model upgrades, or statistical reliability across seeds.',
        'Nullspace and preservation tensor proofs remain bound to unchanged artifacts. Net accuracy can hide wrong-to-correct gains offsetting newly wrong old rows; transition counts supplement the existing matrix audits.',
        'No new PorTAL training, source changes, GPU dispatch, or remote mutations.'
    ],
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({'audit': str(output), **pin(output), 'scoring_rechecks': prediction_count, 'model_files': model_files, 'optimizer_files': optimizer_files, 'collection_files_verified': verified_files}, indent=2))
