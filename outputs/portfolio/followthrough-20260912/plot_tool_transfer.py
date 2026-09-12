import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_compact_control(wave):
    source=wave/'tool-followup-audit.json'
    report=json.loads(source.read_text())
    hybrid=report['followthrough-20260912-skills-compressed-workflow-audit4409']
    control=report['followthrough-20260912-skills-delete-rehearsal-audit4409']
    if hybrid['status']!='passed' or control['status']!='passed':
        raise ValueError('Compact control requires independent raw-record audits')
    arms=[hybrid['details']['compressed_continue'],control['details'],hybrid['details']['compressed_replay']]
    values=[]
    for arm in arms:
        if arm['updates']!=128 or arm['resident_adapter_bytes']!=3670016:
            raise ValueError('Compact control capacity or update count changed')
        panels=arm['test']
        primitive=[sum(v[i] for k,v in panels.items() if k.startswith('primitive-')) for i in (0,1)]
        values.append([primitive,panels['workflow.novel-test']])
    labels=['Continue\n512 current, 0 old','Current-only control\n256 current, 0 old','Replay\n256 current, 256 old']
    colors=['#8497a7','#4478a5','#166f5b']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.9),sharey=True)
    for panel,(axis,title) in enumerate(zip(axes,['Previously learned primitives','Unfamiliar workflow combinations'],strict=True)):
        percentages=[100*row[panel][0]/row[panel][1] for row in values]
        axis.barh(range(3),percentages,color=colors,height=.58)
        axis.set_yticks(range(3),labels,fontsize=10)
        axis.set_xlim(0,115)
        axis.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
        axis.set_title(title,fontweight='bold',fontsize=11,pad=14)
        axis.set_axisbelow(True)
        axis.grid(axis='x',color='#dde3e8',linewidth=.7)
        axis.tick_params(axis='both',length=0)
        for index,row in enumerate(values):
            correct,total=row[panel]
            axis.text(percentages[index]+1.7,index,f'{correct}/{total}',va='center',fontsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Matching current-task exposure narrows the apparent replay benefit',x=.025,y=.97,ha='left',fontsize=14,fontweight='bold')
    fig.text(.025,.896,'Expert adapters → Agent-Dice merge → rank-8 compression → 128 workflow updates',fontsize=10,color='#415465')
    fig.subplots_adjust(left=.235,right=.98,top=.80,bottom=.24,wspace=.17)
    fig.text(.025,.145,'All three start from identical compact factors: 3,670,016 adapter bytes; no dense offset. One calendar stream.',fontsize=9)
    fig.text(.025,.101,'Control and replay use the same current rows and current-loss coefficient. Batch size and optimizer interactions differ.',fontsize=9)
    fig.text(.025,.057,'Fixed final checkpoints; whole JSON answers, execution result and native EOS independently checked after reloading.',fontsize=9)
    for suffix in ('png','pdf','svg'):
        fig.savefig(wave/f'tool-compact-control.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    (wave/'tool-compact-control-figure.json').write_text(json.dumps({'source_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'conditions':labels,'counts':values,'scope':'Single-stream posthoc old-loss ablation; no claim of general superiority or equal lifetime training history.'},indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=Path, required=True)
    args = parser.parse_args()
    source = args.wave/'tool-stream-audit.json'
    report = json.loads(source.read_text())
    runs = sorted(report['runs'].values(),key=lambda x:x['data_seed'])
    if len(runs) != 4 or any(r['status'] != 'passed' for r in runs):
        raise ValueError('Four independently audited streams required')
    arms = ['fresh','relevant','irrelevant','continue','replay','arithmetic','agent_dice']
    labels = ['Fresh learner','Relevant skill','Unrelated skill','Sequential history','Replay history (64)','Arithmetic merge','Agent-Dice merge']
    definitions = [('Before workflow training','validation_initial',32),('Familiar combinations','test',64),('Unfamiliar combinations','novel',64)]
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3,figsize=(13,6.1),sharey=True,gridspec_kw={'wspace':.12})
    all_values = {}
    for axis,(title,field,denominator) in zip(axes,definitions,strict=True):
        counts = np.array([[r['transfers'][arm]['validation_curve'][0][1] if field == 'validation_initial' else r['transfers'][arm][field]['correct'] for r in runs] for arm in arms])
        all_values[field] = counts.tolist()
        display = axis.imshow(counts/denominator,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        axis.set_title(title,pad=14,fontweight='bold',fontsize=11)
        axis.set_xticks(range(4),[f'{r["target_family"].title()}\n{r["data_seed"]}' for r in runs],fontsize=9)
        axis.set_yticks(range(len(arms)),labels,fontsize=10)
        axis.tick_params(axis='both',length=0,pad=7)
        axis.set_xticks(np.arange(-.5,4,1),minor=True)
        axis.set_yticks(np.arange(-.5,len(arms),1),minor=True)
        axis.grid(which='minor',color='white',linewidth=2)
        axis.tick_params(which='minor',bottom=False,left=False)
        for row in range(len(arms)):
            for column in range(4):
                value = int(counts[row,column])
                axis.text(column,row,f'{value}/{denominator}',ha='center',va='center',fontsize=10,color='white' if value/denominator > .6 else '#142332')
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel('Validation at 0 updates' if field == 'validation_initial' else 'Test after 128 workflow updates',labelpad=12,fontsize=9)
    fig.suptitle('Familiar workflow accuracy hides failures on new combinations',x=.04,y=.985,ha='left',fontsize=16,fontweight='bold')
    fig.text(.04,.923,'Four different convention mappings, task orders and datasets; one optimization seed per stream.',fontsize=10,color='#415465')
    fig.subplots_adjust(left=.205,right=.91,bottom=.29,top=.85)
    coloraxis = fig.add_axes([.936,.385,.008,.37])
    colorbar = fig.colorbar(display,cax=coloraxis,ticks=[0,.5,1])
    colorbar.ax.set_yticklabels(['0%','50%','100%'],fontsize=8)
    fig.text(.04,.139,'Rows describe primitive training history. All seven workflow arms train on 512 current examples without rehearsal.',fontsize=9)
    fig.text(.04,.106,'A correct answer is a complete JSON program that executes to the expected state and ends with native EOS.',fontsize=9)
    fig.text(.04,.073,'Unfamiliar command combinations were absent from workflow training. Requests explicitly state the steps.',fontsize=9)
    fig.text(.04,.04,'Merged adapters retain dense offsets; other rows use rank-8 adapters. These rows are not equal in resident memory.',fontsize=9,color='#415465')
    for suffix in ('png','pdf','svg'):
        fig.savefig(args.wave/f'tool-transfer.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    (args.wave/'tool-transfer-figure.json').write_text(json.dumps({'source_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'arms':arms,'streams':[r['data_seed'] for r in runs],'counts':all_values,'denominators':{'validation_initial':32,'test':64,'novel':64},'claim_boundary':'Per-stream counts without pooling four distinct task families into a claim of independent model replication.'},indent=2)+'\n')
    plot_compact_control(args.wave)


if __name__ == '__main__':
    main()
