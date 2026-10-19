"""Describe executed study settings without rewriting historical base configs."""
import argparse
import json
from pathlib import Path

from ca_edgeids.data import sha256_file

STUDIES={
    'controlled':('controlled-binary-100000','selection',120),
    'sampler':('sampler-binary-100000','five-seed-selection',75),
    'novel-host':('novel-host-1000000','selection',30),
}


def execute(study,destination=None,run_base='runs'):
    folder,stage,expected=STUDIES[study]; root=Path(run_base)/folder
    output=Path(destination) if destination else Path('artifacts')/folder/stage/'effective_settings.json'
    if output.exists(): raise FileExistsError(output)
    plan=json.loads((root/'frozen_plan.json').read_text())
    selected=json.loads((root/'selected_models.json').read_text())
    if len(selected)!=expected: raise ValueError('Finish the complete selected-model cohort first')
    seeds=plan.get('model_seeds',plan.get('seeds'))
    datasets={}; models=[]
    for d in plan['datasets']:
        cfg=json.loads((root/d/'resolved_config.json').read_text())
        manifest=json.loads((root/d/'dataset_manifest.json').read_text())
        datasets[d]=dict(csv_sha256=manifest['csv_sha256'],test_row_ids_sha256=manifest['test_row_ids_sha256'],
            feature_columns=cfg['features'],categorical_columns=cfg['categorical'],transformed_width=manifest['transformed_width'],
            class_counts=manifest['class_counts'],actual_split='new-host incident-flow' if study=='novel-host' else 'unordered host-pair hash',
            split_seed=42,preprocessing_fit='current training rows only',
            base_configuration_fields_are_not_the_complete_study_plan=True,
            inherited_base_fields={k:cfg[k] for k in ['seeds','split_protocol','source_audit','learning_rate','repeats'] if k in cfg})
    for r in selected:
        family=r.get('family','GNN'); d=r['dataset']
        cfg=json.loads((root/d/'resolved_config.json').read_text())
        row=dict(dataset=d,family=family,method=r.get('method',r.get('variant','full')),model_seed=r['seed'],
            checkpoint_sha256=r['model_sha256'],fit_source_commit=r.get('fit_source_commit',plan['source_commit']),
            validation_macro_f1=r['validation_macro_f1'])
        if family=='RF':
            row.update(trees=plan.get('rf_trees',100),max_depth=plan.get('rf_depths',{'ton':None,'cse':16,'bot':16})[d],
                class_weight='balanced',bootstrap=True,criterion='gini',max_features='sqrt',n_jobs=1)
        else:
            row.update(learning_rate=r.get('learning_rate',cfg['learning_rate']),hidden=cfg['hidden'],layers=cfg['layers'] if family=='GNN' else 2,
                dropout=cfg['dropout'],weight_decay=cfg['weight_decay'],maximum_epochs=cfg['epochs'],patience=cfg['patience'],
                maximum_targets_per_update=cfg['max_edges_per_batch'],gradient_clip_norm=5,optimizer='Adam',
                loss_normalization='global class-balanced target risk' if study=='sampler' else 'weighted mean within each target batch',
                best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],gradient_updates=r['gradient_updates'])
        models.append(row)
    report=dict(study=study,selected_models_sha256=sha256_file(root/'selected_models.json'),
        model_seeds=seeds,datasets=datasets,models=models,
        interpretation='Dedicated driver arguments, the frozen study plan and individual selected-model records govern execution. Historical base configuration files are preserved; inherited seed, split, source-audit, rate or benchmark fields may be unused or overridden.',
        no_refitting_or_new_selection=True)
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{study}: actual settings for {len(models)} frozen models exported',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',choices=STUDIES,required=True)
    parser.add_argument('--out'); parser.add_argument('--run-base',default='runs'); args=parser.parse_args()
    execute(args.study,args.out,args.run_base)
