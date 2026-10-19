"""Create or verify the exact 100000-flow inputs from official archives."""
import argparse
import json
from pathlib import Path

from ca_edgeids.data import sha256_file
from audit_archive import audit
from text_digest import matches_text_digest


def execute(archive_map=None,check_existing=False):
    locations={} if archive_map is None else json.loads(Path(archive_map).read_text())
    if not check_existing and set(locations)!={'ton','cse','bot'}: raise ValueError('Archive map needs ton, cse and bot paths')
    plan=json.loads(Path('configs/linux_scaling_100000_plan.json').read_text())
    for d in plan['datasets']:
        expected=json.loads(Path(f'docs/provenance/nf-{d}-v2.json').read_text())
        cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())
        target=Path(cfg['csv']); folder=target.parent
        if not check_existing:
            if folder.exists(): raise FileExistsError(f'Refusing to overwrite {folder}; use --check-existing to verify it')
            if sha256_file(locations[d])!=expected['archive_sha256']: raise ValueError('Official archive SHA256 differs')
            audit(locations[d],expected['dataset'],folder,sample_size=100000,seed=2026)
        actual=json.loads(Path(cfg['source_audit']).read_text())
        for key in ['archive_sha256','csv_sha256','source_rows','unique_ip_nodes','source_columns','original_attack_counts','binary_counts']:
            if actual[key]!=expected[key]: raise ValueError(f'{d}: source audit mismatch in {key}')
        frozen=plan['inputs'][d]
        if actual['pilot_rows']!=100000 or actual['pilot_seed']!=2026 or not actual['bagit_sha1_verified']: raise ValueError('Wrong sample protocol')
        if actual['pilot_source_row_ids_sha256']!=frozen['sample_row_ids_sha256']: raise ValueError('Original sampled rows differ')
        if not matches_text_digest(target,frozen['sample_csv_sha256']): raise ValueError('Sample content differs from the frozen study')
        print(f'{d}: sample content, original row-ID digest and recorded source audit verified',flush=True)
    if check_existing: print('Existing-sample verification does not repeat the full archive scan.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--archive-map'); p.add_argument('--check-existing',action='store_true'); a=p.parse_args()
    execute(a.archive_map,a.check_existing)
