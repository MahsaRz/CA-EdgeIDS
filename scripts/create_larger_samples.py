"""Uniform projected samples; full official archive/CSV integrity stays strict."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file
from ca_edgeids.experiment import dump


def execute(archive_map):
    locations=json.loads(Path(archive_map).read_text())
    feature_cfg=json.loads(Path('configs/ton_binary_linux_100000.json').read_text())
    selected=['IPV4_SRC_ADDR','IPV4_DST_ADDR']+feature_cfg['features']+['Label','Attack']
    for d in ['ton','cse','bot']:
        if json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())['features']!=feature_cfg['features']:
            raise ValueError('Feature projections must match across datasets')
        out=Path('data/larger-uniform')/d
        if out.exists(): raise FileExistsError(out)
        provenance=json.loads(Path(f'docs/provenance/nf-{d}-v2.json').read_text())
        archive=Path(locations[d]); out.mkdir(parents=True)
        if sha256_file(archive)!=provenance['archive_sha256']: raise ValueError('Official archive SHA256 differs')
        rng=np.random.default_rng(2026); total=0; sample=None; labels=Counter(); attacks=Counter(); hosts=set()
        with zipfile.ZipFile(archive) as z:
            name=next(n for n in z.namelist() if n.endswith(provenance['dataset']+'.csv'))
            header=pd.read_csv(z.open(name),nrows=0).columns.tolist()
            if header!=provenance['source_columns']: raise ValueError('Official CSV header differs')
            for chunk in pd.read_csv(z.open(name),usecols=selected,chunksize=200000,low_memory=False):
                labels.update(chunk.Label.astype(str).value_counts().to_dict()); attacks.update(chunk.Attack.astype(str).value_counts().to_dict())
                hosts.update(chunk.IPV4_SRC_ADDR.astype(str).unique()); hosts.update(chunk.IPV4_DST_ADDR.astype(str).unique())
                chunk['source_row_id']=np.arange(total,total+len(chunk)); chunk['_priority']=rng.random(len(chunk)); total+=len(chunk)
                sample=chunk if sample is None else pd.concat([sample,chunk],ignore_index=True)
                sample=sample.nsmallest(1000000,'_priority')
                if total%2000000==0: print(f'{d}: {total:,} original rows scanned',flush=True)
            h=hashlib.sha256(); sha1=hashlib.sha1()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block); sha1.update(block)
            manifest=z.read(next(n for n in z.namelist() if n.endswith('/manifest-sha1.txt'))).decode()
            expected=next(line.split(None,1)[0] for line in manifest.splitlines() if line.split(None,1)[-1].strip().endswith(provenance['dataset']+'.csv'))
            if h.hexdigest()!=provenance['csv_sha256'] or sha1.hexdigest()!=expected: raise ValueError('Official CSV checksum differs')
        if total!=provenance['source_rows'] or len(hosts)!=provenance['unique_ip_nodes'] or dict(attacks)!=provenance['original_attack_counts']:
            raise ValueError('Complete-source audit differs')
        old_cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())
        old=pd.read_csv(old_cfg['csv'],usecols=['source_row_id'])
        first=sample.nsmallest(100000,'_priority').source_row_id.sort_values().to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(first,np.sort(old.source_row_id.to_numpy()))
        files=[]
        for size in [300000,1000000]:
            part=sample.nsmallest(size,'_priority').sort_values('source_row_id').drop(columns='_priority')
            path=out/f'{d}-uniform-{size}.csv'; part.to_csv(path,index=False,lineterminator='\n')
            files.append(dict(rows=size,csv=str(path).replace('\\','/'),csv_sha256=sha256_file(path),file_bytes=path.stat().st_size,
                source_row_ids_sha256=hashlib.sha256(part.source_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest(),
                retained_original_development_ids=int(part.source_row_id.isin(old.source_row_id).sum()),
                binary_counts={str(k):int(v) for k,v in part.Label.value_counts().sort_index().items()}))
        dump(out/'source_audit.json',dict(dataset=d,archive_sha256=provenance['archive_sha256'],source_csv_sha256=h.hexdigest(),bagit_sha1_verified=True,
            source_rows=total,source_ip_nodes=len(hosts),source_columns=header,projected_columns=selected,
            sampling_seed=2026,sampling='global smallest uniform priorities; identical row priorities to the original 100000-flow development sample',
            complete_attack_counts=dict(attacks),complete_binary_counts=dict(labels),samples=files,
            original_100000_priority_ids_reproduced=True,projection='only unused source columns omitted; original endpoints/labels/selected features/row IDs retained'))
        print(f'{d}: larger samples and source integrity verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--archive-map',required=True); a=p.parse_args(); execute(a.archive_map)
