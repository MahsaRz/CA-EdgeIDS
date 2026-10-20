"""Check that nested samples preserve development rows and encoder inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ca_edgeids.data import sha256_file


def execute(destination='artifacts/larger-uniform-source-audits/nested-observables.json'):
    output=Path(destination)
    if output.exists(): raise FileExistsError(output)
    records=[]
    for d in ['ton','cse','bot']:
        cfg=json.loads(Path(f'configs/{d}_binary_linux_100000.json').read_text())
        columns=['IPV4_SRC_ADDR','IPV4_DST_ADDR','Label','Attack']+cfg['features']+['source_row_id']
        old=pd.read_csv(cfg['csv'],usecols=columns,low_memory=False).set_index('source_row_id')
        categorical=['IPV4_SRC_ADDR','IPV4_DST_ADDR','Attack']+cfg['categorical']
        numeric=['Label']+[c for c in cfg['features'] if c not in cfg['categorical']]
        for n in [300000,1000000]:
            path=Path(f'data/larger-uniform/{d}/{d}-uniform-{n}.csv')
            new=pd.read_csv(path,usecols=columns,low_memory=False).set_index('source_row_id')
            if not new.index.is_unique or not old.index.isin(new.index).all(): raise ValueError('Original row identities changed')
            retained=new.loc[old.index]
            string_changes={c:int(np.sum(old[c].fillna('<missing>').astype(str).to_numpy()!=retained[c].fillna('<missing>').astype(str).to_numpy())) for c in categorical}
            numeric_changes={c:int(np.sum(~((old[c].to_numpy()==retained[c].to_numpy())|(old[c].isna().to_numpy()&retained[c].isna().to_numpy())))) for c in numeric}
            if any(string_changes.values()) or any(numeric_changes.values()):
                raise ValueError(f'{d}/{n}: encoder-observable changes: {string_changes}; {numeric_changes}')
            records.append(dict(dataset=d,sample_size=n,original_development_rows=len(old),
                source_csv_sha256=sha256_file(path),categorical_representation_changes=string_changes,numerical_value_changes=numeric_changes))
            print(f'{d}/{n}: all original development observables preserved',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(checked_workloads=records,
        scope='All original development endpoints, labels, selected numeric values and categorical encoder representations remain identical in each larger sample; no fitting or predictions'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='artifacts/larger-uniform-source-audits/nested-observables.json')
    execute(parser.parse_args().out)
