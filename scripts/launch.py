#!/usr/bin/env python3
"""Print commands by default. Explicit --execute runs ONLY a selected experiment.
Run concurrently on each allocated node with matching configuration and seeds.
"""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def validate(config):
    assert config['schema_version']==1
    ids=set()
    for e in config['experiments']:
        assert e['id'] not in ids;ids.add(e['id'])
        r=e['resources'];t=e['training']
        assert r['gpus']==r['nodes']*r['gpus_per_node']
        assert t['global_batch']==r['gpus']*t['batch_size']
        assert t['method'] in ['ddp','localsgd','cpdp']
        assert t['sync_period']>=1 and (t['method']!='ddp' or t['sync_period']==1)
        assert 0<=t['warmup_epochs']<=t['epochs'] and 0<=t['slowmo_beta']<1
        assert e['seeds'] and set(e['seeds'])<=set([42,123,777])
        assert e['optimizer']['name']==('AdamW' if t['model']=='vit_s' else 'SGD')

def commands(config, data_root, output_root, experiment=None, seed=None, execute=False):
    validate(config)
    selected=[e for e in config['experiments'] if experiment is None or e['id']==experiment]
    if not selected:raise ValueError('Unknown experiment')
    if execute and experiment is None:raise ValueError('--execute requires --experiment to avoid launching a whole matrix')
    for e in selected:
        r=e['resources'];t=e['training'];n=r['nodes']
        if execute and n>1 and not all(os.environ.get(v) for v in ['MASTER_ADDR','NODE_RANK']):
            raise ValueError('Set MASTER_ADDR and NODE_RANK on every allocated node')
        rank=os.environ.get('NODE_RANK','0');master=os.environ.get('MASTER_ADDR','127.0.0.1' if n==1 else 'MASTER_HOST')
        if execute and not 0<=int(rank)<n:raise ValueError('Invalid NODE_RANK')
        if seed is not None and seed not in e['seeds']:raise ValueError('Seed is not in selected configuration')
        for s in ([seed] if seed is not None else e['seeds']):
            profile='profiling' in config
            entry=ROOT/'src'/('profile_comm.py' if profile else 'train_scaling.py')
            cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node',str(r['gpus_per_node']),
                 '--nnodes',str(n),'--node_rank',rank,'--master_addr',master,
                 '--master_port',os.environ.get('MASTER_PORT','29500'),str(entry)]
            for k,v in t.items():
                if profile and k in ['global_batch','epochs','warmup_epochs']:continue
                if isinstance(v,bool):
                    if v:cmd.append('--'+k.replace('_','-'))
                else:cmd.extend(['--'+k.replace('_','-'),str(v)])
            output=Path(output_root)/e['id']/('s'+str(s))
            cmd.extend(['--seed',str(s),'--data-root',str(data_root)])
            if profile:
                for k,v in config['profiling'].items():cmd.extend(['--'+k.replace('_','-'),str(v)])
                cmd.extend(['--scale-tag',e['id'],'--outfile',str(output/'profile.csv')])
            else:cmd.extend(['--outdir',str(output)])
            yield cmd,e

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--experiment');p.add_argument('--seed',type=int)
    p.add_argument('--data-root',default=os.environ.get('DATA_ROOT',str(ROOT/'data')))
    p.add_argument('--output-root');p.add_argument('--execute',action='store_true')
    a=p.parse_args();path=Path(a.config);config=json.loads(path.read_text())
    output=a.output_root or str(ROOT/'results'/'new_runs'/path.stem)
    for cmd,e in commands(config,a.data_root,output,a.experiment,a.seed,a.execute):
        print('# Resources:',e['resources'],'Dataset:',e['training']['dataset'],flush=True)
        print(shlex.join(cmd),flush=True)
        if a.execute:
            if len(e['resources']['sites'])>1:
                env=os.environ.copy();env.setdefault('NCCL_SOCKET_IFNAME','br0');env.setdefault('NCCL_IB_DISABLE','1');env.setdefault('NCCL_NET_GDR_LEVEL','0')
            else:env=os.environ.copy()
            subprocess.run(cmd,env=env,check=True,cwd=ROOT)
if __name__=='__main__':main()
