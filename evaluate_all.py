from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd
from industrial_inventory_env import generate_student_config
from src.common import evaluate_policy

def load(path):
    spec=importlib.util.spec_from_file_location('p_'+path.stem,path); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod.run_policy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--roll-number',default='DA25M609'); ap.add_argument('--submission-dir',default='submissions'); args=ap.parse_args(); cfg=generate_student_config(args.roll_number)
    scenarios=['stationary','seasonal','trend','shock','mixed']; seeds=list(range(7001,7011)); rows=[]
    for f in ['policy_tabular_q.py','policy_reinforce.py','policy_a2c.py','policy_ppo.py','policy_double_dqn.py']:
        p=Path(args.submission_dir)/f; pol=load(p); rs=evaluate_policy(cfg,pol,seeds,scenarios)
        for r in rs: rows.append({'technique':f.removeprefix('policy_').removesuffix('.py'),'scenario':r.scenario,'seed':r.seed,'episode_cost':r.episode_cost})
    df=pd.DataFrame(rows); print('\nOverall mean cost (lower is better):'); print(df.groupby('technique').episode_cost.agg(['mean','std','min','max']).sort_values('mean').round(2)); print('\nBy scenario:'); print(df.pivot_table(index='technique',columns='scenario',values='episode_cost',aggfunc='mean').round(2)); df.to_csv('validation_results.csv',index=False); print('\nSaved validation_results.csv')
if __name__=='__main__': main()
