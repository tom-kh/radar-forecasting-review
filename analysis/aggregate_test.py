#!/usr/bin/env python3
import argparse, json, statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',default='runs/frozen_test_icra2027')
a=p.parse_args()
root=Path(a.root)
methods=['scratch_transport','jepa_observed','doppler_jepa','reconstruction_transport','synchronous_transport','no_doppler']
fields=[
 ('IoU',lambda d:d['metrics']['iou']),
 ('AP',lambda d:d['metrics']['ap_hist4096']),
 ('F1',lambda d:d['metrics']['f1']),
 ('Recall',lambda d:d['metrics']['recall']),
 ('MoveRec',lambda d:d['moving_footprint_recall']),
 ('H1',lambda d:d['per_horizon'][0]['ap_hist4096']),
 ('H2',lambda d:d['per_horizon'][1]['ap_hist4096']),
 ('H3',lambda d:d['per_horizon'][2]['ap_hist4096']),
]
print('='*122)
print(f"{'METHOD':27s}"+''.join(f'{n:>14s}' for n,_ in fields))
print('='*122)
for m in methods:
    rows=[]
    for s in range(3):
        f=root/m/f'seed_{s}'/'test_clean/metrics.json'
        if f.exists(): rows.append(json.loads(f.read_text()))
    if len(rows)!=3:
        print(f'{m:27s} INCOMPLETE {len(rows)}/3')
        continue
    line=f'{m:27s}'
    for _,fn in fields:
        vals=[fn(r) for r in rows]
        line += f'{statistics.mean(vals):7.4f}±{statistics.stdev(vals):5.4f}'
    print(line)
print('='*122)
print('AP = AP_hist4096 grid-cell AP, NOT nuScenes detection mAP.')
