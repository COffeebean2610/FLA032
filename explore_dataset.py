import os
import sys

dataset_root = os.path.join('dataset', 'image data')
splits = ['train', 'validation', 'test']
summary = {}

for s in splits:
    spath = os.path.join(dataset_root, s)
    summary[s] = {}
    if not os.path.exists(spath):
        continue
    for crop in sorted(os.listdir(spath)):
        cpath = os.path.join(spath, crop)
        if os.path.isdir(cpath):
            for disease in sorted(os.listdir(cpath)):
                dpath = os.path.join(cpath, disease)
                if os.path.isdir(dpath):
                    class_name = f"{crop}___{disease}"
                    files = [f for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))]
                    summary[s][class_name] = len(files)

all_classes = sorted(list(set().union(*[summary[s].keys() for s in splits])))
print(f"Total Unique Classes across dataset: {len(all_classes)}")
print("Class Breakdown:")
header = f"{'Class Name':<55} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}"
print(header)
print("-" * len(header))
tot_train, tot_val, tot_test = 0, 0, 0
for c in all_classes:
    tr = summary['train'].get(c, 0)
    va = summary['validation'].get(c, 0)
    te = summary['test'].get(c, 0)
    tot = tr + va + te
    tot_train += tr
    tot_val += va
    tot_test += te
    print(f"{c:<55} | {tr:<7} | {va:<7} | {te:<7} | {tot:<7}")
print("=" * len(header))
print(f"{'TOTAL':<55} | {tot_train:<7} | {tot_val:<7} | {tot_test:<7} | {tot_train+tot_val+tot_test:<7}")
