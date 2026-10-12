"""Repeatable development comparison. Does not touch Flores devtest."""
import sys, json, gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pandas as pd
from glossweaver.settings import ROOT
from glossweaver.utils import write_json
from glossweaver.evaluation.evaluate import evaluate_model
from glossweaver.inference import Reconstructor

PROBES=['ME GO STORE YESTERDAY','BOY NOT WANT FOOD','SHE NEED HELP','THEY LIKE MUSIC','SHE PUT BOOK ON TABLE','i going school tomorrow','WOMAN BUY CAR RED','I NOT BUY 3 BOOK YESTERDAY','SHE NOT GO HOSPITAL TOMORROW','WE READ 5 BOOK LAST NIGHT','ME WALK LIBRARY NEXT WEEK']

def main():
    out=ROOT/'results/deployment_comparison'; out.mkdir(parents=True,exist_ok=True)
    rows=[]; probes={}
    for model in ['reconstruction-flan','balanced-flan']:
        for dataset,file,limit in [('balanced_development',ROOT/'datasets/processed/balanced_adaptation/validation.csv',120),('untimed_present_diagnostic',ROOT/'datasets/processed/balanced_adaptation/test.csv',None),('time_quantity_regression',ROOT/'datasets/processed/demo_adaptation/test.csv',80)]:
            rows.append(evaluate_model(model,file,out/f'{model}-{dataset}.csv',dataset_name=dataset,split='development_diagnostic',max_examples=limit,num_beams=4,batch_size=16,include_bertscore=False,device='mps',seed=57))
            gc.collect(); torch.mps.empty_cache()
        r=Reconstructor(model,device='mps')
        probes[model]=[{'input':text,'output':r.reconstruct(text)[0]} for text in PROBES]
        del r;gc.collect();torch.mps.empty_cache()
    write_json(out/'summary.json',{'metrics':rows,'probes':probes,'scope':'Development comparison; balanced validation used for early stopping; untimed diagnostics use a narrow generator; no authentic-ASL or final paper claim.'})
    print(json.dumps(probes,indent=2))

if __name__=='__main__': main()
