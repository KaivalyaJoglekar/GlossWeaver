"""Deployment mixture; preserves old evaluation sources and authentic devtest."""
import sys, random, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from glossweaver.settings import ROOT
from glossweaver.utils import read_jsonl, write_json, sha256_file
from glossweaver.data.split_integrity import source_key, group_split, assert_source_isolation


def main():
    rng = random.Random(57)
    reconstruction = ROOT / 'datasets/processed/reconstruction_grouped'
    adaptation = ROOT / 'datasets/processed/demo_adaptation'
    train = pd.read_csv(reconstruction/'train.csv').fillna('').to_dict('records')
    validation = pd.read_csv(reconstruction/'validation.csv').fillna('').to_dict('records')
    test = pd.read_csv(reconstruction/'test.csv').fillna('').to_dict('records')
    timed_train = pd.read_csv(adaptation/'train.csv').fillna('').to_dict('records')
    timed_val = pd.read_csv(adaptation/'validation.csv').fillna('').to_dict('records')
    timed_test = pd.read_csv(adaptation/'test.csv').fillna('').to_dict('records')
    aslg_train = list(read_jsonl(ROOT/'datasets/processed/aslg_grouped/train.jsonl'))
    aslg_val = list(read_jsonl(ROOT/'datasets/processed/aslg_grouped/validation.jsonl'))
    aslg_test = list(read_jsonl(ROOT/'datasets/processed/aslg_grouped/test.jsonl'))
    protected = {source_key(r['gloss']) for r in validation+test+timed_val+timed_test+aslg_val+aslg_test}
    protected.update(source_key(s) for s in ['BOY NOT WANT FOOD','ME GO STORE YESTERDAY','WOMAN BUY CAR RED','i going school tomorrow','I NOT BUY 3 BOOK YESTERDAY'])
    original_train_keys = {source_key(r['gloss']) for r in train+timed_train+aslg_train}
    subjects = [('ME','I',False),('I','I',False),('YOU','You',False),('WE','We',False),('THEY','They',False),('HE','He',True),('SHE','She',True),('BOY','The boy',True),('GIRL','The girl',True),('WOMAN','The woman',True),('MAN','The man',True),('CHILD','The child',True)]
    actions = [('WANT FOOD','want food','wants food'),('NEED HELP','need help','needs help'),('LIKE MUSIC','like music','likes music'),('DRINK WATER','drink water','drinks water'),('EAT RICE','eat rice','eats rice'),('READ BOOK','read a book','reads a book'),('OPEN DOOR','open the door','opens the door'),('CLOSE WINDOW','close the window','closes the window'),('BUY RED CAR','buy a red car','buys a red car'),('PUT BOOK ON TABLE','put the book on the table','puts the book on the table')]
    for place in ['store','school','park','office','hospital','library','market','station']:
        actions.append((f'GO {place.upper()}',f'go to the {place}',f'goes to the {place}'))
        actions.append((f'WALK {place.upper()}',f'walk to the {place}',f'walks to the {place}'))
    rows=[]
    for gloss_subject, subject, singular in subjects:
        for action, verb, third in actions:
            for negative in [False,True]:
                gloss = f'{gloss_subject} '+('NOT ' if negative else '')+action
                if source_key(gloss) in protected: continue
                predicate = ('does not ' if singular else 'do not ')+verb if negative else (third if singular else verb)
                rows.append({'id':f'untimed-{len(rows)}','gloss':gloss,'target':f'{subject} {predicate}.','corpus_origin':'untimed_present_default'})
    groups=group_split(rows,seed=57)
    fresh_train=[r for r in groups if r['split']=='train']
    fresh_val=[r for r in groups if r['split']=='validation' and source_key(r['gloss']) not in original_train_keys]
    fresh_test=[r for r in groups if r['split']=='test' and source_key(r['gloss']) not in original_train_keys]
    held={source_key(r['gloss']) for r in fresh_val+fresh_test}
    def clean(r):
        return 2 <= len(r['target'].split()) <= 22 and not re.search(r'[\[\]"_]|https?://',r['target']) and len(r['gloss'].split())<=26
    def sample(records,n):
        records=[r for r in records if clean(r) and source_key(r['gloss']) not in held]; rng.shuffle(records); return records[:n]
    public=[r for r in train if r['corpus_origin']=='public_librispeech_clean']
    controlled=[r for r in train if r['corpus_origin'].startswith('controlled_')]
    mixed=sample(controlled,4500)+sample(public,3000)+sample(aslg_train,3000)+sample(timed_train,3500)+fresh_train*4
    # Repeat fresh sources intentionally for coverage; IDs are unique and sources stay in one split.
    mixed=[dict(r,id=f'balanced-train-{i}',corpus_origin=r.get('corpus_origin','aslg_pc12')) for i,r in enumerate(mixed) if source_key(r['gloss']) not in protected]
    val=sample(validation,350)+sample(aslg_val,150)+sample(timed_val,150)+fresh_val
    val=[dict(r,id=f'balanced-val-{i}',corpus_origin=r.get('corpus_origin','aslg_pc12')) for i,r in enumerate(val)]
    assert_source_isolation(mixed,val,fresh_test)
    output=ROOT/'datasets/processed/balanced_adaptation'; output.mkdir(parents=True,exist_ok=True)
    for split,records in [('train',mixed),('validation',val),('test',fresh_test)]:
        pd.DataFrame([{k:r[k] for k in ['id','gloss','target','corpus_origin']} for r in records]).to_csv(output/f'{split}.csv',index=False)
    write_json(output/'manifest.json',{'seed':57,'counts':{'train':len(mixed),'validation':len(val),'untimed_test':len(fresh_test)},'policy':'Absent explicit time, present tense is a documented default, not inferred intent. Human Flores splits untouched. Deployment continuation, not matched ablation.','source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in [reconstruction/'train.csv',adaptation/'train.csv',ROOT/'datasets/processed/aslg_grouped/train.jsonl']},'normalized_source_overlap':0, 'untimed_training_repeat_factor':4, 'training_origins':dict(pd.Series([r['corpus_origin'] for r in mixed]).value_counts().items()), 'output_sha256':{name:sha256_file(output/name) for name in ['train.csv','validation.csv','test.csv']}})
    print({'train':len(mixed),'validation':len(val),'untimed_test':len(fresh_test)})

if __name__=='__main__': main()
