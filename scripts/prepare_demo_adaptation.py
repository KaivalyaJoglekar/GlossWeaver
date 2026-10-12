#!/usr/bin/env python3
"""Create a separate synthetic adaptation corpus with broad explicit time cues.
This deployment recipe is not a matched research ablation or authentic-ASL data.
"""
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from glossweaver.settings import ROOT
from glossweaver.data.split_integrity import group_split, source_key, assert_source_isolation
from glossweaver.utils import write_json, sha256_file


def main():
    base = ROOT / 'datasets/processed/reconstruction_grouped'
    old_train = pd.read_csv(base / 'train.csv').to_dict('records')
    old_validation = pd.read_csv(base / 'validation.csv').to_dict('records')
    old_test = pd.read_csv(base / 'test.csv').to_dict('records')
    # Retain prior evaluation partitions and the reported development probes.
    forbidden = {source_key(r['gloss']) for r in old_validation + old_test}
    probes = ['ME GO STORE YESTERDAY', 'BOY NOT WANT FOOD', 'WOMAN BUY CAR RED',
              'i going school tomorrow', 'I NOT BUY 3 BOOK YESTERDAY']
    forbidden.update(map(source_key, probes))
    subjects = [('ME','I'), ('I','I'), ('WE','We'), ('YOU','You'), ('HE','He'),
                ('SHE','She'), ('THEY','They'), ('BOY','The boy'), ('GIRL','The girl'),
                ('WOMAN','The woman'), ('MAN','The man'), ('CHILD','The child')]
    actions = []
    for place in ['store','school','market','park','library','hospital','station','office']:
        for verb, past in [('go','went'),('walk','walked'),('run','ran'),('travel','traveled')]:
            actions.append((f'{verb.upper()} {place.upper()}', f'{verb} to the {place}', f'{past} to the {place}'))
    for verb, past, noun in [('buy','bought','book'),('read','read','book'),('buy','bought','ticket'),('eat','ate','apple')]:
        for number in [1,2,3,5]:
            noun_phrase = f'{number} {noun}' + ('s' if number != 1 else '')
            actions.append((f'{verb.upper()} {number} {noun.upper()}', f'{verb} {noun_phrase}', f'{past} {noun_phrase}'))
    rows = []
    for gloss_subject, subject in subjects:
        for gloss_action, infinitive, past in actions:
            for marker, time, tense in [('YESTERDAY','yesterday','past'),('LAST NIGHT','last night','past'),('TOMORROW','tomorrow','future'),('NEXT WEEK','next week','future')]:
                for negative in [False,True]:
                    gloss = f'{gloss_subject} ' + ('NOT ' if negative else '') + f'{gloss_action} {marker}'
                    if source_key(gloss) in forbidden:
                        continue
                    predicate = ('did not ' + infinitive if negative else past) if tense == 'past' else ('will not ' if negative else 'will ') + infinitive
                    rows.append({'id': f'time-quantity-{len(rows):05d}', 'gloss':gloss,
                                 'target':f'{subject} {predicate} {time}.', 'corpus_origin':'explicit_time_quantity'})
    new_split = group_split(rows)
    seen_base = {source_key(r['gloss']) for r in old_train}
    groups = {s: [r for r in new_split if r['split']==s and (s=='train' or source_key(r['gloss']) not in seen_base)] for s in ['train','validation','test']}
    # Keep an equal amount of old controlled data to retain word-order/condition skills.
    new_keys = {source_key(x['gloss']) for x in new_split}
    selected = [r for r in old_train if str(r['corpus_origin']).startswith('controlled_')
                and source_key(r['gloss']) not in new_keys]
    random.Random(42).shuffle(selected)
    groups['train'] += selected[:len(groups['train'])]
    old_dev = [r for r in old_validation if str(r['corpus_origin']).startswith('controlled_')]
    groups['validation'] += old_dev
    assert_source_isolation(groups['train'],groups['validation'],groups['test'])
    assert not seen_base & {source_key(r['gloss']) for r in groups['test'] + groups['validation'] if r['corpus_origin']=='explicit_time_quantity'}
    output = ROOT / 'datasets/processed/demo_adaptation'; output.mkdir(parents=True,exist_ok=True)
    for split, records in groups.items():
        pd.DataFrame([{key:r[key] for key in ['id','gloss','target','corpus_origin']} for r in records]).to_csv(output/f'{split}.csv',index=False)
    write_json(output/'manifest.json', {'purpose':'separate synthetic deployment adaptation; not matched research',
        'source_sha256':{p.name:sha256_file(p) for p in [base/'train.csv',base/'validation.csv',base/'test.csv']},
        'seed':42,'counts':{s:len(r) for s,r in groups.items()},'excluded_development_probes':probes,
        'normalized_source_overlap':0,'evaluation_excludes_initial_checkpoint_training_sources':True})
    print({s:len(r) for s,r in groups.items()})

if __name__=='__main__': main()
