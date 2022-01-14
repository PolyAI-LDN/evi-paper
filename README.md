## Setup
Get data from Google storage:
```
gsutil -m cp -r gs://poly-data/idnv ../tmp/evi
```

## Evaluate Enrolment

```#bash
python eval_e.py --locale en_GB --nlu risk_averse

python eval_e.py --locale en_GB --nlu risk_seeking
```

Analysis for single-turn:
```#bash
python eval_e.py --locale en_GB --nlu risk_averse --model 1

python eval_e.py --locale en_GB --nlu risk_averse --model 2

python eval_e.py --locale en_GB --nlu risk_averse --model 3
```

## Evaluate Verification

```#bash
python eval_v.py --locale en_GB --nlu risk_averse --model random -p

python eval_v.py --locale en_GB --nlu risk_averse --model exact -p

python eval_v.py --locale en_GB --nlu risk_averse --model fuzzy -p

python eval_v.py --locale en_GB --nlu risk_seeking --model random -p

python eval_v.py --locale en_GB --nlu risk_seeking --model exact -p

python eval_v.py --locale en_GB --nlu risk_seeking --model fuzzy -p
```

Early termination:
```
#  same as above with --thresh 0.0,
using the threshold for the desired security level 
```

## Evaluate Identification

```#bash

python eval_i.py --locale en_GB --nlu risk_averse --model none

python eval_i.py --locale en_GB --nlu risk_seeking --model none

python eval_i.py --locale en_GB --nlu risk_averse --model exact-1

python eval_i.py --locale en_GB --nlu risk_averse --model fuzzy-1

python eval_i.py --locale en_GB --nlu risk_seeking --model exact-1

python eval_i.py --locale en_GB --nlu risk_seeking --model fuzzy-1

python eval_i.py --locale en_GB --nlu risk_averse --model exact-0.5

python eval_i.py --locale en_GB --nlu risk_averse --model fuzzy-0.5

python eval_i.py --locale en_GB --nlu risk_seeking --model exact-0.5

python eval_i.py --locale en_GB --nlu risk_seeking --model fuzzy-0.5

python eval_i.py --locale en_GB --nlu risk_averse --model oracle

python eval_i.py --locale en_GB --nlu risk_seeking --model oracle
```


Oracle for KB:
```#bash
python eval_i.py --locale en_GB --nlu risk_seeking --model none --kbo

python eval_i.py --locale en_GB --nlu risk_seeking --model exact-1 --kbo

python eval_i.py --locale en_GB --nlu risk_seeking --model fuzzy-1 --kbo

python eval_i.py --locale en_GB --nlu risk_seeking --model exact-0.5 --kbo

python eval_i.py --locale en_GB --nlu risk_seeking --model fuzzy-0.5 --kbo

python eval_i.py --locale en_GB --nlu risk_seeking --model oracle --kbo
```