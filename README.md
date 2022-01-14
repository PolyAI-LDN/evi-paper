# EVI

## Dataset

Audios are available here [URL].

## Benchmarks

Scripts to replicate the results of experiments in the paper.

### Enrolment Experiments

```#bash
python eval_e.py --locale en_GB --nlu cautious
python eval_e.py --locale en_GB --nlu seeking
```

Analysis for multi- vs single-turn:
```#bash
python eval_e.py --locale en_GB --nlu cautious --model 0  # multi
python eval_e.py --locale en_GB --nlu cautious --model 1  # single
python eval_e.py --locale en_GB --nlu cautious --model 2  # single
python eval_e.py --locale en_GB --nlu cautious --model 3  # single
```

### Verification Experiments

```#bash
python eval_v.py --locale en_GB --nlu cautious --model random
python eval_v.py --locale en_GB --nlu cautious --model exact
python eval_v.py --locale en_GB --nlu cautious --model fuzzy
python eval_v.py --locale en_GB --nlu seeking --model random
python eval_v.py --locale en_GB --nlu seeking --model exact
python eval_v.py --locale en_GB --nlu seeking --model fuzzy
```

Early termination:
```
# same as above with --thresh 0.0,
# using the threshold for the desired security level 
```

### Identification Experiments

```#bash
python eval_i.py --locale en_GB --nlu cautious --model none
python eval_i.py --locale en_GB --nlu seeking --model none
python eval_i.py --locale en_GB --nlu cautious --model exact-1
python eval_i.py --locale en_GB --nlu cautious --model fuzzy-1
python eval_i.py --locale en_GB --nlu seeking --model exact-1
python eval_i.py --locale en_GB --nlu seeking --model fuzzy-1
python eval_i.py --locale en_GB --nlu cautious --model exact-0.5
python eval_i.py --locale en_GB --nlu cautious --model fuzzy-0.5
python eval_i.py --locale en_GB --nlu seeking --model exact-0.5
python eval_i.py --locale en_GB --nlu seeking --model fuzzy-0.5
python eval_i.py --locale en_GB --nlu cautious --model oracle
python eval_i.py --locale en_GB --nlu seeking --model oracle
```

Analysis with KB oracle:
```#bash
python eval_i.py --locale en_GB --nlu seeking --model none --kbo
python eval_i.py --locale en_GB --nlu seeking --model exact-1 --kbo
python eval_i.py --locale en_GB --nlu seeking --model fuzzy-1 --kbo
python eval_i.py --locale en_GB --nlu seeking --model exact-0.5 --kbo
python eval_i.py --locale en_GB --nlu seeking --model fuzzy-0.5 --kbo
python eval_i.py --locale en_GB --nlu seeking --model oracle --kbo
```