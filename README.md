[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

# EVI

This repo contains the code and data
of the [paper](https://arxiv.org/abs/2204.13496):

*EVI: Multilingual Spoken Dialogue Tasks and Dataset for
Knowledge-Based Enrolment, Verification, and Identification*

## Dataset

This repo contains a challenging spoken multilingual dataset
with 5,506 dialogues in English, Polish, and French
that can be used for benchmarking and developing
knowledge-based enrolment, identification, and identification
for spoken dialogue systems.
The data include the ASR n-best list transcriptions 
and can be used to replicate the results in the paper.

Raw audios are available to download [here](https://poly-public-data.s3.eu-west-2.amazonaws.com/evi-paper/audios.zip),
in case you want to experiment with different ASR systems.

## Benchmarks

This repo includes all scripts
to replicate the results of experiments in the paper.

### Setup

The following scripts assume `python 3.9`.

Install all requirments:

```
pip install -r requirements.txt
```


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

## Citations

When using this dataset in your work,
please cite our [paper](https://arxiv.org/abs/2204.13496):

*EVI: Multilingual Spoken Dialogue Tasks and Dataset for
Knowledge-Based Enrolment, Verification, and Identification*