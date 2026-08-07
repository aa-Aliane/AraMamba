# Tree View:
```
results_finetune
├── ablation_no_pretrain
│   ├── anercorp
│   │   ├── anercorp_best.pt
│   │   ├── anercorp_seed1337.json
│   │   ├── anercorp_seed2024.json
│   │   └── anercorp_seed42.json
│   ├── arcd
│   │   ├── arcd_best.pt
│   │   ├── arcd_seed1337.json
│   │   ├── arcd_seed2024.json
│   │   └── arcd_seed42.json
│   └── xnli_ar
│       ├── xnli_ar_best.pt
│       ├── xnli_ar_seed1337.json
│       ├── xnli_ar_seed2024.json
│       └── xnli_ar_seed42.json
└── ablation_nopretrain
    └── hard
        ├── hard_best.pt
        ├── hard_seed1337.json
        ├── hard_seed2024.json
        └── hard_seed42.json

```

# Content:

## ablation_no_pretrain/anercorp/anercorp_seed1337.json

```json
{
  "task": "anercorp",
  "seed": 1337,
  "world_size": 1,
  "dev_best": 0.6403564608529599,
  "test": {
    "precision": 0.603414313854235,
    "recall": 0.40254051686377573,
    "f1": 0.48292170257488176
  }
}
```


## ablation_no_pretrain/anercorp/anercorp_seed2024.json

```json
{
  "task": "anercorp",
  "seed": 2024,
  "world_size": 1,
  "dev_best": 0.6463730569948186,
  "test": {
    "precision": 0.610204081632653,
    "recall": 0.392904073587385,
    "f1": 0.478017585931255
  }
}
```


## ablation_no_pretrain/anercorp/anercorp_seed42.json

```json
{
  "task": "anercorp",
  "seed": 42,
  "world_size": 1,
  "dev_best": 0.6513994910941475,
  "test": {
    "precision": 0.5980392156862745,
    "recall": 0.4007884362680683,
    "f1": 0.47993705743509046
  }
}
```


## ablation_no_pretrain/arcd/arcd_seed1337.json

```json
{
  "task": "arcd",
  "seed": 1337,
  "world_size": 1,
  "dev_best": 12.433646362875184,
  "test": {
    "exact_match": 0.5698005698005698,
    "f1": 12.571740350299766
  }
}
```


## ablation_no_pretrain/arcd/arcd_seed2024.json

```json
{
  "task": "arcd",
  "seed": 2024,
  "world_size": 1,
  "dev_best": 11.559060264363362,
  "test": {
    "exact_match": 1.4245014245014245,
    "f1": 12.44899791241109
  }
}
```


## ablation_no_pretrain/arcd/arcd_seed42.json

```json
{
  "task": "arcd",
  "seed": 42,
  "world_size": 1,
  "dev_best": 12.635407659046342,
  "test": {
    "exact_match": 1.4245014245014245,
    "f1": 12.51368231769839
  }
}
```


## ablation_no_pretrain/xnli_ar/xnli_ar_seed1337.json

```json
{
  "task": "xnli_ar",
  "seed": 1337,
  "world_size": 4,
  "dev_best": 0.5160696530248686,
  "test": {
    "accuracy": 0.524750499001996,
    "macro_f1": 0.5183211708670074
  }
}
```


## ablation_no_pretrain/xnli_ar/xnli_ar_seed2024.json

```json
{
  "task": "xnli_ar",
  "seed": 2024,
  "world_size": 4,
  "dev_best": 0.5159787702775033,
  "test": {
    "accuracy": 0.5323353293413173,
    "macro_f1": 0.5270614747426249
  }
}
```


## ablation_no_pretrain/xnli_ar/xnli_ar_seed42.json

```json
{
  "task": "xnli_ar",
  "seed": 42,
  "world_size": 4,
  "dev_best": 0.5206109977267102,
  "test": {
    "accuracy": 0.526746506986028,
    "macro_f1": 0.5211555028025114
  }
}
```


## ablation_nopretrain/hard/hard_seed1337.json

```json
{
  "task": "hard",
  "seed": 1337,
  "world_size": 4,
  "dev_best": 0.9493764271919165,
  "test": {
    "accuracy": 0.9541197616119572,
    "macro_f1": 0.9541170007330544
  }
}
```


## ablation_nopretrain/hard/hard_seed2024.json

```json
{
  "task": "hard",
  "seed": 2024,
  "world_size": 4,
  "dev_best": 0.9514561745548077,
  "test": {
    "accuracy": 0.9532683757449626,
    "macro_f1": 0.953266267529376
  }
}
```


## ablation_nopretrain/hard/hard_seed42.json

```json
{
  "task": "hard",
  "seed": 42,
  "world_size": 4,
  "dev_best": 0.9501330794623856,
  "test": {
    "accuracy": 0.9520385961593038,
    "macro_f1": 0.9520349631350964
  }
}
```

