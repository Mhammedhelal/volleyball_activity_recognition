
# Volleyball Activity Recognition

Project layout:

```
volleyball_activity_recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── default.yaml            # base hyperparameters
│   ├── 2group.yaml             # overrides for 2-subgroup style
│   └── 4group.yaml             # overrides for 4-subgroup style
│
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── labels.py           # PERSON_ACTIONS, GROUP_ACTIVITIES, SUBGROUP_ACTIVITIES
│   │   ├── dataset.py          # VolleyballDataset (torch Dataset)
│   │   ├── transforms.py       # crop, resize, normalize pipelines
│   │   └── splits.py           # TRAIN_VIDEOS, VAL_VIDEOS, TEST_VIDEOS
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── person_embedder.py 
│   │   ├── subgroup_pooler.py   
│   │   ├── frame_descriptor.py # FrameDescriptor
│   │   └── hierarchical_model.py    
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── subgroup.py         # make_subgroup_indices()
│   │   ├── metrics.py          # accuracy, confusion matrix helpers
│   │   └── checkpointing.py    # save / load checkpoint logic
│   │
│   └── engine/
│       ├── __init__.py
│       ├── trainer.py          # training loop
│       ├── evaluator.py        # validation / test loop
│       └── losses.py           # combined loss (group + person + subgroup)
│
├── scripts/
│   ├── train.py                # entry point: python scripts/train.py
│   ├── evaluate.py             # entry point: python scripts/evaluate.py
│   └── predict.py              # run inference on a single sample
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_pipeline_smoke_test.ipynb
│   └── 03_results_analysis.ipynb
│
├── outputs/
│   ├── checkpoints/            # model_epoch_*.pt
│   ├── logs/                   # tensorboard / wandb logs
│   └── figures/                # confusion matrices, plots
│
└── tests/
 ├── test_dataset.py
 ├── test_person_embedder.py
 ├── test_subgroup_pooler.py
 ├── test_frame_descriptor.py
 └── test_hierarchical.py
```

Fillers in this repository are placeholders; see `configs/default.yaml` for
default hyperparameters and `src/config.py` for the programmatic defaults.
