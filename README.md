# SADGA


``` bash
mkdir -p dataset third_party
```

Download the dataset: [Spider](https://yale-lily.github.io/spider). Then unzip `spider.zip` into the directory `dataset`.

```
└── dataset
    ├── database
    │   ├── academic
    │   │   ├──academic.sqlite
    │   │   ├──schema.sql
    │   ├── ...
    ├── dev_gold.sql
    ├── dev.json
    ├── README.txt
    ├── tables.json
    ├── train_gold.sql
    ├── train_others.json
    └── train_spider.json
```

Download and unzip [Stanford CoreNLP](https://download.cs.stanford.edu/nlp/software/stanford-corenlp-full-2018-10-05.zip) to the directory `third_party`. Note that this repository requires a JVM to run it.

```
└── third_party
    └── stanford-corenlp-full-2018-10-05
        ├── ...
```


### Create environment

We trained our models on one server with a single NVIDIA GTX 3090 GPU with 24GB GPU memory. In our experiments, we use **python 3.7**,  **torch 1.7.1** with **CUDA version 11.0**. We create conda environment `sadgasql`:

```bash
    conda create -n sadgasql python=3.7
    source activate sadgasql
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Run

All configs of the experiments and models are in the files  `sadga-bert-run.jsonnet`, `sadga-roberta-run.jsonnet`.

##### Step 1. Preprocess

```bash
    python run.py --mode preprocess --config sadga-[roberta|bert]-run.jsonnet
```

##### Step 2. Training

```bash
    python run.py --mode train --config sadga-[roberta|bert]-run.jsonnet
```

- After the training, we can obtain some model-checkpoints in the directory `{logdir}/{model_name}/`, e.g., `sadga_roberta_bs=6_lr=2.4e-04_blr=3.0e-06/model_checkpoint-00000060`.

##### Step 3. Inference

```bash
    python run.py --mode infer --config sadga-[roberta|bert]-run.jsonnet
```



##### Step 4. Eval

```bash
    python run.py --mode eval --config sadga-[roberta|bert]-run.jsonnet
```

You can download the logdir result directory from the link:[logdir](https://drive.google.com/file/d/1KDizYvhEliAgfiGsFopIF_FxL32AGRbO/view?usp=sharing) . Please download it and run it if you are having issue with running the above 4 steps. Add the logdir to the root folder.
##### Step 5. Cosine Similarity
```bash
    python cosineSimilarity.py
```