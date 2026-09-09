# Private Embedding Lookup with Encrypted Compact Queries

Artifact for "Private Embedding Lookup with Encrypted Compact Queries under Fully Homomorphic Encryption" (PoPETs 2027)

This repository provides the source code and documentation for **IVE-PEL**, the private embedding lookup protocol introduced in the paper, together with the baseline method of Kim et al. ("Privacy-Preserving Embedding via Look-up Table Evaluation with Fully Homomorphic Encryption" (ICML 2024)).

The complete artifact, including the preprocessed data used by both implementations, is provided as `Private_Embedding_Lookup.zip` under the **Releases** section of this repository.

---

## Layout

The complete artifact archive has the following layout:

```
Private_Embedding_Lookup/
├── README.md
├── LICENSE
├── ARTIFACT-APPENDIX.md
├── CMakeLists.txt
│
├── ive_pcmm.cpp                  Table 3, ours
├── ICML24_pcmm.cpp               Table 3, baseline (Kim et al.)
├── ive_fasttext_spam.cpp         Table 5, ours
├── ICML24_fasttext_spam.cpp      Table 5, baseline (Kim et al.)
│
├── basis_generation.hpp          pel::ive (Algorithm 1), pel::ICML24
├── pel_helpers.hpp               query generation, dataset paths, precision
├── filtering_data.hpp            Section 6.2 dataset loaders
├── table3_params.hpp             CKKS parameters for Table 3
├── table5_params.hpp             CKKS parameters for Table 5
│
└── data/
    ├── fasttext_spam/            Section 6.2 data
    ├── 50d/                      GloVe.6B.50d      (d = 50)
    ├── 300d/                     GloVe.42B.300d    (d = 300)
    └── gpt2/                     GPT-2             (d = 768)
```

---

## Requirements

The code is written against **HEaaN2**, a homomorphic encryption library from CryptoLab, Inc. HEaaN2 is not included in this archive and cannot be redistributed by us. It is available at **CODE.HEAAN**. Step 0 below explains how to get in.

About 3 GB of disk space is needed for the datasets.

---

## Step 0 — Access CODE.HEAAN

1. Go to <https://code.heaan.io/> and sign up.
2. Click **Go to Workspace**, then **New workspace** at the top right, and choose **HEaaN2 Playground (A100)**.
3. Enter a workspace name and click **Create Workspace**.
4. Wait for the workspace to start. When **code-server** appears under `main`, click it. This opens a browser-based VS Code.

Everything below is done inside that workspace.

---

## Step 1 — Edit CMakeLists.txt

Open

```
/home/user/devkit/examples/heaan2-examples/CMakeLists.txt
```

and add this line at the end, and save:

```
add_subdirectory(my_example/Private_Embedding_Lookup)
```

---

## Step 2 — Download, upload, and unpack the artifact

Download `Private_Embedding_Lookup.zip` from the **Releases** section of this repository.

Upload `Private_Embedding_Lookup.zip` into

```
/home/user/devkit/examples/heaan2-examples/my_example
```

by dragging it onto that folder. The upload may take a few minutes.

Then unpack it in the terminal:

```bash
python3 -m zipfile -e Private_Embedding_Lookup.zip .
```

The result should be:

```
heaan2-examples/
├── CMakeLists.txt                  (edited in Step 1)
└── my_example/
    └── Private_Embedding_Lookup/   (this artifact)
```

---

## Step 3 — Build

```bash
cd /home/user/devkit/examples/heaan2-examples/my_example/Private_Embedding_Lookup
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build
```

---

## Step 4 — Run

### Embedding lookup — Table 3

```
./ive_pcmm     <p> [dataset]
./ICML24_pcmm  <p> [dataset]
```

| Argument | Values |
| --- | --- |
| `p` | `4`, `16`, `64`, `256`, `1024` — sub-table size |
| `dataset` | `glove50d`, `glove300d`, `gpt2` |

### End-to-end FastText inference — Table 5

```
./ive_fasttext_spam
./ICML24_fasttext_spam
```

No arguments: the dataset is Enron-Spam and the sub-table size is fixed to `p = 256` (`log p = 8`), as in Table 5. One batch is processed — 256 examples for our method, 512 for the baseline. Each run prints the time of each stage.

---

## A note on the timings reported in the paper

The measurements in the paper were taken on our own lab server, with HEaaN2 installed locally, on a single pinned CPU core:

```bash
OMP_NUM_THREADS=1 ./ive_pcmm 64 glove300d
```

**They were not measured on CODE.HEAAN.** CODE.HEAAN is a shared service with different hardware, so absolute times obtained there will differ from the numbers in the paper.

---

## Data

Everything under `data/` is preprocessed derivatives of third-party datasets and ready to use.

`data/fasttext_spam/` holds the Section 6.2 experiment: per-example token codes with class labels, the score table in two forms (composed with the DCT matrix for our method, raw for the baseline), and the plaintext model's predictions used as the accuracy reference. Only integer sub-table indices and class labels are stored, not raw email text.

`data/50d/`, `data/300d/` and `data/gpt2/` hold, for each `p`, the embedding table in the same two forms plus the plaintext lookup result used as ground truth for the precision measurement.

Drugs.com Review dataset was used in the paper but is not redistributed because its original usage terms prohibit redistribution. It can be obtained separately from its original source (see our paper)

Training the FastText classifier and compressing the embedding tables are upstream of this artifact and are not included; their outputs are the data files above.

---

## License

See [`LICENSE`](LICENSE).
