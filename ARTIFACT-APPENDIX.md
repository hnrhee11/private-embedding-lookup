# Artifact Appendix

Paper title: **Private Embedding Lookup with Encrypted Compact Queries under Fully Homomorphic Encryption**

Requested Badge(s):

- [x] **Available**
- [ ] **Functional**
- [ ] **Reproduced**

## Description

This artifact accompanies the PoPETs 2027 paper *Private Embedding Lookup with Encrypted Compact Queries under Fully Homomorphic Encryption* by Daehyun Jang, Jaehee Kang, Hanee Rhee, and Jung Hee Cheon.

It contains the implementation of **IVE-PEL**, the private embedding lookup protocol introduced in the paper, together with the baseline method from the ICML 2024 paper *Privacy-Preserving Embedding via Look-up Table Evaluation with Fully Homomorphic Encryption* by Jae-yun Kim, Saerom Park, Joohee Lee, and Jung Hee Cheon. It also includes the preprocessed data used by both implementations.

Four programs are provided:

- `ive_pcmm.cpp` and `ICML24_pcmm.cpp` correspond to the private embedding lookup experiments in Table 3.
- `ive_fasttext_spam.cpp` and `ICML24_fasttext_spam.cpp` correspond to the end-to-end encrypted FastText experiments in Table 5 and Section 6.2.
- `basis_generation.hpp` implements Algorithm 1 and the corresponding baseline construction.
- `table3_params.hpp` and `table5_params.hpp` contain the CKKS parameter sets used in the experiments.

### Security/Privacy Issues and Ethical Concerns

The artifact does not disable security mechanisms or contain malicious code. The programs read local data files, perform homomorphic computations, and print experimental results.

No new human-subject data were collected for this artifact. The included preprocessed Enron-Spam data contain only preprocessed token indices and class labels; no raw email text is distributed. The remaining data are derived from publicly available GloVe and GPT-2 resources.

## Environment

The implementation uses **HEaaN2**, a CKKS library from CryptoLab. HEaaN2 is not redistributed with this artifact and remains subject to CryptoLab's separate license terms. It is available through **CODE.HEAAN** (<https://code.heaan.io/>). Instructions for using the artifact with CODE.HEAAN are provided in `README.md`.

The artifact consists of C++ source files, CMake build configuration, and preprocessed datasets. The source files and documentation are available in the GitHub repository, and the complete artifact archive, including the preprocessed datasets, is provided under the repository's Releases section.

The preprocessing and model-training procedures, including the relevant hyperparameters, are described in detail in the paper; the processed outputs required to run the provided experiments are included in this artifact.

### Accessibility

The artifact is publicly available through the GitHub repository provided with the artifact submission.

The complete artifact, including the source code, `CMakeLists.txt`, preprocessed datasets, `README.md`, `LICENSE`, and this appendix, is provided as `Private_Embedding_Lookup.zip` under the repository's Releases section. The source code and documentation are also available directly in the repository for convenient inspection.

The Drugs.com Review Dataset was used in the paper but is not redistributed because its original terms prohibit redistribution. Its provenance and citation are provided in the paper.

The authors' code is covered by `LICENSE`. Third-party software and data are not covered by that license and remain subject to their original terms, as documented in `README.md`.

## Notes on Reusability

The two private lookup constructions (Ours(ive) and ICML24) are implemented separately in `basis_generation.hpp` and can be incorporated into other CKKS-based pipelines.