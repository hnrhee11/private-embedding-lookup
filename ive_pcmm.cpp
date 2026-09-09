//==============================================================================
// IVE-PEL, server side: the Emb step of Protocol IVE-PEL in
//
//   "Private Embedding Lookup with Encrypted Compact Queries under Fully
//    Homomorphic Encryption"
//
//     (ct_out,k)_{k in [d]} = (PCMM_{U} o IVE)(ct),   U = M_L / sqrt(2p)
//
// One ciphertext of packed queries goes in, d ciphertexts of embedding rows
// come out. This is Figure 2 of the paper:
//
//   (1)(2)   query encryption             Section 5.1
//   (3)      IVE                          Algorithm 1  -> pel::ive
//   (0)(0-b) plaintext table U, offline   footnote 2
//   (4)-(9)  U . IVE(ct)                  PCMM
//   (10)     precision                    Table 3
//
// Notation: d = M (embedding dimension), p = sub-table size, ell = 4
// sub-tables per token, K = ell * p, N = degree, s = N/2 slots.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include "basis_generation.hpp"
#include "pel_helpers.hpp"
#include "table3_params.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <omp.h>




using namespace heaan;

static int runExample(int argc, char *argv[]) try {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <p> [dataset]\n"
                  << "  p       : 4, 16, 64, 256, 1024\n"
                  << "  dataset : " << pel::dataset_names()
                  << "  (default: glove300d)\n"
                  << "  e.g.  " << argv[0] << " 64 gpt2\n";
        return 1;
    }

    omp_set_num_threads(1);
    

    const int p = std::stoi(argv[1]);
    if (p != 4 && p != 16 && p != 64 && p != 256 && p != 1024) {
        std::cerr << "p must be one of 4, 16, 64, 256, 1024\n";
        return 1;
    }

    const std::string dataset_name = (argc >= 3) ? argv[2] : "glove300d";
    const pel::DatasetInfo *dataset = pel::find_dataset(dataset_name);
    if (!dataset) {
        std::cerr << "unknown dataset '" << dataset_name
                  << "'. available: "
                  << pel::dataset_names() << "\n";
        return 1;
    }

    //==========================================================================
    // Parameters -- Table 4 of the paper; see table3_params.hpp
    //==========================================================================
    const u32 log_degree = pel::TABLE3_LOG_DEGREE; // 17
    const u32 log_slots = pel::TABLE3_LOG_SLOTS;   // 16
    const u32 degree = 1u << log_degree;           // N = 131072

    const int M = dataset->M; // d in the paper; fixed by the dataset
    const int ell = 4;        // ell in the paper; fixed to 4 by the datasets
    const int K = p * ell;
    const int num_blocks = ell; // one vector-generation block per sub-table
    const int log_p = (int)std::log2((double)p);

    // The query enters at level log2(p) and the Algorithm 1 L16 constant at
    // level 1. IVE leaves everything at level 1, so all K ciphertexts share a
    // modulus and go straight into PCMM (depth: Table 2).
    const u32 enc_level_seed = (u32)log_p;
    const u32 enc_level_const = 1;

    // q0 is 60-bit so that the integer PCMM accepts it; see table3_params.hpp.
    auto levels = pel::makeTable3Levels(pel::TABLE3_NUM_PRIMES);

    std::cout << "dataset=" << dataset_name << " M=" << M << " p=" << p
              << " degree=" << degree << "\n";
    std::cout << "enc_level(seed)=" << enc_level_seed
              << " enc_level(const)=" << enc_level_const << "\n";
   

    //==========================================================================
    // Key generation. Algorithm 1 needs no rotation: a relinearisation key and
    // a conjugation key (Algorithm 1 L11) suffice.
    //==========================================================================
    // PolyType::SIMPLE is the ordinary RNS system, rescaling one prime at a
    // time; NTTAlgorithm::NORMAL (the default) is the NTT for the usual
    // cyclotomic ring. The conjugate-invariant variant cannot be used here,
    // since Algorithm 1 L11 needs conjugation.
    SKGenParams skgen_params(log_degree, pel::TABLE3_HAMMING_WEIGHT);
    SKGenerator skgen(skgen_params);

    std::cout << "Generate keys ... " << std::flush;
    auto sk = skgen.genKey();

    paramsUtils::SwKeyGenParamsBuilder swkgen_builder;
    swkgen_builder.setRing(log_degree, PolyType::SIMPLE);
    swkgen_builder.setModUpPrimes(pel::TABLE3_QP_MAX_BITS, 0.0);
    auto swkgen_params = swkgen_builder.build(levels.mods, false);
    SwKeyGenerator swkgen(swkgen_params);

    auto relin_key = swkgen.genRelinKey(*sk);
    auto conj_key = swkgen.genConjKey(*sk);
    std::cout << "done\n";

    // Encoder for messages and ciphertexts: slot encoding.
    EncodeParams ecd_params(PolyType::SIMPLE, log_degree, levels);
    EnDecoder encoder(ecd_params);

    // Encoder for the plaintext table U. pcmm accepts coefficient-encoded
    // operands only, so this one sets coeff_encoding. NTTAlgorithm::NORMAL is
    // just the default, repeated to reach the fifth argument.
    EncodeParams mat_ecd_params(PolyType::SIMPLE, log_degree, levels,
                                /*ntt_alg=*/NTTAlgorithm::NORMAL,
                                /*coeff_encoding=*/true);
    EnDecoder mat_encoder(mat_ecd_params);

    DiscreteGaussian noise_dist(/*sigma=*/3.2);
    EncryptParams enc_params(noise_dist);
    EnDecryptor encryptor(enc_params);

    // Public-key encryption, as in Protocol IVE-PEL: the client holds sk, the
    // server sees only ciphertexts. sk is used below only for key generation
    // and for the final decryption. Public-key noise sits about sqrt(N) above
    // secret-key noise.
    EncKeyGenParams enckeygen_params(noise_dist, PolyType::SIMPLE,
                                     levels.mods[levels.top()]);
    EncKeyGenerator enckeygen(enckeygen_params);
    auto enc_key = enckeygen.genKey(*sk);
    HomEval eval{HomEvalParams(levels)}; // arithmetic; needs the prime chain
    HomEvalFlexible eval_flex;           // batching and encoding-flag surgery

    //==========================================================================
    // (1) Random queries: ell blocks of s = N/2 slots, each holding alpha_j
    //==========================================================================
    std::vector<Message> msg;
    msg.reserve(num_blocks);
    for (int b = 0; b < num_blocks; ++b)
        msg.emplace_back(log_slots);

    std::vector<std::vector<int>> vec_int(num_blocks);
    const std::vector<std::uint64_t> seeds = {111, 222, 333, 444};

    for (int b = 0; b < num_blocks; ++b) {
        std::mt19937_64 rng_b(seeds[b]);
        pel::fillRandomIndex(msg[b], vec_int[b], p, rng_b);
    }

    // Algorithm 1 L16: the constant Enc(sqrt(2)), sqrt(2) in every slot
    Message msg_last(log_slots);
    const double cst_last = std::sqrt(2.0);
    for (size_t i = 0; i < (size_t{1} << log_slots); ++i) {
        msg_last[i].real(cst_last);
        msg_last[i].imag(0.0);
    }

    //==========================================================================
    // (2) Encrypt -- Section 5.1. Query at the block base, the Algorithm 1 L16
    //     constant Enc(sqrt(2)) at the block end.
    //==========================================================================
    const auto enc_begin = std::chrono::steady_clock::now();

    std::vector<Ptr<ICiphertext>> ctxt((size_t)K);
    for (int i = 0; i < K; ++i)
        ctxt[(size_t)i] = ICiphertext::make();

    auto base_of = [&](int b) { return b * p; };

    for (int b = 0; b < num_blocks; ++b) {
        const int base = base_of(b);
        compat::Encrypt(encoder, encryptor, *enc_key, msg[b], enc_level_seed,
                     *ctxt[(size_t)base]);
        compat::Encrypt(encoder, encryptor, *enc_key, msg_last, enc_level_const,
                     *ctxt[(size_t)(base + p - 1)]);
    }
    const auto enc_end = std::chrono::steady_clock::now();

    //==========================================================================
    // (3) Vector generation -- Algorithm 1, once per block
    //==========================================================================
    const auto basis_begin = std::chrono::steady_clock::now();
    for (int b = 0; b < num_blocks; ++b)
        pel::ive(eval, *relin_key, *conj_key, ctxt, base_of(b), p);
    const auto basis_end = std::chrono::steady_clock::now();

    // pcmm requires all inputs at the same level (same modulus).
    const u32 lv0 = eval.getLevel(*ctxt[0]);
    for (int i = 1; i < K; ++i) {
        const u32 lv = eval.getLevel(*ctxt[(size_t)i]);
        if (lv != lv0) {
            std::cerr << "[fatal] level mismatch: ctxt[0]=" << lv0 << ", ctxt["
                      << i << "]=" << lv << "\n";
            return 1;
        }
    }
    std::cout << "basis done, all " << K << " ctxt at level " << lv0 << "\n";

    //==========================================================================
    // (0) Plaintext table U [d x K]  (offline)
    //==========================================================================
    // U = M_L / sqrt(2p): M_L is the DCT-transformed embedding table, and the
    // 1/sqrt(2p) undoes the constant factor Algorithm 1 puts on its output.
    const std::string bin_path = pel::table_path(*dataset, p);
    const std::string out_path = pel::reference_path(*dataset, p);

    auto u_colmajor = pel::load_colmajor_f64_bin(bin_path, M, K);
    const double u_scale = 1.0 / std::sqrt(static_cast<double>(p * 2));

    // The file is column-major and Matrix<Real> is row-major, so entries go
    // straight to (r, c).
    Matrix<Real> u_mat(log_degree, (u32)M, (u32)K);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            u_mat.at((u32)r, (u32)c) =
                u_colmajor[(size_t)r + (size_t)M * (size_t)c] * u_scale;

    //==========================================================================
    // (0-b) Encode U  [offline]
    //==========================================================================
    // Footnote 2 of the paper counts preparing the table as a one-time offline
    // cost, so it stays outside the online timer below. U has to be encoded at
    // the modulus and scale the ciphertexts carry, or pcmm rejects it.
    const auto ct_ring = ctxt[0]->ring();
    const auto ct_ecd = ctxt[0]->encoding();

    auto pu = IPtMatrix::make();
    mat_encoder.encode(u_mat, *pu, ct_ring.mod, ct_ecd.scale);

    HomEvalMatrix eval_mat{HomEvalParams(levels)}; // plaintext-ciphertext PCMM

    //==========================================================================
    // (4)-(9) PCMM:  out[d x N] = U[d x K] . IVE(ct)[K x N]   [online]
    //==========================================================================
    const auto pcmm_begin = std::chrono::steady_clock::now();

    // A ciphertext matrix only exists in batched form, so the K ciphertexts
    // are packed into one BatchRLWE ciphertext first.
    std::vector<const ICiphertext *> ct_ptrs;
    ct_ptrs.reserve((size_t)K);
    for (int i = 0; i < K; ++i)
        ct_ptrs.push_back(&*ctxt[(size_t)i]);

    auto batch_in = ICiphertext::make(EncType::BatchRLWE);
    eval_flex.batch(ct_ptrs, *batch_in);

    // pcmm rejects anything but coefficient encoding. It is a linear map on
    // coefficients, so lowering the flag and restoring it later is safe:
    // setDFT touches metadata only, never the polynomial data.
    eval_flex.setDFT(*batch_in, /*dft=*/false);

    auto cv = ICtMatrix::make();
    cv->moveFrom(*batch_in, (u32)K, degree);

    auto cw = ICtMatrix::make();
    eval_mat.pcmm(*pu, *cv, *cw);
    eval_mat.rescale(*cw, *cw); // pcmm leaves the scale squared

    // Unwind: ICtMatrix -> BatchRLWE -> slot encoding -> d ciphertexts
    auto batch_out = ICiphertext::make(EncType::BatchRLWE);
    cw->moveTo(*batch_out);
    eval_flex.setDFT(*batch_out, /*dft=*/true); // back to slot encoding

    std::vector<Ptr<ICiphertext>> ctxt_rst((size_t)M);
    std::vector<ICiphertext *> rst_ptrs;
    rst_ptrs.reserve((size_t)M);
    for (int i = 0; i < M; ++i) {
        ctxt_rst[(size_t)i] = ICiphertext::make();
        rst_ptrs.push_back(&*ctxt_rst[(size_t)i]);
    }
    eval_flex.unbatch(*batch_out, rst_ptrs);

    const auto pcmm_end = std::chrono::steady_clock::now();

    std::cout << "pcmm done, result level = "
              << eval.getLevel(*ctxt_rst[0]) << ", rows = " << M << "\n";

    //==========================================================================
    // (10) Decrypt and compare against the plaintext lookup (Table 3)
    //==========================================================================
    pel::measure_column_vectors_rmse_bits(ctxt_rst, encryptor, encoder, *sk,
                                          out_path, 1e-12);

    const auto to_sec = [](const auto begin, const auto end) {
        return std::chrono::duration<double>(end - begin).count();
    };

    std::cout << "\n=== Latency ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Encrypt          (s): " << to_sec(enc_begin, enc_end) << "\n";
    std::cout << "Basis Generation (s): " << to_sec(basis_begin, basis_end)
              << "\n";
    std::cout << "PCMM             (s): " << to_sec(pcmm_begin, pcmm_end)
              << "\n";

    return 0;
} catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
}

int main(int argc, char *argv[]) { return runExample(argc, argv); }
