//==============================================================================
// End-to-end encrypted FastText inference with IVE-PEL, Section 6.2 and
// Table 5 of
//
//   "Private Embedding Lookup with Encrypted Compact Queries under Fully
//    Homomorphic Encryption"
//
// FastText averages the embedding vectors of a document's tokens and feeds the
// result to a linear classifier. Both steps are linear, so the classifier
// weights are folded into the embedding table offline: the score table U holds
// score[class][sub-table][code] directly. The server then only has to look up
// one score per token and add the scores together.
//
//   (1)(2)  encrypt the token codes           Section 5.1
//   (3)     IVE                               Algorithm 1  -> pel::ive
//   (0)     score table U, offline            footnote 2
//   (4)     U . IVE(ct)                       PCMM
//   (5)     sum over the token axis           7 rotations
//   (6)     decrypt and compare with the plaintext model
//
// Slot layout. Slot (i + stride*k) carries token k of example i, so one
// ciphertext holds stride examples of max_tok tokens each. With stride chosen
// as n / max_tok the slots are exactly filled, and the rotate-sum folds the
// token axis onto slot i, leaving the total score of example i there.
//
// Notation: p = sub-table size, ell = 4 sub-tables per token, K = ell * p,
// N = degree, n = N/2 slots.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include "basis_generation.hpp"
#include "filtering_data.hpp"
#include "pel_helpers.hpp"
#include "table5_params.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <omp.h>




using namespace heaan;

static int runExample() try {
    omp_set_num_threads(1);


    //==========================================================================
    // Parameters -- the "Table 5 / Ours" row of Table 4; see table5_params.hpp
    //==========================================================================
    const u32 log_degree = pel::TABLE5_LOG_DEGREE; // 16
    const u32 log_slots = pel::TABLE5_LOG_SLOTS;   // 15
    const u32 degree = 1u << log_degree;           // 65536
    const int n = 1 << log_slots;                  // 32768 slots

    const int p = pel::FILTER_P;                   // 256
    const int num_blocks = pel::FILTER_NUM_DIGITS; // ell in the paper
    const int K = p * num_blocks;                  // 1024
    const int log_p = (int)std::log2((double)p);   // 8
    const int M_out = pel::FILTER_NUM_CLASSES;     // 2 (ham / spam)
    const int max_tok = pel::FILTER_MAX_TOKENS;    // 128
    const int stride = n / max_tok;                // examples per batch

    // The query enters at level log2(p) and the Algorithm 1 L16 constant at
    // level 1. IVE leaves everything at level 1, so all K ciphertexts share a
    // modulus and go straight into PCMM.
    const u32 enc_level_seed = (u32)log_p;
    const u32 enc_level_const = 1;

    auto levels = pel::makeTable5Levels();

    std::cout << "n=" << n << " p=" << p
              << " degree=" << degree << " stride=" << stride
              << " M_out=" << M_out << "\n";
    std::cout << "enc_level(seed)=" << enc_level_seed
              << " enc_level(const)=" << enc_level_const
              << " (levels.top()=" << levels.top() << ")\n";

    //==========================================================================
    // Key generation: a relinearisation key, a conjugation key for
    // Algorithm 1 L11, and log2(max_tok) rotation keys for the rotate-sum.
    //
    // PolyType::SIMPLE is the ordinary RNS system, rescaling one prime at a
    // time; NTTAlgorithm::NORMAL (the default) is the NTT for the usual
    // cyclotomic ring.
    //==========================================================================
    SKGenParams skgen_params(log_degree, pel::TABLE5_HAMMING_WEIGHT);
    SKGenerator skgen(skgen_params);

    std::cout << "Generating keys ... " << std::flush;
    auto sk = skgen.genKey();

    paramsUtils::SwKeyGenParamsBuilder swkgen_builder;
    swkgen_builder.setRing(log_degree, PolyType::SIMPLE);
    swkgen_builder.setModUpPrimes(pel::TABLE5_QP_MAX_BITS, 0.0);
    auto swkgen_params = swkgen_builder.build(levels.mods, false);
    SwKeyGenerator swkgen(swkgen_params);

    auto relin_key = swkgen.genRelinKey(*sk);
    auto conj_key = swkgen.genConjKey(*sk);

    // A positive step is a left rotation. Steps double so that the sum tree
    // below covers all max_tok token positions in log2(max_tok) steps.
    std::set<i32> rot_steps;
    for (int k = 1; k < max_tok; k <<= 1)
        rot_steps.insert((i32)(k * stride));
    auto rot_keys = swkgen.genRotKeys(*sk, rot_steps);
    std::cout << "done (" << rot_steps.size() << " rotation keys)\n";

    // Encoder for messages and ciphertexts: slot encoding.
    EncodeParams ecd_params(PolyType::SIMPLE, log_degree, levels);
    EnDecoder encoder(ecd_params);

    // Encoder for the score table U. pcmm accepts coefficient-encoded
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
    // (0) Score table U [C x K] and dataset  [offline]
    //==========================================================================
    // Footnote 2 of the paper counts preparing the table as a one-time offline
    // cost, so it is loaded and encoded once, outside the batch loop and
    // outside every timer. U = M_L / sqrt(2p): M_L is the DCT-transformed
    // score table, and the 1/sqrt(2p) undoes the constant factor Algorithm 1
    // puts on its output.
    auto V_colmajor = pel::load_colmajor_f64_bin(
        pel::filterPath(pel::SPAM_SCORE_TRANSFORMED), (std::uint32_t)M_out,
        (std::uint32_t)K);
    const double v_scale = 1.0 / std::sqrt((double)(p * 2));

    // The file is column-major and Matrix<Real> is row-major, so entries go
    // straight to (r, c).
    Matrix<Real> u_mat(log_degree, (u32)M_out, (u32)K);
    for (int r = 0; r < M_out; ++r)
        for (int c = 0; c < K; ++c)
            u_mat.at((u32)r, (u32)c) =
                V_colmajor[(size_t)r + (size_t)M_out * (size_t)c] * v_scale;

    // IVE always ends at level 1, so U can be encoded at that modulus and
    // scale up front. The first batch checks that against the real ciphertexts.
    auto pu = IPtMatrix::make();
    mat_encoder.encode(u_mat, *pu, levels.mods[enc_level_const],
                       levels.scales[enc_level_const]);

    HomEvalMatrix eval_mat{HomEvalParams(levels)};

    auto emails = pel::load_email_codes(pel::filterPath(pel::SPAM_CODES));
    auto ref_preds = pel::load_csv_pred_ids(pel::filterPath(pel::SPAM_CSV));

    const int num_samples = (int)emails.size();
    const int num_batches = (num_samples + stride - 1) / stride;
    std::cout << "Samples: " << num_samples << ", batches: " << num_batches
              << "\n\n";

    //==========================================================================
    // Batch loop
    //==========================================================================
    const auto to_sec = [](const auto b, const auto e) {
        return std::chrono::duration<double>(e - b).count();
    };
    double enc_sec = 0, basis_sec = 0, pcmm_sec = 0, rot_sec = 0;
    int correct = 0, total = 0;
    bool checked = false;

    for (int batch = 0; batch < num_batches; ++batch) {
        const int s_start = batch * stride;
        const int s_end = std::min(s_start + stride, num_samples);
        std::cout << "[Batch " << batch << "] samples " << s_start << " .. "
                  << (s_end - 1) << "\n";

        //----------------------------------------------------------------------
        // (1) Build the queries. This is client-side work, so it is not
        //     timed. Slot (i + stride*k) takes token k of example s_start+i
        //     at sub-table l, as alpha = exp(i*theta(code)); padding slots
        //     stay zero.
        //----------------------------------------------------------------------
        std::vector<Message> msg;
        msg.reserve(num_blocks);
        for (int b = 0; b < num_blocks; ++b)
            msg.emplace_back(log_slots);

        for (int l = 0; l < num_blocks; ++l) {
            for (int i = 0; i < stride; ++i) {
                const int eidx = s_start + i;
                for (int k = 0; k < max_tok; ++k) {
                    const size_t slot = (size_t)(i + stride * k);
                    const bool valid =
                        (eidx < s_end && k < emails[(size_t)eidx].num_tokens);
                    if (valid) {
                        const int c = emails[(size_t)eidx].codes[k][l];
                        const long double sign = (c & 1) ? -1.0L : 1.0L;
                        const long double theta = sign * pel::pi *
                                                  (2.0L * c + 1.0L) /
                                                  (2.0L * (long double)p);
                        msg[(size_t)l][slot].real((double)std::cos(theta));
                        msg[(size_t)l][slot].imag((double)std::sin(theta));
                    } else {
                        msg[(size_t)l][slot].real(0.0);
                        msg[(size_t)l][slot].imag(0.0);
                    }
                }
            }
        }

        // The Algorithm 1 L16 constant, on real token slots only. A padding
        // slot must hold 0 here, or the last basis element of each block would
        // feed a spurious constant into that example's total.
        Message msg_last(log_slots);
        const double cst_last = std::sqrt(2.0);
        for (int i = 0; i < stride; ++i) {
            const int eidx = s_start + i;
            const int nt =
                (eidx < s_end) ? emails[(size_t)eidx].num_tokens : 0;
            for (int k = 0; k < max_tok; ++k) {
                const size_t slot = (size_t)(i + stride * k);
                msg_last[slot].real((k < nt) ? cst_last : 0.0);
                msg_last[slot].imag(0.0);
            }
        }

        //----------------------------------------------------------------------
        // (2) Encrypt
        //----------------------------------------------------------------------
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<Ptr<ICiphertext>> ctxt((size_t)K);
        for (int i = 0; i < K; ++i)
            ctxt[(size_t)i] = ICiphertext::make();

        for (int b = 0; b < num_blocks; ++b) {
            const int base = b * p;
            compat::Encrypt(encoder, encryptor, *enc_key, msg[(size_t)b],
                         enc_level_seed, *ctxt[(size_t)base]);
            compat::Encrypt(encoder, encryptor, *enc_key, msg_last,
                         enc_level_const, *ctxt[(size_t)(base + p - 1)]);
        }
        const auto t1 = std::chrono::steady_clock::now();

        //----------------------------------------------------------------------
        // (3) Vector generation -- Algorithm 1, once per sub-table
        //----------------------------------------------------------------------
        for (int b = 0; b < num_blocks; ++b)
            pel::ive(eval, *relin_key, *conj_key, ctxt,
                                         b * p, p);
        const auto t2 = std::chrono::steady_clock::now();

        // pcmm needs all inputs at one level; also check that the level U was
        // encoded at is the level IVE actually produced.
        {
            const u32 lv0 = eval.getLevel(*ctxt[0]);
            for (int i = 1; i < K; ++i)
                if (eval.getLevel(*ctxt[(size_t)i]) != lv0) {
                    std::cerr << "[fatal] level mismatch at ctxt[" << i
                              << "]\n";
                    return 1;
                }
            if (!checked) {
                if (lv0 != enc_level_const) {
                    std::cerr << "[fatal] basis level " << lv0 << " != "
                              << enc_level_const
                              << " (the level U was encoded at)\n";
                    return 1;
                }
                checked = true;
            }
        }

        //----------------------------------------------------------------------
        // (4) PCMM :  W[M_out x degree] = U[M_out x K] . V[K x degree]
        //----------------------------------------------------------------------
        const auto t3 = std::chrono::steady_clock::now();

        std::vector<const ICiphertext *> ct_ptrs;
        ct_ptrs.reserve((size_t)K);
        for (int i = 0; i < K; ++i)
            ct_ptrs.push_back(&*ctxt[(size_t)i]);

        auto batch_in = ICiphertext::make(EncType::BatchRLWE);
        eval_flex.batch(ct_ptrs, *batch_in);

        // pcmm rejects anything but coefficient encoding. It is a linear map
        // on coefficients, so lowering the flag and restoring it later is
        // safe: setDFT touches metadata only, never the polynomial data.
        eval_flex.setDFT(*batch_in, /*dft=*/false);

        auto cv = ICtMatrix::make();
        cv->moveFrom(*batch_in, (u32)K, degree);

        auto cw = ICtMatrix::make();
        eval_mat.pcmm(*pu, *cv, *cw);
        eval_mat.rescale(*cw, *cw); // pcmm leaves the scale squared

        auto batch_out = ICiphertext::make(EncType::BatchRLWE);
        cw->moveTo(*batch_out);
        eval_flex.setDFT(*batch_out, /*dft=*/true); // back to slot encoding

        std::vector<Ptr<ICiphertext>> ctxt_rst((size_t)M_out);
        std::vector<ICiphertext *> rst_ptrs;
        rst_ptrs.reserve((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            ctxt_rst[(size_t)m] = ICiphertext::make();
            rst_ptrs.push_back(&*ctxt_rst[(size_t)m]);
        }
        eval_flex.unbatch(*batch_out, rst_ptrs);

        const auto t4 = std::chrono::steady_clock::now();

        if (batch == 0)
            std::cout << "  pcmm done, result level = "
                      << eval.getLevel(*ctxt_rst[0]) << ", rows = " << M_out
                      << "\n";

        //----------------------------------------------------------------------
        // (5) Sum over the token axis, as a tree of log2(max_tok) rotations.
        //     Afterwards slot i holds the total score of example s_start+i.
        //----------------------------------------------------------------------
        std::vector<Ptr<ICiphertext>> ctxt_sum((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            ctxt_sum[(size_t)m] = ICiphertext::make();
            ctxt_rst[(size_t)m]->copyTo(*ctxt_sum[(size_t)m]);
            for (int k = 1; k < max_tok; k <<= 1) {
                const i32 step = (i32)(k * stride);
                auto tmp = ICiphertext::make();
                eval.rot(*ctxt_sum[(size_t)m], step, *tmp,
                         *rot_keys.at(step));
                auto acc = ICiphertext::make();
                eval.add(*ctxt_sum[(size_t)m], *tmp, *acc);
                ctxt_sum[(size_t)m] = std::move(acc);
            }
        }
        const auto t5 = std::chrono::steady_clock::now();

        if (batch == 0)
            std::cout << "  rotate-sum done, level = "
                      << eval.getLevel(*ctxt_sum[0])
                      << " (rotation consumes no level)\n";

        enc_sec += to_sec(t0, t1);
        basis_sec += to_sec(t1, t2);
        pcmm_sec += to_sec(t3, t4);
        rot_sec += to_sec(t4, t5);

        //----------------------------------------------------------------------
        // (6) Decrypt and compare with the plaintext model's predictions
        //----------------------------------------------------------------------
        std::vector<Message> dmsg((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            auto dptxt = IPlaintext::make();
            encryptor.decrypt(*ctxt_sum[(size_t)m], *sk, *dptxt);
            encoder.decode(*dptxt, dmsg[(size_t)m]);
            dmsg[(size_t)m].to(Device::CPU);
        }

        for (int i = 0; i < stride; ++i) {
            const int eidx = s_start + i;
            if (eidx >= num_samples)
                break;
            const double s0 = dmsg[0][(size_t)i].real();
            const double s1 = dmsg[1][(size_t)i].real();
            const int pred = (s1 > s0) ? 1 : 0;
            if (pred == ref_preds[(size_t)eidx])
                ++correct;
            ++total;
        }

        std::cout << "  Running: " << correct << "/" << total << " ("
                  << std::fixed << std::setprecision(2)
                  << (100.0 * correct / total) << "%)\n\n";

        break; // one batch is enough to report per-example timings
    }

    //==========================================================================
    // Results
    //==========================================================================
    std::cout << "=== Accuracy vs plaintext pred_id: " << correct << "/"
              << total << " (" << std::fixed << std::setprecision(2)
              << (100.0 * correct / total) << "%) ===\n";

    std::cout << "\n=== Latency (batch 0, " << stride << " examples) ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Encrypt          (s): " << enc_sec << "\n";
    std::cout << "Basis Generation (s): " << basis_sec << "\n";
    std::cout << "PCMM             (s): " << pcmm_sec << "\n";
    std::cout << "Rotate-sum       (s): " << rot_sec << "\n";

    return 0;
} catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
}

int main() { return runExample(); }
