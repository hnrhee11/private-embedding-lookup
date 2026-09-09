#pragma once
//==============================================================================
// Helpers shared by the private embedding lookup examples.
//
//   1. HEaaN2 adapters     compat::Encrypt / Mult / Add
//   2. Generic utilities   pi, safe_neg_log2, percentile, binary file readers
//   3. Datasets            names, dimensions and file paths
//   4. Query generation    the client-side input of each method
//   5. Precision           RMSE of the decrypted result against a reference
//
// Sections 3 to 5 support the Table 3 experiments of
//
//   "Private Embedding Lookup with Encrypted Compact Queries under Fully
//    Homomorphic Encryption"
//
// Sections 1 and 2 have nothing to do with embedding lookup; they only fill
// gaps in the HEaaN2 API and in the standard library. Everything from
// section 3 on is specific to these experiments.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//==============================================================================
// 1. HEaaN2 adapters
//
// Nothing here is specific to embedding lookup. Each function restores
// something the HEaaN2 API does not offer directly, and they are capitalised
// so they stand out from the all-lowercase HEaaN2 API.
//==============================================================================
namespace compat {

/// @brief Encode a message and encrypt it in one call.
/// @details EnDecryptor::encrypt takes a plaintext and reads the level off it,
/// so a message always has to be encoded first.
inline void Encrypt(const heaan::EnDecoder &encoder,
                    const heaan::EnDecryptor &encryptor,
                    const heaan::IEncKey &enc_key, const heaan::Message &msg,
                    heaan::u32 level, heaan::ICiphertext &res) {
    auto ptxt = heaan::IPlaintext::make();
    encoder.encode(msg, *ptxt, level);
    encryptor.encrypt(*ptxt, enc_key, res);
}

/// @details HomEval::mulRescale and HomEval::add throw if their operands sit
/// at different moduli, so the higher operand is leveled down first.
/// @brief Multiply and rescale, leveling down the higher operand if needed.
inline void Mult(const heaan::HomEval &eval, const heaan::ISwKey &relin_key,
                 const heaan::ICiphertext &a, const heaan::ICiphertext &b,
                 heaan::ICiphertext &res) {
    const heaan::u32 lv_a = eval.getLevel(a);
    const heaan::u32 lv_b = eval.getLevel(b);
    if (lv_a == lv_b) {
        eval.mulRescale(a, b, res, relin_key);
        return;
    }
    const heaan::u32 lv_min = std::min(lv_a, lv_b);
    auto aligned = heaan::ICiphertext::make();
    if (lv_a > lv_min) {
        eval.levelDownTo(a, *aligned, lv_min);
        eval.mulRescale(*aligned, b, res, relin_key);
    } else {
        eval.levelDownTo(b, *aligned, lv_min);
        eval.mulRescale(a, *aligned, res, relin_key);
    }
}

/// @brief Add, leveling down the higher operand if needed.
inline void Add(const heaan::HomEval &eval, const heaan::ICiphertext &a,
                const heaan::ICiphertext &b, heaan::ICiphertext &res) {
    const heaan::u32 lv_a = eval.getLevel(a);
    const heaan::u32 lv_b = eval.getLevel(b);
    if (lv_a == lv_b) {
        eval.add(a, b, res);
        return;
    }
    const heaan::u32 lv_min = std::min(lv_a, lv_b);
    auto aligned = heaan::ICiphertext::make();
    if (lv_a > lv_min) {
        eval.levelDownTo(a, *aligned, lv_min);
        eval.add(*aligned, b, res);
    } else {
        eval.levelDownTo(b, *aligned, lv_min);
        eval.add(a, *aligned, res);
    }
}

} // namespace compat

namespace pel {

using u64 = std::uint64_t;

//==============================================================================
// 2. Generic utilities
//==============================================================================

constexpr long double pi = 3.141592653589793238462643383279502884L;

/// @brief -log2(x), reported as +inf when x is zero or negative.
inline double safe_neg_log2(double x) {
    if (!(x > 0.0))
        return std::numeric_limits<double>::infinity();
    return -std::log2(x);
}

/// @brief Linear-interpolated percentile; p01 is a fraction in [0, 1].
/// @details Uses nth_element, so it costs O(size) and reorders the copy it
/// takes by value.
inline double percentile(std::vector<double> v, double p01) {
    if (v.empty())
        return 0.0;
    const double pos = p01 * (double)(v.size() - 1);
    const size_t k = (size_t)std::floor(pos);
    const double a = pos - (double)k;
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    const double vk = v[k];
    if (a == 0.0 || k + 1 >= v.size())
        return vk;
    std::nth_element(v.begin(), v.begin() + (long)(k + 1), v.end());
    const double vk1 = v[k + 1];
    return (1.0 - a) * vk + a * vk1;
}

/// @brief Reads a (rows x cols) column-major float64 file whose header is two
/// uint32 fields, rows then cols.
inline std::vector<double>
load_colmajor_f64_bin(const std::string &path,
                      std::uint32_t expected_rows = 300,
                      std::uint32_t expected_cols = 4096) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        throw std::runtime_error("Failed to open: " + path);

    std::uint32_t rows = 0, cols = 0;
    fin.read(reinterpret_cast<char *>(&rows), sizeof(std::uint32_t));
    fin.read(reinterpret_cast<char *>(&cols), sizeof(std::uint32_t));
    if (!fin)
        throw std::runtime_error("Failed to read header.");

    if (rows != expected_rows || cols != expected_cols) {
        throw std::runtime_error("Unexpected shape: (" + std::to_string(rows) +
                                 "," + std::to_string(cols) + ")");
    }

    std::vector<double> V((size_t)rows * (size_t)cols);
    fin.read(reinterpret_cast<char *>(V.data()),
             (std::streamsize)(V.size() * sizeof(double)));
    if (!fin)
        throw std::runtime_error("Failed to read matrix payload.");

    return V; // col-major
}

/// @brief Reads one row at a time from a (rows x cols) row-major float64 file
/// with the same two-uint32 header.
/// @details The reference results run to hundreds of megabytes, so rows are
/// seeked to and read on demand instead of loading the whole file.
struct BinRowMajorF64 {
    std::FILE *f = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;

    explicit BinRowMajorF64(const std::string &path) {
        f = std::fopen(path.c_str(), "rb");
        if (!f)
            throw std::runtime_error("Failed to open bin: " + path);
        if (std::fread(&rows, sizeof(std::uint32_t), 1, f) != 1)
            throw std::runtime_error("Failed to read rows");
        if (std::fread(&cols, sizeof(std::uint32_t), 1, f) != 1)
            throw std::runtime_error("Failed to read cols");
    }
    ~BinRowMajorF64() {
        if (f)
            std::fclose(f);
    }
    BinRowMajorF64(const BinRowMajorF64 &) = delete;
    BinRowMajorF64 &operator=(const BinRowMajorF64 &) = delete;

    void read_row(std::uint32_t row_index, std::vector<double> &out_row) {
        if (row_index >= rows)
            throw std::runtime_error("row_index out of range");
        const u64 header_bytes = 8;
        const u64 row_bytes = (u64)cols * sizeof(double);
        const u64 offset = header_bytes + (u64)row_index * row_bytes;

        if (std::fseek(f, (long)offset, SEEK_SET) != 0)
            throw std::runtime_error("fseek failed");

        out_row.assign(cols, 0.0);
        const size_t got = std::fread(out_row.data(), sizeof(double), cols, f);
        if (got != cols)
            throw std::runtime_error("Failed to read full row");
    }
};

//==============================================================================
// 3. Datasets
//
// The three embedding tables differ only in their directory, their file-name
// prefix and their dimension d, so they are listed here and selected by name
// on the command line.
//==============================================================================

/// @brief Where one dataset lives and how wide its embeddings are.
struct DatasetInfo {
    const char *dir;    ///< folder under the dataset root
    const char *prefix; ///< leading part of the file names
    int M;              ///< d, the embedding dimension
};

/// @brief Looks a dataset up by the name used on the command line.
inline const DatasetInfo *find_dataset(const std::string &name) {
    static const struct {
        const char *name;
        DatasetInfo info;
    } TABLE[] = {
        {"gpt2", {"gpt2", "gpt2_768d_M4", 768}},
        {"glove300d", {"300d", "42B_300d_M4", 300}},
        {"glove50d", {"50d", "6B_50d_M4", 50}},
    };
    for (const auto &e : TABLE)
        if (name == e.name)
            return &e.info;
    return nullptr;
}

inline std::string dataset_names() { return "gpt2, glove300d, glove50d"; }

// Directory holding the per-dataset folders (50d, 300d, gpt2). CMake passes it
// in as PEL_TABLE_DIR at compile time (see CMakeLists.txt), so no absolute
// path is baked into the source and the tree can be moved.
#ifndef PEL_TABLE_DIR
#error "PEL_TABLE_DIR is not defined; see CMakeLists.txt"
#endif

/// @brief Root directory holding the three dataset folders.
inline const char *dataset_root() { return PEL_TABLE_DIR "/"; }

/// @brief The table M_L that IVE multiplies by: the embedding table composed
/// with the DCT matrix, so that M_L . v_j recovers row j. Callers scale it by
/// 1/sqrt(2p), since Algorithm 1 outputs sqrt(2p) * v_j.
inline std::string table_path(const DatasetInfo &dataset, int p) {
    return std::string(dataset_root()) + dataset.dir + "/" + dataset.prefix +
           "_K" + std::to_string(p) + "_transformed_colmaj_f64.bin";
}

/// @brief The raw embedding table M that the baseline multiplies by, since a
/// one-hot vector already selects a column.
inline std::string codebook_path(const DatasetInfo &dataset, int p) {
    return std::string(dataset_root()) + dataset.dir + "/" + dataset.prefix +
           "_K" + std::to_string(p) + "_col_maj.bin";
}

/// @brief The lookup result computed in plaintext, used as ground truth.
/// @details Column s is the sum of the ell sub-table rows selected by the
/// query in slot s -- the embedding vector the protocol should return.
inline std::string reference_path(const DatasetInfo &dataset, int p) {
    return std::string(dataset_root()) + dataset.dir + "/result_" +
           std::to_string(dataset.M) + "x65536_K" + std::to_string(p) +
           "_row_maj.bin";
}

//==============================================================================
// 4. Query generation
//
// One query value per slot, in the form each method expects.
//==============================================================================

/// @brief Fills every slot with alpha_j = exp(i*theta_j) for a random index j.
/// @details theta_j = (-1)^j * pi * (2j+1) / (2p), the angle assigned to index
/// j in Section 4.1. The alternating sign keeps the values from crowding into
/// one half-plane. These alpha_j are the input ct of Algorithm 1, and in the
/// protocol they are what the client encrypts and sends (Section 5.1).
/// vec_int returns the drawn indices, so a caller can reproduce the lookup in
/// plaintext.
inline void fillRandomIndex(heaan::Message &msg, std::vector<int> &vec_int,
                            int p, std::mt19937_64 &rng) {
    const size_t num_slots = size_t{1} << msg.logSlots();
    vec_int.resize(num_slots);
    std::uniform_int_distribution<int> dist(0, p - 1);

    for (size_t i = 0; i < num_slots; ++i) {
        int j = dist(rng); // the index j
        vec_int[i] = j;

        long double sign = (j & 1) ? -1.0L : 1.0L; // (-1)^j
        long double theta =
            sign * pi * (2.0L * j + 1.0L) / (2.0L * (long double)p);

        msg[i].real((double)std::cos(theta));
        msg[i].imag((double)std::sin(theta));
    }
}

/// @brief Fills every slot with a random index j itself, as a real number.
/// @details The baseline of Kim, Park, Lee and Cheon encrypts the index rather
/// than an angle, and derives a one-hot vector from it with an indicator
/// polynomial.
inline void fillRandomIndex_ICML24(heaan::Message &msg, int p,
                                 std::mt19937_64 &rng) {
    const size_t num_slots = size_t{1} << msg.logSlots();
    std::uniform_int_distribution<int> dist(0, p - 1);
    for (size_t i = 0; i < num_slots; ++i) {
        int j = dist(rng);
        msg[i].real((double)j);
        msg[i].imag(0.0);
    }
}

//==============================================================================
// 5. Precision
//==============================================================================

/// @brief Decrypts the result and reports, in bits, how far each column is
/// from the plaintext lookup stored in ref_bin_path.
/// @details Each of the d ciphertexts holds one row of the output, so a column
/// of the decrypted matrix is one embedding vector. The RMSE is accumulated
/// per column over all d rows and reported as -log2(RMSE), both absolute and
/// relative to the norm of the reference column. This is the quantity
/// Theorem 5.1 bounds: the theorem guarantees an RMS error of at most 1/u per
/// recovered vector, so the absolute figure printed here should stay above
/// log2(u).
inline void measure_column_vectors_rmse_bits(
    const std::vector<heaan::Ptr<heaan::ICiphertext>> &ctxt_rst,
    const heaan::EnDecryptor &encryptor, const heaan::EnDecoder &encoder,
    const heaan::ISecretKey &sk, const std::string &ref_bin_path,
    double tau_rel = 1e-12) {
    BinRowMajorF64 bin(ref_bin_path);

    const size_t R = std::min<size_t>(ctxt_rst.size(), bin.rows);
    const size_t C = (size_t)bin.cols;

    std::vector<long double> sse(C, 0.0L);
    std::vector<long double> ref_sse(C, 0.0L);

    std::vector<double> ref_row;
    std::vector<double> bits_abs(C, 0.0);
    std::vector<double> bits_rel(C, 0.0);

    for (size_t i = 0; i < R; ++i) {
        heaan::Message dmsg;
        {
            auto dptxt = heaan::IPlaintext::make();
            encryptor.decrypt(*ctxt_rst[i], sk, *dptxt);
            encoder.decode(*dptxt, dmsg);
            dmsg.to(heaan::Device::CPU);
        }
        bin.read_row((std::uint32_t)i, ref_row);

        const size_t num_slots = size_t{1} << dmsg.logSlots();
        const size_t Nmin = std::min<size_t>(num_slots, C);
        for (size_t j = 0; j < Nmin; ++j) {
            const double x = dmsg[j].real();
            const double y = ref_row[j];
            const double e = x - y;
            sse[j] += (long double)e * (long double)e;
            ref_sse[j] += (long double)y * (long double)y;
        }
    }

    double min_abs_bits = std::numeric_limits<double>::infinity();
    double min_rel_bits = std::numeric_limits<double>::infinity();
    long double sum_abs_bits = 0.0L, sum_rel_bits = 0.0L;

    for (size_t j = 0; j < C; ++j) {
        const long double mse = sse[j] / (long double)R;
        const double rmse = std::sqrt((double)mse);

        const double b_abs = safe_neg_log2(rmse);
        bits_abs[j] = b_abs;

        const long double ref_mse = ref_sse[j] / (long double)R;
        const double ref_rms = std::sqrt((double)ref_mse);
        const double rrmse = rmse / (ref_rms + tau_rel);

        const double b_rel = safe_neg_log2(rrmse);
        bits_rel[j] = b_rel;

        min_abs_bits = std::min(min_abs_bits, b_abs);
        min_rel_bits = std::min(min_rel_bits, b_rel);
        sum_abs_bits += b_abs;
        sum_rel_bits += b_rel;
    }

    const double mean_abs_bits = (double)(sum_abs_bits / (long double)C);
    const double mean_rel_bits = (double)(sum_rel_bits / (long double)C);

    const double p01_abs = percentile(bits_abs, 0.01);
    const double p05_abs = percentile(bits_abs, 0.05);
    const double p50_abs = percentile(bits_abs, 0.50);

    const double p01_rel = percentile(bits_rel, 0.01);
    const double p05_rel = percentile(bits_rel, 0.05);
    const double p50_rel = percentile(bits_rel, 0.50);

    std::cout << "========== Column-vector (length " << R
              << ") RMSE precision over " << C << " columns ==========\n";
    std::cout << "[ABS RMSE bits = -log2(RMSE)]\n";
    std::cout << "  min=" << min_abs_bits << ", mean=" << mean_abs_bits
              << ", median=" << p50_abs << ", p05=" << p05_abs
              << ", p01=" << p01_abs << "\n";
    std::cout << "[REL RMSE bits = -log2(RRMSE)], RRMSE = RMSE / (ref_rms + "
                 "tau)\n";
    std::cout << "  tau=" << tau_rel << "\n";
    std::cout << "  min=" << min_rel_bits << ", mean=" << mean_rel_bits
              << ", median=" << p50_rel << ", p05=" << p05_rel
              << ", p01=" << p01_rel << "\n";
    std::cout
        << "===========================================================\n";
}

} // namespace pel
