#include "ivf_pq_index.h"
#include "kmeans.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace agentrag {
namespace core {
namespace {

float squared_l2(const float* a, const float* b, size_t dim) {
    float distance = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

template <typename T>
void write_scalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed to write IVF-PQ index");
}

template <typename T>
T read_scalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated IVF-PQ index");
    return value;
}

size_t read_size(std::istream& in) {
    const uint64_t value = read_scalar<uint64_t>(in);
    if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("IVF-PQ index size exceeds platform limit");
    }
    return static_cast<size_t>(value);
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    write_scalar<uint64_t>(out, static_cast<uint64_t>(values.size()));
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!out) throw std::runtime_error("failed to write IVF-PQ vector data");
    }
}

template <typename T>
std::vector<T> read_vector(std::istream& in) {
    const size_t size = read_size(in);
    if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("invalid IVF-PQ vector size");
    }
    std::vector<T> values(size);
    if (!values.empty()) {
        in.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!in) throw std::runtime_error("truncated IVF-PQ vector data");
    }
    return values;
}

}  // namespace

void IVFPQIndex::build(
    const float* vectors,
    size_t n,
    size_t dim,
    size_t n_clusters,
    size_t n_probe,
    size_t n_subvectors,
    size_t n_bits,
    int32_t n_iters)
{
    if (vectors == nullptr || n == 0 || dim == 0) {
        throw std::invalid_argument("vectors, n and dim must be non-empty");
    }
    if (n_clusters == 0 || n_probe == 0) {
        throw std::invalid_argument("n_clusters and n_probe must be greater than zero");
    }
    if (n_subvectors == 0 || dim % n_subvectors != 0) {
        throw std::invalid_argument("dim must be divisible by a positive n_subvectors");
    }
    if (n_bits == 0 || n_bits > 8) {
        throw std::invalid_argument("n_bits must be in [1, 8]");
    }
    if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("n exceeds int32 vector id capacity");
    }

    n_vectors_ = n;
    dim_ = dim;
    n_clusters_ = std::min(n_clusters, n);
    n_probe_ = std::min(n_probe, n_clusters_);

    KMeansResult coarse = kmeans(vectors, n, dim, n_clusters_, n_iters);
    centroids_ = coarse.centroids;

    std::vector<float> residuals(n * dim);
    for (size_t i = 0; i < n; ++i) {
        const size_t cluster = static_cast<size_t>(coarse.assignments[i]);
        const float* vector = vectors + i * dim;
        const float* centroid = centroids_.data() + cluster * dim;
        float* residual = residuals.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) {
            residual[d] = vector[d] - centroid[d];
        }
    }

    pq_.train(residuals.data(), n, dim, n_subvectors, n_bits, n_iters);

    lists_.clear();
    lists_.resize(n_clusters_);
    for (auto& list : lists_) {
        const size_t expected = n / n_clusters_ + 1;
        list.ids.reserve(expected);
        list.codes.reserve(expected * n_subvectors);
    }

    for (size_t i = 0; i < n; ++i) {
        const size_t cluster = static_cast<size_t>(coarse.assignments[i]);
        auto& list = lists_[cluster];
        const auto codes = pq_.encode(residuals.data() + i * dim);
        list.ids.push_back(static_cast<int32_t>(i));
        list.codes.insert(list.codes.end(), codes.begin(), codes.end());
    }
}

std::vector<SearchResult> IVFPQIndex::search(
    const float* query,
    size_t top_k) const
{
    if (query == nullptr) {
        throw std::invalid_argument("query must not be null");
    }
    if (lists_.empty() || top_k == 0) return {};

    std::vector<std::pair<float, size_t>> cluster_distances;
    cluster_distances.reserve(n_clusters_);
    for (size_t cluster = 0; cluster < n_clusters_; ++cluster) {
        cluster_distances.emplace_back(
            squared_l2(query, centroids_.data() + cluster * dim_, dim_),
            cluster);
    }

    const size_t probes = std::min(n_probe_, n_clusters_);
    std::partial_sort(
        cluster_distances.begin(),
        cluster_distances.begin() + probes,
        cluster_distances.end());

    std::vector<SearchResult> candidates;
    std::vector<float> query_residual(dim_);
    for (size_t probe = 0; probe < probes; ++probe) {
        const size_t cluster = cluster_distances[probe].second;
        const auto& list = lists_[cluster];
        if (list.ids.empty()) continue;

        const float* centroid = centroids_.data() + cluster * dim_;
        for (size_t d = 0; d < dim_; ++d) {
            query_residual[d] = query[d] - centroid[d];
        }

        const auto distances = pq_.compute_distances_flat(
            query_residual.data(), list.codes, list.ids.size());
        for (size_t i = 0; i < list.ids.size(); ++i) {
            candidates.push_back({list.ids[i], distances[i]});
        }
    }

    const size_t result_count = std::min(top_k, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + result_count,
        candidates.end(),
        [](const SearchResult& left, const SearchResult& right) {
            if (left.score != right.score) return left.score < right.score;
            return left.id < right.id;
        });
    candidates.resize(result_count);
    return candidates;
}

size_t IVFPQIndex::estimated_memory_bytes() const {
    size_t bytes = centroids_.size() * sizeof(float);
    bytes += pq_.codebooks_.size() * sizeof(float);
    for (const auto& list : lists_) {
        bytes += list.ids.size() * sizeof(int32_t);
        bytes += list.codes.size() * sizeof(uint8_t);
    }
    return bytes;
}

void IVFPQIndex::save(const std::string& path) const {
    if (lists_.empty()) {
        throw std::runtime_error("cannot save an empty IVF-PQ index");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open IVF-PQ index for writing");

    constexpr std::array<char, 8> magic{{'A', 'R', 'I', 'V', 'F', 'P', 'Q', '1'}};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));

    write_scalar<uint64_t>(out, static_cast<uint64_t>(n_vectors_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(dim_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(n_clusters_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(n_probe_));
    write_vector(out, centroids_);

    write_scalar<uint64_t>(out, static_cast<uint64_t>(pq_.dim_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(pq_.n_subvectors_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(pq_.n_bits_));
    write_scalar<uint64_t>(out, static_cast<uint64_t>(pq_.trained_codebook_size_));
    write_vector(out, pq_.codebooks_);

    write_scalar<uint64_t>(out, static_cast<uint64_t>(lists_.size()));
    for (const auto& list : lists_) {
        write_vector(out, list.ids);
        write_vector(out, list.codes);
    }
}

void IVFPQIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open IVF-PQ index for reading");

    constexpr std::array<char, 8> expected{{'A', 'R', 'I', 'V', 'F', 'P', 'Q', '1'}};
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != expected) {
        throw std::runtime_error("invalid IVF-PQ index header");
    }

    IVFPQIndex loaded;
    loaded.n_vectors_ = read_size(in);
    loaded.dim_ = read_size(in);
    loaded.n_clusters_ = read_size(in);
    loaded.n_probe_ = read_size(in);
    loaded.centroids_ = read_vector<float>(in);

    loaded.pq_.dim_ = read_size(in);
    loaded.pq_.n_subvectors_ = read_size(in);
    loaded.pq_.n_bits_ = read_size(in);
    loaded.pq_.trained_codebook_size_ = read_size(in);
    loaded.pq_.codebooks_ = read_vector<float>(in);

    const size_t list_count = read_size(in);
    loaded.lists_.resize(list_count);
    size_t id_count = 0;
    for (auto& list : loaded.lists_) {
        list.ids = read_vector<int32_t>(in);
        list.codes = read_vector<uint8_t>(in);
        id_count += list.ids.size();
    }

    if (loaded.pq_.n_subvectors_ == 0 ||
        loaded.pq_.n_bits_ == 0 || loaded.pq_.n_bits_ > 8 ||
        loaded.dim_ == 0 || loaded.dim_ % loaded.pq_.n_subvectors_ != 0) {
        throw std::runtime_error("invalid IVF-PQ quantizer metadata");
    }

    const size_t full_codebook_size = size_t{1} << loaded.pq_.n_bits_;
    const size_t expected_codebook_values =
        loaded.pq_.n_subvectors_ * full_codebook_size *
        (loaded.dim_ / loaded.pq_.n_subvectors_);

    if (loaded.n_vectors_ == 0 || loaded.dim_ == 0 ||
        loaded.n_clusters_ == 0 || loaded.n_probe_ == 0 ||
        loaded.n_probe_ > loaded.n_clusters_ ||
        loaded.lists_.size() != loaded.n_clusters_ ||
        loaded.centroids_.size() != loaded.n_clusters_ * loaded.dim_ ||
        loaded.pq_.dim_ != loaded.dim_ ||
        loaded.pq_.trained_codebook_size_ == 0 ||
        loaded.pq_.trained_codebook_size_ > full_codebook_size ||
        loaded.pq_.codebooks_.size() != expected_codebook_values ||
        id_count != loaded.n_vectors_) {
        throw std::runtime_error("inconsistent IVF-PQ index metadata");
    }
    for (const auto& list : loaded.lists_) {
        if (list.codes.size() != list.ids.size() * loaded.pq_.n_subvectors_) {
            throw std::runtime_error("inconsistent IVF-PQ inverted list");
        }
    }

    *this = std::move(loaded);
}

}  // namespace core
}  // namespace agentrag
