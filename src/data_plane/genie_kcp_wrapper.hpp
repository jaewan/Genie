#pragma once

#include <cstdint>
#include <functional>

namespace genie {
namespace data_plane {

class KCPWrapper {
public:
    using OutputFn = std::function<int(const char*, int)>;

    KCPWrapper();
    ~KCPWrapper();

    bool initialize(uint32_t conv, OutputFn output_fn);
    int input(const char* data, int len);
    void update(uint32_t current_ms);
    void close();
    int recv(char* buffer, int maxlen);

private:
    OutputFn output_;
    bool enabled_ = false;
#ifdef GENIE_WITH_KCP
    struct IKCPCB* kcp_ = nullptr;
#endif
};

} // namespace data_plane
} // namespace genie


