#include "genie_kcp_wrapper.hpp"
#ifdef GENIE_WITH_KCP
#include <ikcp.h>
#endif

namespace genie {
namespace data_plane {

KCPWrapper::KCPWrapper() {}
KCPWrapper::~KCPWrapper() { close(); }

bool KCPWrapper::initialize(uint32_t conv, OutputFn output_fn) {
    output_ = std::move(output_fn);
#ifdef GENIE_WITH_KCP
    kcp_ = ikcp_create(conv, this);
    if (!kcp_) return false;
    ikcp_nodelay(kcp_, 1, 10, 2, 1);
    ikcp_wndsize(kcp_, 128, 128);
    ikcp_setoutput(kcp_, [](const char* buf, int len, ikcpcb* /*k*/, void* user) -> int {
        auto* self = static_cast<KCPWrapper*>(user);
        return self->output_ ? self->output_(buf, len) : -1;
    });
    enabled_ = true;
#else
    enabled_ = true;
#endif
    return true;
}

int KCPWrapper::input(const char* data, int len) {
#ifdef GENIE_WITH_KCP
    if (kcp_) {
        return ikcp_input(kcp_, data, len);
    }
#endif
    return len;
}

void KCPWrapper::update(uint32_t /*current_ms*/) {
#ifdef GENIE_WITH_KCP
    if (kcp_) {
        // In a full implementation, pass a monotonic ms timestamp
        ikcp_update(kcp_, 0);
    }
#endif
}

void KCPWrapper::close() {
#ifdef GENIE_WITH_KCP
    if (kcp_) {
        ikcp_release(kcp_);
        kcp_ = nullptr;
    }
#endif
    enabled_ = false;
}

int KCPWrapper::recv(char* buffer, int maxlen) {
#ifdef GENIE_WITH_KCP
    if (kcp_) {
        return ikcp_recv(kcp_, buffer, maxlen);
    }
#endif
    return -1;
}

} // namespace data_plane
} // namespace genie


