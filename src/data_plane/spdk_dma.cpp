#include "spdk_dma.hpp"

#ifdef GENIE_WITH_SPDK
#include <spdk/env.h>
#endif

namespace genie { namespace data_plane {

static bool g_spdk_available = false;

bool SpdkDMA::initialize() {
#ifdef GENIE_WITH_SPDK
    if (g_spdk_available) return true;
    spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.shm_id = 0;
    opts.name = "genie-spdk";
    if (spdk_env_init(&opts) == 0) {
        g_spdk_available = true;
    }
#endif
    return g_spdk_available;
}

bool SpdkDMA::available() {
    return g_spdk_available;
}

void* SpdkDMA::alloc(size_t size, size_t align) {
#ifdef GENIE_WITH_SPDK
    if (!g_spdk_available) return nullptr;
    return spdk_dma_zmalloc(size, align, nullptr);
#else
    (void)size; (void)align; return nullptr;
#endif
}

void SpdkDMA::free(void* ptr) {
#ifdef GENIE_WITH_SPDK
    if (ptr) spdk_dma_free(ptr);
#else
    (void)ptr;
#endif
}

} } // namespace


