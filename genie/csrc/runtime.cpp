#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace genie {

struct DpdkApi {
    void* lib_eal{nullptr};
    void* lib_mbuf{nullptr};
    void* lib_mempool{nullptr};

    using rte_eal_init_t = int (*)(int, char**);
    using rte_pktmbuf_pool_create_t = void* (*)(const char*, unsigned, unsigned, uint16_t, uint16_t, int);
    using rte_pktmbuf_alloc_t = void* (*)(void*);

    rte_eal_init_t rte_eal_init{nullptr};
    rte_pktmbuf_pool_create_t rte_pktmbuf_pool_create{nullptr};
    rte_pktmbuf_alloc_t rte_pktmbuf_alloc{nullptr};

    static void* try_dlopen(const std::vector<std::string>& names) {
        for (const auto& n : names) {
            void* h = dlopen(n.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (h) return h;
        }
        return nullptr;
    }

    static void* must_dlsym(void* handle, const char* sym) {
        if (!handle) return nullptr;
        void* f = dlsym(handle, sym);
        return f;
    }

    bool load() {
        lib_eal = try_dlopen({
            "librte_eal.so", "librte_eal.so.20", "librte_eal.so.23", "librte_eal.so.23.0",
            "/usr/lib/x86_64-linux-gnu/librte_eal.so"
        });
        lib_mbuf = try_dlopen({
            "librte_mbuf.so", "librte_mbuf.so.20", "librte_mbuf.so.23", "librte_mbuf.so.23.0",
            "/usr/lib/x86_64-linux-gnu/librte_mbuf.so"
        });
        lib_mempool = try_dlopen({
            "librte_mempool.so", "librte_mempool.so.20", "librte_mempool.so.23", "librte_mempool.so.23.0",
            "/usr/lib/x86_64-linux-gnu/librte_mempool.so"
        });
        // Fallback to monolithic libdpdk if split libs not found
        if (!(lib_eal && lib_mbuf && lib_mempool)) {
            void* libdpdk = try_dlopen({"libdpdk.so", "libdpdk.so.23", "libdpdk.so.24"});
            if (libdpdk) {
                lib_eal = lib_eal ? lib_eal : libdpdk;
                lib_mbuf = lib_mbuf ? lib_mbuf : libdpdk;
                lib_mempool = lib_mempool ? lib_mempool : libdpdk;
            }
        }
        if (!(lib_eal && lib_mbuf && lib_mempool)) return false;
        rte_eal_init = reinterpret_cast<rte_eal_init_t>(must_dlsym(lib_eal, "rte_eal_init"));
        rte_pktmbuf_pool_create = reinterpret_cast<rte_pktmbuf_pool_create_t>(must_dlsym(lib_mbuf, "rte_pktmbuf_pool_create"));
        rte_pktmbuf_alloc = reinterpret_cast<rte_pktmbuf_alloc_t>(must_dlsym(lib_mbuf, "rte_pktmbuf_alloc"));
        return rte_eal_init && rte_pktmbuf_pool_create && rte_pktmbuf_alloc;
    }

    bool ok() const {
        return lib_eal && lib_mbuf && lib_mempool && rte_eal_init && rte_pktmbuf_pool_create && rte_pktmbuf_alloc;
    }
};

struct GpuDevApi {
    void* lib_gpudev{nullptr};
    bool load() {
        lib_gpudev = DpdkApi::try_dlopen({"librte_gpudev.so", "librte_gpudev.so.23", "librte_gpudev.so.23.0"});
        return lib_gpudev != nullptr;
    }
    bool ok() const { return lib_gpudev != nullptr; }
};

struct Runtime {
    DpdkApi dpdk;
    GpuDevApi gpudev;
    bool eal_initialized{false};
    void* default_pool{nullptr};

    Runtime() {
        dpdk.load();
        gpudev.load();
    }

    bool dpdk_available() const { return dpdk.ok(); }
    bool gpudev_available() const { return gpudev.ok(); }

    bool eal_init(const std::vector<std::string>& args) {
        if (!dpdk.ok()) return false;
        if (eal_initialized) return true;
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>("genie-dpdk"));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        int ret = dpdk.rte_eal_init(static_cast<int>(argv.size()), argv.data());
        eal_initialized = ret >= 0;
        return eal_initialized;
    }

    std::uintptr_t create_default_pool(const std::string& name, unsigned n, unsigned data_room) {
        if (!eal_initialized) return 0;
        default_pool = dpdk.rte_pktmbuf_pool_create(name.c_str(), n, /*cache*/256, /*priv*/0, static_cast<uint16_t>(data_room), /*socket*/-1);
        return reinterpret_cast<std::uintptr_t>(default_pool);
    }

    std::uintptr_t alloc_mbuf(std::uintptr_t pool_ptr) {
        if (!eal_initialized) return 0;
        void* pool = pool_ptr ? reinterpret_cast<void*>(pool_ptr) : default_pool;
        if (!pool) return 0;
        void* mbuf = dpdk.rte_pktmbuf_alloc(pool);
        return reinterpret_cast<std::uintptr_t>(mbuf);
    }
};

static Runtime& runtime() {
    static Runtime rt;
    return rt;
}

} // namespace genie

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dpdk_available", []() { return genie::runtime().dpdk_available(); });
    m.def("gpudev_available", []() { return genie::runtime().gpudev_available(); });
    m.def("eal_init", [](const std::vector<std::string>& args) { return genie::runtime().eal_init(args); });
    m.def("create_default_pool", [](const std::string& name, unsigned n, unsigned data_room) {
        return genie::runtime().create_default_pool(name, n, data_room);
    });
    m.def("alloc_mbuf", [](std::uintptr_t pool_ptr) { return genie::runtime().alloc_mbuf(pool_ptr); });
}


