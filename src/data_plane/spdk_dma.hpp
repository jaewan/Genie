#pragma once

#include <cstddef>

namespace genie { namespace data_plane {

class SpdkDMA {
public:
    static bool initialize();
    static void* alloc(size_t size, size_t align = 0x1000);
    static void free(void* ptr);
    static bool available();
};

} } // namespace


