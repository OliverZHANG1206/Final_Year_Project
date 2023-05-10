#include "data_conv.h"

float bytes_to_float(const uint32_t value)
{
    union
	{
        uint32_t u32_value;
        float float32;
    }tmp;

    tmp.u32_value = value;
    return tmp.float32;
}
