#include "crc8.h"

uint8_t crc8_check(uint8_t *ptr, uint8_t len)
{
    uint8_t i;
    uint8_t crc = 0xFF;

    while(len--)
    {
        crc ^= *ptr++;
        for (i = 8; i > 0; --i)
        {
            if (crc & CRC8_INIT)
                crc = (crc << 1) ^ CRC8_POLYNOMIAL;
            else
                crc = (crc << 1);
        }
    }

    return (crc);
}
