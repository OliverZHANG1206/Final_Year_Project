#ifndef COMPONENTS_SCD30_INCLUDE_CRC8_H_
#define COMPONENTS_SCD30_INCLUDE_CRC8_H_

#include <stdio.h>
#include <stdint.h>

#define CRC8_INIT       0x80
#define CRC8_POLYNOMIAL 0x31

uint8_t crc8_check(uint8_t *ptr, uint8_t len);

#endif /* COMPONENTS_SCD30_INCLUDE_CRC8_H_ */
