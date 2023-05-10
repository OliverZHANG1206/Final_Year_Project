#ifndef SCD30_H
#define SCD30_H

#include <stdio.h>
#include <stdint.h>
#include "crc8.h"
#include "esp_log.h"
#include "esp_check.h"
#include "data_conv.h"
#include "driver/i2c.h"
#include "sys/unistd.h"

#define I2C_PORT           0
#define I2C_MASTER_SDA_IO  6
#define I2C_MASTER_SCL_IO  5
#define I2C_MASTER_FREQ_HZ 50000
#define I2C_TX_BUF         0
#define I2C_RX_BUF         0
#define I2C_TIMEOUT_MS     1000

#define I2C_WRITE          0x0
#define I2C_READ           0x1
#define ACK                0x0
#define NACK               0x1
#define ACK_CHECK_EN       0x1              /*!< I2C master will check ack from slave*/
#define ACK_CHECK_DIS      0x0              /*!< I2C master will not check ack from slave */
#define SCD30_ACTIVATE     0x1
#define SCD30_DEACTIVATE   0x0

#define SCD30_ADDR 	                         0x61
#define SCD30_VERSION                        0x0342
#define SCD30_CMD_START_PERIODIC_MEASUREMENT 0x0010
#define SCD30_CMD_STOP_PERIODIC_MEASUREMENT  0x0104
#define SCD30_CMD_READ_MEASUREMENT           0x0300
#define SCD30_CMD_SET_MEASUREMENT_INTERVAL   0x4600
#define SCD30_CMD_SET_TEMPERATURE_OFFSET     0x5403
#define SCD30_CMD_SET_ALTITUDE               0x5102
#define SCD30_CMD_SET_FORCED_RECALIBRATION   0x5204
#define SCD30_CMD_GET_DATA_READY             0x0202
#define SCD30_CMD_AUTO_SELF_CALIBRATION      0x5306
#define SCD30_CMD_VERSION                    0xD100

void SCD30_Init(void);

void scd30_info(void);

bool scd30_send(uint8_t* data, uint8_t size);

bool scd30_receive(uint8_t *data, uint8_t size);

bool scd30_send_without_args(uint16_t cmd);

bool scd30_send_with_args(uint16_t cmd, uint16_t* args, uint16_t size);

bool scd30_receive_words(uint16_t* data, uint8_t size);

void scd30_read_measurement(float* measurement);

void scd30_start_period_measurement(uint16_t pressure);

void scd30_stop_period_measurement(void);

void scd30_set_measurement_interval(uint16_t interval);

void scd30_set_temperature_offset(uint16_t offset);

void scd30_set_altitude(uint16_t altitude);

void scd30_set_forced_recalibration(uint16_t co2_ppm);

void scd30_set_auto_self_calibration(uint16_t status);

bool scd30_get_data_ready(void);

bool scd30_get_auto_self_calibration_status(void);

uint16_t scd30_get_version(void);

uint16_t scd30_get_altitude(void);

uint16_t scd30_get_measurement_interval(void);

uint16_t scd30_get_temperature_offset(void);

uint16_t scd30_get_forced_recalibration_value(void);

#endif /* SCD30_H */

