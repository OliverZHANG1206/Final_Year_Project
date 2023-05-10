#include "scd30.h"

static const char *TAG = "SCD30";

void SCD30_Init(void)
{
	i2c_config_t config =
	{
	    .mode = I2C_MODE_MASTER,
	    .sda_io_num = I2C_MASTER_SDA_IO,         // select GPIO specific to your project
	    .sda_pullup_en = GPIO_PULLUP_ENABLE,
	    .scl_io_num = I2C_MASTER_SCL_IO,         // select GPIO specific to your project
	    .scl_pullup_en = GPIO_PULLUP_ENABLE,
	    .master.clk_speed = I2C_MASTER_FREQ_HZ,  // select frequency specific to your project
	};

	i2c_param_config(I2C_PORT, &config);

	if (i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, I2C_TX_BUF, I2C_RX_BUF, 0) == ESP_OK && scd30_get_version() == SCD30_VERSION)
	    ESP_LOGI(TAG, "Successfully Connected to SCD30 (Version = %x.%x)", SCD30_VERSION >> 8, (uint8_t)SCD30_VERSION);
	else
		ESP_LOGI(TAG, "Failed to Connect SCD30");
}

void scd30_info(void)
{
	printf("       Basic Info of SCD30       \n");
	printf("---------------------------------\n");
	printf("Version: %x.%x\n", SCD30_VERSION >> 8, (uint8_t)SCD30_VERSION);
	printf("Self Calibration: ");
	if (scd30_get_auto_self_calibration_status()) printf("YES\n");
	else printf("NO\n");

	printf("Period Measurement Interval: %ds\n", scd30_get_measurement_interval());
	printf("Altitude: %dm\n", scd30_get_altitude());
	printf("Temperature Offset: %.2fC\n", (float)scd30_get_temperature_offset() / 100);
	printf("CO2 Recalibration: %dppm\n\n", scd30_get_forced_recalibration_value());
}

bool scd30_send(uint8_t* data, uint8_t size)
{
	i2c_cmd_handle_t cmd = i2c_cmd_link_create();
	i2c_master_start(cmd);
	i2c_master_write_byte(cmd, (SCD30_ADDR << 1) | I2C_WRITE, ACK_CHECK_EN);
	i2c_master_write(cmd, data, size, ACK_CHECK_EN);
	i2c_master_stop(cmd);

	esp_err_t result = i2c_master_cmd_begin(I2C_PORT, cmd, 1000 / portTICK_RATE_MS);
	i2c_cmd_link_delete(cmd);

	if (result != ESP_OK) return false;
	else return true;
}

bool scd30_receive(uint8_t *data, uint8_t size)
{
	i2c_cmd_handle_t cmd = i2c_cmd_link_create();
	i2c_master_start(cmd);
	i2c_master_write_byte(cmd, (SCD30_ADDR << 1) | I2C_READ, ACK_CHECK_EN);
	i2c_master_read(cmd, data, size - 1, ACK);
	i2c_master_read_byte(cmd, data + size - 1, NACK);
	i2c_master_stop(cmd);

	esp_err_t result = i2c_master_cmd_begin(I2C_PORT, cmd, 1000 / portTICK_RATE_MS);
	i2c_cmd_link_delete(cmd);

	if (result != ESP_OK) return false;
	else return true;
}

bool scd30_send_with_args(uint16_t cmd, uint16_t* args, uint16_t size)
{
	static uint8_t i = 0;
	uint8_t ptr = 2;
	uint8_t data_serial[2 + size * 3];

	data_serial[0] = cmd >> 8; data_serial[1] = cmd;
	for (i = 0; i < size; i++)
	{
		uint8_t check_box[2] = {args[i] >> 8, args[i]};
		data_serial[ptr++] = args[i] >> 8;
		data_serial[ptr++] = args[i];
		data_serial[ptr++] = crc8_check(check_box, sizeof(check_box));
	}
	return scd30_send(data_serial, 2 + size * 3);
}

bool scd30_send_without_args(uint16_t cmd)
{
	uint8_t data_serial[2] = {cmd >> 8, cmd};
	return scd30_send(data_serial, 2);
}

bool scd30_receive_words(uint16_t* data, uint8_t size)
{
	static uint8_t i = 0;
	uint8_t data_buf[size * 3];

	if (!scd30_receive(data_buf, size * 3)) return false;

	for (i = 0; i < size; i++)
	{
		uint8_t check_box[2] = {data_buf[i * 3], data_buf[i * 3 + 1]};
		if (crc8_check(check_box, sizeof(check_box)) != data_buf[i * 3 + 2]) return false;
		else data[i] = (data_buf[i * 3] << 8 | data_buf[i * 3 + 1]);
	}
	return true;
}

void scd30_start_period_measurement(uint16_t pressure)
{
	if (scd30_send_with_args(SCD30_CMD_START_PERIODIC_MEASUREMENT, &pressure, 1)) ESP_LOGI(TAG, "Start Period Measurement Successfully");
	else ESP_LOGI(TAG, "Failed to Start Period Measurement");
}

void scd30_stop_period_measurement(void)
{
	if (scd30_send_without_args(SCD30_CMD_STOP_PERIODIC_MEASUREMENT)) ESP_LOGI(TAG, "Stop Period Measurement Successfully");
	else ESP_LOGI(TAG, "Failed to Stop Period Measurement");
}

void scd30_set_measurement_interval(uint16_t interval)
{
	if (scd30_send_with_args(SCD30_CMD_SET_MEASUREMENT_INTERVAL, &interval, 1)) ESP_LOGI(TAG, "Set Measurement Period Successfully");
	else ESP_LOGI(TAG, "Failed to Set Measurement Period");
}

void scd30_set_temperature_offset(uint16_t offset)
{
	if (scd30_send_with_args(SCD30_CMD_SET_TEMPERATURE_OFFSET, &offset, 1)) ESP_LOGI(TAG, "Set Temperature Offset Successfully");
	else ESP_LOGI(TAG, "Failed to Set Temperature Offset");
}

void scd30_set_altitude(uint16_t altitude)
{
	if (scd30_send_with_args(SCD30_CMD_SET_ALTITUDE, &altitude, 1)) ESP_LOGI(TAG, "Set Altitude Successfully");
	else ESP_LOGI(TAG, "Failed to Set Altitude");
}

void scd30_set_forced_recalibration(uint16_t co2_ppm)
{
	if (scd30_send_with_args(SCD30_CMD_SET_FORCED_RECALIBRATION, &co2_ppm, 1)) ESP_LOGI(TAG, "Set Recalibration Successfully");
	else ESP_LOGI(TAG, "Failed to Set Recalibration");
}

void scd30_set_auto_self_calibration(uint16_t status)
{
	if (status == SCD30_DEACTIVATE)
	{
		if (scd30_send_with_args(SCD30_CMD_AUTO_SELF_CALIBRATION, &status, 1)) ESP_LOGI(TAG, "Deactivate Auto-calibration Successfully");
		else ESP_LOGI(TAG, "Failed to Deactivate Auto-calibration");
	}
	else
	{
		if (scd30_send_with_args(SCD30_CMD_AUTO_SELF_CALIBRATION, &status, 1)) ESP_LOGI(TAG, "Activate Auto-calibration Successfully");
		else ESP_LOGI(TAG, "Failed to Activate Auto-calibration");
	}
}

bool scd30_get_data_ready(void)
{
	static uint8_t data[3] = {};
	scd30_send_without_args(SCD30_CMD_GET_DATA_READY);

	if (!scd30_receive(data, 3)) return false;

	uint8_t check_dataset[2] = {data[0], data[1]};
	if (crc8_check(check_dataset, sizeof(check_dataset)) != data[2] || (data[0] << 8 | data[1]) == 0x0000) return false;
	else return true;
}

bool scd30_get_auto_self_calibration_status(void)
{
	static uint16_t status;
	scd30_send_without_args(SCD30_CMD_AUTO_SELF_CALIBRATION);
	scd30_receive_words(&status, 1);
	return (status == 0x01) ? true : false;
}

uint16_t scd30_get_version(void)
{
	static uint16_t version = 0;
	scd30_send_without_args(SCD30_CMD_VERSION);
	scd30_receive_words(&version, 1);
	return version;
}

uint16_t scd30_get_measurement_interval(void)
{
	static uint16_t interval;
	scd30_send_without_args(SCD30_CMD_SET_MEASUREMENT_INTERVAL);
	scd30_receive_words(&interval, 1);
	return interval;
}

uint16_t scd30_get_temperature_offset(void)
{
	static uint16_t offset;
	scd30_send_without_args(SCD30_CMD_SET_TEMPERATURE_OFFSET);
	scd30_receive_words(&offset, 1);
	return offset;
}

uint16_t scd30_get_altitude(void)
{
	static uint16_t altitude;
	scd30_send_without_args(SCD30_CMD_SET_ALTITUDE);
	scd30_receive_words(&altitude, 1);
	return altitude;
}

uint16_t scd30_get_forced_recalibration_value(void)
{
	static uint16_t co2_ppm;
	scd30_send_without_args(SCD30_CMD_SET_FORCED_RECALIBRATION);
	scd30_receive_words(&co2_ppm, 1);
	return co2_ppm;
}

void scd30_read_measurement(float* measurement)
{
	static uint16_t data[6] = {};
	scd30_send_without_args(SCD30_CMD_READ_MEASUREMENT);
	scd30_receive_words(data, 6);

	measurement[0] = bytes_to_float(((uint32_t)data[0] << 16 | (uint32_t)data[1])); //CO2
	measurement[1] = bytes_to_float(((uint32_t)data[2] << 16 | (uint32_t)data[3])); //Temperature
	measurement[2] = bytes_to_float(((uint32_t)data[4] << 16 | (uint32_t)data[5])); //Humidity

	ESP_LOGI(TAG, "Get Data %.3f %.3f %.3f", measurement[0], measurement[1], measurement[2]);
}

