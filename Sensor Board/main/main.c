#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "esp_log.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_pm.h"

#include "mqtt.h"
#include "am312.h"
#include "scd30.h"
#include "wifi_config.h"

#define LED_INDICATOR 39

static const char *TAG = "MAIN";

static int state = 0;
//static int ticktime = 0;

static float result[3] = {};
static char data_stream[80] = "";

void Power_management_set()
{
     esp_pm_config_esp32s2_t pm_config = {
        .max_freq_mhz = 160,
        .min_freq_mhz = 80,
        .light_sleep_enable = true,
    };
    esp_pm_configure(&pm_config);
}

void LED_Init()
{
    gpio_set_direction(LED_INDICATOR, GPIO_MODE_OUTPUT);
	gpio_set_level(LED_INDICATOR, state);
}

void app_main(void)
{
    Power_management_set();
    vTaskDelay(500 / portTICK_PERIOD_MS);
    NVS_Init();
    vTaskDelay(500 / portTICK_PERIOD_MS);
    AM312_Init();
    vTaskDelay(500 / portTICK_PERIOD_MS);
    SCD30_Init();
    vTaskDelay(500 / portTICK_PERIOD_MS);
    WIFI_Init();
    vTaskDelay(500 / portTICK_PERIOD_MS);
    MQTT_Init();
    vTaskDelay(2000 / portTICK_PERIOD_MS);
    ESP_LOGI(TAG, "Initialization Completed\r\n");

    scd30_start_period_measurement(0);
    vTaskDelay(500 / portTICK_PERIOD_MS);
    scd30_set_altitude(17);
	vTaskDelay(500 / portTICK_PERIOD_MS);
	scd30_set_auto_self_calibration(0);
	vTaskDelay(500 / portTICK_PERIOD_MS);
	scd30_set_measurement_interval(2);
	vTaskDelay(500 / portTICK_PERIOD_MS);
    ESP_LOGI(TAG, "Configuration Done. Now Start to Transmit Data...\r\n");
    vTaskDelay(2000 / portTICK_PERIOD_MS);

    while (1)
    {
        if (scd30_get_data_ready())
		{
			scd30_read_measurement(result);
			sprintf(data_stream, "%s:%.3f; %.3f; %.3f; %d", SENSOR_ID, result[0], result[1], result[2], get_trigger_num());
            MQTT_Publish(data_stream);
            clear_trigger_num();
            vTaskDelay(5000 / portTICK_PERIOD_MS);
		}
        else vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}
