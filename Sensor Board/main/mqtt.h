#ifndef MQTT_H
#define MQTT_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_system.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "mqtt_client.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"

#define BROKER_URL  "mqtt://broker.emqx.io"
#define BROKER_PORT 1883

#define SENSOR_ID         "DORMITORY"
#define SENSOR_DATA_ID    "/FYP-UNNC/Sensor_data"
#define SENSOR_COMMAND_ID "/FYP-UNNC/Command"

#define MQTT_CONNECTED_COMMAND_BIT BIT0
#define MQTT_CONNECTED_DATA_BIT    BIT1

struct mqtt_topic
{
    char* topic;
    int msg_id; 
};

void MQTT_Init(void);
void MQTT_Publish(char* data);

#endif /* MQTT_H*/
