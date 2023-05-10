#ifndef WIFI_CONFIG_H
#define WIFI_CONFIG_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "nvs_flash.h"

#include "esp_wifi.h"
#include "esp_wpa2.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"
#include "esp_smartconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#define WIFI_UPDATE        512
#define WIFI_SSID_INITIAL "UNNC-IoT"
#define WIFI_PWD_INITIAL  "UNNC@2018"

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
#define WIFI_DONE_BIT      BIT2

#define SSID_SIZE          33
#define PWD_SIZE           65

void NVS_Init(void);
void WIFI_Init(void);

#endif /* WIFI_CONFIG_H*/