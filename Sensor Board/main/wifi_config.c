#include "wifi_config.h"

/* FreeRTOS event group to signal when we are connected & ready to make a request */
static EventGroupHandle_t wifi_event_group;

static wifi_config_t wifi_config;
static char wifi_ssid[SSID_SIZE] = {};
static char wifi_password[PWD_SIZE] = {};

static const char *TAG = "WIFI CONFIG";

void NVS_Init(void)
{
    /* NVS Memory Check */
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) 
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    /* Open NVS */
    nvs_handle wificfg_nvs_handler;
    ESP_ERROR_CHECK(nvs_open("WiFi_cfg", NVS_READWRITE, &wificfg_nvs_handler));

    /* Check if needs to update NVS */
    uint32_t wifi_updateID_current = 0;
    err = nvs_get_u32(wificfg_nvs_handler, "wifi_update", &wifi_updateID_current);
    if (WIFI_UPDATE == wifi_updateID_current) ESP_LOGI(TAG, "wifi_cfg needn't to update");
    else
    {
        ESP_LOGI(TAG, "wifi_cfg update now...");
        ESP_ERROR_CHECK(nvs_set_str(wificfg_nvs_handler, "wifi_ssid", WIFI_SSID_INITIAL));
        ESP_ERROR_CHECK(nvs_set_str(wificfg_nvs_handler, "wifi_passwd", WIFI_PWD_INITIAL));
        ESP_ERROR_CHECK(nvs_set_u32(wificfg_nvs_handler, "wifi_update", WIFI_UPDATE));
        ESP_LOGI(TAG, "wifi_cfg update done.");
    }

    /* Commit Change to NVS */
    ESP_ERROR_CHECK(nvs_commit(wificfg_nvs_handler) ); 
    nvs_close(wificfg_nvs_handler);                    
}

static void NVS_Read(char* id, char* pwd)
{
    size_t length;

    /* Open NVS */
    nvs_handle wificfg_nvs_handler;
    ESP_ERROR_CHECK(nvs_open("WiFi_cfg", NVS_READWRITE, &wificfg_nvs_handler));

    /* Get ssid from NVS */
    length = SSID_SIZE;
    ESP_ERROR_CHECK(nvs_get_str(wificfg_nvs_handler, "wifi_ssid", id, &length));

    /* Get password from NVS */
    length = PWD_SIZE;
    ESP_ERROR_CHECK(nvs_get_str(wificfg_nvs_handler, "wifi_passwd", pwd, &length));

    /* Commit Change */
    ESP_ERROR_CHECK( nvs_commit(wificfg_nvs_handler));
    nvs_close(wificfg_nvs_handler);
}

static void NVS_Write(char* id, char* pwd)
{
    /* Open NVS */
    nvs_handle wificfg_nvs_handler;
    ESP_ERROR_CHECK(nvs_open("WiFi_cfg", NVS_READWRITE, &wificfg_nvs_handler));

    /* Set ssid to NVS */
    ESP_ERROR_CHECK(nvs_set_str(wificfg_nvs_handler,"wifi_ssid", id));

    /* Set password to NVS */
    ESP_ERROR_CHECK(nvs_set_str(wificfg_nvs_handler,"wifi_passwd", pwd));

    /* Commit Change */
    ESP_ERROR_CHECK( nvs_commit(wificfg_nvs_handler)); 
    nvs_close(wificfg_nvs_handler);           
}

static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    /* Parameter to define retry times */
    static int wifi_retry_num = 0;

    /* WIFI Event */
    if (event_base == WIFI_EVENT) 
    {
        if (event_id == WIFI_EVENT_STA_START)
            esp_wifi_connect();
        else if (event_id == WIFI_EVENT_STA_DISCONNECTED)
        {
            esp_wifi_connect();

            /* Update retry times */
            wifi_retry_num++;
            ESP_LOGI(TAG, "Retry to connect to the AP %d times.", wifi_retry_num);

            /* If retry more than 5 times, SET WIFI FAILED BIT TO 1 */
            if (wifi_retry_num > 5) xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);

            /* Clear WIFI CONNECTED BIT */
            xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        }
    } 

    /* WIFI IP Adress Event */
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) 
    {
        wifi_retry_num = 0;
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    } 

    /* WIFI Smart_config Event*/
    else if (event_base == SC_EVENT) 
    {
        /* Scan Smart_config Channel */
        if (event_id == SC_EVENT_SCAN_DONE) 
            ESP_LOGI(TAG, "Scan done");
        
        /* Found Channel */
        else if (event_id == SC_EVENT_FOUND_CHANNEL)
            ESP_LOGI(TAG, "Found channel");

        /* Got SSID and PWD */
        else if (event_id == SC_EVENT_GOT_SSID_PSWD)
        {
            ESP_LOGI(TAG, "Got SSID and password");

            /* Got the data from the smart device */
            smartconfig_event_got_ssid_pswd_t *evt = (smartconfig_event_got_ssid_pswd_t *)event_data;

            /* Copy the data to wifi_config */
            bzero(&wifi_config, sizeof(wifi_config_t));  /* Initialize wifi_config struct */
            memcpy(wifi_config.sta.ssid, evt->ssid, sizeof(wifi_config.sta.ssid));
            memcpy(wifi_config.sta.password, evt->password, sizeof(wifi_config.sta.password));
            wifi_config.sta.bssid_set = evt->bssid_set;
            if (wifi_config.sta.bssid_set == true) 
            {
                memcpy(wifi_config.sta.bssid, evt->bssid, sizeof(wifi_config.sta.bssid));
            }

            /* Print SSID and Password to Laptop */
            memcpy(wifi_ssid, evt->ssid, sizeof(evt->ssid));
            memcpy(wifi_password, evt->password, sizeof(evt->password));
            ESP_LOGI(TAG, "SSID:%s", wifi_ssid);
            ESP_LOGI(TAG, "PASSWORD:%s", wifi_password);

            /* Save the SSID and Password to NVS */
            NVS_Write(wifi_ssid, wifi_password);
            ESP_LOGI(TAG, "Smartconfig save wifi_cfg to NVS");

            /* Using New SSID and Password to connect to the WiFi */
            ESP_ERROR_CHECK( esp_wifi_disconnect() );
            ESP_ERROR_CHECK( esp_wifi_set_config(WIFI_IF_STA, &wifi_config) );
            esp_wifi_connect();
        }

        /* Finish Smart Configuration, Send Response to Smart Config Device */
        else if (event_id == SC_EVENT_SEND_ACK_DONE)
            xEventGroupSetBits(wifi_event_group, WIFI_DONE_BIT);
    } 
}

static void smartconfig_init_start(void)
{
    EventBits_t uxBits;
    ESP_ERROR_CHECK(esp_smartconfig_set_type(SC_TYPE_ESPTOUCH) );
    smartconfig_start_config_t cfg = SMARTCONFIG_START_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_smartconfig_start(&cfg) );
    ESP_LOGI(TAG, "smartconfig start ....... \n");

    uxBits = xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT | WIFI_DONE_BIT, true, false, portMAX_DELAY);
    if (uxBits & WIFI_CONNECTED_BIT) 
    {
        ESP_LOGI(TAG, "WiFi Connected to AP");
        esp_smartconfig_stop();
    }
    if (uxBits & WIFI_DONE_BIT) 
    {
        ESP_LOGI(TAG, "Smartconfig over");
        esp_smartconfig_stop();
    }
}

void WIFI_Init(void)
{
    /* Set WIFI POWER SAVING*/
    esp_wifi_set_ps(WIFI_PS_MAX_MODEM);

    /* Initialize TCP/IP */
    ESP_ERROR_CHECK(esp_netif_init());

    /* Create WIFI FreeRtos Event Group */
    wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Initialize WIFI-STA Protocol */
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    assert(sta_netif);

    /* WiFi Initial Configuration */
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    /* Registing WIFI Event, IP Adress Event and Smart config Event to the handler */
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(SC_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    
    /* Read the SSID and Pwd from NVS and copy them to wifi_config */
    NVS_Read(wifi_ssid, wifi_password);
    bzero(&wifi_config, sizeof(wifi_config_t));  /* Initialize wifi_config struct */
    memcpy(wifi_config.sta.ssid, wifi_ssid, sizeof(wifi_config.sta.ssid));
    memcpy(wifi_config.sta.password, wifi_password, sizeof(wifi_config.sta.password));

    /* Start connecting WIFI */
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Create Event Returning Bits for checking current status */
    EventBits_t wifi_event_bits;
    wifi_event_bits = xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdFALSE, pdFALSE, portMAX_DELAY);

    /* Determine current status */
    /* WIFI Connected Successfully */
    if (wifi_event_bits & WIFI_CONNECTED_BIT)
    {
        ESP_LOGI(TAG, "Connected to AP %s Successfully", wifi_ssid);
        vEventGroupDelete(wifi_event_group);
    }
    /* Failed to connect to WIFI*/
    else if (wifi_event_bits & WIFI_FAIL_BIT)
    {
        ESP_LOGI(TAG, "Failed to connect to AP %s", wifi_ssid);
        smartconfig_init_start();
    }
    /* Timed out */
    else
    {
        ESP_LOGI(TAG, "UNEXPECTED EVENT. Try Smart Config");
        smartconfig_init_start();
    }
}
