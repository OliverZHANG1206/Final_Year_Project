#include "mqtt.h"
#include "am312.h"

static const char *TAG = "MQTT";
static struct mqtt_topic command_receive = {SENSOR_COMMAND_ID, 0};
static struct mqtt_topic data_publish    = {SENSOR_DATA_ID, 0};
static bool cmd_ch_flag = false, data_ch_flag = false;

static EventGroupHandle_t mqtt_event_group;
static esp_mqtt_client_handle_t client;

static void command_task(int length, char* data)
{
	char command_str[20] = "";
	for (int pos = 0; pos <= length - 9; pos++)
		command_str[pos] = *(data + pos + 8);

	if (strncmp(command_str, "Clear", 5) == 0) clear_trigger_num();
	esp_mqtt_client_publish(client, command_receive.topic, "Received", 0, 0, 0);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    ESP_LOGD(TAG, "Event dispatched from event loop base = %s, event_id = %d", base, event_id);
    esp_mqtt_event_handle_t event = event_data;
    //esp_mqtt_client_handle_t client = event->client;

	switch ((esp_mqtt_event_id_t)event_id) 
	{
	    case MQTT_EVENT_CONNECTED:
	        ESP_LOGI(TAG, "MQTT CONNECTED");
			command_receive.msg_id = esp_mqtt_client_subscribe(client, command_receive.topic, 0);
			ESP_LOGI(TAG, "Sent subscribe message successfully, msg_id = %d", command_receive.msg_id);

	        data_publish.msg_id = esp_mqtt_client_subscribe(client, data_publish.topic, 0);
	        ESP_LOGI(TAG, "Sent subscribe message successfully, msg_id = %d", data_publish.msg_id);
	        break;

	    case MQTT_EVENT_DISCONNECTED:
	        ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
	        break;

	    case MQTT_EVENT_SUBSCRIBED:
			if (event->msg_id == command_receive.msg_id)
			{
				ESP_LOGI(TAG, "MQTT SUBSCRIBED TOPIC \"%s\", msg_id=%d", command_receive.topic, event->msg_id);
				xEventGroupSetBits(mqtt_event_group, MQTT_CONNECTED_COMMAND_BIT); cmd_ch_flag = true;
			}
			if (event->msg_id == data_publish.msg_id)
			{
				ESP_LOGI(TAG, "MQTT SUBSCRIBED TOPIC \"%s\", msg_id=%d", data_publish.topic, event->msg_id);
				xEventGroupSetBits(mqtt_event_group, MQTT_CONNECTED_DATA_BIT); data_ch_flag = true;
			}
			if (data_ch_flag && cmd_ch_flag)
			{
				char msg[25] = "Connected to ";
				strcat(msg, SENSOR_ID);
				esp_mqtt_client_publish(client, command_receive.topic, msg, 0, 0, 0);
			}
	        break;

	    case MQTT_EVENT_UNSUBSCRIBED:
	        ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
	        break;

	    case MQTT_EVENT_PUBLISHED:
	        ESP_LOGI(TAG, "MQTT PUBLISHED ON TOPIC \"%s\", msg_id = %d", event->topic, event->msg_id);
	        break;

	    case MQTT_EVENT_DATA:
	        ESP_LOGI(TAG, "TOPIC = %.*s;  DATA = %.*s\r\n", event->topic_len, event->topic, event->data_len, event->data);
			if (strncmp(event->topic, SENSOR_COMMAND_ID, 17) == 0 && strncmp(event->data, SENSOR_ID, 7) == 0) 
				command_task(event->data_len, event->data);
	        break;

	    case MQTT_EVENT_ERROR:
	        ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
	        break;

	    default:
	        ESP_LOGI(TAG, "Other event id:%d", event->event_id);
	        break;
	}
}

void MQTT_Init(void)
{
    /* Create MQTT Structure */
	const esp_mqtt_client_config_t mqtt_cfg =
	{
	    .uri = BROKER_URL,
		.port = BROKER_PORT,
		.keepalive = 180,
		.reconnect_timeout_ms = 5000,
		.disable_auto_reconnect = false,
	};

    /* Create MQTT FreRtos Event*/
	mqtt_event_group = xEventGroupCreate();

    /* Connect to MQTT Broker*/
	client = esp_mqtt_client_init(&mqtt_cfg);
	esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
	esp_mqtt_client_start(client);

	EventBits_t bits = xEventGroupWaitBits(mqtt_event_group, MQTT_CONNECTED_COMMAND_BIT | MQTT_CONNECTED_DATA_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(5000));

	if (bits) ESP_LOGI(TAG, "MQTT Initialized Successfully");
	else ESP_LOGI(TAG, "Failed to Initialize MQTT.");

	vEventGroupDelete(mqtt_event_group);
}

void MQTT_Publish(char* data)
{
	int msg = esp_mqtt_client_publish(client, data_publish.topic, data, 0, 0, 0);

	if (msg == -1) ESP_LOGI(TAG, "Failed to publish");
	else ESP_LOGI(TAG, "Sent publish successful");
}

