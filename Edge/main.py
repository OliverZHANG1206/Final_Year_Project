import os
import time
import schedule
from MQTT_System import MQTT_service

# Global Variable
day_count = 1
stamp = 1


def save(mqtt_service):
    global day_count, stamp
    for sensor_id in mqtt_service.sensor_id:
        path = './Data/' + sensor_id
        if not os.path.exists(path):
            os.makedirs(path)

        filename = "SensorData_" + sensor_id + "_Day" + str(day_count) + ".csv"
        mqtt_service.storage.save_csv(filename=path+"/"+filename, table=sensor_id, stamp=stamp)
        stamp = mqtt_service.storage.current_stamp(table=sensor_id)
    day_count = day_count + 1


# main program
if __name__ == '__main__':
    path = './Data'
    if not os.path.exists(path):
        os.makedirs(path)
    time.sleep(1)

    mqtt_channel = MQTT_service()
    mqtt_channel.init()
    time.sleep(1)

    #mqtt_channel.subscribe(topic="/FYP-UNNC/Command")
    #mqtt_channel.subscribe(topic="/FYP-UNNC/Sensor_data")
    time.sleep(1)

    schedule.every().day.do(save, mqtt_service=mqtt_channel)

    while True:
        schedule.run_pending()
        # time.sleep(1)
