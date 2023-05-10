import re
import random
import paho.mqtt.client as mqtt
from SQLite_Database import Database


class MQTT_service:
    def __init__(self):
        self.broker = 'broker.emqx.io'
        self.port = 1883
        self.topic = []
        self.sensor_id = []
        self.username = 'Yunfan'
        self.password = '123456'
        self.storage = Database()
        self.clint_id = f'python-mqtt-{random.randint(0, 10000)}'
        self.clint = mqtt.Client(self.clint_id)

    def init(self):
        self.clint.username_pw_set(self.username, self.password)
        self.clint.on_connect = self.on_connect
        self.clint.on_message = self.on_message
        self.clint.on_subscribe = self.on_subscribe
        self.clint.connect(self.broker, self.port)
        self.clint.loop_forever()

    def subscribe(self, topic=None, sensor_id=None):
        if sensor_id is not None and sensor_id not in self.sensor_id:
            self.storage.create_table(table=sensor_id)
            self.sensor_id.append(sensor_id)

        if topic is not None and topic not in self.topic:
            self.topic.append(topic)
            self.clint.subscribe(topic)

    def on_message(self, client, userdata, msg):
        print('Received {} From {} Topic'.format(msg.payload.decode(), msg.topic))
        self.data_conv(stream=msg)
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print('Connected to MQTT Broker!')
        else:
            print('Failed to connect, return code {:d}'.format(rc))
        self.clint.subscribe("/FYP-UNNC/Command")
        self.clint.subscribe("/FYP-UNNC/Sensor_data")

    @staticmethod
    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid))

    def data_conv(self, stream):
        # Add Channel
        if stream.topic == "/FYP-UNNC/Command":
            message = stream.payload.decode()
            if len(message.split(" ")) != 3:
                return

            if message.split(" ")[0] != "Connected":
                return

            try:
                self.subscribe(sensor_id=message.split(" ")[2])
            except:
                print("Unexpected Event")
                return

        # Check Stage 1: Not correct topic
        if stream.topic == "/FYP-UNNC/Sensor_data":
            message = stream.payload.decode()
            sensor = message.split(":")[0]

            # Check Stage 2: Sensor ID Invalid
            id_flag = False
            if len(message.split(":")) > 2:
                print('Wrong Message Format')
                return

            for i in self.sensor_id:
                if sensor == i and len(sensor) == len(i):
                    id_flag = True
                    break
            if not id_flag:
                print('Wrong Sensor ID Received')
                return

            # Check Stage 3: Data Format Invalid
            data = message.split(":")[1].split("; ")
            if len(data) != 4:
                print('Wrong Data Received')
                return
            for i in data:
                try:
                    float(i)
                except:
                    print('Wrong Data Received')
                    return

            try:
                self.storage.update_data(table=sensor, content=data)
            except:
                print("Unexpected Event")
                return
