
import redis
import json
import threading
import datetime

class RedisAgent(threading.Thread):
    def __init__(self, r, channels):
        threading.Thread.__init__(self)
        self.redis = r
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(channels)

        self.quit = False
        self.stop_tracking = False

        self.zoom_in = False
        self.zoom_out = False
        self.zoom_x1 = False
        self.autozoom = False

        self.pause = False
        self.play = False
        self.start_recording = False
        self.stop_recording = False

        self.center = False
        self.unlock = False


    def run(self):
        for item in self.pubsub.listen():
            # print(item)
            if(item['type'] == 'message'):
                if(item['data'].decode() == 'KILL'):
                    self.pubsub.unsubscribe()
                    break
                else:
                    self.work(item)

    def work(self, item):
        message = item['data'].decode()
        j = json.loads(message)
        if j['action'] == 'quit':
            self.quit = True

        elif j['action'] == 'stop_tracking':
            self.stop_tracking = True

        elif j['action'] == 'zoom_in':
            self.zoom_in = True
        elif j['action'] == 'zoom_out':
            self.zoom_out = True
        elif j['action'] == 'zoom_x1':
            self.zoom_x1 = True
        elif j['action'] == 'autozoom':
            self.autozoom = True
            if j['value']:
                self.autozoom_enable = True
            else:
                self.autozoom_enable = False

        elif j['action'] == 'pause':
            self.pause = True
        elif j['action'] == 'play':
            self.play = True
        elif j['action'] == 'start_recording':
            self.start_recording = True
        elif j['action'] == 'stop_recording':
            self.stop_recording = True

        elif j['action'] == 'center':
            self.center = True
        elif j['action'] == 'unlock':
            self.unlock = True
        elif j['action'] == 'time':
            print(j['value'])

    def stop(self, channel):
        self.redis.publish(channel, 'KILL')

    def command(self, channel, j):
        jstr = json.dumps(j)
        self.redis.publish(channel, jstr)

    def test(self, channel):
        current_time = datetime.datetime.now().time().isoformat()
        j = {'action': 'time', 'value': current_time}
        jstr = json.dumps(j)
        self.redis.publish(channel, jstr)
