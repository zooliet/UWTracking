import redis
import json

################################################################################

def exit_app(redis, root, event=None):
    root.destroy()
    j = {'action': 'quit'}
    jstr = json.dumps(j)

    for channel in ['uwtec:front', 'uwtec:rear', 'uwtec:side']:
        redis.publish(channel, jstr)
    # if tkinter.messagebox.askokcancel("종료?", "프로그램을 종료합니다."):
    #     root.destroy()

################################################################################

def stop_tracking(redis, channel, event=None):
    j = {'action': 'stop_tracking'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

################################################################################

def zoom_in(redis, channel, event=None):
    j = {'action': 'zoom_in'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def zoom_out(redis, channel, event=None):
    j = {'action': 'zoom_out'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def zoom_x1(redis, channel, event=None):
    j = {'action': 'zoom_x1'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def autozoom(redis, channel, autozoom, event=None):
    if autozoom.get():
        j = {'action': 'autozoom', 'value': True}
    else:
        j = {'action': 'autozoom', 'value': False}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def target_scale(redis, channel, scale, event=None):
    j = {'action': 'target_scale', 'value': scale}
    jstr = json.dumps(j)
    print(jstr)
    # redis.publish(channel, jstr)

################################################################################

def pause(redis, channel, event=None):
    j = {'action': 'pause'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def play(redis, channel, event=None):
    j = {'action': 'play'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def recording(redis, channel, button, event=None):
    if button.config('background')[-1] == '#d9d9d9':
        button.config(state='normal', relief='raised', background='orange')
        j = {'action': 'start_recording'}
    else:
        button.config(background='#d9d9d9')
        button.bind('<Enter>', lambda event, b=button: b.config(background='#d9d9d9'))
        j = {'action': 'stop_recording'}

    jstr = json.dumps(j)
    redis.publish(channel, jstr)

################################################################################

def center(redis, channel, event=None):
    j = {'action': 'center'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

def unlock(redis, channel, event=None):
    j = {'action': 'unlock'}
    jstr = json.dumps(j)
    redis.publish(channel, jstr)

################################################################################
