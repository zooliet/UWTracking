# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

import os
import redis
import json
import re

from tkinter import *
import tkinter.messagebox
import tkinter.filedialog
import tkinter.ttk

class GuiCallBacks:
    def exit_app(self, redis, root, event=None):
        root.destroy()
        j = {'action': 'quit'}
        jstr = json.dumps(j)

        for channel in ['uwtec:front', 'uwtec:rear', 'uwtec:side']:
            redis.publish(channel, jstr)
        # if tkinter.messagebox.askokcancel("종료?", "프로그램을 종료합니다."):
        #     root.destroy()

    ################################################################################

    # def stop_tracking(redis, channel, event=None):
    def stop_tracking(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'stop_tracking'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    ################################################################################

    # def zoom_in(redis, channel, event=None):
    def zoom_in(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'zoom_in'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    # def zoom_out(redis, channel, event=None):
    def zoom_out(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'zoom_out'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    # def zoom_x1(redis, channel, event=None):
    def zoom_x1(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'zoom_x1'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    def autozoom(self, redis, channel, autozoom, event=None):
        if autozoom.get():
            j = {'action': 'autozoom', 'value': True}
        else:
            j = {'action': 'autozoom', 'value': False}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    # def auto_scale(redis, channel, scale, event=None):
    def auto_scale(self, redis, channel, scale, event=None):
        value = scale.get()
        j = {'action': 'auto_scale', 'value': value}
        jstr = json.dumps(j)
        # print(jstr)
        redis.publish(channel, jstr)

    ################################################################################

    def pause(self, redis, channel, event=None):
        j = {'action': 'pause'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    def play(self, redis, channel, event=None):
        j = {'action': 'play'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    def recording(self, redis, channel, button, event=None):
        if button.config('background')[-1] == '#d9d9d9':
            button.config(state='normal', relief='raised', background='orange')
            j = {'action': 'start_recording'}
        else:
            button.config(background='#d9d9d9')
            # button.bind('<Enter>', lambda event, b=button: b.config(background='#d9d9d9'))
            j = {'action': 'stop_recording'}

        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    ################################################################################

    # def center(redis, channel, event=None):
    def center(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'center'}
        jstr = json.dumps(j)
        # print(jstr)
        redis.publish(channel, jstr)

    # def unlock(redis, channel, event=None):
    def unlock(self, redis, channel, button, event=None):
        button.config(state='normal', background='orange')
        button.after(600, lambda: button.config(state='normal', background='#d9d9d9'))

        j = {'action': 'unlock'}
        jstr = json.dumps(j)
        redis.publish(channel, jstr)

    ################################################################################


cb = GuiCallBacks()
redis = redis.Redis()

PROGRAM_NAME = "Tracking Controller"
root = Tk()
root.geometry('1920x340+0+412')
root.overrideredirect(1)
root.title(PROGRAM_NAME)


root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
# root.grid_columnconfigure(1, weight=1)
# root.grid_columnconfigure(2, weight=1)
# root.grid_rowconfigure(0, weight=1)

cameras_frame = Frame(root)
cameras_frame.grid(row=0, column=0, sticky='news')
cameras_frame.grid_columnconfigure(0, weight=1)
cameras_frame.grid_columnconfigure(1, weight=1)
cameras_frame.grid_columnconfigure(2, weight=1)
cameras_frame.grid_rowconfigure(0, weight=1)

common_frame = Frame(root)
common_frame.grid(row=1, column=0, sticky='ew')
common_frame.grid_columnconfigure(0, weight=1)
common_frame.grid_columnconfigure(1, weight=1)
common_frame.grid_columnconfigure(2, weight=1)

cam1 = Frame(cameras_frame, width=640, bd=1, relief=GROOVE)
cam1.grid(row=0, column=0, sticky='news')

cam2 = Frame(cameras_frame, width=640, bd=1, relief=GROOVE)
cam2.grid(row=0, column=1, sticky='news')

cam3 = Frame(cameras_frame, width=640, bd=1, relief=GROOVE)
cam3.grid(row=0, column=2, sticky='news')

################################################################################

cwd = os.getcwd()
cwds = cwd.split("/")
pattern = re.compile('uwtracking', re.IGNORECASE)


if cwds[-1] == 'src':
    suffix =  "/".join(cwds[0:-1]) + "/"
elif (bool(re.match(pattern, cwds[-1]))):
    suffix =  cwd + "/"
else:
    suffix = "/usr/local/src/UWTracking/"

# suffix = "" if cwd.split("/")[-1] == 'UWTracking' else "/usr/local/src/UWTracking/"
# suffix = "" if cwd.split("/")[-1] == 'UWTracking' else "/home/uwtec/work/UWTracking/"

################################################################################

center_icon = PhotoImage(file='{}src/icons/center.png'.format(suffix))
unlock_icon = PhotoImage(file='{}src/icons/unlock.png'.format(suffix))

row = 2
for cam, channel in zip([cam1, cam2, cam3], ['uwtec:rear', 'uwtec:side', 'uwtec:front']):
    Label(cam, text='모터:').grid(row=row, column=0, sticky='e', padx=15, pady=(30, 30))
    # Button(cam, text='센터 지정', image=center_icon, compound='left', command=lambda r=redis, c=channel: cb.center(r, c)).grid(row=row, column=1, sticky='ew')
    button =  Button(cam, text='센터 지정', image=center_icon, compound='left')
    button.grid(row=row, column=1, sticky='ew')
    button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.center(r, c, b))

    # # Button(cam, text='범위 해제', image=unlock_icon, compound='left', command=lambda r=redis, c=channel: cb.unlock(r, c)).grid(row=row, column=2, sticky='ew')
    # button =  Button(cam, text='범위 해제', image=unlock_icon, compound='left')
    # button.grid(row=row, column=2, sticky='ew')
    # button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.unlock(r, c, b))

################################################################################

zoom_in_icon = PhotoImage(file='{}src/icons/zoom_in.png'.format(suffix))
zoom_out_icon = PhotoImage(file='{}src/icons/zoom_out.png'.format(suffix))
zoom_wide_icon = PhotoImage(file='{}src/icons/zoom_wide.png'.format(suffix))

autozoom_1 = BooleanVar()
autozoom_2 = BooleanVar()
autozoom_3 = BooleanVar()

row = 3

for cam, channel, autozoom in zip([cam1, cam2, cam3], ['uwtec:rear', 'uwtec:side', 'uwtec:front'], [autozoom_1, autozoom_2, autozoom_3]):
    # print(cam, channel, autozoom)
    Label(cam, text='줌:').grid(row=row, column=0, sticky='e', padx=15, pady=(30, 30))
    # Button(cam, text='줌 인', image=zoom_in_icon, compound='left', command=lambda r=redis, c=channel: cb.zoom_in(r, c)).grid(row=row, column=1, sticky='ew')
    button = Button(cam, text='줌 인', image=zoom_in_icon, compound='left')
    button.grid(row=row, column=1, sticky='ew')
    button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.zoom_in(r, c, b))

    # Button(cam, text='줌 아웃', image=zoom_out_icon, compound='left', command=lambda r=redis, c=channel: cb.zoom_out(r, c)).grid(row=row, column=2, sticky='ew')
    button = Button(cam, text='줌 아웃', image=zoom_out_icon, compound='left')
    button.grid(row=row, column=2, sticky='ew')
    button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.zoom_out(r, c, b))

    # Button(cam, text='광각 줌', image=zoom_wide_icon, compound='left', command=lambda r=redis, c=channel: cb.zoom_x1(r, c)).grid(row=row, column=3, sticky='ew')
    button = Button(cam, text='광각 줌', image=zoom_wide_icon, compound='left')
    button.grid(row=row, column=3, sticky='ew')
    button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.zoom_x1(r, c, b))

    autozoom.set(False)
    Checkbutton(cam, text='자동 줌(%)', variable=autozoom, command=lambda r=redis, c=channel, a=autozoom: cb.autozoom(r, c, a)).grid(row=row, column=4, sticky='ew')
    # Label(cam1, text='타겟 크기(%)').grid(row=row, column=5, sticky='e')
    scale = Scale(cam, from_=20, to=40, label='', orient=HORIZONTAL)
    scale.grid(row=row, column=5, sticky='ew', padx=(15,0), pady=(15, 15))
    scale.set(30)
    scale.bind("<ButtonRelease-1>", lambda event, r=redis, c=channel, s=scale: cb.auto_scale(r, c, s))

    ################################################################################

stop_icon = PhotoImage(file='{}src/icons/stop.png'.format(suffix))

row = 4
for cam, channel in zip([cam1, cam2, cam3], ['uwtec:rear', 'uwtec:side', 'uwtec:front']):
    Label(cam, text='추적:').grid(row=row, column=0, sticky='e', padx=15, pady=(30, 30))
    # Button(cam, text='추적 중지', image= stop_icon, compound='left', command=lambda r=redis, c=channel: cb.stop_tracking(r, c)).grid(row=row, column=1, sticky='ew')
    button = Button(cam, text='추적 중지', image= stop_icon, compound='left')
    button.grid(row=row, column=1, sticky='ew')
    button.bind('<Button-1>', lambda event, r=redis, c=channel, b=button: cb.stop_tracking(r, c, b))
    # Button(cam, text='범위 해제', image=unlock_icon, compound='left', command=lambda r=redis, c=channel: cb.unlock(r, c)).grid(row=row, column=2, sticky='ew')

################################################################################

# pause_icon = PhotoImage(file='{}src/icons/pause.png'.format(suffix))
# play_icon = PhotoImage(file='{}src/icons/play.png'.format(suffix))
# recording_icon = PhotoImage(file='{}src/icons/recording.png'.format(suffix))
#
# row = 5
# for cam, channel in zip([cam1, cam2, cam3], ['uwtec:rear', 'uwtec:side', 'uwtec:front']):
#     Label(cam, text='플레이어:').grid(row=row, column=0, sticky='e', padx=15, pady=(30, 30))
#     Button(cam, text='영상 정지', image=pause_icon, compound='left', command=lambda r=redis, c=channel: cb.pause(r, c)).grid(row=row, column=1, sticky='ew')
#     Button(cam, text='영상 실행', image=play_icon, compound='left', command=lambda r=redis, c=channel: cb.play(r, c)).grid(row=row, column=2, sticky='ew')
#     button = Button(cam, text='영상 저장', image=recording_icon, compound='left')
#     button.grid(row=row, column=3, sticky='ew')
#     button.bind("<Button-1>", lambda event, r=redis, c=channel, b=button: cb.recording(r, c, b))
#     # Button(cam, text='추적 종료', image=stop_icon, compound='left').grid(row=row, column=3, sticky='ew')
#     # Button(cam, text='목표물 탐색', image=zoom_out_icon, compound='left').grid(row=row, column=4, sticky='ew')

################################################################################

# up_icon = PhotoImage(file='src/icons/up.png')
# down_icon = PhotoImage(file='src/icons/down.png')
# left_icon = PhotoImage(file='src/icons/left.png')
# right_icon = PhotoImage(file='src/icons/right.png')
#
# cam1_pan_tilt = Frame(cam1)
# cam1_pan_tilt.grid(row=4, column=1, columnspan=2, pady=(15, 15))
# Button(cam1_pan_tilt, image=up_icon).grid(row=0, column=1)
# Button(cam1_pan_tilt, image=down_icon).grid(row=2, column=1)
# Button(cam1_pan_tilt, image=left_icon).grid(row=1, column=0)
# Button(cam1_pan_tilt, image=right_icon).grid(row=1, column=2)
#
# cam2_pan_tilt = Frame(cam2)
# cam2_pan_tilt.grid(row=4, column=1, columnspan=2, pady=(15, 15))
# Button(cam2_pan_tilt, image=up_icon).grid(row=0, column=1)
# Button(cam2_pan_tilt, image=down_icon).grid(row=2, column=1)
# Button(cam2_pan_tilt, image=left_icon).grid(row=1, column=0)
# Button(cam2_pan_tilt, image=right_icon).grid(row=1, column=2)
#
# cam3_pan_tilt = Frame(cam3)
# cam3_pan_tilt.grid(row=4, column=1, columnspan=2, pady=(15, 15))
# Button(cam3_pan_tilt, image=up_icon).grid(row=0, column=1)
# Button(cam3_pan_tilt, image=down_icon).grid(row=2, column=1)
# Button(cam3_pan_tilt, image=left_icon).grid(row=1, column=0)
# Button(cam3_pan_tilt, image=right_icon).grid(row=1, column=2)
#
# Button(cam1, text='초기 위치 이동').grid(row=4, column=3, sticky='ew')
# Checkbutton(cam1, text='자동 팬틸트').grid(row=4, column=4, sticky='ew')
#
# Button(cam2, text='초기 위치 이동').grid(row=4, column=3, sticky='ew')
# Checkbutton(cam2, text='자동 팬틸트').grid(row=4, column=4, sticky='ew')
#
# Button(cam3, text='초기 위치 이동').grid(row=4, column=3, sticky='ew')
# Checkbutton(cam3, text='자동 팬틸트').grid(row=4, column=4, sticky='ew')


Button(common_frame, text='종료', height=2, command=lambda: cb.exit_app(redis=redis, root=root)).grid(row=0, column=2,  sticky='e', ipadx=15, pady=15, padx=15)


root.protocol('WM_DELETE_WINDOW', cb.exit_app)
root.mainloop()
