
# construct the argument parse and parse the arguments
import argparse

def parsing():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", help = "camera number")
    ap.add_argument("-p", "--path", help = "path to video file")
    #
    # ap.add_argument("-n", "--num-frames", type=int, default=10000000, help="# of frames to loop over")
    # ap.add_argument("-d", "--display", action="store_true", help="show display")
    #
    ap.add_argument("-m", "--motor", help = "path to motor device")
    ap.add_argument("-z", "--zoom", help = "path to zoom control port")
    ap.add_argument("-w", "--width", type=int, default=640, help="screen width in px")

    ap.add_argument("--kcf", action="store_true", help="Enable KCF tracking")
    ap.add_argument("--dlib", action="store_true", help="Enable DLIB tracking")
    ap.add_argument("--cmt", action="store_true", help="Enable CMT tracking")

    ap.add_argument("--sub", action="store_true", help="Enable sub tracking")

    # ap.add_argument("--color", action="store_true", help="Enable color subtracking")
    # ap.add_argument("--motion", action="store_true", help="Enable Motion subtracking")
    ap.add_argument("--autozoom", action="store_true", help="Enable automatic zoom control")
    #
    ap.add_argument("--view", help = "select the view") # front, rear, side
    # ap.add_argument("--gui", action="store_true", help="Enable GUI control")
    #
    args = vars(ap.parse_args())
    # print('Command:', args)
    return args

def setWidth(width, cfg):
    cfg['WIDTH'] = width
    cfg['HEIGHT'] = width * 9 // 16
    cfg['HALF_WIDTH'] = width // 2
    cfg['HALF_HEIGHT'] = (width * 9 // 16) // 2
    cfg['MIN_SELECTION_WIDTH'] = 16 # or 20, 10
    cfg['MIN_SELECTION_HEIGHT'] = 9 # or 20, 10

def setView(view, cfg):
    if view == 'front':
        cfg['TITLE'] = '전면'
        cfg['WIN_X'] = 1280
        cfg['WIN_Y'] = 0
        cfg['CHANNEL_NAME'] = 'uwtec:front'
        cfg['CAPTURE_NAME'] = 'front'
    elif view == 'rear':
        cfg['TITLE'] = '후면'
        cfg['WIN_X'] = 0
        cfg['WIN_Y'] = 0
        cfg['CHANNEL_NAME'] = 'uwtec:rear'
        cfg['CAPTURE_NAME'] = 'rear'
    elif view == 'side':
        cfg['TITLE'] = '측면'
        cfg['WIN_X'] = 640
        cfg['WIN_Y'] = 0
        cfg['CHANNEL_NAME'] = 'uwtec:side'
        cfg['CAPTURE_NAME'] = 'side'
    else:
        cfg['TITLE'] = '카메라'
        cfg['WIN_X'] = 0
        cfg['WIN_Y'] = 0
        cfg['CHANNEL_NAME'] = 'uwtec:camera'
        cfg['CAPTURE_NAME'] = 'camera'
