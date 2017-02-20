

DEGREE_PER_PULSE = 0.00048 # 0.00048은 현재 사용 모터와 기어비로 결정되는 펄스 당 회전 각도 (degree)
FOVS = [(62.5000/1, 34.5000/1), #1
        (62.5000/2, 34.5000/2), #2
        (0, 0),
        (62.5000/4, 34.5000/4), #4
        (0, 0),
        (0, 0),
        (0, 0),
        (62.5000/8, 34.5000/8), #8
        (0, 0),
        (0, 0),
        (0, 0),
        (62.5000/12, 34.5000/12), #12
        (0, 0),
        (0, 0),
        (0, 0),
        (62.5000/16, 34.5000/16), #16
        (0, 0),
        (0, 0),
        (0, 0),
        (62.5000/20, 34.5000/20), ] #20

HALF_WIDTH = 320
HALF_HEIGHT = 180

def pixel_to_pulse(x_px, y_px, zoom = 1, speed=30000):
    x_degree = x_px/HALF_WIDTH * FOVS[zoom-1][0]/2
    y_degree = y_px/HALF_HEIGHT * FOVS[zoom-1][1]/2

    x = round(x_degree/DEGREE_PER_PULSE)
    y = round(y_degree/DEGREE_PER_PULSE)
    z = f = 0

    SPEED = speed
    d = max(abs(x), abs(y))
    t_sec = d / SPEED # 소요 시간

    print("[MOTOR] ({}px, {}px) => ({:.4f}°, {:.4f}°) => ({}, {}) pulse in {:04.0f}ms".format(x_px, y_px, x_degree, y_degree, x, y, t_sec*1000))
    return x, y, z, f


def pulse_to_pixel(x_pulse, y_pulse, zoom = 1):
    x_degree = x_pulse * DEGREE_PER_PULSE
    y_degree = y_pulse * DEGREE_PER_PULSE

    x_px = round(x_degree * HALF_WIDTH / FOVS[zoom-1][0] * 2)
    y_px = round(y_degree * HALF_HEIGHT / FOVS[zoom-1][1] * 2)

    print("[MOTOR] ({}, {}) pulse => ({:.4f}°, {:.4f}°) => ({}px, {}px)".format(x_pulse, y_pulse, x_degree, y_degree, x_px, y_px))

    return x_px, y_px


if __name__ == '__main__':
    import sys
    x_px = int(sys.argv[1])
    y_px = int(sys.argv[2])
    scale = int(sys.argv[3])
    speed = int(sys.argv[4])

    x_pulse, y_pulse, z_pulse, f_pulse = pixel_to_pulse(x_px, y_px, scale, speed)
    x_px, y_px = pulse_to_pixel(x_pulse, y_pulse, scale)
