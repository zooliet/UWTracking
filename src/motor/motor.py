import struct
import serial
import time

FOVS = [(62.0000/2, 34.5000/2), #1
		(28.5800/2, 16.5000/2), #2
		(20.6666/2, 11.5000/2),
		(14.4000/2, 8.0000/2), #4
		(12.4000/2, 6.9000/2),
		(10.3333/2, 5.7500/2),
		(8.8571/2, 4.9286/2),
		(7.5000/2, 4.1000/2), #8
		(6.8888/2, 3.8333/2),
		(6.2000/2, 3.4500/2),
		(5.6364/2, 3.1364/2),
		(5.1667/2, 2.8750/2), #12
		(4.7692/2, 2.6538/2),
		(4.4286/2, 2.4643/2),
		(4.1333/2, 2.3000/2),
		(3.8000/2, 2.1562/2), #16
		(3.6471/2, 2.1000/2),
		(3.4444/2, 2.0294/2),
		(3.2632/2, 1.9167/2),
		(3.2000/2, 1.8000/2), ] #20

class Motor:
	TABLE = [0,  94, 188, 226,  97,  63, 221, 131, 194, 156, 126,  32, 163, 253,  31,  65,
		157, 195,  33, 127, 252, 162,  64,  30,  95,   1, 227, 189,  62,  96, 130, 220,
		35, 125, 159, 193,  66,  28, 254, 160, 225, 191,  93,   3, 128, 222,  60,  98,
		190, 224,   2,  92, 223, 129,  99,  61, 124,  34, 192, 158,  29,  67, 161, 255,
		70,  24, 250, 164,  39, 121, 155, 197, 132, 218,  56, 102, 229, 187,  89,   7,
		219, 133, 103,  57, 186, 228,   6,  88,  25,  71, 165, 251, 120,  38, 196, 154,
		101,  59, 217, 135,   4,  90, 184, 230, 167, 249,  27,  69, 198, 152, 122,  36,
		248, 166,  68,  26, 153, 199,  37, 123,  58, 100, 134, 216,  91,   5, 231, 185,
		140, 210,  48, 110, 237, 179,  81,  15,  78,  16, 242, 172,  47, 113, 147, 205,
		17,  79, 173, 243, 112,  46, 204, 146, 211, 141, 111,  49, 178, 236,  14,  80,
		175, 241,  19,  77, 206, 144, 114,  44, 109,  51, 209, 143,  12,  82, 176, 238,
		50, 108, 142, 208,  83,  13, 239, 177, 240, 174,  76,  18, 145, 207,  45, 115,
		202, 148, 118,  40, 171, 245,  23,  73,   8,  86, 180, 234, 105,  55, 213, 139,
		87,   9, 235, 181,  54, 104, 138, 212, 149, 203,  41, 119, 244, 170,  72,  22,
		233, 183,  85,  11, 136, 214,  52, 106,  43, 117, 151, 201,  74,  20, 246, 168,
		116,  42, 200, 150,  21,  75, 169, 247, 182, 232,  10,  84, 215, 137, 107,  53]


	# motor contoller의 가용 최대 pulse: 120,000 pulse/1초

	SPEED = 8000 # pulse per second - full speed: 120000, half speed: 600000
	# 주. SPEED는 FOV에 따라 조정되어야 함: 36000 for x1, 18000 for x2 ...
	MOVING_TIME = 0.01 # 0.1 for 100 ms, 0.01 for 10 ms
	MOVING_STEP = 3
	PULSE_LIMIT = int(SPEED * MOVING_TIME)

	WIDTH = 640  # 640x360, 1024x576, 1280x720, 1920x1080
	HEIGHT = WIDTH * 9 // 16
	HALF_WIDTH = WIDTH//2
	HALF_HEIGHT = HEIGHT//2

	def __init__(self, dev = '/dev/ttyUSB0', baud = 115200):
		self.port = serial.Serial(dev, baud, timeout = 0, parity = serial.PARITY_NONE)
		self.sum_of_x_degree = 0
		self.sum_of_y_degree = 0

	def move(self, x = 255, y = 255, z = 0, f = 0,  t = 1, rel = 0xff):
		t = int(t * 1000000) # sec to us
		# print(x, y, t, rel)
		self.sum_of_x_degree += (x * 0.00048)
		self.sum_of_y_degree += (y * 0.00048)

		encoded = list(struct.pack("3i", *[x, y, t]))

		buffer = [
			0xd5, 0x1A, 0x8e,
			encoded[0], encoded[1], encoded[2], encoded[3],
			# (x & 0xff) , ((x >> 8) & 0xff), ((x >> 16) & 0xff), ((x >> 24) & 0xff),
			encoded[4], encoded[5], encoded[6], encoded[7],
			# (y & 0xff) , ((y >> 8) & 0xff), ((y >> 16) & 0xff), ((y >> 24) & 0xff),
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			encoded[8], encoded[9], encoded[10], encoded[11],
			# (t & 0xff) , ((t >> 8) & 0xff), ((t >> 16) & 0xff), ((t >> 24) & 0xff),
			rel, 0xFF
      	]

		self.send_packet(buffer)


	def send_packet(self, buffer):
		crc8 = self.crc8_calc(buffer[2:-1])
		buffer[len(buffer) - 1] = crc8

		bstr = bytes(buffer)

		self.port.write(bstr)


	def crc8_calc(self, data = []):
		crc8 = 0;

		if len(data) == 0:
			test_data = [213, 26, 142, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 66, 15, 0, 0, 55]
			data = test_data[2:-1]

		for x in data:
			crc8 = Motor.TABLE[crc8 ^ x]

		return crc8


	def pixel_to_pulse(self, x_px, y_px, zoom = 1, limit = False):
		# Logitec
		# x_degree = x_px/320 * 66.59215896/2
		# y_degree = y_px/180 * 40.55232596/2

		# UWTec
		# x_degree = x_px/Motor.HALF_WIDTH * 62.0000/2
		# y_degree = y_px/Motor.HALF_HEIGHT * 34.5000/2

		x_degree = x_px/Motor.HALF_WIDTH * FOVS[zoom-1][0]
		y_degree = y_px/Motor.HALF_HEIGHT * FOVS[zoom-1][1]

		# self.sum_of_x_degree += x_degree
		# self.sum_of_y_degree += y_degree

		x = int(x_degree/0.00048)  # 0.00048은 현재 사용 모테와 기어비로 결정되는 초당 회전 각도 (degree)
		y = int(y_degree/0.00048)  # 0.00048은 현재 사용 모테와 기어비로 결정되는 초당 회전 각도 (degree)
		z = f = 0

		# print("({}, {}): ({}px, {}px) => ({:.4f}°, {:.4f}°) => ({}, {}) pulse".format(centerX, centerY, x_px, y_px, x_degree, y_degree, x, y))
		return x, y, z, f

	def track(self, center_to_x, center_to_y, current_zoom):
		if abs(center_to_x) <= 4:
			center_to_x = 0

		if abs(center_to_y) <= 4:
			center_to_y = 0

		if center_to_x == 0 and center_to_y == 0:
			t_sec = 0
		else:
			(x_to, y_to, z_to, f_to) = self.pixel_to_pulse(center_to_x, center_to_y, current_zoom, limit = True)

			# SPEED = 2400 # full speed: 120000, half speed: 600000
			# MAX_MOVING_TIME = 0.03 # 0.1 for 100 ms
			# d = max(abs(x_to), abs(y_to))
			# t_sec = d / SPEED
			#
			# if t_sec > MAX_MOVING_TIME:
			# 	x_to = int(x_to	* (MAX_MOVING_TIME / t_sec))
			# 	y_to = int(y_to	* (MAX_MOVING_TIME / t_sec))
			# 	t_sec = MAX_MOVING_TIME
			#
			# self.move(x = x_to, y = y_to, z = z_to, f = f_to, t = t_sec)

			# 나중에 아래 상수는 맨위로 이동할 것
			FLICTIONLESS_PULSE_PER_SEC = [0,2400, 2000,0,1600,0,0,0,1200,0,0,0,1000,0,0,0,700,0,0,0,600] # pulse / sec # zoom and fov에 따라 가변적이어야 함: 2400 ~ 300
			MOVING_TIME = 0.01 #0.01 # 10 ms
			PULSE_PER_MOVING_TIME = FLICTIONLESS_PULSE_PER_SEC[current_zoom] * MOVING_TIME
			WEIGHTS = [1,3,6,3,1]
			WEIGHTS_TOTAL = sum(WEIGHTS)
			MAX_PULSE = int(PULSE_PER_MOVING_TIME * WEIGHTS_TOTAL)
			MOVING_STEP = len(WEIGHTS)

			if abs(x_to) > MAX_PULSE:
				x_to = MAX_PULSE if x_to >= 0 else -MAX_PULSE

			if abs(y_to) > MAX_PULSE:
				y_to = MAX_PULSE if y_to >= 0 else -MAX_PULSE

			x_pulses = list(map(lambda weight: int(x_to*weight/WEIGHTS_TOTAL), WEIGHTS))
			y_pulses = list(map(lambda weight: int(y_to*weight/WEIGHTS_TOTAL), WEIGHTS))

			print("Pixel: ({},{}) => Pulse: ({},{}) => [{},{}]".format(center_to_x, center_to_y, x_to, y_to, x_pulses, y_pulses))

			for (x_pulse, y_pulse) in zip(x_pulses, y_pulses):
				self.move(x = x_pulse, y = y_pulse, z = 0, f = 0, t = MOVING_TIME)

			t_sec = MOVING_TIME * MOVING_STEP




		return t_sec

	def zoom_x1(self):
		# print('Zoom to x1')
		buffer = [0xff,0x01,0x00,0x40,0x00,0x00,0x41]
		bstr = bytes(buffer)
		self.port.write(bstr)

	def zoom_x20(self):
		# print('Zoom to x20')
		buffer = [0xff,0x01,0x00,0x20,0x00,0x00,0x21]
		bstr = bytes(buffer)
		self.port.write(bstr)

	def zoom(self, x, direction):
		if direction == 'in':
			buffer = [0xff,0x01,0x00,0x20,0x00,0x00,0x21]
			bstr = bytes(buffer)
			self.port.write(bstr)
		else:
			buffer = [0xff,0x01,0x00,0x40,0x00,0x00,0x41]
			bstr = bytes(buffer)
			self.port.write(bstr)

	def zoom_to(self, x):
		print('Zoom to', x)
		if x == 1:
			self.zoom_x1()
		elif x == 20:
			self.zoom_x20()
		else:
			zoom_to_preset = {2: 1, 4: 2, 8: 3, 12: 4, 16: 5}
			preset = zoom_to_preset[x]

			buffer = [0xff,0x01,0x00,0x07,0x00,preset,0x00]
			checksum = 0
			for el in buffer[1: -1]:
				checksum += el

			checksum = checksum % 256
			buffer[-1] = checksum
			bstr = bytes(buffer)
			self.port.write(bstr)

	def stop_zooming(self):
		buffer = [0xff,0x01,0x00,0x00,0x00,0x00,0x01]
		bstr = bytes(buffer)
		self.port.write(bstr)
		# time.sleep(0.1)
