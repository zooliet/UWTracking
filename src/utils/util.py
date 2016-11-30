import numpy as np

def selection_enlarged(mask, x1, y1, x2, y2, ratio):
	frame_height, frame_width = mask.shape

	w = int((x2 - x1) * ratio)
	h = int((y2 - y1) * ratio)

	cx = (x2 + x1) // 2
	cy = (y2 + y1) // 2

	x1 = max(cx - (w // 2), 0)
	x2 = min(cx + (w // 2), frame_width)
	y1 = max(cy - (h // 2), 0)
	y2 = min(cy + (h // 2), frame_height)

	return (x1, y1), (x2, y2)


def distance(pt1, pt2):
	diff = pt2 - pt1
	dist = np.sqrt((diff**2).sum())
	# print(pt1, pt2, dist)
	return dist




