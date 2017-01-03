import numpy as np
import cv2

import sys
PY3 = sys.version_info[0] == 3
if PY3:
  xrange = range

from trackers.kcf_tracker import fhog

# ffttools
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags = ((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!

def real(img):
    return img[:,:,0]

def imag(img):
    return img[:,:,1]

def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:,:,0] = a[:,:,0]*b[:,:,0] - a[:,:,1]*b[:,:,1]
    res[:,:,1] = a[:,:,0]*b[:,:,1] + a[:,:,1]*b[:,:,0]
    return res

def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:,:,0]**2 + b[:,:,1]**2)

    res[:,:,0] = (a[:,:,0]*b[:,:,0] + a[:,:,1]*b[:,:,1]) * divisor
    res[:,:,1] = (a[:,:,1]*b[:,:,0] + a[:,:,0]*b[:,:,1]) * divisor
    return res

def rearrange(img):
    #return np.fft.fftshift(img, axes=(0,1))
    assert(img.ndim==2)
    img_ = np.zeros(img.shape, img.dtype)
    # xh, yh = img.shape[1]/2, img.shape[0]/2
    # img_[0:yh,0:xh], img_[yh:img.shape[0],xh:img.shape[1]] = img[yh:img.shape[0],xh:img.shape[1]], img[0:yh,0:xh]
    # img_[0:yh,xh:img.shape[1]], img_[yh:img.shape[0],0:xh] = img[yh:img.shape[0],0:xh], img[0:yh,xh:img.shape[1]]

    xh, yh = img.shape[1]//2, img.shape[0]//2
    img_[0:yh,0:xh], img_[-yh:img.shape[0],-xh:img.shape[1]] = img[-yh:img.shape[0],-xh:img.shape[1]], img[0:yh,0:xh]
    img_[0:yh,-xh:img.shape[1]], img_[-yh:img.shape[0],0:xh] = img[-yh:img.shape[0],0:xh], img[0:yh,-xh:img.shape[1]]

    return img_

# recttools
def x2(rect):
    return rect[0] + rect[2]

def y2(rect):
    return rect[1] + rect[3]

def limit(rect, limit):
    if(rect[0]+rect[2] > limit[0]+limit[2]):
        rect[2] = limit[0]+limit[2]-rect[0]
    if(rect[1]+rect[3] > limit[1]+limit[3]):
        rect[3] = limit[1]+limit[3]-rect[1]
    if(rect[0] < limit[0]):
        rect[2] -= (limit[0]-rect[0])
        rect[0] = limit[0]
    if(rect[1] < limit[1]):
        rect[3] -= (limit[1]-rect[1])
        rect[1] = limit[1]
    if(rect[2] < 0):
        rect[2] = 0
    if(rect[3] < 0):
        rect[3] = 0
    return rect

def getBorder(original, limited):
    res = [0,0,0,0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert(np.all(np.array(res) >= 0))
    return res

def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0,0,img.shape[1],img.shape[0]])   # modify cutWindow
    assert(cutWindow[2]>0 and cutWindow[3]>0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1]+cutWindow[3], cutWindow[0]:cutWindow[0]+cutWindow[2]]

    if(border != [0,0,0,0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# KCF tracker
class KCFTracker:
    PREV_HISTORY_SIZE = 100

    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self.lambdar = 0.0001   # regularization
        self.padding = 2.5   # extra area surrounding the target
        self.output_sigma_factor = 0.125   # bandwidth of gaussian target

        if(hog):  # HOG feature
            # VOT
            self.interp_factor = 0.012   # linear interpolation factor for adaptation
            self.sigma = 0.6  # gaussian kernel bandwidth
            # TPAMI   #interp_factor = 0.02   #sigma = 0.5
            self.cell_size = 4   # HOG cell size
            self._hogfeatures = True
        else:  # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if(multiscale):
            self.template_size = 96   # template size
            self.scale_step = 1.05   # scale step for multi-scale estimation
            self.scale_weight = 0.96   # to downweight detection scores of other scales for added stability

            self.scale_step_extra = 1.1
            self.scale_weight_extra = 0.91
        elif(fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0,0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0.,0.,0.,0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0,0,0]  #[int,int,int]
        self._scale = 1.   # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])

    def subPixelPeak(self, left, center, right):
        divisor = 2*center - right - left   #float
        return (0 if abs(divisor)<1e-3 else 0.5*(right-left)/divisor)

    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t/(self.size_patch[1]-1)))
        hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t/(self.size_patch[0]-1)))
        hann2d = hann2t * hann1t

        if(self._hogfeatures):
            hann1d = hann2d.reshape(self.size_patch[0]*self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey/2, sizex/2
        output_sigma = np.sqrt(sizex*sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma*output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y-syh)**2, (x-sxh)**2
        res = np.exp(mult * (y+x))
        return fftd(res)

    def gaussianCorrelation(self, x1, x2):
        if(self._hogfeatures):
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in xrange(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB = True)
                caux = real(fftd(caux, True))
                #caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB = True)   # 'conjB=' is necessary!
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if(x1.ndim==3 and x2.ndim==3):
            d = (np.sum(x1[:,:,0]*x1[:,:,0]) + np.sum(x2[:,:,0]*x2[:,:,0]) - 2.0*c) / (self.size_patch[0]*self.size_patch[1]*self.size_patch[2])
        elif(x1.ndim==2 and x2.ndim==2):
            d = (np.sum(x1*x1) + np.sum(x2*x2) - 2.0*c) / (self.size_patch[0]*self.size_patch[1]*self.size_patch[2])

        d = d * (d>=0)
        d = np.exp(-d / (self.sigma*self.sigma))

        return d

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0,0,0,0]   #[int,int,int,int]
        cx = self._roi[0] + self._roi[2]/2  #float
        cy = self._roi[1] + self._roi[3]/2  #float

        if(inithann):
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if(self.template_size > 1):
                if(padded_w >= padded_h):
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if(self._hogfeatures):
                self._tmpl_sz[0] = int(self._tmpl_sz[0] / (2*self.cell_size) * 2*self.cell_size + 2*self.cell_size)
                self._tmpl_sz[1] = int(self._tmpl_sz[1] / (2*self.cell_size) * 2*self.cell_size + 2*self.cell_size)
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0] / 2 * 2)
                self._tmpl_sz[1] = int(self._tmpl_sz[1] / 2 * 2)


        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2]/2)
        extracted_roi[1] = int(cy - extracted_roi[3]/2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)

        # hl1sqi - 짝수 이면서 4의 배수 찾기
        if (self._tmpl_sz[0] // 4) % 2 != 0:
            mod = self._tmpl_sz[0] % 8
            self._tmpl_sz[0] = self._tmpl_sz[0] + (8 - mod)

        if (self._tmpl_sz[1] // 4) % 2 != 0:
            mod = self._tmpl_sz[1] % 8
            self._tmpl_sz[1] = self._tmpl_sz[1] + (8 - mod)

        if(z.shape[1]!=self._tmpl_sz[0] or z.shape[0]!=self._tmpl_sz[1]):
            z = cv2.resize(z, tuple(self._tmpl_sz))

        # print('[KCF] z.shape: {}, tmpl_sz: {}'.format(z.shape, self._tmpl_sz))

        if(self._hogfeatures):
            mapp = {'sizeX':0, 'sizeY':0, 'numFeatures':0, 'map':0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            # print("[KCF] mapp[sizeY]: {}, mapp[sizeX]: {}".format(mapp['sizeY'], mapp['sizeX']))
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            FeaturesMap = mapp['map'].reshape((self.size_patch[0]*self.size_patch[1], self.size_patch[2])).T   # (size_patch[2], size_patch[0]*size_patch[1])
        else:
            if(z.ndim==3 and z.shape[2]==3):
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)   # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
            elif(z.ndim==2):
                FeaturesMap = z   #(size_patch[0], size_patch[1]) #np.int8  #0~255
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if(inithann):
            self.createHanningMats()  # createHanningMats need size_patch

        FeaturesMap = self.hann * FeaturesMap
        return FeaturesMap

    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))

        _, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]   # cv::Point2f, [x,y]  #[float,float]

        if(pi[0]>0 and pi[0]<res.shape[1]-1):
            p[0] += self.subPixelPeak(res[pi[1],pi[0]-1], pv, res[pi[1],pi[0]+1])
        if(pi[1]>0 and pi[1]<res.shape[0]-1):
            p[1] += self.subPixelPeak(res[pi[1]-1,pi[0]], pv, res[pi[1]+1,pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        alphaf = complexDivision(self._prob, fftd(k)+self.lambdar)

        self._tmpl = (1-train_interp_factor)*self._tmpl + train_interp_factor*x
        self._alphaf = (1-train_interp_factor)*self._alphaf + train_interp_factor*alphaf


    def init(self, image):

        self.force_init_flag = False
        self.consecutive_cmt_lost = 0
        self.consecutive_cmt_found = 0
        self.cmt_was_found = False
        self.area = (self.x2-self.x1) * (self.y2-self.y1)


        roi = [self.x1, self.y1, self.x2-self.x1, self.y2-self.y1]
        self._roi = list(map(float, roi))
        assert(roi[2]>0 and roi[3]>0)

        self._tmpl = self.getFeatures(image, 1)
        # print("size_patch[0]: {}, size_patch[1]: {}".format(self.size_patch[0], self.size_patch[1]))
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)

        self.train(self._tmpl, 1.0)
        # print("[KCF] tmpl: {}, prob: {}, alphaf: {}".format(self._tmpl.shape, self._prob.shape, self._alphaf.shape))
        # 주1. prob의 shape: (16,22,2) => 정상, (17,22,2) => 상하로 불안정, (16,24,2) => 좌우로 불안정
        # 주2. prob.shape는 size_patch[0]과 size_patch[1]과 같음
        # 주3. size_patch[]는 getFeatures()에서 결정하므로 반드시 짝수를 return 하도록 해야함
        # 주4. getFeatures()에서 fhog.getFeatureMaps(z,...)를 호출할 때 넘겨지는 z.shape에 의해서 size_patch[]가 결정됨
        # 주5. z.shape를 4로 나누었을때 항상 짝수가 되는 값이 되어야 문제 해결
        # 주6. 이를 위해 self._tmpl_sz[]의 값을 강제로 조정

        self.mean_width = self.x2 - self.x1
        self.mean_height = self.y2 - self.y1

        self.prev_widths = np.array([self.mean_width], dtype=np.int16)
        self.prev_heights = np.array([self.mean_height], dtype=np.int16)

        # self.locations = np.empty((0,2))
        # self.dir_x_count = 0
        # self.dir_y_count = 0

    def update(self, image):
        if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
        if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
        if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 2
        if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2]/2.
        cy = self._roi[1] + self._roi[3]/2.

        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        # if abs(loc[1]) < 0.1:
        # self.locations = np.append(self.locations, [loc], axis=0)
        #
        # if len(self.locations) > self.PREV_HISTORY_SIZE:
        #     self.locations = np.delete(self.locations, (0), axis=0)
        #
        # self.location = np.sum(self.locations, axis=0)
        # print("[KCF] ({:+.4f}, {:+.4f}) =>({:+.4f}, {:+.4f}) peak {:.2f}".format(loc[0], loc[1], self.location[0], self.location[1], peak_value))

        if (self.scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0/self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            # Test at a smaller _scale
            new_loc3, new_peak_value3 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0/1.1))
            # Test at a bigger _scale
            new_loc4, new_peak_value4 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.1))

            new_peak_value1 = self.scale_weight * new_peak_value1
            new_peak_value2 = self.scale_weight * new_peak_value2
            new_peak_value3 = self.scale_weight_extra * new_peak_value3
            new_peak_value4 = self.scale_weight_extra * new_peak_value4

            peaks = [peak_value, new_peak_value1, new_peak_value2, new_peak_value3, new_peak_value4]
            peak_index = peaks.index(max(peaks))
            # print("[KCF] peaks[{}] is selected".format(peak_index))

            if peak_index == 1: # new_peak_value1
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif peak_index == 2: # new_peak_value2
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step
            elif peak_index == 3: # new_peak_value3
                loc = new_loc3
                peak_value = new_peak_value3
                self._scale /= self.scale_step_extra
                self._roi[2] /= self.scale_step_extra
                self._roi[3] /= self.scale_step_extra
            elif peak_index == 4: # new_peak_value4
                loc = new_loc4
                peak_value = new_peak_value4
                self._scale *= self.scale_step_extra
                self._roi[2] *= self.scale_step_extra
                self._roi[3] *= self.scale_step_extra

            # 원래 구현:
            # if(self.scale_weight*new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
            #     loc = new_loc1
            #     peak_value = new_peak_value1
            #     self._scale /= self.scale_step
            #     self._roi[2] /= self.scale_step
            #     self._roi[3] /= self.scale_step
            # elif(self.scale_weight*new_peak_value2 > peak_value):
            #     loc = new_loc2
            #     peak_value = new_peak_value2
            #     self._scale *= self.scale_step
            #     self._roi[2] *= self.scale_step
            #     self._roi[3] *= self.scale_step

        # print("[KCF] loc: ({:+.4f},{:+.4f}), peak_value: {}:{:.2f}, scale: {:.2f}".format(loc[0], loc[1], peak_index, peak_value, self._scale))

        self._roi[0] = cx - self._roi[2]/2.0 + loc[0]*self.cell_size*self._scale
        self._roi[1] = cy - self._roi[3]/2.0 + loc[1]*self.cell_size*self._scale

        if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 1
        if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 1
        if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
        if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
        assert(self._roi[2]>0 and self._roi[3]>0)

        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        # hl1sqi
        return self._roi, peak_value, loc
