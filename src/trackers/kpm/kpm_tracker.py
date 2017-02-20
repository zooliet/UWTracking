import cv2
import numpy as np
import imutils
from utils import util

import itertools
import scipy.spatial
import scipy.cluster



class KPMTracker:
    THR_OUTLIER = 20
    THR_CONF = 0.75
    THR_RATIO = 0.8
    DESC_LENGTH = 512
    MIN_NUM_OF_KEYPOINTS_FOR_BRISK_THRESHOLD = 900 # 900
    PREV_HISTORY_SIZE = 100

    def __init__(self):
        self.estimate_scale = True
        self.estimate_rotation = False

        self.detector = cv2.BRISK_create(10)
        # self.detector = cv2.AKAZE_create()
        # self.detector = cv2.xfeatures2d.SIFT_create()
        self.descriptor = self.detector
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def init_bgsub(self, frame):
        # self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    def update_bgsub(self, frame):
        fgmask = self.fgbg.apply(frame)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        cv2.imshow('sub', fgmask)

    def init_cmt(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        (tl, br) = ((self.x1, self.y1), (self.x2, self.y2))

        self.cX = self.x1 + (self.x2 - self.x1) // 2
        self.cY = self.y1 + (self.y2 - self.y1) // 2

        self.force_init_flag = False
        self.has_result = False
        self.frame_idx = 0

        # Get initial keypoints in whole image
        keypoints_cv = self.detector.detect(gray)
        # Remember keypoints that are in the rectangle as selected keypoints

        if len(keypoints_cv) == 0:
            # print("[CMT] No keypoints found somehow")
            num_selected_keypoints = 0
        else:
            ind = util.in_rect(keypoints_cv, tl, br)
            # keypoints_cv가 tl, br 영역안에 놓여 있는지를 True, False로 표현: [False, False, True, False, True, ....]

            selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
            # tl, br 영역안에 있는 keypoints 만을 수집
            # list(itertools.compress([1,2,3], [True, False, True])) => [1, 3]

            selected_keypoints_cv, self.selected_features = self.descriptor.compute(gray, selected_keypoints_cv)
            # selected_keypoints_cv: [kp1, kp2, ..., kpm]
            # self_selected_features: (m, 64)

            selected_keypoints = util.keypoints_cv_to_np(selected_keypoints_cv)
            # cv2 list => numpy array
            # selected_keypoints: (m, 2)

            num_selected_keypoints = len(selected_keypoints_cv)
            # print("[CMT] num_selected_keypoints is {}".format(num_selected_keypoints))
            # print("[CMT] num_background_keypoints is {}".format(len(background_keypoints_cv)))

        if num_selected_keypoints != 0:
            # Remember keypoints that are not in the rectangle as background keypoints
            background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
            # 전체 keypoints 중에 (tl, br)영역내에 포함되지 않는 keypoints를 계산: [kp1, kp2, kp3, ... n]

            background_keypoints_cv, background_features = self.descriptor.compute(gray, background_keypoints_cv)
            # background keypoints => backgroud features by descriptor
            # background_keypoints_cv: [kp1, kp2, ..., kpn]
            # background_features: (n, 64)

            _ = util.keypoints_cv_to_np(background_keypoints_cv)
            # cv2 list => numpy array: 왜 하는지 모르겠음
            # print("[CMT] num_background_keypoints is {}".format(len(background_keypoints_cv)))

            # Assign each keypoint a class starting from 1, background is 0
            self.selected_classes = np.array(range(num_selected_keypoints)) + 1
            # selected keypoint 마다 고유한 class를 부여하기 위해 class label을 생성
            # selected_classes: (m, ) => [1, 2, 3, 4, 5, 6, ..., m]

            background_classes = np.zeros(len(background_keypoints_cv))
            # background keypoint 에는 모두 class label 0를 부여할 계획
            # background_classes: (n, ) => [0, 0, 0, 0]

            # Stack background features and selected features into database
            if len(background_keypoints_cv) > 0:
                self.features_database = np.vstack((background_features, self.selected_features))
                # features_database: (n, 64) + (m, 64) = (n+m, 64)
            else: # hl1sqi
                self.features_database = self.selected_features

            # Same for classes
            self.classes_database = np.hstack((background_classes, self.selected_classes))
            # database_classes: (n, ) + (m, )  = (n+m, ) => [0,0,..0, 1, 2, 3, 4, m]

            # Get all distances between selected keypoints in squareform
            pdist = scipy.spatial.distance.pdist(selected_keypoints)
            # selected keypoints 모든 쌍에 대한 거리 계산
            # pdist: (mC2, )

            self.squareform = scipy.spatial.distance.squareform(pdist)
            # pdist(mC2, ) => squreform(m, m)

            # # Get all angles between selected keypoints
            # angles = np.empty((num_selected_keypoints, num_selected_keypoints))
            # # angels: (m, m) with all 0's
            #
            # for k1, i1 in zip(selected_keypoints, range(num_selected_keypoints)):
            #     for k2, i2 in zip(selected_keypoints, range(num_selected_keypoints)):
            #         # Compute vector from k1 to k2
            #         v = k2 - k1
            #         # Compute angle of this vector with respect to x axis
            #         angle = np.math.atan2(v[1], v[0])
            #         # Store angle
            #         angles[i1, i2] = angle
            #
            # self.angles = angles

            # Find the center of selected keypoints
            center = np.mean(selected_keypoints, axis=0)
            # Find the center of selected keypoints along axis=0
            # By axis=0, column 별 합계를 계산
            # center: [169.4553833, 179.31629405]

            # Remember the rectangle coordinates relative to the center
            self.center_to_tl = np.array(tl) - center
            self.center_to_tr = np.array([br[0], tl[1]]) - center
            self.center_to_br = np.array(br) - center
            self.center_to_bl = np.array([tl[0], br[1]]) - center
            # center로 부터 주어진 영역의 4꼭지점간의 상대 좌표를 기록


            # Calculate springs of each keypoint
            self.springs = selected_keypoints - center
            # center로 부터 모든 selected keypoints간의 상대 좌표를 기록

            # Set start image for tracking
            self.gray0 = gray

            # Make keypoints 'active' keypoints
            self.active_keypoints = np.copy(selected_keypoints)

            # Attach class information to active keypoints
            self.active_keypoints = np.hstack((selected_keypoints, self.selected_classes[:, None]))
            # selected_keypoints: (m, 2) + selected_clsses[:, None]: (m, 1) = active_keypoints: (m, 3)
            # active keypoints는 [(kp1, 1), (kp2, 2), (kp3, 3)]와 같이 keypoint와 class label이 결합된 형태

            # Remember number of initial keypoints
            self.num_initial_keypoints = len(selected_keypoints_cv)

            self.initial_keypoints = np.copy(self.active_keypoints)
        else:
            self.num_initial_keypoints = 0

    def update_cmt(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        tracked_keypoints, _ = self.track(self.gray0, gray, self.active_keypoints)

        (center, scale_estimate, rotation_estimate, tracked_keypoints) = self.estimate(tracked_keypoints)
        # 입력 tracked_keypoints: Optical FLow 방식으로 걸러낸 tracked keypoints
        # 출력 tracked_keypoints: inliner에 해당하는 tracked keypoints


        # Detect keypoints, compute descriptors
        # keypoints_cv = self.detector.detect(gray)
        # keypoints_cv, features = self.descriptor.compute(gray, keypoints_cv)
        #
        # # Create list of active keypoints
        # active_keypoints = np.zeros((0, 3))
        #
        # # Get the best two matches for each feature
        # matches_all = self.matcher.knnMatch(features, self.features_database, 2)
        # # 전체 화면에 대해 초기 프레임의 features와 지금 프레임의 features를 비교해서 각 feature당 제일 잘 맞는 2개의 match 추출
        # # ([DM01, DM02], [DM11, DM12], ... [DMm1, DMm2])
        #
        # # Get all matches for selected features
        # if not any(np.isnan(center)):
        #     selected_matches_all = self.matcher.knnMatch(features, self.selected_features, len(self.selected_features))
        #     # 초기 프레임의 selected_features와 지금 프레임의 features의 matches 얻어냄


        self.active_keypoints = np.copy(tracked_keypoints)
        self.gray0 = np.copy(gray)

        if False:

            # For each keypoint and its descriptor
            if len(keypoints_cv) > 0:
                transformed_springs = scale_estimate * util.rotate(self.springs, -rotation_estimate)
                for i in range(len(keypoints_cv)):
                    # Retrieve keypoint location
                    location = np.array(keypoints_cv[i].pt)

                    # First: Match over whole image
                    # Compute distances to all descriptors
                    matches = matches_all[i]
                    distances = np.array([m.distance for m in matches])

                    # Convert distances to confidences, do not weight
                    combined = 1 - distances / self.DESC_LENGTH
                    classes = self.classes_database

                    # Get best and second best index
                    try:
                        bestInd = matches[0].trainIdx
                        secondBestInd = matches[1].trainIdx
                    except Exception as e:
                        print(len(matches))

                    # Compute distance ratio according to Lowe
                    try:
                        ratio = (1 - combined[0]) / (1 - combined[1])
                    except:
                        print(combined)
                    # print('[CMT] Pts {}: Distance {} => combined {}: ratio{}'.format(location, distances, combined, ratio))

                    # Extract class of best match
                    keypoint_class = classes[bestInd]
                    # print("[CMT] {}: Class[{}] => {}, Class[{}] => {}".format(i, bestInd, keypoint_class, secondBestInd, classes[secondBestInd]))
                    # print('[CMT] {}: Pts {}: Distance {} => combined {}: ratio: {}'.format(i, location, distances, combined, ratio))

                    # If distance ratio is ok and absolute distance is ok and keypoint class is not background
                    if ratio < self.THR_RATIO and combined[0] > self.THR_CONF and keypoint_class != 0:
                        # print('[CMT] {}: {}: combined {}: ratio: {}, Class[{}] => {}'.format(i, location, combined[0], ratio, bestInd, keypoint_class))
                        # Add keypoint to active keypoints
                        new_kpt = np.append(location, keypoint_class)
                        active_keypoints = np.append(active_keypoints, np.array([new_kpt]), axis=0)

                    # In a second step, try to match difficult keypoints
                    # If structural constraints are applicable
                    if not any(np.isnan(center)):
                        # Compute distances to initial descriptors
                        matches = selected_matches_all[i]
                        distances = np.array([m.distance for m in matches])

                        # Re-order the distances based on indexing
                        idxs = np.argsort(np.array([m.trainIdx for m in matches]))
                        distances = distances[idxs]

                        # Convert distances to confidences
                        confidences = 1 - distances / self.DESC_LENGTH

                        # Compute the keypoint location relative to the object center
                        relative_location = location - center

                        # Compute the distances to all springs
                        displacements = util.L2norm(transformed_springs - relative_location)

                        # For each spring, calculate weight
                        weight = displacements < self.THR_OUTLIER  # Could be smooth function

                        combined = weight * confidences

                        classes = self.selected_classes

                        # Sort in descending order
                        sorted_conf = np.argsort(combined)[::-1]  # reverse

                        # Get best and second best index
                        bestInd = sorted_conf[0]
                        secondBestInd = sorted_conf[1]

                        # Compute distance ratio according to Lowe
                        ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd])

                        # Extract class of best match
                        keypoint_class = classes[bestInd]

                        # If distance ratio is ok and absolute distance is ok and keypoint class is not background
                        if ratio < self.THR_RATIO and combined[bestInd] > self.THR_CONF and keypoint_class != 0:
                            # Add keypoint to active keypoints
                            new_kpt = np.append(location, keypoint_class)

                            # Check whether same class already exists
                            if active_keypoints.size > 0:
                                same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
                                active_keypoints = np.delete(active_keypoints, same_class, axis=0)

                            active_keypoints = np.append(active_keypoints, np.array([new_kpt]), axis=0)

            # If some keypoints have been tracked
            if tracked_keypoints.size > 0:
                # Extract the keypoint classes
                tracked_classes = tracked_keypoints[:, 2]

                # If there already are some active keypoints
                if active_keypoints.size > 0:
                    # Add all tracked keypoints that have not been matched
                    associated_classes = active_keypoints[:, 2]
                    missing = ~np.in1d(tracked_classes, associated_classes)
                    active_keypoints = np.append(active_keypoints, tracked_keypoints[missing, :], axis=0)
                else: # Else use all tracked keypoints
                    active_keypoints = tracked_keypoints

            self.center = center
            self.scale_estimate = scale_estimate
            self.rotation_estimate = rotation_estimate
            self.tracked_keypoints = tracked_keypoints
            self.active_keypoints = active_keypoints
            self.gray0 = gray
            self.frame_idx += 1

            self.tl = (np.nan, np.nan)
            self.tr = (np.nan, np.nan)
            self.br = (np.nan, np.nan)
            self.bl = (np.nan, np.nan)

            self.has_result = False

            if not any(np.isnan(self.center)) and self.active_keypoints.shape[0] > 4: #self.num_initial_keypoints / 10:
                self.has_result = True

                tl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tl[None, :], rotation_estimate).squeeze())
                tr = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tr[None, :], rotation_estimate).squeeze())
                br = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_br[None, :], rotation_estimate).squeeze())
                bl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_bl[None, :], rotation_estimate).squeeze())

                self.tl = tl
                self.tr = tr
                self.bl = bl
                self.br = br

    def track(self, prev_gray, current_gray, keypoints, THR_FB=20, tl=(0,0), br=(0, 0)):
        if type(keypoints) is list:
            keypoints = keypoints_cv_to_np(keypoints)

        num_keypoints = keypoints.shape[0]
        # num_keypoints: m

        # Status of tracked keypoint - True means successfully tracked
        status = [False] * num_keypoints
        # status는 False로 초기화 했다가 잠시 후 유효한 keypoint에 대해서 True로 변경

        # If at least one keypoint is active
        if num_keypoints > 0:
            # Prepare data for opencv:
            # Add singleton dimension
            # Use only first and second column
            # Make sure dtype is float32
            pts = keypoints[:, None, :2].astype(np.float32)
            # keypoints: (m, 3)을 pts: (m, 1, 2) 형태로 변경

            # Calculate forward optical flow for prev_location
            nextPts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, pts, None)
            # nextPts: (m, 1, 2), status: (m, 1)

            # Calculate backward optical flow for prev_location
            pts_back, _, _ = cv2.calcOpticalFlowPyrLK(current_gray, prev_gray, nextPts, None)
            # pts_back: (m, 1, 2), status: (m, 1)

            # Remove singleton dimension
            pts_back = util.squeeze_pts(pts_back)
            pts = util.squeeze_pts(pts)
            nextPts = util.squeeze_pts(nextPts)
            # pts_back, pts, nextPts: (m, 1, 2) => (m, 2)
            status = status.squeeze()
            # status: (m, 1) => (m,)

            # Calculate forward-backward error
            fb_err = np.sqrt(np.power(pts_back - pts, 2).sum(axis=1))
            # fb_err: (m, ) => [ 0.00053231  0.00017795  0.00036748  0.0015567, ... ]

            # Set status depending on fb_err and lk error
            large_fb = fb_err > THR_FB
            # 기준 초과하는 것을 골라내어: [True, False, Flase, True, False, ...]
            # THR_FB 값을 크게 잡으면 멀리 벗어난 keypoints도 모두 수용하는 효과

            status = ~large_fb & status.astype(np.bool)
            # 기준 초과하지 않으면서 status가 True것을 상대로 새롭게 status를 구성
            # status.astype(np.bool): [1, 1, 2] => [True, True, True]

            nextPts = nextPts[status, :]
            # 유효한 status만을 상대로 nextPts 추려냄: (m', 1, 2)
            keypoints_tracked = keypoints[status, :]
            # keypoints: (m, 3)으로 부터 keypoints_tracked: (m', 3)을 만들고,
            keypoints_tracked[:, :2] = nextPts
            # keypoints_tracked의 class label은 그대로 유지하면서 nextPts의 값으로 이전 (active) keypoints 값을 대치
        else:
            keypoints_tracked = np.array([])

        return keypoints_tracked, status

    def estimate(self, keypoints):
        # (tracked) keypoints: (m', 3)

        center = np.array((np.nan, np.nan))
        scale_estimate = np.nan
        med_rot = np.nan

        # At least 2 keypoints are needed for scale
        if len(keypoints) > 1:
            # Extract the keypoint classes
            keypoint_classes = keypoints[:, 2].squeeze().astype(np.int)
            # keypoints(m', 3)에서 마지막 column 값을 추출 => keypoint_classes: (m', ) =>  [1,3,4,5,8,9,...]
            # print("[CMT]", keypoint_classes.shape, keypoint_classes)

            # Retain singular dimension
            if keypoint_classes.size == 1:
                keypoint_classes = keypoint_classes[None]

            # Sort
            ind_sort = np.argsort(keypoint_classes)
            # np_argsort(): sorting 하기 위한 index를 리턴: np.argsort([1,4,3,2]) => [0, 3, 2, 1], np.argsort([1,3,4,7]) => [0, 1, 2, 3]
    		# ind_sort: (m', )

            keypoints = keypoints[ind_sort]
            keypoint_classes = keypoint_classes[ind_sort]
            # OpticalFlow를 통과한 keypoints(+ keypoint classes)의 순서가 바뀌었을 가능성이 있기 때문에 class label 순서대로 재정렬하는 과정

            # Get all combinations of keypoints
            all_combs = np.array([val for val in itertools.product(range(keypoints.shape[0]), repeat=2)])
            # all_combs: (m'^2, 2) => [[0,0], [0,1], [0,2], [0,m'], [1,0], [1,1],...[m',0], [m',1],..., [m',m']]

            # But exclude comparison with itself
            all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]
            # all_combs: (m'^2 - m', 2)

            # Measure distance between allcombs[0] and allcombs[1]
            ind1 = all_combs[:, 0]
            ind2 = all_combs[:, 1]
            # print(class_ind1.shape, class_ind1, sep = " => ")
            # print(class_ind2.shape, class_ind2, sep = " => ")
            # m'가 5 라고 가정하면,
            # ind1: (20,) => [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]
            # ind2: (20,) => [1 2 3 4 0 2 3 4 0 1 3 4 0 1 2 4 0 1 2 3]

            class_ind1 = keypoint_classes[ind1] - 1
            class_ind2 = keypoint_classes[ind2] - 1
            # class_ind1: (20,) => [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]
            # class_ind1: (20,) => [1 2 3 4 0 2 3 4 0 1 3 4 0 1 2 4 0 1 2 3]

            duplicate_classes = class_ind1 == class_ind2

            # all True인 경우를 제외하고는 if문 수행 (정상적인 경우란면 duplicate_classes는 모두 False가 되어야 맞지 않음?)
            if not all(duplicate_classes):
                ind1 = ind1[~duplicate_classes]
                ind2 = ind2[~duplicate_classes]

                class_ind1 = class_ind1[~duplicate_classes]
                class_ind2 = class_ind2[~duplicate_classes]
                # 혹시 있을 duplicate_classes를 제거하기 위한 조치

                pts_allcombs0 = keypoints[ind1, :2]
                pts_allcombs1 = keypoints[ind2, :2]
                # keypoint간 상호 거리 계산을 위해 배열 구성 (자기 자신과의 거리는 배체하기 위해 ind1, ind2가 필요)

                # This distance might be 0 for some combinations,
                # as it can happen that there is more than one keypoint at a single location
                dists = util.L2norm(pts_allcombs0 - pts_allcombs1)
                # dist: (m', )
                # 앞에서(init()) 했듯이 dists = scipy.spatial.distance.pdist(keypoints)를 하고,
                # 여기에 squareform = scipy.spatial.distance.squareform(dists) 하는 방식도 있으나
                # 현재 keypoints.shape은 (m', 3) 이고, 앞에서의 keypoints.shape는 (m', 2) 이므로 추가 변환 과정이 필요하기 때문에 현 방식을 사용

                original_dists = self.squareform[class_ind1, class_ind2]
                # 이전 프레임에서 계산한 self_squareform에서 자기 자신과의 거리를 배제한 정보를 추출하여,
                # print(np.isfinite(original_dists).all())

                scalechange = dists / original_dists
                # 현 프레임의 dists와 비교: 1이면 스케일 변화 없음, 1보다 크면 커짐, 1보다 작으면 작아짐(멀어짐)
                # scalechange: (m', ) => [1, 1, 1, 1,...]

                # # Compute angles
                # angles = np.empty((pts_allcombs0.shape[0]))
                #
                # v = pts_allcombs1 - pts_allcombs0
                # angles = np.arctan2(v[:, 1], v[:, 0])
                # # keypoints간 모든 angle을 계산해서,
                #
                # original_angles = self.angles[class_ind1, class_ind2]
                # angle_diffs = angles - original_angles
                # # 이전 프레임과의 앵글을 비교: 0이면 회전 없음...
                #
                # # # Fix long way angles
                # long_way_angles = np.abs(angle_diffs) > np.math.pi
                # angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(angle_diffs[long_way_angles]) * 2 * np.math.pi
                # # long way angle 교정 (180도보다 크면 360도에서 부호를 바꾸어 계산)

                scale_estimate = np.median(scalechange)
                # 스케일 평균값 계산
                if not self.estimate_scale:
                    scale_estimate = 1;

                # med_rot = np.median(angle_diffs)
                # if not self.estimate_rotation:
                med_rot = 0;

                keypoint_class = keypoints[:, 2].astype(np.int)
                # keypint_class는 현 프레임의 (tracked) keypoints에 대한 class label로써 이전 프레임의 active keypoints와는 다를 수 있다

                votes = keypoints[:, :2] - scale_estimate * (util.rotate(self.springs[keypoint_class - 1], med_rot))
                # votes = keypoints[:, :2] - scale_estimate * self.springs[keypoint_class - 1]
                # keypoints[:, :2]는 현재 프레임의 keypoints
                # print('Current Kepoints', keypoints[:, :2], sep = ':')
                # 위 식에서 sacle_estimate를 1, med_rot를 0으로 가정하면, current_springs = 1 *  rotate(self.springs[keypoint_class - 1], 0)
                # 이 되는데 이는 이전 springs중 현 keypoints 해당하는 springs를 나타냄
                # current_springs = 1 *  rotate(self.springs[keypoint_class - 1], 0) # for testing
                # votes는 현 keypoints과 spring과의 차이로 정의되는데 물리적으로 보면 center에서 spring만큼 멀어지는 방향의 값을 가짐
                # print('Votes', votes, sep=":")

                # Remember all votes including outliers
                self.votes = votes

                # Compute pairwise distance between votes
                pdist = scipy.spatial.distance.pdist(votes)
                # votes간의 상호 거리 계산
                # print('Distance between votes:', pdist, sep = "\n")

                # Compute linkage between pairwise distances
                linkage = scipy.cluster.hierarchy.linkage(pdist)
                # Compute linkage between pairwise distances
                # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
                # print("Linkage:", linkage, sep = "\n")

                # Perform hierarchical distance-based clustering
                T = scipy.cluster.hierarchy.fcluster(linkage, self.THR_OUTLIER, criterion='distance')
                # print("Clustering:", T, sep = "\n")

                # Count votes for each cluster
                cnt = np.bincount(T)  # Dummy 0 label remains
                # print("Count votes for each cluster:", cnt, sep = "\n")

                # Get largest class
                Cmax = np.argmax(cnt)
                # print("Get largest class:", Cmax, sep = "\n")

                # Identify inliers (=members of largest class)
                inliers = T == Cmax
                # inliers = med_dists < THR_OUTLIER
                # print("Inliers:", inliers, sep = "\n")

                # Remember outliers
                self.outliers = keypoints[~inliers, :]

                # Stop tracking outliers
                keypoints = keypoints[inliers, :]

                # Remove outlier votes
                votes = votes[inliers, :]

                # Compute object center
                center = np.mean(votes, axis=0)
            # else:
            #     print("[CMT] All are duplicate.")
        # else:
        #     print("[CMT] At least 2 Keypoints are required for estimation.")

        return (center, scale_estimate, med_rot, keypoints)

    def init(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.gray0 = np.copy(gray)
        self.x01 = self.x1
        self.y01 = self.y1
        self.x02 = self.x2
        self.y02 = self.y2

        #
        # mask = np.zeros(self.gray0.shape[:2], dtype=np.uint8)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x2, self.y2), 255, -1)
        # mask = cv2.bitwise_and(self.gray0, self.gray0, mask=mask)
        # cv2.imshow('Mask', mask)
        # self.kp1, self.desc1 = self.detector.detectAndCompute(gray, mask)
        #

        self.gray0 = gray[self.y1:self.y2, self.x1:self.x2]
        self.x01 = 0
        self.y01 = 0
        self.x02 = self.x2 - self.x1
        self.y02 = self.y2 - self.y1
        cv2.imshow('Template',  self.gray0)
        self.kp1, self.desc1 = self.detector.detectAndCompute(self.gray0, None)

        # print(len(self.kp1))
        self.force_init_flag = False

    def filter_matches(self, kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, list(kp_pairs)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp2, self.desc2 = self.detector.detectAndCompute(gray, None)
        # print(len(self.kp2))

        raw_matches = self.matcher.knnMatch(self.desc1, trainDescriptors = self.desc2, k = 2)
        p1, p2, kp_pairs = self.filter_matches(self.kp1, self.kp2, raw_matches)

        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))

        # vis = explore_match(win, img1, img2, kp_pairs, status, H)
        # def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
        h1, w1 = self.gray0.shape[:2]
        h2, w2 = gray.shape[:2]

        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] = self.gray0
        vis[:h2, w1:w1+w2] = gray
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if H is not None:
            corners = np.float32([[self.x01, self.y01], [self.x02, self.y01], [self.x02, self.y02], [self.x01, self.y02]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = np.ones(len(kp_pairs), np.bool_)
        p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
        for kpp in kp_pairs:
            p1.append(np.int32(kpp[0].pt))
            p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)

        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

        # vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

        cv2.imshow('Keypoints', vis)

    def init_template_matching(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[self.y1:self.y2, self.x1:self.x2]
        # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x2, self.y2), 255, -1)
        # self.template = cv2.bitwise_and(gray, gray, mask=mask)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        cv2.imshow('Mask', self.template)

    def update_template_matching(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.tl = max_loc
        self.br = (self.tl[0]+self.w, self.tl[1]+self.h)
        print(self.tl, self.br)
