# %%
import cv2
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import glob
import skimage
from skimage import measure
import string 
# %%

############ UNCOMMENT FOR RECTIFICATION OF THE IMAGES ############
# nb_vertical = 6
# nb_horizontal = 9

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
# objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpointsL = [] # 2d points in image plane.
# imgpointsR = [] # 2d points in image plane.

# images_l = glob.glob('/home/alex/Alex/DTU/Courses/perception/project/Stereo_calibration_images/left-*.png')
# images_r = glob.glob('/home/alex/Alex/DTU/Courses/perception/project/Stereo_calibration_images/right-*.png')
# left_count = len(images_l)
# images_r.sort()
# images_l.sort()
# print('# of left images:', left_count)
# right_count = len(images_r)
# print('# of left images:', right_count)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# for i in range(left_count):
#     imgL = cv2.imread(images_l[i])
#     imgR = cv2.imread(images_r[i])
#     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#     outputL = imgL.copy()
#     outputR = imgR.copy()

#     retR, cornersR =  cv2.findChessboardCorners(outputR,(nb_vertical,nb_horizontal),None)
#     retL, cornersL = cv2.findChessboardCorners(outputL,(nb_vertical,nb_horizontal),None)

#     if retR and retL:
#         objpoints.append(objp)
#         imgpointsL.append(cornersL)
#         imgpointsR.append(cornersR)
        
#         cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
#         cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
#         cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
#         cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
#         cv2.imshow('cornersR',outputR)
#         cv2.imshow('cornersL',outputL)
#         cv2.waitKey(150)
        
# cv2.destroyAllWindows()

# # calibrating left and right camera
# retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
# retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# flags = 0
# #flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# #flags |= cv2.CALIB_FIX_K1,...,cv2.CALIB_FIX_K6
# flags |= cv2.CALIB_RATIONAL_MODEL
# criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# # This step is performed to calculate the transformation between the two cameras and calculate Essential and Fundamenatl matrix
# ret, newcameramtxL, distL, newcameramtxR, distR, R, T, E, F = \
#     cv2.stereoCalibrate(objpoints, 
#                         imgpointsL, imgpointsR, 
#                         mtxL, distL, mtxR, distR, 
#                         grayL.shape,
#                         criteria=criteria_stereo,
#                         flags=flags)

# # Once we know the transformation between the two cameras we can perform stereo rectification
# R1, R2, P1, P2, Q, roi_left, roi_right = \
#     cv2.stereoRectify(newcameramtxL, distL, 
#                     newcameramtxR, distR,
#                     grayL.shape[::-1], 
#                     R, T, 
#                     flags = cv2.CALIB_ZERO_DISPARITY,
#                     alpha = -1)

# # Compute the mapping required to obtain the undistorted rectified stereo image pair
# xmapL, ymapL = cv2.initUndistortRectifyMap(newcameramtxL, distL, 
#                                         R1, P1, 
#                                         grayL.shape[::-1], 
#                                         cv2.CV_16SC2)
# xmapR, ymapR = cv2.initUndistortRectifyMap(newcameramtxR, distR, 
#                                         R2, P2, 
#                                         grayR.shape[::-1], 
#                                         cv2.CV_16SC2)
# imgL = cv2.imread('/home/alex/Alex/DTU/Courses/perception/project/Stereo_calibration_images/left-0033.png')
# imgR = cv2.imread('/home/alex/Alex/DTU/Courses/perception/project/Stereo_calibration_images/right-0033.png')

# rectL = cv2.remap(imgL, 
#                 xmapL, ymapL, 
#                 cv2.INTER_LANCZOS4, 
#                 cv2.BORDER_CONSTANT, 
#                 0)
# rectR = cv2.remap(imgR, 
#                 xmapR, ymapR, 
#                 cv2.INTER_LANCZOS4, 
#                 cv2.BORDER_CONSTANT, 
#                 0)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
# ax[0].imshow(rectL, cmap='gray')
# ax[0].set_title('Left image')
# ax[1].imshow(rectR, cmap='gray')
# ax[1].set_title('Right image')

# dist_occ_path = '/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_with_occlusions/'
# dist_path = '/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_without_occlusions/'
# dist_list = [dist_occ_path, dist_path]

# for j in range(2):
#     path = dist_list[j]
    
#     templ = path + 'left/*.png'
#     tempr = path + 'right/*.png'
#     images_l = glob.glob(templ)
#     print(images_l)
#     images_l.sort()
#     images_r = glob.glob(tempr)
#     images_r.sort()
#     for i in range(len(images_r)):
#         imgL = cv2.imread(images_l[i])
#         imgR = cv2.imread(images_r[i])

#         rectL = cv2.remap(imgL, 
#                         xmapL, ymapL, 
#                         cv2.INTER_LANCZOS4, 
#                         cv2.BORDER_CONSTANT, 
#                         0)
#         rectR = cv2.remap(imgR, 
#                         xmapR, ymapR, 
#                         cv2.INTER_LANCZOS4, 
#                         cv2.BORDER_CONSTANT, 
#                         0)
        
#         if j == 0 :
#             str_prefix = 'with_'
#         else:
#             str_prefix = 'without_'
        
#         # rectL = cv2.resize(rectL, (256, 256))
#         rectL_str = str(i).zfill(6)+'.png'
#         path_to_save = path + 'left_undist/left_' + str_prefix + rectL_str
#         saved=cv2.imwrite(path_to_save, rectL)
#         print(i,j)
#         print(saved)

#         # rectR = cv2.resize(rectR, (256, 256))

#         rectR_str = str(i).zfill(6)+'.png'
#         path_to_save = path + 'right_undist/right_' + str_prefix + rectR_str
#         cv2.imwrite(path_to_save, rectR)
#         print(i,j)
#         print(saved)

# %%

img_baseline = cv2.imread('/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_with_occlusions/right_undist/right_with_000000.png')
img_baseline_rgb = cv2.cvtColor(img_baseline, cv2.COLOR_BGR2RGB)
img_baseline_hsv = cv2.cvtColor(img_baseline, cv2.COLOR_BGR2HSV)
img_baseline_gray = cv2.cvtColor(img_baseline_rgb, cv2.COLOR_RGB2GRAY)
images = glob.glob('/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_with_occlusions/right_undist/*.png')
images.sort()
list_of_images_with_objects =[]
list_of_images_with_bb =[]
list_of_objects_screenshots = []
Q= np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -6.45245682e+02],
       [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        -3.86842728e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         7.01901305e+02],
       [ 0.00000000e+00,  0.00000000e+00,  2.79959322e-01,
        -0.00000000e+00]])
cnt_in = 0
cnt_out = 0
cnt_obj = 0
klmn_cnt = 0
new_object = False
spf = 0.03317
object_on_conveyor = 0
# img_baseline_conveyor = img_baseline[250:700,400:1150]
# img_baseline_conveyor_hsv = cv2.cvtColor(img_baseline_conveyor, cv2.COLOR_BGR2HSV)
# img_baseline_conveyor_gray = cv2.cvtColor(img_baseline_conveyor,cv2.COLOR_BGR2GRAY)
object_spawn_region = img_baseline_gray[300:350,1025:1125]
object_disap_region = img_baseline_gray[480:700,200:325]
# object_spawn_region_screenshot = img_baseline[250:400,975:1150]

def update(x, P, Z, H, R):
    ### Insert update function
    I_in = np.identity(np.shape(P)[0])
    y = Z - H @ x
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.pinv(S)
    x_update = x + K @ y
    P_update = (I_in - K @ H) @ P
    return x_update, P_update

def predict(x, P, F, u):
    ### insert predict function
    x_predict = F @ x + u
    P_predict = F @ P @ np.transpose(F)
    return x_predict, P_predict
    
    
# Kalman param.
X = np.array([[0], # Position along the x-axis
              [0], # Velocity along the x-axis
              [0], # Acc along the x-axis
              [0], # Position along the y-axis
              [0], # Velocity along the y-axis
              [0]])# Acc along the y-axis
P = np.identity(6)*200000
u = np.array([[0],[0],[0],[0],[0],[0]])
F = np.array(  [[1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]])
H = np.array([[1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0]])
R = np.array([[1],[1]])
I = np.identity(6)
def reset_kalman_depth():

    X_depth = np.array([[0], # Position along the x-axis
                        [0]])# vel along the y-axis
    P_depth = np.identity(2)*200000
    u_depth = np.array([[0],[0]])
    F_depth = np.array(  [[1, 1],
                        [0, 1]])
    H_depth = np.array([[1, 0]])
    R_depth = np.array([[1]])
    I_depth = np.identity(2)
    return X_depth, P_depth, u_depth, F_depth, H_depth, R_depth, I_depth
def reset_kalman_pos():
    
    X_x = np.array([[0], # Position along the x-axis
                        [0]])# vel along the y-axis
    P_x = np.identity(2)*20000
    u_x = np.array([[0],[0]])
    F_x = np.array(  [[1.012, 1],
                        [0, 1]])
    H_x = np.array([[1, 0]])
    R_x = np.array([[1]])
    I_x = np.identity(2)
    return X_x, P_x, u_x, F_x, H_x, R_x, I_x

# X_y = np.array([[0], # Position along the x-axis
#                     [0]])# vel along the y-axis
# P_y = np.identity(2)*20000
# u_y = np.array([[0],[0]])
# F_y = np.array(  [[1.012, 100],
#                       [0, 1]])
# H_y = np.array([[1, 0]])
# R_y = np.array([[1]])
# I_y = np.identity(2)


for image in images:
    img_frame = cv2.imread(image)
    img_frame_vis = img_frame.copy()
    img_frame_vis = cv2.cvtColor(img_frame_vis, cv2.COLOR_BGR2RGB)
    img_frame_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    img_frame_rgb = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    left_path = '/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_with_occlusions/left_undist/'
    img_number = image[103:]
    left_path_to_load = []
    left_path_to_load = str(left_path)+'left_with_'+str(img_number)
    path_process_to_save = '/home/alex/Alex/DTU/Courses/perception/project/' + 'processed/occ_' + img_number


    similarity_map_input = cv2.matchTemplate(img_frame_gray, object_spawn_region, cv2.TM_CCORR_NORMED)
    min_val_in, max_val_in, min_loc_in, max_loc_in = cv2.minMaxLoc(similarity_map_input)
    top_left_in = max_loc_in
    v_int_in, v_dec_in  = divmod(top_left_in[1],1)
    h_int_in, h_dec_in  = divmod(top_left_in[0],1)
    b_box = cv2.rectangle(img_frame_gray.copy(), top_left_in, (top_left_in[0] + object_spawn_region.shape[1], top_left_in[1] + object_spawn_region.shape[0]), (255,0,0), 3)
    # plt.figure(figsize=(12,12))
    # plt.imshow(b_box)
    # print("Spawn top left:")
    # print(v_int_in,h_int_in)
    if v_int_in not in range(300-10,300+10,1) or h_int_in not in range(1025-10,1025+10,1):
        if cnt_in > 25 and object_on_conveyor == 0:
            object_on_conveyor = 1
            new_object = True
            object_area = 0
            object_occluded = 0
            bottom_right_in = (top_left_in[0] + object_spawn_region.shape[1], top_left_in[1] + object_spawn_region.shape[0])
            b_box_in = cv2.rectangle(img_frame_gray.copy(), top_left_in, bottom_right_in, (255,0,0), 3)
            object_screenshot = img_frame[250:400,975:1150]
            list_of_objects_screenshots.append(object_screenshot)
            # plt.figure(figsize=(12,12))
            # plt.imshow(object_screenshot)
            cv2.waitKey(15)
        else: 
            cnt_in=cnt_in+1
    else:
        if cnt_in > 0:
            cnt_in=cnt_in-1
    
    similarity_map_output = cv2.matchTemplate(img_frame_gray, object_disap_region, cv2.TM_CCORR_NORMED)
    min_val_out, max_val_out, min_loc_out, max_loc_out = cv2.minMaxLoc(similarity_map_output)
    top_left_out = max_loc_out
    v_int_out, v_dec_out  = divmod(top_left_out[1],1)
    h_int_out, h_dec_out  = divmod(top_left_out[0],1)
    # print("Disappear top left:")
    # print(v_int_out,h_int_out)
    if v_int_out not in range(480-10,480+10,1) or h_int_out not in range(200-10,200+10,1):
        if cnt_out > 10 and object_on_conveyor == 1:
            object_on_conveyor = 0
            new_object = False
            bottom_right_out = (top_left_out[0] + object_spawn_region.shape[1], top_left_out[1] + object_spawn_region.shape[0])
            b_box_out = cv2.rectangle(img_frame_gray.copy(), top_left_out, bottom_right_out, (255,0,0), 3)
            object_screenshot = img_frame[450:700,200:400]
            list_of_objects_screenshots.append(object_screenshot)
            # plt.figure(figsize=(12,12))
            # plt.imshow(b_box_out)
            cv2.waitKey(15)
        else: 
            cnt_out=cnt_out+1
    else:
        if cnt_out > 0:
            cnt_out=cnt_out-1
    # print(cnt_in,cnt_out)

    
    if object_on_conveyor:

        # img_frame_not_cropped_vis = img_frame.copy()
        # img_frame_not_cropped_vis = cv2.cvtColor(img_frame_not_cropped_vis, cv2.COLOR_BGR2RGB)
        # img_frame = img_frame[250:700,400:1150]
        img_frame_vis = img_frame.copy()
        img_frame_vis = cv2.cvtColor(img_frame_vis, cv2.COLOR_BGR2RGB)
        img_frame_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        # mask = np.zeros_like(img_frame)
        # # Sets image saturation to maximum
        # mask[..., 1] = 255

        #*********** HSV Segmentation *************

        img_frame_diff_hsv = cv2.absdiff(img_baseline_hsv, img_frame_hsv)
        img_frame_diff_bgr = cv2.cvtColor(img_frame_diff_hsv, cv2.COLOR_HSV2BGR)
        img_frame_diff_gray = cv2.cvtColor(img_frame_diff_bgr, cv2.COLOR_BGR2GRAY)
        # plt.figure(figsize = (20,20))
        # plt.imshow(img_frame_diff_gray, cmap='gray')
        
        hsv_th = 30
        hsv_threshold = 30
        threshold_value = 255

        hsv_imask =  img_frame_diff_gray>hsv_th
        hsv_canvas = np.zeros_like(img_frame_gray, np.uint8)
        hsv_canvas[hsv_imask] = img_frame_gray[hsv_imask]
        hsv_thresh = cv2.threshold(hsv_canvas, hsv_threshold, threshold_value, cv2.THRESH_BINARY)[1]
        # plt.figure(figsize = (5,5))
        # plt.imshow(hsv_thresh, cmap='gray')
        hsv_eroded = cv2.erode(hsv_thresh, None, iterations = 2)
        # plt.figure(figsize = (5,5))
        # plt.imshow(hsv_eroded, cmap='gray')
        hsv_dilated = cv2.dilate(hsv_eroded, None, iterations = 2)
        # plt.figure(figsize = (5,5))
        # plt.imshow(hsv_dilated, cmap='gray')

        #*********** BGR Segmentation *************

        img_frame_diff_bgr = cv2.absdiff(img_baseline, img_frame)
        img_frame_diff_gray = cv2.cvtColor(img_frame_diff_bgr, cv2.COLOR_BGR2GRAY)
        # plt.figure(figsize = (20,20))
        # plt.imshow(img_frame_diff_gray, cmap='gray')

        bgr_th = 30
        bgr_threshold = 30
        threshold_value = 255
        
        bgr_imask =  img_frame_diff_gray>bgr_th
        bgr_canvas = np.zeros_like(img_frame_gray, np.uint8)
        bgr_canvas[bgr_imask] = img_frame_gray[bgr_imask]
        bgr_thresh = cv2.threshold(bgr_canvas, bgr_threshold, threshold_value, cv2.THRESH_BINARY)[1]
        # plt.figure(figsize = (5,5))
        # plt.imshow(bgr_thresh, cmap='gray')
        bgr_eroded = cv2.erode(bgr_thresh, None, iterations = 2)
        # plt.figure(figsize = (5,5))
        # plt.imshow(bgr_eroded, cmap='gray'
        bgr_dilated = cv2.dilate(bgr_eroded, None, iterations = 2)
        # plt.figure(figsize = (5,5))
        # plt.imshow(bgr_dilated, cmap='gray')

        #*********** Select Segmentation *************

        if bgr_dilated.mean() > hsv_dilated.mean():
            # print("BGR")
            final_mask = bgr_dilated > 100
            segmented_object_mask= bgr_dilated
            
        else:
            # print("HSV")
            final_mask = hsv_dilated > 100
            segmented_object_mask= hsv_dilated
        segmented_object = np.zeros_like(img_frame_gray, np.uint8)
        segmented_object[final_mask] = img_frame_gray[final_mask]
        # plt.figure(figsize = (5,5))
        # plt.imshow(segmented_object, cmap='gray')

        #*********** Blob Filtering *************

        segmented_object_labeled = measure.label(segmented_object_mask)
        regions = measure.regionprops(segmented_object_labeled)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[1:]:
                segmented_object_labeled[rg.coords[:,0], rg.coords[:,1]] = 0
        # plt.figure(figsize=(15,15))
        # plt.imshow(segmented_object_labeled, cmap='gray')
        one_blob_mask = segmented_object_labeled > 0
        segmented_object_filtered = np.zeros_like(segmented_object, dtype=np.uint8)
        segmented_object_filtered[one_blob_mask] = segmented_object[one_blob_mask]
        ret, lb_mk_thres_1= cv2.threshold(segmented_object_filtered,12,255,cv2.THRESH_BINARY)
        # plt.figure(figsize=(15,15))
        # plt.imshow(lb_mk_thres_1, cmap='gray')

        #*********** Centroid calculation *************

        M = cv2.moments(lb_mk_thres_1)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        segmented_object_filtered_contours= cv2.findContours(segmented_object_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        if len(segmented_object_filtered_contours) > 0:
            x, y, w, h = cv2.boundingRect(segmented_object_filtered_contours[0])
            if new_object:
                object_area = w*h
                X_x, P_x, u_x, F_x, H_x, R_x, I_x = reset_kalman_pos()
                X_y, P_y, u_y, F_y, H_y, R_y, I_y = reset_kalman_pos()
                X_depth, P_depth, u_depth, F_depth, H_depth, R_depth, I_depth = reset_kalman_depth()
                new_object = False
            if w*h > object_area-(object_area*0.1+object_occluded/500):
                padding_width = int(w//20)
                padding_height = int(h//20)
                cv2.circle(img_frame_vis, (cX, cY), 5, (0, 255, 0), -1)
                cv2.rectangle(img_frame_vis,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.rectangle(img_frame_vis,(x-padding_width,y-padding_height),(x+w+padding_width,y+h+padding_height),(225,0,0),2)
                # plt.figure(figsize=(15,15))
                # plt.imshow(img_frame_vis)
                ############### Disparity #############
                
                # print(left_path_to_load)
                left_img_frame = cv2.imread(left_path_to_load)
                left_img_frame_gray = cv2.cvtColor(left_img_frame, cv2.COLOR_BGR2GRAY)
                right_img_frame = cv2.imread(image)
                right_img_frame_gray = cv2.cvtColor(right_img_frame, cv2.COLOR_BGR2GRAY)

                scale_factor = 5
                img_size = (int(left_img_frame.shape[1]/scale_factor), int(left_img_frame.shape[0]/scale_factor))
                img_left = cv2.resize(left_img_frame_gray, img_size, interpolation=cv2.INTER_AREA)
                img_right = cv2.resize(right_img_frame_gray.copy(), img_size, interpolation=cv2.INTER_AREA)
                min_disp = 5
                num_disp = 4 * 16
                block_size = 11
                stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
                stereo.setMinDisparity(min_disp)
                stereo.setDisp12MaxDiff(200)
                stereo.setUniquenessRatio(5)
                stereo.setSpeckleRange(3)
                stereo.setSpeckleWindowSize(25)
                # f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
                # ax_left.imshow(img_left, cmap='gray')
                # ax_right.imshow(img_right, cmap='gray')
                disp = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
                # plt.figure(figsize=(15,15))
                # plt.imshow(disp, cmap='gray')
                depth_estimate = cv2.reprojectImageTo3D(disp, Q)
                cent_x = int((min_disp+2*num_disp+cX)/scale_factor)
                cent_y = int((cY)/scale_factor)
                if cent_x > 250 : cent_x = 250
                if cent_y > 140 : cent_y = 140
                if cent_x < 5 : cent_x = 5
                if cent_y < 5 : cent_y = 5
                # cv2.circle(disp, (cent_x, cent_y), 5, (50, 255, 255), -1)
                # plt.figure(figsize=(15,15))
                # plt.imshow(disp, cmap='gray')
                
                ########### DEPTH ESTIMATION #################

                depth_estimate_roi = depth_estimate[cent_y-3:cent_y+3,cent_x-3:cent_x+3:]
                depth_estimate_roi_mult = np.multiply(depth_estimate_roi,depth_estimate_roi)
                depth_estimate_roi_sum = np.sum(depth_estimate_roi_mult,axis=2)
                depth_estimate_roi_sqrt = np.sqrt(depth_estimate_roi_sum)
                distance = depth_estimate_roi_sqrt.min()/100
                formatted_distance = "{:.2f}".format(distance)
                # distance = sqrt(pow(depth_estimate[cent_y,cent_x,0],2)+pow(depth_estimate[cent_y,cent_x,1],2)+pow(depth_estimate[cent_y,cent_x,2],2))
                print(formatted_distance)
                distance_plot = "Dist: "+formatted_distance+" m"
                cv2.putText(img_frame_vis, distance_plot, (x+padding_width, y-padding_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                ############## KALMAN UPDATE #########

                z_x = np.array([[cX]])
                z_y = np.array([[cY]])
                z_depth = np.array([[distance]])
                X_x,P_x = update(X_x,P_x,z_x,H_x,R_x)
                X_y,P_y = update(X_y,P_y,z_y,H_y,R_y)
                X_depth,P_depth = update(X_depth,P_depth,z_depth,H_depth,R_depth)
                klmn_cnt = klmn_cnt+1
                if object_occluded>10:
                    object_occluded=object_occluded-10
                
            else:
                ########### PLOTING PREDICTION k-1 ##########
                cv2.circle(img_frame_vis, (X_x[0], X_y[0]), 5, (255, 0, 255), -1)
                # cv2.rectangle(img_frame_vis,(x,y),(x+w,y+h),(255,0,255),2)
                formatted_distance = "{:.2f}".format(float(X_depth[0]))
                distance_plot = "Dist: "+formatted_distance+" m"
                cv2.putText(img_frame_vis, distance_plot, (X_x[0]+padding_width, X_y[0]-padding_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                object_occluded = object_occluded+1
        else:
            ########### PLOTING PREDICTION k-1 ##########
            cv2.circle(img_frame_vis, (X_x[0], X_y[0]), 5, (255, 0, 255), -1)
            formatted_distance = "{:.2f}".format(float(X_depth[0]))
            distance_plot = "Dist: "+formatted_distance+" m"
            cv2.putText(img_frame_vis, distance_plot, (X_x[0]+padding_width, X_y[0]-padding_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            object_occluded = object_occluded+1
        print("Current: ", cX, cY,distance)
        print("Updated: ", np.int(X_x[0]), np.int(X_y[0]), X_depth[0])

        ########### KALMAN PREDICTION ##########

        X_x,P_x = predict(X_x,P_x,F_x,u_x)
        X_y,P_y = predict(X_y,P_y,F_y,u_y)
        X_depth,P_depth = predict(X_depth,P_depth,F_depth,u_depth)
        print("Predicted: ", np.int(X_x[0]), np.int(X_y[0]), X_depth[0])
        # cv2.circle(img_frame_vis, (np.int(X_x[0]), np.int(X_y[0])), 5, (255,0,255), -1)
        # cv2.rectangle(img_frame_vis,(cY-np.int(X[3])+x,cX-np.int(X[0])+y),(cY-np.int(X[3])+x+w,cX-np.int(X[0])+y+h),(255,0,255),2)
        # if object_occluded>80:
        #     plt.figure(figsize=(15,15))
        #     plt.imshow(img_frame_vis)
        # path_process_to_save = '/home/alex/Alex/DTU/Courses/perception/project/' + 'processed/occ_' + img_number
        cv2.imwrite(path_process_to_save, cv2.cvtColor(img_frame_vis, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path_process_to_save, cv2.cvtColor(img_frame_vis, cv2.COLOR_RGB2BGR))







# # %%
#             ############### Disparity #############
#             left_path = '/home/alex/Alex/DTU/Courses/perception/project/Stereo_conveyor_without_occlusions/left_undist/'
#             img_number = image[109:]
#             left_path_to_load = []
#             left_path_to_load = str(left_path)+'left_without_'+str(img_number)
#             # print(left_path_to_load)
#             left_img_frame = cv2.imread(left_path_to_load)
#             left_img_frame_gray = cv2.cvtColor(left_img_frame, cv2.COLOR_BGR2GRAY)
#             right_img_frame = cv2.imread(image)
#             right_img_frame_gray = cv2.cvtColor(right_img_frame, cv2.COLOR_BGR2GRAY)

#             img_size = (int(left_img_frame.shape[1]/4), int(left_img_frame.shape[0]/4))
#             img_left = cv2.resize(left_img_frame_gray, img_size, interpolation=cv2.INTER_AREA)
#             img_right = cv2.resize(right_img_frame_gray.copy(), img_size, interpolation=cv2.INTER_AREA)
#             min_disp = 5
#             num_disp = 6 * 16
#             block_size = 9
#             stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
#             stereo.setMinDisparity(min_disp)
#             stereo.setDisp12MaxDiff(200)
#             stereo.setUniquenessRatio(5)
#             stereo.setSpeckleRange(3)
#             stereo.setSpeckleWindowSize(25)
#             # f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
#             # ax_left.imshow(img_left, cmap='gray')
#             # ax_right.imshow(img_right, cmap='gray')
#             disp = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
#             # plt.figure(figsize=(15,15))
#             # plt.imshow(disp, cmap='gray')
#             depth_estimate = cv2.reprojectImageTo3D(disp, Q)
#             cent_x = int((400+cX)/4)
#             cent_y = int((250+cY)/4)
#             # cv2.circle(disp, (centroid_downsampled_x, centroid_downsampled_y), 5, (50, 255, 255), -1)
#             # plt.figure(figsize=(15,15))
#             # plt.imshow(disp, cmap='gray')
#             depth_estimate_roi = depth_estimate[cent_y-3:cent_y+3,cent_x-3:cent_x+3:]
#             depth_estimate_roi_mult = np.multiply(depth_estimate_roi,depth_estimate_roi)
#             depth_estimate_roi_sum = np.sum(depth_estimate_roi_mult,axis=2)
#             depth_estimate_roi_sqrt = np.sqrt(depth_estimate_roi_sum)
#             distance = depth_estimate_roi_sqrt.min()/100
#             formatted_distance = "{:.2f}".format(distance)
#             # distance = sqrt(pow(depth_estimate[cent_y,cent_x,0],2)+pow(depth_estimate[cent_y,cent_x,1],2)+pow(depth_estimate[cent_y,cent_x,2],2))
#             print(formatted_distance)
#             distance_plot = "Dist: "+formatted_distance+" m"
#             # plt.figure(figsize=(15,15))
#             # plt.imshow(img_frame_vis)
#             cv2.putText(img_frame_not_cropped_vis, distance_plot, (400+x+padding_width, 250+y-padding_height), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.rectangle(img_frame_not_cropped_vis,(400+x,250+y),(400+x+w,250+y+h),(0,255,0),2)
#             cv2.circle(img_frame_not_cropped_vis, (400+cX, 250+cY), 5, (0, 255, 0), -1)
#             # plt.figure(figsize=(15,15))
#             # plt.imshow(img_frame_not_cropped_vis)


#             ###### SAVE #######

#             # path_process_to_save = '/home/alex/Alex/DTU/Courses/perception/project/' + 'processed/frame_' + img_number
            
#             # cv2.imwrite(path_process_to_save, cv2.cvtColor(img_frame_not_cropped_vis, cv2.COLOR_RGB2BGR))

    



# # print(list_of_images_with_objects)
# # print(list_of_objects_screenshots)
# cv2.destroyAllWindows()

# # %%

# # img = cv2.imread('1585434283_721692085_Left.png')
# # img_left = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img_baseline_gray = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
# # img_baseline_gray = img_baseline_gray[250:650,400:1200]

# # obj1_roi = cv2.imread('obj1_roi.png')
# # obj1_color = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2RGB)
# # gray_obj1_roi = cv2.cvtColor(obj1_color, cv2.COLOR_RGB2GRAY)
# i=0
# for i in range(len(list_of_images_with_objects)):
#     img_frame_cp = img_frame
#     obj1_roi = list_of_objects_screenshots[i]
#     obj1_roi_cp = obj1_roi
#     obj1_roi_hsv = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2HSV)
#     object_spawn_region_screenshot_hsv = cv2.cvtColor(object_spawn_region_screenshot, cv2.COLOR_BGR2HSV)
#     gray_obj1_roi = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2GRAY)

#     plt.figure(figsize = (5,5))
#     plt.imshow(gray_obj1_roi, cmap = 'gray')

#     ####### HSV DIFF #########
#     hsv_mask = cv2.absdiff(object_spawn_region_screenshot_hsv, obj1_roi_hsv)
#     hsv_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
#     hsv_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_BGR2GRAY)

#     # mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_mask, cmap = 'gray')

#     hsv_th = 30
#     hsv_imask =  hsv_mask>hsv_th

#     hsv_canvas = np.zeros_like(gray_obj1_roi, np.uint8)
#     hsv_canvas[hsv_imask] = gray_obj1_roi[hsv_imask]

#     hsv_threshold = 30
#     threshold_value = 255

#     hsv_thresh = cv2.threshold(hsv_canvas, hsv_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_thresh, cmap='gray')

#     hsv_eroded = cv2.erode(hsv_thresh, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_eroded, cmap='gray')

#     hsv_dilated = cv2.dilate(hsv_eroded, None, iterations = 3)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_dilated, cmap='gray')



#     ####### BGR DIFF #########
#     bgr_mask = cv2.absdiff(object_spawn_region_screenshot, obj1_roi)
#     bgr_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2GRAY)
#     plt.figure(figsize = (5,5))
#     plt.imshow(bgr_mask, cmap='gray')

#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_mask, cmap = 'gray')

#     bgr_th = 28
#     bgr_imask =  bgr_mask>bgr_th

#     bgr_canvas = np.zeros_like(gray_obj1_roi, np.uint8)
#     bgr_canvas[bgr_imask] = gray_obj1_roi[bgr_imask]

#     bgr_threshold = 50
#     threshold_value = 255

#     bgr_thresh = cv2.threshold(bgr_canvas, bgr_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_thresh, cmap='gray')

#     bgr_eroded = cv2.erode(bgr_thresh, None, iterations = 3)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_eroded, cmap='gray')

#     bgr_dilated = cv2.dilate(bgr_eroded, None, iterations = 5)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_dilated, cmap='gray')
#     # to_process = np.zeros_like(gray_obj1_roi, np.uint8)
#     # to_process[dilated] = gray_obj1_roi[dilated]
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(to_process, cmap='gray')

#     if bgr_dilated.mean() > hsv_dilated.mean():
#         print("BGR")
#         final_mask = bgr_dilated > 100
        
#     else:
#         print("HSV")
#         final_mask = hsv_dilated > 100
#     segmented_object = np.zeros_like(gray_obj1_roi, np.uint8)
#     segmented_object[final_mask] = gray_obj1_roi[final_mask]
#     plt.figure(figsize = (5,5))
#     plt.imshow(segmented_object, cmap='gray')


    


#     # img_frame_cp = img_frame
#     # obj1_roi = list_of_objects_screenshots[i]
#     # obj1_roi_cp = obj1_roi
#     # obj1_roi_hsv = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2HSV)
#     # object_spawn_region_screenshot_hsv = cv2.cvtColor(object_spawn_region_screenshot, cv2.COLOR_BGR2HSV)
#     # gray_obj1_roi = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2GRAY)
#     img_frame_to_find = list_of_images_with_objects[i]
#     img_frame_to_find = img_frame_to_find[250:700,400:1150]
#     img_frame_to_find_cp = img_frame_to_find

#     # surf = cv2.xfeatures2d.SURF_create() 
#     sift = cv2.xfeatures2d.SIFT_create()

#     # kp, des = surf.detectAndCompute(img_frame_to_find, None) #Keypoints and descriptors
#     # kp2, des2 = surf.detectAndCompute(segmented_object, None)

#     kp, des = sift.detectAndCompute(img_frame_to_find, None) #Keypoints and descriptors
#     kp2, des2 = sift.detectAndCompute(segmented_object, None)

#     kp_base = cv2.drawKeypoints(img_frame_to_find_cp, kp, None)
#     plt.figure(figsize = (10,10))
#     plt.imshow(kp_base)
    
#     kp_frame = cv2.drawKeypoints(segmented_object, kp2, None)
#     plt.figure(figsize = (10,10))
#     plt.imshow(kp_frame)

#     bf = cv2.BFMatcher()
#     matches = bf.match(des, des2)

#     matches = sorted(matches, key = lambda x:x.distance)

#     img3 = cv2.drawMatches(img_frame_to_find,kp,gray_obj1_roi,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     plt.figure(figsize = (20,20))
#     plt.imshow(img3)

# # %%

# i=0
# for i in range(len(list_of_images_with_objects)):
#     img_frame_cp = img_frame
#     obj1_roi = list_of_objects_screenshots[i]
#     obj1_roi_cp = obj1_roi
#     obj1_roi_hsv = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2HSV)
#     object_spawn_region_screenshot_hsv = cv2.cvtColor(object_spawn_region_screenshot, cv2.COLOR_BGR2HSV)
#     gray_obj1_roi = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2GRAY)

#     plt.figure(figsize = (5,5))
#     plt.imshow(gray_obj1_roi, cmap = 'gray')

#     ####### HSV DIFF #########
#     hsv_mask = cv2.absdiff(object_spawn_region_screenshot_hsv, obj1_roi_hsv)
#     hsv_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
#     hsv_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_BGR2GRAY)

#     # mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_mask, cmap = 'gray')

#     hsv_th = 30
#     hsv_imask =  hsv_mask>hsv_th

#     hsv_canvas = np.zeros_like(gray_obj1_roi, np.uint8)
#     hsv_canvas[hsv_imask] = gray_obj1_roi[hsv_imask]

#     hsv_threshold = 30
#     threshold_value = 255

#     hsv_thresh = cv2.threshold(hsv_canvas, hsv_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_thresh, cmap='gray')

#     hsv_eroded = cv2.erode(hsv_thresh, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_eroded, cmap='gray')

#     hsv_dilated = cv2.dilate(hsv_eroded, None, iterations = 3)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_dilated, cmap='gray')

#     ####### BGR DIFF #########
#     bgr_mask = cv2.absdiff(object_spawn_region_screenshot, obj1_roi)
#     bgr_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2GRAY)

#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_mask, cmap = 'gray')

#     bgr_th = 28
#     bgr_imask =  bgr_mask>bgr_th

#     bgr_canvas = np.zeros_like(gray_obj1_roi, np.uint8)
#     bgr_canvas[bgr_imask] = gray_obj1_roi[bgr_imask]

#     bgr_threshold = 50
#     threshold_value = 255

#     bgr_thresh = cv2.threshold(bgr_canvas, bgr_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_thresh, cmap='gray')

#     bgr_eroded = cv2.erode(bgr_thresh, None, iterations = 3)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_eroded, cmap='gray')

#     bgr_dilated = cv2.dilate(bgr_eroded, None, iterations = 5)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_dilated, cmap='gray')
#     # to_process = np.zeros_like(gray_obj1_roi, np.uint8)
#     # to_process[dilated] = gray_obj1_roi[dilated]
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(to_process, cmap='gray')

#     if bgr_dilated.mean() > hsv_dilated.mean():
#         print("BGR")
#         final_mask = bgr_dilated > 100
        
#     else:
#         print("HSV")
#         final_mask = hsv_dilated > 100
#     segmented_object = np.zeros_like(gray_obj1_roi, np.uint8)
#     segmented_object[final_mask] = gray_obj1_roi[final_mask]
#     plt.figure(figsize = (5,5))
#     plt.imshow(segmented_object, cmap='gray')

#     # img_frame_cp = img_frame
#     # obj1_roi = list_of_objects_screenshots[i]
#     # obj1_roi_cp = obj1_roi
#     # obj1_roi_hsv = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2HSV)
#     # object_spawn_region_screenshot_hsv = cv2.cvtColor(object_spawn_region_screenshot, cv2.COLOR_BGR2HSV)
#     # gray_obj1_roi = cv2.cvtColor(obj1_roi, cv2.COLOR_BGR2GRAY)
#     img_frame_to_find = list_of_images_with_objects[i]
#     img_frame_to_find = img_frame_to_find[250:700,400:1150]
#     img_frame_to_find_cp = img_frame_to_find

#     #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
#     minHessian = 400
#     detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
#     keypoints1, descriptors1 = detector.detectAndCompute(img_frame_to_find, None)
#     keypoints2, descriptors2 = detector.detectAndCompute(segmented_object, None)
#     #-- Step 2: Matching descriptor vectors with a FLANN based matcher
#     # Since SURF is a floating-point descriptor NORM_L2 is used
#     matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
#     knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
#     #-- Filter matches using the Lowe's ratio test
#     ratio_thresh = 0.6
#     good_matches = []
#     for m,n in knn_matches:
#         if m.distance < ratio_thresh * n.distance:
#             good_matches.append(m)
#     #-- Draw matches
#     img_matches = np.empty((max(img_frame_to_find.shape[0], segmented_object.shape[0]), img_frame_to_find.shape[1]+segmented_object.shape[1], 3), dtype=np.uint8)
#     cv2.drawMatches(img_frame_to_find, keypoints1, segmented_object, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     #-- Show detected matches
#     plt.figure(figsize = (20,20))
#     plt.imshow(img_matches)

#     # # surf = cv2.xfeatures2d.SURF_create() 
#     # sift = cv2.xfeatures2d.SIFT_create()

#     # # kp, des = surf.detectAndCompute(img_frame_to_find, None) #Keypoints and descriptors
#     # # kp2, des2 = surf.detectAndCompute(segmented_object, None)

#     # kp, des = sift.detectAndCompute(img_frame_to_find, None) #Keypoints and descriptors
#     # kp2, des2 = sift.detectAndCompute(segmented_object, None)

#     # kp_base = cv2.drawKeypoints(img_frame_to_find_cp, kp, None)
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(kp_base)
    
#     # kp_frame = cv2.drawKeypoints(segmented_object, kp2, None)
#     # plt.figure(figsize = (10,10))
#     # plt.imshow(kp_frame)

#     # bf = cv2.BFMatcher()
#     # matches = bf.match(des, des2)

#     # matches = sorted(matches, key = lambda x:x.distance)

#     # img3 = cv2.drawMatches(img_frame_to_find,kp,gray_obj1_roi,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     # plt.figure(figsize = (20,20))
#     # plt.imshow(img3)


# # %% 
# cnt2 = 0
# for image in images:
#     if cnt2 > 0:
#         prev_img_frame = img_frame
#     img_frame = cv2.imread(image)
#     img_frame = img_frame[250:700,400:1150]
#     img_frame_grey = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
#     if cnt2 != 0:
#         img_frame_diff = cv2.absdiff(img_baseline_conveyor, img_frame)
#         img_frame_diff_gray = cv2.cvtColor(img_frame_diff, cv2.COLOR_BGR2GRAY)
#         plt.figure(figsize = (20,20))
#         plt.imshow(img_frame_diff_gray, cmap='gray')

#         diff_threshold = 20
#         diff_threshold_value = 255

#         img_frame_diff_threshold = cv2.threshold(img_frame_diff_gray, diff_threshold, diff_threshold_value, cv2.THRESH_BINARY)[1]
#         plt.figure(figsize = (20,20))
#         plt.imshow(img_frame_diff_threshold, cmap='gray')

#         diff_eroded = cv2.erode(img_frame_diff_threshold, None, iterations = 3)
#         plt.figure(figsize = (5,5))
#         plt.imshow(diff_eroded, cmap='gray')

#         diff_dilated = cv2.dilate(diff_eroded, None, iterations = 3)
#         plt.figure(figsize = (5,5))
#         plt.imshow(diff_dilated, cmap='gray')
#         if cnt2>1:
#             prev_diff_dilated = diff_dilated
        
#             feat1 = cv2.goodFeaturesToTrack(diff_dilated, maxCorners=50, qualityLevel=0.3, minDistance=7)
#             feat2, status, error = cv2.calcOpticalFlowPyrLK(prev_diff_dilated, diff_dilated, feat1, None)
#         for i in range(len(feat1)):
#             cv2.line(img_frame_grey, (feat1[i][0][0], feat1[i][0][1]), (feat2[i][0][0], feat2[i][0][1]), (0, 0, 0), 1)
#             cv2.circle(img_frame_grey, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 255, 0), -1)

#         plt.figure(figsize=(15,15))
#         plt.imshow(img_frame_grey)
#     cnt2=cnt2+1



# # %% 
# cnt2 = 0
# import skimage
# from skimage import measure

# for image in images[200:210]:
#     if cnt2 > 0:
#         prev_img_frame_gray = img_frame_gray
#     img_frame = cv2.imread(image)
#     img_frame = img_frame[250:700,400:1150]
#     img_baseline_conveyor_gray = cv2.cvtColor(img_baseline_conveyor,cv2.COLOR_BGR2GRAY)
#     img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
#     mask = np.zeros_like(img_frame)
  
#     # Sets image saturation to maximum
#     mask[..., 1] = 255
#     if cnt2 != 0:
        
#         flow = cv2.calcOpticalFlowFarneback(img_baseline_conveyor, img_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
#         # Sets image hue according to the optical flow 
#         # direction
#         mask[..., 0] = angle * 180 / np.pi / 2
        
#         # Sets image value according to the optical flow
#         # magnitude (normalized)
#         mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
#         # Converts HSV to RGB (BGR) color representation
#         rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
#         gray_op = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#         op_threshold = 50
#         op_threshold_value = 255

#         gray_op_thres = cv2.threshold(gray_op, op_threshold, op_threshold_value, cv2.THRESH_BINARY)[1]
        
#         gray_op_eroded = cv2.erode(gray_op_thres, None, iterations = 6)
#         plt.figure(figsize = (15,15))
#         plt.imshow(gray_op_eroded, cmap='gray')

#         gray_op_dilated = cv2.dilate(gray_op_eroded, None, iterations = 7)
#         plt.figure(figsize = (15,15))
#         plt.imshow(gray_op_dilated, cmap='gray')

#         labels_mask = measure.label(gray_op_dilated)                       
#         regions = measure.regionprops(labels_mask)
#         regions.sort(key=lambda x: x.area, reverse=True)
#         if len(regions) > 1:
#             for rg in regions[1:]:
#                 labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
#         labels_mask[labels_mask!=0] = 1
#         ret,lb_mk_thres = cv2.threshold(labels_mask,127,255,0)
#         M = cv2.moments(lb_mk_thres)
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         cv2.circle(labels_mask, (cX, cY), 5, (0, 255, 0), -1)

#         plt.figure(figsize=(15,15))
#         plt.imshow(labels_mask)

#     cnt2=cnt2+1
# # %%
# ############## APPROACH #3 ################
# import skimage
# from skimage import measure

# for image in images:

#     img_frame = cv2.imread(image)
#     img_frame = img_frame[250:700,400:1150]
#     img_frame_vis = img_frame.copy()
#     img_frame_vis = cv2.cvtColor(img_frame_vis, cv2.COLOR_BGR2RGB)
#     img_frame_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
#     img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
#     mask = np.zeros_like(img_frame)
#     # Sets image saturation to maximum
#     mask[..., 1] = 255

#     #*********** HSV Segmentation *************

#     img_frame_diff_hsv = cv2.absdiff(img_baseline_conveyor_hsv, img_frame_hsv)
#     img_frame_diff_bgr = cv2.cvtColor(img_frame_diff_hsv, cv2.COLOR_HSV2BGR)
#     img_frame_diff_gray = cv2.cvtColor(img_frame_diff_bgr, cv2.COLOR_BGR2GRAY)
#     # plt.figure(figsize = (20,20))
#     # plt.imshow(img_frame_diff_gray, cmap='gray')
    
#     hsv_th = 30
#     hsv_threshold = 30
#     threshold_value = 255

#     hsv_imask =  img_frame_diff_gray>hsv_th
#     hsv_canvas = np.zeros_like(img_frame_gray, np.uint8)
#     hsv_canvas[hsv_imask] = img_frame_gray[hsv_imask]
#     hsv_thresh = cv2.threshold(hsv_canvas, hsv_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_thresh, cmap='gray')
#     hsv_eroded = cv2.erode(hsv_thresh, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_eroded, cmap='gray')
#     hsv_dilated = cv2.dilate(hsv_eroded, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(hsv_dilated, cmap='gray')

#     #*********** BGR Segmentation *************

#     img_frame_diff_bgr = cv2.absdiff(img_baseline_conveyor, img_frame)
#     img_frame_diff_gray = cv2.cvtColor(img_frame_diff_bgr, cv2.COLOR_BGR2GRAY)
#     # plt.figure(figsize = (20,20))
#     # plt.imshow(img_frame_diff_gray, cmap='gray')

#     bgr_th = 30
#     bgr_threshold = 30
#     threshold_value = 255

#     bgr_imask =  img_frame_diff_gray>bgr_th
#     bgr_canvas = np.zeros_like(img_frame_gray, np.uint8)
#     bgr_canvas[bgr_imask] = img_frame_gray[bgr_imask]
#     bgr_thresh = cv2.threshold(bgr_canvas, bgr_threshold, threshold_value, cv2.THRESH_BINARY)[1]
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_thresh, cmap='gray')
#     bgr_eroded = cv2.erode(bgr_thresh, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_eroded, cmap='gray'
#     bgr_dilated = cv2.dilate(bgr_eroded, None, iterations = 2)
#     # plt.figure(figsize = (5,5))
#     # plt.imshow(bgr_dilated, cmap='gray')

#     #*********** Select Segmentation *************

#     if bgr_dilated.mean() > hsv_dilated.mean():
#         print("BGR")
#         final_mask = bgr_dilated > 100
#         segmented_object_mask= bgr_dilated
        
#     else:
#         print("HSV")
#         final_mask = hsv_dilated > 100
#         segmented_object_mask= hsv_dilated
#     segmented_object = np.zeros_like(img_frame_gray, np.uint8)
#     segmented_object[final_mask] = img_frame_gray[final_mask]
#     plt.figure(figsize = (5,5))
#     plt.imshow(segmented_object, cmap='gray')

#     #*********** Blob Filtering *************

#     segmented_object_labeled = measure.label(segmented_object_mask)
#     regions = measure.regionprops(segmented_object_labeled)
#     regions.sort(key=lambda x: x.area, reverse=True)
#     if len(regions) > 1:
#         for rg in regions[1:]:
#             segmented_object_labeled[rg.coords[:,0], rg.coords[:,1]] = 0
#     plt.figure(figsize=(15,15))
#     plt.imshow(segmented_object_labeled, cmap='gray')
#     one_blob_mask = segmented_object_labeled > 0
#     segmented_object_filtered = np.zeros_like(segmented_object, dtype=np.uint8)
#     segmented_object_filtered[one_blob_mask] = segmented_object[one_blob_mask]
#     ret, lb_mk_thres_1= cv2.threshold(segmented_object_filtered,12,255,cv2.THRESH_BINARY)
#     # plt.figure(figsize=(15,15))
#     # plt.imshow(lb_mk_thres_1, cmap='gray')
#     M = cv2.moments(lb_mk_thres_1)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#     else:
#         cX, cY = 0, 0
#     cv2.circle(img_frame_vis, (cX, cY), 5, (0, 255, 0), -1)
#     segmented_object_filtered_contours= cv2.findContours(segmented_object_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#     if len(segmented_object_filtered_contours) > 0:
#         cnt_obj = cnt_obj+1
#         x, y, w, h = cv2.boundingRect(segmented_object_filtered_contours[0])
#         padding_width = int(w//5)
#         padding_height = int(h//5)
#         # cv2.rectangle(img_frame_vis,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.rectangle(img_frame_vis,(x-padding_width,y-padding_height),(x+w+padding_width,y+h+padding_height),(225,0,0),2)
#         obj_roi_to_save = img_frame[y-padding_height:y+h+padding_height,x-padding_width:x+w+padding_width]
#         obj_roi_name = 'obj_'+str(cnt_obj).zfill(6)+'.png'
#         path_to_save = '/home/alex/Alex/DTU/Courses/perception/project/' + 'objects/left_' + obj_roi_name
#         cv2.imwrite(path_to_save, obj_roi_to_save)
#     plt.figure(figsize=(15,15))
#     plt.imshow(img_frame_vis)

# # %%
# # img = cv2.imread('1585434279_805531979_Left.png')
# # # convert image to grayscale image
# # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # convert the grayscale image to binary image
# # ret,thresh = cv2.threshold(gray_image,127,255,0)

# # # calculate moments of binary image
# # M = cv2.moments(thresh)

# # # calculate x,y coordinate of center
# # cX = int(M["m10"] / M["m00"])
# # cY = int(M["m01"] / M["m00"])

# # # put text and highlight the center
# # cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
# # cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# # # display the image
# # plt.figure(figsize=(15,15))
# # plt.imshow(img)




# # %%
# # list_of_objects_roii = ['1585434283_721692085_Left.png', '1585434290_857062101_Left.png', '1585434296_598551989_Left.png', '1585434302_705101967_Left.png', '1585434309_475411892_Left.png', '1585434315_681521893_Left.png', '1585434323_547031879_Left.png']
# # for imagen in list_of_objects_roii:
# #     img2 = cv2.imread(imagen)
# #     img_left2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# #     gray_left2 = cv2.cvtColor(img_left2, cv2.COLOR_RGB2GRAY)
# #     object_roi = gray_left2[250:400,975:1150]
# #     plt.figure(figsize=(12,12))
# #     plt.imshow(object_roi,cmap='gray')
# #     cv2.waitKey(15)

# # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # for i in range(left_count):
# #     imgL = cv2.imread(images_l[i])
# #     imgR = cv2.imread(images_r[i])
# #     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# #     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# #     outputL = imgL.copy()
# #     outputR = imgR.copy()

# #     retR, cornersR =  cv2.findChessboardCorners(outputR,(nb_vertical,nb_horizontal),None)
# #     retL, cornersL = cv2.findChessboardCorners(outputL,(nb_vertical,nb_horizontal),None)

# #     if retR and retL:
# #         objpoints.append(objp)
# #         imgpointsL.append(cornersL)
# #         imgpointsR.append(cornersR)
        
# #         cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
# #         cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
# #         cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
# #         cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
# #         cv2.imshow('cornersR',outputR)
# #         cv2.imshow('cornersL',outputL)
# #         cv2.waitKey(150)
        
# # cv2.destroyAllWindows()


# # %%
