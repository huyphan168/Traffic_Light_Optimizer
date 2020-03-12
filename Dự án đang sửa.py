import os
import serial
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import numpy as np
from py.centroidtracker import CentroidTracker
from py.trackableobject import TrackableObject
import dlib

running_on_rpi = False

ser = serial.Serial ('/dev/ttyUSB0', baudrate = 9600, timeout = 3.0)
ser.isOpen()


os_info = os.name()
if os_info[4][:3] == 'arm':
    running_on_rpi = True

# check if optimization is enabled
if not cv2.useOptimized():
    print("By default, OpenCV has not been optimized")
    cv2.setUseOptimized(True)


writer = None
W = None
H = None

observation_mask = None
observation_mask_1 = None
observation_mask_2 = None
display_bounding_boxes = False

display_settings = True
totalFrames = 0
totalOverall = 0
totalOverall_1 = 0

image_for_result = None
CLASSES = ("car")


def predict(frame, net):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()

    predictions = []

    for box_index in range(100):
        if out[box_index + 1] == 0.0:
            break
        base_index = box_index * 7
        if (not np.isfinite(out[base_index]) or
                not np.isfinite(out[base_index + 1]) or
                not np.isfinite(out[base_index + 2]) or
                not np.isfinite(out[base_index + 3]) or
                not np.isfinite(out[base_index + 4]) or
                not np.isfinite(out[base_index + 5]) or
                not np.isfinite(out[base_index + 6])):
            continue
        object_info_overlay = out[base_index:base_index + 7]

        base_index = 0
        class_id = int(object_info_overlay[base_index + 1])
        conf = object_info_overlay[base_index + 2]
        if (conf <= args["confidence"] or class_id != 7):
            continue
        box_left = object_info_overlay[base_index + 3]
        box_top = object_info_overlay[base_index + 4]
        box_right = object_info_overlay[base_index + 5]
        box_bottom = object_info_overlay[base_index + 6]
        prediction_to_append = [class_id, conf, ((box_left, box_top), (box_right, box_bottom))]
        predictions.append(prediction_to_append)

    return predictions

def coverage_rate(mask, rects):
    maskLeft = mask[0][0]
    maskRight = mask[1][0]
    maskTop = mask[0][1]
    maskBot = mask[1][1]

    totalCover = 0
    totalArea = (mask[1][0] - mask[0][0]) * (mask[1][1] - mask[0][1])

    for x in range (maskLeft, maskRight):
        for y in range (maskTop, maskBot):
            for (x_min, y_min, x_max, y_max) in rects:
                if (x_min <= x and x <= x_max and y_min <= y and y <= y_max):
                    totalCover += 1

    return (int) (totalCover * 100 / totalArea)


def resize(frame, width, height=None):
    h, w, _ = frame.shape
    if height is None:
        factor = width * 1.0 / w
        height = int(factor * h )
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame_resized


def crop(frame, top, left, height, width):
    cropped = frame[top:top + height, left: left + width]
    return cropped


def draw_observation_mask(frame, top_left, bottom_right, top_left_1, bottom_right_1, alpha=0.5, color=(0, 0, 255)):
    overlay = frame.copy()
    output = frame.copy()

   #Vẽ 2 hình chữ nhật ROI trên frame chuẩn bị đè lên
    cv2.rectangle(overlay, top_left, bottom_right,
                  color, -1)
    cv2.rectangle(overlay, top_left_1, bottom_right_1,
                  color, -1) 
   #Đè tấm vẽ lên frame
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    return output
# Xây dựng cú pháp riêng
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", default=.5,
                help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
                help="switch to display image on screen")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-s", "--skip-frames", type=int, default=15,
	help="# of skip frames between detections")
ap.add_argument("-r", "--resize", type=str, default=None,
                help="resized frames dimensions, e.g. 320,240")
ap.add_argument("-m", "--mask", type=str, default=None,
                help="observation mask x_min,y_min,x_max,y_max, e.g. 50,70,220,300")
ap.add_argument("-m_1", "--mask_1", type=str, default=None,
                help="observation mask x_min_1,y_min_1,x_max_1,y_max_1, e.g. 50,70,220,300")
ap.add_argument("-m_1", "--mask_2", type=str, default=None,
                help="observation mask x_min_2,y_min_2,x_max_2,y_max_2, e.g. 50,70,220,300")
args = vars(ap.parse_args())

if args["mask"] is not None:
    try:
        x_min, y_min, x_max, y_max = [int(item.replace(" ", "")) for item in args["mask"].split(",")]
        observation_mask = [(x_min, y_min), (x_max, y_max)]
    except ValueError:
        print("Invalid mask format!")
if args["mask_1"] is not None:
    try:
        x_min_1, y_min_1, x_max_1, y_max_1 = [int(item.replace(" ", "")) for item in args["mask_1"].split(",")]
        observation_mask_1 = [(x_min_1, y_min_1), (x_max_1, y_max_1)]
    except ValueError:
        print("Invalid mask_1 format!")
if args["mask_2"] is not None:
    try:
        x_min_2, y_min_2, x_max_2, y_max_2 = [int(item.replace(" ", "")) for item in args["mask_2"].split(",")]
        observation_mask_2 = [(x_min_2, y_min_2), (x_max_2, y_max_2)]
    except ValueError:
        print("Invalid mask format!")
centroidTracker_max_disappeared = 15
centroidTracker_max_distance = 100
ct = CentroidTracker(maxDisappeared=centroidTracker_max_disappeared, maxDistance=centroidTracker_max_distance, mask=observation_mask)
ct_1 = CentroidTracker(maxDisappeared=centroidTracker_max_disappeared, maxDistance=centroidTracker_max_distance, mask=observation_mask_1)
ct_2 = CentroidTracker(maxDisappeared=centroidTracker_max_disappeared, maxDistance=centroidTracker_max_distance, mask=observation_mask_2)
trackers = []
trackers_1 = []
trackers_2 = []
trackableObjects = {}
trackableObjects_1 = {}
trackableObjects_2 = {}
#load model dnn moblienet
net = cv2.dnn.readNet('models/mobilenet-ssd/FP16/mobilenet-ssd.xml', 'models/mobilenet-ssd/FP16/mobilenet-ssd.bin')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
if not args.get("input", False):
    print("[INFO] bat dau video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    print("[INFO] dang mo file...")
    vs = cv2.VideoCapture(args["input"])

time.sleep(1)
fps = FPS().start()
starttimer = time.time()
while True:
    try:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if args["input"] is not None and frame is None:
            break
        if args["resize"] is not None:
            if "," in args["resize"]:
                w, h = [int(item) for item in args["resize"].split(",")]
                frame = resize(frame, width=w, height=h)
            else:
                frame = resize(frame, width=int(args["resize"]))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        H, W, _ = frame.shape

        if display_settings:
            print("[INFO] frame size (W x H): %d x %d" % (W, H))
            preview_image = frame.copy()
            preview_image_file = "screenshots/preview_%d_%d" % (W, H)
            if observation_mask is not None:
                print("Observation mask (bottom left, top right): %s" % str(observation_mask))
                preview_image = draw_observation_mask(image_for_result, observation_mask[0], observation_mask[1],observation_mask_1[0], observation_mask_1[1] )
                preview_image_file += "_mask_%d_%d_%d_%d" % (observation_mask[0][0], observation_mask[0][1], observation_mask[1][0], observation_mask[1][1])
            preview_image_file += ".jpg"
            cv2.imwrite(preview_image_file, preview_image)
            display_settings = False

        if args["display"] > 0 or args["output"] is not None:
            image_for_result = frame.copy()
            if observation_mask is not None:
                image_for_result = draw_observation_mask(image_for_result, observation_mask[0], observation_mask[1],observation_mask_1[0], observation_mask_1[1] )
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)
        status = "Waiting"
        rects = []
        rects_1 = []
        rects_2 = []
        if totalFrames % args["skip_frames"] == 0:
                
            status = "Detecting"
            trackers = []
            trackers_1 = []
            trackers_2 = []
            cropped_frame = frame[observation_mask[0][1]:observation_mask[1][1], observation_mask[0][0]:observation_mask[1][0]]
            cropped_frame_1 = frame[observation_mask_1[0][1]:observation_mask_1[1][1], observation_mask_1[0][0]:observation_mask_1[1][0]]
            cropped_frame_2 = frame[observation_mask_2[0][1]:observation_mask_2[1][1], observation_mask_2[0][0]:observation_mask_2[1][0]]
            predictions = predict(cropped_frame, net)
            predictions_1 = predict(cropped_frame_1, net)
            predictions_2 = predict(cropped_frame_2, net)
            for (i, pred) in enumerate(predictions):
                (class_id, pred_conf, pred_boxpts) = pred
                ((x_min, y_min), (x_max, y_max)) = pred_boxpts
                if pred_conf > args["confidence"]:
                        # Hiển thị prediction trên terminal của raspberry pi
                    print("[INFO] Prediction #{}: confidence={}, "
                          "boxpoints={}".format(i, pred_conf,
                                                pred_boxpts))
                           # Nếu class không phải là xe thì bỏ qua nó
                    if CLASSES[class_id] != "car":
                               continue
                        
                    if observation_mask is not None:
                              mask_width = observation_mask[1][0] - observation_mask[0][0]
                              mask_height = observation_mask[1][1] - observation_mask[0][1]
                              x_min = int(x_min * mask_width) + observation_mask[0][0]
                              y_min = int(y_min * mask_height) + observation_mask[0][1]
                              x_max = int(x_max * mask_width) + observation_mask[0][0]
                              y_max = int(y_max * mask_height) + observation_mask[0][1]
                    else:
                              x_min = int(x_min * W)
                              y_min = int(y_min * H)
                              x_max = int(x_max * W)
                              y_max = int(y_max * H)
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x_min, y_min, x_max, y_max)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
            for (i, pred_1) in enumerate(predictions_1):
                (class_id_1, pred_conf_1, pred_boxpts_1) = pred_1
                ((x_min_1, y_min_1), (x_max_1, y_max_1)) = pred_boxpts_1
                if pred_conf_1 > args["confidence"]:
                    print("[INFO] Prediction #{}: confidence={}, "
                            "boxpoints={}".format(i, pred_conf_1,
                                                pred_boxpts_1))
                    if CLASSES[class_id_1] != "car":
                               continue
                            
                    if observation_mask_1 is not None:
                              mask_width_1 = observation_mask_1[1][0] - observation_mask_1[0][0]
                              mask_height_1 = observation_mask_1[1][1] - observation_mask_1[0][1]
                              x_min_1 = int(x_min_1 * mask_width_1) + observation_mask_1[0][0]
                              y_min_1 = int(y_min_1 * mask_height_1) + observation_mask_1[0][1]
                              x_max_1 = int(x_max_1 * mask_width_1) + observation_mask_1[0][0]
                              y_max_1 = int(y_max_1 * mask_height_1) + observation_mask_1[0][1]
                    else:
                              x_min_1 = int(x_min_1 * W)
                              y_min_1 = int(y_min_1 * H)
                              x_max_1 = int(x_max_1 * W)
                              y_max_1 = int(y_max_1 * H)
                    tracker_1 = dlib.correlation_tracker()
                    rect_1 = dlib.rectangle(x_min_1, y_min_1, x_max_1, y_max_1)
                    tracker_1.start_track(rgb, rect_1)
                    trackers_1.append(tracker_1)
            for (i, pred_2) in enumerate(predictions_2):
                (class_id_2, pred_conf_2, pred_boxpts_2) = pred_2
                ((x_min_2, y_min_2), (x_max_2, y_max_2)) = pred_boxpts_2
                if pred_conf_2 > args["confidence"]:
                    if CLASSES[class_id_2] != "car":
                               continue
                            
                    if observation_mask_2 is not None:
                              mask_width_2 = observation_mask_2[1][0] - observation_mask_2[0][0]
                              mask_height_2 = observation_mask_2[1][1] - observation_mask_2[0][1]
                              x_min_2 = int(x_min_2 * mask_width_2) + observation_mask_2[0][0]
                              y_min_2 = int(y_min_2 * mask_height_2) + observation_mask_2[0][1]
                              x_max_2 = int(x_max_2 * mask_width_2) + observation_mask_2[0][0]
                              y_max_2 = int(y_max_2 * mask_height_2) + observation_mask_2[0][1]
                    else:
                              x_min_2 = int(x_min_2 * W)
                              y_min_2 = int(y_min_2 * H)
                              x_max_2 = int(x_max_2 * W)
                              y_max_2 = int(y_max_2 * H)
                    tracker_2 = dlib.correlation_tracker()
                    rect_2 = dlib.rectangle(x_min_2, y_min_2, x_max_2, y_max_2)
                    tracker_2.start_track(rgb, rect_2)
                    trackers_2.append(tracker_2)
        else:
	
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))
            for tracker_1 in trackers_1:
                status = "Tracking_1"
                tracker_1.update(rgb)
                pos_1 = tracker_1.get_position()
                startX_1 = int(pos_1.left())
                startY_1 = int(pos_1.top())
                endX_1 = int(pos_1.right())
                endY_1 = int(pos_1.bottom())
                rects_1.append((startX_1, startY_1, endX_1, endY_1))
            for tracker_2 in trackers_2:
                tracker_2.update(rgb)
                pos_2 = tracker_2.get_position()
                startX_2 = int(pos_2.left())
                startY_2 = int(pos_2.top())
                endX_2 = int(pos_2.right())
                endY_2 = int(pos_2.bottom())
                rects_2.append((startX_2, startY_2, endX_2, endY_2))
            
        objects = ct.update(rects)
        objects_1 = ct_1.update(rects_1)
        objects_2 = ct_2.update(rects_2)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

        
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                to.centroids.append(centroid)
                if not to.counted:
                    totalOverall += 1
                    to.counted = True


        for (objectID, centroid) in objects_1.items():
            to_1 = trackableObjects.get(objectID, None)
            if to_1 is None:
                to_1 = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to_1.centroids]
                to_1.centroids.append(centroid)
                if not to_1.counted:
                    totalOverall_1 += 1
                    to_1.counted = True


        # Tính độ mật độ phủ đường
        coverage_area = coverage_rate(mask=observation_mask_2, rects= rects_2)
         
            # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        trackableObjects_1[objectID] = to_1

        label = "{}: {:.2f}%".format(CLASSES[class_id], pred_conf * 100)
        label_1 = "{}: {:.2f}%".format(CLASSES[class_id_1], pred_conf * 100)

            # extract information from the prediction boxpoints
        y = y_min - 15 if y_min - 15 > 15 else y_min + 15
#có thể xóa được.
        if image_for_result is not None:
            if display_bounding_boxes:
                    cv2.rectangle(image_for_result, (x_min, y_min), (x_max, y_max),
                                  (255, 0, 0), 2)
                    cv2.putText(image_for_result, label, (x_min, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.rectangle(image_for_result, (x_min_1, y_min_1), (x_max_1, y_max_1),
                                  (255, 0, 0), 2)
                    cv2.putText(image_for_result, label_1, (x_min_1, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    text = "ID {}".format(objectID)
                
                    cv2.putText(image_for_result, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(image_for_result, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        info = [
            ("Làn 1", totalOverall),
	    ("Làn 2", totalOverall_1),
        ]

        if image_for_result is not None:
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image_for_result, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if writer is not None:
            writer.write(image_for_result)
        if args["display"] > 0:
            cv2.imshow("Output", image_for_result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        totalFrames += 1
        fps.update()
        endtimer = time.time()

        # Gửi data cho arduino
        if (endtimer - starttimer >= 60) :
            part1 = (int) (totalOverall * 100 / (totalOverall + totalOverall_1))
            # send data to arduino
            int1 = part1
            int1_encode = b'%d' %int1
            ser.write (int1_encode)
            
            starttimer = time.time()
            totalOverall = 0
            totalOverall_1 = 0


    # thoát khỏi vòng lặp
    except KeyboardInterrupt:
        break
    except AttributeError:
        break
fps.stop()
if args["display"] > 0:
    cv2.destroyAllWindows()
if not args.get("input", False):
    vs.stop()
else:
    vs.release()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


			
