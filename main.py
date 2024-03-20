import os
import random
import time
import cv2
import imutils
import numpy as np

UPLOAD_INPUT_CAMERA = '/adminResources/input/Car4.mp4'
UPLOAD_OUTPUT_CAMERA = '/adminResources/output/'

name = UPLOAD_INPUT_CAMERA.split("/")[2].replace("mp4", "")
cameraOutputFileName = name + '_detected{}.webm'.format(random.randrange(0, 9))
default_confidence = 0.5
default_threshold = 0.3

inputVideo = UPLOAD_INPUT_CAMERA
outputVideo = UPLOAD_OUTPUT_CAMERA + cameraOutputFileName

# load the COCO class labels our model was trained on
labelsPath = 'adminResources/models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the model weights and model configuration
weightsPath = 'D:\illigal_car_parking\illigal\\adminResources\models\yolov4.weights'
configPath = 'D:\illigal_car_parking\illigal\\adminResources\models\yolov4.cfg'
# load our model object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from model
print("[INFO] loading model from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
data_list = []
file = inputVideo.split(os.sep)[-1].split(".")[0]
parking_coordinate = "D:\illigal_car_parking\illigal\adminResources\parking-coordinate"
# f = open("{}{}{}.txt".format(parking_coordinate, os.sep, file), "r")
f=open('D:\illigal_car_parking\illigal\\adminResources\parking-coordinate\\test1.txt','r')
for lines in f:
    lines = eval(lines)
    data_list.append(lines)


def check_point_intersection(x1, y1, x2, y2, x, y):
    """
        :param x1: Roi top left x Coordinate
        :param y1: Roi top left y Coordinate
        :param x2: Roi bottom right x Coordinate
        :param y2: Roi bottom right x Coordinate
        :param x: Object center point x Coordinate
        :param y: Object center point y Coordinate
        :return: If the center point lies within ROI. Returns True or False
    """

    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture('D:\illigal_car_parking\illigal\\adminResources\input\\test1.mp4')

frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
# frame_width = int(vs.get(3))
# print(frame_width)
# frame_height = int(vs.get(4))
fps = 10  # int(vs.get(5))
writer = None
(W, H) = (None, None)
count = 0
frame_rate = 5

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print(total)
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    grabbed, frame = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the Model object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    for i in data_list:
        cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > default_confidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that Model
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, default_confidence,
                            default_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center_x = (x + w // 2)
            center_y = (y + h // 2)

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            if LABELS[classIDs[i]] == "car":
                cv2.circle(frame, (center_x, center_y), 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                for i in data_list:
                    if check_point_intersection(i[0], i[1], i[2], i[3],
                                                center_x, center_y):
                        cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]),
                                      (0, 0, 255), 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        writer = cv2.VideoWriter(outputVideo, fourcc, fps,
                                 (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)
    cv2.imshow('frame', frame)
    result.write(frame)
    count += frame_rate
    vs.set(1, count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
vs.release()
cv2.destroyAllWindows()
