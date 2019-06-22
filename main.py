import cv2
from lane_detect import *
from param import *
import time


video_test = '/home/linus/CV_notTrashCar/output.mp4'


def frame_process(raw_img):
    start_time = time.time()
    # Crop from sky line down
    raw_img = raw_img[sky_line:, :]

    # Hide sensor and car's hood
    raw_img = cv2.rectangle(raw_img, top_left_proximity,
                            bottom_right_proximity, hood_fill_color, -1)
    raw_img = cv2.rectangle(raw_img, top_left_hood,
                            bottom_right_hood, hood_fill_color, -1)
    cv2.imshow('raw', raw_img)

    # Handle shadow by using complex sobel operator
    # combined = get_combined_binary_thresholded_img(
    #     cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)) * 255
    # combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 51, 2)
    # combined = cv2.bitwise_not(combined)

    # Simple color filltering + Canny Edge detection
    combined = easy_lane_preprocess(raw_img)
    line_image, angle = hough_lines(combined, rho, theta,
                                    threshold, min_line_length, max_line_gap)
    test_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    annotated_image = cv2.cvtColor(weighted_img(
        line_image, test_img), cv2.COLOR_RGB2BGR)
    img = annotated_image
    print('FPS:', (1/(time.time()-start_time)))
    print('Angle:', angle)
    print('-----------------------------------')
    return img


def main():
    cap = cv2.VideoCapture(video_test)
    while(cap.isOpened()):
        _, frame = cap.read()
        if frame is not None:
            cv2.imshow('processed_frame', frame_process(frame))
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()
