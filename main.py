import os
import cv2
import numpy as np

video_test = '/Users/lamhoangtung/cds_data/output.mp4'

def main():
    cap = cv2.VideoCapture(video_test)
    while(cap.isOpened()):
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()