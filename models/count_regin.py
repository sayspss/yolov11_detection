import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon


if __name__ == '__main__':
    region_points = {
        "region-01": [(261, 363), (641, 427), (511, 596), (45, 521)],
        "region-02": [(261, 363), (641, 427), (511, 596), (45, 521)],
    }

    regin1 = Polygon(region_points['region-01'])
    regin2 = Polygon(region_points['region-02'])
    im0 = cv2.imread('9.png')

    im0 = cv2.polylines(im0, np.array([region_points['region-01']], np.int32), True, (255, 0, 0), 4)
    cv2.putText(im0, 'danger_region', (region_points['region-01'][0][0], region_points['region-01'][0][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    im0 = cv2.polylines(im0, np.array([region_points['region-02']], np.int32), True, (0, 0, 255), 4)
    cv2.putText(im0, 'danger_region', (region_points['region-02'][0][0], region_points['region-02'][0][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    model = YOLO('yolo11n.pt')
    results = model.predict(im0, classes=[0], conf=0.6)[0]
    r1_count = 0
    r2_count = 0
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
        im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(im0, 'person', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(x1, y1, x2, y2)
        point = Point((int((x1+x2)/2), y2))
        if regin1.contains(point):
            r1_count += 1
            im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 5)
        if regin2.contains(point):
            r2_count += 1
            im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 5)
    cv2.imshow('1', im0)
    cv2.waitKey(0)