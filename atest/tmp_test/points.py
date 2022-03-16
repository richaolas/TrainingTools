import cv2

# import cv2 as cv
import numpy as np

WIDTH = 1920
HEIGHT = 1080
FRAME = 730

loc_mapping = {}
with open('data_loc.txt') as f:
    idx = 0
    for line in f.readlines():
        idx += 1
        frame, x, y = line.split()
        loc_mapping[int(frame)] = (int(x), int(y))

# value

for i in range(1, FRAME):
    if i not in loc_mapping:
        beg = i - 1
        while beg not in loc_mapping:
            beg -= 1
        end = i + 1
        while end not in loc_mapping:
            end += 1
        ratio = (i - beg) / (end - beg)
        x = loc_mapping[beg][0] + ratio * (loc_mapping[end][0] - loc_mapping[beg][0])
        y = loc_mapping[beg][1] + ratio * (loc_mapping[end][1] - loc_mapping[beg][1])
        loc_mapping[i] = (int(x), int(y))

print(loc_mapping)
# def circle_fitness_demo():
# # 创建图像， 绘制初始点
#     image = np.zeros((400, 400, 3), dtype=np.uint8)
#     x = np.array([30, 50, 100, 120])
#     y = np.array([100, 150, 240, 200])
#     for i in range(len(x)):
#         cv.circle(image, (x[i], y[i]), 3, (255, 0, 0), -1, 8, 0)
#         cv.imwrite("D:/curve.png", image)
# # 多项式曲线生成
#     poly = np.poly1d(np.polyfit(x, y, 6))
#     print(poly)
#     # 绘制拟合曲线
#     for t in range(30, 250, 1):
#         y_ = np.int(poly(t))
#         cv.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
#     cv.imshow("fit curve", image)
#     cv.imwrite("D:/fitcurve.png", image)
#     cv.waitKey()
#     cv.destroyAllWindows()
#
# circle_fitness_demo()


head_list = []
tail_list = []
with open('data_record.txt') as f:
    idx = 0
    for line in f.readlines():
        idx += 1

        ret = line.split()
        point = tuple(int(i) for i in ret)
        if idx % 2:
            head_list.append(point)
        else:
            tail_list.append(point)
        # print(line)

print(head_list)
print(tail_list)

x = [i for (i, j) in head_list] + [i for (i, j) in tail_list]
y = [j for (i, j) in head_list] + [j for (i, j) in tail_list]

f1 = np.polyfit(x, y, 2)
print(f1)

def find_nearest_point(x, y):
    min_val = 100000000
    min_idx = -1
    for i, pt in enumerate(head_list):
        if abs(x - pt[0]) < min_val:
            min_idx = i
            min_val = abs(x - pt[0])
    return head_list[min_idx], tail_list[min_idx]


def gety(x):
    global f1
    return f1[2] + f1[1] * x + f1[0] * x * x


def get_tape_points(f, width, beg, end):
    point_list = []
    for i in range(beg, end):
        x = i
        y = int(gety(i)) - width // 2
        point_list.append(x)
        point_list.append(y)

    for i in range(end, beg, -1):
        x = i
        y = int(gety(i)) + width // 2
        point_list.append(x)
        point_list.append(y)

    contours = []
    contour = np.array(point_list).reshape([len(point_list) // 2, 1, 2])
    contours.append(contour)

    return contours


def draw_transparency(frame, contours, color):
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    region = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.drawContours(region, contours, -1, color, -1)
    cv2.add(frame, region, frame, mask)
    #cv2.imshow('mask', frame)
    #cv2.waitKey(0)

# 采样一些点就ok了


vc = cv2.VideoCapture('VID_20210525_110223.mp4')


frame_no = 0

while True:
    ret, frame = vc.read()
    if not ret:
        break

    #if frame_no >= len(head_list):
    #    break

    #head_pt = head_list[frame_no]
    #tail_pt = tail_list[frame_no]
    frame_no += 1

    if frame_no not in loc_mapping:
        continue

    head_pt, tail_pt = find_nearest_point(loc_mapping[frame_no][0], 0)
    deta = loc_mapping[frame_no][0] - head_pt[0]
    print('++++', deta)
    tail_pt = (loc_mapping[frame_no][0] - abs(head_pt[0] - tail_pt[0]), tail_pt[1])
    head_pt = loc_mapping[frame_no]

    # print(frame.shape)
    #point_list = []
    for i in range(frame.shape[1]):
        # print(i, int(gety(i)))
        cv2.circle(frame, (i, int(gety(i))), 2, (255, 0, 0), -1)
        #point_list.append(i)
        #point_list.append(int(gety(i)))

    contours = get_tape_points(f1, 100, tail_pt[0], head_pt[0])
    draw_transparency(frame, contours, (0, 0, 255//2))


    #a = np.array(point_list).reshape([len(point_list) // 2, 1, 2])
    #contours.append(a)
    # contours.append()
    #cv2.drawContours(frame, contours, -1, (255, 255, 255), -1)

    for beg, end in zip(head_list, tail_list):
        cv2.circle(frame, beg, 2, (255, 0, 0), -1)
        cv2.circle(frame, end, 2, (0, 0, 255), -1)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # # Find Contour
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # np.array([100,100])
    # for c in contours:
    #     print(c, type(c), c.shape)

    # print(type(contours[0][0]))

    cv2.imshow('', frame)

    cv2.waitKey(0)
