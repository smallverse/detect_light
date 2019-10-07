#!/usr/bin/python
# python3
import sqlite3
import time
import sys
import uuid
import multiprocessing as mp
import math
from operator import itemgetter  # itemgetter用来去dict中的key，省去了使用lambda函数
from itertools import groupby  # itertool还包含有其他很多函数，比如将多个list联合起来

import cv2
import uuid
import numpy as np
import os

defaultencoding = 'utf-8'
print('defaultencoding:', sys.getdefaultencoding())
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

print('setdefaultencoding,utf-8:', sys.getdefaultencoding())


#######################################################################
t_statistics = '''create table t_statistics
       (
            id char(128) primary key     not null,
            frame_id           char(128)    not null,
            hull_side_num int     not null,
            hull_area real,
            ellipse_area real,
            hull_ellipse_area_ratio real,
            ellipse_x            int     not null,
            ellipse_y            int     not null
       );
       '''
t_conf = '''create table t_conf
       (
            id char(128) primary key     not null,
            ellipse_x            int     not null,
            ellipse_y            int     not null
       );
       '''


def init_db():
    try:
        conn = sqlite3.connect('light_detect.db')
        print("Opened database successfully")

        c = conn.cursor()
        c.execute(t_statistics)
        conn.commit()
        print("Table t_statistics created successfully")

    except BaseException as identifier:
        print(identifier)
    finally:
        conn.close()

    try:
        conn = sqlite3.connect('light_detect.db')
        print("Opened database successfully")

        c = conn.cursor()
        c.execute(t_conf)
        conn.commit()
        print("Table t_conf created successfully")

    except BaseException as identifier:
        print(identifier)
    finally:
        conn.close()


def insert_by_sql(sql, *parameters):
    try:
        conn = sqlite3.connect('light_detect.db')
        c = conn.cursor()
        c.execute(sql, parameters)

        conn.commit()
    except BaseException as identifier:
        print('insert_by_sql:', identifier)
    finally:
        print('insert_by_sql end')
        conn.close()


def select_by_sql(sql):
    try:
        conn = sqlite3.connect('light_detect.db')
        c = conn.cursor()

        cursor = c.execute(sql)
        res = []
        for row in cursor:
            res.append(row)
        return res
    except BaseException as identifier:
        print('select_by_sql:', identifier)
        return None
    finally:
        print('select_by_sql end')
        conn.close()


def del_by_sql(sql):
    try:
        conn = sqlite3.connect('light_detect.db')
        c = conn.cursor()
        c.execute(sql)
        conn.commit()
    except BaseException as identifier:
        print('del_by_sql:', identifier)
    finally:
        print('del_by_sql end')
        conn.close()


HULL_MIN_SIDE_NUM = 5  # 多边形边数的最小值,拟合椭圆最小5边6个点
HULL_MIN_AREA = 1000  # 多边形面积的最小值
MAX_ECCENTRICITY = 0.80  # 离心率最大值
MIN_HULL_ELLIPSE_AREA_RATIO = 0.92  # 凸包椭圆最小面积比


def set_red_light_data(frame):
    """[summary]
    整体思路:读取三分钟，根据指标进行统计，找出最优点，持久化记录作为初始化识别配置
    具体实现：一段时间内循环处理（每帧照片》转为HSV》指定颜色对原图和掩模进行位运算》膨胀腐蚀或开闭运算》获取凸包并过滤》获取拟合椭圆计算相关信息》持久化统计）
    获取拟合椭圆计算相关信息：和凸包的面积比，离心率，坐标等
    Arguments:
        frame {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    print('------find_red_light')
    frame_id = str(uuid.uuid1())
    t1 = time.time()
    # 转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print('hsv', hsv.dtype, hsv.shape)

    # #设定青色的阀值
    # low_range = np.array([78,206,206])
    # high_range = np.array([99,255,255])

    # #设定红色的阀值
    low_range = np.array([0, 123, 100])
    high_range = np.array([5, 255, 255])

    # 根据阀值构建掩模
    mask = cv2.inRange(hsv, low_range, high_range)
    print('mask', mask.dtype, mask.shape)
    # 对原图和掩模进行位运算
    res = cv2.bitwise_and(frame, frame, mask=mask)
    print('bitwise_and res', res.dtype, res.shape)

    # #膨胀
    res = cv2.dilate(res, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (12, 12)))
    # # 开闭运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))  # 矩形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))  # 椭圆结构
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)  # 开运算
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)  # 闭运算

    # #模糊平滑
    # res = cv2.blur(res,(1,1))
    # print('res',res.dtype,res.shape)

    edges = cv2.Canny(res, 128, 256)
    print('edges', edges.dtype, edges.shape)

    print('------time ms:', (time.time()-t1)*1000)

    cnts, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros(frame.shape, dtype=np.uint8)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
    print('cnts0:', len(cnts))

    # 凸包
    hulls = [cv2.convexHull(inst) for inst in cnts]

    # 筛选出边长数>=  HULL_MIN_SIDE_NUM 的多边形
    approxs = [approx for approx in hulls if len(
        approx) >= HULL_MIN_SIDE_NUM]
    print('approxs0:', len(approxs))

    approxs = [approx for approx in approxs if cv2.contourArea(
        approx) > HULL_MIN_AREA]  # 筛选出面积大于 HULL_MIN_AREA 的多边形
    print('approxs1:', len(approxs))

    print('******HULL_MIN_SIDE_NUM,MIN_AREA_NUM',
          HULL_MIN_SIDE_NUM, HULL_MIN_AREA)

    # 拟合椭圆
    all_ellipse = []
    for inst in approxs:
        hull_area = cv2.contourArea(inst)
        hull_side_num = len(inst)
        ellipse = cv2.fitEllipse(inst)
        # print('ellipse:', ellipse)
        (x, y), (MA, ma), angle = ellipse
        a = ma/2
        b = MA/2

        ellipse_area = a * b * np.pi
        hull_ellipse_area_ratio = hull_area/ellipse_area
        ellipse_x = int(x)
        ellipse_y = int(y)

        if hull_ellipse_area_ratio < MIN_HULL_ELLIPSE_AREA_RATIO:
            print('******hull_ellipse_area_ratio < MIN_HULL_ELLIPSE_AREA_RATIO',
                  hull_ellipse_area_ratio, MIN_HULL_ELLIPSE_AREA_RATIO, (int(x), int(y)))
            continue

        # 离心率
        eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        eccentricity = round(eccentricity/a, 2)
        if eccentricity > MAX_ECCENTRICITY:
            print('******eccentricity > MAX_ECCENTRICITY',
                  eccentricity, MAX_ECCENTRICITY, (ellipse_x, ellipse_y))
            continue

        all_ellipse.append(ellipse)

        cv2.ellipse(img, ellipse, (0, 0, 255), 2)  # red
        print('****** hull_area,ellipse_area,hull_ellipse_area_ratio,eccentricity:',
              int(hull_area), int(ellipse_area), hull_ellipse_area_ratio, eccentricity, (ellipse_x, ellipse_y))

        id = str(uuid.uuid1())
        # TODO://持久化
        # id char(128) primary key not null,
        # frame_id           char(128) not null,
        # hull_side_num int not null,
        # hull_area real,
        # ellipse_area real,
        # hull_ellipse_area_ratio real,
        # ellipse_x            int not null,
        # ellipse_y            int not null
        insert_sql = """insert into t_statistics (id,frame_id,hull_side_num,hull_area,ellipse_area,hull_ellipse_area_ratio,ellipse_x,ellipse_y)
        VALUES (?,?,?,?,?,?,?,?)"""
        insert_by_sql(insert_sql, id, frame_id, hull_side_num, hull_area,
                      ellipse_area, hull_ellipse_area_ratio, ellipse_x, ellipse_y)

    cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green
    print('------time ms:', (time.time()-t1)*1000)

    return img


def init_red_conf_by_camera():
    all_data = select_by_sql(
        'select id,frame_id,hull_side_num,hull_area,ellipse_area,hull_ellipse_area_ratio,ellipse_x,ellipse_y from t_statistics')
    print('**********************************************')
    print('all_data count', len(all_data))
    all_obj_data = []
    for inst in all_data:
        print(inst)
        all_obj_data.append({'id': inst[0], 'frame_id': inst[1], 'hull_side_num': inst[2], 'hull_area': inst[3],
                             'ellipse_area': inst[4], 'hull_ellipse_area_ratio': inst[5], 'ellipse_x': inst[6], 'ellipse_y': inst[7]})

    # 在次之前通过调节  HULL_MIN_SIDE_NUM 、HULL_MIN_AREA 、MAX_ECCENTRICITY 、MIN_HULL_ELLIPSE_AREA_RATIO 已经非常准确的可以找出人行道红灯，但还没有很趋近于100%，为了普适一些，不要调节太严格
    # 其实人行道红灯是小人或者圆形，如果摄像头的焦点调节都相似，则拟合椭圆的离心率、凸包/拟合椭圆面积，准确率就很高
    # 理想情况下一组一条数据，特殊画面一组多个，需要根据其他指标判断：监控总时长包含红绿灯交替时段，红灯的个数<group_count排除静止红色椭圆物体（恰巧被遮挡几次暂不考虑），运动的红色椭圆个数比较少，
    lstg = groupby(all_obj_data, itemgetter('frame_id'))
    arr = list([{'key': key, 'data': list(group)} for key, group in lstg])
    group_count = len(arr)
    print('group_count:', group_count)

    # 分类
    final_res = []

    part0_obj_data, part1_obj_data = get_part(all_obj_data[0], all_obj_data)
    final_res.append(part0_obj_data)
    while len(part1_obj_data) > 0:
        part0_obj_data, part1_obj_data = get_part(
            part1_obj_data[0], part1_obj_data)
        final_res.append(part0_obj_data)
        print('len(part1_obj_data):', len(part1_obj_data))

    print('**********************************************')

    print('final_res len:', len(final_res))

    sorts_final_res = sorted(final_res, key=lambda k: len(k), reverse=True)
    # 去除匹配的最高点
    final_points = sorts_final_res[0] if len(
        sorts_final_res[0]) < group_count else sorts_final_res[1]

    print('final_points len:', len(final_points))
    final_x = int(np.mean(
        np.array(list(map(lambda x: x['ellipse_x'], final_points)))))
    final_y = int(np.mean(
        np.array(list(map(lambda x: x['ellipse_y'], final_points)))))
    print('final_x, final_y:', final_x, final_y)

    del_by_sql('delete from t_conf where 1=1;')
    insert_sql = """insert into t_conf (id,ellipse_x,ellipse_y)
        VALUES (?,?,?)"""
    insert_by_sql(insert_sql, str(uuid.uuid1()), final_x, final_y)


UP_DOWN_NUM = 5


def get_part(first, all_obj_data):
    """[summary]
    误差为UP_DOWN_NUM和大于UP_DOWN_NUM的分组
    Arguments:
        first {[type]} -- [description]
        all_obj_data {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    part0_obj_data = list(filter(
        lambda x: (math.fabs(first['ellipse_x']-x['ellipse_x']) <= UP_DOWN_NUM and math.fabs(first['ellipse_y']-x['ellipse_y']) <= UP_DOWN_NUM), all_obj_data))
    part1_obj_data = list(filter(
        lambda x: (math.fabs(x['ellipse_x'] - first['ellipse_x']) > UP_DOWN_NUM or math.fabs(x['ellipse_y'] - first['ellipse_y']) > UP_DOWN_NUM), all_obj_data))

    return part0_obj_data, part1_obj_data


def detect_red_light(frame):
    pass


DETECT_INTERVAL = 0.1  # 0.1s default


def detect_by_camera(q, point_x, point_y):
    print('------detect_by_camera')
    t1 = time.time()
    frame = q.get()
    frame = frame['image']

    # 转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print('hsv', hsv.dtype, hsv.shape)

    # #设定青色的阀值
    # low_range = np.array([78,206,206])
    # high_range = np.array([99,255,255])

    # #设定红色的阀值
    low_range = np.array([0, 123, 100])
    high_range = np.array([5, 255, 255])

    # 根据阀值构建掩模
    mask = cv2.inRange(hsv, low_range, high_range)
    print('mask', mask.dtype, mask.shape)
    # 对原图和掩模进行位运算
    res = cv2.bitwise_and(frame, frame, mask=mask)
    print('bitwise_and res', res.dtype, res.shape)

    print('------time0 ms:', int((time.time()-t1)*1000))

    # #膨胀
    res = cv2.dilate(res, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (12, 12)))

    print('------time1 ms:', int((time.time()-t1)*1000))

    edges = cv2.Canny(res, 128, 256)
    print('edges', edges.dtype, edges.shape)

    print('------time ms:', int((time.time()-t1)*1000))

    cnts, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros(frame.shape, dtype=np.uint8)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
    print('cnts0:', len(cnts))

    # 凸包
    hulls = [cv2.convexHull(inst) for inst in cnts]

    # 筛选出边长数>=  HULL_MIN_SIDE_NUM 的多边形
    approxs = [approx for approx in hulls if len(
        approx) >= HULL_MIN_SIDE_NUM]
    print('approxs0:', len(approxs))

    approxs = [approx for approx in approxs if cv2.contourArea(
        approx) > HULL_MIN_AREA]  # 筛选出面积大于 HULL_MIN_AREA 的多边形
    print('approxs1:', len(approxs))

    print('******HULL_MIN_SIDE_NUM,MIN_AREA_NUM',
          HULL_MIN_SIDE_NUM, HULL_MIN_AREA)

    re = False
    # 拟合椭圆
    all_ellipse = []
    for inst in approxs:
        hull_area = cv2.contourArea(inst)
        hull_side_num = len(inst)
        ellipse = cv2.fitEllipse(inst)
        # print('ellipse:', ellipse)
        (x, y), (MA, ma), angle = ellipse
        int_x = int(x)
        int_y = int(y)
        if math.fabs(int_x-point_x) <= UP_DOWN_NUM and math.fabs(int_y-point_y) <= UP_DOWN_NUM:
            re = True
            print('(int_x, int_y):', (int_x, int_y))
            break

    return re, int((time.time()-t1)*1000)

###########################


def image_put(q, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, img = cap.read()
        # data={'ret':ret,'image_char':img.astype(np.uint8).tostring(),'rtsp_url':rtsp_url,'id':str(uuid.uuid1())}
        data = {'ret': ret, 'image': img,
                'rtsp_url': rtsp_url, 'id': str(uuid.uuid1())}

        q.put(data)
        # q.get() if q.qsize() > 1 else time.sleep(0.01)
        # print('q.qsize():', q.qsize())
        q.get() if q.qsize() > 1 else print('read...')

        # print('time', time.time()*1000)


repo_base = os.path.abspath(os.curdir)

FIND_INTERVAL = 10  # 10s default
FIND_TIME = 5*60*1000  # 5min default
MAX_DETECT_TIME = 10*60*1000  # 10min default

global POINT_MOVE, conf_data, DETECT_START_TIME
POINT_MOVE = False
DETECT_START_TIME = int(time.time()*1000)

conf_data = select_by_sql(
    'select id,ellipse_x,ellipse_y from t_conf')
print('conf_data count', conf_data)


def image_get(q):
    print('------image_get')
    global POINT_MOVE, conf_data, DETECT_START_TIME

    conf_data = init_point_conf(q)

    if len(conf_data) == 1:
        print('******len(conf_data) == 1')
        point_x = conf_data[0][1]
        point_y = conf_data[0][2]

        detect_light(q, point_x, point_y)

    elif len(conf_data) > 1:
        print('******len(conf_data) > 1')


def detect_light(q, point_x, point_y):
    global POINT_MOVE, conf_data, DETECT_START_TIME

    while POINT_MOVE == False:
        print('***************detect_light***************')
        re, use_time = detect_by_camera(q, point_x, point_y)
        if re == False:
            print('--------------------------------------------light is green')
            continue
        if(int(time.time()*1000)-DETECT_START_TIME > MAX_DETECT_TIME):
            print('******light is not working?******')
            POINT_MOVE = True
            # 有可能位置偏移或者灯不工作了
            image_get(q)

        else:
            DETECT_START_TIME = int(time.time()*1000)  # 每次识别成功重置
            # TODO:send msg
            print(
                '---------------------------------------------light is red,use_time:', use_time)
            pass


def init_point_conf(q):
    global POINT_MOVE, conf_data, DETECT_START_TIME

    if conf_data == None or len(conf_data) == 0 or POINT_MOVE == True:  # 如果没有点或者点偏移寻找点

        t1 = time.time()
        del_by_sql('delete from t_statistics where 1=1;')

        while int((time.time()-t1)*1000) <= FIND_TIME:
            print('************************ image_get time.time()', time.time())

            frame = q.get()
            # print('frame:',len(frame['image_char']))
            print('frame:', len(frame['image']))

            img = frame['image']
            print(type(img), img.shape)

            re_img = set_red_light_data(img)

            # 找点,10s
            time.sleep(FIND_INTERVAL)

            print('int((time.time()-t1)*1000):', int((time.time()-t1)*1000))

        print('************************ find_red_light end, time.time()', time.time())

        init_red_conf_by_camera()

        conf_data = select_by_sql(
            'select id,ellipse_x,ellipse_y from t_conf')

        print('conf_data count', len(conf_data))

        DETECT_START_TIME = int((time.time())*1000)  # 识别开始时间在初始化完毕配置时重置

    return conf_data


def run_single_camera(rtsp_url):

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, rtsp_url,)),
                 mp.Process(target=image_get, args=(queue,))]

    [process.start() for process in processes]
    [process.join() for process in processes]


def run_multi_camera(rtsp_urls):
    """[summary]

    Arguments:
        rtsp_urls {[type]} -- [array]
    """
    print('len(rtsp_urls):', len(rtsp_urls))
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=2) for _ in rtsp_urls]

    processes = []
    for queue, rtsp_url in zip(queues, rtsp_urls):
        processes.append(mp.Process(target=image_put, args=(queue, rtsp_url,)))
        processes.append(mp.Process(target=image_get, args=(queue,)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def run():
    init_db()

    type = '白1红.avi'
    # type = '白1绿.avi'
    # type = '晚1红.avi'
    # type = '晚1绿.avi'
    rtsp_url = '/home/zhaijf/local_v/'+type
    print(rtsp_url)
    run_single_camera(rtsp_url)


if __name__ == '__main__':
    run()
