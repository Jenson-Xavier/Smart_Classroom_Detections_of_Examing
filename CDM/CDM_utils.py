import math
import cv2


def cal_Distance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(a, b, c):
    """根据三边长度计算角度（单位：度）"""
    cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    cos_angle = min(cos_angle, 1.0)
    cos_angle = max(cos_angle, -1.0)
    angle = math.acos(cos_angle)
    return math.degrees(angle)


class KeypointsChecker_CDM:
    def __init__(self, rotor_thres, roll_thres, reach_thres):
        self.rotor_thres = rotor_thres
        self.roll_thres = roll_thres
        self.reach_thres = reach_thres

    def run(self, keypoints):
        is_rotor, is_roll, is_reach = False, False, False
        rotor_rate, roll_rate, reach_rate = 0, 0, 0

        leye, reye = keypoints[1], keypoints[2]
        lear, rear = keypoints[3], keypoints[4]
        head = keypoints[0]
        lshldr, rshldr = keypoints[5], keypoints[6]
        lelbow, relbow = keypoints[7], keypoints[8]
        lhand, rhand = keypoints[9], keypoints[10]

        # 转头判定
        if (leye[0] < lear[0] and reye[0] < rear[0]) or (leye[0] > lear[0] and reye[0] > rear[0]):
            D_leye2ear = cal_Distance(leye[0], lear[0], leye[1], lear[1])
            D_reye2ear = cal_Distance(reye[0], rear[0], reye[1], rear[1])
            D_eye2eye = cal_Distance(leye[0], reye[0], leye[1], reye[1])
            rotor_rate = min(D_leye2ear, D_reye2ear) / max(D_leye2ear, D_reye2ear)
            rotor_rate = round(rotor_rate, 3)

            if rotor_rate > self.rotor_thres:
                is_rotor = True

        # 侧倾判定
        D_5_6 = cal_Distance(lshldr[0], rshldr[0], lshldr[1], rshldr[1])
        D_0_5 = cal_Distance(lshldr[0], head[0], lshldr[1], head[1])
        D_0_6 = cal_Distance(rshldr[0], head[0], rshldr[1], head[1])

        angle_0 = calculate_angle(D_0_5, D_0_6, D_5_6)
        angle_5 = calculate_angle(D_5_6, D_0_5, D_0_6)
        angle_6 = calculate_angle(D_5_6, D_0_6, D_0_5)

        if angle_5 > self.roll_thres or angle_6 > self.roll_thres:
            is_roll = True
        if is_roll:
            roll_rate = max(angle_5, angle_6) / 160
            roll_rate = round(roll_rate, 3)

        # 伸手判定
        if (lhand[0] > lelbow[0] > lshldr[0]) or (rhand[0] < relbow[0] < rshldr[0]):
            # 计算上臂与下臂夹角
            D_6_8 = cal_Distance(rshldr[0], relbow[0], rshldr[1], relbow[1])
            D_8_10 = cal_Distance(relbow[0], rhand[0], relbow[1], rhand[1])
            D_6_10 = cal_Distance(rshldr[0], rhand[0], rshldr[1], rhand[1])

            angle_8 = calculate_angle(D_6_8, D_8_10, D_6_10)

            D_5_7 = cal_Distance(lshldr[0], lelbow[0], lshldr[1], lelbow[1])
            D_7_9 = cal_Distance(lelbow[0], lhand[0], lelbow[1], lhand[1])
            D_5_9 = cal_Distance(lshldr[0], lhand[0], lshldr[1], lhand[1])

            angle_7 = calculate_angle(D_5_7, D_7_9, D_5_9)

            if ((angle_7 > self.reach_thres and lhand[0] > lelbow[0] > lshldr[0])
                    or (angle_8 > self.reach_thres and rhand[0] < relbow[0] < rshldr[0])):
                is_reach = True
                reach_rate = max(angle_7, angle_8) / 180
                reach_rate = round(reach_rate, 3)

        result = is_rotor | is_roll

        return int(result), (rotor_rate, roll_rate, reach_rate)


