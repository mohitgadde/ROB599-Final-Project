#!/usr/bin/env python3
# encoding:utf-8
import sys
sys.path.append('/home/ubuntu/Sensor/')
sys.path.append('/home/ubuntu/ArmPi/')
import time
import socket
import json
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from ArmIK.InverseKinematics import *
from ArmIK.Transform import getAngle
from ArmIK.Transform import convertCoordinate
from mpl_toolkits.mplot3d import Axes3D
from HiwonderSDK.Board import setBusServoPulse, getBusServoPulse

#机械臂根据逆运动学算出的角度进行移动
ik = IK('arm')
#设置连杆长度
l1 = ik.l1 + 0.75
l4 = ik.l4 - 0.15
ik.setLinkLength(L1=l1, L4=l4)

serverIP = "192.168.91.178" # Mohit PC IP
clientAddress = ("192.168.91.251", 20001) #ArmPi IP
serverPort = 20001
buffersize = 256
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPClientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
UDPClientSocket.bind(clientAddress)

class Camera():

    def __init__(self, serverIP="127.0.0.1", serverPort=20001):
        print("Camera Server TCP Initialized.")
        self.serverIP = serverIP
        self.serverPort = serverPort
        self.msgOut = {"edges":None}
        self.buffersize = 1024

        # setup socket
        # assume only single to single connection with the robot
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.socket.bind((self.serverIP, self.serverPort))
        self.socket.listen(1)
        self.clientSocket, self.clientIP = self.socket.accept()
        print("Connection established with client!")
        print("Client ip", self.clientIP)
        self.camera_fcn()

    def send(self):
        # clientMsg = self.clientSocket.recv(self.buffersize)
        self.get_data()
        data = json.dumps(self.msgOut).encode()
        # send reply to client
        print("Msg to Client ", data)
        self.clientSocket.send(data)
        time.sleep(1)

    def get_data(self):
        self.msgOut["x"] = round(np.random.random(), 2)
        self.msgOut["y"] = round(np.random.random(), 2)

    def __close__(self):
        self.socket.close()

class ArmIK:
    servo3Range = (0, 1000, 0, 240) #脉宽， 角度
    servo4Range = (0, 1000, 0, 240)
    servo5Range = (0, 1000, 0, 240)
    servo6Range = (0, 1000, 0, 240)

    def __init__(self):
        self.setServoRange()

    def setServoRange(self, servo3_Range=servo3Range, servo4_Range=servo4Range, servo5_Range=servo5Range, servo6_Range=servo6Range):
        # 适配不同的舵机
        self.servo3Range = servo3_Range
        self.servo4Range = servo4_Range
        self.servo5Range = servo5_Range
        self.servo6Range = servo6_Range
        self.servo3Param = (self.servo3Range[1] - self.servo3Range[0]) / (self.servo3Range[3] - self.servo3Range[2])
        self.servo4Param = (self.servo4Range[1] - self.servo4Range[0]) / (self.servo4Range[3] - self.servo4Range[2])
        self.servo5Param = (self.servo5Range[1] - self.servo5Range[0]) / (self.servo5Range[3] - self.servo5Range[2])
        self.servo6Param = (self.servo6Range[1] - self.servo6Range[0]) / (self.servo6Range[3] - self.servo6Range[2])

    def transformAngelAdaptArm(self, theta3, theta4, theta5, theta6):
        #将逆运动学算出的角度转换为舵机对应的脉宽值
        servo3 = int(round(theta3 * self.servo3Param + (self.servo3Range[1] + self.servo3Range[0])/2))
        if servo3 > self.servo3Range[1] or servo3 < self.servo3Range[0] + 60:
            logger.info('servo3(%s)超出范围(%s, %s)', servo3, self.servo3Range[0] + 60, self.servo3Range[1])
            return False

        servo4 = int(round(theta4 * self.servo4Param + (self.servo4Range[1] + self.servo4Range[0])/2))
        if servo4 > self.servo4Range[1] or servo4 < self.servo4Range[0]:
            logger.info('servo4(%s)超出范围(%s, %s)', servo4, self.servo4Range[0], self.servo4Range[1])
            return False

        servo5 = int(round((self.servo5Range[1] + self.servo5Range[0])/2 - (90.0 - theta5) * self.servo5Param))
        if servo5 > ((self.servo5Range[1] + self.servo5Range[0])/2 + 90*self.servo5Param) or servo5 < ((self.servo5Range[1] + self.servo5Range[0])/2 - 90*self.servo5Param):
            logger.info('servo5(%s)超出范围(%s, %s)', servo5, self.servo5Range[0], self.servo5Range[1])
            return False
        
        if theta6 < -(self.servo6Range[3] - self.servo6Range[2])/2:
            servo6 = int(round(((self.servo6Range[3] - self.servo6Range[2])/2 + (90 + (180 + theta6))) * self.servo6Param))
        else:
            servo6 = int(round(((self.servo6Range[3] - self.servo6Range[2])/2 - (90 - theta6)) * self.servo6Param))
        if servo6 > self.servo6Range[1] or servo6 < self.servo6Range[0]:
            logger.info('servo6(%s)超出范围(%s, %s)', servo6, self.servo6Range[0], self.servo6Range[1])
            return False

        return {"servo3": servo3, "servo4": servo4, "servo5": servo5, "servo6": servo6}

    def servosMove(self, servos, movetime=None):
        #驱动3,4,5,6号舵机转动
        time.sleep(0.02)
        if movetime is None:
            max_d = 0
            for i in  range(0, 4):
                d = abs(getBusServoPulse(i + 3) - servos[i])
                if d > max_d:
                    max_d = d
            movetime = int(max_d*4)
        setBusServoPulse(3, servos[0], movetime)
        setBusServoPulse(4, servos[1], movetime)
        setBusServoPulse(5, servos[2], movetime)
        setBusServoPulse(6, servos[3], movetime)

        return movetime

    def setPitchRange(self, coordinate_data, alpha1, alpha2, da = 1):
        #给定坐标coordinate_data和俯仰角的范围alpha1，alpha2, 自动在范围内寻找到的合适的解
        #如果无解返回False,否则返回对应舵机角度,俯仰角
        #坐标单位cm， 以元组形式传入，例如(0, 5, 10)
        #da为俯仰角遍历时每次增加的角度
        x, y, z = coordinate_data
        if alpha1 >= alpha2:
            da = -da
        for alpha in np.arange(alpha1, alpha2, da):#遍历求解
            result = ik.getRotationAngle((x, y, z), alpha)
            if result:
                theta3, theta4, theta5, theta6 = result['theta3'], result['theta4'], result['theta5'], result['theta6']
                servos = self.transformAngelAdaptArm(theta3, theta4, theta5, theta6)
                if servos != False:
                    return servos, alpha

        return False

    def setPitchRangeMoving(self, coordinate_data, alpha, alpha1, alpha2, movetime=None):
        #给定坐标coordinate_data和俯仰角alpha,以及俯仰角范围的范围alpha1, alpha2，自动寻找最接近给定俯仰角的解，并转到目标位置
        #如果无解返回False,否则返回舵机角度、俯仰角、运行时间
        #坐标单位cm， 以元组形式传入，例如(0, 5, 10)
        #alpha为给定俯仰角
        #alpha1和alpha2为俯仰角的取值范围
        #movetime为舵机转动时间，单位ms, 如果不给出时间，则自动计算
        x, y, z = coordinate_data
        result1 = self.setPitchRange((x, y, z), alpha, alpha1)
        result2 = self.setPitchRange((x, y, z), alpha, alpha2)
        if result1 != False:
            data = result1
            if result2 != False:
                if abs(result2[1] - alpha) < abs(result1[1] - alpha):
                    data = result2
        else:
            if result2 != False:
                data = result2
            else:
                return False
        servos, alpha = data[0], data[1]

        movetime = self.servosMove((servos["servo3"], servos["servo4"], servos["servo5"], servos["servo6"]), movetime)

        return servos, alpha, movetime
    
    #for test
    def drawMoveRange2D(self, x_min, x_max, dx, y_min, y_max, dy, z, a_min, a_max, da):
        # 测试可到达点, 以2d图形式展现，z固定
        #测试可到达点, 以3d图形式展现，如果点过多，3d图会比较难旋转
        plt.show()
        try:
            for y in np.arange(y_min, y_max, dy):
                for x in np.arange(x_min, x_max, dx):
                    x_,y_ = convertCoordinate(x,y,(640, 480))
                    result = self.setPitchRange((-10, y_, z), a_min, a_min, da)
                    print(result)
                    if result:
                        plt.scatter(x, y, s=np.pi, c='r')
                        plt.draw()

            plt.xlabel('X Label')
            plt.ylabel('Y Label')

            plt.show()
        except Exception as e:
            print(e)
            pass

    def drawMoveRange3D(self, x_min, x_max, dx, y_min, y_max, dy, z_min, z_max, dz, a_min, a_max, da):
        #测试可到达点, 以3d图形式展现，如果点过多，3d图会比较难旋转
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            for z in np.arange(z_min, z_max, dz):
                for y in np.arange(y_min, y_max, dy):
                    for x in np.arange(x_min, x_max, dx):
                        # x_,y_ = convertCoordinate((x,y),(640, 480))
                        result = self.setPitchRange((x, y, z), a_min, a_max, da)
                        if result:
                            ax.scatter(x, y, z, s=np.pi, c='r')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()
        except Exception as e:
            print(e)
            pass
            
def drawTraingle():
    print(AK.setPitchRangeMoving((-10, 12, 20), 0, -90, 0, 1000))
    time.sleep(3)
    print(AK.setPitchRangeMoving((-10, 12, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-8, 12, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-6, 12, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-4, 12, 4), 0, -90,0, 1000))
    time.sleep(2)
    print("base of triangle")
    print(AK.setPitchRangeMoving((-5, 13, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-6, 15, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-7, 17, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print("two completed")
    print(AK.setPitchRangeMoving((-8, 15, 4), 0, -90,0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-9, 13, 4), 0, -90, 0, 1000))
    time.sleep(1.5)
    print(AK.setPitchRangeMoving((-10, 12, 4), 0, -90,0, 1000))
    time.sleep(1.5)

def drawRectangle():
	print(AK.setPitchRangeMoving((-10, 12, 20), 0, -90, 0, 1000))
	print("side 1")
	time.sleep(3)
	print(AK.setPitchRangeMoving((-10, 12, 4), 0, -90,0, 1000))
	time.sleep(1.5)
	print(AK.setPitchRangeMoving((-8, 12, 4), 0, -90,0, 1000))
	time.sleep(1.5)
	print(AK.setPitchRangeMoving((-6, 12, 4), 0, -90,0, 1000))
	time.sleep(1.5)
	print(AK.setPitchRangeMoving((-4, 12, 4), 0, -90,0, 1000))
	time.sleep(2)
	print("side 2")
	print(AK.setPitchRangeMoving((-12, 20, 4), 0, -90,0, 1000))
	time.sleep(1)
	print(AK.setPitchRangeMoving((-12, 18, 4), 0, -90,0, 1000))
	time.sleep(2)
	print(AK.setPitchRangeMoving((-12, 16, 4), 0, -90,0, 1000))
	time.sleep(1)
	print(AK.setPitchRangeMoving((-12, 14, 4), 0, -90,0, 1000))
	time.sleep(1)
	print(AK.setPitchRangeMoving((-12, 12, 4), 0, -90,0, 1000))
	time.sleep(1)
	print(AK.setPitchRangeMoving((-12, 10, 4), 0, -90,0, 1000))
	print("side 3") 
	time.sleep(2)
	
if __name__ == "__main__":
    AK = ArmIK()
    camera = Camera()
    print("Exit")
