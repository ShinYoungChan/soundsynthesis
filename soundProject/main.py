import numpy as np
import math
import librosa
import soundfile as sf
from collections import deque
import copy
import matplotlib.pyplot as plt
import json
import time
import datetime
import os


class Clustering:
    delta = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

    def init(self, Frame, Row, Col):
        self.frame = Frame
        self.row = Row
        self.col = Col
        self.div = 1 / self.row

    def velsize(self, a):
        return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

    def dist2scalar(self, soundpos, userpos):
        relative = [d - u for d, u in zip(soundpos, userpos)]
        dScalar = math.sqrt(relative[0] ** 2 + relative[1] ** 2)
        return dScalar

    def distanceCalc(self, beforeCenter, currentCenter, beforeVel, currentVel):
        accList = []
        for idx, i in enumerate(currentCenter):
            mindist = 1000000
            minId = 0
            for id, j in enumerate(beforeCenter):
                dist = self.dist2scalar(i, j)
                if mindist > dist:
                    mindist = dist
                    minId = id
            accList.append([d - u for d, u in zip(currentVel[idx], beforeVel[minId])])
        return accList

    def grid3by3(self, x, y, arr):
        particleSum = arr[x][y]
        for i in self.delta:
            mx = x + i[0]
            my = y + i[1]
            if mx < 0 or mx > self.row - 1 or my < 0 or my > self.col - 1: continue
            particleSum += arr[mx][my]
        return particleSum

    def oriClustering(self, particle):
        labelList = []
        for i in range(self.row):
            for j in range(self.col):
                count = 0
                level = []
                if particle[i][j] != 0:
                    for k in self.delta:
                        mx = i + k[0]
                        my = j + k[1]
                        if mx < 0 or mx > self.row - 1 or my < 0 or my > self.col - 1:
                            continue
                        if particle[mx][my] != 0:
                            for ind, m in enumerate(labelList):
                                if (mx, my) in m:
                                    if ind in level:
                                        break
                                    level.append(ind)
                                    count += 1
                                    break
                    level.sort()
                    if count == 0:
                        labelList.append([(i, j)])
                    elif count == 1:
                        labelList[level[0]].append((i, j))
                    else:
                        asd = [(i, j)]
                        for lev in reversed(level):
                            asd.extend(labelList[lev])
                            labelList.pop(lev)
                        labelList.append(asd)
        return labelList

    def myClustering(self, candidate):
        labelList = []
        for i in range(self.row):
            for j in range(self.col):
                count = 0
                level = []
                if (i, j) in candidate:
                    for k in self.delta:
                        mx = i + k[0]
                        my = j + k[1]
                        if mx < 0 or mx > self.row - 1 or my < 0 or my > self.col - 1:
                            continue
                        if (mx, my) in candidate:
                            for ind, m in enumerate(labelList):
                                if (mx, my) in m:
                                    if ind in level:
                                        break
                                    level.append(ind)
                                    count += 1
                                    break
                    if count == 0:
                        labelList.append([(i, j)])
                    elif count == 1:
                        labelList[level[0]].append((i, j))
                    else:
                        level.sort()
                        asd = [(i, j)]
                        for lev in reversed(level):
                            asd.extend(labelList[lev])
                            labelList.pop(lev)
                        labelList.append(asd)
        # print("labelList = ", labelList)
        # print("listsize = ", len(labelList))
        # print("0 size = ", len(labelList[0]))
        return labelList

    def setclustering(self, sceneLength, readPath, savePath):
        clusteringData = {'center': [], 'acc': [], 'vel': [], 'vellength': [], 'gridNum': [], 'particleNum': []}
        centerList = []
        accList = []
        velList = []
        vellenList = []
        gridNumList = []
        particleNumList = []
        beforeFrameVel = []
        beforeFrameCenter = []
        for f in range(0, sceneLength, self.frame):
            print("frame = ", f)
            sceneframe = readPath + str(f) + ".txt"
            vel = np.zeros((self.row, self.col, 3))
            particle = np.zeros((self.row, self.col))
            labelids = np.zeros((self.row, self.col)) - 1
            center = np.zeros((self.row, self.col, 2))
            with open(sceneframe, 'r', encoding='utf-8') as read_file:
                read_file.readline()
                for idx, i in enumerate(read_file):
                    posvel = i.split()
                    posX = float(posvel[0])
                    posY = float(posvel[1])
                    posZ = float(posvel[2])
                    velX = float(posvel[3])
                    velY = float(posvel[4])
                    velZ = float(posvel[5])
                    x = math.ceil(posX / self.div) - 1
                    y = math.ceil(posZ / self.div) - 1
                    vel[x][y][0] += velX
                    vel[x][y][1] += velY
                    vel[x][y][2] += velZ
                    particle[x][y] += 1
                    center[x][y][0] += posX
                    center[x][y][1] += posZ

            # 후보 찾기
            candidate = []
            find = False
            for i in range(self.row):
                for j in range(self.col):
                    check = False
                    if self.grid3by3(i, j, particle) // 1000 >= 25:
                        check = True
                        find = True

                    if check:
                        candidate.append((i, j))

            labelList = []
            if find:
                labelList = self.myClustering(candidate)
            else:
                labelList = self.oriClustering(particle)
            q = deque()
            centerids = [[0 for j in range(2)] for i in range(len(labelList))]
            particleids = [0 for i in range(len(labelList))]
            gridsize = [0 for _ in range(len(labelList))]
            for i in range(len(labelList)):
                for j in range(len(labelList[i])):
                    labelids[labelList[i][j][0]][labelList[i][j][1]] = i
                    gridsize[i] += 1
                    q.append(labelList[i][j])
                    centerids[i][0] += center[labelList[i][j][0]][labelList[i][j][1]][0]
                    centerids[i][1] += center[labelList[i][j][0]][labelList[i][j][1]][1]
                    particleids[i] += particle[labelList[i][j][0]][labelList[i][j][1]]
            # clustering
            while len(q):
                x, y = q.popleft()
                for k in self.delta:
                    mx = x + k[0]
                    my = y + k[1]
                    if mx < 0 or mx > self.row - 1 or my < 0 or my > self.col - 1: continue
                    if labelids[mx][my] == -1 and particle[mx][my] != 0:
                        q.append([mx, my])
                        labelids[mx][my] = labelids[x][y]
                        gridsize[int(labelids[x][y])] += 1
                        centerids[int(labelids[mx][my])][0] += center[mx][my][0]
                        centerids[int(labelids[mx][my])][1] += center[mx][my][1]
                        particleids[int(labelids[mx][my])] += particle[mx][my]
            velsum = [[0, 0, 0] for _ in range(len(labelList))]
            vellen = [0 for _ in range(len(labelList))]
            for i in range(self.row):
                for j in range(self.col):
                    if labelids[i][j] != -1:
                        l = int(labelids[i][j])
                        velsum[l][0] += vel[i][j][0]
                        velsum[l][1] += vel[i][j][1]
                        velsum[l][2] += vel[i][j][2]
                        vellen[l] += self.velsize(vel[i][j])
            # 무게 중심
            centerGravity = []
            for i in range(len(centerids)):
                centerGravity.append([centerids[i][0] / particleids[i], centerids[i][1] / particleids[i]])
            distance = []
            # 사용자와 무게 중심사이 거리
            for i in range(len(centerGravity)):
                distance.append(math.sqrt((1.5 - centerGravity[i][0]) ** 2 + (1.5 - centerGravity[i][1]) ** 2))

            # 가속도 계산
            acc = []
            if len(beforeFrameVel) > 0:
                acc = copy.deepcopy(self.distanceCalc(beforeFrameCenter, centerGravity, beforeFrameVel, velsum))
            else:
                acc = copy.deepcopy(velsum)

            beforeFrameVel = copy.deepcopy(velsum)
            beforeFrameCenter = copy.deepcopy(centerGravity)

            centerList.append(centerGravity)
            accList.append(acc)
            velList.append(velsum)
            vellenList.append(vellen)
            gridNumList.append(gridsize)
            particleNumList.append(particleids)

        clusteringData['center'] = centerList
        clusteringData['acc'] = accList
        clusteringData['vel'] = velList
        clusteringData['vellength'] = vellenList
        clusteringData['gridNum'] = gridNumList
        clusteringData['particleNum'] = particleNumList
        with open(savePath, 'w', encoding="utf-8") as write_file:
            json.dump(clusteringData, write_file)


# gamma correction  감마값을 density로하고 x값을 사운드 데이터로 설정
def gammacurve(gamma, duration):
    gain = 1
    offset = 0
    x = [i / duration for i in range(0, duration)]
    y = [gain * i ** gamma + offset for i in reversed(x)]
    return y


def interpolation(p1, p2, d1):
    return (1 - d1) * p1 + d1 * p2


def getcos(velocity, center, user):
    center.insert(1, 0.0)
    relativepos = [d - u for d, u in zip(center, user)]
    dot = velocity[0] * relativepos[0] + velocity[1] * relativepos[1]
    vScalar = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    uScalar = math.sqrt(relativepos[0] ** 2 + relativepos[1] ** 2)
    return dot / (vScalar * uScalar + 1.0e-20)


def distscalar(soundpos, userpos):
    # print(soundpos)
    if len(soundpos) == 2:
        soundpos.insert(1, 0.0)
    relative = [d - u for d, u in zip(soundpos, userpos)]
    dScalar = math.sqrt(relative[0] ** 2 + relative[1] ** 2 + relative[2] ** 2)
    return dScalar


def magtodb(mag):
    return 10 * math.log10(mag ** 2)


def dbtomag(db):
    return (10 ** (db / 10)) ** 0.5


def single_amp(sound, curve):
    nsound = list(map(lambda x: sound[x] * curve[x], range(len(sound))))
    return nsound


if __name__ == '__main__':
    start = time.time()
    # clips = []
    # Lmax = 0
    # Lmin = 1
    # Ls = []
    # clipData = {'clip': [], 'Ls': []}
    # for i in range(1, 15):
    #     sPath = "C:/Users/zxc78/Desktop/소리모음/파도소리/wave" + str(i) + ".wav"
    #     try:
    #         sound, sr = librosa.load(sPath)
    #     except:
    #         print(i)
    #         continue
    #     for j in range(0, len(sound), 4456 + 4456):
    #         y1 = sound[j:j + 4456 + 4456].tolist()
    #         if len(y1) < 4456 + 4456: break
    #         clips.append(y1)
    #         Lsum = sum(list(map(abs, y1)))
    #         Ls.append(Lsum)
    #         if Lmax < Lsum:
    #             Lmax = Lsum
    #         elif Lmin > Lsum:
    #             Lmin = Lsum
    # for i in range(len(Ls)):
    #     # print("Ls = ", Ls[i], ", Lmax = ", Lmax, ", Lmin = ", Lmin)
    #     Ls[i] = (Ls[i] - Lmin) / (Lmax - Lmin)
    #     if Ls[i] < 1.0e-5:
    #         Ls[i] = 100
    # clipData['clip'] = clips
    # clipData['Ls'] = Ls
    # path = "C:/Users/zxc78/PycharmProjects/watersimulation/clipData/clip.json"
    # with open(path, 'w', encoding="utf-8") as write_file:
    #     json.dump(clipData, write_file)
    ############################################
    path = "C:/Users/zxc78/PycharmProjects/watersimulation/clipData/clip.json"
    with open(path, 'r', encoding="utf-8") as read_file:
        clipData = json.load(read_file)
    clips = clipData['clip']
    Ls = clipData['Ls']
    for i in range(len(Ls)):
        if Ls[i] < 1.0e-5:
            Ls[i] = 100
    cooltime = [0 for _ in range(len(clips))]
    view = [0.5, 0.5, 0.5]
    timestep = 0.2
    isSaved = False
    isTimedelay = True
    # sceneFrameList
    # scenePath = os.listdir("E:/soundFrame/scene1 (tornado)")
    # readPath = "E:/soundFrame/scene1 (tornado)/scene1_frame"
    # scenePath = os.listdir("E:/soundFrame/scene2 (u-shaped corridor)")
    # readPath = "E:/soundFrame/scene2 (u-shaped corridor)/scene2_frame"
    # scenePath = os.listdir("E:/soundFrame/scene3 (moving two boxes)")
    # readPath = "E:/soundFrame/scene3 (moving two boxes)/scene3_frame"
    # scenePath = os.listdir("E:/soundFrame/scene4 (spinning emitter)")
    # readPath = "E:/soundFrame/scene4 (spinning emitter)/scene4_frame"
    # scenePath = os.listdir("E:/soundFrame/scene5 (propeller)")
    # readPath = "E:/soundFrame/scene5 (propeller)/scene5_frame"
    savePath = "./sceneData/scene5.json"
    # dirlength = len(scenePath)
    clusteringData = Clustering()
    clusteringData.init(12, 30, 30)  # frame, row, col
    if isSaved:
        print("save")
        # clusteringData.setclustering(dirlength, readPath, savePath)
    else:
        with open(savePath, 'r', encoding="utf-8") as read_file:
            data = json.load(read_file)
        centerGravity = data['center']
        velList = data['vel']  # velocity vector
        accList = data['acc']  # acc vector
        velLength = data['vellength']  # velocity scalar
        gridsize = data['gridNum']
        particlenum = data['particleNum']
        # 총 읽은 프레임 갯수
        frameLength = len(centerGravity)
        maxAcc = 0
        waveSound = []
        acc = []
        vel = []
        accsize = []
        # 가속도 최대값 찾기
        # for i in range(len(accList)):
        for i in range(len(velList)):
            if not accList[i]:
                accsize = [0]
            else:
                # accsize = [clusteringData.velsize(a) for a in accList[i]]
                accsize = [clusteringData.velsize(a) for a in velList[i]]
            acc.append(accsize)
            if maxAcc < max(accsize):
                maxAcc = max(accsize)
        for i in range(0, frameLength):
            soundList = []
            if acc[i] == [0]:
                soundList = [0 for _ in range(4456 + 4456)]
            else:
                # density = [p / g for p, g in zip(particlenum[i], gridsize[i])]
                for idx, a in enumerate(acc[i]):  # 원래는 velsum 지금은 가속도로 계산하기에 acc 활용중
                    v1 = a / maxAcc * 0.8
                    Lmap = 5.733e21 * math.exp(-((v1 - 3.223) / 0.3117) ** 2) + 8.345e7 * math.exp(
                        -((v1 - 9.621) / 1.985) ** 2)
                    sList = []
                    soundCan = []
                    period = 0
                    flag = True
                    while flag:
                        for j in range(len(clips)):
                            if abs(Lmap - Ls[j]) < 0.01 + 0.05 * period and cooltime[j] == 0:
                                soundCan.append([abs(Lmap - Ls[j]), clips[j], j])
                                flag = False
                        period += 1

                    soundCan = min(soundCan)
                    sList = copy.deepcopy(soundCan[1])
                    cooltime[soundCan[2]] = 30
                    # print(Lmap, Ls[soundCan[2]], soundCan[2])
                    # if not soundList[i]:
                    #     soundList[i].extend(sList)
                    # else:
                    #     # 로그 씌워서 더할 것.
                    #     soundList[i] = [s1 + s2 for s1, s2 in zip(soundList[i], sList)]
                    # for i in range(len(sList)):
                    #     print("sList = ", sList[i], ", db = ", magtodb(sList[i]), ", mag = ", dbtomag(magtodb(sList[i])))
                    #     sList[i] = 10 ** (sList[i] / 10)
                    r = 1.0
                    dist = distscalar(centerGravity[i][idx], view) / r
                    cos = getcos(accList[i][idx], centerGravity[i][idx], view)
                    dist *= (cos + 2) / 3
                    if dist > 1.0:
                        for j in range(len(sList)):
                            if sList[j] > 0:
                                m = magtodb(sList[j])
                                avg = m - 20 * math.log10(dist)
                                sList[j] = dbtomag(avg)
                            elif sList[j] < 0:
                                m = magtodb(-sList[j])
                                avg = m - 20 * math.log10(dist)
                                sList[j] = -dbtomag(avg)
                    if not soundList:
                        soundList.extend(sList)
                    else:
                        soundList = [s1 + s2 for s1, s2 in zip(soundList, sList)]
                for c in range(len(cooltime)):
                    if cooltime[c] > 0:
                        cooltime[c] -= 1
                soundList = [qwe / len(velList[i]) for qwe in soundList]
                # soundList = [qwe / len(accList[i]) for qwe in soundList]
            print(timestep)
            # interpolation
            if not waveSound:
                waveSound.extend(soundList)
            else:
                waveSound[-4456:] = [interpolation(s1, s2, idx / 4456) for idx, (s1, s2) in
                                     enumerate(zip(waveSound[-4456:], soundList[:4456]))]
                waveSound.extend(soundList[4456:])
            timestep += 0.2

        scenewave = 'scene5wavesvel08testtest.wav'
        waveSound[44560:66840] = [0 for _ in range(22280)]
        if isTimedelay:
            print("timedelay")
            timeline = []
            glist = []
            curvelist = []
            for i in range(0, len(waveSound), 4456):
                timeline.append(i)
            for i in range(0, len(timeline)-1):
                if len(waveSound[timeline[i]:timeline[i] + 4456]) < 4456: break
                pNum = True
                gamma = (sum(waveSound[timeline[i+1]:timeline[i+1] + 4456]) - sum(
                    waveSound[timeline[i]:timeline[i] + 4456])) / (timeline[i+1] - timeline[i])
                gamma += 0.0000001
                if gamma < 0:
                    pNum = False
                if not pNum:
                    gamma *= -1
                while gamma < 1.0:
                    gamma *= 10
                if pNum:
                    gamma = 1 / gamma
                curvelist.append(gammacurve(gamma, 4456))
            delaytime = 4456
            for i in range(0, len(timeline)-1):
                # if len(waveSound[timeline[i]:timeline[i] + 4456]) < 4456: break
                pSound = single_amp(waveSound[timeline[i]:timeline[i] + 4456], curvelist[i])
                print(timeline[i])
                # plt.plot(pSound)
                # plt.show()
                # print("timeline = ", timeline[i+1], ", 1second delay = ", timeline[i+1]+22280)
                waveSound[timeline[i]+44560:timeline[i] + 44560+4456] = [s1 + s2 for (s1, s2) in
                                                             zip(waveSound[timeline[i]+44560:timeline[i] + 44560+4456], pSound)]
            plt.plot(waveSound)
            plt.show()
            scenewave = scenewave[:-4] + 'timedelay.wav'
        sr = 22050
        sf.write(scenewave, waveSound, sr, subtype='PCM_16')
    end = time.time()
    sec = end - start
    result = datetime.timedelta(seconds=sec)
    print(result)
