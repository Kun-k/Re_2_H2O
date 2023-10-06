import pandas as pd
import numpy as np
import os
import argparse

def route_cut_highD(rootpath, filenum=2, num_agents=2, dis_threshold=30, starttime=None):
    """
    Args:
        rootpath (str): Root path of the 'tracks' directory in the highD dataset,
                        For example, if the path is "./highD-dataset/data/02_tracks",
                        you should input "./highD-dataset/data/".
        filenum (int): File number in the highD dataset.
        num_agents (int): Number of vehicles in the truncated traffic flow.
        dis_threshold (int, meter): Maximum distance threshold between vehicles in the initial state.
        starttime (int): Starting time step for truncation. If None or an unreasonable time is provided,
                         a random time step will be generated.

    Return:
        Save the truncated traffic flow in the original file format and return the file path.
        Saved file name: "(filenum)_(num_agents)_(dis_threshold)_(starttime).csv"
        File saving path: rootpath + "dis_(dis_threshold)_car_(num_agents)/(filenum)_track_cut/"
    """

    if os.path.exists(rootpath + "dis_" + str(dis_threshold) + "_car_" + str(num_agents)) is False:
        os.mkdir(rootpath + "dis_" + str(dis_threshold) + "_car_" + str(num_agents))
    savepath = rootpath + "dis_" + str(dis_threshold) + "_car_" + str(num_agents) + "/%02d" % filenum + "_track_cut/"
    if os.path.exists(savepath) is False:
        os.mkdir(savepath)
    savepath = savepath + "%02d" % filenum + \
               "_" + str(num_agents) + \
               "_" + str(dis_threshold) + \
               "_" + str(starttime) + ".csv"

    filepath = rootpath + "%02d" % filenum
    track_file_name = filepath + "_tracks.csv"
    track_data = pd.DataFrame(pd.read_csv(track_file_name))
    track_data = track_data[track_data.xVelocity > 0]
    track_data = track_data.sort_values(by=["frame", "x"], ascending=True, ignore_index=True)
    times_try = 50
    for i in range(times_try):
        if (starttime is None) or track_data[track_data.frame == starttime].shape[0] < num_agents:
            starttime = np.random.randint(track_data["frame"][0], track_data["frame"][track_data.shape[0] - 1])
        else:
            track_data_cut = track_data[track_data.frame >= starttime].reset_index(drop=True)
            x_list = np.sort(np.array(track_data_cut.x[0: num_agents]))
            delta_x = x_list[1:num_agents] - x_list[0:num_agents - 1]
            if (abs(delta_x) < dis_threshold).all():
                break
            starttime = np.random.randint(track_data["frame"][0], track_data["frame"][track_data.shape[0] - 1])
        if i == times_try - 1:
            return None
    carid = []
    for i in range(num_agents):
        carid.append(track_data_cut.id[i])
    track_data_cut = track_data_cut[track_data_cut.id.isin(carid)]
    track_data_cut = track_data_cut.sort_values(by=["id", "frame"], ascending=True, ignore_index=True)
    if os.path.exists(savepath) is False:
        track_data_cut.to_csv(savepath, encoding="GBK")
        return savepath
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rootpath', type=str, default="./highD-dataset/data/")
    parser.add_argument('num_agents', type=int, default=2)
    parser.add_argument('dis_threshold', type=float, default=25)
    args = parser.parse_args()
    rootpath = args.rootpath
    num_agents = args.num_agents
    dis_threshold = args.dis_threshold

    road_3_filenum = [1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 统计3车道highD数据
    for filenum in road_3_filenum:
        for i in range(20):
            j = 0
            filepath = route_cut_highD(rootpath, filenum, num_agents=num_agents, dis_threshold=dis_threshold)
            while filepath is None and j < 10:
                filepath = route_cut_highD(rootpath, filenum, num_agents=num_agents, dis_threshold=dis_threshold)
                j += 1
            if filenum is None:
                print(None)
            else:
                print("%02d" % filenum, ", ", i, ",", filepath)
