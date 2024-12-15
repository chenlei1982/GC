"""
Export a custom dims feature matrix
"""
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'
if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This program is used to export a custom dims feature matrix.", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datapath", type=str, help="The path of dataset.", required=True)
    parser.add_argument("-i", "--idxpath", type=str, help="The path of indexes file.", required=True)
    parser.add_argument("-b", "--start", type=int, help="An integer number specifying at which position to start. Default is 0", required=True, default=0)
    parser.add_argument("-s", "--step", type=int, help="An integer number specifying the incrementation. Default is 1", default=1)
    parser.add_argument("-e", "--stop", type=int, help="An integer number specifying at which position to endt.", default=10)
    args = parser.parse_args()

    df_features = pd.read_csv(args.datapath, encoding='utf8') # 读取整体特征矩阵
    df_idxs = pd.read_table(args.idxpath, header=None) # 读取特征索引文件

    print("\nThe shape of features matrix: ", df_features.shape)
    print("\nThe shape of indexes file: ", df_idxs.shape)
    print("\nStart:{0}, Stop:{1}, Step:{2}".format(args.start, args.stop, args.step))
    print("\nStart exporting custom features matrix...")

    # 根据步长大小等参数迭代输出不同维度特征矩阵
    for x in range(args.start, args.stop+1, args.step):
        if x == 0:
            pass
        else:
            new_features = df_features.iloc[:, np.insert(df_idxs.loc[:(x-1)].values.T[0], 0, 0)]
            new_features.to_csv('{0}.csv'.format(x), index=None)
            # print(df_features.iloc[:, np.insert(df_idxs.loc[:(x-1)].values.T[0], 0, 0)])
    
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(((end_time - start_time) / 60), (end_time - start_time)))