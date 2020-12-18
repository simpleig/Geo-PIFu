import argparse
from Constants import consts
import numpy as np
import os
import glob
import json
import pdb # pdb.set_trace()

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--totalNumFrame', type=int, default="108720", help="total data number: N*M'*4 = 6795*4*4 = 108720")
    parser.add_argument('--trainingDataRatio', type=float, default="0.8")
    parser.add_argument('--datasetDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender")
    parser.add_argument('--resultsDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender/deepHumanResults/expName")

    args = parser.parse_args()

    return args

def get_training_test_indices(args, shuffle):

    # sanity check for args.totalNumFrame
    assert(os.path.exists(args.datasetDir))
    totalNumFrameTrue = len(glob.glob(args.datasetDir+"/config/*.json"))
    assert((args.totalNumFrame == totalNumFrameTrue) or (args.totalNumFrame == totalNumFrameTrue+len(consts.black_list_images)//4))

    max_idx = args.totalNumFrame # total data number: N*M'*4 = 6795*4*4 = 108720
    indices = np.asarray(range(max_idx))
    assert(len(indices)%4 == 0)

    testing_flag = (indices >= args.trainingDataRatio*max_idx)
    testing_inds = indices[testing_flag] # 21744 testing indices: array of [86976, ..., 108719]
    testing_inds = testing_inds.tolist()
    if shuffle: np.random.shuffle(testing_inds)
    assert(len(testing_inds) % 4 == 0)

    training_inds = indices[np.logical_not(testing_flag)] # 86976 training indices: array of [0, ..., 86975]
    training_inds = training_inds.tolist()
    if shuffle: np.random.shuffle(training_inds)
    assert(len(training_inds) % 4 == 0)

    return training_inds, testing_inds

def main(args):

    # get training/test data indices
    training_inds, testing_inds = get_training_test_indices(args=args,shuffle=False)
    
    # init.
    norm_cos_dis_front = []
    norm_l2_dis_front  = []
    chamfer_dis_list, estV_2_gtM_dis_list = [], []

    # read eval metrics
    count = 0
    for idx in testing_inds:

        # logs
        if ("%06d"%(idx)) in consts.black_list_images: continue
        expName = args.resultsDir.split("/")[-1]
        print("%s read metrics %06d/%06d..." % (expName, count, len(testing_inds)-len(consts.black_list_images)))
        count += 1

        # get path of evalMetrics.json
        evalMetricsPath            = "%s/%06d_evalMetrics.json"            % (args.resultsDir,idx)
        evalMetricsPath_additional = "%s/%06d_evalMetrics_additional.json" % (args.resultsDir,idx)
        assert(os.path.exists(evalMetricsPath) and os.path.exists(evalMetricsPath_additional))

        # read evalMetrics.json
        with open(evalMetricsPath) as f: evalMetrics = json.load(f)
        norm_cos_dis_ft = evalMetrics["norm_cos_dis_ft"]
        norm_l2_dis_ft  = evalMetrics["norm_l2_dis_ft"]
        with open(evalMetricsPath_additional) as f: evalMetrics_additional = json.load(f)
        chamfer_dis    = evalMetrics_additional["chamfer_dis"]
        estV_2_gtM_dis = evalMetrics_additional["estV_2_gtM_dis"]

        # save eval metrics
        norm_cos_dis_front.append(norm_cos_dis_ft[0])
        norm_l2_dis_front.append(norm_l2_dis_ft[0])

        chamfer_dis_list.append(chamfer_dis)
        estV_2_gtM_dis_list.append(estV_2_gtM_dis)

    # compute average eval metrics
    avgEvalMetrics = {"avg_norm_cos_dis_front": np.mean(norm_cos_dis_front),
                      "avg_norm_l2_dis_front": np.mean(norm_l2_dis_front),
                      "avg_chamfer_dis": np.mean(chamfer_dis_list),
                      "avg_estV_2_gtM_dis": np.mean(estV_2_gtM_dis_list)
                     }
    print("\n\n\n\nAverage evaluation metrics:\n\n{}".format(avgEvalMetrics))
    avgEvalMetricsPath = "%s/avgEvalMetrics.json" % (args.resultsDir)
    with open(avgEvalMetricsPath, 'w') as outfile: json.dump(avgEvalMetrics, outfile)
    visualCheck = True
    if visualCheck:
        print("check average eval metrics json results...")
        os.system("cp %s ./examples/avgEvalMetrics.json" % (avgEvalMetricsPath))
        pdb.set_trace()

if __name__ == '__main__':

    # parse args.
    args = parse_args()

    # main function
    main(args=args)