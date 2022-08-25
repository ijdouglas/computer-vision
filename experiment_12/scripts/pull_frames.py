import argparse
import pandas as pd
import os.path as osp
import os
from shutil import copy

multi_root = "/data/drives/multiwork"

def get_subj_table():
    df = pd.read_csv(osp.join(multi_root, "subject_table.txt"), sep='\t', names=["subID", "exp", "date", "kidID"])
    return df

def main(args):
    subj_table = get_subj_table()
    frames = pd.read_csv(args.frames_file)

    data_txt = pd.DataFrame(columns=["path", "obj_id", "subj", "img_num", "event_id", 'instanceID'])

    eid = 0

    for s, subj in frames.groupby("subj"):
        print(s)
        st_entry = subj_table.query("subID == {}".format(s)).iloc[0]
        subdir = "__{}_{}".format(st_entry.date, int(st_entry.kidID))
        subj_multi_dir = osp.join(multi_root, "experiment_{}".format(args.exp), "included", subdir)
        cam07 = osp.join(subj_multi_dir, "cam07_frames_p")
        cam08 = osp.join(subj_multi_dir, "cam08_frames_p")

        cam07_out = osp.join(args.out_dir, subdir, "cam07_frames_p")
        cam08_out = osp.join(args.out_dir, subdir, "cam08_frames_p")

        if not osp.isdir(cam07_out):
            os.makedirs(cam07_out)
        if not osp.isdir(cam08_out):
            os.makedirs(cam08_out)

        for i, inst in subj.groupby("instanceID"):
            start = inst.onset.iloc[0]
            end = inst.offset.iloc[0]

            # only use complete 90 frame chunks (i.e. 3 seconds centered around labeling)
            if end - start == args.frame_count or args.frame_count is None:
                frames = list(range(start, end+1))
                for f in frames:
                    img = "img_{}.jpg".format(f)
                    try:
                        copy(osp.join(cam07, img), osp.join(cam07_out, img))
                        copy(osp.join(cam08, img), osp.join(cam08_out, img))
                    except Exception:
                        print("failed to copy file: {}".format("img_{}.jpg".format(f)))

                    data_txt = data_txt.append({
                        "path": osp.join(subdir, "cam07_frames_p", img),
                        "obj_id": inst.label.iloc[0],
                        "subj": osp.join(subdir, "cam07_frames_p"),
                        "img_num": f, "event_id": eid, "instanceID": i
                        }, ignore_index=True)

                    # eid += 1

                    data_txt = data_txt.append({
                        "path": osp.join(subdir, "cam08_frames_p", img),
                        "obj_id": inst.label.iloc[0],
                        "subj": osp.join(subdir, "cam08_frames_p"),
                        "img_num": f, "event_id": eid+1, "instanceID": i
                        }, ignore_index=True)

                    # eid += 1

                eid += 2

    return data_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_file")
    parser.add_argument("--out_dir")
    parser.add_argument("--exp")
    parser.add_argument("--frame_count", default=None)


    args = parser.parse_args()

    df = main(args)

    df.to_csv("../data/dataset_txt/exp{}_attending_3s_whole_img_full.csv".format(args.exp), index=False)
