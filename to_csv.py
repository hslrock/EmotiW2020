import os
import re
import pandas as pd
import argparse

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


base="data/videos/"
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory",default="Test", required=False)
args = vars(ap.parse_args())

target=args["directory"]

file_list=os.listdir(base+target)
file_list.sort(key=natural_keys)
file_list_filtered=[]

for i in file_list:
    if i.endswith('.mp4'):
        file_list_filtered.append(os.path.splitext(i)[0])
df=pd.DataFrame(file_list_filtered ,columns=["Vid_name"])
df.to_csv(target+" files.csv",index=False)


