import argparse
import FFMPEGFrames


#Code Used from https://github.com/alibugra/frame-extraction


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output",default="data/images", required=False)
ap.add_argument("-ow", "--wav_output",default="data/audio", required=False)
ap.add_argument("-i", "--input",default="data/videos", required=False)
ap.add_argument("-f", "--fps",default=5, required=False)
args = vars(ap.parse_args())

input = args["input"]
output=args["output"]
fps = args["fps"]
wav_output=args["wav_output"]

f = FFMPEGFrames.FFMPEGFrames(output,wav_output,input,fps)
f.extract_all_video()
