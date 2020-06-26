import os
import subprocess

#Code Used from https://github.com/alibugra/frame-extraction
class FFMPEGFrames:
    def __init__(self, output,wav_output,folder_path,fps,):
        self.output = output
        self.wav_output=wav_output
        self.folder_path=folder_path
        self.labels = [f for f in os.listdir(folder_path) if not os.path.isfile(f)]
        self.fps=fps ##Number of frame per second
        
        
                    
    def extract_all_video(self): #Extract all videos in master folder
        for label in self.labels:
            self.extract_folder(label)
            
            
    def extract_folder(self,label): #Extract videos in sub-folder (label)
        videos_path=os.path.join(self.folder_path,label)
        for file in os.listdir(videos_path):
            if file.endswith(".mp4"):
                self.extract_frames(os.path.join(videos_path,file),self.fps)
                self.extract_wav(os.path.join(videos_path,file),label)

    def extract_wav(self,input,label): #Extracting wav files
        output = input.split(self.folder_path)[-1].split('.')[0]
        if not os.path.exists(self.wav_output+'/'+label):
            os.makedirs(self.wav_output+'/'+label)
                                                                  
        command = "ffmpeg -i " +input+ " -vn -acodec pcm_s16le -ar 44100 -ac 2 "+self.wav_output+output+".wav"
        response = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE).stdout.read()
        
    def extract_frames(self, input, fps): #Extracting video frames
        output = input.split(self.folder_path)[-1].split('.')[0]

        if not os.path.exists(self.output + output):
            os.makedirs(self.output + output)

        query = "ffmpeg -i " + input + " -vf fps=" + str(fps) + " " + self.output + output + "/%06d.jpg"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        
