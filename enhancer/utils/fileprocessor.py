import glob
import os
import numpy as np
from scipy.io import wavfile

class Fileprocessor:

    def __init__(
        self,
        clean_dir,
        noisy_dir,
        sr = 16000,
        matching_function = None
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sr = sr
        self.matching_function = matching_function

    @classmethod
    def from_name(cls,
                name:str,
                clean_dir,
                noisy_dir,
                sr,
                matching_function=None
        ):

        if name.lower() == "vctk":
            return cls(clean_dir,noisy_dir,sr, Fileprocessor.match_vtck)
        elif name.lower() == "dns-2020":
            return cls(clean_dir,noisy_dir,sr, Fileprocessor.match_dns2020)
        else:
            return cls(clean_dir,noisy_dir,sr, matching_function)

    def prepare_matching_dict(self):

        if self.matching_function is None:
            raise ValueError("Not a valid matching function")

        return self.matching_function(self.clean_dir,self.noisy_dir,self.sr)

    @staticmethod
    def match_vtck(clean_path,noisy_path,sr):

        matching_wavfiles = list()
        clean_filenames = [file.split('/')[-1] for file in glob.glob(os.path.join(clean_path,"*.wav"))]
        noisy_filenames = [file.split('/')[-1] for file in glob.glob(os.path.join(noisy_path,"*.wav"))]
        common_filenames = np.intersect1d(noisy_filenames,clean_filenames)

        for file_name in common_filenames:

             sr_clean, clean_file = wavfile.read(os.path.join(clean_path,file_name))
             sr_noisy, noisy_file = wavfile.read(os.path.join(noisy_path,file_name))
             if ((clean_file.shape[-1]==noisy_file.shape[-1]) and 
                    (sr_clean==sr) and 
                        (sr_noisy==sr)):
                matching_wavfiles.append(
                                    {"clean":os.path.join(clean_path,file_name),"noisy":os.path.join(noisy_path,file_name),
                                    "duration":clean_file.shape[-1]/sr}
                                    )
        return matching_wavfiles

    @staticmethod
    def match_dns2020(clean_path,noisy_path,sr):
        
        matching_wavfiles = dict()
        clean_filenames = [file.split('/')[-1] for file in glob.glob(os.path.join(clean_path,"*.wav"))]
        for clean_file in clean_filenames:
            noisy_filenames = glob.glob(os.path.join(noisy_path,f"*_{clean_file}.wav"))
            for noisy_file in noisy_filenames:

                sr_clean, clean_file = wavfile.read(os.path.join(clean_path,clean_file))
                sr_noisy, noisy_file = wavfile.read(noisy_file)
                if ((clean_file.shape[-1]==noisy_file.shape[-1]) and 
                        (sr_clean==sr) and 
                            (sr_noisy==sr)):
                    matching_wavfiles.update(
                                        {"clean":os.path.join(clean_path,clean_file),"noisy":noisy_file,
                                        "duration":clean_file.shape[-1]/sr}
                                        )

        return matching_wavfiles




        
        


        


