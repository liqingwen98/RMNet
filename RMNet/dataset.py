from torch.utils.data import Dataset
import linecache
import os
import numpy as np

base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def clear_linecache():
    linecache.clearcache()

def _parse_one_read_level_feature(feature, kmer):
    words = feature.strip().split("||")
    
    mean_sig = np.array([float(x) for x in words[0].split(",")])[0:5]
    median_sig = np.array([float(x) for x in words[1].split(",")])[0:5]
    std_sig = np.array([float(x) for x in words[2].split(",")])[0:5]
    len_sig = np.array([int(x) for x in words[3].split(",")])[0:5]
    qual = np.array([int(x) for x in words[5].split(",")])[0:5]
    mis = np.array([int(x) for x in words[6].split(",")])[0:5]
    ins = np.array([int(x) for x in words[7].split(",")])[0:5]
    dele = np.array([int(x) for x in words[8].split(",")])[0:5]

    return np.concatenate((kmer.reshape(-1, len(kmer)), 
                            mean_sig.reshape(-1, len(kmer)), median_sig.reshape(-1, len(kmer)),
                            std_sig.reshape(-1, len(kmer)), len_sig.reshape(-1, len(kmer)),
                            qual.reshape(-1, len(kmer)), mis.reshape(-1, len(kmer)), 
                            ins.reshape(-1, len(kmer)), dele.reshape(-1, len(kmer)),
                           ), 
                           axis=0)


def generate_features_line(line, sampleing=False):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:3])
    coverage = int(words[3])
    kmer = np.array([base2code[x] for x in words[4]])[0:5]
    features = np.array(words[5:-1])
    assert len(features) == coverage
    if sampleing:
        features = features[np.random.choice(coverage, 20, replace=False)]
    features = np.array([_parse_one_read_level_feature(x, kmer) for x in features])

    return sampleinfo, features


class RMNetDataSet(Dataset):
    def __init__(self, filename):

        self._filename = os.path.abspath(filename)
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = generate_features_line(line)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass
