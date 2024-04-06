import pickle
PATH="/root/OpenPCDet/output/pillar_mamba_pkl"


def save_to_pkl(data, fname, path=PATH):
    with open('%s/%s.pkl'%(path, fname), 'wb') as f:
        pickle.dump(data, f)

def load_from_pkl(fname, path=PATH):
    with open('%s/%s.pkl'%(path, fname), 'rb') as f:
        return pickle.load(f)