import numpy as np
import dill

def save_dict(dictionary, path):
    keys = list(dictionary.keys())

    values = [list(tens) for tens in list(dictionary.values())]
    values = np.array(values)
    keys_path = path + '_keys'
    values_path = path + '_values'

    with open(keys_path, 'wb+') as f:
        dill.dump(keys, f)
    np.save(values_path, values)