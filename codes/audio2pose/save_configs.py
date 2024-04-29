import pickle
from utils import config

args = config.parse_args()

with open('camn_config.obj', 'wb') as config_file:
    pickle.dump(args, config_file)