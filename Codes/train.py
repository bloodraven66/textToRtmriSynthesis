import os, sys, time
import argparse 
from data_prep import data_handler
from common.utils import read_yaml, get_model, get_trainer


parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--yaml', type=str, default='config/hparams.yaml')
parser.add_argument('--sub', type=str, default=None)
parser.add_argument('--device', type=str, default=None)
args = parser.parse_args()

def main():
    
    cfg = read_yaml(args.yaml)
    if args.sub is not None:
        cfg = cfg + {'data':{'filter':args.sub, 'apply_filter':True}}
    if args.device is not None:
        assert args.device in ['cuda:0', 'cuda:1']
        cfg = cfg + {'common':{'device':args.device}} 

    loaders = data_handler.collect(cfg)
    model = get_model(cfg)
    
    trainer = get_trainer(cfg)
    
    operate = trainer.Operate(cfg)
    operate.trainer(model, loaders)

if __name__ == '__main__':
    main()
