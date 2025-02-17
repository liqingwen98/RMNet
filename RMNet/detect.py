import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import DataLoader
from .model import SiteLevelRMNet
from .dataset import RMNetDataSet


def detect_m6a(args):
    model = SiteLevelRMNet()
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(args.ckpt)
    model.read_level_model.load_state_dict(checkpoint, strict=True)

    test_dataset = RMNetDataSet(args.input_file)
    test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        site_info = []
        pred_m6a = []
        for batch_features_all in tqdm(test_dl):
            features, sampl_info = batch_features_all[1].cuda(), batch_features_all[0]
            y_pred = model(features)
            y_pred = y_pred.tolist()
            for i in range(len(y_pred)):
                if y_pred[i] < 0.5:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            site_info.extend(sampl_info)
            pred_m6a.extend(y_pred)
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
        with open(os.path.join(args.save_dir, 'result.tsv'), 'a') as file:
            for row in range(len(site_info)):
                file.write(site_info[row]+'\t'+str(pred_m6a[row])+'\n')

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", default='', help="Path to the input file.", required=True)
    parser.add_argument("--save_dir", default='', help="Directory to save the output results.", required=True)
    parser.add_argument("--ckpt", default='', help="Path to the checkpoint file.", required=True)
    return parser


def main():
    args = argparser().parse_args()
    detect_m6a(args)

if __name__ == '__main__':
    main()
