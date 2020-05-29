import os
from models import create_model
from datasets import create_dataset, get_character_names
import option_parser
import torch
from tqdm import tqdm


def eval(eval_seq, save_dir, test_device='cpu'):
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq
    args.save_dir = save_dir
    character_names = get_character_names(args)

    dataset = create_dataset(args, character_names)

    model = create_model(args, character_names, dataset)
    model.load(epoch=20000)

    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(motions)
        model.test()


if __name__ == '__main__':
    parser = option_parser.get_parser()
    args = parser.parse_args()
    eval(args.eval_seq, args.save_dir, args.cuda_device)
