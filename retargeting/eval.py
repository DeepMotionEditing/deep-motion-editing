import os
from models import create_model
from datasets import create_dataset, get_character_names
import option_parser
import torch


def main():
    parser = option_parser.get_parser()
    args = parser.parse_args()
    test_device = args.cuda_device
    eval_seq = args.eval_seq

    para_path = os.path.join(args.save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = parser.parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq
    character_names = get_character_names(args)

    dataset = create_dataset(args, character_names)

    model = create_model(args, character_names, dataset)
    model.load(epoch=20000)

    for i, motions in enumerate(dataset):
        print('[{}/4] Running on test {}'.format(args.eval_seq, i))

        model.set_input(motions)
        model.test()


if __name__ == '__main__':
    main()
