def get_character_names(args):
    if args.is_train:
        raise Exception('Training data coming soon')

    else:
        characters = [['BigVegas', 'BigVegas', 'BigVegas', 'BigVegas'],  ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]
        tmp = characters[1][args.eval_seq]
        characters[1][args.eval_seq] = characters[1][0]
        characters[1][0] = tmp

        return characters


def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData

    if args.is_train:
        raise Exception('Training data coming soon')
    else:
        return TestData(args, character_names)


def get_test_set():
    file = open('./datasets/Mixamo/test_list.txt', 'r')
    list = file.readlines()
    list = [f[:-1] for f in list]
    return list
