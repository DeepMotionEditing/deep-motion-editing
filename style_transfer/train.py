import torch
import os
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
import argparse
import importlib

from tensorboardX import SummaryWriter
from data_loader import get_dataloader
from itertools import cycle

from py_utils import write_loss, print_composite, to_float
from probe.latent_plot_utils import get_all_plots
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config')
    return parser.parse_args()

def main(args):
    config_module = importlib.import_module(args.config)
    config = config_module.Config()

    # Load experiment setting
    config.initialize(args)
    max_iter = config.max_iter

    # Dataloader

    train_content_loader = get_dataloader(config, 'train')
    train_class_loader = get_dataloader(config, 'train')
    test_content_loader = get_dataloader(config, 'test')
    test_class_loader = get_dataloader(config, 'test', shuffle=True)
    trainfull_content_loader = get_dataloader(config, 'trainfull', shuffle=True)
    trainfull_class_loader = get_dataloader(config, 'trainfull', shuffle=True)
    test_rec_loader = get_dataloader(config, 'test',  shuffle=True)
    rec_loader = cycle(test_rec_loader)

    # Trainer
    trainer = Trainer(config)
    print("here!")

    tr_info = open(os.path.join(config.info_dir, "info-network"), "w")
    print(trainer.model, file=tr_info)
    tr_info.close()

    trainer.to(config.device)
    iterations = trainer.resume()

    # Summary Writer
    train_writer = SummaryWriter(os.path.join(config.tb_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(config.tb_dir, 'test'))

    layout = {'adversarial acc & loss': {
        'acc': ['Multiline', ['gen_acc_all', 'dis_acc_all']],
        'adv_loss': ['Multiline', ['gen_loss_adv', 'dis_loss_adv_all']]},
        'reconstruction loss':
            { 'gen_loss_recon_all': ['Multiline', ['gen_loss_recon_all']],
              'gen_loss_recon_r': ['Multiline', ['gen_loss_recon_r']],
              'gen_loss_recon_s': ['Multiline', ['gen_loss_recon_s']],
              'gen_loss_recon_u': ['Multiline', ['gen_loss_recon_u']]}
    }
    train_writer.add_custom_scalars(layout)

    it = iterations
    cyc_train_content_loader = cycle(train_content_loader)
    cyc_train_class_loader = cycle(train_class_loader)

    while True:
        it = it + 1
        co_data = next(cyc_train_content_loader)
        cl_data = next(cyc_train_class_loader)

        d_acc = trainer.dis_update(co_data, cl_data)
        g_acc = trainer.gen_update(co_data, cl_data)

        if (iterations + 1) % config.log_freq == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)
            rec_data = next(rec_loader)
            loss_dict, _ = trainer.test_rec(rec_data)
            for key, value in loss_dict.items():
                test_writer.add_scalar(key, value, iterations + 1)

        if ((iterations + 1) % config.mt_save_iter == 0 or (
                iterations + 1) % config.mt_display_iter == 0):
            if (iterations + 1) % config.mt_save_iter == 0:
                key_str = '%08d' % (iterations + 1)
            else:
                key_str = 'current'

            with torch.no_grad():

                """latent codes"""  # !!!!! TD: add a separate function, merge with plot_clusters

                vis_dicts = {}
                for phase, co_loader, cl_loader, writer in [['train', train_content_loader, train_class_loader, train_writer],
                                                            ['test', test_content_loader, test_class_loader, test_writer]]:

                    vis_dict = None
                    for t, tcl_data in enumerate(cl_loader):
                        vis_codes = trainer.get_latent_codes(tcl_data)
                        if vis_dict is None:
                            vis_dict = {}
                            for key, value in vis_codes.items():
                                vis_dict[key] = [value]
                        else:
                            for key, value in vis_codes.items():
                                vis_dict[key].append(value)
                    for key, value in vis_dict.items():
                        if phase == "test" and key == "content_code":
                            continue
                        if key == "meta":
                            secondary_keys = value[0].keys()
                            num = len(value)
                            vis_dict[key] = {secondary_key: [to_float(item) for i in range(num) for item in value[i][secondary_key]]
                                             for secondary_key in secondary_keys}
                        else:
                            vis_dict[key] = torch.cat(vis_dict[key], 0)
                            vis_dict[key] = vis_dict[key].cpu().numpy()
                            vis_dict[key] = to_float(vis_dict[key].reshape(vis_dict[key].shape[0], -1))

                    vis_dicts[phase] = vis_dict

                writers = {"train": train_writer, "test": test_writer}
                get_all_plots(vis_dicts, os.path.join(config.output_dir, key_str), writers, iterations + 1)

                """outputs"""
                for phase, co_loader, cl_loader in [['trainfull', trainfull_content_loader, trainfull_class_loader],
                                                    ['test', test_content_loader, test_class_loader]]:
                    for status in ["3d", "2d"]:
                        name = "%s_%s_%s" % (phase, key_str, status)
                        outputs = {}
                        for t, (tco_data, tcl_data) in enumerate(zip(co_loader, cl_loader)):
                            if t >= config.test_batch_n:
                                break
                            cur_outputs = trainer.test(tco_data, tcl_data, status)
                            for key in cur_outputs.keys():
                                output = cur_outputs[key]
                                if key not in outputs:
                                    outputs[key] = []
                                if isinstance(output, torch.Tensor):
                                    outputs[key].append(output.reshape(output.shape[1:]))
                                else:
                                    outputs[key].append(output)

                        output_path = os.path.join(config.output_dir, name)
                        print("%s saved" % name)
                        torch.save(outputs, output_path)

        if (iterations + 1) % config.save_freq == 0:
            trainer.save(iterations)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
