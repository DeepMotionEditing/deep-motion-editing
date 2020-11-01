import torch
from models.enc_and_dec import AE, StaticEncoder
from models.vanilla_gan import Discriminator
from models.skeleton import build_edge_topology
from models.Kinematics import ForwardKinematics
from datasets.bvh_parser import BVH_file
from option_parser import get_std_bvh
import os


class IntegratedModel:
    # origin_offsets should have shape num_skeleton * J * 3
    def __init__(self, args, joint_topology, origin_offsets: torch.Tensor, device, characters):
        self.args = args
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.fk = ForwardKinematics(args, self.edges)

        self.height = [] # for normalize ee_loss
        self.real_height = []
        for char in characters:
            if args.use_sep_ee:
                h = BVH_file(get_std_bvh(dataset=char)).get_ee_length()
            else:
                h = BVH_file(get_std_bvh(dataset=char)).get_height()
            if args.ee_loss_fact == 'learn':
                h = torch.tensor(h, dtype=torch.float)
            else:
                h = torch.tensor(h, dtype=torch.float, requires_grad=False)
            self.real_height.append(BVH_file(get_std_bvh(dataset=char)).get_height())
            self.height.append(h.unsqueeze(0))
        self.real_height = torch.tensor(self.real_height, device=device)
        self.height = torch.cat(self.height, dim=0)
        self.height = self.height.to(device)
        if not args.use_sep_ee: self.height.unsqueeze_(-1)
        if args.ee_loss_fact == 'learn': self.height_para = [self.height]
        else: self.height_para = []

        if not args.simple_operator:
            self.auto_encoder = AE(args, topology=self.edges).to(device)
            self.discriminator = Discriminator(args, self.edges).to(device)
            self.static_encoder = StaticEncoder(args, self.edges).to(device)
        else:
            raise Exception('Conventional operator not yet implemented')

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters()) + self.height_para

    def D_parameters(self):
        return list(self.discriminator.parameters())

    def save(self, path, epoch):
        from option_parser import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.height, os.path.join(path, 'height.pt'))
        torch.save(self.auto_encoder.state_dict(), os.path.join(path, 'auto_encoder.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))
        torch.save(self.static_encoder.state_dict(), os.path.join(path, 'static_encoder.pt'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        if epoch is None:
            all = [int(q) for q in os.listdir(path) if os.path.isdir(path + q)]
            if len(all) == 0:
                raise Exception('Empty loading path')
            epoch = sorted(all)[-1]

        path = os.path.join(path, str(epoch))
        print('loading from epoch {}......'.format(epoch))

        self.auto_encoder.load_state_dict(torch.load(os.path.join(path, 'auto_encoder.pt'),
                                                     map_location=self.args.cuda_device))
        self.static_encoder.load_state_dict(torch.load(os.path.join(path, 'static_encoder.pt'),
                                                       map_location=self.args.cuda_device))
        print('load succeed!')
