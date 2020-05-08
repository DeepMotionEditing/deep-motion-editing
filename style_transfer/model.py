import torch
import torch.nn as nn
import numpy as np

from networks import PatchDis, JointGen


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.use_rotloss = config.use_rotloss
        if self.use_rotloss:
            self.rrec_w = config.rrec_w

        self.gen = JointGen(config)
        self.dis = PatchDis(config)

        self.gan_w = config.gan_w
        self.rec_w = config.rec_w
        self.feat_w = config.feat_w
        self.qt_w = config.qt_w
        self.triplet_w = config.triplet_w
        self.joint_w = config.joint_w
        self.alpha = config.twist_alpha / 180.0 * np.pi
        self.tw_w = config.twist_w
        self.trans_p = config.trans_weight
        self.rec_p = 1.0 - self.trans_p
        self.device = config.device
        self.mse = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=config.triplet_margin)

        self.nets = [self.gen, self.dis]

        self.iter = 0

    @staticmethod
    def euler(q, order="yzx"): # angles: [B, J * 4, T]
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        if order == "xyz":
            ex = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            ey = torch.asin(torch.clamp(2 * (q0 * q2 - q3 * q1), min = -1, max = 1))
            ez = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
            return torch.stack([ex, ez], dim=-1)[:, :, 1:]
        elif order == "yzx":
            ex = torch.atan2(2 * (q1 * q0 - q2 * q3),
                             -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            ey = torch.atan2(2 * (q2 * q0 - q1 * q3),
                             q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            ez = torch.asin(torch.clamp(2 * (q1 * q2 + q3 * q0), min = -1, max = 1))
            return ey[:, :, 1:]  # [B, T, J - 1] .. exclude the root joint?
        else:
            raise Exception("Unknown Euler order!")

    def twist_loss(self, normed):
        euler_y = self.euler(normed)  # [B, T, J - 1]
        diff = torch.clamp(torch.abs(euler_y) - self.alpha, min = 0)
        loss = torch.mean(diff ** 2)
        return loss

    def rot_loss_from_raw(self, raw): # from [B, J * 4 + ?, T] to [B, T, J, 4]
        rot = raw.reshape(raw.shape[0], -1, 4, raw.shape[-1]).permute(0, 3, 1, 2)
        rot_norm = torch.norm(rot, dim=-1, keepdim=True) # [B, T, J, 1]
        l_qt = torch.mean((rot_norm - 1.0) ** 2)
        normed = rot / rot_norm
        l_twist = self.twist_loss(normed)
        return l_qt, l_twist

    @staticmethod
    def recon_criterion(predict, target):
        return torch.mean(torch.abs(predict - target))

    @staticmethod
    def split_pos_glb(raw):  # raw: [B, (J - 1) * 3 + 4, T]
        return raw[:, :-4, :], raw[:, -4:, :]

    @staticmethod
    def merge_pos_glb(pos, glb): # [B, (J - 1) * 3, T], [B, 4, T]
        return torch.cat([pos, glb], dim=-2)

    @staticmethod
    def convert_to_disc(raw): # convert [B, (J - 1) * 3 + 4 (or 3, in 3dpos), T] to representations for the discriminator
        vraw = raw[..., 1:] - raw[..., :-1]  # local velocity -> this differentiation is the same for positions & rotations
        return torch.cat([raw[..., 0:1], vraw], dim=-1)

    def weighted_average(self, trans, rec):
        return self.trans_p * trans + self.rec_p * rec

    def forward(self, co_data, cl_data, mode):
        '''Train forward step'''

        for net in self.nets:
            net.train()
            net.to(self.device)

        """
        stylestr, ostylestr = ("style3d", "style3draw") if self.iter % 2 == 0 else ("style2d", "style2draw")
        same_style = "same_style" + ("3d" if self.iter % 4 < 2 else "2d")
        diff_style = "diff_style" + ("3d" if self.iter % 8 < 4 else "2d")
        """
        stylestr, ostylestr = ("style3d", "style3draw") if self.iter % 2 == 0 else ("style2d", "style2draw")
        same_style = "same_style" + ("3d" if self.iter % 2 == 0 else "2d")
        diff_style = "diff_style" + ("3d" if self.iter % 2 == 0 else "2d")

        if mode == 'gen_update':

            self.iter += 1

            # joint positions to calc l_rec, glb info to complete the output
            xo, xglb = self.split_pos_glb(co_data["style3draw"])
            la = co_data["label"]
            lb = cl_data["label"]

            # encode
            c_xa = self.gen.enc_content(co_data["content"])
            s_xa = self.gen.enc_style(co_data[stylestr], stylestr[-2:]) # xxx, "2d" / "3d"
            s_xb = self.gen.enc_style(cl_data[stylestr], stylestr[-2:])
            s_xpos = self.gen.enc_style(co_data[same_style], same_style[-2:])  # another motion with the same label

            # decode
            xt, rxt = self.gen.decode(c_xa, s_xb)  # translation
            xr, rxr = self.gen.decode(c_xa, s_xa)  # reconstruction
            xs, rxs = self.gen.decode(c_xa, s_xpos)  # reconstruction from another motion

            # quaternion loss -> norm = 1
            l_qt_t, l_tw_t = self.rot_loss_from_raw(rxt)
            l_qt_r, l_tw_r = self.rot_loss_from_raw(rxr)
            l_qt_s, l_tw_s = self.rot_loss_from_raw(rxs)

            l_qt_rec = (l_qt_r + l_qt_s) / 2.0
            l_qt = self.weighted_average(l_qt_t, l_qt_rec)
            l_tw_rec = (l_tw_r + l_tw_s) / 2.0
            l_tw = self.weighted_average(l_tw_t, l_tw_rec)

            xtf = self.merge_pos_glb(xt, xglb)
            xrf = self.merge_pos_glb(xr, xglb)
            xsf = self.merge_pos_glb(xs, xglb)

            # input to discriminator
            da = self.convert_to_disc(co_data["style3draw"])
            db = self.convert_to_disc(cl_data["style3draw"])
            dt = self.convert_to_disc(xtf)
            dr = self.convert_to_disc(xrf)
            ds = self.convert_to_disc(xsf)

            # adversarial loss
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(dt, lb)
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(dr, la)
            l_adv_s, gacc_s, xs_gan_feat = self.dis.calc_gen_loss(ds, la)
            l_adv_rec = (l_adv_r + l_adv_s) / 2.0
            l_adv = self.weighted_average(l_adv_t, l_adv_rec)
            gacc_rec = (gacc_r + gacc_s) / 2.0
            gacc = self.weighted_average(gacc_t, gacc_rec)

            # feature loss
            _, xb_gan_feat = self.dis(db, cl_data["label"])
            _, xa_gan_feat = self.dis(da, co_data["label"])

            l_ft_t = self.recon_criterion(xt_gan_feat.mean(2),
                                          xb_gan_feat.mean(2))
            l_ft_r = self.recon_criterion(xr_gan_feat.mean(2),
                                          xa_gan_feat.mean(2))
            l_ft_s = self.recon_criterion(xs_gan_feat.mean(2),
                                          xa_gan_feat.mean(2))
            l_ft_rec = (l_ft_r + l_ft_s) / 2.0
            l_ft = self.weighted_average(l_ft_t, l_ft_rec)

            # reconstruction loss

            l_r_rec = self.recon_criterion(xr, xo)
            l_s_rec = self.recon_criterion(xs, xo)
            l_rec = (l_r_rec + l_s_rec) / 2.0

            # reconstruction loss for rotations!
            if self.use_rotloss:
                rxo, _ = self.split_pos_glb(co_data["contentraw"]) # nrot: xx + 4
                l_r_rrec = self.recon_criterion(rxr, rxo)
                l_s_rrec = self.recon_criterion(rxs, rxo)
                l_rrec = (l_r_rrec + l_s_rrec) / 2.0

            # triplet loss

            anchor = s_xa
            pos = s_xpos
            neg = self.gen.enc_style(co_data[diff_style], diff_style[-2:])

            l_triplet = self.triplet_loss(anchor, pos, neg)

            # joint loss

            otherstr = "style2d" if stylestr == "style3d" else "style3d"
            s_other = self.gen.enc_style(co_data[otherstr], otherstr[-2:])

            l_joint = self.mse(s_other, s_xa)

            # summary

            l_total = (self.gan_w * l_adv +
                       self.rec_w * l_rec +
                       self.feat_w * l_ft +
                       self.qt_w * l_qt +
                       self.tw_w * l_tw +
                       self.triplet_w * l_triplet +
                       self.joint_w * l_joint)

            if self.use_rotloss:
                l_total += self.rrec_w * l_rrec

            l_total.backward()

            ret_dict = {
                'gen_loss_total': l_total,
                'gen_loss_adv': l_adv,
                'gen_loss_recon_all': l_rec,
                'gen_loss_recon_r': l_r_rec,
                'gen_loss_recon_s': l_s_rec,
                'gen_loss_feature_all': l_ft,
                'gen_loss_feature_r': l_ft_r,
                'gen_loss_feature_s': l_ft_s,
                'gen_loss_feature_t': l_ft_t,
                'gen_loss_quaternion': l_qt,
                'gen_loss_twist': l_tw,
                'gen_loss_triplet': l_triplet,
                'gen_loss_joint': l_joint,
                'gen_acc_all': gacc,
                'gen_acc_rec': gacc_rec,
                'gen_acc_t': gacc_t
            }

            if self.use_rotloss:
                ret_dict["gen_loss_recon_rot_all"] = l_rrec

            return ret_dict

        elif mode == 'dis_update':
            xb = cl_data["style3draw"]
            xa = co_data["style3draw"]
            lb = cl_data["label"]
            la = co_data["label"]
            xb.requires_grad_()
            db = self.convert_to_disc(xb)
            l_real_p, acc_r, resp_r = self.dis.calc_dis_real_loss(db, lb)
            l_real = self.gan_w * l_real_p
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, db)
            l_reg = 10 * l_reg_pre
            l_reg.backward(retain_graph=True)
            with torch.no_grad():
                xo, xglb = self.split_pos_glb(xa)
                c_xa = self.gen.enc_content(co_data["content"])
                s_xa = self.gen.enc_style(co_data[stylestr], stylestr[-2:])
                s_xb = self.gen.enc_style(cl_data[stylestr], stylestr[-2:])

                xt, rxt = self.gen.decode(c_xa, s_xb)
                xr, rxr = self.gen.decode(c_xa, s_xa)

                dt = self.convert_to_disc(self.merge_pos_glb(xt, xglb))
                dr = self.convert_to_disc(self.merge_pos_glb(xr, xglb))

            l_fake_p_r, acc_f_r, resp_f_r = self.dis.calc_dis_fake_loss(dr.detach(), la)
            l_fake_p_t, acc_f_t, resp_f_t = self.dis.calc_dis_fake_loss(dt.detach(), lb)
            l_fake_p = (l_fake_p_r + l_fake_p_t) / 2.0
            acc_f = (acc_f_r + acc_f_t) / 2.0
            l_fake = self.gan_w * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            loss_dict = {
                'dis_loss_total': l_total,
                'dis_loss_adv_all': l_fake_p + l_real_p,
                'dis_loss_adv_real': l_real_p,
                'dis_loss_adv_fake': l_fake_p,
                'dis_loss_reg': l_reg_pre,
                'dis_acc_all': acc,
                'dis_acc_real': acc_r,
                'dis_acc_fake': acc_f
            }
            return loss_dict
        else:
            assert 0, 'not support operation'

    def test_rec(self, data):
        '''For plotting the reconstruction curve on test data'''
        self.eval()
        self.gen.eval()

        xtgt = data["style3draw"]
        x = data["content"]
        stylestr = "style3d" if self.iter % 2 == 0 else "style2d"
        same_style = "same_" + stylestr
        y = data[stylestr]
        yp = data[same_style]

        xo, xglb = self.split_pos_glb(xtgt)

        with torch.no_grad():
            c_x = self.gen.enc_content(x)
            s_x = self.gen.enc_style(y, stylestr[-2:])
            s_p = self.gen.enc_style(yp, stylestr[-2:])
            xr, rxr = self.gen.decode(c_x, s_x)
            loss_r = self.recon_criterion(xr, xo)
            xs, rxs = self.gen.decode(c_x, s_p)
            loss_s = self.recon_criterion(xs, xo)

        loss = (loss_r + loss_s) / 2.0
        loss_dict = {'gen_loss_recon_all': loss,
                     'gen_loss_recon_r': loss_r,
                     'gen_loss_recon_s': loss_s}
        return loss_dict, {}

    def test(self, co_data, cl_data, status):
        '''For producing results'''
        self.eval()
        self.gen.eval()

        xtgt = co_data["style3draw"]
        xa = co_data["content"]
        stylestr = "style" + status
        if stylestr in co_data:
            content_stylestr = stylestr
        else:
            content_stylestr = "style3d"
        ya = co_data[content_stylestr]
        yb = cl_data[stylestr]

        xo, xglb = self.split_pos_glb(xtgt)
        c_xa = self.gen.enc_content(xa)
        s_xa = self.gen.enc_style(ya, content_stylestr[-2:])
        s_xb = self.gen.enc_style(yb, stylestr[-2:])

        _, rxt = self.gen.decode(c_xa, s_xb)
        _, rxr = self.gen.decode(c_xa, s_xa)

        full_r = self.merge_pos_glb(rxr, xglb)
        full_t = self.merge_pos_glb(rxt, xglb)

        self.train()

        out_dict = {
            "content_meta": co_data["meta"],
            "style_meta": cl_data["meta"],
            "foot_contact": co_data["foot_contact"],
            "content": co_data["contentraw"],
            "recon": full_r,
            "trans": full_t,
        }

        if status == "3d":
            out_dict["style"] = cl_data["contentraw"]
        else:
            out_dict["style"] = cl_data["style2draw"]

        return out_dict


    def test_interpolation(self, co_data, cl_b, cl_c, num=3):
        '''For interpolation'''
        self.eval()
        self.gen.eval()

        xtgt = co_data["style3draw"]
        xa = co_data["content"]

        stylestr = "style3d"

        ostylestr = stylestr + "raw"
        ya = co_data[stylestr]
        yb, yc = cl_b[stylestr], cl_c[stylestr]
        ybo, yco = cl_b[ostylestr], cl_c[ostylestr]

        with torch.no_grad():

            xo, xglb = self.split_pos_glb(xtgt)
            c_xa = self.gen.enc_content(xa)
            s_xa = self.gen.enc_style(ya, stylestr[-2:])
            s_xb = self.gen.enc_style(yb, stylestr[-2:])
            s_xc = self.gen.enc_style(yc, stylestr[-2:])

            def itp(s, t, num):
                num = num + 1.0
                d = (t - s) / num
                ret = []
                for i in range(int(num + 1)):
                    ret.append(s + i * d)
                return ret

            def itp_outs(s, t, num, c, glb):
                scodes = itp(s, t, num)
                xs = []
                rxs = []
                for s in scodes:
                    x, rx = self.gen.decode(c, s)
                    x = self.merge_pos_glb(x, glb)
                    xs.append(x)
                    rxs.append(rx)
                return xs, rxs

            sab_outs = itp_outs(s_xa, s_xb, num, c_xa, xglb)
            sbc_outs = itp_outs(s_xb, s_xc, num, c_xa, xglb)

        self.train()

        rd = {
            "content": [],
            "content_rot": [],
            "foot_contact": [],
            "content_label": [],
            "style_label": [],
            "style2_label": [],
            "style": [],
            "style2": [],
            "style_rot": [],
            "style2_rot": [],
            "trans": [],
            "trans_rot": [],
            "info": []
        }

        cont_label = co_data["label"]
        style_label = cl_b["label"]
        style2_label = cl_c["label"]
        ft = co_data["foot_contact"]
        cont_rot = co_data["contentraw"]
        style_rot = cl_b["contentraw"]
        style2_rot = cl_c["contentraw"]

        for prefix, outs in zip(["ab", "bc"], [sab_outs, sbc_outs]):
            for i, (x, xr) in enumerate(zip(*outs)):
                rd["content"].append(xtgt)
                rd["style"].append(ybo)
                rd["style2"].append(yco)
                rd["content_rot"].append(cont_rot)
                rd["foot_contact"].append(ft)
                rd["content_label"].append(cont_label)
                rd["style_label"].append(style_label)
                rd["style2_label"].append(style2_label)
                rd["style_rot"].append(style_rot)
                rd["style2_rot"].append(style2_rot)
                rd["trans"].append(x)
                rd["trans_rot"].append(xr)
                rd["info"].append(prefix + "_%d" % i)

        return rd

    def get_latent_codes(self, data):
        '''For latent code extraction'''
        return self.gen.get_latent_codes(data)



