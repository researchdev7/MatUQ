import os
import argparse
import time
import sys
import json
import copy
import random
from random import sample
import numpy as np
import pprint
import csv
import yaml
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import swa_utils
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, ExponentialLR, StepLR, LambdaLR
from timm.scheduler import create_scheduler_v2
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.layers import BondExpansion
import schnetpack as spk
from models.cgcnn import load_cgcnn_data, collate_pool, get_cgcnn_train_val_test_loader, CrystalGraphConvNet
from models.megnet import load_megnet_data, MEGNetDataset, megnet_collate_fn_graph, get_megnet_train_val_test_loader, MEGNet
from models.schnet import LoadSchnetData, atoms_collate_fn, get_schnet_train_val_test_loader, SchNet_Output, NeuralNetworkPotential
from models.deepergatgnn import DeeperGATGNNData, get_deepergatgnn_train_val_test_loader, DEEP_GATGNN
from models.alignn import load_alignn_data, get_alignn_train_val_test_loader, ALIGNN
from models.dimenetpp import LoadDimeNetPPData, collate_dimenetpp, get_dimenetpp_train_val_test_loader, DimeNetPlusPlus
from models.sodnet import SODNetData, get_sodnet_train_val_test_loader, SODNet
from models.matformer import load_matformer_data, get_matformer_train_val_test_loader, Matformer
from models.potnet.potnet import load_potnet_data, get_potnet_train_val_test_loader, PotNet
from models.comformer import load_comformer_data, get_comformer_train_val_test_loader, eComFormer, iComFormer
from models.crystalframer.crystalframer import CrystalFramerData, get_crystalframer_train_val_test_loader, CrystalFramer
from utils import mae, AverageMeter, save_checkpoint, Normalizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"The model will run on the {device} device")


def nig_nll(y, mu, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll

def nig_reg(y, mu, v, alpha):
    error = F.l1_loss(y, mu, reduction="none")
    evi = 2 * v + alpha
    return error * evi

def evidential_regresssion_loss(pred, y, coeff=0.2):
    mu = torch.flatten(pred[0])
    v = torch.flatten(pred[1])
    alpha = torch.flatten(pred[2])
    beta = torch.flatten(pred[3])
    _y = y.view(-1)

    loss_nll = nig_nll(_y, mu, v, alpha, beta)
    loss_reg = nig_reg(_y, mu, v, alpha)
    return loss_nll.mean() + (coeff * loss_reg.mean())


def train(train_loader, model, config, optimizer, epoch, normalizer, scheduler=None, swa_model=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config["Models"]["model"] in ["DeeperGATGNN", "SODNet", "PotNet", "CrystalFramer"]:
            target = data.y
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = data.to(device)
            else:
                input_var = data
        else:
            input, target, _ = data

        if config["Models"]["model"] == "CGCNN":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda(),
                             [crys_idx.cuda() for crys_idx in input[3]])
            else:
                input_var = input
        elif config["Models"]["model"] == "MEGNet":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].to(device),
                             input[1].to(device),
                             input[2].to(device))
            else:
                input_var = input
            input_var[0].edata["lattice"] = torch.repeat_interleave(input_var[1], input_var[0].batch_num_edges(), dim=0)
            input_var[0].edata["pbc_offshift"] = (input_var[0].edata["pbc_offset"].unsqueeze(dim=-1) * input_var[0].edata["lattice"]).sum(dim=1)
            input_var[0].ndata["pos"] = (
                    input_var[0].ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(input_var[1], input_var[0].batch_num_nodes(),
                                                                                       dim=0)
            ).sum(dim=1)
            input_var = (input_var[0], input_var[2])
        elif config["Models"]["model"] == "SchNet":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = {}
                for key in input:
                    input_var[key] = input[key].cuda()
            else:
                input_var = input
        elif config["Models"]["model"] == "ALIGNN":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = [input[0].to(device), input[1].to(device)]
            else:
                input_var = input
        elif config["Models"]["model"] in ["DimeNetPP", "Matformer", "ComFormer"]:
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda())
            else:
                input_var = input
        # normalize target
        target = torch.flatten(target)
        target_normed = normalizer.norm(target)

        if not args.disable_cuda and torch.cuda.is_available():
            target_var = target_normed.to(device)
        else:
            target_var = target_normed

        # compute output
        if config["Models"]["model"] in ["SchNet", "DeeperGATGNN", "ALIGNN", "Matformer", "PotNet", "ComFormer", "CrystalFramer"]:
            output = model(input_var)
        elif config["Models"]["model"] in ["SODNet"]:
            output = model(batch=input_var.batch, edge_occu=input_var.edge_occu, f_in=input_var.x,
                    edge_src=input_var.edge_src, edge_dst=input_var.edge_dst, edge_attr=input_var.edge_attr,
                    edge_vec=input_var.edge_vec, edge_num = input_var.edge_num)
        else:
            output = model(*input_var)

        if config["Models"]["evidential"]=="True":
            loss = evidential_regresssion_loss(output, target_var, config["Models"]["coeff"])
        else:
            loss = torch.nn.MSELoss()(output, target_var)

        torch.cuda.synchronize()
        # measure accuracy and record loss
        losses.update(loss.data.cpu(), target.size(0))
        if config["Models"]["evidential"]=="False":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if config["Models"]["model"] == "CrystalFramer":
            if config["Models"]["clip_norm"] > 0:
                total_norm = nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["Models"]["clip_norm"])
            if config["Models"]["clip_grad"] > 0:
                nn.utils.clip_grad.clip_grad_value_(model.parameters(), config["Models"]["clip_grad"])

        optimizer.step()

        if config["Models"]["model"] == "CrystalFramer":
            swa_enabled = config["Models"]["swa_epochs"] + epoch >= config["Models"]["epochs"]
            if swa_enabled:
                swa_model.update_parameters(model)
            if scheduler is not None and not swa_enabled:
                scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if config["Models"]["evidential"]=="True":
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data_Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data_Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )


def validate(val_loader, model, config, normalizer, save_dir, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    uncert_errors = AverageMeter()
    epi_uncert_errors = AverageMeter()
    ale_uncert_errors = AverageMeter()
    mc_uncert_errors = AverageMeter()
    mc_mae_errors = AverageMeter()
    mc_der_uncert_errors = AverageMeter()
    mc_der_uncert_a_errors = AverageMeter()
    mc_der_uncert_e_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_uncerts = []
        test_epi_uncerts = []
        test_ale_uncerts = []
        test_mc_uncerts = []
        test_mc_preds = []
        test_mc_der_uncerts = []
        test_mc_der_uncerts_a = []
        test_mc_der_uncerts_e = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        if config["Models"]["model"] in ["DeeperGATGNN", "SODNet", "PotNet", "CrystalFramer"]:
            target = data.y
            if config["Models"]["model"] in ["DeeperGATGNN", "PotNet", "CrystalFramer"]:
                batch_cif_ids = data.structure_id
            else:
                batch_cif_ids = data.name
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = data.to(device)
            else:
                input_var = data
        else:
            input, target, batch_cif_ids = data

        if config["Models"]["model"] == "CGCNN":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda(),
                             [crys_idx.cuda() for crys_idx in input[3]])
            else:
                input_var = input
        elif config["Models"]["model"] == "MEGNet":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].to(device),
                             input[1].to(device),
                             input[2].to(device))
            else:
                input_var = input
            input_var[0].edata["lattice"] = torch.repeat_interleave(input_var[1], input_var[0].batch_num_edges(), dim=0)
            input_var[0].edata["pbc_offshift"] = (input_var[0].edata["pbc_offset"].unsqueeze(dim=-1) * input_var[0].edata["lattice"]).sum(dim=1)
            input_var[0].ndata["pos"] = (
                    input_var[0].ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(input_var[1], input_var[0].batch_num_nodes(),
                                                                                       dim=0)
            ).sum(dim=1)
            input_var = (input_var[0], input_var[2])
        elif config["Models"]["model"] == "SchNet":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = {}
                for key in input:
                    input_var[key] = input[key].cuda()
            else:
                input_var = input
        elif config["Models"]["model"] == "ALIGNN":
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = [input[0].to(device), input[1].to(device)]
            else:
                input_var = input
        elif config["Models"]["model"] in ["DimeNetPP", "Matformer", "ComFormer"]:
            if not args.disable_cuda and torch.cuda.is_available():
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda())
            else:
                input_var = input

        target = torch.flatten(target)
        target_normed = normalizer.norm(target)

        if not args.disable_cuda and torch.cuda.is_available():
                target_var = target_normed.to(device)
        else:
                target_var = target_normed

        # compute output
        if test:
            with torch.no_grad():
                if config["Models"]["model"] in ["SchNet", "DeeperGATGNN", "ALIGNN", "Matformer", "PotNet", "ComFormer", "CrystalFramer"]:
                    output = model(input_var)
                elif config["Models"]["model"] in ["SODNet"]:
                    output = model(batch=input_var.batch, edge_occu=input_var.edge_occu, f_in=input_var.x,
                                   edge_src=input_var.edge_src, edge_dst=input_var.edge_dst,
                                   edge_attr=input_var.edge_attr,
                                   edge_vec=input_var.edge_vec, edge_num=input_var.edge_num)
                else:
                    output = model(*input_var)
            if config["Models"]["evidential"]=="True":
                mu=torch.flatten(output[0])
                v=torch.flatten(output[1])
                alpha=torch.flatten(output[2])
                beta=torch.flatten(output[3])

                torch.cuda.synchronize()
                v = v.detach().cpu().numpy()
                alpha = alpha.detach().cpu().numpy()
                beta = beta.detach().cpu().numpy()
                var = beta / (v * (alpha - 1))
                output = mu.cpu()
                epi_uncert = torch.tensor(var, dtype=torch.float32)
                ale_uncert = torch.tensor(var*v, dtype=torch.float32)
                uncert = torch.tensor(var+var*v, dtype=torch.float32)
                prediction, mc_uncert, _uncert, _epi_uncert, _ale_uncert = mc_dropout_predict(input_var, model, config,
                                                                                          n_dropout=50)
            # 计算MAE和有MC_Dropout的MAE
            mae_error = mae(normalizer.denorm(output.data), target)
            mae_errors.update(mae_error, target.size(0))
            mc_mae_error = mae(normalizer.denorm(prediction), target)
            mc_mae_errors.update(mc_mae_error, target.size(0))

            # 更新不确定性
            uncert_errors.update(uncert.mean(), target.size(0))
            epi_uncert_errors.update(epi_uncert.mean(), target.size(0))
            ale_uncert_errors.update(ale_uncert.mean(), target.size(0))
            mc_uncert_errors.update(mc_uncert.mean(), target.size(0))
            mc_der_uncert_errors.update(_uncert.mean(), target.size(0))
            mc_der_uncert_e_errors.update(_epi_uncert.mean(), target.size(0))
            mc_der_uncert_a_errors.update(_ale_uncert.mean(), target.size(0))


            test_pred = normalizer.denorm(output.data.cpu())
            test_prediction = normalizer.denorm(prediction)
            test_target = target
            test_uncert = uncert
            test_epi_uncert = epi_uncert
            test_ale_uncert = ale_uncert
            test_mc_uncert = mc_uncert
            test_mc_der_uncert = _uncert
            test_mc_der_uncert_a = _ale_uncert
            test_mc_der_uncert_e = _epi_uncert
            test_preds += test_pred.view(-1).tolist()
            test_mc_preds += test_prediction.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
            test_uncerts += test_uncert.view(-1).tolist()
            test_mc_uncerts += test_mc_uncert.view(-1).tolist()
            test_mc_der_uncerts += test_mc_der_uncert.view(-1).tolist()
            test_epi_uncerts += test_epi_uncert.view(-1).tolist()
            test_ale_uncerts += test_ale_uncert.view(-1).tolist()
            test_mc_der_uncerts_e += test_mc_der_uncert_e.view(-1).tolist()
            test_mc_der_uncerts_a += test_mc_der_uncert_a.view(-1).tolist()

        else:
            with torch.no_grad():
                if config["Models"]["model"] in ["SchNet", "DeeperGATGNN", "ALIGNN", "Matformer", "PotNet", "ComFormer", "CrystalFramer"]:
                    output = model(input_var)
                elif config["Models"]["model"] in ["SODNet"]:
                    output = model(batch=input_var.batch, edge_occu=input_var.edge_occu, f_in=input_var.x,
                                   edge_src=input_var.edge_src, edge_dst=input_var.edge_dst,
                                   edge_attr=input_var.edge_attr,
                                   edge_vec=input_var.edge_vec, edge_num=input_var.edge_num)
                else:
                    output = model(*input_var)
            if config["Models"]["evidential"]=="True":
                loss = evidential_regresssion_loss(output, target_var, config["Models"]["coeff"])
                torch.cuda.synchronize()
                mae_error = mae(normalizer.denorm(torch.flatten(output[0]).data.cpu()), target)
                mae_errors.update(mae_error, target.size(0))
            else:
                loss = torch.nn.MSELoss()(output, target_var)
                torch.cuda.synchronize()
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                mae_errors.update(mae_error, target.size(0))
            losses.update(loss.data.cpu().item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not test:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f}), MC_MAE {mc_mae_errors.val:.3f} ({mc_mae_errors.avg:.3f})\t'
                      'Unc {uncert_errors.val:.3f} ({uncert_errors.avg:.3f})\t'
                      'Unc_der_mc {mc_der_uncert_errors.val:.3f} ({mc_der_uncert_errors.avg:.3f}), Unc_mc {mc_uncert_errors.val:.3f} ({mc_uncert_errors.avg:.3f})\t'
                      'Unc_e {epi_uncert_errors.val:.3f} ({epi_uncert_errors.avg:.3f}), Unc_a {ale_uncert_errors.val:.3f} ({ale_uncert_errors.avg:.3f})\t'
                      'Unc_e_mc {mc_der_uncert_e_errors.val:.3f} ({mc_der_uncert_e_errors.avg:.3f}), Unc_a_mc {mc_der_uncert_a_errors.val:.3f} ({mc_der_uncert_a_errors.avg:.3f})'.format(i, len(val_loader),
                    batch_time=batch_time, mae_errors=mae_errors, mc_mae_errors=mc_mae_errors, uncert_errors=uncert_errors,
                    mc_der_uncert_errors=mc_der_uncert_errors,mc_uncert_errors=mc_uncert_errors,
                    epi_uncert_errors=epi_uncert_errors, ale_uncert_errors=ale_uncert_errors,
                    mc_der_uncert_e_errors=mc_der_uncert_e_errors, mc_der_uncert_a_errors=mc_der_uncert_a_errors))

    if test:
        star_label = '**'
        with open(save_dir + '/' + 'test_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('test_cif_ids', 'test_targets', 'test_preds', 'test_mc_preds', 'test_mc_uncerts',
                             'test_uncerts', 'test_mc_der_uncerts', 'test_epi_uncerts', 'test_ale_uncerts', 'test_mc_der_uncerts_e', 'test_mc_der_uncerts_a'))
            for cif_id, target, pred, mc_pred, mc_uncertainty, uncertainty, mc_der_uncertainty, epi_uncertainty, ale_uncertainty, mc_der_uncerts_e, mc_der_uncerts_a  in zip(test_cif_ids, test_targets,
                test_preds, test_mc_preds, test_mc_uncerts, test_uncerts, test_mc_der_uncerts, test_epi_uncerts, test_ale_uncerts, test_mc_der_uncerts_e, test_mc_der_uncerts_a):
                writer.writerow((cif_id, target, pred, mc_pred, mc_uncertainty, uncertainty, mc_der_uncertainty, epi_uncertainty, ale_uncertainty, mc_der_uncerts_e, mc_der_uncerts_a))
    else:
        star_label = '*'

    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                    mae_errors=mae_errors))

    if not test:
        if config["Models"]["evidential"]=="True":
            print(' {star} Loss {losses.avg:.3f}'.format(star=star_label,
                                                         losses=losses))
            return losses.avg
        else:
            return mae_errors.avg
    else:
        if config["Models"]["evidential"]=="True":
            print(' {star} MC_MAE {mc_mae_errors.avg:.3f}'.format(star=star_label,
                                                                  mc_mae_errors=mc_mae_errors))
            print(' {star} Uncertainty {uncert_errors.avg:.3f}, epi_Uncertainty {epi_uncert_errors.avg:.3f}, ale_Uncertainty {ale_uncert_errors.avg:.3f}'.format(star=star_label,
                                                            uncert_errors=uncert_errors, epi_uncert_errors=epi_uncert_errors, ale_uncert_errors=ale_uncert_errors))
            print(' {star} MC_Uncertainty {mc_uncert_errors.avg:.3f}, MC_DER_Uncertainty {mc_der_uncert_errors.avg:.3f}, MC_E_Uncertainty {mc_der_uncert_e_errors.avg:.3f}, MC_A_Uncertainty {mc_der_uncert_a_errors.avg:.3f}'.format(star=star_label,
                    mc_uncert_errors=mc_uncert_errors, mc_der_uncert_errors=mc_der_uncert_errors, mc_der_uncert_e_errors=mc_der_uncert_e_errors, mc_der_uncert_a_errors=mc_der_uncert_a_errors))
            return (mae_errors.avg, mc_mae_errors.avg, mc_uncert_errors.avg, uncert_errors.avg, epi_uncert_errors.avg,
                    ale_uncert_errors.avg, mc_der_uncert_errors.avg, mc_der_uncert_e_errors.avg, mc_der_uncert_a_errors.avg)

def mc_dropout_predict(input_var, model, config, n_dropout):

    # 启用dropout
    model.train()
    predictions = []
    for t in range(n_dropout):
        with torch.no_grad():
            if config["Models"]["model"] in ["SchNet", "DeeperGATGNN", "ALIGNN", "Matformer", "PotNet", "ComFormer", "CrystalFramer"]:
                pred = model(input_var)
            elif config["Models"]["model"] in ["SODNet"]:
                pred = model(batch=input_var.batch, edge_occu=input_var.edge_occu, f_in=input_var.x,
                               edge_src=input_var.edge_src, edge_dst=input_var.edge_dst,
                               edge_attr=input_var.edge_attr,
                               edge_vec=input_var.edge_vec, edge_num=input_var.edge_num)
            else:
                pred = model(*input_var)
            predictions.append(pred)
    if config["Models"]["evidential"] == "True":
        _mu = []
        _v = []
        _alpha = []
        _beta = []
        for n in range(n_dropout):
            _mu.append(torch.flatten(predictions[n][0]))
            _v.append(torch.flatten(predictions[n][1]))
            _alpha.append(torch.flatten(predictions[n][2]))
            _beta.append(torch.flatten(predictions[n][3]))
        prediction = torch.stack(_mu).mean(dim=0).view(-1)
        mc_uncert = torch.stack(_mu).var(dim=0).view(-1)
        _v = torch.stack(_v).mean(dim=0).view(-1)
        _alpha = torch.stack(_alpha).mean(dim=0).view(-1)
        _beta = torch.stack(_beta).mean(dim=0).view(-1)

        torch.cuda.synchronize()

        prediction = prediction.data.cpu()
        mc_uncert = mc_uncert.data.cpu()
        _v = _v.detach().cpu().numpy()
        _alpha = _alpha.detach().cpu().numpy()
        _beta = _beta.detach().cpu().numpy()
        _var = _beta / (_v * (_alpha - 1))
        _epi_uncert = torch.tensor(_var, dtype=torch.float32)
        _ale_uncert = torch.tensor(_var*_v, dtype=torch.float32)
        _uncert = torch.tensor(_var + _var * _v, dtype=torch.float32)
        return prediction, mc_uncert, _uncert, _epi_uncert, _ale_uncert
    else:
        predictions = torch.stack(predictions)
        prediction = predictions.mean(dim=0).view(-1)
        mc_uncert = predictions.var(dim=0).view(-1)
        torch.cuda.synchronize()
        prediction = prediction.data.cpu()
        mc_uncert = mc_uncert.data.cpu()
        return prediction, mc_uncert

def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    start_time = time.time()
    print("Start time:", start_time)

    parser = argparse.ArgumentParser(description="OOD MatBench with UQ")
    parser.add_argument("--task", default="mp_gap", type=str, help="dielectric, elasticity, perovskites, jdft2d, supercon3d, mp_gap")
    parser.add_argument("--data_path", default="./data", type=str, help="path to data")
    parser.add_argument("--config_path", default="config.yml", type=str, help="path to config file")
    parser.add_argument("--model", default="CrystalFramer", type=str, help="CGCNN, SchNet, MEGNet, DeeperGATGNN,"
                                    "ALIGNN, DimeNetPP, SODNet, Matformer, PotNet, ComFormer, CrystalFramer")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--epochs", default=None, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=None, type=float, help="initial learning rate")
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    args = parser.parse_args(sys.argv[1:])

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)

    assert os.path.exists(args.config_path), (
            "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr

    if args.model != None:
        config["Models"] = config["Models"].get(args.model)

    print("Settings: ")
    pprint.pprint(config)
    with open(args.task + "_" + args.model + "_settings.txt", "w") as log_file:
        pprint.pprint(config, log_file)

    data_process_start_time = time.time()
    if config["Models"]["model"]=="CGCNN":
        dataset = load_cgcnn_data(args.data_path, args.task)
        #print(dataset[0])
        print("--- %s seconds for processing data ---" % (time.time() - data_process_start_time))
        # obtain target value normalizer
        sample_data_list = [dataset[i] for i in range(len(dataset))]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
    elif config["Models"]["model"] == "MEGNet":
        cif_ids, targets, crystals = load_megnet_data(args.data_path, task=args.task)
        elem_list = get_element_list(crystals)
        converter = Structure2Graph(element_types=elem_list, cutoff=config["Models"]["cutoff"])
        dataset = MEGNetDataset(
            structures=crystals,
            targets=targets,
            ids=cif_ids,
            converter=converter,
            save_dir=args.data_path,
            task = args.task,
        )
        normalizer = Normalizer(torch.stack(targets, dim=0))
    elif config["Models"]["model"] == "SchNet":
        dataset = LoadSchnetData(args.data_path, args.task)
        print("--- %s seconds for processing data ---" % (time.time() - data_process_start_time))
        sample_data_list = [dataset[i] for i in range(len(dataset))]
        _, targets, _ = atoms_collate_fn(sample_data_list)
        normalizer = Normalizer(targets)
    elif config["Models"]["model"] == "DeeperGATGNN":
        dataset = DeeperGATGNNData(args.data_path, args.task, config)
        sample_target = torch.stack([dataset[i].y for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "ALIGNN":
        dataset = load_alignn_data(args.data_path, args.task, config["Models"])
        sample_target = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "DimeNetPP":
        dataset = LoadDimeNetPPData(args.data_path, args.task)
        sample_data_list = [dataset[i] for i in range(len(dataset))]
        _, sample_target, _ = collate_dimenetpp(sample_data_list)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "SODNet":
        dataset = SODNetData(args.data_path, args.task)
        sample_target = torch.stack([dataset[i].y for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "Matformer":
        dataset = load_matformer_data(args.data_path, args.task, config["Models"])
        sample_target = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "PotNet":
        dataset = load_potnet_data(args.data_path, args.task, config)
        sample_target = torch.stack([dataset[i].y for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "ComFormer":
        dataset = load_comformer_data(args.data_path, args.task, config["Models"])
        sample_target = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)
    elif config["Models"]["model"] == "CrystalFramer":
        dataset = CrystalFramerData(args.data_path, args.task)
        sample_target = torch.stack([dataset[i].y for i in range(len(dataset))], dim=0)
        normalizer = Normalizer(sample_target)


    #oods = ["LOCO", "SparseXcluster", "SparseYcluster", "SparseXsingle", "SparseYsingle"]
    oods = ["LOCO"]
    task_mae = {}
    task_uncert = {}
    task_uncert_E = {}
    task_uncert_A = {}
    task_mae_mc = {}
    task_uncert_mc = {}
    task_uncert_mc_der = {}
    task_uncert_mc_E = {}
    task_uncert_mc_A = {}
    for ood in oods:
        train_sets_file = "folds/" + args.task + "_folds/train/OFM_" + args.task + "_" + ood + "_target_clusters50_train.json"
        valid_sets_file = "folds/" + args.task + "_folds/val/OFM_" + args.task + "_" + ood + "_target_clusters50_val.json"
        test_sets_file = "folds/" + args.task + "_folds/test/OFM_" + args.task + "_" + ood + "_target_clusters50_test.json"
        assert os.path.exists(train_sets_file), (
            "OOD train set indexs file not found"
        )
        assert os.path.exists(valid_sets_file), (
            "OOD val set indexs file not found"
        )
        assert os.path.exists(test_sets_file), (
            "OOD test set indexs file not found"
        )
        with open(train_sets_file, 'r', encoding='utf-8') as fileT:
            train_indexs = json.load(fileT)
        with open(valid_sets_file, 'r', encoding='utf-8') as fileV:
            val_indexs = json.load(fileV)
        with open(test_sets_file, 'r', encoding='utf-8') as fileT:
            test_indexs = json.load(fileT)

        mae_errors=[]
        uncert_errors=[]
        epi_uncert_errors=[]
        ale_uncert_errors=[]
        mc_mae_errors=[]
        mc_uncert_errors=[]
        mc_der_uncert_errors=[]
        mc_der_uncert_a_errors=[]
        mc_der_uncert_e_errors=[]

        for i in range(50):
            train_index = train_indexs[str(i)]
            val_index = val_indexs[str(i)]
            test_index = test_indexs[str(i)]
            pin_m = not args.disable_cuda and torch.cuda.is_available()

            if config["Models"]["model"]=="CGCNN":
                train_loader, val_loader, test_loader = get_cgcnn_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    collate_fn=collate_pool, batch_size=config["Models"]["batch_size"], pin_memory=pin_m)

                model = CrystalGraphConvNet(
                    orig_atom_fea_len=orig_atom_fea_len,
                    nbr_fea_len=nbr_fea_len,
                    atom_fea_len=config["Models"]["atom_fea_len"],
                    n_conv=config["Models"]["n_conv"],
                    h_fea_len=config["Models"]["h_fea_len"],
                    n_h=config["Models"]["n_h"],
                    evidential=config["Models"]["evidential"],
                    classification=False)

            elif config["Models"]["model"]=="MEGNet":
                train_loader, val_loader, test_loader = get_megnet_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    collate_fn=megnet_collate_fn_graph, batch_size=config["Models"]["batch_size"], pin_memory=pin_m)

                # setup the embedding layer for node attributes
                #node_embed = torch.nn.Embedding(len(elem_list), 16)
                # define the bond expansion
                bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=config["Models"]["cutoff"],
                                               num_centers=config["Models"]["dim_edge_embedding"],
                                               width=config["Models"]["gauss_width"])
                hidden_layer_sizes_input=tuple(int(i) for i in config["Models"]["hidden_layer_sizes_input"])
                hidden_layer_sizes_conv=tuple(int(c) for c in config["Models"]["hidden_layer_sizes_conv"])
                hidden_layer_sizes_output=tuple(int(o) for o in config["Models"]["hidden_layer_sizes_output"])
                # setup the architecture of MEGNet model
                model = MEGNet(
                    dim_node_embedding=config["Models"]["dim_node_embedding"],
                    dim_edge_embedding=config["Models"]["dim_edge_embedding"],
                    dim_state_embedding=config["Models"]["dim_state_embedding"],
                    nblocks=config["Models"]["nblocks"],
                    hidden_layer_sizes_input=hidden_layer_sizes_input,
                    hidden_layer_sizes_conv=hidden_layer_sizes_conv,
                    nlayers_set2set=config["Models"]["nlayers_set2set"],
                    niters_set2set=config["Models"]["niters_set2set"],
                    hidden_layer_sizes_output=hidden_layer_sizes_output,
                    is_classification=False,
                    bond_expansion=bond_expansion,
                    mc_dropout=config["Models"]["mc_dropout"],
                    element_types=elem_list,
                    evidential=config["Models"]["evidential"],
                )
            elif config["Models"]["model"]=="SchNet":
                train_loader, val_loader, test_loader = get_schnet_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    collate_fn=atoms_collate_fn, batch_size=config["Models"]["batch_size"], pin_memory=pin_m)

                pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
                radial_basis = spk.nn.GaussianRBF(n_rbf=config["Models"]["n_rbf"], cutoff=config["Models"]["cutoff"])
                schnet = spk.representation.SchNet(
                    n_atom_basis=config["Models"]["n_atom_basis"], n_interactions=config["Models"]["n_interactions"],
                    radial_basis=radial_basis,
                    cutoff_fn=spk.nn.CosineCutoff(config["Models"]["cutoff"])
                )
                pred = SchNet_Output(n_in=config["Models"]["n_atom_basis"],evidential=config["Models"]["evidential"])

                model = NeuralNetworkPotential(
                    representation=schnet,
                    input_modules=[pairwise_distance],
                    output_modules=[pred],
                    evidential=config["Models"]["evidential"],
                )
            elif config["Models"]["model"]=="DeeperGATGNN":
                train_loader, val_loader, test_loader = get_deepergatgnn_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)

                model = DEEP_GATGNN(dataset, evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"]=="ALIGNN":
                train_loader, val_loader, test_loader = get_alignn_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = ALIGNN(evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"] == "DimeNetPP":
                train_loader, val_loader, test_loader = get_dimenetpp_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    collate_fn=collate_dimenetpp, batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = DimeNetPlusPlus(evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"]=="SODNet":
                train_loader, val_loader, test_loader = get_sodnet_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = SODNet(evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"]=="Matformer":
                train_loader, val_loader, test_loader = get_matformer_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = Matformer(evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"]=="PotNet":
                train_loader, val_loader, test_loader = get_potnet_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = PotNet(evidential=config["Models"]["evidential"], **(config["Models"]["model_setting"]))
            elif config["Models"]["model"] == "ComFormer":
                train_loader, val_loader, test_loader = get_comformer_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                if config["Models"]["name"]=="iComFormer":
                    model = iComFormer(evidential=config["Models"]["evidential"], **(config["Models"]["i_model_setting"]))
                elif config["Models"]["name"]=="eComFormer":
                    model = eComFormer(evidential=config["Models"]["evidential"], **(config["Models"]["e_model_setting"]))
                else:
                    model_name = config["Models"]["name"]
                    raise ValueError(f"Unknown model name: '{model_name}'")
            elif config["Models"]["model"]=="CrystalFramer":
                train_loader, val_loader, test_loader = get_crystalframer_train_val_test_loader(
                    dataset=dataset, train_indexs=train_index, val_indexs=val_index, test_indexs=test_index,
                    batch_size=config["Models"]["batch_size"], pin_memory=pin_m)
                model = CrystalFramer(**(config["Models"]["model_setting"]), lattice_args=config["Models"]["lattice_params"], evidential=config["Models"]["evidential"])
                swa_model = swa_utils.AveragedModel(model)

            if not args.disable_cuda and torch.cuda.is_available():
                model.cuda()

            parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
            print(f'Number of M parameters = {parameters:,}')

            def group_decay(model):
                """Omit weight decay from bias and batchnorm params."""
                decay, no_decay = [], []
                for name, p in model.named_parameters():
                    if "bias" in name or "bn" in name or "norm" in name:
                        no_decay.append(p)
                    else:
                        decay.append(p)

                return [
                    {"params": decay},
                    {"params": no_decay, "weight_decay": 0},
                ]
            def crystalframer_group_decay(model, weight_decay):
                decay, nodecay = [], []
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.SyncBatchNorm)):
                        nodecay.extend(m.parameters(False))
                    else:
                        for name, param in m.named_parameters(recurse=False):
                            if "bias" in name:
                                nodecay.append(param)
                            else:
                                decay.append(param)
                return [
                    {'params': nodecay},
                    {'params': decay, 'weight_decay': weight_decay},
                ]

            def sodnet_wd(model, weight_decay):
                skip = {}
                if hasattr(model, 'no_weight_decay'):
                    skip = model.no_weight_decay()
                def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
                    decay = []
                    no_decay = []
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue  # frozen weights
                        if (name.endswith(".bias") or name.endswith(".affine_weight")
                                or name.endswith(".affine_bias") or name.endswith('.mean_shift')
                                or 'bias.' in name
                                or name in skip_list):
                            no_decay.append(param)
                        else:
                            decay.append(param)
                    return [
                        {'params': no_decay, 'weight_decay': 0.},
                        {'params': decay, 'weight_decay': weight_decay}]
                parameters = add_weight_decay(model, weight_decay, skip)
                return parameters

            if config["Models"]["optimizer"] == "SGD":
                optimizer = optim.SGD(model.parameters(), config["Models"]["lr"],
                                      momentum=config["Models"]["momentum"],
                                      weight_decay=config["Models"]["weight_decay"])
            elif config["Models"]["optimizer"] == "Adam":
                amsgrad = config["Models"]["amsgrad"] == "True"
                optimizer = optim.Adam(model.parameters(), config["Models"]["lr"],
                                       weight_decay=config["Models"]["weight_decay"], amsgrad=amsgrad)
            elif config["Models"]["optimizer"] == "AdamW":
                if config["Models"]["model"] in ["ALIGNN", "Matformer", "PotNet", "ComFormer"]:
                    params = group_decay(model)
                    optimizer = optim.AdamW(params, config["Models"]["lr"],
                                       weight_decay=config["Models"]["weight_decay"])
                elif config["Models"]["model"] == "SODNet":
                    params = sodnet_wd(model, config["Models"]["weight_decay"])
                    optimizer = optim.AdamW(params, config["Models"]["lr"], weight_decay=0.)
                elif config["Models"]["model"] == "CrystalFramer":
                    params = crystalframer_group_decay(model, config["Models"]["weight_decay"])
                    optimizer = optim.AdamW(params, config["Models"]["lr"], config["Models"]["adam_betas"])
                else:
                    optimizer = optim.AdamW(model.parameters(), config["Models"]["lr"],
                                        weight_decay=config["Models"]["weight_decay"])
            else:
                raise NameError('Only SGD/Adam/AdamW is allowed as optimizer')

            # optionally resume from a checkpoint
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    args.start_epoch = checkpoint['epoch']
                    best_error = checkpoint['best_error']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    normalizer.load_state_dict(checkpoint['normalizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
            if config["Models"]["model"] == "CGCNN":
                scheduler = MultiStepLR(optimizer, milestones=config["Models"]["lr_milestones"], gamma=0.1)
            elif config["Models"]["model"] == "MEGNet":
                scheduler = CosineAnnealingLR(optimizer, T_max=config["Models"]["T_max"],
                                              eta_min=config["Models"]["lr"]*config["Models"]["decay_alpha"])
            elif config["Models"]["model"] == "SchNet":
                scheduler = spk.train.ReduceLROnPlateau(optimizer, factor=config["Models"]["factor"],
                                                        patience=config["Models"]["patience"],
                                                        threshold=config["Models"]["threshold"],
                                                        cooldown=config["Models"]["cooldown"])
            elif config["Models"]["model"] == "DeeperGATGNN":
                scheduler = ReduceLROnPlateau(optimizer, **(config["Models"]["scheduler_cfg"]))
            elif config["Models"]["model"] in ["ALIGNN", "Matformer", "PotNet", "ComFormer"]:
                steps_per_epoch = len(train_loader)
                pct_start = config["Models"]["warmup_steps"] / (config["Models"]["epochs"] * steps_per_epoch)
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=config["Models"]["lr"],
                    epochs=config["Models"]["epochs"],
                    steps_per_epoch=steps_per_epoch,
                    pct_start=pct_start,
                    #pct_start=0.3,
                )
            elif config["Models"]["model"] == "DimeNetPP":
                #scheduler = ExponentialLR(optimizer, gamma=config["Models"]["decay_rate"])
                scheduler = StepLR(
                    optimizer, config["Models"]["step_size"], gamma=config["Models"]["gamma"]
                )
            elif config["Models"]["model"] == "SODNet":
                scheduler, _ = create_scheduler_v2(optimizer, **(config["Models"]["scheduler_cfg"]))
            elif config["Models"]["model"] == "CrystalFramer":
                decay = config["Models"]["sch_params"][0]
                f = lambda t: (decay / (decay + t)) ** 0.5
                scheduler = LambdaLR(optimizer, f)

            if not os.path.exists(
                    'results/' + args.model + '/' + args.task + '/' + ood + '/fold_' + str(i)):
                os.makedirs('results/' + args.model + '/' + args.task + '/' + ood + '/fold_' + str(i))
            save_dir = 'results/' + args.model + '/' + args.task + '/' + ood + '/fold_' + str(i)

            best_error = 1e10
            stopping_monitor = 0
            for epoch in range(args.start_epoch, config["Models"]["epochs"]):
                if config["Models"]["model"] == "SODNet":
                    scheduler.step(epoch)

                # train for one epoch

                if config["Models"]["model"] == "CrystalFramer":
                    train(train_loader, model, config, optimizer, epoch, normalizer, scheduler, swa_model)
                else:
                    train(train_loader, model, config, optimizer, epoch, normalizer)
                # evaluate on validation set
                error = validate(val_loader, model, config, normalizer, save_dir)

                if error != error:
                    print('Exit due to NaN')
                    sys.exit(1)

                if config["Models"]["scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(error)
                elif config["Models"]["scheduler"] not in ["cosine", "LambdaLR"]:
                    scheduler.step()

                # remember the best error and save checkpoint
                is_best = error < best_error
                best_error = min(error, best_error)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_error': best_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': vars(args)
                }, is_best=is_best, filedir=save_dir)

                if is_best == False:
                    stopping_monitor += 1
                else:
                    stopping_monitor = 0

                if stopping_monitor > 50:
                    print('Training ends on epoch {0} because the best result remains unchanged after 50 iterations.'.format(epoch+1))
                    break

            # test best model
            print('---------Evaluate Model on Test Set---------------')
            best_checkpoint = torch.load(save_dir + '/' + 'model_best.pth.tar')
            model.load_state_dict(best_checkpoint['state_dict'])
            (mae_error, mc_mae_error, mc_uncert_error, uncert_error, epi_uncert_error, ale_uncert_error,
             mc_der_uncert_error, mc_der_uncert_e_error, mc_der_uncert_a_error) = validate(test_loader, model, config,
                                                                                    normalizer, save_dir, test=True)

            mae_errors.append(mae_error)
            uncert_errors.append(uncert_error)
            epi_uncert_errors.append(epi_uncert_error)
            ale_uncert_errors.append(ale_uncert_error)

            mc_mae_errors.append(mc_mae_error)
            mc_uncert_errors.append(mc_uncert_error)
            mc_der_uncert_errors.append(mc_der_uncert_error)
            mc_der_uncert_a_errors.append(mc_der_uncert_a_error)
            mc_der_uncert_e_errors.append(mc_der_uncert_e_error)


        mae_errors = torch.stack(mae_errors).view(-1).tolist()
        uncert_errors = torch.stack(uncert_errors).view(-1).tolist()
        epi_uncert_errors = torch.stack(epi_uncert_errors).view(-1).tolist()
        ale_uncert_errors = torch.stack(ale_uncert_errors).view(-1).tolist()
        mc_mae_errors=torch.stack(mc_mae_errors).view(-1).tolist()
        mc_uncert_errors=torch.stack(mc_uncert_errors).view(-1).tolist()
        mc_der_uncert_errors=torch.stack(mc_der_uncert_errors).view(-1).tolist()
        mc_der_uncert_a_errors=torch.stack(mc_der_uncert_a_errors).view(-1).tolist()
        mc_der_uncert_e_errors=torch.stack(mc_der_uncert_e_errors).view(-1).tolist()

        #print('mae_errors: {mae_errors:5f}, uncert_errors: {uncert_errors:5f}\t'
        #      'epi_uncert_errors: {epi_uncert_errors:5f}, ale_uncert_errors: {ale_uncert_errors:5f}\t'
        #      'mc_mae_errors: {mc_mae_errors:5f}, mc_uncert_errors: {mc_uncert_errors:5f}\t'
        #      'mc_der_uncert_errors: {mc_der_uncert_errors:5f}, '
        #      'mc_der_uncert_a_errors: {mc_der_uncert_a_errors:5f}, '
        #      'mc_der_uncert_e_errors: {mc_der_uncert_e_errors:5f}'.format(mae_errors=mae_errors,
        #        uncert_errors=uncert_errors,epi_uncert_errors=epi_uncert_errors,ale_uncert_errors=ale_uncert_errors,
        #        mc_mae_errors=mc_mae_errors,mc_uncert_errors=mc_uncert_errors,mc_der_uncert_errors=mc_der_uncert_errors,
        #        mc_der_uncert_a_errors=mc_der_uncert_a_errors,mc_der_uncert_e_errors=mc_der_uncert_e_errors))

        mae = np.array(mae_errors, dtype=float).mean()
        mae_std = np.array(mae_errors, dtype=float).std()
        uncertainty = np.array(uncert_errors, dtype=float).mean()
        uncertainty_std = np.array(uncert_errors, dtype=float).std()
        epi_uncertainty = np.array(epi_uncert_errors, dtype=float).mean()
        epi_uncertainty_std = np.array(epi_uncert_errors, dtype=float).std()
        ale_uncertainty = np.array(ale_uncert_errors, dtype=float).mean()
        ale_uncertainty_std = np.array(ale_uncert_errors, dtype=float).std()
        mc_mae = np.array(mc_mae_errors, dtype=float).mean()
        mc_mae_std = np.array(mc_mae_errors, dtype=float).std()
        mc_uncertainty = np.array(mc_uncert_errors, dtype=float).mean()
        mc_uncertainty_std = np.array(mc_uncert_errors, dtype=float).std()
        mc_der_uncertainty = np.array(mc_der_uncert_errors, dtype=float).mean()
        mc_der_uncertainty_std = np.array(mc_der_uncert_errors, dtype=float).std()
        mc_e_uncertainty = np.array(mc_der_uncert_e_errors, dtype=float).mean()
        mc_e_uncertainty_std = np.array(mc_der_uncert_e_errors, dtype=float).std()
        mc_a_uncertainty = np.array(mc_der_uncert_a_errors, dtype=float).mean()
        mc_a_uncertainty_std = np.array(mc_der_uncert_a_errors, dtype=float).std()



        with open('results/' + args.model + '/' + args.task + '/' + ood + '/' + 'test_metrics.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('mae', 'uncertainty', 'epi_uncertainty', 'ale_uncertainty', 'mc_mae', 'mc_uncertainty', 'mc_der_uncertainty','mc_e_uncertainty','mc_a_uncertainty'))
            for mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err, mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err in zip(mae_errors, uncert_errors, epi_uncert_errors,
                ale_uncert_errors, mc_mae_errors, mc_uncert_errors, mc_der_uncert_errors, mc_der_uncert_e_errors, mc_der_uncert_a_errors):
                writer.writerow((mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err, mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err))
            writer.writerow(('mae(std)', 'uncertainty(std)', 'epi(std)', 'ale(std)', 'mc_mae(std)', 'mc_uncertainty(std)', 'mc_der_uncertainty(std)', 'mc_e(std)', 'mc_a(std)'))
            writer.writerow((f'{mae}({mae_std})', f'{uncertainty}({uncertainty_std})', f'{epi_uncertainty}({epi_uncertainty_std})', f'{ale_uncertainty}({ale_uncertainty_std})',
                             f'{mc_mae}({mc_mae_std})', f'{mc_uncertainty}({mc_uncertainty_std})',
                             f'{mc_der_uncertainty}({mc_der_uncertainty_std})', f'{mc_e_uncertainty}({mc_e_uncertainty_std})', f'{mc_a_uncertainty}({mc_a_uncertainty_std})'))

        task_mae[ood]=(mae,mae_std)
        task_uncert[ood]=(uncertainty,uncertainty_std)
        task_uncert_E[ood]=(epi_uncertainty,epi_uncertainty_std)
        task_uncert_A[ood]=(ale_uncertainty,ale_uncertainty_std)
        task_mae_mc[ood]=(mc_mae,mc_mae_std)
        task_uncert_mc[ood]=(mc_uncertainty,mc_uncertainty_std)
        task_uncert_mc_der[ood]=(mc_der_uncertainty,mc_der_uncertainty_std)
        task_uncert_mc_E[ood]=(mc_e_uncertainty,mc_e_uncertainty_std)
        task_uncert_mc_A[ood]=(mc_a_uncertainty,mc_a_uncertainty_std)

    print(args.task+'_mae：', task_mae)
    print(args.task+'_uncert：', task_uncert)
    print(args.task+'_uncert_E:', task_uncert_E)
    print(args.task+'_uncert_A:', task_uncert_A)
    print(args.task+'_mae_mc：', task_mae_mc)
    print(args.task+'_uncert_mc：', task_uncert_mc)
    print(args.task+'_uncert_mc_der：', task_uncert_mc_der)
    print(args.task+'_uncert_mc_E:', task_uncert_mc_E)
    print(args.task+'_uncert_mc_A:', task_uncert_mc_A)
    print("--- %s seconds for the entire experiment time  ---" % (time.time() - start_time))