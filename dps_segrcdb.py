import argparse 
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.segrcdb_semi import SemiDataset
from model.semseg.dpt import DPT
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='UniMatch V2 Pre-training (Standard CutMix)')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True 

    # --- Model Setup ---
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    
    state_dict = torch.load('dinov2_base.pth')
    model.backbone.load_state_dict(state_dict)
        
    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    # --- Criterion ---
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=255).cuda(local_rank)

    # --- Dataset & Loader ---
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    total_iters = len(trainloader_l) * cfg['epochs']
    epoch = -1
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}'.format(epoch))
        
        total_loss  = AverageMeter()
        trainloader_l.sampler.set_epoch(epoch)

        model.train()

        for i, (img_s1, img_s2, mask_gt, cutmix_box1, cutmix_box2) in enumerate(trainloader_l):
            
            img_s1, img_s2 = img_s1.cuda(), img_s2.cuda()
            mask_gt = mask_gt.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            
            img_s1_flip = img_s1.flip(0)
            img_s2_flip = img_s2.flip(0)
            mask_gt_flip = mask_gt.flip(0)

            # --- Mask CutMix Logic ---
            target_mask_s1 = mask_gt.clone()
            target_mask_s2 = mask_gt.clone()

            box1_mask = (cutmix_box1 == 1)
            box2_mask = (cutmix_box2 == 1)

            target_mask_s1[box1_mask] = mask_gt_flip[box1_mask]
            target_mask_s2[box2_mask] = mask_gt_flip[box2_mask]

            # --- Image CutMix Logic ---
            box1_mask_expanded = box1_mask.unsqueeze(1).expand_as(img_s1)
            box2_mask_expanded = box2_mask.unsqueeze(1).expand_as(img_s2)

            img_s1[box1_mask_expanded] = img_s1_flip[box1_mask_expanded]
            img_s2[box2_mask_expanded] = img_s2_flip[box2_mask_expanded]

            # --- Forward ---
            pred_s1, pred_s2 = model(torch.cat((img_s1, img_s2)), comp_drop=True).chunk(2)
            
            # --- Loss Calculation (Manual Reduction) ---
            
            # S1 Loss
            loss_s1_pixelwise = criterion(pred_s1, target_mask_s1) # [B, H, W]
            valid_mask_s1 = (target_mask_s1 != 256).float()        
            loss_s1 = (loss_s1_pixelwise * valid_mask_s1).sum() / (valid_mask_s1.sum() + 1e-6)

            # S2 Loss
            loss_s2_pixelwise = criterion(pred_s2, target_mask_s2)
            valid_mask_s2 = (target_mask_s2 != 256).float()
            loss_s2 = (loss_s2_pixelwise * valid_mask_s2).sum() / (valid_mask_s2.sum() + 1e-6)
            
            loss = (loss_s1 + loss_s2) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader_l) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)

            if (i % (len(trainloader_l) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg))

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            if epoch % 10 == 0 or epoch == cfg['epochs'] - 1:
                torch.save(checkpoint, os.path.join(args.save_path, 'segrcdb_' +str(epoch)+ '.pth'))

if __name__ == '__main__':
    main()