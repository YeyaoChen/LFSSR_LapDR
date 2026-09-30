import torch
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, ValSetDataLoader
from collections import OrderedDict
import imageio
from scipy.io import savemat
import numpy as np
import random
from torchkeras import VLog
import time


def main(args):
    ''' 建立保存模型的文件夹 Create Dir for Save '''
    log_dir, val_dir, checkpoints_dir = create_dir(args)

    ''' 训练日志 Logger '''
    logger = Logger(log_dir, args)

    ''' 设置在GPU上训练 CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' 导入训练集 / DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True)

    ''' 导入验证集 / DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    val_dataset = ValSetDataLoader(args)
    val_loaders = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers,
                                               batch_size=1, shuffle=False)
    logger.log_string("The number of validation data is: %d" % len(val_loaders))

    ''' 设置随机种子 seed setup'''
    setup_seed(42)
    print(f"Random seed has been set to: 42")

    ''' 导入定义好的网络模型 MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    model_path = 'model.' + args.model_name
    model_lfssr = importlib.import_module(model_path)
    net = model_lfssr.get_model(args)

    ''' 构建Adam优化器 Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    ''' 导入预训练的模型 Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(model_lfssr.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
        net = net.to(device)
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cuda:0')
            start_epoch = checkpoint['epoch']

            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v

            # load params
            net.load_state_dict(new_state_dict)
            net = net.to(device)
            # load optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.log_string('Use pretrain model!')
            print(ckpt_path)
        except:
            net.apply(model_lfssr.weights_init)
            net = net.to(device)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass

    ''' 将网络放入GPU中 '''
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    ''' LOSS LOADING '''
    criterion = model_lfssr.get_loss(args).to(device)

    # ####################################################################################################
    vlog1 = VLog(epochs=args.epoch, monitor_metric='train_psnr', monitor_mode='max', save_path='train')
    vlog2 = VLog(epochs=args.epoch, monitor_metric='val_psnr', monitor_mode='max', save_path='val')

    vlog1.log_start()
    vlog2.log_start()
    # ##########################################################################################################
    ''' 开始训练 TRAINING & TEST '''
    for idx_epoch in range(start_epoch, args.epoch):

        print(f"Epoch: {idx_epoch + 1}, Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        ''' Training '''
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))
        logger.log_string('\nStart training...')
        loss_epoch_train, psnr_epoch_train = train(train_loader, device, net, criterion, optimizer, logger)
        logger.log_string('The %dth Train,  loss is: %.5f' % (idx_epoch + 1, loss_epoch_train))
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train))

        vlog1.log_step({'train_loss': loss_epoch_train, 'train_psnr': psnr_epoch_train})
        time.sleep(0.05)
        vlog1.log_epoch({'train_psnr': psnr_epoch_train})

        ''' 训练完一个epoch，调整学习率 scheduler '''
        scheduler.step()

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
                args.model_name, args.angRes, args.angRes, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' 训练完一个epoch，开始验证 Validation '''
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))
        logger.log_string('\nStart validating...')
        step = 1
        if idx_epoch % step == 0 or idx_epoch > args.epoch - step:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()
                psnr_testset = []

                epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                epoch_dir.mkdir(exist_ok=True)

                LF_epoch_save_dir = epoch_dir.joinpath('Results/')
                LF_epoch_save_dir.mkdir(exist_ok=True)

                psnr_iter_test, LF_name, _, _ = test(val_loaders, device, net, LF_epoch_save_dir)
                excel_file.write_sheet('val', LF_name, psnr_iter_test)

                psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                psnr_testset.append(psnr_epoch_test)

                logger.log_string('The %dth val, psnr is %.3f' % (idx_epoch + 1, psnr_epoch_test))
                vlog2.log_step({'val_psnr': psnr_epoch_test})
                time.sleep(0.05)
                vlog2.log_epoch({'val_psnr': psnr_epoch_test})
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
                pass
            pass
        pass
    vlog1.log_end()
    vlog2.log_end()
    pass


''' 训练函数 '''
def train(train_loader, device, net, criterion, optimizer, logger):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []

    for idx_iter, (data, label, ref_y, hr_down4, hr_down2) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        # '''Memory Usage'''
        # torch.cuda.empty_cache()

        data = data.to(device)             # low resolution  [b, 1, 160, 160]
        label = label.to(device)           # high resolution [b, 1, 640, 640]
        ref = ref_y.to(device)             # 2D HR ref image [b, 1, 128, 128]

        out_lf = net(data, ref)
        loss_total = criterion(out_lf, label)

        loss_iter_train.append(loss_total.data.cpu())
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        psnr = cal_metrics(args, label, out_lf)
        psnr_iter_train.append(psnr.sum()/np.sum(psnr>0))
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train


''' 验证/测试函数 '''
def test(test_loader, device, net, epoch_dir=None):

    LF_iter_test = []
    psnr_iter_test, psnr_iter_test_allviews = [], []
    single_sence_time_avg = []

    for idx_iter, (Lr_SAI_y, Hr_SAI_y, ref_y, Sr_SAI_cbcr, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        Lr_SAI_y = Lr_SAI_y.squeeze()         # numU, numV, h*angRes, w*angRes
        H, W = Lr_SAI_y.shape
        h = H//5
        w = W//5
        Lr_SAI_y = rearrange(Lr_SAI_y,'(an1 h) (an2 w)->(h an1) (w an2)', an1=5, an2=5, h=h, w=w)
        Hr_SAI_y = Hr_SAI_y
        ref_y = ref_y[0, 0, :, :]
        Sr_SAI_cbcr = Sr_SAI_cbcr

        spa_bound = 4

        ''' Crop LFs into Patches '''
        subLFin, lf_row_nums, lf_col_nums = crop_patch(Lr_SAI_y, args.angRes, args.patch_size_for_test, spa_bound)
        ref_2d_vol, ref_row_nums, ref_col_nums = crop_patch(ref_y, 1, args.patch_size_for_test * args.scale_factor, spa_bound * args.scale_factor)

        patch_num = subLFin.shape[-1]
        patch_ref = ref_2d_vol.shape[1]

        _, _, h_Hr, w_HR = Hr_SAI_y.shape

        h_patch = (96 + spa_bound * args.scale_factor) * args.angRes
        w_patch = (96 + spa_bound * args.scale_factor) * args.angRes

        subLFout = torch.zeros([1, 1, h_patch, w_patch], dtype=torch.float32).to(device)

        net = net.to(args.device)
        ''' SR the Patches '''
        for i in range(0, patch_num, args.minibatch_for_test):
            hr_lf = torch.tensor(subLFin[:, :, i])
            tmp_ref = torch.tensor(ref_2d_vol[:, :, i])

            H, W = hr_lf.shape
            h = H // 5
            w = W // 5
            tmp = rearrange(hr_lf,'(h an1) (w an2)->(an1 h) (an2 w)', an1=5, an2=5, h=h, w=w)
            tmp = tmp[None, None, :, :]
            tmp_ref = tmp_ref[None, None, :, :]
            with torch.no_grad():
                output = net(tmp.to(device), tmp_ref.to(device))
                output_ = rearrange(output,'1 1 (an1 h) (an2 w)->1 1 (h an1) (w an2)', an1=5, an2=5, h=patch_ref, w=patch_ref)
                subLFout = torch.cat([subLFout, output_],dim=0)
        subLFout = rearrange(subLFout,'num 1 h w->h w 1 num').cpu()
        subLFout = subLFout[:,:,:,1:]

        ''' Restore the Patches to LFs '''
        Sr_4D_y = merge_patch(subLFout, lf_row_nums, lf_col_nums, Hr_SAI_y.shape[2], Hr_SAI_y.shape[3], args.angRes,
                              96, spa_bound * args.scale_factor, 1)
        Sr_SAI_y = rearrange(Sr_4D_y, '(h an1) (w an2) 1-> 1 (an1 h) (an2 w)', an1=5, an2=5, h=h_Hr//5, w=w_HR//5)
        # Sr_SAI_y = torch.from_numpy(Sr_SAI_y[None,:,:,:])

        ''' Calculate the PSNR '''
        # psnr = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        # psnr_mean = psnr.sum() / np.sum(psnr > 0)

        # psnr_iter_test.append(psnr_mean)
        # LF_iter_test.append(LF_name[0])
        # psnr_iter_test_allviews.append(psnr)

        # LF_scence_name = f"{int(LF_name[0]):03}" if LF_name[0].isdigit() else LF_name[0]

        ''' Save RGB '''
        # if epoch_dir is not None:
        #     save_dir_ = epoch_dir.joinpath(LF_scence_name)
        #     save_dir_.mkdir(exist_ok=True)
        #
        #     Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
        #     Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype('uint8')
        #     Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes, a2=args.angRes)
        #
        #     # save all views
        #     for i in range(args.angRes):
        #         for j in range(args.angRes):
        #             img = Sr_4D_rgb[i, j, :, :, :]
        #             path = str(save_dir_) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.png'
        #             imageio.imwrite(path, img)
        #             pass
        #         pass
        #     pass
        # pass

    return psnr_iter_test, LF_iter_test, psnr_iter_test_allviews, single_sence_time_avg


'''Set Set the random number seed'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from config import args
    main(args)
