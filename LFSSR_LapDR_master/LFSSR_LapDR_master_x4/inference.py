import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import ToTensor
import imageio
from tqdm import tqdm
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader, MultiValSetDataLoader, ValSetDataLoader


def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('Inference')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)


    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    Test_Dataset = ValSetDataLoader(args)
    Test_Loaders = torch.utils.data.DataLoader(dataset=Test_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=False)
    # test_Names, test_Loaders, length_of_tests = MultiValSetDataLoader(args)
    # print("The number of test data is: %d" % Test_Loaders)
    test_name = 'Test'


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():

        # for index, test_name in enumerate(test_Names):
        #     test_loader = test_Loaders[index]

        save_dir = result_dir.joinpath(test_name)
        save_dir.mkdir(exist_ok=True)

        test(Test_Loaders, device, net, save_dir)
    pass


def test(test_loader, device, net, save_dir=None):
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, ref_y, Sr_SAI_cbcr, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        ref_y = ref_y[0, 0, :, :]
        spa_bound = 2
        Lr_SAI_y = Lr_SAI_y.squeeze()  # numU, numV, h*angRes, w*angRes
        H, W = Lr_SAI_y.shape
        h = H // 5
        w = W // 5
        Lr_SAI_y = rearrange(Lr_SAI_y, '(an1 h) (an2 w)->(h an1) (w an2)', an1=5, an2=5, h=h, w=w)
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLFin, lf_row_nums, lf_col_nums = crop_patch(Lr_SAI_y, args.angRes, args.patch_size_for_test, spa_bound)
        ref_2d_vol, ref_row_nums, ref_col_nums = crop_patch(ref_y, 1, args.patch_size_for_test * args.scale_factor,
                                                            spa_bound * 8)
        patch_num = subLFin.shape[-1]
        patch_ref = ref_2d_vol.shape[1]
        _, _, h_Hr, w_HR = Hr_SAI_y.shape

        h_patch = (96 + 16) * 5
        w_patch = (96 + 16) * 5
        subLFout = torch.zeros([1, 1, h_patch, w_patch], dtype=torch.float32)
        net = net.to('cuda:0')

        ''' SR the Patches '''
        for i in range(0, patch_num, args.minibatch_for_test):
            hr_lf = torch.tensor(subLFin[:, :, i])
            tmp_ref =torch.tensor(ref_2d_vol[:, :, i])

            H, W = hr_lf.shape
            h = H // 5
            w = W // 5
            tmp = rearrange(hr_lf,'(h an1) (w an2)->(an1 h) (an2 w)', an1=5, an2=5,h=h,w=w)
            tmp = tmp[None, None, :, :]
            tmp_ref = tmp_ref[None, None, :, :]
            with torch.no_grad():
                '''‘
                output [1,1,520,520] 
                '''
                output, _, _ = net(tmp.to(device), tmp_ref.to(device))
                output = rearrange(output[0], '1 1 (an1 h) (an2 w)->1 1 (h an1) (w an2)', an1=5, an2=5, h=patch_ref, w=patch_ref)

                # subLFout[i:min(i + args.minibatch_for_test, num), :, :, :] = output
                subLFout = torch.cat([subLFout, output.to('cpu')], dim=0)
        # subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)
        subLFout = rearrange(subLFout, 'num 1 h w->h w 1 num').cpu()
        subLFout = subLFout[:, :, :, 1:]
        _,_,high, width= Sr_SAI_cbcr.shape
        ''' Restore the Patches to LFs '''
        Sr_4D_y = merge_patch(subLFout, lf_row_nums, lf_col_nums, Hr_SAI_y.shape[2], Hr_SAI_y.shape[3], args.angRes, 96,
                              spa_bound * 8, 1)
        # Sr_4D_y = LFintegrate(subLFout, args.angRes, args.patch_size_for_test * args.scale_factor,
        #                       args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes, Hr_SAI_y.size(-1)//args.angRes)
        # Sr_4D_y 的维度是 2160 3120 1
        Sr_SAI_y = rearrange(Sr_4D_y, '(h an1) (w an2) 1-> 1 (an1 h) (an2 w)', an1=5, an2=5, h=h_Hr // 5, w=w_HR // 5)
        Sr_SAI_y = torch.from_numpy(Sr_SAI_y[None, :, :, :])

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes, a2=args.angRes)

            # save all views
            for i in range(args.angRes):
                for j in range(args.angRes):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(save_dir_) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    from config import args
    args.scale_factor = 8
    args.path_for_val = 'F:/LFSSR_HI_master/results/Real_validation/data_for_inference/'

    # args.data_name = 'NTIRE_Val_Real'
    args.model_name = 'LFSSR_LapDR'
    args.path_pre_pth = './pretrain/LFSSR_LapDR_5x5_8x_model.pth'
    main(args)
