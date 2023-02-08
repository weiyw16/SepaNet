# Machine learning based P- and S- wave separation
#              Author: Yanwen Wei
#              Email: wei_yanwen@163.com
#              Date: 2021-12-28
#

import os
import torch
import datetime
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
torch.set_default_dtype(torch.float32)
# local
from cusdataloader import *
from paras import parameters
from utils import * # network

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'

def local_para(opt=None):
    ## change here ###
    opt.which_epoch = 1
    opt.train_model_id = 2023
    opt.test_model_id = 1
    # opt.continue_net_path_G = f'trainOn_m{opt.train_model_id}/NNmodel-epoch-' + str(opt.which_epoch) + '-G.pth'
    opt.continue_net_path_G = 'model/latest_net_G.pth'
    # output
    opt.out_name = 'Test'
    opt.prepredix = f'Results_trainOn_m{opt.train_model_id}_'
    ##################

    opt.data_path = 'data/test_model_' + str(opt.test_model_id)
    opt.max_data = len(os.listdir(opt.data_path + "/vz"))
    opt.batch_size = 1
    opt.gpuid = 2
    opt.nph = 2
    opt.noh = 1
    opt.nt = 2048
    opt.nr = 126
    opt.num_workers = 4
    opt.num_gpu = 1
    opt.data_start = 0
    opt.jsample = 1
    opt.save_testout = 1
    opt.seed = 0
    opt.validate_interval = 1
    opt.sample_interval = 1

    return opt

def main():
    # parameters
    opt = parameters().get_parser() # command line input parameters
    opt = local_para(opt) # local paramters
    opt.out_predix = opt.prepredix + 'testOn_m' + str(opt.test_model_id) + '_epoch_' + str(opt.which_epoch)

    print("------ All read-in parameters ------\n", opt.__dict__)
    torch.cuda.set_device(opt.gpuid)
    mydevice = torch.device('cuda:' + str(opt.gpuid) if torch.cuda.is_available() else "cpu")
    print(f"------ using device: {mydevice} ------")

    # read in data
    if opt.nph == 2:
        myDataset = npy_2to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                             opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    elif opt.nph == 1:
        myDataset = npy_1to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                            opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    else:
        print("ERROR: opt.nph should be 1 or 2.")

    mySampler = RandomSampler(myDataset)
    mydataloader = DataLoader( myDataset, batch_size=int(opt.batch_size),
                               shuffle=None, sampler=mySampler, batch_sampler=None,
                               num_workers=int(opt.num_workers), pin_memory=False,
                               drop_last=False, timeout=0, worker_init_fn=None)

    # <editor-fold === define network === >
    net_G = FCN8s_conv_upsample_out2_v2_k5_c5(opt.nph, opt.nop)
    init_net(net_G, gpu_ids=[opt.gpuid])
    print(f"Network: {net_G.__class__.__name__}")
    # summary(net_G, (opt.nph, opt.nt, opt.nr+2), device='cuda')

    # check for if requiring gradient
    set_requires_grad(net_G)

    # load in
    modelG_loadin = torch.load(opt.continue_net_path_G, map_location=lambda storage, loc: storage.cuda(opt.gpuid))
    net_G.load_state_dict(modelG_loadin)

    # save output
    save_dir = os.path.join(os.getcwd(), opt.out_predix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # # run
    print("------ test begin ------")
    start_time = datetime.datetime.now()
    r2lp = []; r2ls = []; out_fname = []
    for w, (batchdata_B, batchdata_A1, batchdata_A2, batchfname) in enumerate(mydataloader):
        # print("readin data")
        batchdata_B, batchdata_A1, batchdata_A2 = batchdata_B.to(mydevice), batchdata_A1.to(mydevice), batchdata_A2.to(mydevice)
        set_requires_grad(net_G, False)
        batchfake_A1, batchfake_A2 = net_G(batchdata_B)
        for this_subbatch in range(opt.batch_size):
            fname = (batchfname[this_subbatch]).numpy()
            data_B = batchdata_B[this_subbatch]
            data_A1 = batchdata_A1[this_subbatch]
            fake_A1 = batchfake_A1[this_subbatch]
            data_A2 = batchdata_A2[this_subbatch]
            fake_A2 = batchfake_A2[this_subbatch]

            #     evaluation
            out_fname.append(fname)
            r2lp.append(r2_score(data_A1.cpu().detach().numpy().flatten(), fake_A1.cpu().detach().numpy().flatten()) )
            r2ls.append(r2_score(data_A2.cpu().detach().numpy().flatten(), fake_A2.cpu().detach().numpy().flatten()) )

            if opt.save_testout:
                this_name_p = os.path.join(save_dir, str(fname)) + "_P"
                this_name_s = os.path.join(save_dir, str(fname)) + "_S"
                this_name_p_res = os.path.join(save_dir, str(fname)) + "_P_res"
                this_name_s_res = os.path.join(save_dir, str(fname)) + "_S_res"
                #        np.save(this_name, y_predict.detach().numpy())
                with open(this_name_p, "wb") as f:
                    f.write(fake_A1.cpu().detach().numpy())
                with open(this_name_s, "wb") as f:
                    f.write(fake_A2.cpu().detach().numpy())
                with open(this_name_p_res, "wb") as f:
                    f.write((data_A1 - fake_A1).cpu().detach().numpy())
                with open(this_name_s_res, "wb") as f:
                    f.write((data_A2 - fake_A2).cpu().detach().numpy())

    out_data = zip(*[out_fname, r2lp, r2ls])
    out_table = pd.DataFrame(index=range(opt.max_data), columns=['num', 'r2p', 'r2s'], data=out_data)
    out_table.to_csv(save_dir + '/R2.csv')

    r2p = np.array(r2lp); r2s = np.array(r2ls)

    print("Test: Rank %d: [Epoch %d] The average of [r2p score: %f] [r2s score: %f]" \
          % (opt.gpuid, opt.which_epoch, np.mean(r2p), np.mean(r2s)) )
    print("Test: Rank %d: [Epoch %d] The median of [r2p score: %f] [r2s score: %f]" \
          % (opt.gpuid, opt.which_epoch, np.median(r2p),np.median(r2s)))
    print("Test: Rank %d: [Epoch %d] The maximum of: [r2p score: %f at %d] [r2s score: %f at %d]" \
          % (opt.gpuid, opt.which_epoch, np.max(r2p), np.argmax(r2p), np.max(r2s), np.argmax(r2s)))
    print("Test: Rank %d: [Epoch %d] The minimum of: [r2p score: %f at %d] [r2s score: %f at %d]" \
          % (opt.gpuid, opt.which_epoch, np.min(r2p), np.argmin(r2p), np.min(r2s), np.argmin(r2s)))

    for ii in range(opt.max_data):
        print("Num:   ", out_fname[ii], "r2 score p, s:   ", r2lp[ii], r2ls[ii])
    print("------ finished ------")


if __name__ == '__main__':
    main()




