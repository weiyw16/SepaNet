# Machine learning based P- and S- wave separation
#              Author: Yanwen Wei
#              Email: wei_yanwen@163.com
#              Date: 2021-12-28
#

import os
import torch
import datetime
from sklearn.metrics import r2_score
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
torch.set_default_dtype(torch.float32)
# local
from paras import parameters
from cusdataloader import *
from utils import * # network

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'

def local_para(opt=None):
    ## data parameter ##
    # input
    opt.train_model_id = 2023
    opt.test_model_id = 1
    # output
    opt.out_name = 'NNmodel'  # model name (change if same training model exist)
    opt.prepredix = f'trainOn_m{opt.train_model_id}'  # dir name
    ####################
    opt.data_path = f'data/fortrain_model_{opt.train_model_id}'
    opt.data_path_valid = f'data/test_model_{opt.test_model_id}'
    opt.max_data = len(os.listdir(opt.data_path + "/vz"))
    opt.max_data_valid = len(os.listdir(opt.data_path_valid + "/vz"))

    opt.if_continue = False  # True
    opt.continue_net_path_G = "old_model.pth"
    opt.epoches = 3
    opt.batch_size = 1
    opt.gpuid = 2
    opt.nph = 2 # input 2 channel
    opt.noh = 1 # set 1 if multi-task learning
    opt.nt = 2048
    opt.nr = 126
    opt.num_workers = 4
    opt.num_gpu = 1

    opt.data_start = 0
    opt.jsample = 1
    opt.lr = 0.001
    opt.beta1 = 0.5

    opt.do_validate = 1
    opt.data_start_valid = 0
    opt.jsample_valid = 10
    opt.seed = 0
    opt.validate_interval = 1
    opt.sample_interval = 1

    return opt

def main():
    # parameters
    opt = parameters().get_parser() # add parameters through command line
    opt = local_para(opt) # add through scripts, prior than command line
    opt.out_predix = opt.prepredix #+ str(opt.train_model_id)

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
    if opt.nph == 2:
        myDataset_valid = npy_2to2_Dataloader_getindex(opt.data_path_valid, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data_valid, opt.data_norm_type, opt.data_start_valid, opt.jsample_valid, 0)
    elif opt.nph == 1:
        myDataset_valid = npy_1to2_Dataloader_getindex(opt.data_path_valid, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data_valid, opt.data_norm_type, opt.data_start_valid, opt.jsample_valid, 0)
    else:
        print("ERROR: opt.nph should be 1 or 2.")
    mySampler_valid = RandomSampler(myDataset_valid)
    mydataloader_valid = DataLoader(myDataset, batch_size=int(opt.batch_size),
                              shuffle=None, sampler=mySampler_valid, batch_sampler=None,
                              num_workers=int(opt.num_workers), pin_memory=False,
                              drop_last=False, timeout=0, worker_init_fn=None)


    # <editor-fold === define network === >
    net_G = FCN8s_conv_upsample_out2_v2_k5_c5(opt.nph, opt.nop)
    init_net(net_G, gpu_ids=[opt.gpuid])
    print(f"Network: {net_G.__class__.__name__}")
    # summary(net_G, (opt.nph, opt.nt, opt.nr+2), device='cuda')

    # check for if requiring gradient
    set_requires_grad(net_G, requires_grad=True)  #not sure

    # other configuration
    metric_gan = torch.nn.MSELoss().to(mydevice)
    optimizer = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer.zero_grad()

    scheduler_G = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # save output
    save_dir = os.path.join(os.getcwd(), opt.out_predix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # # run
    print("------ training begin ------")
    start_time = datetime.datetime.now()
    for epoch in range(opt.epoch_start, opt.epoches):
        batch_i = 1 # local counting
        for data_B, data_A1, data_A2, fname in mydataloader:

            data_B, data_A1, data_A2 = data_B.to(mydevice), data_A1.to(mydevice), data_A2.to(mydevice)
            # print(f"The shape of data is B {data_B.shape}; A1 {data_A1.shape}; A2 {data_A2.shape}; fnames {fname.numpy()}")
            fake_A1, fake_A2 = net_G(data_B)

            loss_G1 = metric_gan(fake_A1, data_A1)  # * opt.lambda_L1
            loss_G2 = metric_gan(fake_A2, data_A2)  # * opt.lambda_L1

            loss = loss_G1 + loss_G2
            loss.backward()

            # step
            optimizer.step()
            optimizer.zero_grad()

            # evaluation
            r2_p = r2_score(data_A1.cpu().detach().numpy().flatten(), fake_A1.cpu().detach().numpy().flatten())
            r2_s = r2_score(data_A2.cpu().detach().numpy().flatten(), fake_A2.cpu().detach().numpy().flatten())
            elapsed_time = datetime.datetime.now() - start_time
            print("Rank %d: [Epoch %d/%d] [Batch %d/%d] [G loss: %f, %f] [r2 score: p %f, s %f]time: % s" \
                  %(opt.gpuid, epoch, opt.epoches, batch_i, int(opt.max_data / opt.batch_size), \
                    loss_G1, loss_G2, r2_p, r2_s, elapsed_time))
            batch_i += 1

        ## validate
        if opt.do_validate and epoch % opt.validate_interval == 0:

            r2lp = []; r2ls = []; loss_p = []; loss_s = []

            for data_B, data_A1, data_A2, fname in mydataloader_valid:
                data_B, data_A1, data_A2 = data_B.to(mydevice), data_A1.to(mydevice), data_A2.to(mydevice)
                ## forward
                set_requires_grad(net_G, False)
                fake_A1, fake_A2 = net_G(data_B)
                loss_p.append(metric_gan(fake_A1, data_A1))
                loss_s.append(metric_gan(fake_A2, data_A2))

            # evaluation
            r2lp.append(r2_score(data_A1.cpu().detach().numpy().flatten(), fake_A1.cpu().detach().numpy().flatten()) )
            r2ls.append(r2_score(data_A2.cpu().detach().numpy().flatten(), fake_A2.cpu().detach().numpy().flatten()) )
            r2p = np.array(r2lp); r2s = np.array(r2ls)
            print("Validation: Rank %d: [Epoch %d/%d] [loss S: %f] [loss S: %f]" % \
                  (opt.gpuid, epoch, opt.epoches, torch.mean(torch.stack(loss_p)), torch.mean(torch.stack(loss_s)) ) )
            print("Validation: Rank %d: [Epoch %d/%d] The average of [r2p score: %f] [r2s score:% f]" % (opt.gpuid, epoch, opt.epoches, np.mean(r2p), np.mean(r2s) ) )
            print("Validation: Rank %d: [Epoch %d/%d] The median of [r2p score: %f] [r2s score: %f]" % (opt.gpuid, epoch, opt.epoches,  np.median(r2p), np.median(r2s) ) )
            print("Validation: Rank %d: [Epoch %d/%d] The maximum of: [r2p score: %f at %d] [r2s score: % f at % d]" % \
                  (opt.gpuid, epoch, opt.epoches, np.max(r2p), np.argmax(r2p), np.max(r2s), np.argmax(r2s) ))
            print("Validation: Rank %d: [Epoch %d/%d] The minimum of: [r2p score: %f at %d] [r2s score: % f at % d]" % \
                  (opt.gpuid, epoch, opt.epoches, np.min(r2p), np.argmin(r2p), np.min(r2s), np.argmin(r2s) ))
            set_requires_grad(net_G, True)
        ## save

        if epoch % opt.sample_interval == 0:
            netsave_name = os.path.join(save_dir, opt.out_name) + "-epoch-" + str(epoch)
            torch.save(net_G.cpu().state_dict(), netsave_name + "-G.pth")
            net_G.cuda(opt.gpuid)

            print("------ model saved ------")

        # lr step
        scheduler_G.step()
        # end batch

    print("------ finished ------")






if __name__ == '__main__':
    main()




