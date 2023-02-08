# inverse gradient for interpretating the PS wave separation
#              Author: Yanwen Wei
#              Email: wei_yanwen@163.com
#              Date: 2021-12-28
#
import torch
import  numpy as np

class npy_2to2_Dataloader_getindex():
    def __init__(self, data_path, nt, nr, nph, nop, max_data, data_norm_type, data_start, jsample, seek_num=0):
        self.data_path_B1 = data_path + "/vz"
        self.data_path_B2 = data_path + "/vx"
        self.data_path_A1 = data_path + "/div"
        self.data_path_A2 = data_path + "/curl"
        self.nt = nt
        self.nr = nr
        self.nph = nph  ## 2
        self.nop = nop  ## 1
        self.max_data = max_data
        self.norm_type = data_norm_type
        self.data_start = data_start
        self.jsample = jsample
        self.seek_num = seek_num

    def __len__(self):
        return self.max_data

    def get2exp(self, num):
        for exp in range(0, 12):
            if num <= 2 ** exp:
                return 2 ** exp
                break
            if num > 2 ** 11:
                return 2 ** 11 + 2 ** 10

    def __getitem__(self, index):
        this_index = self.data_start + index * self.jsample
        B_tensor = torch.zeros((self.nph, self.get2exp(self.nt), self.get2exp(self.nr)))
        A1_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        A2_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        B1 = np.load(self.data_path_B1 + "/vz_" + str(this_index) + ".npy")
        B2 = np.load(self.data_path_B2 + "/vx_" + str(this_index) + ".npy")
        A1 = np.load(self.data_path_A1 + "/div_" + str(this_index) + ".npy")
        A2 = np.load(self.data_path_A2 + "/curl_" + str(this_index) + ".npy")

        start_a = int((self.get2exp(self.nt) - self.nt) / 2)
        start_b = int((self.get2exp(self.nr) - self.nr) / 2)
        B_tensor[0, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B1)
        B_tensor[1, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B2)
        A1_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A1)
        A2_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A2)
        return B_tensor, A1_tensor, A2_tensor, this_index

class npy_1to2_Dataloader_getindex():
    def __init__(self, data_path, nt, nr, nph, nop, max_data, data_norm_type, data_start, jsample, seek_num=0):
        self.data_path_B1 = data_path + "/vz"
        # self.data_path_B2 = data_path + "/vx"
        self.data_path_A1 = data_path + "/div"
        self.data_path_A2 = data_path + "/curl"
        self.nt = nt
        self.nr = nr
        self.nph = 1 #nph  ## constant
        self.nop = 1 #nop  ## constant
        self.max_data = max_data
        self.norm_type = data_norm_type
        self.data_start = data_start
        self.jsample = jsample
        self.seek_num = seek_num

    def __len__(self):
        return self.max_data

    def get2exp(self, num):
        for exp in range(0, 12):
            if num <= 2 ** exp:
                return 2 ** exp
                break
            if num > 2 ** 11:
                return 2 ** 11 + 2 ** 10

    def __getitem__(self, index):
        this_index = self.data_start + index * self.jsample
        B_tensor = torch.zeros((self.nph, self.get2exp(self.nt), self.get2exp(self.nr)))
        A1_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        A2_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        B1 = np.load(self.data_path_B1 + "/vz_" + str(this_index) + ".npy")
        # B2 = np.load(self.data_path_B2 + "/vx_" + str(this_index) + ".npy")
        A1 = np.load(self.data_path_A1 + "/div_" + str(this_index) + ".npy")
        A2 = np.load(self.data_path_A2 + "/curl_" + str(this_index) + ".npy")

        start_a = int((self.get2exp(self.nt) - self.nt) / 2)
        start_b = int((self.get2exp(self.nr) - self.nr) / 2)
        B_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B1)
        # B_tensor[1, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B2)
        A1_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A1)
        A2_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A2)
        return B_tensor, A1_tensor, A2_tensor, this_index


