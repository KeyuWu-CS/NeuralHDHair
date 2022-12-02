from options.test_options import TestOptions
from dataload import data_loader
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
opt=TestOptions().parse()
gpu_str = [str(i) for i in opt.gpu_ids]
gpu_str = ','.join(gpu_str)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
dataloader=data_loader(opt)
if opt.model_name=="GrowingNet":
    g_slover = GrowingNetSolver()
elif opt.model_name=="HairSpatNet":
    g_slover = HairSpatNetSolver()
elif opt.model_name=='HairModeling':
    g_slover=HairModelingHDSolver()


g_slover.initialize(opt)
g_slover.test(dataloader)