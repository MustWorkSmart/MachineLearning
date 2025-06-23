'''
steps (for windows; this is for multi-machine, each machine could have multi-GPU):

1) use ipconfig to get master node's IP address and update os.environ['MASTER_ADDR'] below

2) update this below (and other lines nearby) as needed:
    parser.add_argument('-g', '--gpus', default = 4, type=int, help='number of GPUs in a machine')
    
3) < these python main.py commands has been replaced by torchrun commands, see below >
For the master machine with an IP of updated below with such done in #1 above, use the following command:
python main.py --mid=0
For the other machine with another IP address, use the following command:
python main.py --mid=1
..  and --mid=2 for the next machine, and so on

note:
tried setting GLOO_SOCKET_IFNAME in environment variables to Wi-Fi, but it did not work
-> deleted such later on
'''

'''
Background info:

https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html

pre-reqs:
https://docs.pytorch.org/tutorials/beginner/dist_overview.html
https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html
https://docs.pytorch.org/docs/main/notes/ddp.html
'''

'''
from https://docs.pytorch.org//tutorials/intermediate/ddp_tutorial.html:
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py

modified to:
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=192.168.1.158:12345 main.py --mid=0
and for the 2nd machine:
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=192.168.1.158:12345 main.py --mid=1

'''

# Windows Defender Firewall with Advanced Security
# Add new rules (both inbound and outbound) for the port 12345 (the port used in the code below)

'''
Had issue of the 1st machine not able to ping into the 2nd one, although the 2nd one could ping into the 1st one, followed these from google AI:

Configure the Windows Defender Firewall
The most frequent culprit behind a one-way ping is the Windows Defender Firewall. By default, it may be configured to block incoming ICMP echo requests, which are the technical term for ping requests.

Solution:

You don't need to disable your entire firewall to fix this. Instead, you can create a specific inbound rule to allow these requests.


Open the Start Menu and search for "Windows Defender Firewall with Advanced Security" and open it.
In the left pane, click on Inbound Rules.
In the right pane, click on New Rule....
In the New Inbound Rule Wizard, select Custom and click Next.
Select All programs and click Next.
For the Protocol type, select ICMPv4 from the dropdown menu.
Click on the Customize... button.
Select Specific ICMP types, and then check the box for Echo Request. Click OK and then Next.
Leave the Scope settings as they are (Any IP address for both local and remote) unless you want to restrict which devices can ping you. Click Next.
Ensure Allow the connection is selected and click Next.
Choose the network profiles to which this rule applies (Private, Domain, Public). It's recommended to apply this at least to the Private profile. Click Next.
Give the rule a descriptive name, such as "Allow Inbound Ping," and click Finish.
Once this rule is enabled, your computer should start responding to pings.
'''

#also did this in the 2nd machine which did/does NOT have Wi-Fi/wireless set up:
#set GLOO_SOCKET_IFNAME=Ethernet
#such machine got this error (now resolved):
#RuntimeError: [enforce fail at C:\actions-runner\_work\pytorch\pytorch\pytorch\third_party\gloo\gloo\transport\uv\device.cc:164] false. Unable to find address for: Wi-Fi

import argparse
import os
import datetime

from my_net import *
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch import optim
#from torch.distributed.optim import DistributedOptimizer
#got this:
#ImportError: cannot import name 'DistributedOptimizer' from 'torch.distributed.optim' (C:\ProgramData\anaconda3\envs\pytorch_gpu_py312\Lib\site-packages\torch\distributed\optim\__init__.py)
#commented out the above line as it's NOT used below anyway for now
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DDP_sampler

train_all_set = datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))]))
train_set, val_set = torch.utils.data.random_split( train_all_set,
        			 [50000, 10000])

test_set = datasets.MNIST('./mnist_data', download=True, train=False,
              transform = transforms.Compose([transforms.ToTensor(), 
              transforms.Normalize((0.1307,),(0.3081,))]))

def net_setup():
	os.environ['MASTER_ADDR'] = '192.168.1.158'  # replace with the master node's IP address
	os.environ['MASTER_PORT'] = '12345'
	#os.environ['MASTER_PORT'] = '139'
	#os.environ['MASTER_PORT'] = '0' 
	#to to discover an available port automatically by default
	#https://github.com/pytorch/pytorch/issues/73320
	
	#for 139 used above, got this error with socket codes below:
	#OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
	#import socket
	#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	#server_address = (os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']))
	#sock.bind(server_address)

def checkpointing(rank, epoch, net, optimizer, loss):
	path = f"model{rank}.pt"
	torch.save({
				'epoch':epoch,
				'model_state':net.state_dict(),
				'loss': loss,
				'optim_state': optimizer.state_dict(),
				}, path)
	print(f"Checkpointing model {rank} done.")

def load_checkpoint(rank, machines, local_rank):
	path = f"model{rank}.pt"
	checkpoint = torch.load(path)
	#model = torch.nn.DataParallel(MyNet(), device_ids=[rank%machines])
	model = torch.nn.DataParallel(MyNet(), device_ids=[local_rank]) #seems like local_rank should be used instead
	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)

	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	model.load_state_dict(checkpoint['model_state'])
	optimizer.load_state_dict(checkpoint['optim_state'])
	return model, optimizer, epoch, loss
	
def validation(model, val_set):
	model.eval()
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
	correct_total = 0
	with torch.no_grad():
		for idx, (data, target) in enumerate(val_loader):
			output = model(data)
			predict = output.argmax(dim=1, keepdim=True).cuda()
			target = target.cuda()
			correct = predict.eq(target.view_as(predict)).sum().item() 
			#view_as is a PyTorch function that reshapes a tensor to have the same size as another tensor
			correct_total += correct
		acc = correct_total/len(val_loader.dataset)
	print(f"Validation Accuracy {acc}")

def train(local_rank, args):
	torch.manual_seed(123)
	world_size = args.machines*args.gpus
	rank = args.mid * args.gpus + local_rank
	print(f"INFO from train: world_size is {world_size} and rank is {rank} and mid[machine id number] is {args.mid} with {args.gpus} GPU")
	#dist.init_process_group('nccl', rank =rank, world_size = world_size,
	#seems like windows does not support 'nccl' backend, so using 'gloo' instead
	dist.init_process_group('gloo', rank =rank, world_size = world_size,
						 	init_method='tcp://192.168.1.158:12345',
							#init_method='tcp://192.168.1.158:0',
                            timeout=datetime.timedelta(seconds=60))
	print(f"INFO from train: dist.init_process_group done for rank {rank}")
	torch.cuda.set_device(local_rank)
	model = MyNet()
	local_train_sampler = DDP_sampler(datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))])), rank = rank, num_replicas = world_size) 
	local_train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))])),
							batch_size = 128,
							shuffle = False,
							sampler = local_train_sampler)

	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)
	model = DDP(model, device_ids=[local_rank])

	for epoch in range(args.epochs):
		#print(f"Epoch {epoch}")
		print(f"Epoch {epoch+1} of {args.epochs} training on GPU {rank}")
		for idx, (data, target) in enumerate(local_train_loader):
			data = data.cuda()
			target = target.cuda()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward() #pytorch/DDP triggers gradient synchronization across all processes
			optimizer.step()
			print(f"batch {idx} training :: loss {loss.item()}")
		checkpointing(rank, epoch, model, optimizer, loss.item())
		validation(model, val_set)
	print("Training Done!")
	dist.destroy_process_group()
	
def test(local_rank, args):
	world_size = args.machines*args.gpus
	rank = args.mid * args.gpus + local_rank
	print(f"INFO from test: world_size is {world_size} and rank is {rank}")
	#dist.init_process_group('nccl', rank =rank, world_size = world_size,
	#seems like windows does not support 'nccl' backend, so using 'gloo' instead
	dist.init_process_group('gloo', rank =rank, world_size = world_size,
						 	init_method='tcp://192.168.1.158:12345',
							#init_method='tcp://192.168.1.158:0',
                            timeout=datetime.timedelta(seconds=60))

	torch.cuda.set_device(local_rank)
	print(f"Load checkpoint {rank}")
	#model, optimizer, epoch, loss = load_checkpoint(rank, args.machines)
	model, optimizer, epoch, loss = load_checkpoint(rank, args.machines, local_rank) #seems like local_rank should be there
	print("Checkpoint loading done!")

	local_test_sampler = DDP_sampler(test_set, rank = rank, num_replicas = world_size)

	model.eval()
	local_test_loader = torch.utils.data.DataLoader(test_set, 
							batch_size=128,
							shuffle = False, 
							sampler = local_test_sampler)
	correct_total = 0
	with torch.no_grad():
		for idx, (data, target) in enumerate(local_test_loader):
			output = model(data)
			predict = output.argmax(dim=1, keepdim=True).cuda()
			target = target.cuda()
			correct = predict.eq(target.view_as(predict)).sum().item()
			correct_total += correct
		acc = correct_total/len(local_test_loader.dataset)
	print(f"GPU {rank}, Test Accuracy {acc}")
	print("Test Done!")
	dist.destroy_process_group()

def main():
	parser = argparse.ArgumentParser(description = 'distributed data parallel training')
	parser.add_argument('-m', '--machines', default=2, type=int, help='number of machines')
	#parser.add_argument('-g', '--gpus', default = 4, type=int, help='number of GPUs in a machine')
	parser.add_argument('-g', '--gpus', default = 1, type=int, help='number of GPUs in a machine')
	parser.add_argument('-id', '--mid', default = 0, type=int, help='machine id number')
	#parser.add_argument('-e', '--epochs', default = 10, type = int, help='number of epochs')
	parser.add_argument('-e', '--epochs', default = 2, type = int, help='number of epochs')
	args = parser.parse_args()
	net_setup()
	mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
	mp.spawn(test, nprocs=args.gpus, args=(args,), join=True)

if __name__ == '__main__':
    main()
