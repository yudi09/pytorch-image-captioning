import os
import torch
import pickle
import argparse
from PIL import Image
import torch.nn as nn
from utils import get_cnn
from Decoder import RNN
from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from DataLoader import DataLoader, shuffle_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i')
	parser.add_argument('-model')
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	args = parser.parse_args()

	with open(os.path.join(args.model, 'vocab.pkl'), 'rb') as f:
	    vocab = pickle.load(f)

	transform = transforms.Compose([transforms.Resize((224, 224)), 
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])
	image = transform(Image.open(args.i))
	
	embedding_dim = 512
	vocab_size = vocab.index
	hidden_dim = 512
	model_name = args.model
	cnn = get_cnn(architecture = model_name, embedding_dim = embedding_dim)
	lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
	           vocab_size = vocab_size)
	# cnn.eval()

	image = image.unsqueeze(0)
	
	# image = Variable(image)
	if torch.cuda.is_available():
		with torch.cuda.device(args.gpu_device):
			cnn.cuda()
			lstm.cuda()
			image = Variable(image).cuda()
	else:
		image = Variable(image)

	iteration = args.epoch
	cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
	lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
	cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file)))
	lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file)))

	
	cnn_out = cnn(image)
	ids_list = lstm.greedy(cnn_out)
	print(vocab.get_sentence(ids_list))