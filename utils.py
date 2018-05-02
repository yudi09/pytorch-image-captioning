from Vgg import Vgg
from Resnet import Resnet
from Alexnet import Alexnet
from DenseNet import DenseNet
from Inception import Inception
from Resnet152 import Resnet152
from SqueezeNet import SqueezeNet


def get_cnn(architecture = 'resnet18', embedding_dim = 300):
	if architecture == 'resnet18':
		cnn = Resnet(embedding_dim = embedding_dim)
	elif architecture == 'resnet152':
		cnn = Resnet152(embedding_dim = embedding_dim)
	elif architecture == 'alexnet':
		cnn = Alexnet(embedding_dim = embedding_dim) 
	elif architecture == 'vgg':
		cnn = Vgg(embedding_dim = embedding_dim) 
	elif architecture == 'inception':
		cnn = Inception(embedding_dim = embedding_dim) 
	elif architecture == 'squeeze':
		cnn = SqueezeNet(embedding_dim = embedding_dim) 
	elif architecture == 'dense':
		cnn = DenseNet(embedding_dim = embedding_dim) 
	return cnn