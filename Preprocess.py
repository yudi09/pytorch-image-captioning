import os
import json
import time
import numpy as np
from PIL import Image
from shutil import copyfile


def read_captions(filepath):
	captions_dict = {}
	with open(filepath) as f:
		for line in f:
			line_split = line.split(sep='\t', maxsplit=1)
			caption = line_split[1][:-1]
			id_image = line_split[0].split(sep='#')[0]
			if id_image not in captions_dict:
				captions_dict[id_image] = [caption]
			else:
				captions_dict[id_image].append(caption)
	return captions_dict

def get_ids(filepath):
	ids = []
	with open(filepath) as f:
		for line in f:
			ids.append(line[:-1])
	return ids

def copyfiles(dir_output, dir_input, ids):
	if not os.path.exists(dir_output):
		os.makedirs(dir_output)
	for cur_id in ids:
		path_input = os.path.join(dir_input, cur_id)
		path_output = os.path.join(dir_output, cur_id)
		copyfile(path_input, path_output)

def write_captions(dir_output, ids, captions_dict):
	output_path = os.path.join(dir_output, 'captions.txt')
	output = []
	for cur_id in ids:
		cur_dict = {cur_id: captions_dict[cur_id]}
		output.append(json.dumps(cur_dict))
		
	with open(output_path, mode='w') as f:
		f.write('\n'.join(output))

def segregate(dir_images, filepath_token, captions_path_input):
	dir_output = {'train': 'train',
				  'dev'  : 'dev',
				  'test' : 'test'
				 }
	
	# id [caption1, caption2, ..]
	captions_dict = read_captions(filepath_token)
	
	# train, dev, test images mixture
	images = os.listdir(dir_images)
	
	# read ids
	ids_train = get_ids(captions_path_input['train'])
	ids_dev = get_ids(captions_path_input['dev'])
	ids_test = get_ids(captions_path_input['test'])
	
	# copy images to respective dirs
	copyfiles(dir_output['train'], dir_images, ids_train)
	copyfiles(dir_output['dev'], dir_images, ids_dev)
	copyfiles(dir_output['test'], dir_images, ids_test)
	
	# write id
	write_captions(dir_output['train'], ids_train, captions_dict)
	write_captions(dir_output['dev'], ids_dev, captions_dict)
	write_captions(dir_output['test'], ids_test, captions_dict)

def load_captions(captions_dir):
	caption_file = os.path.join(captions_dir, 'captions.txt')
	captions_dict = {}
	with open(caption_file) as f:
		for line in f:
			cur_dict = json.loads(line)
			for k, v in cur_dict.items():
				captions_dict[k] = v
	return captions_dict

if __name__ == '__main__':
	dir_images = 'images'
	dir_text = 'text'
	filename_token = 'Flickr8k.token.txt'
	filename_train = 'Flickr_8k.trainImages.txt'
	filename_dev = 'Flickr_8k.devImages.txt'
	filename_test = 'Flickr_8k.testImages.txt'
	filepath_token = os.path.join(dir_text, filename_token)
	captions_path_input = {'train': os.path.join(dir_text, filename_train),
						   'dev': os.path.join(dir_text, filename_dev),
						   'test': os.path.join(dir_text, filename_test)
						  }
	
	tic = time.time()
	segregate(dir_images, filepath_token, captions_path_input)
	toc = time.time()
	print('time: %.2f mins' %((toc-tic)/60))