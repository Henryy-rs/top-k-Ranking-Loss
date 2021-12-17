import torch
from torch.autograd import Variable
import argparse
import os
import logging
import numpy as np
import skimage.io as io

from utils.utils import build_transforms, get_torch_device
from utils.load_model import load_feature_extractor


def extract_feature():
	device = get_torch_device()
	net = load_feature_extractor(MODEL_TYPE, MODEL_PATH, device).eval()

	# current location
	temp_path = os.path.join(os.getcwd(), 'temp')
	if not os.path.exists(temp_path):
		os.mkdir(temp_path)

	video_list = []
	for video_folder in os.listdir(VIDEO_DIR):
		temp_folder = os.path.join(temp_path, video_folder)

		if not os.path.exists(temp_folder):
			os.mkdir(temp_folder)

		for video in os.listdir(os.path.join(VIDEO_DIR, video_folder)):
			video_list.append(os.path.join(video_folder, video))

	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	error_fid = open('error.txt', 'w')
	num_videos = len(video_list)

	for video_num, video_name in enumerate(video_list):
		video_path = os.path.join(VIDEO_DIR, video_name)
		video_name = video_name.split('.')[0]
		video_folder = video_name.split('\\')[0]
		frame_path = os.path.join(temp_path, video_name)

		if not os.path.exists(frame_path):
			os.mkdir(frame_path)

		print('Extracting video frames ...')
		# using ffmpeg to extract video frames into a temporary folder
		# example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
		os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%6d.jpg')

		print('Extracting features ...')
		total_frames = len(os.listdir(frame_path))
		if total_frames == 0:
			error_fid.write(video_name+'\n')
			print('Fail to extract frames for video: %s'%video_name)
			continue

		valid_frames = total_frames / nb_frames * nb_frames
		n_feat = valid_frames / nb_frames
		n_batch = n_feat / BATCH_SIZE

		if n_feat - n_batch*BATCH_SIZE > 0:
			n_batch = n_batch + 1

		n_batch = int(n_batch)
		n_feat = int(n_feat)
		features = []
		print("progress: {}/{}, current_video: {}, n_batch={}, batch_size={}".format(video_num+1, num_videos, video_name, n_batch, BATCH_SIZE))

		for i in range(n_batch-1):
			input_blobs = None
			# print("n_batch={}, cur_batch={}".format(n_batch, i))
			for j in range(BATCH_SIZE):
				clip = np.array([io.imread(os.path.join(frame_path, 'image_{:06d}.jpg'.format(k))) for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
				clip = build_transforms(mode=MODEL_TYPE)(torch.from_numpy(clip))
				clip = clip[None, :]

				if input_blobs is None:
					input_blobs = clip
				else:
					input_blobs = torch.cat([input_blobs, clip], dim=0)

			input_blobs = Variable(input_blobs).to(device)
			batch_output = net(input_blobs)
			batch_feature = batch_output.data.cpu()
			features.append(batch_feature)
			torch.cuda.empty_cache()

		# The last batch
		input_blobs = None
		# print("n_batch={}, cur_batch={}".format(n_batch, i+1))

		for j in range(n_feat-(n_batch-1)*BATCH_SIZE):
			clip = np.array([io.imread(os.path.join(frame_path, 'image_{:06d}.jpg'.format(k))) for k in range(((n_batch-1)*BATCH_SIZE+j) * nb_frames+1, min(((n_batch-1)*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
			clip = build_transforms(mode=MODEL_TYPE)(torch.from_numpy(clip))
			clip = clip[None, :]

			if input_blobs is None:
				input_blobs = clip
			else:
				input_blobs = torch.cat([input_blobs, clip], dim=0)

		input_blobs = Variable(input_blobs).to(device)
		batch_output = net(input_blobs)
		batch_feature = batch_output.data.cpu()
		features.append(batch_feature)
		torch.cuda.empty_cache()

		features = torch.cat(features, 0)
		features = features.numpy()
		segments_feature = []
		num = 32
		thirty2_shots = np.round(np.linspace(0, len(features) - 1, num=num+1)).astype(int)
		for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
			if ss == ee:
				temp_vect = features[min(ss, features.shape[0] - 1), :]
			else:
				temp_vect = features[ss:ee, :].mean(axis=0)

			temp_vect = temp_vect / np.linalg.norm(temp_vect)
			if np.linalg.norm == 0:
				logging.error("Feature norm is 0")
				exit()
			if len(temp_vect) != 0:
				segments_feature.append(temp_vect.tolist())

		path = os.path.join(OUTPUT_DIR, f"{video_name}.txt")
		video_folder_dir = os.path.join(OUTPUT_DIR, video_folder)
		if not os.path.isdir(video_folder_dir):
			os.mkdir(video_folder_dir)

		with open(path, 'w') as fp:
			for d in segments_feature:
				d = [str(x) for x in d]
				fp.write(' '.join(d) + '\n')

		print('%s has been processed...'%video_name)

		# clear temp frame folders
		os.system('rmdir /s /q ' + frame_path)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print('******--------- Extract 3dResNet features ------*******')
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='features', help='Output directory')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type=str, default='D:\\UCF-Crime', help='Input Video directory')
	parser.add_argument('-m', '--MODEL_PATH', dest='MODEL_PATH', type=str, default='pretrained/r3d152_KM_200ep.pth', help='Model path')
	parser.add_argument('-t', '--MODEL_TYPE', type=str, required=True, help="type of feature extractor", choices=['c3d', 'i3d', 'mfnet', '3dResNet'])
	parser.add_argument('-b', '--BATCH_SIZE', default=32, help='the batch size')
	args = parser.parse_args()
	params = vars(args) 	# convert to ordinary dict
	
	OUTPUT_DIR = params['OUTPUT_DIR']
	MODEL_PATH = params['MODEL_PATH']
	MODEL_TYPE = params['MODEL_TYPE']
	VIDEO_DIR = params['VIDEO_DIR']
	BATCH_SIZE = int(params['BATCH_SIZE'])
	nb_frames = 16

	extract_feature()
