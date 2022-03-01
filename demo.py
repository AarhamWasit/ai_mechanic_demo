import argparse
import torch
import pandas as pd
import numpy as np
import librosa
import data_loader.cascade_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

import datasets.cascade_split_dataset as cd

from tqdm import tqdm
from parse_config import ConfigParser
from utils import MetricTracker
from sklearn.metrics import confusion_matrix

SAMPLE_RATE = 48000

def main(config, args):
    
    logger = config.get_logger('train')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    #checkpoint = np.load(config.resume, allow_pickle=True)
    #checkpoint = checkpoint.to(device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    total_loss = 0.0

    cascade_dataset = cd.CascadeSplitDataset('')

    target_dict = cascade_dataset.target_dict
    target_names = list(target_dict.keys())
    label_map = cascade_dataset.label_map
    class_names = list(label_map.keys())

    writer_names = []
    loss_names = ['overall_loss']
    metric_names = []
    metrics = [m.__name__ for m in metric_ftns]

    for name in target_names:
        loss_names.append('loss'+'/'+name)

    for met in metrics:
        for name in target_names:
            metric_names.append(met+'/'+name)

    writer_names = loss_names + metric_names

    val_metrics = MetricTracker(*writer_names)

    val_metrics.reset()

    cm_dict = {}

    demo_path = '/scratch/adam/git/ai_mechanic_demo/sample/demo_full.wav'
    #demo_path = '/scratch/adam/git/ai_mechanic_demo/sample/demo_3s.wav'

    #demo_path = config.sample

    with torch.no_grad():

        signal, _ = librosa.load(demo_path, sr=SAMPLE_RATE)

        # num_mfcc (int): Number of coefficients to extract
        # n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        # hop_length (int): Sliding window for FFT. Measured in # of samples
        num_mfcc=13
        n_fft=2048
        hop_length=512

        features = librosa.feature.mfcc(signal, SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T

        data = torch.tensor(features[np.newaxis, np.newaxis, :])

        target = torch.zeros((1,1,11), dtype=torch.long)

        data, target = data.to(device), target.to(device)

        output = model(data)

        # computing loss, metrics on test set
        overall_loss, loss_dict = loss_fn(output, target)

        #val_metrics.update('overall_loss', overall_loss.item())
        
        batch_size = data.shape[0]
        
        total_loss += overall_loss.item() * batch_size

        for loss_key in loss_dict:
            curr_index = target_dict[loss_key]
            curr_loss = loss_dict[loss_key]
            curr_output = output[curr_index]
            curr_target = torch.flatten(target[:,:,curr_index])
            #print(loss_key, curr_index, curr_output.shape, curr_target.shape)
            #print('loss_'+loss_key)
            val_metrics.update('loss/'+loss_key, curr_loss.item())

            curr_pred = torch.argmax(curr_output, dim=1)
            print(loss_key, label_map[loss_key][curr_pred.cpu().numpy()[0]])
            assert curr_pred.shape[0] == len(curr_target)
            #print(pd.unique(curr_pred.cpu().numpy()), pd.unique(curr_target.cpu().numpy()))
            cm = confusion_matrix(curr_target.cpu().numpy(), curr_pred.cpu().numpy())

            if loss_key in cm_dict:
                cm_dict[loss_key] += cm
            else:
                cm_dict[loss_key] = cm

            for met in metric_ftns:
                #print(met.__name__+'_'+loss_key)
                val_metrics.update(met.__name__+'/'+loss_key, met(curr_output, curr_target))
        
        #for i, metric in enumerate(metric_fns):
        #    total_metrics[i] += metric(output, target) * batch_size

    #print(cm_dict)

    #n_samples = len(data_loader.sampler)
    #log = {'loss': total_loss / n_samples}
    #log.update({
    #    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    #})
    
    log = val_metrics.result()
    
    logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--sample', default=None, type=str,
                      help='sample audio as wav file')

    config = ConfigParser.from_args(args)
    main(config, args)
