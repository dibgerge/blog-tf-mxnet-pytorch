import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import TorchDataset, TensorflowDataset, MxnetDataset
import time


def pytorch_model(batch_size=10, num_workers=4):
    from torchvision.models import resnet50, resnet101, densenet121
    import torch

    models_fcns = [resnet50, resnet101, densenet121]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = pd.DataFrame(columns=[f.__name__ for f in models_fcns],
                           index=['error', 'total_time', 'pred_time_mean', 'pred_time_std'])

    for model_fun in models_fcns:
        model = model_fun(pretrained=True).to(device)
        model.eval()

        dataset = TorchDataset(batch_size=batch_size, num_workers=num_workers)
        total_images = 0
        running_corrects = 0.0
        error = 0
        pred_times = []

        with tqdm(total=len(dataset)//batch_size, postfix=[dict(error='0')]) as t:
            total_time = time.time()
            for images, labels in dataset:
                images = images.to(device)
                start_time = time.time()
                preds = model(images)
                pred_times.append(time.time()-start_time)
                running_corrects += sum(preds.cpu().argmax(dim=1) == labels).numpy()
                total_images += batch_size
                error = (1.0 - running_corrects/total_images)*100.0
                t.postfix[0]['error'] = f'{error:.2f}'
                t.update()
            total_time = time.time() - total_time
        print(f'\nFinal Error: {error:.3f}')
        results[model_fun.__name__] = [error, total_time, np.mean(pred_times), np.std(pred_times)]
    results.to_csv('pytorch_results.csv')


def tensorflow_model(batch_size=10):
    from tensorflow.python.keras.applications.resnet50 import ResNet50
    from tensorflow.python.keras.applications.densenet import DenseNet121
    from tensorflow.python.keras.applications.mobilenet import MobileNet

    from tensorflow.python.keras.applications.resnet50 import preprocess_input as res_preprocess
    from tensorflow.python.keras.applications.densenet import preprocess_input as dense_preprocess
    from tensorflow.python.keras.applications.mobilenet import preprocess_input as mobile_preprocess

    model_funcs = [ResNet50, MobileNet, DenseNet121]
    model_names = ['resnet50', 'mobilenet', 'densenet121']
    preprocess_funcs = [res_preprocess, mobile_preprocess, dense_preprocess]
    results = pd.DataFrame(columns=model_names, index=['error', 'total_time'])

    for model_fun, model_name, preprocess_fun in zip(model_funcs, model_names, preprocess_funcs):
        data = TensorflowDataset(batch_size=batch_size, preprocess=preprocess_fun)
        model = model_fun()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print('starting')
        total_time = time.time()
        res = model.evaluate_generator(data,
                                       workers=4,
                                       use_multiprocessing=False,
                                       verbose=True)
        total_time = time.time() - total_time
        error = 100*(1 - res[1])
        print('Error: {:.2f}'.format(error))
        results[model_name] = [error, total_time]
    results.to_csv('tensorflow_results.csv')


def mxnet_model(batch_size=10, num_workers=4):
    import mxnet as mx
    import gluoncv

    model_names = ['ResNet50_v1', 'ResNet101_v1', 'MobileNet1.0', 'DenseNet121']

    results = pd.DataFrame(columns=model_names,
                           index=['error', 'total_time', 'pred_time_mean', 'pred_time_std'])

    for model_name in model_names:
        net = gluoncv.model_zoo.get_model(model_name, pretrained=True, ctx=mx.gpu(0))
        dataset = MxnetDataset(batch_size=batch_size, num_workers=num_workers)

        total_images = 0
        running_corrects = 0.0
        error = 0
        pred_times = []

        with tqdm(total=len(dataset)//batch_size, postfix=[dict(error='0')]) as t:
            total_time = time.time()
            for images, labels in dataset:
                images = images.copyto(mx.gpu(0))

                start_time = time.time()
                pred = net(images)
                pred_times.append(time.time()-start_time)

                running_corrects += sum(pred.asnumpy().argmax(axis=1) == labels.asnumpy())
                total_images += batch_size
                error = (1.0 - running_corrects/total_images)*100.0
                t.postfix[0]['error'] = f'{error:.2f}'
                t.update()
            total_time = time.time() - total_time
        print(f'\nFinal Error: {error:.3f}')
        results[model_name] = [error, total_time, np.mean(pred_times), np.std(pred_times)]
    results.to_csv('mxnet_results.csv')


if __name__ == '__main__':
    pytorch_model()
    tensorflow_model()
    mxnet_model()
