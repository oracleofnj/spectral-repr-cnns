import pickle
import requests
import tarfile
import os
import numpy as np
import tensorflow as tf
from IPython.display import display, HTML

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def download_cifar10(download_100=False):
    """Download cifar-10 tarzip file and unzip it."""
    if download_100:
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    else:
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split("/")[-1]
    fpath = os.path.join(BASE_DIR, filename)

    if os.path.exists(fpath):
        print('file already downloaded..')
        return

    # download file:
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # unzip file:
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()


def load_cifar10(num_batches=5,
                 get_test_data=True,
                 channels_last=True):
    """Load the cifar data.

    Args:
        num_batches: int, the number of batches of data to return
        get_test_data: bool, whether to return test data
    Returns:
        (images, labels) it get_test_data False
        (images, labels, test_images, test_labels) otherwise
        images are numpy arrays of shape:
                    (num_images, num_channels, width, height)
        labels are 1D numpy arrays
    """
    assert num_batches <= 5
    # download if not exists:
    download_cifar10()

    # load batches in order:
    dirpath = os.path.join(BASE_DIR, 'cifar-10-batches-py')
    images = None
    for i in range(1, num_batches + 1):
        print('getting batch {0}'.format(i))
        filename = 'data_batch_{0}'.format(i)
        fpath = os.path.join(dirpath, filename)
        with open(fpath, 'rb') as f:
            content = pickle.load(f, encoding='bytes')
        if images is None:
            images = content[b'data']
            labels = content[b'labels']
        else:
            images = np.vstack([images, content[b'data']])
            labels.extend(content[b'labels'])
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format:
    images = images.reshape(-1, 3, 32, 32)

    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        images = np.moveaxis(images, 1, -1)

    if not get_test_data:
        return images, labels

    filename = 'test_batch'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'labels'])

    # normalize:
    test_images = test_images / 255.
    # make channels last:
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels


def load_cifar100(get_test_data=True,
                  channels_last=True):
    """Load the cifar 100 data (not in batches).

    Args:
        get_test_data: bool, whether to return test data
    Returns:
        (images, labels) it get_test_data False
        (images, labels, test_images, test_labels) otherwise
        images are numpy arrays of shape:
                    (num_images, num_channels, width, height)
        labels are 1D numpy arrays
    """
    download_cifar10(download_100=True)

    # load batches in order:
    dirpath = os.path.join(BASE_DIR, 'cifar-100-python')
    images = None
    filename = 'train'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    if images is None:
        images = content[b'data']
        labels = content[b'fine_labels']
    # convert to labels:
    labels = np.asarray(labels)
    # convert to RGB format:
    images = images.reshape(-1, 3, 32, 32)

    # normalize data by dividing by 255:
    images = images / 255.
    if channels_last:
        images = np.moveaxis(images, 1, -1)

    if not get_test_data:
        return images, labels

    filename = 'test'
    fpath = os.path.join(dirpath, filename)
    with open(fpath, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    test_images = content[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.asarray(content[b'fine_labels'])

    # normalize:
    test_images = test_images / 255.
    # make channels last:
    if channels_last:
        test_images = np.moveaxis(test_images, 1, -1)

    return images, labels, test_images, test_labels


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def.
    This function has been taken from the homework assignments
    given by the TAs"""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph.
    This function has been taken from the homework assignments
    given by the TAs"""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1000px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
