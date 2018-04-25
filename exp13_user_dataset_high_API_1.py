'''
import user's own dataset using high level API ---- From Tensor

reference:
https://www.tensorflow.org/programmers_guide/datasets
'''

import tensorflow as tf
import numpy as np
import glob
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def __read__(file, label):
    image = cv2.imread(file.decode(), cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, 2)
    image = np.float32(image) / 255.
    return (image, label)

if __name__ == "__main__":
    ## load into memory
    print('\nloading data into memory ...\n')
    data_path = 'my_data'
    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()
    labels = np.random.randint(0,2, size=(len(files),1))

    ## construct dataset
    print('\nconstructing Dataset & Iterator ...\n')
    # 1. build Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    # 2. preprocess with Dataset.map()
    dataset = dataset.map(
        lambda x, y: tf.py_func(__read__, [x, y], [tf.float32, y.dtype])
    )
    # 3. multiple epochs & batching
    dataset = dataset.repeat(10) # 10 epoches
    dataset = dataset.batch(64) # batch size: 64
    # 4. shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)

    ## construct iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    ## read data
    print('\nreading dataset ...\n')
    img_batch = tf.placeholder(tf.float32, [None, 180, 180, 1])
    tf.summary.image(name='display', tensor=img_batch, max_outputs=6)
    step = 0
    with tf.Session() as sess:
        summary = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter('logs', sess.graph)
        while True:
            try:
                batch_img, batch_label = sess.run(next_element)
                # print(batch_img.shape)
                # print(batch_label.shape)
                # print(type(batch_label[0,0]))
            except tf.errors.OutOfRangeError:
                print('\nEnd of dataset\n')
                break
            summ = sess.run(summary, {img_batch: batch_img})
            summ_writer.add_summary(summ, step)
            summ_writer.flush()
            step += 1
