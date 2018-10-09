import matplotlib
matplotlib.use('Agg')
import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def read_data(path):
    with open(path) as f:
        text = f.read()
    return text

def preprocess(text, freq=5):
    """
    Preprocessing the data
    :param text: data
    :param freq: threshold
    :return:
    """
    text = text.lower()
    text = text.replace('.', '<PERIOD>')
    text = text.replace(',', '<COMMA>')
    text = text.replace('"', '<QUOTATION_MARK>')
    text = text.replace(';', '<SEMICOLON>')
    text = text.replace('!', '<EXCLAMATION_MARK>')
    text = text.replace('?', '<QUESTION_MARK>')
    text = text.replace('(', '<LEFT_PAREN>')
    text = text.replace(')', '<RIGHT_PAREN>')
    text = text.replace('--', '<HYPHENS>')
    text = text.replace(':', '<COLON>')

    words = text.split()
    print('Original number of word is: {}'.format(len(words)))
    words_count = Counter(words)
    trimmed_words = [word for word in words if words_count[word] > freq]

    return trimmed_words


def build_vocab(words):
    file = open('vocab.txt', 'w')
    dictionary = dict()
    count = [('UNK', -1)]
    index = 0
    count.extend(Counter(words).most_common())
    for word, _n in count:
        dictionary[word] = index
        index += 1
        file.write(word + '\n')
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary



def convert_words_to_index(words, dictionary):
    return [dictionary[w] if w in dictionary else 0  for w in words]



def subsample(t, threshold, index_words):
    t = 1e-5
    threshold = 0.8
    total_count = len(index_words)
    words_count = Counter(index_words).most_common()
    words_freq = {word: count / total_count for word, count in words_count}
    words_prob = {word: 1 - np.sqrt(t / words_freq[word]) for word, _ in words_count}
    trim_words = [word for word in index_words if words_prob[word] < threshold]
    return trim_words

def generate_sample(index_words, context_window_size):
    for index, center in enumerate(index_words):
        real_window_size = random.randint(1, context_window_size)
        for target in index_words[max(0, index - real_window_size):index]:
            yield center, target
        for target in index_words[index+1:index + real_window_size+1]:
            yield center, target

def gen_batch(batch_size, context_window_size, sample):
    while True:
        center_batch = np.zeros(batch_size, dtype = np.int32)
        target_batch = np.zeros([batch_size, 1], dtype = np.int32)
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(sample, 'None')
        yield center_batch, target_batch


def word2vec(index_words, batch_size, epoches, context_window_size,
             embedding_size, vocab_size, num_sampled, num_skip, index_dictionary):
    # get the dataset

    # construct input
    centers = tf.placeholder(tf.int32, shape=[None], name='centers')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

    # construct network/inference
    with tf.variable_scope('embdding_matrix'):
        embed_matrix = tf.get_variable(name='embed_matrix', shape=[vocab_size, embedding_size],
                                       dtype=tf.float32,initializer=tf.random_uniform_initializer())
        embedded = tf.nn.embedding_lookup(embed_matrix, centers, name='embed')
    # define the loss function2
    with tf.variable_scope('loss'):
        nge_weight = tf.get_variable(name='nge_weight', shape=[vocab_size, embedding_size],
                                     dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0 / (embedding_size ** 0.5)))
        nge_bias = tf.get_variable(name='nge_bias', initializer=tf.zeros([vocab_size]))

        nge_loss = tf.nn.sampled_softmax_loss(weights=nge_weight, biases=nge_bias, labels=targets, inputs=embedded,
                                              num_sampled=num_sampled, num_classes=vocab_size)
        nge_loss = tf.reduce_mean(nge_loss)
    # define the optimizer
    #optimizer = tf.train.AdamOptimizer().minimize(nge_loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(nge_loss)
    # training
    start = time.time()
    sample = generate_sample(index_words, context_window_size)
    genbatch = gen_batch(batch_size, context_window_size, sample)

    valid_size = 16
    valid_window = 100
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window),
                                                             valid_size//2))
    valid_size = len(valid_examples)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), 1, keep_dims=True))

    normalized_embedding = embed_matrix / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


    with tf.Session() as sess:
        # initiliaze all the variables
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        for epoch in range(epoches):
            training_sample = next(genbatch)
            if (training_sample == 'None'):
                sample = None
                sample = generate_sample(index_words, context_window_size)
                genbatch = None
                genbatch = gen_batch(batch_size, context_window_size, sample)
            inputs, labels = training_sample
            feed = {centers: inputs, targets: labels}
            loss_batch, _ = sess.run([nge_loss, optimizer], feed_dict=feed)
            total_loss += loss_batch
            if (epoch + 1) % num_skip == 0:
                print('Average loss at step {}: {:5.1f}'.format(epoch, total_loss / num_skip))
                total_loss = 0.0

        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = index_dictionary[valid_examples[i]]
            top_k = 8
            nearest = (-sim[i, :]).argsort()[1: top_k + 1]
            log = 'Nearest to [%s]:'% valid_word
            for k in range(top_k):
                close_word = index_dictionary[nearest[k]]
                log = '%s %s,'% (log, close_word)
            print(log)
        embed_mat = sess.run(normalized_embedding)
    viz_words = 100
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
    fig, ax = plt.subplots(figsize=(14,14))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(index_dictionary[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

    plt.savefig('data/fig.png')




# batch_size = 128
# epoches = 100000
# context_window_size = 1
# embedding_size = 128
# vocab_size = len(dictionary)
# num_sampled = 64
# num_skip = 5000
# word2vec(index_words,batch_size, epoches, context_window_size, embedding_size, vocab_size,num_sampled,num_skip)

if __name__ == '__main__':
    filepath = 'data/text8'
    text = read_data(filepath)
    words = preprocess(text)
    #print(words[:20])
    dictionary, index_dictionary = build_vocab(words)
    index_words = convert_words_to_index(words, dictionary)
    batch_size = 128
    epoches = 200000
    context_window_size = 1
    embedding_size = 128
    vocab_size = len(dictionary)
    num_sampled = 64
    num_skip = 5000
    word2vec(index_words,batch_size, epoches, context_window_size, embedding_size,
             vocab_size,num_sampled,num_skip, index_dictionary)
