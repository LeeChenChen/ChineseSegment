import re
import numpy as np
import json
from collections import Counter,defaultdict
import tqdm
import os
import tensorflow as tf
import sys, getopt

def set_config():
    #设置在3卡上运行,占用上限50%
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.5
    return  config

def get_data(path):
    all_tags=[]
    all_words=[]
    stops=u'，。！？；、：,\.!\?;:\n'
    i=0
    with open(path,'r') as f:
        txt=[line.strip(' ') for line in re.split('(['+stops+'])',f.read()) if line.strip(' ')]
        for line in txt:
            i+=1
            all_words.append('')
            all_tags.append('')
            for word in re.split(' +',line):
                all_words[-1]+=word
                if len(word)==1:
                    all_tags[-1]+='S'
                else :
                    all_tags[-1]+='B'+(len(word)-2)*'M'+'E'

    lens=[len(i) for i in all_words]

    lens=np.argsort(lens)[::-1]#从大到小排序
    all_words=[all_words[i] for i in lens]
    all_tags=[all_tags[i] for i in lens]
    return  all_words,all_tags

def data2batch(all_words,all_tags,word_id,tag2vec,batch_size=256):
    # batch_size = 256
    l=len(all_words[0])
    x=[]
    y=[]
    for i in range(len(all_words)):
        if len(all_words[i])!=l or len(x)==batch_size:
            yield x,y
            x=[]
            y=[]
            l=len(all_words[i])
        x.append([word_id[j] for j in all_words[i]])
        y.append([tag2vec[j] for j in all_tags[i]])

    
def word2dic(all_words):
    min_count=2
    word_count=Counter(''.join(all_words))
    word_count=Counter({word:index for word,index in word_count.items() if index>=min_count})
    word_id=defaultdict(int)
    id = 0
    for i in  word_count.most_common():
        id+=1
        word_id[i[0]]=id
    vacabulary_size=len(word_id)+1
    return word_count,word_id,vacabulary_size


def get_test_data(all_words,all_tags,word_id,tag2vec):
    x=[]
    y=[]
    for i in range(len(all_words)):
        x.extend([word_id.get(j,4735) for j in all_words[i]])
        y.extend([tag2vec[j] for j in all_tags[i]])
    return [x],[y]
    
def build_graph(vacabulary_size):
    embedding_size = 128
    keep_prob = tf.placeholder(tf.float32)

    # embedding layer
    embeddings = tf.Variable(tf.random_uniform([vacabulary_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, None])
    embedded = tf.nn.embedding_lookup(embeddings, x)
    embedded_dropout = tf.nn.dropout(embedded, keep_prob)
    
    # converlution layer1
    W1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
    b1 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0), dtype=tf.float32)
    y1 = tf.nn.relu(tf.nn.conv1d(embedded_dropout, W1, stride=1, padding='SAME') + b1)
    
    # converlution layer2
    W2 = tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size / 4)], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([int(embedding_size / 4)], -1.0, 1.0))
    y2 = tf.nn.relu(tf.nn.conv1d(y1, W2, stride=1, padding='SAME') + b2)
    
    # converlution layer3
    W3 = tf.Variable(tf.random_uniform([3, int(embedding_size / 4), 4], -1.0, 1.0))
    b3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
    y = tf.nn.softmax(tf.nn.conv1d(y2, W3, stride=1, padding='SAME') + b3)
  
    return x,y,keep_prob

def cnn_train(vacabulary_size,all_words,all_tags,word_id,tag2vec,config,epoch=50):
    
    x,y,keep_prob = build_graph(vacabulary_size)
    
    # loss
    y_ = tf.placeholder(tf.float32, shape=[None, None, 4])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-20))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 2), tf.argmax(y_, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    
    # sess.run(init)
    saver=tf.train.Saver()
    saver.restore(sess,'./model/frist_model.ckpt')
    
    # epoch = 50
    for i in range(epoch):
        temp_data = tqdm.tqdm(data2batch(all_words, all_tags, word_id, tag2vec,batch_size=512), desc=u'Epcho %s,Accuracy:0.0' % (i + 1))
        k = 0
        accs = []
        for x_data, y_data in temp_data:
            k += 1
            if k % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: x_data, y_: y_data, keep_prob: 1})
                accs.append(acc)
                temp_data.set_description('Epcho %s, Accuracy: %s' % (i + 1, acc))
            sess.run(train_step, feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
        print(u'Epcho %s Mean Accuracy: %s' % (i + 1, np.mean(accs)))

    # checkpoint
    saver = tf.train.Saver()
    saver.save(sess, './model/frist_model.ckpt')

def model_test(vac_size,x_data,y_data,predict=False):
    x,y_pre,keep_prob = build_graph(vac_size)
    y=tf.placeholder(tf.float32,shape=[None,None,4])

    config=set_config()
    sess=tf.Session(config=config)
    saver=tf.train.Saver()
    saver.restore(sess,'./model/frist_model.ckpt')

    correct_pre=tf.equal(tf.argmax(y,2),tf.argmax(y_pre,2))
    acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))

    if predict:
        result = sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        return result

    else:
        sess.run(y_pre, feed_dict={x: x_data, keep_prob: 0.5})
        scores = sess.run(acc, feed_dict={x: x_data, y: y_data, keep_prob: 1.0})
        print(scores)


def viterbi(result,trans_pro):
    nodes=[dict(zip( ('S','B','M','E'),i )) for i in result]
    paths=nodes[0]
    for t in range(1,len(nodes)):
        path_old=paths.copy()
        paths={}
        for i in nodes[t]:
            nows={}
            for j in path_old:
                if j[-1]+i in trans_pro:
                    nows[j+i]=path_old[j]+nodes[t][i]+trans_pro[j[-1]+i]
            pro,key=max([(nows[key],key) for key,value in nows.items()])
            paths[key]=pro
    best_pro,best_path=max([(paths[key],key)for key,value in paths.items()])
    return best_path

def segword(txt,best_path):
    begin,end=0,0
    seg_word=[]
    for index,char in enumerate(txt):
        signal=best_path[index]
        if signal=='B':
            begin=index
        elif signal=='E':
            seg_word.append(txt[begin:index+1])
            end=index+1
        elif signal=='S':
            seg_word.append(char)
            end=index+1
    if end<len(txt):
        seg_word.append(txt[end:])
    return seg_word

def cnn_seg(txt):
    word_id=json.load(open('word_id.json','r'))
    vacabulary_size=len(word_id)+1
    trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}
    trans_pro={state:np.log(num) for state,num in trans_pro.items()}

    txt2id=[[word_id.get(word,4735)for word in txt]]
    result=model_test(vacabulary_size,x_data=txt2id,y_data=None,predict=True)
    result = result[0, :, :]
    best_path=viterbi(result,trans_pro)

    return  segword(txt,best_path)

def cnn_test(path="./data/msr_test_gold.utf8"):
    all_words, all_tags = get_data(path)
    word_id=json.load(open('word_id.json','r'))
    vacabulary_size=len(word_id)+1
    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    x, y = get_test_data(all_words, all_tags, word_id, tag2vec)
    model_test(vacabulary_size, x_data=x, y_data=y)

if __name__ == '__main__':
    mode = sys.argv[1:]
    if mode[0] == 'train':
        all_words, all_tags=get_data('data/msr_training.utf8')
        word_count,word_id,vacabulary_size=word2dic(all_words)
        tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
        
        json_str = json.dumps(word_id)
        with open('word_id.json', 'w') as json_file:
            json_file.write(json_str)
        
        config=set_config()
        cnn_train(vacabulary_size,all_words,all_tags,word_id,tag2vec,config,epoch=10)
        
    elif mode[0] == 'test':
        #cnn_test()
        print(cnn_seg("但众所周知，基于字标注法的分词，需要标签语料训练，训练完之后，就适应那一批语料了，比较难拓展到新领域；又或者说，如果发现有分错的地方，则没法很快调整过来。"))
    