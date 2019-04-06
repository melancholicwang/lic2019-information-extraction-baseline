import sys
sys.path.append('/root/melan/compt/ccf-2019/spo/base/information-extraction/bin/so_labeling')
from spo_data_reader import DataReader 
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')

def index_dic_inverse(index_dic):
    return {v:k for k,v in index_dic.items()}

data_generator = DataReader(
        wordemb_dict_path='../dict/word_idx',
        postag_dict_path='../dict/postag_dict',
        label_dict_path='../dict/label_dict',
        p_eng_dict_path='../dict/p_eng',
        train_data_list_path='../data/train_data.p',
        test_data_list_path='../data/dev_data.p', out_newm_path='')
ttt = data_generator.get_train_reader()
dtt = data_generator.get_test_reader(need_input=False)

train_fw = codecs.open('../data/train_data_nts.txt', 'w')
dev_fw = codecs.open('../data/dev_data_nts.txt', 'w')

idx_word_d = index_dic_inverse(data_generator._feature_dict['wordemb_dict'])
idx_label_d = index_dic_inverse(data_generator._feature_dict['so_label_dict'])

def get_bert_train(tgenerator, fw):
    for index, features in enumerate(tgenerator()):
        print(features)
        word_idx_list, postag_list, p_idx, label_list = features
        line0 = '\t'.join([idx_word_d[ix] for ix in word_idx_list])
        line1 = '\t'.join([idx_label_d[ix] for ix in label_list])
        print(line0)
        print(line1)
        fw.write(line0+'\n')
        fw.write(line1+'\n')

get_bert_train(ttt, train_fw)
get_bert_train(dtt, dev_fw)
