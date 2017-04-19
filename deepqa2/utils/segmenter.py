# coding:utf8
'''  
Segmenter with Chinese  
'''

import jieba  
import langid


def segment_chinese_sentence(sentence):  
    '''
    Return segmented sentence.
    '''
    seg_list = jieba.cut(sentence, cut_all=False)
    seg_sentence = u" ".join(seg_list)
    return seg_sentence.strip().encode('utf8')


def process_sentence(sentence):  
    '''
    Only process Chinese Sentence.
    '''
    if langid.classify(sentence)[0] == 'zh':
        return segment_chinese_sentence(sentence)
    return sentence

if __name__ == "__main__":  
    print(process_sentence('飞雪连天射白鹿'))
    print(process_sentence('I have a pen.'))