import numpy as np
from collections import defaultdict
import copy


class IBMModel1:
    def __init__(self):
        # 翻译概率 t(e|f)
        self.translation_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.vocab_f = set()
        self.vocab_e = set()

    def initialize_translation_probs(self, corpus_f, corpus_e):
        """
        初始化翻译概率 t(e|f) 为均匀分布
        """
        # 统计词汇表
        for f_sentence, e_sentence in zip(corpus_f, corpus_e):
            for f_word in f_sentence:
                self.vocab_f.add(f_word)
            for e_word in e_sentence:
                self.vocab_e.add(e_word)

        initial_prob = 1.0 / len(self.vocab_e)
        for f_word in self.vocab_f:
            for e_word in self.vocab_e:
                self.translation_probs[f_word][e_word] = initial_prob

    def train(self, corpus_f, corpus_e, num_iterations=10):
        """
        使用 EM 算法训练 IBM 模型 1
        """
        self.initialize_translation_probs(corpus_f, corpus_e)

        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}")
            count_ef = defaultdict(lambda: defaultdict(float))
            total_f = defaultdict(float)

            # E 步骤
            for f_sentence, e_sentence in zip(corpus_f, corpus_e):
                for f_word in f_sentence:
                    # 计算当前 f_word 对所有 e_word 的翻译概率之和
                    total_prob = sum(self.translation_probs[f_word][e_word] for e_word in e_sentence)
                    for e_word in e_sentence:
                        # 计算 p(e|f) = t(e|f) / sum(t(e’|f))
                        delta = self.translation_probs[f_word][e_word] / total_prob
                        count_ef[f_word][e_word] += delta
                        total_f[f_word] += delta

            # M 步骤
            for f_word in self.vocab_f:
                for e_word in self.vocab_e:
                    if total_f[f_word] > 0:
                        self.translation_probs[f_word][e_word] = count_ef[f_word][e_word] / total_f[f_word]

            # 可选：打印部分翻译概率以观察训练过程
            sample_f = next(iter(self.vocab_f))
            print(f"t({sample_f}|{sample_f}) = {self.translation_probs[sample_f][sample_f]:.4f}")

    def translate(self, sentence_f):
        """
        翻译一个中文句子为英文
        """
        translation = []
        for f_word in sentence_f:
            if f_word in self.translation_probs:
                # 找到具有最高翻译概率的 e_word
                e_word = max(self.translation_probs[f_word], key=self.translation_probs[f_word].get)
                translation.append(e_word)
            else:
                # 未见过的词保持不变或使用特殊标记
                translation.append(f_word)
        return translation


def main():
    # 简单的中英平行语料
    corpus_zh = [
        ['我', '爱', '猫'],
        ['我', '爱', '狗'],
        ['你', '爱', '狗'],
        ['他', '喜欢', '猫']
    ]

    corpus_en = [
        ['I', 'love', 'cats'],
        ['I', 'love', 'dogs'],
        ['you', 'love', 'dogs'],
        ['he', 'likes', 'cats']
    ]

    # 创建 IBM 模型 1 实例并训练
    ibm1 = IBMModel1()
    ibm1.train(corpus_zh, corpus_en, num_iterations=20)

    # 测试翻译
    test_sentences = [
        ['我', '喜欢', '狗'],
        ['你', '爱', '猫']
    ]

    for sentence in test_sentences:
        translation = ibm1.translate(sentence)
        print("\n中文句子:", ' '.join(sentence))
        print("翻译成英文:", ' '.join(translation))


if __name__ == "__main__":
    main()
