import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict

# 下载必要的 NLTK 数据
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def get_wordnet_pos(word):
    """
    将词性标注转换为 WordNet 格式。
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_sentence(sentence):
    """
    对句子进行分词和词形还原。
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence.lower())
    lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return lemmatized


def meteor_score(reference, hypothesis):
    """
    计算简化版的 METEOR 评分。

    参数:
    - reference: 参考译文（字符串）
    - hypothesis: 机器译文（字符串）

    返回:
    - METEOR 分数（0 到 100）
    """
    # 预处理
    ref_tokens = preprocess_sentence(reference)
    hyp_tokens = preprocess_sentence(hypothesis)

    # 创建匹配集
    ref_matches = defaultdict(int)
    hyp_matches = defaultdict(int)

    # 精确匹配
    for token in hyp_tokens:
        if token in ref_tokens and ref_matches[token] < ref_tokens.count(token):
            hyp_matches[token] += 1
            ref_matches[token] += 1

    # 词形还原匹配已经在预处理时完成

    # 计算匹配数量
    matches = sum(hyp_matches.values())

    # 计算精确率和召回率
    precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0

    # 计算 F1 分数
    if precision + recall > 0:
        f1 = (10 * precision * recall) / (recall + 9 * precision)
    else:
        f1 = 0

    # 简化版 METEOR 分数
    meteor = f1 * 100
    return meteor


def main_meteor():
    # 示例 1
    reference1 = "This is a test"
    hypothesis1 = "This is a test"
    score1 = meteor_score(reference1, hypothesis1)
    print(f"参考译文: \"{reference1}\"")
    print(f"机器译文: \"{hypothesis1}\"")
    print(f"METEOR 分数: {score1:.2f}\n")

    # 示例 2
    reference2 = "It is a guide to action which ensures that the military always obeys the commands of the party"
    hypothesis2 = "It is a guide to action that ensures that the military will forever heed Party commands"
    score2 = meteor_score(reference2, hypothesis2)
    print(f"参考译文: \"{reference2}\"")
    print(f"机器译文: \"{hypothesis2}\"")
    print(f"METEOR 分数: {score2:.2f}\n")

    # 示例 3
    reference3 = "The quick brown fox jumps over the lazy dog"
    hypothesis3 = "A fast dark fox leaps over a sleepy dog"
    score3 = meteor_score(reference3, hypothesis3)
    print(f"参考译文: \"{reference3}\"")
    print(f"机器译文: \"{hypothesis3}\"")
    print(f"METEOR 分数: {score3:.2f}\n")


if __name__ == "__main__":
    main_meteor()
