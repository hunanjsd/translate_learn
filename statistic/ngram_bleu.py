import math
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def tokenize_sentence(sentence):
    """
    对句子进行分词和小写化处理。
    """
    return word_tokenize(sentence.lower())


def get_ngrams(tokens, n):
    """
    生成 n-gram 列表。
    """
    return list(ngrams(tokens, n))


def count_ngrams(ngrams_list):
    """
    计算 n-gram 的频率。
    """
    counts = defaultdict(int)
    for ng in ngrams_list:
        counts[ng] += 1
    return counts


def compute_clip_counts(candidate_counts, reference_counts):
    """
    计算修剪后的 n-gram 匹配数。
    """
    clip_counts = defaultdict(int)
    for ng in candidate_counts:
        if ng in reference_counts:
            clip_counts[ng] = min(candidate_counts[ng], reference_counts[ng])
    return clip_counts


def compute_precision(clip_counts, total_candidate_ngrams):
    """
    计算 n-gram 的精确率。
    """
    if total_candidate_ngrams == 0:
        return 0
    return sum(clip_counts.values()) / total_candidate_ngrams


def brevity_penalty(candidate_len, reference_len):
    """
    计算简短惩罚 BP。
    """
    if candidate_len > reference_len:
        return 1
    elif candidate_len == 0:
        return 0
    else:
        return math.exp(1 - reference_len / candidate_len)


def calculate_bleu(candidate, references, max_n=4):
    """
    计算 BLEU 分数。

    参数:
    - candidate: 机器翻译结果（字符串）
    - references: 参考译文列表（列表的列表）
    - max_n: 使用的最大 n-gram
    """
    candidate_tokens = tokenize_sentence(candidate)
    reference_tokens = [tokenize_sentence(ref) for ref in references]

    # 选择参考译文中与候选译文长度最接近的一个
    ref_lengths = [len(ref) for ref in reference_tokens]
    closest_ref_len = min(ref_lengths, key=lambda x: (abs(x - len(candidate_tokens)), x))

    precisions = []
    for n in range(1, max_n + 1):
        # 生成 n-gram
        cand_ngrams = get_ngrams(candidate_tokens, n)
        cand_counts = count_ngrams(cand_ngrams)

        # 统计所有参考译文的 n-gram 频率
        max_ref_counts = defaultdict(int)
        for ref in reference_tokens:
            ref_ngrams = get_ngrams(ref, n)
            ref_counts = count_ngrams(ref_ngrams)
            for ng in ref_counts:
                if ref_counts[ng] > max_ref_counts[ng]:
                    max_ref_counts[ng] = ref_counts[ng]

        # 计算修剪后的 n-gram 匹配数
        clip_counts = compute_clip_counts(cand_counts, max_ref_counts)

        # 计算精确率
        precision = compute_precision(clip_counts, len(cand_ngrams))
        precisions.append(precision)

    # 几何平均
    if min(precisions) > 0:
        geo_mean = math.exp(sum((1.0 / max_n) * math.log(p) for p in precisions))
    else:
        geo_mean = 0

    # 简短惩罚
    bp = brevity_penalty(len(candidate_tokens), closest_ref_len)

    # 最终 BLEU 分数
    bleu = bp * geo_mean
    return bleu * 100  # 转换为 0-100 分数


def main_custom_bleu():
    # 参考译文
    references = [
        "This is a test",
        "This is test"
    ]

    # 机器翻译结果
    candidate = "This is a test"

    bleu_score = calculate_bleu(candidate, references)

    print(f"参考译文: {references}")
    print(f"机器译文: {candidate}")
    print(f"自定义 BLEU 分数: {bleu_score:.2f}")

    # 另一个示例
    references2 = [
        "It is a guide to action which ensures that the military always obeys the commands of the party",
        "It is the guiding principle which guarantees the military forces always being under the command of the Party"
    ]
    candidate2 = "It is a guide to action that ensures that the military will forever heed Party commands"

    bleu_score2 = calculate_bleu(candidate2, references2)
    print(f"\n参考译文: {references2}")
    print(f"机器译文: {candidate2}")
    print(f"自定义 BLEU 分数: {bleu_score2:.2f}")


if __name__ == "__main__":
    main_custom_bleu()
