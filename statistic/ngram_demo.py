import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random

# 下载必要的 NLTK 数据
# nltk.download('punkt')


def preprocess_corpus(corpus, n):
    """
    对语料库进行预处理，包括分词和添加起始、结束标记。
    """
    processed_corpus = []
    for sentence in corpus:
        tokens = word_tokenize(sentence.lower())  # 小写化和分词
        # 添加起始和结束标记
        tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
        processed_corpus.append(tokens)
    return processed_corpus


def build_ngram_counts(processed_corpus, n):
    """
    构建 n-gram 和 (n-1)-gram 的频率统计。
    """
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for sentence in processed_corpus:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n])
            context = tuple(sentence[i:i + n - 1])
            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts


def calculate_ngram_probabilities_laplace(ngram_counts, context_counts, n, vocabulary):
    """
    计算每个 n-gram 的条件概率 P(w_n | w_1, w_2, ..., w_{n-1})，并应用 Laplace 平滑。
    """
    ngram_probabilities = defaultdict(lambda: defaultdict(float))
    V = len(vocabulary)  # 词汇表大小

    for context in context_counts:
        for word in vocabulary:
            ngram = context + (word,)
            count = ngram_counts.get(ngram, 0)
            # 应用 Laplace 平滑
            probability = (count + 1) / (context_counts[context] + V)
            ngram_probabilities[context][word] = probability

    return ngram_probabilities


def generate_sentence(ngram_probabilities, n):
    """
    生成一个句子，基于训练好的 n-gram 条件概率。
    """
    sentence = ['<s>'] * (n - 1)
    while True:
        context = tuple(sentence[-(n - 1):])
        if context not in ngram_probabilities:
            break  # 无法继续生成
        next_words = list(ngram_probabilities[context].keys())
        probabilities = list(ngram_probabilities[context].values())
        next_word = random.choices(next_words, weights=probabilities, k=1)[0]
        if next_word == '</s>':
            break
        sentence.append(next_word)
    return ' '.join(sentence[n - 1:])


def main():
    # 示例文本数据
    corpus = [
        "I love natural language processing",
        "I love machine learning",
        "Natural language processing is fascinating",
        "Machine learning and natural language processing are closely related",
        "I enjoy learning new things in machine learning and NLP"
    ]

    n = 3  # 使用三元模型（trigram）

    # 数据预处理
    processed_corpus = preprocess_corpus(corpus, n)
    print("预处理后的语料库:")
    for sentence in processed_corpus:
        print(sentence)

    # 构建 n-gram 统计
    ngram_counts, context_counts = build_ngram_counts(processed_corpus, n)

    print("\nTrigram 频率:")
    for trigram, count in ngram_counts.items():
        print(f"{trigram}: {count}")

    print("\nBigram 上下文频率:")
    for context, count in context_counts.items():
        print(f"{context}: {count}")

    # 构建词汇表
    vocabulary = set()
    for sentence in processed_corpus:
        for word in sentence:
            vocabulary.add(word)
    print(f"\n词汇表大小 (V): {len(vocabulary)}")
    print(f"词汇表: {vocabulary}")

    # 计算条件概率并应用 Laplace 平滑
    ngram_probabilities = calculate_ngram_probabilities_laplace(ngram_counts, context_counts, n, vocabulary)

    print("\nTrigram 条件概率（应用 Laplace 平滑）:")
    for context, probs in ngram_probabilities.items():
        for word, prob in probs.items():
            print(f"P({word} | {' '.join(context)}) = {prob:.4f}")

    # 生成文本
    print("\n生成的句子:")
    for _ in range(5):
        print(generate_sentence(ngram_probabilities, n))


if __name__ == "__main__":
    main()
