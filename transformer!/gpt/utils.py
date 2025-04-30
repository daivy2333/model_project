chars = []
stoi = {}
itos = {}

def build_vocab(text):
    # 声明全局变量 chars, stoi, itos
    global chars, stoi, itos
    # 将输入文本中的所有字符去重并排序，存储在 chars 列表中
    chars = sorted(list(set(text)))
    # 在字符集中添加一个 'UNK' 字符（代表未知字符）
    chars.append('UNK')
    # 创建一个字典 stoi，将每个字符映射到其在 chars 列表中的索引
    stoi = {ch: i for i, ch in enumerate(chars)}
    # 创建一个字典 itos，将每个索引映射回其对应的字符
    itos = {i: ch for ch, i in stoi.items()}
    # 返回字符集的大小，即 chars 列表的长度
    return len(chars)

def encode(s):
    # 处理不可识别的字符，如果字符不在 vocab 中，使用 'UNK' 对应的索引
    return [stoi.get(c, stoi['UNK']) for c in s]  # 默认为 'UNK' 的索引

def decode(l):
    # 将索引列表 l 解码为对应的字符
    return ''.join([itos[i] for i in l])
