import torch
import torch.nn as nn
import pandas as pd
import re
import torch.optim as optim
import torch.nn.functional as F
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 하이퍼파라미터 설정
PAD_token = 0  # 패딩 토큰
SOS_token = 1  # 문장 시작 토큰
EOS_token = 2  # 문장 끝 토큰
UNK_token = 3  # 알 수 없는 토큰
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
data = pd.read_csv('./end.csv')
selected_columns = ['name', '업태구분명', 'naver_review1', 'naver_review2', 'naver_review3', 
                    'naver_review4', 'naver_review5', 'k_review1', 'k_review2', 
                    'k_review3', 'k_review4', 'k_review5']
data = data[selected_columns]

# 데이터 전처리
data = data.fillna('')

# 텍스트 정제 함수
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'\W', ' ', text)   # 특수 문자 제거
    text = text.lower().strip()
    return text

# 정제 함수 적용
for col in selected_columns[2:]:
    data[col] = data[col].apply(clean_text)

data['text'] = data[['업태구분명'] + selected_columns[2:]].apply(lambda x: ' '.join(x), axis=1)

# 레이블 인코딩
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['name'])

# 데이터 분할
train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)

# 토크나이저 설정
tokenizer = get_tokenizer('basic_english')

# 토큰 생성 함수
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 어휘 사전 생성
VOCAB = build_vocab_from_iterator(yield_tokens(train_texts), specials=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
VOCAB.set_default_index(VOCAB['<UNK>'])
VOCAB_SIZE = len(VOCAB)

# PyTorch 데이터셋 클래스
class QADataset(Dataset):
    def __init__(self, pairs, vocab, tokenizer, input_seq_len, target_seq_len):
        self.pairs = pairs
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        question_tokens = [self.vocab[token] for token in tokenizer(question)]
        answer_tokens = [self.vocab[token] for token in tokenizer(answer)]
        enc_src = self.pad_sequence(question_tokens + [self.vocab['<EOS>']], self.input_seq_len)
        dec_src = self.pad_sequence([self.vocab['<SOS>']] + answer_tokens, self.target_seq_len)
        trg = self.pad_sequence([self.vocab['<SOS>']] + answer_tokens + [self.vocab['<EOS>']], self.target_seq_len)
        return enc_src, dec_src, trg

    def pad_sequence(self, seq, max_len):
        return F.pad(torch.tensor(seq), (0, max_len - len(seq)), value=self.vocab['<PAD>'])

# 단어 및 위치 임베딩 레이어
class WordPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, emb_size, device):
        super(WordPositionEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_size, device=device)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pos_emb = torch.zeros(max_seq_len, emb_size)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_embedding', pos_emb)

    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        pos_embeddings = self.position_embedding[:x.size(1), :]
        embeddings = word_embeddings + pos_embeddings
        return embeddings

# 멀티-헤드 어텐션 메커니즘 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.head_dim = emb_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size)

    def forward(self, values, keys, queries, mask=None):
        batch_size = queries.shape[0]
        values = values.reshape(batch_size, self.heads, -1, self.head_dim)
        keys = keys.reshape(batch_size, self.heads, -1, self.head_dim)
        queries = queries.reshape(batch_size, self.heads, -1, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(batch_size, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

# 트랜스포머 블록
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 인코더
class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, drop_out, device):
        super(Encoder, self).__init__()
        self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device)
        self.layers = nn.ModuleList([TransformerBlock(emb_size, heads, forward_expansion, drop_out) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, mask):
        out = self.dropout(self.embedding(x))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

# 디코더 블록
class DecoderBlock(nn.Module):
    def __init__(self, emb_size, heads, forward_expansion, drop_out):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, heads)
        self.norm = nn.LayerNorm(emb_size)
        self.transformer_block = TransformerBlock(emb_size, heads, forward_expansion, drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
        
# 다중 디코더 블록으로 구성된 디코더
class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, n_layers, heads, forward_expansion, 
                 drop_out, device):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = WordPositionEmbedding(vocab_size, seq_len, emb_size, device)
        self.layers = nn.ModuleList([
            DecoderBlock(emb_size, heads, forward_expansion, drop_out) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, X, enc_out, src_mask, trg_mask):
        out = self.dropout(self.embedding(X))

        # 각 디코더 블록을 처리합니다.
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        # 어휘 사전으로 매핑하는 출력 레이어
        out = self.fc_out(out)
        return out
# 인코더와 디코더를 결합한 전체 트랜스포머 모델
class TransformerScratch(nn.Module):
    def __init__(self, inp_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, emb_size, 
                 n_layers=1, heads=1, forward_expansion=1, drop_out=0.2, max_seq_len=100, 
                 device=torch.device('cuda')):
        super(TransformerScratch, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(inp_vocab_size, max_seq_len, emb_size, n_layers, heads, 
                               forward_expansion, drop_out, device).to(device)
        self.decoder = Decoder(trg_vocab_size, max_seq_len, emb_size, n_layers, heads, 
                               forward_expansion, drop_out, device).to(device)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        batch_size, trg_seq_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len))).expand(
            batch_size, 1, trg_seq_len, trg_seq_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out
# 모델의 단일 학습 단계
def step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device):
    enc_src = enc_src.to(device)
    dec_src = dec_src.to(device)
    trg = trg.to(device)

    # 모델을 통해 순전파 계산을 수행합니다.
    logits = model(enc_src, dec_src)

    # SOS 토큰을 대상에서 제외하고 마지막 logit을 제거하여 대상과 일치시킵니다.
    logits = logits[:, :-1, :].contiguous()
    trg = trg[:, 1:].contiguous()

    loss = loss_fn(logits.view(-1, logits.shape[-1]), trg.view(-1))

    # 정확도 계산
    non_pad_elements = (trg != VOCAB['<PAD>']).nonzero(as_tuple=True)
    correct_predictions = (logits.argmax(dim=2) == trg).sum().item()
    accuracy = correct_predictions / len(non_pad_elements[0])

    return loss, accuracy
# 하나의 에포크 동안의 학습 루프
def train_step(model, iterator, optimizer, loss_fn, clip, VOCAB, device):
    model.train()  # 모델을 학습 모드로 설정합니다.
    epoch_loss = 0
    epoch_acc = 0

    for i, batch in enumerate(iterator):
        enc_src, dec_src, trg = batch

        # 기울기를 초기화합니다.
        optimizer.zero_grad()

        loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

        # 역방향 계산을 수행합니다.
        loss.backward()

        # 기울기 폭발을 방지하기 위해 기울기를 클리핑합니다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터를 업데이트합니다.
        optimizer.step()

        # 손실과 정확도를 누적합니다.
        epoch_loss += loss.item()
        epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
# 모델 학습 루프
def train(model, train_loader, optimizer, loss_fn, clip, epochs, VOCAB, device, val_loader=None):
    """
    모델을 지정된 에포크 수 동안 학습하고 선택적으로 평가합니다.

    Args:
        model (nn.Module): 학습할 모델.
        train_loader (DataLoader): 학습 데이터에 대한 DataLoader.
        optimizer (Optimizer): 모델 가중치를 업데이트하는 옵티마이저.
        loss_fn (function): 오류를 계산하는 손실 함수.
        clip (float): 기울기의 최대 허용 값 (기울기 폭발 방지).
        epochs (int): 모델을 학습할 총 에포크 수.
        VOCAB (dict): 어휘 사전 정보가 포함된 사전.
        device (torch.device): 모델을 학습할 디바이스 (CPU/GPU).
        val_loader (DataLoader, optional): 검증 데이터에 대한 DataLoader. None이면 검증을 생략합니다.

    Returns:
        nn.Module: 학습된 모델.
    """
    for epoch in range(epochs):
        # 하나의 에포크 동안 학습을 수행합니다.
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, clip, VOCAB, device)

        # 결과를 기록할 문자열을 준비합니다.
        result = f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%'

        # 검증 로더가 제공된 경우 검증을 수행합니다.
        if val_loader:
            eval_loss, eval_acc = evaluate_step(model, val_loader, loss_fn, VOCAB, device)
            result += f' || Eval Loss: {eval_loss:.3f} | Eval Acc: {eval_acc * 100:.2f}%'

        # 현재 에포크의 결과를 로그에 기록합니다.
        print(f'Epoch: {epoch + 1:02}')
        print(result)

    return model
# 평가 단계
def evaluate_step(model, iterator, loss_fn, VOCAB, device):
    model.eval()  # 모델을 평가 모드로 설정합니다.
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():  # 기울기 계산을 비활성화합니다.
        for i, batch in enumerate(iterator):
            enc_src, dec_src, trg = batch

            loss, accuracy = step(model, enc_src, dec_src, trg, loss_fn, VOCAB, device)

            # 손실과 정확도를 누적합니다.
            epoch_loss += loss.item()
            epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
# 모델 입력 준비 함수
def prepare_model_input(question, VOCAB, max_length=50, device='cuda'):
    # 입력 질문을 토큰화
    tokenized_question = text_to_tokens(question)
    enc_src = tokenized_question + [VOCAB['<EOS>']]  # EOS 토큰을 끝에 추가
    # 인코더 소스 길이가 최대 길이를 초과하지 않도록 보장
    if len(enc_src) > max_length:
        enc_src = enc_src[:max_length]  # 시퀀스를 최대 길이로 자름
    padded_enc_src = F.pad(torch.LongTensor(enc_src), (0, max_length - len(enc_src)), mode='constant',
                           value=VOCAB['<PAD>']).unsqueeze(0).to(device)  # 패딩 및 디바이스로 이동
    # 디코더 입력을 <SOS> 토큰으로 시작하는 자리 표시자를 준비
    dec_src = torch.LongTensor([VOCAB['<SOS>']]).unsqueeze(0).to(device)

    return padded_enc_src, dec_src
# 트랜스포머와 채팅하는 함수
def chat_with_transformer(model, VOCAB, max_length=50, temperature=1.0, device='cpu'):
    model.eval().to(device)

    while True:  # 채팅 세션을 위한 무한 루프 시작
        question = input("You: ")  # 사용자로부터 입력 받음
        if question.lower() == "bye":  # 사용자가 대화를 끝내고 싶어하는지 확인
            print("Bot: Goodbye!")
            break  # 사용자가 'bye'라고 하면 루프를 종료
        # 모델 입력 준비
        enc_src, dec_src = prepare_model_input(question, VOCAB=VOCAB, max_length=max_length, 
                                               device=device)

        generated_answer = []
        with torch.no_grad():
            for _ in range(max_length):
                logits = model(enc_src, dec_src)
                # 마지막 토큰만 고려하도록 조정
                predictions = F.softmax(logits[:, -1, :] / temperature, dim=1)  
                predicted_token = torch.multinomial(predictions, num_samples=1).squeeze(1)
#                 predicted_token = torch.argmax(predictions, dim=1)

                if predicted_token.item() == VOCAB['<EOS>']:
                    break  # EOS 토큰이 예측되면 토큰 생성을 중지
                # 디코더 입력을 업데이트
                dec_src = torch.cat([dec_src, predicted_token.unsqueeze(-1)], dim=1)  
                generated_answer.append(predicted_token.item())

                response = tokens_to_text(generated_answer, VOCAB)  # 토큰 ID를 텍스트로 변환
        print(f"Bot: {response}")  # 봇의 응답을 출력
# 데이터 저장 함수
def save_data(data, path="./dataset/data2.pkl"):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {path}")
# 데이터 로드 함수
def load_data(path="./dataset/data2.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from {path}")
    return data
# 데이터 로드 및 기본 전처리
df = pd.read_csv('./dataset/dialogs.txt', sep='\t', names=['Question', 'Answer'])
df['QUESTION_CLEAN'] = df['Question'].apply(clean_text)
df['ANSWER_CLEAN'] = df['Answer'].apply(clean_text)

# 모든 문장을 토큰화
tokenizer = get_tokenizer('basic_english')
special_tokens = ['<SOS', '<EOS>', '<UNK>', '<PAD>']

# 질문과 답변을 쌍으로 만듭니다.
qa_pairs = list(zip(df['QUESTION_CLEAN'], df['ANSWER_CLEAN']))

# 학습 및 검증 세트로 분리
train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.01, random_state=42)

# 편의를 위해 질문과 답변을 분리
train_questions, train_answers = zip(*train_pairs)
val_questions, val_answers = zip(*val_pairs)

# 어휘 사전 구축
train_texts = train_questions + train_answers + val_questions + val_answers
VOCAB = build_vocab_from_iterator(yield_tokens(train_texts), 
                                  specials=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
VOCAB.set_default_index(VOCAB['<UNK>'])

VOCAB_SIZE = len(VOCAB)
get_max_length = lambda train_texts: max(len(text.split()) for text in train_texts)
INPUT_SEQ_LEN = TARGET_SEQ_LEN = get_max_length(train_texts)

print('VOCAB_SIZE:', VOCAB_SIZE)
print('INPUT_SEQ_LEN:', INPUT_SEQ_LEN)
print('TARGET_SEQ_LEN:', TARGET_SEQ_LEN)
save_data([VOCAB, TARGET_SEQ_LEN], path="./dataset/vocab_simple_transformer_v7_new2.pkl")

train_dataset = QADataset(train_pairs, VOCAB, tokenizer, INPUT_SEQ_LEN, TARGET_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

transformer = TransformerScratch(
    inp_vocab_size=VOCAB_SIZE,
    trg_vocab_size=VOCAB_SIZE,
    src_pad_idx=VOCAB['<PAD>'],
    trg_pad_idx=VOCAB['<PAD>'],
    emb_size=256,
    n_layers=2,
    heads=8,
    forward_expansion=4,
    drop_out=0.05,
    max_seq_len=TARGET_SEQ_LEN,
    device=device
).to(device)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'], reduction='mean')
optimizer = optim.Adam(transformer.parameters(), lr=0.00001)

transformer = train(transformer, train_dataloader, optimizer, loss_function, clip=1, 
                    epochs=10000, VOCAB=VOCAB, device=device)

# 모델 저장
torch.save(transformer.state_dict(), './models/simple_transformer_v7_new2.pth')

chat_with_transformer(transformer, VOCAB, max_length=TARGET_SEQ_LEN, temperature=1.5, device=device)
tokenizer = get_tokenizer('basic_english')
VOCAB, TARGET_SEQ_LEN = load_data(path="./models/simple_transformer_v7_new_final/vocab_simple_transformer_v7_new.pkl")
VOCAB_SIZE = len(VOCAB)
transformer = TransformerScratch(
    inp_vocab_size=VOCAB_SIZE,
    trg_vocab_size=VOCAB_SIZE,
    src_pad_idx=VOCAB['<PAD>'],
    trg_pad_idx=VOCAB['<PAD>'],
    emb_size=256,
    n_layers=2,
    heads=8,
    forward_expansion=4,
    drop_out=0.05,
    max_seq_len=TARGET_SEQ_LEN,
    device=device
).to(device)
transformer.load_state_dict(torch.load('./models/simple_transformer_v7_new_final/simple_transformer_v7_new.pth'))
chat_with_transformer(transformer, VOCAB, max_length=TARGET_SEQ_LEN, temperature=1.5, device=device)chatbot_Trans
