
import streamlit as st
import streamlit.components.v1 as html
import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import time 

# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# 페이지의 기본 설정 구성
st.set_page_config(
 layout="wide",
 page_title='오늘 이거 먹어')


# # 방법 1 progress bar 
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.05)
#   # 0.05 초 마다 1씩증가

#######################################################################################################
#### 모델 불러오기 ####

device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=17,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):

        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def softmax(vals, idx):
    valscpu = vals.detach().cpu().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100


def testModel(model, seq):
    cate = ["곱창","국수","돈카츠", "디저트","라멘","버거", "베이커리", "분식", "스시", "아시아음식", "양식", "전골", "중식", "치킨", "타코", "한식", "해산물"]
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(tok, max_len, pad=True, pair=False)
    tokenized = transform(tmp)

    modelload.eval()
    result = modelload(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device)) 
    idx = result.argmax().cpu().item() 
    result2 = F.softmax(result, dim=1).sort()
    return cate[idx], softmax(result,idx) 

# 모델을 불러와서 한번만 로드하고 캐시에 저장하기
@st.cache_resource
def cache_model(path, modelname):
    modelload = torch.load("/content/drive/MyDrive/final project/model/model6.pt") # gpu사용시
    modelload.eval()
    return modelload

modelload = cache_model('/content/drive/MyDrive/final project/model/','model6.pt')

# 카카오 api
@st.cache_resource
def elec_location(region,page_num):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    params = {'query': region,'page': page_num}
    headers = {"Authorization": "KakaoAK 6dd31dbd3f7b90aed3f5591fdde29527"}

    places = requests.get(url, params=params, headers=headers).json()['documents']
    total = requests.get(url, params=params, headers=headers).json()['meta']['total_count']

    return places

def elec_info(places):
    X = []
    Y = []
    stores = []
    road_address = []
    place_url = []
    ID = []
    for place in places:
        X.append(float(place['x']))
        Y.append(float(place['y']))
        stores.append(place['place_name'])
        road_address.append(place['road_address_name'])
        place_url.append(place['place_url'])
        ID.append(place['id'])

    ar = np.array([ID,stores, X, Y, road_address,place_url]).T
    df = pd.DataFrame(ar, columns = ['ID','stores', 'X', 'Y','road_address','place_url'])
    return df

def keywords(location_name):
    df = None
    page_num = int(1)
    for loca in location_name:
        for page in range(1,page_num+1):
            local_name = elec_location(loca, page)
            local_elec_info = elec_info(local_name)

            if df is None:
                df = local_elec_info
            elif local_elec_info is None:
                continue
            else:
                df = pd.concat([df, local_elec_info],join='outer', ignore_index = True)
    return df

def make_map(dfs, m):
    
    minimap = MiniMap() 
    m.add_child(minimap)

    for i in range(len(dfs)):
        folium.Marker([dfs['Y'][i],dfs['X'][i]],
                      tooltip=dfs['stores'][i],
                      popup = '<iframe width="1000" height="400" src="' + df['place_url'][i] + '"title="YouTube video player" frameborder="0" allow="accelerometer; autoplay;  clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                      ).add_to(m)
    return m

#######################################################################################################

st.sidebar.header('Side Menu')
tab1, tab2 = st.tabs(['search', 'map'])

# user 입력값 저장
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

if 'user_location_input' not in st.session_state:
    st.session_state['user_location_input'] = ''

with st.sidebar:
        when = st.selectbox('식사 시간은 언제인가요?', ['아침', '점심', '저녁'])
        #location = st.text_input('지금 계신 지역은 어디인가요?', value = '', placeholder = '근처 지하철 역을 입력해주세요', key='user_location_input')

with tab1:
    st.subheader('오늘도 무엇을 먹을지 고민하고 계신가요?')

    value = st.text_input('지금 생각나는 키워드를 입력하고 Enter를 눌러주세요!', placeholder = 'Ex) 육즙이 팡팡 터지는 고소한 음식이 먹고싶어.', key='user_input')
    test, val = testModel(model ,st.session_state.user_input)

    if value:
       st.header(test, '이 음식은 어떠신가요?')

       st.write(test, '추천드립니다.', '신뢰도는', round(val, 2), '% 입니다.')

with tab2:
    st.header('여기는 지도가 나올겁니다.')
    location = st.text_input('지금 계신 지역은 어디인가요?', value = '', placeholder = '근처 지하철 역을 입력해주세요', key='user_location_input')
    user_location = st.session_state.user_location_input

    if location:
        kakao_location = [user_location + ' ' + test]
        df = keywords(kakao_location)

        lat = 0
        lon = 0

        for i in df['Y']:
            lat += float(i)
        for j in df['X']:
            lon += float(j)

        lat = lat/len(df['Y'])
        lon = lon/len(df['X'])

        m = folium.Map(kakao_location=[lat, lon],   # 기준좌표: current_location
                      zoom_start=5)
        make_map = make_map(df, m)

        st_folium(make_map, width = 1000, height = 500, zoom=16, center = [lat, lon])
