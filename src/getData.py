#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '..')


# ## 1. Load Package 

# In[2]:


from src.conf import *
from src.utils import *
from tqdm import tqdm


# In[3]:


display_all_dataframe()


# In[ ]:


get_ipython().system('git clone https://github.com/vncorenlp/VnCoreNLP.git')
get_ipython().system('wget -q https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip -P data/')
get_ipython().system('unzip data/UIT-ViSFD.zip -d data')


# ## 2. Transform Data

# In[9]:


TextProcessing = TextProcessing()


# In[11]:


ls_df = []
new_labels = GetNewLabels()
for data_name in tqdm(['Train', 'Dev', 'Test']):
    df = pd.read_csv(f'data/{data_name}.csv')
    df = df[['comment', 'label']]

    for label in new_labels:
        df[label] = 0

    df = df.apply(lambda row: labels2onehot(row, raw_label='label'), axis=1) 

    df['tokenize'] = df['comment'].apply(lambda text: TextProcessing.clean_text(text, remove_stopwords=False))
    # df['stop_words_remove'] = df['comment'].apply(lambda text: clean_text(text, remove_stopwords=True))

    df = df[['tokenize', *new_labels]]
    ls_df.append(df)

df_train, df_valid, df_test = ls_df


# ## 3. Check

# In[13]:


df_train.tail(5)


# In[14]:


df_train.to_csv('data/processed_train.csv', index=False)
df_valid.to_csv('data/processed_valid.csv', index=False)
df_test.to_csv('data/processed_test.csv', index=False)


# In[ ]:




