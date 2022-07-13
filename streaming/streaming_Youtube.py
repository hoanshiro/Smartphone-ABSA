#!/usr/bin/env python
# coding: utf-8

# In[6]:


# !pip install pytube
# !pip install youtube-comment-downloader
import socket
import json
from pytube import YouTube
from pytube import Search
import subprocess
import pandas as pd
# In[7]:
try:
    cmd = "rm -r data"
    returned_value = subprocess.call(cmd, shell=True) 
    #get_ipython().system('rm -r data')


# In[8]:
except:
    cmd = "mkdir data"
    returned_value = subprocess.call(cmd, shell=True) 
#get_ipython().system('mkdir data')


# In[11]:



# In[ ]:





# In[13]:


keywords = ['điện thoại bphone']

def sent_data(s: socket.socket, keywords):
    result_ids = []
    for keyword in keywords:
        _search = Search(keyword)
        for result in _search.results:
            result_ids.append('https://www.youtube.com/watch?v=' + result.video_id)
            file_name = f'data/{result.video_id}.json'
            cmd = f"youtube-comment-downloader --youtubeid  {result.video_id} --output {file_name} --language vi"
            returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
            df = pd.read_json(file_name,  lines=True)
            df['text'].apply(lambda comment: s.send(comment.encode("utf-8")))


# In[16]:


if __name__ == "__main__":
    s = socket.socket()
    host = "127.0.0.1"
    port = 5555
    s.bind((host, port))
    
    print("Listening on port: %s" %str(port))
    
    s.listen(5) # wait for client connection
    c, addr = s.accept() #Establish connection with client
    
    print("Received request from: " + str(addr))
    sent_data(c, keywords=['điện thoại bphone'])


# In[17]:


# import pandas as pd
# import glob

# path = os.getcwd() # use your path
# all_files = glob.glob(os.path.join(path , "*.json"))

# li = []

# for filename in all_files:
#     df = pd.read_json(filename,  lines=True)
#     li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)


# In[18]:


# frame.head(5)

