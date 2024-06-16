from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters,MessageHandler
from openai import OpenAI as OpenAI1

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import os
import json
import glob
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import paramiko
import json, os
import shutil
import time
from time import sleep
import requests
from xml.etree import ElementTree
import numpy as np
from datetime import datetime, timedelta

# # Start
# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     '''响应start命令'''
#     text = '你好~我是一个bot'
#     print(update.message.text.replace("/start",""))
#     await context.bot.send_message(chat_id=update.effective_chat.id,text=text)

# help
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Responds to the /help command'''
    help_info = {
        "🤖 Chat with me!": "`/chat <your question>`",
        "📰 Get article recommendations!": [
            {"/rag <keywords>": "Content-based article retrieval"},
            {"/new <keywords>": "Explore the latest articles"}
        ],
    }

    help_text = "**Available Commands**\n" 
    for key, value in help_info.items():
        help_text += f"*{key}*\n" 
        if isinstance(value, list):
            for item in value:
                for k, v in item.items():
                    help_text += f"  •  `{k}`: {v}\n"  
        else:
            help_text += f"  •  {value}\n" 

    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text, parse_mode='Markdown') 

# Chat——DeepSeek
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''响应用户消息'''
    text = update.message.text.replace("/chat","")
    response = client_deepseek.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": text},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)
    res=response.choices[0].message.content
    username = update.message.from_user.username
    final_response = f"{res} @{username}"
    await context.bot.send_message(chat_id=update.effective_chat.id,text=final_response)

# async def chat2(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     '''响应用户消息'''
#     text = update.message.text.replace("/chat2","")
#     response = client_gpt.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": text},
#   ],
#     max_tokens=1024,
#     temperature=0.7,
#     stream=False
# )
#     res=response.choices[0].message.content
#     await context.bot.send_message(chat_id=update.effective_chat.id,text=res)

# rec1
async def rag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''响应用户消息'''
    query = update.message.text.replace("/rag","")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=False
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        chain_type="stuff",
        retriever=database.as_retriever(),
        memory=memory
        )
    print(conversation_chain)

    query = optimize_query(query)
    response = conversation_chain.run(query)
    username = update.message.from_user.username
    response1 = f"{response} @{username}"
    print("rag is done")

    await context.bot.send_message(chat_id=update.effective_chat.id,text=response1)

def optimize_query(user_input: str) -> str:
    optimized_query = user_input.strip()
    return optimized_query

def read_doc(folder_path, doc, text_splitter):
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for _, article in enumerate(data):
                chunks = text_splitter.split_text(str(article))
                for chunk in chunks:
                    metadata = {
                    "url" : str(article['link']) ,
                    "Title" : str(article['title']),
                    "Authors" : str(article['authors']),
                    "Subjects" : str(article['subjects']),
                    "SubmitDate": str(article['published']),
                    }
                    doc.append(Document(page_content=chunk, metadata=metadata))
# init_database
def init_database():
    json_file_path = "source" 
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)   
    doc = []
    read_doc(json_file_path, doc, text_splitter)
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # You can choose your preferred embedding
    print("---initial database---")
    db = Chroma.from_documents(doc, embedding_model)
    return db

# rec2
async def new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''Responds to user messages with a list of recommended articles.'''
    text = update.message.text.replace("/new ", "")  # Extract keywords
    res_list = get_recommand(text)

    response = f"Here are the latest articles recommended for **'{text}'**:\n\n"
    for i, article in enumerate(res_list):
        response += f"*{i+1}. {article['Title']}*\n"  
        response += f"<a href='{article['URL']}'>🔗 Article Link</a>\n\n" 

    username = update.message.from_user.username
    response1 = f"{response} @{username}" 
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response1, parse_mode='HTML')

def move_file_with_timestamp(src_path, dest_dir):
    file_name, file_extension = os.path.splitext(os.path.basename(src_path))
    timestamp = time.strftime("%Y%m%d%H%M%S")
    new_file_name = f"{file_name}_{timestamp}{file_extension}"
    dest_path = os.path.join(dest_dir, new_file_name)
    if os.path.exists(dest_path):
        os.remove(dest_path)
        print(f"Removed existing file at {dest_path}")
    
    try:
        shutil.move(src_path, dest_path)
        print(f"File {src_path} has been moved to {dest_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_papers_for_day(year, month, day):
    start_index = 0
    max_results = 100
    all_entries = []

    while True:
        url = f"http://export.arxiv.org/api/query?search_query=cat:cs.*+AND+submittedDate:[{year}{month:02d}{day:02d}0000+TO+{year}{month:02d}{day:02d}2359]&start={start_index}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        try:
            response = requests.get(url)
            root = ElementTree.fromstring(response.content)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            break

        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        if not entries:
            break

        for entry in entries:
            paper_info = {
                'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                'link': entry.find('{http://www.w3.org/2005/Atom}link').attrib['href'],
                'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                'subjects': [category.attrib['term'] for category in entry.findall('{http://www.w3.org/2005/Atom}category')],
                'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                'published': entry.find('{http://www.w3.org/2005/Atom}published').text
            }
            all_entries.append(paper_info)

        start_index += len(entries)
        sleeptime = np.random.uniform(0.8, 1.2)
        time.sleep(sleeptime)
    return all_entries

def save_results(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {filepath}")

def crawl(day, recrawl=True):
    directory = 'source'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # filepath = os.path.join(directory, 'arxiv_papers_daily.json')
    start_date = datetime.now() - timedelta(days=day)
    end_date = datetime.now()

    current_date = start_date
    while current_date <= end_date:
        results = []
        if not current_date == end_date:
            if not recrawl:
                current_date += timedelta(days=1)
                continue
        year = current_date.year
        month = current_date.month
        day = current_date.day
        print(f"Fetching papers for {year}-{month:02d}-{day:02d}")
        results.extend(fetch_papers_for_day(year, month, day))
        filepath = os.path.join(directory, f'arxiv_papers_{year}-{month:02d}-{day:02d}.json')
        current_date += timedelta(days=1)
        save_results(results, filepath)

def get_targets(day):
    start_date = datetime.now() - timedelta(days=day)
    end_date = datetime.now()
    current_date = start_date
    result_paths = []
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        filepath = f"arxiv_papers_{year}-{month:02d}-{day:02d}.json"
        # 检查是否有这个文件
        if os.path.exists("source/"+filepath):
            result_paths.append(filepath)
        else:
            print(f"File {filepath} does not exist")
        current_date += timedelta(days=1)
    print("why?")
    return result_paths

def get_recommand(text):
    # hostname = 'connect.westb.seetacloud.com'
    hostname = 'your host name'
    port = 00000
    username = 'your username'
    password = 'your password'  # 或者使用私钥文件
    

    # 初始化 SSH 客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # text_data = "{'name': 'xiaomangguo', 'age': 20}"
    # json_data = json.dumps(text_data)
    # 把text写入query.txt文件
    with open('query/query.txt', 'w') as f:
        f.write(text)

    # try:
    #     while True:
    print("Waiting for query")
    day = 4
    recrawl = False
    # 如果生成了query.txt文件，就把它传到服务器上
    if os.path.isfile("query/query.txt"):
        # 连接到远程服务器
        ssh.connect(hostname, port, username, password)
        # 执行远程命令
        # stdin, stdout, stderr = ssh.exec_command('ls -l')
        # print(stdout.read().decode())
        sftp = ssh.open_sftp()
        
        # 把target papers传到服务器上,即要从哪些文章中挑选出来
        # localpath = 'source/CS_500.json'
        print(f"Ask for {day}-days latest papers, recrawl = {recrawl}")
        crawl(day = day, recrawl=recrawl)
        target_papers_path = get_targets(day = day)
        for target_paper in target_papers_path:
            print(target_paper)
            remotepath = f'/root/autodl-fs/DSP/data/source/{target_paper}'
            sftp.put("source/"+target_paper, remotepath)
        # sftp.put(localpath, remotepath)
        print("Target papers already set!")

        # 把query.json传到服务器上。上传后移动到history_query文件夹
        localpath = 'query/query.txt'
        remotepath = '/root/autodl-fs/DSP/data/query.txt'
        sftp.put(localpath, remotepath)
        move_file_with_timestamp(localpath, 'history_query')
        print("Query already received!")

        # 等待服务器的结果，即等待服务器生成result.json文件
        # 读取服务器的文件，如果有就读取，没有就等待
        # 读完之后删除。再接着等待query
        # 检测远程文件夹是否有json文件，如果有，就把它移到本地
        print("Waiting for result")
        remote_forder = '/root/autodl-fs/DSP/data/output'
        localpath = 'result/result.json'
        files = sftp.listdir(remote_forder)
        print(files)
        while 'priority_queue.json' not in files:
            files = sftp.listdir(remote_forder)
            print(files)
            sleep(1)
            # print("Waiting for result")
        sleep(1)
        # remotepath = os.path.join(remote_forder, 'priority_queue.json')
        remotepath = remote_forder+ "/priority_queue.json"
        sftp.get(remotepath, localpath)  # 下载远程文件到本地
        sftp.remove(remotepath)  # 删除远程文件
        print("Result already received!")
        # 用dataframe读取json文件，然后删掉json文件
        with open(localpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # os.remove(localpath)  # 删除本地文件
        ssh.close()
    return data
    # return data

# Unknown
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles unknown commands, prompting the user to use /help.
    """
    await context.bot.send_message(chat_id=update.effective_chat.id, 
                                   text="Sorry, I don't understand your command. Please use /help to see available commands.")
    
#Handlers
# start_handler = CommandHandler('start', start) 
unknown_handler = MessageHandler(filters.COMMAND, unknown)
chat_handler = CommandHandler('chat', chat)
# chat_handler2 = CommandHandler('chat2', chat2)
help_handler = CommandHandler('help', help)
rec1_handler = CommandHandler('rag', rag)
rec2_handler = CommandHandler('new', new)

#Bot
TOKEN='Telegram Bot API key HERE'
application = ApplicationBuilder().token(TOKEN).build()

# 注册 handler
# application.add_handler(start_handler)
application.add_handler(chat_handler)
# application.add_handler(chat_handler2)
application.add_handler(help_handler)
application.add_handler(rec1_handler)
application.add_handler(rec2_handler)
application.add_handler(unknown_handler)

# DeepSeek API
client_deepseek = OpenAI1(api_key="Deepseek API key HERE", base_url="https://api.deepseek.com")
# OpenAI API
client_gpt = OpenAI1(api_key="OPENAI API key HERE")
os.environ["OPENAI_API_KEY"] = "OPENAI API key HERE"
llm_model = OpenAI(model="gpt-3.5-turbo-instruct")
database = init_database()
# run!
print("Bot is running...")
application.run_polling()