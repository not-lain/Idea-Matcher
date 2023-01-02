import streamlit as st
import numpy as np
import pickle
import pandas as pd 
import logging
from gensim.models import TfidfModel
from nltk import corpus
from nltk import download
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
import pickle 
from gensim.corpora import Dictionary
from tqdm import tqdm


# ['Title', 'Type', 'Sector','Key words', 'Problem/Opportunity', 
#           'Description', 'Added Value','Impact']

columns = ["Key words","Title","Description"]


path = "./dependencies/"


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Import and download stopwords from NLTK.




def file_path(column, variable_name, path=path):
    return path+"".join(column.split())+"_"+variable_name+".pickle"


df = pd.read_excel('./Example of the original database (1).xlsx')
df = df.iloc[:, :11]




# Importing the variables only ONCE

if 'model' not in st.session_state:
    with st.spinner("# Loading api .."):
        st.session_state.model = api.load('word2vec-google-news-300')
        
if 'stopwords' not in st.session_state:
    with st.spinner('loadig stopwords ...'):
        download('stopwords')  # Download stopwords list.
        stop_words = corpus.stopwords.words('english')
        portuguese = corpus.stopwords.words('portuguese')
        stop_words.extend(portuguese)
        st.session_state.stopwords = stop_words
    
stop_words = st.session_state.stopwords
model = st.session_state.model




def preprocess(sentence,stop_words=stop_words):
    sentence = str(sentence)
    return [w for w in sentence.lower().split() if w not in stop_words]

def save_variable(column,variable_name,variable): 
    p = file_path(column,variable_name,path = path)
    with open(p,"wb") as f :
        pickle.dump(variable,f)
        
        
def createtheAI(column,model=model):
    sentences = df[column].values
    processed_sentences = []
    for sentence in sentences : 
        processed_sentences.append(preprocess(sentence))
    # Define dictionary and create bag of words
    dictionary = Dictionary(processed_sentences)
    bow = [dictionary.doc2bow(sentence) for sentence in processed_sentences]
    # Creating the Term Frequency - Inverse Document Frequency
    tfidf = TfidfModel(bow)
    # Term Indexing and Similarity Matrix
    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    # Saving the envirenmental variables
    save_variable(column, "termsim_matrix", termsim_matrix)
    save_variable(column,"tfidf",tfidf)
    save_variable(column,"dictionary",dictionary)

# loads variables 
def load_variables(column):
    l = ["termsim_matrix","tfidf","dictionary"]
    paths = []
    for variable_name in l :
        paths.append(file_path(column, variable_name))
    with open(paths[0],"rb") as f :
        termsim_matrix = pickle.load(f)
    with open(paths[1],"rb") as f :
        tfidf = pickle.load(f)
    with open(paths[2],"rb") as f :
        dictionary = pickle.load(f)
        
    return termsim_matrix, tfidf, dictionary

# preprocessing the input
def prepare_input(s, dictionary, tfidf):
    precessed_input = preprocess(s)
    bow_input = dictionary.doc2bow(precessed_input)
    tfidf_input = tfidf[bow_input]
    return tfidf_input


def calculate_similarity(s1, s2, column):
    termsim_matrix, tfidf, dictionary = load_variables("Title")
    in1 = prepare_input(s1, dictionary, tfidf)
    in2 = prepare_input(s2, dictionary, tfidf)
    similarity = termsim_matrix.inner_product(
        in1, in2, normalized=(True, True))
    return similarity


def similarity_between_two_rows(idx1, idx2, available_columns=columns):
    sim = 0
    for column in available_columns:
        s1 = df.loc[idx1, column]
        s2 = df.loc[idx2, column]
        sim += calculate_similarity(s1, s2, column)
    sim = sim/len(available_columns)
    return sim
def similarity_with_new_row(new_row,Thresh=0,sim=False, columns=columns,df=df):
    n = len(df)
    # Contains a list with sim coef of the new row in comparison with all the rows
    similarities = [] 
    for i in tqdm(range(n)):
        s = 0
        for column in columns:
            s1 = df.loc[i, column]
            s2 = new_row[column]
            s+= calculate_similarity(s1, s2, column)
        similarities.append(s/len(columns))
    if sim == True :
        m = max(similarities)
        indx = similarities.index(m)
        return(m,indx)
    else :     
        similars = []
        for i in tqdm(range(n)):
            if similarities[i] > Thresh : 
                l = list(df.loc[i,:].values)
                similars.append(l)
        return similars

       


def update_sim_dic(columns):
    n = len(df)
    d = {}
    bar1 = st.progress(0)
    bar2 = st.progress(0)
    for i in tqdm(range(0,n-1)):
        bar2.progress(i/(n-1))
        for j in range(i+1,n):
            s = similarity_between_two_rows(i,j,columns)
            d[f"{i},{j}"] = s
            bar1.progress((j-i)/(n-1))
    with open(path+'d.pickle','wb') as f : 
        pickle.dump(d,f)


# loads variables
def load_variables(column):
    l = ["termsim_matrix", "tfidf", "dictionary"]
    paths = []
    for variable_name in l:
        paths.append(file_path(column, variable_name))
    with open(paths[0], "rb") as f:
        termsim_matrix = pickle.load(f)
    with open(paths[1], "rb") as f:
        tfidf = pickle.load(f)
    with open(paths[2], "rb") as f:
        dictionary = pickle.load(f)

    return termsim_matrix, tfidf, dictionary

# preprocessing the input


def prepare_input(s, dictionary, tfidf):
    precessed_input = preprocess(s)
    bow_input = dictionary.doc2bow(precessed_input)
    tfidf_input = tfidf[bow_input]
    return tfidf_input


def calculate_similarity(s1, s2, column):
    termsim_matrix, tfidf, dictionary = load_variables(column)
    in1 = prepare_input(s1, dictionary, tfidf)
    in2 = prepare_input(s2, dictionary, tfidf)
    similarity = termsim_matrix.inner_product(
        in1, in2, normalized=(True, True))
    return similarity


def similarity_between_two_rows(idx1, idx2, available_columns=columns):
    sim = 0
    for column in available_columns:
        s1 = df.loc[idx1, column]
        s2 = df.loc[idx2, column]
        sim += calculate_similarity(s1, s2, column)
    sim = sim/len(available_columns)
    return sim

# loads the dictionary that contains the similarity coefficients
def load_sim_dictionary():
    with open(path+'d.pickle','rb') as f : 
        d = pickle.load(f)
    return d

# Outputs keys for the similar ideas
def similar_ideas(thresh=0):
    d = load_sim_dictionary()
    v = list(d.values())
    k = list(d.keys())
    l = []
    for i in range(len(v)): 
        if v[i] > thresh : 
            l.append(k[i])
    return l 


# Outputs the names of the users that have similar ideas
def users_with_sim_ideas(thresh, rows=False):
    l = similar_ideas(thresh)
    names = []
    for i in l:
        tmp = []
        indexes = i.split(',')
        if rows == True:
            emp1 = list(df.loc[int(indexes[0]), :].values)
            emp2 = list(df.loc[int(indexes[1]), :].values)
            tmp = [emp1, emp2]
            names.append(tmp)
        else:
            emp1 = df.loc[int(indexes[0]), 'Employee Name']
            emp2 = df.loc[int(indexes[1]), 'Employee Name']
            tmp = [emp1, emp2]
            names.append(tmp)

    return names

def TrainTheAi(columns):
    my_bar = st.progress(0)
    percent_complete = 0
    for i in columns:
        percent_complete += 1/(len(columns)+1)
        my_bar.progress(percent_complete)
        # Calling the function to create the AI here
        # All variables are saved in the dependencies folder
        createtheAI(i)
    my_bar.progress(1.0)
       

### streamlit 

def main():

    st.write('# Idea Matcher')

    #columns
    st.set_option('deprecation.showfileUploaderEncoding', False)
    columns = st.multiselect(
        'What columns are you going to use ?',
        ['Title', 'Type', 'Sector','Key words', 'Problem/Opportunity', 
            'Description', 'Added Value','Impact'],
        ["Key words", "Title", "Description"])

    st.write('You selected:', columns)

    #button to train
    if st.button("update the AI"):
        st.text('updating the AI ...')
        TrainTheAi(columns)

    if st.button("update the similarity dictionary"):
        st.text('updating the similarity dictionary ...')
        update_sim_dic(columns)
        d = load_sim_dictionary()
        k = list(d.keys())
        v = list(d.values())
        m = max(v)
        idx = v.index(m)
        max_combination = k[idx]
        st.text(f"""
        maximum similarity coeff is {m:.4} 
        which can be found when comaparing the lines { max_combination } of the dataset""")

    thresh = st.number_input('Insert min similarity coeff')
    st.write('The current number is ', thresh)
    
    rows = st.checkbox('Show rows ?')
    
    if st.button(f'show users with sim ideas with similarity coeff > {str(thresh)}'):
        names = users_with_sim_ideas(thresh,rows)
        st.write(names)
    
    
    
    st.write("# new user similarity")
    df_cols = df.columns
    new_row = {}
    for i in df_cols : 
        new_row[i] =st.text_input(i,str(df.loc[0,i]))
    
    
    if st.button(f'calculate maximum similarity'):
        m,indx = similarity_with_new_row(new_row, thresh,sim = True)
        st.write(f'''maximum similarity to this data is {m} 
                 which can be found at line {indx} ''')
        
        
    if st.button(f'show similar ideas to this new data with coeff = {thresh} '):
        s = similarity_with_new_row(new_row, thresh)
        st.write(s)
            
    
    


if __name__ == "__main__":
    main()


