import os
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("ticks",{'axes.grid' : True})
LOG_PATH = "logdir"

def get_all_eval_files(LOG_PATH):
    files_dict = {}
    for dir in os.listdir(LOG_PATH):
        path = os.path.join(LOG_PATH, dir)
        model = ""
        if (os.path.isdir(path)):
            if("roberta" in dir.split("_")):
                model = "roberta"
            if("bert" in dir.split("_")):
                model = "bert"
            dirs = os.listdir(path)
            assert "res" in dirs
            folder_path = os.path.join(path, "res")
            files = []
            for file_path in os.listdir(folder_path):
                if "acc" in file_path:
                    files.append(os.path.join(folder_path, file_path))
            files_dict[model] = files
    return files_dict

def get_similarity(s1, s2):
    X_words = word_tokenize(s1)
    Y_words = word_tokenize(s2)

    w1 =[]
    w2 =[]

    # remove repeated words from the string
    X_words_set = set(X_words)
    Y_words_set = set(Y_words)

    all_words = X_words_set.union(Y_words_set)
    for w in all_words:
        if w in X_words_set: w1.append(1)
        else: w1.append(0)
        if w in Y_words_set: w2.append(1)
        else: w2.append(0)
    c = 0
    # cosine formula
    for i in range(len(all_words)):
        c+= w1[i]*w2[i]
    similarity = c / float((sum(w1)*sum(w2))**0.5)
    return similarity

def read_file(file_path):
    f = open(file_path, "r")
    data = f.read()
    rows = data.split("\n")
    pred_list = []
    y_list = []
    for row in rows:
        if "pred" in row:
            pred_list.append(row)
        if "gold" in row:
            y_list.append(row)
    assert len(pred_list) == len(y_list)
    l = len(pred_list)
    sim_list = []
    #similarity of prediction with original
    for i in range(l):
        pred = pred_list[i].split(":")[1]
        y = y_list[i].split(":")[1]
        similarity = get_similarity(pred, y)
        sim_list.append(similarity)
    return sum(sim_list) / l

def plotBertandRoberta(df):
    ax = sns.lineplot(data=df, x='Step', y='Similarity', hue='PretrainedModel')
    ax.tick_params(axis='x', rotation=90)
    plt.show()

def compareBertandRoberta(files):
    values = []
    if files.get("bert"):
        all_files = files["bert"]
        for file_path in all_files:
            step = file_path.split(".")[2].split("_")[-1]
            step = step.replace("step", "")
            avg = read_file(file_path)
            values.append([step, "bert", avg])
    if files.get("roberta"):
        all_files = files["roberta"]
        for file_path in all_files:
            step = file_path.split(".")[2].split("_")[-1]
            step = step.replace("step", "")
            avg = read_file(file_path)
            values.append([step, "roberta", avg])
    
    df = pd.DataFrame(values, columns=['Step', 'PretrainedModel', 'Similarity'])
    df = df.sort_values('Step', key=pd.to_numeric)
    return df

def read_file_level(file_path):
    f = open(file_path, "r")
    data = f.read()
    rows = data.split("\n")
    pred_list = []
    y_list = []
    for row in rows:
        if "pred" in row:
            pred_list.append(row)
        if "gold" in row:
            y_list.append(row)
    assert len(pred_list) == len(y_list)
    l = len(pred_list)
    sim_list = {}
    #similarity of prediction with original for every difficulty level
    for i in range(l):
        pred = pred_list[i].split(":")[1]
        y = y_list[i].split(":")[1]
        type_ques = pred_list[i].split(" ")[0]
        similarity = get_similarity(pred, y)
        if sim_list.get(type_ques):
            li = sim_list[type_ques]
            li.append(similarity)
            sim_list[type_ques] = li
        else:
            sim_list[type_ques] = [similarity]
    return sum(sim_list['easy']) / len(sim_list['easy']), sum(sim_list['medium']) / len(sim_list['medium']), sum(sim_list['hard']) / len(sim_list['hard']), sum(sim_list['extra']) / len(sim_list['extra'])

def compareBertandRobertaWithDifficulty(files):
    values = []
    if files.get("bert"):
        all_files = files["bert"]
        for file_path in all_files:
            step = file_path.split(".")[2].split("_")[-1]
            step = step.replace("step", "")
            easy, medium, hard, extra = read_file_level(file_path)
            values.append([step, "bert", "easy", easy])
            values.append([step, "bert", "medium", medium])
            values.append([step, "bert", "hard", hard])
            values.append([step, "bert", "extra", extra])

    if files.get("roberta"):
        all_files = files["roberta"]
        for file_path in all_files:
            step = file_path.split(".")[2].split("_")[-1]
            step = step.replace("step", "")
            easy, medium, hard, extra = read_file_level(file_path)
            values.append([step, "roberta", "easy", easy])
            values.append([step, "roberta", "medium", medium])
            values.append([step, "roberta", "hard", hard])
            values.append([step, "roberta", "extra", extra])
    
    df_level = pd.DataFrame(values, columns=['Step', 'PretrainedModel', 'Difficulty', 'Similarity'])
    df_level = df_level.sort_values('Step', key=pd.to_numeric)
    return df_level

def plotBertandRobertaWithDifficulty(df):
    g = sns.FacetGrid(df, col='PretrainedModel', hue='Difficulty', height=4, aspect= 1.5)
    g = g.map(sns.lineplot, 'Step', 'Similarity', hue_order=[0, 1, 2, 3])
    g.set_xticklabels(rotation=90)
    g.add_legend()
    plt.show()

def main():
    # get all prediction files from bert and roberta
    files = get_all_eval_files(LOG_PATH)
    # get average similarity scores of bert and roberta
    df = compareBertandRoberta(files)
    #plot
    plotBertandRoberta(df)
    # get average similarity scores of bert and roberta with different difficulty levels.
    df_difficulty = compareBertandRobertaWithDifficulty(files)
    #plot
    plotBertandRobertaWithDifficulty(df_difficulty)

if __name__ == "__main__":
    main()
