
import snscrape.modules.twitter as sntwitter
from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
from sklearn.metrics import mean_absolute_error
from langdetect import detect

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a la API de Predicción de sentimiento de tweets"

#1. Genera un resumen de tweets positivos/negativos en base a los inputs del usuario
@app.route('/v2/predict', methods=['GET'])
def predict():
    modelo = pickle.load(open('sentiment_model','rb'))

    account = request.args.get('account', None)
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    
    query = f"{account} since:{start_date} until:{end_date}"

    if account is None or start_date is None or end_date is None:
        return "Missing args, the input values are needed to predict. Please enter \n account: @XXXX \n start_date: YYYY-MM-DD \n end_date: YYYY-MM-DD"
    else:
        tweets = []
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweet_id = tweet.id
            text = tweet.rawContent
            date = tweet.date.strftime('%Y-%m-%d %H:%M:%S')
            author_id = tweet.user.id
            author_name = tweet.user.displayname
            author_username = tweet.user.username
            retweets = tweet.retweetCount
            replies = tweet.replyCount
            likes = tweet.likeCount
            quotes = tweet.quoteCount

            tweets.append([tweet_id, text, date, author_id, author_name, author_username, retweets, replies, likes, quotes])
            
            
            # Romper el bucle si se alcanza el límite de tweets deseados
            if len(tweets) >= 10000:
                break
            
    tweets_es = []

    for tweet in tweets:
        if detect(tweet[1]) == 'es':
            tweets_es.append(tweet)

    # Crear un DataFrame a partir de la lista de tweets
    columns = ["ID", "Text", "Date", "Author ID", "Author Name", "Author Username", "Retweets", "Replies", "Likes", "Quotes"]
    tweets_df = pd.DataFrame(tweets_es, columns=columns)
    
    resultado=[]
    for i in range(len(tweets_df["Text"])):
        pred=modelo.predict([tweets_df.iloc[i][1]])
        resultado.append(pred[0])
    df_res=pd.DataFrame(resultado, columns=["predict"])
    sum_pred=df_res.value_counts()
    
    resultado_prob = [modelo.predict_proba([tweets_df.iloc[i][1]])[0].tolist() for i in range(len(tweets_df["Text"]))]
    df_prob=pd.DataFrame(resultado_prob, columns=["0", "1"])
    df_prob["0"] = df_prob["0"].astype(float)
    indices_top = df_prob["0"].nlargest(5).index
    indices_bottom = df_prob["0"].nsmallest(5).index

    
    return (
    f"Se han encontrado {sum_pred[0]+sum_pred[1]} tweets. {sum_pred[0]} positivos y {sum_pred[1]} negativos\n\n\n"
    "Los tweets más claramente negativos\n"
    + "*"*50 + "\n"
    "1.-\n"
    + tweets_df.iloc[indices_bottom[0]][1] + "\n\n"
    "2.-\n"
    + tweets_df.iloc[indices_bottom[1]][1] + "\n\n"
    "3.-\n"
    + tweets_df.iloc[indices_bottom[2]][1] + "\n\n"
    "4.-\n"
    + tweets_df.iloc[indices_bottom[3]][1] + "\n\n"
    "5.-\n"
    + tweets_df.iloc[indices_bottom[4]][1] + "\n\n\n"
    "Los tweets más claramente positivos\n"
    + "*"*50 + "\n"
    "1.-\n"
    + tweets_df.iloc[indices_top[0]][1] + "\n\n"
    "2.-\n"
    + tweets_df.iloc[indices_top[1]][1] + "\n\n"
    "3.-\n"
    + tweets_df.iloc[indices_top[2]][1] + "\n\n"
    "4.-\n"
    + tweets_df.iloc[indices_top[3]][1] + "\n\n"
    "5.-\n"
    + tweets_df.iloc[indices_top[4]][1] + "\n\n"
    )
        
        
#2. Muestra la tabla de tweets
@app.route('/v2/bbdd_tweets', methods=["GET", "POST"])
def bbdd_tweets():
    connection = sqlite3.connect('tweets_thebridge_new.db')
    cursor = connection.cursor()
    
    query="SELECT * from tweets"

    result=cursor.execute(query).fetchall()
    connection.commit()
    connection.close()
    
    return str(result)

#3. Muestra la tabla de usuarios
@app.route('/v2/bbdd_users', methods=["GET", "POST"])
def bbdd_users():
    connection = sqlite3.connect('tweets_thebridge_new.db')
    cursor = connection.cursor()
    
    query="SELECT * from users"

    result=cursor.execute(query).fetchall()
    connection.commit()
    connection.close()
    
    return str(result)

#4. Hace un análisis de los tweets guardados en la bbdd
@app.route('/v2/analysis_bbdd', methods=["GET", "POST"])
def analysis_bbdd():
    connection = sqlite3.connect('tweets_thebridge_new.db')
    cursor = connection.cursor()
    modelo = pickle.load(open('sentiment_model','rb'))
    query="SELECT * from tweets"
    
    cursor.execute(query).fetchall()
    connection.commit()
    df = pd.read_sql_query(query,connection)
    
    columns = ["ID", "Text", "Date", "Author ID", "Author Name", "Author Username", "Retweets", "Replies", "Likes", "Quotes"]
    tweets_df = pd.DataFrame(df, columns=columns)
    
    resultado=[]
    for i in range(len(tweets_df["Text"])):
        pred=modelo.predict([tweets_df.iloc[i][1]])
        resultado.append(pred[0])
    df_res=pd.DataFrame(resultado, columns=["predict"])
    sum_pred=df_res.value_counts()
    
    resultado_prob = [modelo.predict_proba([tweets_df.iloc[i][1]])[0].tolist() for i in range(len(tweets_df["Text"]))]
    df_prob=pd.DataFrame(resultado_prob, columns=["0", "1"])
    df_prob["0"] = df_prob["0"].astype(float)
    indices_top = df_prob["0"].nlargest(5).index
    indices_bottom = df_prob["0"].nsmallest(5).index

    
    return (
    f"Se han encontrado {sum_pred[0]+sum_pred[1]} tweets. {sum_pred[0]} positivos y {sum_pred[1]} negativos\n\n\n"
    "Los tweets más claramente negativos\n"
    + "*"*50 + "\n"
    "1.-\n"
    + tweets_df.iloc[indices_bottom[0]][1] + "\n\n"
    "2.-\n"
    + tweets_df.iloc[indices_bottom[1]][1] + "\n\n"
    "3.-\n"
    + tweets_df.iloc[indices_bottom[2]][1] + "\n\n"
    "4.-\n"
    + tweets_df.iloc[indices_bottom[3]][1] + "\n\n"
    "5.-\n"
    + tweets_df.iloc[indices_bottom[4]][1] + "\n\n\n"
    "Los tweets más claramente positivos\n"
    + "*"*50 + "\n"
    "1.-\n"
    + tweets_df.iloc[indices_top[0]][1] + "\n\n"
    "2.-\n"
    + tweets_df.iloc[indices_top[1]][1] + "\n\n"
    "3.-\n"
    + tweets_df.iloc[indices_top[2]][1] + "\n\n"
    "4.-\n"
    + tweets_df.iloc[indices_top[3]][1] + "\n\n"
    "5.-\n"
    + tweets_df.iloc[indices_top[4]][1] + "\n\n"
    )
    

app.run()  
    