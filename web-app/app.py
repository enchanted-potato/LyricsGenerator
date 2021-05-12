import os
import json
from flask import Flask, render_template, request, session
from src.get_lyrics import get_lyrics, capitalise_list
from src.train import Model_NN

app = Flask(__name__)

app.secret_key = "hello"

with open(os.path.join('..', 'config.json')) as jf:
    config = json.load(jf)

root_dir = config['paths']['root_path']


@app.route('/')
def web_page():
    return render_template('simple.html')


@app.route('/artist', methods=['GET', 'POST'])
def insert_text():
    if request.method=='POST':
        artist_name = request.form.get("artist")
        number_songs = request.form.get("number")

        try:
            titles, lyrics, no_of_songs = get_lyrics(artist = artist_name,
                                                     max_no_songs = number_songs,
                                                     path_to_txt=root_dir + config['paths']['lyrics_path'])
            train = True



        except:
            train = False
            titles= []
            no_of_songs = None
    print(train)

    return render_template('simple.html', titles=titles, no_songs=no_of_songs, is_post=True, train=train)


@app.route('/train', methods=['GET', 'POST'])
def train():
    model_obj = Model_NN(config['model_config'],
                         root_dir + config['paths']['lyrics_path'],
                         root_dir + config['paths']['glove_path'],
                         root_dir + config['paths']['data_path'])

    model_obj.load_glove()

    model_obj.data_sequences()

    session['max_sequence_length'] = model_obj.prepare_data()

    model_obj.custom_embedding_layer()

    model_obj.train_model()

    # we need to save this to use it in predict function
    model_obj.predict_model()

    no_lines = int(request.form.get("lines"))
    predictions = capitalise_list(model_obj.generate_lyrics(no_lines,
                                                            config,
                                                            session['max_sequence_length']))

    session['predictions'] = predictions
    with open(root_dir + config['paths']['data_path'] + "/predictions.txt", "w") as output:
        output.write(str(predictions))

    if len(predictions)>0:
        predicted=True
    else:
        predicted=False

    return render_template('result.html', prediction = predicted)

@app.route('/result', methods=['GET', 'POST'])
def predict():

    lyrics = session.get('predictions', None)

    return render_template('result.html', lyrics = lyrics)

@app.route('/results', methods=['GET', 'POST'])
def predict_extra():
    if request.method=='POST':

        no_lines = int(request.form.get("lines"))

        # lyrics = session.get('predictions', None)
        # print(len(lyrics))

        lyrics_new = (capitalise_list(Model_NN.generate_lyrics(no_lines,
                                                              config,
                                                              session['max_sequence_length'])))

        if len(lyrics_new) > 0:
            predicted = True
        else:
            predicted = False

    return render_template('result.html', new_prediction = predicted, lyrics = lyrics_new)



if __name__ == "__main__":
    with open(os.path.join('..', 'config.json')) as jf:
        config = json.load(jf)

    root_dir = config['paths']['root_path']
    app.run(debug=True)
