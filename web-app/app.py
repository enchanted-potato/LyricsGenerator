import json
import os

from flask import Flask, render_template, request, session

from src.get_lyrics import capitalise_list, get_lyrics
from src.train import Model_NN

config_path = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(config_path, "config.json")) as jf:
    config = json.load(jf)

root_dir = config["paths"]["root_path"]

app = Flask(__name__)

app.secret_key = "hello"


@app.route("/")
def web_page():
    return render_template("simple.html")


@app.route("/artist", methods=["POST"])
def insert_text():
    if request.method == "POST":
        artist_name = request.form.get("artist")
        number_songs = request.form.get("number")

        try:
            titles, lyrics, no_of_songs = get_lyrics(
                artist=artist_name,
                max_no_songs=number_songs,
                path_to_txt=root_dir + "/Data/lyrics_new.txt",
            )
            train = True

        except:
            train = False
            titles = []
            no_of_songs = None
    print(train)
    return render_template(
        "simple.html", titles=titles, no_songs=no_of_songs, is_post=True, train=train
    )


@app.route("/train", methods=["POST"])
def train():
    model_obj = Model_NN(
        config["model_config"],
        root_dir + config["paths"]["lyrics_path"],
        root_dir + config["paths"]["glove_path"],
    )

    model_obj.load_glove()

    model_obj.data_sequences()

    model_obj.prepare_data()

    model_obj.custom_embedding_layer()

    model_obj.train_model()

    model_obj.predict_model()

    no_lines = int(request.form.get("lines"))
    predictions = capitalise_list(model_obj.generate_lyrics(no_lines))

    session["predictions"] = predictions

    if len(predictions) > 0:
        predicted = True
    else:
        predicted = False

    return render_template("result.html", prediction=predicted)


@app.route("/result", methods=["POST"])
def predict():
    lyrics = session.get("predictions", None)

    return render_template("result.html", lyrics=lyrics)


def main():
    app.run(debug=True)
    return None


if __name__ == "__main__":
    main()
