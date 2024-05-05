import lyricsgenius as lg


def _save_lyrics_to_file(lyrics, path_to_txt):
    try:
        with open(path_to_txt, "w", encoding="utf-8") as file:
            file.write("\n\n<|endoftext|>\n\n".join(lyrics))
        print("Lyrics saved successfully.")
    except Exception as e:
        print("Failed to save lyrics:", e)

def get_lyrics(
    artist,
    max_no_songs,
    path_to_txt="/Data/lyrics_new.txt",
    excluded_terms=["(Remix)", "(Live)"],
    skip_non_songs=True,
    remove_section_headers=True,
):
    assert isinstance(artist, str)
    assert isinstance(path_to_txt, str)

    genius = lg.Genius(
        "fJ8UODl_rAGEKCKP-Xpy9IOj0sYLKxXjuZMPGzolumJmsw_mqiflsWFREcWjsWEE",
        skip_non_songs=skip_non_songs,
        excluded_terms=excluded_terms,
        remove_section_headers=remove_section_headers,
    )

    try:
        # Attempt to search for the artist and retrieve songs
        artist_info = genius.search_artist(artist, max_songs=int(max_no_songs), sort="popularity")
    except Exception as e:
        # If an exception occurs during artist search, print an error message and return None
        print("An error occurred while searching for the artist:", e)
        return None, None, 0

    try:
        # Extract titles and lyrics from retrieved songs
        songs = artist_info.songs
        titles = [song.title for song in songs]
        lyrics = [song.lyrics for song in songs]
        total_no_songs = len(lyrics)
    except Exception as e:
        # If an exception occurs while extracting titles and lyrics, print an error message and return None
        print("An error occurred while extracting titles and lyrics:", e)
        return None, None, 0

    try:
        # Attempt to save lyrics to a file
        _save_lyrics_to_file(lyrics, path_to_txt)
    except Exception as e:
        # If an exception occurs while saving lyrics to a file, print an error message
        print("An error occurred while saving lyrics to a file:", e)

    # If everything is successful, return titles, lyrics, and total number of songs
    return titles, lyrics, total_no_songs


def capitalise_list(text_list):
    return [lyric.capitalize() for lyric in text_list]
