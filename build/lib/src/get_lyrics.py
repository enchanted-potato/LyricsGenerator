import lyricsgenius as lg


def get_lyrics(artist, max_no_songs, path_to_txt="C:/Users/kristia.karakatsani/PycharmProjects/TokioHotel/Data/lyrics_new.txt",
               excluded_terms=["(Remix)", "(Live)"],skip_non_songs=True, remove_section_headers=True):

    assert(isinstance(artist, str))
    assert(isinstance(path_to_txt, str))


    genius = lg.Genius('fJ8UODl_rAGEKCKP-Xpy9IOj0sYLKxXjuZMPGzolumJmsw_mqiflsWFREcWjsWEE',
                       skip_non_songs = skip_non_songs,
                       excluded_terms = excluded_terms,
                       remove_section_headers = remove_section_headers
    )
    try:
        songs = (genius.search_artist(artist, max_songs=int(max_no_songs), sort='popularity')).songs
        titles = [song.title for song in songs]
        lyrics = [song.lyrics for song in songs]
        total_no_songs = len(lyrics)
        try:
            file = open(path_to_txt, 'w', encoding='utf-8')
            file.write("\n\n<|endoftext|>\n\n".join(lyrics))
            print(f"Saved lyrics")
            return titles, lyrics, total_no_songs
        except:
            print("Found lyrics but did not save text")

    except:
        print("Failed")
        titles = None
        lyrics = None
        total_no_songs =0
        return titles, lyrics, total_no_songs




def capitalise_list(text_list):

    return [lyric.capitalize() for lyric in text_list]
