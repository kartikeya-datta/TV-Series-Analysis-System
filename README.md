- In this I will Develop a website using different features and technolagies I have laerned to create a universe of one my favourite anime namely naruto.

## Theme Analysis Process in the jupyter notebook:

* **Model Loading and Setup:**
    * The "facebook/bart-large-mnli" model was loaded for zero-shot classification, configured for PyTorch and to run on a GPU if available.
    * A list of themes (`theme_list`) was defined for the classification task.
    * A test classification was performed on a sample sentence to ensure the model was working.
* **Subtitle Data Processing:**
    * Subtitle files were loaded from the `../data/Subtitles/` directory.
    * The subtitle data was cleaned, removing header information and formatting characters.
    * The cleaned dialogue was concatenated into single strings for each episode.
    * The episode number was also extracted from each file name.
    * The data was then loaded into a pandas DataFrame.
* **Sentence Batching:**
    * The dialogue for each episode was split into sentences using NLTK's `sent_tokenize`.
    * To accommodate the model's token limit, sentences were grouped into batches of 20.
* **Theme Inference:**
    * The zero-shot classification model was applied to each batch of sentences.
    * The model's output (theme scores) was aggregated and averaged to produce a thematic profile for each episode.
* **DataFrame Enrichment:**
    * The resulting theme data was then added to the original DataFrame, to create a DataFrame that contained both the original subtitle data and the theme classification data.