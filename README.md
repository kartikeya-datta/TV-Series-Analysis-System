- In this I will Develop a website using different features and technolagies I have laerned to create a universe of one my favourite anime namely naruto.

## Theme Analysis Process in the jupyter notebook:

* **Model Loading and Setup:**
    * The ![facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model was loaded for zero-shot classification, configured for PyTorch and to run on a GPU if available. (I have used google colab for GPU powered tasks😄)
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


- **I am going to use Gradio for creating an interactive demo of the model.**
- I am using gradio to make a visuyal representation of the themes I want to choose and the inference of our model on the selected themes based on our subtitles. 
- I have provided witht three text fields to enter namely themes, Subtitles and the path to save our file.
- Using the data we will plot a Bar-Graph of all the themes to be visualised and their score based on the output of zero-shot classifier. 
_ The sample themes I have picked are friendship, hope, sacrifice, Loss, battle, self development, betrayal, love, dialogue. with a subtitle script path /Users/kartikeyadatta/Documents/Projects/Naruto/code/data/Subtitles anssd I saved my file to /Users/kartikeyadatta/Documents/Projects/Naruto/code/stubs/theme_classifier_output.csv

- I have used the GPU of google colab to compute my theme classification outputnand stored it in the stubs folder to save the current data and use it to represenrt the theme classsification.

![Theme Classification output](data/images/themeClassificationOutput.png)

- **Creation of Named Entity Recognition**

- Loaded the spaCy transformer model ("en_core_web_trf") to perform NER on the script dataset.

- Extracted person names ("PERSON" label) and store first names in a set to avoid duplicates.

-  Character Network Generation

- Created a sliding window over sentences (size 10) to establish relationships based on character proximity.

- Stored relationships in a DataFrame and aggregate connections.

- Visualized the character network using NetworkX and PyVis, generating an interactive HTML graph.

**Output**

- The output is a dynamic, interactive network graph (naruto.html) showing character relationships based on script proximity.

- Have succesfully create a network of connection between characters in the naruto series.

- Here is a visualisation of the Character relation graph based on their interaction in the subtitles.

![Character Relation](data/images/CharacterNersGraphs.png)

- **Now am gaing to create a jutsu classification to refere to.**

- Simplified the jutsu classification to the major three, which is **Ninjutsu** **Taijutsu** and **Genjutsu**.

- cleaned the data and removed any noise from the json file if there any.

- Tokenized the input for the model to implemet using preprocessing from scikit-learn.

- I split the data into training and testing sets.

- Shall  be using ![distilbertbase-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) model for my tokenisation and fed forwarn to the nueral network.

- Defined a preprocessing function to handle text input efficiently.

- Converted Pandas DataFrames to Hugging Face Datasets and applied the tokenizer to the datasets for model readiness.

- Got class weights to penalise the less data more as the data is more squed towards ninjutsu.

