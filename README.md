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


- **I am going to use Gradio for creating an interactive demo of the model.**
- I am using gradio to make a visuyal representation of the themes I want to choose and the inference of our model on the selected themes based on our subtitles. 
- I have provided witht three text fields to enter namely themes, Subtitles and the path to save our file.
- Using the data we will plot a Bar-Graph of all the themes to be visualised and their score based on the output of zero-shot classifier. 
_ The sample themes I have picked are friendship, hope, sacrifice, Loss, battle, self development, betrayal, love, dialogue. with a subtitle script path /Users/kartikeyadatta/Documents/Projects/Naruto/code/data/Subtitles anssd I saved my file to /Users/kartikeyadatta/Documents/Projects/Naruto/code/stubs/theme_classifier_output.csv

- I have used the GPU of google colab to compute my theme classification outputnand stored it in the stubs folder to save the current data and use it to represenrt the theme classsification.


