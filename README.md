- In tihs I will Develop a website using different features and technolagies I have laerned to create a universe of one my favourite anime namely naruto.

- I have done some we scraping using crawler to get information about different jutsus used in the ninja world of naruto. To classify and define different jutsus used. i\I preffered web scrapping as I could not fing a proper datset to use to get the entire list of jutsus, hence I had to scrape it from the web itself.

- I will be applying zero shot classifier. I will tsake two inputs. The premise  which is the text and the hypothesis which is the class. The output will be three different classes, namely Entailment, nuetral and controdiction. I will be using the hugging face library to do this.

- I am using a NLP model namely [facebook-Bart](https://huggingface.co/facebook/bart-large-mnli)
I am finetuning the transfoemer model on different themes [friendhip", "hope", "sacrifice", "Loss", "battle", "self developmet", "betrayal", "love", "dialogue"]

- Created a dataframe by loading it into our loading the subtitles into our jupyter notebook and cleaning the subtitle data.

- Created a new colums stating the themes of the subtitles we have with huggingface library.