# Automatic Sexism Detection

For a detailed overview of the project and the meanings of the terms referred to below, please refer to the provided presentation document, i.e. ```automatic_sexism_detection.pdf```.

The original dataset csv files are included in the folder ```Call me sexist but```, and the csv files containing the GPT-generated items are contained in the folder ```gpt```. The requisite data pre-processing and cleaning is done automatically on the fly when running ```task1.py``` or ```task3.py```.

- ```task1.py``` produces the binary (sexist / not sexist) and fine-grained (type of sexism) classification results while using only scale items from the original 'Call me sexist but' dataset. Also prints a list of different interesting tweets (e.g. some of the tweets with the top and lowest sexist/non-sexist scores, tweets falling near the classification threshold, etc.). For the default values of ```k``` and ```ε```, these tweets have been saved in ```task1_tweets_of_interest.txt```.

- ```task2.ipynb``` is the code used for generating and saving as csv files the GPT-generated items. We generate both sexist and non-sexist (neutral) sentences/tweet-like items using GPT.

- ```task3.py``` produces the tweets' binary classification results when using all available scale items - the original scales from 'Call me sexist but' dataset, as well as all the GPT-generated items. The choice to use all of these as scale items was made after our evaluations showed that this resulted in arguably the best classification performance overall. Similar to ```task1.py```, running ```task3.py``` also produces a list of various tweets of interest. For the default values of ```k``` and ```ε```, these tweets have been saved in ```task3_tweets_of_interest.txt```.

- ```tsne_plots.py``` contains the code to produce t-SNE plots to visualize the different categories of tweets/sentences in embedding space, e.g. to see how sexist vs non-sexist tweets/scale items are situated relative to each other. To choose which t-SNE plot to generate out of the six shown in the presentation document, the parameter ```which_plot``` in line 10 can be set to an integer value from 1 till 6.

## Parameters and Saving Similarity Scores

The default values of the parameters ```k``` and ```ε``` of the classifier can be changed, separately for ```task1.py``` and ```task3.py``` (lines 13-14 in both). Furthermore, since the computation of sentence encodings and similarities can be time-consuming, we provide the option to save the cosine similarity scores in ```task1.py``` and ```task3.py```. There is a boolean variable ```load``` in these scripts (line 11 in both): if ```load=False```, it will save the scores in the ```scores``` folder; if ```load=True```, it will load these files generated from the last ```load=False``` run, instead of re-computing encodings and similarities. By default, we do not provide pre-computed scores due to their large size, and therefore have set ```load=False```, but after running a file once, it can be set to ```load=True``` afterwards. However, if you wish to re-run a classifier with changed parameters, please first change it to ```load=False``` again, otherwise the changes will not take effect.
