Train_models outlines the process used to train save and ensemble our three different LLM models.

Generate_counterfactuals builds on this by loading in the pretrained models and their classification
to save on processing time and then outlines the prompting pipeline used to generate and
evaluate counterfactual sentences by utilizing the OpenAi API.

The front end folder contains the code for the flask app as well as the html code that goes with it.
This html file is also hosted at https://my.up.ist.psu.edu/ggw5057/DS440/index.html

The Data folder contains the datasets the original three models were fintuned on. Our plan is to 
retrain these LLM's using the same training process but with a political dataset rathter than
one that soley focuses on COVID-19 so that the models can be generalizded to a wider variety of topics.
