Train_models outlines the process used to train save and ensemble our three different LLM models.

Generate_counterfactuals builds on this by loading in the pretrained models and their classification
to save on processing time and then outlines the prompting pipeline used to generate and
evaluate counterfactual sentences by utilizing the OpenAi API.

The dataset used for training and validation can be found under the Data directory.
