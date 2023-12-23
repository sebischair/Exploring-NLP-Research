# Exploring-NLP-Research
Repository of the RANLP 2023 paper "Exploring the Landscape of Natural Language Processing Research"

# NLP Taxonomy Classifier

This is a fine-tuned BERT-based language model to classify NLP-related research papers according to concepts included in the [NLP taxonomy](#nlp-taxonomy). 
It is a multi-label classifier that can predict concepts from all levels of the NLP taxonomy. 
If the model identifies a lower-level concept, it did learn to predict both the lower-level concept and its hypernyms in the NLP taxonomy.
The model is fine-tuned on a weakly labeled dataset of 178,521 scientific papers from the ACL Anthology, the arXiv cs.CL domain, and Scopus.
Prior to fine-tuning, the model is initialized with weights from [allenai/specter2_base](https://huggingface.co/allenai/specter2_base).

📄 Paper: [Exploring the Landscape of Natural Language Processing Research (RANLP 2023)](https://aclanthology.org/2023.ranlp-1.111).

🤗 Model: [https://huggingface.co/TimSchopf/nlp_taxonomy_classifier](https://huggingface.co/TimSchopf/nlp_taxonomy_classifier)

💾 Data: [https://huggingface.co/datasets/TimSchopf/nlp_taxonomy_data](https://huggingface.co/datasets/TimSchopf/nlp_taxonomy_data)

<a name="#nlp-taxonomy"/></a>
## NLP Taxonomy

![NLP taxonomy](https://github.com/sebischair/Exploring-NLP-Research/blob/main/figures/NLP-Taxonomy.jpg?raw=true)


## How to use the fine-tuned model

### Get predictions by loading the model directly
```python
from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('TimSchopf/nlp_taxonomy_classifier')
model = BertForSequenceClassification.from_pretrained('TimSchopf/nlp_taxonomy_classifier')

# prepare data
papers = [{'title': 'Attention Is All You Need', 'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.'},
          {'title': 'SimCSE: Simple Contrastive Learning of Sentence Embeddings', 'abstract': 'This paper presents SimCSE, a simple contrastive learning framework that greatly advances state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation, and removing it leads to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language inference datasets into our contrastive learning framework by using "entailment" pairs as positives and "contradiction" pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT base achieve an average of 76.3% and 81.6% Spearmans correlation respectively, a 4.2% and 2.2% improvement compared to the previous best results. We also show -- both theoretically and empirically -- that the contrastive learning objective regularizes pre-trained embeddings anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.'}]
# concatenate title and abstract with [SEP] token
title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]


def predict_nlp_concepts(model, tokenizer, texts: List[str], batch_size=8, device=None, shuffle_data=False):
    """
    helper function for predicting NLP concepts of scientific papers
    """
    
    # tokenize texts
    def tokenize_dataset(sentences, tokenizer):
        sentences_num = len(sentences)
        dataset = []
        for i in range(sentences_num):
            
            sentence = tokenizer(sentences[i], padding="max_length", truncation=True, return_tensors='pt', max_length=model.config.max_position_embeddings)
            
            # get input_ids, token_type_ids, and attention_mask
            input_ids = sentence['input_ids'][0]
            token_type_ids = sentence['token_type_ids'][0]
            attention_mask = sentence['attention_mask'][0]

            dataset.append((input_ids, token_type_ids, attention_mask))
        return dataset

    tokenized_data = tokenize_dataset(sentences=texts, tokenizer=tokenizer)
    
    # get the individual input formats for the model
    input_ids = torch.stack([x[0] for x in tokenized_data])
    token_type_ids = torch.stack([x[1] for x in tokenized_data])
    attention_mask_ids = torch.stack([x[2].to(torch.float) for x in tokenized_data])
    
    # convert input to DataLoader
    input_dataset = []
    for i in range(len(input_ids)):
        data = {}
        data['input_ids'] = input_ids[i]
        data['token_type_ids'] = token_type_ids[i]
        data['attention_mask'] = attention_mask_ids[i]
        input_dataset.append(data)

    dataloader = DataLoader(input_dataset, shuffle=shuffle_data, batch_size=batch_size)
    
    # predict data
    if not device:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.eval()
    y_pred = torch.tensor([]).to(device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids_batch = batch['input_ids']
        token_type_ids_batch = batch['token_type_ids']
        mask_ids_batch = batch['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=mask_ids_batch, token_type_ids=token_type_ids_batch)

        logits = outputs.logits
        predictions = torch.round(torch.sigmoid(logits))
        y_pred = torch.cat([y_pred,predictions])
        
    
    # get prediction class names
    prediction_indices_list = []
    for prediction in y_pred:
        prediction_indices_list.append((prediction == torch.max(prediction)).nonzero(as_tuple=True)[0])

    prediction_class_names_list = []
    for prediction_indices in prediction_indices_list:
        prediction_class_names = []
        for prediction_idx in prediction_indices:
            prediction_class_names.append(model.config.id2label[int(prediction_idx)])
        prediction_class_names_list.append(prediction_class_names)

    return y_pred, prediction_class_names_list

# predict concepts of NLP papers
numerical_predictions, class_name_predictions = predict_nlp_concepts(model=model, tokenizer=tokenizer, texts=title_abs)
```
### Use a pipeline to get predictions

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="TimSchopf/nlp_taxonomy_classifier")

# prepare data
papers = [{'title': 'Attention Is All You Need', 'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.'},
          {'title': 'SimCSE: Simple Contrastive Learning of Sentence Embeddings', 'abstract': 'This paper presents SimCSE, a simple contrastive learning framework that greatly advances state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation, and removing it leads to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language inference datasets into our contrastive learning framework by using "entailment" pairs as positives and "contradiction" pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT base achieve an average of 76.3% and 81.6% Spearmans correlation respectively, a 4.2% and 2.2% improvement compared to the previous best results. We also show -- both theoretically and empirically -- that the contrastive learning objective regularizes pre-trained embeddings anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.'}]
# concatenate title and abstract with [SEP] token
title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]

pipe(title_abs, return_all_scores=True)
```
## Evaluation Results

The model was evaluated on a manually labeled test set of 828 different EMNLP 2022 papers. The following shows the average evaluation results for classifying papers according to the NLP taxonomy on three different training runs. Since the distribution of classes is very unbalanced, we report micro scores.

* **F1:** 93.21
* **Recall:** 93.99
* **Precision:** 92.46

## License
BSD 3-Clause License

## Citation information
When citing our work in academic papers and theses, please use this BibTeX entry:
``` 
@inproceedings{schopf-etal-2023-exploring,
    title = "Exploring the Landscape of Natural Language Processing Research",
    author = "Schopf, Tim  and
      Arabi, Karim  and
      Matthes, Florian",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.111",
    pages = "1034--1045",
    abstract = "As an efficient approach to understand, generate, and process natural language texts, research in natural language processing (NLP) has exhibited a rapid spread and wide adoption in recent years. Given the increasing research work in this area, several NLP-related approaches have been surveyed in the research community. However, a comprehensive study that categorizes established topics, identifies trends, and outlines areas for future research remains absent. Contributing to closing this gap, we have systematically classified and analyzed research papers in the ACL Anthology. As a result, we present a structured overview of the research landscape, provide a taxonomy of fields of study in NLP, analyze recent developments in NLP, summarize our findings, and highlight directions for future work.",
}
``` 
