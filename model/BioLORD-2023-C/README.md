---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- medical
- biology
language: en
license: other
license_name: ihtsdo-and-nlm-licences
license_link: https://www.nlm.nih.gov/databases/umls.html
datasets:
- FremyCompany/BioLORD-Dataset
- FremyCompany/AGCT-Dataset
widget:
- source_sentence: bartonellosis
  sentences:
  - cat scratch disease
  - cat scratch wound
  - tick-borne orbivirus fever
  - cat fur

---

| 🙏 If you are able to, please help me [fund my open research](https://gofund.me/1f2d6803). 🙏 Thank you for your generosity! 🤗  |
|-----------------------------------------------------------------------------------------------------------------------------------|

# FremyCompany/BioLORD-2023-C
This model was trained using BioLORD, a new pre-training strategy for producing meaningful representations for clinical sentences and biomedical concepts. 

State-of-the-art methodologies operate by maximizing the similarity in representation of names referring to the same concept, and preventing collapse through contrastive learning. However, because biomedical names are not always self-explanatory, it sometimes results in non-semantic representations. 

BioLORD overcomes this issue by grounding its concept representations using definitions, as well as short descriptions derived from a multi-relational knowledge graph consisting of biomedical ontologies. Thanks to this grounding, our model produces more semantic concept representations that match more closely the hierarchical structure of ontologies. BioLORD-2023 establishes a new state of the art for text similarity on both clinical sentences (MedSTS) and biomedical concepts (EHR-Rel-B).

This model is based on [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and was further finetuned on the [BioLORD-Dataset](https://huggingface.co/datasets/FremyCompany/BioLORD-Dataset) and LLM-generated definitions from the [Automatic Glossary of Clinical Terminology (AGCT)](https://huggingface.co/datasets/FremyCompany/AGCT-Dataset).

## Sibling models

This model is accompanied by other models in the BioLORD-2023 series, which you might want to check:

- [BioLORD-2023-M](https://huggingface.co/FremyCompany/BioLORD-2023-M) (multilingual model; distilled from BioLORD-2023)
- [BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023) (best model after model averaging)
- [BioLORD-2023-S](https://huggingface.co/FremyCompany/BioLORD-2023-S) (best hyperparameters; no model averaging)
- [BioLORD-2023-C](https://huggingface.co/FremyCompany/BioLORD-2023-C) (contrastive training only; for NEL tasks; this model)

You can also take a look at last year's model and paper:

- [BioLORD-2022](https://huggingface.co/FremyCompany/BioLORD-STAMB2-v1) (also known as BioLORD-STAMB2-v1)

## Training strategy

### Summary of the 3 phases
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f04e8865d08220171a0ad3f/my94lNjxATRU_Rg5knUZ8.png)

### Contrastive phase: details
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f04e8865d08220171a0ad3f/_jE2ETcXkLvYLr7TeOdci.png)

### Self-distallation phase: details
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f04e8865d08220171a0ad3f/7xuqi231RB0OzvcxK3bf-.png)

## Citation
This model accompanies the [BioLORD-2023: Learning Ontological Representations from Definitions](https://arxiv.org/abs/2311.16075) paper. When you use this model, please cite the original paper as follows:

```latex
@article{remy-etal-2023-biolord,
    author = {Remy, François and Demuynck, Kris and Demeester, Thomas},
    title = "{BioLORD-2023: semantic textual representations fusing large language models and clinical knowledge graph insights}",
    journal = {Journal of the American Medical Informatics Association},
    pages = {ocae029},
    year = {2024},
    month = {02},
    issn = {1527-974X},
    doi = {10.1093/jamia/ocae029},
    url = {https://doi.org/10.1093/jamia/ocae029},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocae029/56772025/ocae029.pdf},
}
```

## Usage (Sentence-Transformers)
This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search. This model has been finentuned for the biomedical domain. While it preserves a good ability to produce embeddings for general-purpose text, it will be more useful to you if you are trying to process medical documents such as EHR records or clinical notes. Both sentences and phrases can be embedded in the same latent space.

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:
```
pip install -U sentence-transformers
```
Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
sentences = ["Cat scratch injury", "Cat scratch disease", "Bartonellosis"]

model = SentenceTransformer('FremyCompany/BioLORD-2023-C')
embeddings = model.encode(sentences)
print(embeddings)
```
## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.
```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ["Cat scratch injury", "Cat scratch disease", "Bartonellosis"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023-C')
model = AutoModel.from_pretrained('FremyCompany/BioLORD-2023-C')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:")
print(sentence_embeddings)
```

## License
My own contributions for this model are covered by the MIT license.
However, given the data used to train this model originates from UMLS and SnomedCT, you will need to ensure you have proper licensing of UMLS and SnomedCT before using this model. Both UMLS and SnomedCT are free of charge in most countries, but you might have to create an account and report on your usage of the data yearly to keep a valid license.