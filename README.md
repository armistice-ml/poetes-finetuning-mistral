# Fine-tuning d'un LLM pour écrire à la manière de poètes français

> Adaptation de **Mistral 7B Instruct v0.3** au style de **Victor Hugo, Charles Baudelaire et Paul Verlaine**, avec **QLoRA** (Unsloth), sur un GPU gratuit de Google Colab (T4).

[Ouvrir le notebook dans Colab](https://colab.research.google.com/drive/1TBDHH_WuGNfnk1rNZtDKVYRCxPjIjhta#scrollTo=V32yqBC-oz58) · [Adaptateurs LoRA sur Hugging Face](LIEN_HUGGING_FACE)

---

## Objectif

Les modèles de langage généralistes écrivent des poèmes corrects mais génériques. Ce projet cherche à **spécialiser un modèle open source dans la poésie française du XIXᵉ siècle**, en lui donnant une consigne du type :

> *Écris un poème dans le style de Charles Baudelaire. Sujet : la mer.*

C'est mon **premier fine-tuning de bout en bout**. Il me sert à comprendre chaque étape : construction d'un jeu de données, choix du modèle, QLoRA, entraînement, évaluation et limites.

## Résultats en bref

| Perplexité sur 45 extraits de validation inédits | Hugo | Baudelaire | Verlaine | Ensemble |
|---|---|---|---|---|
| Modèle de base | 12,5 | 12,4 | 14,2 | 12,9 |
| Après fine-tuning | 9,0 | 9,4 | 10,3 | **9,5** |

- La perplexité baisse d'environ **26 %** (plus bas = mieux).
- Aucun vers du jeu d'entraînement n'est recopié littéralement dans les six poèmes générés pour l'évaluation.
- **Test de conditionnement par auteur** : pour chaque poème de validation, la perplexité est plus basse quand la consigne nomme le *vrai* auteur que lorsqu'elle en nomme un autre, pour les trois poètes (9,57 en moyenne contre 9,86, soit environ 3 %). Le modèle utilise donc le nom de l'auteur, mais **faiblement** : il a surtout appris un registre poétique général du XIXᵉ siècle.

### Exemple (consigne : « Écris un poème dans le style de Victor Hugo. Sujet : la mer. », graine 0)

Avant fine-tuning :

```
Sur les bords de ce monde immense,
Où les vents meurent et naissent,
L'océan, sombre et effrayant,
A son souffle me dorent.
```

Après fine-tuning :

```
C'est la mer.
Et le vent a tous les tons de la musique.
Elle est profonde et verte comme la nuit;
Elle a les pleurs de l'infini en ses yeux;
```

Ce que l'on observe : un vocabulaire et des anaphores plus proches du XIXᵉ siècle, et un découpage en vraies strophes. Ce que l'on n'observe pas : des alexandrins rimés réguliers (voir *Limites*).

## Méthode

### 1. Données

- **Source** : textes du domaine public de [Project Gutenberg](https://www.gutenberg.org/) :
  - Baudelaire, *Les Fleurs du Mal* (ebook n° 6099)
  - Hugo, *Les Contemplations* (n° 29843 et 29844) et *La Légende des siècles*, tomes I et II (n° 72885 et 76396)
  - Verlaine, *Œuvres complètes, volume 1* (n° 15112)
- **Traitement** :
  - Retrait des en-têtes légaux, des préfaces, des dédicaces et des marqueurs d'italique.
  - Découpage en poèmes. Les poèmes trop longs sont coupés en extraits de 48 vers au maximum, toujours entre deux strophes.
  - Écartement de la prose (test de la majuscule en début de vers) et des répliques de théâtre.
  - Rééquilibrage : 300 extraits au maximum par auteur.
  - Retrait des extraits de plus de 1024 tokens.
- **Consignes** : le sujet de chaque extrait est généré par le modèle de base (*rétro-traduction d'instructions*), puis nettoyé et filtré. Trois formulations de consigne sont utilisées pour éviter que le modèle mémorise une phrase unique.
- **Découpage** : 15 extraits par auteur en validation (45 au total) ; le reste en entraînement ([à compléter : nombre d'exemples d'entraînement]).

### 2. Entraînement

| Élément | Valeur |
|---|---|
| Modèle de base | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (4 bits) |
| Méthode | QLoRA via Unsloth et TRL (`SFTTrainer`) |
| LoRA | rang 16, alpha 16, sur les 7 projections (q, k, v, o, gate, up, down) |
| Entraînement | 2 epochs, taux d'apprentissage 2e-4, taille de lot effective 8, `adamw_8bit` |
| Longueur maximale | 2048 tokens |
| Perte calculée sur | le poème uniquement (pas sur la consigne) |
| Matériel | Google Colab, GPU T4 |

Observation sur la courbe d'apprentissage : l'essentiel du gain vient de la première epoch. Pendant la seconde, la perte d'entraînement chute alors que celle de validation reste plate (début de mémorisation sans surapprentissage marqué). Pour une prochaine version, une seule epoch suffirait.

### 3. Évaluation

- Perplexité par auteur sur la validation, avant et après l'entraînement.
- Comparaison de poèmes générés, avec et sans adaptateurs, avec la même graine.
- Détection des vers recopiés mot pour mot depuis le jeu d'entraînement.
- Test du conditionnement par auteur (matrice des perplexités selon l'auteur nommé dans la consigne).

## Limites

- **Jeu de validation petit** (15 extraits par auteur) et **un seul entraînement** : les écarts entre auteurs sont dans le bruit de mesure.
- **Fuite possible entre entraînement et validation** : les extraits d'un même long poème de *La Légende des siècles* peuvent se retrouver des deux côtés, ce qui rend la validation un peu optimiste pour Hugo.
- **La perplexité ne mesure ni les rimes, ni le mètre, ni la qualité littéraire.** Le modèle écrit rarement des alexandrins rimés, et des fautes de langue subsistent (« Un étendue monstrueuse »).
- **Voix individuelles peu marquées** : les poèmes de plusieurs auteurs peuvent commencer de la même façon.
- **Les sujets des consignes sont générés par le modèle de base** : certains sont approximatifs, surtout pour les poèmes archaïques ou très métaphoriques.
- **Aucune évaluation humaine.**
- Le test de copie ne détecte que les vers recopiés à l'identique (plus de 25 caractères).

## Pistes d'amélioration

- Se concentrer sur **un seul auteur** pour obtenir une voix plus nette.
- Ajouter des recueils (par exemple *Les Orientales* ou *Les Châtiments* de Hugo) pour augmenter le nombre de textes par auteur.
- Mesurer la **versification** : nombre de syllabes par vers et taux de rimes.
- Utiliser une validation plus large, séparée par poème et non par extrait.
- Ajouter une évaluation humaine, en aveugle.
- Comparer avec une version **Base** du modèle, ou avec un autre modèle (Qwen3 8B, Llama 3.1 8B).

## Utilisation

Installer Unsloth, puis charger les adaptateurs :

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="CHEMIN_OU_ID_HUGGING_FACE_DES_ADAPTATEURS",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user",
             "content": "Écris un poème dans le style de Paul Verlaine. Sujet : l'automne."}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

sortie = model.generate(input_ids=inputs, max_new_tokens=300,
                        temperature=0.8, top_p=0.9, do_sample=True)
print(tokenizer.decode(sortie[0][inputs.shape[1]:], skip_special_tokens=True))
```

Les adaptateurs ne fonctionnent qu'avec le modèle de base indiqué ci-dessus (**Mistral 7B Instruct v0.3**).

## Contenu du dépôt

```
.
├── notebook.ipynb      # notebook complet, exécutable dans Colab
├── data/
│   ├── train.jsonl     # exemples d'entraînement (format conversationnel)
│   └── val.jsonl       # exemples de validation (45)
└── README.md
```

Les adaptateurs LoRA peuvent dépasser la limite de 100 Mo par fichier de GitHub : hébergez-les plutôt sur Hugging Face et indiquez le lien ci-dessus.

## Sources et licences

- **Textes** : Victor Hugo, Charles Baudelaire et Paul Verlaine, domaine public. Éditions numériques de [Project Gutenberg](https://www.gutenberg.org/).
- **Modèle de base** : Mistral 7B Instruct v0.3 (Mistral AI), licence Apache 2.0, quantifié en 4 bits par [Unsloth](https://github.com/unslothai/unsloth).
- **Adaptateurs LoRA** : licence Apache 2.0, publiés sur [Hugging Face](https://huggingface.co/armistice-ml/poetes-mistral-7b-lora).
- **Code de ce dépôt** : licence MIT (voir le fichier [LICENSE](LICENSE)).

## Ce que j'ai appris

- Construire un jeu de données propre à partir de textes bruts : c'est l'étape la plus longue, et celle qui pèse le plus sur le résultat.
- Pourquoi on choisit un modèle *Instruct* et comment fonctionne le *chat template*.
- Le principe de QLoRA : un modèle quantifié en 4 bits et de petits adaptateurs entraînables.
- Lire une courbe d'apprentissage et repérer le début de la mémorisation.
- Vérifier ce que le modèle utilise vraiment, par exemple avec le test du conditionnement par auteur.
- Évaluer honnêtement un modèle, y compris quand le résultat est modeste.

## Auteur

**Armistice ANAGONOU** · [GitHub](https://github.com/armistice-ml) · [Hugging Face](https://huggingface.co/armistice-ml) · [Portfolio](URL_DU_PORTFOLIO) · [LinkedIn](URL_LINKEDIN)
