Description
-----------

Ce script Python contient plusieurs fonctions pour l'analyse de texte en français. Il utilise des bibliothèques telles que NLTK, OpenAI, NumPy, Pandas, Matplotlib et Streamlit pour fournir des fonctionnalités telles que le calcul de la richesse lexicale, grammaticale et verbale, la génération de texte, la détection de texte généré et la visualisation de données.

Fonctionnalités
---------------

* Calcul de la richesse lexicale, grammaticale et verbale d'un texte.
* Génération de texte en utilisant l'API OpenAI.
* Détection de texte généré automatiquement.
* Visualisation de données à l'aide de Matplotlib et Streamlit.
* Analyse statistique de textes.

Dépendances
-----------

* NLTK
* OpenAI
* NumPy
* Pandas
* Matplotlib
* Streamlit

Installation
------------

1. Clonez ce dépôt.
2. Installez les dépendances en utilisant `pip install -r requirements.txt`.
3. Ajoutez votre clé API OpenAI à l'environnement ou remplacez-la directement dans le script.

Utilisation
-----------

Importez les fonctions souhaitées depuis le script et utilisez-les selon vos besoins. Assurez-vous d'avoir installé toutes les dépendances requises avant d'exécuter le script.

Contribution
------------

Les contributions sont les bienvenues ! Si vous rencontrez des problèmes ou si vous souhaitez ajouter de nouvelles fonctionnalités, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.

License
-------

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.

Ce script contient plusieurs fonctions pour l'analyse de texte en français. Voici un bref aperçu des fonctionnalités :

1. `calculate_stats` : calcule la richesse lexicale et la richesse générale d'un ou plusieurs textes et affiche un graphique en utilisant Streamlit.
2. `kl_div_with_exponential_transform` : calcule la divergence KL entre deux matrices avec une transformation exponentielle.
3. `shifted_exp` : fonction d'aide pour la transformation exponentielle.
4. `scaled_manhattan_distance` : calcule la distance de Manhattan entre deux matrices en utilisant un facteur d'échelle.
5. `nettoyer_texte` : nettoie un texte en supprimant les chiffres, les caractères spéciaux, les mots vides et en convertissant le texte en minuscules.
6. `create_markdown_table` : crée un tableau Markdown à partir d'un dictionnaire de mesures de similarité.
7. `generation2` et `generation` : génèrent du texte en utilisant l'API OpenAI.
8. `grammatical_richness` : calcule la richesse grammaticale d'un texte.
9. `verbal_richness` : calcule la richesse verbale d'un texte.
10. `lexical_field` : calcule le champ lexical d'un texte.
11. `lexical_richness` : calcule la richesse lexicale d'un texte.
12. `highlight_text2` et `highlight_text` : mettent en surbrillance des mots dans un texte en utilisant des balises HTML.
13. `lexical_richness_normalized` et `lexical_richness_normalized_ttr` : calculent la richesse lexicale normalisée d'un texte.
14. `lexical_richness_normalized_only` : calcule la richesse lexicale normalisée d'un texte en utilisant uniquement la densité lexicale, la densité grammaticale et la densité verbale.
15. `count_words` : compte le nombre de mots dans un texte.
16. `compute_text_metrics` : calcule les métriques d'un texte en utilisant la richesse lexicale, grammaticale et verbale.
17. `measure_lexical_richness` : calcule la richesse lexicale d'un texte en utilisant la ponctuation spécifiée.
18. `plot_text_relations_2d` : affiche un graphique 2D des relations entre les textes en utilisant Streamlit.
19. `detect_generated_text` : détecte si un texte a été généré automatiquement en utilisant des métriques de richesse lexicale.
20. `plot_texts_3d` : affiche un graphique 3D des relations entre les textes en utilisant Streamlit.

Ce script nécessite les bibliothèques suivantes : nltk, PyPDF2, openai, numpy, pandas, matplotlib, streamlit.
