# Dag Van De Wetenschap 2023

At the [Dag Van De Wetenschap 2023](https://www.dagvandewetenschap.be/activiteiten/universiteit-gent-kunnen-computers-gebarentaal-begrijpen-op-locatie), [IDLab-AIRO](https://airo.ugent.be/) demonstrated a prototype of a search tool for the [VGT-NL](https://woordenboek.vlaamsegebarentaal.be/) dictionary.

This prototype is based on [this paper](https://users.ugent.be/~mcdcoste/assets/2023095125.pdf), with an updated sign language recognition model that has better performance.

In this repository, you will find:

- Code to train the sign language recognition model.
- A link to the model checkpoint.
- Code to create a vector database from the dictionary.
- Code of the prototype, including frontend and backend. This includes a data collection pipeline.
- Code to evaluate the search system with a set of query videos.

## LICENCE

The code in this repository is licenced under the MIT licence.

## Citation

If you find this code useful, please consider citing our work:

```
@inproceedings{de2023querying,
  title={Querying a sign language dictionary with videos using dense vector search},
  author={De Coster, Mathieu and Dambre, Joni},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
