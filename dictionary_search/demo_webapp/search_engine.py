import dataclasses
import glob
import os
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


@dataclasses.dataclass
class DatabaseEntry:
    embedding: np.ndarray
    gloss: str

# We do not want certain glosses to be included, because they are place names or may otherwise be unsuitable for
# this demo.
UNWANTED_GLOSSES = [
    'KOERSEL(LIM)-B-16630',
    'BLANKENBERGE(WVL)-A-1576',
    'HEULE(WVL)-A-4938',
    'PROSTITUEE-C-16287',
    'MELSBROEK(VLB)-A-7428',
    'DOORNIK(BEL)-A-3102',
    'MARSEILLE(FRA)-C-15475',
    'NIEUWPOORT(WVL)-B-19123',
    'HOUTHALEN(LIM)-C-5137',
    'KOKSIJDE(WVL)-A-6193'
]


class SearchEngine:
    """Allows searching the database."""

    def __init__(self, database_path: str, max_entries: int):
        self.database = []

        # We first add the signs that are in the query set and then continue to add unique signs until we reach max_entries.
        # We do this in a deterministic manner of course...
        db_entries = [os.path.join(database_path, 'HEBBEN-A-4801.npy'),  # In corpus.
                      os.path.join(database_path, 'TELEFONEREN-D-11870.npy'),  # In corpus.
                      os.path.join(database_path, 'HAAS-A-16146.npy'),  # In corpus.
                      os.path.join(database_path, 'STRAAT-A-11560.npy'),  # In corpus.
                      os.path.join(database_path, 'PAARD-A-8880.npy'),  # In corpus.
                      os.path.join(database_path, 'HOND-A-5052.npy'),  # In corpus.
                      os.path.join(database_path, 'RUSTEN-B-10250.npy'),  # In corpus.
                      os.path.join(database_path, 'SCHOOL-A-10547.npy'),  # In corpus.
                      os.path.join(database_path, 'ONTHOUDEN-A-8420.npy'),  # In corpus.
                      os.path.join(database_path, 'WAT-A-13657.npy'),  # In corpus.

                      os.path.join(database_path, 'BOUWEN-G-1906.npy'),  # Not in corpus.
                      os.path.join(database_path, 'WAAROM-A-13564.npy'),  # Not in corpus.
                      os.path.join(database_path, 'MELK-B-7418.npy'),  # Not in corpus.
                      os.path.join(database_path, 'VALENTIJN-A-16235.npy'),  # Not in corpus.
                      os.path.join(database_path, 'HERFST-B-4897.npy'),  # Not in corpus.
                      os.path.join(database_path, 'VLIEGTUIG-B-13187.npy'),  # Not in corpus.
                      os.path.join(database_path, 'KLEPELBEL-A-1166.npy'),  # Not in corpus.
                      os.path.join(database_path, 'POES-G-9372.npy'),  # Not in corpus.
                      os.path.join(database_path, 'MOEDER-A-7676.npy'),  # Not in corpus.
                      os.path.join(database_path, 'VADER-G-8975.npy')]  # Not in corpus.

        all_db_entries: List[str] = sorted(glob.glob(os.path.join(database_path, '*.npy')))
        random.seed(42)
        random.shuffle(all_db_entries)
        index: int = 0
        while max_entries == -1 or len(db_entries) < max_entries:
            if all_db_entries[index] not in db_entries:  # I know, not ideal, but it'll have to do.
                # Additional check: for this demo we do not want to include numbers and some other glosses.
                is_number = os.path.basename(all_db_entries[index]).split('-')[0].isnumeric()
                otherwise_unwanted = os.path.basename(all_db_entries[index]).split('.')[0] in UNWANTED_GLOSSES
                if not is_number and not otherwise_unwanted:
                    print(all_db_entries[index])
                    db_entries.append(all_db_entries[index])
            index += 1
        print(f'Collected {len(db_entries)} database entries.')
        for db_entry in db_entries:
            label: str = os.path.splitext(os.path.basename(db_entry))[0]
            self.database.append(DatabaseEntry(np.load(db_entry), label))

    def get_results(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Perform a search. Return a maximum number of results.

        :param embedding: The query embedding.
        :returns: The search results, a list of glosses and their distance to the query."""
        distances = []
        for i, key in enumerate(self.database):
            eucdist = np.linalg.norm(embedding - key.embedding)

            distances.append((eucdist, i))
        ordered = sorted(distances, key=lambda tup: tup[0])  # Sort by ascending distance.
        results = [(self.database[i].gloss, float(d)) for d, i in ordered]
        return results
