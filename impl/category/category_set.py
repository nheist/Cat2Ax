import util
from collections import namedtuple, defaultdict
import impl.category.store as cat_store
import impl.category.nlp as cat_nlp
from spacy.tokens import Doc
import operator
from typing import Tuple, Optional


CategorySet = namedtuple('CategorySet', ['parent', 'categories', 'pattern'])


def get_category_sets() -> list:
    """Return a list of category sets found in DBpedia."""
    return [cs for category_sets in _get_parent_to_category_set_mapping().values() for cs in category_sets]


def _get_parent_to_category_set_mapping() -> dict:
    global __CATEGORY_SETS__
    if '__CATEGORY_SETS__' not in globals():
        __CATEGORY_SETS__ = util.load_or_create_cache('dbpedia_category_sets', _compute_category_sets)

    return __CATEGORY_SETS__


def _compute_category_sets() -> dict:
    """Iterate over DBpedia categories and identify all category sets.

    1) Retrieve all usable categories (i.e. categories that are not used for maintenance/organisational purposes)
    2) Normalize their names by removing by-phrases (e.g. "X by genre", "Y by country")
    3) For each category, retrieve all its children and search for name patterns (see '_find_child_sets')
    """
    category_sets = {}
    for cat in cat_store.get_usable_cats():
        children = {c for c in cat_store.get_children(cat) if cat_store.is_usable(c)}
        children_docs = {c: _remove_by_phrase(cat_nlp.parse_category(c)) for c in children}
        child_sets = _find_child_sets(cat, children_docs)
        if child_sets:
            category_sets[cat] = child_sets
    return category_sets


def _remove_by_phrase(doc: Doc) -> Doc:
    by_indices = [w.i for w in doc if w.text == 'by']
    if len(by_indices) == 0:
        return doc
    last_by_index = by_indices[-1]
    if last_by_index == 0 or last_by_index == len(doc) - 1:
        return doc
    word_after_by = doc[last_by_index+1]
    if word_after_by.text.istitle() or word_after_by.text == 'the' or word_after_by.tag_ == 'NNS':
        return doc
    return doc[:last_by_index].as_doc()


def _find_child_sets(parent: str, category_docs: dict, current_pattern=((), ())) -> list:
    """Identify sets of child categories by recursively checking whether a pattern is shared by multiple categories."""
    if len(category_docs) < 2:
        return []

    front_grp, front_word = _find_best_group(category_docs, len(current_pattern[0]))
    back_grp, back_word = _find_best_group(category_docs, -len(current_pattern[1]) - 1)

    if len(front_grp) >= len(back_grp):
        grp = front_grp
        new_pattern = (current_pattern[0] + (front_word,), current_pattern[1])
    else:
        grp = back_grp
        new_pattern = (current_pattern[0], (back_word,) + current_pattern[1])

    # stop pattern search if categories are divided into too many groups
    count = len(grp)
    score = count / len(category_docs)
    if count < 2 or score < .5:
        if current_pattern[0] or current_pattern[1]:
            return [CategorySet(parent=parent, categories=set(category_docs), pattern=current_pattern)]
        else:
            return []

    # continue pattern search if division of categories was helpful
    grouped_docs = {c: doc for c, doc in category_docs.items() if c in grp}
    ungrouped_docs = {c: doc for c, doc in category_docs.items() if c not in grp}
    return _find_child_sets(parent, grouped_docs, new_pattern) + _find_child_sets(parent, ungrouped_docs, current_pattern)


def _find_best_group(category_docs: dict, idx: int) -> Tuple[set, Optional[str]]:
    """Locate the best group of categories by checking which words appear most frequently at the current index."""
    word_counts = defaultdict(lambda: 0)
    for d in category_docs.values():
        if len(d) > idx and len(d) >= -idx:  # take positive and negative indices into accout
            word_counts[d[idx].text] += 1

    if not word_counts:
        return set(), None

    most_frequent_word = max(word_counts.items(), key=operator.itemgetter(1))[0]
    return {c for c, d in category_docs.items() if len(d) > idx and len(d) >= -idx and d[idx].text == most_frequent_word}, most_frequent_word
