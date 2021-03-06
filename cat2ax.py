import datetime
import util
from collections import defaultdict
import operator
import numpy as np
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.category.store as cat_store
import impl.category.category_set as cat_set
import impl.category.nlp as cat_nlp
import impl.util.nlp as nlp_util
import impl.util.rdf as rdf_util
import pandas as pd
from spacy.tokens import Doc


def run_extraction():
    """Run the Cat2Ax extraction procedure and create result files for relation/type axioms and assertions.

    The extraction is performed in four steps:
    1) Identify candidate category sets that share a textual pattern
    2) Find characteristic properties and types for candidate sets and combine them to patterns
    3) Apply patterns to all categories to extract axioms
    4) Apply axioms to their respective categories to extract assertions
    """
    util.get_logger().debug('Step 1: Candidate Selection')
    candidate_sets = cat_set.get_category_sets()

    util.get_logger().debug('Step 2: Pattern Mining')
    patterns = _extract_patterns(candidate_sets)

    util.get_logger().debug('Step 3: Pattern Application')
    relation_axioms, type_axioms = _extract_axioms(patterns)

    util.get_logger().debug('Step 4: Axiom Application & Post-Filtering')
    relation_assertions, type_assertions = _extract_assertions(relation_axioms, type_axioms)

    util.get_logger().debug('Finished extraction - persisting results..')
    pd.DataFrame(data=relation_axioms, columns=['cat', 'pred', 'val', 'confidence']).to_csv(util.get_results_file('results.cat2ax.relation_axioms'), sep=';', index=False)
    pd.DataFrame(data=type_axioms, columns=['cat', 'pred', 'val', 'confidence']).to_csv(util.get_results_file('results.cat2ax.type_axioms'), sep=';', index=False)

    df_relation_assertions = pd.DataFrame(data=relation_assertions, columns=['sub', 'pred', 'val'])
    df_relation_assertions.to_csv(util.get_results_file('results.cat2ax.relation_assertions'), sep=';', index=False)
    rdf_util.write_triple_file(df_relation_assertions, util.get_results_file('results.cat2ax.relation_assertion_triples'))

    df_type_assertions = pd.DataFrame(data=type_assertions, columns=['sub', 'pred', 'val'])
    df_type_assertions.to_csv(util.get_results_file('results.cat2ax.type_assertions'), sep=';', index=False)
    rdf_util.write_triple_file(df_type_assertions, util.get_results_file('results.cat2ax.type_assertion_triples'))


# --- PATTERN MINING ---

def _extract_patterns(candidate_sets: list) -> dict:
    """Return extracted property/type patterns for each set in `candidate_sets`."""
    patterns = defaultdict(lambda: {'preds': defaultdict(list), 'types': defaultdict(list)})

    for parent, categories, (first_words, last_words) in candidate_sets:
        property_scores = defaultdict(list)
        type_scores = defaultdict(list)
        type_lexicalisation_scores = _get_type_lexicalisation_scores(first_words + last_words)

        categories_with_matches = {cat: _get_match_for_category(cat, first_words, last_words) for cat in categories}
        categories_with_matches = {cat: match for cat, match in categories_with_matches.items() if cat_store.is_usable(cat) and match}
        for cat, match in categories_with_matches.items():
            statistics = cat_store.get_statistics(cat)
            resource_lexicalisation_scores = _get_resource_lexicalisation_scores(match)
            # compute property scores
            for (pred, val), prop_freq in statistics['property_frequencies'].items():
                if val in resource_lexicalisation_scores:
                    property_scores[pred].append(prop_freq * resource_lexicalisation_scores[val])
            # compute type scores
            for t, type_freq in statistics['type_frequencies'].items():
                type_scores[t].append(type_freq * type_lexicalisation_scores[t])
        if property_scores:
            # pad frequencies to get the correct median
            property_scores = {pred: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for pred, freqs in property_scores.items()}
            pred, freqs = max(property_scores.items(), key=lambda x: np.median(x[1]))
            # extract pattern with best median
            med = np.median(freqs)
            if dbp_util.is_dbp_type(pred) and med > 0:
                for _ in categories_with_matches:
                    patterns[(tuple(first_words), tuple(last_words))]['preds'][pred].append(med)
        if type_scores:
            # pad frequencies to get the correct median
            type_scores = {t: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for t, freqs in type_scores.items()}
            # extract pattern with best median
            max_median = max(np.median(freqs) for freqs in type_scores.values())
            types = {t for t, freqs in type_scores.items() if np.median(freqs) >= max_median}
            if max_median > 0:
                for _ in categories_with_matches:
                    for t in types:
                        patterns[(tuple(first_words), tuple(last_words))]['types'][t].append(max_median)
    return patterns


def _get_match_for_category(category: str, first_words: tuple, last_words: tuple) -> str:
    """Return variable part of the category name."""
    doc = cat_set._remove_by_phrase(cat_nlp.parse_category(category))
    return doc[len(first_words):len(doc)-len(last_words)].text


def _get_resource_lexicalisation_scores(text: str) -> dict:
    """Return resource lexicalisation scores for a given `text`."""
    lexicalisation_scores = {}
    if not text:
        return lexicalisation_scores
    lexicalisation_scores[text] = 1
    direct_match = dbp_store.resolve_redirect(dbp_util.name2resource(text))
    if direct_match in dbp_store.get_resources():
        lexicalisation_scores[direct_match] = 1
    for lex_match, frequency in sorted(dbp_store.get_inverse_lexicalisations(text.lower()).items(), key=operator.itemgetter(1)):
        lexicalisation_scores[lex_match] = frequency
    return lexicalisation_scores


def _get_type_lexicalisation_scores(words: list) -> dict:
    """Return type lexicalisation scores for a given set of `words`."""
    lexicalisation_scores = defaultdict(lambda: 0)
    for lemma in [nlp_util.parse(w)[0].lemma_ for w in words]:
        for t, score in dbp_store.get_type_lexicalisations(lemma).items():
            lexicalisation_scores[t] += score
    total_scores = sum(lexicalisation_scores.values())
    return defaultdict(lambda: 0, {t: score / total_scores for t, score in lexicalisation_scores.items()})


# --- PATTERN APPLICATION ---

def _extract_axioms(patterns: dict) -> tuple:
    """Return axioms extracted by applying `patterns` to all categories."""
    relation_axioms = set()
    type_axioms = set()

    # process front/back/front+back patterns individually to reduce computational complexity
    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        _fill_dict(front_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        _fill_dict(back_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        _fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    for cat in cat_store.get_usable_cats():
        cat_doc = cat_set._remove_by_phrase(cat_nlp.parse_category(cat))
        cat_prop_axioms = []
        cat_type_axioms = []

        front_prop_axiom, front_type_axiom = _find_axioms(front_pattern_dict, cat, cat_doc)
        if front_prop_axiom:
            cat_prop_axioms.append(front_prop_axiom)
        if front_type_axiom:
            cat_type_axioms.append(front_type_axiom)

        back_prop_axiom, back_type_axiom = _find_axioms(back_pattern_dict, cat, cat_doc)
        if back_prop_axiom:
            cat_prop_axioms.append(back_prop_axiom)
        if back_type_axiom:
            cat_type_axioms.append(back_type_axiom)

        enclosing_prop_axiom, enclosing_type_axiom = _find_axioms(enclosing_pattern_dict, cat, cat_doc)
        if enclosing_prop_axiom:
            cat_prop_axioms.append(enclosing_prop_axiom)
        if enclosing_type_axiom:
            cat_type_axioms.append(enclosing_type_axiom)

        # apply selection of axioms (multiple axioms can be selected if they do not contradict each other)
        prop_axioms_by_pred = {a[1]: {x for x in cat_prop_axioms if x[1] == a[1]} for a in cat_prop_axioms}
        for pred, similar_prop_axioms in prop_axioms_by_pred.items():
            if dbp_store.is_object_property(pred):
                res_labels = {a[2]: dbp_store.get_label(a[2]) for a in similar_prop_axioms}
                similar_prop_axioms = {a for a in similar_prop_axioms if all(res_labels[a[2]] == val or res_labels[a[2]] not in val for val in res_labels.values())}
            best_prop_axiom = max(similar_prop_axioms, key=operator.itemgetter(3))
            relation_axioms.add(best_prop_axiom)

        best_type_axiom = None
        for type_axiom in sorted(cat_type_axioms, key=operator.itemgetter(3), reverse=True):
            if not best_type_axiom or type_axiom[2] in dbp_store.get_transitive_subtypes(best_type_axiom[2]):
                best_type_axiom = type_axiom
        if best_type_axiom:
            type_axioms.add(best_type_axiom)

    return relation_axioms, type_axioms


def _get_confidence_pattern_set(pattern_set: dict, has_front: bool, has_back: bool) -> dict:
    """Return pattern confidences per pattern."""
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds, types = axiom_patterns['preds'], axiom_patterns['types']
            preds_sum = sum(len(freqs) for freqs in preds.values())
            types_sum = sum(len(freqs) for freqs in types.values())
            result[pattern] = {
                'preds': defaultdict(lambda: 0, {p: len(freqs) / preds_sum for p, freqs in preds.items()}),
                'types': defaultdict(lambda: 0, {t: len(freqs) / types_sum for t, freqs in types.items()})
            }
    return result


MARKER_HIT = '_marker_hit_'
MARKER_REVERSE = '_marker_reverse_'
def _fill_dict(dictionary: dict, elements: list, leaf):
    """Recursively fill a dictionary with a given sequence of elements and finally apply/append `leaf`."""
    if not elements:
        if callable(leaf):
            if MARKER_REVERSE not in dictionary:
                dictionary[MARKER_REVERSE] = {}
            leaf(dictionary[MARKER_REVERSE])
        else:
            dictionary[MARKER_HIT] = leaf
    else:
        if elements[0] not in dictionary:
            dictionary[elements[0]] = {}
        _fill_dict(dictionary[elements[0]], elements[1:], leaf)


def _detect_pattern(pattern_dict: dict, words: list) -> tuple:
    """Search for a pattern of `words` in `pattern_dict` and return if found - else return None."""
    pattern_length = 0
    ctx = pattern_dict
    for word in words:
        if word in ctx:
            ctx = ctx[word]
            pattern_length += 1
            continue
        if MARKER_HIT in ctx:
            return ctx[MARKER_HIT], pattern_length
        if MARKER_REVERSE in ctx:
            preds, back_pattern_length = _detect_pattern(ctx[MARKER_REVERSE], list(reversed(words)))
            return preds, (pattern_length, back_pattern_length)
        return None, None
    return None, None


def _get_axioms_for_cat(axiom_patterns: dict, cat: str, text_diff: str, words_same: list) -> tuple:
    """Return axioms by applying the best matching pattern to a category."""
    pattern_confidence = util.get_config('cat2ax.pattern_confidence')
    prop_axiom = None
    type_axiom = None

    statistics = cat_store.get_statistics(cat)
    pred_patterns = axiom_patterns['preds']
    resource_lexicalisation_scores = _get_resource_lexicalisation_scores(text_diff)
    props_scores = {(p, v): prop_freq * pred_patterns[p] * resource_lexicalisation_scores[v] for (p, v), prop_freq in statistics['property_frequencies'].items() if p in pred_patterns and v in resource_lexicalisation_scores}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= pattern_confidence:
        pred, val = prop
        prop_axiom = (cat, pred, val, max_prop_score)

    type_patterns = axiom_patterns['types']
    type_lexicalisation_scores = _get_type_lexicalisation_scores(words_same)
    types_scores = {t: type_freq * type_patterns[t] * type_lexicalisation_scores[t] for t, type_freq in statistics['type_frequencies'].items() if t in type_patterns}
    t, max_type_score = max(types_scores.items(), key=operator.itemgetter(1), default=(None, 0))
    if max_type_score >= pattern_confidence:
        type_axiom = (cat, rdf_util.PREDICATE_TYPE, t, max_type_score)

    return prop_axiom, type_axiom


def _find_axioms(pattern_dict: dict, cat: str, cat_doc: Doc) -> tuple:
    """Iterate over possible patterns to extract and return best axioms."""
    cat_words = [w.text for w in cat_doc]
    axiom_patterns, pattern_lengths = _detect_pattern(pattern_dict, cat_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = cat_doc[front_pattern_idx:back_pattern_idx].text
        words_same = []
        if front_pattern_idx:
            words_same += cat_words[:front_pattern_idx]
        if back_pattern_idx:
            words_same += cat_words[back_pattern_idx:]
        return _get_axioms_for_cat(axiom_patterns, cat, text_diff, words_same)
    return None, None


# --- AXIOM APPLICATION & POST-FILTERING ---

def _extract_assertions(relation_axioms: set, type_axioms: set) -> tuple:
    """Return assertions by applying the axioms to their respective categories."""
    relation_assertions = {(res, pred, val) for cat, pred, val, _ in relation_axioms for res in cat_store.get_resources(cat)}
    new_relation_assertions = {(res, pred, val) for res, pred, val in relation_assertions if pred not in dbp_store.get_properties(res) or val not in dbp_store.get_properties(res)[pred]}

    type_assertions = {(res, rdf_util.PREDICATE_TYPE, t) for cat, pred, t, _ in type_axioms for res in cat_store.get_resources(cat)}
    new_type_assertions = {(res, pred, t) for res, pred, t in type_assertions if t not in {tt for t in dbp_store.get_types(res) for tt in dbp_store.get_transitive_supertype_closure(t)} and t != rdf_util.CLASS_OWL_THING}
    new_type_assertions_transitive = {(res, pred, tt) for res, pred, t in new_type_assertions for tt in dbp_store.get_transitive_supertype_closure(t) if tt not in {ott for t in dbp_store.get_types(res) for ott in dbp_store.get_transitive_supertype_closure(t)} and tt != rdf_util.CLASS_OWL_THING}

    # post-filtering
    filtered_new_relation_assertions = {(res, pred, val) for res, pred, val in new_relation_assertions if pred not in dbp_store.get_properties(res) or not dbp_store.is_functional(pred)}
    filtered_new_type_assertions = {(res, pred, t) for res, pred, t in new_type_assertions_transitive if not dbp_store.get_disjoint_types(t).intersection(dbp_store.get_transitive_types(res))}

    return filtered_new_relation_assertions, filtered_new_type_assertions


# --- START SCRIPT ---

if __name__ == '__main__':
    now = datetime.datetime.now()
    util.get_logger().info('Started Cat2Ax extraction.')

    run_extraction()

    duration = (datetime.datetime.now() - now).seconds // 60
    util.get_logger().info(f'Finished Cat2Ax extraction after {duration} minutes.')
