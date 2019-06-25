# TODO: document+structure


import util
import datetime
import pandas as pd
import re
from collections import defaultdict
import operator
import impl.util.nlp as nlp_util
import impl.category.store as cat_store
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.util.rdf as rdf_util
from nltk.metrics.distance import edit_distance


def run_extraction():
    util.get_logger().debug('Step 1: Pattern Extraction')
    patterns = _extract_patterns()

    util.get_logger().debug('Step 2: Axiom Extraction')
    relation_axioms = _extract_axioms(patterns)

    util.get_logger().debug('Step 3: Assertion Extraction')
    relation_assertions = _extract_assertions(relation_axioms)

    pd.DataFrame(data=relation_axioms, columns=['cat', 'pred', 'val']).to_csv(util.get_results_file('results.catriple.relation_axioms'), sep=';', index=False)

    df_relation_assertions = pd.DataFrame(data=relation_assertions, columns=['sub', 'pred', 'val'])
    df_relation_assertions.to_csv(util.get_results_file('results.catriple.relation_assertions'), sep=';', index=False)
    rdf_util.write_triple_file(df_relation_assertions, util.get_results_file('results.catriple.relation_assertion_triples'))


# --- PATTERN EXTRACTION ---

def _extract_patterns():
    patterns = {}
    for cat in cat_store.get_usable_cats():
        X_and_Z = _find_X_and_Z(cat)
        if X_and_Z:
            X, Z = X_and_Z
            subcats = [cat for cat in cat_store.get_children(cat) if cat_store.is_usable(cat)]
            for subcat in subcats:
                Y = _find_Y(X, subcat)
                if Y:
                    if cat in patterns:
                        patterns[cat][2][subcat] = Y.text
                    else:
                        patterns[cat] = (X.text, Z, {subcat: Y.text})
    return patterns


def _find_X_and_Z(cat_uri):
    cat = nlp_util.parse(cat_store.get_label(cat_uri))
    by_indices = [w.i for w in cat if w.text == 'by']
    if len(by_indices) > 1:
        return None
    elif len(by_indices) == 1:  # "X by Z"
        by_index = by_indices[0]
        X = cat[:by_index]
        Z_span = cat[by_index+1:]
        Z_chunks = list(Z_span.noun_chunks)
        Z = Z_chunks[0].root.lemma_ if Z_chunks else Z_span[0].lemma_
    else:  # "X"
        X = cat
        Z = None
    return X, Z


def _find_Y(X, subcat_uri):
    if X.text.lower() not in cat_store.get_label(subcat_uri).lower():
        return None
    subcat = nlp_util.parse(cat_store.get_label(subcat_uri))
    if subcat.text.lower().endswith(' ' + X.text.lower()):  # "YX"
        if subcat[-(len(X)+1)].pos_ == 'ADP':
            return None
        return subcat[:-len(X)]
    elif subcat.text.lower().startswith(X.text.lower() + ' '):  # "X <prep> Y"
        adp_indices = [w.i for w in subcat if w.pos_ == 'ADP']
        if len(adp_indices) != 1:
            return None
        adp_index = adp_indices[0]
        Y = subcat[adp_index + 1:]
        if subcat[adp_index].text == 'by':
            childcats = cat_store.get_children(subcat_uri)
            resources = cat_store.get_resources(subcat_uri)
            predicate_labels = {dbp_store.get_label(pred) for res in resources for pred in dbp_store.get_properties(res)}
            if len(childcats) * 10 >= len(resources) or any(Y.text.lower() in p for p in predicate_labels):
                return None
        return Y
    return None


# --- AXIOM EXTRACTION ---

def _extract_axioms(patterns):
    axioms = {}

    for cat, (sub, pred, subcats) in patterns.items():
        if pred:  # simple mapping of label to predicate (P1+2)
            if pred.lower() in predicate_names:
                axioms[cat] = (sub, predicate_names[pred.lower()], subcats)
        else:  # Voting required to discover Z (P3+4)
            predicate_counts = defaultdict(lambda: 0)
            for subcat, value in subcats.items():
                value = normalize_val(value)
                for res in cat_store.get_resources(subcat):
                    for pred, values in dbp_store.get_properties(res).items():
                        normalized_values = {normalize_val(val) for val in values}
                        if value in normalized_values:
                            predicate_counts[pred] += 1
            if predicate_counts:
                pred = max(predicate_counts.items(), key=operator.itemgetter(1))[0]
                axioms[cat] = (sub, pred, subcats)

    # map values to dbpedia resources if necessary (only possible if we have an object property)
    valid_axioms = {}

    for cat in axioms:
        _, pred, subcats = axioms[cat]
        if dbp_store.is_object_property(pred):
            for subcat, obj in subcats.items():
                obj_uri = dbp_util.name2resource(obj)
                if obj_uri in dbp_store.get_resources():
                    if cat in valid_axioms:
                        valid_axioms[cat][1][subcat] = obj_uri
                    else:
                        valid_axioms[cat] = (pred, {subcat: obj_uri})
        else:
            valid_axioms[cat] = (pred, subcats)

    return {(cat, pred, val) for pred, cat_vals in valid_axioms.values() for cat, val in cat_vals.items()}


predicate_names = {dbp_util.type2name(pred).lower(): pred for pred in dbp_store.get_all_predicates()}
normalizer_regex = re.compile(r'[^a-zA-Z0-9]')
def normalize_val(val: str) -> str:
    if val.startswith('http://dbpedia.org/'):
        val = dbp_util.object2name(val)
    return normalizer_regex.sub('', val).lower()


# --- ASSERTION EXTRACTION ---

def _extract_assertions(axioms):
    assertions = []

    for cat, pred, value in axioms:
        new_val = normalize_val(value)
        for res in cat_store.get_resources(cat):
            res_props = dbp_store.get_properties(res)

            if pred in res_props:
                existing_values = {normalize_val(val) for val in res_props[pred]}
                if any((new_val in ex_val) or (ex_val in new_val) for ex_val in existing_values):
                    continue

                if any(edit_distance(new_val, ex_val) < 3 for ex_val in existing_values):
                    continue

                if existing_values.intersection(nlp_util.get_synonyms(new_val)):
                    continue

                new_val_words = normalize_to_words(new_val)
                if any(new_val_words.intersection(normalize_to_words(ex_val)) for ex_val in existing_values):
                    continue

            assertions.append((res, pred, value))
    return assertions


def normalize_to_words(val: str) -> set:
    if dbp_util.is_dbp_type(val):
        val = dbp_util.type2name(val)
    return {normalizer_regex.sub('', word).lower() for word in val.split()}


# --- START SCRIPT ---

if __name__ == '__main__':
    now = datetime.datetime.now()
    util.get_logger().info('Started Catriple extraction.')

    run_extraction()

    duration = (datetime.datetime.now() - now).seconds // 60
    util.get_logger().info(f'Finished Catriple extraction after {duration} minutes.')
