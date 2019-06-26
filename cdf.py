import util
import datetime
import operator
from collections import defaultdict
import pandas as pd
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.category.store as cat_store
import impl.util.rdf as rdf_util


def run_extraction():
    """Run the C-DF extraction procedure and create result files for relation/type axioms and assertions.

    The extraction is performed in two steps:
    1) Find defining features (DFs) of a category, i.e. sets of types/relations that are frequent in the category and globally infrequent
    2) Use DFs to extract rules and subsequently apply the rules to extract axioms and assertions
    """
    util.get_logger().debug('Step 1: Defining Feature Extraction')
    cat_dfs = _extract_cat_dfs()
    direct_axioms = {(cat, pred, val) for cat, (df, _) in cat_dfs.items() for (pred, val) in df}

    util.get_logger().debug('Step 2: Rule Extraction')
    rule_axioms = _extract_axioms_with_rules(cat_dfs)
    all_axioms = rule_axioms | {(cat, pred, val) for cat, pred, val in direct_axioms if cat not in rule_axioms}
    all_assertions = {(res, pred, val) for cat, pred, val in all_axioms for res in cat_store.get_resources(cat)}

    util.get_logger().debug('Finished extraction - persisting results..')
    relation_axioms = {ax for ax in all_axioms if ax[1] != rdf_util.PREDICATE_TYPE}
    pd.DataFrame(data=relation_axioms, columns=['cat', 'pred', 'val']).to_csv(util.get_results_file('results.cdf.relation_axioms'), sep=';', index=False)

    type_axioms = {ax for ax in all_axioms if ax[1] == rdf_util.PREDICATE_TYPE}
    pd.DataFrame(data=type_axioms, columns=['cat', 'pred', 'val']).to_csv(util.get_results_file('results.cdf.type_axioms'), sep=';', index=False)

    relation_assertions = {a for a in all_assertions if a[1] != rdf_util.PREDICATE_TYPE}
    df_relation_assertions = pd.DataFrame(data=relation_assertions, columns=['sub', 'pred', 'val'])
    df_relation_assertions.to_csv(util.get_results_file('results.cdf.relation_assertions'), sep=';', index=False)
    rdf_util.write_triple_file(df_relation_assertions, util.get_results_file('results.cdf.relation_assertion_triples'))

    type_assertions = {a for a in all_assertions if a[1] == rdf_util.PREDICATE_TYPE}
    df_type_assertions = pd.DataFrame(data=type_assertions, columns=['sub', 'pred', 'val'])
    df_type_assertions.to_csv(util.get_results_file('results.cdf.type_assertions'), sep=';', index=False)
    rdf_util.write_triple_file(df_type_assertions, util.get_results_file('results.cdf.type_assertion_triples'))


# --- DEFINING FEATURE EXTRACTION ---

def _extract_cat_dfs() -> dict:
    """Return DFs of categories that are frequent in the category and infrequent globally."""
    cat_df_candidates = {}
    alpha = util.get_config('cdf.alpha')

    for cat in cat_store.get_usable_cats():
        df_candidates = {}

        if len(cat_store.get_resources(cat)) < 2:
            # discard a category if it has at most one resource (as there is not enough evidence)
            continue

        # collect base features for DF generation
        cat_stats = cat_store.get_statistics(cat)
        base_props = {prop for prop, freq in cat_stats['property_frequencies'].items() if freq >= alpha}
        base_types = {(rdf_util.PREDICATE_TYPE, t) for t, freq in cat_stats['type_frequencies'].items() if freq >= alpha}
        independent_base_types = dbp_store.get_independent_types({val[1] for val in base_types})
        base_types = {val for val in base_types if val[1] in independent_base_types}
        base_features = base_props | base_types

        if len(base_features) > 20:
            # discard a category if there are way too many base features (as computational complexity is too high)
            continue
        df_candidates.update({(prop,): (cat_stats['property_counts'][prop], cat_stats['property_frequencies'][prop]) for prop in base_props})
        df_candidates.update({(t,): (cat_stats['type_counts'][t[1]], cat_stats['type_frequencies'][t[1]]) for t in base_types})

        # iteratively look for promising DFs
        current_features = {(f,) for f in base_features}
        current_features_strings = {_get_feature_set_as_string(f_set) for f_set in current_features}
        while True:
            new_features = {}
            new_features_strings = set()
            for cf in current_features:
                for bf in base_features:
                    if bf not in cf:
                        nf = cf + (bf,)
                        nf_string = _get_feature_set_as_string(nf)
                        if nf_string not in new_features_strings:
                            if all(_get_feature_set_as_string(set(nf).difference({elem})) in current_features_strings for elem in nf):
                                nf_count = _get_overall_features_count(nf, cat=cat)
                                nf_freq = nf_count / len(cat_store.get_resources(cat))
                                if nf_freq > alpha:
                                    new_features[nf] = (nf_count, nf_freq)
                                    new_features_strings.add(nf_string)

            if not new_features:
                break
            current_features = set(new_features)
            current_features_strings = new_features_strings
            df_candidates.update(new_features)

        if df_candidates:
            cat_df_candidates[cat] = df_candidates

    # find best DFs by scoring them
    cat_df_candidate_scores = {}
    for cat, candidates in cat_df_candidates.items():
        candidate_scores = {}
        for features, (count, freq) in candidates.items():
            overall_count = _get_overall_features_count(features)
            candidate_scores[features] = freq * count / overall_count if overall_count > 0 else 0
        cat_df_candidate_scores[cat] = candidate_scores

    cat_dfs = {}
    for cat, candidate_dfs in cat_df_candidate_scores.items():
        best_df, score = max(candidate_dfs.items(), key=operator.itemgetter(1), default=(None, 0))
        if score > alpha:
            cat_dfs[cat] = (best_df, score)
    return cat_dfs


# create an index of resources and properties that converts string-uris to integers in order to speed up indexing and reduce complexity
resource_features = {res: {(k, v) for k, values in props.items() for v in values} for res, props in dbp_store.get_resource_property_mapping().items()}
for res in resource_features:
    for t in dbp_store.get_types(res):
        resource_features[res].add((rdf_util.PREDICATE_TYPE, t))

resource_to_idx_dict = {res: i for res, i in zip(resource_features, range(len(resource_features)))}

feature_to_idx_dict = defaultdict(set)
for res, feats in resource_features.items():
    res_idx = resource_to_idx_dict[res]
    for f in feats:
        feature_to_idx_dict[f].add(res_idx)


def _get_overall_features_count(feats: tuple, cat: str = None) -> int:
    """Return global count of features."""
    valid_res_idxs = set()
    if cat:
        valid_res_idxs = {resource_to_idx_dict[res] for res in cat_store.get_resources(cat) if res in resource_to_idx_dict}

    for f in feats:
        res_idxs_with_f = feature_to_idx_dict[f]
        valid_res_idxs = valid_res_idxs.intersection(res_idxs_with_f) if valid_res_idxs else res_idxs_with_f
    return len(valid_res_idxs)


def _get_feature_set_as_string(feature_set: set) -> str:
    """Return feature set as unique identifier to ease indexing."""
    return ''.join(sorted([f[0] + f[1] for f in feature_set]))


# --- RULE EXTRACTION ---

def _extract_axioms_with_rules(cat_dfs: dict) -> set:
    """Return axioms genered by applying C-DF rules."""

    # generate rule candidates by extracting shared pre-/postfixes
    cdf_rule_candidates = defaultdict(lambda: defaultdict(lambda: 0))
    for cat, (df, _) in cat_dfs.items():
        cat_label = cat_store.get_label(cat)
        for f in {f for f in df if f[0] != rdf_util.PREDICATE_TYPE}:
            if dbp_util.is_dbp_resource(f[1]):
                f_label = dbp_store._get_label_mapping()[f[1]] if f[1] in dbp_store._get_label_mapping() else dbp_util.object2name(f[1])
            else:
                f_label = f[1]
            if f_label in cat_label:
                first_words = cat_label[:cat_label.index(f_label)].strip()
                first_words = tuple(first_words.split(' ')) if first_words else tuple()
                last_words = cat_label[cat_label.index(f_label) + len(f_label):].strip()
                last_words = tuple(last_words.split(' ')) if last_words else tuple()
                if first_words or last_words:
                    f_types = dbp_store.get_independent_types(dbp_store.get_types(f[1])) if dbp_util.is_dbp_resource(f[1]) else set()
                    f_type = f_types.pop() if f_types else None
                    cdf_rule_candidates[(first_words, last_words)][((f[0], f_type), tuple(set(df).difference({f})))] += 1

    # filter rules using the threshold parameters min_support and beta
    cdf_rules = {}
    min_support = util.get_config('cdf.min_support')
    beta = util.get_config('cdf.beta')
    for word_patterns in cdf_rule_candidates:
        total_support = sum(cdf_rule_candidates[word_patterns].values())
        valid_axiom_patterns = [pattern for pattern, support in cdf_rule_candidates[word_patterns].items() if support >= min_support and (support / total_support) >= beta]

        if len(valid_axiom_patterns) > 0:
            cdf_rules[word_patterns] = valid_axiom_patterns[0]

    # apply the patterns to all categories in order to extract axioms
    # (the rules are applied individually depending on whether the pattern is at the front, back, or front+back in order to reduce computational complexity)
    cdf_front_patterns = {word_patterns: axiom_pattern for word_patterns, axiom_pattern in cdf_rules.items() if word_patterns[0] and not word_patterns[1]}
    cdf_front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in cdf_front_patterns.items():
        _fill_dict(cdf_front_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    cdf_back_patterns = {word_patterns: axiom_pattern for word_patterns, axiom_pattern in cdf_rules.items() if not word_patterns[0] and word_patterns[1]}
    cdf_back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in cdf_back_patterns.items():
        _fill_dict(cdf_back_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    cdf_enclosing_patterns = {word_patterns: axiom_pattern for word_patterns, axiom_pattern in cdf_rules.items() if word_patterns[0] and word_patterns[1]}
    cdf_enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in cdf_enclosing_patterns.items():
        _fill_dict(cdf_enclosing_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    rule_axioms = set()
    for cat in cat_store.get_usable_cats():
        rule_axioms.update(_apply_rules(cdf_front_pattern_dict, cat))
        rule_axioms.update(_apply_rules(cdf_back_pattern_dict, cat))
        rule_axioms.update(_apply_rules(cdf_enclosing_pattern_dict, cat))
    return rule_axioms


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


def _apply_rules(pattern_dict: dict, cat: str) -> set:
    """Apply rules form `pattern_dict` and return the implied axioms."""
    cat_words = cat_store.get_label(cat).split(' ')

    axiom_patterns, pattern_lengths = _detect_pattern(pattern_dict, cat_words)
    if not axiom_patterns:
        return set()

    (pred, pred_type), additional_axioms = axiom_patterns
    front_pattern_idx = pattern_lengths[0] or None
    back_pattern_idx = -1 * pattern_lengths[1] or None
    resource = ' '.join(cat_words[front_pattern_idx:back_pattern_idx])

    if pred_type:
        resource = dbp_util.name2resource(resource)
        if resource not in dbp_store.get_resources() or pred_type not in dbp_store.get_transitive_types(resource):
            return set()
    return {(cat, pred, resource)} | {(cat, pred, val) for pred, val in additional_axioms}


# --- START SCRIPT ---

if __name__ == '__main__':
    now = datetime.datetime.now()
    util.get_logger().info('Started C-DF extraction.')

    run_extraction()

    duration = (datetime.datetime.now() - now).seconds // 60
    util.get_logger().info(f'Finished C-DF extraction after {duration} minutes.')
