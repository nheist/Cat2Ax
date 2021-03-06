from . import util as cat_util
import impl.util.rdf as rdf_util
import impl.dbpedia.store as dbp_store
import util
from collections import defaultdict


def get_categories() -> set:
    global __CATEGORIES__
    if '__CATEGORIES__' not in globals():
        initializer = lambda: set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_TYPE))
        __CATEGORIES__ = util.load_or_create_cache('dbpedia_categories', initializer)

    return __CATEGORIES__


def get_usable_cats() -> set:
    """Return only usable categories (cf. 'is_usable')."""
    return {cat for cat in get_categories() if is_usable(cat)}


def is_usable(category: str) -> bool:
    """Return categories that are no maintenance or organisational ones (using indicators in the category name)."""
    indicators = ['wikipedia', 'wikiproject', 'lists', 'redirects', 'mediawiki', 'template', 'user', 'portal', 'categories', 'articles', 'pages', 'navigational', 'stubs']
    return category not in get_maintenance_categories() and all(indicator not in category.lower() for indicator in indicators)


def get_label(category: str) -> str:
    global __CATEGORY_LABELS__
    if '__CATEGORY_LABELS__' not in globals():
        __CATEGORY_LABELS__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_SKOS_LABEL)

    return __CATEGORY_LABELS__[category] if category in __CATEGORY_LABELS__ else cat_util.category2name(category)


def get_resources(category: str) -> set:
    global __CATEGORY_RESOURCES__
    if '__CATEGORY_RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        __CATEGORY_RESOURCES__ = util.load_or_create_cache('dbpedia_category_resources', initializer)

    return __CATEGORY_RESOURCES__[category]


def get_children(category: str) -> set:
    global __CHILDREN__
    if '__CHILDREN__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_BROADER, reverse_key=True)
        __CHILDREN__ = util.load_or_create_cache('dbpedia_category_children', initializer)

    return __CHILDREN__[category].difference({category})


def get_maintenance_categories() -> set:
    global __MAINTENANCE_CATS__
    if '__MAINTENANCE_CATS__' not in globals():
        __MAINTENANCE_CATS__ = set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.maintenance_categories')], rdf_util.PREDICATE_TYPE))

    return __MAINTENANCE_CATS__


def get_statistics(category: str) -> dict:
    """Return information about the amounts/frequencies of types and properties of a category's resources."""
    global __CATEGORY_STATISTICS__
    if '__CATEGORY_STATISTICS__' not in globals():
        __CATEGORY_STATISTICS__ = util.load_or_create_cache('dbpedia_category_statistics', _compute_category_statistics)
    return __CATEGORY_STATISTICS__[category]


def _compute_category_statistics() -> dict:
    category_statistics = {}
    for cat in get_categories():
        type_counts = defaultdict(int)
        property_counts = defaultdict(int)

        resources = get_resources(cat)
        for res in resources:
            resource_statistics = dbp_store.get_statistics(res)
            for t in resource_statistics['types']:
                type_counts[t] += 1
            for prop in resource_statistics['properties']:
                property_counts[prop] += 1
        category_statistics[cat] = {
            'type_counts': type_counts,
            'type_frequencies': defaultdict(float, {t: t_count / len(resources) for t, t_count in type_counts.items()}),
            'property_counts': property_counts,
            'property_frequencies': defaultdict(float, {prop: p_count / len(resources) for prop, p_count in property_counts.items()}),
        }
    return category_statistics
