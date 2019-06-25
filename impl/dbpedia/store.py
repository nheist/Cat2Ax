import util
import impl.util.rdf as rdf_util
from . import util as dbp_util
from collections import defaultdict
import networkx as nx
from typing import Optional
import functools


# DBpedia resources


def get_resources() -> set:
    global __RESOURCES__
    if '__RESOURCES__' not in globals():
        __RESOURCES__ = set(_get_label_mapping()) | set(get_resource_property_mapping())
    return __RESOURCES__


def get_label(dbp_object: str) -> str:
    global __RESOURCE_LABELS__
    if '__RESOURCE_LABELS__' not in globals():
        __RESOURCE_LABELS__ = dict(_get_label_mapping())
        __RESOURCE_LABELS__.update(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_LABEL))
    return __RESOURCE_LABELS__[dbp_object] if dbp_object in __RESOURCE_LABELS__ else dbp_util.object2name(dbp_object)


def _get_label_mapping() -> dict:
    global __RESOURCE_LABEL_MAPPING__
    if '__RESOURCE_LABEL_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.labels')], rdf_util.PREDICATE_LABEL)
        __RESOURCE_LABEL_MAPPING__ = util.load_or_create_cache('dbpedia_resource_labels', initializer)

    return __RESOURCE_LABEL_MAPPING__


def get_inverse_lexicalisations(text: str) -> dict:
    global __RESOURCE_INVERSE_LEXICALISATIONS__
    if '__RESOURCE_INVERSE_LEXICALISATIONS__' not in globals():
        __RESOURCE_INVERSE_LEXICALISATIONS__ = util.load_or_create_cache('dbpedia_resource_inverse_lexicalisations', _compute_inverse_lexicalisations)
    return __RESOURCE_INVERSE_LEXICALISATIONS__[text.lower()] if text.lower() in __RESOURCE_INVERSE_LEXICALISATIONS__ else {}


def _compute_inverse_lexicalisations():
    inverse_lexicalisation_dict = rdf_util.create_multi_val_freq_dict_from_rdf([util.get_data_file('files.dbpedia.anchor_texts')], rdf_util.PREDICATE_ANCHOR_TEXT, reverse_key=True)
    for lex, resources in inverse_lexicalisation_dict.items():
        for res in set(resources.keys()):
            redirect_res = resolve_redirect(res)
            if res != redirect_res:
                if redirect_res in inverse_lexicalisation_dict[lex]:
                    inverse_lexicalisation_dict[lex][redirect_res] += inverse_lexicalisation_dict[lex][res]
                else:
                    inverse_lexicalisation_dict[lex][redirect_res] = inverse_lexicalisation_dict[lex][res]
                del inverse_lexicalisation_dict[lex][res]
    return inverse_lexicalisation_dict


def get_types(dbp_resource: str) -> set:
    return {t for t in _get_resource_type_mapping()[dbp_resource] if dbp_util.is_dbp_type(t)}


def _get_resource_type_mapping() -> dict:
    global __RESOURCE_TYPE_MAPPING__
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [
            util.get_data_file('files.dbpedia.instance_types'),
            util.get_data_file('files.dbpedia.transitive_instance_types'),
        ]
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)

    return __RESOURCE_TYPE_MAPPING__


def get_transitive_types(dbp_resource: str) -> set:
    """Return a resource's types as well as the transitive closure of these types."""
    transitive_types = {tt for t in get_types(dbp_resource) for tt in get_transitive_supertype_closure(t)}
    return {t for t in transitive_types if dbp_util.is_dbp_type(t)}


def get_properties(dbp_resource: str) -> dict:
    """Return all properties where `dbp_resource` is the subject."""
    return get_resource_property_mapping()[dbp_resource]


def resolve_redirect(dbp_resource: str, visited=None) -> str:
    """Return the resource to which `dbp_resource` redirects (if any) or `dbp_resource` itself."""
    global __REDIRECTS__
    if '__REDIRECTS__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.redirects')], rdf_util.PREDICATE_REDIRECTS)
        __REDIRECTS__ = util.load_or_create_cache('dbpedia_resource_redirects', initializer)

    if dbp_resource in __REDIRECTS__:
        visited = visited or set()
        if dbp_resource not in visited:
            return resolve_redirect(__REDIRECTS__[dbp_resource], visited | {dbp_resource})
    return dbp_resource


def get_statistics(dbp_resource: str) -> dict:
    """Return information about the types and properties of `dbp_resource`."""
    global __RESOURCE_STATISTICS__
    if '__RESOURCE_STATISTICS__' not in globals():
        __RESOURCE_STATISTICS__ = {}
    if dbp_resource not in __RESOURCE_STATISTICS__:
        __RESOURCE_STATISTICS__[dbp_resource] = {
            'types': get_transitive_types(dbp_resource),
            'properties': {(pred, val) for pred, values in get_properties(dbp_resource).items() for val in values},
        }
    return __RESOURCE_STATISTICS__[dbp_resource]


# DBpedia property


def get_resource_property_mapping() -> dict:
    global __RESOURCE_PROPERTY_MAPPING__
    if '__RESOURCE_PROPERTY_MAPPING__' not in globals():
        property_files = [util.get_data_file('files.dbpedia.mappingbased_literals'), util.get_data_file('files.dbpedia.mappingbased_objects')]
        initializer = lambda: rdf_util.create_dict_from_rdf(property_files)
        __RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_resource_properties', initializer)

    return __RESOURCE_PROPERTY_MAPPING__


def get_domain(dbp_predicate: str) -> Optional[str]:
    global __PREDICATE_DOMAIN__
    if '__PREDICATE_DOMAIN__' not in globals():
        __PREDICATE_DOMAIN__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DOMAIN)

    return __PREDICATE_DOMAIN__[dbp_predicate] if dbp_predicate in __PREDICATE_DOMAIN__ else None


def get_range(dbp_predicate: str) -> Optional[str]:
    global __PREDICATE_RANGE__
    if '__PREDICATE_RANGE__' not in globals():
        __PREDICATE_RANGE__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_RANGE)

    return __PREDICATE_RANGE__[dbp_predicate] if dbp_predicate in __PREDICATE_RANGE__ else None


def is_object_property(dbp_predicate: str) -> bool:
    global __OBJECT_PROPERTY__
    if '__OBJECT_PROPERTY__' not in globals():
        __OBJECT_PROPERTY__ = defaultdict(lambda: False)
    if dbp_predicate not in __OBJECT_PROPERTY__:
        if get_range(dbp_predicate):
            __OBJECT_PROPERTY__[dbp_predicate] = dbp_util.is_dbp_type(get_range(dbp_predicate))
        else:
            for props in get_resource_property_mapping().values():
                if dbp_predicate in props:
                    __OBJECT_PROPERTY__[dbp_predicate] = dbp_util.is_dbp_resource(props[dbp_predicate].pop())
                    break

    return __OBJECT_PROPERTY__[dbp_predicate]


def is_functional(dbp_predicate: str) -> bool:
    global __PREDICATE_FUNCTIONAL__
    if '__PREDICATE_FUNCTIONAL__' not in globals():
        __PREDICATE_FUNCTIONAL__ = util.load_or_create_cache('dbpedia_functional_predicates', _create_functional_predicate_dict)

    return __PREDICATE_FUNCTIONAL__[dbp_predicate] if dbp_predicate in __PREDICATE_FUNCTIONAL__ else False


def _create_functional_predicate_dict():
    predicate_resources_count = {pred: 0 for pred in get_all_predicates()}
    predicate_nonfunctional_count = {pred: 0 for pred in get_all_predicates()}

    resource_property_mapping = get_resource_property_mapping()
    for r in resource_property_mapping:
        for pred in resource_property_mapping[r]:
            predicate_resources_count[pred] += 1
            if len(resource_property_mapping[r][pred]) > 1:
                predicate_nonfunctional_count[pred] += 1

    return {pred: (predicate_nonfunctional_count[pred] / predicate_resources_count[pred]) < .05 for pred in get_all_predicates()}


def get_all_predicates() -> set:
    global __PREDICATES__
    if '__PREDICATES__' not in globals():
        __PREDICATES__ = {pred for props in get_resource_property_mapping().values() for pred in props}

    return __PREDICATES__


# DBpedia types

def get_all_types() -> set:
    return set(_get_type_graph().nodes)


def get_independent_types(dbp_types: set) -> set:
    """Return only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_supertypes(dbp_type: str) -> set:
    type_graph = _get_type_graph()
    return set(type_graph.predecessors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_supertypes(dbp_type: str) -> set:
    global __TRANSITIVE_SUPERTYPE_MAPPING__
    if '__TRANSITIVE_SUPERTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUPERTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = nx.ancestors(type_graph, dbp_type) if dbp_type in type_graph else set()

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_transitive_supertype_closure(dbp_type: str) -> set:
    """Return `dbp_type` itself and its transitive supertypes."""
    return {dbp_type} | get_transitive_supertypes(dbp_type)


def get_subtypes(dbp_type: str) -> set:
    type_graph = _get_type_graph()
    return set(type_graph.successors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_subtypes(dbp_type: str) -> set:
    global __TRANSITIVE_SUBTYPE_MAPPING__
    if '__TRANSITIVE_SUBTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUBTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = nx.descendants(type_graph, dbp_type) if dbp_type in type_graph else set()

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_transitive_subtype_closure(dbp_type: str) -> set:
    """Return `dbp_type` itself and its transitive subtypes."""
    return {dbp_type} | get_transitive_subtypes(dbp_type)


def get_equivalent_types(dbp_type: str) -> set:
    global __EQUIVALENT_TYPE_MAPPING__
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS, reflexive=True)

    return {dbp_type} | __EQUIVALENT_TYPE_MAPPING__[dbp_type]


REMOVED_DISJOINTNESS_AXIOMS = [{'http://dbpedia.org/ontology/Agent', 'http://dbpedia.org/ontology/Place'}]
ADDED_DISJOINTNESS_AXIOMS = [{'http://dbpedia.org/ontology/Person', 'http://dbpedia.org/ontology/Place'}, {'http://dbpedia.org/ontology/Family', 'http://dbpedia.org/ontology/Place'}]
def get_disjoint_types(dbp_type: str) -> set:
    """Return all types that are disjoint with `dbp_type` (excluding the wrong disjointness Agent<->Place)."""
    global __DISJOINT_TYPE_MAPPING__
    if '__DISJOINT_TYPE_MAPPING__' not in globals():
        __DISJOINT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DISJOINT_WITH, reflexive=True)
        # add/remove custom axioms
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {k: {v for v in values if {k, v} not in REMOVED_DISJOINTNESS_AXIOMS} for k, values in __DISJOINT_TYPE_MAPPING__.items()})
        for a, b in ADDED_DISJOINTNESS_AXIOMS:
            __DISJOINT_TYPE_MAPPING__[a].add(b)
            __DISJOINT_TYPE_MAPPING__[b].add(a)

        # completing the subtype of each type with the subtypes of its disjoint types
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {t: {st for dt in disjoint_types for st in get_transitive_subtypes(dt)} for t, disjoint_types in __DISJOINT_TYPE_MAPPING__.items()})

    return __DISJOINT_TYPE_MAPPING__[dbp_type]


def get_type_lexicalisations(lemma: str) -> dict:
    """Return the type lexicalisation score for a set of lemmas (i.e. the probabilities of types given `lemmas`)."""
    global __TYPE_LEXICALISATIONS__
    if '__TYPE_LEXICALISATIONS__' not in globals():
        __TYPE_LEXICALISATIONS__ = util.load_cache('dbpedia_type_lexicalisations')

    return __TYPE_LEXICALISATIONS__[lemma] if lemma in __TYPE_LEXICALISATIONS__ else {}


def _get_type_graph() -> nx.DiGraph:
    """Return the initialised graph of DBpedia types."""
    global __TYPE_GRAPH__
    if '__TYPE_GRAPH__' not in globals():
        subtype_mapping = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # add missing types (i.e. those, that do not have subclasses at all)
        all_types = rdf_util.create_set_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS)
        subtype_mapping.update({et: set() for t in all_types for et in get_equivalent_types(t) if et not in subtype_mapping})
        # completing subtypes with subtypes of equivalent types
        subtype_mapping = {t: {est for et in get_equivalent_types(t) for st in subtype_mapping[et] for est in get_equivalent_types(st)} for t in set(subtype_mapping)}
        # remove non-dbpedia types from ontology
        subtype_mapping = {t: {st for st in sts if dbp_util.is_dbp_type(st) or st == rdf_util.CLASS_OWL_THING} for t, sts in subtype_mapping.items() if dbp_util.is_dbp_type(t) or t == rdf_util.CLASS_OWL_THING}
        __TYPE_GRAPH__ = nx.DiGraph(incoming_graph_data=[(t, st) for t, sts in subtype_mapping.items() for st in sts])

    return __TYPE_GRAPH__
