import impl.util.rdf as rdf_util
from impl.dbpedia.util import NAMESPACE_DBP_RESOURCE

NAMESPACE_DBP_CATEGORY = NAMESPACE_DBP_RESOURCE + 'Category:'


def category2name(category: str) -> str:
    return rdf_util.uri2name(category, NAMESPACE_DBP_CATEGORY)
