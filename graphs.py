import util
import datetime
import impl.dbpedia.store as dbp_store
import impl.category.store as cat_store
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


def generate_graphs():
    """Generate the graphs of the paper 'Uncovering the Semantics of Wikipedia Categories'

    Note that there will be some deviations between the generated graphs and the ones in the paper
    as we continued to improve our extraction methods after the submission of the paper.
    """
    _generate_minimum_confidence_graph()
    _generate_dbpedia_coverage_graph()
    _generate_dbpedia_unknown_resources_graph()


def _generate_minimum_confidence_graph():
    """Create graph of Figure 3"""
    # this data is taken from the manually extracted evaluation results on
    # http://data.dws.informatik.uni-mannheim.de/Cat2Ax/evaluation/min_confidence_evaluation/
    columns = ['confidence', 'relationAxioms', 'relationPrecision', 'typeAxioms', 'typePrecision']
    data = [
        (0.1, 234358, 96, 367768, 100),
        (0.09, 240881, 95.95, 376717, 99.95),
        (0.08, 247986, 95.89, 386653, 99.90),
        (0.07, 255530, 95.83, 397302, 99.74),
        (0.06, 263563, 95.72, 409936, 99.57),
        (0.05, 272659, 95.53, 429175, 99.50),
        (0.04, 282658, 95.05, 450695, 99.23),
        (0.03, 294018, 94.39, 476175, 98.85),
        (0.02, 307288, 94.11, 507462, 98.30),
        (0.01, 324942, 93.56, 547986, 96.65),
    ]

    # create line chart of precisions
    df = pd.DataFrame(data=data, columns=columns)
    df[['typePrecision', 'relationPrecision']].plot.line()
    ax = plt.axes()
    ax.legend(['Precision (type)', 'Precision (relation)'], loc=(.025, .65), fontsize=12)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Precision of axioms [%]', fontsize=12)
    plt.yticks(fontsize=11)

    # create line chart of amounts
    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    df[['typeAxioms', 'relationAxioms']].plot.line(ax=ax2, linestyle=':')
    ax2.legend(['Amount (type)', 'Amount (relation)'], loc=(.025, .15), fontsize=12)
    plt.ylabel('Amount of axioms', fontsize=12)
    plt.yticks(fontsize=11)

    # format and store
    ax.xaxis.set_ticklabels(df['confidence'], fontsize=11)
    ax.xaxis.grid()
    ax.yaxis.grid()
    plt.savefig(util.get_results_file('results.graphs.minimum_confidence'), bbox_inches='tight')


def _generate_dbpedia_coverage_graph():
    """Create graph of Figure 4a"""
    # retrieve data from extracted axioms and assertions
    cat2ax_relation_axioms = pd.read_csv(util.get_results_file('results.cat2ax.relation_axioms'), sep=';')
    cat2ax_type_axioms = pd.read_csv(util.get_results_file('results.cat2ax.type_axioms'), sep=';')
    cat2ax_relation_triples = pd.read_csv(util.get_results_file('results.cat2ax.relation_assertions'), sep=';')
    cat2ax_type_triples = pd.read_csv(util.get_results_file('results.cat2ax.type_assertions'), sep=';')

    catriple_relation_axioms = pd.read_csv(util.get_results_file('results.catriple.relation_axioms'), sep=';')
    catriple_relation_triples = pd.read_csv(util.get_results_file('results.catriple.relation_assertions'), sep=';')

    cdf_relation_axioms = pd.read_csv(util.get_results_file('results.cdf.relation_axioms'), sep=';')
    cdf_type_axioms = pd.read_csv(util.get_results_file('results.cdf.type_axioms'), sep=';')
    cdf_relation_triples = pd.read_csv(util.get_results_file('results.cdf.relation_assertions'), sep=';')
    cdf_type_triples = pd.read_csv(util.get_results_file('results.cdf.type_assertions'), sep=';')

    # retrieve unique entity counts
    cat2ax_cat_count = len(set(cat2ax_relation_axioms['cat'].unique()) | set(cat2ax_type_axioms['cat'].unique()))
    catriple_cat_count = len(set(catriple_relation_axioms['cat'].unique()))
    cdf_cat_count = len(set(cdf_relation_axioms['cat'].unique()) | set(cdf_type_axioms['cat'].unique()))
    total_cat_count = len(cat_store.get_usable_cats())

    cat2ax_preds = cat2ax_relation_triples.groupby(by='pred').count()
    cat2ax_pred_count = len(cat2ax_preds[cat2ax_preds['sub'] >= 100].index)
    catriple_preds = catriple_relation_triples.groupby(by='pred').count()
    catriple_pred_count = len(catriple_preds[catriple_preds['sub'] >= 100].index)
    cdf_preds = cdf_relation_triples.groupby(by='pred').count()
    cdf_pred_count = len(cdf_preds[cdf_preds['sub'] >= 100].index)
    total_pred_count = len(dbp_store.get_all_predicates())

    cat2ax_res_count = len(set(cat2ax_relation_triples['sub'].unique()) | set(cat2ax_type_triples['sub'].unique()))
    catriple_res_count = len(set(catriple_relation_triples['sub'].unique()))
    cdf_res_count = len(set(cdf_relation_triples['sub'].unique()) | set(cdf_type_triples['sub'].unique()))
    total_res_count = len(dbp_store.get_resources())

    # initialise bars
    bars_ca = [cat2ax_cat_count / total_cat_count, cat2ax_res_count / total_res_count, cat2ax_pred_count / total_pred_count]
    bars_ct = [catriple_cat_count / total_cat_count, catriple_res_count / total_res_count, catriple_pred_count / total_pred_count]
    bars_cdf = [cdf_cat_count / total_cat_count, cdf_res_count / total_res_count, cdf_pred_count / total_pred_count]

    # arrange bars
    bar_width = 0.25
    r1 = np.arange(len(bars_ca))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # make plot
    plt.figure(figsize=(8, 5))
    plt.bar(r1, bars_ca, color='#2d7f5e', width=bar_width, edgecolor='white', label='Cat2Ax')
    plt.bar(r2, bars_ct, color='darkgrey', width=bar_width, edgecolor='white', label='Catriple')
    plt.bar(r3, bars_cdf, color='black', width=bar_width, edgecolor='white', label='C-DF')
    plt.ylabel('Fraction of items covered', fontsize=16)
    plt.xticks([r + bar_width for r in range(len(bars_ca))], ['(1) Categories', '(2) Resources', '(3) Properties'], fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    ax = plt.gca()
    ax.yaxis.grid()

    plt.savefig(util.get_results_file('results.graphs.dbpedia_coverage'), bbox_inches='tight')


def _generate_dbpedia_unknown_resources_graph():
    """Create graph of Figure 4b"""
    # retrieve data from extracted assertions
    cat2ax_relation_triples = pd.read_csv(util.get_results_file('results.cat2ax.relation_assertions'), sep=';')
    cat2ax_new_relation_resources = len({r for r in cat2ax_relation_triples['sub'].unique() if not dbp_store.get_properties(r)})
    cat2ax_type_triples = pd.read_csv(util.get_results_file('results.cat2ax.type_assertions'), sep=';')
    cat2ax_new_type_resources = len({r for r in cat2ax_type_triples['sub'].unique() if not dbp_store.get_types(r)})

    catriple_relation_triples = pd.read_csv(util.get_results_file('results.catriple.relation_assertions'), sep=';')
    catriple_new_relation_resources = len({r for r in catriple_relation_triples['sub'].unique() if not dbp_store.get_properties(r)})

    cdf_relation_triples = pd.read_csv(util.get_results_file('results.cdf.relation_assertions'), sep=';')
    cdf_new_relation_resources = len({r for r in cdf_relation_triples['sub'].unique() if not dbp_store.get_properties(r)})
    cdf_type_triples = pd.read_csv(util.get_results_file('results.cdf.type_assertions'), sep=';')
    cdf_new_type_resources = len({r for r in cdf_type_triples['sub'].unique() if not dbp_store.get_types(r)})

    # initialise bars
    bars_ca = [cat2ax_new_relation_resources, cat2ax_new_type_resources]
    bars_ct = [catriple_new_relation_resources, 0]
    bars_cdf = [cdf_new_relation_resources, cdf_new_type_resources]

    # arrange bars
    bar_width = 0.25
    r1 = np.arange(len(bars_ca))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # make plot
    plt.figure(figsize=(8, 5))
    plt.bar(r1, bars_ca, color='#2d7f5e', width=bar_width, edgecolor='white', label='Cat2Ax')
    plt.bar(r2, bars_ct, color='darkgrey', width=bar_width, edgecolor='white', label='Catriple')
    plt.bar(r3, bars_cdf, color='black', width=bar_width, edgecolor='white', label='C-DF')
    plt.ylabel('Amount of resources', fontsize=16)
    plt.xticks([r + bar_width for r in range(len(bars_ca))], ['(1) Relations', '(2) Types'], fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    ax = plt.gca()
    ax.yaxis.grid()

    plt.savefig(util.get_results_file('results.graphs.dbpedia_unknown_resources'), bbox_inches='tight')


# --- START SCRIPT ---

if __name__ == '__main__':
    now = datetime.datetime.now()
    util.get_logger().info('Started graph generation.')

    generate_graphs()

    duration = (datetime.datetime.now() - now).seconds // 60
    util.get_logger().info(f'Finished graph generation after {duration} minutes.')
